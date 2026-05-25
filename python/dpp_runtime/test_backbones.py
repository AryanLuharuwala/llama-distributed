"""Unit tests for backbone adapters (CF12-G).

These tests don't load actual model weights — they exercise the
adapter selection, partition logic, and metadata invariants by
constructing minimal mock pipeline objects.  End-to-end image
generation per adapter would need GPU + model weights and lives in
the integration test suite (run on rtxserver).
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpp_runtime.backbones import (
    BackboneAdapter, SDXLAdapter, SD15Adapter, SD3Adapter, FluxAdapter,
    PixArtSigmaAdapter, AnimateDiffAdapter, SVDAdapter,
    CogVideoXAdapter, HunyuanVideoAdapter, MochiAdapter,
    LTXVideoAdapter, WanAdapter,
    adapter_for_pipeline_class, detect_pipeline_class,
    _NoOpModule,
)


def _mock_sdxl_pipe():
    """Build a mock pipe that looks like an SDXL pipeline to the adapter."""
    pipe = MagicMock()
    pipe.unet.down_blocks = [MagicMock() for _ in range(3)]
    pipe.unet.up_blocks = [MagicMock() for _ in range(3)]
    pipe.unet.mid_block = MagicMock()
    pipe.unet.conv_in = MagicMock()
    pipe.unet.conv_norm_out = MagicMock()
    pipe.unet.conv_out = MagicMock()
    pipe.unet.config.in_channels = 4
    return pipe


def _mock_dit_pipe(n_blocks=10, secondary=0):
    pipe = MagicMock()
    pipe.transformer.transformer_blocks = [MagicMock() for _ in range(n_blocks)]
    pipe.transformer.pos_embed = MagicMock()
    pipe.transformer.x_embedder = MagicMock()
    pipe.transformer.context_embedder = MagicMock()
    pipe.transformer.time_text_embed = MagicMock()
    pipe.transformer.norm_out = MagicMock()
    pipe.transformer.proj_out = MagicMock()
    if secondary:
        pipe.transformer.single_transformer_blocks = [MagicMock() for _ in range(secondary)]
    return pipe


class TestAdapterRegistry(unittest.TestCase):
    """All advertised pipelines map to a non-abstract adapter."""

    EXPECTED = {
        "StableDiffusionXLPipeline": SDXLAdapter,
        "StableDiffusionPipeline": SD15Adapter,
        "StableDiffusion3Pipeline": SD3Adapter,
        "FluxPipeline": FluxAdapter,
        "PixArtSigmaPipeline": PixArtSigmaAdapter,
        "PixArtAlphaPipeline": PixArtSigmaAdapter,  # alias
        "StableVideoDiffusionPipeline": SVDAdapter,
        "AnimateDiffPipeline": AnimateDiffAdapter,
        "AnimateDiffSDXLPipeline": AnimateDiffAdapter,  # alias
        "CogVideoXPipeline": CogVideoXAdapter,
        "HunyuanVideoPipeline": HunyuanVideoAdapter,
        "MochiPipeline": MochiAdapter,
        "LTXPipeline": LTXVideoAdapter,
        "WanPipeline": WanAdapter,
    }

    def test_all_mapped(self):
        for name, cls in self.EXPECTED.items():
            self.assertIs(adapter_for_pipeline_class(name), cls,
                          f"pipeline class {name!r} should map to {cls.__name__}")

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            adapter_for_pipeline_class("PotatoPipeline")


class TestPipelineDetection(unittest.TestCase):
    """detect_pipeline_class falls back to model-id heuristics."""

    def test_override_wins(self):
        self.assertEqual(detect_pipeline_class("anything", "/tmp", override="FluxPipeline"),
                         "FluxPipeline")

    def test_env_var_wins_over_heuristic(self):
        # Env-var path short-circuits before any diffusers import.
        with patch.dict(os.environ, {"DPP_PIPELINE": "PixArtSigmaPipeline"}):
            self.assertEqual(detect_pipeline_class("stable-diffusion-xl", "/tmp"),
                             "PixArtSigmaPipeline")

    def test_heuristic_routing(self):
        # detect_pipeline_class falls through to model-id heuristic when
        # diffusers' load_config either fails or returns no _class_name.
        # We install a stub diffusers.DiffusionPipeline whose load_config
        # raises — that drops us into the heuristic without needing the
        # real diffusers (which can't import in some CI envs due to
        # xformers ABI drift).
        stub_mod = types.ModuleType("diffusers")
        class _DP:
            @staticmethod
            def load_config(*a, **kw):
                raise RuntimeError("stubbed")
        stub_mod.DiffusionPipeline = _DP
        cases = [
            ("stabilityai/stable-diffusion-xl-base-1.0", "StableDiffusionXLPipeline"),
            ("stabilityai/sdxl-turbo", "StableDiffusionXLPipeline"),
            ("runwayml/stable-diffusion-v1-5", "StableDiffusionPipeline"),
            ("stabilityai/stable-diffusion-3-medium", "StableDiffusion3Pipeline"),
            ("black-forest-labs/FLUX.1-dev", "FluxPipeline"),
            ("PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", "PixArtSigmaPipeline"),
            ("stabilityai/stable-video-diffusion-img2vid", "StableVideoDiffusionPipeline"),
            ("guoyww/animatediff-motion-adapter-v1-5", "AnimateDiffPipeline"),
            ("THUDM/CogVideoX-2b", "CogVideoXPipeline"),
            ("tencent/HunyuanVideo", "HunyuanVideoPipeline"),
            ("genmo/mochi-1-preview", "MochiPipeline"),
            ("Lightricks/LTX-Video", "LTXPipeline"),
            ("Wan-AI/Wan2.1-T2V-14B", "WanPipeline"),
        ]
        with patch.dict(sys.modules, {"diffusers": stub_mod}):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("DPP_PIPELINE", None)
                for model_id, expected in cases:
                    got = detect_pipeline_class(model_id, "/tmp")
                    self.assertEqual(got, expected,
                                     f"{model_id} → expected {expected} got {got}")


class TestAdapterMetadata(unittest.TestCase):
    """Pin invariants subclasses promise (CFG, video-ness, residuals)."""

    def test_image_adapters_not_video(self):
        for cls in (SDXLAdapter, SD15Adapter, SD3Adapter, FluxAdapter, PixArtSigmaAdapter):
            self.assertFalse(cls.IS_VIDEO, f"{cls.__name__} should not be video")

    def test_video_adapters_are_video(self):
        for cls in (SVDAdapter, AnimateDiffAdapter, CogVideoXAdapter,
                    HunyuanVideoAdapter, MochiAdapter, LTXVideoAdapter, WanAdapter):
            self.assertTrue(cls.IS_VIDEO, f"{cls.__name__} should be video")

    def test_dit_adapters_have_no_residuals(self):
        for cls in (SD3Adapter, FluxAdapter, PixArtSigmaAdapter,
                    CogVideoXAdapter, HunyuanVideoAdapter, MochiAdapter,
                    LTXVideoAdapter, WanAdapter):
            self.assertFalse(cls.HAS_RESIDUALS, f"{cls.__name__} should have no skip residuals")

    def test_unet_adapters_have_residuals(self):
        for cls in (SDXLAdapter, SD15Adapter, AnimateDiffAdapter, SVDAdapter):
            self.assertTrue(cls.HAS_RESIDUALS, f"{cls.__name__} should have skip residuals")

    def test_flux_no_cfg(self):
        self.assertFalse(FluxAdapter.DOES_CFG)

    def test_hunyuanvideo_no_cfg(self):
        self.assertFalse(HunyuanVideoAdapter.DOES_CFG)

    def test_sdxl_vae_needs_fp32(self):
        self.assertTrue(SDXLAdapter.VAE_NEEDS_FP32)

    def test_dit_video_default_latent_channels_16(self):
        # Most video DiTs use 16-channel latents (vs 4 for SDXL).
        self.assertEqual(CogVideoXAdapter.LATENT_CHANNELS, 16)
        self.assertEqual(HunyuanVideoAdapter.LATENT_CHANNELS, 16)


class TestSDXLPartition(unittest.TestCase):
    """Partition logic: unowned blocks → _NoOpModule, owned untouched."""

    def test_total_blocks(self):
        a = SDXLAdapter(_mock_sdxl_pipe(), device="cpu", dtype="fp16")
        self.assertEqual(a.total_blocks(), 7)  # 3 + 1 + 3

    def test_index_to_module(self):
        a = SDXLAdapter(_mock_sdxl_pipe(), device="cpu", dtype="fp16")
        self.assertEqual(a.index_to_module(0), ("down", 0))
        self.assertEqual(a.index_to_module(2), ("down", 2))
        self.assertEqual(a.index_to_module(3), ("mid", 0))
        self.assertEqual(a.index_to_module(4), ("up", 0))
        self.assertEqual(a.index_to_module(6), ("up", 2))

    def test_partition_drops_unowned(self):
        pipe = _mock_sdxl_pipe()
        a = SDXLAdapter(pipe, device="cpu", dtype="fp16")
        owned_down_orig = pipe.unet.down_blocks[1]
        unowned_down_orig = pipe.unet.down_blocks[0]
        a.apply_partition(block_lo=1, block_hi=4)  # owns down[1], down[2], mid
        self.assertIs(pipe.unet.down_blocks[1], owned_down_orig)
        self.assertIsInstance(pipe.unet.down_blocks[0], _NoOpModule)
        self.assertIsInstance(pipe.unet.up_blocks[0], _NoOpModule)
        # Stage 1 (not first, not last): conv_in + conv_out replaced.
        self.assertIsInstance(pipe.unet.conv_in, _NoOpModule)
        self.assertIsInstance(pipe.unet.conv_out, _NoOpModule)

    def test_partition_first_keeps_conv_in(self):
        pipe = _mock_sdxl_pipe()
        a = SDXLAdapter(pipe, device="cpu", dtype="fp16")
        orig_conv_in = pipe.unet.conv_in
        a.apply_partition(block_lo=0, block_hi=3)
        self.assertIs(pipe.unet.conv_in, orig_conv_in)
        # But conv_out goes (not last).
        self.assertIsInstance(pipe.unet.conv_out, _NoOpModule)

    def test_partition_last_keeps_conv_out(self):
        pipe = _mock_sdxl_pipe()
        a = SDXLAdapter(pipe, device="cpu", dtype="fp16")
        orig_conv_out = pipe.unet.conv_out
        a.apply_partition(block_lo=4, block_hi=7)
        self.assertIs(pipe.unet.conv_out, orig_conv_out)
        self.assertIsInstance(pipe.unet.conv_in, _NoOpModule)


class TestDiTPartition(unittest.TestCase):

    def test_flat_dit_total(self):
        a = PixArtSigmaAdapter(_mock_dit_pipe(n_blocks=28), device="cpu", dtype="fp16")
        self.assertEqual(a.total_blocks(), 28)

    def test_flux_two_pool_total(self):
        a = FluxAdapter(_mock_dit_pipe(n_blocks=19, secondary=38), device="cpu", dtype="fp16")
        self.assertEqual(a.total_blocks(), 19 + 38)

    def test_flux_secondary_index(self):
        a = FluxAdapter(_mock_dit_pipe(n_blocks=19, secondary=38), device="cpu", dtype="fp16")
        self.assertEqual(a.index_to_module(0), ("primary", 0))
        self.assertEqual(a.index_to_module(18), ("primary", 18))
        self.assertEqual(a.index_to_module(19), ("secondary", 0))
        self.assertEqual(a.index_to_module(56), ("secondary", 37))

    def test_dit_partition_replaces_unowned(self):
        pipe = _mock_dit_pipe(n_blocks=28)
        a = PixArtSigmaAdapter(pipe, device="cpu", dtype="fp16")
        a.apply_partition(block_lo=10, block_hi=20)
        for i in range(28):
            blk = pipe.transformer.transformer_blocks[i]
            if 10 <= i < 20:
                self.assertNotIsInstance(blk, _NoOpModule, f"block {i} should be owned")
            else:
                self.assertIsInstance(blk, _NoOpModule, f"block {i} should be NoOp")

    def test_dit_partition_first_keeps_patch_embed(self):
        pipe = _mock_dit_pipe(n_blocks=28)
        orig_pos_embed = pipe.transformer.pos_embed
        a = PixArtSigmaAdapter(pipe, device="cpu", dtype="fp16")
        a.apply_partition(block_lo=0, block_hi=14)
        # First stage keeps pos_embed.
        self.assertIs(pipe.transformer.pos_embed, orig_pos_embed)
        # Last-stage attributes get NoOpped on non-last.
        self.assertIsInstance(pipe.transformer.norm_out, _NoOpModule)
        self.assertIsInstance(pipe.transformer.proj_out, _NoOpModule)

    def test_dit_partition_last_keeps_proj_out(self):
        pipe = _mock_dit_pipe(n_blocks=28)
        orig_proj_out = pipe.transformer.proj_out
        a = PixArtSigmaAdapter(pipe, device="cpu", dtype="fp16")
        a.apply_partition(block_lo=14, block_hi=28)
        self.assertIs(pipe.transformer.proj_out, orig_proj_out)
        self.assertIsInstance(pipe.transformer.pos_embed, _NoOpModule)


class TestUNetPayloadHandlesVideoRanks(unittest.TestCase):
    """The UNetStepPayload codec is rank-agnostic — verify a 5-D video
    sample tensor round-trips."""

    def test_5d_round_trip(self):
        from dpp_runtime.unet_payload import UNetStepPayload, UNetTensor
        # [1 batch, 16 frames, 4 channels, 32 h, 32 w]
        dims = (1, 16, 4, 32, 32)
        n = 1 * 16 * 4 * 32 * 32 * 2  # fp16 bytes
        sample = UNetTensor(dims=dims, data=bytes([0x42]) * n)
        p = UNetStepPayload(sample=sample, residuals=[], step_idx=5, timestep=987.5)
        got = UNetStepPayload.decode(p.encode())
        self.assertEqual(got.sample.dims, dims)
        self.assertEqual(len(got.sample.data), n)
        self.assertEqual(got.step_idx, 5)


class TestNoOpModule(unittest.TestCase):

    def test_noop_call_raises(self):
        n = _NoOpModule()
        with self.assertRaises(RuntimeError):
            n()

    def test_noop_to_self(self):
        n = _NoOpModule()
        self.assertIs(n.to("cuda"), n)

    def test_noop_introspection_safe(self):
        n = _NoOpModule()
        # diffusers occasionally walks these — must not raise.
        list(n.parameters())
        list(n.named_parameters())
        list(n.children())
        list(n.modules())


if __name__ == "__main__":
    unittest.main()
