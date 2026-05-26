"""Backbone adapters: pluggable per-architecture load + partition + forward.

The CF12-A→F path was SDXL-only.  CF12-G abstracts the model-specific
bits behind a `BackboneAdapter` interface so the Worker stays
architecture-agnostic and new diffusion families can be added with one
class plus a pipeline-class-name → adapter mapping.

Two tiers of support land here:

  Tier 1 — same DPP machinery, different block index/loader:
    • SD 1.5 / 2.x  (UNet2DConditionModel, single text encoder)
    • SDXL          (UNet2DConditionModel + add_embedding, dual TE)
    • SD 3 / 3.5    (MM-DiT, triple TE — T5 + 2× CLIP)
    • Flux.1        (MM-DiT + single_transformer_blocks, T5 + CLIP)
    • PixArt-α/Σ    (DiT, single T5 encoder)

  Tier 2 — same machinery but 5-D video latents and/or motion modules:
    • Stable Video Diffusion (SVD)         — image-conditioned video UNet
    • AnimateDiff (SD1.5 + SDXL variants)  — UNet + motion adapter
    • CogVideoX                             — DiT video
    • HunyuanVideo                          — DiT video
    • Mochi-1                               — DiT video
    • LTX-Video                             — DiT video
    • Wan 2.1                               — DiT video

All adapters share these invariants:

  • The wire payload (UNetStepPayload) is rank-agnostic.  5-D video
    tensors [B, F, C, H, W] flow through it unchanged — the "UNet"
    name is historical.
  • Residuals are only emitted by UNet families.  DiT adapters return
    `residuals_out = []`, and DiT partitions ignore `residuals_in`.
  • Last-stage CFG combine + scheduler.step always lives in
    `_unet_first_loopback` on the Worker side (CF12-E loopback shape).
    Adapters just expose `cfg_combine` and the loop-shape stays the
    same across backbones.
  • For models without classifier-free guidance (e.g. Flux), the
    adapter sets `DOES_CFG = False`; the Worker skips the uncond
    branch and ships only the cond hidden state through.
"""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# ─── _NoOpModule mirrored locally so backbones.py is self-contained ──────────


class _NoOpModule:
    """Placeholder for a block this stage doesn't own — see worker._NoOpModule."""
    def __init__(self):
        pass

    def __call__(self, *a, **kw):
        raise RuntimeError("_NoOpModule called — partition bug")

    def to(self, *a, **kw):
        return self

    def eval(self):
        return self

    def parameters(self):
        return iter(())

    def named_parameters(self, *a, **kw):
        return iter(())

    def children(self):
        return iter(())

    def modules(self):
        return iter(())


def _torch_dtype(s: str):
    import torch
    return {"fp16": torch.float16, "f16": torch.float16,
            "bf16": torch.bfloat16, "fp32": torch.float32}[s]


# ─── BackboneAdapter base ───────────────────────────────────────────────────


@dataclass
class EncodedPrompt:
    """Normalized text-encode output across all backbones.

    `embeds` is [N, T, D] stacked cond+uncond (or just cond for non-CFG).
    `pooled` is [N, D_pool] optional.
    `extras` is a dict of any per-backbone additional tensors that need
    to ride along to the UNet/DiT stage (e.g. SVD image_embeds, Flux
    text_ids, SD3 negative_pooled).  The Worker ships each tensor as a
    separate tok_seq frame keyed by name; the receiving stage rebuilds
    the dict.
    """
    embeds: Any                          # torch.Tensor [N, T, D]
    pooled: Optional[Any] = None         # torch.Tensor [N, D] or None
    extras: Dict[str, Any] = field(default_factory=dict)


class BackboneAdapter:
    """Abstract per-architecture adapter.

    The Worker holds one instance per loaded role.  Subclasses override
    the methods that differ; defaults match the SDXL UNet path so
    UNet-style backbones only need to override loader + tweaks.
    """

    # Class-level metadata (subclasses override) ─────────────────────────────
    PIPELINE_CLASS: str = ""             # diffusers class name
    BACKBONE_ATTR: str = "unet"          # attribute on pipe ("unet" / "transformer")
    HAS_RESIDUALS: bool = True           # UNet skip connections
    DOES_CFG: bool = True                # classifier-free guidance pair
    IS_VIDEO: bool = False               # 5-D latents
    VAE_NEEDS_FP32: bool = False         # SDXL/SD1.5 VAEs prefer fp32 to avoid NaN
    LATENT_CHANNELS: int = 4             # most pipes; some video models = 16

    # ── construction ────────────────────────────────────────────────────────

    def __init__(self, pipe, device: str, dtype: str, model: str = "", cache: str = ""):
        self.pipe = pipe
        self.device = device
        self.dtype = dtype
        # model + cache are needed for lazy-loading a VAE encoder on the
        # TE rig (img2img path).  Filled by make_adapter().
        self.model_name = model
        self.cache_dir = cache
        # Lazy VAE encoder: populated by encode_init_image() the first
        # time img2img is used on a rig that didn't load the VAE in
        # load_for_role (typically the text_encoder rig).
        self._lazy_vae = None

    @property
    def backbone(self):
        return getattr(self.pipe, self.BACKBONE_ATTR)

    # ── img2img init-image encode (default impl) ────────────────────────────

    def encode_init_image(self, url_or_path: str, h: int, w: int):
        """Fetch an image, resize to (h, w), VAE-encode → init latent.

        Runs on the text_encoder rig: that role's pipe is loaded with
        vae=None, so we lazy-load just the VAE component from the same
        model id and cache the AutoencoderKL instance on the adapter.

        Returns a [1, C, H/8, W/8] tensor in self.dtype on self.device.
        """
        import io as _io
        import torch
        from PIL import Image
        # Fetch image bytes.
        if url_or_path.startswith(("http://", "https://", "data:")):
            if url_or_path.startswith("data:"):
                import base64
                head, _, b64 = url_or_path.partition(",")
                raw = base64.b64decode(b64)
            else:
                import urllib.request
                with urllib.request.urlopen(url_or_path, timeout=30) as r:
                    raw = r.read()
        else:
            with open(url_or_path, "rb") as f:
                raw = f.read()
        img = Image.open(_io.BytesIO(raw)).convert("RGB").resize((w, h), Image.LANCZOS)
        import numpy as np
        arr = np.asarray(img).astype("float32") / 127.5 - 1.0  # [-1, 1]
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        vae = getattr(self.pipe, "vae", None)
        if vae is None:
            if self._lazy_vae is None:
                from diffusers import AutoencoderKL
                vae_dtype = torch.float32 if self.VAE_NEEDS_FP32 else _torch_dtype(self.dtype)
                self._lazy_vae = AutoencoderKL.from_pretrained(
                    self.model_name, subfolder="vae",
                    torch_dtype=vae_dtype, cache_dir=self.cache_dir,
                ).to(self.device)
            vae = self._lazy_vae
        target_dtype = torch.float32 if self.VAE_NEEDS_FP32 else next(vae.parameters()).dtype
        t = t.to(self.device, dtype=target_dtype)
        with torch.no_grad():
            lat = vae.encode(t).latent_dist.sample()
        lat = lat * vae.config.scaling_factor
        return lat.to(dtype=_torch_dtype(self.dtype))

    # ── load (class method — chooses the right diffusers pipeline) ──────────

    @classmethod
    def load_for_role(cls, role: str, model: str, device: str, dtype: str, cache: str):
        """Return a diffusers pipeline loaded with only the components
        needed for `role` ∈ {text_encoder, unet, vae}.  Drops the others
        to None so we don't pull weights we won't use.

        Subclasses override to wire the correct pipeline class and the
        correct component-skip kwargs.
        """
        raise NotImplementedError(f"{cls.__name__}.load_for_role")

    # ── role-scoped HF snapshot (CF12-B) ────────────────────────────────────
    #
    # `from_pretrained(unet=None, vae=None, ...)` only skips *loading* the
    # excluded components into RAM — diffusers' resolver still calls
    # snapshot_download() under the hood and pulls every file in the repo.
    # For multi-rig role chains that's wasteful: a TE rig downloads the
    # UNet weights it will never touch.  We pre-warm the HF cache with a
    # role-scoped snapshot (allow_patterns filters to just the directories
    # this role uses) before any from_pretrained call; the subsequent
    # from_pretrained call then resolves entirely from the local cache.

    @classmethod
    def allow_patterns_for_role(cls, role: str) -> List[str]:
        """HuggingFace allow_patterns globs for `role`.  Subclasses
        override per pipeline-layout.  Default returns a permissive
        glob that fetches the whole repo (safe fallback).
        """
        return ["*"]

    @classmethod
    def _snapshot_for_role(cls, model: str, cache: str, role: str) -> None:
        """Pre-warm the HF cache with only the files this role needs.

        Skipped (no-op) when:
          • `model` is a local path (not a hub repo id), or
          • huggingface_hub isn't importable for some reason, or
          • the patterns list is empty.

        Any download error is logged and swallowed — the downstream
        from_pretrained will retry under its own resolver and raise a
        clearer error if the model truly isn't reachable.
        """
        # Heuristic: local paths contain os.sep or start with "./".  HF
        # repo ids look like "org/name" (single slash, no os.sep).
        if not model or os.path.isabs(model) or model.startswith((".", "/")) or os.path.exists(model):
            return
        try:
            from huggingface_hub import snapshot_download
        except Exception:  # pragma: no cover - hf_hub is a diffusers dep
            return
        patterns = cls.allow_patterns_for_role(role)
        if not patterns:
            return
        try:
            os.makedirs(cache, exist_ok=True)
            snapshot_download(
                repo_id=model,
                cache_dir=cache,
                allow_patterns=patterns,
            )
            log.info("snapshot_download(%s, role=%s) ok — patterns=%s",
                     model, role, patterns)
        except Exception as e:
            log.warning("snapshot_download(%s, role=%s) failed: %s — "
                        "falling through to from_pretrained", model, role, e)

    # ── partition (UNet-style by default) ───────────────────────────────────

    def total_blocks(self) -> int:
        """Number of linearised partitionable blocks."""
        bb = self.backbone
        n_down = len(getattr(bb, "down_blocks", []))
        n_up = len(getattr(bb, "up_blocks", []))
        n_mid = 1 if getattr(bb, "mid_block", None) is not None else 0
        return n_down + n_mid + n_up

    def index_to_module(self, idx: int) -> Tuple[str, int]:
        """Map linearised idx → ("down"|"mid"|"up", sub_idx)."""
        bb = self.backbone
        n_down = len(bb.down_blocks)
        if idx < n_down:
            return "down", idx
        if idx == n_down:
            return "mid", 0
        return "up", idx - n_down - 1

    def apply_partition(self, block_lo: int, block_hi: int) -> None:
        """Replace unowned blocks with `_NoOpModule` to free VRAM."""
        bb = self.backbone
        total = self.total_blocks()
        is_first = block_lo == 0
        is_last = block_hi == total
        for idx in range(total):
            kind, sub = self.index_to_module(idx)
            if block_lo <= idx < block_hi:
                continue
            if kind == "down":
                bb.down_blocks[sub] = _NoOpModule()
            elif kind == "mid":
                bb.mid_block = _NoOpModule()
            elif kind == "up":
                bb.up_blocks[sub] = _NoOpModule()
        if not is_first and hasattr(bb, "conv_in"):
            bb.conv_in = _NoOpModule()
        if not is_last:
            if hasattr(bb, "conv_norm_out"):
                bb.conv_norm_out = _NoOpModule()
            if hasattr(bb, "conv_out"):
                bb.conv_out = _NoOpModule()
        log.info(
            "%s partition: blocks [%d,%d) of %d (first=%s last=%s)",
            self.__class__.__name__, block_lo, block_hi, total, is_first, is_last,
        )

    # ── text encode (override per backbone) ─────────────────────────────────

    def encode_prompt(self, prompt: str, *, negative_prompt: str = "") -> EncodedPrompt:
        raise NotImplementedError(f"{self.__class__.__name__}.encode_prompt")

    # ── init latents ────────────────────────────────────────────────────────

    def init_latents(self, *, h: int, w: int, frames: int, seed: int):
        """Build initial noise.  Image: [1,C,H/8,W/8].  Video: [1,F,C,H/8,W/8]."""
        import torch
        scheduler = self.pipe.scheduler
        gen = torch.Generator(device=self.device).manual_seed(seed)
        ch = self.LATENT_CHANNELS
        dtype = _torch_dtype(self.dtype)
        if self.IS_VIDEO:
            shape = (1, frames, ch, h // 8, w // 8)
        else:
            shape = (1, ch, h // 8, w // 8)
        latents = torch.randn(shape, generator=gen, device=self.device, dtype=dtype)
        if hasattr(scheduler, "init_noise_sigma"):
            latents = latents * scheduler.init_noise_sigma
        return latents

    # ── forward_range (override per backbone block shape) ───────────────────

    def forward_range(self, *, block_lo: int, block_hi: int, sample, timestep,
                      encoder_hidden_states, added_cond_kwargs,
                      residuals_in) -> Tuple[Any, List[Any]]:
        """Run blocks [block_lo, block_hi) on `sample`.  Returns (sample, residuals_out)."""
        raise NotImplementedError(f"{self.__class__.__name__}.forward_range")

    # ── cfg combine + step (last-stage logic, identical across most) ────────

    def cfg_combine(self, noise_pred, cfg: float):
        if not self.DOES_CFG:
            return noise_pred
        npr, pr = noise_pred.chunk(2)
        return npr + cfg * (pr - npr)

    # ── VAE decode (image vs video) ─────────────────────────────────────────

    def vae_decode(self, latents) -> bytes:
        """Decode latents → bytes ready to send over the wire.

        Images: PNG bytes.  Video: MP4 bytes (H.264 via ffmpeg-python or
        imageio).  Subclasses override only if the VAE has a non-standard
        scaling factor or post-processing.
        """
        import torch
        from PIL import Image
        vae = self.pipe.vae
        if self.VAE_NEEDS_FP32:
            vae.to(dtype=torch.float32)
            target_dtype = torch.float32
        else:
            target_dtype = next(vae.parameters()).dtype
        latents = latents.to(target_dtype) / vae.config.scaling_factor
        with torch.no_grad():
            image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = (image.cpu().permute(0, 2, 3, 1).float().numpy() * 255).round().astype("uint8")
        buf = io.BytesIO()
        Image.fromarray(image[0]).save(buf, format="PNG")
        return buf.getvalue()


# ─── Tier 1: SDXL ───────────────────────────────────────────────────────────


class SDXLAdapter(BackboneAdapter):
    PIPELINE_CLASS = "StableDiffusionXLPipeline"
    BACKBONE_ATTR = "unet"
    HAS_RESIDUALS = True
    DOES_CFG = True
    VAE_NEEDS_FP32 = True
    LATENT_CHANNELS = 4

    @classmethod
    def allow_patterns_for_role(cls, role):
        common = ["*.json", "*.txt", "scheduler/*", "model_index.json"]
        if role == "text_encoder":
            return common + [
                "text_encoder/*", "text_encoder_2/*",
                "tokenizer/*", "tokenizer_2/*",
            ]
        if role == "unet":
            return common + ["unet/*"]
        if role == "vae":
            return common + ["vae/*"]
        return ["*"]

    @classmethod
    def load_for_role(cls, role, model, device, dtype, cache):
        from diffusers import StableDiffusionXLPipeline
        import torch
        os.makedirs(cache, exist_ok=True)
        cls._snapshot_for_role(model, cache, role)
        td = _torch_dtype(dtype)
        if role == "text_encoder":
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model, torch_dtype=td, cache_dir=cache, use_safetensors=True,
                unet=None, vae=None,
            )
            pipe.text_encoder.to(device)
            pipe.text_encoder_2.to(device)
        elif role == "unet":
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model, torch_dtype=td, cache_dir=cache, use_safetensors=True,
                text_encoder=None, text_encoder_2=None, vae=None,
            )
            pipe.unet.to(device)
        elif role == "vae":
            # SDXL base VAE: must use fp32 to avoid NaN.
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model, torch_dtype=torch.float32, cache_dir=cache, use_safetensors=True,
                text_encoder=None, text_encoder_2=None, unet=None,
            )
            pipe.vae.to(device, dtype=torch.float32)
        else:
            raise ValueError(role)
        return pipe

    def encode_prompt(self, prompt, *, negative_prompt=""):
        import torch
        with torch.no_grad():
            pe, npe, ppe, nppe = self.pipe.encode_prompt(
                prompt=prompt, prompt_2=None,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt, negative_prompt_2="",
            )
        embeds = torch.cat([npe, pe], dim=0)
        pooled = torch.cat([nppe, ppe], dim=0)
        return EncodedPrompt(embeds=embeds, pooled=pooled)

    def forward_range(self, *, block_lo, block_hi, sample, timestep,
                      encoder_hidden_states, added_cond_kwargs, residuals_in):
        """SDXL UNet partitioned forward.  Same shape as the original
        worker._unet_forward_range — kept here so future SDXL tweaks
        (e.g. ControlNet residual injection) have one home."""
        import torch
        unet = self.backbone
        total = self.total_blocks()
        is_first = block_lo == 0
        is_last = block_hi == total

        emb = None
        residuals = list(residuals_in)
        if is_first:
            sample = unet.conv_in(sample)
            t_emb = unet.time_proj(
                timestep.expand(sample.shape[0]) if timestep.dim() == 0 else timestep,
            )
            t_emb = t_emb.to(dtype=sample.dtype)
            emb = unet.time_embedding(t_emb)
            if hasattr(unet, "add_embedding") and added_cond_kwargs is not None:
                add_emb_in = torch.cat([
                    added_cond_kwargs["text_embeds"],
                    unet.add_time_proj(added_cond_kwargs["time_ids"].flatten()).reshape(
                        added_cond_kwargs["text_embeds"].shape[0], -1,
                    ).to(sample.dtype),
                ], dim=-1)
                emb = emb + unet.add_embedding(add_emb_in)
            residuals = [sample]

        for idx in range(block_lo, min(block_hi, total)):
            kind, sub = self.index_to_module(idx)
            if kind == "down":
                block = unet.down_blocks[sub]
                if hasattr(block, "has_cross_attention") and block.has_cross_attention:
                    sample, res = block(
                        hidden_states=sample, temb=emb,
                        encoder_hidden_states=encoder_hidden_states,
                    )
                else:
                    sample, res = block(hidden_states=sample, temb=emb)
                residuals.extend(res)
            elif kind == "mid":
                sample = unet.mid_block(
                    sample, emb,
                    encoder_hidden_states=encoder_hidden_states,
                )
            elif kind == "up":
                block = unet.up_blocks[sub]
                n_res = len(block.resnets)
                res_samples = tuple(residuals[-n_res:])
                residuals = residuals[:-n_res]
                if hasattr(block, "has_cross_attention") and block.has_cross_attention:
                    sample = block(
                        hidden_states=sample, temb=emb,
                        res_hidden_states_tuple=res_samples,
                        encoder_hidden_states=encoder_hidden_states,
                        upsample_size=None,
                    )
                else:
                    sample = block(
                        hidden_states=sample, temb=emb,
                        res_hidden_states_tuple=res_samples,
                        upsample_size=None,
                    )

        if is_last:
            sample = unet.conv_norm_out(sample)
            sample = unet.conv_act(sample)
            sample = unet.conv_out(sample)
        return sample, residuals


# ─── Tier 1: SD 1.5 / 2.x ───────────────────────────────────────────────────


class SD15Adapter(BackboneAdapter):
    PIPELINE_CLASS = "StableDiffusionPipeline"
    BACKBONE_ATTR = "unet"
    HAS_RESIDUALS = True
    DOES_CFG = True
    VAE_NEEDS_FP32 = True
    LATENT_CHANNELS = 4

    @classmethod
    def load_for_role(cls, role, model, device, dtype, cache):
        from diffusers import StableDiffusionPipeline
        import torch
        os.makedirs(cache, exist_ok=True)
        td = _torch_dtype(dtype)
        kwargs = dict(torch_dtype=td, cache_dir=cache, use_safetensors=True,
                      safety_checker=None, requires_safety_checker=False)
        if role == "text_encoder":
            pipe = StableDiffusionPipeline.from_pretrained(model, unet=None, vae=None, **kwargs)
            pipe.text_encoder.to(device)
        elif role == "unet":
            pipe = StableDiffusionPipeline.from_pretrained(model, text_encoder=None, vae=None, **kwargs)
            pipe.unet.to(device)
        elif role == "vae":
            pipe = StableDiffusionPipeline.from_pretrained(
                model, text_encoder=None, unet=None,
                torch_dtype=torch.float32, cache_dir=cache, use_safetensors=True,
                safety_checker=None, requires_safety_checker=False,
            )
            pipe.vae.to(device, dtype=torch.float32)
        else:
            raise ValueError(role)
        return pipe

    def encode_prompt(self, prompt, *, negative_prompt=""):
        import torch
        with torch.no_grad():
            pe, npe = self.pipe.encode_prompt(
                prompt=prompt, device=self.device,
                num_images_per_prompt=1, do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
        embeds = torch.cat([npe, pe], dim=0)
        return EncodedPrompt(embeds=embeds, pooled=None)

    def forward_range(self, *, block_lo, block_hi, sample, timestep,
                      encoder_hidden_states, added_cond_kwargs, residuals_in):
        """SD 1.5/2.x UNet partitioned forward.  Identical shape to SDXL
        minus the add_embedding path."""
        import torch
        unet = self.backbone
        total = self.total_blocks()
        is_first = block_lo == 0
        is_last = block_hi == total

        emb = None
        residuals = list(residuals_in)
        if is_first:
            sample = unet.conv_in(sample)
            t_emb = unet.time_proj(
                timestep.expand(sample.shape[0]) if timestep.dim() == 0 else timestep,
            )
            t_emb = t_emb.to(dtype=sample.dtype)
            emb = unet.time_embedding(t_emb)
            residuals = [sample]

        for idx in range(block_lo, min(block_hi, total)):
            kind, sub = self.index_to_module(idx)
            if kind == "down":
                block = unet.down_blocks[sub]
                if hasattr(block, "has_cross_attention") and block.has_cross_attention:
                    sample, res = block(
                        hidden_states=sample, temb=emb,
                        encoder_hidden_states=encoder_hidden_states,
                    )
                else:
                    sample, res = block(hidden_states=sample, temb=emb)
                residuals.extend(res)
            elif kind == "mid":
                sample = unet.mid_block(
                    sample, emb, encoder_hidden_states=encoder_hidden_states,
                )
            elif kind == "up":
                block = unet.up_blocks[sub]
                n_res = len(block.resnets)
                res_samples = tuple(residuals[-n_res:])
                residuals = residuals[:-n_res]
                if hasattr(block, "has_cross_attention") and block.has_cross_attention:
                    sample = block(
                        hidden_states=sample, temb=emb,
                        res_hidden_states_tuple=res_samples,
                        encoder_hidden_states=encoder_hidden_states,
                        upsample_size=None,
                    )
                else:
                    sample = block(
                        hidden_states=sample, temb=emb,
                        res_hidden_states_tuple=res_samples,
                        upsample_size=None,
                    )
        if is_last:
            sample = unet.conv_norm_out(sample)
            sample = unet.conv_act(sample)
            sample = unet.conv_out(sample)
        return sample, residuals


# ─── DiT block partition mixin ──────────────────────────────────────────────


class _DiTAdapter(BackboneAdapter):
    """Common partition shape for diffusers DiT-style backbones.

    DiT models in diffusers expose a flat `transformer_blocks` list (and
    sometimes additional pools like Flux's `single_transformer_blocks`).
    The block index is just `transformer_blocks[i]`; partitioning means
    None-ing out unowned blocks.  Subclasses override `forward_range` for
    model-specific patch-embed / output-projection logic.
    """
    BACKBONE_ATTR = "transformer"
    HAS_RESIDUALS = False
    DOES_CFG = True  # most DiTs do CFG; Flux is the exception
    VAE_NEEDS_FP32 = False
    LATENT_CHANNELS = 4

    # Optional secondary block list (Flux).  Empty by default.
    SECONDARY_ATTR: str = ""

    def total_blocks(self) -> int:
        n = len(self.backbone.transformer_blocks)
        if self.SECONDARY_ATTR:
            n += len(getattr(self.backbone, self.SECONDARY_ATTR))
        return n

    def index_to_module(self, idx: int) -> Tuple[str, int]:
        n_primary = len(self.backbone.transformer_blocks)
        if idx < n_primary:
            return "primary", idx
        return "secondary", idx - n_primary

    def apply_partition(self, block_lo: int, block_hi: int) -> None:
        bb = self.backbone
        total = self.total_blocks()
        is_first = block_lo == 0
        is_last = block_hi == total
        # None unowned transformer blocks.  diffusers' DiTs iterate the
        # ModuleList directly so we replace with _NoOpModule; the
        # forward_range skips by index, so the NoOp body is never called.
        n_primary = len(bb.transformer_blocks)
        for i in range(n_primary):
            if not (block_lo <= i < block_hi):
                bb.transformer_blocks[i] = _NoOpModule()
        if self.SECONDARY_ATTR:
            sec = getattr(bb, self.SECONDARY_ATTR)
            for j in range(len(sec)):
                idx = n_primary + j
                if not (block_lo <= idx < block_hi):
                    sec[j] = _NoOpModule()
        # Only stage-0 needs patch embed / pos embed / time embed.
        if not is_first:
            for attr in ("pos_embed", "x_embedder", "patch_embed", "context_embedder",
                         "time_text_embed", "time_embedder"):
                if hasattr(bb, attr):
                    setattr(bb, attr, _NoOpModule())
        # Only the last stage needs the output projection.
        if not is_last:
            for attr in ("norm_out", "proj_out", "scale_shift_table"):
                if hasattr(bb, attr):
                    setattr(bb, attr, _NoOpModule())
        log.info(
            "%s DiT partition: blocks [%d,%d) of %d (first=%s last=%s)",
            self.__class__.__name__, block_lo, block_hi, total, is_first, is_last,
        )


# ─── Tier 1: SD 3 / 3.5 (MM-DiT) ────────────────────────────────────────────


class SD3Adapter(_DiTAdapter):
    PIPELINE_CLASS = "StableDiffusion3Pipeline"
    LATENT_CHANNELS = 16

    @classmethod
    def allow_patterns_for_role(cls, role):
        common = ["*.json", "*.txt", "scheduler/*", "model_index.json"]
        if role == "text_encoder":
            return common + [
                "text_encoder/*", "text_encoder_2/*", "text_encoder_3/*",
                "tokenizer/*", "tokenizer_2/*", "tokenizer_3/*",
            ]
        if role == "unet":
            return common + ["transformer/*"]
        if role == "vae":
            return common + ["vae/*"]
        return ["*"]

    @classmethod
    def load_for_role(cls, role, model, device, dtype, cache):
        from diffusers import StableDiffusion3Pipeline
        os.makedirs(cache, exist_ok=True)
        cls._snapshot_for_role(model, cache, role)
        td = _torch_dtype(dtype)
        kwargs = dict(torch_dtype=td, cache_dir=cache, use_safetensors=True)
        if role == "text_encoder":
            # SD3 has T5 + 2× CLIP.  Load all three; skip transformer + vae.
            pipe = StableDiffusion3Pipeline.from_pretrained(model, transformer=None, vae=None, **kwargs)
            pipe.text_encoder.to(device)
            pipe.text_encoder_2.to(device)
            if pipe.text_encoder_3 is not None:
                pipe.text_encoder_3.to(device)
        elif role == "unet":  # "unet" role means "backbone" — for SD3 it's the transformer.
            pipe = StableDiffusion3Pipeline.from_pretrained(
                model, text_encoder=None, text_encoder_2=None, text_encoder_3=None, vae=None, **kwargs,
            )
            pipe.transformer.to(device)
        elif role == "vae":
            pipe = StableDiffusion3Pipeline.from_pretrained(
                model, text_encoder=None, text_encoder_2=None, text_encoder_3=None,
                transformer=None, **kwargs,
            )
            pipe.vae.to(device)
        else:
            raise ValueError(role)
        return pipe

    def encode_prompt(self, prompt, *, negative_prompt=""):
        import torch
        with torch.no_grad():
            out = self.pipe.encode_prompt(
                prompt=prompt, prompt_2=None, prompt_3=None,
                negative_prompt=negative_prompt, negative_prompt_2="", negative_prompt_3="",
                do_classifier_free_guidance=True, device=self.device,
            )
        # SD3 returns (prompt_embeds, negative_prompt_embeds,
        #              pooled_prompt_embeds, negative_pooled_prompt_embeds)
        pe, npe, ppe, nppe = out
        embeds = torch.cat([npe, pe], dim=0)
        pooled = torch.cat([nppe, ppe], dim=0)
        return EncodedPrompt(embeds=embeds, pooled=pooled)

    def forward_range(self, *, block_lo, block_hi, sample, timestep,
                      encoder_hidden_states, added_cond_kwargs, residuals_in):
        """SD3 MM-DiT partitioned forward.

        Each MM-DiT block jointly attends over [image-tokens | text-tokens]
        and emits both back.  Stage 0 patch-embeds the latents and time-
        embeds the timestep; the last stage applies norm_out + proj_out
        and unpatches.  Intermediate stages just iterate blocks.
        """
        import torch
        tf = self.backbone
        total = self.total_blocks()
        is_first = block_lo == 0
        is_last = block_hi == total

        # `sample` on entry to stage-0 is the latent [B, C, H, W].
        # On intermediate stages it's the post-patch hidden state
        # [B, S, D] — the wire payload carries whatever shape we emit.
        if is_first:
            # Patch embed + positional embedding.
            hidden = tf.pos_embed(sample)  # [B, S, D]
            # Pooled text_embeds + timestep → conditioning embedding.
            temb = tf.time_text_embed(timestep, added_cond_kwargs["pooled_projections"])
            encoder_hidden_states = tf.context_embedder(encoder_hidden_states)
        else:
            hidden = sample
            temb = added_cond_kwargs["temb"]  # cached from stage 0 on subsequent stages

        n_primary = len(tf.transformer_blocks)
        for idx in range(block_lo, min(block_hi, total)):
            kind, sub = self.index_to_module(idx)
            if kind == "primary":
                blk = tf.transformer_blocks[sub]
                encoder_hidden_states, hidden = blk(
                    hidden_states=hidden,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                )

        if is_last:
            hidden = tf.norm_out(hidden, temb)
            hidden = tf.proj_out(hidden)
            # Unpatchify — diffusers' SD3 transformer has _unpatchify
            # under that name in 0.30+; older versions need manual rearrange.
            if hasattr(tf, "_unpatchify"):
                hidden = tf._unpatchify(hidden)
            elif hasattr(tf, "unpatchify"):
                hidden = tf.unpatchify(hidden)
            sample = hidden
        else:
            sample = hidden
        return sample, []


# ─── Tier 1: Flux ───────────────────────────────────────────────────────────


class FluxAdapter(_DiTAdapter):
    PIPELINE_CLASS = "FluxPipeline"
    DOES_CFG = False  # Flux uses guidance_scale baked into the model (distilled)
    LATENT_CHANNELS = 16
    SECONDARY_ATTR = "single_transformer_blocks"

    @classmethod
    def allow_patterns_for_role(cls, role):
        common = ["*.json", "*.txt", "scheduler/*", "model_index.json"]
        if role == "text_encoder":
            return common + [
                "text_encoder/*", "text_encoder_2/*",
                "tokenizer/*", "tokenizer_2/*",
            ]
        if role == "unet":
            return common + ["transformer/*"]
        if role == "vae":
            return common + ["vae/*"]
        return ["*"]

    @classmethod
    def load_for_role(cls, role, model, device, dtype, cache):
        from diffusers import FluxPipeline
        os.makedirs(cache, exist_ok=True)
        cls._snapshot_for_role(model, cache, role)
        td = _torch_dtype(dtype)
        kwargs = dict(torch_dtype=td, cache_dir=cache, use_safetensors=True)
        if role == "text_encoder":
            pipe = FluxPipeline.from_pretrained(model, transformer=None, vae=None, **kwargs)
            pipe.text_encoder.to(device)
            pipe.text_encoder_2.to(device)
        elif role == "unet":
            pipe = FluxPipeline.from_pretrained(
                model, text_encoder=None, text_encoder_2=None, vae=None, **kwargs,
            )
            pipe.transformer.to(device)
        elif role == "vae":
            pipe = FluxPipeline.from_pretrained(
                model, text_encoder=None, text_encoder_2=None, transformer=None, **kwargs,
            )
            pipe.vae.to(device)
        else:
            raise ValueError(role)
        return pipe

    def encode_prompt(self, prompt, *, negative_prompt=""):
        import torch
        with torch.no_grad():
            # Flux encode_prompt returns (prompt_embeds, pooled_prompt_embeds, text_ids)
            # No CFG → no negative embeds; embeds shape is [1, T, D].
            pe, ppe, text_ids = self.pipe.encode_prompt(
                prompt=prompt, prompt_2=None, device=self.device, num_images_per_prompt=1,
            )
        return EncodedPrompt(embeds=pe, pooled=ppe, extras={"text_ids": text_ids})

    def forward_range(self, *, block_lo, block_hi, sample, timestep,
                      encoder_hidden_states, added_cond_kwargs, residuals_in):
        """Flux MM-DiT partitioned forward.

        Flux has two transformer block pools: `transformer_blocks` (joint
        image+text attention) followed by `single_transformer_blocks`
        (image-only after the text is folded in).  The linearised index
        spans both: 0..N1-1 = primary, N1..N1+N2-1 = secondary.
        """
        import torch
        tf = self.backbone
        total = self.total_blocks()
        is_first = block_lo == 0
        is_last = block_hi == total

        if is_first:
            hidden = tf.x_embedder(sample)  # [B, S_img, D]
            txt_ids = added_cond_kwargs["text_ids"]
            img_ids = added_cond_kwargs["img_ids"]
            temb = tf.time_text_embed(
                timestep, added_cond_kwargs.get("guidance"),
                added_cond_kwargs["pooled_projections"],
            )
            encoder_hidden_states = tf.context_embedder(encoder_hidden_states)
            ids = torch.cat([txt_ids, img_ids], dim=0)
            image_rotary_emb = tf.pos_embed(ids)
        else:
            hidden = sample
            temb = added_cond_kwargs["temb"]
            image_rotary_emb = added_cond_kwargs.get("image_rotary_emb")

        n_primary = len(tf.transformer_blocks)
        for idx in range(block_lo, min(block_hi, total)):
            kind, sub = self.index_to_module(idx)
            if kind == "primary":
                blk = tf.transformer_blocks[sub]
                encoder_hidden_states, hidden = blk(
                    hidden_states=hidden,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                )
            elif kind == "secondary":
                # Single-stream blocks: input is the concat of text+image
                # tokens (text first), output is the same shape.  diffusers
                # handles the cat/split inside the FluxSingleTransformerBlock,
                # so on first secondary call we need to cat once.
                if sub == 0:
                    hidden = torch.cat([encoder_hidden_states, hidden], dim=1)
                blk = getattr(tf, self.SECONDARY_ATTR)[sub]
                hidden = blk(
                    hidden_states=hidden, temb=temb,
                    image_rotary_emb=image_rotary_emb,
                )
                # On the LAST secondary block, strip the text prefix off so
                # downstream unpatchify gets only image tokens.
                if sub == len(getattr(tf, self.SECONDARY_ATTR)) - 1:
                    n_txt = encoder_hidden_states.shape[1]
                    hidden = hidden[:, n_txt:, :]

        if is_last:
            hidden = tf.norm_out(hidden, temb)
            hidden = tf.proj_out(hidden)
        return hidden, []


# ─── Tier 1: PixArt-α / PixArt-Σ / Sana (single-T5 DiT) ─────────────────────


class PixArtSigmaAdapter(_DiTAdapter):
    PIPELINE_CLASS = "PixArtSigmaPipeline"
    LATENT_CHANNELS = 4

    @classmethod
    def load_for_role(cls, role, model, device, dtype, cache):
        from diffusers import PixArtSigmaPipeline
        os.makedirs(cache, exist_ok=True)
        td = _torch_dtype(dtype)
        kwargs = dict(torch_dtype=td, cache_dir=cache, use_safetensors=True)
        if role == "text_encoder":
            pipe = PixArtSigmaPipeline.from_pretrained(model, transformer=None, vae=None, **kwargs)
            pipe.text_encoder.to(device)
        elif role == "unet":
            pipe = PixArtSigmaPipeline.from_pretrained(
                model, text_encoder=None, vae=None, **kwargs,
            )
            pipe.transformer.to(device)
        elif role == "vae":
            pipe = PixArtSigmaPipeline.from_pretrained(
                model, text_encoder=None, transformer=None, **kwargs,
            )
            pipe.vae.to(device)
        else:
            raise ValueError(role)
        return pipe

    def encode_prompt(self, prompt, *, negative_prompt=""):
        import torch
        with torch.no_grad():
            pe, pmask, npe, npmask = self.pipe.encode_prompt(
                prompt=prompt, do_classifier_free_guidance=True,
                negative_prompt=negative_prompt, device=self.device,
            )
        embeds = torch.cat([npe, pe], dim=0)
        # PixArt uses attention masks — pack into extras.
        mask = torch.cat([npmask, pmask], dim=0)
        return EncodedPrompt(embeds=embeds, pooled=None, extras={"attention_mask": mask})

    def forward_range(self, *, block_lo, block_hi, sample, timestep,
                      encoder_hidden_states, added_cond_kwargs, residuals_in):
        """PixArt-Σ DiT partitioned forward.

        Single-stream DiT: patch_embed → blocks (cross-attend to T5) →
        norm_out + proj_out → unpatchify.  No skip connections.
        """
        import torch
        tf = self.backbone
        total = self.total_blocks()
        is_first = block_lo == 0
        is_last = block_hi == total

        # Latent grid before patchification — needed to unpatchify at the end.
        patch_size = int(tf.config.patch_size)
        out_channels = int(tf.out_channels)
        latent_h = sample.shape[-2] // patch_size
        latent_w = sample.shape[-1] // patch_size

        if is_first:
            hidden = tf.pos_embed(sample)  # [B, S, D]
            # Timestep + (optionally) resolution/aspect ratio embedding.
            # adaln_single expects timestep as a 1d tensor [B]; scheduler
            # gives a 0-d scalar, so expand.
            ts = timestep
            if not torch.is_tensor(ts):
                ts = torch.tensor([ts], device=sample.device)
            if ts.dim() == 0:
                ts = ts[None]
            if ts.shape[0] == 1 and sample.shape[0] > 1:
                ts = ts.expand(sample.shape[0])
            emb_kwargs = {}
            if "resolution" in added_cond_kwargs:
                emb_kwargs["resolution"] = added_cond_kwargs["resolution"]
            if "aspect_ratio" in added_cond_kwargs:
                emb_kwargs["aspect_ratio"] = added_cond_kwargs["aspect_ratio"]
            temb, embedded_timestep = tf.adaln_single(
                ts, added_cond_kwargs.get("added_cond_kwargs", {}),
                batch_size=sample.shape[0], hidden_dtype=sample.dtype,
            )
            # Project encoder_hidden_states once.
            encoder_hidden_states = tf.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(
                sample.shape[0], -1, hidden.shape[-1],
            )
        else:
            hidden = sample
            temb = added_cond_kwargs["temb"]
            embedded_timestep = added_cond_kwargs.get("embedded_timestep")

        attn_mask = added_cond_kwargs.get("attention_mask")
        for idx in range(block_lo, min(block_hi, total)):
            kind, sub = self.index_to_module(idx)
            if kind == "primary":
                blk = tf.transformer_blocks[sub]
                hidden = blk(
                    hidden_states=hidden,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=attn_mask,
                    timestep=temb,
                )

        if is_last:
            # PixArt uses scale_shift_table + adaln-style output norm.
            shift, scale = (tf.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)
            hidden = tf.norm_out(hidden) * (1 + scale) + shift
            hidden = tf.proj_out(hidden)
            # Match diffusers PixArtTransformer2DModel: squeeze a singleton
            # dim if present, then unpatchify back to [B, C, H, W].
            if hidden.dim() == 4 and hidden.shape[1] == 1:
                hidden = hidden.squeeze(1)
            B = hidden.shape[0]
            hidden = hidden.reshape(
                B, latent_h, latent_w, patch_size, patch_size, out_channels,
            )
            hidden = torch.einsum("nhwpqc->nchpwq", hidden)
            hidden = hidden.reshape(
                B, out_channels, latent_h * patch_size, latent_w * patch_size,
            )
            # Learned-sigma models predict 2*C channels (noise + variance);
            # keep noise only, matching PixArtSigmaPipeline.
            if out_channels == 2 * self.LATENT_CHANNELS:
                hidden, _ = hidden.chunk(2, dim=1)
        return hidden, []


# ─── Tier 2: AnimateDiff (SD UNet + motion adapter) ─────────────────────────


class AnimateDiffAdapter(SD15Adapter):
    """SD 1.5 / SDXL with a MotionAdapter injecting temporal layers
    between spatial down/mid/up blocks.  The block index is identical
    to the parent UNet (motion modules ride inside each UNet block);
    `IS_VIDEO=True` flips latent shape to 5-D.
    """
    PIPELINE_CLASS = "AnimateDiffPipeline"
    IS_VIDEO = True
    LATENT_CHANNELS = 4

    @classmethod
    def load_for_role(cls, role, model, device, dtype, cache):
        # AnimateDiff loads the SD UNet + MotionAdapter into a single
        # AnimateDiffPipeline.  Model id is expected to point at the
        # composed pipeline checkpoint (HuggingFace ships pre-baked
        # ones, e.g. "guoyww/animatediff-motion-adapter-v1-5-3" paired
        # with a base SD1.5 model — we accept a single id of the
        # composed checkpoint for simplicity).
        from diffusers import AnimateDiffPipeline
        os.makedirs(cache, exist_ok=True)
        td = _torch_dtype(dtype)
        kwargs = dict(torch_dtype=td, cache_dir=cache, use_safetensors=True)
        if role == "text_encoder":
            pipe = AnimateDiffPipeline.from_pretrained(model, unet=None, vae=None, **kwargs)
            pipe.text_encoder.to(device)
        elif role == "unet":
            pipe = AnimateDiffPipeline.from_pretrained(
                model, text_encoder=None, vae=None, **kwargs,
            )
            pipe.unet.to(device)
        elif role == "vae":
            pipe = AnimateDiffPipeline.from_pretrained(
                model, text_encoder=None, unet=None, **kwargs,
            )
            pipe.vae.to(device)
        else:
            raise ValueError(role)
        return pipe

    def vae_decode(self, latents) -> bytes:
        """Video VAE decode: latents [1, F, C, H/8, W/8] → MP4 bytes."""
        import torch
        # Reshape to per-frame: [F, C, H/8, W/8].
        if latents.dim() == 5:
            latents = latents.squeeze(0)
        latents = latents.to(torch.float32) / self.pipe.vae.config.scaling_factor
        with torch.no_grad():
            frames = self.pipe.vae.decode(latents).sample  # [F, 3, H, W]
        frames = (frames / 2 + 0.5).clamp(0, 1)
        frames = (frames.cpu().permute(0, 2, 3, 1).float().numpy() * 255).round().astype("uint8")
        return _encode_mp4(frames, fps=8)


# ─── Tier 2: Stable Video Diffusion ─────────────────────────────────────────


class SVDAdapter(BackboneAdapter):
    """UNetSpatioTemporalConditionModel — SDXL-ish spatial layout with
    extra temporal layers folded into each block.  Conditioning is an
    image (CLIP image embed + VAE image embed), not text.
    """
    PIPELINE_CLASS = "StableVideoDiffusionPipeline"
    BACKBONE_ATTR = "unet"
    HAS_RESIDUALS = True
    DOES_CFG = True  # SVD does CFG over the image conditioning (min/max).
    IS_VIDEO = True
    LATENT_CHANNELS = 4

    @classmethod
    def load_for_role(cls, role, model, device, dtype, cache):
        from diffusers import StableVideoDiffusionPipeline
        os.makedirs(cache, exist_ok=True)
        td = _torch_dtype(dtype)
        kwargs = dict(torch_dtype=td, cache_dir=cache, use_safetensors=True,
                      variant=("fp16" if dtype in ("fp16", "f16") else None))
        if role == "text_encoder":
            # SVD uses image_encoder, not a text encoder.  Same role
            # name though — we'll repurpose the slot.
            pipe = StableVideoDiffusionPipeline.from_pretrained(model, unet=None, vae=None, **kwargs)
            pipe.image_encoder.to(device)
        elif role == "unet":
            pipe = StableVideoDiffusionPipeline.from_pretrained(
                model, image_encoder=None, vae=None, **kwargs,
            )
            pipe.unet.to(device)
        elif role == "vae":
            pipe = StableVideoDiffusionPipeline.from_pretrained(
                model, image_encoder=None, unet=None, **kwargs,
            )
            pipe.vae.to(device)
        else:
            raise ValueError(role)
        return pipe

    def encode_prompt(self, prompt, *, negative_prompt=""):
        """For SVD, `prompt` is a path/URL to the conditioning image —
        the Worker must hand us the actual PIL.Image upstream.  Returning
        an empty EncodedPrompt keeps the existing TE plumbing; the
        Worker's SVD path emits image embeds via a dedicated control
        frame (out of scope for this adapter's text-shaped contract).

        TODO: refactor the TE→UNet wire to carry typed conditioning
        (text-or-image) instead of always-text.
        """
        raise NotImplementedError("SVD text prompts not supported; pass image conditioning via image_embed control frame")

    def vae_decode(self, latents) -> bytes:
        import torch
        if latents.dim() == 5:
            # [1, F, C, H/8, W/8] → [F, C, H/8, W/8]
            latents = latents.squeeze(0)
        latents = latents.to(torch.float32) / self.pipe.vae.config.scaling_factor
        with torch.no_grad():
            # SVD VAE expects num_frames kwarg.
            frames = self.pipe.vae.decode(latents, num_frames=latents.shape[0]).sample
        frames = (frames / 2 + 0.5).clamp(0, 1)
        frames = (frames.cpu().permute(0, 2, 3, 1).float().numpy() * 255).round().astype("uint8")
        return _encode_mp4(frames, fps=7)


# ─── Tier 2: video DiTs (CogVideoX / HunyuanVideo / Mochi / LTX / Wan) ──────


class _VideoDiTAdapter(_DiTAdapter):
    """Common shape for the newer 5-D-latent video DiTs.

    All share the diffusers pattern:
      transformer.transformer_blocks: ModuleList
      transformer.patch_embed or transformer.x_embedder: 3-D patch projection
      transformer.norm_out + transformer.proj_out: output head
      VAE produces [B, C, F, H, W] (note: F before H/W on most)
      Scheduler is flow-matching (FlowMatchEulerDiscrete or similar)

    Subclasses pin PIPELINE_CLASS and the diffusers pipeline import path,
    plus any per-model encode_prompt quirks.
    """
    IS_VIDEO = True
    LATENT_CHANNELS = 16  # most are 16

    def init_latents(self, *, h, w, frames, seed):
        import torch
        gen = torch.Generator(device=self.device).manual_seed(seed)
        ch = self.LATENT_CHANNELS
        # diffusers convention for video DiTs: [B, C, F, H/8, W/8].
        # Some pipes (Mochi) use a temporal compression factor; we
        # delegate to the pipe's prepare_latents when available.
        if hasattr(self.pipe, "prepare_latents"):
            try:
                lat = self.pipe.prepare_latents(
                    batch_size=1, num_channels_latents=ch,
                    num_frames=frames, height=h, width=w,
                    dtype=_torch_dtype(self.dtype), device=self.device,
                    generator=gen, latents=None,
                )
                return lat
            except TypeError:
                pass  # signature varies; fall through
        shape = (1, ch, frames, h // 8, w // 8)
        latents = torch.randn(shape, generator=gen, device=self.device,
                              dtype=_torch_dtype(self.dtype))
        if hasattr(self.pipe.scheduler, "init_noise_sigma"):
            latents = latents * self.pipe.scheduler.init_noise_sigma
        return latents

    def vae_decode(self, latents) -> bytes:
        import torch
        latents = latents.to(torch.float32)
        if hasattr(self.pipe.vae.config, "scaling_factor"):
            latents = latents / self.pipe.vae.config.scaling_factor
        if hasattr(self.pipe.vae.config, "shift_factor") and self.pipe.vae.config.shift_factor:
            latents = latents + self.pipe.vae.config.shift_factor
        with torch.no_grad():
            frames = self.pipe.vae.decode(latents, return_dict=False)[0]
        # Frames may come back as [B, C, F, H, W] — rearrange to [F, H, W, C].
        if frames.dim() == 5:
            frames = frames.squeeze(0).permute(1, 2, 3, 0)  # [F, H, W, C]
        elif frames.dim() == 4:
            frames = frames.permute(0, 2, 3, 1)             # [F, H, W, C]
        frames = (frames / 2 + 0.5).clamp(0, 1)
        frames = (frames.cpu().float().numpy() * 255).round().astype("uint8")
        return _encode_mp4(frames, fps=getattr(self, "DEFAULT_FPS", 24))

    def forward_range(self, *, block_lo, block_hi, sample, timestep,
                      encoder_hidden_states, added_cond_kwargs, residuals_in):
        """Generic video-DiT partitioned forward.

        Stage 0 runs patch_embed + time embedding.  Intermediate stages
        iterate transformer_blocks on the hidden state.  Last stage
        applies norm_out + proj_out and unpatchifies.

        Each video DiT family has minor signature differences inside
        the block (e.g. CogVideoX expects `image_rotary_emb`, HunyuanVideo
        expects `freqs_cis`, Mochi/LTX use plain `temb`).  Subclasses can
        override this method to thread the right kwargs; the default
        path tries the common kwargs and falls back to plain
        `block(hidden, encoder_hidden, temb)`.
        """
        import torch
        tf = self.backbone
        total = self.total_blocks()
        is_first = block_lo == 0
        is_last = block_hi == total

        if is_first:
            # Try the common patch-embed attribute names.
            patch_embed = (getattr(tf, "patch_embed", None)
                           or getattr(tf, "x_embedder", None)
                           or getattr(tf, "pos_embed", None))
            if patch_embed is None:
                raise RuntimeError(f"{self.__class__.__name__}: no patch_embed found")
            hidden = patch_embed(sample) if not isinstance(patch_embed, _NoOpModule) else sample
            # Common time embedding entry points.
            time_embed = (getattr(tf, "time_text_embed", None)
                          or getattr(tf, "time_embedder", None)
                          or getattr(tf, "timestep_embedder", None))
            if time_embed is not None and not isinstance(time_embed, _NoOpModule):
                try:
                    temb = time_embed(timestep, added_cond_kwargs.get("pooled_projections"))
                except TypeError:
                    temb = time_embed(timestep)
            else:
                temb = timestep
        else:
            hidden = sample
            temb = added_cond_kwargs.get("temb", timestep)

        n_primary = len(tf.transformer_blocks)
        rope = added_cond_kwargs.get("image_rotary_emb") or added_cond_kwargs.get("freqs_cis")
        for idx in range(block_lo, min(block_hi, total)):
            kind, sub = self.index_to_module(idx)
            if kind != "primary":
                continue
            blk = tf.transformer_blocks[sub]
            try:
                hidden, encoder_hidden_states = blk(
                    hidden_states=hidden,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=rope,
                )
            except TypeError:
                # Older signature: returns just hidden.
                try:
                    hidden = blk(
                        hidden_states=hidden,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                    )
                except TypeError:
                    hidden = blk(hidden, encoder_hidden_states, temb)

        if is_last:
            if hasattr(tf, "norm_out") and not isinstance(tf.norm_out, _NoOpModule):
                try:
                    hidden = tf.norm_out(hidden, temb)
                except TypeError:
                    hidden = tf.norm_out(hidden)
            if hasattr(tf, "proj_out") and not isinstance(tf.proj_out, _NoOpModule):
                hidden = tf.proj_out(hidden)
            # Unpatchify if the transformer exposes one.
            for attr in ("_unpatchify", "unpatchify"):
                if hasattr(tf, attr):
                    fn = getattr(tf, attr)
                    if not isinstance(fn, _NoOpModule):
                        hidden = fn(hidden)
                        break
        return hidden, []


class CogVideoXAdapter(_VideoDiTAdapter):
    PIPELINE_CLASS = "CogVideoXPipeline"
    DEFAULT_FPS = 8

    @classmethod
    def load_for_role(cls, role, model, device, dtype, cache):
        from diffusers import CogVideoXPipeline
        return _load_video_pipe_for_role(CogVideoXPipeline, role, model, device, dtype, cache)

    def encode_prompt(self, prompt, *, negative_prompt=""):
        import torch
        with torch.no_grad():
            pe, npe = self.pipe.encode_prompt(
                prompt=prompt, negative_prompt=negative_prompt,
                do_classifier_free_guidance=True, num_videos_per_prompt=1,
                device=self.device, dtype=_torch_dtype(self.dtype),
            )
        embeds = torch.cat([npe, pe], dim=0)
        return EncodedPrompt(embeds=embeds, pooled=None)


class HunyuanVideoAdapter(_VideoDiTAdapter):
    PIPELINE_CLASS = "HunyuanVideoPipeline"
    DEFAULT_FPS = 24
    DOES_CFG = False  # HunyuanVideo is a distilled / guidance-free model

    @classmethod
    def load_for_role(cls, role, model, device, dtype, cache):
        from diffusers import HunyuanVideoPipeline
        return _load_video_pipe_for_role(HunyuanVideoPipeline, role, model, device, dtype, cache)

    def encode_prompt(self, prompt, *, negative_prompt=""):
        import torch
        with torch.no_grad():
            # HunyuanVideo uses an LLM + CLIP text encoder pair.
            out = self.pipe.encode_prompt(
                prompt=prompt, prompt_2=None, device=self.device,
                num_videos_per_prompt=1, dtype=_torch_dtype(self.dtype),
            )
        # encode_prompt returns (prompt_embeds, pooled_prompt_embeds,
        # prompt_attention_mask) in 0.32+.  No CFG, no negative.
        pe = out[0]
        ppe = out[1] if len(out) > 1 else None
        mask = out[2] if len(out) > 2 else None
        extras = {}
        if mask is not None:
            extras["attention_mask"] = mask
        return EncodedPrompt(embeds=pe, pooled=ppe, extras=extras)


class MochiAdapter(_VideoDiTAdapter):
    PIPELINE_CLASS = "MochiPipeline"
    DEFAULT_FPS = 30

    @classmethod
    def load_for_role(cls, role, model, device, dtype, cache):
        from diffusers import MochiPipeline
        return _load_video_pipe_for_role(MochiPipeline, role, model, device, dtype, cache)

    def encode_prompt(self, prompt, *, negative_prompt=""):
        import torch
        with torch.no_grad():
            pe, pmask, npe, nmask = self.pipe.encode_prompt(
                prompt=prompt, negative_prompt=negative_prompt,
                do_classifier_free_guidance=True, device=self.device,
                num_videos_per_prompt=1,
            )
        embeds = torch.cat([npe, pe], dim=0)
        mask = torch.cat([nmask, pmask], dim=0)
        return EncodedPrompt(embeds=embeds, pooled=None, extras={"attention_mask": mask})


class LTXVideoAdapter(_VideoDiTAdapter):
    PIPELINE_CLASS = "LTXPipeline"
    DEFAULT_FPS = 24

    @classmethod
    def load_for_role(cls, role, model, device, dtype, cache):
        from diffusers import LTXPipeline
        return _load_video_pipe_for_role(LTXPipeline, role, model, device, dtype, cache)

    def encode_prompt(self, prompt, *, negative_prompt=""):
        import torch
        with torch.no_grad():
            pe, pmask, npe, nmask = self.pipe.encode_prompt(
                prompt=prompt, negative_prompt=negative_prompt,
                do_classifier_free_guidance=True, device=self.device,
                num_videos_per_prompt=1,
            )
        embeds = torch.cat([npe, pe], dim=0)
        mask = torch.cat([nmask, pmask], dim=0)
        return EncodedPrompt(embeds=embeds, pooled=None, extras={"attention_mask": mask})


class WanAdapter(_VideoDiTAdapter):
    PIPELINE_CLASS = "WanPipeline"
    DEFAULT_FPS = 16

    @classmethod
    def load_for_role(cls, role, model, device, dtype, cache):
        # Wan landed in diffusers ≥0.32; import inside so older
        # diffusers versions don't break import-time.
        from diffusers import WanPipeline
        return _load_video_pipe_for_role(WanPipeline, role, model, device, dtype, cache)

    def encode_prompt(self, prompt, *, negative_prompt=""):
        import torch
        with torch.no_grad():
            out = self.pipe.encode_prompt(
                prompt=prompt, negative_prompt=negative_prompt,
                do_classifier_free_guidance=True, device=self.device,
                num_videos_per_prompt=1,
            )
        pe = out[0]
        npe = out[1] if len(out) > 1 else None
        if npe is not None:
            embeds = torch.cat([npe, pe], dim=0)
        else:
            embeds = pe
        return EncodedPrompt(embeds=embeds, pooled=None)


# ─── Shared video-pipe loader ──────────────────────────────────────────────


def _load_video_pipe_for_role(PipelineClass, role: str, model: str,
                              device: str, dtype: str, cache: str):
    """Generic loader used by the video-DiT adapters.

    Drops components outside `role` and moves the kept ones to `device`.
    Component attribute names vary across pipelines — we try a small
    set of common ones for each role.
    """
    import torch
    os.makedirs(cache, exist_ok=True)
    td = _torch_dtype(dtype)
    base_kwargs = dict(torch_dtype=td, cache_dir=cache, use_safetensors=True)

    # Build per-role skip kwargs by introspecting the pipeline class's
    # __init__ signature.  Any constructor arg whose name matches a
    # known component slot gets passed None when we don't need it.
    import inspect
    sig = inspect.signature(PipelineClass.__init__)
    slot_names = set(sig.parameters.keys()) - {"self"}

    text_slots = {"text_encoder", "text_encoder_2", "text_encoder_3",
                  "tokenizer", "tokenizer_2", "tokenizer_3", "image_encoder"}
    backbone_slots = {"transformer", "unet"}
    vae_slots = {"vae"}

    skip = {}
    if role == "text_encoder":
        for s in backbone_slots | vae_slots:
            if s in slot_names:
                skip[s] = None
    elif role == "unet":
        for s in text_slots | vae_slots:
            if s in slot_names:
                # Some text encoders don't accept None (no default).
                # Pass None anyway — diffusers handles this.
                skip[s] = None
    elif role == "vae":
        for s in text_slots | backbone_slots:
            if s in slot_names:
                skip[s] = None
    else:
        raise ValueError(role)

    pipe = PipelineClass.from_pretrained(model, **base_kwargs, **skip)

    if role == "text_encoder":
        for name in text_slots:
            mod = getattr(pipe, name, None)
            if mod is not None and hasattr(mod, "to"):
                mod.to(device)
    elif role == "unet":
        for name in backbone_slots:
            mod = getattr(pipe, name, None)
            if mod is not None and hasattr(mod, "to"):
                mod.to(device)
    elif role == "vae":
        for name in vae_slots:
            mod = getattr(pipe, name, None)
            if mod is not None and hasattr(mod, "to"):
                mod.to(device)
    return pipe


# ─── Adapter selection ─────────────────────────────────────────────────────


_ADAPTERS: Dict[str, type] = {}


def _register(cls):
    _ADAPTERS[cls.PIPELINE_CLASS] = cls
    return cls


for _cls in (SDXLAdapter, SD15Adapter, SD3Adapter, FluxAdapter, PixArtSigmaAdapter,
             AnimateDiffAdapter, SVDAdapter,
             CogVideoXAdapter, HunyuanVideoAdapter, MochiAdapter,
             LTXVideoAdapter, WanAdapter):
    _register(_cls)
# Aliases — diffusers ships multiple pipeline classes for the same family.
_ADAPTERS["PixArtAlphaPipeline"] = PixArtSigmaAdapter
_ADAPTERS["StableDiffusion2Pipeline"] = SD15Adapter
_ADAPTERS["AnimateDiffSDXLPipeline"] = AnimateDiffAdapter
_ADAPTERS["StableDiffusion3ImgToImgPipeline"] = SD3Adapter
_ADAPTERS["FluxImg2ImgPipeline"] = FluxAdapter


def detect_pipeline_class(model: str, cache: str, override: str = "") -> str:
    """Return the canonical diffusers pipeline class name for `model`.

    Strategy:
      1. If `override` is non-empty, trust it (operator-set escape hatch).
      2. Try the `DPP_PIPELINE` env var.
      3. Read model_index.json via DiffusionPipeline.load_config — this
         is the only authoritative source short of loading weights.
      4. Fall back to string match on the model id.
    """
    if override:
        return override
    env = os.environ.get("DPP_PIPELINE", "")
    if env:
        return env
    try:
        from diffusers import DiffusionPipeline
        cfg = DiffusionPipeline.load_config(model, cache_dir=cache)
        if isinstance(cfg, dict) and "_class_name" in cfg:
            return cfg["_class_name"]
    except Exception as e:
        log.warning("detect_pipeline_class: load_config failed: %s", e)

    # Last-resort heuristic on the model id.  Order matters — check
    # narrower keywords first so e.g. "sdxl-turbo" routes to SDXL, not SD15.
    m = model.lower()
    if "stable-video-diffusion" in m or m.startswith("svd"):
        return "StableVideoDiffusionPipeline"
    if "animatediff" in m or "animate-diff" in m:
        return "AnimateDiffPipeline"
    if "cogvideox" in m or "cogvideo" in m:
        return "CogVideoXPipeline"
    if "hunyuanvideo" in m or "hunyuan-video" in m:
        return "HunyuanVideoPipeline"
    if "mochi" in m:
        return "MochiPipeline"
    if "ltx-video" in m or "ltxvideo" in m or m.startswith("ltx/"):
        return "LTXPipeline"
    if "wan2" in m or m.startswith("wan/"):
        return "WanPipeline"
    if "flux" in m:
        return "FluxPipeline"
    if "stable-diffusion-3" in m or m.startswith("sd3") or "sd-3" in m:
        return "StableDiffusion3Pipeline"
    if "pixart" in m:
        return "PixArtSigmaPipeline"
    if "xl" in m and ("stable-diffusion" in m or "sdxl" in m):
        return "StableDiffusionXLPipeline"
    if "stable-diffusion" in m or m.startswith("sd"):
        return "StableDiffusionPipeline"
    # Default — assume SDXL since that's what the trunk built first.
    return "StableDiffusionXLPipeline"


def adapter_for_pipeline_class(name: str) -> type:
    if name not in _ADAPTERS:
        raise ValueError(f"no DPP adapter for pipeline class {name!r}; "
                         f"known: {sorted(_ADAPTERS.keys())}")
    return _ADAPTERS[name]


def make_adapter(role: str, model: str, device: str, dtype: str,
                 cache: str, override_pipeline: str = "") -> BackboneAdapter:
    """One-stop entry point: detects pipeline class, loads the right
    components, returns a ready BackboneAdapter."""
    pcn = detect_pipeline_class(model, cache, override=override_pipeline)
    cls = adapter_for_pipeline_class(pcn)
    pipe = cls.load_for_role(role, model, device, dtype, cache)
    return cls(pipe, device=device, dtype=dtype, model=model, cache=cache)


# ─── MP4 encoder shared by video adapters ──────────────────────────────────


def _encode_mp4(frames, fps: int = 24) -> bytes:
    """Encode an HxWxC uint8 array of frames as MP4 bytes (H.264).

    Tries imageio-ffmpeg first (zero-config); falls back to opencv if
    imageio isn't installed.  Both are runtime deps — if neither is
    present, raises with a clear message so the operator knows what to
    `pip install`.
    """
    # imageio path (preferred — minimal install, no system ffmpeg needed
    # since imageio-ffmpeg vendors the binary).
    try:
        import imageio
        buf = io.BytesIO()
        with imageio.get_writer(buf, format="ffmpeg", mode="I",
                                fps=fps, codec="libx264",
                                pixelformat="yuv420p", quality=8) as w:
            for fr in frames:
                w.append_data(fr)
        return buf.getvalue()
    except ImportError:
        pass
    try:
        import cv2
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(path, fourcc, fps, (w, h))
        for fr in frames:
            out.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        out.release()
        with open(path, "rb") as f:
            data = f.read()
        os.unlink(path)
        return data
    except ImportError:
        raise RuntimeError(
            "video adapter needs imageio[ffmpeg] or opencv-python — "
            "install one: pip install imageio-ffmpeg"
        )
