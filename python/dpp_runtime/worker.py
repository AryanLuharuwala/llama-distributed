"""Per-role worker: owns one slice of a diffusion model.

Three roles, three implementations.  All share the same handle(frame) ->
iterator-of-frames contract so the dispatcher in __main__.py stays role-
agnostic.

The handle() contract:

  - role=text_encoder:
      INPUT  : ActvFrame{type=ACT, dtype=BYTES, payload=prompt utf8,
                         flags=is_prompt|end_of_prompt|dpp_latent}
      OUTPUT : ActvFrame{type=ACT, dtype=F16, dims=[2,T,D],
                         payload=cond+uncond embeds, flags=dpp_latent}

  - role=unet:
      INPUT  : ActvFrame{type=ACT, dtype=F16, dims=[2,T,D], ...embeds...}
      OUTPUT : ActvFrame{type=ACT, dtype=F16, dims=[1,4,H/8,W/8],
                         flags=dpp_latent|dpp_final}
      (Runs the full N-step denoising loop internally before emitting.)

  - role=vae:
      INPUT  : ActvFrame{type=ACT, ...latent..., flags=dpp_final}
      OUTPUT : ActvFrame{type=ACT, dtype=BYTES, payload=PNG bytes,
                         flags=dpp_image}
              ActvFrame{type=DONE}

Frames carry a `req_id` so multiple concurrent generations interleave
safely on one runtime process.
"""

from __future__ import annotations

import io
import logging
import os
import struct
from dataclasses import dataclass
from typing import Iterable, Optional

from .wire import (
    ActvFrame,
    DTYPE_BYTES,
    DTYPE_F16,
    FLAG_DPP_FINAL,
    FLAG_DPP_IMAGE,
    FLAG_DPP_LATENT,
    TYPE_ACT,
    TYPE_DONE,
    TYPE_ERROR,
)

log = logging.getLogger(__name__)


def _np_dtype(dtype: int):
    import numpy as np
    return {0: np.float32, 1: np.float16, 2: None, 3: None, 4: None}[dtype] or np.float16


def _torch_dtype(dtype_str: str):
    import torch
    return {"fp16": torch.float16, "f16": torch.float16,
            "bf16": torch.bfloat16, "fp32": torch.float32}[dtype_str]


def _tensor_to_payload(t):
    """Move tensor to CPU, contig, fp16 little-endian-on-the-wire (we encode
    big-endian dims in the header; the payload itself is host-native bytes).
    """
    import torch
    t = t.detach().to(torch.float16).contiguous().cpu().numpy()
    return t.tobytes(), list(t.shape)


def _payload_to_tensor(payload: bytes, dims, device, dtype_str):
    import numpy as np
    import torch
    arr = np.frombuffer(payload, dtype=np.float16).reshape(dims)
    return torch.from_numpy(arr.copy()).to(device=device, dtype=_torch_dtype(dtype_str))


@dataclass
class Worker:
    role: str
    model: str
    block_lo: int
    block_hi: int
    device: str
    dtype: str
    cache: str
    pipe: object = None  # the loaded diffusers pipeline (or slice thereof)

    # ─── load ──────────────────────────────────────────────────────────────

    def preload(self) -> None:
        """Pull just the weights this role needs."""
        os.makedirs(self.cache, exist_ok=True)
        log.info("preload role=%s model=%s device=%s", self.role, self.model, self.device)
        if self.role == "text_encoder":
            self._load_text_encoder()
        elif self.role == "unet":
            self._load_unet()
        elif self.role == "vae":
            self._load_vae()
        else:
            raise ValueError(f"unknown role {self.role}")

    def _load_text_encoder(self):
        from diffusers import StableDiffusionXLPipeline
        import torch
        # We only need the text encoders + tokenizers — load the full pipe
        # with components=None for the rest to save VRAM.
        pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model,
            torch_dtype=_torch_dtype(self.dtype),
            cache_dir=self.cache,
            use_safetensors=True,
            unet=None,
            vae=None,
        )
        # Move the encoders to GPU; tokenizers are CPU-only.
        pipe.text_encoder.to(self.device)
        pipe.text_encoder_2.to(self.device)
        self.pipe = pipe

    def _load_unet(self):
        from diffusers import StableDiffusionXLPipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model,
            torch_dtype=_torch_dtype(self.dtype),
            cache_dir=self.cache,
            use_safetensors=True,
            text_encoder=None,
            text_encoder_2=None,
            vae=None,
        )
        pipe.unet.to(self.device)
        # Pre-warm the scheduler.
        self.pipe = pipe

    def _load_vae(self):
        from diffusers import StableDiffusionXLPipeline
        import torch
        # SDXL's base VAE produces NaN in fp16 — well-known issue. Always
        # load and run the VAE in fp32, regardless of the pipeline dtype the
        # rest of the stages use.
        pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model,
            torch_dtype=torch.float32,
            cache_dir=self.cache,
            use_safetensors=True,
            text_encoder=None,
            text_encoder_2=None,
            unet=None,
        )
        pipe.vae.to(self.device, dtype=torch.float32)
        self.pipe = pipe

    # ─── handle ────────────────────────────────────────────────────────────

    def handle(self, frame: ActvFrame) -> Iterable[ActvFrame]:
        try:
            if self.role == "text_encoder":
                yield from self._handle_te(frame)
            elif self.role == "unet":
                yield from self._handle_unet(frame)
            elif self.role == "vae":
                yield from self._handle_vae(frame)
        except Exception as e:
            log.exception("handle failed: %s", e)
            yield ActvFrame(type=TYPE_ERROR, req_id=frame.req_id,
                            stage=frame.stage, dtype=DTYPE_BYTES,
                            payload=str(e).encode())

    def _handle_te(self, frame):
        prompt = frame.payload.decode("utf-8", errors="replace")
        # Use the pipeline's encode_prompt — gives us (prompt_embeds,
        # negative_prompt_embeds, pooled, negative_pooled) sized for SDXL.
        import torch
        with torch.no_grad():
            pe, npe, ppe, nppe = self.pipe.encode_prompt(
                prompt=prompt, prompt_2=None,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt="", negative_prompt_2="",
            )
        # Stack cond+uncond on dim 0 → shape [2,T,D].  Pooled embeds also
        # go in a separate small frame.
        stacked = torch.cat([npe, pe], dim=0)
        payload, dims = _tensor_to_payload(stacked)
        yield ActvFrame(
            type=TYPE_ACT, req_id=frame.req_id, stage=frame.stage + 1,
            tok_seq=0, dtype=DTYPE_F16,
            flags=FLAG_DPP_LATENT,
            dims=dims, payload=payload,
        )
        # Pooled embeds in a follow-up tok_seq=1 frame.
        pooled = torch.cat([nppe, ppe], dim=0)
        ppayload, pdims = _tensor_to_payload(pooled)
        yield ActvFrame(
            type=TYPE_ACT, req_id=frame.req_id, stage=frame.stage + 1,
            tok_seq=1, dtype=DTYPE_F16,
            flags=FLAG_DPP_LATENT,
            dims=pdims, payload=ppayload,
        )

    def _handle_unet(self, frame):
        # We expect two incoming frames per request (tok_seq 0 = stacked
        # embeds, tok_seq 1 = pooled).  Buffer per req_id.
        buf = getattr(self, "_te_buf", None)
        if buf is None:
            buf = {}
            self._te_buf = buf
        slot = buf.setdefault(frame.req_id, {})
        slot[frame.tok_seq] = frame
        if 0 not in slot or 1 not in slot:
            return  # wait for both
        embeds = _payload_to_tensor(slot[0].payload, slot[0].dims, self.device, self.dtype)
        pooled = _payload_to_tensor(slot[1].payload, slot[1].dims, self.device, self.dtype)
        del buf[frame.req_id]

        # Pull config out of env (the agent should re-emit it per req — for
        # now use defaults).  TODO: route config through a control frame.
        steps = int(os.environ.get("DPP_STEPS", "30"))
        cfg = float(os.environ.get("DPP_CFG", "7.5"))
        w = int(os.environ.get("DPP_W", "1024"))
        h = int(os.environ.get("DPP_H", "1024"))
        seed = int(os.environ.get("DPP_SEED", "42"))

        import torch
        npe, pe = embeds[0:1], embeds[1:2]
        nppe, ppe = pooled[0:1], pooled[1:2]

        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(steps, device=self.device)

        gen = torch.Generator(device=self.device).manual_seed(seed)
        latents = torch.randn(
            (1, self.pipe.unet.config.in_channels, h // 8, w // 8),
            generator=gen, device=self.device, dtype=_torch_dtype(self.dtype),
        ) * scheduler.init_noise_sigma

        # SDXL added_cond_kwargs: time_ids + text_embeds (pooled).
        add_time_ids = torch.tensor(
            [[h, w, 0, 0, h, w]], device=self.device, dtype=_torch_dtype(self.dtype),
        )

        for t in scheduler.timesteps:
            lat_in = torch.cat([latents, latents], dim=0)
            lat_in = scheduler.scale_model_input(lat_in, t)
            added = {
                "text_embeds": torch.cat([nppe, ppe], dim=0),
                "time_ids":    torch.cat([add_time_ids, add_time_ids], dim=0),
            }
            with torch.no_grad():
                noise_pred = self.pipe.unet(
                    lat_in, t,
                    encoder_hidden_states=torch.cat([npe, pe], dim=0),
                    added_cond_kwargs=added,
                ).sample
            npr, pr = noise_pred.chunk(2)
            noise_pred = npr + cfg * (pr - npr)
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        payload, dims = _tensor_to_payload(latents)
        yield ActvFrame(
            type=TYPE_ACT, req_id=frame.req_id, stage=frame.stage + 1,
            tok_seq=0, dtype=DTYPE_F16,
            flags=FLAG_DPP_LATENT | FLAG_DPP_FINAL,
            dims=dims, payload=payload,
        )

    def _handle_vae(self, frame):
        if frame.flags & FLAG_DPP_FINAL == 0:
            return
        import torch
        from PIL import Image
        # Always decode in fp32 — SDXL base VAE in fp16 returns NaN (cast
        # warning + black image). Latents came in as fp16 from the unet but
        # converting to fp32 here is cheap and stable.
        latents = _payload_to_tensor(frame.payload, frame.dims, self.device, self.dtype)
        latents = latents.to(torch.float32) / self.pipe.vae.config.scaling_factor
        with torch.no_grad():
            image = self.pipe.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = (image.cpu().permute(0, 2, 3, 1).float().numpy() * 255).round().astype("uint8")
        buf = io.BytesIO()
        Image.fromarray(image[0]).save(buf, format="PNG")
        png = buf.getvalue()
        yield ActvFrame(
            type=TYPE_ACT, req_id=frame.req_id, stage=frame.stage + 1,
            tok_seq=0, dtype=DTYPE_BYTES,
            flags=FLAG_DPP_IMAGE,
            payload=png,
        )
        yield ActvFrame(type=TYPE_DONE, req_id=frame.req_id, stage=frame.stage + 1)
