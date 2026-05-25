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
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional

from .wire import (
    ActvFrame,
    DTYPE_BYTES,
    DTYPE_F16,
    FLAG_DPP_CONFIG,
    FLAG_DPP_FINAL,
    FLAG_DPP_IMAGE,
    FLAG_DPP_LATENT,
    FLAG_DPP_LOOP,
    TYPE_ACT,
    TYPE_DONE,
    TYPE_ERROR,
)
from .unet_payload import UNetStepPayload, UNetTensor

log = logging.getLogger(__name__)

# CF12-G: BackboneAdapter is imported lazily inside preload() so unit
# tests that mock the Worker don't pull in diffusers at import time.


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
    # CF12-G: per-architecture adapter — set by preload(), used by all
    # role handlers for load / partition / forward.  None in unit tests
    # that exercise emit() without loading weights.
    adapter: object = None
    # Operator-set escape hatch: force a specific diffusers pipeline class
    # name (e.g. "FluxPipeline") instead of auto-detecting from
    # model_index.json.  Pulled from DPP_PIPELINE env at preload().
    pipeline_override: str = ""
    # Set by __main__._serve to a function that ships JSON progress events
    # back to the agent over the runtime socket.  None during preload and
    # in unit tests — emit() is a no-op when unset.
    progress: Optional[Callable[[dict], None]] = None

    def emit(self, event: str, req_id: int = 0, stage_idx: int = 0,
             step_idx: int = -1, total_steps: int = 0, msg: str = "") -> None:
        """Send one progress event to the agent (which forwards to the server).

        Silently dropped if no progress hook is installed (e.g. unit tests).
        """
        if self.progress is None:
            return
        try:
            self.progress({
                "kind":       "dpp_progress",
                "req_id":     int(req_id),
                "stage_idx":  int(stage_idx),
                "role":       self.role,
                "block_lo":   int(self.block_lo),
                "block_hi":   int(self.block_hi),
                "step_idx":   int(step_idx),
                "total_steps": int(total_steps),
                "event":      event,
                "msg":        msg,
            })
        except Exception as e:
            log.warning("progress emit failed: %s", e)

    # ─── load ──────────────────────────────────────────────────────────────

    def preload(self) -> None:
        """Pull just the weights this role needs.

        CF12-G: delegates to the BackboneAdapter selected for this model.
        The adapter picks the right diffusers pipeline class, loads only
        the components the role needs, and applies the block partition
        for UNet/DiT role when block_lo>=0.
        """
        from .backbones import make_adapter
        os.makedirs(self.cache, exist_ok=True)
        if not self.pipeline_override:
            self.pipeline_override = os.environ.get("DPP_PIPELINE", "")
        log.info("preload role=%s model=%s device=%s (override=%r)",
                 self.role, self.model, self.device, self.pipeline_override)
        self.adapter = make_adapter(
            role=self.role, model=self.model, device=self.device,
            dtype=self.dtype, cache=self.cache,
            override_pipeline=self.pipeline_override,
        )
        self.pipe = self.adapter.pipe
        if self.role in ("unet",) and self.block_lo >= 0 and self.block_hi >= 0:
            self.adapter.apply_partition(self.block_lo, self.block_hi)
        log.info("preload done: adapter=%s pipe=%s",
                 self.adapter.__class__.__name__, type(self.pipe).__name__)

    # ─── handle ────────────────────────────────────────────────────────────

    def handle(self, frame: ActvFrame) -> Iterable[ActvFrame]:
        try:
            # FLAG_DPP_CONFIG: agent→worker control frame carrying the
            # per-request generation config (steps, cfg_scale, w, h, seed,
            # …) as utf-8 JSON.  Cache it keyed by req_id; subsequent
            # ACT frames for that req_id consume it.  Emits no frames.
            if frame.flags & FLAG_DPP_CONFIG:
                import json as _json
                try:
                    cfg = _json.loads(frame.payload.decode("utf-8"))
                    if not isinstance(cfg, dict):
                        cfg = {}
                except Exception:
                    cfg = {}
                if not hasattr(self, "_cfg_buf") or self._cfg_buf is None:
                    self._cfg_buf = {}
                self._cfg_buf[frame.req_id] = cfg
                return
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
        """CF12-G: delegate text encoding to the adapter.

        Wire shape is unchanged for backbones with pooled embeds:
          tok_seq 0 → embeds [N,T,D]   (cond+uncond stacked)
          tok_seq 1 → pooled  [N,D]    (cond+uncond stacked)
          tok_seq 2+ → extras (any per-backbone tensors; keyed by name
                                via the dtype-bytes payload header.)

        Backbones without pooled embeds (e.g. SD 1.5, PixArt) emit only
        tok_seq 0; the downstream UNet waits for 0 only.  Backbones
        with extras (Flux text_ids, PixArt attention_mask) ship them
        on tok_seq 2,3,... — the extras dict is serialised as a small
        registry header on tok_seq=2 followed by the raw tensors.
        """
        self.emit("enter", req_id=frame.req_id, stage_idx=frame.stage,
                  msg="encode_prompt")
        prompt = frame.payload.decode("utf-8", errors="replace")
        encoded = self.adapter.encode_prompt(prompt=prompt)
        # tok_seq 0 — stacked embeds.
        payload, dims = _tensor_to_payload(encoded.embeds)
        yield ActvFrame(
            type=TYPE_ACT, req_id=frame.req_id, stage=frame.stage + 1,
            tok_seq=0, dtype=DTYPE_F16, flags=FLAG_DPP_LATENT,
            dims=dims, payload=payload,
        )
        # tok_seq 1 — pooled embeds (if present).  We always emit a
        # tok_seq=1 frame for shape symmetry — backbones without pooled
        # embeds send a zero-rank sentinel so the UNet's pair-buffer
        # logic still triggers.
        if encoded.pooled is not None:
            ppayload, pdims = _tensor_to_payload(encoded.pooled)
            yield ActvFrame(
                type=TYPE_ACT, req_id=frame.req_id, stage=frame.stage + 1,
                tok_seq=1, dtype=DTYPE_F16, flags=FLAG_DPP_LATENT,
                dims=pdims, payload=ppayload,
            )
        else:
            yield ActvFrame(
                type=TYPE_ACT, req_id=frame.req_id, stage=frame.stage + 1,
                tok_seq=1, dtype=DTYPE_F16, flags=FLAG_DPP_LATENT,
                dims=[0], payload=b"",
            )
        # tok_seq 2+ — extras.  Each extra rides as its own frame; the
        # name registry header on tok_seq=2 lists them in order.
        if encoded.extras:
            names = list(encoded.extras.keys())
            header = ("\n".join(names) + "\n").encode("utf-8")
            yield ActvFrame(
                type=TYPE_ACT, req_id=frame.req_id, stage=frame.stage + 1,
                tok_seq=2, dtype=DTYPE_BYTES, flags=FLAG_DPP_LATENT,
                payload=header,
            )
            for i, name in enumerate(names):
                t = encoded.extras[name]
                epayload, edims = _tensor_to_payload(t)
                yield ActvFrame(
                    type=TYPE_ACT, req_id=frame.req_id, stage=frame.stage + 1,
                    tok_seq=3 + i, dtype=DTYPE_F16, flags=FLAG_DPP_LATENT,
                    dims=edims, payload=epayload,
                )
        # img2img: if the per-req config carried an init_image_url, the TE
        # rig is responsible for fetching + VAE-encoding it (since this is
        # the only rig that can lazy-load a VAE for the prompt-side
        # pipeline — the UNet rig may not have the VAE at all).  The
        # encoded init latent rides on a reserved tok_seq=99 so the
        # downstream UNet stage 0 (or single-rig handler) consumes it as
        # the denoise starting point instead of pure noise.
        rcfg_te = getattr(self, "_cfg_buf", None) or {}
        rcfg_te = rcfg_te.get(frame.req_id, {}) if isinstance(rcfg_te, dict) else {}
        init_url = rcfg_te.get("init_image_url") or ""
        if init_url:
            try:
                w_te = int(rcfg_te.get("width") or os.environ.get("DPP_W", "1024"))
                h_te = int(rcfg_te.get("height") or os.environ.get("DPP_H", "1024"))
                self.emit("step", req_id=frame.req_id, stage_idx=frame.stage,
                          msg="init image encode")
                init_lat = self.adapter.encode_init_image(init_url, h=h_te, w=w_te)
                ipayload, idims = _tensor_to_payload(init_lat)
                yield ActvFrame(
                    type=TYPE_ACT, req_id=frame.req_id, stage=frame.stage + 1,
                    tok_seq=99, dtype=DTYPE_F16, flags=FLAG_DPP_LATENT,
                    dims=idims, payload=ipayload,
                )
            except Exception as e:
                log.warning("init_image encode failed (%s); falling back to pure noise", e)
        self.emit("exit", req_id=frame.req_id, stage_idx=frame.stage,
                  msg="prompt encoded")

    def _handle_unet(self, frame):
        # Multi-stage UNet (block_lo >= 0) routes through the partitioned
        # forward; single-stage (block_lo == -1) runs the full denoise
        # loop in-rig using the adapter's forward_range.  The two paths
        # share buffer keys per req_id and per-frame extras (tok_seq 2+).
        if self.block_lo >= 0:
            yield from self._handle_unet_partitioned(frame)
            return
        yield from self._handle_unet_single(frame)

    def _handle_unet_single(self, frame):
        """Single-rig denoise loop using the adapter's forward_range.

        Buffers TE frames (tok_seq 0=embeds, 1=pooled, 2=extras-header,
        3..=extras tensors) until the full set is in, then runs N steps
        and emits final latents.
        """
        import torch
        buf = getattr(self, "_te_buf", None)
        if buf is None:
            buf = {}
            self._te_buf = buf
        slot = buf.setdefault(frame.req_id, {})
        slot[frame.tok_seq] = frame
        # Detect "all frames received": tok_seq 0 + 1 always; tok_seq 2
        # (extras header) optional; if present, wait for all named
        # extras before kicking off.  tok_seq 99 (init latent for
        # img2img) — wait for it only when the per-req config signalled
        # init_image_url; pure t2i never ships a tok_seq=99.
        if 0 not in slot or 1 not in slot:
            return
        extras_header = slot.get(2)
        extras_expected = []
        if extras_header is not None:
            extras_expected = [n for n in extras_header.payload.decode("utf-8").split("\n") if n]
            for i in range(len(extras_expected)):
                if (3 + i) not in slot:
                    return  # still waiting for an extras tensor
        # Peek at the buffered config (without popping) to decide whether
        # we need to wait for the img2img init-latent frame.
        _cfg_peek = (getattr(self, "_cfg_buf", None) or {}).get(frame.req_id) or {}
        if _cfg_peek.get("init_image_url") and 99 not in slot:
            return  # still waiting for init latent
        del buf[frame.req_id]

        embeds = _payload_to_tensor(slot[0].payload, slot[0].dims, self.device, self.dtype)
        pooled = None
        if slot[1].dims and slot[1].dims[0] > 0:
            pooled = _payload_to_tensor(slot[1].payload, slot[1].dims, self.device, self.dtype)
        extras = {}
        for i, name in enumerate(extras_expected):
            f = slot[3 + i]
            extras[name] = _payload_to_tensor(f.payload, f.dims, self.device, self.dtype)

        # Per-request config from the agent's FLAG_DPP_CONFIG frame takes
        # precedence over env vars (which are the rig-wide fallback).
        # cfg_scale=0 is meaningful (SDXL-Turbo) — explicit-presence checks
        # use `in rcfg` so a literal 0 isn't fallback-clobbered.
        rcfg = getattr(self, "_cfg_buf", None) or {}
        rcfg = rcfg.pop(frame.req_id, {}) if isinstance(rcfg, dict) else {}
        steps = int(rcfg["steps"]) if rcfg.get("steps") else \
                int(os.environ.get("DPP_STEPS", "30"))
        cfg = float(rcfg["cfg_scale"]) if "cfg_scale" in rcfg and rcfg["cfg_scale"] is not None \
              else float(os.environ.get("DPP_CFG", "7.5"))
        w = int(rcfg["width"]) if rcfg.get("width") else int(os.environ.get("DPP_W", "1024"))
        h = int(rcfg["height"]) if rcfg.get("height") else int(os.environ.get("DPP_H", "1024"))
        frames = int(rcfg.get("frames") or os.environ.get("DPP_FRAMES", "16"))
        seed = int(rcfg["seed"]) if "seed" in rcfg and rcfg["seed"] is not None \
               else int(os.environ.get("DPP_SEED", "42"))

        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(steps, device=self.device)
        # img2img: tok_seq=99 carries a VAE-encoded init latent from TE.
        # When present we add noise per `strength` and skip the first
        # (1 - strength) fraction of the schedule.  Pure t2i ignores it.
        init_lat_slot = slot.get(99)
        timesteps = scheduler.timesteps
        start_idx = 0
        if init_lat_slot is not None and init_lat_slot.dims:
            try:
                strength = float(rcfg.get("strength") or 0.0)
            except Exception:
                strength = 0.0
            if strength <= 0.0 or strength > 1.0:
                strength = 0.75
            init_latent = _payload_to_tensor(
                init_lat_slot.payload, init_lat_slot.dims, self.device, self.dtype,
            )
            start_idx = max(0, min(steps - 1, int(round(steps * (1.0 - strength)))))
            gen = torch.Generator(device=self.device).manual_seed(seed)
            noise = torch.randn(init_latent.shape, generator=gen,
                                device=self.device, dtype=init_latent.dtype)
            latents = scheduler.add_noise(
                init_latent, noise, timesteps[start_idx:start_idx + 1],
            )
            timesteps = timesteps[start_idx:]
        else:
            latents = self.adapter.init_latents(h=h, w=w, frames=frames, seed=seed)

        # Build added_cond_kwargs once.  SDXL needs time_ids + text_embeds;
        # other backbones use the extras dict (Flux text_ids, PixArt
        # attention_mask, etc.).
        added = {}
        if self.adapter.__class__.__name__ == "SDXLAdapter":
            add_time_ids = torch.tensor(
                [[h, w, 0, 0, h, w]], device=self.device, dtype=_torch_dtype(self.dtype),
            )
            if self.adapter.DOES_CFG and pooled is not None:
                added = {
                    "text_embeds": pooled,
                    "time_ids":    torch.cat([add_time_ids, add_time_ids], dim=0),
                }
        # Stack encoder_hidden_states for CFG backbones; pass-through otherwise.
        ehs = embeds

        total_blocks = self.adapter.total_blocks()
        eff_steps = len(timesteps)
        self.emit("enter", req_id=frame.req_id, stage_idx=frame.stage,
                  total_steps=eff_steps, msg="single-rig denoise start")
        for step_i, t in enumerate(timesteps):
            if self.adapter.DOES_CFG:
                lat_in = torch.cat([latents, latents], dim=0)
            else:
                lat_in = latents
            lat_in = scheduler.scale_model_input(lat_in, t)
            with torch.no_grad():
                sample, _ = self.adapter.forward_range(
                    block_lo=0, block_hi=total_blocks,
                    sample=lat_in, timestep=t,
                    encoder_hidden_states=ehs,
                    added_cond_kwargs=(added if added else {**extras}),
                    residuals_in=[],
                )
            noise_pred = self.adapter.cfg_combine(sample, cfg)
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            self.emit("step", req_id=frame.req_id, stage_idx=frame.stage,
                      step_idx=step_i, total_steps=eff_steps)
        self.emit("exit", req_id=frame.req_id, stage_idx=frame.stage,
                  total_steps=eff_steps, msg="denoise complete")

        payload, dims = _tensor_to_payload(latents)
        yield ActvFrame(
            type=TYPE_ACT, req_id=frame.req_id, stage=frame.stage + 1,
            tok_seq=0, dtype=DTYPE_F16,
            flags=FLAG_DPP_LATENT | FLAG_DPP_FINAL,
            dims=dims, payload=payload,
        )

    # ─── Multi-stage UNet (CF12-B) ─────────────────────────────────────────
    #
    # Wire shape across N UNet stages:
    #
    #   Stage 0 receives stacked embeds + pooled (tok_seq 0, 1) from the
    #   TE stage exactly like the single-stage case.  It then enters the
    #   denoising loop: for each step it runs blocks [block_lo, block_hi),
    #   packs (sample, residuals) into an UNetStepPayload, and emits it
    #   downstream.  Middle stages decode, run their slice, repack, and
    #   forward.  The last stage runs its slice, applies conv_norm_out +
    #   conv_out to get noise_pred, does cfg + scheduler.step locally,
    #   and on non-final steps loops back to stage 0 via a fresh
    #   downstream emit.  This means the chain is ring-shaped per step
    #   rather than purely linear — see _next_unet_target().
    #
    # For simplicity we cache encoder_hidden_states + pooled + time_ids
    # on every stage (broadcast at request init).  ~64 KiB extra per
    # rig is cheap vs. shipping per step.

    def _handle_unet_partitioned(self, frame):
        buf = getattr(self, "_pp_buf", None)
        if buf is None:
            buf = {}
            self._pp_buf = buf
        # First frame for a req is the dpp_route activation kick.  The
        # first stage gets text-encoder embeds (tok_seq 0/1), an optional
        # extras header (tok_seq 2) plus extras tensors (tok_seq 3+).
        # Middle and last stages get UNetStepPayload-bearing frames
        # (tok_seq 0); the first stage also gets loopback frames
        # (FLAG_DPP_LOOP set) carrying the per-step noise prediction
        # from the last UNet.
        if frame.flags & (FLAG_DPP_LATENT | FLAG_DPP_LOOP) == 0:
            return
        is_first = self.block_lo == 0
        if is_first and (frame.flags & FLAG_DPP_LOOP) != 0:
            yield from self._unet_first_loopback(frame)
            return
        if is_first and (frame.flags & FLAG_DPP_LOOP) == 0 and frame.tok_seq >= 0 and frame.dtype in (DTYPE_F16, DTYPE_BYTES):
            # Buffering text-encoder frames.  Wait for the canonical
            # pair (0, 1) plus all extras the header (tok_seq=2)
            # promises.  Adapters without extras only ship tok_seq 0+1.
            slot = buf.setdefault(frame.req_id, {})
            slot[frame.tok_seq] = frame
            if 0 not in slot or 1 not in slot:
                return
            extras_header = slot.get(2)
            extras_names: List[str] = []
            if extras_header is not None:
                extras_names = [n for n in extras_header.payload.decode("utf-8").split("\n") if n]
                for i in range(len(extras_names)):
                    if (3 + i) not in slot:
                        return
            # img2img: wait for tok_seq=99 if the per-req config signalled
            # an init_image_url.
            _cfg_peek = (getattr(self, "_cfg_buf", None) or {}).get(frame.req_id) or {}
            if _cfg_peek.get("init_image_url") and 99 not in slot:
                return
            yield from self._unet_stage0_kickoff(frame.req_id, slot, extras_names, frame.stage)
            del buf[frame.req_id]
            return
        # Intermediate / last stage: payload is a UNetStepPayload.
        yield from self._unet_step_through(frame)

    def _unet_first_loopback(self, frame):
        """First UNet stage: noise_pred came back from last stage.  Apply
        scheduler.step + cfg combine to advance latents one step, then
        kick off the next forward step.
        """
        import torch
        st = getattr(self, "_pp_state", {}).get(frame.req_id)
        if st is None:
            log.error("loopback: no state for req=%d", frame.req_id)
            return
        self.emit("step", req_id=frame.req_id, stage_idx=frame.stage,
                  step_idx=st["step_idx"],
                  total_steps=len(self.pipe.scheduler.timesteps),
                  msg="loopback combine")
        # Decode noise_pred (cfg-stacked for CFG backbones; single for Flux).
        noise_pred = _payload_to_tensor(frame.payload, frame.dims, self.device, self.dtype)
        # cfg_scale is stashed at kickoff (per-req config or env fallback).
        # Use `in st` so a literal 0 (SDXL-Turbo) isn't fallback-clobbered.
        cfg = float(st["cfg"]) if "cfg" in st else float(os.environ.get("DPP_CFG", "7.5"))
        combined = self.adapter.cfg_combine(noise_pred, cfg)
        scheduler = self.pipe.scheduler
        timestep = scheduler.timesteps[st["step_idx"]]
        new_latents = scheduler.step(combined, timestep, st["latents"]).prev_sample
        st["latents"] = new_latents
        st["step_idx"] += 1
        # If we just consumed the last step, emit the final latents
        # forward to VAE (and don't loop again).
        if st["step_idx"] >= len(scheduler.timesteps):
            self.emit("exit", req_id=frame.req_id, stage_idx=frame.stage,
                      total_steps=len(scheduler.timesteps),
                      msg="denoise complete → VAE")
            payload, dims = _tensor_to_payload(new_latents)
            yield ActvFrame(
                type=TYPE_ACT, req_id=frame.req_id, stage=st["next_stage"],
                tok_seq=0, dtype=DTYPE_F16,
                flags=FLAG_DPP_LATENT | FLAG_DPP_FINAL,
                dims=dims, payload=payload,
            )
            self._pp_state.pop(frame.req_id, None)
            return
        # Not done — emit the next forward step.
        yield from self._emit_step(frame.req_id, st["next_stage"] - 1)

    def _unet_stage0_kickoff(self, req_id: int, slot: dict, extras_names: List[str], stage_idx: int):
        """First stage of the first denoise step: build latents, run our blocks.

        CF12-G: backbone-aware.  Pulls latent shape from the adapter
        (image vs video) and builds adapter-specific added_cond_kwargs
        (SDXL time_ids vs DiT pooled_projections vs Flux text_ids).
        """
        import torch
        # Per-request config from the agent's FLAG_DPP_CONFIG frame takes
        # precedence over env vars; cfg_scale=0 (SDXL-Turbo) preserved via
        # explicit-presence checks.  cfg is stashed in _pp_state so the
        # loopback (which sees only a noise_pred frame) can re-read it.
        rcfg = getattr(self, "_cfg_buf", None) or {}
        rcfg = rcfg.pop(req_id, {}) if isinstance(rcfg, dict) else {}
        steps = int(rcfg["steps"]) if rcfg.get("steps") else \
                int(os.environ.get("DPP_STEPS", "30"))
        cfg = float(rcfg["cfg_scale"]) if "cfg_scale" in rcfg and rcfg["cfg_scale"] is not None \
              else float(os.environ.get("DPP_CFG", "7.5"))
        self.emit("enter", req_id=req_id, stage_idx=stage_idx,
                  step_idx=0, total_steps=steps, msg="stage0 kickoff")

        embeds = _payload_to_tensor(slot[0].payload, slot[0].dims, self.device, self.dtype)
        pooled = None
        if slot[1].dims and slot[1].dims[0] > 0:
            pooled = _payload_to_tensor(slot[1].payload, slot[1].dims, self.device, self.dtype)
        extras = {}
        for i, name in enumerate(extras_names):
            f = slot[3 + i]
            extras[name] = _payload_to_tensor(f.payload, f.dims, self.device, self.dtype)

        w = int(rcfg["width"]) if rcfg.get("width") else int(os.environ.get("DPP_W", "1024"))
        h = int(rcfg["height"]) if rcfg.get("height") else int(os.environ.get("DPP_H", "1024"))
        frames = int(rcfg.get("frames") or os.environ.get("DPP_FRAMES", "16"))
        seed = int(rcfg["seed"]) if "seed" in rcfg and rcfg["seed"] is not None \
               else int(os.environ.get("DPP_SEED", "42"))

        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(steps, device=self.device)
        # img2img: tok_seq=99 carries a VAE-encoded init latent from TE.
        # Apply scheduler.add_noise per `strength` and start the denoise
        # loop at the matching timestep; pure t2i goes through the
        # adapter's noise sampler.  `start_idx` is stashed in _pp_state so
        # _emit_step/_unet_first_loopback can index the right timestep
        # and total-step count.
        start_idx = 0
        init_slot = slot.get(99)
        if init_slot is not None and init_slot.dims:
            try:
                strength = float(rcfg.get("strength") or 0.0)
            except Exception:
                strength = 0.0
            if strength <= 0.0 or strength > 1.0:
                strength = 0.75
            init_latent = _payload_to_tensor(
                init_slot.payload, init_slot.dims, self.device, self.dtype,
            )
            start_idx = max(0, min(steps - 1, int(round(steps * (1.0 - strength)))))
            gen = torch.Generator(device=self.device).manual_seed(seed)
            noise = torch.randn(init_latent.shape, generator=gen,
                                device=self.device, dtype=init_latent.dtype)
            latents = scheduler.add_noise(
                init_latent, noise, scheduler.timesteps[start_idx:start_idx + 1],
            )
        else:
            latents = self.adapter.init_latents(h=h, w=w, frames=frames, seed=seed)

        # Per-adapter added_cond_kwargs (cached for the duration of
        # the request).  SDXL needs time_ids + text_embeds; DiTs use
        # pooled_projections.  Extras (Flux text_ids, PixArt
        # attention_mask) ride through unchanged.
        added_cond_kwargs = dict(extras)
        if self.adapter.__class__.__name__ == "SDXLAdapter" and pooled is not None:
            add_time_ids = torch.tensor(
                [[h, w, 0, 0, h, w]], device=self.device, dtype=_torch_dtype(self.dtype),
            )
            added_cond_kwargs.update({
                "text_embeds": pooled,
                "time_ids":    torch.cat([add_time_ids, add_time_ids], dim=0),
            })
        elif pooled is not None:
            # Common name across SD3/Flux/Hunyuan.
            added_cond_kwargs["pooled_projections"] = pooled

        # Stash per-req state.  Cache embeds_full so middle/last stages
        # can rebuild added_cond_kwargs without re-shipping pooled.
        self._pp_state = getattr(self, "_pp_state", {})
        self._pp_state[req_id] = {
            "embeds_full":     embeds,
            "pooled":          pooled,
            "extras":          extras,
            "added_cond_kwargs": added_cond_kwargs,
            "latents":         latents,
            "step_idx":        start_idx,
            "start_idx":       start_idx,
            "next_stage":      stage_idx + 1,
            "cfg":             cfg,
        }
        yield from self._emit_step(req_id, stage_idx)

    def _emit_step(self, req_id: int, stage_idx: int):
        """Stage 0 prepares the step-K payload and forwards it downstream."""
        import torch
        st = self._pp_state[req_id]
        scheduler = self.pipe.scheduler
        step_idx = st["step_idx"]
        t = scheduler.timesteps[step_idx].item()
        latents = st["latents"]
        if self.adapter.DOES_CFG:
            lat_in = torch.cat([latents, latents], dim=0)
        else:
            lat_in = latents
        lat_in = scheduler.scale_model_input(lat_in, scheduler.timesteps[step_idx])
        sample, residuals = self.adapter.forward_range(
            block_lo=self.block_lo, block_hi=self.block_hi,
            sample=lat_in,
            timestep=scheduler.timesteps[step_idx],
            encoder_hidden_states=st["embeds_full"],
            added_cond_kwargs=st["added_cond_kwargs"],
            residuals_in=[],
        )
        payload = self._pack_step_payload(sample, residuals, step_idx, t,
                                          is_final_step=False)
        yield ActvFrame(
            type=TYPE_ACT, req_id=req_id, stage=stage_idx + 1,
            tok_seq=0, dtype=DTYPE_BYTES,
            flags=FLAG_DPP_LATENT,
            payload=payload,
        )

    def _unet_step_through(self, frame):
        """Middle / last stage processes one step's hidden state."""
        import torch
        try:
            step = UNetStepPayload.decode(frame.payload)
        except ValueError as e:
            log.error("unet step decode: %s", e)
            return
        total = self.adapter.total_blocks()
        is_last = self.block_hi == total
        self.emit("step", req_id=frame.req_id, stage_idx=frame.stage,
                  step_idx=step.step_idx,
                  msg=("last" if is_last else "mid"))
        st = self._pp_state.setdefault(frame.req_id, {}) if hasattr(self, "_pp_state") else None
        if st is None:
            self._pp_state = {}
            st = self._pp_state.setdefault(frame.req_id, {})
        # Materialize tensors.
        sample = self._tensor_from_upld(step.sample)
        residuals_in = [self._tensor_from_upld(r) for r in step.residuals]
        # Run our blocks via the backbone adapter.
        sample, residuals_out = self.adapter.forward_range(
            block_lo=self.block_lo, block_hi=self.block_hi,
            sample=sample,
            timestep=torch.tensor(step.timestep, device=self.device,
                                  dtype=_torch_dtype(self.dtype)),
            encoder_hidden_states=st.get("embeds_full"),
            added_cond_kwargs=st.get("added_cond_kwargs"),
            residuals_in=residuals_in,
        )
        if not is_last:
            payload = self._pack_step_payload(sample, residuals_out,
                                              step.step_idx, step.timestep,
                                              is_final_step=False)
            yield ActvFrame(
                type=TYPE_ACT, req_id=frame.req_id, stage=frame.stage + 1,
                tok_seq=0, dtype=DTYPE_BYTES,
                flags=FLAG_DPP_LATENT,
                payload=payload,
            )
            return
        # Last stage: sample IS the cfg-stacked noise prediction
        # (shape [2, C, H/8, W/8]).  Ship it back to the first UNet
        # stage via the loopback flag.  The first stage owns the
        # scheduler state and applies cfg combine + scheduler.step
        # locally (see _unet_first_loopback).  The dispatcher routes
        # FLAG_DPP_LOOP frames to dppPeer.loopbackAgent.  When the first
        # stage completes the final step, it emits FLAG_DPP_FINAL
        # forward to VAE — we never reach is_final on the last UNet
        # stage's emit path.
        payload_bytes, dims = _tensor_to_payload(sample)
        yield ActvFrame(
            type=TYPE_ACT, req_id=frame.req_id, stage=frame.stage + 1,
            tok_seq=0, dtype=DTYPE_F16,
            flags=FLAG_DPP_LATENT | FLAG_DPP_LOOP,
            dims=dims, payload=payload_bytes,
        )

    def _pack_step_payload(self, sample, residuals, step_idx: int, timestep: float, is_final_step: bool) -> bytes:
        sample_bytes, sample_dims = _tensor_to_payload(sample)
        upld_residuals = []
        for r in residuals:
            rb, rd = _tensor_to_payload(r)
            upld_residuals.append(UNetTensor(dims=tuple(rd), data=rb))
        upld = UNetStepPayload(
            sample=UNetTensor(dims=tuple(sample_dims), data=sample_bytes),
            residuals=upld_residuals,
            step_idx=step_idx,
            timestep=float(timestep),
            is_final_step=is_final_step,
        )
        return upld.encode()

    def _tensor_from_upld(self, ut: UNetTensor):
        import numpy as np
        import torch
        arr = np.frombuffer(ut.data, dtype=np.float16).reshape(ut.dims)
        return torch.from_numpy(arr.copy()).to(device=self.device, dtype=_torch_dtype(self.dtype))

    def _handle_vae(self, frame):
        """VAE decode → bytes via adapter.

        Image adapters return PNG; video adapters return MP4 (H.264).
        The FLAG_DPP_IMAGE flag is kept regardless — downstream consumers
        (ws.go, comfy_adapter.cpp) treat it as "image-or-video payload
        complete" and use Content-Type sniffing for the actual mime.
        """
        if frame.flags & FLAG_DPP_FINAL == 0:
            return
        self.emit("enter", req_id=frame.req_id, stage_idx=frame.stage,
                  msg="vae decode")
        latents = _payload_to_tensor(frame.payload, frame.dims, self.device, self.dtype)
        out_bytes = self.adapter.vae_decode(latents)
        yield ActvFrame(
            type=TYPE_ACT, req_id=frame.req_id, stage=frame.stage + 1,
            tok_seq=0, dtype=DTYPE_BYTES,
            flags=FLAG_DPP_IMAGE,
            payload=out_bytes,
        )
        yield ActvFrame(type=TYPE_DONE, req_id=frame.req_id, stage=frame.stage + 1)
        self.emit("exit", req_id=frame.req_id, stage_idx=frame.stage,
                  msg=("video emitted" if self.adapter.IS_VIDEO else "image emitted"))
