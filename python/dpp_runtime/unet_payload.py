"""Inter-stage UNet hidden-state codec (CF12-C).

When the UNet is split across N pipeline-parallel rigs, each forward
pass per denoise step ships the running activation across the chain.
The payload carries:

  - sample tensor                — the current activation (fp16, NCHW)
  - residuals stack              — k residual tensors emitted by down
                                   blocks earlier in the chain, to be
                                   consumed by the matching up blocks
  - step_idx                     — which scheduler step we're on
                                   (last stage uses it to know when
                                    to emit FINAL)
  - timestep                     — diffuser t value for this step,
                                   sent so middle stages don't need
                                   to re-derive it from step_idx

The encoder_hidden_states / pooled / time_ids tensors are NOT in this
payload — they're broadcast once at request setup as separate ACTV
frames (tok_seq 1, 2, 3) before stepping begins, so every stage caches
them and avoids re-shipping ~64 KiB per step.

Header layout (big-endian):

    magic         u32   = b"UPLD"
    ver           u8    = 1
    flags         u8    bit 0 = is_final_step (last stage should emit
                                 FLAG_DPP_FINAL after this step)
    n_residuals   u16
    step_idx      u32
    timestep_f32  f32
    sample_rank   u8
    sample_dims   u32 × sample_rank
    sample_nbytes u32
    per residual:
      res_rank   u8
      res_dims   u32 × res_rank
      res_nbytes u32
    payload:
      residuals (in stack order, bottom-up) concatenated
      sample bytes

Sample dtype is always fp16; residuals match.  fp32 paths convert
on encode/decode (cheap — done once per residual per step).

This codec is intentionally not part of wire.py — ACTV is the
*transport*; this is the *payload format* for a specific frame type
(UNet hidden state).  Keeping them separate means future payload
schemas (e.g. video latents with temporal residuals) plug in without
muddying the activation framing.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List, Tuple

MAGIC = 0x55504C44  # b"UPLD"
VER = 1

FLAG_FINAL_STEP = 0x01


@dataclass
class UNetTensor:
    """One residual or sample tensor in the payload.

    dims is row-major (NCHW for sample, same for residuals — diffusers
    always emits residuals as NCHW).  data is raw fp16 little-endian
    host bytes (matches what _tensor_to_payload writes).
    """
    dims: Tuple[int, ...]
    data: bytes


@dataclass
class UNetStepPayload:
    sample: UNetTensor
    residuals: List[UNetTensor] = field(default_factory=list)
    step_idx: int = 0
    timestep: float = 0.0
    is_final_step: bool = False

    def encode(self) -> bytes:
        flags = FLAG_FINAL_STEP if self.is_final_step else 0
        out = bytearray()
        out += struct.pack(
            ">IBBHIf",
            MAGIC, VER, flags,
            len(self.residuals),
            self.step_idx,
            self.timestep,
        )
        out += _encode_tensor_header(self.sample)
        for r in self.residuals:
            out += _encode_tensor_header(r)
        # Residuals in stack order, then sample.
        for r in self.residuals:
            out += r.data
        out += self.sample.data
        return bytes(out)

    @classmethod
    def decode(cls, buf: bytes) -> "UNetStepPayload":
        if len(buf) < 16:
            raise ValueError("upld: short header")
        (magic, ver, flags, n_res, step_idx, timestep) = struct.unpack(">IBBHIf", buf[:16])
        if magic != MAGIC:
            raise ValueError(f"upld: bad magic {magic:08x}")
        if ver != VER:
            raise ValueError(f"upld: bad ver {ver}")
        off = 16
        sample_dims, sample_nbytes, off = _decode_tensor_header(buf, off)
        residuals_meta: List[Tuple[Tuple[int, ...], int]] = []
        for _ in range(n_res):
            dims, nbytes, off = _decode_tensor_header(buf, off)
            residuals_meta.append((dims, nbytes))
        residuals: List[UNetTensor] = []
        for dims, nbytes in residuals_meta:
            if off + nbytes > len(buf):
                raise ValueError("upld: truncated residual")
            residuals.append(UNetTensor(dims=dims, data=bytes(buf[off:off + nbytes])))
            off += nbytes
        if off + sample_nbytes > len(buf):
            raise ValueError("upld: truncated sample")
        sample = UNetTensor(dims=sample_dims, data=bytes(buf[off:off + sample_nbytes]))
        return cls(
            sample=sample,
            residuals=residuals,
            step_idx=step_idx,
            timestep=timestep,
            is_final_step=bool(flags & FLAG_FINAL_STEP),
        )


def _encode_tensor_header(t: UNetTensor) -> bytes:
    rank = len(t.dims)
    out = struct.pack(">B", rank)
    out += b"".join(struct.pack(">I", d) for d in t.dims)
    out += struct.pack(">I", len(t.data))
    return out


def _decode_tensor_header(buf: bytes, off: int) -> Tuple[Tuple[int, ...], int, int]:
    if off + 1 > len(buf):
        raise ValueError("upld: short tensor header")
    rank = buf[off]
    off += 1
    if off + 4 * rank + 4 > len(buf):
        raise ValueError("upld: truncated tensor dims")
    dims = struct.unpack(">" + "I" * rank, buf[off:off + 4 * rank])
    off += 4 * rank
    (nbytes,) = struct.unpack(">I", buf[off:off + 4])
    off += 4
    return dims, nbytes, off
