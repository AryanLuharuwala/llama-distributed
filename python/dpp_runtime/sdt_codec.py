"""SDT / SDCD codec — python counterpart of include/sdt_codec.h.

Lets a python dpp_runtime worker hop into/out of sd.cpp role-workers
mid-pipeline.  Wire format is identical (big-endian header, LE payload)
so a sd.cpp rig and a python rig agree byte-for-byte.

UPLD is already covered by `dpp_runtime.unet_payload.UNetStepPayload`
— sd.cpp's UPLD encoder/decoder is wire-compatible with it.

This module has no torch dependency; tensors arrive as raw bytes +
shape + dtype and the caller is responsible for reshaping into a
framework-native tensor.  Keeping torch out of the codec means
test_sdt_codec runs in a vanilla python env.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ─── dtype tags (mirror C++ SdtDType) ─────────────────────────────────────
DT_F32  = 0
DT_F16  = 1
DT_BF16 = 2
DT_I32  = 3
DT_I64  = 4
DT_U8   = 5

DTYPE_BYTES = {DT_F32: 4, DT_F16: 2, DT_BF16: 2, DT_I32: 4, DT_I64: 8, DT_U8: 1}
DTYPE_NAME  = {DT_F32: "f32", DT_F16: "f16", DT_BF16: "bf16",
               DT_I32: "i32", DT_I64: "i64", DT_U8: "u8"}
NAME_DTYPE  = {v: k for k, v in DTYPE_NAME.items()}

SDT_MAGIC  = 0x53445431  # "SDT1"
SDT_VER    = 1
SDCD_MAGIC = 0x53444344  # "SDCD"
SDCD_VER   = 1


# ─── SDT ───────────────────────────────────────────────────────────────────

@dataclass
class SdtTensor:
    dtype: int                       # one of DT_*
    dims: Tuple[int, ...]            # row-major
    data: bytes                      # LE host bytes; len == prod(dims)*DTYPE_BYTES[dtype]

    def expected_nbytes(self) -> int:
        n = DTYPE_BYTES[self.dtype]
        for d in self.dims:
            n *= int(d)
        return n


def sdt_encode(t: SdtTensor) -> bytes:
    if len(t.dims) > 8:
        raise ValueError("sdt: rank > 8")
    want = t.expected_nbytes()
    if len(t.data) != want:
        raise ValueError(f"sdt: data {len(t.data)} != dims*dtype {want}")
    rank = len(t.dims)
    out = bytearray()
    out += struct.pack(">IBBBB", SDT_MAGIC, SDT_VER, t.dtype, rank, 0)
    for d in t.dims:
        out += struct.pack(">I", int(d))
    out += struct.pack(">I", want)
    out += t.data
    return bytes(out)


def sdt_decode(buf: bytes) -> SdtTensor:
    if len(buf) < 8:
        raise ValueError("sdt: short header")
    magic, ver, dtype, rank, _ = struct.unpack(">IBBBB", buf[:8])
    if magic != SDT_MAGIC:
        raise ValueError(f"sdt: bad magic {magic:08x}")
    if ver != SDT_VER:
        raise ValueError(f"sdt: bad ver {ver}")
    if rank > 8:
        raise ValueError("sdt: rank > 8")
    off = 8
    if len(buf) < off + 4 * rank + 4:
        raise ValueError("sdt: truncated dims")
    dims = struct.unpack(">" + "I" * rank, buf[off:off + 4 * rank])
    off += 4 * rank
    (nbytes,) = struct.unpack(">I", buf[off:off + 4])
    off += 4
    t = SdtTensor(dtype=dtype, dims=tuple(dims), data=b"")
    want = t.expected_nbytes()
    if want != nbytes:
        raise ValueError(f"sdt: nbytes {nbytes} != dims*dtype {want}")
    if len(buf) < off + nbytes:
        raise ValueError("sdt: truncated payload")
    t.data = bytes(buf[off:off + nbytes])
    return t


# ─── SDCD ──────────────────────────────────────────────────────────────────

@dataclass
class SdcdFrame:
    kv: Dict[str, str] = field(default_factory=dict)
    tensors: Dict[str, SdtTensor] = field(default_factory=dict)
    # Preserve order for byte-roundtrip equality with the C++ encoder.
    _order: List[str] = field(default_factory=list)

    def set(self, name: str, t: SdtTensor) -> None:
        if name not in self.tensors:
            self._order.append(name)
        self.tensors[name] = t

    def get(self, name: str) -> Optional[SdtTensor]:
        return self.tensors.get(name)


def sdcd_encode(f: SdcdFrame) -> bytes:
    if len(f.kv) > 0xFFFF:
        raise ValueError("sdcd: too many kv")
    if len(f.tensors) > 0xFFFF:
        raise ValueError("sdcd: too many tensors")

    out = bytearray()
    out += struct.pack(">IBBHHH",
                       SDCD_MAGIC, SDCD_VER, 0,
                       len(f.kv), len(f.tensors), 0)

    for k, v in f.kv.items():
        kb = k.encode("utf-8")
        vb = v.encode("utf-8")
        if len(kb) > 0xFFFF or len(vb) > 0xFFFF:
            raise ValueError("sdcd: kv string > 64 KiB")
        out += struct.pack(">HH", len(kb), len(vb))
        out += kb
        out += vb

    order = f._order if f._order else list(f.tensors.keys())
    for name in order:
        nb = name.encode("utf-8")
        if len(nb) > 0xFFFF:
            raise ValueError("sdcd: name > 64 KiB")
        out += struct.pack(">H", len(nb))
        out += nb
        out += sdt_encode(f.tensors[name])
    return bytes(out)


def sdcd_decode(buf: bytes) -> SdcdFrame:
    if len(buf) < 12:
        raise ValueError("sdcd: short header")
    magic, ver, _flags, n_kv, n_t, _reserved = struct.unpack(">IBBHHH", buf[:12])
    if magic != SDCD_MAGIC:
        raise ValueError(f"sdcd: bad magic {magic:08x}")
    if ver != SDCD_VER:
        raise ValueError(f"sdcd: bad ver {ver}")
    off = 12

    frame = SdcdFrame()
    for _ in range(n_kv):
        if len(buf) < off + 4:
            raise ValueError("sdcd: kv hdr")
        klen, vlen = struct.unpack(">HH", buf[off:off + 4])
        off += 4
        if len(buf) < off + klen + vlen:
            raise ValueError("sdcd: kv body")
        k = buf[off:off + klen].decode("utf-8")
        off += klen
        v = buf[off:off + vlen].decode("utf-8")
        off += vlen
        frame.kv[k] = v

    for _ in range(n_t):
        if len(buf) < off + 2:
            raise ValueError("sdcd: tname len")
        (nlen,) = struct.unpack(">H", buf[off:off + 2])
        off += 2
        if len(buf) < off + nlen:
            raise ValueError("sdcd: tname body")
        name = buf[off:off + nlen].decode("utf-8")
        off += nlen

        # Parse inner SDT to know how far to advance.
        if len(buf) < off + 8:
            raise ValueError("sdcd: nested sdt short")
        _m, _v, _dtype, rank, _r = struct.unpack(">IBBBB", buf[off:off + 8])
        inner = off + 8 + 4 * rank
        if len(buf) < inner + 4:
            raise ValueError("sdcd: nested sdt nbytes")
        (nbytes,) = struct.unpack(">I", buf[inner:inner + 4])
        end = inner + 4 + nbytes
        if len(buf) < end:
            raise ValueError("sdcd: nested sdt body")
        t = sdt_decode(buf[off:end])
        frame.set(name, t)
        off = end

    return frame


def partition_blocks(total, stages):
    """Even contiguous partition of [0, total) into `stages` ranges, front-
    loading the remainder. Single source of truth shared by the N-way split
    tests/validators (mirrors the Go partitionUNetBlocks + the C++ schedule).
    Returns [(lo, hi), ...] or [] for invalid inputs."""
    if stages < 1 or total < stages:
        return []
    base, extra, cursor, out = total // stages, total % stages, 0, []
    for i in range(stages):
        size = base + (1 if i < extra else 0)
        out.append((cursor, cursor + size))
        cursor += size
    return out
