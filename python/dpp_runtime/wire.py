"""ACTV frame codec — mirrors server/activation.go.

Big-endian framing.  Header is 20 bytes fixed + 4*rank dim bytes + 4-byte
payload_len; payload follows.

Magic = b"ACTV", ver = 1.

Type codes:  1=act, 2=token, 3=done, 4=error
DType codes: 0=f32, 1=f16, 2=bf16, 3=q8_0, 4=bytes
Flag bits:   0x01=is_prompt, 0x02=kv_append, 0x04=end_of_prompt,
             0x08=dpp_latent, 0x10=dpp_final, 0x20=dpp_image
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List

MAGIC = 0x41435456  # b"ACTV"
VER = 1

TYPE_ACT = 1
TYPE_TOKEN = 2
TYPE_DONE = 3
TYPE_ERROR = 4

DTYPE_F32 = 0
DTYPE_F16 = 1
DTYPE_BF16 = 2
DTYPE_Q8_0 = 3
DTYPE_BYTES = 4

FLAG_IS_PROMPT = 0x01
FLAG_KV_APPEND = 0x02
FLAG_END_OF_PROMPT = 0x04
FLAG_DPP_LATENT = 0x08
FLAG_DPP_FINAL = 0x10
FLAG_DPP_IMAGE = 0x20


@dataclass
class ActvFrame:
    type: int = TYPE_ACT
    req_id: int = 0
    stage: int = 0
    tok_seq: int = 0
    dtype: int = DTYPE_BYTES
    flags: int = 0
    dims: List[int] = field(default_factory=list)
    payload: bytes = b""

    def encode(self) -> bytes:
        rank = len(self.dims)
        hdr = struct.pack(
            ">IBBHHIBBBBxx",
            MAGIC, VER, self.type,
            self.req_id, self.stage,
            self.tok_seq,
            self.dtype, rank, self.flags, 0,
        )
        dim_bytes = b"".join(struct.pack(">I", d) for d in self.dims)
        plen = struct.pack(">I", len(self.payload))
        return hdr + dim_bytes + plen + self.payload

    @classmethod
    def decode(cls, buf: bytes) -> "ActvFrame":
        if len(buf) < 20:
            raise ValueError("actv: short header")
        (magic, ver, typ, req_id, stage, tok_seq, dtype, rank, flags, _rsv) = struct.unpack(">IBBHHIBBBBxx", buf[:20])
        if magic != MAGIC:
            raise ValueError(f"actv: bad magic {magic:08x}")
        if ver != VER:
            raise ValueError(f"actv: bad ver {ver}")
        need = 20 + 4 * rank + 4
        if len(buf) < need:
            raise ValueError("actv: truncated header")
        dims = list(struct.unpack(">" + "I" * rank, buf[20:20 + 4 * rank])) if rank else []
        (plen,) = struct.unpack(">I", buf[20 + 4 * rank:24 + 4 * rank])
        if len(buf) < need + plen:
            raise ValueError("actv: truncated payload")
        payload = bytes(buf[need:need + plen])
        return cls(type=typ, req_id=req_id, stage=stage, tok_seq=tok_seq,
                   dtype=dtype, flags=flags, dims=dims, payload=payload)


def recv_frame(sock) -> ActvFrame:
    """Read one length-prefixed ACTV frame from a socket.

    Frames on the local agent↔runtime socket are framed as `u32_be length
    || ACTV-bytes` so the runtime can read exactly one frame per call.
    """
    hdr = _readn(sock, 4)
    (n,) = struct.unpack(">I", hdr)
    if n > (64 << 20):
        raise ValueError(f"actv: oversize frame {n}")
    body = _readn(sock, n)
    return ActvFrame.decode(body)


def send_frame(sock, f: ActvFrame) -> None:
    buf = f.encode()
    sock.sendall(struct.pack(">I", len(buf)) + buf)


def _readn(sock, n: int) -> bytes:
    out = bytearray()
    while len(out) < n:
        chunk = sock.recv(n - len(out))
        if not chunk:
            raise ConnectionError("dpp socket closed mid-frame")
        out.extend(chunk)
    return bytes(out)
