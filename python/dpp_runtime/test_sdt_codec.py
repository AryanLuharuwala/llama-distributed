"""Round-trip tests for SDT / SDCD wire codecs (matches C++ sdt_codec)."""

from __future__ import annotations

import struct

import pytest

from dpp_runtime import sdt_codec as sc


def _make_tensor(dtype: int, dims):
    nbytes = sc.DTYPE_BYTES[dtype]
    for d in dims:
        nbytes *= d
    data = bytes((i * 31 + 7) & 0xFF for i in range(nbytes))
    return sc.SdtTensor(dtype=dtype, dims=tuple(dims), data=data)


def test_sdt_roundtrip_f16_nchw():
    t = _make_tensor(sc.DT_F16, [1, 4, 64, 64])  # SDXL-ish latent
    enc = sc.sdt_encode(t)
    dec = sc.sdt_decode(enc)
    assert dec.dtype == sc.DT_F16
    assert dec.dims == (1, 4, 64, 64)
    assert dec.data == t.data


def test_sdt_roundtrip_u8_image():
    t = _make_tensor(sc.DT_U8, [1, 512, 512, 3])
    enc = sc.sdt_encode(t)
    dec = sc.sdt_decode(enc)
    assert dec.dtype == sc.DT_U8
    assert dec.dims == (1, 512, 512, 3)
    assert dec.data == t.data


def test_sdt_bad_magic():
    enc = sc.sdt_encode(sc.SdtTensor(dtype=sc.DT_U8, dims=(1,), data=b"\x01"))
    bad = b"XXXX" + enc[4:]
    with pytest.raises(ValueError, match="bad magic"):
        sc.sdt_decode(bad)


def test_sdt_truncated_payload():
    t = _make_tensor(sc.DT_F32, [2, 2])
    enc = sc.sdt_encode(t)
    with pytest.raises(ValueError, match="truncated payload"):
        sc.sdt_decode(enc[:-2])


def test_sdt_size_mismatch_on_encode():
    bad = sc.SdtTensor(dtype=sc.DT_F32, dims=(2, 2), data=b"\x00" * 3)
    with pytest.raises(ValueError, match=r"data 3 != dims\*dtype 16"):
        sc.sdt_encode(bad)


def test_sdcd_basic_roundtrip():
    f = sc.SdcdFrame()
    f.kv["backbone"] = "sdxl"
    f.kv["role"] = "te"
    embeds = _make_tensor(sc.DT_F16, [1, 77, 2048])
    embeds.data = embeds.data[:embeds.expected_nbytes()]
    pooled = _make_tensor(sc.DT_F16, [1, 1280])
    pooled.data = pooled.data[:pooled.expected_nbytes()]
    f.set("prompt_embeds", embeds)
    f.set("pooled", pooled)

    enc = sc.sdcd_encode(f)
    dec = sc.sdcd_decode(enc)
    assert dec.kv == {"backbone": "sdxl", "role": "te"}
    assert dec.get("prompt_embeds").dims == (1, 77, 2048)
    assert dec.get("pooled").dims == (1, 1280)
    assert dec.get("prompt_embeds").data == embeds.data


def test_sdcd_empty_frame():
    f = sc.SdcdFrame()
    enc = sc.sdcd_encode(f)
    dec = sc.sdcd_decode(enc)
    assert dec.kv == {}
    assert dec.tensors == {}


def test_sdcd_kv_only():
    f = sc.SdcdFrame()
    f.kv["prompt"] = "a cat on a windowsill"
    f.kv["neg"] = ""
    f.kv["cfg_split"] = "1"
    enc = sc.sdcd_encode(f)
    dec = sc.sdcd_decode(enc)
    assert dec.kv == f.kv


def test_sdcd_preserves_tensor_order():
    f = sc.SdcdFrame()
    for name in ["z", "a", "m"]:
        t = _make_tensor(sc.DT_U8, [2])
        t.data = t.data[:2]
        f.set(name, t)
    enc = sc.sdcd_encode(f)
    dec = sc.sdcd_decode(enc)
    assert list(dec.tensors.keys()) == ["z", "a", "m"]


def test_sdcd_bad_magic():
    f = sc.SdcdFrame()
    enc = bytearray(sc.sdcd_encode(f))
    enc[0] = 0
    with pytest.raises(ValueError, match="bad magic"):
        sc.sdcd_decode(bytes(enc))


def test_sdt_dtype_table_complete():
    # If a new dtype is added to the C side, the table here must follow.
    for k, v in sc.DTYPE_BYTES.items():
        assert v > 0
        assert k in sc.DTYPE_NAME
