"""Unit tests for the inter-stage UNet payload codec.

Covers round-trip of sample + residuals, FINAL flag bit, empty
residual list (first-stage case), and rejection of malformed inputs.
Runs without torch/numpy — pure stdlib so it executes in any CI shape.
"""

from __future__ import annotations

import pytest

from dpp_runtime.unet_payload import UNetStepPayload, UNetTensor


def _t(dims, fill: int = 0xAB):
    """Synthesise a tensor of the given dims filled with `fill`.

    Size assumes fp16 (2 bytes per element).  We don't need real
    floats — we're testing the codec, not the model.
    """
    n = 1
    for d in dims:
        n *= d
    return UNetTensor(dims=tuple(dims), data=bytes([fill]) * (n * 2))


def test_roundtrip_no_residuals():
    p = UNetStepPayload(sample=_t([1, 4, 128, 128]), residuals=[], step_idx=0, timestep=999.0)
    got = UNetStepPayload.decode(p.encode())
    assert got.sample.dims == (1, 4, 128, 128)
    assert got.residuals == []
    assert got.step_idx == 0
    assert abs(got.timestep - 999.0) < 1e-3
    assert got.is_final_step is False


def test_roundtrip_with_residuals():
    p = UNetStepPayload(
        sample=_t([1, 1280, 16, 16], fill=0x12),
        residuals=[
            _t([1, 320, 128, 128], fill=0x01),
            _t([1, 640, 64, 64], fill=0x02),
            _t([1, 1280, 32, 32], fill=0x03),
        ],
        step_idx=7,
        timestep=420.5,
        is_final_step=True,
    )
    blob = p.encode()
    got = UNetStepPayload.decode(blob)
    assert got.sample.dims == (1, 1280, 16, 16)
    assert got.sample.data[0] == 0x12
    assert len(got.residuals) == 3
    assert got.residuals[0].dims == (1, 320, 128, 128)
    assert got.residuals[0].data[0] == 0x01
    assert got.residuals[2].dims == (1, 1280, 32, 32)
    assert got.residuals[2].data[0] == 0x03
    assert got.step_idx == 7
    assert abs(got.timestep - 420.5) < 1e-3
    assert got.is_final_step is True


def test_decode_rejects_bad_magic():
    blob = bytearray(b"\x00\x00\x00\x00" + b"\x01\x00\x00\x00" + b"\x00\x00\x00\x00\x00\x00\x00\x00")
    with pytest.raises(ValueError, match="bad magic"):
        UNetStepPayload.decode(bytes(blob))


def test_decode_rejects_truncated():
    p = UNetStepPayload(sample=_t([1, 4, 8, 8]), residuals=[_t([1, 4, 4, 4])], step_idx=1, timestep=0.5)
    blob = p.encode()
    with pytest.raises(ValueError, match="truncated"):
        UNetStepPayload.decode(blob[:-8])


def test_final_flag_bit_isolated():
    # is_final_step=False must clear the flag; True must set it.  No
    # other bits in the flag byte are reserved yet, so the byte should
    # be exactly 0 or 1.
    no_flag = UNetStepPayload(sample=_t([1, 1, 1, 1]), is_final_step=False).encode()
    yes_flag = UNetStepPayload(sample=_t([1, 1, 1, 1]), is_final_step=True).encode()
    # Flag byte is at offset 5 (after 4-byte magic + 1-byte ver).
    assert no_flag[5] == 0x00
    assert yes_flag[5] == 0x01
