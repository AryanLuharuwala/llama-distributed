"""CF12-W6d — Python ↔ C++ wire compatibility tests for the 2-rig sd.cpp
block-split daemon path.

These tests are model-free.  They cover:

  1. The worker's `sdr_caps` response carries the new `block_split` /
     `block_total` fields introduced in CF12-W6c.  When the binary is
     built with DIST_HAVE_SDCPP_SPLIT the advert is `block_total == 2`
     and `block_split == true`; otherwise the loader falls through to
     `block_total == 1` and `block_split == false`.  The test accepts
     both flavours so it runs in CI before the ggml patch lands.

  2. Python encodes an `sdcpp_step_x` SDCD frame (kind, step_idx,
     timestep + a single fp32 "x" tensor) and the C++ daemon mirrors
     the dispatch error contract — the dispatch we want to exercise
     fires *after* `ensure_loaded`, so without a real model we assert
     the wire shape itself parses cleanly in Python (encode ↔ decode).

  3. The `sdr_sample_blocks` error paths return `sdcpp_error` (not a
     crash, not a hang) when the daemon can't find a model.  This is
     the safety net for planners that broadcast block-split commands
     to a rig that has no UNet checkpoint yet.

End-to-end half0/half1 numerical equivalence lives in
`test_sdcpp_unet_split.py` (gated on DIST_SDCPP_MODEL_*).
"""

from __future__ import annotations

import base64
import json
import os
import struct

import pytest

from dpp_runtime.test_sdcpp_daemon import Daemon, WORKER_BIN, need_worker
from dpp_runtime import sdt_codec as sc


# ─── helpers ──────────────────────────────────────────────────────────────

def _encode_step_x_frame(x_dims, *, step_idx: int, timestep: float) -> bytes:
    """Mirror of include/sdcpp_split_wire.h::sdcpp_x_to_sdcd in python.

    Caller drives the scheduler and emits one of these per (step, lo=0)
    block call — sd.cpp's `sd_role_sample_blocks` decodes the frame with
    `sdcpp_sdcd_to_x` and recovers the step metadata.
    """
    n = 1
    for d in x_dims:
        n *= int(d)
    data = struct.pack("<%df" % n, *(0.0 for _ in range(n)))
    t = sc.SdtTensor(dtype=sc.DT_F32, dims=tuple(int(d) for d in x_dims), data=data)
    frame = sc.SdcdFrame()
    frame.kv["kind"]     = "sdcpp_step_x"
    frame.kv["step_idx"] = str(int(step_idx))
    frame.kv["timestep"] = repr(float(timestep))
    frame.set("x", t)
    return sc.sdcd_encode(frame)


# ─── caps ─────────────────────────────────────────────────────────────────

@need_worker
def test_caps_advertises_block_split_fields():
    """CF12-W6c added block_split / block_total to the caps advert.  The
    planner uses these to decide whether to fan UNet block ranges out."""
    d = Daemon(WORKER_BIN)
    try:
        resp = d.request({"cmd": "sdr_caps", "req_id": 7})
        assert resp["kind"] == "sdcpp_caps"
        assert resp["req_id"] == 7

        assert "block_split" in resp, f"caps missing block_split: {resp}"
        assert "block_total" in resp, f"caps missing block_total: {resp}"
        assert isinstance(resp["block_split"], bool)
        assert isinstance(resp["block_total"], int)

        # When patched (DIST_HAVE_SDCPP_SPLIT) the advert is split-capable;
        # otherwise the worker honestly says no.  The two states are
        # mutually consistent.
        if resp["block_split"]:
            assert resp["block_total"] >= 2, resp
        else:
            assert resp["block_total"] == 1, resp
    finally:
        d.close()


# ─── wire roundtrip (python-only; proves the python codec matches the
#     C++ encoder so a cross-language daemon hop is byte-equal) ────────────

def test_step_x_frame_python_roundtrip():
    """Encode an `sdcpp_step_x` SDCD frame in python and decode it back —
    the C++ daemon reads exactly the same bytes via
    `dist::sdcpp_sdcd_to_x`, so a byte-equal python roundtrip is the
    cheap-and-fast version of the cross-language smoke."""
    buf = _encode_step_x_frame((1, 4, 8, 8), step_idx=3, timestep=14.5)
    frame = sc.sdcd_decode(buf)
    assert frame.kv["kind"]     == "sdcpp_step_x"
    assert frame.kv["step_idx"] == "3"
    assert float(frame.kv["timestep"]) == pytest.approx(14.5)

    x = frame.get("x")
    assert x is not None
    assert x.dtype == sc.DT_F32
    assert x.dims  == (1, 4, 8, 8)
    assert len(x.data) == 1 * 4 * 8 * 8 * 4

    # Byte-stability: re-encode must be identical (kv order + tensor
    # order are preserved by SdcdFrame).
    assert sc.sdcd_encode(frame) == buf


def test_half0_carry_frame_python_roundtrip():
    """Same idea for the `upld_sdcpp_half0` carry SDCD — h + emb + hs.N
    tensors plus an hs_count meta field.  Mirrors what the half0 rig
    emits and the half1 rig consumes."""
    frame = sc.SdcdFrame()
    frame.kv["kind"]     = "upld_sdcpp_half0"
    frame.kv["hs_count"] = "2"

    def _f32_zeros(dims):
        n = 1
        for d in dims: n *= d
        return sc.SdtTensor(dtype=sc.DT_F32, dims=tuple(dims),
                            data=struct.pack("<%df" % n, *(0.0 for _ in range(n))))

    frame.set("h",    _f32_zeros((1, 4, 8, 8)))
    frame.set("emb",  _f32_zeros((1, 320)))
    frame.set("hs.0", _f32_zeros((1, 4, 8, 8)))
    frame.set("hs.1", _f32_zeros((1, 4, 4, 4)))

    buf = sc.sdcd_encode(frame)
    out = sc.sdcd_decode(buf)
    assert out.kv["hs_count"] == "2"
    assert set(out.tensors.keys()) == {"h", "emb", "hs.0", "hs.1"}
    assert out.get("hs.1").dims == (1, 4, 4, 4)


# ─── dispatch error contract (no model required) ─────────────────────────

@need_worker
def test_sample_blocks_missing_model_errors_cleanly():
    """Without a `model_path` the daemon must come back with
    `sdcpp_error` — never a crash, never a silent timeout."""
    d = Daemon(WORKER_BIN)
    try:
        resp = d.request({
            "cmd": "sdr_sample_blocks",
            "req_id": 80,
            "block_lo": 0,
            "block_hi": 1,
            "block_total": 2,
            "sdcd_b64": "",
            "upld_b64": "",
        })
        assert resp["kind"]   == "sdcpp_error"
        assert resp["req_id"] == 80
        assert "model_path" in resp.get("error", "").lower()
    finally:
        d.close()


@need_worker
def test_sample_blocks_bad_model_errors_cleanly():
    """Real-looking but nonexistent path: the loader fails and the daemon
    must return the error rather than mask it as success."""
    d = Daemon(WORKER_BIN)
    try:
        # Feed a real step_x wire payload so the dispatch *can* progress
        # past the b64 decode — the load step is the one we want to see
        # fail.
        step_x = _encode_step_x_frame((1, 4, 8, 8), step_idx=0, timestep=1.0)
        resp = d.request({
            "cmd": "sdr_sample_blocks",
            "req_id": 81,
            "model_path": "/nonexistent/path/model.gguf",
            "block_lo": 0,
            "block_hi": 1,
            "block_total": 2,
            "sdcd_b64": "",
            "upld_b64": base64.b64encode(step_x).decode("ascii"),
            "steps": 4,
            "cfg": 5.0,
            "seed": 1,
            "step_idx": 0,
            "timestep": 1.0,
        })
        assert resp["kind"]   == "sdcpp_error"
        assert resp["req_id"] == 81
    finally:
        d.close()


# ─── end-to-end half0 → half1 wire chain (gated on a real model) ─────────

@need_worker
def test_half0_half1_wire_chain_smoke():
    """When DIST_SDCPP_MODEL_SD15 (or SDXL) is set we run the full split:

        TE → half0 (emits SDCD carry) → half1 (emits SDT noise_pred)

    for a single denoise step.  This catches drift between the C++
    encoders/decoders and the python codec at the SDCD boundary —
    half0's output must be a frame python can parse, and half1 must
    accept the same bytes that python could have generated.
    """
    model = (os.environ.get("DIST_SDCPP_MODEL_SD15")
             or os.environ.get("DIST_SDCPP_MODEL_SDXL"))
    if not model or not os.path.exists(model):
        pytest.skip("no real SD model; skip wire-chain smoke")

    d = Daemon(WORKER_BIN)
    try:
        te = d.request({
            "cmd": "sdr_encode_text",
            "req_id": 90,
            "model_path": model,
            "prompt": "a still life with apples",
        }, timeout=600.0)
        assert te["kind"] == "sdcpp_role_done", te
        sdcd_b64 = te["frame_b64"]

        # Tell the daemon the model so block_total is real.
        caps = d.request({"cmd": "sdr_caps", "req_id": 91})
        block_total = int(caps.get("block_total", 2))
        if block_total < 2:
            pytest.skip("worker not compiled with DIST_HAVE_SDCPP_SPLIT")
        half_cut = block_total // 2

        # Build an sdcpp_step_x frame.  sd::Tensor uses WHCN layout (not
        # NCHW), so SD1.5 at 512×512 latents → (W=64, H=64, C=4, N=1).
        step_x = _encode_step_x_frame((64, 64, 4, 1), step_idx=0, timestep=999.0)

        half0 = d.request({
            "cmd": "sdr_sample_blocks",
            "req_id": 92,
            "model_path": model,
            "sdcd_b64": sdcd_b64,
            "upld_b64": base64.b64encode(step_x).decode("ascii"),
            "block_lo": 0,
            "block_hi": half_cut,
            "block_total": block_total,
            "steps": 4,
            "cfg": 5.0,
            "seed": 1234,
            "sampler": "euler_a",
            "step_idx": 0,
            "timestep": 999.0,
        }, timeout=900.0)
        assert half0["kind"] == "sdcpp_role_done", half0
        carry_b64 = half0["frame_b64"]

        # Validate the half0 carry parses in python — same bytes the
        # half1 rig will be fed.
        carry = sc.sdcd_decode(base64.b64decode(carry_b64))
        assert carry.kv.get("kind") == "upld_sdcpp_half0"
        assert "hs_count" in carry.kv
        assert carry.get("h")   is not None
        assert carry.get("emb") is not None

        half1 = d.request({
            "cmd": "sdr_sample_blocks",
            "req_id": 93,
            "model_path": model,
            "sdcd_b64": sdcd_b64,
            "upld_b64": carry_b64,
            "block_lo": half_cut,
            "block_hi": block_total,
            "block_total": block_total,
            "steps": 4,
            "cfg": 5.0,
            "seed": 1234,
            "sampler": "euler_a",
            "step_idx": 0,
            "timestep": 999.0,
        }, timeout=900.0)
        assert half1["kind"] == "sdcpp_role_done", half1

        noise_pred = sc.sdt_decode(base64.b64decode(half1["frame_b64"]))
        # SD1.5 noise pred is WHCN with trailing N=1 dropped → rank-3.
        assert noise_pred.dtype in (sc.DT_F32, sc.DT_F16, sc.DT_BF16)
        assert len(noise_pred.dims) in (3, 4)
    finally:
        d.close()
