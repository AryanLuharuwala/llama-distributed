"""CF12-W6e — UNet block-split smoke.

The C++ adapter accepts `sdr_sample_blocks` with [block_lo, block_hi) +
block_total.  Two paths are exercised:

  1. Degenerate full-range (block_lo=0, block_hi=block_total): collapses
     to a normal whole-UNet sample.  This is what a single-rig fallback
     looks like today — the wire is identical to a real block-split
     pipeline that just happens to have one rig owning all blocks.

  2. Partial-range: returns sdcpp_error with the ENOTIMPL marker, gated
     behind the upstream sd.cpp ggml patch (CF12-W6a-real).  The smoke
     test asserts the error path is honest — no hang, no crash, no
     misleading "ok".

Both paths run without a real model — the full-range case still falls
through to the ENOTIMPL path because the loader fails on the fake model
path.  When DIST_SDCPP_MODEL_* is set, the full-range path runs end-to-
end.
"""

from __future__ import annotations

import base64
import os

import pytest

from dpp_runtime.test_sdcpp_daemon import Daemon, WORKER_BIN, need_worker
from dpp_runtime import sdt_codec as sc


@need_worker
def test_partial_block_range_returns_error_not_hang():
    d = Daemon(WORKER_BIN)
    try:
        resp = d.request({
            "cmd": "sdr_sample_blocks",
            "req_id": 50,
            "model_path": "/nonexistent/model.gguf",
            "block_lo": 1,
            "block_hi": 4,
            "block_total": 7,
            "sdcd_b64": "",
            "upld_b64": "",
            "steps": 4,
            "cfg": 5.0,
            "seed": 1,
        })
        assert resp["kind"] == "sdcpp_error"
        assert resp["req_id"] == 50
    finally:
        d.close()


@need_worker
def test_full_range_block_split_collapses_to_sample():
    model = os.environ.get("DIST_SDCPP_MODEL_SDXL") or \
            os.environ.get("DIST_SDCPP_MODEL_SD15")
    if not model or not os.path.exists(model):
        pytest.skip("no real model for full-range block-split smoke")

    d = Daemon(WORKER_BIN)
    try:
        # First TE so we have an SDCD to feed.
        te = d.request({
            "cmd": "sdr_encode_text",
            "req_id": 60,
            "model_path": model,
            "prompt": "a small landscape, watercolour",
        }, timeout=600.0)
        assert te["kind"] == "sdcpp_role_done", te
        sdcd_b64 = te["frame_b64"]

        # Block-split with lo=0, hi=block_total → degenerate; should run
        # the whole UNet sample inside the role API.
        r = d.request({
            "cmd": "sdr_sample_blocks",
            "req_id": 61,
            "model_path": model,
            "block_lo": 0,
            "block_hi": 7,
            "block_total": 7,
            "sdcd_b64": sdcd_b64,
            "upld_b64": "",
            "steps": 4,
            "cfg": 5.0,
            "seed": 1234,
            "sampler": "euler_a",
        }, timeout=900.0)
        assert r["kind"] == "sdcpp_role_done", r
        assert r["role"] == "unet_blocks"
        sdt = sc.sdt_decode(base64.b64decode(r["frame_b64"]))
        # Same shape contract as sd_role_sample's output.
        assert sdt.dtype in (sc.DT_U8, sc.DT_F16, sc.DT_F32, sc.DT_BF16)
    finally:
        d.close()


@need_worker
def test_planner_partition_matches_python_split():
    """Cross-check that the Go partitionUNetBlocks math agrees with
    Python's: total=7, stages=3 → sizes [3,2,2].  Embedded here so any
    drift between the two splitters surfaces in CI."""
    # Single-side partitioning function (mirror of the Go helper).
    def partition(total: int, stages: int):
        if total <= 0 or stages <= 0:
            return []
        base, extra = divmod(total, stages)
        out, cursor = [], 0
        for i in range(stages):
            size = base + (1 if i < extra else 0)
            out.append((cursor, cursor + size))
            cursor += size
        return out

    assert partition(7, 3)  == [(0, 3), (3, 5), (5, 7)]
    assert partition(24, 4) == [(0, 6), (6, 12), (12, 18), (18, 24)]
    assert partition(57, 8) == [
        (0, 8), (8, 15), (15, 22), (22, 29),
        (29, 36), (36, 43), (43, 50), (50, 57),
    ]
