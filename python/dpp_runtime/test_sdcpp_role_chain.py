"""CF12-W5 — multi-daemon E2E smoke for the sd.cpp role-split chain.

Spawns three separate dist-sdcpp-worker daemons (one each for TE, UNet,
VAE) and walks an image generation through them, mirroring how the
server's planSdcppRoleChain dispatches `sdcpp_role_route` to three
different rigs.

The actual model load is gated on DIST_SDCPP_MODEL_SDXL (or any of the
per-family env vars from test_sdcpp_daemon) — without a model the test
verifies wire compatibility only (TE returns an error, UNet refuses
empty SDCD, VAE refuses empty SDT).
"""

from __future__ import annotations

import base64
import json
import os
from typing import Optional

import pytest

from dpp_runtime.test_sdcpp_daemon import (
    Daemon,
    WORKER_BIN,
    need_worker,
)
from dpp_runtime import sdt_codec as sc


def _model_path() -> Optional[str]:
    # Same per-family env vars as test_sdcpp_daemon; pick whichever exists.
    for var in (
        "DIST_SDCPP_MODEL_SDXL",
        "DIST_SDCPP_MODEL_SD15",
        "DIST_SDCPP_MODEL_SD3",
        "DIST_SDCPP_MODEL_FLUX",
        "DIST_SDCPP_MODEL_PIXART",
    ):
        v = os.environ.get(var)
        if v and os.path.exists(v):
            return v
    return None


@need_worker
def test_role_chain_wire_error_paths():
    """Without a model, the three daemons should still wire-handshake and
    refuse work cleanly — no hangs, all errors carry the matching req_id."""
    te = Daemon(WORKER_BIN)
    unet = Daemon(WORKER_BIN)
    vae = Daemon(WORKER_BIN)
    try:
        resp_te = te.request({
            "cmd": "sdr_encode_text",
            "req_id": 1,
            "model_path": "/nonexistent/te.gguf",
            "prompt": "x",
        })
        assert resp_te["kind"] == "sdcpp_error"
        assert resp_te["req_id"] == 1

        resp_u = unet.request({
            "cmd": "sdr_sample",
            "req_id": 2,
            "model_path": "/nonexistent/unet.gguf",
            "sdcd_b64": "",
        })
        assert resp_u["kind"] == "sdcpp_error"
        assert resp_u["req_id"] == 2

        resp_v = vae.request({
            "cmd": "sdr_decode_latent",
            "req_id": 3,
            "model_path": "/nonexistent/vae.gguf",
            "sdt_b64": "",
        })
        assert resp_v["kind"] == "sdcpp_error"
        assert resp_v["req_id"] == 3
    finally:
        te.close(); unet.close(); vae.close()


@need_worker
def test_role_chain_end_to_end():
    model = _model_path()
    if not model:
        pytest.skip("no DIST_SDCPP_MODEL_* env var pointing at a real GGUF")

    te = Daemon(WORKER_BIN)
    unet = Daemon(WORKER_BIN)
    vae = Daemon(WORKER_BIN)
    try:
        # ── TE on daemon A ────────────────────────────────────────────────
        r1 = te.request({
            "cmd": "sdr_encode_text",
            "req_id": 10,
            "model_path": model,
            "prompt": "a cat on a windowsill, oil painting",
            "negative_prompt": "",
        }, timeout=600.0)
        assert r1["kind"] == "sdcpp_role_done", r1
        assert r1["role"] == "te"
        sdcd_b64 = r1["frame_b64"]

        # ── UNet on daemon B ──────────────────────────────────────────────
        r2 = unet.request({
            "cmd": "sdr_sample",
            "req_id": 11,
            "model_path": model,
            "sdcd_b64": sdcd_b64,
            "width": 256,
            "height": 256,
            "steps": 4,
            "cfg": 5.0,
            "seed": 1234,
            "sampler": "euler_a",
        }, timeout=900.0)
        assert r2["kind"] == "sdcpp_role_done", r2
        assert r2["role"] == "unet"
        sdt_b64 = r2["frame_b64"]

        # ── VAE on daemon C ───────────────────────────────────────────────
        r3 = vae.request({
            "cmd": "sdr_decode_latent",
            "req_id": 12,
            "model_path": model,
            "sdt_b64": sdt_b64,
        }, timeout=600.0)
        assert r3["kind"] == "sdcpp_role_done", r3
        img = sc.sdt_decode(base64.b64decode(r3["frame_b64"]))
        assert img.dtype == sc.DT_U8
        assert len(img.dims) == 4 and img.dims[-1] in (3, 4)
    finally:
        te.close(); unet.close(); vae.close()
