"""CF12-W1i — per-model smoke tests for dist-sdcpp-worker.

The worker is a C++ binary spawned via stdin/stdout JSON-line protocol.
These tests exercise the role API (`sdr_caps` / `sdr_encode_text` /
`sdr_sample` / `sdr_decode_latent`) without depending on a GGUF model
file being present — caps and error paths run unconditionally, while the
end-to-end paths are skipped when the per-family env var is unset.

Env vars (each gates the matching family's smoke test):

    DIST_SDCPP_BIN           path to dist-sdcpp-worker (default: search build dirs)
    DIST_SDCPP_MODEL_SD15    GGUF path for SD1.5
    DIST_SDCPP_MODEL_SDXL    GGUF path for SDXL
    DIST_SDCPP_MODEL_SD3     GGUF path for SD3
    DIST_SDCPP_MODEL_FLUX    GGUF path for Flux
    DIST_SDCPP_MODEL_PIXART  GGUF path for PixArt
"""

from __future__ import annotations

import base64
import json
import os
import select
import subprocess
import time
from pathlib import Path
from typing import Optional

import pytest


# ─── helpers ──────────────────────────────────────────────────────────────

def _resolve_worker_bin() -> Optional[str]:
    env = os.environ.get("DIST_SDCPP_BIN")
    if env and os.access(env, os.X_OK):
        return env
    here = Path(__file__).resolve()
    repo = here.parents[2]  # python/dpp_runtime/X -> repo root
    for cand in [
        repo / "build-sdcpp" / "dist-sdcpp-worker",
        repo / "build" / "dist-sdcpp-worker",
    ]:
        if cand.exists() and os.access(cand, os.X_OK):
            return str(cand)
    return None


WORKER_BIN = _resolve_worker_bin()
need_worker = pytest.mark.skipif(
    WORKER_BIN is None,
    reason="dist-sdcpp-worker not built (expected build-sdcpp/dist-sdcpp-worker)",
)


class Daemon:
    """Light wrapper around the worker's --daemon stdin/stdout protocol."""

    def __init__(self, bin_path: str):
        self.proc = subprocess.Popen(
            [bin_path, "--daemon"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        self._buf = b""
        # Wait for sdcpp_ready handshake.
        ready = self._next_json(timeout=5.0)
        assert ready.get("kind") == "sdcpp_ready", f"bad handshake: {ready}"

    def _next_json(self, timeout: float = 30.0) -> dict:
        deadline = time.monotonic() + timeout
        fd = self.proc.stdout.fileno()
        while True:
            nl = self._buf.find(b"\n")
            if nl >= 0:
                line = self._buf[:nl]
                self._buf = self._buf[nl + 1:]
                if not line.strip():
                    continue
                return json.loads(line.decode("utf-8"))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("daemon stdout idle")
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                continue
            chunk = os.read(fd, 65536)
            if not chunk:
                raise RuntimeError("daemon stdout closed")
            self._buf += chunk

    def send(self, msg: dict) -> None:
        line = (json.dumps(msg) + "\n").encode("utf-8")
        self.proc.stdin.write(line)
        self.proc.stdin.flush()

    def request(self, msg: dict, timeout: float = 60.0) -> dict:
        self.send(msg)
        return self._next_json(timeout=timeout)

    def close(self) -> None:
        try:
            self.send({"cmd": "quit"})
        except Exception:
            pass
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


# ─── unconditional protocol smoke ─────────────────────────────────────────

@need_worker
def test_daemon_handshake_and_caps():
    d = Daemon(WORKER_BIN)
    try:
        resp = d.request({"cmd": "sdr_caps", "req_id": 1})
        assert resp["kind"] == "sdcpp_caps"
        assert resp["req_id"] == 1
        assert "te" in resp["roles"]
        assert "unet" in resp["roles"]
        assert "vae" in resp["roles"]
        assert resp["sdt_ver"] >= 1
        assert resp["upld_ver"] >= 1
    finally:
        d.close()


@need_worker
def test_daemon_missing_model_path_errors():
    d = Daemon(WORKER_BIN)
    try:
        resp = d.request({"cmd": "sdr_encode_text", "req_id": 42,
                          "prompt": "a cat"})
        assert resp["kind"] == "sdcpp_error"
        assert resp["req_id"] == 42
        assert "model_path" in resp.get("error", "").lower()
    finally:
        d.close()


@need_worker
def test_daemon_bad_model_path_errors():
    d = Daemon(WORKER_BIN)
    try:
        resp = d.request({
            "cmd": "sdr_encode_text",
            "req_id": 43,
            "model_path": "/nonexistent/path/to/model.gguf",
            "prompt": "a cat",
        })
        assert resp["kind"] == "sdcpp_error"
        assert resp["req_id"] == 43
    finally:
        d.close()


@need_worker
def test_daemon_block_split_not_implemented_yet():
    """Sliceable UNet is gated behind CF12-W6a/b.  The role API should
    refuse politely instead of crashing."""
    d = Daemon(WORKER_BIN)
    try:
        resp = d.request({
            "cmd": "sdr_sample_blocks",
            "req_id": 44,
            "model_path": "/nonexistent/path/to/model.gguf",
            "block_lo": 0,
            "block_hi": 3,
            "block_total": 7,
            "sdcd_b64": "",
            "upld_b64": "",
        })
        assert resp["kind"] == "sdcpp_error"
        assert resp["req_id"] == 44
    finally:
        d.close()


# ─── per-family end-to-end (gated on a real model file) ──────────────────

@pytest.mark.parametrize(
    "env_var,family",
    [
        ("DIST_SDCPP_MODEL_SD15",   "sd1.5"),
        ("DIST_SDCPP_MODEL_SDXL",   "sdxl"),
        ("DIST_SDCPP_MODEL_SD3",    "sd3"),
        ("DIST_SDCPP_MODEL_FLUX",   "flux"),
        ("DIST_SDCPP_MODEL_PIXART", "pixart"),
    ],
)
@need_worker
def test_daemon_te_unet_vae_chain(env_var, family):
    model = os.environ.get(env_var)
    if not model or not os.path.exists(model):
        pytest.skip(f"{env_var} not set or model not present ({family})")

    d = Daemon(WORKER_BIN)
    try:
        # 1) TE
        te = d.request({
            "cmd": "sdr_encode_text",
            "req_id": 100,
            "model_path": model,
            "prompt": "a cat on a windowsill, oil painting",
            "negative_prompt": "",
            "cfg_split": 0,
            "clip_skip": -1,
        }, timeout=600.0)
        assert te["kind"] == "sdcpp_role_done", f"TE failed: {te}"
        assert te["role"] == "te"
        sdcd_b64 = te["frame_b64"]
        assert isinstance(sdcd_b64, str) and len(sdcd_b64) > 0
        # Verify the SDCD payload parses with our python codec.
        from dpp_runtime import sdt_codec as sc
        sdcd = sc.sdcd_decode(base64.b64decode(sdcd_b64))
        assert sdcd.kv.get("role") == "te"

        # 2) UNet
        unet = d.request({
            "cmd": "sdr_sample",
            "req_id": 101,
            "model_path": model,
            "sdcd_b64": sdcd_b64,
            "width": 256,
            "height": 256,
            "steps": 4,
            "cfg": 5.0,
            "seed": 1234,
            "sampler": "euler_a",
        }, timeout=900.0)
        assert unet["kind"] == "sdcpp_role_done", f"UNet failed: {unet}"
        assert unet["role"] == "unet"
        sdt_b64 = unet["frame_b64"]
        sdt = sc.sdt_decode(base64.b64decode(sdt_b64))
        assert sdt.dtype in (sc.DT_U8, sc.DT_F16, sc.DT_F32, sc.DT_BF16)

        # 3) VAE — accepts the SDT and emits an image SDT.
        vae = d.request({
            "cmd": "sdr_decode_latent",
            "req_id": 102,
            "model_path": model,
            "sdt_b64": sdt_b64,
        }, timeout=600.0)
        assert vae["kind"] == "sdcpp_role_done", f"VAE failed: {vae}"
        img = sc.sdt_decode(base64.b64decode(vae["frame_b64"]))
        # NHWC u8 image.
        assert img.dtype == sc.DT_U8
        assert len(img.dims) == 4
        assert img.dims[-1] in (3, 4)
    finally:
        d.close()
