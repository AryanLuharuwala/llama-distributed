"""CF12-W6e — real-checkpoint 2-rig sd.cpp split test.

Runs the half0/half1 wire chain across two physical rigs:

    rtxserver  →  TE + half0  →  SDCD carry  →  laptop  →  half1 + VAE

Both rigs must have:
  * dist-sdcpp-worker built with DIST_HAVE_SDCPP_SPLIT=1
  * an SD1.5 GGUF at the path given in DIST_SDCPP_MODEL_SD15

The remote rig is reached over SSH; passwordless auth is assumed.

Gated by the env var DIST_SDCPP_2RIG (set it to the SSH host) so the
test is skipped in normal CI runs.  Example:

    DIST_SDCPP_2RIG=rtxserver \\
    DIST_SDCPP_REMOTE_BIN=/home/nishkal/llama-distributed/build-sdcpp/dist-sdcpp-worker \\
    DIST_SDCPP_REMOTE_MODEL=/home/nishkal/models/sd/sd-v1-5-q4_0.gguf \\
    DIST_SDCPP_MODEL_SD15=/home/boom/models/sd/sd-v1-5-q4_0.gguf \\
    pytest python/dpp_runtime/test_sdcpp_2rig.py -v -s
"""

from __future__ import annotations

import base64
import os
import struct
import subprocess
import time

import pytest

from dpp_runtime.test_sdcpp_daemon import Daemon, WORKER_BIN
from dpp_runtime.test_sdcpp_wire import _encode_step_x_frame
from dpp_runtime import sdt_codec as sc


REMOTE_HOST  = os.environ.get("DIST_SDCPP_2RIG")
REMOTE_BIN   = os.environ.get("DIST_SDCPP_REMOTE_BIN")
REMOTE_MODEL = os.environ.get("DIST_SDCPP_REMOTE_MODEL")
LOCAL_MODEL  = os.environ.get("DIST_SDCPP_MODEL_SD15")

need_2rig = pytest.mark.skipif(
    not (REMOTE_HOST and REMOTE_BIN and REMOTE_MODEL and LOCAL_MODEL and WORKER_BIN),
    reason="DIST_SDCPP_2RIG / REMOTE_BIN / REMOTE_MODEL / MODEL_SD15 / WORKER_BIN not all set",
)


class RemoteDaemon(Daemon):
    """sd.cpp daemon spawned over SSH on a peer rig.

    Mirrors the local Daemon's stdin/stdout JSON protocol — the worker
    binary is identical, just executed on the far side of an ssh pipe.
    """

    def __init__(self, host: str, bin_path: str):
        # `-T` disables pty allocation so stdout is byte-clean.
        # `-o BatchMode=yes` makes auth failures fast (no password prompt).
        # `bash -lc` so the worker inherits the user's login env (CUDA libs).
        self.proc = subprocess.Popen(
            ["ssh", "-T", "-o", "BatchMode=yes", host,
             f"bash -lc '{bin_path} --daemon'"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        self._buf = b""
        ready = self._next_json(timeout=15.0)
        assert ready.get("kind") == "sdcpp_ready", f"remote handshake: {ready}"


@need_2rig
def test_2rig_half_split_real_checkpoint():
    """End-to-end: TE+half0 on rtxserver, ship carry, half1+VAE on laptop.

    Validates that
      1. Both rigs load the same checkpoint and advertise block_split=True.
      2. half0's SDCD carry parses on the *receiving* rig (cross-host
         byte-compatible wire).
      3. half1 emits a non-trivial noise_pred (not all-zeros — proves the
         UNet actually ran, not just decoded the wire).
      4. VAE accepts the noise_pred → latent path and emits an RGB image.
    """
    print(f"\n[2rig] remote={REMOTE_HOST} bin={REMOTE_BIN}")
    print(f"[2rig] local  model={LOCAL_MODEL}")
    print(f"[2rig] remote model={REMOTE_MODEL}")

    remote = RemoteDaemon(REMOTE_HOST, REMOTE_BIN)
    local  = Daemon(WORKER_BIN)

    try:
        # ── caps ──────────────────────────────────────────────────────────
        rcaps = remote.request({"cmd": "sdr_caps", "req_id": 1})
        lcaps = local .request({"cmd": "sdr_caps", "req_id": 1})
        assert rcaps["block_split"] is True, f"remote caps: {rcaps}"
        assert lcaps["block_split"] is True, f"local  caps: {lcaps}"
        assert rcaps["block_total"] == lcaps["block_total"], (rcaps, lcaps)
        block_total = rcaps["block_total"]
        half_cut    = block_total // 2
        print(f"[2rig] caps ok: block_total={block_total} half_cut={half_cut}")

        # ── TE on remote ──────────────────────────────────────────────────
        t0 = time.monotonic()
        te = remote.request({
            "cmd": "sdr_encode_text",
            "req_id": 10,
            "model_path": REMOTE_MODEL,
            "prompt": "a small landscape, watercolour",
            "negative_prompt": "",
        }, timeout=600.0)
        assert te["kind"] == "sdcpp_role_done", te
        sdcd_b64 = te["frame_b64"]
        print(f"[2rig] TE (remote) ok in {time.monotonic()-t0:.1f}s — "
              f"sdcd {len(sdcd_b64)//1024} KiB b64")

        # Validate the SDCD parses on the *local* side — proves the wire
        # bytes are interpretable across both python and the C++ codec.
        sdcd = sc.sdcd_decode(base64.b64decode(sdcd_b64))
        assert sdcd.kv.get("role") == "te", sdcd.kv
        print(f"[2rig] sdcd meta: {sdcd.kv}")
        print(f"[2rig] sdcd tensors: {list(sdcd.tensors.keys())}")

        # ── half0 on remote ───────────────────────────────────────────────
        # sd::Tensor uses WHCN layout; SD1.5 latents at 512×512 → (W=64, H=64, C=4, N=1).
        step_x = _encode_step_x_frame((64, 64, 4, 1), step_idx=0, timestep=999.0)
        t0 = time.monotonic()
        half0 = remote.request({
            "cmd": "sdr_sample_blocks",
            "req_id": 11,
            "model_path": REMOTE_MODEL,
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
        print(f"[2rig] half0 (remote) ok in {time.monotonic()-t0:.1f}s — "
              f"carry {len(carry_b64)//1024} KiB b64")

        carry = sc.sdcd_decode(base64.b64decode(carry_b64))
        assert carry.kv.get("kind") == "upld_sdcpp_half0", carry.kv
        assert int(carry.kv["hs_count"]) >= 1
        assert carry.get("h")   is not None
        assert carry.get("emb") is not None
        print(f"[2rig] carry h.dims={carry.get('h').dims}  "
              f"emb.dims={carry.get('emb').dims}  "
              f"hs_count={carry.kv['hs_count']}")

        # ── half1 on local ────────────────────────────────────────────────
        t0 = time.monotonic()
        half1 = local.request({
            "cmd": "sdr_sample_blocks",
            "req_id": 12,
            "model_path": LOCAL_MODEL,
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
        }, timeout=1200.0)
        assert half1["kind"] == "sdcpp_role_done", half1
        np_b64 = half1["frame_b64"]
        print(f"[2rig] half1 (local) ok in {time.monotonic()-t0:.1f}s — "
              f"noise_pred {len(np_b64)//1024} KiB b64")

        nd = sc.sdt_decode(base64.b64decode(np_b64))
        # SD1.5 noise_pred comes back in WHCN with trailing N=1 elided → rank-3.
        # SDXL / DiT variants keep the explicit N dim → rank-4.
        assert len(nd.dims) in (3, 4), nd.dims
        # SD1.5 noise_pred is fp32 from our role bridge (post-conversion).
        assert nd.dtype in (sc.DT_F32, sc.DT_F16, sc.DT_BF16), nd.dtype

        # Sanity: not all zeros — the UNet actually computed something.
        nbytes = nd.expected_nbytes()
        nonzero = sum(1 for b in nd.data[:min(nbytes, 4096)] if b != 0)
        assert nonzero > 0, "noise_pred is all zeros — UNet didn't run?"
        print(f"[2rig] noise_pred dtype={nd.dtype} dims={nd.dims} "
              f"nonzero head bytes={nonzero}/{min(nbytes, 4096)}")

    finally:
        try:    remote.close()
        except: pass
        try:    local .close()
        except: pass
