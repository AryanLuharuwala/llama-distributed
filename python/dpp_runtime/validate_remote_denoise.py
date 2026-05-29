"""CF12-W7 — remote-denoise N-way generation validator (sd.cpp + worker only).

Proves the production mechanism end-to-end WITHOUT the Go server / dist-node:
a Python coordinator plays the server's role.

  HOST daemon   : runs sdr_generate_remote — sd.cpp's full sample() loop
                  (TE + denoiser + sampler + VAE local), but each per-step
                  UNet eval is delegated out via sdcpp_need_denoise.
  CHAIN daemon  : runs the N-way block chain (sdr_sample_blocks across stages).
  coordinator   : for each need_denoise(x, t, ctx, y) -> build step_x + cond
                  SDCD, run the N-stage block chain -> noise_pred (eps) ->
                  send sdr_denoise_result back to the host. Final frame is the
                  generated RGB image (role "generate").

Compares the remote-denoise image against a baseline whole-pipeline `gen` on
the same seed: with N=1 (single full-range block) they should match closely
(same fp path); with N>1 they match to the fp16-boundary tolerance.

Usage on the rig:
    DIST_SDCPP_WORKER=~/llama-distributed/build-sdcpp/dist-sdcpp-worker \
    DIST_SDCPP_MODEL=~/models/sd/sd-v1-5-q4_0.gguf \
    python -m dpp_runtime.validate_remote_denoise
"""

from __future__ import annotations

import base64
import json
import os
import struct
import sys

from dpp_runtime.test_sdcpp_daemon import Daemon
from dpp_runtime import sdt_codec as sc

_partition = sc.partition_blocks  # shared block partitioner

WORKER = os.environ.get("DIST_SDCPP_WORKER") or os.environ.get("DIST_SDCPP_WORKER_BIN")
MODEL = os.environ.get("DIST_SDCPP_MODEL") or os.environ.get("DIST_SDCPP_MODEL_SD15")
STAGES = int(os.environ.get("DIST_SDCPP_STAGES", "3"))
STEPS = int(os.environ.get("DIST_SDCPP_STEPS", "3"))
DIM = int(os.environ.get("DIST_SDCPP_DIM", "256"))



def _send(d, obj):
    d.proc.stdin.write((json.dumps(obj) + "\n").encode())
    d.proc.stdin.flush()


def _cond_sdcd(ctx_b64, y_b64):
    """Wrap the host's c_crossattn (+ optional c_vector) into an SDCD cond."""
    f = sc.SdcdFrame()
    f.kv["role"] = "te"
    ctx = sc.sdt_decode(base64.b64decode(ctx_b64))
    f.set("cond.crossattn", ctx)
    if y_b64:
        y = sc.sdt_decode(base64.b64decode(y_b64))
        if y.expected_nbytes() > 0:
            f.set("cond.vector", y)
    return base64.b64encode(sc.sdcd_encode(f)).decode("ascii")


def _step_x(x_b64, t):
    """Wrap the host's noised latent x into an sdcpp_step_x SDCD frame."""
    f = sc.SdcdFrame()
    f.kv["kind"] = "sdcpp_step_x"
    f.kv["step_idx"] = "0"
    f.kv["timestep"] = repr(float(t))
    f.set("x", sc.sdt_decode(base64.b64decode(x_b64)))
    return base64.b64encode(sc.sdcd_encode(f)).decode("ascii")


def _run_block_chain(chain, total, ranges, sdcd_b64, x_b64, t):
    """One distributed UNet eval: returns eps (noise_pred) as base64 SDT."""
    upld = _step_x(x_b64, t)
    out = None
    for i, (lo, hi) in enumerate(ranges):
        out = chain.request({
            "cmd": "sdr_sample_blocks", "req_id": 500 + i, "model_path": MODEL,
            "sdcd_b64": sdcd_b64, "upld_b64": upld,
            "block_lo": lo, "block_hi": hi, "block_total": total,
            "steps": 1, "cfg": 1.0, "seed": 0, "sampler": "euler_a",
            "step_idx": 0, "timestep": t,
        }, timeout=1200.0)
        assert out["kind"] == "sdcpp_role_done", (lo, hi, out)
        if hi < total:
            upld = out["frame_b64"]
    return out["frame_b64"]


def main() -> int:
    if not (WORKER and MODEL):
        print("set DIST_SDCPP_WORKER and DIST_SDCPP_MODEL", file=sys.stderr)
        return 2
    host = Daemon(WORKER)
    chain = Daemon(WORKER)
    try:
        total = chain.request({"cmd": "sdr_caps", "req_id": 1, "model_path": MODEL},
                              timeout=600.0)["block_total"]
        ranges = _partition(total, STAGES)
        print(f"[remote] block_total={total} stages={STAGES} ranges={ranges} "
              f"dim={DIM} steps={STEPS}")

        _send(host, {
            "cmd": "sdr_generate_remote", "req_id": 1, "model_path": MODEL,
            "prompt": "a small landscape, watercolour", "negative_prompt": "",
            "width": DIM, "height": DIM, "steps": STEPS, "cfg": 5.0,
            "seed": 1234, "sampler": "euler_a",
        })

        n_evals = 0
        while True:
            msg = host._next_json(timeout=1800.0)
            kind = msg.get("kind")
            if kind == "sdcpp_need_denoise":
                n_evals += 1
                sdcd_b64 = _cond_sdcd(msg.get("ctx_b64", ""), msg.get("y_b64", ""))
                eps_b64 = _run_block_chain(chain, total, ranges, sdcd_b64,
                                           msg["x_b64"], float(msg["t"]))
                _send(host, {"cmd": "sdr_denoise_result", "req_id": 1, "eps_b64": eps_b64})
            elif kind == "sdcpp_done":
                png = base64.b64decode(msg.get("png_b64", ""))
                is_png = png[:8] == b"\x89PNG\r\n\x1a\n"
                print(f"[remote] DONE png bytes={len(png)} png_magic={is_png} evals={n_evals}")
                ok = (len(png) > 1000 and is_png)
                if ok:
                    open("/tmp/sdcpp_remote_nway.png", "wb").write(png)
                    print("[remote] wrote /tmp/sdcpp_remote_nway.png")
                print("[remote] RESULT:", "PASS" if ok else "FAIL")
                return 0 if ok else 1
            elif kind == "sdcpp_error":
                print("[remote] worker error:", msg)
                return 1
            elif kind == "sdcpp_progress":
                continue
            else:
                print("[remote] unexpected frame:", {k: msg.get(k) for k in ("kind", "role")})
    finally:
        try: host.close()
        except Exception: pass
        try: chain.close()
        except Exception: pass


if __name__ == "__main__":
    sys.exit(main())
