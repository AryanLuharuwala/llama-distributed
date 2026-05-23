"""Entry point: `python -m dpp_runtime --role text_encoder|unet|vae [opts]`.

Launched by the C++ dist-node when a dpp_route control frame arrives.
Opens a TCP socket on 127.0.0.1:<port> and serves one or more requests.
Lifecycle is keyed by req_id sent in every frame; the runtime reuses the
loaded model across requests.

Usage (the agent supplies these via argv):

    python -m dpp_runtime \
        --role unet \
        --model stabilityai/stable-diffusion-xl-base-1.0 \
        --port 47781 \
        [--block-lo 0] [--block-hi -1] \
        [--device cuda] [--dtype fp16] \
        [--cache /var/tmp/dpp]

The control plane addresses each dpp_route to one of these processes.
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import threading

log = logging.getLogger("dpp_runtime")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--role", required=True, choices=["text_encoder", "unet", "vae"])
    p.add_argument("--model", default="")
    p.add_argument("--port", type=int, default=0,
                   help="loopback port the agent connects to; 0 = auto-pick & print")
    p.add_argument("--block-lo", type=int, default=-1)
    p.add_argument("--block-hi", type=int, default=-1)
    p.add_argument("--device", default=os.environ.get("DPP_DEVICE", "cuda"))
    p.add_argument("--dtype", default=os.environ.get("DPP_DTYPE", "fp16"))
    p.add_argument("--cache", default=os.environ.get("DPP_CACHE", "/var/tmp/dpp"))
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")

    # Lazy import — torch/diffusers are heavy, only the worker needs them.
    from dpp_runtime.worker import Worker

    worker = Worker(
        role=args.role,
        model=args.model,
        block_lo=args.block_lo,
        block_hi=args.block_hi,
        device=args.device,
        dtype=args.dtype,
        cache=args.cache,
    )
    worker.preload()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", args.port))
    srv.listen(4)
    actual_port = srv.getsockname()[1]

    # The agent reads stdout for the listening port — line-buffered.
    print(f"DPP_LISTEN {actual_port}", flush=True)
    log.info("dpp_runtime[%s] listening on 127.0.0.1:%d", args.role, actual_port)

    try:
        while True:
            conn, addr = srv.accept()
            log.info("accept %s", addr)
            t = threading.Thread(target=_serve, args=(conn, worker), daemon=True)
            t.start()
    except KeyboardInterrupt:
        log.info("shutting down")
        srv.close()
        sys.exit(0)


def _serve(conn: socket.socket, worker) -> None:
    from dpp_runtime.wire import recv_frame, send_frame
    try:
        while True:
            frame = recv_frame(conn)
            for out in worker.handle(frame):
                send_frame(conn, out)
    except (ConnectionError, OSError) as e:
        log.info("conn closed: %s", e)
    except Exception as e:
        log.exception("worker crash: %s", e)
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
