"""Tests for the PROG sideband.

Verifies that send_progress produces the length-prefixed PROG-magic
frame that the dist-node's reader_loop expects, and that the JSON body
decodes back to the original event dict.
"""

import io
import json
import socket
import struct
import threading

from .wire import PROGRESS_MAGIC, send_progress


def test_progress_frame_shape():
    a, b = socket.socketpair()
    try:
        ev = {"kind": "dpp_progress", "req_id": 7, "stage_idx": 1,
              "role": "unet", "block_lo": 0, "block_hi": 3,
              "step_idx": 4, "total_steps": 30, "event": "step", "msg": ""}
        send_progress(a, ev)
        # Read length + body off the peer.
        hdr = b.recv(4)
        (n,) = struct.unpack(">I", hdr)
        assert 0 < n < (1 << 20), f"unreasonable len {n}"
        body = b.recv(n)
        assert len(body) == n
        magic = struct.unpack(">I", body[:4])[0]
        assert magic == PROGRESS_MAGIC, f"bad magic {magic:08x}"
        got = json.loads(body[4:].decode("utf-8"))
        assert got == ev, f"roundtrip mismatch: {got}"
    finally:
        a.close()
        b.close()


def test_progress_multiple_back_to_back():
    """Two events on the same conn must be framed independently."""
    a, b = socket.socketpair()
    try:
        send_progress(a, {"event": "enter", "stage_idx": 0})
        send_progress(a, {"event": "step",  "stage_idx": 0, "step_idx": 1})
        for expected in ("enter", "step"):
            hdr = b.recv(4)
            (n,) = struct.unpack(">I", hdr)
            body = b.recv(n)
            magic = struct.unpack(">I", body[:4])[0]
            assert magic == PROGRESS_MAGIC
            got = json.loads(body[4:].decode("utf-8"))
            assert got["event"] == expected
    finally:
        a.close()
        b.close()


def test_worker_emit_no_hook_is_noop():
    """Worker.emit() with no progress hook installed must not raise."""
    from .worker import Worker
    w = Worker(role="unet", model="x", block_lo=0, block_hi=3,
               device="cpu", dtype="fp16", cache="/tmp")
    # progress hook left as None; should silently drop.
    w.emit("enter", req_id=1, stage_idx=0)
    w.emit("step",  req_id=1, stage_idx=0, step_idx=5, total_steps=30)


def test_worker_emit_with_hook_passes_through():
    from .worker import Worker
    captured = []
    w = Worker(role="unet", model="x", block_lo=2, block_hi=5,
               device="cpu", dtype="fp16", cache="/tmp",
               progress=captured.append)
    w.emit("step", req_id=42, stage_idx=3, step_idx=7, total_steps=30,
           msg="hi")
    assert len(captured) == 1
    ev = captured[0]
    assert ev["kind"] == "dpp_progress"
    assert ev["req_id"] == 42 and ev["stage_idx"] == 3
    assert ev["role"] == "unet"
    assert ev["block_lo"] == 2 and ev["block_hi"] == 5
    assert ev["step_idx"] == 7 and ev["total_steps"] == 30
    assert ev["event"] == "step" and ev["msg"] == "hi"


def test_worker_emit_hook_failure_doesnt_kill_worker():
    from .worker import Worker
    def bad(_ev):
        raise RuntimeError("simulated socket EPIPE")
    w = Worker(role="vae", model="x", block_lo=-1, block_hi=-1,
               device="cpu", dtype="fp16", cache="/tmp",
               progress=bad)
    # Should swallow the exception — a flaky progress link must never
    # tank inference.
    w.emit("enter", req_id=1, stage_idx=0)
