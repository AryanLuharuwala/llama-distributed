#pragma once
//
// Diffusion pipeline-parallel adapter.
//
// One DppAdapter instance lives in dist-node.  When a `dpp_route` JSON
// control message arrives over the WS, the adapter:
//
//   1. Looks up (or spawns) a Python runtime process for the requested
//      (role, model) pair.  Each runtime opens a loopback TCP socket; the
//      adapter is the sole local client of that socket.
//   2. Forwards subsequent ACTV frames that target this req_id into the
//      runtime via the socket (length-prefixed wire).
//   3. A reader thread per runtime drains response frames off the socket
//      and pushes them into a cross-thread outbox.  The main loop (the WS
//      writer) drains that outbox each tick.
//
// The C++ side never decodes the ACTV payload — it ferries opaque bytes.
// All tensor work happens in the Python runtime.
//
// Multi-backend hook point: today `Runtime::launch_python` runs
//   python -m dpp_runtime --role=… --model=…
// A future backend (torchrun + NCCL on a LAN cluster, Ray, etc.) just
// supplies a different launch command — the wire contract is unchanged.

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace dist {

// One frame, opaque to the adapter.
struct DppFrame {
    std::vector<uint8_t> bytes;
};

// A single Python runtime process the adapter is talking to.
//
// Lifecycle: created by `handle_dpp_route` in a *pending* state (ready=false).
// A background launcher thread does the fork+exec+wait-for-DPP_LISTEN+connect
// dance, then flips ready=true and drains buffered frames.  This keeps the
// dist-node main loop responsive to WS pings while a multi-GB model loads.
struct DppRuntime {
    std::string role;
    std::string model;
    int         pid       = -1;
    int         sock_fd   = -1;
    uint16_t    port      = 0;
    std::thread reader;
    std::thread launcher;
    std::atomic<bool> stop{false};
    std::atomic<bool> ready{false};
    std::atomic<bool> failed{false};

    std::mutex              pending_mu;
    std::vector<DppFrame>   pending;

    ~DppRuntime();
    void close();
    // send_or_buffer: if the runtime is ready, send on the socket; otherwise
    // park the frame on `pending` so the launcher can drain it once connected.
    bool send_or_buffer(const DppFrame& f);
    // raw_send is unconditional — caller knows the socket is up.
    bool raw_send(const DppFrame& f);
};

class DppAdapter {
public:
    DppAdapter();
    ~DppAdapter();

    // Pull pending outbound frames the adapter wants ws-sent.  Thread-safe.
    // Returns frames that should be ws.send_binary'd on the main thread.
    std::vector<DppFrame> drain_outbox();

    // Pull pending JSON progress lines (one per dpp_progress event the
    // worker emitted on the runtime socket via the "PROG" sideband).
    // Each string is a complete JSON object ready for ws.send_text.
    std::vector<std::string> drain_text_outbox();

    // Handle a `dpp_route` JSON message.  Returns true if it was a valid
    // dpp_route and a runtime was spawned/reused; false on parse error.
    // The `python_bin` is the interpreter to invoke (default: "python3");
    // the `module_path` is the directory containing the `dpp_runtime`
    // package (so PYTHONPATH includes it).
    bool handle_dpp_route(const std::string& json_msg,
                          const std::string& python_bin,
                          const std::string& module_path,
                          std::string& err_out);

    // Forward an incoming binary ACTV frame to whichever runtime owns the
    // req_id encoded in the frame.  Returns true if delivered.
    bool dispatch_actv(const uint8_t* data, size_t n);

    // Used by capability advertising (hello frame).  Reports whether
    // diffusion-PP is locally available (python3 + the dpp_runtime module
    // importable).
    static bool probe_local_caps(const std::string& python_bin,
                                 const std::string& module_path,
                                 std::string& err);

    // Companion probe for the C++ diffusion backend.  Reports whether
    // dist-sdcpp-worker is runnable on this rig (binary exists, ggml
    // backends initialise).  Used when python+diffusers is not available
    // (e.g. paramshakti cn01) — the rig still advertises `sdcpp:1` so the
    // control plane can route a whole-pipeline (single-rig) diffusion job
    // here.  Caps-style: model_load is not performed.
    static bool probe_local_sdcpp_caps(const std::string& worker_bin,
                                       std::string& backend_info,
                                       std::string& err);

    // Handle an `sdcpp_route` JSON message — single-rig whole-pipeline
    // diffusion via the vendored sd.cpp backend.  Forks dist-sdcpp-worker
    // with the request params, waits for the PNG to be produced, then
    // emits a PROG frame (with the PNG bytes embedded as JSON) + a DONE
    // marker so the server's dpp dispatch surface can drain it the same
    // way it drains a python-DPP run.  Synchronous from the caller's
    // perspective — runs the worker in a detached thread internally.
    //
    // Resident-daemon path (#254): the first call to handle_sdcpp_route
    // forks dist-sdcpp-worker once with `--daemon`, then every subsequent
    // request is piped over stdin as a single JSON line.  The model stays
    // resident in the worker between requests (it only reloads when the
    // model_path changes), which is the actual perf win — sd.cpp model
    // load is multi-GB and dominates the per-image cost on CPU rigs.
    //
    // A reader thread inside the adapter parses the worker's stdout lines
    // and pushes JSON events into text_outbox_ (sdcpp_progress, sdcpp_done,
    // sdcpp_error) so the server's existing dpp text-outbox surface drains
    // them unchanged.  For sdcpp_done, the adapter slurps the PNG the
    // worker wrote and base64-embeds it on the way out — keeps the wire
    // text-only for this path.
    bool handle_sdcpp_route(const std::string& json_msg,
                            const std::string& worker_bin,
                            std::string& err_out);

    // CF12-W3 — per-role dispatch.  Parses {role, req_id, model_path, ...}
    // and forwards the matching `sdr_encode_text` / `sdr_sample` /
    // `sdr_sample_blocks` / `sdr_decode_latent` / `sdr_caps` command to
    // the resident sdcpp daemon.  Binary payloads (SDCD cond, SDT latent,
    // UPLD block-state) ride as base64 strings on the same JSON line —
    // the daemon decodes and re-encodes them on the way back through the
    // text outbox as `sdcpp_role_done`.
    //
    // role must be one of: "te" | "unet" | "unet_blocks" | "vae" | "caps".
    // Returns true if the command was queued; false (with err_out set) on
    // parse error or daemon spawn failure.
    bool handle_sdcpp_role_route(const std::string& json_msg,
                                 const std::string& worker_bin,
                                 std::string& err_out);

private:
    // Key by (role + ":" + model).  Reusing the same runtime across
    // requests keeps the model resident in VRAM.
    std::map<std::string, std::unique_ptr<DppRuntime>> runtimes_;
    std::mutex runtimes_mu_;

    // ((req_id, stage_idx) → runtime_key).  Established at dpp_route time —
    // one entry per (req, stage) the agent is assigned to.  Lookup is keyed
    // on BOTH req_id and stage_idx so that a single agent owning multiple
    // consecutive stages can pipeline them locally without re-routing
    // through the WS.
    std::map<uint32_t, std::string> req_to_runtime_; // key = (req_id<<16)|stage_idx
    std::mutex req_mu_;
    static uint32_t reqStageKey(uint16_t r, uint16_t s) {
        return ((uint32_t)r << 16) | (uint32_t)s;
    }

    // Frames ready for ws.send_binary on the main thread.
    std::vector<DppFrame> outbox_;
    std::mutex outbox_mu_;

    // JSON progress lines ready for ws.send_text on the main thread.
    // Populated by reader_loop when a "PROG"-magic frame arrives on the
    // runtime socket — see python/dpp_runtime/wire.py:send_progress.
    std::vector<std::string> text_outbox_;
    std::mutex text_outbox_mu_;

    // Shared resident worker for the C++ diffusion backend.  Forward-decl
    // only — the SdcppDaemon owns child pid + pipe fds + a reader thread
    // and lives entirely inside diffusion_pp_adapter.cpp.  Lazily spawned
    // on the first sdcpp_route.
    std::unique_ptr<struct SdcppDaemon> sdcpp_daemon_;
    std::mutex sdcpp_daemon_mu_;

    // Reader for the resident sdcpp worker.  Static so it can run on its
    // own std::thread; takes the adapter+daemon by pointer.
    static void sdcpp_daemon_reader(DppAdapter* self,
                                    struct SdcppDaemon* d);

    DppRuntime* find_runtime_locked(const std::string& key);
    // Background launcher: forks + exec's the python runtime, waits for the
    // DPP_LISTEN line, connects the loopback socket, then starts the reader
    // thread and flips ready=true.  On failure, sets failed=true and pushes an
    // error ACTV frame to the outbox so the caller (server) gets unstuck
    // instead of hanging until the 5-minute drain timeout.
    void launch_python(DppRuntime* rt,
                       const std::string& python_bin,
                       const std::string& module_path);

    void reader_loop(DppRuntime* rt);
    // Build an opaque ACTV TYPE_ERROR frame ([req_id], msg) ready for ws.send_binary.
    static DppFrame make_error_frame(uint16_t req_id, const std::string& msg);
};

} // namespace dist
