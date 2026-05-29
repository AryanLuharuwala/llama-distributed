// comfy_adapter.h — bridge between gpunet-node and a locally-running ComfyUI
//
// The control plane sends a JSON workflow (`graph`) to the rig over WS as
// `{"kind":"comfy_run","job_id":N,"workflow":"image|video","graph":"..."}`.
// The rig POSTs that graph to ComfyUI's HTTP API (default http://127.0.0.1:8188),
// polls for completion, downloads the output files, and streams them back to
// the control plane as `comfy_result` frames carrying base64-encoded payloads.
//
// All HTTP is plain socket-level — no libcurl dependency.  ComfyUI exposes
// the API on http (not https) so this is enough.
//
// Environment:
//   DIST_COMFY_URL   — base URL for the local ComfyUI server.
//                      Default: http://127.0.0.1:8188
//   DIST_WITH_COMFYUI — when truthy (1/true/yes) the agent advertises
//                       `comfy_caps` on hello regardless of probe success
//                       (lets the server schedule jobs while ComfyUI is
//                       warming up; the job itself will fail back to the
//                       server if the API is genuinely down).
//
// Lifetime: a single ComfyClient is reused for the agent's life.  The class
// is thread-compatible (use one per thread) but not thread-safe; the agent
// drives it from a single dispatcher thread.

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace dist {

struct ComfyProbe {
    bool        ok      = false;
    std::string version;
    std::vector<std::string> models;   // ckpt / unet / vae files visible to ComfyUI
    std::string error;
};

// One output frame from a ComfyUI workflow execution.  Forwarded by the
// caller as a `comfy_result` WS frame with base64-encoded `data`.
struct ComfyResult {
    std::string filename;   // basename only (no path)
    std::string mime;       // sniffed from extension; "" if unknown
    std::vector<uint8_t> data;
};

// Callback invoked for each output file as it lands.
//   Return false to abort streaming further outputs.
using ComfyOnResult = std::function<bool(const ComfyResult&)>;

class ComfyClient {
public:
    // base_url like "http://127.0.0.1:8188" — trailing slash optional.
    explicit ComfyClient(std::string base_url);

    // Hit /system_stats + /object_info to confirm the server is alive and
    // collect a model list.  Cheap (no model load).  Times out fast so
    // hello-frame emission isn't blocked on a stalled ComfyUI.
    ComfyProbe probe(int timeout_ms = 1500);

    // Submit a workflow graph (already serialized JSON for the
    // `{"prompt": <graph>}` body) and stream every produced file back via
    // `on_result`.  Blocks until the run finishes, errors, or exceeds
    // `total_timeout_ms`.
    //
    // Returns "" on success, otherwise a one-line error description.
    std::string run(const std::string& graph_json,
                    int total_timeout_ms,
                    ComfyOnResult on_result);

    const std::string& base_url() const { return base_; }

    // Raw proxy — invoked from the agent's WS reader when a
    // `comfy_meta` frame arrives.  Method must be GET or POST; the
    // path is validated against an allowlist on the server side
    // before reaching us (see server/comfy_meta.go), and we re-check
    // here for defence-in-depth.  Returns body + *status (0 on
    // transport failure / non-allowed path).
    std::string proxy(const std::string& method,
                      const std::string& path,
                      const std::string& body,
                      int timeout_ms,
                      int* status);

private:
    std::string base_;
    std::string host_;
    uint16_t    port_ = 8188;
    std::string path_prefix_;   // empty unless base contains a path

    // HTTP primitives — implemented in comfy_adapter.cpp on raw sockets so
    // we don't pull in libcurl.  Each returns the response body; status
    // code goes into *status (0 on transport failure).
    std::string http_get (const std::string& path, int timeout_ms, int* status = nullptr);
    std::string http_post(const std::string& path,
                          const std::string& content_type,
                          const std::string& body,
                          int timeout_ms, int* status = nullptr);
};

// Convenience: build a ComfyClient from the DIST_COMFY_URL env var (with
// the documented default).  Returned by value; cheap.
ComfyClient make_default_comfy_client();

// Convenience: read DIST_WITH_COMFYUI / DIST_COMFY_FORCE truthy values.
bool comfy_force_enabled();

}  // namespace dist
