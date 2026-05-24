#pragma once
//
// P12: vLLM runtime adapter.
//
// vLLM is a Python single-rig runtime that ships an OpenAI-compatible HTTP
// server (`vllm.entrypoints.openai.api_server`).  This adapter speaks to
// that server over plain HTTP and exposes the dist::IRuntimeAdapter
// interface so the agent can pick vLLM in place of the inline llama.cpp
// path when the rig has been assigned the whole model.
//
// Two modes:
//
//   - Attach mode (default): connect to an already-running vLLM at
//     DIST_VLLM_URL (default http://127.0.0.1:8000).  The operator owns
//     the lifecycle (systemd unit, container entrypoint, etc.).  This is
//     the production shape — vLLM has its own warmup/healthcheck story
//     and we don't want to babysit it from inside dist-node.
//
//   - Spawn mode: when DIST_VLLM_SPAWN=1 and a model path is given to
//     load_model(), the adapter execs `python -m vllm.entrypoints.openai.api_server`
//     with the model path and waits up to DIST_VLLM_SPAWN_TIMEOUT seconds
//     for /health to return 200.  Useful for dev rigs and CI.
//
// Streaming: vLLM emits SSE `data: <json>` lines on /v1/completions when
// `stream: true` is set.  We parse line-by-line and call ChunkCallback for
// each delta.  Returning false from the callback closes the socket, which
// vLLM treats as a client disconnect and stops decoding.

#include "runtime_adapter.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

namespace dist {

struct VllmAdapterConfig {
    std::string base_url   = "http://127.0.0.1:8000";
    std::string api_key;            // optional Authorization bearer
    int  connect_timeout_ms = 2000;
    int  request_timeout_ms = 0;    // 0 → no read timeout (long generations)

    // Spawn-mode knobs.  Ignored when spawn == false.
    bool spawn              = false;
    std::string python_bin  = "python3";
    int  spawn_ready_timeout_s = 60;
    std::string extra_args; // forwarded to vllm.entrypoints.openai.api_server
};

class VllmAdapter : public IRuntimeAdapter {
public:
    explicit VllmAdapter(VllmAdapterConfig cfg);
    ~VllmAdapter() override;

    std::string name() const override { return "vllm"; }
    std::string load_model(const std::string& model_path) override;
    std::string generate(const RuntimeRequest& req, ChunkCallback cb) override;
    void close() override;

    // Cheap GET /health probe — also called from load_model() to validate
    // the server is reachable.  Returns true on HTTP 200.
    bool probe(int timeout_ms = 1500);

    const VllmAdapterConfig& config() const { return cfg_; }

private:
    VllmAdapterConfig cfg_;
    std::string host_;
    uint16_t    port_ = 8000;
    std::string path_prefix_;
    std::string served_model_; // remembered from load_model

    // Owned child process when spawn=true.  -1 when not spawned.
#if !defined(_WIN32)
    int child_pid_ = -1;
#else
    void* child_handle_ = nullptr;
#endif
    std::mutex   spawn_mu_;
    std::atomic<bool> closed_{false};

    bool parse_base_url();
    std::string spawn_server(const std::string& model_path);
    std::string wait_for_ready(int timeout_s);
};

// Convenience: VllmAdapterConfig populated from the documented env vars.
VllmAdapterConfig vllm_config_from_env();

} // namespace dist
