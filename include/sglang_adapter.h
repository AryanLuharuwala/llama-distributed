#pragma once
//
// P13: SGLang runtime adapter.
//
// SGLang ships a Python OpenAI-compatible server like vLLM, plus its
// native `/generate` endpoint that surfaces prompt-prefix cache hits
// in the response `meta_info`.  We use the native endpoint so the
// control plane can route follow-up turns to the same rig that
// already holds the prefix in its radix-tree cache — that's what
// makes SGLang shine for multi-turn chat workloads.
//
// Native /generate request (streaming):
//   POST {base}/generate
//     {
//       "text": "<prompt>",
//       "sampling_params": {
//         "max_new_tokens": N, "temperature": T, "top_p": P,
//         "top_k": K, "repetition_penalty": R, "stop": [...]
//       },
//       "stream": true
//     }
//   →  SSE  data: { "text": "...delta...",
//                   "meta_info": {
//                     "prompt_tokens": M,
//                     "completion_tokens": N,
//                     "cached_tokens": K,         ← prefix-cache hit count
//                     "finish_reason": { "type": "stop"|"length" }
//                   } }
//      ... terminated by  data: [DONE]
//
// The adapter exposes the latest `cached_tokens` value from a stream
// via `last_cached_tokens()` so the agent can surface it on the
// inference-finished frame; the server uses that to update its
// per-rig prefix-cache score.

#include "runtime_adapter.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

namespace dist {

struct SglangAdapterConfig {
    std::string base_url   = "http://127.0.0.1:30000";
    std::string api_key;
    int  connect_timeout_ms = 2000;

    // Spawn-mode knobs.  When spawn=true the adapter forks
    //   python -m sglang.launch_server --model-path <model_path> ...
    bool spawn              = false;
    std::string python_bin  = "python3";
    int  spawn_ready_timeout_s = 90;  // SGLang warmup is heavier than vLLM
    std::string extra_args;
};

class SglangAdapter : public IRuntimeAdapter {
public:
    explicit SglangAdapter(SglangAdapterConfig cfg);
    ~SglangAdapter() override;

    std::string name() const override { return "sglang"; }
    std::string load_model(const std::string& model_path) override;
    std::string generate(const RuntimeRequest& req, ChunkCallback cb) override;
    void close() override;

    bool probe(int timeout_ms = 1500);

    // Most recent `meta_info.cached_tokens` observed from a stream.
    // Zero on a fresh request that didn't hit the radix cache.  The
    // control plane uses this for prefix-aware routing scores.
    int64_t last_cached_tokens() const { return last_cached_tokens_.load(); }
    int64_t last_prompt_tokens() const { return last_prompt_tokens_.load(); }
    int64_t last_completion_tokens() const { return last_completion_tokens_.load(); }

    const SglangAdapterConfig& config() const { return cfg_; }

private:
    SglangAdapterConfig cfg_;
    std::string host_;
    uint16_t    port_ = 30000;
    std::string path_prefix_;
    std::string served_model_;

#if !defined(_WIN32)
    int child_pid_ = -1;
#else
    void* child_handle_ = nullptr;
#endif
    std::mutex spawn_mu_;
    std::atomic<bool>     closed_{false};
    std::atomic<int64_t>  last_cached_tokens_{0};
    std::atomic<int64_t>  last_prompt_tokens_{0};
    std::atomic<int64_t>  last_completion_tokens_{0};

    bool parse_base_url();
    std::string spawn_server(const std::string& model_path);
    std::string wait_for_ready(int timeout_s);
};

SglangAdapterConfig sglang_config_from_env();

} // namespace dist
