#pragma once
//
// P14: TensorRT-LLM runtime adapter.
//
// TensorRT-LLM compiles a per-model engine plan and runs it from
// either its own C++ runtime API or via Triton Inference Server's
// `tensorrtllm_backend`.  The C++ runtime API is fastest but
// pulls in nvinfer / cudart / TRT-LLM static libs that are NVIDIA-
// only and heavy enough to make the build host pick up GPU
// dependencies it shouldn't need.  Triton, by contrast, is just a
// process that exposes a stable HTTP/gRPC interface (the operator
// runs it; we talk to it).  We pick Triton.
//
// The shape:
//
//   POST {base}/v2/models/{model}/generate_stream
//        Content-Type: application/json
//        {
//          "text_input": "<prompt>",
//          "parameters": {
//            "max_tokens": N, "temperature": T, "top_p": P,
//            "top_k": K, "stop_words": [...],
//            "stream": true
//          }
//        }
//   →  SSE  data: { "text_output": "<delta>", "model_name": "...",
//                   "finish_reason": "...", "finished": true|false }
//
// Triton's TRT-LLM backend emits one event per generated token
// (or per "n" tokens when batched).  We forward each event as a
// RuntimeChunk delta and treat `"finished": true` as the
// end-of-stream signal — Triton does *not* send "[DONE]".
//
// Path quirk: the served model name has to match the directory
// under Triton's model repository (e.g. "ensemble", "llama-3-8b").
// We take it from DIST_TRTLLM_MODEL (no env-derivable default).

#include "runtime_adapter.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

namespace dist {

struct TrtLlmAdapterConfig {
    std::string base_url   = "http://127.0.0.1:8000";
    std::string model_name = "ensemble";  // Triton model repo entry
    std::string api_key;
    int  connect_timeout_ms = 2000;
};

class TrtLlmAdapter : public IRuntimeAdapter {
public:
    explicit TrtLlmAdapter(TrtLlmAdapterConfig cfg);
    ~TrtLlmAdapter() override;

    std::string name() const override { return "trtllm"; }
    std::string load_model(const std::string& model_path) override;
    std::string generate(const RuntimeRequest& req, ChunkCallback cb) override;
    void close() override;

    bool probe(int timeout_ms = 1500);

    const TrtLlmAdapterConfig& config() const { return cfg_; }

private:
    TrtLlmAdapterConfig cfg_;
    std::string host_;
    uint16_t    port_ = 8000;
    std::string path_prefix_;
    std::atomic<bool> closed_{false};

    bool parse_base_url();
};

TrtLlmAdapterConfig trtllm_config_from_env();

} // namespace dist
