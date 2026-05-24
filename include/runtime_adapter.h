#pragma once
//
// P12: Pluggable inference-runtime adapter.
//
// dist-node ships with llama.cpp as the default backend — see pp_engine.{h,cpp}
// and node_agent.cpp.  llama.cpp is the right call for the project's core
// use case: pipeline-parallel inference across heterogeneous rigs, with
// activations flowing over the wire as bf16/fp16 tensors.  But on rigs
// where the whole model fits in local VRAM, the OSS world's
// high-throughput single-node runtime is vLLM (PagedAttention, continuous
// batching, prefix caching).  vLLM is a *single-rig* runtime; it can't do
// pipeline parallelism across networks.  So the right way to compose them
// is to keep llama.cpp for the multi-rig pipeline path and add vLLM as a
// *runtime adapter* that the node selects when it has been assigned the
// entire model and the user (or operator) has opted in.
//
// This header defines the interface every adapter implements.  Concrete
// adapters live in:
//
//   - src/pp_engine.cpp                  — implicit llama.cpp adapter
//                                          (still inline; will be lifted
//                                          to a LlamaCppAdapter class as
//                                          a follow-up so the rest of
//                                          the agent code stops branching
//                                          on the runtime type)
//   - src/vllm_adapter.cpp               — vLLM HTTP-bridge adapter
//                                          (this PR)
//   - src/sglang_adapter.cpp (P13)       — SGLang adapter, similar HTTP
//                                          bridge but with prompt-cache
//                                          metadata in the response
//   - src/trtllm_adapter.cpp (P14)       — TensorRT-LLM adapter, in-process
//                                          via the C++ runtime API
//
// Design notes:
//
//   1. The interface is **string-in, token-stream-out**.  vLLM/SGLang/TRT
//      all expose this shape natively; llama.cpp can be wrapped to
//      match it.  This means the adapter can't sit *inside* a pipeline
//      stage — it must own the whole forward path.  That's the
//      intended split: pipeline-parallel runs use llama.cpp; full-model
//      runs can pick any adapter.
//
//   2. Token IDs use llama.cpp's `llama_token` type (int32) for
//      compatibility with the rest of the code that processes outputs.
//      Adapters that return only text reproduce-tokenize via the
//      shared tokenizer that came with the GGUF.
//
//   3. Streaming uses a callback rather than an iterator so we can
//      cancel mid-stream by returning `false` from the callback.
//      This matches the cancellation shape /v1/chat/completions
//      already has on the server side.

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace dist {

// ─── Request shape ────────────────────────────────────────────────────

struct RuntimeRequest {
    // Prompt: either a raw string (adapter tokenizes) or pre-tokenized
    // ids.  When both are set, prompt_tokens wins — the string is kept
    // around only for log/trace context.
    std::string             prompt;
    std::vector<int32_t>    prompt_tokens;

    // Sampling.  Defaults give "deterministic + reasonable".
    int     max_tokens          = 256;
    float   temperature         = 0.7f;
    float   top_p               = 1.0f;
    int     top_k               = 40;
    float   repetition_penalty  = 1.0f;
    int     seed                = -1;  // -1 → adapter picks per-request

    // Optional stop-string list.  Matched against the *decoded* string,
    // not token ids, because vLLM/SGLang only accept strings here.
    std::vector<std::string> stop;

    // User-facing request id, used for log correlation across the
    // server log line + the runtime's own logs.  Free-form short string.
    std::string request_id;
};

// ─── Streaming output ─────────────────────────────────────────────────

// RuntimeChunk is one delta emitted as the model decodes.  Adapters
// must call the callback at least once per token (or per text fragment
// for adapters that don't expose token ids).
//
// Returning false from the callback aborts the request — the adapter
// must stop decoding and return promptly.  This is the cancellation
// path used by SSE client disconnects.
struct RuntimeChunk {
    std::string         text;       // decoded text since the last chunk
    int32_t             token_id;   // -1 when adapter only exposes text
    bool                final;      // last chunk in the stream
    std::string         finish_reason; // "stop" | "length" | "cancelled" | ""
};

using ChunkCallback = std::function<bool(const RuntimeChunk&)>;

// ─── Adapter interface ────────────────────────────────────────────────

// IRuntimeAdapter is the minimum surface a runtime backend must provide.
// All methods are blocking unless documented otherwise; concurrency is
// up to the caller (dist-node serializes per-request, so adapters don't
// need to be reentrant on a single instance).
class IRuntimeAdapter {
public:
    virtual ~IRuntimeAdapter() = default;

    // human-readable name for logs and the /api/runtimes endpoint.
    virtual std::string name() const = 0;

    // Load (or attach to) a model at `model_path`.  Path semantics are
    // adapter-specific: llama.cpp wants a GGUF, vLLM wants either an HF
    // repo id or a local directory with safetensors + config.  Caller
    // must call load_model exactly once before generate.
    //
    // Returns empty string on success, error message on failure.
    virtual std::string load_model(const std::string& model_path) = 0;

    // Stream a completion for `req` into `cb`.  The callback may be
    // invoked from a worker thread owned by the adapter.  Returns the
    // final finish_reason on success, error message prefixed with
    // "error:" on failure.
    virtual std::string generate(const RuntimeRequest& req,
                                 ChunkCallback cb) = 0;

    // Best-effort shutdown.  Adapters that own a subprocess (vLLM,
    // SGLang) terminate it here; in-process adapters tear down their
    // model handle.  Called from the node-agent's shutdown path.
    virtual void close() = 0;
};

// ─── Factory ──────────────────────────────────────────────────────────

// Runtime selection comes from the environment so the operator can pick
// per-rig without recompiling.  DIST_RUNTIME=llama-cpp (default) | vllm
// | sglang | trtllm.  Unknown values log a warning and fall back to
// llama.cpp.
enum class RuntimeKind {
    LlamaCpp,
    VLLM,
    SGLang,
    TRTLLM,
};

RuntimeKind runtime_kind_from_env(const std::string& env_value);

// Construct an adapter of the requested kind.  Returns nullptr if the
// adapter isn't compiled into this build (e.g., TRT-LLM not available
// on the build host) — callers must handle the nullptr by falling back
// to the default llama.cpp inline path.
std::unique_ptr<IRuntimeAdapter> make_runtime_adapter(RuntimeKind kind);

} // namespace dist
