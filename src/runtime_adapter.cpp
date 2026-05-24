// runtime_adapter.cpp — factory and env-driven selection for the
// IRuntimeAdapter family.  See include/runtime_adapter.h for the design
// notes; this file owns the small bits of glue that don't belong in
// either the header or in any one adapter's TU.
//
// vLLM, SGLang, and TRT-LLM-via-Triton adapters are wired here (P12 – P14).
// The LlamaCpp kind still returns nullptr, which is the documented signal
// that the caller takes the inline pp_engine path.

#include "runtime_adapter.h"
#include "vllm_adapter.h"
#include "sglang_adapter.h"
#include "trtllm_adapter.h"

#include <algorithm>
#include <cctype>
#include <iostream>

namespace dist {

namespace {

std::string to_lower(std::string s) {
    for (auto& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

} // namespace

RuntimeKind runtime_kind_from_env(const std::string& env_value) {
    std::string v = to_lower(env_value);
    if (v.empty() || v == "llama-cpp" || v == "llama" || v == "llamacpp")
        return RuntimeKind::LlamaCpp;
    if (v == "vllm") return RuntimeKind::VLLM;
    if (v == "sglang") return RuntimeKind::SGLang;
    if (v == "trtllm" || v == "tensorrt-llm" || v == "tensorrtllm")
        return RuntimeKind::TRTLLM;
    std::cerr << "[runtime] unknown DIST_RUNTIME='" << env_value
              << "', falling back to llama-cpp\n";
    return RuntimeKind::LlamaCpp;
}

std::unique_ptr<IRuntimeAdapter> make_runtime_adapter(RuntimeKind kind) {
    switch (kind) {
        case RuntimeKind::LlamaCpp:
            // Still served inline through pp_engine + node_agent; the
            // LlamaCppAdapter wrapper class is a follow-up.  Returning
            // nullptr is the documented signal that callers must take
            // the legacy path.
            return nullptr;
        case RuntimeKind::VLLM:
            return std::make_unique<VllmAdapter>(vllm_config_from_env());
        case RuntimeKind::SGLang:
            return std::make_unique<SglangAdapter>(sglang_config_from_env());
        case RuntimeKind::TRTLLM:
            return std::make_unique<TrtLlmAdapter>(trtllm_config_from_env());
    }
    return nullptr;
}

} // namespace dist
