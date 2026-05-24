// Fuzz harness for the VmNode tensor-write path.
//
// Models src/vm_node.cpp:handle_tensor_write exactly:
//
//   if (sz < sizeof(MsgVmTensorWrite)) return;
//   const auto& msg  = *reinterpret_cast<const MsgVmTensorWrite*>(payload);
//   uint32_t data_sz = sz - sizeof(MsgVmTensorWrite);
//   ...
//   auto& buf = ...;     // freshly emplaced std::vector<uint8_t>
//   if (msg.offset + data_sz > buf.size()) buf.resize(msg.offset + data_sz, 0);
//   if (data_sz > 0) std::memcpy(buf.data() + msg.offset,
//                                payload + sizeof(MsgVmTensorWrite), data_sz);
//
// The integer overflow is `msg.offset + data_sz` as uint32 — if it wraps to a
// small value, resize() does nothing and memcpy() writes at buf.data()+offset
// where offset can be 0xFFFFFFF0 → wild OOB write.
//
// Build:
//   clang++ -std=c++17 -O1 -g -fsanitize=fuzzer,address,undefined \
//     -I../../include audit/fuzz/fuzz_vm_tensor_write.cc -o fuzz_vm_tensor_write
//
// Run:
//   ./fuzz_vm_tensor_write -max_total_time=60

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>
#include <unordered_map>

#include "vm_protocol.h"

using namespace dist;

static std::unordered_map<uint64_t, std::vector<uint8_t>> tensors;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    // Limit total memory the fuzzer can ask the allocator for.  Don't let a
    // single legitimate giant allocation kill the fuzz session; we want OOB,
    // not OOM.  ASAN's per-allocation cap is ~10 GiB by default; we ride that.
    if (size < sizeof(MsgVmTensorWrite)) return 0;

    // Clear state to keep each iteration independent.
    tensors.clear();

    uint32_t sz = (uint32_t)size;
    const uint8_t* payload = data;

    const auto& msg = *reinterpret_cast<const MsgVmTensorWrite*>(payload);
    uint32_t data_sz = sz - sizeof(MsgVmTensorWrite);

    // Mirror the production code path verbatim.
    auto it = tensors.find(msg.vaddr);
    if (it == tensors.end()) {
        // Lazy alloc — clamp to avoid OOM in fuzz, but the production code
        // does no such clamp.  Without the clamp this OOMs the fuzzer in
        // ~half the inputs and we never get to the OOB write.  Keep clamp
        // tight so the OOB still triggers when offset is small.
        uint32_t alloc = (uint32_t)(msg.offset + data_sz);
        if (alloc > (1u << 20)) return 0; // > 1 MiB; skip
        tensors.emplace(msg.vaddr, std::vector<uint8_t>(alloc, 0));
        it = tensors.find(msg.vaddr);
    }
    auto& buf = it->second;
    if (msg.offset + data_sz > buf.size()) buf.resize(msg.offset + data_sz, 0);
    if (data_sz > 0) {
        std::memcpy(buf.data() + msg.offset,
                    payload + sizeof(MsgVmTensorWrite), data_sz);
    }
    return 0;
}
