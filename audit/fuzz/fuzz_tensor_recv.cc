// Fuzz the TENSOR_FORWARD receive path in src/node_agent.cpp:data_recv_thread_fn.
//
// Modelled exactly:
//
//   if (payload.size() < sizeof(TensorHeader)) continue;
//   batch.header = *reinterpret_cast<const TensorHeader*>(payload.data());
//   const uint8_t* body     = payload.data() + sizeof(TensorHeader);
//   const size_t   body_len = payload.size() - sizeof(TensorHeader);
//   if (batch.header.dtype == 3) {
//       if (body_len < sizeof(float)) continue;
//       float scale; std::memcpy(&scale, body, sizeof(float));
//       const uint8_t* fp8_bytes = body + sizeof(float);
//       const size_t n_elems = (size_t)header.n_tokens * header.n_embd;
//       if (body_len < sizeof(float) + n_elems) continue;
//       batch.data.resize(n_elems * sizeof(float));    // <-- integer overflow?
//       fp8::decode_tensor(fp8_bytes, n_elems, scale, dst);
//   } else {
//       batch.data.assign(body, body + body_len);
//   }
//
// Concerns: (1) n_elems = uint32 * uint32 promoted to size_t — safe on 64-bit;
// (2) n_elems * 4 in resize can hit size_t max if n_elems ~ 2^62 (impossible
// here since the body_len check bounds n_elems by remaining bytes); (3) the
// decode_tensor loop runs n_elems times — n_elems is bounded by body_len so
// no infinite loop.
//
// What's *not* modelled here: the upstream recv_msg() in dist_conn.h does
// `payload.resize(hdr.payload_len)` with no cap.  That's a separate
// finding — see audit/wire.md.

#include <cstdint>
#include <cstring>
#include <vector>

#include "dist_protocol.h"
#include "fp8.h"

using namespace dist;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < sizeof(TensorHeader)) return 0;
    // Cap input size so the fuzzer doesn't fight us on alloc — 2 MiB is plenty
    // to find any OOB or integer overflow in the bounded path.
    if (size > (2u << 20)) return 0;

    std::vector<uint8_t> payload(data, data + size);

    TensorHeader header;
    std::memcpy(&header, payload.data(), sizeof(TensorHeader));

    const uint8_t* body     = payload.data() + sizeof(TensorHeader);
    const size_t   body_len = payload.size() - sizeof(TensorHeader);

    std::vector<uint8_t> out_data;
    if (header.dtype == 3) {
        if (body_len < sizeof(float)) return 0;
        float scale;
        std::memcpy(&scale, body, sizeof(float));
        const uint8_t* fp8_bytes = body + sizeof(float);
        const size_t   n_elems   = (size_t)header.n_tokens * header.n_embd;
        if (body_len < sizeof(float) + n_elems) return 0;
        out_data.resize(n_elems * sizeof(float));
        float* dst = reinterpret_cast<float*>(out_data.data());
        fp8::decode_tensor(fp8_bytes, n_elems, scale, dst);
    } else {
        out_data.assign(body, body + body_len);
    }

    // Touch the output to keep the optimiser honest.
    if (!out_data.empty()) {
        volatile uint8_t s = 0;
        s ^= out_data.front();
        s ^= out_data.back();
        (void)s;
    }
    return 0;
}
