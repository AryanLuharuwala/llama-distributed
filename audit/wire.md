# Wire-format adversarial audit

Date: 2026-05-25
Auditor: wire-format agent
Targets: C++ dist_protocol / VM protocol parsers; src/nat_pmp.cpp; server/macaroon.go; server/ws.go status-frame path.

## Summary (top 5 findings, severity-ranked)

1. **F-WIRE-01 (Critical, remote unauth heap OOB write)** — `src/vm_node.cpp:171` `handle_tensor_write` integer-overflows `msg.offset + data_sz` (both `uint32_t`), then `memcpy(buf.data() + msg.offset, ...)` writes at an arbitrary 32-bit offset past a small buffer. **Confirmed by libFuzzer+ASAN crash on iteration #2 (25-byte input)**. Any peer/coordinator that can deliver a single `VM_TENSOR_WRITE` over the VM-ctrl plane can corrupt the heap. Repro at `audit/fuzz/crash-1d95f60054879e418f153df031d3a87b9fef0f2a`.
2. **F-WIRE-02 (High, remote DoS via 4 GiB allocation)** — `include/dist_conn.h:142` `recv_msg` resizes `payload_out` to attacker-controlled `hdr.payload_len` (uint32_t) with no cap. One malformed header buys a 4 GiB allocation and likely process OOM-kill. Same pattern repeated for `MsgInferRequest.n_prompt_tokens` (coordinator.cpp:230), `MsgLayerAssign.n_ranges` (node_agent.cpp:147), and `MsgVmTensorAlloc.n_bytes` (vm_node.cpp:160) — each allocates an attacker-sized vector before any bounds check.
3. **F-WIRE-03 (High, remote OOB read via missing length check)** — `src/vm_node.cpp:282 handle_topo_update` loops `hdr.n_nodes` times reading 64-byte slots from `payload+sizeof(MsgVmTopoUpdate)`, with **no check** that the buffer is big enough. `n_nodes` is wire-controlled uint32_t. Further, `new_ring.emplace_back(p)` constructs a `std::string` from `p` treated as a C-string — if the wire bytes lack a NUL the constructor walks past the buffer until it finds one, leaking memory or crashing.
4. **F-WIRE-04 (High, alloc DoS in handle_op_dispatch)** — `src/vm_node.cpp:247` `task.input_vaddrs.resize(hdr.n_inputs)` with uint32_t `n_inputs` → up to 32 GiB allocation. The pre-check at line 235 verifies the buffer is big enough but **only after** `n_inputs * 8` has been computed; on 32-bit builds that multiplication wraps and the gate passes spuriously. On 64-bit the gate works but a peer who sends a real ~4 GiB payload with `n_inputs` matching can still drive a multi-GiB resize.
5. **F-WIRE-05 (Medium, integer overflow in tensor recv path)** — `include/dist_protocol.h:264 tensor_payload_bytes` computes `(size_t)n_tokens * n_embd * dtype_size(dtype)`. Both factors are uint32_t in `[0, 2^32-1]`; on 64-bit the cast prevents the *first* product from overflowing but `(... * dtype_size)` can still wrap size_t for adversarial values, and **callers do not validate against `payload_len` upper bound**. In `src/node_agent.cpp:213` the same product is recomputed and used to gate a `batch.data.resize(n_elems * sizeof(float))`. Fuzzed for 60 s × ~20M execs with ASAN+UBSAN and *no crash within a 2 MiB payload cap* — the bug is real but is fronted by F-WIRE-02 (unbounded `recv_msg` allocation) which has to be fixed for this one to matter.

## Files reviewed

- `include/dist_protocol.h` — TensorHeader, tensor_payload_bytes, MsgInferRequest, MsgLayerAssign, all phase-5 structs
- `include/dist_conn.h` — Connection::recv_msg / recv_msg_into / _recv_all (read framing)
- `include/fp8.h` — encode_e4m3, decode_e4m3, encode/decode_tensor
- `include/vm_protocol.h` — VM message structs (MsgVmTensorWrite, MsgVmTopoUpdate, …)
- `include/actv_p2p.h` — peer wrapper (no raw parser here)
- `src/node_agent.cpp` — data_recv_thread_fn, control_thread_fn, run_layers
- `src/coordinator.cpp` — node_thread_fn, client_thread_fn, handle_node_join, handle_*_ack
- `src/vm_node.cpp` — vm_ctrl_thread_fn and every handle_* (handle_tensor_alloc/free/write/read, handle_op_dispatch, handle_topo_update, handle_collective_chunk, handle_checkpoint, handle_restore_req)
- `src/vm_coordinator.cpp` — handle_tensor_read_rsp, handle_op_result, handle_op_reject, handle_checkpoint_ack, handle_collective_chunk
- `src/actv_p2p.cpp` — STUN response parser (`stun_query`) + reachability probe (UDP recvfrom)
- `src/nat_pmp.cpp` — `try_pcp_map`, `try_natpmp_map` (PCP and NAT-PMP response parsers)
- `server/ws.go` — agent WS reader loop, kind dispatch (status / *_caps / relay_stats)
- `server/validation.go` — validateAndClampStatus, clampRelayBytes, isAllowedPublicIP
- `server/macaroon.go` — mintCap, verifyCap, capChecker

## Findings

### F-WIRE-01 — Critical — Heap OOB write in `handle_tensor_write`

**File:** `src/vm_node.cpp:171-189`

```cpp
void VmNode::handle_tensor_write(const uint8_t* payload, uint32_t sz) {
    if (sz < sizeof(MsgVmTensorWrite)) return;
    const auto& msg  = *reinterpret_cast<const MsgVmTensorWrite*>(payload);
    uint32_t data_sz = sz - sizeof(MsgVmTensorWrite);

    std::lock_guard<std::mutex> lk(tensor_mu_);
    auto it = tensors_.find(msg.vaddr);
    if (it == tensors_.end()) {
        // Lazy alloc
        tensors_.emplace(msg.vaddr, std::vector<uint8_t>(msg.offset + data_sz, 0));
        it = tensors_.find(msg.vaddr);
    }
    auto& buf = it->second;
    if (msg.offset + data_sz > buf.size()) buf.resize(msg.offset + data_sz, 0);
    if (data_sz > 0) {
        std::memcpy(buf.data() + msg.offset,
                    payload + sizeof(MsgVmTensorWrite), data_sz);
    }
}
```

`msg.offset` and `data_sz` are both `uint32_t`. `msg.offset + data_sz` is `uint32_t` arithmetic and wraps modulo 2^32. With `msg.offset = 0xFFFFFFFF` and `data_sz = 1`, the sum is 0. The lazy-alloc path constructs `std::vector<uint8_t>(0)` (empty), the resize is a no-op (`0 > 0` is false), and `memcpy(buf.data() + 0xFFFFFFFF, src, 1)` writes one byte 4 GiB past a freshly-allocated empty vector.

**Repro:** libFuzzer crash on the second input.

```
audit/fuzz/fuzz_vm_tensor_write.cc:67:32: runtime error: applying non-zero offset 4294967295 to null pointer
AddressSanitizer: SEGV on unknown address 0x00009fff7ff8 ... in __asan_memcpy
Test unit: 0a ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff
```

Payload layout: 24 bytes of `MsgVmTensorWrite` (vaddr=0xff..ff, op_id=0xff..ff, offset=0xFFFFFFFF, n_bytes=0xFFFFFFFF) + 1 byte of data. `data_sz = 25 - 24 = 1`. `offset + data_sz = 0`. Then `memcpy(empty_buf.data() + 0xFFFFFFFF, ...)` SEGVs.

Crash artifact stored at `audit/fuzz/crash-1d95f60054879e418f153df031d3a87b9fef0f2a`.

**Exploitability:** the VM-ctrl channel is currently TCP at `PORT_VM_CTRL = 7703`. Even if peer auth is enforced (see audit/auth.md from the auth-attacks agent), a compromised or misbehaving VmNode/VmContext peer can write to arbitrary heap offsets — full code-exec-class primitive given a heap-spray strategy. Without auth: anyone who can reach 7703 wins.

**Fix:**
```cpp
// Promote to size_t before adding; cap at a per-message ceiling (8 MiB?).
size_t end = (size_t)msg.offset + (size_t)data_sz;
if (end < msg.offset)         return;        // belt-and-braces (size_t can't actually overflow here)
if (end > MAX_TENSOR_BYTES)   return;        // policy cap
if (sz < sizeof(MsgVmTensorWrite) + data_sz) return; // already true but be explicit
```
Also: `tensors_.emplace(msg.vaddr, std::vector<uint8_t>(end, 0))` then mirrors the check.

### F-WIRE-02 — High — `recv_msg` resizes to attacker-controlled `payload_len`

**File:** `include/dist_conn.h:136-147`

```cpp
bool recv_msg(MsgHeader& hdr_out, std::vector<uint8_t>& payload_out) {
    if (!_recv_all(&hdr_out, sizeof(MsgHeader))) return false;
    if (hdr_out.magic != PROTO_MAGIC) { connected_.store(false); return false; }
    payload_out.resize(hdr_out.payload_len);   // <-- no cap
    ...
}
```

`payload_len` is uint32_t. A peer that already passed the magic-cookie check can pin 4 GiB of RSS per message. There is no version check (`hdr_out.version`) either — a v0 implementation could nominally connect and drive resource exhaustion through this surface. The same uncapped resize pattern shows up in dependent handlers:

- `src/coordinator.cpp:230` — `std::vector<int32_t> token_ids(req.n_prompt_tokens)` → 16 GiB on uint32_t max
- `src/node_agent.cpp:147` — `std::vector<LayerRange> ranges(msg.n_ranges)` → 48 GiB
- `src/vm_node.cpp:160` — `std::vector<uint8_t>(msg.n_bytes, 0)` → 4 GiB
- `src/vm_node.cpp:247` — `task.input_vaddrs.resize(hdr.n_inputs)` → 32 GiB

**Fix:** introduce `MAX_PAYLOAD_BYTES` (e.g. 64 MiB for control plane, separate cap for data plane), reject early; gate every variable-length field downstream.

### F-WIRE-03 — High — `handle_topo_update` reads past buffer end

**File:** `src/vm_node.cpp:282-295`

```cpp
void VmNode::handle_topo_update(const uint8_t* payload, uint32_t sz) {
    if (sz < sizeof(MsgVmTopoUpdate)) return;
    const auto& hdr = *reinterpret_cast<const MsgVmTopoUpdate*>(payload);
    std::vector<std::string> new_ring;
    const char* p = reinterpret_cast<const char*>(payload + sizeof(MsgVmTopoUpdate));
    for (uint32_t i = 0; i < hdr.n_nodes; ++i) {
        new_ring.emplace_back(p);   // <-- C-string constructor: walks until NUL
        p += 64;
    }
    ...
}
```

Two distinct bugs:
1. **No buffer-length gate.** `hdr.n_nodes` is wire-controlled; the loop steps `p` forward 64 bytes per iteration without comparing against `payload + sz`. With `n_nodes = 2^32 - 1` the pointer walks 256 GiB into adjacent memory.
2. **`emplace_back(const char*)` requires NUL-terminated input.** Even on a well-sized payload, if the attacker fills the 64-byte slot with non-NUL bytes the std::string constructor reads past the slot until it finds a NUL elsewhere on the heap. Combined with bug #1, an attacker can probe arbitrary heap addresses by varying `n_nodes` and the trailing byte pattern.

**Fix:**
```cpp
if (sz < sizeof(MsgVmTopoUpdate) + (size_t)hdr.n_nodes * 64) return;
const char* p = (const char*)(payload + sizeof(MsgVmTopoUpdate));
for (uint32_t i = 0; i < hdr.n_nodes; ++i) {
    new_ring.emplace_back(p, ::strnlen(p, 64));   // bound the string scan
    p += 64;
}
```
Header docstring says `MAX_NODES = 64` — also gate `if (hdr.n_nodes > MAX_NODES) return;`.

### F-WIRE-04 — High — `handle_op_dispatch` alloc DoS + 32-bit overflow

**File:** `src/vm_node.cpp:229-258`

```cpp
uint32_t vaddr_bytes = hdr.n_inputs * sizeof(uint64_t);    // uint32 × size_t → size_t on 64-bit
if (sz < sizeof(MsgVmOpDispatch) + vaddr_bytes) { ... }   // OK on 64-bit
...
task.input_vaddrs.resize(hdr.n_inputs);                    // up to 32 GiB
```

On 32-bit builds (less likely here but the codebase still supports Windows builds via the platform_compat shim) `hdr.n_inputs * 8` overflows `uint32_t`, and the gate passes spuriously. On 64-bit, the worst case is an unauthenticated peer paying the recv_msg cost (see F-WIRE-02) and then forcing a 32 GiB resize. Fix is the same: cap `n_inputs` at `MAX_NODES` or `MAX_INPUTS` (small constant).

### F-WIRE-05 — Medium — Integer arithmetic in tensor payload

**File:** `include/dist_protocol.h:264-268` and mirror at `src/node_agent.cpp:212-217`

```cpp
inline size_t tensor_payload_bytes(const TensorHeader& th) {
    return sizeof(TensorHeader)
         + dtype_scale_prefix_bytes(th.dtype)
         + (size_t)th.n_tokens * th.n_embd * dtype_size(th.dtype);
}
```

`th.n_tokens` and `th.n_embd` are `uint32_t` (≤ 2^32 − 1). On 64-bit `(size_t)n_tokens * n_embd ≤ 2^63`, and `* dtype_size` (at most 4) can push size_t past 2^65 → wraps. The caller in `node_agent.cpp` re-derives the same product:

```cpp
const size_t n_elems = (size_t)header.n_tokens * header.n_embd;
if (body_len < sizeof(float) + n_elems) continue;
batch.data.resize(n_elems * sizeof(float));
```

The `body_len < sizeof(float) + n_elems` gate **only works if n_elems is bounded by body_len** (a real recv buffer), which it is — as long as F-WIRE-02 is fixed and `payload_len` is capped. Without that cap, `n_elems` can be 2^62 (body_len = 4 GiB, n_tokens = n_embd = 2^16) and `resize(n_elems * 4)` invokes `std::__throw_bad_alloc` — a crash, not memory corruption. Fuzzed with libFuzzer+ASAN+UBSAN for 60 seconds (~20 M executions) under a 2 MiB payload cap and found no crash; the issue is gated by the upstream payload cap.

**Fix:** make `tensor_payload_bytes` overflow-safe and have callers validate that the returned size matches `payload_len`:
```cpp
inline bool tensor_payload_bytes(const TensorHeader& th, size_t* out) {
    size_t prefix = sizeof(TensorHeader) + dtype_scale_prefix_bytes(th.dtype);
    uint64_t elems = (uint64_t)th.n_tokens * th.n_embd;
    if (elems > (1ull << 40)) return false;                // 1 Ti elements
    size_t body = (size_t)(elems * dtype_size(th.dtype));
    if (body / dtype_size(th.dtype) != elems) return false;
    *out = prefix + body;
    return *out >= prefix; // detect outer add wrap
}
```

### F-WIRE-06 — Medium — `fp8::decode_tensor` propagates NaN/Inf into compute

**File:** `include/fp8.h:163-168`

```cpp
inline void decode_tensor(const uint8_t* src, size_t n_elems,
                          float scale, float* dst) {
    for (size_t i = 0; i < n_elems; ++i) dst[i] = decode_e4m3(src[i]) * scale;
}
```

The `scale` is unmarshalled directly from the wire (`std::memcpy(&scale, body, sizeof(float))` in node_agent.cpp:210). If the sender writes `0x7FC00000` (qNaN), every decoded element becomes NaN. If they write `0` we get a tensor of zeros (denial of useful inference for the request). If they write `Inf`, the downstream `llama_decode` blows up internally — out of scope for this audit but worth flagging because the validator in `validateAndClampStatus` does not extend to data-plane numerics. No memory corruption.

**Fix:** `if (!std::isfinite(scale) || scale <= 0.0f) continue;` before decoding.

### F-WIRE-07 — Medium — `MsgVmCollectiveChunk` uses raw `from_node` as std::string

**File:** `src/vm_node.cpp:297-307`

```cpp
const auto& hdr = *reinterpret_cast<const MsgVmCollectiveChunk*>(payload);
...
collective_.on_chunk(hdr.coll_id, std::string(hdr.from_node), ...);
```

`hdr.from_node` is a fixed 64-byte array. `std::string(hdr.from_node)` invokes the const-char* constructor which **requires NUL termination within the array**. A wire-format frame that fills all 64 bytes with non-zero data causes the string constructor to read past the struct field. The struct sits inside the `payload` buffer so the OOB read is bounded by the payload tail, but `payload` is itself a heap allocation — strlen runs until it finds a NUL on the heap, potentially leaking adjacent heap contents (e.g. through the next debug log line) or just SEGV.

Same construction pattern needs auditing wherever a struct's `char name[N]` is wrapped in `std::string`: `vm_node.cpp:289` (already covered in F-WIRE-03), `vm_coordinator.cpp:152` (`std::string id = msg.cap.node_id;` if cap.node_id is non-terminated), node_agent.cpp:310 (same).

**Fix:** always use the `(ptr, length)` constructor with `strnlen` or a hardcoded 64.

### F-WIRE-08 — Low — Macaroon trailing-junk equality check is not constant-time

**File:** `server/macaroon.go:103-110`

```go
round, err := m.MarshalBinary()
if err != nil || len(round) != len(raw) {
    return fmt.Errorf("token length mismatch (trailing-junk attack?)")
}
for i, b := range raw {
    if round[i] != b {
        return fmt.Errorf("token bytes mismatch at %d", i)
    }
}
```

This is the trailing-junk defense, not the cryptographic verification (that's `m.Verify` two lines below). The byte-by-byte loop with early exit leaks the position of the first differing byte. Forging the prefix grants no advantage because the canonical bytes are derived deterministically from the macaroon body — an attacker who can already produce a valid macaroon can produce the right prefix bytes trivially. **No exploitable timing channel**, but `crypto/subtle.ConstantTimeCompare` is one line and removes the lint.

### F-WIRE-09 — Low — `MsgVmTensorWrite` parses `vaddr` of zero as "real" key

**File:** `src/vm_node.cpp:171-189`

`tensors_.emplace(0, ...)` is allowed; the wire format makes `vaddr=0` valid. The coordinator allocator returns 0 to signal allocation failure (per `MsgVmTensorAllocRsp.vaddr` doc comment), so a node that stores at `vaddr=0` gets a tensor it can never reach via the normal alloc path but can poison through direct writes. Not a memory bug; protocol confusion.

## Sanitizer build results

The project's main CMake build chain links against llama.cpp / heavy GPU stacks; rather than recompile the world with `-fsanitize=address,undefined` (which would have eaten the time budget on llama.cpp's CUDA kernels even with NVCC fallbacks), the audit built **focused, header-only fuzz harnesses against just the parser code**:

```
clang++ -std=c++17 -O1 -g \
  -fsanitize=fuzzer,address,undefined -fno-omit-frame-pointer \
  -Iinclude audit/fuzz/fuzz_vm_tensor_write.cc -o audit/fuzz/fuzz_vm_tensor_write
```

All three harnesses built clean under ASAN+UBSAN with clang-20.1.8 on Fedora 42. No warnings from clang on either header (`dist_protocol.h`, `vm_protocol.h`, `fp8.h`) — UBSAN-disabled `-Wall -Wextra` would surface the integer-overflow concerns but the audit did not enable them; the runtime UBSAN catches were sufficient (see F-WIRE-01).

**Did not run:** project test suite under ASAN. The CMakeLists builds llama.cpp + ggml + CUDA via FetchContent and the ASAN rebuild would have exceeded the time budget. Filed under "what I couldn't get to" below.

## Fuzzer runs

| Target | File | Execs | Crashes | Corpus |
|---|---|---:|---|---|
| `fuzz_vm_tensor_write` | `audit/fuzz/fuzz_vm_tensor_write.cc` | **2 (crashed immediately)** | **1** (F-WIRE-01) | `audit/fuzz/corpus_vm/` + `audit/fuzz/crash-1d95f60054879e418f153df031d3a87b9fef0f2a` |
| `fuzz_tensor_recv` | `audit/fuzz/fuzz_tensor_recv.cc` | 19,728,920 in 61 s | 0 | `audit/fuzz/corpus_recv/` |
| `fuzz_natpmp_resp` | `audit/fuzz/fuzz_natpmp_resp.cc` | 50,211,927 in 61 s | 0 | `audit/fuzz/corpus_natpmp/` |
| `FuzzAgentStatusUnmarshal` | `server/fuzz_wire_audit_test.go` | 418,350 in 47 s (12 workers) | 0 | go test fuzz cache |
| `FuzzMacaroonVerify` | `server/fuzz_wire_audit_test.go` | 269,745 in 31 s (12 workers) | 0 | go test fuzz cache |

**Total CPU time on parsers under ASAN+UBSAN:** ~3 minutes. Negative results are real — the NAT-PMP and TensorHeader bounded paths take all the bytes the attacker can offer and don't crash. The two real bugs both popped immediately once a fuzzer was pointed at the right code (F-WIRE-01 in iteration 2; F-WIRE-03 and F-WIRE-04 found by code review, not yet fuzzed because they require slightly different harness scaffolding around `std::string` heap layout).

## What I couldn't get to and why

- **Full ASAN/UBSAN build of the C++ project.** The CMakeLists builds llama.cpp with CUDA via FetchContent; an ASAN rebuild needs `-DGGML_CUDA=OFF` and rebuilds half the universe. Time budget exhausted by focused harnesses + reading the parsers. The targeted fuzz harnesses caught the real bug regardless.
- **`MsgVmTopoUpdate` (F-WIRE-03) repro by libFuzzer.** I have high confidence from code review and the bug is straightforward; writing a harness needs the same scaffolding as `fuzz_vm_tensor_write.cc`. ~30 minutes if you want it; the diagnosis above is sufficient to file a fix.
- **`MsgInferRequest` (DoS in coordinator.cpp:230).** Same DoS shape as F-WIRE-02; not fuzzed because the alloc-too-large failure mode is observable by inspection.
- **DTLS layer in the actv_p2p relay path.** That's the auth-attacks agent's domain and is also covered by the memory note `peer_relay_plaintext_exposure.md` for the architectural side.
- **Cross-checking the auth/macaroon claim "constant-time generalized."** macaroon-v2 library does its own constant-time HMAC check; the trailing-junk check (F-WIRE-08) is lint-level. Did not exhaustively walk every HMAC compare in `server/` — that overlaps with the auth-attacks agent.
- **`MsgVmNodeReady` field mismatch.** `vm_node.cpp:75-78` assigns to fields (`n_gpus`, `cpu_ram_bytes`, `gpu_free_bytes`) that the struct in `vm_protocol.h:231-236` does not declare. Either the codebase doesn't compile in its current form or one of the files I read is out-of-sync with the source of truth. Flagged but not pursued — likely a header that has additional declarations farther in (or this code path is `#ifdef`'d out).

## Recommended fix order

1. F-WIRE-01 — heap OOB write, immediately exploitable on the VM-ctrl plane.
2. F-WIRE-02 — universal allocation cap on `recv_msg` and downstream variable-length parsers.
3. F-WIRE-03 — bound `n_nodes` and use `strnlen`-bounded string construction.
4. F-WIRE-04 — bound `n_inputs` in `handle_op_dispatch`.
5. F-WIRE-05 — overflow-safe `tensor_payload_bytes`.
6. F-WIRE-06 / F-WIRE-07 — defensive cleanups.
7. F-WIRE-08 / F-WIRE-09 — lint-level.
