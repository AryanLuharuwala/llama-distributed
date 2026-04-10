#pragma once
/**
 * vm_context.h
 *
 * VmContext — client-facing handle to the distributed virtual machine.
 *
 * VmContext is the single object a user program creates to interact with
 * the cluster.  It talks to a VmCoordinator over TCP (PORT_VM_CTRL) and
 * exposes a clean, synchronous API:
 *
 *   VmContext ctx("coordinator-host");
 *
 *   // Allocate a 1-MiB float32 tensor anywhere in the cluster
 *   uint64_t w = ctx.alloc(1<<20, DTYPE_F32);
 *
 *   // Write initial weights
 *   ctx.write(w, data.data(), data.size());
 *
 *   // Submit a matrix-multiply op
 *   auto fut = ctx.submit_op(VmOpType::MatMul, {w, x}, {});
 *   auto res = fut.get();
 *
 *   // All-reduce a gradient across all nodes
 *   auto cfut = ctx.all_reduce(data, DTYPE_F32, ReduceOp::Sum);
 *   auto cres = cfut.wait();
 *
 *   // Trigger a checkpoint
 *   ctx.checkpoint().wait();
 *
 *   // Run inference end-to-end through the pipeline layer
 *   ctx.infer(token_ids, max_tokens, [](int32_t tok){ ... });
 *
 * All blocking calls have a configurable timeout (default 30 s).
 * The internal connection is kept alive; reconnection is attempted once
 * on any send/recv failure.
 */

#include "vm_protocol.h"
#include "vm_scheduler.h"
#include "vm_collective.h"
#include "vm_fault.h"
#include "dist_conn.h"

#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace dist {

// ─── Convenience aliases ─────────────────────────────────────────────────────

using TokenStreamCb = std::function<void(int32_t token_id, bool is_last)>;

struct VmContextConfig {
    std::string coordinator_host;
    uint16_t    vm_ctrl_port  = PORT_VM_CTRL;
    uint16_t    api_port      = PORT_API;          // for infer requests
    std::chrono::milliseconds default_timeout { 30000 };
};

// ─── VmContext ────────────────────────────────────────────────────────────────

class VmContext {
public:
    explicit VmContext(VmContextConfig cfg);
    ~VmContext();

    // ── Tensor management ────────────────────────────────────────────────────

    // Allocate tensor in the cluster. Returns vaddr (0 on failure).
    uint64_t alloc(uint32_t n_bytes, uint16_t dtype,
                   const std::string& preferred_node = "");

    // Free a previously allocated tensor.
    void free(uint64_t vaddr);

    // Write host data into a cluster tensor.
    bool write(uint64_t vaddr, const uint8_t* data, uint32_t n_bytes);

    // Read cluster tensor data to host. Returns empty on failure/timeout.
    std::vector<uint8_t> read(uint64_t vaddr,
        std::chrono::milliseconds timeout = std::chrono::milliseconds(30000));

    // ── Compute ops ──────────────────────────────────────────────────────────

    // Submit a distributed op. Returns a future for the output bytes.
    OpFuture submit_op(VmOpType op_type,
                       std::vector<uint64_t> input_vaddrs,
                       std::vector<uint8_t>  op_payload,
                       uint32_t              expected_output_bytes = 0);

    // ── Collective ops ────────────────────────────────────────────────────────

    // All-reduce data across all nodes. Returns a future for the reduced data.
    CollectiveFuture all_reduce(std::vector<uint8_t> data,
                                uint16_t dtype,
                                ReduceOp reduce_op = ReduceOp::Sum);

    // Broadcast data from this context to all nodes.
    CollectiveFuture broadcast(std::vector<uint8_t> data, uint16_t dtype);

    // ── Fault tolerance ───────────────────────────────────────────────────────

    // Trigger a cluster-wide checkpoint. Returns a future for the result.
    CheckpointFuture checkpoint();

    // ── Inference ────────────────────────────────────────────────────────────

    // Run autoregressive inference end-to-end.
    // Blocks until max_tokens generated or EOS. Calls cb for each token.
    void infer(const std::vector<int32_t>& prompt_tokens,
               uint32_t max_tokens,
               TokenStreamCb cb,
               std::chrono::milliseconds timeout = std::chrono::milliseconds(120000));

    // ── Cluster info ──────────────────────────────────────────────────────────

    std::vector<std::string> live_nodes();

private:
    // ── Connection management ─────────────────────────────────────────────────
    bool ensure_connected();
    void reconnect();

    // ── Recv thread ───────────────────────────────────────────────────────────
    void recv_thread_fn();

    // ── Message handlers ─────────────────────────────────────────────────────
    void on_tensor_alloc_rsp (const uint8_t* p, uint32_t sz);
    void on_tensor_read_rsp  (const uint8_t* p, uint32_t sz);
    void on_op_result        (const uint8_t* p, uint32_t sz);
    void on_op_reject        (const uint8_t* p, uint32_t sz);
    void on_checkpoint_done  (const uint8_t* p, uint32_t sz);
    void on_collective_done  (const uint8_t* p, uint32_t sz);
    void on_infer_token      (const uint8_t* p, uint32_t sz);
    void on_infer_done       (const uint8_t* p, uint32_t sz);

    VmContextConfig cfg_;
    std::atomic<bool> running_  { false };
    std::atomic<uint64_t> next_op_id_   { 1 };
    std::atomic<uint64_t> next_coll_id_ { 1 };
    std::atomic<uint64_t> next_req_id_  { 1 };

    // ── VM ctrl connection ────────────────────────────────────────────────────
    mutable std::mutex                 conn_mu_;
    std::unique_ptr<Connection>        vm_conn_;

    // ── Pending read promises (vaddr → promise) ───────────────────────────────
    mutable std::mutex                                              read_mu_;
    std::unordered_map<uint64_t,
        std::shared_ptr<std::promise<std::vector<uint8_t>>>>        read_promises_;

    // ── Pending op promises (op_id → promise) ────────────────────────────────
    mutable std::mutex                                              op_mu_;
    std::unordered_map<uint64_t,
        std::shared_ptr<std::promise<OpResult>>>                    op_promises_;

    // ── Pending checkpoint promises (snap_id → promise) ───────────────────────
    mutable std::mutex                                              ckpt_mu_;
    std::unordered_map<uint64_t,
        std::shared_ptr<std::promise<CheckpointResult>>>            ckpt_promises_;

    // ── Pending collective promises (coll_id → promise) ───────────────────────
    mutable std::mutex                                              coll_mu_;
    std::unordered_map<uint64_t,
        std::shared_ptr<std::promise<CollectiveResult>>>            coll_promises_;

    // ── Pending infer stream (req_id → callback + done_promise) ──────────────
    struct InferPending {
        TokenStreamCb cb;
        std::shared_ptr<std::promise<void>> done_promise;
    };
    mutable std::mutex                                              infer_mu_;
    std::unordered_map<uint64_t, InferPending>                      infer_pending_;

    // ── Recv thread ───────────────────────────────────────────────────────────
    std::thread recv_thread_;
};

} // namespace dist
