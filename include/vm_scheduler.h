#pragma once
/**
 * vm_scheduler.h
 *
 * VmScheduler — distributed work dispatcher.
 *
 * Accepts op submissions, selects the best node using a data-locality +
 * load-balance policy, dispatches via VM_OP_DISPATCH, and returns
 * OpFuture handles that callers can wait on.
 *
 * Policy (in priority order):
 *   1. Data locality: prefer node that owns the most input tensor bytes.
 *   2. Load: break ties by lowest gpu_util.
 *   3. Capacity: break ties by most free_vram.
 *
 * Retry: on VM_OP_REJECT the op is re-dispatched to the next-best node
 * (up to MAX_RETRIES times, then the future is failed).
 */

#include "vm_protocol.h"
#include "vm_tensor_registry.h"
#include "dist_conn.h"
#include "dist_queue.h"

#include <atomic>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace dist {

static constexpr int VM_SCHED_MAX_RETRIES = 3;

// ─── OpFuture ────────────────────────────────────────────────────────────────

struct OpResult {
    bool                 success = false;
    std::vector<uint8_t> output;   // result tensor bytes
    std::string          error;
};

// Returned to callers of VmScheduler::submit().
// Internally backed by std::shared_future so it can be copied/shared.
struct OpFuture {
    uint64_t op_id = 0;

    bool is_ready() const {
        if (!fut_.valid()) return true;
        return fut_.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    }

    // Blocks until result or timeout. Returns nullopt on timeout.
    std::optional<OpResult> get(std::chrono::milliseconds timeout
                                    = std::chrono::milliseconds(30000)) {
        if (!fut_.valid()) return std::nullopt;
        if (fut_.wait_for(timeout) != std::future_status::ready) return std::nullopt;
        return fut_.get();
    }

    // Internal: constructed by VmScheduler
    explicit OpFuture(uint64_t id,
                      std::shared_future<OpResult> f)
        : op_id(id), fut_(std::move(f)) {}
    OpFuture() = default;

private:
    std::shared_future<OpResult> fut_;
};

// ─── VmScheduler ─────────────────────────────────────────────────────────────

class VmScheduler {
public:
    explicit VmScheduler(VmTensorRegistry& registry);

    // ── Node management ──────────────────────────────────────────────────────

    // Register a node with its current capabilities and a control connection.
    void add_node(const std::string& node_id,
                  const NodeCapability& cap,
                  std::shared_ptr<Connection> ctrl_conn);

    // Remove a node (e.g. on failure). In-flight ops for that node get retried.
    void remove_node(const std::string& node_id);

    // Update load metrics (called on heartbeat).
    void update_load(const std::string& node_id,
                     float gpu_util, uint64_t free_vram);

    // ── Op submission ────────────────────────────────────────────────────────

    // Submit an op to be dispatched to the best node.
    // input_vaddrs: virtual addresses of input tensors.
    // op_payload: serialised description of the op (VmOpType + params).
    // expected_output_bytes: hint for result buffer sizing.
    OpFuture submit(uint64_t request_id,
                    std::vector<uint64_t> input_vaddrs,
                    std::vector<uint8_t>  op_payload,
                    VmOpType              op_type,
                    uint32_t              expected_output_bytes = 0);

    // ── Result callbacks (called by VmCoordinator recv loop) ─────────────────

    // Node completed an op successfully.
    void on_result(uint64_t op_id,
                   std::vector<uint8_t> output_bytes);

    // Node rejected the op (queue full, OOM, etc.). Re-dispatch.
    void on_reject(uint64_t op_id, const std::string& rejecting_node);

    // Node failed (called by FaultManager). Fail/retry all its in-flight ops.
    void on_node_failed(const std::string& node_id);

    // ── Accessors ────────────────────────────────────────────────────────────

    size_t pending_op_count() const;
    std::vector<std::string> node_ids() const;

private:
    struct NodeStat {
        std::string                  id;
        NodeCapability               cap;
        std::shared_ptr<Connection>  ctrl_conn;
        float                        gpu_util     = 0.0f;
        uint64_t                     free_vram    = 0;
        uint32_t                     inflight_ops = 0;
        bool                         alive        = true;
    };

    struct PendingOp {
        uint64_t                     op_id;
        uint64_t                     request_id;
        std::vector<uint64_t>        input_vaddrs;
        std::vector<uint8_t>         op_payload;
        VmOpType                     op_type;
        uint32_t                     expected_output_bytes;
        std::string                  dispatched_to;
        int                          retry_count = 0;
        std::shared_ptr<std::promise<OpResult>> promise;
    };

    // Select the best available node for input_vaddrs.
    // Returns "" if no node is available.
    std::string select_node(const std::vector<uint64_t>& input_vaddrs,
                             const std::string& exclude = "");

    // Send VM_OP_DISPATCH to target_node.
    void dispatch_op(PendingOp& op, const std::string& target_node);

    mutable std::mutex                              mu_;
    VmTensorRegistry&                               registry_;
    std::atomic<uint64_t>                           next_op_id_ { 1 };
    std::unordered_map<std::string, NodeStat>        nodes_;
    std::unordered_map<uint64_t, PendingOp>          pending_;
};

} // namespace dist
