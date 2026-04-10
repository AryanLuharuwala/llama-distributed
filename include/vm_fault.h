#pragma once
/**
 * vm_fault.h
 *
 * FaultManager — checkpoint, recovery, and node-failure handling.
 *
 * Checkpoint protocol (2-phase commit):
 *   1. Coordinator calls begin_checkpoint().
 *   2. FaultManager broadcasts VM_CHECKPOINT_SNAP to all live nodes.
 *   3. Each node persists its local tensor store and replies VM_CHECKPOINT_ACK.
 *   4. Once all ACKs arrive (or timeout), the checkpoint is marked committed
 *      and the promise resolves.
 *
 * Node failure recovery:
 *   1. VmCoordinator calls on_node_failed(node_id).
 *   2. FaultManager calls VmTensorRegistry::migrate_all_from() to reassign
 *      tensor ownership records to surviving nodes.
 *   3. FaultManager calls VmScheduler::on_node_failed() so in-flight ops
 *      are retried on other nodes.
 *   4. FaultManager attempts to restore tensors from the last committed
 *      checkpoint by sending VM_RESTORE_REQ to the new owning nodes.
 *
 * Topology change notifications:
 *   After any membership change (join or fail) FaultManager broadcasts
 *   VM_TOPO_UPDATE to all surviving nodes so their collective rings update.
 */

#include "vm_protocol.h"
#include "vm_tensor_registry.h"
#include "vm_scheduler.h"
#include "dist_conn.h"

#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dist {

// ─── CheckpointFuture ────────────────────────────────────────────────────────

struct CheckpointResult {
    bool        success  = false;
    uint64_t    snap_id  = 0;
    std::string error;
};

struct CheckpointFuture {
    uint64_t snap_id = 0;

    bool is_done() const {
        if (!fut_.valid()) return true;
        return fut_.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    }

    std::optional<CheckpointResult> wait(
            std::chrono::milliseconds timeout = std::chrono::milliseconds(60000)) {
        if (!fut_.valid()) return std::nullopt;
        if (fut_.wait_for(timeout) != std::future_status::ready) return std::nullopt;
        return fut_.get();
    }

    explicit CheckpointFuture(uint64_t id, std::shared_future<CheckpointResult> f)
        : snap_id(id), fut_(std::move(f)) {}
    CheckpointFuture() = default;

private:
    std::shared_future<CheckpointResult> fut_;
};

// ─── FaultManager ────────────────────────────────────────────────────────────

class FaultManager {
public:
    /**
     * get_conn      : callback → Connection* for a node_id (or nullptr).
     * registry      : shared tensor registry (FaultManager updates ownership).
     * scheduler     : shared scheduler (FaultManager triggers op retries).
     */
    using GetConnFn       = std::function<Connection*(const std::string& node_id)>;
    using NodeListFn      = std::function<std::vector<std::string>()>;
    using SendTopoFn      = std::function<void(const std::vector<std::string>& live_nodes)>;

    FaultManager(GetConnFn      get_conn,
                 NodeListFn     live_nodes_fn,
                 VmTensorRegistry& registry,
                 VmScheduler&      scheduler);

    // ── Checkpoint ───────────────────────────────────────────────────────────

    // Initiate a checkpoint across all live nodes.
    // Returns a future that resolves when all nodes ACK (or on timeout/failure).
    CheckpointFuture begin_checkpoint();

    // Called by the VmCoordinator recv loop when a VM_CHECKPOINT_ACK arrives.
    void on_checkpoint_ack(uint64_t snap_id, const std::string& node_id);

    // ── Node lifecycle ───────────────────────────────────────────────────────

    // Called when a new node joins. Broadcasts updated topology to all nodes.
    void on_node_joined(const std::string& node_id);

    // Called when a node is detected dead (heartbeat timeout or disconnect).
    // Migrates tensor ownership, retries ops, broadcasts new topology.
    void on_node_failed(const std::string& node_id);

    // ── Last committed checkpoint ─────────────────────────────────────────────

    // Returns the snap_id of the last fully committed checkpoint (0 = none).
    uint64_t last_committed_snap_id() const;

    // Returns the tensor snapshot from the last committed checkpoint.
    std::vector<TensorDesc> last_snapshot() const;

private:
    struct PendingCheckpoint {
        uint64_t                                     snap_id;
        std::unordered_set<std::string>              waiting_for; // nodes that haven't ACKed
        std::vector<TensorDesc>                      snap;        // registry snapshot at issue time
        std::shared_ptr<std::promise<CheckpointResult>> promise;
    };

    // Send VM_TOPO_UPDATE to all live nodes.
    void broadcast_topo_update();

    // Send VM_RESTORE_REQ to new owning node for each migrated tensor.
    void send_restore_requests(
            const std::vector<std::pair<uint64_t, std::string>>& migrations);

    GetConnFn          get_conn_;
    NodeListFn         live_nodes_fn_;
    VmTensorRegistry&  registry_;
    VmScheduler&       scheduler_;

    mutable std::mutex mu_;

    std::atomic<uint64_t>                                next_snap_id_ { 1 };
    std::unordered_map<uint64_t, PendingCheckpoint>      pending_snaps_;

    // Last committed checkpoint
    uint64_t                 committed_snap_id_ = 0;
    std::vector<TensorDesc>  committed_snap_;
};

} // namespace dist
