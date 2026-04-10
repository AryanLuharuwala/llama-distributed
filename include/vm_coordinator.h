#pragma once
/**
 * vm_coordinator.h
 *
 * VmCoordinator — wraps the base Coordinator and adds the VM control plane.
 *
 * Responsibilities:
 *   1. Owns VmTensorRegistry, VmScheduler, CollectiveEngine, FaultManager.
 *   2. Listens on PORT_VM_CTRL (7703) for VM control messages from nodes:
 *        VM_TENSOR_ALLOC / FREE / WRITE / READ
 *        VM_OP_RESULT / VM_OP_REJECT
 *        VM_CHECKPOINT_ACK
 *        VM_NODE_READY
 *   3. Exposes alloc_tensor / free_tensor / write_tensor / read_tensor /
 *      submit_op / all_reduce / broadcast / checkpoint to callers (VmContext).
 *   4. Hooks into Coordinator::remove_dead_node() via vm_hook_ callback
 *      to trigger FaultManager::on_node_failed().
 *
 * Design:
 *   Composition: holds Coordinator by value (not inheritance).
 *   vm_ctrl_listener_ runs on PORT_VM_CTRL; each node that joins via the
 *   base Coordinator also connects on PORT_VM_CTRL so VmCoordinator can
 *   forward VM messages to them.
 *
 *   The data plane for collectives (VM_COLLECTIVE_CHUNK) goes directly
 *   node-to-node via PORT_DATA — the coordinator is NOT in that path.
 */

#include "coordinator.h"
#include "vm_protocol.h"
#include "vm_tensor_registry.h"
#include "vm_scheduler.h"
#include "vm_collective.h"
#include "vm_fault.h"
#include "dist_conn.h"

#include <atomic>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace dist {

struct VmCoordinatorConfig {
    CoordinatorConfig base;            // forwarded to Coordinator
    uint16_t          vm_ctrl_port = PORT_VM_CTRL;
    std::string       bind_host    = "0.0.0.0";
};

class VmCoordinator {
public:
    explicit VmCoordinator(VmCoordinatorConfig cfg);
    ~VmCoordinator();

    void run();   // blocks until stop()
    void stop();

    // ── VM tensor ops (called by VmContext via the API plane) ─────────────────

    // Allocate a virtual tensor. Returns vaddr (0 on failure).
    uint64_t alloc_tensor(uint32_t n_bytes, uint16_t dtype,
                          const std::string& preferred_node = "");

    // Free a virtual tensor.
    void free_tensor(uint64_t vaddr);

    // Write bytes into a virtual tensor.  Sends VM_TENSOR_WRITE to owning node.
    bool write_tensor(uint64_t vaddr, const uint8_t* data, uint32_t n_bytes);

    // Read bytes from a virtual tensor. Sends VM_TENSOR_READ, awaits response.
    std::vector<uint8_t> read_tensor(uint64_t vaddr,
        std::chrono::milliseconds timeout = std::chrono::milliseconds(30000));

    // Submit a compute op. Returns OpFuture.
    OpFuture submit_op(uint64_t request_id,
                       std::vector<uint64_t> input_vaddrs,
                       std::vector<uint8_t>  op_payload,
                       VmOpType              op_type,
                       uint32_t              expected_output_bytes = 0);

    // AllReduce across all live nodes. Returns collective future.
    CollectiveFuture all_reduce(uint64_t coll_id,
                                const std::vector<std::string>& participants,
                                std::vector<uint8_t> data,
                                uint16_t dtype,
                                ReduceOp reduce_op = ReduceOp::Sum);

    // Broadcast from root to all. Returns collective future.
    CollectiveFuture broadcast(uint64_t coll_id,
                               const std::vector<std::string>& participants,
                               const std::string& root_node,
                               std::vector<uint8_t> data,
                               uint16_t dtype);

    // Trigger a cluster-wide checkpoint.
    CheckpointFuture checkpoint();

    // Live node list from registry.
    std::vector<std::string> live_nodes() const;

    // Next collective ID (monotonically increasing).
    uint64_t next_coll_id() { return next_coll_id_.fetch_add(1); }

private:
    // ── VM ctrl server ───────────────────────────────────────────────────────
    void vm_ctrl_accept_fn();
    void vm_ctrl_node_fn(std::string node_id, std::shared_ptr<Connection> conn);

    // ── Message handlers ─────────────────────────────────────────────────────
    void handle_node_ready      (const std::string& node_id, const MsgVmNodeReady& msg);
    void handle_tensor_alloc_rsp(const std::string& node_id, const uint8_t* payload, uint32_t sz);
    void handle_tensor_read_rsp (const std::string& node_id, const uint8_t* payload, uint32_t sz);
    void handle_op_result       (const std::string& node_id, const uint8_t* payload, uint32_t sz);
    void handle_op_reject       (const std::string& node_id, const uint8_t* payload, uint32_t sz);
    void handle_checkpoint_ack  (const std::string& node_id, const uint8_t* payload, uint32_t sz);
    void handle_collective_chunk(const std::string& node_id, const uint8_t* payload, uint32_t sz);

    // ── Helpers ──────────────────────────────────────────────────────────────
    Connection* get_vm_conn(const std::string& node_id);

    VmCoordinatorConfig cfg_;
    std::atomic<bool>   running_  { false };

    // ── Base coordinator ─────────────────────────────────────────────────────
    Coordinator coordinator_;

    // ── VM subsystems ────────────────────────────────────────────────────────
    VmTensorRegistry  registry_;
    VmScheduler       scheduler_;
    CollectiveEngine  collective_;
    FaultManager      fault_;

    // ── VM ctrl connections (one per node) ───────────────────────────────────
    mutable std::mutex                                   vm_conn_mu_;
    std::unordered_map<std::string, std::shared_ptr<Connection>> vm_conns_;

    // ── Pending read requests (vaddr → promise for read data) ────────────────
    mutable std::mutex                                          read_mu_;
    std::unordered_map<uint64_t, std::shared_ptr<std::promise<std::vector<uint8_t>>>> read_promises_;

    // ── VM ctrl listener ─────────────────────────────────────────────────────
    Listener     vm_ctrl_listener_;
    std::thread  vm_ctrl_accept_thread_;
    std::vector<std::thread> vm_ctrl_node_threads_;

    std::atomic<uint64_t> next_coll_id_ { 1 };
};

} // namespace dist
