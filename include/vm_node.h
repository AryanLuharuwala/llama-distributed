#pragma once
/**
 * vm_node.h
 *
 * VmNode — wraps NodeAgent and adds the VM control plane on each worker node.
 *
 * Responsibilities:
 *   1. Run the base NodeAgent (pipeline stage logic, layer execution).
 *   2. Connect to VmCoordinator on PORT_VM_CTRL and send VM_NODE_READY.
 *   3. Listen for VM control messages:
 *        VM_TENSOR_ALLOC / FREE / WRITE / READ  — manage local tensor store
 *        VM_OP_DISPATCH                          — execute a distributed op
 *        VM_CHECKPOINT_SNAP                      — persist local state and ACK
 *        VM_RESTORE_REQ                          — restore tensor from snapshot
 *        VM_TOPO_UPDATE                          — update collective ring
 *   4. Participate in collective operations (AllReduce etc.) via
 *      CollectiveEngine, which routes VM_COLLECTIVE_CHUNK directly to peers.
 *
 * Local tensor store:
 *   A simple map from vaddr → vector<uint8_t> holding the raw bytes.
 *   Tensors live in CPU RAM here; the ggml execution path (exec_op) copies
 *   data to GPU for computation and back after.
 *
 * Threading:
 *   base NodeAgent threads (control, data_recv, compute, data_send, hb)
 *   vm_ctrl_thread_  : recv/dispatch VM control messages from coordinator
 *   vm_exec_thread_  : execute VM_OP_DISPATCH tasks off the critical path
 */

#include "node_agent.h"
#include "vm_protocol.h"
#include "vm_collective.h"
#include "dist_conn.h"
#include "dist_queue.h"

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace dist {

struct VmNodeConfig {
    NodeAgentConfig base;

    std::string vm_coordinator_host;
    uint16_t    vm_ctrl_port      = PORT_VM_CTRL;

    uint32_t    exec_queue_depth  = 8;  // max queued ops before back-pressure
};

// A VM op task queued for the exec thread
struct VmOpTask {
    uint64_t              op_id;
    uint64_t              request_id;
    VmOpType              op_type;
    std::vector<uint64_t> input_vaddrs;
    std::vector<uint8_t>  op_payload;
    uint32_t              n_output_bytes;
};

class VmNode {
public:
    explicit VmNode(VmNodeConfig cfg);
    ~VmNode();

    void run();   // blocks until stop()
    void stop();

private:
    // ── VM ctrl ──────────────────────────────────────────────────────────────
    void vm_ctrl_thread_fn();
    void vm_exec_thread_fn();

    // ── Message handlers ─────────────────────────────────────────────────────
    void handle_tensor_alloc(const uint8_t* payload, uint32_t sz);
    void handle_tensor_free (const uint8_t* payload, uint32_t sz);
    void handle_tensor_write(const uint8_t* payload, uint32_t sz);
    void handle_tensor_read (const uint8_t* payload, uint32_t sz);
    void handle_op_dispatch (const uint8_t* payload, uint32_t sz);
    void handle_checkpoint  (const uint8_t* payload, uint32_t sz);
    void handle_restore_req (const uint8_t* payload, uint32_t sz);
    void handle_topo_update (const uint8_t* payload, uint32_t sz);
    void handle_collective_chunk(const uint8_t* payload, uint32_t sz);

    // ── Op execution ─────────────────────────────────────────────────────────
    std::vector<uint8_t> exec_op(const VmOpTask& task);

    // ── Checkpoint / restore ──────────────────────────────────────────────────
    void persist_snapshot(uint64_t snap_id);
    void restore_from_snap(uint64_t snap_id, uint64_t vaddr,
                           uint32_t n_bytes, uint16_t dtype);

    // ── Collective peer lookup ────────────────────────────────────────────────
    Connection* get_peer_conn(const std::string& peer_id);

    // ── Helpers ──────────────────────────────────────────────────────────────
    void send_op_result(uint64_t op_id, const std::vector<uint8_t>& output);
    void send_op_reject(uint64_t op_id);
    NodeCapability build_vm_capability() const;

    VmNodeConfig cfg_;
    NodeAgent    agent_;

    std::atomic<bool> running_ { false };

    // ── VM ctrl connection to coordinator ────────────────────────────────────
    std::unique_ptr<Connection> vm_ctrl_conn_;

    // ── Local tensor store (vaddr → raw bytes) ───────────────────────────────
    mutable std::mutex                                  tensor_mu_;
    std::unordered_map<uint64_t, std::vector<uint8_t>>  tensors_;

    // ── Snapshot store (snap_id → tensor map) ────────────────────────────────
    mutable std::mutex                                                     snap_mu_;
    std::unordered_map<uint64_t,
        std::unordered_map<uint64_t, std::vector<uint8_t>>>               snapshots_;

    // ── Peer connections for collective ──────────────────────────────────────
    mutable std::mutex                                   peer_mu_;
    std::unordered_map<std::string, std::unique_ptr<Connection>> peer_conns_;

    // ── Live ring for collective ──────────────────────────────────────────────
    std::mutex                  ring_mu_;
    std::vector<std::string>    ring_;    // ordered node list from last TOPO_UPDATE

    // ── Collective engine ─────────────────────────────────────────────────────
    CollectiveEngine collective_;

    // ── Op exec queue ─────────────────────────────────────────────────────────
    BoundedQueue<VmOpTask> exec_queue_;

    // ── Threads ──────────────────────────────────────────────────────────────
    std::thread vm_ctrl_thread_;
    std::thread vm_exec_thread_;
};

} // namespace dist
