/**
 * vm_coordinator.cpp
 *
 * VmCoordinator implementation.
 *
 * Startup sequence:
 *   1. VmCoordinator::run() starts the base Coordinator in a thread.
 *   2. vm_ctrl_accept_fn() listens on PORT_VM_CTRL; each VmNode connects
 *      here after it has connected to the base Coordinator.
 *   3. When a VM_NODE_READY arrives we register the node with the registry
 *      and the scheduler, and notify the FaultManager.
 *
 * VM message flow:
 *   API client (VmContext) → VmCoordinator::alloc_tensor() etc.
 *   VmCoordinator → VM_TENSOR_WRITE / VM_OP_DISPATCH → VmNode (via vm_conns_)
 *   VmNode → VM_OP_RESULT / VM_TENSOR_READ_RSP → VmCoordinator recv thread
 *   VmCoordinator resolves waiting promises / futures.
 */

#include "vm_coordinator.h"

#include <cstring>
#include <iostream>
#include <stdexcept>

namespace dist {

// ─── Constructor ─────────────────────────────────────────────────────────────

VmCoordinator::VmCoordinator(VmCoordinatorConfig cfg)
    : cfg_(std::move(cfg))
    , coordinator_(cfg_.base)
    , scheduler_(registry_)
    , collective_(
          // The coordinator does NOT participate in the data ring.
          // "coordinator" pseudo-node id for completeness.
          "coordinator",
          [this](const std::string& nid) -> Connection* { return get_vm_conn(nid); })
    , fault_(
          [this](const std::string& nid) -> Connection* { return get_vm_conn(nid); },
          [this]() { return live_nodes(); },
          registry_,
          scheduler_)
{}

VmCoordinator::~VmCoordinator() { stop(); }

// ─── Run / Stop ──────────────────────────────────────────────────────────────

void VmCoordinator::run() {
    running_.store(true);

    // Bind VM ctrl listener
    vm_ctrl_listener_.bind_and_listen(cfg_.bind_host, cfg_.vm_ctrl_port);
    std::cout << "[VmCoordinator] VM ctrl port " << cfg_.vm_ctrl_port << "\n";

    vm_ctrl_accept_thread_ = std::thread([this]{ vm_ctrl_accept_fn(); });

    // Run base coordinator (blocks)
    coordinator_.run();
}

void VmCoordinator::stop() {
    if (!running_.exchange(false)) return;
    coordinator_.stop();
    vm_ctrl_listener_.close();
    if (vm_ctrl_accept_thread_.joinable()) vm_ctrl_accept_thread_.join();
    for (auto& t : vm_ctrl_node_threads_) if (t.joinable()) t.join();
}

// ─── VM ctrl listener ─────────────────────────────────────────────────────────

void VmCoordinator::vm_ctrl_accept_fn() {
    while (running_.load()) {
        auto conn = vm_ctrl_listener_.accept_one();
        if (!conn) continue;

        // Wait for VM_NODE_READY to know the node_id
        std::vector<uint8_t> buf;
        MsgType mtype;
        if (!conn->recv_msg(mtype, buf) ||
            static_cast<VmMsgType>(mtype) != VmMsgType::VM_NODE_READY) {
            continue;
        }
        if (buf.size() < sizeof(MsgVmNodeReady)) continue;
        const auto& rdy = *reinterpret_cast<const MsgVmNodeReady*>(buf.data());
        std::string node_id(rdy.node_id);

        auto conn_ptr = std::make_shared<Connection>(std::move(*conn));
        {
            std::lock_guard<std::mutex> lk(vm_conn_mu_);
            vm_conns_[node_id] = conn_ptr;
        }

        handle_node_ready(node_id, rdy);

        std::string nid_copy = node_id;
        auto raw_conn = conn_ptr;
        vm_ctrl_node_threads_.emplace_back(
            [this, nid_copy, raw_conn]{ vm_ctrl_node_fn(nid_copy, raw_conn); });
    }
}

void VmCoordinator::vm_ctrl_node_fn(std::string node_id,
                                     std::shared_ptr<Connection> conn) {
    while (running_.load() && conn->is_connected()) {
        std::vector<uint8_t> buf;
        MsgType mtype;
        if (!conn->recv_msg(mtype, buf)) break;

        auto vm_type = static_cast<VmMsgType>(mtype);
        const uint8_t* p = buf.data();
        uint32_t sz      = (uint32_t)buf.size();

        switch (vm_type) {
        case VmMsgType::VM_TENSOR_READ_RSP:
            handle_tensor_read_rsp(node_id, p, sz);
            break;
        case VmMsgType::VM_OP_RESULT:
            handle_op_result(node_id, p, sz);
            break;
        case VmMsgType::VM_OP_REJECT:
            handle_op_reject(node_id, p, sz);
            break;
        case VmMsgType::VM_CHECKPOINT_ACK:
            handle_checkpoint_ack(node_id, p, sz);
            break;
        case VmMsgType::VM_COLLECTIVE_CHUNK:
            handle_collective_chunk(node_id, p, sz);
            break;
        default:
            break;
        }
    }

    // Node disconnected
    std::cout << "[VmCoordinator] VM ctrl disconnect: " << node_id << "\n";
    fault_.on_node_failed(node_id);
    registry_.unregister_node(node_id);
    scheduler_.remove_node(node_id);
    {
        std::lock_guard<std::mutex> lk(vm_conn_mu_);
        vm_conns_.erase(node_id);
    }
}

// ─── Message handlers ────────────────────────────────────────────────────────

void VmCoordinator::handle_node_ready(const std::string& node_id,
                                       const MsgVmNodeReady& msg) {
    NodeCapability cap{};
    cap.n_gpus = msg.n_gpus;
    for (uint32_t i = 0; i < msg.n_gpus && i < 8; ++i)
        cap.gpu_free_bytes[i] = msg.gpu_free_bytes[i];
    cap.cpu_ram_bytes = msg.cpu_ram_bytes;

    registry_.register_node(node_id);
    scheduler_.add_node(node_id, cap, get_vm_conn(node_id)
        ? std::shared_ptr<Connection>(std::shared_ptr<Connection>{}, get_vm_conn(node_id))
        : nullptr);
    fault_.on_node_joined(node_id);

    std::cout << "[VmCoordinator] node ready: " << node_id
              << " gpus=" << msg.n_gpus << "\n";
}

void VmCoordinator::handle_tensor_read_rsp(const std::string& /*node_id*/,
                                            const uint8_t* payload, uint32_t sz) {
    if (sz < sizeof(MsgVmTensorReadRsp)) return;
    const auto& rsp = *reinterpret_cast<const MsgVmTensorReadRsp*>(payload);
    uint32_t data_bytes = sz - sizeof(MsgVmTensorReadRsp);
    std::vector<uint8_t> data(payload + sizeof(MsgVmTensorReadRsp),
                               payload + sizeof(MsgVmTensorReadRsp) + data_bytes);

    std::lock_guard<std::mutex> lk(read_mu_);
    auto it = read_promises_.find(rsp.vaddr);
    if (it != read_promises_.end()) {
        it->second->set_value(std::move(data));
        read_promises_.erase(it);
    }
}

void VmCoordinator::handle_op_result(const std::string& node_id,
                                      const uint8_t* payload, uint32_t sz) {
    if (sz < sizeof(MsgVmOpResult)) return;
    const auto& msg = *reinterpret_cast<const MsgVmOpResult*>(payload);
    uint32_t data_bytes = sz - sizeof(MsgVmOpResult);
    std::vector<uint8_t> output(payload + sizeof(MsgVmOpResult),
                                 payload + sizeof(MsgVmOpResult) + data_bytes);
    scheduler_.on_result(msg.op_id, std::move(output));

    // Update load metrics from the result message
    scheduler_.update_load(node_id, msg.gpu_util, msg.free_vram);
}

void VmCoordinator::handle_op_reject(const std::string& node_id,
                                      const uint8_t* payload, uint32_t sz) {
    if (sz < sizeof(MsgVmOpReject)) return;
    const auto& msg = *reinterpret_cast<const MsgVmOpReject*>(payload);
    scheduler_.on_reject(msg.op_id, node_id);
}

void VmCoordinator::handle_checkpoint_ack(const std::string& node_id,
                                           const uint8_t* payload, uint32_t sz) {
    if (sz < sizeof(MsgVmCheckpointAck)) return;
    const auto& msg = *reinterpret_cast<const MsgVmCheckpointAck*>(payload);
    fault_.on_checkpoint_ack(msg.snap_id, node_id);
}

void VmCoordinator::handle_collective_chunk(const std::string& node_id,
                                             const uint8_t* payload, uint32_t sz) {
    if (sz < sizeof(MsgVmCollectiveChunk)) return;
    const auto& hdr = *reinterpret_cast<const MsgVmCollectiveChunk*>(payload);
    uint32_t chunk_bytes = sz - sizeof(MsgVmCollectiveChunk);
    collective_.on_chunk(hdr.coll_id,
                         node_id,
                         hdr.phase,
                         hdr.chunk_offset,
                         payload + sizeof(MsgVmCollectiveChunk),
                         chunk_bytes);
}

// ─── Public VM ops ───────────────────────────────────────────────────────────

uint64_t VmCoordinator::alloc_tensor(uint32_t n_bytes, uint16_t dtype,
                                      const std::string& preferred_node) {
    uint64_t vaddr = registry_.alloc(n_bytes, dtype, preferred_node);
    if (!vaddr) return 0;

    std::string owner = registry_.owning_node(vaddr);
    Connection* conn  = get_vm_conn(owner);
    if (!conn || !conn->is_connected()) return vaddr; // owner will allocate on first write

    MsgVmTensorAlloc msg{};
    msg.vaddr   = vaddr;
    msg.n_bytes = n_bytes;
    msg.dtype   = dtype;

    try {
        conn->send_msg(static_cast<MsgType>(VmMsgType::VM_TENSOR_ALLOC),
                       reinterpret_cast<uint8_t*>(&msg), sizeof(msg));
    } catch (...) {}

    return vaddr;
}

void VmCoordinator::free_tensor(uint64_t vaddr) {
    std::string owner = registry_.owning_node(vaddr);
    Connection* conn  = get_vm_conn(owner);
    if (conn && conn->is_connected()) {
        MsgVmTensorFree msg{};
        msg.vaddr = vaddr;
        try {
            conn->send_msg(static_cast<MsgType>(VmMsgType::VM_TENSOR_FREE),
                           reinterpret_cast<uint8_t*>(&msg), sizeof(msg));
        } catch (...) {}
    }
    registry_.free(vaddr);
}

bool VmCoordinator::write_tensor(uint64_t vaddr, const uint8_t* data,
                                  uint32_t n_bytes) {
    std::string owner = registry_.owning_node(vaddr);
    Connection* conn  = get_vm_conn(owner);
    if (!conn || !conn->is_connected()) return false;

    // Payload: MsgVmTensorWrite header + data bytes
    uint32_t total = sizeof(MsgVmTensorWrite) + n_bytes;
    std::vector<uint8_t> buf(total);
    auto& hdr     = *reinterpret_cast<MsgVmTensorWrite*>(buf.data());
    hdr.vaddr     = vaddr;
    hdr.offset    = 0;
    hdr.n_bytes   = n_bytes;
    if (n_bytes > 0) std::memcpy(buf.data() + sizeof(MsgVmTensorWrite), data, n_bytes);

    try {
        conn->send_msg(static_cast<MsgType>(VmMsgType::VM_TENSOR_WRITE),
                       buf.data(), total);
        return true;
    } catch (...) {
        return false;
    }
}

std::vector<uint8_t> VmCoordinator::read_tensor(uint64_t vaddr,
                                                  std::chrono::milliseconds timeout) {
    std::string owner = registry_.owning_node(vaddr);
    Connection* conn  = get_vm_conn(owner);
    if (!conn || !conn->is_connected()) return {};

    auto promise = std::make_shared<std::promise<std::vector<uint8_t>>>();
    auto fut     = promise->get_future();

    {
        std::lock_guard<std::mutex> lk(read_mu_);
        read_promises_[vaddr] = promise;
    }

    MsgVmTensorRead msg{};
    msg.vaddr   = vaddr;
    msg.offset  = 0;
    msg.n_bytes = 0; // 0 = read all

    try {
        conn->send_msg(static_cast<MsgType>(VmMsgType::VM_TENSOR_READ),
                       reinterpret_cast<uint8_t*>(&msg), sizeof(msg));
    } catch (...) {
        std::lock_guard<std::mutex> lk(read_mu_);
        read_promises_.erase(vaddr);
        return {};
    }

    if (fut.wait_for(timeout) != std::future_status::ready) {
        std::lock_guard<std::mutex> lk(read_mu_);
        read_promises_.erase(vaddr);
        return {};
    }
    return fut.get();
}

OpFuture VmCoordinator::submit_op(uint64_t request_id,
                                   std::vector<uint64_t> input_vaddrs,
                                   std::vector<uint8_t>  op_payload,
                                   VmOpType              op_type,
                                   uint32_t              expected_output_bytes) {
    return scheduler_.submit(request_id, std::move(input_vaddrs),
                             std::move(op_payload), op_type, expected_output_bytes);
}

CollectiveFuture VmCoordinator::all_reduce(uint64_t coll_id,
                                            const std::vector<std::string>& participants,
                                            std::vector<uint8_t> data,
                                            uint16_t dtype,
                                            ReduceOp reduce_op) {
    // The coordinator initiates via CollectiveEngine; each node receives
    // VM_COLLECTIVE_INIT (sent by VmNode) and participates on its own.
    // This returns a future that resolves when the coordinator's own
    // "virtual participation" completes (or immediately if it's a passthrough).
    return collective_.all_reduce(coll_id, participants, std::move(data),
                                  dtype, reduce_op);
}

CollectiveFuture VmCoordinator::broadcast(uint64_t coll_id,
                                           const std::vector<std::string>& participants,
                                           const std::string& root_node,
                                           std::vector<uint8_t> data,
                                           uint16_t dtype) {
    return collective_.broadcast(coll_id, participants, root_node,
                                 std::move(data), dtype);
}

CheckpointFuture VmCoordinator::checkpoint() {
    return fault_.begin_checkpoint();
}

std::vector<std::string> VmCoordinator::live_nodes() const {
    return registry_.live_nodes();
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

Connection* VmCoordinator::get_vm_conn(const std::string& node_id) {
    std::lock_guard<std::mutex> lk(vm_conn_mu_);
    auto it = vm_conns_.find(node_id);
    if (it == vm_conns_.end()) return nullptr;
    return it->second.get();
}

} // namespace dist
