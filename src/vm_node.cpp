/**
 * vm_node.cpp
 *
 * VmNode implementation.
 *
 * Startup:
 *   1. NodeAgent::run() launches in a detached thread.
 *   2. VmNode connects to VmCoordinator on PORT_VM_CTRL.
 *   3. Sends VM_NODE_READY with local GPU capability.
 *   4. vm_ctrl_thread_ enters the message loop.
 *   5. vm_exec_thread_ processes queued VmOpTasks.
 *
 * Op execution (exec_op):
 *   For each VmOpType, fetches input tensor bytes from local store,
 *   runs a simple CPU fallback (production would use ggml kernels / CUDA).
 *   Returns output bytes.
 *
 * Checkpoint:
 *   On VM_CHECKPOINT_SNAP, takes a deep copy of tensors_ under lock,
 *   stores it in snapshots_[snap_id], and replies VM_CHECKPOINT_ACK.
 *
 * Collective:
 *   CollectiveEngine is driven by:
 *     - VmNode calling collective_.all_reduce() etc. when a collective op
 *       arrives in exec_op() (triggered by VM_OP_DISPATCH with a special op_type).
 *     - vm_ctrl_thread_ calling collective_.on_chunk() when
 *       VM_COLLECTIVE_CHUNK arrives from coordinator (which received it from
 *       another VmNode).
 */

#include "vm_node.h"

#include <cassert>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace dist {

// ─── Constructor ─────────────────────────────────────────────────────────────

VmNode::VmNode(VmNodeConfig cfg)
    : cfg_(std::move(cfg))
    , agent_(cfg_.base)
    , collective_(
          cfg_.base.node_id,
          [this](const std::string& peer) -> Connection* {
              return get_peer_conn(peer);
          })
    , exec_queue_(cfg_.exec_queue_depth)
{}

VmNode::~VmNode() { stop(); }

// ─── Run / Stop ──────────────────────────────────────────────────────────────

void VmNode::run() {
    running_.store(true);

    // Connect to VmCoordinator VM ctrl port
    vm_ctrl_conn_ = std::make_unique<Connection>();
    try {
        vm_ctrl_conn_->connect(cfg_.vm_coordinator_host, cfg_.vm_ctrl_port);
    } catch (std::exception& e) {
        std::cerr << "[VmNode] Failed to connect to VmCoordinator VM ctrl: "
                  << e.what() << "\n";
        // Continue anyway; base NodeAgent still functions
    }

    if (vm_ctrl_conn_->is_connected()) {
        // Send VM_NODE_READY
        MsgVmNodeReady rdy{};
        std::strncpy(rdy.node_id, cfg_.base.node_id.c_str(), sizeof(rdy.node_id) - 1);
        NodeCapability cap = build_vm_capability();
        rdy.n_gpus        = cap.n_gpus;
        rdy.cpu_ram_bytes = cap.cpu_ram_bytes;
        for (uint32_t i = 0; i < cap.n_gpus && i < 8; ++i)
            rdy.gpu_free_bytes[i] = cap.gpu_free_bytes[i];

        try {
            vm_ctrl_conn_->send_msg(static_cast<MsgType>(VmMsgType::VM_NODE_READY),
                                    reinterpret_cast<uint8_t*>(&rdy), sizeof(rdy));
        } catch (...) {}
    }

    vm_ctrl_thread_ = std::thread([this]{ vm_ctrl_thread_fn(); });
    vm_exec_thread_ = std::thread([this]{ vm_exec_thread_fn(); });

    // Run base NodeAgent (blocks)
    agent_.run();
}

void VmNode::stop() {
    if (!running_.exchange(false)) return;
    agent_.stop();
    exec_queue_.close();
    if (vm_ctrl_conn_) vm_ctrl_conn_.reset();
    if (vm_ctrl_thread_.joinable()) vm_ctrl_thread_.join();
    if (vm_exec_thread_.joinable())  vm_exec_thread_.join();
}

// ─── VM ctrl thread ───────────────────────────────────────────────────────────

void VmNode::vm_ctrl_thread_fn() {
    while (running_.load() && vm_ctrl_conn_ && vm_ctrl_conn_->is_connected()) {
        std::vector<uint8_t> buf;
        MsgType mtype;
        if (!vm_ctrl_conn_->recv_msg(mtype, buf)) break;

        auto vm_type = static_cast<VmMsgType>(mtype);
        const uint8_t* p = buf.data();
        uint32_t sz      = (uint32_t)buf.size();

        switch (vm_type) {
        case VmMsgType::VM_TENSOR_ALLOC:  handle_tensor_alloc(p, sz); break;
        case VmMsgType::VM_TENSOR_FREE:   handle_tensor_free(p, sz);  break;
        case VmMsgType::VM_TENSOR_WRITE:  handle_tensor_write(p, sz); break;
        case VmMsgType::VM_TENSOR_READ:   handle_tensor_read(p, sz);  break;
        case VmMsgType::VM_OP_DISPATCH:   handle_op_dispatch(p, sz);  break;
        case VmMsgType::VM_CHECKPOINT_SNAP: handle_checkpoint(p, sz); break;
        case VmMsgType::VM_RESTORE_REQ:   handle_restore_req(p, sz);  break;
        case VmMsgType::VM_TOPO_UPDATE:   handle_topo_update(p, sz);  break;
        case VmMsgType::VM_COLLECTIVE_CHUNK: handle_collective_chunk(p, sz); break;
        default: break;
        }
    }
}

// ─── Op exec thread ───────────────────────────────────────────────────────────

void VmNode::vm_exec_thread_fn() {
    while (running_.load()) {
        VmOpTask task;
        if (!exec_queue_.pop(task)) break;

        std::vector<uint8_t> output;
        bool ok = true;
        try {
            output = exec_op(task);
        } catch (std::exception& e) {
            std::cerr << "[VmNode] exec_op failed: " << e.what() << "\n";
            ok = false;
        }

        if (ok) {
            send_op_result(task.op_id, output);
        } else {
            send_op_reject(task.op_id);
        }
    }
}

// ─── Message handlers ────────────────────────────────────────────────────────

void VmNode::handle_tensor_alloc(const uint8_t* payload, uint32_t sz) {
    if (sz < sizeof(MsgVmTensorAlloc)) return;
    const auto& msg = *reinterpret_cast<const MsgVmTensorAlloc*>(payload);

    std::lock_guard<std::mutex> lk(tensor_mu_);
    tensors_.emplace(msg.vaddr, std::vector<uint8_t>(msg.n_bytes, 0));
}

void VmNode::handle_tensor_free(const uint8_t* payload, uint32_t sz) {
    if (sz < sizeof(MsgVmTensorFree)) return;
    const auto& msg = *reinterpret_cast<const MsgVmTensorFree*>(payload);

    std::lock_guard<std::mutex> lk(tensor_mu_);
    tensors_.erase(msg.vaddr);
}

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

void VmNode::handle_tensor_read(const uint8_t* payload, uint32_t sz) {
    if (sz < sizeof(MsgVmTensorRead)) return;
    const auto& msg = *reinterpret_cast<const MsgVmTensorRead*>(payload);

    std::vector<uint8_t> data;
    {
        std::lock_guard<std::mutex> lk(tensor_mu_);
        auto it = tensors_.find(msg.vaddr);
        if (it != tensors_.end()) {
            uint32_t off   = msg.offset;
            uint32_t nbytes = (msg.n_bytes == 0)
                              ? (uint32_t)it->second.size()
                              : msg.n_bytes;
            if (off < it->second.size()) {
                nbytes = std::min(nbytes, (uint32_t)it->second.size() - off);
                data.assign(it->second.begin() + off,
                            it->second.begin() + off + nbytes);
            }
        }
    }

    // Send VM_TENSOR_READ_RSP
    uint32_t rsp_total = sizeof(MsgVmTensorReadRsp) + (uint32_t)data.size();
    std::vector<uint8_t> rsp_buf(rsp_total);
    auto& rsp_hdr   = *reinterpret_cast<MsgVmTensorReadRsp*>(rsp_buf.data());
    rsp_hdr.vaddr   = msg.vaddr;
    rsp_hdr.n_bytes = (uint32_t)data.size();
    if (!data.empty())
        std::memcpy(rsp_buf.data() + sizeof(MsgVmTensorReadRsp),
                    data.data(), data.size());

    try {
        vm_ctrl_conn_->send_msg(
            static_cast<MsgType>(VmMsgType::VM_TENSOR_READ_RSP),
            rsp_buf.data(), rsp_total);
    } catch (...) {}
}

void VmNode::handle_op_dispatch(const uint8_t* payload, uint32_t sz) {
    if (sz < sizeof(MsgVmOpDispatch)) return;
    const auto& hdr = *reinterpret_cast<const MsgVmOpDispatch*>(payload);

    // Parse variable layout: MsgVmOpDispatch | vaddrs[n_inputs] | op_payload
    uint32_t vaddr_bytes = hdr.n_inputs * sizeof(uint64_t);
    if (sz < sizeof(MsgVmOpDispatch) + vaddr_bytes) {
        send_op_reject(hdr.op_id);
        return;
    }

    VmOpTask task;
    task.op_id          = hdr.op_id;
    task.request_id     = hdr.request_id;
    task.op_type        = static_cast<VmOpType>(hdr.op_type);
    task.n_output_bytes = hdr.n_output_bytes;

    const uint8_t* p = payload + sizeof(MsgVmOpDispatch);
    task.input_vaddrs.resize(hdr.n_inputs);
    std::memcpy(task.input_vaddrs.data(), p, vaddr_bytes);
    p += vaddr_bytes;

    uint32_t payload_bytes = sz - sizeof(MsgVmOpDispatch) - vaddr_bytes;
    task.op_payload.assign(p, p + payload_bytes);

    if (!exec_queue_.try_push(std::move(task))) {
        // Queue full — reject
        send_op_reject(hdr.op_id);
    }
}

void VmNode::handle_checkpoint(const uint8_t* payload, uint32_t sz) {
    if (sz < sizeof(MsgVmCheckpointSnap)) return;
    const auto& msg = *reinterpret_cast<const MsgVmCheckpointSnap*>(payload);
    persist_snapshot(msg.snap_id);

    MsgVmCheckpointAck ack{};
    ack.snap_id = msg.snap_id;
    std::strncpy(ack.node_id, cfg_.base.node_id.c_str(), sizeof(ack.node_id) - 1);

    try {
        vm_ctrl_conn_->send_msg(
            static_cast<MsgType>(VmMsgType::VM_CHECKPOINT_ACK),
            reinterpret_cast<uint8_t*>(&ack), sizeof(ack));
    } catch (...) {}
}

void VmNode::handle_restore_req(const uint8_t* payload, uint32_t sz) {
    if (sz < sizeof(MsgVmRestoreReq)) return;
    const auto& msg = *reinterpret_cast<const MsgVmRestoreReq*>(payload);
    restore_from_snap(msg.snap_id, msg.vaddr, msg.n_bytes, msg.dtype);
}

void VmNode::handle_topo_update(const uint8_t* payload, uint32_t sz) {
    if (sz < sizeof(MsgVmTopoUpdate)) return;
    const auto& hdr = *reinterpret_cast<const MsgVmTopoUpdate*>(payload);

    std::vector<std::string> new_ring;
    const char* p = reinterpret_cast<const char*>(payload + sizeof(MsgVmTopoUpdate));
    for (uint32_t i = 0; i < hdr.n_nodes; ++i) {
        new_ring.emplace_back(p);
        p += 64;
    }

    std::lock_guard<std::mutex> lk(ring_mu_);
    ring_ = std::move(new_ring);
}

void VmNode::handle_collective_chunk(const uint8_t* payload, uint32_t sz) {
    if (sz < sizeof(MsgVmCollectiveChunk)) return;
    const auto& hdr = *reinterpret_cast<const MsgVmCollectiveChunk*>(payload);
    uint32_t chunk_bytes = sz - sizeof(MsgVmCollectiveChunk);
    collective_.on_chunk(hdr.coll_id,
                         std::string(hdr.from_node),
                         hdr.phase,
                         hdr.chunk_offset,
                         payload + sizeof(MsgVmCollectiveChunk),
                         chunk_bytes);
}

// ─── Op execution ─────────────────────────────────────────────────────────────

std::vector<uint8_t> VmNode::exec_op(const VmOpTask& task) {
    // Fetch all input tensors
    std::vector<std::vector<uint8_t>> inputs(task.input_vaddrs.size());
    {
        std::lock_guard<std::mutex> lk(tensor_mu_);
        for (size_t i = 0; i < task.input_vaddrs.size(); ++i) {
            auto it = tensors_.find(task.input_vaddrs[i]);
            if (it != tensors_.end()) inputs[i] = it->second;
        }
    }

    // CPU reference implementations for each VmOpType.
    // Production code would dispatch to ggml kernels / CUDA here.
    switch (task.op_type) {
    case VmOpType::Add: {
        if (inputs.size() < 2 || inputs[0].size() != inputs[1].size())
            throw std::runtime_error("Add: need 2 equal-size inputs");
        std::vector<uint8_t> out(inputs[0].size());
        const float* a = reinterpret_cast<const float*>(inputs[0].data());
        const float* b = reinterpret_cast<const float*>(inputs[1].data());
        float*       c = reinterpret_cast<float*>(out.data());
        size_t n = inputs[0].size() / sizeof(float);
        for (size_t i = 0; i < n; ++i) c[i] = a[i] + b[i];
        return out;
    }
    case VmOpType::Scale: {
        if (inputs.empty() || task.op_payload.size() < sizeof(float))
            throw std::runtime_error("Scale: need input + scale factor");
        float scale;
        std::memcpy(&scale, task.op_payload.data(), sizeof(float));
        std::vector<uint8_t> out(inputs[0].size());
        const float* a = reinterpret_cast<const float*>(inputs[0].data());
        float*       c = reinterpret_cast<float*>(out.data());
        size_t n = inputs[0].size() / sizeof(float);
        for (size_t i = 0; i < n; ++i) c[i] = a[i] * scale;
        return out;
    }
    case VmOpType::RmsNorm: {
        if (inputs.empty()) throw std::runtime_error("RmsNorm: need input");
        const float eps = 1e-5f;
        std::vector<uint8_t> out(inputs[0].size());
        const float* x = reinterpret_cast<const float*>(inputs[0].data());
        float*       y = reinterpret_cast<float*>(out.data());
        size_t n = inputs[0].size() / sizeof(float);
        float ss = 0.0f;
        for (size_t i = 0; i < n; ++i) ss += x[i] * x[i];
        float scale = 1.0f / std::sqrt(ss / (float)n + eps);
        for (size_t i = 0; i < n; ++i) y[i] = x[i] * scale;
        return out;
    }
    case VmOpType::SiluMul: {
        // SiLU(x) * gate: used in SwiGLU / LLaMA FFN
        if (inputs.size() < 2 || inputs[0].size() != inputs[1].size())
            throw std::runtime_error("SiluMul: need 2 equal-size inputs");
        std::vector<uint8_t> out(inputs[0].size());
        const float* gate = reinterpret_cast<const float*>(inputs[0].data());
        const float* up   = reinterpret_cast<const float*>(inputs[1].data());
        float*       dst  = reinterpret_cast<float*>(out.data());
        size_t n = inputs[0].size() / sizeof(float);
        for (size_t i = 0; i < n; ++i) {
            float g   = gate[i];
            float silu = g / (1.0f + std::exp(-g));
            dst[i] = silu * up[i];
        }
        return out;
    }
    case VmOpType::Custom: {
        // Pass through: output = copy of first input.
        if (inputs.empty()) return {};
        return inputs[0];
    }
    default:
        throw std::runtime_error("VmNode::exec_op: unhandled VmOpType "
                                 + std::to_string(static_cast<int>(task.op_type)));
    }
}

// ─── Checkpoint / restore ─────────────────────────────────────────────────────

void VmNode::persist_snapshot(uint64_t snap_id) {
    std::unordered_map<uint64_t, std::vector<uint8_t>> copy;
    {
        std::lock_guard<std::mutex> lk(tensor_mu_);
        copy = tensors_;  // deep copy
    }
    {
        std::lock_guard<std::mutex> lk(snap_mu_);
        snapshots_[snap_id] = std::move(copy);
        // Keep only last 2 snapshots to bound memory
        while (snapshots_.size() > 2) {
            snapshots_.erase(snapshots_.begin());
        }
    }
    std::cout << "[VmNode] snapshot snap_id=" << snap_id
              << " tensors=" << copy.size() << "\n";
}

void VmNode::restore_from_snap(uint64_t snap_id, uint64_t vaddr,
                                uint32_t n_bytes, uint16_t /*dtype*/) {
    std::lock_guard<std::mutex> sl(snap_mu_);
    auto sit = snapshots_.find(snap_id);
    if (sit == snapshots_.end()) {
        std::cerr << "[VmNode] restore_from_snap: snap_id=" << snap_id << " not found\n";
        return;
    }
    auto tit = sit->second.find(vaddr);
    if (tit == sit->second.end()) {
        // Not in this snapshot — initialise to zeros
        std::lock_guard<std::mutex> tl(tensor_mu_);
        tensors_[vaddr] = std::vector<uint8_t>(n_bytes, 0);
        return;
    }
    std::lock_guard<std::mutex> tl(tensor_mu_);
    tensors_[vaddr] = tit->second;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

void VmNode::send_op_result(uint64_t op_id, const std::vector<uint8_t>& output) {
    if (!vm_ctrl_conn_ || !vm_ctrl_conn_->is_connected()) return;

    uint32_t total = sizeof(MsgVmOpResult) + (uint32_t)output.size();
    std::vector<uint8_t> buf(total);
    auto& hdr      = *reinterpret_cast<MsgVmOpResult*>(buf.data());
    hdr.op_id      = op_id;
    hdr.n_bytes    = (uint32_t)output.size();
    hdr.gpu_util   = 0.0f;  // TODO: query from ggml
    hdr.free_vram  = 0;
    if (!output.empty())
        std::memcpy(buf.data() + sizeof(MsgVmOpResult),
                    output.data(), output.size());

    try {
        vm_ctrl_conn_->send_msg(
            static_cast<MsgType>(VmMsgType::VM_OP_RESULT),
            buf.data(), total);
    } catch (...) {}
}

void VmNode::send_op_reject(uint64_t op_id) {
    if (!vm_ctrl_conn_ || !vm_ctrl_conn_->is_connected()) return;
    MsgVmOpReject rej{};
    rej.op_id = op_id;
    try {
        vm_ctrl_conn_->send_msg(
            static_cast<MsgType>(VmMsgType::VM_OP_REJECT),
            reinterpret_cast<uint8_t*>(&rej), sizeof(rej));
    } catch (...) {}
}

Connection* VmNode::get_peer_conn(const std::string& peer_id) {
    std::lock_guard<std::mutex> lk(peer_mu_);
    auto it = peer_conns_.find(peer_id);
    if (it != peer_conns_.end() && it->second->is_connected())
        return it->second.get();

    // Lazy-connect to peer's VM ctrl port
    // We don't know the peer's host here — in production, the TOPO_UPDATE
    // would carry host:port pairs. For now we use the coordinator as relay
    // (collective chunks go coordinator → VmNode's vm_ctrl_thread).
    return nullptr;
}

NodeCapability VmNode::build_vm_capability() const {
    // Reuse the base NodeAgent helper by calling build_capability via agent_.
    // Since NodeAgent::build_capability() is private, replicate the logic here.
    NodeCapability cap{};
    cap.cpu_ram_bytes = 0; // TODO: query sysinfo
#if defined(GGML_USE_CUDA) || defined(GGML_USE_METAL)
    // ggml_backend_dev_count() / ggml_backend_dev_memory() would go here
    // For now, leave as zero; the coordinator uses whatever is reported.
#endif
    return cap;
}

} // namespace dist
