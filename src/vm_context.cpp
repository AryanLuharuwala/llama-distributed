/**
 * vm_context.cpp
 *
 * VmContext — thin client that speaks to a VmCoordinator over TCP.
 *
 * Connection model:
 *   One persistent TCP connection to VmCoordinator:PORT_VM_CTRL.
 *   VmContext sends requests (alloc, write, submit_op, …) and a background
 *   recv_thread_ dispatches responses back to waiting promises.
 *
 *   Inference requests go to PORT_API (base Coordinator) as MsgInferRequest
 *   and tokens stream back as MsgInferToken / MsgInferDone.
 *
 * Thread safety:
 *   All public methods are thread-safe (protected by conn_mu_ + per-map mutexes).
 */

#include "vm_context.h"

#include <cstring>
#include <iostream>
#include <stdexcept>

namespace dist {

// ─── Constructor / Destructor ─────────────────────────────────────────────────

VmContext::VmContext(VmContextConfig cfg)
    : cfg_(std::move(cfg))
{
    running_.store(true);
    ensure_connected();
    recv_thread_ = std::thread([this]{ recv_thread_fn(); });
}

VmContext::~VmContext() {
    running_.store(false);
    {
        std::lock_guard<std::mutex> lk(conn_mu_);
        if (vm_conn_) vm_conn_.reset();
    }
    if (recv_thread_.joinable()) recv_thread_.join();
}

// ─── Connection ───────────────────────────────────────────────────────────────

bool VmContext::ensure_connected() {
    std::lock_guard<std::mutex> lk(conn_mu_);
    if (vm_conn_ && vm_conn_->is_connected()) return true;
    try {
        vm_conn_ = std::make_unique<Connection>();
        vm_conn_->connect(cfg_.coordinator_host, cfg_.vm_ctrl_port);
        std::cout << "[VmContext] connected to " << cfg_.coordinator_host
                  << ":" << cfg_.vm_ctrl_port << "\n";
        return true;
    } catch (std::exception& e) {
        std::cerr << "[VmContext] connect failed: " << e.what() << "\n";
        vm_conn_.reset();
        return false;
    }
}

void VmContext::reconnect() {
    std::cerr << "[VmContext] reconnecting...\n";
    {
        std::lock_guard<std::mutex> lk(conn_mu_);
        vm_conn_.reset();
    }
    ensure_connected();
}

// ─── Recv thread ──────────────────────────────────────────────────────────────

void VmContext::recv_thread_fn() {
    while (running_.load()) {
        Connection* c = nullptr;
        {
            std::lock_guard<std::mutex> lk(conn_mu_);
            c = vm_conn_.get();
        }
        if (!c || !c->is_connected()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        std::vector<uint8_t> buf;
        MsgType mtype;
        if (!c->recv_msg(mtype, buf)) {
            reconnect();
            continue;
        }

        auto vm_type = static_cast<VmMsgType>(mtype);
        const uint8_t* p = buf.data();
        uint32_t sz      = (uint32_t)buf.size();

        switch (vm_type) {
        case VmMsgType::VM_TENSOR_READ_RSP:  on_tensor_read_rsp(p, sz);  break;
        case VmMsgType::VM_OP_RESULT:        on_op_result(p, sz);        break;
        case VmMsgType::VM_OP_REJECT:        on_op_reject(p, sz);        break;
        case VmMsgType::VM_COLLECTIVE_DONE:  on_collective_done(p, sz);  break;
        case VmMsgType::VM_CHECKPOINT_ACK:   on_checkpoint_done(p, sz);  break;
        default:
            // Check base protocol types (infer tokens)
            if (mtype == static_cast<MsgType>(MsgTypeEnum::INFER_TOKEN))
                on_infer_token(p, sz);
            else if (mtype == static_cast<MsgType>(MsgTypeEnum::INFER_DONE))
                on_infer_done(p, sz);
            break;
        }
    }
}

// ─── Tensor management ────────────────────────────────────────────────────────

uint64_t VmContext::alloc(uint32_t n_bytes, uint16_t dtype,
                           const std::string& preferred_node) {
    if (!ensure_connected()) return 0;

    // VmContext sends VM_TENSOR_ALLOC to coordinator; coordinator allocates
    // in the registry and forwards to owning node. Coordinator replies with
    // VM_TENSOR_ALLOC_RSP containing the vaddr.
    // For simplicity, we use a synchronous promise pattern here.

    // Coordinator returns vaddr synchronously via VM_TENSOR_ALLOC_RSP.
    // We'll register a pending promise keyed on a temporary local id.
    // To avoid a separate request-id for allocs, we use 0 (only one outstanding
    // alloc at a time is safe; for production use a proper counter).
    uint64_t temp_key = next_op_id_.fetch_add(1);

    auto promise = std::make_shared<std::promise<std::vector<uint8_t>>>();
    auto fut     = promise->get_future();
    {
        std::lock_guard<std::mutex> lk(read_mu_);
        read_promises_[temp_key] = promise;
    }

    MsgVmTensorAlloc msg{};
    msg.vaddr   = temp_key;   // coordinator interprets vaddr=0 as alloc-new
    msg.n_bytes = n_bytes;
    msg.dtype   = dtype;
    if (!preferred_node.empty())
        std::strncpy(msg.preferred_node, preferred_node.c_str(),
                     sizeof(msg.preferred_node) - 1);

    {
        std::lock_guard<std::mutex> lk(conn_mu_);
        if (!vm_conn_) return 0;
        try {
            vm_conn_->send_msg(static_cast<MsgType>(VmMsgType::VM_TENSOR_ALLOC),
                               reinterpret_cast<uint8_t*>(&msg), sizeof(msg));
        } catch (...) {
            read_promises_.erase(temp_key);
            return 0;
        }
    }

    if (fut.wait_for(cfg_.default_timeout) != std::future_status::ready) {
        std::lock_guard<std::mutex> lk(read_mu_);
        read_promises_.erase(temp_key);
        return 0;
    }
    auto rsp = fut.get();
    if (rsp.size() < sizeof(uint64_t)) return 0;
    uint64_t vaddr = 0;
    std::memcpy(&vaddr, rsp.data(), sizeof(vaddr));
    return vaddr;
}

void VmContext::free(uint64_t vaddr) {
    if (!ensure_connected()) return;
    MsgVmTensorFree msg{};
    msg.vaddr = vaddr;
    std::lock_guard<std::mutex> lk(conn_mu_);
    if (!vm_conn_) return;
    try {
        vm_conn_->send_msg(static_cast<MsgType>(VmMsgType::VM_TENSOR_FREE),
                           reinterpret_cast<uint8_t*>(&msg), sizeof(msg));
    } catch (...) {}
}

bool VmContext::write(uint64_t vaddr, const uint8_t* data, uint32_t n_bytes) {
    if (!ensure_connected()) return false;
    uint32_t total = sizeof(MsgVmTensorWrite) + n_bytes;
    std::vector<uint8_t> buf(total);
    auto& hdr   = *reinterpret_cast<MsgVmTensorWrite*>(buf.data());
    hdr.vaddr   = vaddr;
    hdr.offset  = 0;
    hdr.n_bytes = n_bytes;
    if (n_bytes > 0) std::memcpy(buf.data() + sizeof(MsgVmTensorWrite), data, n_bytes);

    std::lock_guard<std::mutex> lk(conn_mu_);
    if (!vm_conn_) return false;
    try {
        vm_conn_->send_msg(static_cast<MsgType>(VmMsgType::VM_TENSOR_WRITE),
                           buf.data(), total);
        return true;
    } catch (...) { return false; }
}

std::vector<uint8_t> VmContext::read(uint64_t vaddr,
                                      std::chrono::milliseconds timeout) {
    if (!ensure_connected()) return {};

    auto promise = std::make_shared<std::promise<std::vector<uint8_t>>>();
    auto fut     = promise->get_future();
    {
        std::lock_guard<std::mutex> lk(read_mu_);
        read_promises_[vaddr] = promise;
    }

    MsgVmTensorRead msg{};
    msg.vaddr   = vaddr;
    msg.offset  = 0;
    msg.n_bytes = 0; // 0 = all

    {
        std::lock_guard<std::mutex> lk(conn_mu_);
        if (!vm_conn_) return {};
        try {
            vm_conn_->send_msg(static_cast<MsgType>(VmMsgType::VM_TENSOR_READ),
                               reinterpret_cast<uint8_t*>(&msg), sizeof(msg));
        } catch (...) {
            std::lock_guard<std::mutex> rlk(read_mu_);
            read_promises_.erase(vaddr);
            return {};
        }
    }

    if (fut.wait_for(timeout) != std::future_status::ready) {
        std::lock_guard<std::mutex> lk(read_mu_);
        read_promises_.erase(vaddr);
        return {};
    }
    return fut.get();
}

// ─── Compute ops ──────────────────────────────────────────────────────────────

OpFuture VmContext::submit_op(VmOpType op_type,
                               std::vector<uint64_t> input_vaddrs,
                               std::vector<uint8_t>  op_payload,
                               uint32_t              expected_output_bytes) {
    uint64_t op_id = next_op_id_.fetch_add(1);

    auto promise = std::make_shared<std::promise<OpResult>>();
    std::shared_future<OpResult> fut = promise->get_future().share();

    {
        std::lock_guard<std::mutex> lk(op_mu_);
        op_promises_[op_id] = promise;
    }

    uint32_t vaddr_bytes = (uint32_t)(input_vaddrs.size() * sizeof(uint64_t));
    uint32_t total       = sizeof(MsgVmOpDispatch) + vaddr_bytes
                           + (uint32_t)op_payload.size();
    std::vector<uint8_t> buf(total);
    auto& hdr              = *reinterpret_cast<MsgVmOpDispatch*>(buf.data());
    hdr.op_id              = op_id;
    hdr.request_id         = next_req_id_.fetch_add(1);
    hdr.op_type            = static_cast<uint32_t>(op_type);
    hdr.n_inputs           = (uint32_t)input_vaddrs.size();
    hdr.n_output_bytes     = expected_output_bytes;
    hdr.op_payload_bytes   = (uint32_t)op_payload.size();

    uint8_t* p = buf.data() + sizeof(MsgVmOpDispatch);
    if (!input_vaddrs.empty()) std::memcpy(p, input_vaddrs.data(), vaddr_bytes);
    p += vaddr_bytes;
    if (!op_payload.empty())   std::memcpy(p, op_payload.data(), op_payload.size());

    {
        std::lock_guard<std::mutex> lk(conn_mu_);
        if (!vm_conn_) {
            OpResult r; r.success = false; r.error = "not connected";
            promise->set_value(r);
        } else {
            try {
                vm_conn_->send_msg(static_cast<MsgType>(VmMsgType::VM_OP_DISPATCH),
                                   buf.data(), total);
            } catch (...) {
                OpResult r; r.success = false; r.error = "send failed";
                promise->set_value(r);
            }
        }
    }
    return OpFuture(op_id, fut);
}

// ─── Collective ops ───────────────────────────────────────────────────────────

CollectiveFuture VmContext::all_reduce(std::vector<uint8_t> data,
                                        uint16_t dtype,
                                        ReduceOp reduce_op) {
    uint64_t coll_id = next_coll_id_.fetch_add(1);

    auto promise = std::make_shared<std::promise<CollectiveResult>>();
    std::shared_future<CollectiveResult> fut = promise->get_future().share();
    {
        std::lock_guard<std::mutex> lk(coll_mu_);
        coll_promises_[coll_id] = promise;
    }

    // Build VM_COLLECTIVE_INIT: tell coordinator to initiate AllReduce
    uint32_t total = sizeof(MsgVmCollectiveInit) + (uint32_t)data.size();
    std::vector<uint8_t> buf(total);
    auto& hdr         = *reinterpret_cast<MsgVmCollectiveInit*>(buf.data());
    hdr.coll_id       = coll_id;
    hdr.op            = static_cast<uint8_t>(CollectiveOp::AllReduce);
    hdr.reduce_op     = static_cast<uint8_t>(reduce_op);
    hdr.dtype         = dtype;
    hdr.data_bytes    = (uint32_t)data.size();
    if (!data.empty())
        std::memcpy(buf.data() + sizeof(MsgVmCollectiveInit),
                    data.data(), data.size());

    std::lock_guard<std::mutex> lk(conn_mu_);
    if (!vm_conn_) {
        CollectiveResult r; r.success = false; r.error = "not connected";
        promise->set_value(r);
    } else {
        try {
            vm_conn_->send_msg(static_cast<MsgType>(VmMsgType::VM_COLLECTIVE_INIT),
                               buf.data(), total);
        } catch (...) {
            CollectiveResult r; r.success = false; r.error = "send failed";
            promise->set_value(r);
        }
    }
    return CollectiveFuture(coll_id, fut);
}

CollectiveFuture VmContext::broadcast(std::vector<uint8_t> data, uint16_t dtype) {
    uint64_t coll_id = next_coll_id_.fetch_add(1);

    auto promise = std::make_shared<std::promise<CollectiveResult>>();
    std::shared_future<CollectiveResult> fut = promise->get_future().share();
    {
        std::lock_guard<std::mutex> lk(coll_mu_);
        coll_promises_[coll_id] = promise;
    }

    uint32_t total = sizeof(MsgVmCollectiveInit) + (uint32_t)data.size();
    std::vector<uint8_t> buf(total);
    auto& hdr         = *reinterpret_cast<MsgVmCollectiveInit*>(buf.data());
    hdr.coll_id       = coll_id;
    hdr.op            = static_cast<uint8_t>(CollectiveOp::Broadcast);
    hdr.reduce_op     = 0;
    hdr.dtype         = dtype;
    hdr.data_bytes    = (uint32_t)data.size();
    if (!data.empty())
        std::memcpy(buf.data() + sizeof(MsgVmCollectiveInit),
                    data.data(), data.size());

    std::lock_guard<std::mutex> lk(conn_mu_);
    if (!vm_conn_) {
        CollectiveResult r; r.success = false; r.error = "not connected";
        promise->set_value(r);
    } else {
        try {
            vm_conn_->send_msg(static_cast<MsgType>(VmMsgType::VM_COLLECTIVE_INIT),
                               buf.data(), total);
        } catch (...) {
            CollectiveResult r; r.success = false; r.error = "send failed";
            promise->set_value(r);
        }
    }
    return CollectiveFuture(coll_id, fut);
}

// ─── Checkpoint ───────────────────────────────────────────────────────────────

CheckpointFuture VmContext::checkpoint() {
    uint64_t snap_id = next_op_id_.fetch_add(1);

    auto promise = std::make_shared<std::promise<CheckpointResult>>();
    std::shared_future<CheckpointResult> fut = promise->get_future().share();
    {
        std::lock_guard<std::mutex> lk(ckpt_mu_);
        ckpt_promises_[snap_id] = promise;
    }

    MsgVmCheckpointSnap msg{};
    msg.snap_id   = snap_id;
    msg.n_tensors = 0; // coordinator fills in

    std::lock_guard<std::mutex> lk(conn_mu_);
    if (!vm_conn_) {
        CheckpointResult r; r.success = false; r.error = "not connected";
        promise->set_value(r);
    } else {
        try {
            vm_conn_->send_msg(static_cast<MsgType>(VmMsgType::VM_CHECKPOINT_SNAP),
                               reinterpret_cast<uint8_t*>(&msg), sizeof(msg));
        } catch (...) {
            CheckpointResult r; r.success = false; r.error = "send failed";
            promise->set_value(r);
        }
    }
    return CheckpointFuture(snap_id, fut);
}

// ─── Inference ────────────────────────────────────────────────────────────────

void VmContext::infer(const std::vector<int32_t>& prompt_tokens,
                       uint32_t max_tokens,
                       TokenStreamCb cb,
                       std::chrono::milliseconds timeout) {
    if (!ensure_connected()) {
        if (cb) cb(-1, true);
        return;
    }

    uint64_t req_id = next_req_id_.fetch_add(1);
    auto done_promise = std::make_shared<std::promise<void>>();
    auto done_fut     = done_promise->get_future();

    {
        std::lock_guard<std::mutex> lk(infer_mu_);
        infer_pending_[req_id] = { std::move(cb), done_promise };
    }

    // Build MsgInferRequest payload: header + token ids
    uint32_t n_tokens   = (uint32_t)prompt_tokens.size();
    uint32_t tok_bytes  = n_tokens * sizeof(int32_t);
    uint32_t total      = sizeof(MsgInferRequest) + tok_bytes;
    std::vector<uint8_t> buf(total);

    auto& hdr           = *reinterpret_cast<MsgInferRequest*>(buf.data());
    hdr.request_id      = req_id;
    hdr.n_prompt_tokens = n_tokens;
    hdr.max_gen_tokens  = max_tokens;
    hdr.temperature     = 0.8f;
    hdr.top_p           = 0.95f;

    std::memcpy(buf.data() + sizeof(MsgInferRequest),
                prompt_tokens.data(), tok_bytes);

    {
        std::lock_guard<std::mutex> lk(conn_mu_);
        if (!vm_conn_) {
            std::lock_guard<std::mutex> ilk(infer_mu_);
            infer_pending_.erase(req_id);
            if (cb) cb(-1, true);
            return;
        }
        try {
            vm_conn_->send_msg(static_cast<MsgType>(MsgTypeEnum::INFER_REQUEST),
                               buf.data(), total);
        } catch (...) {
            std::lock_guard<std::mutex> ilk(infer_mu_);
            infer_pending_.erase(req_id);
            if (cb) cb(-1, true);
            return;
        }
    }

    // Wait for completion
    if (done_fut.wait_for(timeout) != std::future_status::ready) {
        std::lock_guard<std::mutex> ilk(infer_mu_);
        infer_pending_.erase(req_id);
    }
}

// ─── Cluster info ─────────────────────────────────────────────────────────────

std::vector<std::string> VmContext::live_nodes() {
    // We don't maintain a local list — coordinator knows.
    // In production, request via a VM_TOPO_QUERY message.
    return {};
}

// ─── Response handlers ────────────────────────────────────────────────────────

void VmContext::on_tensor_alloc_rsp(const uint8_t* p, uint32_t sz) {
    if (sz < sizeof(MsgVmTensorAllocRsp)) return;
    const auto& msg = *reinterpret_cast<const MsgVmTensorAllocRsp*>(p);

    std::lock_guard<std::mutex> lk(read_mu_);
    // Alloc responses are keyed by the temp_key we sent as vaddr
    // and carry back the real vaddr. We stored the promise under temp_key.
    // The coordinator echoes the request vaddr in req_vaddr field.
    auto it = read_promises_.find(msg.req_vaddr);
    if (it == read_promises_.end()) return;
    std::vector<uint8_t> rsp(sizeof(uint64_t));
    uint64_t real_vaddr = msg.vaddr;
    std::memcpy(rsp.data(), &real_vaddr, sizeof(real_vaddr));
    it->second->set_value(std::move(rsp));
    read_promises_.erase(it);
}

void VmContext::on_tensor_read_rsp(const uint8_t* p, uint32_t sz) {
    if (sz < sizeof(MsgVmTensorReadRsp)) return;
    const auto& msg    = *reinterpret_cast<const MsgVmTensorReadRsp*>(p);
    uint32_t data_bytes = sz - sizeof(MsgVmTensorReadRsp);
    std::vector<uint8_t> data(p + sizeof(MsgVmTensorReadRsp),
                               p + sizeof(MsgVmTensorReadRsp) + data_bytes);

    std::lock_guard<std::mutex> lk(read_mu_);
    auto it = read_promises_.find(msg.vaddr);
    if (it == read_promises_.end()) return;
    it->second->set_value(std::move(data));
    read_promises_.erase(it);
}

void VmContext::on_op_result(const uint8_t* p, uint32_t sz) {
    if (sz < sizeof(MsgVmOpResult)) return;
    const auto& msg    = *reinterpret_cast<const MsgVmOpResult*>(p);
    uint32_t data_bytes = sz - sizeof(MsgVmOpResult);

    OpResult r;
    r.success = true;
    r.output.assign(p + sizeof(MsgVmOpResult),
                    p + sizeof(MsgVmOpResult) + data_bytes);

    std::lock_guard<std::mutex> lk(op_mu_);
    auto it = op_promises_.find(msg.op_id);
    if (it == op_promises_.end()) return;
    it->second->set_value(std::move(r));
    op_promises_.erase(it);
}

void VmContext::on_op_reject(const uint8_t* p, uint32_t sz) {
    if (sz < sizeof(MsgVmOpReject)) return;
    const auto& msg = *reinterpret_cast<const MsgVmOpReject*>(p);

    OpResult r;
    r.success = false;
    r.error   = "op rejected by cluster";

    std::lock_guard<std::mutex> lk(op_mu_);
    auto it = op_promises_.find(msg.op_id);
    if (it == op_promises_.end()) return;
    it->second->set_value(std::move(r));
    op_promises_.erase(it);
}

void VmContext::on_checkpoint_done(const uint8_t* p, uint32_t sz) {
    if (sz < sizeof(MsgVmCheckpointAck)) return;
    const auto& msg = *reinterpret_cast<const MsgVmCheckpointAck*>(p);

    CheckpointResult r;
    r.success = true;
    r.snap_id = msg.snap_id;

    std::lock_guard<std::mutex> lk(ckpt_mu_);
    auto it = ckpt_promises_.find(msg.snap_id);
    if (it == ckpt_promises_.end()) return;
    it->second->set_value(std::move(r));
    ckpt_promises_.erase(it);
}

void VmContext::on_collective_done(const uint8_t* p, uint32_t sz) {
    if (sz < sizeof(MsgVmCollectiveDone)) return;
    const auto& msg    = *reinterpret_cast<const MsgVmCollectiveDone*>(p);
    uint32_t data_bytes = sz - sizeof(MsgVmCollectiveDone);

    CollectiveResult r;
    r.success = true;
    r.data.assign(p + sizeof(MsgVmCollectiveDone),
                  p + sizeof(MsgVmCollectiveDone) + data_bytes);

    std::lock_guard<std::mutex> lk(coll_mu_);
    auto it = coll_promises_.find(msg.coll_id);
    if (it == coll_promises_.end()) return;
    it->second->set_value(std::move(r));
    coll_promises_.erase(it);
}

void VmContext::on_infer_token(const uint8_t* p, uint32_t sz) {
    if (sz < sizeof(MsgInferToken)) return;
    const auto& msg = *reinterpret_cast<const MsgInferToken*>(p);

    std::lock_guard<std::mutex> lk(infer_mu_);
    auto it = infer_pending_.find(msg.request_id);
    if (it == infer_pending_.end()) return;
    if (it->second.cb) it->second.cb(msg.token_id, false);
}

void VmContext::on_infer_done(const uint8_t* p, uint32_t sz) {
    if (sz < sizeof(MsgInferDone)) return;
    const auto& msg = *reinterpret_cast<const MsgInferDone*>(p);

    std::lock_guard<std::mutex> lk(infer_mu_);
    auto it = infer_pending_.find(msg.request_id);
    if (it == infer_pending_.end()) return;
    if (it->second.cb) it->second.cb(-1, true);
    it->second.done_promise->set_value();
    infer_pending_.erase(it);
}

} // namespace dist
