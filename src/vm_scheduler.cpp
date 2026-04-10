/**
 * vm_scheduler.cpp
 */

#include "vm_scheduler.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>

namespace dist {

VmScheduler::VmScheduler(VmTensorRegistry& registry)
    : registry_(registry)
{}

// ─── Node management ─────────────────────────────────────────────────────────

void VmScheduler::add_node(const std::string& node_id,
                            const NodeCapability& cap,
                            std::shared_ptr<Connection> ctrl_conn) {
    std::lock_guard<std::mutex> lk(mu_);
    NodeStat& n   = nodes_[node_id];
    n.id          = node_id;
    n.cap         = cap;
    n.ctrl_conn   = std::move(ctrl_conn);
    n.alive       = true;
    n.gpu_util    = 0.0f;
    // Initialise free_vram from capability
    n.free_vram   = 0;
    for (uint32_t i = 0; i < cap.n_gpus && i < 8; ++i)
        n.free_vram += cap.gpu_free_bytes[i];
    std::cout << "[VmScheduler] added node " << node_id
              << " free_vram=" << n.free_vram / (1024*1024) << "MiB\n";
}

void VmScheduler::remove_node(const std::string& node_id) {
    on_node_failed(node_id);
    std::lock_guard<std::mutex> lk(mu_);
    nodes_.erase(node_id);
}

void VmScheduler::update_load(const std::string& node_id,
                               float gpu_util, uint64_t free_vram) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = nodes_.find(node_id);
    if (it != nodes_.end()) {
        it->second.gpu_util  = gpu_util;
        it->second.free_vram = free_vram;
    }
}

// ─── Op submission ────────────────────────────────────────────────────────────

OpFuture VmScheduler::submit(uint64_t request_id,
                               std::vector<uint64_t> input_vaddrs,
                               std::vector<uint8_t>  op_payload,
                               VmOpType              op_type,
                               uint32_t              expected_output_bytes) {
    uint64_t op_id = next_op_id_.fetch_add(1);

    auto promise = std::make_shared<std::promise<OpResult>>();
    std::shared_future<OpResult> fut = promise->get_future().share();
    OpFuture result(op_id, fut);

    PendingOp op;
    op.op_id                  = op_id;
    op.request_id             = request_id;
    op.input_vaddrs           = std::move(input_vaddrs);
    op.op_payload             = std::move(op_payload);
    op.op_type                = op_type;
    op.expected_output_bytes  = expected_output_bytes;
    op.promise                = promise;

    {
        std::lock_guard<std::mutex> lk(mu_);
        std::string target = select_node(op.input_vaddrs);
        if (target.empty()) {
            OpResult r;
            r.success = false;
            r.error   = "no nodes available";
            promise->set_value(r);
            return result;
        }
        dispatch_op(op, target);
        pending_.emplace(op_id, std::move(op));
    }
    return result;
}

// ─── Result callbacks ────────────────────────────────────────────────────────

void VmScheduler::on_result(uint64_t op_id, std::vector<uint8_t> output_bytes) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = pending_.find(op_id);
    if (it == pending_.end()) return;

    PendingOp& op = it->second;
    auto nit = nodes_.find(op.dispatched_to);
    if (nit != nodes_.end() && nit->second.inflight_ops > 0)
        nit->second.inflight_ops--;

    OpResult r;
    r.success = true;
    r.output  = std::move(output_bytes);
    op.promise->set_value(std::move(r));
    pending_.erase(it);
}

void VmScheduler::on_reject(uint64_t op_id, const std::string& rejecting_node) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = pending_.find(op_id);
    if (it == pending_.end()) return;

    PendingOp& op = it->second;
    auto nit = nodes_.find(rejecting_node);
    if (nit != nodes_.end() && nit->second.inflight_ops > 0)
        nit->second.inflight_ops--;

    op.retry_count++;
    if (op.retry_count >= VM_SCHED_MAX_RETRIES) {
        OpResult r;
        r.success = false;
        r.error   = "op rejected by " + std::to_string(op.retry_count) + " nodes";
        op.promise->set_value(std::move(r));
        pending_.erase(it);
        return;
    }

    std::string next = select_node(op.input_vaddrs, rejecting_node);
    if (next.empty()) {
        OpResult r;
        r.success = false;
        r.error   = "no alternative node for retry";
        op.promise->set_value(std::move(r));
        pending_.erase(it);
        return;
    }
    dispatch_op(op, next);
}

void VmScheduler::on_node_failed(const std::string& node_id) {
    std::lock_guard<std::mutex> lk(mu_);
    auto nit = nodes_.find(node_id);
    if (nit != nodes_.end()) nit->second.alive = false;

    // Find all ops dispatched to this node and retry them
    std::vector<uint64_t> affected;
    for (auto& [oid, op] : pending_) {
        if (op.dispatched_to == node_id) affected.push_back(oid);
    }
    for (uint64_t oid : affected) {
        auto it = pending_.find(oid);
        if (it == pending_.end()) continue;
        PendingOp& op = it->second;
        op.retry_count++;
        if (op.retry_count >= VM_SCHED_MAX_RETRIES) {
            OpResult r;
            r.success = false;
            r.error   = "node " + node_id + " failed during op";
            op.promise->set_value(std::move(r));
            pending_.erase(it);
            continue;
        }
        std::string next = select_node(op.input_vaddrs, node_id);
        if (next.empty()) {
            OpResult r;
            r.success = false;
            r.error   = "node failed, no fallback";
            op.promise->set_value(std::move(r));
            pending_.erase(it);
            continue;
        }
        dispatch_op(op, next);
    }
}

// ─── Accessors ───────────────────────────────────────────────────────────────

size_t VmScheduler::pending_op_count() const {
    std::lock_guard<std::mutex> lk(mu_);
    return pending_.size();
}

std::vector<std::string> VmScheduler::node_ids() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<std::string> ids;
    for (auto& [id, _] : nodes_) ids.push_back(id);
    return ids;
}

// ─── Private: node selection ─────────────────────────────────────────────────

std::string VmScheduler::select_node(const std::vector<uint64_t>& input_vaddrs,
                                      const std::string& exclude) {
    // Score each live node.
    // locality_score[node] = bytes of input tensors owned by that node.
    std::unordered_map<std::string, uint64_t> locality;
    for (uint64_t va : input_vaddrs) {
        auto desc = registry_.lookup(va);
        if (!desc) continue;
        locality[desc->owning_node] += desc->n_bytes;
    }

    std::string best;
    double      best_score = -1.0;

    for (auto& [nid, stat] : nodes_) {
        if (!stat.alive) continue;
        if (nid == exclude) continue;
        if (!stat.ctrl_conn || !stat.ctrl_conn->is_connected()) continue;

        // Locality contribution (normalised to [0,1])
        double loc = 0.0;
        if (!input_vaddrs.empty()) {
            uint64_t total_input_bytes = 0;
            for (uint64_t va : input_vaddrs) {
                auto d = registry_.lookup(va);
                if (d) total_input_bytes += d->n_bytes;
            }
            if (total_input_bytes > 0)
                loc = (double)locality[nid] / total_input_bytes;
        }

        // Load contribution (lower gpu_util is better)
        double load = 1.0 - (double)std::min(stat.gpu_util, 1.0f);

        // Inflight penalty
        double inflight_pen = 1.0 / (1.0 + stat.inflight_ops);

        // Combined score: locality is most important
        double score = 0.6*loc + 0.25*load + 0.15*inflight_pen;

        if (score > best_score) {
            best_score = score;
            best       = nid;
        }
    }
    return best;
}

// ─── Private: dispatch ───────────────────────────────────────────────────────

void VmScheduler::dispatch_op(PendingOp& op, const std::string& target_node) {
    auto nit = nodes_.find(target_node);
    if (nit == nodes_.end() || !nit->second.alive) return;
    NodeStat& n = nit->second;
    if (!n.ctrl_conn || !n.ctrl_conn->is_connected()) return;

    op.dispatched_to = target_node;
    n.inflight_ops++;

    // Build VM_OP_DISPATCH payload:
    // MsgVmOpDispatch | n_inputs * uint64_t vaddrs | op_payload
    uint32_t vaddr_bytes = (uint32_t)(op.input_vaddrs.size() * sizeof(uint64_t));
    uint32_t total_pay   = sizeof(MsgVmOpDispatch) + vaddr_bytes
                           + (uint32_t)op.op_payload.size();

    std::vector<uint8_t> buf(total_pay);
    auto& hdr = *reinterpret_cast<MsgVmOpDispatch*>(buf.data());
    hdr.op_id                = op.op_id;
    hdr.request_id           = op.request_id;
    hdr.op_type              = static_cast<uint32_t>(op.op_type);
    hdr.n_inputs             = (uint32_t)op.input_vaddrs.size();
    hdr.n_output_bytes       = op.expected_output_bytes;
    hdr.op_payload_bytes     = (uint32_t)op.op_payload.size();

    uint8_t* p = buf.data() + sizeof(MsgVmOpDispatch);
    std::memcpy(p, op.input_vaddrs.data(), vaddr_bytes);
    p += vaddr_bytes;
    if (!op.op_payload.empty())
        std::memcpy(p, op.op_payload.data(), op.op_payload.size());

    try {
        n.ctrl_conn->send_msg(static_cast<MsgType>(VmMsgType::VM_OP_DISPATCH),
                              buf.data(), total_pay);
    } catch (...) {
        n.alive = false;
    }
}

} // namespace dist
