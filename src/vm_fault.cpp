/**
 * vm_fault.cpp
 *
 * FaultManager implementation.
 *
 * Checkpoint 2PC:
 *   begin_checkpoint() snapshots the registry, broadcasts VM_CHECKPOINT_SNAP
 *   to all live nodes, and waits for each to reply VM_CHECKPOINT_ACK.
 *   If all ACK within timeout the checkpoint is committed; otherwise failed.
 *
 * Node failure:
 *   on_node_failed() calls registry_.migrate_all_from() to reassign ownership
 *   records, then scheduler_.on_node_failed() to retry in-flight ops, then
 *   sends VM_RESTORE_REQ to the new owners so they can fetch tensor data
 *   from the last committed checkpoint, then broadcasts a topology update.
 */

#include "vm_fault.h"

#include <iostream>
#include <cstring>

namespace dist {

FaultManager::FaultManager(GetConnFn      get_conn,
                            NodeListFn     live_nodes_fn,
                            VmTensorRegistry& registry,
                            VmScheduler&      scheduler)
    : get_conn_(std::move(get_conn))
    , live_nodes_fn_(std::move(live_nodes_fn))
    , registry_(registry)
    , scheduler_(scheduler)
{}

// ─── Checkpoint ──────────────────────────────────────────────────────────────

CheckpointFuture FaultManager::begin_checkpoint() {
    uint64_t snap_id = next_snap_id_.fetch_add(1);

    auto promise = std::make_shared<std::promise<CheckpointResult>>();
    std::shared_future<CheckpointResult> fut = promise->get_future().share();

    std::vector<std::string> nodes = live_nodes_fn_();
    if (nodes.empty()) {
        CheckpointResult r;
        r.success = true;
        r.snap_id = snap_id;
        promise->set_value(r);
        return CheckpointFuture(snap_id, fut);
    }

    // Snapshot the registry before broadcasting
    std::vector<TensorDesc> snap = registry_.snapshot();

    PendingCheckpoint pc;
    pc.snap_id    = snap_id;
    pc.snap       = snap;
    pc.promise    = promise;
    for (auto& n : nodes) pc.waiting_for.insert(n);

    {
        std::lock_guard<std::mutex> lk(mu_);
        pending_snaps_.emplace(snap_id, std::move(pc));
    }

    // Build VM_CHECKPOINT_SNAP message (no payload beyond header)
    MsgVmCheckpointSnap msg{};
    msg.snap_id     = snap_id;
    msg.n_tensors   = (uint32_t)snap.size();

    for (auto& nid : nodes) {
        Connection* conn = get_conn_(nid);
        if (!conn || !conn->is_connected()) {
            on_checkpoint_ack(snap_id, nid); // treat disconnected as ACK-fail
            continue;
        }
        try {
            conn->send_msg(static_cast<MsgType>(VmMsgType::VM_CHECKPOINT_SNAP),
                           reinterpret_cast<uint8_t*>(&msg), sizeof(msg));
        } catch (...) {
            std::cerr << "[FaultManager] VM_CHECKPOINT_SNAP send failed to " << nid << "\n";
            on_checkpoint_ack(snap_id, nid); // treat as responded (failed node)
        }
    }

    return CheckpointFuture(snap_id, fut);
}

void FaultManager::on_checkpoint_ack(uint64_t snap_id, const std::string& node_id) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = pending_snaps_.find(snap_id);
    if (it == pending_snaps_.end()) return;

    PendingCheckpoint& pc = it->second;
    pc.waiting_for.erase(node_id);

    if (pc.waiting_for.empty()) {
        // All nodes ACKed — commit
        committed_snap_id_ = snap_id;
        committed_snap_    = std::move(pc.snap);

        CheckpointResult r;
        r.success = true;
        r.snap_id = snap_id;
        pc.promise->set_value(std::move(r));
        pending_snaps_.erase(it);

        std::cout << "[FaultManager] checkpoint snap_id=" << snap_id << " committed\n";
    }
}

// ─── Node lifecycle ───────────────────────────────────────────────────────────

void FaultManager::on_node_joined(const std::string& node_id) {
    (void)node_id;
    broadcast_topo_update();
}

void FaultManager::on_node_failed(const std::string& node_id) {
    std::cout << "[FaultManager] node failed: " << node_id << "\n";

    // 1. Migrate tensor ownership in the registry
    auto migrations = registry_.migrate_all_from(node_id);

    // 2. Retry in-flight ops via scheduler
    scheduler_.on_node_failed(node_id);

    // 3. Send restore requests to new owners
    send_restore_requests(migrations);

    // 4. Broadcast updated topology
    broadcast_topo_update();
}

// ─── Accessors ────────────────────────────────────────────────────────────────

uint64_t FaultManager::last_committed_snap_id() const {
    std::lock_guard<std::mutex> lk(mu_);
    return committed_snap_id_;
}

std::vector<TensorDesc> FaultManager::last_snapshot() const {
    std::lock_guard<std::mutex> lk(mu_);
    return committed_snap_;
}

// ─── Private ─────────────────────────────────────────────────────────────────

void FaultManager::broadcast_topo_update() {
    std::vector<std::string> live = live_nodes_fn_();

    // Build VM_TOPO_UPDATE: header + node_id strings (null-terminated, 64 bytes each)
    // MsgVmTopoUpdate has n_nodes and a variable-length list of node_id[64] strings.
    uint32_t id_stride = 64;
    uint32_t payload_size = sizeof(MsgVmTopoUpdate) + (uint32_t)live.size() * id_stride;
    std::vector<uint8_t> buf(payload_size, 0);

    auto& hdr = *reinterpret_cast<MsgVmTopoUpdate*>(buf.data());
    hdr.n_nodes = (uint32_t)live.size();

    uint8_t* ptr = buf.data() + sizeof(MsgVmTopoUpdate);
    for (auto& nid : live) {
        std::strncpy(reinterpret_cast<char*>(ptr), nid.c_str(), id_stride - 1);
        ptr += id_stride;
    }

    for (auto& nid : live) {
        Connection* conn = get_conn_(nid);
        if (!conn || !conn->is_connected()) continue;
        try {
            conn->send_msg(static_cast<MsgType>(VmMsgType::VM_TOPO_UPDATE),
                           buf.data(), payload_size);
        } catch (...) {
            std::cerr << "[FaultManager] VM_TOPO_UPDATE send failed to " << nid << "\n";
        }
    }
}

void FaultManager::send_restore_requests(
        const std::vector<std::pair<uint64_t, std::string>>& migrations) {

    if (migrations.empty() || committed_snap_id_ == 0) return;

    // Build a quick vaddr→TensorDesc map from the last snapshot
    std::unordered_map<uint64_t, TensorDesc> snap_map;
    {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto& d : committed_snap_) snap_map.emplace(d.vaddr, d);
    }

    for (auto& [vaddr, new_node] : migrations) {
        auto sit = snap_map.find(vaddr);
        if (sit == snap_map.end()) continue;
        const TensorDesc& d = sit->second;

        Connection* conn = get_conn_(new_node);
        if (!conn || !conn->is_connected()) continue;

        MsgVmRestoreReq req{};
        req.snap_id  = committed_snap_id_;
        req.vaddr    = vaddr;
        req.n_bytes  = d.n_bytes;
        req.dtype    = d.dtype;

        try {
            conn->send_msg(static_cast<MsgType>(VmMsgType::VM_RESTORE_REQ),
                           reinterpret_cast<uint8_t*>(&req), sizeof(req));
        } catch (...) {
            std::cerr << "[FaultManager] VM_RESTORE_REQ send failed vaddr=" << vaddr
                      << " to " << new_node << "\n";
        }
    }
}

} // namespace dist
