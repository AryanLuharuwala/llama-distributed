/**
 * vm_collective.cpp
 *
 * Ring-AllReduce, AllGather, Broadcast implementation.
 *
 * Each phase of ring-allreduce:
 *
 *   N nodes, data_bytes = N * shard_bytes.
 *
 *   Scatter-reduce (N-1 rounds):
 *     Round r: node i sends shard (i - r) % N to node (i+1) % N.
 *              node i receives shard (i - r - 1) % N from node (i-1) % N.
 *              node i accumulates received shard into its accumulator.
 *     After N-1 rounds: node i holds fully-reduced shard i.
 *
 *   All-gather (N-1 rounds):
 *     Round r: node i sends its current result shard (which is now
 *              the fully-reduced shard (i - (N-1) - r) % N) to node (i+1)%N.
 *              node i receives and copies (no reduction).
 *     After N-1 rounds: node i holds the full reduced tensor.
 *
 * For AllGather: only the gather phase runs (no accumulation).
 * For Broadcast: root sends full data N-1 times (no ring needed, simpler tree).
 */

#include "vm_collective.h"

#include <cassert>
#include <cstring>
#include <iostream>

namespace dist {

CollectiveEngine::CollectiveEngine(std::string local_id, GetConnFn get_conn)
    : local_id_(std::move(local_id))
    , get_conn_(std::move(get_conn))
{}

// ─── Initiate: AllReduce ─────────────────────────────────────────────────────

CollectiveFuture CollectiveEngine::all_reduce(
        uint64_t coll_id,
        const std::vector<std::string>& participants,
        std::vector<uint8_t> data,
        uint16_t dtype,
        ReduceOp reduce_op) {

    auto promise = std::make_shared<std::promise<CollectiveResult>>();
    std::shared_future<CollectiveResult> fut = promise->get_future().share();

    int my_pos = -1;
    for (int i = 0; i < (int)participants.size(); ++i) {
        if (participants[i] == local_id_) { my_pos = i; break; }
    }
    if (my_pos < 0 || participants.size() < 2) {
        CollectiveResult r;
        r.success = true;
        r.data    = std::move(data); // nothing to reduce
        promise->set_value(std::move(r));
        return CollectiveFuture(coll_id, fut);
    }

    uint32_t N           = (uint32_t)participants.size();
    uint32_t data_bytes  = (uint32_t)data.size();
    uint32_t shard_bytes = data_bytes / N; // each node's shard

    CollState s;
    s.coll_id              = coll_id;
    s.op                   = CollectiveOp::AllReduce;
    s.reduce_op            = reduce_op;
    s.dtype                = dtype;
    s.ring                 = participants;
    s.my_pos               = my_pos;
    s.shard_bytes          = shard_bytes;
    s.data_bytes           = data_bytes;
    s.accumulator          = std::move(data);  // start with own data
    s.scatter_rounds_done  = 0;
    s.gather_rounds_done   = 0;
    s.scatter_phase_done   = false;
    s.done                 = false;
    s.promise              = promise;

    // Kick off scatter phase: send our initial shard to ring-next
    uint32_t shard_idx = (uint32_t)my_pos;
    uint32_t offset    = shard_idx * shard_bytes;
    {
        std::lock_guard<std::mutex> lk(mu_);
        active_.emplace(coll_id, std::move(s));
        auto& cs = active_.at(coll_id);
        send_chunk(cs, 0, cs.accumulator.data() + offset, offset, shard_bytes);
    }
    return CollectiveFuture(coll_id, fut);
}

// ─── Initiate: AllGather ─────────────────────────────────────────────────────

CollectiveFuture CollectiveEngine::all_gather(
        uint64_t coll_id,
        const std::vector<std::string>& participants,
        std::vector<uint8_t> local_shard,
        uint16_t dtype) {

    auto promise = std::make_shared<std::promise<CollectiveResult>>();
    std::shared_future<CollectiveResult> fut = promise->get_future().share();

    int my_pos = -1;
    for (int i = 0; i < (int)participants.size(); ++i) {
        if (participants[i] == local_id_) { my_pos = i; break; }
    }
    if (my_pos < 0 || participants.size() < 2) {
        CollectiveResult r;
        r.success = true;
        r.data    = std::move(local_shard);
        promise->set_value(std::move(r));
        return CollectiveFuture(coll_id, fut);
    }

    uint32_t N           = (uint32_t)participants.size();
    uint32_t shard_bytes = (uint32_t)local_shard.size();
    uint32_t data_bytes  = shard_bytes * N;

    CollState s;
    s.coll_id              = coll_id;
    s.op                   = CollectiveOp::AllGather;
    s.reduce_op            = ReduceOp::Sum; // unused
    s.dtype                = dtype;
    s.ring                 = participants;
    s.my_pos               = my_pos;
    s.shard_bytes          = shard_bytes;
    s.data_bytes           = data_bytes;
    s.gather_buf.resize(data_bytes, 0);
    // Place own shard at correct position in gather_buf
    std::memcpy(s.gather_buf.data() + (size_t)my_pos * shard_bytes,
                local_shard.data(), shard_bytes);
    s.accumulator          = std::move(local_shard);
    s.scatter_phase_done   = true; // no scatter needed
    s.gather_rounds_done   = 0;
    s.done                 = false;
    s.promise              = promise;

    {
        std::lock_guard<std::mutex> lk(mu_);
        active_.emplace(coll_id, std::move(s));
        auto& cs = active_.at(coll_id);
        // Kick off gather: send own shard to ring-next
        send_chunk(cs, 1, cs.accumulator.data(),
                   (uint32_t)cs.my_pos * shard_bytes, shard_bytes);
    }
    return CollectiveFuture(coll_id, fut);
}

// ─── Initiate: Broadcast ─────────────────────────────────────────────────────

CollectiveFuture CollectiveEngine::broadcast(
        uint64_t coll_id,
        const std::vector<std::string>& participants,
        const std::string& root_node,
        std::vector<uint8_t> data,
        uint16_t dtype) {

    auto promise = std::make_shared<std::promise<CollectiveResult>>();
    std::shared_future<CollectiveResult> fut = promise->get_future().share();

    bool is_root = (local_id_ == root_node);
    if (!is_root) {
        // Non-root: wait for gather phase to deliver the data.
        // We still need a CollState so on_chunk knows what to do.
    }

    int my_pos = -1;
    for (int i = 0; i < (int)participants.size(); ++i) {
        if (participants[i] == local_id_) { my_pos = i; break; }
    }
    if (my_pos < 0) {
        CollectiveResult r;
        r.success = false;
        r.error   = "local node not in participants";
        promise->set_value(std::move(r));
        return CollectiveFuture(coll_id, fut);
    }

    uint32_t data_bytes = is_root ? (uint32_t)data.size() : 0;

    CollState s;
    s.coll_id             = coll_id;
    s.op                  = CollectiveOp::Broadcast;
    s.reduce_op           = ReduceOp::Sum; // unused
    s.dtype               = dtype;
    s.ring                = participants;
    s.my_pos              = my_pos;
    s.shard_bytes         = data_bytes; // full data for broadcast
    s.data_bytes          = data_bytes;
    s.accumulator         = is_root ? std::move(data) : std::vector<uint8_t>{};
    s.scatter_phase_done  = true; // broadcast = gather only
    s.gather_rounds_done  = 0;
    s.done                = false;
    s.promise             = promise;

    {
        std::lock_guard<std::mutex> lk(mu_);
        active_.emplace(coll_id, std::move(s));
        if (is_root) {
            auto& cs = active_.at(coll_id);
            // Root sends full data to ring-next
            send_chunk(cs, 1, cs.accumulator.data(), 0, data_bytes);
        }
        // Non-root: wait for chunk from ring-prev
    }
    return CollectiveFuture(coll_id, fut);
}

// ─── Chunk reception ─────────────────────────────────────────────────────────

void CollectiveEngine::on_chunk(uint64_t           coll_id,
                                 const std::string& from_node,
                                 uint8_t            phase,
                                 uint32_t           chunk_offset,
                                 const uint8_t*     chunk_data,
                                 uint32_t           chunk_bytes) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = active_.find(coll_id);
    if (it == active_.end()) return;
    CollState& s = it->second;
    if (s.done) return;

    if (phase == 0) {
        // Scatter-reduce phase
        advance_scatter(s, chunk_data, chunk_offset, chunk_bytes);
    } else {
        // Gather phase
        advance_gather(s, chunk_data, chunk_offset, chunk_bytes);
    }

    if (s.done) active_.erase(it);
}

// ─── Private: scatter advance ────────────────────────────────────────────────

void CollectiveEngine::advance_scatter(CollState& s,
                                        const uint8_t* data,
                                        uint32_t offset, uint32_t bytes) {
    uint32_t N = (uint32_t)s.ring.size();

    // Accumulate received shard into our accumulator
    if (offset + bytes <= (uint32_t)s.accumulator.size()) {
        reduce_inplace(s.accumulator.data() + offset, data, bytes,
                       s.dtype, s.reduce_op);
    }
    s.scatter_rounds_done++;

    if (s.scatter_rounds_done >= N - 1) {
        // Scatter phase done; begin gather phase
        s.scatter_phase_done = true;

        if (s.op == CollectiveOp::AllReduce) {
            // Initialise gather_buf with our current accumulator state
            s.gather_buf = s.accumulator;
            // Send our reduced shard to ring-next
            uint32_t my_shard_off = (uint32_t)s.my_pos * s.shard_bytes;
            send_chunk(s, 1,
                       s.gather_buf.data() + my_shard_off,
                       my_shard_off, s.shard_bytes);
        }
    } else {
        // Forward received shard to ring-next (pass-through in scatter)
        send_chunk(s, 0, data, offset, bytes);
    }
}

// ─── Private: gather advance ─────────────────────────────────────────────────

void CollectiveEngine::advance_gather(CollState& s,
                                       const uint8_t* data,
                                       uint32_t offset, uint32_t bytes) {
    uint32_t N = (uint32_t)s.ring.size();

    // Copy received shard into gather_buf
    if (s.gather_buf.size() < s.data_bytes) s.gather_buf.resize(s.data_bytes, 0);
    if (offset + bytes <= (uint32_t)s.gather_buf.size()) {
        std::memcpy(s.gather_buf.data() + offset, data, bytes);
    }
    s.gather_rounds_done++;

    if (s.gather_rounds_done >= N - 1) {
        // Gather complete
        resolve(s);
    } else {
        // Forward to ring-next
        send_chunk(s, 1, data, offset, bytes);
    }
}

// ─── Private: resolve future ─────────────────────────────────────────────────

void CollectiveEngine::resolve(CollState& s) {
    s.done = true;
    CollectiveResult r;
    r.success = true;
    r.data    = (s.op == CollectiveOp::AllGather || s.op == CollectiveOp::Broadcast)
                ? std::move(s.gather_buf)
                : std::move(s.gather_buf.empty() ? s.accumulator : s.gather_buf);
    s.promise->set_value(std::move(r));
}

// ─── Private: send chunk ─────────────────────────────────────────────────────

void CollectiveEngine::send_chunk(CollState& s, uint8_t phase,
                                   const uint8_t* data,
                                   uint32_t offset, uint32_t bytes) {
    Connection* conn = get_conn_(s.ring_next());
    if (!conn || !conn->is_connected()) return;

    // Build VM_COLLECTIVE_CHUNK payload
    std::vector<uint8_t> payload(sizeof(MsgVmCollectiveChunk) + bytes);
    auto& hdr = *reinterpret_cast<MsgVmCollectiveChunk*>(payload.data());
    hdr.coll_id      = s.coll_id;
    hdr.chunk_offset = offset;
    hdr.chunk_bytes  = bytes;
    hdr.phase        = phase;
    std::strncpy(hdr.from_node, local_id_.c_str(), sizeof(hdr.from_node) - 1);
    if (bytes > 0) std::memcpy(payload.data() + sizeof(MsgVmCollectiveChunk), data, bytes);

    try {
        conn->send_msg(static_cast<MsgType>(VmMsgType::VM_COLLECTIVE_CHUNK),
                       payload.data(), (uint32_t)payload.size());
    } catch (...) {
        std::cerr << "[CollectiveEngine] send_chunk failed for coll_id=" << s.coll_id << "\n";
    }
}

} // namespace dist
