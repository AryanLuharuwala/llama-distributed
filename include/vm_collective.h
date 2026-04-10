#pragma once
/**
 * vm_collective.h
 *
 * CollectiveEngine — ring-based AllReduce, AllGather, and Broadcast.
 *
 * Runs on each VmNode and on the VmCoordinator (coord side just tracks
 * completion, doesn't participate in data movement).
 *
 * Ring AllReduce (bandwidth-optimal, O(N) messages):
 *   Phase 1 — Scatter-reduce: each node sends its shard to ring-next,
 *             receives and accumulates from ring-prev. After N-1 rounds
 *             each node holds a fully-reduced shard of size data/N.
 *   Phase 2 — All-gather: each node forwards its reduced shard. After
 *             N-1 rounds every node holds the full reduced tensor.
 *
 * AllGather: only Phase 2 (each node starts with its own shard).
 * Broadcast: root sends data to ring-next N-1 times (no reduction).
 *
 * All data messages use VM_COLLECTIVE_CHUNK over the existing PORT_DATA
 * connections so no new sockets need to be opened.
 *
 * Threading: CollectiveEngine has NO threads. It is driven by the caller:
 *   - initiate_*() sets up state and sends the first chunk.
 *   - on_chunk()   is called from the node's data_recv_thread whenever a
 *                  VM_COLLECTIVE_CHUNK arrives. It accumulates, forwards,
 *                  and resolves the future when done.
 */

#include "vm_protocol.h"
#include "dist_conn.h"

#include <atomic>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace dist {

// ─── CollectiveFuture ────────────────────────────────────────────────────────

struct CollectiveResult {
    bool                 success = false;
    std::vector<uint8_t> data;    // reduced/gathered tensor bytes
    std::string          error;
};

struct CollectiveFuture {
    uint64_t coll_id = 0;

    bool is_done() const {
        if (!fut_.valid()) return true;
        return fut_.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    }

    std::optional<CollectiveResult> wait(
            std::chrono::milliseconds timeout = std::chrono::milliseconds(30000)) {
        if (!fut_.valid()) return std::nullopt;
        if (fut_.wait_for(timeout) != std::future_status::ready) return std::nullopt;
        return fut_.get();
    }

    explicit CollectiveFuture(uint64_t id, std::shared_future<CollectiveResult> f)
        : coll_id(id), fut_(std::move(f)) {}
    CollectiveFuture() = default;

private:
    std::shared_future<CollectiveResult> fut_;
};

// ─── CollectiveEngine ────────────────────────────────────────────────────────

class CollectiveEngine {
public:
    /**
     * local_id      : this node's identity in the ring
     * get_conn      : callback that returns a Connection* for a given node_id,
     *                 or nullptr if the node is unreachable.
     *                 CollectiveEngine does NOT own the connections.
     */
    using GetConnFn = std::function<Connection*(const std::string& node_id)>;

    CollectiveEngine(std::string local_id, GetConnFn get_conn);

    // ── Initiate collectives ─────────────────────────────────────────────────

    // AllReduce: every node calls this with its own shard of `data`.
    // participants: ordered ring (the *same* order on every node).
    // Returns a future that resolves when this node has the full reduced tensor.
    CollectiveFuture all_reduce(
        uint64_t                       coll_id,   // assigned by VmCoordinator
        const std::vector<std::string>& participants,
        std::vector<uint8_t>            data,
        uint16_t                        dtype,
        ReduceOp                        reduce_op = ReduceOp::Sum);

    // AllGather: each node provides its shard; result is concatenation of all.
    CollectiveFuture all_gather(
        uint64_t                       coll_id,
        const std::vector<std::string>& participants,
        std::vector<uint8_t>            local_shard,
        uint16_t                        dtype);

    // Broadcast: root sends `data` to all; non-root nodes pass empty data.
    CollectiveFuture broadcast(
        uint64_t                       coll_id,
        const std::vector<std::string>& participants,
        const std::string&              root_node,
        std::vector<uint8_t>            data,
        uint16_t                        dtype);

    // ── Chunk reception (called by data recv loop) ───────────────────────────

    void on_chunk(uint64_t           coll_id,
                  const std::string& from_node,
                  uint8_t            phase,
                  uint32_t           chunk_offset,
                  const uint8_t*     chunk_data,
                  uint32_t           chunk_bytes);

private:
    struct CollState {
        uint64_t                 coll_id;
        CollectiveOp             op;
        ReduceOp                 reduce_op;
        uint16_t                 dtype;
        std::vector<std::string> ring;        // ordered participants
        int                      my_pos;      // this node's index in ring
        uint32_t                 shard_bytes; // bytes per shard (data_bytes / N)
        uint32_t                 data_bytes;  // total tensor bytes
        std::vector<uint8_t>     accumulator; // running partial result
        std::vector<uint8_t>     gather_buf;  // for AllGather result
        uint32_t                 scatter_rounds_done = 0;
        uint32_t                 gather_rounds_done  = 0;
        bool                     scatter_phase_done  = false;
        bool                     done                = false;
        std::shared_ptr<std::promise<CollectiveResult>> promise;

        // Ring helpers
        const std::string& ring_next() const {
            return ring[(my_pos + 1) % ring.size()];
        }
        const std::string& ring_prev() const {
            int n = (int)ring.size();
            return ring[(my_pos - 1 + n) % n];
        }
    };

    // Send a chunk to ring-next
    void send_chunk(CollState& s, uint8_t phase,
                    const uint8_t* data, uint32_t offset, uint32_t bytes);

    // Advance scatter phase after receiving a chunk
    void advance_scatter(CollState& s, const uint8_t* data,
                         uint32_t offset, uint32_t bytes);

    // Advance gather phase after receiving a chunk
    void advance_gather(CollState& s, const uint8_t* data,
                         uint32_t offset, uint32_t bytes);

    // Resolve the future with the final result
    void resolve(CollState& s);

    std::string  local_id_;
    GetConnFn    get_conn_;

    mutable std::mutex                            mu_;
    std::atomic<uint64_t>                         next_coll_id_ { 1 };
    std::unordered_map<uint64_t, CollState>        active_;
};

} // namespace dist
