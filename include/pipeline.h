#pragma once
/**
 * pipeline.h
 *
 * PipelineStage: ties together the compute and I/O threads for one node.
 *
 * The key insight is double-buffering:
 *
 *   Slot A: GPU computing layer N
 *   Slot B: network receiving next activation batch (overlapped)
 *
 * While GPU executes the current batch, the NIC is already streaming
 * in the next batch. As soon as GPU finishes, we swap slots — zero wait.
 *
 *   ┌────────┐   recv_queue   ┌─────────┐   send_queue   ┌────────┐
 *   │  NET   │ ─────────────> │  GPU    │ ─────────────> │  NET   │
 *   │  RX    │                │ COMPUTE │                │  TX    │
 *   └────────┘                └─────────┘                └────────┘
 *     thread                    thread                     thread
 *       (fills slot B)           (drains slot A,           (drains send_queue)
 *                                 fills send_queue)
 *
 * Layer weight prefetch:
 *   When layers don't fit in VRAM, weights are kept in pinned CPU RAM.
 *   LayerCache tracks which layers are currently in VRAM.
 *   The prefetch thread moves the next-needed layers to VRAM while
 *   the current layers are executing — one step ahead.
 */

#include "dist_protocol.h"
#include "dist_queue.h"
#include "node_agent.h"

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace dist {

// ─── Layer cache (CPU-RAM <-> GPU VRAM management) ──────────────────────────

struct LayerSlot {
    uint32_t  layer_idx   = UINT32_MAX;  // which model layer is loaded here
    void*     gpu_buf     = nullptr;     // pointer to VRAM buffer (ggml_backend_buffer)
    bool      in_use      = false;       // GPU is computing with this slot right now
};

/**
 * LayerCache manages a fixed pool of GPU VRAM slots.
 * When a layer is needed and all slots are full, it evicts the LRU slot.
 *
 * For now this wraps simple bookkeeping; actual GPU memory management
 * is done through llama.cpp's llama_model_params.tensor_split mechanism
 * at model load time. A deeper integration would call ggml_backend_buffer_*
 * directly.
 */
class LayerCache {
public:
    explicit LayerCache(uint32_t n_slots) : slots_(n_slots) {}

    // Returns true if layer is already in a VRAM slot.
    bool is_resident(uint32_t layer_idx) const;

    // Mark layer as needed soon; triggers async prefetch if not resident.
    // Calls the provided load_fn(layer_idx) synchronously if needed.
    using LoadFn = std::function<void(uint32_t layer_idx)>;
    using EvictFn = std::function<void(uint32_t layer_idx)>;

    void ensure_resident(uint32_t layer_idx,
                         const LoadFn& load_fn,
                         const EvictFn& evict_fn);

    // Mark a slot as in-use (cannot be evicted).
    void lock(uint32_t layer_idx);
    void unlock(uint32_t layer_idx);

private:
    mutable std::mutex       mu_;
    std::vector<LayerSlot>   slots_;
    // LRU tracking: layer_idx -> last_access_time
    std::unordered_map<uint32_t, uint64_t> lru_;
    uint64_t tick_ = 0;

    int find_slot(uint32_t layer_idx) const; // -1 if not found; call with mu_ held
    int find_lru_evictable() const;          // -1 if all in-use; call with mu_ held
};

// ─── PipelineStats ───────────────────────────────────────────────────────────

struct PipelineStats {
    std::atomic<uint64_t> tokens_processed  { 0 };
    std::atomic<uint64_t> batches_processed { 0 };
    std::atomic<uint64_t> recv_bytes        { 0 };
    std::atomic<uint64_t> send_bytes        { 0 };
    std::atomic<uint64_t> compute_us        { 0 }; // microseconds in GPU compute
    std::atomic<uint64_t> recv_wait_us      { 0 }; // stalled waiting for upstream
    std::atomic<uint64_t> send_wait_us      { 0 }; // stalled waiting on send queue
};

// ─── PipelineStage ──────────────────────────────────────────────────────────

struct PipelineConfig {
    uint32_t   layer_first      = 0;
    uint32_t   layer_last       = 0;
    uint32_t   n_vram_slots     = 4;   // number of layer-weight VRAM slots
    uint32_t   queue_depth      = 4;   // recv/send queue depth (double-buffering)
    uint32_t   prefetch_ahead   = 2;   // layers to prefetch ahead
    bool       is_first_stage   = false;
    bool       is_last_stage    = false;
};

/**
 * PipelineStage wraps a NodeAgent and adds:
 *   - Explicit double-buffering control
 *   - Layer weight prefetch scheduling
 *   - Per-stage throughput statistics
 */
class PipelineStage {
public:
    explicit PipelineStage(NodeAgent* agent, PipelineConfig cfg);
    ~PipelineStage();

    void start();
    void stop();

    // Push a batch to this stage (for the first stage, called by API layer).
    bool push_input(ActivationBatch batch);

    // Pop a completed batch (for the last stage, or for testing).
    std::optional<ActivationBatch> pop_output(std::chrono::milliseconds timeout);

    const PipelineStats& stats() const { return stats_; }
    void print_stats(std::ostream& out) const;

private:
    void prefetch_thread_fn();
    void compute_thread_fn();

    // Schedule prefetch for the next N layers after current
    void schedule_prefetch(uint32_t current_layer);

    NodeAgent*       agent_;
    PipelineConfig   cfg_;
    PipelineStats    stats_;

    LayerCache       layer_cache_;

    // Queues shared with NodeAgent
    BoundedQueue<ActivationBatch> input_queue_;   // -> compute
    BoundedQueue<ActivationBatch> output_queue_;  // compute ->

    std::atomic<bool>  running_  { false };
    std::thread        prefetch_thread_;
    std::thread        compute_thread_;

    // Current layer being prefetched
    std::atomic<uint32_t> prefetch_cursor_ { 0 };
};

} // namespace dist
