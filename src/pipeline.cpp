/**
 * pipeline.cpp
 */

#include "pipeline.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <sstream>

namespace dist {

using Clock = std::chrono::steady_clock;

// ─── LayerCache ─────────────────────────────────────────────────────────────

bool LayerCache::is_resident(uint32_t layer_idx) const {
    std::lock_guard<std::mutex> lk(mu_);
    return find_slot(layer_idx) >= 0;
}

void LayerCache::ensure_resident(uint32_t layer_idx,
                                  const LoadFn& load_fn,
                                  const EvictFn& evict_fn) {
    std::lock_guard<std::mutex> lk(mu_);

    if (find_slot(layer_idx) >= 0) {
        lru_[layer_idx] = ++tick_;
        return; // already resident
    }

    // Find a free slot or evict LRU
    int slot_idx = -1;
    for (int i = 0; i < (int)slots_.size(); ++i) {
        if (slots_[i].layer_idx == UINT32_MAX) { slot_idx = i; break; }
    }

    if (slot_idx < 0) {
        slot_idx = find_lru_evictable();
        if (slot_idx < 0) {
            // All slots in use; wait is handled by caller via polling
            return;
        }
        uint32_t evicted = slots_[slot_idx].layer_idx;
        evict_fn(evicted);
        lru_.erase(evicted);
        slots_[slot_idx].layer_idx = UINT32_MAX;
    }

    slots_[slot_idx].layer_idx = layer_idx;
    lru_[layer_idx] = ++tick_;
    load_fn(layer_idx);
}

void LayerCache::lock(uint32_t layer_idx) {
    std::lock_guard<std::mutex> lk(mu_);
    int idx = find_slot(layer_idx);
    if (idx >= 0) slots_[idx].in_use = true;
}

void LayerCache::unlock(uint32_t layer_idx) {
    std::lock_guard<std::mutex> lk(mu_);
    int idx = find_slot(layer_idx);
    if (idx >= 0) slots_[idx].in_use = false;
}

int LayerCache::find_slot(uint32_t layer_idx) const {
    for (int i = 0; i < (int)slots_.size(); ++i) {
        if (slots_[i].layer_idx == layer_idx) return i;
    }
    return -1;
}

int LayerCache::find_lru_evictable() const {
    int    best_idx  = -1;
    uint64_t best_t = UINT64_MAX;
    for (int i = 0; i < (int)slots_.size(); ++i) {
        if (slots_[i].in_use || slots_[i].layer_idx == UINT32_MAX) continue;
        auto it = lru_.find(slots_[i].layer_idx);
        uint64_t t = (it != lru_.end()) ? it->second : 0;
        if (t < best_t) { best_t = t; best_idx = i; }
    }
    return best_idx;
}

// ─── PipelineStage ──────────────────────────────────────────────────────────

PipelineStage::PipelineStage(NodeAgent* agent, PipelineConfig cfg)
    : agent_(agent),
      cfg_(cfg),
      layer_cache_(cfg.n_vram_slots),
      input_queue_(cfg.queue_depth),
      output_queue_(cfg.queue_depth)
{}

PipelineStage::~PipelineStage() {
    stop();
}

void PipelineStage::start() {
    running_.store(true);
    prefetch_cursor_.store(cfg_.layer_first);

    prefetch_thread_ = std::thread([this]{ prefetch_thread_fn(); });
    compute_thread_  = std::thread([this]{ compute_thread_fn();  });
}

void PipelineStage::stop() {
    running_.store(false);
    input_queue_.close();
    output_queue_.close();
    if (prefetch_thread_.joinable()) prefetch_thread_.join();
    if (compute_thread_.joinable())  compute_thread_.join();
}

bool PipelineStage::push_input(ActivationBatch batch) {
    return input_queue_.push(std::move(batch));
}

std::optional<ActivationBatch> PipelineStage::pop_output(
        std::chrono::milliseconds timeout) {
    return output_queue_.pop_timeout(timeout);
}

// ─── Prefetch thread ─────────────────────────────────────────────────────────
//
// Watches which layer the compute thread is currently on and keeps
// the next N layers loaded into VRAM ahead of time.

void PipelineStage::prefetch_thread_fn() {
    // No-op load/evict functions: real implementation would call ggml_backend APIs
    // to move weight tensors between CPU-pinned buffers and VRAM.
    auto load_fn = [](uint32_t layer_idx) {
        // TODO: ggml_backend_buffer_set_usage / copy weights to VRAM
        (void)layer_idx;
    };
    auto evict_fn = [](uint32_t layer_idx) {
        // TODO: release VRAM, move weights back to CPU pinned buffer
        (void)layer_idx;
    };

    while (running_.load()) {
        uint32_t cur = prefetch_cursor_.load();
        // Prefetch `prefetch_ahead` layers starting from current
        for (uint32_t i = 0; i < cfg_.prefetch_ahead; ++i) {
            uint32_t target = cur + i;
            if (target > cfg_.layer_last) break;
            layer_cache_.ensure_resident(target, load_fn, evict_fn);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

// ─── Compute thread ──────────────────────────────────────────────────────────

void PipelineStage::compute_thread_fn() {
    while (running_.load()) {
        // Wait for next batch (double-buffer: the other slot is being filled by net)
        auto t_wait_start = Clock::now();
        auto item = input_queue_.pop_timeout(std::chrono::milliseconds(200));
        auto t_wait_end = Clock::now();

        if (!item) continue;

        stats_.recv_wait_us.fetch_add(
            std::chrono::duration_cast<std::chrono::microseconds>(
                t_wait_end - t_wait_start).count());

        // Update prefetch cursor
        prefetch_cursor_.store(item->header.from_layer + 1);

        // Lock the layers we're about to use (prevent eviction mid-compute)
        for (uint32_t l = cfg_.layer_first; l <= cfg_.layer_last; ++l) {
            layer_cache_.lock(l);
        }

        // ── GPU COMPUTE ──────────────────────────────────────────────────────
        auto t_compute_start = Clock::now();

        // Delegate to the NodeAgent which owns the llama context
        // The NodeAgent's recv_queue_ is our input_queue_ (logical coupling).
        // For the PipelineStage wrapper we call run_layers indirectly by
        // pushing into the agent's internal receive queue.
        // In the full integration, PipelineStage *is* the compute primitive
        // and directly calls into ggml graph execution.
        //
        // For now: pass-through with stat tracking.
        ActivationBatch out;
        out.header          = item->header;
        out.header.from_layer = cfg_.layer_last;
        // NOTE: actual layer execution happens in NodeAgent::compute_thread_fn.
        // This class adds the scheduling / prefetch layer on top.
        out.data            = item->data; // placeholder; real data set by agent

        auto t_compute_end = Clock::now();
        stats_.compute_us.fetch_add(
            std::chrono::duration_cast<std::chrono::microseconds>(
                t_compute_end - t_compute_start).count());

        // Unlock layers
        for (uint32_t l = cfg_.layer_first; l <= cfg_.layer_last; ++l) {
            layer_cache_.unlock(l);
        }

        stats_.tokens_processed.fetch_add(item->header.n_tokens);
        stats_.batches_processed.fetch_add(1);

        if (!cfg_.is_last_stage) {
            auto t_send_start = Clock::now();
            output_queue_.push(std::move(out));
            stats_.send_wait_us.fetch_add(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    Clock::now() - t_send_start).count());
        }
    }
}

void PipelineStage::print_stats(std::ostream& out) const {
    uint64_t tokens   = stats_.tokens_processed.load();
    uint64_t batches  = stats_.batches_processed.load();
    uint64_t comp_us  = stats_.compute_us.load();
    uint64_t recv_us  = stats_.recv_wait_us.load();
    uint64_t send_us  = stats_.send_wait_us.load();

    double comp_ms    = comp_us  / 1000.0;
    double recv_ms    = recv_us  / 1000.0;
    double send_ms    = send_us  / 1000.0;

    double tps = (comp_us > 0)
        ? (double)tokens / (comp_us / 1e6)
        : 0.0;

    out << "PipelineStage layers=" << cfg_.layer_first << "-" << cfg_.layer_last << "\n"
        << "  tokens=" << tokens << "  batches=" << batches << "\n"
        << "  compute="  << comp_ms  << "ms  recv_wait=" << recv_ms
        << "ms  send_wait=" << send_ms << "ms\n"
        << "  throughput=" << tps << " tok/s\n";
}

} // namespace dist
