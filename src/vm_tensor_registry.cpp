/**
 * vm_tensor_registry.cpp
 */

#include "vm_tensor_registry.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace dist {

// ─── Node registration ───────────────────────────────────────────────────────

void VmTensorRegistry::register_node(const std::string& node_id) {
    std::lock_guard<std::mutex> lk(mu_);
    node_bytes_.emplace(node_id, uint64_t{0});
}

void VmTensorRegistry::unregister_node(const std::string& node_id) {
    std::lock_guard<std::mutex> lk(mu_);
    node_bytes_.erase(node_id);
    // Note: tensors previously owned by this node still exist in table_
    // but their owning_node will no longer appear in node_bytes_.
    // FaultManager calls migrate_all_from() before unregister_node().
}

std::vector<std::string> VmTensorRegistry::live_nodes() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<std::string> ids;
    ids.reserve(node_bytes_.size());
    for (auto& [id, _] : node_bytes_) ids.push_back(id);
    return ids;
}

// ─── Tensor lifecycle ────────────────────────────────────────────────────────

uint64_t VmTensorRegistry::alloc(uint32_t n_bytes, uint16_t dtype,
                                  const std::string& preferred_node) {
    std::lock_guard<std::mutex> lk(mu_);
    if (node_bytes_.empty()) return 0;

    std::string target = pick_node(preferred_node);
    if (target.empty()) return 0;

    uint64_t vaddr = next_vaddr_.fetch_add(1);

    TensorDesc desc;
    desc.vaddr        = vaddr;
    desc.owning_node  = target;
    desc.n_bytes      = n_bytes;
    desc.dtype        = dtype;
    desc.valid        = true;

    table_.emplace(vaddr, desc);
    node_bytes_[target] += n_bytes;
    return vaddr;
}

void VmTensorRegistry::free(uint64_t vaddr) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = table_.find(vaddr);
    if (it == table_.end()) return;

    const TensorDesc& d = it->second;
    auto nit = node_bytes_.find(d.owning_node);
    if (nit != node_bytes_.end()) {
        nit->second = (nit->second >= d.n_bytes) ? nit->second - d.n_bytes : 0;
    }
    table_.erase(it);
}

// ─── Lookup ──────────────────────────────────────────────────────────────────

std::optional<TensorDesc> VmTensorRegistry::lookup(uint64_t vaddr) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = table_.find(vaddr);
    if (it == table_.end()) return std::nullopt;
    return it->second;
}

std::string VmTensorRegistry::owning_node(uint64_t vaddr) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = table_.find(vaddr);
    return (it != table_.end()) ? it->second.owning_node : "";
}

// ─── Migration ───────────────────────────────────────────────────────────────

void VmTensorRegistry::migrate(uint64_t vaddr, const std::string& new_node) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = table_.find(vaddr);
    if (it == table_.end()) return;

    TensorDesc& d = it->second;
    const std::string old_node = d.owning_node;

    // Update byte accounting
    auto old_it = node_bytes_.find(old_node);
    if (old_it != node_bytes_.end()) {
        old_it->second = (old_it->second >= d.n_bytes) ? old_it->second - d.n_bytes : 0;
    }
    auto new_it = node_bytes_.find(new_node);
    if (new_it != node_bytes_.end()) {
        new_it->second += d.n_bytes;
    }

    d.owning_node = new_node;
}

std::vector<std::pair<uint64_t, std::string>>
VmTensorRegistry::migrate_all_from(const std::string& dead_node) {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<std::pair<uint64_t, std::string>> result;

    for (auto& [vaddr, desc] : table_) {
        if (desc.owning_node != dead_node) continue;

        // Pick the least-loaded alive node (excluding dead_node)
        std::string best;
        uint64_t    best_bytes = std::numeric_limits<uint64_t>::max();
        for (auto& [nid, nbytes] : node_bytes_) {
            if (nid == dead_node) continue;
            if (nbytes < best_bytes) { best_bytes = nbytes; best = nid; }
        }
        if (best.empty()) continue;

        // Update accounting
        auto old_it = node_bytes_.find(dead_node);
        if (old_it != node_bytes_.end()) {
            old_it->second = (old_it->second >= desc.n_bytes)
                             ? old_it->second - desc.n_bytes : 0;
        }
        node_bytes_[best] += desc.n_bytes;
        desc.owning_node = best;
        result.emplace_back(vaddr, best);
    }
    return result;
}

// ─── Snapshot / restore ──────────────────────────────────────────────────────

std::vector<TensorDesc> VmTensorRegistry::snapshot() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<TensorDesc> out;
    out.reserve(table_.size());
    for (auto& [_, d] : table_) out.push_back(d);
    return out;
}

void VmTensorRegistry::restore(const std::vector<TensorDesc>& snap) {
    std::lock_guard<std::mutex> lk(mu_);
    table_.clear();
    node_bytes_.clear();
    uint64_t max_vaddr = 0;
    for (const TensorDesc& d : snap) {
        table_.emplace(d.vaddr, d);
        node_bytes_[d.owning_node] += d.n_bytes;
        if (d.vaddr > max_vaddr) max_vaddr = d.vaddr;
    }
    // Ensure next_vaddr_ won't collide with restored addresses
    next_vaddr_.store(max_vaddr + 1);
}

// ─── Stats ───────────────────────────────────────────────────────────────────

size_t VmTensorRegistry::total_tensors() const {
    std::lock_guard<std::mutex> lk(mu_);
    return table_.size();
}

uint64_t VmTensorRegistry::bytes_on_node(const std::string& node_id) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = node_bytes_.find(node_id);
    return (it != node_bytes_.end()) ? it->second : 0;
}

// ─── Private helpers ─────────────────────────────────────────────────────────

std::string VmTensorRegistry::pick_node(const std::string& preferred) const {
    // Try preferred first
    if (!preferred.empty() && node_bytes_.count(preferred)) {
        return preferred;
    }
    // Least bytes allocated
    std::string best;
    uint64_t    best_bytes = std::numeric_limits<uint64_t>::max();
    for (auto& [nid, nbytes] : node_bytes_) {
        if (nbytes < best_bytes) { best_bytes = nbytes; best = nid; }
    }
    return best;
}

} // namespace dist
