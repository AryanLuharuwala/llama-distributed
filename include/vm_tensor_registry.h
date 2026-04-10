#pragma once
/**
 * vm_tensor_registry.h
 *
 * VmTensorRegistry — the unified virtual address space.
 *
 * Maps 64-bit virtual addresses to (owning_node, n_bytes, dtype).
 * Lives inside the VmCoordinator and is the authority for placement.
 *
 * Placement policy: least-bytes-allocated node (simple load balancing),
 * with optional per-call preference hint.
 *
 * Thread-safe for concurrent alloc/free/migrate/lookup.
 */

#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace dist {

struct TensorDesc {
    uint64_t    vaddr        = 0;
    std::string owning_node;
    uint64_t    n_bytes      = 0;
    uint16_t    dtype        = 0;   // 0=f32, 1=f16, 2=bf16
    bool        valid        = false;
};

class VmTensorRegistry {
public:
    VmTensorRegistry() = default;

    // ── Node registration ────────────────────────────────────────────────────

    // Register a node so the registry can place tensors on it.
    void register_node(const std::string& node_id);

    // Remove a node (used during fault recovery; its tensors will be migrated).
    void unregister_node(const std::string& node_id);

    // Returns all currently registered node IDs.
    std::vector<std::string> live_nodes() const;

    // ── Tensor lifecycle ─────────────────────────────────────────────────────

    // Allocate a virtual tensor.
    // preferred_node: if non-empty and alive, prefer that node.
    // Returns vaddr on success, 0 if no nodes available.
    uint64_t alloc(uint32_t n_bytes, uint16_t dtype,
                   const std::string& preferred_node = "");

    // Free a virtual address. No-op if not found.
    void free(uint64_t vaddr);

    // ── Lookup ───────────────────────────────────────────────────────────────

    // Returns a copy of the descriptor, or nullopt if not found.
    std::optional<TensorDesc> lookup(uint64_t vaddr) const;

    // Returns the owning node for a vaddr, or "" if not found.
    std::string owning_node(uint64_t vaddr) const;

    // ── Migration (fault recovery) ───────────────────────────────────────────

    // Reassign ownership of vaddr to new_node.
    // Called by FaultManager when old_node dies.
    void migrate(uint64_t vaddr, const std::string& new_node);

    // Migrate all tensors owned by dead_node to the least-loaded surviving node.
    // Returns list of migrated (vaddr, new_node) pairs.
    std::vector<std::pair<uint64_t, std::string>>
        migrate_all_from(const std::string& dead_node);

    // ── Snapshot / restore ───────────────────────────────────────────────────

    // Snapshot the entire table (for checkpointing).
    std::vector<TensorDesc> snapshot() const;

    // Restore from a snapshot (used after full cluster restart).
    void restore(const std::vector<TensorDesc>& snap);

    // ── Stats ────────────────────────────────────────────────────────────────

    size_t total_tensors() const;
    uint64_t bytes_on_node(const std::string& node_id) const;

private:
    // Pick the node with the fewest allocated bytes.
    // Call with mu_ held.
    std::string pick_node(const std::string& preferred) const;

    mutable std::mutex mu_;
    std::atomic<uint64_t> next_vaddr_ { 1 };

    std::unordered_map<uint64_t, TensorDesc>  table_;
    std::unordered_map<std::string, uint64_t> node_bytes_;  // node_id -> total bytes
};

} // namespace dist
