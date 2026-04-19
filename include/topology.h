#pragma once
/**
 * topology.h
 *
 * Latency-aware multi-level ring formation for global-scale pools.
 *
 *   Level 0: intra-rack          (sub-ms RTT)
 *   Level 1: intra-zone / DC     (single-digit ms)
 *   Level 2: intra-region        (tens of ms)
 *   Level 3: inter-region        (100ms+)
 *
 * Each level runs its own ring so that collectives (all-reduce, all-gather)
 * exchange the bulk of their traffic inside the lowest-latency tier and only
 * cross higher tiers with summarised data.
 *
 * The registry is kept in memory by the coordinator.  Nodes self-report a
 * TopologyHello on join; the coordinator may also probe pairs and collect
 * TopologyLatency samples to refine the groupings.
 */

#include "dist_protocol.h"

#include <array>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace dist {

struct NodeLocation {
    std::string node_id;
    std::string region;          // "us-west"
    std::string zone;            // "us-west-2a"
    std::string rack;            // free-form; can be empty
    float       lat_deg = 0.f;
    float       lon_deg = 0.f;
    uint32_t    bandwidth_mbps = 0;
    bool        behind_nat = false;
};

// Which tier a pair of nodes sits in.  Lower = cheaper to communicate.
enum class TopologyTier : uint8_t {
    Rack     = 0,
    Zone     = 1,
    Region   = 2,
    Global   = 3,
};

// A ring at one tier.  The ordering is computed from nearest-neighbour
// latencies so that adjacent ring members are low-RTT peers.
struct Ring {
    TopologyTier             tier = TopologyTier::Global;
    std::string              group_key;     // e.g. "us-west/us-west-2a"
    std::vector<std::string> members;       // node_ids in ring order
};

class TopologyRegistry {
public:
    // Called from the coordinator's join handler.
    void upsert(const NodeLocation& loc);

    // Called when a node leaves or is declared dead.
    void remove(const std::string& node_id);

    // Record a latency sample from src -> dst (ms).
    void record_latency(const std::string& src,
                        const std::string& dst,
                        float rtt_ms);

    // Look up a node's reported location (copy).
    std::optional<NodeLocation> location(const std::string& node_id) const;

    // Return the best-known RTT between two nodes, or nullopt if never probed.
    std::optional<float> rtt_ms(const std::string& a, const std::string& b) const;

    // Classify a pair by tier using reported geography + measured RTT.
    TopologyTier classify(const std::string& a, const std::string& b) const;

    // Build rings at every tier.  Result is tier-grouped.
    // The member ordering within each ring uses nearest-neighbour TSP over RTT.
    std::vector<Ring> build_rings() const;

    // JSON dump for the dashboard.
    std::string to_json() const;

    // Per-tier RTT heuristics — members within this bound are grouped together
    // when no geography tag is available.
    struct Thresholds {
        float rack_rtt_ms   = 0.5f;
        float zone_rtt_ms   = 3.0f;
        float region_rtt_ms = 30.0f;
    };
    void set_thresholds(const Thresholds& t);

private:
    mutable std::mutex                                mtx_;
    std::unordered_map<std::string, NodeLocation>     nodes_;
    // Double-keyed RTT map.  Key is "a|b" with a<b lexicographic.
    std::unordered_map<std::string, float>            rtt_;
    Thresholds                                        thr_;

    static std::string rtt_key(const std::string& a, const std::string& b);

    // Internal: group by geo tag; fall back to RTT clustering when tag missing.
    std::vector<std::vector<std::string>>
        partition_by_tier(TopologyTier tier) const;

    // Greedy nearest-neighbour ordering of one group by measured RTT.
    std::vector<std::string>
        order_ring(const std::vector<std::string>& group) const;
};

// Helpers used by the node side to fill out a TopologyHello payload from
// environment variables + optional GeoIP file.
MsgTopologyHello make_topology_hello(const std::string& node_id);

} // namespace dist
