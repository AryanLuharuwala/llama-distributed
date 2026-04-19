#pragma once
/**
 * cluster_monitor.h
 *
 * ClusterMonitor — thread-safe stats aggregator embedded in the Coordinator.
 *
 * Updated by:
 *   - on_node_join()       when a node registers
 *   - on_heartbeat()       from node heartbeats (HEARTBEAT messages)
 *   - on_node_left()       when a node disconnects
 *   - on_token_generated() for each generated token
 *   - on_tensor_forward()  for data-plane byte accounting
 *
 * Read by:
 *   - DashboardServer (HTTP SSE)
 *   - ClusterStats snapshots (CLUSTER_STATS_RSP)
 *
 * All public methods are thread-safe.
 */

#include "dist_protocol.h"

#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace dist {

struct LiveNodeStat {
    NodeStatEntry  entry;
    std::chrono::steady_clock::time_point last_seen;

    // Rolling TPS window: (timestamp, token_count) pairs over last 10 s
    std::deque<std::pair<std::chrono::steady_clock::time_point, uint64_t>> tps_window;
};

class ClusterMonitor {
public:
    ClusterMonitor() = default;

    // ── Called by Coordinator ─────────────────────────────────────────────────

    void on_node_join(const std::string& node_id,
                      const std::string& addr,
                      const NodeCapability& cap,
                      uint16_t data_port);

    void on_heartbeat(const std::string& node_id,
                      const MsgHeartbeat& hb);

    void on_node_left(const std::string& node_id);

    // Call when the coordinator assigns layers to a node.
    void on_layer_assign(const std::string& node_id,
                         uint32_t layer_first, uint32_t layer_last);

    // Call when model is confirmed loaded on a node.
    void on_model_loaded(const std::string& node_id,
                         const std::string& model_name,
                         uint32_t n_layers_total);

    // Call per generated token (may be called from multiple threads).
    void on_token_generated(const std::string& node_id, uint32_t n = 1);

    // Call when a tensor is forwarded to account for data-plane bytes.
    void on_tensor_forward(const std::string& from_node,
                           uint64_t bytes);

    // ── Snapshot (for HTTP dashboard + CLUSTER_STATS_RSP) ─────────────────────

    // Fill buf with MsgClusterStats + NodeStatEntry[].
    // Returns the number of bytes written.
    std::vector<uint8_t> snapshot() const;

    // Quick JSON summary for the HTTP dashboard endpoint.
    std::string to_json() const;

    // Returns a "join token" URL-safe string (coordinator-ip:port).
    std::string join_token(const std::string& public_host,
                           uint16_t ctrl_port) const;

private:
    void update_rolling_tps(LiveNodeStat& ns, uint64_t new_tokens);

    mutable std::mutex mu_;

    std::unordered_map<std::string, LiveNodeStat> nodes_;

    // Cluster-wide lifetime counters
    std::atomic<uint64_t> cluster_tokens_total_ { 0 };

    // Rolling cluster TPS
    mutable std::deque<std::pair<std::chrono::steady_clock::time_point, uint64_t>>
        cluster_tps_window_;

    // Active model
    std::string model_name_;
    uint32_t    n_layers_total_ = 0;

    // Active inference requests
    std::atomic<uint32_t> active_requests_ { 0 };

    friend class Coordinator;  // coordinator can bump active_requests_
};

} // namespace dist
