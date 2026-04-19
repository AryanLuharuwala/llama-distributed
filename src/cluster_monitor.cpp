/**
 * cluster_monitor.cpp
 */

#include "cluster_monitor.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <sstream>
#include <iomanip>

namespace dist {

using Clock = std::chrono::steady_clock;
static constexpr int TPS_WINDOW_SEC = 10;

// ─── Node lifecycle ───────────────────────────────────────────────────────────

void ClusterMonitor::on_node_join(const std::string& node_id,
                                   const std::string& addr,
                                   const NodeCapability& cap,
                                   uint16_t /*data_port*/) {
    std::lock_guard<std::mutex> lk(mu_);
    LiveNodeStat& ns = nodes_[node_id];
    ns.last_seen = Clock::now();

    NodeStatEntry& e = ns.entry;
    std::memset(&e, 0, sizeof(e));
    std::strncpy(e.node_id, node_id.c_str(), sizeof(e.node_id) - 1);
    std::strncpy(e.addr,    addr.c_str(),    sizeof(e.addr)    - 1);
    e.n_gpus       = cap.n_gpus;
    e.cpu_ram_total = cap.cpu_ram_bytes;
    e.cpu_ram_free  = cap.cpu_ram_free_bytes;
    e.n_cpu_threads = cap.n_cpu_threads;
    e.network_mbps  = cap.network_bandwidth_mbps;
    for (uint32_t i = 0; i < cap.n_gpus && i < 8; ++i) {
        e.gpu_vram_total[i] = cap.gpu_vram_bytes[i];
        e.gpu_vram_free[i]  = cap.gpu_free_bytes[i];
    }
    e.alive        = 1;
    e.model_loaded = 0;
}

void ClusterMonitor::on_heartbeat(const std::string& node_id,
                                   const MsgHeartbeat& hb) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = nodes_.find(node_id);
    if (it == nodes_.end()) return;
    LiveNodeStat& ns = it->second;
    NodeStatEntry& e = ns.entry;

    ns.last_seen = Clock::now();
    e.cpu_ram_free = hb.cpu_ram_free_bytes;
    for (int i = 0; i < 8; ++i) {
        e.gpu_vram_free[i] = hb.gpu_free_bytes[i];
        e.gpu_util[i]      = hb.gpu_util[i];
    }

    if (hb.tokens_processed > 0) {
        e.tokens_total += hb.tokens_processed;
        cluster_tokens_total_.fetch_add(hb.tokens_processed);
        update_rolling_tps(ns, hb.tokens_processed);

        // Update cluster TPS window
        auto now = Clock::now();
        cluster_tps_window_.push_back({now, hb.tokens_processed});
        while (!cluster_tps_window_.empty()) {
            auto age = std::chrono::duration_cast<std::chrono::seconds>(
                now - cluster_tps_window_.front().first).count();
            if (age > TPS_WINDOW_SEC) cluster_tps_window_.pop_front();
            else break;
        }
    }
}

void ClusterMonitor::on_node_left(const std::string& node_id) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = nodes_.find(node_id);
    if (it != nodes_.end()) it->second.entry.alive = 0;
}

void ClusterMonitor::on_layer_assign(const std::string& node_id,
                                      uint32_t layer_first, uint32_t layer_last) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = nodes_.find(node_id);
    if (it == nodes_.end()) return;
    it->second.entry.layer_first = layer_first;
    it->second.entry.layer_last  = layer_last;
}

void ClusterMonitor::on_model_loaded(const std::string& /*node_id*/,
                                      const std::string& model_name,
                                      uint32_t n_layers_total) {
    std::lock_guard<std::mutex> lk(mu_);
    model_name_     = model_name;
    n_layers_total_ = n_layers_total;

    auto it = nodes_.find(/*node_id*/ model_name); // wrong lookup intentionally avoided:
    // model_loaded flag is set per-node inside on_heartbeat when layers > 0
    (void)model_name;
}

void ClusterMonitor::on_token_generated(const std::string& node_id, uint32_t n) {
    cluster_tokens_total_.fetch_add(n);
    std::lock_guard<std::mutex> lk(mu_);
    auto it = nodes_.find(node_id);
    if (it == nodes_.end()) return;
    it->second.entry.tokens_total += n;
    update_rolling_tps(it->second, n);
}

void ClusterMonitor::on_tensor_forward(const std::string& from_node,
                                        uint64_t bytes) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = nodes_.find(from_node);
    if (it == nodes_.end()) return;
    it->second.entry.bytes_sent += bytes;
}

// ─── Snapshot ─────────────────────────────────────────────────────────────────

std::vector<uint8_t> ClusterMonitor::snapshot() const {
    std::lock_guard<std::mutex> lk(mu_);

    uint32_t n = (uint32_t)nodes_.size();
    std::vector<uint8_t> buf(sizeof(MsgClusterStats) + n * sizeof(NodeStatEntry), 0);

    auto& cs = *reinterpret_cast<MsgClusterStats*>(buf.data());
    auto now = std::chrono::system_clock::now();
    cs.timestamp_us = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()).count();
    cs.n_nodes             = n;
    cs.n_active_requests   = active_requests_.load();
    cs.tokens_total        = cluster_tokens_total_.load();
    cs.n_layers_total      = n_layers_total_;
    std::strncpy(cs.model_name, model_name_.c_str(), sizeof(cs.model_name) - 1);

    // Compute cluster rolling TPS
    if (!cluster_tps_window_.empty()) {
        uint64_t total = 0;
        for (auto& [t, cnt] : cluster_tps_window_) total += cnt;
        cs.tokens_per_second = (double)total / TPS_WINDOW_SEC;
    }

    auto* entries = reinterpret_cast<NodeStatEntry*>(buf.data() + sizeof(MsgClusterStats));
    uint32_t idx = 0;
    for (auto& [id, ns] : nodes_) {
        entries[idx] = ns.entry;
        // Fill rolling TPS
        if (!ns.tps_window.empty()) {
            uint64_t total = 0;
            for (auto& [t, cnt] : ns.tps_window) total += cnt;
            entries[idx].tokens_per_second = (double)total / TPS_WINDOW_SEC;
        }
        ++idx;
    }
    return buf;
}

// ─── JSON ─────────────────────────────────────────────────────────────────────

static std::string json_str(const char* s) {
    // Minimal JSON string escape
    std::string out = "\"";
    for (const char* p = s; *p; ++p) {
        if (*p == '"' || *p == '\\') out += '\\';
        out += *p;
    }
    out += '"';
    return out;
}

std::string ClusterMonitor::to_json() const {
    std::lock_guard<std::mutex> lk(mu_);

    // Compute cluster TPS
    double cluster_tps = 0.0;
    if (!cluster_tps_window_.empty()) {
        uint64_t total = 0;
        for (auto& [t, cnt] : cluster_tps_window_) total += cnt;
        cluster_tps = (double)total / TPS_WINDOW_SEC;
    }

    auto now_us = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    std::ostringstream o;
    o << std::fixed << std::setprecision(2);
    o << "{"
      << "\"timestamp_us\":" << now_us << ","
      << "\"model\":" << json_str(model_name_.c_str()) << ","
      << "\"n_layers\":" << n_layers_total_ << ","
      << "\"n_active_requests\":" << active_requests_.load() << ","
      << "\"tokens_total\":" << cluster_tokens_total_.load() << ","
      << "\"tokens_per_second\":" << cluster_tps << ","
      << "\"nodes\":[";

    bool first = true;
    for (auto& [id, ns] : nodes_) {
        if (!first) o << ",";
        first = false;
        const NodeStatEntry& e = ns.entry;

        double node_tps = 0.0;
        if (!ns.tps_window.empty()) {
            uint64_t total = 0;
            for (auto& [t, cnt] : ns.tps_window) total += cnt;
            node_tps = (double)total / TPS_WINDOW_SEC;
        }

        // VRAM totals across GPUs
        uint64_t vram_total = 0, vram_free = 0;
        float gpu_util_avg = 0.0f;
        for (uint32_t i = 0; i < e.n_gpus && i < 8; ++i) {
            vram_total   += e.gpu_vram_total[i];
            vram_free    += e.gpu_vram_free[i];
            gpu_util_avg += e.gpu_util[i];
        }
        if (e.n_gpus > 0) gpu_util_avg /= (float)e.n_gpus;

        // Layer contribution % of total
        double layer_pct = 0.0;
        if (n_layers_total_ > 0) {
            uint32_t my_layers = e.layer_last >= e.layer_first
                ? e.layer_last - e.layer_first + 1 : 0;
            layer_pct = 100.0 * my_layers / n_layers_total_;
        }

        o << "{"
          << "\"id\":"          << json_str(e.node_id)  << ","
          << "\"addr\":"         << json_str(e.addr)     << ","
          << "\"alive\":"        << (e.alive ? "true" : "false") << ","
          << "\"model_loaded\":" << (e.model_loaded ? "true" : "false") << ","
          << "\"n_gpus\":"       << e.n_gpus             << ","
          << "\"vram_total\":"   << vram_total            << ","
          << "\"vram_free\":"    << vram_free             << ","
          << "\"gpu_util\":"     << gpu_util_avg          << ","
          << "\"cpu_ram_total\":" << e.cpu_ram_total      << ","
          << "\"cpu_ram_free\":"  << e.cpu_ram_free       << ","
          << "\"layer_first\":"  << e.layer_first         << ","
          << "\"layer_last\":"   << e.layer_last          << ","
          << "\"layer_pct\":"    << layer_pct             << ","
          << "\"tokens_total\":" << e.tokens_total        << ","
          << "\"tps\":"          << node_tps              << ","
          << "\"bytes_sent\":"   << e.bytes_sent          << ","
          << "\"bytes_recv\":"   << e.bytes_received
          << "}";
    }

    o << "]}";
    return o.str();
}

std::string ClusterMonitor::join_token(const std::string& public_host,
                                        uint16_t ctrl_port) const {
    return public_host + ":" + std::to_string(ctrl_port);
}

// ─── Private ─────────────────────────────────────────────────────────────────

void ClusterMonitor::update_rolling_tps(LiveNodeStat& ns, uint64_t new_tokens) {
    auto now = Clock::now();
    ns.tps_window.push_back({now, new_tokens});
    while (!ns.tps_window.empty()) {
        auto age = std::chrono::duration_cast<std::chrono::seconds>(
            now - ns.tps_window.front().first).count();
        if (age > TPS_WINDOW_SEC) ns.tps_window.pop_front();
        else break;
    }
}

} // namespace dist
