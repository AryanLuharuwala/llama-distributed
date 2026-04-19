#include "topology.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>

namespace dist {

// ─── TopologyRegistry ───────────────────────────────────────────────────────

void TopologyRegistry::set_thresholds(const Thresholds& t) {
    std::lock_guard<std::mutex> g(mtx_);
    thr_ = t;
}

void TopologyRegistry::upsert(const NodeLocation& loc) {
    std::lock_guard<std::mutex> g(mtx_);
    nodes_[loc.node_id] = loc;
}

void TopologyRegistry::remove(const std::string& node_id) {
    std::lock_guard<std::mutex> g(mtx_);
    nodes_.erase(node_id);
    for (auto it = rtt_.begin(); it != rtt_.end();) {
        if (it->first.find(node_id) != std::string::npos) it = rtt_.erase(it);
        else ++it;
    }
}

std::string TopologyRegistry::rtt_key(const std::string& a, const std::string& b) {
    return a < b ? (a + "|" + b) : (b + "|" + a);
}

void TopologyRegistry::record_latency(const std::string& src,
                                       const std::string& dst,
                                       float rtt_ms) {
    if (src == dst) return;
    std::lock_guard<std::mutex> g(mtx_);
    auto k = rtt_key(src, dst);
    auto it = rtt_.find(k);
    // EWMA to smooth jitter.
    if (it == rtt_.end()) rtt_[k] = rtt_ms;
    else                  it->second = 0.7f * it->second + 0.3f * rtt_ms;
}

std::optional<NodeLocation>
TopologyRegistry::location(const std::string& node_id) const {
    std::lock_guard<std::mutex> g(mtx_);
    auto it = nodes_.find(node_id);
    if (it == nodes_.end()) return std::nullopt;
    return it->second;
}

std::optional<float>
TopologyRegistry::rtt_ms(const std::string& a, const std::string& b) const {
    std::lock_guard<std::mutex> g(mtx_);
    auto it = rtt_.find(rtt_key(a, b));
    if (it == rtt_.end()) return std::nullopt;
    return it->second;
}

TopologyTier
TopologyRegistry::classify(const std::string& a, const std::string& b) const {
    std::lock_guard<std::mutex> g(mtx_);
    auto la = nodes_.find(a);
    auto lb = nodes_.find(b);
    if (la != nodes_.end() && lb != nodes_.end()) {
        const auto& A = la->second;
        const auto& B = lb->second;
        if (!A.rack.empty() && A.rack == B.rack && A.zone == B.zone
            && A.region == B.region) return TopologyTier::Rack;
        if (!A.zone.empty() && A.zone == B.zone && A.region == B.region)
            return TopologyTier::Zone;
        if (!A.region.empty() && A.region == B.region)
            return TopologyTier::Region;
        return TopologyTier::Global;
    }
    // Fall back to RTT bands.
    auto it = rtt_.find(rtt_key(a, b));
    if (it == rtt_.end()) return TopologyTier::Global;
    float r = it->second;
    if (r <= thr_.rack_rtt_ms)   return TopologyTier::Rack;
    if (r <= thr_.zone_rtt_ms)   return TopologyTier::Zone;
    if (r <= thr_.region_rtt_ms) return TopologyTier::Region;
    return TopologyTier::Global;
}

std::vector<std::vector<std::string>>
TopologyRegistry::partition_by_tier(TopologyTier tier) const {
    // Called under mtx_ held by caller.
    std::unordered_map<std::string, std::vector<std::string>> buckets;
    for (const auto& [id, loc] : nodes_) {
        std::string key;
        switch (tier) {
            case TopologyTier::Rack:
                key = loc.region + "/" + loc.zone + "/" + loc.rack;
                if (loc.rack.empty()) key.clear();
                break;
            case TopologyTier::Zone:
                key = loc.region + "/" + loc.zone;
                if (loc.zone.empty()) key.clear();
                break;
            case TopologyTier::Region:
                key = loc.region;
                if (loc.region.empty()) key.clear();
                break;
            case TopologyTier::Global:
                key = "global";
                break;
        }
        if (!key.empty()) buckets[key].push_back(id);
    }
    std::vector<std::vector<std::string>> out;
    out.reserve(buckets.size());
    for (auto& [_, v] : buckets) {
        if (v.size() >= 2) out.push_back(std::move(v));
    }
    return out;
}

std::vector<std::string>
TopologyRegistry::order_ring(const std::vector<std::string>& group) const {
    // Called under mtx_ held by caller.
    if (group.size() <= 2) return group;

    std::vector<std::string> ring;
    ring.reserve(group.size());
    std::vector<bool> used(group.size(), false);

    ring.push_back(group[0]);
    used[0] = true;

    for (size_t i = 1; i < group.size(); ++i) {
        const std::string& last = ring.back();
        int   best_idx = -1;
        float best_rtt = std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < group.size(); ++j) {
            if (used[j]) continue;
            auto it = rtt_.find(rtt_key(last, group[j]));
            float r = it == rtt_.end()
                        ? std::numeric_limits<float>::infinity()
                        : it->second;
            if (r < best_rtt) { best_rtt = r; best_idx = (int)j; }
        }
        if (best_idx < 0) {
            // No RTT data — take any remaining.
            for (size_t j = 0; j < group.size(); ++j)
                if (!used[j]) { best_idx = (int)j; break; }
        }
        used[best_idx] = true;
        ring.push_back(group[best_idx]);
    }
    return ring;
}

std::vector<Ring> TopologyRegistry::build_rings() const {
    std::lock_guard<std::mutex> g(mtx_);
    std::vector<Ring> out;
    for (auto tier : {TopologyTier::Rack, TopologyTier::Zone,
                      TopologyTier::Region, TopologyTier::Global}) {
        auto groups = partition_by_tier(tier);
        for (auto& grp : groups) {
            Ring r;
            r.tier      = tier;
            r.members   = order_ring(grp);
            // Build a stable group_key from the first member's location.
            auto it = nodes_.find(r.members.front());
            if (it != nodes_.end()) {
                const auto& L = it->second;
                switch (tier) {
                    case TopologyTier::Rack:
                        r.group_key = L.region + "/" + L.zone + "/" + L.rack;
                        break;
                    case TopologyTier::Zone:
                        r.group_key = L.region + "/" + L.zone;
                        break;
                    case TopologyTier::Region:
                        r.group_key = L.region;
                        break;
                    case TopologyTier::Global:
                        r.group_key = "global";
                        break;
                }
            }
            out.push_back(std::move(r));
        }
    }
    return out;
}

std::string TopologyRegistry::to_json() const {
    auto rings = build_rings();
    std::ostringstream os;
    os << "{\"rings\":[";
    for (size_t i = 0; i < rings.size(); ++i) {
        if (i) os << ',';
        const auto& r = rings[i];
        os << "{\"tier\":" << (int)r.tier
           << ",\"group\":\"" << r.group_key << "\""
           << ",\"members\":[";
        for (size_t j = 0; j < r.members.size(); ++j) {
            if (j) os << ',';
            os << "\"" << r.members[j] << "\"";
        }
        os << "]}";
    }
    os << "],\"nodes\":[";
    {
        std::lock_guard<std::mutex> g(mtx_);
        size_t k = 0;
        for (const auto& [id, L] : nodes_) {
            if (k++) os << ',';
            os << "{\"id\":\"" << id << "\""
               << ",\"region\":\"" << L.region << "\""
               << ",\"zone\":\""   << L.zone   << "\""
               << ",\"rack\":\""   << L.rack   << "\""
               << ",\"nat\":"      << (L.behind_nat?"true":"false")
               << "}";
        }
    }
    os << "]}";
    return os.str();
}

// ─── make_topology_hello ────────────────────────────────────────────────────

static void copy_env(char* dst, size_t cap, const char* var) {
    const char* v = std::getenv(var);
    if (!v) { dst[0] = '\0'; return; }
    std::strncpy(dst, v, cap - 1);
    dst[cap - 1] = '\0';
}

MsgTopologyHello make_topology_hello(const std::string& node_id) {
    MsgTopologyHello h{};
    std::strncpy(h.node_id, node_id.c_str(), MAX_NODE_ID_LEN - 1);

    copy_env(h.region, MAX_REGION_LEN, "DIST_REGION");
    copy_env(h.zone,   MAX_ZONE_LEN,   "DIST_ZONE");
    copy_env(h.rack,   MAX_RACK_LEN,   "DIST_RACK");

    const char* lat = std::getenv("DIST_LAT");
    const char* lon = std::getenv("DIST_LON");
    h.lat_deg = lat ? std::strtof(lat, nullptr) : 0.f;
    h.lon_deg = lon ? std::strtof(lon, nullptr) : 0.f;

    const char* bw = std::getenv("DIST_BANDWIDTH_MBPS");
    h.bandwidth_mbps_self = bw ? (uint32_t)std::strtoul(bw, nullptr, 10) : 0;

    const char* nat = std::getenv("DIST_BEHIND_NAT");
    h.behind_nat = (nat && (*nat == '1' || *nat == 't' || *nat == 'T')) ? 1 : 0;

    return h;
}

} // namespace dist
