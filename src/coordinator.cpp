/**
 * coordinator.cpp
 */

#include "coordinator.h"
#include "node_agent.h"  // for ActivationBatch

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <sys/stat.h>

namespace dist {

using Clock = std::chrono::steady_clock;

// ─── Construction / destruction ─────────────────────────────────────────────

Coordinator::Coordinator(CoordinatorConfig cfg)
    : cfg_(std::move(cfg))
{}

Coordinator::~Coordinator() {
    stop();
}

// ─── Public API ─────────────────────────────────────────────────────────────

void Coordinator::run() {
    running_.store(true);

    // Load tokens if an auth file was configured.
    if (!cfg_.token_file.empty()) {
        if (!tokens_.load(cfg_.token_file)) {
            std::cerr << "[Coordinator] WARNING: could not read token file "
                      << cfg_.token_file << " — starting with empty token store\n";
        }
        auth_required_ = tokens_.size() > 0;
        std::cout << "[Coordinator] auth: "
                  << (auth_required_ ? "ENABLED" : "disabled")
                  << " (" << tokens_.size() << " tokens loaded)\n";
    }
    if (cfg_.issue_receipts) ensure_issuer_secret();

    control_listener_.bind_and_listen(cfg_.control_port);
    api_listener_.bind_and_listen(cfg_.api_port);

    std::cout << "[Coordinator] control port " << cfg_.control_port
              << ", API port " << cfg_.api_port << "\n";

    // Start dashboard HTTP server
    if (cfg_.dashboard_port > 0) {
        DashboardConfig dcfg;
        dcfg.bind_host    = cfg_.bind_host;
        dcfg.http_port    = cfg_.dashboard_port;
        dcfg.public_host  = cfg_.public_host.empty() ? cfg_.bind_host : cfg_.public_host;
        dcfg.ctrl_port    = cfg_.control_port;
        dcfg.vm_mode      = cfg_.vm_mode;
        dashboard_ = std::make_unique<DashboardServer>(std::move(dcfg), monitor_);
        dashboard_->start();
    }

    accept_thread_     = std::thread([this]{ accept_thread_fn();     });
    api_accept_thread_ = std::thread([this]{ api_accept_thread_fn(); });
    monitor_thread_    = std::thread([this]{ monitor_thread_fn();    });

    accept_thread_.join();
    api_accept_thread_.join();
    monitor_thread_.join();

    for (auto& t : node_threads_) if (t.joinable()) t.join();
}

void Coordinator::stop() {
    running_.store(false);
    if (dashboard_) { dashboard_->stop(); dashboard_.reset(); }
    control_listener_.stop();
    api_listener_.stop();
    {
        std::lock_guard<std::mutex> lk(nodes_mu_);
        for (auto& [id, node] : nodes_) {
            if (node.conn) node.conn->close();
        }
    }
}

// ─── Thread: accept node control connections ─────────────────────────────────

void Coordinator::accept_thread_fn() {
    while (running_.load()) {
        auto conn = control_listener_.accept_one();
        if (!conn) continue;

        // Gate on auth first (no-op if auth not required).
        std::string deny_reason;
        if (!authenticate_incoming(*conn, &deny_reason)) {
            std::cerr << "[Coordinator] auth rejected: " << deny_reason << "\n";
            conn->close();
            continue;
        }

        // Peek the first message to get the NODE_JOIN
        MsgHeader hdr{};
        std::vector<uint8_t> payload;
        if (!conn->recv_msg(hdr, payload)) continue;
        if (static_cast<MsgType>(hdr.msg_type) != MsgType::NODE_JOIN) continue;
        if (payload.size() < sizeof(MsgNodeJoin)) continue;

        const auto& join_msg = *reinterpret_cast<const MsgNodeJoin*>(payload.data());

        // Extract peer address from the connection (best-effort; use node_id if blank)
        std::string addr = join_msg.cap.node_id; // use node_id as address key

        handle_node_join(addr, std::move(conn),
                         *reinterpret_cast<const MsgNodeJoin*>(payload.data()));
    }
}

// ─── Thread: per-node message loop ──────────────────────────────────────────

void Coordinator::node_thread_fn(std::string node_id) {
    NodeInfo* node = nullptr;
    {
        std::lock_guard<std::mutex> lk(nodes_mu_);
        node = find_node(node_id);
    }
    if (!node) return;

    MsgHeader            hdr{};
    std::vector<uint8_t> payload;

    while (running_.load()) {
        {
            std::lock_guard<std::mutex> lk(nodes_mu_);
            node = find_node(node_id);
            if (!node || !node->alive || !node->conn) break;
        }

        bool ok;
        {
            // We need to recv outside the lock (blocking call).
            // Read conn ptr safely.
            Connection* raw = nullptr;
            {
                std::lock_guard<std::mutex> lk(nodes_mu_);
                node = find_node(node_id);
                if (!node) break;
                raw = node->conn.get();
            }
            ok = raw && raw->recv_msg(hdr, payload);
        }

        if (!ok) {
            std::cerr << "[Coordinator] lost connection to " << node_id << "\n";
            remove_dead_node(node_id);
            return;
        }

        switch (static_cast<MsgType>(hdr.msg_type)) {
        case MsgType::HEARTBEAT:
            if (payload.size() >= sizeof(MsgHeartbeat))
                handle_heartbeat(node_id,
                    *reinterpret_cast<const MsgHeartbeat*>(payload.data()));
            break;
        case MsgType::ASSIGN_ACK:
            if (payload.size() >= sizeof(MsgAck))
                handle_assign_ack(node_id,
                    *reinterpret_cast<const MsgAck*>(payload.data()));
            break;
        case MsgType::LOAD_ACK:
            if (payload.size() >= sizeof(MsgAck))
                handle_load_ack(node_id,
                    *reinterpret_cast<const MsgAck*>(payload.data()));
            break;
        case MsgType::NODE_LEAVE:
            remove_dead_node(node_id);
            return;
        case MsgType::TOPOLOGY_HELLO:
            if (payload.size() >= sizeof(MsgTopologyHello))
                handle_topology_hello(
                    *reinterpret_cast<const MsgTopologyHello*>(payload.data()));
            break;
        case MsgType::TOPOLOGY_LATENCY:
            if (payload.size() >= sizeof(MsgTopologyLatency))
                handle_topology_latency(
                    *reinterpret_cast<const MsgTopologyLatency*>(payload.data()));
            break;
        default:
            break;
        }
    }
}

// ─── Thread: accept API client connections ──────────────────────────────────

void Coordinator::api_accept_thread_fn() {
    while (running_.load()) {
        auto conn = api_listener_.accept_one();
        if (!conn) continue;
        std::shared_ptr<Connection> shared(conn.release());
        std::thread([this, c = std::move(shared)]() mutable {
            client_thread_fn(std::move(c));
        }).detach();
    }
}

// ─── Thread: per-client inference handler ───────────────────────────────────

void Coordinator::client_thread_fn(std::shared_ptr<Connection> client_conn) {
    MsgHeader            hdr{};
    std::vector<uint8_t> payload;

    while (running_.load() && client_conn->is_connected()) {
        if (!client_conn->recv_msg(hdr, payload)) break;

        if (static_cast<MsgType>(hdr.msg_type) == MsgType::INFER_REQUEST) {
            if (payload.size() < sizeof(MsgInferRequest)) continue;
            const auto& req = *reinterpret_cast<const MsgInferRequest*>(payload.data());

            size_t tokens_offset = sizeof(MsgInferRequest);
            size_t n_token_bytes = req.n_prompt_tokens * sizeof(int32_t);
            if (payload.size() < tokens_offset + n_token_bytes) continue;

            std::vector<int32_t> token_ids(req.n_prompt_tokens);
            memcpy(token_ids.data(), payload.data() + tokens_offset, n_token_bytes);

            route_infer_request(client_conn, req, token_ids);
        }
    }
}

// ─── Thread: health monitor ─────────────────────────────────────────────────

void Coordinator::monitor_thread_fn() {
    while (running_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        auto now = Clock::now();
        std::vector<std::string> dead;

        {
            std::lock_guard<std::mutex> lk(nodes_mu_);
            for (auto& [id, node] : nodes_) {
                if (!node.alive) continue;
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - node.last_heartbeat).count();
                if (elapsed > DEAD_NODE_MS) {
                    std::cerr << "[Coordinator] node " << id
                              << " missed heartbeat (" << elapsed << "ms), marking dead\n";
                    dead.push_back(id);
                }
            }
        }

        for (auto& id : dead) remove_dead_node(id);

        // Auto-assign model if configured and enough nodes are ready
        if (!cfg_.auto_model_path.empty()) {
            std::lock_guard<std::mutex> plk(plan_mu_);
            if (!plan_active_) {
                std::lock_guard<std::mutex> nlk(nodes_mu_);
                if (nodes_.size() >= cfg_.min_nodes) {
                    // Count total VRAM to estimate n_layers needed
                    // (we use a fixed n_layers from config for now)
                    uint32_t n_layers = 32; // TODO: read from model file
                    active_plan_ = make_plan(cfg_.auto_model_path,
                                             cfg_.auto_model_name,
                                             n_layers,
                                             cfg_.auto_n_ctx);
                    plan_active_ = true;
                    // Release nlk before pushing (push locks nodes_mu_ too)
                }
            }
        }

        // Push plan outside the lock
        {
            std::lock_guard<std::mutex> plk(plan_mu_);
            if (plan_active_ && !active_plan_.ready) {
                // Check if all nodes have loaded
                bool all_loaded = true;
                {
                    std::lock_guard<std::mutex> nlk(nodes_mu_);
                    for (auto& nid : active_plan_.pipeline_order) {
                        NodeInfo* n = find_node(nid);
                        if (!n || !n->model_loaded) { all_loaded = false; break; }
                    }
                }
                if (all_loaded) {
                    active_plan_.ready = true;
                    wire_pipeline(active_plan_);
                    std::cout << "[Coordinator] pipeline ready!\n";
                }
            }
        }
    }
}

// ─── Node management ─────────────────────────────────────────────────────────

void Coordinator::handle_node_join(const std::string& addr,
                                   std::unique_ptr<Connection> conn,
                                   const MsgNodeJoin& msg) {
    std::string id = msg.cap.node_id;
    std::cout << "[Coordinator] node joined: " << id
              << " gpus=" << msg.cap.n_gpus
              << " data_port=" << msg.data_port << "\n";

    {
        std::lock_guard<std::mutex> lk(nodes_mu_);
        NodeInfo& node          = nodes_[id];
        node.id                 = id;
        node.cap                = msg.cap;
        node.data_port          = msg.data_port;
        node.addr               = addr;
        node.last_heartbeat     = Clock::now();
        node.alive              = true;
        node.model_loaded       = false;
        node.conn               = std::move(conn);
    }

    monitor_.on_node_join(id, addr, msg.cap, msg.data_port);

    node_threads_.emplace_back([this, id]{ node_thread_fn(id); });

    // If we have a pending plan to push, send assignment to this new node
    {
        std::lock_guard<std::mutex> plk(plan_mu_);
        if (plan_active_) {
            push_plan(active_plan_);
        }
    }
}

void Coordinator::handle_heartbeat(const std::string& node_id,
                                   const MsgHeartbeat& msg) {
    {
        std::lock_guard<std::mutex> lk(nodes_mu_);
        NodeInfo* n = find_node(node_id);
        if (!n) return;
        n->last_heartbeat = Clock::now();
        for (uint32_t i = 0; i < n->cap.n_gpus && i < 8; ++i)
            n->cap.gpu_free_bytes[i] = msg.gpu_free_bytes[i];
    }
    monitor_.on_heartbeat(node_id, msg);
}

void Coordinator::handle_assign_ack(const std::string& node_id,
                                    const MsgAck& msg) {
    std::cout << "[Coordinator] node " << node_id
              << " ASSIGN_ACK: " << (msg.success ? "OK" : "FAIL")
              << (msg.success ? "" : std::string(" ") + msg.message) << "\n";
    if (msg.success) {
        // Model is loading; model_loaded will be set via LOAD_ACK
    }
}

void Coordinator::handle_load_ack(const std::string& node_id,
                                  const MsgAck& msg) {
    std::cout << "[Coordinator] node " << node_id
              << " LOAD_ACK: " << (msg.success ? "OK" : "FAIL") << "\n";
    if (msg.success) {
        {
            std::lock_guard<std::mutex> lk(nodes_mu_);
            NodeInfo* n = find_node(node_id);
            if (n) n->model_loaded = true;
        }
        std::string mname, mpath;
        uint32_t n_layers = 0;
        {
            std::lock_guard<std::mutex> plk(plan_mu_);
            mname   = active_plan_.model_name;
            n_layers = active_plan_.n_layers;
        }
        monitor_.on_model_loaded(node_id, mname, n_layers);
    }
}

void Coordinator::remove_dead_node(const std::string& node_id) {
    {
        std::lock_guard<std::mutex> lk(nodes_mu_);
        NodeInfo* n = find_node(node_id);
        if (!n) return;
        n->alive = false;
        if (n->conn) n->conn->close();
    }
    std::cout << "[Coordinator] removed dead node: " << node_id << "\n";
    monitor_.on_node_left(node_id);
    topology_.remove(node_id);
    if (vm_hook_) vm_hook_(node_id);  // notify VmCoordinator
}

// ─── Planning ────────────────────────────────────────────────────────────────

ModelPlan Coordinator::make_plan(const std::string& model_path,
                                  const std::string& model_name,
                                  uint32_t n_layers,
                                  uint32_t n_ctx) {
    // Called with nodes_mu_ held (or only from monitor which holds it).
    // Partition layers proportional to total GPU VRAM on each node.

    ModelPlan plan;
    plan.model_path = model_path;
    plan.model_name = model_name;
    plan.n_layers   = n_layers;
    plan.n_ctx      = n_ctx;

    // Collect live nodes and their total VRAM
    struct NodeVRAM { std::string id; uint64_t vram; };
    std::vector<NodeVRAM> candidates;
    for (auto& [id, node] : nodes_) {
        if (!node.alive) continue;
        uint64_t total = 0;
        for (uint32_t i = 0; i < node.cap.n_gpus; ++i)
            total += node.cap.gpu_vram_bytes[i];
        if (total == 0) total = 1; // CPU-only node gets minimum share
        candidates.push_back({id, total});
    }

    if (candidates.empty()) return plan;

    uint64_t total_vram = 0;
    for (auto& c : candidates) total_vram += c.vram;

    // Sort by VRAM descending so biggest node gets first layers (embedding is heavy)
    std::sort(candidates.begin(), candidates.end(),
              [](const NodeVRAM& a, const NodeVRAM& b){ return a.vram > b.vram; });

    // Proportional layer assignment
    uint32_t layer_cursor = 0;
    for (size_t i = 0; i < candidates.size(); ++i) {
        uint32_t share;
        if (i == candidates.size() - 1) {
            share = n_layers - layer_cursor; // give remainder to last node
        } else {
            share = (uint32_t)((double)candidates[i].vram / total_vram * n_layers);
            share = std::max(share, 1u);
        }

        LayerRange range;
        range.layer_first = layer_cursor;
        range.layer_last  = layer_cursor + share - 1;
        range.gpu_index   = 0;

        plan.assignment[candidates[i].id] = { range };
        plan.pipeline_order.push_back(candidates[i].id);
        layer_cursor += share;

        if (layer_cursor >= n_layers) break;
    }

    std::cout << "[Coordinator] plan for model=" << model_name
              << " n_layers=" << n_layers << ":\n";
    for (auto& nid : plan.pipeline_order) {
        for (auto& r : plan.assignment[nid]) {
            std::cout << "  " << nid << " -> layers "
                      << r.layer_first << "-" << r.layer_last << "\n";
        }
    }

    return plan;
}

void Coordinator::push_plan(const ModelPlan& plan) {
    // Send LAYER_ASSIGN to each node in the plan
    std::lock_guard<std::mutex> lk(nodes_mu_);
    for (auto& nid : plan.pipeline_order) {
        NodeInfo* node = find_node(nid);
        if (!node || !node->alive || !node->conn) continue;

        const auto& ranges = plan.assignment.at(nid);

        // Build payload: MsgLayerAssign + ranges
        std::vector<uint8_t> payload(sizeof(MsgLayerAssign)
                                     + ranges.size() * sizeof(LayerRange));
        auto& msg = *reinterpret_cast<MsgLayerAssign*>(payload.data());
        strncpy(msg.node_id, nid.c_str(), sizeof(msg.node_id) - 1);
        strncpy(msg.model_path, plan.model_path.c_str(), sizeof(msg.model_path) - 1);
        strncpy(msg.model_name, plan.model_name.c_str(), sizeof(msg.model_name) - 1);
        msg.n_layer_total = plan.n_layers;
        msg.n_ranges      = (uint32_t)ranges.size();
        msg.n_ctx         = plan.n_ctx;

        memcpy(payload.data() + sizeof(MsgLayerAssign),
               ranges.data(),
               ranges.size() * sizeof(LayerRange));

        node->conn->send_msg(MsgType::LAYER_ASSIGN,
                             payload.data(), (uint32_t)payload.size());

        // Update monitor with layer assignment
        if (!ranges.empty()) {
            monitor_.on_layer_assign(nid, ranges.front().layer_first,
                                          ranges.back().layer_last);
        }
    }
}

void Coordinator::wire_pipeline(const ModelPlan& plan) {
    // After all nodes have loaded, tell each node the address of its
    // downstream neighbour. We encode this as a small custom control message.
    // For simplicity, we send LAYER_ASSIGN again with next-node info appended
    // as a text "next_host:port" — in a real system this would be a dedicated msg.
    // Here we just print the wiring and the nodes derive it from the join order.
    // TODO: define a WIRE_PIPELINE message and send next-node address to each node.
    std::cout << "[Coordinator] pipeline wired (nodes communicate directly on data port)\n";
    // In full implementation: send MsgPipelineWire to each node containing
    // the host:port of the next node so data_send_thread can connect.
}

// ─── Inference routing ───────────────────────────────────────────────────────

void Coordinator::route_infer_request(std::shared_ptr<Connection> client,
                                       const MsgInferRequest& req,
                                       const std::vector<int32_t>& token_ids) {
    std::lock_guard<std::mutex> plk(plan_mu_);
    if (!plan_active_ || !active_plan_.ready || active_plan_.pipeline_order.empty()) {
        MsgInferError err{};
        err.request_id = req.request_id;
        strncpy(err.message, "no model loaded", sizeof(err.message) - 1);
        client->send_msg(MsgType::INFER_ERROR, &err, sizeof(err));
        return;
    }

    uint64_t req_id = next_request_id_.fetch_add(1);

    // Register pending infer state
    {
        std::lock_guard<std::mutex> ilk(infer_mu_);
        InferState& s   = pending_infers_[req_id];
        s.request_id    = req_id;
        s.client_conn   = client;
        s.max_gen_tokens = req.max_gen_tokens;
    }

    // Build activation batch for the first stage
    ActivationBatch batch;
    batch.header.request_id = req_id;
    batch.header.from_layer = 0;
    batch.header.seq_pos    = 0;
    batch.header.n_tokens   = (uint32_t)token_ids.size();
    batch.header.n_embd     = 0; // first stage uses token ids, not embeddings
    batch.header.dtype      = 0;
    batch.header.is_last    = 0;

    // Payload: raw int32_t token ids
    batch.data.resize(token_ids.size() * sizeof(int32_t));
    memcpy(batch.data.data(), token_ids.data(), batch.data.size());

    // Send to first node's data port
    const std::string& first_node_id = active_plan_.pipeline_order.front();
    NodeInfo* first_node;
    {
        std::lock_guard<std::mutex> nlk(nodes_mu_);
        first_node = find_node(first_node_id);
        if (!first_node || !first_node->alive) {
            MsgInferError err{};
            err.request_id = req_id;
            strncpy(err.message, "first pipeline node unavailable", sizeof(err.message) - 1);
            client->send_msg(MsgType::INFER_ERROR, &err, sizeof(err));
            return;
        }

        // Connect to first node's data port and send the activation
        // (in a full implementation this connection is kept persistent)
        Connection data_conn;
        data_conn.connect(first_node->addr, first_node->data_port);
        data_conn.send_msg2(
            MsgType::TENSOR_FORWARD,
            &batch.header, sizeof(TensorHeader),
            batch.data.data(), (uint32_t)batch.data.size()
        );
        // data_conn closes at end of scope; node should handle reconnect gracefully
    }
}

void Coordinator::on_token_generated(uint64_t req_id, int32_t token_id,
                                      uint32_t pos, bool is_last) {
    std::lock_guard<std::mutex> lk(infer_mu_);
    auto it = pending_infers_.find(req_id);
    if (it == pending_infers_.end()) return;

    InferState& s = it->second;

    MsgInferToken tok{};
    tok.request_id = req_id;
    tok.token_id   = (uint32_t)token_id;
    tok.token_pos  = pos;
    tok.logprob    = 0.0f;
    s.client_conn->send_msg(MsgType::INFER_TOKEN, &tok, sizeof(tok));
    s.tokens_generated++;
    monitor_.on_token_generated("", 1); // cluster-level accounting

    if (is_last || s.tokens_generated >= s.max_gen_tokens) {
        MsgInferDone done{};
        done.request_id        = req_id;
        done.n_tokens_generated = s.tokens_generated;
        s.client_conn->send_msg(MsgType::INFER_DONE, &done, sizeof(done));
        pending_infers_.erase(it);
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

NodeInfo* Coordinator::find_node(const std::string& id) {
    auto it = nodes_.find(id);
    return (it != nodes_.end()) ? &it->second : nullptr;
}

// ─── Phase-5: auth + topology ───────────────────────────────────────────────

bool Coordinator::authenticate_incoming(Connection& conn,
                                         std::string* out_reason) {
    if (!auth_required_) return true;

    // 1. Generate a fresh nonce and send the challenge.
    MsgAuthChallenge challenge{};
    random_bytes(challenge.nonce, sizeof(challenge.nonce));
    challenge.issued_at_us =
        (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

    try {
        conn.send_msg(MsgType::AUTH_CHALLENGE, &challenge, sizeof(challenge));
    } catch (...) {
        if (out_reason) *out_reason = "failed to send challenge";
        return false;
    }

    // 2. Receive the response with a short timeout.
    MsgHeader hdr{};
    std::vector<uint8_t> payload;
    if (!conn.recv_msg(hdr, payload)
        || static_cast<MsgType>(hdr.msg_type) != MsgType::AUTH_RESPONSE
        || payload.size() < sizeof(MsgAuthResponse)) {
        if (out_reason) *out_reason = "no AUTH_RESPONSE received";
        return false;
    }
    const auto& resp = *reinterpret_cast<const MsgAuthResponse*>(payload.data());

    uint64_t now_us =
        (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

    uint32_t granted = 0;
    std::string reason;
    bool ok = verify_auth_response(challenge.nonce, resp, tokens_,
                                    now_us, &granted, &reason);

    MsgAuthResult result{};
    result.accepted      = ok ? 1 : 0;
    result.scope_granted = granted;
    result.expires_at_us = now_us + 3600ull * 1000000ull; // 1h session
    std::strncpy(result.reason, reason.c_str(), sizeof(result.reason) - 1);
    conn.send_msg(MsgType::AUTH_RESULT, &result, sizeof(result));

    if (!ok) { if (out_reason) *out_reason = reason; return false; }

    // Require SCOPE_JOIN for a node connection.
    if ((granted & SCOPE_JOIN) == 0) {
        if (out_reason) *out_reason = "token lacks SCOPE_JOIN";
        return false;
    }
    return true;
}

void Coordinator::handle_topology_hello(const MsgTopologyHello& msg) {
    NodeLocation loc;
    loc.node_id        = std::string(msg.node_id,
                                     strnlen(msg.node_id, MAX_NODE_ID_LEN));
    loc.region         = std::string(msg.region,
                                     strnlen(msg.region, MAX_REGION_LEN));
    loc.zone           = std::string(msg.zone,
                                     strnlen(msg.zone, MAX_ZONE_LEN));
    loc.rack           = std::string(msg.rack,
                                     strnlen(msg.rack, MAX_RACK_LEN));
    loc.lat_deg        = msg.lat_deg;
    loc.lon_deg        = msg.lon_deg;
    loc.bandwidth_mbps = msg.bandwidth_mbps_self;
    loc.behind_nat     = msg.behind_nat != 0;
    topology_.upsert(loc);
}

void Coordinator::handle_topology_latency(const MsgTopologyLatency& msg) {
    std::string a(msg.src_node, strnlen(msg.src_node, MAX_NODE_ID_LEN));
    std::string b(msg.dst_node, strnlen(msg.dst_node, MAX_NODE_ID_LEN));
    topology_.record_latency(a, b, msg.rtt_ms);
}

void Coordinator::ensure_issuer_secret() {
    if (has_issuer_secret_) return;
    std::string path = cfg_.token_file.empty()
                        ? std::string("coordinator.issuer")
                        : cfg_.token_file + ".issuer";

    // Try to load existing key.
    {
        std::ifstream f(path);
        std::string line;
        if (f && std::getline(f, line)) {
            auto raw = from_hex(line);
            if (raw.size() == 32) {
                std::memcpy(issuer_secret_.data(), raw.data(), 32);
                has_issuer_secret_ = true;
                return;
            }
        }
    }

    // Generate new.
    random_bytes(issuer_secret_.data(), issuer_secret_.size());
    std::ofstream f(path);
    if (f) {
        f << to_hex(issuer_secret_.data(), issuer_secret_.size()) << "\n";
        f.close();
        ::chmod(path.c_str(), 0600);
    }
    has_issuer_secret_ = true;
    std::cout << "[Coordinator] generated issuer secret at " << path << "\n";
}

MsgContribReceipt Coordinator::make_receipt(const NodeInfo& n,
                                             uint64_t window_start_us,
                                             uint64_t window_end_us,
                                             uint64_t tokens,
                                             uint64_t bytes) {
    MsgContribReceipt r{};
    std::strncpy(r.node_id, n.id.c_str(), MAX_NODE_ID_LEN - 1);
    // Tenant unknown at this layer (Pass A) — leave blank.
    r.window_start_us       = window_start_us;
    r.window_end_us         = window_end_us;
    r.tokens_processed      = tokens;
    r.layer_bytes_forwarded = bytes;
    // layer_seconds = assigned_layers * window_seconds — rough work proxy.
    uint64_t layers_assigned = 0;
    for (auto& rng : n.assigned_ranges)
        layers_assigned += (rng.layer_last - rng.layer_first + 1);
    r.layer_seconds = layers_assigned *
                      ((window_end_us - window_start_us) / 1000000ull);
    std::strncpy(r.issuer_id, cfg_.public_host.c_str(), MAX_TOKEN_ID_LEN - 1);

    if (has_issuer_secret_) sign_receipt(r, issuer_secret_);
    return r;
}

} // namespace dist
