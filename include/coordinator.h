#pragma once
/**
 * coordinator.h
 *
 * The Coordinator is the single central service in the cluster.
 * It is NOT a performance-critical path — it only handles cluster management.
 * All tensor data flows directly between nodes.
 *
 * Responsibilities:
 *  1. Accept node registrations and track their capabilities.
 *  2. When a model is requested (or pre-configured), compute a layer
 *     partitioning plan and push LAYER_ASSIGN to each node.
 *  3. Watch heartbeats; mark nodes dead after DEAD_NODE_MS silence.
 *  4. Accept inference requests from the API plane and route them to
 *     the first pipeline stage.
 *  5. Return generated tokens to the requesting client.
 *
 * Threading:
 *   accept_thread_    : accepts new control connections
 *   per-node thread   : one thread per connected node (recv heartbeats etc.)
 *   api_accept_thread_: accepts client API connections
 *   per-client thread : one thread per API client
 *   monitor_thread_   : periodic health checks, dead-node detection
 */

#include "dist_protocol.h"
#include "dist_conn.h"
#include "dist_queue.h"

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace dist {

struct NodeInfo {
    std::string              id;
    NodeCapability           cap;
    uint16_t                 data_port;
    std::string              addr;        // IP resolved from accept()

    // Layer assignment
    std::vector<LayerRange>  assigned_ranges;
    bool                     model_loaded = false;

    // Liveness
    std::chrono::steady_clock::time_point last_heartbeat;
    bool                     alive = true;

    std::unique_ptr<Connection> conn; // coordinator -> node control channel
};

struct ModelPlan {
    std::string              model_name;
    std::string              model_path;
    uint32_t                 n_layers;
    uint32_t                 n_ctx;
    // node_id -> layer ranges assigned
    std::map<std::string, std::vector<LayerRange>> assignment;
    // ordered pipeline: node_ids front-to-back
    std::vector<std::string> pipeline_order;
    bool                     ready = false;  // true when all nodes ACKed
};

// Pending inference request tracked by coordinator
struct InferState {
    uint64_t                 request_id;
    std::shared_ptr<Connection> client_conn;
    uint32_t                 tokens_generated = 0;
    uint32_t                 max_gen_tokens   = 256;
    bool                     done             = false;
};

struct CoordinatorConfig {
    std::string bind_host       = "0.0.0.0";
    uint16_t    control_port    = PORT_CONTROL;
    uint16_t    api_port        = PORT_API;

    // If non-empty, auto-assign this model on startup once enough nodes join.
    std::string auto_model_path;
    std::string auto_model_name;
    uint32_t    auto_n_ctx      = 4096;
    uint32_t    min_nodes       = 1;       // wait for this many nodes before auto-assign
};

class Coordinator {
public:
    explicit Coordinator(CoordinatorConfig cfg);
    ~Coordinator();

    void run();
    void stop();

private:
    // ── Threads ──────────────────────────────────────────────────────────────
    void accept_thread_fn();
    void node_thread_fn(std::string node_id);
    void api_accept_thread_fn();
    void client_thread_fn(std::shared_ptr<Connection> client_conn);
    void monitor_thread_fn();

    // ── Node management ──────────────────────────────────────────────────────
    void handle_node_join(const std::string& addr,
                          std::unique_ptr<Connection> conn,
                          const MsgNodeJoin& msg);
    void handle_heartbeat(const std::string& node_id, const MsgHeartbeat& msg);
    void handle_assign_ack(const std::string& node_id, const MsgAck& msg);
    void handle_load_ack(const std::string& node_id, const MsgAck& msg);
    void remove_dead_node(const std::string& node_id);

    // ── Planning ─────────────────────────────────────────────────────────────
    // Partition model layers across available nodes proportional to their VRAM.
    ModelPlan make_plan(const std::string& model_path,
                        const std::string& model_name,
                        uint32_t n_layers,
                        uint32_t n_ctx);

    void push_plan(const ModelPlan& plan);

    // Tell each node the address of its downstream neighbour.
    void wire_pipeline(const ModelPlan& plan);

    // ── Inference routing ────────────────────────────────────────────────────
    void route_infer_request(std::shared_ptr<Connection> client,
                              const MsgInferRequest& req,
                              const std::vector<int32_t>& token_ids);
    void on_token_generated(uint64_t req_id, int32_t token_id,
                             uint32_t pos, bool is_last);

    // ── Helpers ──────────────────────────────────────────────────────────────
    NodeInfo* find_node(const std::string& id);  // call with nodes_mu_ held

    // Optional hook called when a node is removed (used by VmCoordinator).
    std::function<void(const std::string&)> vm_hook_;
    friend class VmCoordinator;

    CoordinatorConfig cfg_;
    std::atomic<bool> running_ { false };

    // Node registry
    mutable std::mutex                           nodes_mu_;
    std::unordered_map<std::string, NodeInfo>    nodes_;

    // Active model plan
    std::mutex   plan_mu_;
    ModelPlan    active_plan_;
    bool         plan_active_ = false;

    // Pending infer requests
    std::mutex   infer_mu_;
    std::unordered_map<uint64_t, InferState> pending_infers_;
    std::atomic<uint64_t> next_request_id_ { 1 };

    // Listeners
    Listener control_listener_;
    Listener api_listener_;

    // Threads
    std::thread             accept_thread_;
    std::thread             api_accept_thread_;
    std::thread             monitor_thread_;
    std::vector<std::thread> node_threads_;
};

} // namespace dist
