#pragma once
/**
 * node_agent.h
 *
 * The NodeAgent runs on each GPU server.
 *
 * Responsibilities:
 *  1. Connect to the Coordinator and register capacity.
 *  2. Receive a layer-range assignment and load those weights.
 *  3. Listen on the data port for incoming activation tensors.
 *  4. Execute assigned layers on the local GPU.
 *  5. Forward the output activation tensor to the next node (or return
 *     logits if this is the last stage).
 *  6. Send periodic heartbeats to the Coordinator.
 *
 * Threading model:
 *   control_thread_  : recv messages from coordinator, send heartbeats
 *   data_recv_thread_: accept+recv TENSOR_FORWARD from previous node
 *   compute_thread_  : pop from recv_queue_, run layers, push to send_queue_
 *   data_send_thread_: pop from send_queue_, send TENSOR_FORWARD to next node
 */

#include "dist_protocol.h"
#include "dist_conn.h"
#include "dist_queue.h"

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

// Forward declarations for llama.cpp types
struct llama_model;
struct llama_context;

namespace dist {

// A pending activation batch moving through the pipeline
struct ActivationBatch {
    TensorHeader          header;
    std::vector<uint8_t>  data;   // raw tensor bytes (n_tokens * n_embd * dtype_size)
};

// Callback invoked by the last node when generation is complete
using TokenCallback = std::function<void(uint64_t req_id, int32_t token_id,
                                         uint32_t pos, bool is_last)>;

struct NodeAgentConfig {
    std::string coordinator_host;
    uint16_t    coordinator_port = PORT_CONTROL;

    std::string bind_host        = "0.0.0.0";
    uint16_t    data_port        = PORT_DATA;

    std::string node_id;           // defaults to hostname:pid if empty

    uint32_t    compute_queue_depth = 4;   // double-buffering depth
    uint32_t    n_cpu_offload_layers = 0;  // layers kept in CPU RAM (0 = GPU-only)

    // llama.cpp context params (set after layer assignment)
    uint32_t    n_ctx      = 4096;
    uint32_t    n_batch    = 512;
    uint32_t    n_ubatch   = 512;
    int32_t     n_gpu_layers = 999; // offload all assigned layers to GPU

    // Phase-5 auth.  If token_id is empty, the node makes no attempt to
    // authenticate; the coordinator will also proceed if auth is disabled.
    std::string token_id;
    std::string token_secret_hex;   // 64 hex chars = 32 bytes
};

class NodeAgent {
public:
    explicit NodeAgent(NodeAgentConfig cfg);
    ~NodeAgent();

    // Start all threads; blocks until stop() is called.
    void run();
    void stop();

    // Inject token callback (used when this node is the last stage).
    void set_token_callback(TokenCallback cb) { token_cb_ = std::move(cb); }

    const std::string& node_id() const { return cfg_.node_id; }

private:
    // ── Threads ──────────────────────────────────────────────────────────────
    void control_thread_fn();
    void data_recv_thread_fn();
    void compute_thread_fn();
    void data_send_thread_fn();
    void heartbeat_loop();

    // ── Control plane handlers ───────────────────────────────────────────────
    void handle_layer_assign(const MsgLayerAssign& msg,
                             const std::vector<LayerRange>& ranges);
    void handle_load_model(const std::string& model_path);
    void handle_shutdown();

    // ── Compute ─────────────────────────────────────────────────────────────
    // Run assigned layers on one activation batch.
    // Returns the output activation (or logits if last stage).
    ActivationBatch run_layers(const ActivationBatch& in);

    // ── Helpers ─────────────────────────────────────────────────────────────
    void send_node_join();
    void send_heartbeat();
    NodeCapability build_capability() const;
    void connect_to_next_node();

    // Respond to an AUTH_CHALLENGE if the coordinator sends one as the first
    // message.  Returns true if auth succeeded or wasn't required.
    bool perform_auth_if_challenged();

    // Send a TOPOLOGY_HELLO based on DIST_* environment variables.
    void send_topology_hello();

    // ── State ────────────────────────────────────────────────────────────────
    NodeAgentConfig cfg_;
    TokenCallback   token_cb_;

    std::atomic<bool> running_     { false };
    std::atomic<bool> model_loaded_{ false };

    // Layer assignment
    uint32_t              layer_first_ = 0;
    uint32_t              layer_last_  = 0;
    uint32_t              n_layer_total_ = 0;
    bool                  is_first_stage_ = false;  // holds embedding
    bool                  is_last_stage_  = false;  // produces logits

    // Next node in the pipeline
    std::string           next_node_host_;
    uint16_t              next_node_port_ = 0;
    std::unique_ptr<Connection> next_conn_;

    // llama.cpp handles (owned)
    llama_model*   lm_  = nullptr;
    llama_context* lctx_ = nullptr;

    // Connections
    std::unique_ptr<Connection> coord_conn_;  // to coordinator
    Listener                    data_listener_;

    // Queues
    BoundedQueue<ActivationBatch> recv_queue_;  // network -> compute
    BoundedQueue<ActivationBatch> send_queue_;  // compute -> network

    // Worker threads
    std::thread control_thread_;
    std::thread data_recv_thread_;
    std::thread compute_thread_;
    std::thread data_send_thread_;
    std::thread heartbeat_thread_;
};

} // namespace dist
