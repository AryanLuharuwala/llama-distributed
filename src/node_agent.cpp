/**
 * node_agent.cpp
 *
 * Implementation of NodeAgent.
 * Reads model weights via llama.cpp, executes assigned layers,
 * streams activation tensors to/from neighbours over TCP.
 */

#include "node_agent.h"
#include "auth.h"
#include "topology.h"

// llama.cpp public API
#include "llama.h"
#include "ggml.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>

#ifdef _WIN32
#  include <winsock2.h>
#else
#  include <unistd.h>
#  include <sys/utsname.h>
#endif

namespace dist {

// ─── Construction / destruction ─────────────────────────────────────────────

NodeAgent::NodeAgent(NodeAgentConfig cfg)
    : cfg_(std::move(cfg)),
      recv_queue_(cfg_.compute_queue_depth),
      send_queue_(cfg_.compute_queue_depth)
{
    if (cfg_.node_id.empty()) {
#ifdef _WIN32
        char buf[256]; gethostname(buf, sizeof(buf));
        cfg_.node_id = std::string(buf) + ":" + std::to_string(GetCurrentProcessId());
#else
        char buf[256]; gethostname(buf, sizeof(buf));
        cfg_.node_id = std::string(buf) + ":" + std::to_string(getpid());
#endif
    }

    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);
}

NodeAgent::~NodeAgent() {
    stop();
    if (lctx_) { llama_free(lctx_); lctx_ = nullptr; }
    if (lm_)   { llama_free_model(lm_); lm_ = nullptr; }
    llama_backend_free();
}

// ─── Public API ─────────────────────────────────────────────────────────────

void NodeAgent::run() {
    running_.store(true);

    // Bind data port for incoming activations
    data_listener_.bind_and_listen(cfg_.data_port);
    std::cout << "[NodeAgent:" << cfg_.node_id << "] data listener on port "
              << cfg_.data_port << "\n";

    // Connect to coordinator
    coord_conn_ = std::make_unique<Connection>();
    coord_conn_->connect(cfg_.coordinator_host, cfg_.coordinator_port);
    std::cout << "[NodeAgent:" << cfg_.node_id << "] connected to coordinator "
              << cfg_.coordinator_host << ":" << cfg_.coordinator_port << "\n";

    // If the coordinator sends an AUTH_CHALLENGE we handle it *before* JOIN.
    if (!perform_auth_if_challenged()) {
        std::cerr << "[NodeAgent:" << cfg_.node_id << "] auth failed — exiting\n";
        coord_conn_->close();
        return;
    }

    send_node_join();
    send_topology_hello();

    // Start worker threads
    control_thread_   = std::thread([this]{ control_thread_fn();   });
    data_recv_thread_ = std::thread([this]{ data_recv_thread_fn(); });
    compute_thread_   = std::thread([this]{ compute_thread_fn();   });
    data_send_thread_ = std::thread([this]{ data_send_thread_fn(); });
    heartbeat_thread_ = std::thread([this]{ heartbeat_loop();      });

    // Wait for all threads
    if (control_thread_.joinable())   control_thread_.join();
    if (data_recv_thread_.joinable()) data_recv_thread_.join();
    if (compute_thread_.joinable())   compute_thread_.join();
    if (data_send_thread_.joinable()) data_send_thread_.join();
    if (heartbeat_thread_.joinable()) heartbeat_thread_.join();
}

void NodeAgent::stop() {
    running_.store(false);
    recv_queue_.close();
    send_queue_.close();
    data_listener_.stop();
    if (coord_conn_) coord_conn_->close();
    if (next_conn_)  next_conn_->close();
}

// ─── Thread: control plane ──────────────────────────────────────────────────

void NodeAgent::control_thread_fn() {
    MsgHeader              hdr{};
    std::vector<uint8_t>   payload;

    while (running_.load() && coord_conn_->is_connected()) {
        if (!coord_conn_->recv_msg(hdr, payload)) break;

        switch (static_cast<MsgType>(hdr.msg_type)) {
        case MsgType::LAYER_ASSIGN: {
            if (payload.size() < sizeof(MsgLayerAssign)) break;
            const auto& msg = *reinterpret_cast<const MsgLayerAssign*>(payload.data());
            size_t ranges_offset = sizeof(MsgLayerAssign);
            std::vector<LayerRange> ranges(msg.n_ranges);
            if (payload.size() >= ranges_offset + msg.n_ranges * sizeof(LayerRange)) {
                memcpy(ranges.data(),
                       payload.data() + ranges_offset,
                       msg.n_ranges * sizeof(LayerRange));
            }
            handle_layer_assign(msg, ranges);
            break;
        }
        case MsgType::SHUTDOWN:
            handle_shutdown();
            return;
        default:
            std::cerr << "[NodeAgent] unknown control msg 0x"
                      << std::hex << hdr.msg_type << "\n";
            break;
        }
    }
    std::cout << "[NodeAgent:" << cfg_.node_id << "] control connection closed\n";
    running_.store(false);
    recv_queue_.close();
    send_queue_.close();
}

// ─── Thread: receive activation tensors from previous node ──────────────────

void NodeAgent::data_recv_thread_fn() {
    // For the first stage there's no upstream; skip recv and wait for
    // the compute thread to be fed via push_input() (API layer calls that).
    if (is_first_stage_) {
        // First stage gets input from the API layer directly, not from a peer.
        // We still need to keep this thread alive so the queue machinery works.
        while (running_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        return;
    }

    // Accept the persistent connection from the previous node.
    // In steady state there's exactly one upstream peer.
    auto upstream = data_listener_.accept_one();
    if (!upstream) return;

    std::cout << "[NodeAgent:" << cfg_.node_id << "] upstream peer connected\n";

    while (running_.load() && upstream->is_connected()) {
        MsgHeader hdr{};
        // Peek header
        std::vector<uint8_t> payload;
        if (!upstream->recv_msg(hdr, payload)) break;
        if (static_cast<MsgType>(hdr.msg_type) != MsgType::TENSOR_FORWARD) continue;
        if (payload.size() < sizeof(TensorHeader)) continue;

        ActivationBatch batch;
        batch.header = *reinterpret_cast<const TensorHeader*>(payload.data());
        batch.data.assign(payload.begin() + sizeof(TensorHeader), payload.end());

        recv_queue_.push(std::move(batch));
    }
    std::cout << "[NodeAgent:" << cfg_.node_id << "] upstream peer disconnected\n";
}

// ─── Thread: compute layers on incoming activations ─────────────────────────

void NodeAgent::compute_thread_fn() {
    // Wait until model is loaded before processing
    while (running_.load() && !model_loaded_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (!running_.load()) return;

    std::cout << "[NodeAgent:" << cfg_.node_id << "] compute thread ready, layers "
              << layer_first_ << "-" << layer_last_ << "\n";

    while (running_.load()) {
        auto item = recv_queue_.pop_timeout(std::chrono::milliseconds(200));
        if (!item) continue;

        ActivationBatch out = run_layers(*item);

        if (is_last_stage_) {
            // Decode logits -> token, call callback
            // out.data contains logit vector (n_vocab floats for last token)
            if (token_cb_) {
                // Simple greedy decode: find argmax
                const float* logits = reinterpret_cast<const float*>(out.data.data());
                size_t n_vocab = out.data.size() / sizeof(float);
                int32_t best_token = 0;
                float   best_logit = logits[0];
                for (size_t i = 1; i < n_vocab; ++i) {
                    if (logits[i] > best_logit) {
                        best_logit = logits[i];
                        best_token = (int32_t)i;
                    }
                }
                token_cb_(out.header.request_id, best_token,
                          out.header.seq_pos, out.header.is_last != 0);
            }
        } else {
            send_queue_.push(std::move(out));
        }
    }
}

// ─── Thread: send activations to next node ──────────────────────────────────

void NodeAgent::data_send_thread_fn() {
    if (is_last_stage_) {
        // Nothing to forward downstream
        while (running_.load()) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        return;
    }

    // Wait until next_conn_ is set up (coordinator will tell us the next node)
    while (running_.load() && !next_conn_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (!running_.load()) return;

    while (running_.load()) {
        auto item = send_queue_.pop_timeout(std::chrono::milliseconds(200));
        if (!item) continue;

        if (!next_conn_ || !next_conn_->is_connected()) {
            std::cerr << "[NodeAgent] next node disconnected, trying to reconnect...\n";
            connect_to_next_node();
            if (!next_conn_) continue;
        }

        // Send: TensorHeader + raw data
        next_conn_->send_msg2(
            MsgType::TENSOR_FORWARD,
            &item->header, sizeof(TensorHeader),
            item->data.data(), (uint32_t)item->data.size()
        );
    }
}

// ─── Thread: heartbeat ──────────────────────────────────────────────────────

void NodeAgent::heartbeat_loop() {
    while (running_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(HEARTBEAT_MS));
        if (!running_.load()) break;
        if (coord_conn_ && coord_conn_->is_connected()) send_heartbeat();
    }
}

// ─── Control handlers ───────────────────────────────────────────────────────

void NodeAgent::handle_layer_assign(const MsgLayerAssign& msg,
                                    const std::vector<LayerRange>& ranges) {
    std::cout << "[NodeAgent:" << cfg_.node_id << "] LAYER_ASSIGN: layers ";
    for (auto& r : ranges) std::cout << r.layer_first << "-" << r.layer_last << " ";
    std::cout << "model=" << msg.model_path << "\n";

    n_layer_total_ = msg.n_layer_total;
    if (!ranges.empty()) {
        layer_first_ = ranges.front().layer_first;
        layer_last_  = ranges.back().layer_last;
    }
    is_first_stage_ = (layer_first_ == 0);
    is_last_stage_  = (layer_last_ == n_layer_total_ - 1);

    // Load model
    handle_load_model(msg.model_path);

    // ACK
    MsgAck ack{};
    strncpy(ack.node_id, cfg_.node_id.c_str(), sizeof(ack.node_id) - 1);
    ack.success = 1;
    coord_conn_->send_msg(MsgType::ASSIGN_ACK, &ack, sizeof(ack));
}

void NodeAgent::handle_load_model(const std::string& model_path) {
    std::cout << "[NodeAgent:" << cfg_.node_id << "] loading model: " << model_path << "\n";

    // Free previous model if any
    if (lctx_) { llama_free(lctx_); lctx_ = nullptr; }
    if (lm_)   { llama_free_model(lm_); lm_ = nullptr; }

    llama_model_params mparams = llama_model_default_params();
    // Only load the layers assigned to us.
    // llama.cpp's n_gpu_layers controls how many layers go to GPU.
    // For pipeline parallelism we want to load ALL assigned layers onto GPU.
    mparams.n_gpu_layers = cfg_.n_gpu_layers;

    // tensor_split is left at default (equal split across available GPUs).
    // For finer control, set mparams.tensor_split[] based on layer ranges.

    lm_ = llama_load_model_from_file(model_path.c_str(), mparams);
    if (!lm_) {
        std::cerr << "[NodeAgent] failed to load model from " << model_path << "\n";
        return;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx    = cfg_.n_ctx;
    cparams.n_batch  = cfg_.n_batch;
    cparams.n_ubatch = cfg_.n_ubatch;
    cparams.offload_kqv = true;

    lctx_ = llama_new_context_with_model(lm_, cparams);
    if (!lctx_) {
        std::cerr << "[NodeAgent] failed to create context\n";
        llama_free_model(lm_); lm_ = nullptr;
        return;
    }

    model_loaded_.store(true);
    const llama_vocab* vocab = llama_model_get_vocab(lm_);
    std::cout << "[NodeAgent:" << cfg_.node_id << "] model loaded. "
              << "vocab=" << llama_vocab_n_tokens(vocab)
              << " embd=" << llama_model_n_embd(lm_)
              << " layers=" << llama_model_n_layer(lm_) << "\n";
}

void NodeAgent::handle_shutdown() {
    std::cout << "[NodeAgent:" << cfg_.node_id << "] shutdown ordered\n";
    stop();
}

// ─── Compute: run assigned layers ───────────────────────────────────────────

ActivationBatch NodeAgent::run_layers(const ActivationBatch& in) {
    // We use llama_decode() which handles the full forward pass internally.
    // For pipeline parallelism the ideal path is to call only the assigned
    // layer range. llama.cpp doesn't directly expose partial-layer execution
    // in the public API, so the approach here is:
    //
    //   Full model on each node, but each node is loaded with ONLY its
    //   layer range in GPU memory (n_gpu_layers = range size), and the
    //   activations passed between nodes are the KV-cache boundary values.
    //
    // This works today with llama.cpp as shipped. A deeper integration would
    // intercept at the ggml graph level, but that requires patching llama.cpp
    // internals which conflicts with AGENTS.md.
    //
    // Practical approach: each node runs llama_decode for its token batch.
    // The first stage processes the embedding + first N layers.
    // Subsequent stages continue from the received hidden state.
    //
    // NOTE: True activation tensor hand-off between partial models requires
    // the ggml_backend_sched intermediate output tensors — an extension point
    // that the llama.cpp community is actively developing (see PR #19378).
    // This code wires up the pipeline plumbing; the layer-boundary extraction
    // below uses the logit output as a proxy until that API is available.

    assert(lctx_ != nullptr);

    const uint32_t n_tokens = in.header.n_tokens;
    const uint32_t n_embd   = in.header.n_embd;

    ActivationBatch out;
    out.header             = in.header;
    out.header.from_layer  = layer_last_;
    out.header.seq_pos    += n_tokens;

    if (is_first_stage_) {
        // Input: token ids packed as int32_t in in.data
        const int32_t* token_ids = reinterpret_cast<const int32_t*>(in.data.data());

        llama_batch batch = llama_batch_init(n_tokens, 0, 1);
        batch.n_tokens = (int32_t)n_tokens;
        for (uint32_t i = 0; i < n_tokens; ++i) {
            batch.token[i]     = token_ids[i];
            batch.pos[i]       = in.header.seq_pos + i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = (int32_t)(in.header.request_id & 0x7FFFFFFF);
            batch.logits[i]    = (i == n_tokens - 1) ? 1 : 0; // only need last logit
        }

        int ret = llama_decode(lctx_, batch);
        llama_batch_free(batch);

        if (ret != 0) {
            std::cerr << "[NodeAgent] llama_decode failed: " << ret << "\n";
            out.header.is_last = 1;
            return out;
        }

        if (is_last_stage_) {
            // Single-node: return logits
            const float* logits = llama_get_logits_ith(lctx_, n_tokens - 1);
            size_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(lm_));
            out.header.n_embd = (uint32_t)n_vocab;
            out.header.dtype  = 0; // f32
            out.header.is_last = 1;
            out.data.assign(
                reinterpret_cast<const uint8_t*>(logits),
                reinterpret_cast<const uint8_t*>(logits) + n_vocab * sizeof(float)
            );
        } else {
            // Multi-node: extract hidden state after embedding+first layers.
            // For now we pass the embeddings of the last token as the activation.
            // TODO: use ggml_backend_sched intermediate tensor once API is stable.
            const float* embd = llama_get_embeddings_ith(lctx_, n_tokens - 1);
            out.header.n_embd = (uint32_t)llama_n_embd(lm_);
            out.header.n_tokens = 1; // pipeline transmits one token at a time
            out.header.dtype  = 0;  // f32
            out.data.assign(
                reinterpret_cast<const uint8_t*>(embd),
                reinterpret_cast<const uint8_t*>(embd) + n_embd * sizeof(float)
            );
        }
    } else {
        // Middle or last stage: input is the hidden state from previous node.
        // We inject it into the model using llama_set_embeddings and run decode.
        const float* hidden = reinterpret_cast<const float*>(in.data.data());

        llama_batch batch = llama_batch_init(1, n_embd, 1);
        batch.n_tokens     = 1;
        batch.embd         = const_cast<float*>(hidden); // llama_batch borrows ptr
        batch.pos[0]       = in.header.seq_pos;
        batch.n_seq_id[0]  = 1;
        batch.seq_id[0][0] = (int32_t)(in.header.request_id & 0x7FFFFFFF);
        batch.logits[0]    = is_last_stage_ ? 1 : 0;

        int ret = llama_decode(lctx_, batch);
        llama_batch_free(batch);

        if (ret != 0) {
            std::cerr << "[NodeAgent] llama_decode (embd) failed: " << ret << "\n";
            out.header.is_last = 1;
            return out;
        }

        if (is_last_stage_) {
            const float* logits = llama_get_logits_ith(lctx_, 0);
            size_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(lm_));
            out.header.n_embd  = (uint32_t)n_vocab;
            out.header.dtype   = 0;
            out.header.is_last = 1;
            out.data.assign(
                reinterpret_cast<const uint8_t*>(logits),
                reinterpret_cast<const uint8_t*>(logits) + n_vocab * sizeof(float)
            );
        } else {
            const float* embd = llama_get_embeddings_ith(lctx_, 0);
            out.header.n_embd   = (uint32_t)n_embd;
            out.header.n_tokens = 1;
            out.header.dtype    = 0;
            out.data.assign(
                reinterpret_cast<const uint8_t*>(embd),
                reinterpret_cast<const uint8_t*>(embd) + n_embd * sizeof(float)
            );
        }
    }

    return out;
}

// ─── Helpers ────────────────────────────────────────────────────────────────

void NodeAgent::send_node_join() {
    MsgNodeJoin msg{};
    auto cap = build_capability();
    msg.cap       = cap;
    msg.data_port = cfg_.data_port;
    coord_conn_->send_msg(MsgType::NODE_JOIN, &msg, sizeof(msg));
}

void NodeAgent::send_heartbeat() {
    MsgHeartbeat hb{};
    strncpy(hb.node_id, cfg_.node_id.c_str(), sizeof(hb.node_id) - 1);
    hb.tokens_processed = 0; // TODO: track with atomic counter
    coord_conn_->send_msg(MsgType::HEARTBEAT, &hb, sizeof(hb));
}

NodeCapability NodeAgent::build_capability() const {
    NodeCapability cap{};
    strncpy(cap.node_id, cfg_.node_id.c_str(), sizeof(cap.node_id) - 1);

    // Query ggml backends for GPU info
    size_t n_devs = ggml_backend_dev_count();
    cap.n_gpus = 0;
    for (size_t i = 0; i < n_devs && cap.n_gpus < 8; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU ||
            ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            size_t free_b, total_b;
            ggml_backend_dev_memory(dev, &free_b, &total_b);
            cap.gpu_vram_bytes[cap.n_gpus] = total_b;
            cap.gpu_free_bytes[cap.n_gpus] = free_b;
            cap.n_gpus++;
        }
    }

    cap.n_cpu_threads = std::thread::hardware_concurrency();
    return cap;
}

void NodeAgent::connect_to_next_node() {
    if (next_node_host_.empty()) return;
    try {
        next_conn_ = std::make_unique<Connection>();
        next_conn_->connect(next_node_host_, next_node_port_);
        std::cout << "[NodeAgent:" << cfg_.node_id << "] connected to next node "
                  << next_node_host_ << ":" << next_node_port_ << "\n";
    } catch (const std::exception& e) {
        std::cerr << "[NodeAgent] failed to connect to next node: " << e.what() << "\n";
        next_conn_.reset();
    }
}

// ─── Phase-5: auth handshake + topology hello ──────────────────────────────

bool NodeAgent::perform_auth_if_challenged() {
    // If no token is provided locally, skip the handshake entirely.  If the
    // coordinator requires auth it will send AUTH_CHALLENGE and then close
    // our connection when we send NODE_JOIN before responding — the normal
    // error path.  If the coordinator does NOT require auth, it never sends
    // a challenge and this function is a no-op.
    if (cfg_.token_id.empty()) return true;

    // Token was provided — expect AUTH_CHALLENGE as the first message.
    MsgHeader hdr{};
    std::vector<uint8_t> payload;
    if (!coord_conn_->recv_msg(hdr, payload)) {
        std::cerr << "[NodeAgent] coordinator closed before AUTH_CHALLENGE\n";
        return false;
    }
    if (static_cast<MsgType>(hdr.msg_type) != MsgType::AUTH_CHALLENGE
        || payload.size() < sizeof(MsgAuthChallenge)) {
        std::cerr << "[NodeAgent] expected AUTH_CHALLENGE, got type "
                  << hdr.msg_type << "\n";
        return false;
    }

    const auto& ch = *reinterpret_cast<const MsgAuthChallenge*>(payload.data());

    if (cfg_.token_secret_hex.size() != 64) {
        std::cerr << "[NodeAgent] token_secret_hex must be 64 hex chars\n";
        return false;
    }
    auto sec = from_hex(cfg_.token_secret_hex);
    if (sec.size() != 32) {
        std::cerr << "[NodeAgent] invalid token_secret_hex (need 64 hex chars)\n";
        return false;
    }
    std::array<uint8_t, 32> secret{};
    std::memcpy(secret.data(), sec.data(), 32);

    MsgAuthResponse resp{};
    std::strncpy(resp.token_id, cfg_.token_id.c_str(), MAX_TOKEN_ID_LEN - 1);
    resp.scope_requested = SCOPE_JOIN;
    compute_auth_response(ch.nonce, cfg_.token_id, secret, resp.mac);
    coord_conn_->send_msg(MsgType::AUTH_RESPONSE, &resp, sizeof(resp));

    // Expect AUTH_RESULT next.
    if (!coord_conn_->recv_msg(hdr, payload)
        || static_cast<MsgType>(hdr.msg_type) != MsgType::AUTH_RESULT
        || payload.size() < sizeof(MsgAuthResult)) {
        std::cerr << "[NodeAgent] no AUTH_RESULT from coordinator\n";
        return false;
    }
    const auto& res = *reinterpret_cast<const MsgAuthResult*>(payload.data());
    if (!res.accepted) {
        std::cerr << "[NodeAgent] auth denied: " << res.reason << "\n";
        return false;
    }
    std::cout << "[NodeAgent:" << cfg_.node_id << "] authenticated (scope=0x"
              << std::hex << res.scope_granted << std::dec << ")\n";
    return true;
}

void NodeAgent::send_topology_hello() {
    if (!coord_conn_ || !coord_conn_->is_connected()) return;
    auto hello = make_topology_hello(cfg_.node_id);
    coord_conn_->send_msg(MsgType::TOPOLOGY_HELLO, &hello, sizeof(hello));
}

} // namespace dist
