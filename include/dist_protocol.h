#pragma once
/**
 * dist_protocol.h
 *
 * Wire protocol for the distributed inference pipeline.
 *
 * All multi-byte integers are little-endian.
 * Every message starts with a MsgHeader, followed by a payload
 * whose layout is described by the msg_type field.
 *
 * Transport: persistent TCP connections.
 *   Coordinator  <-> NodeAgent : control plane  (port 7700)
 *   NodeAgent    <-> NodeAgent : data plane      (port 7701)
 *   Client       <-> Coordinator : API plane     (port 7702)
 */

#include <cstdint>
#include <cstring>

namespace dist {

// ─── Constants ──────────────────────────────────────────────────────────────

static constexpr uint16_t PORT_CONTROL = 7700;
static constexpr uint16_t PORT_DATA    = 7701;
static constexpr uint16_t PORT_API     = 7702;

static constexpr uint32_t PROTO_MAGIC   = 0xD157C0DE; // "DIST CODE"
static constexpr uint32_t PROTO_VERSION = 1;

static constexpr uint32_t MAX_NODES        = 64;
static constexpr uint32_t MAX_LAYERS       = 256;
static constexpr uint32_t MAX_MODEL_NAME   = 128;
static constexpr uint32_t MAX_NODE_ID_LEN  = 64;
static constexpr uint32_t HEARTBEAT_MS     = 2000;
static constexpr uint32_t DEAD_NODE_MS     = 8000;

// ─── Message types ──────────────────────────────────────────────────────────

enum class MsgType : uint16_t {
    // Control plane (node -> coordinator)
    NODE_JOIN        = 0x0001,  // Node announces itself
    NODE_LEAVE       = 0x0002,  // Graceful shutdown
    HEARTBEAT        = 0x0003,  // Keepalive + stats update

    // Control plane (coordinator -> node)
    LAYER_ASSIGN     = 0x0010,  // Assign layer range to node
    LOAD_MODEL       = 0x0011,  // Load model weights for assigned layers
    UNLOAD_MODEL     = 0x0012,  // Free model weights
    SHUTDOWN         = 0x0013,  // Coordinator ordering node shutdown

    // Control plane (coordinator -> coordinator / internal)
    ASSIGN_ACK       = 0x0020,  // Node confirms assignment
    LOAD_ACK         = 0x0021,  // Node confirms model loaded
    ERROR            = 0x00FF,

    // Data plane (node -> node): activation tensors
    TENSOR_FORWARD   = 0x0100,  // Hidden state flowing forward through pipeline
    TENSOR_BACKWARD  = 0x0101,  // (future: gradients for fine-tuning)
    TENSOR_ACK       = 0x0102,  // Receiver confirms tensor received & queued

    // API plane (client -> coordinator)
    INFER_REQUEST    = 0x0200,
    INFER_TOKEN      = 0x0201,  // Streaming: one generated token
    INFER_DONE       = 0x0202,
    INFER_ERROR      = 0x0203,
};

// ─── Packed message header (always first) ───────────────────────────────────

#pragma pack(push, 1)

struct MsgHeader {
    uint32_t magic;       // PROTO_MAGIC
    uint32_t version;     // PROTO_VERSION
    uint16_t msg_type;    // cast to MsgType
    uint16_t flags;       // reserved, 0
    uint32_t payload_len; // bytes following this header
    uint64_t seq;         // monotonically increasing per-connection sequence number
};

static_assert(sizeof(MsgHeader) == 24, "MsgHeader size changed");

// ─── Control: NODE_JOIN ─────────────────────────────────────────────────────

struct NodeCapability {
    char     node_id[MAX_NODE_ID_LEN]; // unique name, e.g. hostname:pid
    uint32_t n_gpus;
    uint64_t gpu_vram_bytes[8];        // per-GPU VRAM total
    uint64_t gpu_free_bytes[8];        // per-GPU VRAM free
    uint64_t cpu_ram_bytes;            // total system RAM
    uint64_t cpu_ram_free_bytes;
    uint32_t n_cpu_threads;
    uint32_t network_bandwidth_mbps;   // self-reported or 0
    uint8_t  supports_rdma;
    uint8_t  _pad[3];
};

struct MsgNodeJoin {
    NodeCapability cap;
    uint16_t       data_port;  // port this node listens on for TENSOR_FORWARD
    uint8_t        _pad[2];
};

// ─── Control: HEARTBEAT ─────────────────────────────────────────────────────

struct MsgHeartbeat {
    char     node_id[MAX_NODE_ID_LEN];
    uint64_t gpu_free_bytes[8];  // current free VRAM
    uint64_t cpu_ram_free_bytes;
    float    gpu_util[8];        // 0.0-1.0 utilisation per GPU
    uint32_t tokens_processed;   // since last heartbeat
};

// ─── Control: LAYER_ASSIGN ──────────────────────────────────────────────────

struct LayerRange {
    uint32_t layer_first; // inclusive
    uint32_t layer_last;  // inclusive
    uint32_t gpu_index;   // which local GPU to use (0 = first)
};

struct MsgLayerAssign {
    char       node_id[MAX_NODE_ID_LEN];
    char       model_path[256];          // GGUF file path (or empty if using cache)
    char       model_name[MAX_MODEL_NAME];
    uint32_t   n_layer_total;            // total layers in the model
    uint32_t   n_ranges;                 // how many LayerRange entries follow
    uint32_t   n_ctx;                    // context window size
    // followed by n_ranges * sizeof(LayerRange) bytes
};

// ─── Control: ASSIGN_ACK / LOAD_ACK ─────────────────────────────────────────

struct MsgAck {
    char    node_id[MAX_NODE_ID_LEN];
    uint8_t success;
    char    message[128]; // error description if !success
};

// ─── Data plane: TENSOR_FORWARD ─────────────────────────────────────────────
//
// Layout of a TENSOR_FORWARD payload:
//   [ TensorHeader ] [ tensor bytes (n_elements * element_size) ]
//
// For pipeline parallelism the tensor is the hidden state after layer `from_layer`.
// Shape is always [n_tokens, n_embd] in row-major order.

struct TensorHeader {
    uint64_t request_id;   // ties this tensor to an inference request
    uint32_t from_layer;   // last layer computed (0 = embedding output)
    uint32_t seq_pos;      // position in sequence (for KV cache indexing)
    uint32_t n_tokens;     // rows
    uint32_t n_embd;       // cols (hidden dimension)
    uint16_t dtype;        // 0=f32, 1=f16, 2=bf16
    uint8_t  is_last;      // 1 if this is the final stage output
    uint8_t  _pad;
    // followed by n_tokens * n_embd * sizeof(dtype) bytes
};

static_assert(sizeof(TensorHeader) == 32, "TensorHeader size changed");

// ─── API plane: INFER_REQUEST ───────────────────────────────────────────────

struct MsgInferRequest {
    uint64_t request_id;        // assigned by client, echoed back
    uint32_t n_prompt_tokens;   // number of token ids that follow this struct
    uint32_t max_gen_tokens;
    float    temperature;
    float    top_p;
    int32_t  seed;
    char     model_name[MAX_MODEL_NAME];
    // followed by n_prompt_tokens * sizeof(int32_t) token ids
};

struct MsgInferToken {
    uint64_t request_id;
    uint32_t token_id;
    uint32_t token_pos;   // position in generated sequence
    float    logprob;
};

struct MsgInferDone {
    uint64_t request_id;
    uint32_t n_tokens_generated;
    double   time_to_first_token_ms;
    double   tokens_per_second;
};

struct MsgInferError {
    uint64_t request_id;
    char     message[256];
};

#pragma pack(pop)

// ─── Helpers ────────────────────────────────────────────────────────────────

inline MsgHeader make_header(MsgType type, uint32_t payload_len, uint64_t seq = 0) {
    MsgHeader h{};
    h.magic       = PROTO_MAGIC;
    h.version     = PROTO_VERSION;
    h.msg_type    = static_cast<uint16_t>(type);
    h.flags       = 0;
    h.payload_len = payload_len;
    h.seq         = seq;
    return h;
}

inline size_t dtype_size(uint16_t dtype) {
    switch (dtype) {
        case 0: return 4; // f32
        case 1: return 2; // f16
        case 2: return 2; // bf16
        default: return 4;
    }
}

inline size_t tensor_payload_bytes(const TensorHeader& th) {
    return sizeof(TensorHeader)
         + (size_t)th.n_tokens * th.n_embd * dtype_size(th.dtype);
}

} // namespace dist
