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

    // Dashboard / monitoring plane (HTTP server polls these internally)
    CLUSTER_STATS_REQ = 0x0300, // request full cluster snapshot
    CLUSTER_STATS_RSP = 0x0301, // response: MsgClusterStats + NodeStat[]
    NODE_STATS_REQ    = 0x0302, // request one node's live stats
    NODE_STATS_RSP    = 0x0303, // response: MsgNodeStats

    // ─── Phase 5: global-scale / production ──────────────────────────────────

    // Auth (node -> coordinator and coordinator -> node)
    AUTH_CHALLENGE    = 0x0400, // coord asks node to prove token possession
    AUTH_RESPONSE     = 0x0401, // node replies with HMAC over challenge
    AUTH_RESULT       = 0x0402, // coord: accepted / denied with reason

    // Topology (node -> coordinator)
    TOPOLOGY_HELLO    = 0x0410, // node's region/zone/rack/geo hints
    TOPOLOGY_LATENCY  = 0x0411, // probe round-trip sample, node <-> node

    // Contribution accounting (coordinator -> node)
    CONTRIB_RECEIPT   = 0x0420, // signed receipt: layers*tokens*uptime

    // Federation (regional coord <-> global coord)
    FEDERATION_HELLO  = 0x0430, // regional announces itself to global
    FEDERATION_STATS  = 0x0431, // regional pushes rollup stats upward
    FEDERATION_ROUTE  = 0x0432, // global steers a client to a region

    // Weight P2P (node <-> node)
    WEIGHT_PEER_QUERY     = 0x0440, // who has layer L for model M?
    WEIGHT_PEER_ADVERTISE = 0x0441, // I have layers [a..b] for model M
    WEIGHT_CHUNK_REQUEST  = 0x0442, // please send me bytes [off..off+len]
    WEIGHT_CHUNK_DATA     = 0x0443, // here are the bytes

    // Rate limit (coordinator -> client / coordinator -> coordinator)
    RATE_LIMIT_QUOTA  = 0x0450, // current per-tenant bucket state
};

// Alias for use outside the dist_protocol.h include (vm_context etc.)
using MsgTypeEnum = MsgType;

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

static_assert(sizeof(TensorHeader) == 28, "TensorHeader size changed");

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

// ─── Dashboard / stats structs ───────────────────────────────────────────────

#pragma pack(push, 1)

// Per-node live stats — sent in CLUSTER_STATS_RSP and NODE_STATS_RSP.
struct NodeStatEntry {
    char     node_id[MAX_NODE_ID_LEN];
    char     addr[64];              // IP:port
    uint32_t n_gpus;
    uint64_t gpu_vram_total[8];     // bytes
    uint64_t gpu_vram_free[8];      // bytes
    float    gpu_util[8];           // 0–1
    uint64_t cpu_ram_total;
    uint64_t cpu_ram_free;
    uint32_t n_cpu_threads;
    uint32_t network_mbps;
    uint64_t tokens_total;          // lifetime tokens processed by this node
    double   tokens_per_second;     // rolling 10-second average
    uint32_t layer_first;           // first assigned layer
    uint32_t layer_last;            // last assigned layer
    uint64_t bytes_received;        // bytes of activation tensors received
    uint64_t bytes_sent;            // bytes of activation tensors forwarded
    uint8_t  alive;
    uint8_t  model_loaded;
    uint8_t  _pad[6];
};

struct MsgClusterStats {
    uint64_t     timestamp_us;        // unix microseconds
    uint32_t     n_nodes;             // number of NodeStatEntry entries that follow
    uint32_t     n_active_requests;   // in-flight inference requests right now
    uint64_t     tokens_total;        // cluster lifetime total
    double       tokens_per_second;   // cluster aggregate TPS (rolling)
    char         model_name[128];
    uint32_t     n_layers_total;
    // followed by n_nodes * sizeof(NodeStatEntry)
};

struct MsgNodeStats {
    NodeStatEntry entry;
};

// ─── Phase 5: auth, topology, receipts, federation, weight-p2p ──────────────

static constexpr uint32_t AUTH_CHALLENGE_BYTES = 32;
static constexpr uint32_t AUTH_MAC_BYTES       = 32; // HMAC-SHA256
static constexpr uint32_t MAX_REGION_LEN       = 32;
static constexpr uint32_t MAX_ZONE_LEN         = 32;
static constexpr uint32_t MAX_RACK_LEN         = 32;
static constexpr uint32_t MAX_TOKEN_ID_LEN     = 64;
static constexpr uint32_t MAX_TENANT_LEN       = 64;

// Capability scopes bit-encoded in the token; enforced by coordinator.
enum AuthScope : uint32_t {
    SCOPE_NONE      = 0,
    SCOPE_JOIN      = 1u << 0, // may join the pool as a worker
    SCOPE_CLIENT    = 1u << 1, // may submit inference requests
    SCOPE_FEDERATE  = 1u << 2, // may act as a regional coordinator uplink
    SCOPE_ADMIN     = 1u << 3, // may issue new tokens (root only)
};

// AUTH_CHALLENGE: coord -> node. 32 bytes of random nonce.
struct MsgAuthChallenge {
    uint8_t nonce[AUTH_CHALLENGE_BYTES];
    uint64_t issued_at_us;
};

// AUTH_RESPONSE: node -> coord. HMAC_SHA256(token_secret, nonce || token_id).
struct MsgAuthResponse {
    char     token_id[MAX_TOKEN_ID_LEN];   // which token was used
    uint8_t  mac[AUTH_MAC_BYTES];          // HMAC over challenge+token_id
    uint32_t scope_requested;              // bitfield of AuthScope
    uint8_t  _pad[4];
};

// AUTH_RESULT: coord -> node.
struct MsgAuthResult {
    uint8_t  accepted;          // 1 = ok, 0 = denied
    uint8_t  _pad[3];
    uint32_t scope_granted;
    uint64_t expires_at_us;     // when this session expires
    char     reason[128];       // human-readable denial reason
};

// TOPOLOGY_HELLO: node -> coord. Self-reported location + hints.
struct MsgTopologyHello {
    char     node_id[MAX_NODE_ID_LEN];
    char     region[MAX_REGION_LEN];   // e.g. "us-west"
    char     zone[MAX_ZONE_LEN];       // e.g. "us-west-2a"
    char     rack[MAX_RACK_LEN];       // free-form; empty if unknown
    float    lat_deg;                  // GeoIP optional
    float    lon_deg;
    uint32_t bandwidth_mbps_self;      // measured, not reported
    uint8_t  behind_nat;               // 1 = node cannot accept inbound
    uint8_t  _pad[3];
};

// TOPOLOGY_LATENCY: a latency probe sample from node A to node B.
struct MsgTopologyLatency {
    char     src_node[MAX_NODE_ID_LEN];
    char     dst_node[MAX_NODE_ID_LEN];
    float    rtt_ms;
    uint64_t measured_at_us;
};

// CONTRIB_RECEIPT: coord -> node. Signed record of work performed in a window.
// HMAC is over (node_id | window_start | window_end | tokens | layer_bytes).
struct MsgContribReceipt {
    char     node_id[MAX_NODE_ID_LEN];
    char     tenant[MAX_TENANT_LEN];
    uint64_t window_start_us;
    uint64_t window_end_us;
    uint64_t tokens_processed;     // output tokens this node contributed to
    uint64_t layer_bytes_forwarded;
    uint64_t layer_seconds;         // layers_assigned * seconds_up (work units)
    uint8_t  mac[AUTH_MAC_BYTES];   // issuer signature
    char     issuer_id[MAX_TOKEN_ID_LEN]; // which coordinator signed
    uint8_t  _pad[4];
};

// FEDERATION_HELLO: regional coord -> global coord.
struct MsgFederationHello {
    char     region[MAX_REGION_LEN];
    char     coord_id[MAX_NODE_ID_LEN];
    char     public_host[128];
    uint16_t ctrl_port;
    uint16_t api_port;
    uint16_t dashboard_port;
    uint8_t  _pad[2];
    uint8_t  mac[AUTH_MAC_BYTES];
};

// FEDERATION_STATS: regional -> global rollup.
struct MsgFederationStats {
    char     region[MAX_REGION_LEN];
    uint32_t n_nodes;
    uint32_t n_active_requests;
    uint64_t tokens_total;
    double   tokens_per_second;
    uint64_t total_vram_bytes;
    uint64_t free_vram_bytes;
    uint64_t timestamp_us;
};

// FEDERATION_ROUTE: global -> client (piggybacked on rendezvous HTTP too).
struct MsgFederationRoute {
    char     region[MAX_REGION_LEN];
    char     coord_host[128];
    uint16_t api_port;
    uint16_t dashboard_port;
    uint8_t  _pad[4];
    float    score;                // higher = better match
};

// WEIGHT_PEER_QUERY: node A asks the coord for peers holding a layer range.
struct MsgWeightPeerQuery {
    char     model_name[MAX_MODEL_NAME];
    uint32_t layer_first;
    uint32_t layer_last;
};

// WEIGHT_PEER_ADVERTISE: coord replies with a set of peers (one message per
// peer, or the client concatenates; keep it fixed-size per record).
struct MsgWeightPeerAdvertise {
    char     peer_node_id[MAX_NODE_ID_LEN];
    char     peer_host[128];
    uint16_t peer_data_port;
    uint8_t  _pad[2];
    uint32_t layer_first;
    uint32_t layer_last;
    uint64_t total_bytes;
};

// WEIGHT_CHUNK_REQUEST: node A -> node B.
struct MsgWeightChunkRequest {
    char     model_name[MAX_MODEL_NAME];
    uint32_t layer_index;
    uint64_t offset;
    uint64_t length;
};

// WEIGHT_CHUNK_DATA: node B -> node A. Payload is length bytes of weight data.
struct MsgWeightChunkData {
    uint32_t layer_index;
    uint32_t _pad;
    uint64_t offset;
    uint64_t length;
    // followed by `length` bytes
};

// RATE_LIMIT_QUOTA: coordinator reports a tenant's current bucket state.
struct MsgRateLimitQuota {
    char     tenant[MAX_TENANT_LEN];
    uint64_t tokens_remaining;
    uint64_t refill_per_sec;
    uint64_t reset_at_us;
    uint8_t  priority;             // 0=low, 1=normal, 2=high
    uint8_t  _pad[7];
};

#pragma pack(pop)

} // namespace dist
