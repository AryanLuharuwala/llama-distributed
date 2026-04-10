#pragma once
/**
 * vm_protocol.h
 *
 * VM-layer wire protocol extensions.
 * Extends dist_protocol.h without modifying it.
 *
 * All new message types live in the 0x0300+ range.
 * The same MsgHeader framing from dist_protocol.h is used.
 *
 * New port:
 *   PORT_VM_CTRL = 7703  — VM control plane (VmCoordinator <-> VmNode <-> VmContext)
 *
 * Data-plane collectives reuse PORT_DATA (7701) connections.
 */

#include "dist_protocol.h"
#include <cstdint>

namespace dist {

static constexpr uint16_t PORT_VM_CTRL = 7703;

// ─── Extended MsgType values ─────────────────────────────────────────────────
//
// These complement the existing MsgType enum.
// Cast to/from uint16_t when placing in MsgHeader.msg_type.

enum class VmMsgType : uint16_t {
    // Tensor memory management (VmContext <-> VmCoordinator)
    VM_TENSOR_ALLOC    = 0x0300,  // request: alloc a virtual tensor
    VM_TENSOR_ALLOC_RSP= 0x0301,  // response: vaddr assigned
    VM_TENSOR_FREE     = 0x0302,  // free a virtual tensor
    VM_TENSOR_WRITE    = 0x0303,  // write host data into vaddr
    VM_TENSOR_READ     = 0x0304,  // read vaddr data to host
    VM_TENSOR_READ_RSP = 0x0305,  // response carrying data

    // Op dispatch (VmContext -> VmCoordinator -> VmNode)
    VM_OP_DISPATCH     = 0x0310,  // schedule op on best node
    VM_OP_RESULT       = 0x0311,  // op finished, result bytes
    VM_OP_REJECT       = 0x0312,  // node too busy, re-dispatch elsewhere

    // Collective operations (VmCoordinator orchestrates)
    VM_COLLECTIVE_INIT  = 0x0320, // start a collective (coord -> all participants)
    VM_COLLECTIVE_CHUNK = 0x0321, // ring chunk (node -> node via PORT_DATA)
    VM_COLLECTIVE_DONE  = 0x0322, // node signals completion to coord
    VM_COLLECTIVE_RSP   = 0x0323, // coord sends final result to VmContext

    // Fault tolerance
    VM_CHECKPOINT_SNAP = 0x0330,  // coord -> node: write checkpoint now
    VM_CHECKPOINT_ACK  = 0x0331,  // node -> coord: checkpoint written/failed
    VM_RESTORE_REQ     = 0x0332,  // coord -> node: restore from checkpoint
    VM_RESTORE_ACK     = 0x0333,  // node -> coord: restore done

    // Topology
    VM_NODE_READY      = 0x0340,  // node -> coord: VM layer is up
    VM_TOPO_UPDATE     = 0x0341,  // coord -> all: cluster topology changed

    // Tensor migration (coord -> node: take ownership of vaddr)
    VM_TENSOR_MIGRATE  = 0x0350,
};

// Collective operation types
enum class CollectiveOp : uint8_t {
    AllReduce = 0,
    AllGather = 1,
    Broadcast = 2,
};

// Reduce operations used by AllReduce
enum class ReduceOp : uint8_t {
    Sum = 0,
    Max = 1,
    Min = 2,
};

// Op types for VM_OP_DISPATCH
enum class VmOpType : uint32_t {
    MatMul    = 0,
    Add       = 1,
    Scale     = 2,
    Softmax   = 3,
    RmsNorm   = 4,
    SiluMul   = 5,   // SwiGLU activation
    RopeFwd   = 6,
    Custom    = 255, // raw ggml subgraph bytes in payload
};

// ─── Packed message structs ───────────────────────────────────────────────────

#pragma pack(push, 1)

// VM_TENSOR_ALLOC  (VmContext -> VmCoordinator)
struct MsgVmTensorAlloc {
    uint64_t client_tag;          // echoed back in RSP so client matches async response
    uint32_t n_bytes;
    uint16_t dtype;               // 0=f32 1=f16 2=bf16
    uint8_t  _pad[2];
    char     hint_node[64];       // preferred owning node (may be empty)
};

// VM_TENSOR_ALLOC_RSP  (VmCoordinator -> VmContext)
struct MsgVmTensorAllocRsp {
    uint64_t client_tag;
    uint64_t vaddr;               // 0 if allocation failed
    char     owning_node[64];
};

// VM_TENSOR_FREE
struct MsgVmTensorFree {
    uint64_t vaddr;
};

// VM_TENSOR_WRITE  (VmContext -> VmCoordinator, then forwarded to owning node)
// Payload: this struct followed by n_bytes of data
struct MsgVmTensorWrite {
    uint64_t vaddr;
    uint64_t op_id;               // for async ack
    uint32_t offset;
    uint32_t n_bytes;
};

// VM_TENSOR_READ (VmContext -> VmCoordinator)
struct MsgVmTensorRead {
    uint64_t vaddr;
    uint64_t op_id;
    uint32_t offset;
    uint32_t n_bytes;
};

// VM_TENSOR_READ_RSP (VmCoordinator -> VmContext)
// Payload: this struct followed by n_bytes of data
struct MsgVmTensorReadRsp {
    uint64_t vaddr;
    uint64_t op_id;
    uint32_t n_bytes;
    uint8_t  success;
    uint8_t  _pad[3];
};

// VM_OP_DISPATCH (VmCoordinator -> VmNode)
// Payload layout: this struct | n_inputs * uint64_t (input vaddrs) | op_payload_bytes
struct MsgVmOpDispatch {
    uint64_t op_id;
    uint64_t request_id;
    uint32_t op_type;             // cast to VmOpType
    uint32_t n_inputs;
    uint32_t n_output_bytes;      // expected output size
    uint32_t op_payload_bytes;    // bytes of op description following vaddrs
};

// VM_OP_RESULT (VmNode -> VmCoordinator -> VmContext)
// Payload: this struct followed by n_output_bytes of result data
struct MsgVmOpResult {
    uint64_t op_id;
    uint64_t request_id;
    uint32_t n_output_bytes;
    uint8_t  success;
    uint8_t  _pad[3];
};

// VM_OP_REJECT (VmNode -> VmCoordinator)
struct MsgVmOpReject {
    uint64_t op_id;
    char     node_id[64];
};

// VM_COLLECTIVE_INIT (VmCoordinator -> all participants)
// Followed by n_participants * 64-byte node_id strings
struct MsgVmCollectiveInit {
    uint64_t coll_id;
    uint8_t  coll_op;             // cast to CollectiveOp
    uint8_t  reduce_op;           // cast to ReduceOp (for AllReduce)
    uint16_t dtype;
    uint32_t n_participants;
    uint32_t data_bytes;          // total tensor bytes being reduced/gathered
    char     root_node[64];       // only for Broadcast
};

// VM_COLLECTIVE_CHUNK (node -> next node in ring, over PORT_DATA)
// Payload: this struct followed by chunk_bytes of partial-sum data
struct MsgVmCollectiveChunk {
    uint64_t coll_id;
    char     from_node[64];
    uint32_t chunk_offset;
    uint32_t chunk_bytes;
    uint8_t  phase;               // 0=scatter, 1=gather
    uint8_t  _pad[3];
};

// VM_COLLECTIVE_DONE (VmNode -> VmCoordinator)
// Payload: this struct followed by data_bytes of result (AllReduce/AllGather result)
struct MsgVmCollectiveDone {
    uint64_t coll_id;
    char     node_id[64];
    uint32_t data_bytes;
    uint8_t  success;
    uint8_t  _pad[3];
};

// VM_CHECKPOINT_SNAP (VmCoordinator -> VmNode)
struct MsgVmCheckpointSnap {
    uint64_t snap_id;
    char     storage_path[256];   // base path; node appends its own node_id
};

// VM_CHECKPOINT_ACK (VmNode -> VmCoordinator)
struct MsgVmCheckpointAck {
    uint64_t snap_id;
    char     node_id[64];
    uint8_t  success;
    char     message[128];
};

// VM_RESTORE_REQ (VmCoordinator -> VmNode)
struct MsgVmRestoreReq {
    uint64_t snap_id;
    char     storage_path[256];
};

// VM_TOPO_UPDATE (VmCoordinator -> all nodes and clients)
// Followed by n_nodes * sizeof(NodeCapability) bytes
struct MsgVmTopoUpdate {
    uint32_t epoch;               // monotonically increasing
    uint32_t n_nodes;
    uint32_t n_pipeline_stages;   // updated pipeline order
    // followed by: n_nodes NodeCapability structs
    // followed by: n_pipeline_stages * 64-byte node_id strings (pipeline order)
};

// VM_NODE_READY (VmNode -> VmCoordinator)
struct MsgVmNodeReady {
    char node_id[64];
    uint16_t vm_data_port;        // port VmNode listens on for collective chunks
    uint8_t  _pad[6];
};

// VM_TENSOR_MIGRATE (VmCoordinator -> VmNode)
// Tells node it now owns vaddr; data may follow or be fetched from checkpoint
// Payload: this struct followed by data_bytes of tensor data (if available)
struct MsgVmTensorMigrate {
    uint64_t vaddr;
    uint32_t n_bytes;
    uint16_t dtype;
    uint8_t  has_data;            // 1 if tensor data follows this struct
    uint8_t  _pad;
};

#pragma pack(pop)

// ─── Helpers ─────────────────────────────────────────────────────────────────

inline MsgHeader make_vm_header(VmMsgType type, uint32_t payload_len, uint64_t seq = 0) {
    return make_header(static_cast<MsgType>(type), payload_len, seq);
}

// Reduce two f32 arrays in-place: acc[i] = op(acc[i], src[i])
inline void reduce_f32(float* acc, const float* src, size_t n, ReduceOp op) {
    switch (op) {
    case ReduceOp::Sum:
        for (size_t i = 0; i < n; ++i) acc[i] += src[i];
        break;
    case ReduceOp::Max:
        for (size_t i = 0; i < n; ++i) acc[i] = acc[i] > src[i] ? acc[i] : src[i];
        break;
    case ReduceOp::Min:
        for (size_t i = 0; i < n; ++i) acc[i] = acc[i] < src[i] ? acc[i] : src[i];
        break;
    }
}

// dtype-dispatching reduce (f32 only for now; extend for f16/bf16)
inline void reduce_inplace(uint8_t* acc, const uint8_t* src,
                            uint32_t n_bytes, uint16_t dtype, ReduceOp op) {
    if (dtype == 0 && n_bytes % 4 == 0) {
        reduce_f32(reinterpret_cast<float*>(acc),
                   reinterpret_cast<const float*>(src),
                   n_bytes / 4, op);
    }
    // TODO: f16/bf16 via half-precision libraries
}

} // namespace dist
