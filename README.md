# llama-distributed

Distributed LLM inference across a pool of GPU servers.

Models that don't fit on any single machine are split layer-by-layer across
as many nodes as you have. Each node holds a slice of the weights; activation
tensors flow through the pipeline stage-by-stage. An optional **VM layer**
makes the entire cluster appear as one unified computer with a shared virtual
address space, collective operations (AllReduce / AllGather), and fault tolerance.

---

## Quick start

```bash
git clone --recursive https://github.com/your-org/llama-distributed
cd llama-distributed
./scripts/build.sh          # CPU build (takes ~5 min for llama.cpp)
./scripts/build.sh --cuda   # CUDA build
```

Binaries land in `./build/`.

> **No separate llama.cpp checkout needed.** It ships as a git submodule at
> `third_party/llama.cpp`. A single `--recursive` clone is all that is required.

---

## Running — base pipeline mode

```bash
# 1. Start the coordinator on one machine
./build/dist-coordinator \
  --model  /models/llama-3-70b.gguf \
  --min-nodes 3

# 2. Start a node agent on each GPU server
./build/dist-node --server <coordinator-ip>

# 3. Send a prompt from anywhere
./build/dist-client \
  --server <coordinator-ip> \
  --prompt "Explain ring-AllReduce in one paragraph." \
  --max-tokens 256
```

Tokens stream to stdout as they are generated.

---

## Running — VM / unified-computer mode

VM mode adds distributed tensor memory, collective ops, and checkpointing on
top of the base pipeline.

```bash
# Coordinator with VM layer
./build/dist-vm-coordinator \
  --model  /models/llama-3-70b.gguf \
  --min-nodes 3

# Node agents with VM layer (same flag set as dist-node)
./build/dist-vm-node --server <coordinator-ip>

# Client API (C++ — VmContext)
VmContext ctx({"coordinator-ip"});
ctx.infer(token_ids, 256, [](int32_t tok, bool last){
    std::cout << tok;
});
ctx.checkpoint().wait();   // snapshot entire cluster state
```

---

## Single-machine test (3 processes, localhost)

```bash
# Terminal 1
./build/dist-coordinator --model model.gguf --min-nodes 2

# Terminal 2
./build/dist-node --server 127.0.0.1 --data-port 7701

# Terminal 3
./build/dist-node --server 127.0.0.1 --data-port 7704

# Terminal 4
./build/dist-client --server 127.0.0.1 --prompt "Hello!"
```

---

## Build options

| CMake flag | Default | Description |
|---|---|---|
| `DIST_USE_CUDA` | OFF | CUDA (NVIDIA GPU) acceleration |
| `DIST_USE_METAL` | OFF | Metal (Apple GPU) acceleration |
| `DIST_USE_VULKAN` | OFF | Vulkan acceleration |
| `DIST_LLAMA_SOURCE` | AUTO | `AUTO` = submodule · `SYSTEM` = installed · `/path` = explicit source tree |
| `DIST_BUILD_TESTS` | OFF | Build unit tests |

### Build script shorthand

```bash
./scripts/build.sh [flags]

  --cuda            Enable CUDA
  --metal           Enable Metal
  --vulkan          Enable Vulkan
  --debug           Debug build with symbols
  --jobs N          Parallel build jobs (default: nproc)
  --prefix PATH     Install prefix (default: ./install)
  --llama PATH      External llama.cpp source tree instead of the submodule
```

### Using a pre-installed llama.cpp

```bash
cmake -S . -B build \
  -DDIST_LLAMA_SOURCE=SYSTEM \
  -DLLAMA_INSTALL_PREFIX=/usr/local
cmake --build build
```

### Using a custom llama.cpp fork

```bash
cmake -S . -B build \
  -DDIST_LLAMA_SOURCE=/path/to/my/llama.cpp \
  -DDIST_USE_CUDA=ON
cmake --build build
```

---

## Architecture

```
  Client
    │  TCP :7702  (INFER_REQUEST / INFER_TOKEN)
    ▼
┌──────────────────────────────────────────────┐
│      Coordinator  (or  VmCoordinator)        │
│  ● node registry & heartbeat monitor         │
│  ● VRAM-proportional layer planner           │
│  ● inference request router                  │
│  ● VM: tensor registry, op scheduler,        │
│        collective orchestrator, fault mgr    │
└──────────┬───────────────────────────────────┘
           │ TCP :7700  LAYER_ASSIGN
           │ TCP :7703  VM_CTRL  (VM mode)
     ┌─────┴──────┐
     ▼            ▼
┌─────────┐   ┌─────────┐   ┌─────────┐
│ Node 0  │──▶│ Node 1  │──▶│ Node N  │
│ L0–L19  │   │ L20–L39 │   │ L40–L79 │
│  GPU 0  │   │  GPU 1  │   │  GPU 2  │
└─────────┘   └─────────┘   └─────────┘
  activations flow right  TCP :7701
  collectives flow in a ring  (VM mode)
```

### Layer partitioning

VRAM-proportional: a 40 GB node gets twice as many layers as a 20 GB node.
The first node holds the embedding table; the last node produces logits.

### Double-buffering

- **RX thread** fills a bounded queue with incoming activation tensors.
- **Compute thread** drains it, runs GPU kernels, pushes to the send queue.
- **TX thread** forwards output activations to the next node.

GPU compute and NIC receive overlap completely.

### VM layer — unified-computer mode

| Component | Responsibility |
|---|---|
| `VmTensorRegistry` | 64-bit virtual address → `(node, bytes, dtype)` |
| `VmScheduler` | Data-locality op dispatch with automatic retry |
| `CollectiveEngine` | Ring-AllReduce, AllGather, Broadcast (bandwidth-optimal) |
| `FaultManager` | Checkpoint 2PC; tensor migration on node failure |
| `VmCoordinator` | Owns all VM subsystems; listens on PORT_VM_CTRL 7703 |
| `VmNode` | Local tensor store; executes dispatched ops |
| `VmContext` | Client-facing C++ API |

---

## Wire protocol

All messages are length-prefixed TCP frames:

```
[ MsgHeader (24 bytes) ][ payload (variable) ]

MsgHeader:
  magic       uint32   0xD157C0DE
  version     uint32   1
  msg_type    uint16   MsgType enum
  flags       uint16   reserved
  payload_len uint32   bytes after header
  seq         uint64   per-connection sequence number
```

| Port | Purpose |
|---|---|
| 7700 | Coordinator control — NODE_JOIN, HEARTBEAT, LAYER_ASSIGN |
| 7701 | Data plane — TENSOR_FORWARD (node → node) |
| 7702 | API plane — INFER_REQUEST / INFER_TOKEN / INFER_DONE |
| 7703 | VM control — tensor alloc/write/read, op dispatch, collectives, checkpoints |

---

## Relationship to llama.cpp

`llama-distributed` is an **adapter layer on top of llama.cpp**, not a fork:

- llama.cpp is the computation engine (GGUF loading, GPU kernels, quantisation).
- `NodeAgent` owns a `llama_model*` / `llama_context*` and calls `llama_decode()`
  on its assigned layer slice.
- **Zero modifications** to llama.cpp are made. It is used purely as a library.
- To update llama.cpp: `git -C third_party/llama.cpp pull && cmake --build build`

---

## File structure

```
llama-distributed/
├── CMakeLists.txt                    top-level build (all targets)
├── cmake/
│   └── FindLlamaCpp.cmake            find pre-installed llama.cpp
├── scripts/
│   └── build.sh                      one-command build helper
├── third_party/
│   └── llama.cpp/                    git submodule (ggml-org/llama.cpp)
├── include/
│   ├── dist_protocol.h               base wire format (ports 7700–7702)
│   ├── dist_conn.h                   TCP framing + Listener
│   ├── dist_queue.h                  bounded MPMC queue
│   ├── node_agent.h                  pipeline worker
│   ├── coordinator.h                 cluster coordinator
│   ├── pipeline.h                    layer cache + double-buffering
│   ├── vm_protocol.h                 VM wire format (port 7703)
│   ├── vm_tensor_registry.h          virtual address space
│   ├── vm_scheduler.h                op dispatch + retry
│   ├── vm_collective.h               ring collectives
│   ├── vm_fault.h                    checkpoint + fault recovery
│   ├── vm_coordinator.h              VM coordinator
│   ├── vm_node.h                     VM worker
│   └── vm_context.h                  client API
└── src/
    ├── coordinator.cpp
    ├── node_agent.cpp
    ├── pipeline.cpp
    ├── vm_tensor_registry.cpp
    ├── vm_scheduler.cpp
    ├── vm_collective.cpp
    ├── vm_fault.cpp
    ├── vm_coordinator.cpp
    ├── vm_node.cpp
    ├── vm_context.cpp
    ├── dist_coordinator_main.cpp     → dist-coordinator
    ├── dist_node_main.cpp            → dist-node
    ├── dist_client_main.cpp          → dist-client
    ├── dist_vm_coordinator_main.cpp  → dist-vm-coordinator
    └── dist_vm_node_main.cpp         → dist-vm-node
```

---

## Roadmap

- [ ] Partial model loading — assign weight tensors per node via `tensor_buft_overrides`
- [ ] Pre-connected data sockets — connect node pairs at startup
- [ ] Multi-request interleaving — batched decode across concurrent sessions
- [ ] RDMA transport — ibverbs / UCX for InfiniBand
- [ ] Python bindings for `VmContext`
- [ ] Web dashboard — node status, VRAM, throughput via HTTP/SSE
