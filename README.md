# llama-distributed

Distributed inference platform for running large language models across a pool of GPU servers.

Each server holds a slice of the model's layers. Activation tensors flow through the pipeline stage-by-stage, token by token, allowing models far too large for any single machine to be served collaboratively.

---

## Architecture

```
  Client
    │  (TCP :7702)
    ▼
┌───────────────────────────────────────┐
│           Coordinator                 │
│  - Node registry & heartbeat monitor  │
│  - Layer partitioning planner         │
│  - Inference request router           │
└────────────┬──────────────────────────┘
             │ LAYER_ASSIGN (TCP :7700)
    ┌────────┴────────┐
    ▼                 ▼
┌─────────┐       ┌─────────┐       ┌─────────┐
│  Node 0 │──────▶│  Node 1 │──────▶│  Node N │
│ L0..L19 │  act  │ L20..L39│  act  │ L40..L79│
│  GPU 0  │  ten  │  GPU 1  │  ten  │  GPU 2  │
└─────────┘  sor  └─────────┘  sor  └─────────┘
              (TCP :7701)
```

### Three executables

| Binary | Role |
|---|---|
| `dist-coordinator` | Central daemon. Run one per cluster. |
| `dist-node` | Node agent. Run one per GPU server. |
| `dist-client` | CLI client that sends prompts and streams tokens. |

### Layer partitioning

The coordinator measures each node's total VRAM, then assigns layers proportionally. A node with 40 GB VRAM gets twice as many layers as a node with 20 GB. The first node also handles the embedding table; the last node produces logits.

### Double-buffering

Each node has two concurrency domains:

- **Network RX thread** fills a bounded queue with incoming activation tensors.
- **Compute thread** drains the queue, runs GPU kernels, pushes to send queue.
- **Network TX thread** drains the send queue and forwards to the next node.

While GPU is computing batch N, the NIC is already receiving batch N+1 — zero stall.

### CPU offload (large models)

For models that exceed GPU VRAM, `LayerCache` keeps weights in CPU pinned RAM and streams them to VRAM on demand. A prefetch thread keeps the next N layers warm to hide transfer latency.

---

## Build

### Prerequisites

- CMake ≥ 3.14
- C++17 compiler (GCC ≥ 9, Clang ≥ 10, MSVC 2019)
- llama.cpp source tree (automatically used if sibling directory)

### Minimal CPU-only build

```bash
cd llama-distributed
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### CUDA build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DDIST_USE_CUDA=ON
cmake --build build -j$(nproc)
```

### Custom llama.cpp path

```bash
cmake -B build -DLLAMA_CPP_DIR=/path/to/llama.cpp
```

---

## Running

### 1. Start the coordinator (any machine, reachable by all nodes)

```bash
./build/dist-coordinator \
  --model /data/models/llama3-70b-q4_k_m.gguf \
  --model-name llama3-70b \
  --context 8192 \
  --min-nodes 3
```

The coordinator waits for `--min-nodes` nodes to connect before assigning layers.

### 2. Start node agents (one per GPU server)

On server 1 (192.168.1.10):
```bash
./build/dist-node \
  --server 192.168.1.100 \
  --data-port 7701 \
  --n-gpu-layers 999
```

On server 2 (192.168.1.11):
```bash
./build/dist-node \
  --server 192.168.1.100 \
  --data-port 7701 \
  --n-gpu-layers 999
```

On server 3 (192.168.1.12):
```bash
./build/dist-node \
  --server 192.168.1.100 \
  --data-port 7701 \
  --n-gpu-layers 999
```

Once all 3 nodes connect, the coordinator assigns layer ranges, each node loads its slice of the model, and the pipeline is ready.

### 3. Send inference requests

```bash
./build/dist-client \
  --server 192.168.1.100 \
  --model llama3-70b \
  --prompt "Explain quantum entanglement in simple terms." \
  --max-tokens 512 \
  --temp 0.7
```

Tokens stream to stdout as they are generated.

---

## Single-machine test (3 processes)

For local testing without multiple servers, run coordinator and nodes all on localhost, using different data ports:

```bash
# Terminal 1
./build/dist-coordinator --model model.gguf --model-name test --min-nodes 2

# Terminal 2
./build/dist-node --server 127.0.0.1 --data-port 7701

# Terminal 3
./build/dist-node --server 127.0.0.1 --data-port 7703

# Terminal 4
./build/dist-client --server 127.0.0.1 --model test --prompt "Hello!"
```

---

## Wire protocol

All messages are TCP frames:

```
[ MsgHeader (24 bytes) ][ payload (variable) ]

MsgHeader:
  magic       uint32  0xD157C0DE
  version     uint32  1
  msg_type    uint16  see MsgType enum
  flags       uint16  reserved
  payload_len uint32  bytes after header
  seq         uint64  per-connection sequence number
```

Key message types:

| Type | Direction | Purpose |
|---|---|---|
| `NODE_JOIN` | Node → Coordinator | Registration + capability report |
| `HEARTBEAT` | Node → Coordinator | Liveness + stats |
| `LAYER_ASSIGN` | Coordinator → Node | Layer range + model path |
| `TENSOR_FORWARD` | Node → Node | Activation batch (hidden state) |
| `INFER_REQUEST` | Client → Coordinator | Prompt tokens + generation params |
| `INFER_TOKEN` | Coordinator → Client | One generated token (streaming) |
| `INFER_DONE` | Coordinator → Client | End of generation + stats |

---

## File structure

```
llama-distributed/
├── CMakeLists.txt
├── include/
│   ├── dist_protocol.h    # Wire format, all packed structs
│   ├── dist_conn.h        # TCP connection with framing
│   ├── dist_queue.h       # Bounded MPMC queue
│   ├── node_agent.h       # Node agent declaration
│   ├── coordinator.h      # Coordinator declaration
│   └── pipeline.h         # Pipeline stage + layer cache
└── src/
    ├── node_agent.cpp             # Node agent: load model, run layers, stream tensors
    ├── coordinator.cpp            # Coordinator: plan, route, monitor
    ├── pipeline.cpp               # Double-buffering, layer prefetch, stats
    ├── dist_coordinator_main.cpp  # Coordinator binary entry point
    ├── dist_node_main.cpp         # Node binary entry point
    └── dist_client_main.cpp       # Client binary entry point
```

---

## Limitations and roadmap

### Current state
- Pipeline plumbing, protocol, and threading model are complete.
- Each node loads the **full model** — true layer-slice loading requires extracting weight tensors by layer range from the GGUF file, which is a straightforward but time-consuming extension.
- Activation hand-off between nodes uses `llama_get_embeddings_ith()` (the hidden state after the last layer computed). For true mid-model slicing, intercept at the `ggml_backend_sched` graph split boundary — this is being developed upstream (see llama.cpp PR #19378).

### Next steps
1. **Partial model loading** — use `llama_model_params.tensor_buft_overrides` to keep only assigned layers in VRAM; the rest stays on CPU or is unmapped.
2. **Persistent data connections** — pre-connect each node to its downstream neighbour at startup rather than per-request.
3. **Multi-request pipelining** — multiple in-flight requests interleaved across the same pipeline (batched decode).
4. **RDMA transport** — replace TCP with ibverbs/UCX for sub-microsecond tensor transfer on InfiniBand clusters.
5. **Fault recovery** — reassign layers from a dead node to remaining nodes.
6. **Web dashboard** — expose coordinator metrics (node status, VRAM, throughput) via HTTP/SSE.
