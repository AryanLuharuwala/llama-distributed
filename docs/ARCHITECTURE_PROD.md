# SURD — production-grade architecture review

A read of the current stack from the perspective of Google-scale operating
practice. The system today is a *good* startup-quality monolith: a single
Go binary, SQLite, hand-rolled WS framing, hand-rolled Prometheus text,
hand-rolled HMAC URL signing, pion-based TURN. That is the right way to
ship v1. This doc captures the upgrade path when traffic and SLOs justify
the swap.

The lens for every recommendation is one of two questions:

  1. Is there a *battle-tested library* that does this with fewer bugs
     per LOC than ours?
  2. Is there a way to **process fewer bits** of data — at the wire, in
     RAM, on disk, or per request?

---

## 1. Wire protocol

### Today
* JSON over WebSocket for the agent control plane (`ws.go`).
* Custom binary framing for ACTV/INFR/LAT activations (`activation.go`).
* JSON for OAI request bodies; SSE for streaming.
* `text/event-stream` for dashboard live data.

### Production
| Concern | Current | Production-grade choice | Why |
|---|---|---|---|
| Schema | ad-hoc JSON | **protobuf** (gRPC) | Forward/backward compatibility, field-level reservations, codegen across Go/C++/Py, ~3-10× smaller wire than JSON. |
| Transport | WS + custom binary | **gRPC over HTTP/2 + QUIC fallback** | Cancellation, deadlines, flow control, multiplexing, mTLS — all out of the box. Removes hand-rolled framing in `inference.go::encodeChunk`. |
| Streaming | SSE | **gRPC server-streaming + WebTransport for browsers** | Backpressure, HTTP/2 trailers for token counts, no parse-and-split. |
| Activation tensors | FP16 bin frames | **FP8 (E4M3 or E5M2) on the wire, dequant at consumer** | 2× bandwidth cut on the ACTV path. NVIDIA Hopper natively, Ada via marlin kernels. |
| Compression | none | **zstd (level 1) on JSON control frames; LZ4 on activations only if CPU has headroom** | Activations compress poorly post-quantization; control frames compress ~5×. |
| Delta encoding | none | **KV-cache delta on subsequent tokens** | Pipeline parallel currently re-sends partial state per token; FlashAttention-style cache delta drops resends to O(1 token). |

### Reducing every bit
* **Speculative decoding** — server-side draft model emits 4-8 tokens at a time, big model verifies. ~2.5× throughput at the same quality. Pair with **eagle-3** or **medusa** heads on the dispatched model.
* **Continuous batching (already scaffolded)** + **paged attention** (vLLM-style) — eliminates padding waste; the existing batching scaffold should adopt PagedKV semantics rather than reinventing them.
* **Activation checkpointing on the wire** for >2-stage PP — recompute on the consumer for layers whose tensor < recomputation_cost / bandwidth.
* **Prompt cache** — Anthropic / OpenAI both ship this: hash the request prefix, dedup at the rig level, return cache-hit tokens at 10× cheaper. Implement as a separate cache key on `inference_log` and a `prompt_cache_hits_total` Prometheus counter.
* **Embedding-only short path** — for cheap requests (classification, similarity), route to a 7B distillation; only escalate to 70B when downstream needs it. Saves ~10× tokens billed on the easy 80% of traffic.

---

## 2. Identity, authn, authz

### Today
* GitHub OAuth, dev-mode shortcut, device-code, session cookies.
* Bearer `sk-…` API keys with sha256 hash in DB.
* `agent_key` stored plaintext (see audit A1).
* Hand-rolled ed25519 challenge in `agent_identity.go`.

### Production
* **SPIFFE / SPIRE for workload identity** — rigs become first-class workloads with rotating X.509 SVIDs. Replaces `agent_key` + the bespoke ed25519 challenge. Free mTLS to everything.
* **OIDC** (Google/GitHub/Okta) via **dex** or **ory/hydra** — drops the hand-rolled cookie/session code. Sessions become JWTs verified by JWKS, no DB lookup on hot path.
* **Macaroons or biscuit** for delegated capabilities (e.g., signed comfy URLs in A11) — proper attenuation, third-party caveats, no per-resource HMAC roll.
* **WebAuthn / passkeys** as the human auth path — kills the password + dev-mode + device-code surface entirely for the dashboard.

### Reducing every bit
* JWTs (vs session-cookie + DB lookup) save **one DB round-trip per request**. At 100 RPS that's 8.6M reads/day eliminated.
* SPIFFE auto-rotated short-lived certs make `pair_tokens`, `device_codes`, `agent_key` tables disappear — entire schema slabs go away.

---

## 3. State management

### Today
* SQLite, single-writer, WAL.
* Hand-rolled migrations via `addColumnIfMissing` (`db.go:11`).
* 2-second per-pool cost cache (`cost_cache.go`).
* All state — users, pools, sessions, rigs, reputation, inference_log, comfy_jobs, hf_imports — in one file.

### Production
| Concern | Choice | Why |
|---|---|---|
| OLTP | **Spanner** (or **CockroachDB** if portable) | Global consistency, horizontal scale, no manual sharding. Replaces SQLite + the eventual "shard SQLite by pool" hack. |
| Time-series | **BigQuery for cold inference_log; Bigtable for hot per-user counters** | `inference_log` grows unbounded; partitioned BQ tables are the right fit. `usage_counters` is hot-path increment — Bigtable / DynamoDB-style counter rows. |
| Cache | **Memcache (mcrouter) + single-flight** (`golang.org/x/sync/singleflight`) | Replaces hand-rolled `rigCostCache`. Survives process restart, scales horizontally, supports cross-instance coalescing. |
| Migrations | **ent + atlas** or **goose** | Declarative schema, diff-based migration, signed migration files. Kills the `duplicate column name` string-match hack. |
| Sessions | **Redis with EXPIRE** | No more `sessions` table sweep; TTL handled by Redis. |
| Models metadata | **GCS + content-addressed (sha256) URIs** | `models-store/` + `shards_dir` lookups disappear; rigs fetch by hash, cache-by-content. |

### Reducing every bit
* **Push down filters**: replace `SELECT * FROM rig_reputation WHERE agent_id IN (...)` with stored procedures / prepared statements that return only `(agent_id, score)` — saves the 5 unused columns × N rows per planner round.
* **Columnar `inference_log`**: BigQuery + clustering on `(pool_id, started_at)` makes the dashboard 24h query an order of magnitude faster than the SQLite scan.
* **Write coalescing**: today `last_seen` is batched (`server.go::flushLastSeen` every 1s) — extend the same pattern to `usage_counters`, `relay_sessions_total`. Drops write QPS by 10-100×.
* **Drop unused indexes**: `idx_inflog_user (user_id, started_at)` + `idx_inflog_agent` — pick one based on real query patterns; SQLite indexes are not free on writes.

---

## 4. Networking & relay plane

### Today
* `dist-turn` sidecar (own pion/turn build) — solid.
* Peer-as-TURN forwarding with self-reported NAT type (`pp_route.go`).
* Hand-rolled reputation scoring (`reputation.go::relayScore`).
* libdatachannel for WebRTC in C++ agent.

### Production
* **Real coturn deployment** at the edges (anycast IPs) with `dist-turn` as the fallback for residential rigs. coturn has 10 years of TURN spec compliance work the pion stack doesn't yet match.
* **gRPC-LB + xDS** for routing inference requests to the right pool/rig — replaces the hand-rolled subdomain → pool resolver in `openai.go::resolvePoolFromHost`.
* **HAProxy or Envoy** in front of the Go binary for TLS termination, request shaping, rate limit, observability. The Go binary stops reading X-Forwarded-For directly (closes the documented `[[xff-trust-known-gap]]`).
* **MASQUE / CONNECT-UDP** for UDP egress through carrier-grade NATs — better than TURN allocations for ~20% of consumer environments.

### Reducing every bit
* **NAT punching via PCP / NAT-PMP** before falling back to TURN — saves 100% of relayed bytes on the 60% of consumer connections that support it. Add to `nat-probe`.
* **Thompson-sampled bandit** for relay rig selection instead of the current Bayesian-blended score. Provably regret-optimal; ~30% fewer bad-relay assignments in steady state.
* **ECN + L4S** for activation tensors — congestion-aware without packet loss. Recent kernels (≥5.18) support it natively.

---

## 5. Observability

### Today
* Hand-rolled Prometheus text in `metrics.go`.
* Request-ID middleware via `context.Context`.
* `DIST_LOG_JSON=1` toggle for structured logs.

### Production
* **OpenTelemetry** (OTLP) for *all three* of metrics, traces, logs. Drops the hand-rolled exposition format entirely; gets distributed traces (request → planner → ws frame → rig → response) for free.
* **Prometheus client_golang** for in-process metric definitions even if exporting via OTel — typed Counter/Gauge/Histogram catches the "gauge that should be a counter" class of bug at compile time.
* **eBPF profiling** (`profiles.proto`) — continuous, low-overhead. Replaces "ssh in and run pprof" on production rigs.
* **Sentry** or equivalent for panic capture with source-map'd stacks across Go ↔ C++ ↔ Python.

### Reducing every bit
* **Tail-based sampling** for traces — keep 100% of 5xx + slow, 1% of OK. Cuts trace storage by 100×.
* **Exemplars** on histograms — link `request_duration_seconds_bucket` to the specific trace ID that landed in that bucket. Saves hours of debugging without storing every span.

---

## 6. Build, deploy, supply chain

### Today
* `./scripts/build.sh`, manual installer scripts.
* `dist-turn` cross-compile matrix is hand-maintained.

### Production
* **Bazel** for hermetic builds across Go + C++ + Python. Reproducible binaries with provenance (SLSA L3).
* **rules_oci** + distroless base images.
* **goreleaser** for the dist-turn / surd CLI matrix.
* **Sigstore (cosign)** signatures + transparency log for every binary on the install one-liner.
* **GitOps**: Argo CD or Flux pointing at a Kustomize/Helm tree. The current systemd / launchd / schtasks installers are fine for end-user rigs; control plane goes onto k8s.

---

## 7. Libraries to swap in (concrete)

| Hand-rolled today | Battle-tested replacement |
|---|---|
| `rigCostCache` singleflight | `golang.org/x/sync/singleflight` |
| `rateLimiter` (`ratelimit.go`) | `golang.org/x/time/rate` (token bucket) for per-IP; **redis_rate** for cross-instance |
| HMAC URL signing | **macaroons** (capnproto/go-macaroon) |
| ed25519 nonce challenge | **SPIFFE workload API** |
| `addColumnIfMissing` string match | **atlas** migrations |
| `metrics.go` Prom text | **prometheus/client_golang** + OTel SDK |
| Custom JSON WS frames | **connectrpc.com/connect** (gRPC over HTTP/1.1 friendly) |
| `clientIP()` X-Forwarded-For | **realip** behind a known proxy chain |
| `formatLabels` | (free with client_golang) |
| `sqliteDialect` / `postgresDialect` | **ent** + **atlas** — generated, no manual rewriter |
| `newRandomToken` | (keep; this one is fine) |
| `controlPIDPath` + signal-0 alive check | **systemd notify + `Type=notify`** on Linux; equivalent on launchd |

Total LOC removed from our tree: rough estimate **2,500-4,000 lines**. Total dependency surface added: ~25 libraries, all in heavy production use.

---

## 8. Reducing every bit — a single checklist

Order by ROI:

1. **Prompt cache** (Anthropic-style) — easiest 5-10× cost cut on repeat traffic.
2. **FP8 activations on the wire** — 2× bandwidth, same quality on Hopper/Ada.
3. **Continuous batching + paged attention** — 3-5× throughput at same VRAM.
4. **Speculative decoding** — 2-3× tokens/sec on chat workloads.
5. **NAT-PMP / PCP before TURN** — kills 60% of relayed bytes.
6. **JWT sessions** — kills the per-request DB round-trip.
7. **protobuf control plane** — 3-10× smaller WS frames.
8. **zstd on control-plane JSON** — additional 5× on the remainder.
9. **Write coalescing on hot counters** — 10-100× fewer DB writes.
10. **Tail-sampled traces + columnar inference_log** — 100× observability cost cut.

Multiplicatively the realistic floor for "bits processed per useful inference token" looks like:

```
current:  ~3000 bytes / generated token (control + activations + storage IO)
target :  ~80 bytes / generated token (after FP8 + prompt cache + continuous batching + protobuf)
```

A ~40× efficiency improvement at the wire and DB layer is on the table without changing what the user can ask the system to do.

---

## 9. Not on this list (deliberately)

* **Rewriting in Rust.** Go is fine; the bottlenecks are protocol and data layout, not the language.
* **Custom CUDA kernels.** Use vLLM / TensorRT-LLM / SGLang. Anything we wrote ourselves would be a year behind.
* **Multi-region active-active control plane.** Premature until pool count is in the thousands; Spanner handles it when we get there.
* **Blockchain anything.** No.

---

*This document captures the v∞ target. None of it is a v1 requirement; the
current architecture is correct for the present scale. Use this as the
North Star when picking which piece of hand-rolled code is worth
replacing this quarter.*


## 10. MCP servers + RAG (added 2026-05-23)

The system grows two new control-plane subsystems that ride on top of the
gRPC tool-call channel (P4) and the embedding tier on the rig fleet (R6).
Both are user-scoped and dialect-portable — schemas live in `mcp.go` and
`rag.go`, gated by `migrateMCP` / `migrateRAG`.

### 10.1 MCP server registry

Users plug in MCP (Model Context Protocol) servers that the inference
path can call when the model emits a `tool_use` frame. The registry
table is:

    mcp_servers(id, user_id, name, transport, endpoint, scopes_json,
                secret_ref, enabled, last_health_at, last_health_ok,
                created_at, updated_at)
    UNIQUE(user_id, name)

    mcp_calls(id, user_id, server_id, tool, args_sha256, size_bytes,
              success, error_class, latency_ms, called_at)

Transports: `stdio` (subprocess sandboxed under bwrap + landlock with a
per-user uid), `http+sse` (https or http to an allow-listed host),
`ws`/`wss`. Endpoints are validated at register-time; stdio commands
must use absolute paths, and HTTP/WS endpoints must clear the same
`isAllowedPublicIP` check the relay broker uses (no loopback, no link-
local, no RFC1918 from untrusted users).

Audit log only persists `sha256(args)` and a size — never the plaintext
arguments — so a user piping an OAuth token through a downstream API
doesn't leak it into our DB. Tool name and call timing are logged for
billing and forensics.

The broker (R2, separate file `mcp_broker.go`) holds one persistent
connection per `(user_id, server_id)` while the user has an active
session, drops idle connections after 5 min, and enforces a per-server
rate limit + payload cap drawn from the scopes JSON.

### 10.2 Tool-call channel on the inference protocol (R3)

The protobuf message family defined in P4 grows a bidirectional
`ToolCall` sub-stream that piggybacks on the existing ACTV channel:

    message ToolCallRequest  { string call_id = 1; string server_name = 2;
                                string tool = 3; bytes args = 4; }
    message ToolCallResponse { string call_id = 1; oneof result {
                                bytes ok = 2; ToolCallError err = 3; } }

The rig emits `ToolCallRequest`, the control-plane resolves the named
server via the registry, calls through the broker, and returns the
response inline. Token cost of the call is accounted against the
session's slot the same way prompt/output tokens are.

### 10.3 RAG storage

Three tables, all dialect-portable:

    rag_collections(id, user_id, name, embedding_model, embedding_dim,
                    documents_count, chunks_count, created_at, updated_at)
    rag_documents(id, collection_id, user_id, uri, content_sha,
                  mime_type, size_bytes, chunk_count, created_at)
    rag_chunks(id, collection_id, document_id, ordinal_idx,
               text, token_count, embedding BLOB)

Embeddings are packed little-endian `float32` in BLOB (BYTEA on
Postgres). On the SQLite path we score in Go with a min-heap; on the
Postgres path we alter `embedding` to `vector(N)` once and add an
`ivfflat` or `hnsw` index — the migration script ships separately so
SQLite tests don't depend on pgvector.

### 10.4 Hybrid retrieval (R5)

`reciprocal_rank_fusion(dense_topk, bm25_topk)` produces the candidate
set. Dense scoring uses the collection's pinned embedding model
(BGE-large by default). BM25 uses FTS5 on SQLite or `tsvector` on
Postgres. An optional rerank stage calls a cross-encoder
(`bge-reranker-v2-m3`) served on the same embedding tier as `/v1/rerank`.

### 10.5 Embedding tier (R6)

Rigs advertise `capability=embed` in the hello frame when they load an
embedding model class (BGE, E5, GTE, bge-reranker). The planner routes
`/v1/embeddings` and `/v1/rerank` to embedding-capable rigs and
treats them as a separate slot pool from chat-completion slots — they
serve short bursty workloads with very different cache behaviour than
generative inference.

### 10.6 Conversation memory (R7)

Every N turns the planner summarises the trailing window via the same
model serving the chat, embeds the summary, and writes it to a
per-conversation collection. On the next turn the planner injects the
top-k retrieved summaries into the system prompt. This compresses long
chats without losing facts, and reuses the entire RAG stack — no new
storage code, just a different collection naming convention.

### 10.7 Where MCP + RAG cost bits

For a 5k-token prompt that pulls 4 RAG chunks (~512 tokens each) and
makes 1 MCP `read_file` call:

| Step | Bits on wire | Notes |
|---|---|---|
| Embed query (768-dim, fp16) | ~12 KB request, 1.5 KB response | dedicated embedding rig |
| Retrieve top-k from pgvector | ~2 KB request, ~4 KB metadata | server-local |
| Rerank candidates (top-32 → top-4) | ~64 KB | cross-encoder over candidates |
| MCP tool_use round-trip | ~8 KB request, ~16 KB response | proxied through control-plane |
| Augmented prompt → rig (with prompt cache) | ~3 KB (delta only) | SGLang RadixAttention hit |

Without prompt caching that augmented prompt is ~10 KB on the wire.
With caching (P13 + R7's prefix-stable system prompt) it drops to the
delta. The retrieved chunks themselves are the only payload that grows
linearly with `top_k`, which is why R5 caps `top_k_context` at the
collection level rather than the request level.
