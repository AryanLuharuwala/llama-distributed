# Deploying dist-server (prod branch)

This branch ships the topology the `main` branch defers: front-proxy
+ trusted X-Forwarded-For chain. The `main` branch runs as a single
container with no proxy in front; on `prod` you put envoy at the edge
and dist-server sits behind it.

Why bother? Without a front-proxy, `X-Forwarded-For` is either ignored
(no source-IP attribution past one NAT hop) or trusted blindly (any
client can spoof its rate-limit key, audit log, and device-code
ratchet). Envoy resolves the question by being the only thing that
ever appends to XFF, and dist-server enforces that with
`DIST_TRUSTED_PROXIES`.

## Files in this directory

| Path | What it is |
| --- | --- |
| `envoy/envoy.yaml` | Production envoy config — TLS terminate on :443, h1/h2 ALPN, WebSocket + SSE upgrade support, h1 to upstream. |
| `otel/collector.yaml` | OpenTelemetry Collector (Contrib) — receives OTLP from dist-server and fans out to your tracing/metrics backend of choice. |
| `docker-compose.envoy.yaml` | Four-service stack (envoy + dist-server + redis + otel-collector) with the right network isolation and trust set. |
| `README.md` | This file. |

## Five-minute deploy

```bash
# 1. Get a TLS cert.  Anything that drops fullchain.pem + privkey.pem in
#    a directory works.  Letsencrypt via certbot is the usual path:
sudo certbot certonly --standalone -d your.domain

# 2. Point compose at the cert dir + your secrets:
cat > deploy/.env <<EOF
DIST_PUBLIC_URL=https://your.domain
DIST_SESSION_SECRET=$(openssl rand -hex 32)
TLS_DIR=/etc/letsencrypt/live/your.domain
EOF

# 3. Launch:
docker compose --env-file deploy/.env -f deploy/docker-compose.envoy.yaml up -d

# 4. Sanity check:
curl -sf https://your.domain/api/health    # ⇒ {"ok":true,...}
curl -sf https://your.domain/api/meta      # ⇒ build/host/env
```

If the second curl shows your real IP in the dist-server logs — not
`172.x.x.x` (the envoy egress) — the trust chain is working.

## DIST_TRUSTED_PROXIES — the one knob that matters

`DIST_TRUSTED_PROXIES` is a comma-separated CIDR list. Dist-server
honors `X-Forwarded-For` only for connections whose TCP peer falls in
this set, and walks the header right-to-left to extract the real
client.

Three topologies, three settings:

| Topology | Set it to |
| --- | --- |
| envoy on same host as dist-server, compose default bridge | `172.16.0.0/12,127.0.0.1/32` |
| envoy on a dedicated VM, dist-server on another VM in the same VPC | the envoy VM's CIDR, e.g. `10.0.1.0/24` |
| envoy + Cloudflare in front | envoy CIDR **plus** Cloudflare's edge ranges (see [Cloudflare IPs](https://www.cloudflare.com/ips/)) — and set `xff_num_trusted_hops: 1` in envoy.yaml |

**Empty `DIST_TRUSTED_PROXIES`** → XFF is ignored; rate limits and
audit logs key off `r.RemoteAddr`. Safe but useless once you're behind
a real proxy.

## DIST_REDIS_URL — the cross-instance rate-limit backend

`DIST_REDIS_URL` (e.g. `redis://redis:6379/0`) makes the per-IP token
buckets for device-code approve, device-code poll, OAuth start, and WS
hello-fail retries shared across replicas. Without it each replica has
its own in-memory `sync.Map` and an attacker behind a CDN gets N× the
configured budget when you scale to N pods.

The bundled compose file ships a sidecar Redis with AOF
(`appendfsync everysec`, 256 MB cap, allkeys-LRU eviction). For real
multi-region deploys, point `DIST_REDIS_URL` at a managed Redis cluster
(Elasticache, Memorystore, Upstash) and drop the sidecar — the dialer
honors `redis://`, `rediss://`, and `unix://` schemes via go-redis URL
parsing.

**On failure:** the limiter logs once per second and fails open (allows
the request). The reasoning: a Redis outage shouldn't be a DoS
amplifier. If you'd rather fail closed, that's a one-line change in
`server/ip_ratelimit.go` (`allow()` would return `ok` from the backend
directly without the err-suppression).

Sessions and device codes are already DB-backed in `dist-server`
(SQLite or Postgres via `DIST_DB_DRIVER=postgres`), so the only state
that ever needed Redis is the rate-limit token buckets. A horizontally
scaled deploy looks like:

```
envoy ─┬─> dist-server (replica 1) ─┐
       ├─> dist-server (replica 2) ─┼─> postgres   (sessions, device codes, audit)
       └─> dist-server (replica N) ─┤
                                    └─> redis      (rate-limit buckets)
```

## Schema migrations — versioned files

Two-layer migration story:

1. **Legacy baseline** (`server/db.go` + the `migrate*()` siblings).
   Idempotent `CREATE TABLE IF NOT EXISTS` statements; run on every
   boot to handle the union of all tables that existed before
   versioning. Existing deploys are already at the baseline.
2. **Versioned files** (`server/migrations/*.sql`). All new schema
   changes — `ALTER TABLE`, `CREATE INDEX`, new tables — go here as
   atlas-named SQL files. Applied in lexical order, each once per DB,
   tracked in the `schema_migrations` table.

Authoring a new migration:

```bash
# Either hand-write it:
cat > server/migrations/20260525120000_add_foo_index.sql <<EOF
CREATE INDEX idx_rigs_pool_id ON rigs(pool_id);
EOF

# Or let atlas generate it from a schema diff:
atlas migrate diff add_foo_index \
  --env prod \
  --to "postgres://reader:pw@host/dbname?sslmode=disable" \
  --dev-url "docker://postgres/16/dev"
```

Either way, commit the file and redeploy. `dist-server` applies it on
the next boot and refuses to start if the SQL fails — partial state is
rolled back. Concurrent boots are safe: the `UNIQUE(version)` on
`schema_migrations` breaks the race.

The `atlas.hcl` at the repo root defines `prod` and `sqlite`
environments for the operator's local atlas CLI. The runtime applier
in `server/schema_migrations.go` is ~150 lines and ships in the binary
— no atlas runtime dependency.

## OTEL_EXPORTER_OTLP_ENDPOINT — traces + metrics out

dist-server ships an OpenTelemetry SDK that activates the moment
`OTEL_EXPORTER_OTLP_ENDPOINT` is set. The bundled compose points it at
the included `otel-collector` sidecar (`http://otel-collector:4318`),
which by default just logs everything to stdout. Edit
`deploy/otel/collector.yaml` to fan out to your real backend — there
are commented-out exporter stanzas for Tempo (traces),
Prometheus (metrics), and Loki (logs).

What you get out of the box:

- **Traces**: every HTTP request via `otelhttp` middleware
  (method + path span, status code, duration). 10% sampling by default;
  override with `OTEL_TRACES_SAMPLER_ARG`.
- **Metrics**:
  - `http.server.duration` / `http.server.active_requests` — from
    `otelhttp`.
  - `dist.ratelimit.allow` / `.deny` / `.error` — per (bucket, backend)
    so you can graph "device-poll denies per minute" or "redis backend
    fail-open events".
  - `dist.ws.active` — live WebSocket count, labeled `kind=agent|browser`.
  - `dist.inference.requests` — settled inferences by (model, status).
- **Resource attributes**: `service.name`, `service.version`,
  `service.instance.id` (hostname), `deployment.environment`. Override
  via `DIST_OTEL_SERVICE` / `DIST_OTEL_VERSION` / `DIST_OTEL_ENV`.

Unset `OTEL_EXPORTER_OTLP_ENDPOINT` to disable telemetry entirely —
every instrument call becomes a no-op and no exporter goroutines run.

## /ws/agent — proto subprotocol (distpool.proto.v1)

Rigs that ship the proto wire opt in by sending
`Sec-WebSocket-Protocol: distpool.proto.v1` on the WS handshake.  The
server lists it in its `Subprotocols` and echoes the token back on the
101 response per RFC 6455 §4.2.2.  When the negotiation succeeds the
control frames switch from JSON-over-TEXT to length-delimited proto
(`distpool.ctrl.v1`) carried in WebSocket BINARY frames; ACTV relay
bytes stay on BINARY too, distinguished by the fact that they don't
parse as a ClientFrame.

Legacy rigs send no subprotocol header and stay on JSON — byte-for-byte
identical to the pre-P4 wire.  There is no flag day; rigs roll over
independently.

Schemas live in `proto/distpool/ctrl/v1/ctrl.proto`; regenerate Go
bindings with `buf generate` (`buf.gen.yaml` writes into
`server/ctrlpb/`).  Frames whose payloads are still evolving
(`comfy_result`, `comfy_caps`, `p2p_signal`) ride inside the proto as
opaque JSON bytes so the server doesn't need a re-deploy every time a
new field is added.

## DIST_OIDC_* — federation via dex / hydra / Okta / Auth0

Setting `DIST_OIDC_ISSUER` swaps the built-in GitHub/Google OAuth pair
out for a real OIDC OP. The server runs discovery at boot
(`/.well-known/openid-configuration`), caches the JWKS, and exposes:

- `/auth/oidc` — browser code flow (PKCE S256, state HMAC'd to the
  session secret like the GitHub flow).
- `/auth/oidc/callback` — exchanges code → id_token, verifies signature
  + audience + expiry, upserts a row keyed on (issuer, sub), mints the
  session cookie.
- `Authorization: Bearer <jwt>` on any `/api` endpoint — the bearer
  middleware shape-checks the token (three base64url segments) before
  hitting the verifier, so `sk-...` api keys still take the cheap path.

The four env vars:

| Var | What it is |
| --- | --- |
| `DIST_OIDC_ISSUER` | OP issuer URL, e.g. `https://dex.example.com`. Empty disables OIDC. |
| `DIST_OIDC_CLIENT_ID` | Confidential client ID registered with the OP. |
| `DIST_OIDC_CLIENT_SECRET` | Paired secret. |
| `DIST_OIDC_BEARER_AUDIENCE` | Optional. Expected `aud` claim on JWT bearers. Defaults to `DIST_OIDC_CLIENT_ID`. |
| `DIST_OIDC_LABEL` | Optional. Label rendered on the /auth page button. Defaults to `SSO`. |

Discovery failures at boot log once and continue — `/auth/oidc*` returns
501 in that mode so the operator can correct the issuer URL and restart
without crash-looping the deploy. GitHub and Google OAuth paths are
untouched and keep working alongside OIDC for the duration of any
migration.

The `users` table gets two columns from the
`20260524000002_oidc_users.sql` migration: `oidc_issuer` and
`oidc_subject`, both TEXT, joined by a partial UNIQUE index. Legacy
GitHub/Google rows leave both columns NULL; the partial index skips
NULL pairs so the dual-mode table doesn't collide.

## DIST_SPIFFE_* — workload identity for rigs

When set, rigs can authenticate via a SPIFFE SVID instead of (or in
addition to) the paired `agent_key`. Two SVID flavours are accepted:

| Flavour | Path | What you wire |
| --- | --- | --- |
| JWT-SVID | `/ws/agent` hello frame, `spiffe_token` field | `DIST_SPIFFE_JWKS_URL` points at your SPIRE server's OIDC discovery provider; dist-server verifies + refreshes the bundle every 10 min. |
| X.509-SVID | envoy XFCC header | Envoy terminates mTLS, validates the client cert against the SPIRE bundle, then forwards the verified cert metadata in `x-forwarded-client-cert`. Dist-server only trusts XFCC when the TCP peer is in `DIST_TRUSTED_PROXIES`. |

Env vars:

| Var | What it is |
| --- | --- |
| `DIST_SPIFFE_TRUST_DOMAIN` | Required to enable. The trust domain the server accepts SVIDs for, e.g. `prod.example.com`. SVIDs from other trust domains are rejected (no federation in this revision). |
| `DIST_SPIFFE_JWKS_URL` | The SPIRE OIDC discovery provider's JWKS endpoint, e.g. `https://spire-oidc.example.com/keys`. Required for JWT-SVID; X.509-SVID via XFCC works without it. |
| `DIST_SPIFFE_AUDIENCE` | Expected `aud` claim on JWT-SVIDs. Defaults to `DIST_PUBLIC_URL`. |

Current revision is **additive**: a verified SVID gets recorded on the
rig row (`rigs.spiffe_id`) alongside the existing `agent_key`. The
agent_key path is unchanged; it remains the load-bearing credential
during the SPIRE rollout. A follow-up commit can flip the dependency so
SVID alone is sufficient once the fleet has rotated through.

The `rigs` table gets a `spiffe_id TEXT` column from
`20260524000003_rig_spiffe.sql` with a partial UNIQUE index — same
pattern as the OIDC users table.

### Envoy mTLS pass-through (X.509-SVID)

To use the X.509 path, point envoy at your SPIRE trust bundle and have
it forward the verified cert metadata:

```yaml
# envoy.yaml, inside the https listener's downstream TLS context:
require_client_certificate: true
common_tls_context:
  validation_context:
    trusted_ca: { filename: /etc/envoy/spire-bundle.pem }

# inside http_connection_manager:
forward_client_cert_details: SANITIZE_SET
set_current_client_cert_details:
  uri: true
  subject: true
```

That config makes envoy:

1. Require a client cert on the mTLS handshake.
2. Validate it against the SPIRE bundle.
3. Sanitize any inbound XFCC, then append its own `x-forwarded-client-cert`
   with `URI=spiffe://...` populated from the cert's SAN.

`SANITIZE_SET` (not `APPEND_FORWARD`) is important — it stops an
upstream proxy from injecting a forged XFCC and laundering it through us.

## WebSocket + SSE caveats

The dashboard and every rig hold long-lived connections:

- `/ws/agent` — bidirectional rig ↔ server, lasts as long as the rig
  is online (hours to weeks).
- `/api/me/rigs/stream` — SSE push of fleet updates to the browser,
  reconnects every few seconds if dropped.

Envoy's `idle_timeout: 600s` on both the connection manager and the
upstream cluster lets these survive multi-minute silences. If you put
another layer in front of envoy (e.g. a cloud load balancer) confirm
**its** idle timeout is ≥ 10 minutes too — most cloud LBs default to
60s and will kill the rig WS every minute, manifesting as endless
reconnect storms on the rig side.

## Upgrading from the single-container topology

If you were running the `main`-branch image directly with `docker run
-p 443:8080 …`, the migration is:

1. Stop the running container; the SQLite db on the bind-mount carries
   over.
2. Move the TLS work from whatever was doing it (caddy, traefik,
   manual) into `envoy/envoy.yaml`.
3. Set `DIST_TRUSTED_PROXIES` to the envoy CIDR.
4. `docker compose up -d` with the new file.

The wire protocol is unchanged. Rigs reconnect to the same hostname,
authenticate with their stored agent keys, and resume serving.

## Cross-driver SQL (P10)

The control-plane DB layer speaks SQLite (default), Postgres, CockroachDB
(via the Postgres wire protocol), and Cloud Spanner (via the PGAdapter
sidecar).  Driver selection is via `DIST_DB_DRIVER` + `DIST_DB_DSN`:

| Backend     | `DIST_DB_DRIVER`     | DSN example                                                     |
|-------------|----------------------|-----------------------------------------------------------------|
| SQLite      | `sqlite3` (default)  | `file:state/distpool.sqlite?_journal_mode=WAL`                  |
| Postgres    | `postgres`           | `postgres://user:pass@host:5432/distpool?sslmode=require`       |
| CockroachDB | `cockroach`          | `postgres://user:pass@host:26257/distpool?sslmode=verify-full`  |
| Spanner     | `spanner`            | `postgres://localhost:5432/<project>/<instance>/<database>` (PGAdapter) |

How the cross-driver machinery works:

- **`server/dialect.go`** — DDL and query rewriters per dialect.  DDL
  rewrite handles SQLite-isms (`INTEGER PRIMARY KEY AUTOINCREMENT` →
  `BIGSERIAL PRIMARY KEY`, `BLOB` → `BYTEA`).  Query rewrite turns `?`
  placeholders into `$1, $2, …` for Postgres-family drivers, respecting
  single-quoted string literals.

- **`server/dialect_wrap.go`** — `*server` methods (`s.dbExec`,
  `s.dbQuery`, `s.dbQueryRow`, plus `Ctx` variants, plus `s.txExec` /
  `s.txQueryRow`) auto-apply `RewriteQuery` so callers can keep using
  `?` placeholders everywhere.  These are no-ops on SQLite.

- **Serializable retry** (`s.dbDoTx`) — Cockroach (and Spanner) can fail
  any transaction with SQLSTATE 40001 / `transaction aborted` under
  contention.  `dbDoTx` runs a closure inside `Begin/Commit` with a
  bounded retry loop (4 attempts, 50ms × n backoff) on retryable errors.
  SQLite never returns 40001 so the retry loop is a no-op for the
  default deploy.

- **Schema migrations** (`server/schema_migrations.go`) — embedded
  `.sql` files under `migrations/` are run through `RewriteDDL` before
  application, so the same file works on SQLite and Postgres.  Spanner
  is supported via the same path but the production recipe is to
  author hand-tuned migrations under `migrations/spanner/` (Spanner
  has no AUTOINCREMENT — PKs use UUIDs or bit-reverse sequences).

What's deferred:

- **Helm chart** — package the (envoy + dist-server + redis + postgres)
  topology for k8s with proper PDBs, HPA, and a sidecar Redis or
  Memorystore reference.

## Analytics warehouse + hot counters (P11)

The control plane fans out every inference event to an analytics sink
that streams to BigQuery, and exposes a pluggable hot-counter store
that can be swapped from SQLite to Bigtable without touching the
inference path.

Both are opt-in: with no env, the binary is a single-container deploy
that writes to local SQLite as always.

### BigQuery streaming (`DIST_BQ_*`)

| Env                   | Required | Default          | What                                |
|-----------------------|----------|------------------|-------------------------------------|
| `DIST_BQ_PROJECT`     | yes      | (sink disabled)  | GCP project hosting the dataset.    |
| `DIST_BQ_DATASET`     | yes      | (sink disabled)  | BQ dataset (must exist + partitioned recommended). |
| `DIST_BQ_TABLE`       | no       | `inference_log`  | Destination table name.             |

Auth: standard Google application default credentials.  On GKE, use
Workload Identity for the deployment's service account.  On Cloud Run,
the runtime service account is picked up automatically.  Locally,
`GOOGLE_APPLICATION_CREDENTIALS` pointed at a service-account JSON
works.  Scope: `bigquery.insertdata` (write-only, no read).

Recommended BQ table schema (partition by DAY on `started_at`,
cluster on `(user_id, agent_id)`):

```sql
CREATE TABLE `<project>.<dataset>.inference_log` (
  id              INT64,
  user_id         INT64,
  pool_id         INT64,
  agent_user_id   INT64,
  agent_id        STRING,
  input_tokens    INT64,
  output_tokens   INT64,
  started_at      INT64,
  finished_at     INT64,
  latency_ms      INT64,
  status          STRING,
  model_name      STRING,
  region          STRING
)
PARTITION BY TIMESTAMP_TRUNC(TIMESTAMP_SECONDS(started_at), DAY)
CLUSTER BY user_id, agent_id;
```

Behaviour:

- Fully async — the inference path never blocks on BQ.
- Buffered (default 10_000 rows); rows beyond the buffer are dropped
  with a metric (`Sink.Dropped()`).
- Batched (default 500 rows or 5s flush interval).
- 5xx and per-row insert errors are logged but not retried — the next
  batch carries on.  This is intentional: long retry chains during a
  BQ outage would inflate the buffer and risk OOM.

### Bigtable hot counters (`DIST_BT_*`)

| Env                   | Required | Default | What                                       |
|-----------------------|----------|---------|--------------------------------------------|
| `DIST_BT_PROJECT`     | yes      | (off)   | GCP project hosting the Bigtable instance. |
| `DIST_BT_INSTANCE`    | yes      | (off)   | Bigtable instance id.                      |
| `DIST_BT_TABLE`       | yes      | (off)   | Table name; created externally.            |

Today the in-tree Bigtable implementation is a **stub that proxies to
the SQLite counter store** and tracks call volume in memory.  This lets
operators wire the env vars in advance and watch real QPS on `/metrics`
before flipping the build-tagged real Bigtable adapter on (separate
commit; adds the cloud.google.com/go/bigtable dep).

Row-key format the eventual adapter will use:

```
<period>#<zero-padded-user-id>
e.g. 202605#0000000000000000042
```

Period-first ordering means a single row-range scan returns every
user for a billing period — the same shape any external reporting
view (BigQuery federation over Bigtable) will want.

## Pluggable inference runtime (P12)

`dist-node` ships with llama.cpp as the default backend.  It is the
right choice for the project's core use case: pipeline-parallel
inference across heterogeneous rigs, with bf16 / fp16 activations
flowing over the ACTV wire.  But on rigs where the whole model fits
in local VRAM, the high-throughput single-rig runtime in the OSS
world is **vLLM** (PagedAttention, continuous batching, prefix
caching).  vLLM is single-rig — it cannot do pipeline parallelism
across networks — so the right composition is:

- multi-rig pipeline runs keep using llama.cpp (inline path through
  `pp_engine.cpp` / `node_agent.cpp`);
- full-model runs on a single rig can pick any **runtime adapter**
  that implements `dist::IRuntimeAdapter`
  (`include/runtime_adapter.h`).

The vLLM adapter (`src/vllm_adapter.cpp`) speaks to vLLM's
OpenAI-compatible HTTP server (`vllm.entrypoints.openai.api_server`)
over plain HTTP/SSE.  No libcurl dependency — the agent already
hand-rolls HTTP for ComfyUI and we keep the same shape here.

### Environment

| Env                          | Default                  | Notes                                                                            |
|------------------------------|--------------------------|----------------------------------------------------------------------------------|
| `DIST_RUNTIME`               | `llama-cpp`              | `llama-cpp` (default) \| `vllm` \| `sglang` (P13) \| `trtllm` (P14)              |
| `DIST_VLLM_URL`              | `http://127.0.0.1:8000`  | Base URL of the vLLM OpenAI server.  Setting this auto-enables the `vllm_caps` advertisement on `hello`. |
| `DIST_VLLM_API_KEY`          | unset                    | Bearer for vLLM's `--api-key` mode.                                              |
| `DIST_VLLM_SPAWN`            | `0`                      | When truthy, the adapter forks `python -m vllm.entrypoints.openai.api_server …`. |
| `DIST_VLLM_PYTHON`           | `python3`                | Interpreter used in spawn mode.                                                  |
| `DIST_VLLM_EXTRA_ARGS`       | empty                    | Whitespace-split, appended verbatim to the vLLM CLI in spawn mode.               |
| `DIST_VLLM_SPAWN_TIMEOUT`    | `60`                     | Seconds to wait for vLLM `/health` after spawn.                                  |
| `DIST_VLLM_CONNECT_TIMEOUT_MS` | `2000`                 | Socket-connect timeout for probe and request open.                               |

### Capability advertisement

When `DIST_RUNTIME=vllm` or `DIST_VLLM_URL` is set, dist-node probes
the local vLLM server at startup and emits a `vllm_caps` frame on
the agent WS:

```
{"kind":"vllm_caps","ok":true,"base_url":"http://127.0.0.1:8000"}
```

The control plane uses this to know which rigs can take full-model
jobs through the vLLM path.  Probe failure logs a warning but does
not crash the agent — the rig falls back to the llama.cpp inline
path the same way it would without `DIST_RUNTIME` set.

### Streaming protocol

The adapter posts to `/v1/completions` with `"stream": true` and
parses SSE `data: {...}` frames line by line.  Each delta becomes
one `RuntimeChunk` passed to the caller's `ChunkCallback`; returning
`false` from the callback closes the socket and stops decoding (the
SSE cancellation path used by `/v1/chat/completions` already).

### What's intentionally not in this PR

The agent's existing inference path still routes through
`pp_engine`.  P12 establishes the adapter interface and a working
vLLM bridge; lifting the inline llama.cpp path behind a
`LlamaCppAdapter` and switching the dispatcher to call the
adapter trait by default is the P12-follow-up.  Until then, set
`DIST_RUNTIME=vllm` only on rigs where the operator has manually
wired the vLLM endpoint into their server-side routing.

## SGLang adapter + prefix-cache routing (P13)

P13 adds a second runtime adapter — SGLang — and uses one of its
native features to make the control plane smarter about which rig
serves a follow-up turn.

### Why SGLang gets its own adapter (not OpenAI-compat)

SGLang's `vllm`-style OpenAI endpoints work, but they strip the
prompt-prefix-cache hit count from the response.  Its native
`/generate` endpoint exposes `meta_info.cached_tokens`, which is the
exact signal the routing layer needs: it tells us how many tokens of
the prompt SGLang served straight out of its radix-tree cache
without re-prefilling.  That's the difference between a 50ms first
token and a 2s first token for chat workloads with long shared
system prompts.

### Routing flow

1. Agent boots with `DIST_RUNTIME=sglang` (or `DIST_SGLANG_URL`
   set).  `dist-node` probes the local SGLang `/health` endpoint
   and, on success, emits `{"kind":"sglang_caps","ok":true,
   "base_url":"...","prefix_cache":true}` on the agent WS.
2. The control plane records the cap in `sglang_caps` (parallel to
   `comfy_caps`).
3. After each inference finishes, the agent's adapter exposes the
   stream's last `cached_tokens` via `SglangAdapter::last_cached_tokens()`.
   The server upserts `(user_id, prefix_hash, rig_id) → cached_tokens`
   into `prefix_affinity`.
4. On a follow-up request, the dispatcher hashes the prompt's first
   512 bytes (`promptPrefixHash`) and consults `prefixAffinityRig()`
   for a sticky route.  Same hash → same rig → prefix cache hit.

The table is bounded by `prefixAffinityCapPerUser` (32 distinct
prefixes per owner); beyond that window the oldest entries are
pruned because the rig's radix cache will have evicted them anyway.

### Environment

| Env                              | Default                   | Notes                                         |
|----------------------------------|---------------------------|-----------------------------------------------|
| `DIST_RUNTIME=sglang`            | unset                     | Enables sglang_caps advertisement.            |
| `DIST_SGLANG_URL`                | `http://127.0.0.1:30000`  | Base URL of `sglang.launch_server`.           |
| `DIST_SGLANG_API_KEY`            | unset                     | Bearer for SGLang's `--api-key` flag.         |
| `DIST_SGLANG_SPAWN`              | `0`                       | Truthy → adapter forks `python -m sglang.launch_server …`. |
| `DIST_SGLANG_PYTHON`             | `python3`                  | Interpreter for spawn mode.                   |
| `DIST_SGLANG_EXTRA_ARGS`         | empty                     | Verbatim CLI args appended to launch_server.  |
| `DIST_SGLANG_SPAWN_TIMEOUT`      | `90`                      | Seconds to wait for `/health` after spawn.    |
| `DIST_SGLANG_CONNECT_TIMEOUT_MS` | `2000`                    | Socket-connect timeout.                       |

### Privacy / scoping

`prefix_affinity` is keyed by `user_id` first.  Owner A's prompt
hash never matches owner B's, even if they happen to share a
verbatim system prompt — the row keys are disjoint by design.  The
hash is over the first 512 prompt bytes only, so a long shared
prefix followed by per-user PII produces a stable hash without
storing the PII anywhere.

## TensorRT-LLM via Triton (P14)

P14 adds a third runtime tier — TensorRT-LLM — for pinned
production models where the engine plan and prefix-cache config
are owned by the operator and rebuilt rarely.  TRT-LLM compiles a
model to a fused engine plan and runs it under NVIDIA's runtime;
the fastest path to that runtime from outside is **Triton
Inference Server's `tensorrtllm_backend`**, which exposes a stable
HTTP/gRPC surface.  dist-node talks to Triton over HTTP+SSE and
does **not** link against `nvinfer` / `cudart` / the TRT-LLM C++
runtime — keeps the dist-node binary CPU-only and the build host
free of NVIDIA dependencies.

### Wire

`POST {base_url}/v2/models/{model_name}/generate_stream` with

```json
{
  "text_input": "<prompt>",
  "parameters": {
    "max_tokens": 64,
    "temperature": 0.7,
    "top_p": 1.0,
    "stop_words": ["</s>"],
    "stream": true
  }
}
```

The server replies with SSE events of the form

```
data: {"text_output":"<delta>","model_name":"...","finished":false}
```

and signals end-of-stream with `"finished": true` (Triton does
**not** send `[DONE]`).

### Environment

| variable                          | default                   | meaning                                                                          |
| --------------------------------- | ------------------------- | -------------------------------------------------------------------------------- |
| `DIST_RUNTIME`                    | `llama-cpp`               | Set to `trtllm` (also `tensorrt-llm` / `tensorrtllm`) to enable.                 |
| `DIST_TRTLLM_URL`                 | `http://127.0.0.1:8000`   | Triton HTTP base URL.  Setting this auto-enables the `trtllm_caps` advertisement.|
| `DIST_TRTLLM_MODEL`               | `ensemble`                | Name under Triton's model repository (must match the served model directory).    |
| `DIST_TRTLLM_API_KEY`             | unset                     | Bearer token for Triton's HTTP API (if the operator fronts it with auth).        |
| `DIST_TRTLLM_CONNECT_TIMEOUT_MS`  | `2000`                    | Socket-connect timeout.                                                          |

### Capability advertisement

At pair-time dist-node probes Triton's `/v2/health/ready` and
emits a `trtllm_caps` frame:

```json
{"kind":"trtllm_caps","ok":true,"base_url":"http://127.0.0.1:8000","model":"ensemble"}
```

The control plane stores it in the same `sglang_caps` table as
the vLLM tier (no per-prefix `cached_tokens` channel — Triton
owns prefix caching internally and does not surface a stable
counter), so the dispatcher can route full-model jobs to the
Triton-backed rig.

### Why not link the TRT-LLM C++ runtime directly?

In-process TRT-LLM is faster than HTTP-via-Triton.  It is also
the path that drags `nvinfer`, `cudart`, and the TRT-LLM static
libs onto every machine that wants to build dist-node — including
the CI host that has no GPU and shouldn't.  HTTP is a clean
process boundary: the operator builds and runs Triton with all
the NVIDIA bits, and the dist-node binary stays portable.  When
the latency cost matters more than the build-host story (large
GPU fleets, dedicated inference pods), a follow-up can ship a
DIST_USE_TRTLLM CMake flag that pulls the in-process runtime in
behind a build option.

## NAT-PMP / PCP port mapping (P17)

Before letting an ACTV peer-relay session fall back to a TURN
server (with the byte-counting and relay-credit machinery that
implies), try to ask the upstream router to open a direct
external port via **PCP** (RFC 6887) or **NAT-PMP** (RFC 6886).
Most consumer routers speak one of the two when port-forwarding
UPnP is enabled; when they do, the rig gets an externally
reachable `(ip:port)` at the cost of a single UDP round-trip
and the router itself does the forwarding (no relay hop, no
operator-funded TURN bandwidth).

### Tools

| binary         | what it does                                                                                                    |
| -------------- | --------------------------------------------------------------------------------------------------------------- |
| `nat-portmap`  | Operator probe: prints `external_ip:port`, granted lifetime, and which protocol succeeded.  `--keep` to renew.  |
| `nat-pingpong` | Already-existing UDP reachability tester (P-task earlier).  Use it once `nat-portmap` reports a mapping.        |

### Wire shape

`include/nat_pmp.h` exposes:

```cpp
std::optional<MappedPort> try_map_udp(internal_port,
                                      suggested_ext_port,
                                      lifetime_s,
                                      gateway = "",      // auto-discovers on Linux
                                      timeout_ms = 1500);

class PortMapper { /* renews halfway through granted lifetime */ };
```

The probe tries PCP MAP (version 2) first; on `UNSUPP_VERSION`
or silence, it falls back to NAT-PMP MAP (version 0).  Both
target UDP/5351 on the gateway.  Default gateway is read from
`/proc/net/route` on Linux; set `DIST_PORTMAP_GATEWAY=<ip>` to
override (also accepted by `nat-portmap --gateway`).

### When this matters

- Residential / SOHO networks where the rig sits behind a single
  NAT and the router supports port mapping.
- Carrier-grade NAT (CGNAT) **does not** support these protocols
  in general — for those rigs you'll still need TURN.
- Inside a corporate firewall, PCP is rarely enabled — admins
  should open the port explicitly instead.

### Why this lives in dist_common

The library is small (~14 KB compiled) and the ACTV path needs
it the moment a peer is opened.  Building it into `dist_common`
lets `dist-node`, `nat-portmap`, and future ICE-candidate
injection code all share the same implementation.

## Speculative decoding capability (P16)

Speculative decoding (Medusa heads / Eagle / draft-model) lets a
rig produce K candidate tokens per target-model forward pass.
Real-world acceptance rates of 1.5–3× turn long completions
from "interactive" into "fast."  The heads themselves live in
the backing runtime — vLLM, SGLang, TensorRT-LLM, or a
llama.cpp draft-model pair — and we don't try to reimplement
them.  P16 wires the control-plane half: rigs claim a method
and an expected draft size; the dispatcher will use that to
prefer them for latency-sensitive endpoints.

### Agent environment

| variable                      | meaning                                                       |
| ----------------------------- | ------------------------------------------------------------- |
| `DIST_SPEC_DECODING`          | `medusa` \| `eagle` \| `draft_model` \| `lookahead` \| `none`  |
| `DIST_SPEC_DRAFT_TOKENS`      | K = how many tokens the rig speculates per step (default `4`) |
| `DIST_SPEC_ACCEPT_HINT`       | optional float in `[0,1]`, the operator's measured accept rate|

`DIST_SPEC_DECODING=1` (truthy alias) is treated as `medusa`,
matching the most common single-model integration.

### Wire frame

Sent once on pair-time:

```json
{
  "kind": "spec_caps",
  "ok": true,
  "method": "medusa",
  "draft_tokens": 4,
  "accept_rate_hint": 0.62
}
```

Server stores it in `spec_caps (user_id, agent_id, ok, method,
draft_tokens, accept_rate_hint, updated_at)`.  Unknown methods
and out-of-range numbers are normalized server-side — agents
cannot inject arbitrary strings into the column.

### Routing (follow-up)

This PR establishes the surface end-to-end (agent advert →
server table → tested upserts) but does **not** yet wire
`spec_caps.ok` into dispatch ranking.  The intended use is in a
follow-up to bias the dispatcher toward `ok=1` rigs when the
incoming request is small (`max_tokens < 256`, `stream=true`)
and tie-break by `draft_tokens * accept_rate_hint` as a proxy
for expected per-step token gain.

## FP8 (E4M3) ACTV compression (P15)

Set `DIST_ACTV_FP8=1` on a rig to opt-in to FP8 E4M3 hidden-state
compression on the ACTV wire.  Hidden states are normally streamed
between pipeline stages as fp32 (`dtype=0`).  With FP8 enabled, the
sender quantizes each per-step hidden vector to E4M3 with a
per-tensor float32 scale factor, and the receiver decodes back to
fp32 before feeding the next stage.  Logits are **not** compressed
— the vocab softmax is sensitive to precision and the wire cost is
dominated by per-step hidden vectors anyway.

### Wire shape

When `TensorHeader.dtype == 3` the payload is:

```
TensorHeader { dtype=3, n_tokens, n_embd, ... }
float32       per_tensor_scale
uint8_t[n]    E4M3 bytes   (n = n_tokens * n_embd)
```

The receiver recovers `f32_value = decode_e4m3(byte) * scale`.
Scale is chosen so the largest |x| in the tensor maps to ±448
(the E4M3 max-normal), preserving as much dynamic range as the
format allows.

### Bandwidth

| dtype     | bytes/element | example: 8192 hidden × 32 steps |
| --------- | ------------- | -------------------------------- |
| fp32 (0)  | 4             | 1.00 MB                          |
| fp16 (1)  | 2             | 0.50 MB                          |
| fp8  (3)  | 1 + scale     | 0.25 MB + 4 B                    |

In practice the 4-byte per-tensor scale is amortized across
several thousand elements, so the realized savings are ≥3.99× vs
fp32 and ≥1.99× vs fp16.

### Round-trip quality

On N(0, σ²) hidden states (the practical distribution after
RMSNorm), mean absolute error sits around 2% of mean(|x|) with
no per-element relative errors above 10% on the |x| > 0.1 tail.
That's below the noise floor of greedy / temperature sampling at
the next stage's softmax.

### When to enable

- High-latency cross-region peer-relay rigs where the SDK doubles
  the data on each leg.
- Low-end uplinks where 16 KB/step vs 64 KB/step is a real cost.
- Anywhere bandwidth, not GPU, is the bottleneck.

Leave **off** when:

- All rigs share a LAN or a tight DC mesh (CPU encode/decode cost
  outweighs wire savings).
- The model is sensitive enough that even sub-1% activation error
  shows up in eval metrics — measure first.

### Format

We track the OCP MX-FP8 / NVIDIA E4M3 convention:

- 1 sign bit, 4 exponent bits, 3 mantissa bits
- exponent bias = 7
- max normal = 448 (`exp=15, mant=6`)
- NaN encoding: `exp=15, mant=7` only — no Inf
- subnormal min = 2^-9

The encoder is pure C++17 scalar code in `include/fp8.h`; no
NVIDIA intrinsics are required and the dist-node binary stays
CPU-portable.

## URL capabilities (P7)

Shard download URLs (`/models/{id}/shards/{file}`) and Comfy output
URLs (`/comfy/out/{id}/{file}`) are signed with macaroon-based
capability tokens carried in the `cap=` query parameter.  Caveats:

- `path=<shard|comfy-out>` — namespaces tokens so a shard cap cannot
  be replayed against a comfy URL.
- `model=`/`job=` — resource id.
- `uid=` — for comfy URLs, binds the cap to a specific user; the
  handler also requires that the calling session match this uid, so
  a leaked URL pasted into a public page cannot be opened by anyone
  but the original user.
- `file=` — exact filename, case-sensitive.
- `exp=` — wall-clock expiry.

The HMAC root key is derived from `DIST_SESSION_SECRET` (the same
secret used by the legacy `?exp=&sig=` URLs), so rotating that
secret invalidates every outstanding cap in one move.

The legacy `?exp=&sig=` (shard) and `?v=2&uid=&exp=&sig=` (comfy)
URL shapes are still accepted within 24 h of server start so URLs
minted just before a rolling deploy don't 401 mid-fetch.  After the
window, only macaroon caps verify.
