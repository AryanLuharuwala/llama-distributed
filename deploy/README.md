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

## What's NOT here yet

Items deferred to later prod-branch commits, in rough priority order:

- **Helm chart** — package the (envoy + dist-server + redis + postgres)
  topology for k8s with proper PDBs, HPA, and a sidecar Redis or
  Memorystore reference.
- **Spanner / CockroachDB migration path** (P10) — once SQLite
  becomes the bottleneck.  The dialect abstraction (server/dialect.go)
  is the first-mile of this work; remaining gaps are query-time `?`
  placeholders and Postgres connection pooling tuning.

See the `Pn` tasks in the project board for the full plan.

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
