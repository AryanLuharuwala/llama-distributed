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
- **OIDC** (P5) — federation via dex/hydra, replacing the
  GitHub/Google OAuth pair.
- **SPIFFE/SPIRE workload identity** (P6) — rig identity via SVID
  instead of paired agent keys.

See the `Pn` tasks in the project board for the full plan.
