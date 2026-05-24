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
| `docker-compose.envoy.yaml` | Two-service stack (envoy + dist-server) with the right network isolation and trust set. |
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

This is the first commit on the `prod` branch. Items deferred to
later prod-branch commits, in rough priority order:

- **Redis-backed sessions, device codes, and rate limits** (P9) —
  required before you can horizontally scale dist-server to multiple
  replicas behind envoy. Right now each dist-server instance has its
  own in-memory rate-limit map; a multi-replica deploy is racy.
- **Helm chart** — once Redis lands, package the topology for k8s.
- **OIDC** (P5) — federation via dex/hydra, replacing the
  GitHub/Google OAuth pair.
- **SPIFFE/SPIRE workload identity** (P6) — rig identity via SVID
  instead of paired agent keys.

See the `Pn` tasks in the project board for the full plan.
