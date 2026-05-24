# coturn anycast edges (P18)

The ACTV peer-to-peer transport tries, in order:

1. **Host candidates** — direct LAN reachability.
2. **PCP / NAT-PMP mapped candidates** — router-forwarded
   (see `../README.md#nat-pmp--pcp-port-mapping-p17`).
3. **STUN-discovered server-reflexive candidates** — through
   public STUN endpoints (`stun.l.google.com`, `stun.cloudflare.com`).
4. **TURN relayed candidates** — through this tier (or via the
   in-tree `dist-turn` binary at `cmd/dist-turn/`, which is a
   pion/turn-based standalone relay for operators who don't want
   to run coturn).
5. **dist-server peer relay** — last-resort, double-DTLS path.

Steps 4 and 5 are the slowest and most expensive.  P18 reduces
the cost of step 4 by deploying coturn at multiple geographic
edges behind a shared anycast IP, so a client on the east coast
hits an east-coast coturn instead of a single global TURN.  The
in-tree `cmd/dist-turn/` binary (Go, pion/turn) is the
operator-owned alternative when running coturn isn't an option
(licensing, hosting policy, etc.) — see the file header for the
sidecar and standalone modes.

## Files

| file                              | purpose                                                                                 |
| --------------------------------- | --------------------------------------------------------------------------------------- |
| `coturn.conf`                     | Template with env-var placeholders.  Drop in `/etc/coturn/turnserver.conf` on each edge. |
| `docker-compose.coturn.yaml`      | Run-one-coturn-edge compose file using host networking and Let's Encrypt TLS material. |

## Anycast deployment topology

Each edge runs the same coturn container with the same
`static-auth-secret`.  The hosts each announce the same `/32`
(or `/128` for v6) into BGP from their respective POPs.  Clients
resolve `turn.dist.example.com` → the anycast IP and land on
whichever edge is closest by BGP, no GeoDNS required.

For TURN — unlike STUN — every UDP packet in a session must reach
the **same edge** that owns the allocation.  Anycast works for
this only if the upstream BGP/ECMP layer hashes by 5-tuple
(src-ip, src-port, dst-ip, dst-port, proto).  Operators who can't
guarantee that should run anycast for STUN only (port 3478 UDP
without auth) and unicast TURN behind a separate GeoDNS record.

## Auth

REST-style ephemeral credentials (`use-auth-secret` +
`static-auth-secret`).  dist-server mints time-limited
`(username, password)` pairs from the same secret and hands them
to rigs via `/api/turn-credential`.  The wire format is:

    username = "<expiry-unix-seconds>:<rig-id>"
    password = base64( HMAC-SHA1(secret, username) )

The shared secret never leaves the operator's machines.

## TLS material

coturn loads `fullchain.pem` + `privkey.pem` from
`/etc/letsencrypt/live/$DIST_TURN_REALM/`.  Renew with the same
`certbot renew` cron the rest of the stack uses; restart coturn
or send it SIGUSR2 to reload.

## Performance knobs

| knob                          | default               | when to tune                                                  |
| ----------------------------- | --------------------- | ------------------------------------------------------------- |
| `min-port` / `max-port`       | 49152–65535           | Narrow if your firewall can't pinhole 16k ports.              |
| `total-quota`                 | 10000                 | Per-edge concurrent allocation cap.                           |
| `bps-capacity`                | 1 Gbps                | Edge-wide bandwidth cap.                                      |
| `allocation-default-lifetime` | 600 s                 | Lower for ephemeral chat, higher for long pipeline runs.      |

## dist-turn (Go binary, pion/turn)

`cmd/dist-turn/main.go` is the in-tree TURN relay — a real
RFC 5766 implementation via the pion/turn library, with two
operating modes:

  - **Sidecar mode** — started by `dist-node` on relay-capable
    rigs.  Takes a per-rig HMAC secret over the CLI and forwards
    TURN traffic on UDP/3478.  Credentials are minted by
    dist-server (`mintRigTURNCreds`) and pushed to the rig in
    its welcome frame.

  - **Standalone mode** — a TURN-only node, no GPU.  Pairs with
    dist-server like any other agent (`--token=<pair-token>` on
    first run, then `agent.id` / `agent.key` persisted under
    `--state-dir`), advertises `relay_only=true`, heartbeats
    `coturn_port` + `public_ip` so the planner picks it up.

Both modes statically link, no coturn dependency, builds for
Linux / macOS / Windows out of the same source tree.  Operators
who run coturn for anycast TURN still benefit from `dist-turn`
in sidecar mode on residential relay rigs (the anycast tier
handles the common case; rig-local sidecars cover the rest).
