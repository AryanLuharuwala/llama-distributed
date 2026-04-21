# llama-distributed — end-to-end flow

This is the reference doc for what actually happens between a user clicking
buttons in the dashboard and tokens streaming out of a pooled rig. It also
catalogues the friction points so we can keep cutting them down.

---

## 1. System overview

Three kinds of process:

| Process           | Runs on                                | What it does                                                                   |
|-------------------|-----------------------------------------|--------------------------------------------------------------------------------|
| `dist-server`     | Control plane (your apex host)          | Serves dashboard UI, OAuth, pair tokens, pool routing, OpenAI-compat endpoints |
| `dist-node`       | Each rig (laptop / desktop / GPU box)   | Loads model shards, runs inference, keeps a WebSocket open to the server      |
| Browser / API client | End user                              | Consumes the dashboard or `/v1/*` via an API key                               |

Data stores: one SQLite file (`distpool.sqlite`) on the control plane. No
other infra required for a single-region deployment.

---

## 2. The happy-path flow (what the user actually sees)

```
┌──────────────┐   1. GitHub OAuth / dev login
│   Browser    │─────────────────────────────▶
│ (dashboard)  │                                  ┌──────────────────┐
│              │   2. "Generate install command" │                  │
│              │─────────────────────────────▶   │   dist-server    │
│              │ ◀───────── one-liner ────────   │  (control plane) │
│              │                                  │                  │
└──────────────┘                                  │  SQLite          │
                                                  │  - users         │
 paste one-liner                                  │  - pair_tokens   │
        │                                         │  - rigs          │
        ▼                                         │  - pools         │
┌──────────────┐   3. curl install.sh             │  - api_keys      │
│ Rig laptop   │ ──────────────────────────────▶  │                  │
│              │   4. download dist-node tarball  │                  │
│              │ ◀──── from GitHub Releases ────  └──────────────────┘
│              │             │                            ▲
│              │             │ 5. dist-node --pair …      │
│   dist-node  │─────────────┴───────────── WebSocket ────┘
│   running    │   6. presents token, becomes rig #N
└──────────────┘   7. subsequent runs reconnect via saved agent_key
```

8. User clicks **+ New pool** → server assigns a slug (e.g. `alice-alpha`),
   returns `base_url`.
9. User drags the rig into the pool.
10. User clicks **+ New key** → server mints `sk-dist-…` (shown once).
11. Any OpenAI SDK pointed at `https://<slug>.<apex>/v1` with that bearer
    key routes chat requests to whichever rig in the pool is online.

---

## 3. Step-by-step: what each click actually does

### 3.1 Login

- `GET /` serves the embedded SPA.
- GitHub OAuth round-trip in prod; `-dev` flag exposes `POST /auth/dev` for
  local work (no GitHub app needed).
- Session cookie is an HMAC-signed user id.

### 3.2 Generate install command

- Browser → `POST /api/install_command`.
- Server mints a 20-byte pair token, TTL 5 min, stores it in `pair_tokens`
  with `user_id`.
- Response has two ready-to-paste strings:
  ```
  bash: curl -fsSL http://<apex>/install.sh | sh -s -- --pair 'distpool://pair?token=XXX&server=ws://<apex>/ws/agent'
  pwsh: $p='distpool://…'; iwr http://<apex>/install.ps1 -UseB | iex
  ```
- UI tabs between the two.

### 3.3 User pastes on the rig

- `install.sh` (or `install.ps1`):
  1. Detects OS + arch + accelerator (nvidia-smi → cuda; darwin-arm → metal).
  2. Resolves `https://github.com/<repo>/releases/latest` → tag.
  3. Downloads `llama-distributed-<tag>-<target>.tar.gz`.
  4. Extracts to `~/.local/share/llama-distributed` (linux) /
     `~/Library/Application Support/llama-distributed` (mac) /
     `%LOCALAPPDATA%\llama-distributed` (win).
  5. Writes a per-user service unit (systemd / launchd / Scheduled Task).
  6. Exec's `dist-node --pair <url>`.

### 3.4 First-time pair handshake

- `dist-node` dials the WebSocket URL from the deep link.
- Sends `{ "type": "pair", "token": "XXX", "hostname": "...", "gpus": […] }`.
- Server calls `consumePairToken(token)` — single-use, TTL-guarded — gets
  the `user_id`, inserts a row in `rigs` with a fresh random `agent_key`,
  sends `{ "type": "paired", "rig_id": N, "agent_key": "…" }`.
- Agent persists `agent_key` to a local config file.

### 3.5 Subsequent reconnects

- `dist-node` dials `/ws/agent`, sends `{ "type": "auth", "agent_key": "…" }`.
- Server looks up the rig, marks online, subscribes it to events.
- Reconnect with backoff when the WS drops.

### 3.6 Create a pool

- Browser → `POST /api/pools {name, visibility}`.
- Server runs `pickPoolSlug(userHandle, name)` — slugify + collision check —
  and inserts. Returns `{id, slug, base_url}`.
- `base_url` in prod = `https://<slug>.<apex>/v1`.
  In dev/LAN = `http://<public_url>/v1?pool=<slug>` (no wildcard DNS needed).

### 3.7 Attach a rig to a pool

- Browser → `POST /api/pools/{id}/rigs {rig_id}`.
- Server writes `pool_rigs` row, pushes a `pool_rigs_changed` event down
  every open `/ws/browser` for the pool's members.

### 3.8 Mint API key

- Browser → `POST /api/api_keys {label}`.
- Server generates `sk-dist-<48 hex>`, stores the **sha256 hash** + a 12-char
  plaintext prefix for display. The plaintext is returned exactly **once**.
- List endpoint never reveals the full key — only prefix + metadata.
- Revoke = `DELETE /api/api_keys/{id}`.

### 3.9 Inference request

- Client hits e.g. `POST https://alice-alpha.surds.co.in/v1/chat/completions`
  with `Authorization: Bearer sk-dist-…`.
- `resolvePoolFromHost(r)` parses `<slug>` from `r.Host` against the
  configured apex. Fallback order for LAN/dev: `X-Pool-Slug` header →
  `?pool=<slug>` query param.
- `authOAI()` hashes the bearer, looks up the `api_keys` row, checks pool
  membership of that user, bumps `last_used_at`.
- Server picks an online rig in the pool, forwards the request over its WS,
  streams the response back as SSE (OpenAI chunk format + `[DONE]`) or
  buffered JSON.

---

## 4. Where the current friction is (and how to remove it)

Rated by how much the user has to *do*, not by eng effort.

### 4.1 Installer depends on a public GitHub release

**Friction:** the one-liner 404's until we cut a tagged release. Today that
means editing CI, pushing tags, waiting ~15 min.

**Remove it:**
- **(a)** Serve `dist-node` tarballs directly from the control plane at
  `/releases/<target>.tar.gz`. Either embed them (`//go:embed`) for small
  builds, or mount a directory and rsync artifacts in. The installer
  already accepts `--repo` / `--version` — point it at
  `${DIST_PUBLIC_URL}/releases/` instead of GitHub when the dashboard host
  serves them. Wins: no GitHub dependency, works on LAN, works behind NAT.
- **(b)** Keep GitHub as the source of truth but let the dashboard proxy
  `/releases/*` through to the current GitHub Release. Cache once per tag.
- **(c)** Cut a **prerelease via `workflow_dispatch`** so the installer
  works before we're ready to ship a stable tag. (Already wired up —
  see `.github/workflows/release.yml`.)

Recommended: do **(c)** first (5 min, unblocks tonight), then **(a)** as the
real fix.

### 4.2 Two buttons on the dashboard do similar-looking things

**Friction:** "Install agent" (new rig) vs. "Pair this machine" (already
installed) are both on the landing screen. Users don't know which to click.

**Remove it:** collapse into one **"Add rig"** card with two tabs/labels:
- *First time on this machine* → shows the one-liner.
- *Already installed dist-node* → shows the `distpool://` deep link +
  QR code.

The backend stays identical; they're both `POST /api/install_command`
responses, just presented differently.

### 4.3 Pair token expires in 5 minutes

**Friction:** user generates the one-liner, gets distracted, pastes 6 min
later → "pair token expired". Forces a round trip.

**Remove it:**
- Bump TTL to 30 min for install-command tokens (keep 5 min for raw
  `/api/pair` deep links if you want).
- Or: **detect expiry on the install page itself** — poll the `expires_at`
  from the response, gray the card out when it lapses, auto-regenerate on
  click. Already have `expires_at` in the response; just wire the UI timer.

### 4.4 User has to pick OS themselves

**Friction:** the Install card shows both bash and pwsh tabs. Easy, but the
dashboard already sees `User-Agent`.

**Remove it:** default the active tab based on `navigator.platform` /
`navigator.userAgent`. Keep both tabs visible for cross-device paste-to-
phone-to-rig cases.

### 4.5 User has to SSH to the rig to paste

**Friction:** you're on your dashboard laptop, the rig is across the room.
Right now you ssh over, paste, done.

**Remove it (nice-to-have, not P0):**
- Show a **QR code** for the one-liner. Phone scans → gets the URL →
  opens a second "copy this to your rig" page. Still manual, but no
  typing.
- Or publish a signed URL that, when opened in a browser on the rig,
  does `window.location = "distpool://…"` — which the per-OS installer
  registers as a protocol handler. Then scanning the QR on the rig's
  own browser auto-fires the installer. This is the "Zoom join a meeting"
  pattern.

### 4.6 Attaching a rig to a pool is a second click

**Friction:** rig pairs → shows up in "Your rigs" → user then creates a
pool → drags rig in.

**Remove it:** on the Install card, add a dropdown *"Pool to auto-join
after pairing"* (default: *None — attach later* or, if they have one pool,
default to that). Server: pair handshake accepts an optional `pool_id` in
the token payload; if present, inserts `pool_rigs` atomically after the
pair succeeds. UI disappears a click for the common case.

### 4.7 API key flow still shows a prompt() dialog

**Friction:** clicking "+ New key" pops up the browser's native prompt. Ugly
and easy to fat-finger.

**Remove it:** inline form — label input + Create button in the card, using
the same pattern as the new pool modal. 15-line change in `ui.html`.

### 4.8 `base_url` in dev uses `?pool=` which many SDKs strip

**Friction:** `openai.OpenAI(base_url="http://host:8080/v1?pool=slug")` —
some SDKs append `/chat/completions` without preserving query string.

**Remove it:**
- Accept `/v1/<slug>/…` as a synonym for `?pool=<slug>` on the control
  plane. Then dev users get a clean `base_url = http://host:8080/v1/alice-alpha`.
- Keep the subdomain form as the canonical prod URL.

### 4.9 No "copy OpenAI snippet" for the pool

**Friction:** user got base URL and an API key. Now they have to write the
Python / curl themselves.

**Remove it:** on the pool detail modal, add a **"Use with"** subsection
with three collapsible code blocks pre-filled:
```python
from openai import OpenAI
client = OpenAI(base_url="https://alice-alpha.surds.co.in/v1",
                api_key="sk-dist-…")
```
…a `curl` version, and a Node version. Copy buttons.

### 4.10 Firewall / port 8080 friction on self-host

**Friction:** running on LAN, the other laptops can't reach the apex until
the user opens the firewall hole. We hit this tonight.

**Remove it:** print a **"sharing" pre-flight check** on startup:
- Try listening.
- Try dialing ourselves on the bound address from a goroutine.
- If dial fails and `firewalld` is present, log the exact `firewall-cmd`
  line to paste. Same for Windows Defender.

Small: the user still pastes one command, but the error moves from
`connection refused on the other laptop` → `copy this line to open the port`.

---

## 5. After these passes, the zero-friction flow is:

1. Open `https://apex`. Sign in.
2. Click **Add rig**. Pick the pool to auto-join from a dropdown. One-liner
   appears; QR next to it.
3. On the rig: paste (or scan). Installer runs, agent auto-pairs, auto-joins.
4. Back on dashboard: rig is green, in the pool.
5. Click **Get OpenAI snippet**. Paste into your code. Ship.

Five clicks, one paste, one copy. No firewall surgery, no SSH, no token
juggling, no CI waits.

---

## 6. What's out of scope for now

- Multi-region / geo-routing: today all traffic hits one control plane.
- Bring-your-own-TLS: you terminate TLS upstream (Caddy/Cloudflare) and
  forward `Host:` untouched. That's a docs item, not a code item.
- Rig → rig tensor parallel: already wired by `llama.cpp` upstream, but
  pool-level topology negotiation isn't in the dashboard yet.
- Billing / rate limits per API key.
