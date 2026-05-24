# Auth/Authz adversarial audit

Date: 2026-05-25
Reviewer: adversarial sweep — authn/authz boundary
Branch: `prod` (HEAD a4db35d)
Scope: HTTP / WS surface reachable by an unauthenticated remote attacker.

Tests written under `//go:build adversarial`; run with:

```
cd server && go test -tags=adversarial -run TestAdversarial -v
```

Result at audit time: **2 FAIL (demonstrate real holes) / 11 PASS (confirm or
document defended-in-depth gaps).**

---

## Summary (top 5 findings, severity-ranked)

1. **[CRITICAL] Macaroon caveats not enforced when absent** — `capChecker` only
   complains about *unexpected* caveat keys; if `model=`, `file=`, `exp=`, or
   `uid=` is missing from the macaroon, the verifier ignores the gap and
   accepts the cap as a universal grant. An attacker (or a future bug in the
   minter) who can produce a macaroon with only `path=shard` set can read any
   shard for any model with no expiry. Requires sessionSecret or a minting
   path that drops caveats.
   - file: `server/macaroon.go:126-152`
   - PoC: `TestAdversarial_CapMissingCaveat` (FAIL)
2. **[HIGH] Macaroon `exp=` caveat is optional in the same way** — a cap
   minted without an `exp=` caveat is honored forever. Same root cause as #1.
   - file: `server/macaroon.go:126-152`
   - PoC: `TestAdversarial_CapMissingExp` (FAIL)
3. **[MEDIUM] Agent-ID squatting locks legitimate rigs out of resume** — the
   resume query in `ws.go` is `WHERE agent_id = ? LIMIT 1` while the rigs
   table only enforces `UNIQUE(user_id, agent_id)`. An attacker who learns
   a victim's `agent_id` and registers it on their own account can flip the
   LIMIT-1 winner; the victim's `agent_key` then fails the hash check.
   Persistent reconnect denial.
   - file: `server/ws.go:709`, schema in `server/db.go:44-53`
   - PoC: `TestAdversarial_AgentIDSquattingDoS` (PASS — documents the gap)
4. **[MEDIUM] No CSRF / Origin / auth check on POST `/auth/logout`** — anyone
   who can hand the server a session cookie value (referer leak, sub-domain
   XSS, log scraping) can terminate the session. SameSite=Lax is the only
   mitigation, and it doesn't cover curl/replay scenarios.
   - file: `server/server.go:785-791`
   - PoC: `TestAdversarial_LogoutCSRF` (PASS — documents)
5. **[MEDIUM] Device-approve race binds rig to first approver** — a user_code
   shown to the victim can be approved by *any* logged-in user. ~38 bits of
   entropy (30^8) and a 6s per-IP throttle slow but don't prevent a race
   between an attacker with the code and the legitimate operator.
   - file: `server/device_code.go:109-211`
   - PoC: `TestAdversarial_DeviceApproveCrossUser` (PASS — first-approver
     wins, victim 409s)

---

## Endpoints reviewed

| Endpoint | Auth | Notes |
|---|---|---|
| `POST /auth/dev` | none (devMode gate) | Confirmed fail-closed when devMode off (BUGS_FOUND #2). |
| `POST /auth/logout` | cookie only | No CSRF / Origin check — finding #4. |
| `GET /auth/github`, `/auth/github/callback` | state cookie | State HMAC binds IP+ts; not session. |
| `GET /auth/google`, `/auth/google/callback` | state cookie | Same shape as GitHub. |
| `GET /auth/oidc`, `/auth/oidc/callback` | state + PKCE verifier cookies | PKCE used; no session binding on state. |
| `POST /api/pair` | session | One-shot consume verified. |
| `POST /api/device/code` | public | Uses `cfg.publicURL` (Host-injection closed per BUGS_FOUND #1). |
| `POST /api/device/approve` | session + ipRL | First-approver-wins race (finding #5). |
| `POST /api/device/token` | public + ipRL + one-shot consume | One-shot consume verified. |
| `POST /api/agent/api_key` | agent_key bearer | **No LimitReader** on body (`device_code.go:370`); no per-rig RL on key-mint. Low. |
| `GET /api/agent/pools` | agent_key bearer | Returns base_url derived from `r.Host` (low — caller can't poison other agents). |
| `GET /models/{id}/shards/{file}` | cap (or legacy HMAC ≤24h) | No session check; cap is bearer. |
| `GET /comfy/out/{id}/{file}` | cap OR (legacy v2 HMAC + session uid) OR (legacy v1 HMAC, no uid, 24h grace) | v1 grace is cross-user replayable; v2/cap require session uid match. |
| `GET /ws/agent` | pair token (hello) OR agent_key hash + optional ed25519 challenge (resume) | A1's hash + constant-time compare confirmed at `agent_key.go:70-84`; replay protected. |
| `GET /ws/browser` | session cookie | Same-origin only by acceptOptions. |
| `/v1/*` | API key bearer only | CORS `*` safe — no cookie path. |
| `POST /api/install_command`, `/api/firewall_check`, `/api/usage`, `/api/me*` | session | Standard. |
| `GET /api/firewall_check` | session | Read-only. |

---

## Findings

### Finding 1 — [CRITICAL] macaroon caveats not enforced when absent

- **file:line:** `server/macaroon.go:126-152` (`capChecker`), called from
  `verifyShardCap` / `verifyComfyOutputCap` (`macaroon.go:165`, `:194`).
- **What:** `capChecker(expected)` returns a check function that
  - errors on caveats whose keys aren't in `expected`,
  - errors on caveats whose values don't match `expected`,
  - **does nothing about expected keys that are absent from the macaroon.**
  `macaroon.v2.Verify` only calls the checker for caveats *present* in the
  token. A macaroon that contains zero caveats verifies cleanly against any
  expected map — and a macaroon that contains only `path=shard` verifies as
  a cap for *any* `(modelID, file)` pair, with no expiry.
- **Attack scenario:** primary risk is supply-chain / future regression: any
  bug, refactor, or rogue minting helper that produces a cap without all
  caveats grants forever-access to arbitrary shards. Direct exploitation by
  an unauthenticated attacker requires `sessionSecret`, in which case they
  also have legacy-HMAC and session-token forgery — so today this is
  defense-in-depth, but it's load-bearing the moment minting code grows.
- **PoC:** `TestAdversarial_CapMissingCaveat` (FAIL at audit time).
- **Suggested fix:** in `verifyCap`, track which expected keys appear in the
  caveat list (via a `seen` map fed inside the checker closure) and assert
  every expected key was seen before returning nil. Existing callers pass
  `path`, `model`, `file`, `exp` (and `uid` / `job` for comfy) — all required.

### Finding 2 — [HIGH] missing `exp=` caveat produces an eternal cap

- **file:line:** same as #1; same root cause.
- **Attack scenario:** mint helpers in `macaroon.go:156-209` always include
  `exp=`, so the gap requires a future minting path that drops it (or
  attacker-controlled forgery). Once present, the cap never expires —
  defeats the entire "short-lived URLs" property.
- **PoC:** `TestAdversarial_CapMissingExp` (FAIL).
- **Suggested fix:** the `seen` fix from #1 covers this. As a belt-and-braces
  measure, `verifyCap` could refuse any cap whose caveat list is empty or
  doesn't include `exp=`.

### Finding 3 — [MEDIUM] agent-ID squatting denies rig resume

- **file:line:** `server/ws.go:708-715`, schema at `server/db.go:44-53`.
- **What:** `UNIQUE(user_id, agent_id)` allows the same `agent_id` for
  different users. The resume query `SELECT … FROM rigs WHERE agent_id = ?
  LIMIT 1` returns whichever row SQLite's rowid order surfaces first. If an
  attacker registers a row with the victim's `agent_id` on their own
  account, the victim's `agent_key` may not match the row LIMIT-1 picks,
  failing `verifyAgentKeyWithFallback`. The hello/resume rate limiter then
  back-pressures the victim out of reconnect.
- **Attack scenario:** attacker needs to learn the victim's `agent_id` — for
  rigs created via dist-turn it's `relay-<8-byte hex>` (64 bits); for
  dist-node device-code flow it's `<hostname>:<4-byte token>` (32 bits +
  hostname). Hostname leakage (telemetry, public swarm page) makes this
  reachable. The attacker also needs a valid pair token / device approval
  on their own account to insert the squatting row.
- **PoC:** `TestAdversarial_AgentIDSquattingDoS` (PASS — documents the gap).
- **Suggested fix:** index `agent_key_hash` (already done — see
  `db.go:192`) and resume by `WHERE agent_key_hash = ?`; cross-check
  `agent_id` *after* identity is established. This is what `agentFromRequest`
  already does for the REST path (`device_code.go:340-353`); the WS resume
  flow should match.

### Finding 4 — [MEDIUM] `/auth/logout` is CSRFable / unauthenticated

- **file:line:** `server/server.go:785-791`.
- **What:** the handler reads the cookie value, runs `DELETE FROM sessions
  WHERE id = ?` against it verbatim, then clears the cookie. No Origin /
  Referer / CSRF-token check; no verification that the caller is the
  legitimate holder of that session.
- **Attack scenario:** sub-domain XSS, log leak, or attacker on the path
  (TLS-terminating proxy with verbose access logs) hands the cookie value
  to a curl, the victim is logged out. SameSite=Lax stops vanilla
  cross-site form-POST CSRF, but does nothing against a server-side curl.
- **PoC:** `TestAdversarial_LogoutCSRF` (PASS).
- **Suggested fix:** require the caller to pass a CSRF token (double-submit
  cookie or stored-in-session token) and/or check `Origin` against the
  configured `publicURL`. As a minimum, require the cookie *and* a
  same-origin Origin header.

### Finding 5 — [MEDIUM] device-approve race binds rig to first approver

- **file:line:** `server/device_code.go:109-211`.
- **What:** `handleDeviceApprove` accepts a `user_code` from any logged-in
  user and binds the rig to *that* user. The legit operator may be
  redirected through `/auth?next=/device?code=…` before reaching the
  approve page — that's seconds of window. Anyone who saw the
  `user_code` (over-the-shoulder, screen-share, screenshot in a
  Slack channel) can race to the approve endpoint first.
- **Attack scenario:** social engineering — attacker convinces operator to
  read out the user_code, then approves it on their own account before
  the operator gets to the dashboard. Rig now reconnects to the
  attacker's account, not the operator's.
- **PoC:** `TestAdversarial_DeviceApproveCrossUser` (PASS).
- **Suggested fix:** require the approving user to demonstrate co-location
  with the device — e.g. the rig signs a challenge from the
  device_code page and the approver echoes that signature. Or, simpler:
  bind the device_code row to the IP that minted it and require the
  approver to come from the same /24. Both alter the UX so the
  short-term fix is just a clearer dashboard warning.

### Finding 6 — [LOW] `oauth_state` is bound to client IP only

- **file:line:** `server/oauth.go:31-70`, used at
  `oauth.go:141,186`, `oauth_google.go:40,85`, `oidc.go:137,185`.
- **What:** `mintOAuthState(clientIP)` builds an HMAC over `ts + clientIP`.
  No session identifier, no per-flow nonce stored server-side. Two
  callers on the same NAT'd IP (mobile carrier, corporate egress) can
  swap state tokens.
- **Attack scenario:** a hostile co-tenant on the same egress IP captures
  the victim's `state` (e.g. via a malicious Wi-Fi captive portal that
  records the Referer) and replays it in their own browser to walk
  through their own OAuth flow — the result is they create an account
  bound to *their* GitHub identity in the victim's session UI, which is
  the wrong threat model but does break the documented "binds to client".
- **PoC:** `TestAdversarial_OauthStateNoSessionBinding` (PASS — confirms
  documented behavior).
- **Suggested fix:** also bind to a `state_nonce` cookie (random
  per-start, HttpOnly + Secure + SameSite=Lax), so the state value is
  bound to both the IP and the actual browser instance.

### Finding 7 — [LOW] `oauth_next` admits `/\evil.com`

- **file:line:** `server/oauth.go:154,253`,
  `server/oauth_google.go:51,140`, `server/oidc.go:153,244`.
- **What:** the redirect-target validator is
  `strings.HasPrefix(v, "/") && !strings.HasPrefix(v, "//")`. A path
  like `/\evil.com` passes — and several browsers normalise `\` to `/`
  in the URL parsing step, turning the eventual Location header into
  `//evil.com` (protocol-relative open redirect).
- **PoC:** `TestAdversarial_OpenRedirectViaNext` (PASS — flag-only).
- **Suggested fix:** also reject paths starting with `/\\` or containing
  any backslash, or run the value through `url.Parse` and refuse hosts.

### Finding 8 — [LOW] garbage XFF hop becomes literal rate-limit key

- **file:line:** `server/trusted_proxy.go:140-147`.
- **What:** if a hop in `X-Forwarded-For` isn't a parseable IP,
  `trustedClientIP` returns the raw string. An attacker upstream of a
  trusted proxy that fails to validate XFF can cycle bucket keys
  arbitrarily.
- **Attack scenario:** Requires misconfigured front proxy (envoy/nginx
  not stripping inbound XFF). With Envoy + the documented
  `xff_num_trusted_hops: 0` config, the attacker's XFF is overwritten,
  so this is theoretical for well-configured deploys.
- **PoC:** `TestAdversarial_TrustedProxyGarbageXFF` (PASS).
- **Suggested fix:** on parse failure, fall back to `r.RemoteAddr` rather
  than returning garbage. Or skip the entry and keep walking right-to-
  left.

### Finding 9 — [LOW] NUL byte not blocked by `isSafeShardFile`

- **file:line:** `server/models.go:362-375`,
  `server/comfy.go:1211-1223`.
- **What:** validator checks for `/`, `\\`, `.`, `..` and suffix, but
  not for NUL. A name like `stage-0\x00.gguf` slips through validation;
  Go's `os.Open` rejects the NUL at syscall time so this isn't directly
  exploitable, but it's a hygiene fix.
- **PoC:** `TestAdversarial_NULByteInShardFile` (PASS).
- **Suggested fix:** add `if strings.ContainsRune(f, 0) { return false }`.

### Finding 10 — [LOW] `handleAgentMintAPIKey` has no LimitReader

- **file:line:** `server/device_code.go:370`.
- **What:** `json.NewDecoder(r.Body).Decode(&body)` — no `io.LimitReader`,
  unlike every other auth'd JSON endpoint per BUGS_FOUND #3.
- **Attack scenario:** any agent_key holder can post a multi-gigabyte
  body; the `Label` string is buffered into RAM and then persisted to
  the api_keys table.
- **Suggested fix:** `io.LimitReader(r.Body, 4<<10)` to match
  `handleMintAPIKey`.

### Finding 11 — [LOW] IPv6 /64 bucketing is per-/128

- **file:line:** `server/ip_ratelimit.go:95-132`,
  `server/trusted_proxy.go:111-155`.
- **What:** the rate-limit bucket key is the literal IP string. IPv6
  clients with a /64 allocation (the standard SLAAC residential block)
  have 2^64 addresses they can spread credential-stuffing across before
  hitting any aggregate limit.
- **Attack scenario:** brute-force `/auth/dev` (if devMode is on),
  `/api/device/approve` `user_code` enumeration, WS hello-fail spam —
  all benefit from /64 spray.
- **Suggested fix:** bucket IPv6 addresses by `/64` (and IPv4 by `/24`?)
  for the credential-sensitive endpoints (deviceApprove, helloFail).
  Leave per-/128 for the read-only endpoints.

---

## Tests written

File: `server/adversarial_auth_test.go` (build tag `adversarial`).

| Test | Asserts / documents | Result |
|---|---|---|
| `TestAdversarial_CapMissingCaveat` | verifyShardCap should reject a macaroon missing `model=` / `file=` / `exp=` caveats | **FAIL** (demonstrates hole) |
| `TestAdversarial_CapMissingExp` | verifyShardCap should reject a macaroon missing `exp=` | **FAIL** (demonstrates hole) |
| `TestAdversarial_LogoutCSRF` | logout requires only the cookie; no Origin/CSRF token | PASS (documents) |
| `TestAdversarial_AgentIDSquattingDoS` | LIMIT-1 resume + multi-user `agent_id` lets one user lock another out | PASS (documents) |
| `TestAdversarial_LegacyHMACGraceCrossUser` | v1 comfy URL within grace window has no session check | PASS (documents) |
| `TestAdversarial_OauthStateNoSessionBinding` | state binds IP-only, not session | PASS (confirms) |
| `TestAdversarial_DeviceApproveCrossUser` | first-approver wins; victim 409s | PASS (confirms) |
| `TestAdversarial_PairTokenReuseAcrossSessions` | pair token is one-shot | PASS (asserts fix) |
| `TestAdversarial_OpenRedirectViaNext` | `oauth_next` admits `/\evil.com` | PASS (documents) |
| `TestAdversarial_TrustedProxyGarbageXFF` | garbage hop becomes literal bucket key | PASS (documents) |
| `TestAdversarial_OAuthStateDotSplitting` | malformed state strings rejected without panic | PASS (asserts robust parser) |
| `TestAdversarial_PathTraversalShardFile` | common traversal shapes rejected | PASS (asserts fix) |
| `TestAdversarial_NULByteInShardFile` | NUL byte handling | PASS (documents) |

To reproduce:

```
cd server
go test -tags=adversarial -run TestAdversarial -v
```

---

## Known gaps re-confirmed vs. closed

| Gap from MEMORY.md | Status on `prod` HEAD |
|---|---|
| XFF Trust Known Gap | **CLOSED** on prod: `trusted_proxy.go` wired into both `clientIP` and `remoteIPForRateLimit`; empty trust set ⇒ XFF ignored. The "garbage hop" subcase (Finding 8) survives but is hygiene-only. |
| Peer-Relay Plaintext Exposure | **STILL OPEN** by design; not in scope for this audit, but I confirmed `pp_route.go` still terminates DTLS on both legs. |
| BUGS_FOUND #1 Host-header phishing in device-code | **CLOSED** — both `handleDeviceCodeMint` and `handleDeviceToken` now use `cfg.publicURL`. |
| BUGS_FOUND #2 `/auth/dev` open by default | **CLOSED** — `devDefault=false`, defense-in-depth gate in handler. |
| BUGS_FOUND #3 unbounded bodies | **CLOSED for the listed endpoints** but **`handleAgentMintAPIKey` was missed** — see Finding 10. |
| A1 agent_key hashed + const-time compared | **CONFIRMED** at `agent_key.go:70-84` — `subtle.ConstantTimeCompare` for both hashed and legacy paths. |
| A3 device-code RL | **CONFIRMED** at `device_code.go:110,218` — `ipRL.deviceApprove` (1/6s burst 3) and `ipRL.devicePoll` (1/2s burst 10). |
| A7 oauth state HMAC | **CONFIRMED** at `oauth.go:31-70`; one extension (Finding 6) suggests also binding to session. |
| Single-use pair token | **CONFIRMED** by `TestAdversarial_PairTokenReuseAcrossSessions`. |
| Macaroon strict round-trip (trailing-junk attack defence) | **CONFIRMED** at `macaroon.go:99-110`. |
| Macaroon caveat-completeness | **NEW finding** (#1, #2) — the round-trip check defends against trailing-junk, but the *missing-caveat* shape was never asserted. |

---

## What I couldn't get to and why

- **End-to-end WebSocket hello/resume PoC**: writing a real WS-client test
  requires either spinning up a real `httptest.Server` + `websocket.Dial`
  or simulating the full codec handshake in-process. I dropped this in
  favour of asserting the SQL-shape DoS directly (Finding 3) and trusting
  `agent_key.go`'s hash + `subtle.ConstantTimeCompare` for the
  hello-replay scenario. The constant-time + hash compare is short
  enough to read by eye; replay is impossible without the plaintext key.
- **OIDC JWKS / token-replay tests**: would need a fake OP. The OIDC
  call sites pass through `go-oidc`'s `prov.Verifier(...)`, so the
  audit trusted that library. Note: `userFromOIDCBearer` auto-upserts
  on first sight, which means any valid JWT for the configured
  audience creates a local user. Operators must double-check the
  `DIST_OIDC_BEARER_AUDIENCE` env (or it defaults to the OAuth client
  ID, which is wider than ideal — see `oidc.go:52-56`).
- **`r.URL.Path` `%2f` decoding for shard/comfy files**: Go's net/http
  1.22 ServeMux uses `r.URL.Path` (decoded) for PathValue, which means
  a literal `/` in the filename can't survive the router. Verified by
  reading the mux code path; didn't write a black-box repro.
- **Header / chunked-encoding boundary fuzzing** (the "phase C" item):
  was redirected to the parallel wire-format fuzz agent per the
  delegation note in the prompt.
- **Real CSRF browser repro**: the SameSite=Lax + cookie-only
  defence is well-understood; static analysis was sufficient.
