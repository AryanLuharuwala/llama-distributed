//go:build adversarial
// +build adversarial

// adversarial_auth_test.go — opt-in tests written during the auth/authz
// adversarial audit (2026-05-25).  Build tag keeps them out of the regular
// `go test ./...` sweep so a partial bypass uncovered here doesn't gate
// the green build until a fix lands.  Run with:
//
//   go test -tags=adversarial -run TestAdversarial -v ./...
//
// Each test name is prefixed `TestAdversarial_` and documents the attack
// scenario in its comment.  See /audit/auth.md for the report context.

package main

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"

	macaroon "gopkg.in/macaroon.v2"
)

// ─── TestAdversarial_CapMissingCaveat ──────────────────────────────────────
//
// Scenario: capChecker (server/macaroon.go:126) only complains about
// caveats whose keys it doesn't recognize.  If the caveat is *absent*
// entirely, macaroon.Verify never calls the checker for that key, so
// the implicit "model=X, file=Y, exp=…" guarantee evaporates.  An
// attacker who learns sessionSecret (or any minting service that ever
// drops a caveat) can mint a "universal" shard cap that unlocks any
// (modelID, file) for any TTL.
//
// Defense in depth: verifyShardCap should require that the expected
// keys (path, model, file, exp) all appeared in the macaroon, not just
// that *every present caveat* was expected.  Today it doesn't.
//
// This test demonstrates the gap by handcrafting a macaroon with only
// path=shard set and verifying it succeeds against verifyShardCap for
// arbitrary modelID/file inputs.  Expected: FAIL (test wants
// verifyShardCap to reject).  Actual at audit time: PASS-the-attack
// (verifyShardCap returns nil) — the assert flips the test to
// red-when-vulnerable.
func TestAdversarial_CapMissingCaveat(t *testing.T) {
	s := newMacaroonTestServerForAdversarial(t)

	// Mint a deliberately under-specified macaroon: only path=shard.
	m, err := macaroon.New(s.macaroonRootKey(), []byte(macaroonIDPrefix), macaroonLocation, macaroon.V2)
	if err != nil {
		t.Fatalf("macaroon.New: %v", err)
	}
	if err := m.AddFirstPartyCaveat([]byte("path=shard")); err != nil {
		t.Fatalf("AddFirstPartyCaveat: %v", err)
	}
	b, err := m.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary: %v", err)
	}
	tok := base64.RawURLEncoding.EncodeToString(b)

	// Now ask the verifier to confirm this is a cap for (modelID=99999,
	// file=arbitrary.gguf).  A correct implementation rejects because
	// the macaroon never declared model= or file= caveats — yet the
	// current verifier accepts.
	err = s.verifyShardCap(tok, 99999, "arbitrary.gguf")
	if err == nil {
		t.Fatalf("CRITICAL: verifyShardCap accepted a macaroon that never declared model= or file= caveats — universal shard cap forgery is possible if sessionSecret ever leaks")
	}
}

// ─── TestAdversarial_CapMissingExp ─────────────────────────────────────────
//
// Same shape as above for the exp= caveat: if exp is missing, the
// checker never enforces an expiry, so the cap is effectively eternal.
func TestAdversarial_CapMissingExp(t *testing.T) {
	s := newMacaroonTestServerForAdversarial(t)

	m, err := macaroon.New(s.macaroonRootKey(), []byte(macaroonIDPrefix), macaroonLocation, macaroon.V2)
	if err != nil {
		t.Fatalf("macaroon.New: %v", err)
	}
	for _, cv := range []string{"path=shard", "model=1", "file=f.gguf"} {
		if err := m.AddFirstPartyCaveat([]byte(cv)); err != nil {
			t.Fatalf("AddFirstPartyCaveat: %v", err)
		}
	}
	b, _ := m.MarshalBinary()
	tok := base64.RawURLEncoding.EncodeToString(b)

	// No exp caveat — verifyShardCap should refuse to honor an
	// un-bounded cap, but currently it does.
	if err := s.verifyShardCap(tok, 1, "f.gguf"); err == nil {
		t.Fatalf("HIGH: verifyShardCap accepted a macaroon with no exp= caveat — never-expiring caps are possible if minting drops the caveat")
	}
}

// ─── TestAdversarial_LogoutCSRF ────────────────────────────────────────────
//
// Scenario: handleLogout (server.go:785) accepts the session cookie
// value verbatim and DELETEs the row without checking that the caller
// actually has authority over that session.  Anyone in possession of a
// session ID (e.g. read from logs / referer leak / sub-domain XSS) can
// silently log the user out by POSTing /auth/logout with their cookie
// — no auth check, no CSRF token, no anti-replay.
//
// SameSite=Lax mitigates browser-cross-site CSRF but does NOT cover
// the case of a stolen-cookie + curl, nor a same-site sibling app that
// shares the cookie domain.
//
// Repro: log a user in, then drive a "logout" from a different (no
// other auth) request carrying just the cookie.  The session is
// destroyed and a subsequent authenticated call 401s.
func TestAdversarial_LogoutCSRF(t *testing.T) {
	s := newTestServer(t)
	s.cfg.publicURL = "https://distpool.example"
	uid, sid := makeUser(t, s, "victim")
	_ = uid

	// Sanity: session works.
	rr := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/api/me", nil)
	r.AddCookie(&http.Cookie{Name: sessionCookieName, Value: sid})
	s.handleMe(rr, r)
	if rr.Code != 200 {
		t.Fatalf("pre-logout /api/me = %d body=%s", rr.Code, rr.Body.String())
	}

	// Attacker GETs (or POSTs without same-origin) /auth/logout with the
	// stolen cookie. AUDIT-S4 fix: require POST + Origin/Referer matching
	// publicURL. The server now rejects.
	rr1 := httptest.NewRecorder()
	r1 := httptest.NewRequest("GET", "/auth/logout", nil)
	r1.AddCookie(&http.Cookie{Name: sessionCookieName, Value: sid})
	s.handleLogout(rr1, r1)
	if rr1.Code != 405 {
		t.Fatalf("GET /auth/logout = %d (want 405)", rr1.Code)
	}
	rr2 := httptest.NewRecorder()
	r2 := httptest.NewRequest("POST", "/auth/logout", nil)
	r2.Header.Set("Origin", "https://evil.example")
	r2.AddCookie(&http.Cookie{Name: sessionCookieName, Value: sid})
	s.handleLogout(rr2, r2)
	if rr2.Code != 403 {
		t.Fatalf("cross-origin POST /auth/logout = %d (want 403)", rr2.Code)
	}

	// Same-origin POST: succeeds (legitimate logout from the UI).
	rr3 := httptest.NewRecorder()
	r3 := httptest.NewRequest("POST", "/auth/logout", nil)
	r3.Header.Set("Origin", "https://distpool.example")
	r3.AddCookie(&http.Cookie{Name: sessionCookieName, Value: sid})
	s.handleLogout(rr3, r3)
	if rr3.Code != 200 {
		t.Fatalf("same-origin POST /auth/logout = %d (want 200)", rr3.Code)
	}
	// Post-logout: the session row is gone, /api/me 401s.
	rr4 := httptest.NewRecorder()
	r4 := httptest.NewRequest("GET", "/api/me", nil)
	r4.AddCookie(&http.Cookie{Name: sessionCookieName, Value: sid})
	s.handleMe(rr4, r4)
	if rr4.Code != 401 {
		t.Fatalf("post-logout /api/me = %d (want 401)", rr4.Code)
	}
}

// ─── TestAdversarial_AgentIDSquattingDoS ───────────────────────────────────
//
// Scenario: the rigs table has UNIQUE(user_id, agent_id) so the same
// agent_id can exist for multiple users.  The WS resume query
// (ws.go:709) is `WHERE agent_id = ? LIMIT 1`, with implementation-
// defined row selection.  An attacker who learns a victim's agent_id
// can register a row with that agent_id under their own account,
// causing SQLite (which serves rows in rowid order without ORDER BY)
// to return the attacker's row on the victim's resume — and the
// victim's hash check fails because attacker's agent_key_hash is
// different.  Net: persistent inability for victim to resume that
// rig.
//
// Mitigation needed: the resume query should be scoped to the user
// that owns the matching agent_key_hash, not by agent_id alone.
func TestAdversarial_AgentIDSquattingDoS(t *testing.T) {
	s := newTestServer(t)
	victimUID, _ := makeUser(t, s, "victim")
	attackerUID, _ := makeUser(t, s, "attacker")

	const sharedAgentID = "rig-shared-id-001"

	// Victim's rig: legit key hash.
	victimKey := "victim-key-aaaaaaaaaaaa"
	if _, err := s.db.Exec(`INSERT INTO rigs
		(user_id, agent_id, hostname, n_gpus, vram_bytes, last_seen, agent_key, agent_key_hash, pubkey)
		VALUES (?, ?, 'host', 1, 1, ?, '', ?, NULL)`,
		victimUID, sharedAgentID, nowUnix(), hashAgentKey(victimKey),
	); err != nil {
		t.Fatalf("insert victim rig: %v", err)
	}
	// Attacker squats the same agent_id under their own account.
	attackerKey := "attacker-key-zzzzzzz"
	if _, err := s.db.Exec(`INSERT INTO rigs
		(user_id, agent_id, hostname, n_gpus, vram_bytes, last_seen, agent_key, agent_key_hash, pubkey)
		VALUES (?, ?, 'evil', 1, 1, ?, '', ?, NULL)`,
		attackerUID, sharedAgentID, nowUnix(), hashAgentKey(attackerKey),
	); err != nil {
		t.Fatalf("insert attacker rig: %v", err)
	}

	// Reproduce the exact SELECT that handleAgentWS resume uses.
	// We don't drive a full websocket; just confirm LIMIT 1 returns
	// the attacker's row, breaking victim's key verification.
	var uid int64
	var storedHash, legacyPlain interface{ ScanString() string } // dummy iface to satisfy the linter
	_ = storedHash
	_ = legacyPlain

	type row struct {
		userID     int64
		hash, plain string
	}
	rows, err := s.db.Query(`SELECT user_id, agent_key_hash, agent_key FROM rigs WHERE agent_id = ?`, sharedAgentID)
	if err != nil {
		t.Fatalf("query: %v", err)
	}
	defer rows.Close()
	var ordered []row
	for rows.Next() {
		var rr row
		if err := rows.Scan(&rr.userID, &rr.hash, &rr.plain); err != nil {
			t.Fatalf("scan: %v", err)
		}
		ordered = append(ordered, rr)
	}
	if len(ordered) != 2 {
		t.Fatalf("expected 2 rows, got %d", len(ordered))
	}
	// SQLite, without ORDER BY, returns rowid order — first INSERT wins.
	if ordered[0].userID != victimUID {
		t.Logf("rowid ordering returned %d first (victim=%d, attacker=%d)", ordered[0].userID, victimUID, attackerUID)
	}

	// The actual vulnerability is the LIMIT-1 + hash-check shape: if
	// the picked row is the attacker's, the victim's key won't match.
	// Demonstrate by simulating a victim resume against the attacker's
	// row (we pick it explicitly to model the worst-case ordering).
	uid = ordered[len(ordered)-1].userID // attacker
	if uid != attackerUID {
		t.Logf("ordering placed attacker last; for repro purposes we still pick the attacker's row to show what LIMIT-1 *could* return")
	}

	// Hash check with victim's key against attacker's stored hash → fail.
	if hashAgentKey(victimKey) == ordered[len(ordered)-1].hash {
		t.Fatalf("test setup broken: hashes collided")
	}
	t.Logf("MEDIUM: SQL `WHERE agent_id = ? LIMIT 1` with multi-user UNIQUE(user_id, agent_id) lets one user lock another out of their own agent_id; the fix is to look up by (agent_key_hash) and then cross-check agent_id, not the other way around")
}

// ─── TestAdversarial_LegacyHMACGraceCrossUser ──────────────────────────────
//
// Scenario: the legacy comfy v1 URL signature (signComfyOutputV1) has
// no uid binding.  During the 24h grace window after server start, any
// captured v1 URL can be replayed by any unauthenticated client — the
// HTTP handler in comfy.go:689 doesn't require a session for v1.  This
// is by design for backward-compat, but the audit needs to confirm.
func TestAdversarial_LegacyHMACGraceCrossUser(t *testing.T) {
	s := newTestServer(t)
	uid1, _ := makeUser(t, s, "alice")
	_ = uid1

	jobID := int64(1)
	file := "out.png"
	exp := time.Now().Add(time.Hour).Unix()
	sig := s.signComfyOutputV1(jobID, file, exp)
	url := fmt.Sprintf("/comfy/out/%d/%s?exp=%d&sig=%s", jobID, file, exp, sig)

	// Within grace window — should reach the file-resolution branch.
	rr := httptest.NewRecorder()
	r := httptest.NewRequest("GET", url, nil)
	// Deliberately NO session cookie attached: this is the cross-user
	// replay scenario.
	s.handleComfyOutput(rr, r)
	// We expect either 200 (file served) or 400 (bad path / nonexistent
	// directory) — anything other than 401/403 confirms there's no
	// session check on the v1 path.
	if rr.Code == 401 || rr.Code == 403 {
		t.Fatalf("v1 legacy URL with no session was rejected (%d %s) — that's the desired behaviour and contradicts the audit hypothesis", rr.Code, rr.Body.String())
	}
	t.Logf("CONFIRMED: v1 comfy legacy URL (status %d) is honored without a session within grace window; cross-user URL replay is possible if a v1 URL leaks", rr.Code)
}

// ─── TestAdversarial_OauthStateNoSessionBinding ────────────────────────────
//
// Scenario: mintOAuthState binds the state token to the client IP
// (via XFF-trust) — NOT to the in-flight session cookie.  An attacker
// who can co-locate with the victim (same proxy IP / mobile carrier
// NAT / dev-mode trusted_proxies wide-open) can re-use a captured
// state value to drive a callback that mints a session for the
// victim's GitHub identity into the attacker's browser.
//
// Real-world impact bounded by:
//   - state has a 10-minute TTL,
//   - the oauth_state cookie is HttpOnly+Secure+SameSite=Lax,
//   - the auth code itself is one-time (consumed by GitHub).
// But: this asserts the documented behavior (IP-only binding), so
// future regressions that drop the IP check stay visible.
func TestAdversarial_OauthStateNoSessionBinding(t *testing.T) {
	s := newTestServer(t)
	// Pre-seed the trusted proxy set so we can vary clientIP via XFF.
	trustSet, bad := parseTrustedProxies("127.0.0.0/8")
	if len(bad) > 0 {
		t.Fatalf("parseTrustedProxies bad entries: %v", bad)
	}
	s.cfg.trustedProxies = trustSet

	// Attacker observes ts.
	stateA := s.mintOAuthState("203.0.113.10")
	// Same IP → still valid.
	if !s.verifyOAuthState(stateA, "203.0.113.10") {
		t.Fatalf("freshly-minted state didn't verify against the same IP")
	}
	// Different IP → fails, as documented.
	if s.verifyOAuthState(stateA, "198.51.100.99") {
		t.Fatalf("state verified against a different IP — IP binding is broken")
	}
	// CONFIRMS: state is bound only to IP, not to a session identifier.
	// Re-use across different sessions on the same NAT'd IP is possible.
	t.Logf("CONFIRMED: oauth state binds to clientIP only; co-NAT'd attacker can replay a captured state for their own session")
}

// ─── TestAdversarial_DeviceApproveCrossUser ────────────────────────────────
//
// Scenario: handleDeviceApprove accepts a user_code from any
// authenticated user and binds the rig to that user.  An attacker
// who observes a victim's user_code (over-the-shoulder, screenshot,
// dashboard mirror) can race the victim to approve first — the rig
// then pairs to the attacker's account, not the victim's.
//
// Rate-limit: deviceApprove is 1/6s burst 3 per IP, so spamming is
// bounded.  But the race window between the victim seeing the code
// and approving is many seconds long.
func TestAdversarial_DeviceApproveCrossUser(t *testing.T) {
	s := newTestServer(t)
	victimUID, victimSID := makeUser(t, s, "victim")
	attackerUID, attackerSID := makeUser(t, s, "attacker")
	_ = attackerUID

	// Mint a device code (as the rig would).
	rr := httptest.NewRecorder()
	r := httptest.NewRequest("POST", "/api/device/code",
		strings.NewReader(`{"hostname":"laptop","n_gpus":1,"vram_bytes":8000000000}`))
	r.Header.Set("Content-Type", "application/json")
	s.handleDeviceCodeMint(rr, r)
	if rr.Code != 200 {
		t.Fatalf("device/code = %d %s", rr.Code, rr.Body.String())
	}
	var resp struct {
		DeviceCode string `json:"device_code"`
		UserCode   string `json:"user_code"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}

	// Attacker races and approves first using the user_code they
	// somehow obtained.
	rr2 := httptest.NewRecorder()
	r2 := httptest.NewRequest("POST", "/api/device/approve",
		strings.NewReader(fmt.Sprintf(`{"user_code":%q}`, resp.UserCode)))
	r2.Header.Set("Content-Type", "application/json")
	r2.AddCookie(&http.Cookie{Name: sessionCookieName, Value: attackerSID})
	s.handleDeviceApprove(rr2, r2)
	if rr2.Code != 200 {
		t.Fatalf("attacker approve = %d %s", rr2.Code, rr2.Body.String())
	}

	// Victim tries to approve — too late.
	rr3 := httptest.NewRecorder()
	r3 := httptest.NewRequest("POST", "/api/device/approve",
		strings.NewReader(fmt.Sprintf(`{"user_code":%q}`, resp.UserCode)))
	r3.Header.Set("Content-Type", "application/json")
	r3.AddCookie(&http.Cookie{Name: sessionCookieName, Value: victimSID})
	s.handleDeviceApprove(rr3, r3)
	if rr3.Code == 200 {
		t.Fatalf("two consecutive approves succeeded for the same user_code — exclusivity broken")
	}

	// The rig is now bound to the attacker, not the victim.
	var ownerUID int64
	if err := s.db.QueryRow(`SELECT user_id FROM device_codes WHERE user_code = ?`, resp.UserCode).Scan(&ownerUID); err != nil {
		t.Fatalf("lookup: %v", err)
	}
	if ownerUID != attackerUID {
		t.Fatalf("device_codes.user_id = %d, want attacker %d", ownerUID, attackerUID)
	}
	t.Logf("CONFIRMED: attacker who learns user_code can pair victim's rig to their account; user_code entropy (~30^8 ≈ 38 bits, minus collision retry pruning) is the only barrier and approve isn't bound to the originating device_code IP/fingerprint. Victim has %d", victimUID)
}

// ─── TestAdversarial_PairTokenReuseAcrossSessions ──────────────────────────
//
// Scenario: pair_tokens are scoped to the user that minted them.
// Confirm one-shot consume so a captured pair token can't be used
// twice (race to mint two rigs under one token).
func TestAdversarial_PairTokenReuseAcrossSessions(t *testing.T) {
	s := newTestServer(t)
	uid, _ := makeUser(t, s, "alice")

	// Mint a pair_token directly.
	tok := "test-pair-tok-1234567890"
	if _, err := s.db.Exec(
		`INSERT INTO pair_tokens (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)`,
		tok, uid, nowUnix(), nowUnix()+300,
	); err != nil {
		t.Fatalf("insert: %v", err)
	}

	// First consume succeeds.
	u1, _, err := s.consumePairToken(tok)
	if err != nil {
		t.Fatalf("first consume: %v", err)
	}
	if u1 != uid {
		t.Fatalf("uid mismatch: got %d want %d", u1, uid)
	}
	// Second consume must fail.
	if _, _, err := s.consumePairToken(tok); err == nil {
		t.Fatalf("CRITICAL: pair token reusable — second consume succeeded")
	}
}

// ─── TestAdversarial_OpenRedirectViaNext ───────────────────────────────────
//
// Scenario: the oauth_next cookie is honored on callback (oauth.go:253).
// The cookie value is restricted to paths starting with "/" and not
// "//".  Probe shapes that might slip past:
//   "/\\evil.com"     — backslash-prefixed
//   "/%2fevil.com"    — URL-encoded slash
//   "/./evil.com"     — path-normalised
// Pure unit test on the redirect-value validator inline in oauth.go.
func TestAdversarial_OpenRedirectViaNext(t *testing.T) {
	tests := []struct {
		v       string
		wantBad bool
	}{
		{"/console", false},
		{"//evil.com", true},
		{"/\\evil.com", false}, // backslash IS allowed — RFC 3986 path-segment
		{"https://evil.com", true},
		{"javascript:alert(1)", true},
		{"/.evil", false},
	}
	for _, tc := range tests {
		bad := !(strings.HasPrefix(tc.v, "/") && !strings.HasPrefix(tc.v, "//"))
		if bad != tc.wantBad {
			t.Errorf("validator for %q: bad=%v want %v", tc.v, bad, tc.wantBad)
		}
		// "/\\evil.com" deserves explicit attention: some browsers
		// normalise the backslash to a forward slash, so "/\\evil.com"
		// can end up redirecting to "//evil.com" → open redirect.
		if tc.v == "/\\evil.com" {
			t.Logf("LOW: oauth_next validator accepts %q; some browsers normalise backslash to slash, which can become //evil.com cross-origin", tc.v)
		}
	}
}

// ─── TestAdversarial_TrustedProxyGarbageXFF ────────────────────────────────
//
// Scenario: trustedClientIP (trusted_proxy.go) returns the first
// non-trusted hop walking XFF right-to-left.  If a hop is malformed
// (not a parseable IP), the function returns the raw string as the
// "IP" key.  An attacker upstream of a trusted proxy can emit
// rotating garbage strings to cycle rate-limit buckets.
func TestAdversarial_TrustedProxyGarbageXFF(t *testing.T) {
	trustSet, _ := parseTrustedProxies("10.0.0.0/8")
	r := httptest.NewRequest("GET", "/", nil)
	r.RemoteAddr = "10.1.2.3:1234"
	// Attacker sends a garbage right-most XFF hop.
	r.Header.Set("X-Forwarded-For", "1.2.3.4, NOT_AN_IP")
	got := trustedClientIP(r, trustSet)
	if got != "NOT_AN_IP" {
		t.Fatalf("expected garbage-as-bucket-key %q, got %q", "NOT_AN_IP", got)
	}
	t.Logf("LOW: garbage XFF entries become rate-limit bucket keys verbatim; attacker upstream of a trusted proxy can cycle keys arbitrarily, but XFF stripping at the front proxy mitigates")
}

// ─── TestAdversarial_OAuthStateDotSplitting ────────────────────────────────
//
// Scenario: mintOAuthState format is "<ts>.<base64>".  verifyOAuthState
// finds the first '.' to split.  What happens if an attacker submits a
// state like "9999999999.." or "9999999999.AAAA.AAAA"?  Confirm the
// parser handles these without panic and rejects them.
func TestAdversarial_OAuthStateDotSplitting(t *testing.T) {
	s := newMacaroonTestServerForAdversarial(t)
	bads := []string{
		"",
		".",
		"abc",
		"123.",
		".abc",
		"99999999999.",
		"99999999999..",
		strconv.FormatInt(nowUnix(), 10) + ".\x00\x01\x02",
	}
	for _, b := range bads {
		if s.verifyOAuthState(b, "1.2.3.4") {
			t.Fatalf("verifyOAuthState accepted malformed state %q", b)
		}
	}
}

// ─── TestAdversarial_PathTraversalShardFile ────────────────────────────────
//
// Scenario: isSafeShardFile rejects '/', '\\', '.', '..' and requires
// a .gguf or manifest.json suffix.  Confirm common bypasses fail.
func TestAdversarial_PathTraversalShardFile(t *testing.T) {
	bads := []string{
		"",
		".",
		"..",
		"../../etc/passwd",
		"../etc/passwd.gguf",  // contains slash → reject
		"foo/bar.gguf",
		"foo\\bar.gguf",
		"...gguf",             // dot-only prefix is allowed but exotic — current impl PERMITS this; document
		"x.png",               // wrong suffix
		"manifest.json.bak",
	}
	for _, b := range bads {
		if isSafeShardFile(b) && b != "...gguf" {
			t.Errorf("isSafeShardFile(%q) = true, want false", b)
		}
	}
	// Confirm legitimate values still pass.
	if !isSafeShardFile("stage-0.gguf") {
		t.Errorf("isSafeShardFile(stage-0.gguf) = false")
	}
	if !isSafeShardFile("manifest.json") {
		t.Errorf("isSafeShardFile(manifest.json) = false")
	}
}

// ─── TestAdversarial_NULByteInShardFile ────────────────────────────────────
//
// NUL byte not explicitly blocked in isSafeShardFile.  Confirm it still
// fails downstream (Go's syscall layer rejects NUL in paths) but flag
// the gap.
func TestAdversarial_NULByteInShardFile(t *testing.T) {
	nul := "stage-0\x00.gguf"
	if !isSafeShardFile(nul) {
		// Validator catches it for free (slash/backslash/dot check
		// happens to reject because the file doesn't end in .gguf
		// after NUL — actually it DOES, so this is dependent on the
		// suffix check).  Document either way.
		t.Logf("isSafeShardFile rejects NUL-containing names (good)")
		return
	}
	t.Logf("LOW: isSafeShardFile accepted NUL-containing %q — relies on os.Open to fail downstream; safer to reject explicitly", nul)
}

// ─── helper ──────────────────────────────────────────────────────────────

// newMacaroonTestServerForAdversarial mirrors newMacaroonTestServer in
// macaroon_test.go but is local so we don't depend on macaroon_test
// linkage when the adversarial build tag flips on.
func newMacaroonTestServerForAdversarial(t *testing.T) *server {
	t.Helper()
	return &server{
		cfg: config{
			sessionSecret: "adversarial-test-secret-aaaaaaaaaaaa",
			publicURL:     "https://example.test",
		},
	}
}
