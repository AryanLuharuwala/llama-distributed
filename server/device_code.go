package main

// Device code flow (RFC 8628 in spirit).  Used by `dist-node login` so a rig
// can self-pair without a copy-paste pair URL from the dashboard.
//
//   1. Rig hits POST /api/device/code → server mints (device_code, user_code,
//      verification_url) and persists a row in device_codes.
//   2. Rig prints the verification_url + user_code, opens the page in a
//      browser, and starts polling POST /api/device/token.
//   3. User (logged in to the dashboard) confirms the user_code via
//      POST /api/device/approve, which fills in user_id + approved=1 and
//      mints a long-lived agent_key bound to the rig row.
//   4. The next /api/device/token poll returns {agent_id, agent_key, server}
//      which the rig stashes locally and uses for all future WS resumes.

import (
	"crypto/rand"
	_ "embed"
	"encoding/json"
	"fmt"
	"io"
	"math/big"
	"net/http"
	"net/url"
	"strings"
	"time"
)

//go:embed assets/device.html
var devicePageHTML []byte

const (
	deviceCodeTTL      = 10 * time.Minute
	deviceCodePollMin  = 3 // seconds
	deviceUserCodeSize = 8 // chars: ABCD-1234 = 9 with the dash
)

// userCode returns an 8-char base32-like string formatted as "XXXX-XXXX".
// We use a Crockford-ish alphabet (no I/L/O/0/1) so codes are easy to type.
func newUserCode() string {
	const alphabet = "ABCDEFGHJKMNPQRSTVWXYZ23456789"
	var b strings.Builder
	for i := 0; i < deviceUserCodeSize; i++ {
		if i == 4 {
			b.WriteByte('-')
		}
		n, _ := rand.Int(rand.Reader, big.NewInt(int64(len(alphabet))))
		b.WriteByte(alphabet[n.Int64()])
	}
	return b.String()
}

// POST /api/device/code  (public)
// Body: { "hostname":"…", "n_gpus":N, "vram_bytes":N } — same shape as the
// rig's hello frame; we use it to seed the rig row at approval time so the
// dashboard can show meaningful info even before the rig connects.
func (s *server) handleDeviceCodeMint(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Hostname  string `json:"hostname"`
		NGPUs     int    `json:"n_gpus"`
		VRAMBytes int64  `json:"vram_bytes"`
	}
	_ = json.NewDecoder(io.LimitReader(r.Body, 4<<10)).Decode(&body)

	deviceCode := newRandomToken(24)
	// Retry a few times in case of user_code collision (UNIQUE constraint).
	var userCode string
	now := nowUnix()
	exp := now + int64(deviceCodeTTL.Seconds())
	for i := 0; i < 5; i++ {
		userCode = newUserCode()
		_, err := s.dbExec(
			`INSERT INTO device_codes
			   (device_code, user_code, hostname, n_gpus, vram_bytes,
			    created_at, expires_at)
			 VALUES (?, ?, ?, ?, ?, ?, ?)`,
			deviceCode, userCode, body.Hostname, body.NGPUs, body.VRAMBytes,
			now, exp,
		)
		if err == nil {
			break
		}
		if i == 4 {
			writeErr(w, 500, err.Error())
			return
		}
	}

	// Use the canonical publicURL — NOT r.Host — to defeat Host-header
	// injection.  An attacker who can set r.Host would otherwise control the
	// verification URL printed by every rig (phishing).  publicURL is
	// configured at boot from a trusted source.
	verification := strings.TrimRight(s.cfg.publicURL, "/") + "/device"

	writeJSON(w, 200, map[string]any{
		"device_code":               deviceCode,
		"user_code":                 userCode,
		"verification_url":          verification,
		"verification_url_complete": verification + "?code=" + userCode,
		"expires_in":                int(deviceCodeTTL.Seconds()),
		"interval":                  deviceCodePollMin,
	})
}

// POST /api/device/approve  (authenticated)
// Body: { "user_code":"ABCD-1234" }
// Marks the row approved + binds the current user_id, mints an agent_key
// and a stable agent_id, and inserts/updates the rigs row.
func (s *server) handleDeviceApprove(w http.ResponseWriter, r *http.Request) {
	if s.ipRL != nil && !s.ipRL.deviceApprove.allow(s.remoteIPForRateLimit(r)) {
		writeErr(w, 429, "too many approval attempts — slow down")
		return
	}
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	var body struct {
		UserCode string `json:"user_code"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<10)).Decode(&body); err != nil || body.UserCode == "" {
		writeErr(w, 400, "user_code required")
		return
	}
	body.UserCode = strings.ToUpper(strings.TrimSpace(body.UserCode))

	var (
		id        int64
		approved  int
		expiresAt int64
		hostname  string
		nGPUs     int
		vramBytes int64
	)
	err := s.dbQueryRow(
		`SELECT id, approved, expires_at, hostname, n_gpus, vram_bytes
		 FROM device_codes WHERE user_code = ?`, body.UserCode,
	).Scan(&id, &approved, &expiresAt, &hostname, &nGPUs, &vramBytes)
	if err != nil {
		writeErr(w, 404, "unknown code")
		return
	}
	if nowUnix() > expiresAt {
		writeErr(w, 410, "code expired — run `dist-node login` again")
		return
	}
	if approved == 1 {
		writeErr(w, 409, "code already approved")
		return
	}

	// Mint a stable agent_id + a fresh agent_key bound to this user.
	agentID := fmt.Sprintf("%s:%s", strFallback(hostname, "rig"), newRandomToken(4))
	agentKey := newRandomToken(24)

	tx, err := s.db.Begin()
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	defer tx.Rollback()

	if _, err := s.txExec(tx,
		`UPDATE device_codes
		    SET approved = 1, user_id = ?, agent_id = ?, agent_key = ?,
		        approved_at = ?
		  WHERE id = ?`,
		u.ID, agentID, agentKey, nowUnix(), id,
	); err != nil {
		writeErr(w, 500, err.Error())
		return
	}

	// Pre-create the rig row so the dashboard shows it immediately, even
	// before the rig connects via WS.  We store only the hash here; the
	// plaintext lives in device_codes.agent_key for exactly one polled
	// fetch (see handleDeviceToken below, which nulls it out on read).
	//
	// dist-cli (operator seat) posts n_gpus=-1 to mean "I'm not a compute
	// rig, don't touch capabilities".  COALESCE(NULLIF(...,-1), existing)
	// keeps the row's prior n_gpus/vram_bytes when the sentinel is set, so
	// an operator login on the same host as a compute node can't erase its
	// advertised GPU count.
	akHash := hashAgentKey(agentKey)
	if _, err := s.txExec(tx, `INSERT INTO rigs
		(user_id, agent_id, hostname, n_gpus, vram_bytes, last_seen, agent_key, agent_key_hash)
		VALUES (?, ?, ?, COALESCE(NULLIF(?, -1), 0), COALESCE(NULLIF(?, -1), 0), ?, '', ?)
		ON CONFLICT (user_id, agent_id) DO UPDATE SET
		   hostname       = excluded.hostname,
		   n_gpus         = COALESCE(NULLIF(excluded.n_gpus, 0),     rigs.n_gpus),
		   vram_bytes     = COALESCE(NULLIF(excluded.vram_bytes, 0), rigs.vram_bytes),
		   agent_key      = '',
		   agent_key_hash = excluded.agent_key_hash`,
		u.ID, agentID, hostname, nGPUs, vramBytes, nowUnix(), akHash,
	); err != nil {
		writeErr(w, 500, err.Error())
		return
	}

	if err := tx.Commit(); err != nil {
		writeErr(w, 500, err.Error())
		return
	}

	writeJSON(w, 200, map[string]any{
		"ok":       true,
		"agent_id": agentID,
		"hostname": hostname,
	})
}

// POST /api/device/token  (public, polled by the rig)
// Body: { "device_code":"…" }
// Returns 428 (precondition_required) until approved, then 200 with
// {agent_id, agent_key, server}.
func (s *server) handleDeviceToken(w http.ResponseWriter, r *http.Request) {
	if s.ipRL != nil && !s.ipRL.devicePoll.allow(s.remoteIPForRateLimit(r)) {
		writeErr(w, 429, "poll too fast — back off")
		return
	}
	var body struct {
		DeviceCode string `json:"device_code"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<10)).Decode(&body); err != nil || body.DeviceCode == "" {
		writeErr(w, 400, "device_code required")
		return
	}

	var (
		approved  int
		expiresAt int64
		agentID   *string
		agentKey  *string
	)
	err := s.dbQueryRow(
		`SELECT approved, expires_at, agent_id, agent_key
		   FROM device_codes WHERE device_code = ?`, body.DeviceCode,
	).Scan(&approved, &expiresAt, &agentID, &agentKey)
	if err != nil {
		// Unify the "unknown code" and "expired code" responses: both are
		// terminal for the client and disclosing the difference lets an
		// attacker enumerate valid (but unapproved/unconsumed) device_codes
		// from a leaked log or a 410 vs 404 timing side-channel.  device_code
		// is 24 bytes of randomness so the enumeration window is small, but
		// the unified response is free defence-in-depth.
		writeErr(w, 410, "device_code invalid or expired")
		return
	}
	if nowUnix() > expiresAt {
		writeErr(w, 410, "device_code invalid or expired")
		return
	}
	if approved == 0 {
		// RFC 8628 says 'authorization_pending'.  We use 428 so curl
		// clients see a clear "not yet" without it looking like a hard
		// error.
		writeJSON(w, 428, map[string]any{"status": "pending"})
		return
	}

	// Consume-once: null out the plaintext column the moment the rig
	// fetches it.  A second poll for the same device_code returns 410.
	// We do the CAS-style UPDATE … WHERE agent_key IS NOT NULL so two
	// concurrent polls race to read-and-null exactly once.
	res, err := s.dbExec(
		`UPDATE device_codes SET agent_key = NULL
		   WHERE device_code = ? AND agent_key IS NOT NULL`,
		body.DeviceCode,
	)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	if n, _ := res.RowsAffected(); n == 0 {
		// Someone else already consumed it (or it was never set).  Use the
		// same "invalid or expired" wording as the lookup-miss path so the
		// client can't distinguish a stolen-by-a-race code from one that
		// never existed.
		writeErr(w, 410, "device_code invalid or expired")
		return
	}

	// Like verification_url above: use publicURL (trusted), not r.Host
	// (attacker-controllable).  A rig that's poisoned here would reconnect
	// to the attacker's WS endpoint forever.
	server := httpToWS(strings.TrimRight(s.cfg.publicURL, "/")) + "/ws/agent"

	writeJSON(w, 200, map[string]any{
		"agent_id":  *agentID,
		"agent_key": *agentKey,
		"server":    server,
	})
}

// GET /device  (HTML page; resolves via the device-pair approval UI)
//
// The page handles the ?code=XXXX-XXXX deeplink the CLI shows: it
// fetches the pair preview, asks the operator to confirm, and POSTs
// the approval.  Unauthed callers bounce through /auth with a `next`
// pointer back to /device so the code survives the OAuth round-trip.
func (s *server) handleDevicePage(w http.ResponseWriter, r *http.Request) {
	if _, ok := s.userFromRequest(r); !ok {
		next := "/device"
		if q := r.URL.RawQuery; q != "" {
			next = "/device?" + q
		}
		http.Redirect(w, r, "/auth?next="+url.QueryEscape(next), http.StatusFound)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	_, _ = w.Write(devicePageHTML)
}

func strFallback(s, d string) string {
	if s == "" {
		return d
	}
	return s
}

// agentFromHeader resolves an agent_key (passed as Bearer or X-Agent-Key) to
// (user, agent_id).  Used by /api/agent/* endpoints so a rig can self-serve
// API keys + pool URLs without holding a browser session.
func (s *server) agentFromRequest(r *http.Request) (uid int64, agentID string, ok bool) {
	key := r.Header.Get("X-Agent-Key")
	if key == "" {
		if b := bearerFromRequest(r); strings.HasPrefix(b, "ak-") || (b != "" && !strings.HasPrefix(b, "sk-")) {
			key = b
		}
	}
	if key == "" {
		return 0, "", false
	}
	// Look up by hash, not plaintext — the indexed agent_key_hash column
	// is the canonical bearer mapping post-A1.  Falls back to the plaintext
	// column only for rows that the boot-time backfill hasn't touched yet
	// (e.g. an in-flight reconnect on a freshly-migrated DB).
	hash := hashAgentKey(key)
	if err := s.dbQueryRow(
		`SELECT user_id, agent_id FROM rigs WHERE agent_key_hash = ? LIMIT 1`, hash,
	).Scan(&uid, &agentID); err == nil {
		return uid, agentID, true
	}
	if err := s.dbQueryRow(
		`SELECT user_id, agent_id FROM rigs
		   WHERE (agent_key_hash IS NULL OR agent_key_hash = '')
		     AND agent_key = ? LIMIT 1`, key,
	).Scan(&uid, &agentID); err != nil {
		return 0, "", false
	}
	return uid, agentID, true
}

// POST /api/agent/api_key   { "label":"…" }
// Header: Authorization: Bearer <agent_key>   (or X-Agent-Key: <agent_key>)
// Mints a sk-dist-* API key bound to the user that owns the agent.  Used by
// `dist-node url` so a freshly-logged-in rig can self-serve a Bearer token
// for /v1/chat/completions without anyone touching the dashboard.
func (s *server) handleAgentMintAPIKey(w http.ResponseWriter, r *http.Request) {
	uid, agentID, ok := s.agentFromRequest(r)
	if !ok {
		writeErr(w, 401, "bad or missing agent_key")
		return
	}
	var body struct {
		Label string `json:"label"`
	}
	// Cap the request body so an agent_key holder can't post a multi-GiB
	// label string. Matches the convention applied to other auth'd JSON
	// endpoints (BUGS_FOUND #3); 8 KiB is generous for a label.
	_ = json.NewDecoder(io.LimitReader(r.Body, 8*1024)).Decode(&body)
	if len(body.Label) > 256 {
		body.Label = body.Label[:256]
	}
	if body.Label == "" {
		body.Label = "rig:" + agentID
	}
	plain, id, err := s.mintAPIKey(uid, body.Label)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	writeJSON(w, 200, map[string]any{
		"id":     id,
		"key":    plain,
		"prefix": plain[:12],
		"label":  body.Label,
	})
}

// GET /api/agent/pools
// Returns the pools this agent's user can use for /v1/* — both pools the user
// owns and any pool whose member list includes them.  Each row carries the
// canonical base_url for the OpenAI-compat endpoint.
func (s *server) handleAgentListPools(w http.ResponseWriter, r *http.Request) {
	uid, _, ok := s.agentFromRequest(r)
	if !ok {
		writeErr(w, 401, "bad or missing agent_key")
		return
	}
	rows, err := s.dbQuery(`
		SELECT DISTINCT p.id, p.name, COALESCE(p.slug,''), p.visibility
		  FROM pools p
		  LEFT JOIN pool_members m ON m.pool_id = p.id
		 WHERE p.owner_id = ? OR m.user_id = ?
		 ORDER BY p.id`,
		uid, uid)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	defer rows.Close()
	type poolOut struct {
		ID         int64  `json:"id"`
		Name       string `json:"name"`
		Slug       string `json:"slug"`
		Visibility string `json:"visibility"`
		BaseURL    string `json:"base_url"`
	}
	scheme := "https"
	if r.TLS == nil && (strings.HasPrefix(r.Host, "localhost") || strings.HasPrefix(r.Host, "127.")) {
		scheme = "http"
	}
	var out []poolOut
	for rows.Next() {
		var p poolOut
		if err := rows.Scan(&p.ID, &p.Name, &p.Slug, &p.Visibility); err == nil {
			ref := p.Slug
			if ref == "" {
				ref = fmt.Sprintf("%d", p.ID)
			}
			p.BaseURL = fmt.Sprintf("%s://%s/v1/%s", scheme, r.Host, ref)
			out = append(out, p)
		}
	}
	writeJSON(w, 200, map[string]any{"pools": out})
}
