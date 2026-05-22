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
	"encoding/json"
	"fmt"
	"math/big"
	"net/http"
	"strings"
	"time"
)

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
	_ = json.NewDecoder(r.Body).Decode(&body)

	deviceCode := newRandomToken(24)
	// Retry a few times in case of user_code collision (UNIQUE constraint).
	var userCode string
	now := nowUnix()
	exp := now + int64(deviceCodeTTL.Seconds())
	for i := 0; i < 5; i++ {
		userCode = newUserCode()
		_, err := s.db.Exec(
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

	scheme := "https"
	if r.TLS == nil && !strings.HasPrefix(r.Host, "localhost") {
		scheme = "https" // ACA terminates TLS; r.TLS is nil but we're behind https
	}
	if strings.HasPrefix(r.Host, "localhost") || strings.HasPrefix(r.Host, "127.") {
		scheme = "http"
	}
	verification := fmt.Sprintf("%s://%s/device", scheme, r.Host)

	writeJSON(w, 200, map[string]any{
		"device_code":              deviceCode,
		"user_code":                userCode,
		"verification_url":         verification,
		"verification_url_complete": verification + "?code=" + userCode,
		"expires_in":               int(deviceCodeTTL.Seconds()),
		"interval":                 deviceCodePollMin,
	})
}

// POST /api/device/approve  (authenticated)
// Body: { "user_code":"ABCD-1234" }
// Marks the row approved + binds the current user_id, mints an agent_key
// and a stable agent_id, and inserts/updates the rigs row.
func (s *server) handleDeviceApprove(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	var body struct {
		UserCode string `json:"user_code"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body.UserCode == "" {
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
	err := s.db.QueryRow(
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

	if _, err := tx.Exec(
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
	// before the rig connects via WS.
	if _, err := tx.Exec(`INSERT INTO rigs
		(user_id, agent_id, hostname, n_gpus, vram_bytes, last_seen, agent_key)
		VALUES (?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT (user_id, agent_id) DO UPDATE SET
		   hostname   = excluded.hostname,
		   n_gpus     = excluded.n_gpus,
		   vram_bytes = excluded.vram_bytes,
		   agent_key  = excluded.agent_key`,
		u.ID, agentID, hostname, nGPUs, vramBytes, nowUnix(), agentKey,
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
	var body struct {
		DeviceCode string `json:"device_code"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body.DeviceCode == "" {
		writeErr(w, 400, "device_code required")
		return
	}

	var (
		approved  int
		expiresAt int64
		agentID   *string
		agentKey  *string
	)
	err := s.db.QueryRow(
		`SELECT approved, expires_at, agent_id, agent_key
		   FROM device_codes WHERE device_code = ?`, body.DeviceCode,
	).Scan(&approved, &expiresAt, &agentID, &agentKey)
	if err != nil {
		writeErr(w, 404, "unknown device_code")
		return
	}
	if nowUnix() > expiresAt {
		writeErr(w, 410, "code expired")
		return
	}
	if approved == 0 {
		// RFC 8628 says 'authorization_pending'.  We use 428 so curl
		// clients see a clear "not yet" without it looking like a hard
		// error.
		writeJSON(w, 428, map[string]any{"status": "pending"})
		return
	}

	wsHost := r.Host
	scheme := "wss"
	if strings.HasPrefix(wsHost, "localhost") || strings.HasPrefix(wsHost, "127.") {
		scheme = "ws"
	}
	server := fmt.Sprintf("%s://%s/ws/agent", scheme, wsHost)

	writeJSON(w, 200, map[string]any{
		"agent_id":  *agentID,
		"agent_key": *agentKey,
		"server":    server,
	})
}

// GET /device  (HTML page; resolves via the dashboard's UI)
// Falls through to the main UI which detects ?code=… and shows the
// "Approve this rig?" confirmation card.
func (s *server) handleDevicePage(w http.ResponseWriter, r *http.Request) {
	// Reuse the same HTML — the UI checks location.pathname/search.
	s.handleIndex(w, r)
}

func strFallback(s, d string) string {
	if s == "" {
		return d
	}
	return s
}
