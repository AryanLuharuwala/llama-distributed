package main

import (
	"bufio"
	"context"
	"crypto/rand"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"log"
	"net"
	"net/http"
	"strconv"
	"sync"
	"time"
)

// ─── Server ─────────────────────────────────────────────────────────────────

type server struct {
	cfg config
	db  *sql.DB
	hub *hub
}

func newServer(cfg config, db *sql.DB) *server {
	return &server{cfg: cfg, db: db, hub: newHub()}
}

func (s *server) router() http.Handler {
	mux := http.NewServeMux()

	// UI
	mux.HandleFunc("GET /", s.handleIndex)

	// Auth
	mux.HandleFunc("GET /auth/github", s.handleGithubStart)
	mux.HandleFunc("GET /auth/github/callback", s.handleGithubCallback)
	mux.HandleFunc("POST /auth/logout", s.handleLogout)
	if s.cfg.devMode {
		mux.HandleFunc("POST /auth/dev", s.handleDevLogin)
	}

	// Pairing (browser mints, agent consumes)
	mux.HandleFunc("POST /api/pair", s.handlePairMint)

	// Sessions / me
	mux.HandleFunc("GET /api/me", s.handleMe)
	mux.HandleFunc("GET /api/rigs", s.handleListRigs)

	// Inference + usage
	mux.HandleFunc("POST /api/infer", s.handleInfer)
	mux.HandleFunc("POST /api/infer_pp", s.handleInferPP)
	mux.HandleFunc("GET /api/usage", s.handleUsage)
	mux.HandleFunc("GET /api/inference_log", s.handleInferenceLog)

	// P2P signalling (WebRTC offer/answer relay)
	mux.HandleFunc("POST /api/signal", s.handleSignal)

	// Models + shard server
	mux.HandleFunc("POST /api/models", s.handleRegisterModel)
	mux.HandleFunc("GET /api/models", s.handleListModels)
	mux.HandleFunc("GET /models/{id}/manifest.json", s.handleModelManifest)
	mux.HandleFunc("GET /models/{id}/shards/{file}", s.handleShardDownload)

	// Pools
	mux.HandleFunc("POST /api/pools", s.handleCreatePool)
	mux.HandleFunc("GET /api/pools", s.handleListPools)
	mux.HandleFunc("GET /api/pools/{id}", s.handlePoolDetail)
	mux.HandleFunc("POST /api/pools/{id}/invite", s.handlePoolInvite)
	mux.HandleFunc("POST /api/pools/join", s.handlePoolJoin)
	mux.HandleFunc("POST /api/pools/{id}/rigs", s.handlePoolAttachRig)
	mux.HandleFunc("DELETE /api/pools/{id}/rigs/{rigID}", s.handlePoolDetachRig)

	// WebSocket endpoints
	mux.HandleFunc("GET /ws/browser", s.handleBrowserWS)
	mux.HandleFunc("GET /ws/agent", s.handleAgentWS)
	mux.HandleFunc("GET /ws/client", s.handleClientWS)

	// Health
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte("ok"))
	})

	return withLogging(mux)
}

// ─── Helpers ────────────────────────────────────────────────────────────────

func newRandomToken(nBytes int) string {
	b := make([]byte, nBytes)
	if _, err := rand.Read(b); err != nil {
		panic(err)
	}
	return hex.EncodeToString(b)
}

func nowUnix() int64 { return time.Now().Unix() }

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

func withLogging(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t0 := time.Now()
		sr := &statusRecorder{ResponseWriter: w, status: 200}
		h.ServeHTTP(sr, r)
		log.Printf("%s %s -> %d in %s",
			r.Method, r.URL.Path, sr.status, time.Since(t0))
	})
}

type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (r *statusRecorder) WriteHeader(code int) {
	r.status = code
	r.ResponseWriter.WriteHeader(code)
}

// Hijack passes through to the underlying ResponseWriter so that
// WebSocket upgrades (which require http.Hijacker) continue to work
// when this middleware is in the chain.
func (r *statusRecorder) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	h, ok := r.ResponseWriter.(http.Hijacker)
	if !ok {
		return nil, nil, http.ErrNotSupported
	}
	return h.Hijack()
}

func (r *statusRecorder) Flush() {
	if f, ok := r.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// ─── Sessions ───────────────────────────────────────────────────────────────

const sessionCookieName = "distpool_session"
const sessionTTL = 30 * 24 * time.Hour

type user struct {
	ID          int64
	GithubLogin string
	DisplayName string
}

func (s *server) createSession(userID int64) (string, time.Time, error) {
	sid := newRandomToken(32)
	expires := time.Now().Add(sessionTTL)
	_, err := s.db.Exec(
		`INSERT INTO sessions (id, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)`,
		sid, userID, nowUnix(), expires.Unix(),
	)
	if err != nil {
		return "", time.Time{}, err
	}
	return sid, expires, nil
}

func (s *server) userFromRequest(r *http.Request) (*user, bool) {
	c, err := r.Cookie(sessionCookieName)
	if err != nil || c.Value == "" {
		return nil, false
	}
	var (
		uid   int64
		exp   int64
	)
	err = s.db.QueryRow(
		`SELECT user_id, expires_at FROM sessions WHERE id = ?`,
		c.Value,
	).Scan(&uid, &exp)
	if err != nil || exp < nowUnix() {
		return nil, false
	}
	u := &user{}
	var gh sql.NullString
	err = s.db.QueryRow(
		`SELECT id, github_login, display_name FROM users WHERE id = ?`,
		uid,
	).Scan(&u.ID, &gh, &u.DisplayName)
	if err != nil {
		return nil, false
	}
	u.GithubLogin = gh.String
	return u, true
}

func (s *server) setSessionCookie(w http.ResponseWriter, sid string, expires time.Time) {
	http.SetCookie(w, &http.Cookie{
		Name:     sessionCookieName,
		Value:    sid,
		Path:     "/",
		HttpOnly: true,
		Expires:  expires,
		SameSite: http.SameSiteLaxMode,
	})
}

func (s *server) clearSessionCookie(w http.ResponseWriter) {
	http.SetCookie(w, &http.Cookie{
		Name:   sessionCookieName,
		Value:  "",
		Path:   "/",
		MaxAge: -1,
	})
}

// ─── Handlers: misc ─────────────────────────────────────────────────────────

func (s *server) handleIndex(w http.ResponseWriter, _ *http.Request) {
	b, err := uiFS.ReadFile("ui.html")
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write(b)
}

func (s *server) handleMe(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	writeJSON(w, 200, map[string]any{
		"id":           u.ID,
		"display_name": u.DisplayName,
		"github_login": u.GithubLogin,
	})
}

func (s *server) handleListRigs(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	rows, err := s.db.Query(
		`SELECT agent_id, hostname, n_gpus, vram_bytes, last_seen
		 FROM rigs WHERE user_id = ? ORDER BY last_seen DESC`,
		u.ID,
	)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	defer rows.Close()
	type rig struct {
		AgentID   string `json:"agent_id"`
		Hostname  string `json:"hostname"`
		NGPUs     int    `json:"n_gpus"`
		VRAMBytes int64  `json:"vram_bytes"`
		LastSeen  int64  `json:"last_seen"`
		Online    bool   `json:"online"`
	}
	var rigs []rig
	for rows.Next() {
		var r rig
		if err := rows.Scan(&r.AgentID, &r.Hostname, &r.NGPUs, &r.VRAMBytes, &r.LastSeen); err != nil {
			continue
		}
		r.Online = s.hub.agentOnline(u.ID, r.AgentID)
		rigs = append(rigs, r)
	}
	writeJSON(w, 200, map[string]any{"rigs": rigs})
}

// Dev login — POST {display_name} — only enabled when DIST_GITHUB_CLIENT is unset.
func (s *server) handleDevLogin(w http.ResponseWriter, r *http.Request) {
	var body struct {
		DisplayName string `json:"display_name"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body.DisplayName == "" {
		writeErr(w, 400, "display_name required")
		return
	}
	res, err := s.db.Exec(
		`INSERT INTO users (github_login, display_name, created_at) VALUES (NULL, ?, ?)`,
		body.DisplayName, nowUnix(),
	)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	uid, _ := res.LastInsertId()
	sid, exp, err := s.createSession(uid)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	s.setSessionCookie(w, sid, exp)
	writeJSON(w, 200, map[string]any{"id": uid, "display_name": body.DisplayName})
}

func (s *server) handleLogout(w http.ResponseWriter, r *http.Request) {
	if c, err := r.Cookie(sessionCookieName); err == nil {
		_, _ = s.db.Exec(`DELETE FROM sessions WHERE id = ?`, c.Value)
	}
	s.clearSessionCookie(w)
	writeJSON(w, 200, map[string]string{"ok": "true"})
}

// ─── Handlers: pairing ──────────────────────────────────────────────────────

// POST /api/pair — logged-in browser asks the server for a short-lived token
// to embed in a distpool:// deep link.
func (s *server) handlePairMint(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	token := newRandomToken(20)
	expires := time.Now().Add(5 * time.Minute)
	_, err := s.db.Exec(
		`INSERT INTO pair_tokens (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)`,
		token, u.ID, nowUnix(), expires.Unix(),
	)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	// Public URL the agent will WebSocket to.
	// If cfg.publicURL starts with http://, the WS endpoint is ws://; https→wss.
	wsURL := httpToWS(s.cfg.publicURL) + "/ws/agent"
	deepLink := "distpool://pair?token=" + token + "&server=" + wsURL
	writeJSON(w, 200, map[string]any{
		"token":      token,
		"deep_link":  deepLink,
		"expires_at": expires.Unix(),
	})
}

func httpToWS(u string) string {
	if len(u) > 7 && u[:7] == "http://" {
		return "ws://" + u[7:]
	}
	if len(u) > 8 && u[:8] == "https://" {
		return "wss://" + u[8:]
	}
	return u
}

// consumePairToken returns the user_id that minted `token`, deletes it (one-shot),
// or returns 0 if the token is unknown, expired, or already used.
func (s *server) consumePairToken(token string) (int64, error) {
	var uid, exp int64
	err := s.db.QueryRow(
		`SELECT user_id, expires_at FROM pair_tokens
		 WHERE token = ? AND used_at IS NULL`,
		token,
	).Scan(&uid, &exp)
	if err != nil {
		return 0, err
	}
	if exp < nowUnix() {
		return 0, errPairExpired
	}
	if _, err := s.db.Exec(
		`UPDATE pair_tokens SET used_at = ? WHERE token = ?`,
		nowUnix(), token,
	); err != nil {
		return 0, err
	}
	return uid, nil
}

var errPairExpired = &pairErr{"pair token expired"}

type pairErr struct{ msg string }

func (e *pairErr) Error() string { return e.msg }

// ─── Reap loop ──────────────────────────────────────────────────────────────

func (s *server) reapLoop(ctx context.Context) {
	t := time.NewTicker(1 * time.Minute)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			_, _ = s.db.Exec(`DELETE FROM sessions    WHERE expires_at < ?`, nowUnix())
			_, _ = s.db.Exec(`DELETE FROM pair_tokens WHERE expires_at < ?`, nowUnix())
		}
	}
}

// ─── Hub ────────────────────────────────────────────────────────────────────
//
// Tracks live WebSocket connections so the browser-side can subscribe to events
// from agents owned by the same user.

type hub struct {
	mu       sync.RWMutex
	agents   map[agentKey]*agentConn    // (userID, agentID) -> conn
	browsers map[int64][]*browserConn   // userID -> connections
}

type agentKey struct {
	userID  int64
	agentID string
}

func newHub() *hub {
	return &hub{
		agents:   make(map[agentKey]*agentConn),
		browsers: make(map[int64][]*browserConn),
	}
}

func (h *hub) registerAgent(a *agentConn) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.agents[agentKey{a.userID, a.agentID}] = a
}

func (h *hub) unregisterAgent(a *agentConn) {
	h.mu.Lock()
	defer h.mu.Unlock()
	k := agentKey{a.userID, a.agentID}
	if cur, ok := h.agents[k]; ok && cur == a {
		delete(h.agents, k)
	}
}

func (h *hub) agentOnline(userID int64, agentID string) bool {
	h.mu.RLock()
	defer h.mu.RUnlock()
	_, ok := h.agents[agentKey{userID, agentID}]
	return ok
}

// findAgent returns the live agentConn for (userID, agentID) if online.
func (h *hub) findAgent(userID int64, agentID string) (*agentConn, bool) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	a, ok := h.agents[agentKey{userID, agentID}]
	return a, ok
}

// countOnlineRigsInPool counts how many pool_rigs are currently online.
func (s *server) countOnlineRigsInPool(poolID int64) int {
	rows, err := s.db.Query(`
		SELECT r.user_id, r.agent_id FROM pool_rigs pr
		JOIN rigs r ON r.id = pr.rig_id WHERE pr.pool_id = ?`, poolID)
	if err != nil {
		return 0
	}
	defer rows.Close()
	n := 0
	for rows.Next() {
		var uid int64
		var aid string
		if err := rows.Scan(&uid, &aid); err != nil {
			continue
		}
		if s.hub.agentOnline(uid, aid) {
			n++
		}
	}
	return n
}

// pickOnlineRigInPool returns a live agentConn in the pool (for relay target
// selection).  Returns (nil, false) if none are online.
func (s *server) pickOnlineRigInPool(poolID int64) (*agentConn, bool) {
	rows, err := s.db.Query(`
		SELECT r.user_id, r.agent_id FROM pool_rigs pr
		JOIN rigs r ON r.id = pr.rig_id WHERE pr.pool_id = ?`, poolID)
	if err != nil {
		return nil, false
	}
	defer rows.Close()
	for rows.Next() {
		var uid int64
		var aid string
		if err := rows.Scan(&uid, &aid); err != nil {
			continue
		}
		if a, ok := s.hub.findAgent(uid, aid); ok {
			return a, true
		}
	}
	return nil, false
}

func (h *hub) registerBrowser(b *browserConn) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.browsers[b.userID] = append(h.browsers[b.userID], b)
}

func (h *hub) unregisterBrowser(b *browserConn) {
	h.mu.Lock()
	defer h.mu.Unlock()
	bs := h.browsers[b.userID]
	for i, x := range bs {
		if x == b {
			h.browsers[b.userID] = append(bs[:i], bs[i+1:]...)
			break
		}
	}
}

func (h *hub) broadcastToUser(userID int64, kind string, payload any) {
	h.mu.RLock()
	bs := append([]*browserConn(nil), h.browsers[userID]...)
	h.mu.RUnlock()
	msg := map[string]any{"kind": kind, "payload": payload}
	for _, b := range bs {
		b.send(msg)
	}
}

// ─── Debug ──────────────────────────────────────────────────────────────────

var _ = strconv.Itoa // silence unused import when handlers grow
