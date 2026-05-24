package main

import (
	"bufio"
	"context"
	"crypto/rand"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// ─── Server ─────────────────────────────────────────────────────────────────

type server struct {
	cfg     config
	db      *sql.DB
	dialect sqlDialect // set in main() after openDB; defaults to SQLite for tests via newServer
	hub     *hub

	// Pending last_seen updates, batched to cut SQLite WAL contention under
	// load.  Key is (user_id, agent_id); value is the latest unix timestamp
	// we observed on that agent's status frame.  Flushed every 1s by
	// reapLoop into a single multi-row UPDATE.
	lastSeenMu sync.Mutex
	lastSeen   map[agentKey]int64

	// HF importer: in-flight job cancellation registry.
	hfJobs *importJobs

	// ComfyUI jobs: in-flight image/video generation jobs.
	comfyJobs *comfyJobs

	// Active relay assignments — populated when pp_route sends
	// p2p_relay_assign, drained on release or relay-rig disconnect.  Used
	// to attribute bytes back to the right reputation row and to detect
	// mid-session failures.
	relays *activeRelays

	// Per-pool rig-cost snapshot cache.  See cost_cache.go for the
	// rationale; short TTL trades up to ~2s of plan-staleness for a big
	// reduction in DB load on the planPipeline / planDPP hot paths.
	costCache *rigCostCache

	// Process start time — surfaced via /metrics for uptime tracking and
	// used by /readyz to ignore the initial warm-up window when SQLite
	// might still be migrating.
	startedAt time.Time

	// Per-IP token-bucket limiters for endpoints reached before auth
	// (device-code approve/poll, oauth_start, hello-fail).  Constructed
	// in newServer; consumed by handleDeviceApprove + handleDeviceToken
	// + handleGithubStart + the ws auth path.
	ipRL *ipRateLimiterSet

	// Per-rig billing-drift estimator.  Records (reported, server-cap)
	// from each settled inference; once 100 samples accumulate, rigs
	// that over-report by >5% get flagged in rig_quarantine and stop
	// earning reputation credit.  See tokenization.go.
	drift *rigDriftTable

	// Live MCP broker connections, keyed by (uid, server_id).  Lazy-init
	// on first call; janitor closes idle entries past mcpConnTTL.
	brokers *brokerRegistry

	// Flipped to true when graceful shutdown begins.  /readyz watches
	// this so load balancers can stop sending new connections while the
	// process drains in-flight work.
	shuttingDown atomic.Bool
}

func newServer(cfg config, db *sql.DB) *server {
	return &server{
		cfg:       cfg,
		db:        db,
		dialect:   sqliteDialect{}, // default; main() overrides after dialectFor()
		hub:       newHub(),
		lastSeen:  make(map[agentKey]int64),
		hfJobs:    newImportJobs(),
		comfyJobs: newComfyJobs(),
		relays:    newActiveRelays(),
		costCache: newRigCostCache(2 * time.Second),
		ipRL:      newIPRateLimiterSet(cfg.rateBackend),
		drift:     newRigDriftTable(),
		startedAt: time.Now(),
	}
}

// markLastSeen records that (uid, agentID) was heard from at now.  O(1),
// lock-free on the hot path after taking a single mutex.
func (s *server) markLastSeen(uid int64, agentID string) {
	s.lastSeenMu.Lock()
	s.lastSeen[agentKey{uid, agentID}] = nowUnix()
	s.lastSeenMu.Unlock()
}

// flushLastSeen writes all pending last_seen updates in one transaction.
// Called from reapLoop every 1s and once more on shutdown.
func (s *server) flushLastSeen() {
	s.lastSeenMu.Lock()
	if len(s.lastSeen) == 0 {
		s.lastSeenMu.Unlock()
		return
	}
	pending := s.lastSeen
	s.lastSeen = make(map[agentKey]int64, len(pending))
	s.lastSeenMu.Unlock()

	tx, err := s.db.Begin()
	if err != nil {
		// Put everything back; we'll retry next tick.
		s.lastSeenMu.Lock()
		for k, v := range pending {
			if cur, ok := s.lastSeen[k]; !ok || v > cur {
				s.lastSeen[k] = v
			}
		}
		s.lastSeenMu.Unlock()
		return
	}
	stmt, err := tx.Prepare(`UPDATE rigs SET last_seen = ? WHERE user_id = ? AND agent_id = ?`)
	if err != nil {
		_ = tx.Rollback()
		return
	}
	defer stmt.Close()
	for k, t := range pending {
		if _, err := stmt.Exec(t, k.userID, k.agentID); err != nil {
			log.Printf("flushLastSeen: %v", err)
		}
	}
	if err := tx.Commit(); err != nil {
		log.Printf("flushLastSeen commit: %v", err)
	}
}

func (s *server) router() http.Handler {
	mux := http.NewServeMux()

	// UI
	mux.HandleFunc("GET /", s.handleIndex)

	// Auth
	mux.HandleFunc("GET /auth", s.handleAuthPage)
	mux.HandleFunc("GET /api/auth/status", s.handleAuthStatus)
	mux.HandleFunc("GET /api/meta", s.handleMeta)
	mux.HandleFunc("GET /auth/github", s.handleGithubStart)
	mux.HandleFunc("GET /auth/github/callback", s.handleGithubCallback)
	mux.HandleFunc("GET /auth/google", s.handleGoogleStart)
	mux.HandleFunc("GET /auth/google/callback", s.handleGoogleCallback)
	mux.HandleFunc("GET /auth/oidc", s.handleOIDCStart)
	mux.HandleFunc("GET /auth/oidc/callback", s.handleOIDCCallback)
	mux.HandleFunc("POST /auth/logout", s.handleLogout)
	if s.cfg.devMode {
		mux.HandleFunc("POST /auth/dev", s.handleDevLogin)
	}

	// Pairing (browser mints, agent consumes)
	mux.HandleFunc("POST /api/pair", s.handlePairMint)

	// Device-code flow (`dist-node login`): rig mints code, user confirms.
	mux.HandleFunc("POST /api/device/code", s.handleDeviceCodeMint)
	mux.HandleFunc("POST /api/device/approve", s.handleDeviceApprove)
	mux.HandleFunc("POST /api/device/token", s.handleDeviceToken)
	mux.HandleFunc("GET /device", s.handleDevicePage)

	// Agent-key authenticated endpoints (used by `dist-node url`).
	mux.HandleFunc("POST /api/agent/api_key", s.handleAgentMintAPIKey)
	mux.HandleFunc("GET /api/agent/pools", s.handleAgentListPools)

	// Install flow
	mux.HandleFunc("GET /install.sh", s.handleInstallSh)
	mux.HandleFunc("GET /install.ps1", s.handleInstallPs1)
	mux.HandleFunc("GET /build.sh", s.handleBuildSh)
	mux.HandleFunc("GET /build.ps1", s.handleBuildPs1)
	// Lightweight `surd`-only installers (Phase 1.4 — 3-step UX).
	mux.HandleFunc("GET /setup.sh", s.handleSetupSh)
	mux.HandleFunc("GET /setup.zsh", s.handleSetupZsh)
	mux.HandleFunc("GET /setup.ps1", s.handleSetupPs1)
	// Friendly UI for the one-liner above (OS picker, pool/invite fields).
	mux.HandleFunc("GET /install", s.handleInstallPage)
	mux.HandleFunc("POST /api/install_command", s.handleInstallCommand)
	mux.HandleFunc("GET /api/install_targets", s.handleInstallTargets)
	// Release tarball proxy — rigs fetch short-name tarballs from the control
	// plane rather than GitHub.  Cache on disk.
	mux.HandleFunc("GET /releases/{name}", s.handleReleaseAsset)

	// API keys (dashboard-minted, used on /v1/* subdomain endpoints).
	mux.HandleFunc("POST /api/api_keys", s.handleMintAPIKey)
	mux.HandleFunc("GET /api/api_keys", s.handleListAPIKeys)
	mux.HandleFunc("DELETE /api/api_keys/{id}", s.handleRevokeAPIKey)

	// OpenAI-compatible endpoints.  Three addressing modes:
	//   1. Subdomain: <slug>.<apex>/v1/models                (prod)
	//   2. Path:      /v1/<slug>/models                      (works anywhere)
	//   3. Header:    /v1/models + X-Pool-Slug: <slug>       (dev / curl)
	// Requests to the apex with no slug resolution are 404'd by the handler.
	mux.HandleFunc("GET /v1/models", s.handleOAIModels)
	mux.HandleFunc("POST /v1/chat/completions", s.handleOAIChat)
	mux.HandleFunc("GET /v1/{slug}/models", s.handleOAIModels)
	mux.HandleFunc("POST /v1/{slug}/chat/completions", s.handleOAIChat)

	// Sessions / me
	mux.HandleFunc("GET /api/me", s.handleMe)
	mux.HandleFunc("GET /api/rigs", s.handleListRigs)
	mux.HandleFunc("DELETE /api/rigs/{agentID}", s.handleForgetRig)

	// Inference + usage
	mux.HandleFunc("POST /api/infer", s.handleInfer)
	mux.HandleFunc("POST /api/infer_pp", s.handleInferPP)
	mux.HandleFunc("POST /api/infer_dpp", s.handleInferDPP)
	mux.HandleFunc("GET /api/usage", s.handleUsage)
	mux.HandleFunc("GET /api/inference_log", s.handleInferenceLog)

	// P2P signalling (WebRTC offer/answer relay)
	mux.HandleFunc("POST /api/signal", s.handleSignal)

	// Models + shard server
	mux.HandleFunc("POST /api/models", s.handleRegisterModel)
	mux.HandleFunc("GET /api/models", s.handleListModels)
	mux.HandleFunc("GET /models/{id}/manifest.json", s.handleModelManifest)
	mux.HandleFunc("GET /models/{id}/shards/{file}", s.handleShardDownload)
	// Shard-peers — list of online rigs (under the requester's user) that
	// claim to have this shard.  Used by rigs spinning up to prefer P2P
	// over origin downloads.  See shard_index.go.
	mux.HandleFunc("GET /api/models/{name}/peers", s.handleShardPeers)
	// Fetch plan — server returns an ordered list of source URLs (peers
	// first, origin last) + recommended chunk size so a rig can do
	// parallel range-GET against many sources.  See shard_fetch_plan.go.
	mux.HandleFunc("GET /api/models/{name}/fetch-plan", s.handleShardFetchPlan)

	// HuggingFace import flow.
	mux.HandleFunc("POST /api/hf/token", s.handleHFSetToken)
	mux.HandleFunc("GET /api/hf/token", s.handleHFGetToken)
	mux.HandleFunc("GET /api/hf/search", s.handleHFSearch)
	mux.HandleFunc("POST /api/hf/resolve", s.handleHFResolve)
	mux.HandleFunc("POST /api/hf/import", s.handleHFImport)
	mux.HandleFunc("GET /api/hf/jobs", s.handleHFListJobs)
	mux.HandleFunc("GET /api/hf/jobs/{id}", s.handleHFJobDetail)
	mux.HandleFunc("POST /api/hf/jobs/{id}/cancel", s.handleHFJobCancel)

	// ComfyUI image/video gen.
	mux.HandleFunc("POST /api/comfy/workflows", s.handleComfyRegisterWorkflow)
	mux.HandleFunc("GET /api/comfy/workflows", s.handleComfyListWorkflows)
	mux.HandleFunc("POST /api/comfy/models", s.handleComfyRegisterModel)
	mux.HandleFunc("GET /api/comfy/models", s.handleComfyListModels)
	mux.HandleFunc("POST /api/comfy/generate", s.handleComfyGenerate)
	mux.HandleFunc("GET /api/comfy/jobs", s.handleComfyListJobs)
	mux.HandleFunc("GET /api/comfy/jobs/{id}", s.handleComfyJobDetail)
	mux.HandleFunc("POST /api/comfy/jobs/{id}/cancel", s.handleComfyJobCancel)
	mux.HandleFunc("GET /comfy/out/{id}/{file}", s.handleComfyOutput)
	mux.HandleFunc("POST /v1/images/generations", s.handleOAIImageGen)
	mux.HandleFunc("POST /v1/{slug}/images/generations", s.handleOAIImageGen)

	// Pools
	mux.HandleFunc("POST /api/pools", s.handleCreatePool)
	mux.HandleFunc("GET /api/pools", s.handleListPools)
	mux.HandleFunc("GET /api/pools/{id}", s.handlePoolDetail)
	mux.HandleFunc("POST /api/pools/{id}/invite", s.handlePoolInvite)
	mux.HandleFunc("GET /api/invites/{token}", s.handlePoolInvitePreview)
	mux.HandleFunc("POST /api/pools/join", s.handlePoolJoin)
	mux.HandleFunc("GET /join/{token}", s.handleJoinPage)
	mux.HandleFunc("POST /api/pools/{id}/rigs", s.handlePoolAttachRig)
	mux.HandleFunc("DELETE /api/pools/{id}/rigs/{rigID}", s.handlePoolDetachRig)

	// WebSocket endpoints
	mux.HandleFunc("GET /ws/browser", s.handleBrowserWS)
	mux.HandleFunc("GET /ws/agent", s.handleAgentWS)
	mux.HandleFunc("GET /ws/client", s.handleClientWS)

	// Public swarm dashboard — no auth.  Aggregates live telemetry across
	// every connected rig (Petals-style global view).
	mux.HandleFunc("GET /api/swarm", s.handleSwarmStats)
	mux.HandleFunc("GET /swarm", s.handleSwarmPage)

	// Dashboard read-only APIs (futuristic /console UI).
	mux.HandleFunc("GET /api/me/rigs", s.handleMeRigs)
	mux.HandleFunc("GET /api/me/rigs/stream", s.handleMeRigsStream)
	mux.HandleFunc("GET /api/me/earnings", s.handleMeEarnings)
	mux.HandleFunc("GET /api/pools/{id}/topology", s.handlePoolTopology)
	mux.HandleFunc("GET /api/pools/{id}/plan", s.handlePoolPlanGet)
	mux.HandleFunc("PUT /api/pools/{id}/plan", s.handlePoolPlanPut)
	mux.HandleFunc("DELETE /api/pools/{id}/plan", s.handlePoolPlanDelete)
	mux.HandleFunc("GET /api/pools/{id}/sessions", s.handlePoolSessions)
	mux.HandleFunc("GET /api/console/network", s.handleConsoleNetwork)
	mux.HandleFunc("GET /api/install/oneliner", s.handleInstallOneliner)
	// Desktop-widget combined feed (session OR agent-key auth).
	mux.HandleFunc("GET /api/widget/state", s.handleWidgetState)
	mux.HandleFunc("GET /console", s.handleConsolePage)
	mux.HandleFunc("GET /observatory", s.handleObservatoryPage)
	mux.HandleFunc("GET /nexus", s.handleNexusPage)
	mux.HandleFunc("GET /playground", s.handlePlaygroundPage)

	// RAG control plane.
	mux.HandleFunc("POST /api/rag/collections", s.handleRAGCreateCollection)
	mux.HandleFunc("GET /api/rag/collections", s.handleRAGListCollections)
	mux.HandleFunc("DELETE /api/rag/collections/{id}", s.handleRAGDeleteCollection)
	mux.HandleFunc("POST /api/rag/collections/{id}/documents", s.handleRAGUploadDocument)
	mux.HandleFunc("GET /api/rag/collections/{id}/documents", s.handleRAGListDocuments)
	mux.HandleFunc("DELETE /api/rag/collections/{id}/documents/{doc_id}", s.handleRAGDeleteDocument)
	mux.HandleFunc("POST /api/rag/collections/{id}/search", s.handleRAGSearch)
	mux.HandleFunc("POST /api/rag/collections/{id}/hybrid_search", s.handleRAGHybridSearch)

	// MCP control plane.  Registry CRUD for user-owned tool servers,
	// plus the broker call path that proxies JSON-RPC through to them.
	mux.HandleFunc("GET /api/mcp/servers", s.handleMCPListServers)
	mux.HandleFunc("POST /api/mcp/servers", s.handleMCPCreateServer)
	mux.HandleFunc("GET /api/mcp/servers/{id}", s.handleMCPGetServer)
	mux.HandleFunc("PUT /api/mcp/servers/{id}", s.handleMCPUpdateServer)
	mux.HandleFunc("DELETE /api/mcp/servers/{id}", s.handleMCPDeleteServer)
	mux.HandleFunc("POST /api/mcp/{server}/call", s.handleMCPCall)

	// Health + readiness + metrics.  /healthz is a cheap liveness
	// probe (always 200 if the process is up); /readyz pings the DB and
	// flips during graceful shutdown so LBs can drain; /metrics emits
	// Prometheus text exposition.
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte("ok"))
	})
	mux.HandleFunc("GET /readyz", s.handleReadyz)
	mux.HandleFunc("GET /metrics", s.handleMetrics)

	// On-demand LAN reachability check.  Re-runs the boot-time firewall
	// probe; used by the dashboard's "Test LAN reachability" button so
	// users can verify a fix without restarting the server.
	mux.HandleFunc("GET /api/firewall_check", func(w http.ResponseWriter, r *http.Request) {
		if _, ok := s.userFromRequest(r); !ok {
			writeErr(w, 401, "not logged in")
			return
		}
		writeJSON(w, 200, firewallProbe(s.cfg.addr))
	})

	return withRequestID(withLogging(withSecurityHeaders(withCORSForV1(mux))))
}

// withSecurityHeaders sets common defense-in-depth headers on every
// response.  Cheap, well-understood, and missing them is a frequent
// audit finding even on otherwise-clean services.
//
//	X-Content-Type-Options:  stops MIME sniffing on user-uploaded blobs
//	                         (model shards, comfy outputs).
//	X-Frame-Options:         the dashboard isn't an embeddable widget;
//	                         clickjacking defence.
//	Referrer-Policy:         keep referer leaks to a minimum.
//	Strict-Transport-Security: only set when the request came in over
//	                           HTTPS (TLS-terminating proxy or built-in
//	                           TLS); harmless on dev HTTP.
//
// We deliberately skip Content-Security-Policy here — the dashboard
// uses some inline scripts and locking it down without an audit risks
// breaking the UI.  Add later when there's time to inventory sources.
func withSecurityHeaders(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("X-Frame-Options", "DENY")
		w.Header().Set("Referrer-Policy", "strict-origin-when-cross-origin")
		// HSTS is meaningful only when the connection actually used TLS.
		// `r.TLS != nil` covers in-process TLS; `X-Forwarded-Proto: https`
		// covers a TLS-terminating reverse proxy (the common deploy).
		if r.TLS != nil || strings.EqualFold(r.Header.Get("X-Forwarded-Proto"), "https") {
			w.Header().Set("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
		}
		h.ServeHTTP(w, r)
	})
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

// requestIDKey is the context key for the per-request correlation ID.
// We use an unexported type so callers can't collide on the value.
type requestIDKey struct{}

// requestIDFromContext returns the per-request ID set by withRequestID,
// or "" if no ID is in scope.  Handlers / log lines can include this so
// a problem reported by a user can be traced back through middleware.
func requestIDFromContext(ctx context.Context) string {
	if v, ok := ctx.Value(requestIDKey{}).(string); ok {
		return v
	}
	return ""
}

// withRequestID assigns a stable correlation ID per request.  Honours an
// inbound X-Request-ID when present (so reverse proxies or callers can
// propagate one); otherwise mints a fresh 16-byte hex.  Echoed in the
// response header so clients can correlate without parsing logs.
func withRequestID(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rid := r.Header.Get("X-Request-ID")
		if rid == "" || len(rid) > 64 {
			rid = newRandomToken(8) // 16 hex chars
		}
		w.Header().Set("X-Request-ID", rid)
		ctx := context.WithValue(r.Context(), requestIDKey{}, rid)
		h.ServeHTTP(w, r.WithContext(ctx))
	})
}

// logJSON is the toggle for structured access logs.  Defaulted off so dev
// runs stay human-readable; set DIST_LOG_JSON=1 in production.
var logJSON = os.Getenv("DIST_LOG_JSON") == "1"

func withLogging(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t0 := time.Now()
		sr := &statusRecorder{ResponseWriter: w, status: 200}
		h.ServeHTTP(sr, r)
		dur := time.Since(t0)
		rid := requestIDFromContext(r.Context())
		if logJSON {
			entry := map[string]any{
				"ts":          t0.UTC().Format(time.RFC3339Nano),
				"level":       "info",
				"event":       "http_request",
				"request_id":  rid,
				"method":      r.Method,
				"path":        r.URL.Path,
				"status":      sr.status,
				"duration_ms": dur.Milliseconds(),
				"remote":      r.RemoteAddr,
			}
			b, _ := json.Marshal(entry)
			fmt.Fprintln(os.Stdout, string(b))
			return
		}
		log.Printf("[%s] %s %s -> %d in %s",
			rid, r.Method, r.URL.Path, sr.status, dur)
	})
}

// withCORSForV1 emits permissive CORS headers on /v1/* paths so any OpenAI
// client can call the pool endpoints from a browser.  Safe because auth on
// /v1/* is Bearer-only — cookies are blocked by SameSite=Lax, so allowing
// `*` origin doesn't expose any cookie-authenticated surface.
func withCORSForV1(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/v1/") {
			w.Header().Set("Access-Control-Allow-Origin", "*")
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Authorization, Content-Type, X-Pool-Slug")
			w.Header().Set("Access-Control-Max-Age", "600")
			if r.Method == http.MethodOptions {
				w.WriteHeader(http.StatusNoContent)
				return
			}
		}
		h.ServeHTTP(w, r)
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
	_, err := s.dbExec(
		`INSERT INTO sessions (id, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)`,
		sid, userID, nowUnix(), expires.Unix(),
	)
	if err != nil {
		return "", time.Time{}, err
	}
	return sid, expires, nil
}

func (s *server) userFromRequest(r *http.Request) (*user, bool) {
	// Headless callers (dist-cli, scripts) authenticate via
	// Authorization: Bearer sk-dist-...  — the same api_key minted by
	// /api/api_keys or /api/agent/api_key.  Browsers send cookies; CLIs
	// send bearers.  We accept either so the dashboard endpoints can be
	// driven from a script without a browser session.
	if bearer := bearerFromRequest(r); bearer != "" {
		// Two bearer shapes are accepted:
		//   sk-dist-...  → static api_key, cheap DB lookup
		//   eyJ...       → JWT minted by the configured OIDC OP,
		//                  verified via JWKS.  Distinguished by
		//                  looksLikeJWT — three dot-separated segments.
		if looksLikeJWT(bearer) {
			if u, ok := s.userFromOIDCBearer(r); ok {
				return u, true
			}
		}
		if u, ok := s.userFromAPIKey(bearer); ok {
			return u, true
		}
	}
	c, err := r.Cookie(sessionCookieName)
	if err != nil || c.Value == "" {
		return nil, false
	}
	var (
		uid int64
		exp int64
	)
	err = s.dbQueryRow(
		`SELECT user_id, expires_at FROM sessions WHERE id = ?`,
		c.Value,
	).Scan(&uid, &exp)
	if err != nil || exp < nowUnix() {
		return nil, false
	}
	u := &user{}
	var gh sql.NullString
	err = s.dbQueryRow(
		`SELECT id, github_login, display_name FROM users WHERE id = ?`,
		uid,
	).Scan(&u.ID, &gh, &u.DisplayName)
	if err != nil {
		return nil, false
	}
	u.GithubLogin = gh.String
	return u, true
}

// secureCookies returns true when publicURL is HTTPS, so we should set
// the Secure flag on every Set-Cookie we emit.  Plain-HTTP deployments
// (dev, behind-a-trusted-proxy testing) don't get the flag because
// browsers will refuse to send it back and the user is silently logged
// out.  This is the single source of truth — every SetCookie site
// reads from here.
func (s *server) secureCookies() bool {
	return strings.HasPrefix(strings.ToLower(s.cfg.publicURL), "https://")
}

func (s *server) setSessionCookie(w http.ResponseWriter, sid string, expires time.Time) {
	http.SetCookie(w, &http.Cookie{
		Name:     sessionCookieName,
		Value:    sid,
		Path:     "/",
		HttpOnly: true,
		Secure:   s.secureCookies(),
		Expires:  expires,
		SameSite: http.SameSiteLaxMode,
	})
}

func (s *server) clearSessionCookie(w http.ResponseWriter) {
	http.SetCookie(w, &http.Cookie{
		Name:     sessionCookieName,
		Value:    "",
		Path:     "/",
		HttpOnly: true,
		Secure:   s.secureCookies(),
		MaxAge:   -1,
	})
}

// ─── Handlers: misc ─────────────────────────────────────────────────────────

func (s *server) handleIndex(w http.ResponseWriter, r *http.Request) {
	// Authed users land on the live operations console. Unauthed users
	// see the editorial sign-in page (which then bounces them to /nexus
	// once a session is minted).
	if _, ok := s.userFromRequest(r); ok {
		http.Redirect(w, r, "/nexus", http.StatusFound)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	_, _ = w.Write(authPageHTML)
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
	rows, err := s.dbQuery(
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

// DELETE /api/rigs/{agentID} — forget a rig owned by the current user.
//
// Idempotent: returns 200 with deleted=true/false.  We refuse to drop a rig
// that's currently online (the live WS connection would just re-register it
// on the next heartbeat) — caller should `dist-node logout` on the rig first
// or wait for it to disconnect.  Removes the rig row, its pool memberships,
// shard-cache entries, and any API keys minted via /api/agent/api_key by
// this rig identity.
func (s *server) handleForgetRig(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	agentID := r.PathValue("agentID")
	if agentID == "" {
		writeErr(w, 400, "agent_id required")
		return
	}
	if s.hub.agentOnline(u.ID, agentID) {
		writeErr(w, 409, "rig is currently online — disconnect it first")
		return
	}
	var rigID int64
	err := s.dbQueryRow(
		`SELECT id FROM rigs WHERE user_id = ? AND agent_id = ?`,
		u.ID, agentID,
	).Scan(&rigID)
	if err != nil {
		writeJSON(w, 200, map[string]any{"deleted": false, "reason": "not_found"})
		return
	}
	tx, err := s.db.Begin()
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	defer tx.Rollback()
	// Pool membership rows + shard cache index reference rig_id; drop first
	// so the foreign-key-less SQLite schema doesn't leave dangling rows.
	_, _ = s.txExec(tx, `DELETE FROM pool_rigs WHERE rig_id = ?`, rigID)
	_, _ = s.txExec(tx, `DELETE FROM rig_shards WHERE rig_id = ?`, rigID)
	if _, err := s.txExec(tx,
		`DELETE FROM rigs WHERE id = ? AND user_id = ?`, rigID, u.ID,
	); err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	if err := tx.Commit(); err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	writeJSON(w, 200, map[string]any{"deleted": true, "agent_id": agentID})
}

// Dev login — POST {display_name}.  Opt-in only via DIST_DEV_MODE=1 / --dev.
// Defense-in-depth: even if the route is wired by accident, refuse when devMode
// is off.  Combined with the router gate this is belt-and-suspenders.
func (s *server) handleDevLogin(w http.ResponseWriter, r *http.Request) {
	if !s.cfg.devMode {
		writeErr(w, 404, "not found")
		return
	}
	var body struct {
		DisplayName string `json:"display_name"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 4<<10)).Decode(&body); err != nil {
		writeErr(w, 400, "display_name required")
		return
	}
	clean, err := sanitizeDisplayName(body.DisplayName)
	if err != nil {
		writeErr(w, 400, err.Error())
		return
	}
	body.DisplayName = clean
	res, err := s.dbExec(
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
		_, _ = s.dbExec(`DELETE FROM sessions WHERE id = ?`, c.Value)
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
	_, err := s.dbExec(
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
	if len(u) >= 7 && strings.EqualFold(u[:7], "http://") {
		return "ws://" + u[7:]
	}
	if len(u) >= 8 && strings.EqualFold(u[:8], "https://") {
		return "wss://" + u[8:]
	}
	return u
}

// consumePairToken returns (user_id, pool_id, error).  pool_id is 0 when the
// token was not pinned to a pool.  Uses UPDATE ... RETURNING so the read and
// the "mark used" write are a single atomic statement — two concurrent
// consumers of the same token can never both succeed.
func (s *server) consumePairToken(token string) (int64, int64, error) {
	var uid, exp int64
	var poolID sql.NullInt64
	now := nowUnix()
	err := s.dbQueryRow(
		`UPDATE pair_tokens
		    SET used_at = ?
		  WHERE token = ? AND used_at IS NULL AND expires_at >= ?
		  RETURNING user_id, expires_at, pool_id`,
		now, token, now,
	).Scan(&uid, &exp, &poolID)
	if err == sql.ErrNoRows {
		// Row either doesn't exist, was already used, or expired.  We can
		// disambiguate expiry vs missing/used with a follow-up read, but the
		// caller treats both the same way (prompt the user to regenerate),
		// so keep it one round-trip.
		return 0, 0, errPairExpired
	}
	if err != nil {
		return 0, 0, err
	}
	var pid int64
	if poolID.Valid {
		pid = poolID.Int64
	}
	return uid, pid, nil
}

var errPairExpired = &pairErr{"pair token expired"}

type pairErr struct{ msg string }

func (e *pairErr) Error() string { return e.msg }

// ─── Reap loop ──────────────────────────────────────────────────────────────

func (s *server) reapLoop(ctx context.Context) {
	reap := time.NewTicker(1 * time.Minute)
	defer reap.Stop()
	flush := time.NewTicker(1 * time.Second)
	defer flush.Stop()
	// Hourly: prune idle reputation rows + reap stale relay assignments.
	// Both are bounded-growth defences; running them less often than the
	// per-minute table reap keeps the hot loop quick.
	slow := time.NewTicker(1 * time.Hour)
	defer slow.Stop()
	for {
		select {
		case <-ctx.Done():
			s.flushLastSeen()
			return
		case <-flush.C:
			s.flushLastSeen()
		case <-reap.C:
			_, _ = s.dbExec(`DELETE FROM sessions    WHERE expires_at < ?`, nowUnix())
			_, _ = s.dbExec(`DELETE FROM pair_tokens WHERE expires_at < ?`, nowUnix())
			// Reap any relay assignment older than maxRelayAssignmentAge —
			// the session leaked (client disappeared without release).
			if s.relays != nil {
				stale := s.relays.reapStale(nowUnix(), maxRelayAssignmentAge)
				for _, a := range stale {
					s.recordRelayFailure(a.AgentID)
					log.Printf("relay assignment %s/%s reaped after %ds (no release frame)",
						a.AgentID, a.SessionID, nowUnix()-a.StartedAt)
				}
			}
			// Comfy jobs that have gone unmoved for ~15 minutes are
			// orphaned (handler crashed, rig died without disconnect
			// being observed, etc.).  Fail them so users see a
			// terminal state instead of staring at a "running" row
			// that will never update.
			if n, err := s.reapStaleComfyJobs(15 * 60); err == nil && n > 0 {
				log.Printf("comfy_jobs: reaped %d stale rows", n)
			}
		case <-slow.C:
			// 30-day idle eviction.  Reputation can always be rebuilt; the
			// goal is to keep the table from holding forever rows for
			// agents that did one session and vanished.
			if n, err := s.pruneIdleReputation(30 * 24 * 3600); err == nil && n > 0 {
				log.Printf("rig_reputation: pruned %d idle rows", n)
			}
		}
	}
}

// ─── Hub ────────────────────────────────────────────────────────────────────
//
// Tracks live WebSocket connections so the browser-side can subscribe to events
// from agents owned by the same user.

type hub struct {
	mu       sync.RWMutex
	agents   map[agentKey]*agentConn  // (userID, agentID) -> conn
	browsers map[int64][]*browserConn // userID -> connections
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

// findAgentByID looks up the first online agentConn whose agentID matches,
// across all users.  P2P signaling needs this because the planner can pair
// rigs from different users within the same pool.  When agent IDs collide
// across users (theoretically possible — they're free-form strings the rig
// chose), we return the first match; rigs MUST treat their agent_id as
// globally unique within their pool for P2P to work.
func (h *hub) findAgentByID(agentID string) *agentConn {
	h.mu.RLock()
	defer h.mu.RUnlock()
	for k, a := range h.agents {
		if k.agentID == agentID {
			return a
		}
	}
	return nil
}

// snapshotAgents returns a slice copy of every online agentConn — safe to
// iterate without holding the hub lock.  Used by the swarm dashboard
// aggregator, which must not block hot-path operations on the hub.
func (h *hub) snapshotAgents() []*agentConn {
	h.mu.RLock()
	defer h.mu.RUnlock()
	out := make([]*agentConn, 0, len(h.agents))
	for _, a := range h.agents {
		out = append(out, a)
	}
	return out
}

// countOnlineRigsInPool counts how many pool_rigs are currently online.
func (s *server) countOnlineRigsInPool(poolID int64) int {
	rows, err := s.dbQuery(`
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

// pickAndReserveRig picks the least-loaded rig in the pool AND atomically
// attaches `ip` to it (acquireInferSlot).  Skips rigs whose slots are full
// or that are currently bound as a relay/pp/dpp peer.  Returns nil if no
// rig has spare capacity.  The caller still owns the inflight increment
// for the chosen rig — we don't touch it here.
//
// This is the single entry point that lets batching pay off: when N
// concurrent /v1/chat requests hit the same multi-slot rig, each call to
// pickAndReserveRig admits up to MaxConcurrent of them onto the same
// agentConn instead of bouncing the later ones with 503.
func (s *server) pickAndReserveRig(poolID int64, ip *inferPeer) *agentConn {
	rows, err := s.dbQuery(`
		SELECT r.user_id, r.agent_id FROM pool_rigs pr
		JOIN rigs r ON r.id = pr.rig_id
		WHERE pr.pool_id = ?
		ORDER BY RANDOM()`, poolID)
	if err != nil {
		return nil
	}
	defer rows.Close()
	type cand struct {
		ac   *agentConn
		load int32
		cap  int
	}
	var cands []cand
	for rows.Next() {
		var uid int64
		var aid string
		if err := rows.Scan(&uid, &aid); err != nil {
			continue
		}
		a, ok := s.hub.findAgent(uid, aid)
		if !ok {
			continue
		}
		st := a.snapshotStatus()
		maxConc := st.MaxConcurrent
		if maxConc <= 0 {
			maxConc = 1
		}
		cands = append(cands, cand{ac: a, load: a.loadInflight(), cap: maxConc})
	}
	// Sort by least loaded.  Slice is small (handful of rigs) so a single
	// linear scan beats sort.Slice overhead.
	for i := range cands {
		for j := i + 1; j < len(cands); j++ {
			if cands[j].load < cands[i].load {
				cands[i], cands[j] = cands[j], cands[i]
			}
		}
	}
	for _, c := range cands {
		if c.ac.acquireInferSlot(ip, c.cap) {
			return c.ac
		}
	}
	return nil
}

// pickOnlineRigInPool returns the least-loaded live agentConn in the pool.
// Load == number of in-flight requests the server has dispatched to that
// agent.  Ties are broken by RANDOM() order in the DB scan so a cold pool
// doesn't always route to the same rig.  Caller MUST call incInflight()
// before dispatching and decInflight() when the request terminates.
func (s *server) pickOnlineRigInPool(poolID int64) (*agentConn, bool) {
	rows, err := s.dbQuery(`
		SELECT r.user_id, r.agent_id FROM pool_rigs pr
		JOIN rigs r ON r.id = pr.rig_id
		WHERE pr.pool_id = ?
		ORDER BY RANDOM()`, poolID)
	if err != nil {
		return nil, false
	}
	defer rows.Close()
	var best *agentConn
	var bestLoad int32 = 1 << 30
	for rows.Next() {
		var uid int64
		var aid string
		if err := rows.Scan(&uid, &aid); err != nil {
			continue
		}
		a, ok := s.hub.findAgent(uid, aid)
		if !ok {
			continue
		}
		if load := a.loadInflight(); load < bestLoad {
			best = a
			bestLoad = load
			if load == 0 {
				break // can't beat zero; stop scanning
			}
		}
	}
	if best == nil {
		return nil, false
	}
	return best, true
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
