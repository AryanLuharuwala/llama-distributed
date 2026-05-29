package main

import (
	"context"
	"crypto/rand"
	"database/sql"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"log"
	"net"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coder/websocket"
)

// authFailJitter sleeps a small randomised amount to defeat timing-based
// user enumeration on the hello/resume paths.  Anywhere from 75ms to
// 250ms — small enough not to harm honest reconnects, large enough to
// dwarf any branch-timing differential between "no such rig" and
// "wrong key".
func authFailJitter() {
	var n uint16
	_ = binary.Read(rand.Reader, binary.BigEndian, &n)
	// 75ms base, up to ~250ms total.
	d := 75*time.Millisecond + time.Duration(n%176)*time.Millisecond
	time.Sleep(d)
}

// failWSAuth records a hello/resume failure against the per-IP rate
// limiter, jitters the close so an attacker can't tell which auth
// branch they hit, sends an error frame, and closes.
func (s *server) failWSAuth(ctx context.Context, conn *websocket.Conn, ip, msg string) {
	if s.ipRL != nil {
		s.ipRL.helloFail.allow(ip)
	}
	authFailJitter()
	_ = wsjsonWrite(ctx, conn, map[string]any{
		"kind": "error", "message": msg,
	})
	_ = conn.Close(websocket.StatusPolicyViolation, msg)
}

// clientIP returns the request's originating IP.  When the server has a
// configured trusted-proxy set (DIST_TRUSTED_PROXIES), XFF is walked
// right-to-left with each trusted hop stripped, returning the first
// untrusted address — the actual client.  Otherwise XFF is ignored and
// we use r.RemoteAddr (the TCP peer).
//
// trustedClientIP returns "" only if r.RemoteAddr can't be parsed; that's
// effectively impossible for a real http.Request.
func (s *server) clientIP(r *http.Request) string {
	if s == nil || s.cfg.trustedProxies == nil {
		// Pre-init or in a test that wired a partial server — fall back
		// to the pre-trust-set behavior so we don't panic.
		host, _, err := net.SplitHostPort(r.RemoteAddr)
		if err != nil {
			return r.RemoteAddr
		}
		return host
	}
	return trustedClientIP(r, s.cfg.trustedProxies)
}

// b64DecodeStd is a small adapter so callers don't have to import
// encoding/base64 explicitly.
func b64DecodeStd(s string) ([]byte, error) {
	return base64.StdEncoding.DecodeString(s)
}

// ─── Browser WebSocket ──────────────────────────────────────────────────────

type browserConn struct {
	userID int64
	conn   *websocket.Conn
	outCh  chan any
	closed chan struct{}
	once   sync.Once
}

// send tries a non-blocking enqueue first; if the buffer is full the browser
// is too slow to keep up and we close the socket so the client reconnects and
// re-fetches state on open rather than silently missing events.  Much better
// than the previous silent drop, which left dashboards permanently stale.
func (b *browserConn) send(v any) {
	select {
	case b.outCh <- v:
	default:
		log.Printf("browser uid=%d: outCh full, closing to force client re-sync", b.userID)
		b.close()
	}
}

func (b *browserConn) close() {
	b.once.Do(func() { close(b.closed) })
}

func (s *server) handleBrowserWS(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	conn, err := websocket.Accept(w, r, &websocket.AcceptOptions{
		// Same-origin only; cookie auth already proved it.
		InsecureSkipVerify: false,
	})
	if err != nil {
		return
	}
	wsConnDelta(r.Context(), "browser", 1)
	defer wsConnDelta(context.Background(), "browser", -1)

	bc := &browserConn{
		userID: u.ID,
		conn:   conn,
		outCh:  make(chan any, 64),
		closed: make(chan struct{}),
	}
	s.hub.registerBrowser(bc)
	defer s.hub.unregisterBrowser(bc)

	ctx := r.Context()
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	// Writer
	go func() {
		for {
			select {
			case <-bc.closed:
				return
			case <-ctx.Done():
				return
			case msg := <-bc.outCh:
				wctx, c := context.WithTimeout(ctx, 5*time.Second)
				err := wsjsonWrite(wctx, conn, msg)
				c()
				if err != nil {
					bc.close()
					return
				}
			}
		}
	}()

	// Reader — keep the socket alive, discard any inbound messages for M1.
	for {
		_, _, err := conn.Read(ctx)
		if err != nil {
			bc.close()
			break
		}
	}
	_ = conn.Close(websocket.StatusNormalClosure, "")
}

// ─── Agent WebSocket ────────────────────────────────────────────────────────
//
// Protocol (JSON messages, agent <-> server):
//
//   First message from agent:
//     {"kind":"hello", "token":"<pair token>", "agent_id":"host:pid",
//      "hostname":"...", "n_gpus":N, "vram_bytes":...}
//
//   Server replies:
//     {"kind":"welcome", "user_id":N, "display_name":"..."}   on success
//     {"kind":"error", "message":"..."}                       on failure
//
//   Afterwards, agent periodically sends:
//     {"kind":"status", "n_gpus":N, "gpu_util":[...], "tokens_sec":...}
//
//   Server may push:
//     {"kind":"command", "cmd":"ping"|"join-pool"|...}

type agentConn struct {
	userID   int64
	agentID  string
	hostname string
	conn     *websocket.Conn
	outCh    chan any    // outbound JSON (text)
	binCh    chan []byte // outbound binary (relay from client)
	closed   chan struct{}
	once     sync.Once

	// Load tracking for request routing.  Incremented by the dispatcher before
	// a request is forwarded to this agent, decremented when the response
	// finishes (success, error, or client disconnect).  Read lock-free via
	// atomic.Int32.
	inflight atomic.Int32

	// Relay binding: if non-nil, binary frames from this agent are forwarded
	// to the bound client.  Protected by peerMu.
	peerMu     sync.RWMutex
	peer       *clientConn
	inferPeers map[uint16]*inferPeer // reqID → peer; len()==0 means idle.
	ppPeer     *ppPeer               // set while this agent is a stage in an /api/infer_pp request
	dppPeer    *dppPeer              // set while this agent is a stage in a diffusion-PP request

	// ComfyUI result channels — keyed by job_id.  The dispatcher subscribes
	// before sending `comfy_run` and unsubscribes when the job finishes.
	// comfy_result frames from the rig are routed into the right channel.
	comfyMu      sync.Mutex
	comfyResults map[int64]chan comfyResultMsg

	// ComfyUI native-API proxy responses — keyed by correlation id.  See
	// CF11: the control plane wraps ComfyUI's metadata endpoints
	// (system_stats/object_info/queue/embeddings/interrupt/free) so SDK
	// clients don't need direct rig access.  Each request gets a fresh
	// uuid; the rig echoes it back in the comfy_meta_result frame.
	comfyMetaResults map[string]chan comfyMetaMsg

	// Latest telemetry snapshot, populated from `status` frames.  Read by
	// the swarm dashboard aggregator.  Server-derived fields (remoteIP,
	// pairedAt) are written once at connect; agent-reported fields are
	// overwritten on every status frame.
	statusMu sync.RWMutex
	live     liveStatus
	remoteIP string
	pairedAt int64

	// Negotiated wire-protocol version (1 if rig was legacy / omitted the
	// field).  Future incompatible frame changes branch on this.
	protocolVersion int
}

func (a *agentConn) updateStatus(st agentStatus) {
	a.statusMu.Lock()
	defer a.statusMu.Unlock()
	a.live.UpdatedAt = nowUnix()
	a.live.TokensPS = st.TokensPS
	if len(st.GPUUtil) > 0 {
		a.live.GPUUtil = st.GPUUtil
	}
	if st.NGPUs > 0 {
		a.live.NGPUs = st.NGPUs
	}
	if st.GPUModel != "" {
		a.live.GPUModel = st.GPUModel
	}
	if st.VRAMTotal > 0 {
		a.live.VRAMTotal = st.VRAMTotal
	}
	if st.VRAMFree > 0 {
		a.live.VRAMFree = st.VRAMFree
	}
	if st.UptimeSec > 0 {
		a.live.UptimeSec = st.UptimeSec
	}
	if len(st.RolesHeld) > 0 {
		a.live.RolesHeld = st.RolesHeld
	}
	if len(st.ModelsHeld) > 0 {
		a.live.ModelsHeld = st.ModelsHeld
	}
	a.live.Inflight = st.Inflight
	if st.BWUpKbps > 0 {
		a.live.BWUpKbps = st.BWUpKbps
	}
	if st.BWDnKbps > 0 {
		a.live.BWDnKbps = st.BWDnKbps
	}
	a.live.LastError = st.LastError
	if st.NATType != "" {
		a.live.NATType = st.NATType
	}
	a.live.RelayCapable = st.RelayCapable
	a.live.CoturnPort = st.CoturnPort
	if st.PublicIP != "" {
		a.live.PublicIP = st.PublicIP
	}
	if st.MaxConcurrent > 0 {
		a.live.MaxConcurrent = st.MaxConcurrent
	}
}

func (a *agentConn) snapshotStatus() liveStatus {
	a.statusMu.RLock()
	defer a.statusMu.RUnlock()
	cp := a.live
	if cp.GPUUtil != nil {
		cp.GPUUtil = append([]float64(nil), cp.GPUUtil...)
	}
	if cp.RolesHeld != nil {
		cp.RolesHeld = append([]string(nil), cp.RolesHeld...)
	}
	if cp.ModelsHeld != nil {
		cp.ModelsHeld = append([]string(nil), cp.ModelsHeld...)
	}
	return cp
}

// comfyResultMsg is a single frame from a rig running a ComfyUI workflow.
//   - Data + FileName populated on a result file frame.
//   - Final = true marks the last frame; Err non-nil means the workflow failed.
type comfyResultMsg struct {
	FileName string
	Data     []byte
	Final    bool
	Err      error
}

// comfyMetaMsg is a single ComfyUI native-API proxy response.  Body
// holds the raw response body bytes; HTTPStatus is the upstream HTTP
// status (0 if the rig couldn't reach its local ComfyUI).  Err is
// populated when the rig itself fails (queue full, panic, parse error)
// — distinct from upstream non-2xx, which carries a body + status.
type comfyMetaMsg struct {
	HTTPStatus int
	Body       []byte
	Err        error
}

func (a *agentConn) subscribeComfyMeta(corrID string) chan comfyMetaMsg {
	a.comfyMu.Lock()
	defer a.comfyMu.Unlock()
	if a.comfyMetaResults == nil {
		a.comfyMetaResults = make(map[string]chan comfyMetaMsg)
	}
	ch := make(chan comfyMetaMsg, 1)
	a.comfyMetaResults[corrID] = ch
	return ch
}

func (a *agentConn) unsubscribeComfyMeta(corrID string) {
	a.comfyMu.Lock()
	defer a.comfyMu.Unlock()
	if ch, ok := a.comfyMetaResults[corrID]; ok {
		delete(a.comfyMetaResults, corrID)
		close(ch)
	}
}

// deliverComfyMetaResult routes a comfy_meta_result frame to its
// subscriber.  Same close-safety pattern as deliverComfyResult.
func (a *agentConn) deliverComfyMetaResult(msg map[string]any) bool {
	corr, _ := msg["corr_id"].(string)
	if corr == "" {
		return false
	}
	out := comfyMetaMsg{}
	if v, ok := msg["status"].(float64); ok {
		out.HTTPStatus = int(v)
	}
	if b64, ok := msg["body_b64"].(string); ok && b64 != "" {
		if data, err := b64DecodeStd(b64); err == nil {
			out.Body = data
		} else {
			out.Err = err
		}
	} else if body, ok := msg["body"].(string); ok {
		// Backwards-compat path: rigs that haven't shipped the b64
		// variant inline a UTF-8 string.  Metadata responses (JSON)
		// are always UTF-8 so this is safe; binary responses always
		// take the b64 path.
		out.Body = []byte(body)
	}
	if errMsg, ok := msg["error"].(string); ok && errMsg != "" {
		out.Err = errors.New(errMsg)
	}
	a.comfyMu.Lock()
	defer a.comfyMu.Unlock()
	ch, ok := a.comfyMetaResults[corr]
	if !ok {
		return false
	}
	select {
	case ch <- out:
	default:
	}
	return true
}

func (a *agentConn) subscribeComfy(jobID int64) chan comfyResultMsg {
	a.comfyMu.Lock()
	defer a.comfyMu.Unlock()
	if a.comfyResults == nil {
		a.comfyResults = make(map[int64]chan comfyResultMsg)
	}
	ch := make(chan comfyResultMsg, 16)
	a.comfyResults[jobID] = ch
	return ch
}

func (a *agentConn) unsubscribeComfy(jobID int64) {
	a.comfyMu.Lock()
	defer a.comfyMu.Unlock()
	if ch, ok := a.comfyResults[jobID]; ok {
		delete(a.comfyResults, jobID)
		close(ch)
	}
}

// deliverComfyResult routes a comfy_result frame to its subscriber.
// Returns true if the frame was handled (so the WS reader can skip the
// generic broadcast path).
func (a *agentConn) deliverComfyResult(msg map[string]any) bool {
	jobF, _ := msg["job_id"].(float64)
	if jobF == 0 {
		return false
	}
	jobID := int64(jobF)
	out := comfyResultMsg{}
	if name, ok := msg["file"].(string); ok {
		out.FileName = name
	}
	if b64, ok := msg["data_b64"].(string); ok {
		// Inline import keeps this file's import list minimal — base64 is
		// already a transitive dep via openai.go.
		if data, err := b64DecodeStd(b64); err == nil {
			out.Data = data
		} else {
			out.Err = err
		}
	}
	if final, _ := msg["final"].(bool); final {
		out.Final = true
		if errMsg, ok := msg["error"].(string); ok && errMsg != "" {
			out.Err = errors.New(errMsg)
		}
	}
	// Hold comfyMu across the send so a concurrent ac.close() — which
	// takes comfyMu, drops the map, then closes each channel after
	// releasing the lock — can't close `ch` between our map lookup
	// and our send.  The send is non-blocking; total time-under-lock
	// is microseconds.
	a.comfyMu.Lock()
	defer a.comfyMu.Unlock()
	ch, ok := a.comfyResults[jobID]
	if !ok {
		return false
	}
	select {
	case ch <- out:
	default:
	}
	return true
}

func (a *agentConn) incInflight() { a.inflight.Add(1) }
func (a *agentConn) decInflight() { a.inflight.Add(-1) }
func (a *agentConn) loadInflight() int32 { return a.inflight.Load() }

func (a *agentConn) send(v any) {
	select {
	case a.outCh <- v:
	default:
	}
}

// trySend is like send but returns false when the outbound queue is
// full or the agent has been closed.  Callers that can't tolerate
// silent drops (comfy dispatch, infer PP) should prefer this so they
// can fail fast instead of waiting for a timeout.  See audit P21-CF8.
func (a *agentConn) trySend(v any) bool {
	select {
	case <-a.closed:
		return false
	default:
	}
	select {
	case a.outCh <- v:
		return true
	case <-a.closed:
		return false
	default:
		return false
	}
}

func (a *agentConn) sendBin(b []byte) bool {
	select {
	case a.binCh <- b:
		return true
	default:
		return false
	}
}

func (a *agentConn) setPeer(c *clientConn) {
	a.peerMu.Lock()
	a.peer = c
	a.peerMu.Unlock()
}

func (a *agentConn) getPeer() *clientConn {
	a.peerMu.RLock()
	defer a.peerMu.RUnlock()
	return a.peer
}

func (a *agentConn) close() {
	a.once.Do(func() {
		close(a.closed)
		// Fail-fast every in-flight handler bound to this rig: LLM chat,
		// LLM PP-stage, diffusion-PP stage, and comfy job subscribers
		// all sit on per-request channels that the agent's read loop is
		// the sole producer for.  Without this fan-out, a rig crash
		// pins clients on their per-handler deadlines (2 min for chat,
		// 10 min for comfy) instead of surfacing the disconnect
		// immediately — and holds rate-limit slots in the meantime.
		a.peerMu.Lock()
		peers := a.inferPeers
		a.inferPeers = make(map[uint16]*inferPeer)
		pp := a.ppPeer
		a.ppPeer = nil
		dpp := a.dppPeer
		a.dppPeer = nil
		a.peerMu.Unlock()
		for _, ip := range peers {
			ip.close()
		}
		if pp != nil {
			pp.close()
		}
		if dpp != nil {
			dpp.close()
		}

		// Comfy job channels — we don't track the receiver's lifecycle
		// the way we do for inference handlers, so close each one and
		// let the consumer's `if !ok { … rig disconnected …}` branch
		// run.  Holds comfyMu so a racing deliverComfyResult either
		// sees an open chan and lands its message, or sees a deleted
		// key and drops cleanly — never sends on a closed chan.
		a.comfyMu.Lock()
		jobs := a.comfyResults
		a.comfyResults = nil
		meta := a.comfyMetaResults
		a.comfyMetaResults = nil
		a.comfyMu.Unlock()
		for _, ch := range jobs {
			close(ch)
		}
		for _, ch := range meta {
			close(ch)
		}
	})
}

// agentHello is the first WS frame sent by dist-node.  Two shapes share the
// struct:
//   {kind:"hello",  token:"…",     agent_id, hostname, …}  — pair bootstrap
//   {kind:"resume", agent_key:"…", agent_id, hostname, …}  — every reconnect
type agentHello struct {
	Kind      string `json:"kind"`
	Token     string `json:"token,omitempty"`
	AgentKey  string `json:"agent_key,omitempty"`
	AgentID   string `json:"agent_id"`
	Hostname  string `json:"hostname"`
	NGPUs     int    `json:"n_gpus"`
	VRAMBytes int64  `json:"vram_bytes"`
	// Signed-identity fields (optional during the rollout window).
	//   PubkeyB64 — sent once on first pair so the server can store it.
	//                On every resume, ignored unless the rig is rotating
	//                its key (out of scope today).
	PubkeyB64 string `json:"pubkey,omitempty"`

	// Wire protocol version the rig is built against.  Optional today
	// (legacy rigs leave it zero, treated as v1).  Future server changes
	// that break compatibility — new ACTV frame layout, mandatory
	// E2E encryption, etc. — bump serverProtocolMax and refuse rigs
	// below serverProtocolMin.  See protocol_version.go.
	ProtocolVersion int    `json:"protocol_version,omitempty"`
	ClientBuild     string `json:"client_build,omitempty"` // free-form build tag

	// CachedShards is what the rig has on disk at connect time.  Server
	// indexes these in rig_shards so the planner can prefer P2P fetch.
	// Empty / omitted on legacy rigs.
	CachedShards []cachedShardEntry `json:"cached_shards,omitempty"`

	// SPIFFEToken is the JWT-SVID a SPIFFE-enabled rig fetched from its
	// local workload API socket.  Empty on legacy rigs.  When present,
	// the server verifies it against the configured trust bundle and
	// uses the SPIFFE ID as the rig's identity instead of (or in
	// addition to) the paired agent_key.  See server/spiffe.go.
	SPIFFEToken string `json:"spiffe_token,omitempty"`
}

// cachedShardEntry is one row of CachedShards.  Wire-only; we project it
// into the rig_shards table.
type cachedShardEntry struct {
	ModelName string `json:"model"`
	File      string `json:"file"`
	SizeBytes int64  `json:"size_bytes,omitempty"`
}

type agentStatus struct {
	Kind     string    `json:"kind"`
	TokensPS float64   `json:"tokens_sec"`
	GPUUtil  []float64 `json:"gpu_util"`
	NGPUs    int       `json:"n_gpus"`

	// Optional richer telemetry — emitted by newer dist-node builds.  Older
	// rigs leave these zero/empty; the swarm dashboard treats them as
	// "unknown" but still counts the node.
	GPUModel    string   `json:"gpu_model,omitempty"`     // e.g. "RTX 3050 Laptop GPU"
	VRAMTotal   int64    `json:"vram_total,omitempty"`    // bytes
	VRAMFree    int64    `json:"vram_free,omitempty"`     // bytes
	UptimeSec   int64    `json:"uptime_sec,omitempty"`    // dist-node process uptime
	RolesHeld   []string `json:"roles,omitempty"`         // ["text_encoder","unet",...]
	ModelsHeld  []string `json:"models,omitempty"`        // huggingface repo ids served
	Inflight    int      `json:"inflight,omitempty"`      // requests being processed
	BWUpKbps    int64    `json:"bw_up_kbps,omitempty"`    // measured upload bw
	BWDnKbps    int64    `json:"bw_dn_kbps,omitempty"`    // measured download bw
	LastError   string   `json:"last_error,omitempty"`

	// Peer-relay candidacy.  A rig that has a public IPv4 or a friendly
	// (cone) NAT can volunteer to forward ACTV frames between peers that
	// are themselves both behind symmetric NATs.  Reported by dist-node
	// after its initial ICE gathering succeeds; values are advisory only.
	//
	//   NATType ∈ {"open", "cone", "symmetric", "blocked", "unknown"}
	//   RelayCapable — true iff the rig is willing to forward for others.
	NATType      string `json:"nat_type,omitempty"`
	RelayCapable bool   `json:"relay_capable,omitempty"`

	// TURN sidecar — when dist-node successfully spawns its bundled
	// dist-turn server, this is the UDP port it's listening on.  The
	// server combines the port with agentConn.remoteIP to assemble the
	// `turn:<ip>:<port>` URL it ships in ICE entries.  Zero means the
	// rig isn't running a TURN sidecar.
	CoturnPort int `json:"coturn_port,omitempty"`

	// PublicIP is the rig's server-reflexive (STUN-discovered) IPv4.
	// The server prefers this over the WS source IP when forming the
	// TURN URL — it's the correct UDP-reachable address even when WS
	// traffic flows through a different NAT mapping (1:1 NAT on EC2,
	// asymmetric routing, multi-homed hosts).
	PublicIP string `json:"public_ip,omitempty"`

	// CachedShards: delta-update of the rig's shard cache, sent any time
	// the rig finishes pulling a new shard.  We treat it as authoritative
	// (the rig knows what's on its own disk better than us), so a status
	// frame with non-empty CachedShards causes us to upsert the listed
	// entries — it does NOT clear unmentioned ones.  Disconnect clears
	// the rig's row set entirely; that's the only deletion path.
	CachedShards []cachedShardEntry `json:"cached_shards,omitempty"`

	// MaxConcurrent is how many simultaneous /v1/chat or /api/infer
	// requests this rig is willing to multiplex.  Defaults to 1 on
	// legacy rigs (preserves the historical "agent busy" semantics);
	// rigs running llama.cpp in --parallel N mode advertise N.  The
	// server uses len(inferPeers) < MaxConcurrent to admit new
	// requests instead of the old "any in-flight = busy" check.
	MaxConcurrent int `json:"max_concurrent,omitempty"`
}

// liveStatus is the server-side snapshot of an agent's latest telemetry
// plus a few server-derived fields (connecting IP, geo, paired-at).  It's
// protected by the agentConn.statusMu RWMutex and is the only structure
// the /api/swarm endpoint reads from when aggregating swarm-wide stats.
type liveStatus struct {
	UpdatedAt  int64    `json:"updated_at"`
	TokensPS   float64  `json:"tokens_sec"`
	GPUUtil    []float64 `json:"gpu_util,omitempty"`
	NGPUs      int      `json:"n_gpus"`
	GPUModel   string   `json:"gpu_model,omitempty"`
	VRAMTotal  int64    `json:"vram_total,omitempty"`
	VRAMFree   int64    `json:"vram_free,omitempty"`
	UptimeSec  int64    `json:"uptime_sec,omitempty"`
	RolesHeld  []string `json:"roles,omitempty"`
	ModelsHeld []string `json:"models,omitempty"`
	Inflight   int      `json:"inflight,omitempty"`
	BWUpKbps   int64    `json:"bw_up_kbps,omitempty"`
	BWDnKbps   int64    `json:"bw_dn_kbps,omitempty"`
	LastError  string   `json:"last_error,omitempty"`

	// Peer-relay candidacy (see agentStatus for semantics).  Populated from
	// each status frame so a rig can change its mind (e.g. roam from cone
	// to symmetric NAT) without re-pairing.
	NATType      string `json:"nat_type,omitempty"`
	RelayCapable bool   `json:"relay_capable,omitempty"`
	CoturnPort   int    `json:"coturn_port,omitempty"`
	PublicIP     string `json:"public_ip,omitempty"`

	// MaxConcurrent — rig-advertised parallel inference slots.  Zero
	// means "legacy, treat as 1".  See agentStatus.MaxConcurrent.
	MaxConcurrent int `json:"max_concurrent,omitempty"`
}

func (s *server) handleAgentWS(w http.ResponseWriter, r *http.Request) {
	srcIP := s.clientIP(r)

	// Per-IP throttle before any handshake work.  Burst of 3, refill 12s
	// (≈5/min) — generous for an honest rig that reconnects on transient
	// network failure, but a brute-force attacker spamming hello/resume
	// frames hits the wall after a few tries.  Brief jitter on close so
	// the rejection can't double as a timing oracle.
	if s.ipRL != nil && !s.ipRL.helloFail.peek(srcIP) {
		authFailJitter()
		// Accept then immediately close — we still need a websocket
		// session to send a clean status code to the client.
		conn, err := websocket.Accept(w, r, &websocket.AcceptOptions{
			InsecureSkipVerify: true,
			OriginPatterns:     []string{"*"},
		})
		if err == nil {
			_ = conn.Close(websocket.StatusTryAgainLater, "too many auth attempts; back off")
		}
		return
	}

	// The agent is not a browser — accept any origin.  Auth is via pair token
	// in the first message.
	//
	// Subprotocol negotiation: rigs that ship the proto wire opt in by
	// advertising `Sec-WebSocket-Protocol: distpool.proto.v1` on the
	// handshake.  We list it in Subprotocols so coder/websocket echoes
	// it back in the 101 response (RFC 6455 §4.2.2).  Legacy rigs send
	// no subprotocol header and stay on JSON.  See ws_codec.go.
	conn, err := websocket.Accept(w, r, &websocket.AcceptOptions{
		InsecureSkipVerify: true,
		OriginPatterns:     []string{"*"},
		Subprotocols:       []string{protoSubprotocolV1},
	})
	if err != nil {
		return
	}
	codec := codecForSubprotocol(conn.Subprotocol())
	wsConnDelta(r.Context(), "agent", 1)
	defer wsConnDelta(context.Background(), "agent", -1)

	// Default coder/websocket read limit is 32 KiB.  Comfy result frames
	// carry base64-encoded image/video bytes — a 512×512 PNG is ~750 KiB
	// after base64 and small videos run multi-MB.  Cap at 32 MiB so a
	// single oversized frame can't OOM the server, but real comfy outputs
	// fit comfortably.
	conn.SetReadLimit(32 << 20)

	ctx := r.Context()
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	// First message must arrive within 10s.  Accept either hello (first-time
	// pair) or resume (every subsequent reconnect).  Codec-aware so a
	// proto-negotiated rig can send its hello as binary.
	readCtx, rc := context.WithTimeout(ctx, 10*time.Second)
	var hello agentHello
	err = codecReadHello(readCtx, conn, codec, &hello)
	rc()
	if err != nil || hello.AgentID == "" ||
		(hello.Kind != "hello" && hello.Kind != "resume") ||
		(hello.Kind == "hello" && hello.Token == "") ||
		(hello.Kind == "resume" && hello.AgentKey == "") {
		s.failWSAuth(ctx, conn, srcIP, "expected hello or resume frame")
		return
	}

	// Protocol-version negotiation.  Rejecting too-old rigs *here* (before
	// any DB work) keeps a malformed rig from polluting our pair-token
	// store and gives the operator a clear upgrade message.
	negotiated, ok := negotiateProtocol(hello.ProtocolVersion)
	if !ok {
		_ = codecWrite(ctx, conn, codec, map[string]any{
			"kind":                "error",
			"message":             "dist-node is too old; upgrade to a newer build",
			"server_protocol_min": serverProtocolMin,
			"server_protocol_max": serverProtocolMax,
		})
		_ = conn.Close(websocket.StatusPolicyViolation, "protocol_version too old")
		log.Printf("rejected agent %s: protocol v%d (server requires ≥v%d)",
			hello.AgentID, hello.ProtocolVersion, serverProtocolMin)
		return
	}

	var (
		uid           int64
		preferPoolID  int64
		agentKey      string // sent back in welcome on first pair only
		isFirstPair   bool
	)

	// pubkeyBytes is set on first-pair (from hello.PubkeyB64) and on every
	// resume (from the rigs row).  Used below for signature verification.
	var pubkeyBytes []byte

	if hello.Kind == "hello" {
		var err2 error
		uid, preferPoolID, err2 = s.consumePairToken(hello.Token)
		if err2 != nil {
			s.failWSAuth(ctx, conn, srcIP, "bad pair token")
			return
		}
		// Mint a fresh agent_key that the rig persists for future reconnects.
		agentKey = newRandomToken(24)
		isFirstPair = true
		// Optional: the rig uploads its ed25519 pubkey on first pair.  If
		// missing we still accept (older rigs); we just can't verify sigs
		// on subsequent resumes for this rig.
		if hello.PubkeyB64 != "" {
			if pk, err := decodePubkeyB64(hello.PubkeyB64); err == nil {
				pubkeyBytes = pk
			} else {
				log.Printf("agent %s: ignoring bad pubkey: %v", hello.AgentID, err)
			}
		}
	} else {
		// resume: look up by agent_key_hash (the *secret* identifier) and
		// cross-check that the presented agent_id matches the row. An older
		// version keyed the WHERE on agent_id, which let an attacker who
		// learned the victim's agent_id squat the row from their own account.
		// Keying on the hash ties the lookup to the secret. A legacy fallback
		// covers rigs whose agent_key_hash hasn't been backfilled yet (NULL
		// hash, plaintext agent_key in legacy column).
		presentedHash := hashAgentKey(hello.AgentKey)
		var pkRaw []byte
		var storedHash sql.NullString
		var legacyPlain sql.NullString
		var rowAgentID string
		err2 := s.dbQueryRow(
			`SELECT user_id, pubkey, agent_key_hash, agent_key, agent_id
			   FROM rigs
			  WHERE agent_key_hash = ?
			  LIMIT 1`,
			presentedHash,
		).Scan(&uid, &pkRaw, &storedHash, &legacyPlain, &rowAgentID)
		if err2 != nil {
			// Backfill rescue: fall back to the legacy agent_id lookup if the
			// modern hash-keyed lookup missed (covers rows still on plaintext).
			err2 = s.dbQueryRow(
				`SELECT user_id, pubkey, agent_key_hash, agent_key, agent_id
				   FROM rigs WHERE agent_id = ? LIMIT 1`,
				hello.AgentID,
			).Scan(&uid, &pkRaw, &storedHash, &legacyPlain, &rowAgentID)
		}
		if err2 != nil || !verifyAgentKeyWithFallback(storedHash, legacyPlain, hello.AgentKey) {
			s.failWSAuth(ctx, conn, srcIP, "bad agent_key — re-run the pair flow from the dashboard")
			return
		}
		if rowAgentID != hello.AgentID {
			s.failWSAuth(ctx, conn, srcIP, "agent_id does not match this agent_key")
			return
		}
		pubkeyBytes = pkRaw

		// If the rig has a registered pubkey, demand a signed challenge.
		// Rigs paired before this feature shipped have a NULL pubkey and
		// fall through on agent_key alone — we log a warning so the
		// operator can see which rigs need to re-pair to get signed IDs.
		if len(pubkeyBytes) > 0 {
			nonce, ts, err := mintChallenge()
			if err != nil {
				log.Printf("agent %s: mint challenge: %v", hello.AgentID, err)
				_ = conn.Close(websocket.StatusInternalError, "challenge")
				return
			}
			if err := codecWrite(ctx, conn, codec, map[string]any{
				"kind":  "challenge",
				"nonce": nonce,
				"ts":    ts,
			}); err != nil {
				return
			}
			sigCtx, scancel := context.WithTimeout(ctx, challengeTimeout)
			var sigMsg struct {
				Kind   string `json:"kind"`
				SigB64 string `json:"sig"`
			}
			err = codecReadInto(sigCtx, conn, codec, &sigMsg)
			scancel()
			if err != nil || sigMsg.Kind != "sig" || sigMsg.SigB64 == "" {
				s.failWSAuth(ctx, conn, srcIP, "missing or bad sig frame")
				return
			}
			sigBytes, err := decodeSigB64(sigMsg.SigB64)
			if err != nil {
				s.failWSAuth(ctx, conn, srcIP, "sig decode failed")
				return
			}
			if err := verifyAgentSig(pubkeyBytes, hello.AgentID, nonce, ts, sigBytes); err != nil {
				log.Printf("agent %s: sig verify failed: %v", hello.AgentID, err)
				s.failWSAuth(ctx, conn, srcIP, "sig verify failed")
				return
			}
		} else {
			log.Printf("agent %s: no registered pubkey — accepting on agent_key alone (legacy rig)", hello.AgentID)
		}
	}

	// Upsert the rig.  On first pair, also persist the new agent_key + the
	// uploaded ed25519 pubkey (if any); on resume, leave both in place.
	if isFirstPair {
		var pkArg any = nil
		if len(pubkeyBytes) > 0 {
			pkArg = pubkeyBytes
		}
		// Store hash, not plaintext.  We blank out the legacy agent_key
		// column so a DB dump never exposes the bearer credential.
		akHash := hashAgentKey(agentKey)
		_, err = s.dbExec(`INSERT INTO rigs
			(user_id, agent_id, hostname, n_gpus, vram_bytes, last_seen, agent_key, agent_key_hash, pubkey)
			VALUES (?, ?, ?, ?, ?, ?, '', ?, ?)
			ON CONFLICT (user_id, agent_id)
			DO UPDATE SET hostname        = excluded.hostname,
			              n_gpus          = excluded.n_gpus,
			              vram_bytes      = excluded.vram_bytes,
			              last_seen       = excluded.last_seen,
			              agent_key       = '',
			              agent_key_hash  = excluded.agent_key_hash,
			              pubkey          = COALESCE(excluded.pubkey, rigs.pubkey)`,
			uid, hello.AgentID, hello.Hostname, hello.NGPUs, hello.VRAMBytes, nowUnix(), akHash, pkArg,
		)
	} else {
		_, err = s.dbExec(`UPDATE rigs SET hostname = ?, n_gpus = ?, vram_bytes = ?, last_seen = ?
			WHERE user_id = ? AND agent_id = ?`,
			hello.Hostname, hello.NGPUs, hello.VRAMBytes, nowUnix(), uid, hello.AgentID,
		)
	}
	if err != nil {
		log.Printf("rig upsert: %v", err)
	}

	// SPIFFE workload identity (P6).  Additive in this revision: we
	// verify the SVID + record the resulting SPIFFE ID on the rig row,
	// but agent_key remains the load-bearing credential.  A follow-up
	// can flip the dependency so a valid SVID replaces agent_key
	// entirely — for now we just persist the binding so operators can
	// roll out SPIRE without breaking the existing fleet.
	if s.cfg.spiffe != nil {
		if id, ok := s.spiffeFromAgentHello(r, &hello); ok {
			if _, derr := s.dbExec(
				`UPDATE rigs SET spiffe_id = ? WHERE user_id = ? AND agent_id = ?`,
				id.String(), uid, hello.AgentID,
			); derr != nil {
				log.Printf("rig %s: spiffe_id write: %v", hello.AgentID, derr)
			} else {
				log.Printf("agent %s: bound to SPIFFE ID %s", hello.AgentID, id)
			}
		}
	}

	// Auto-attach this rig to the pool the browser pre-selected during
	// install.  Only runs on first pair — resume reconnects keep whatever
	// pool_rigs rows already exist.
	if isFirstPair && preferPoolID != 0 {
		var rigRowID int64
		if err := s.dbQueryRow(
			`SELECT id FROM rigs WHERE user_id = ? AND agent_id = ?`,
			uid, hello.AgentID,
		).Scan(&rigRowID); err == nil {
			if _, isMember := s.userIsMember(preferPoolID, uid); isMember {
				if _, err := s.dbExec(
					`INSERT OR IGNORE INTO pool_rigs (pool_id, rig_id, added_at) VALUES (?, ?, ?)`,
					preferPoolID, rigRowID, nowUnix(),
				); err != nil {
					log.Printf("pool auto-attach: %v", err)
				} else {
					log.Printf("auto-attached rig %d to pool %d", rigRowID, preferPoolID)
				}
			}
		}
	}

	// Fetch display name for welcome.
	var displayName string
	_ = s.dbQueryRow(`SELECT display_name FROM users WHERE id = ?`, uid).
		Scan(&displayName)

	welcome := map[string]any{
		"kind":             "welcome",
		"user_id":          uid,
		"display_name":     displayName,
		"protocol_version": negotiated,
		"server_protocol_min": serverProtocolMin,
		"server_protocol_max": serverProtocolMax,
	}
	if isFirstPair {
		// Only surface the key on the frame that mints it.  On resume we
		// don't re-emit — the rig already has it on disk.
		welcome["agent_key"] = agentKey
	}
	// TURN sidecar secret — derived per-rig from the operator's master
	// secret so a compromised rig only exposes its own credentials.  The
	// derivation is HMAC-SHA256(masterSecret, agent_id) hex-encoded; the
	// server applies the same derivation when minting creds targeted at
	// this rig's sidecar (see rigTURNSecret + pickPeerRelays).
	if s.cfg.turnSecret != "" {
		welcome["turn_secret"] = s.rigTURNSecret(hello.AgentID)
		if s.cfg.turnRealm != "" {
			welcome["turn_realm"] = s.cfg.turnRealm
		} else {
			welcome["turn_realm"] = "dist"
		}
	}
	_ = codecWrite(ctx, conn, codec, welcome)

	ac := &agentConn{
		userID:          uid,
		agentID:         hello.AgentID,
		hostname:        hello.Hostname,
		conn:            conn,
		outCh:           make(chan any, 64),
		binCh:           make(chan []byte, 64),
		closed:          make(chan struct{}),
		inferPeers:      make(map[uint16]*inferPeer),
		remoteIP:        s.clientIP(r),
		pairedAt:        nowUnix(),
		protocolVersion: negotiated,
	}
	ac.live.NGPUs = hello.NGPUs
	ac.live.VRAMTotal = hello.VRAMBytes
	s.hub.registerAgent(ac)
	// A new rig changes which pools have which capacity; flush the
	// per-pool cost snapshot cache so planPipeline doesn't ignore the
	// rig for up to costCache.ttl.  Cheap (map reset under a mutex).
	if s.costCache != nil {
		s.costCache.invalidateAll()
	}

	// Shard cache index — the rig tells us what's on disk so we can
	// route future shard fetches to peers instead of the origin.  We
	// clear before upsert so a rig that GC'd between sessions doesn't
	// leave dangling claims.  Empty CachedShards on a legacy rig leaves
	// the cleared state intact, which is the correct default.
	s.clearRigShards(uid, hello.AgentID)
	if len(hello.CachedShards) > 0 {
		s.upsertRigShards(uid, hello.AgentID, hello.CachedShards)
	}
	defer func() {
		s.hub.unregisterAgent(ac)
		// The plan can no longer route through this rig; drop cached
		// cost snapshots so the next request observes the absence
		// without waiting out the TTL.  Stage-level safety (hub.findAgent
		// re-check before dispatch) already protects against the small
		// window where the cache could still mention this rig.
		if s.costCache != nil {
			s.costCache.invalidateAll()
		}
		// Index claims are advisory and become misleading the moment
		// the rig leaves — clear them now so /api/models/.../peers
		// doesn't point downloaders at a dead WS.
		s.clearRigShards(uid, hello.AgentID)
		// Any active relay session this rig was serving is now dead.
		// Mark each as a failure for the rig's reputation row — that
		// pulls its score down for the next planner round so we stop
		// picking it until it stabilises.  The compute stages on
		// either side will observe the dropped P2P channel and the
		// client will retry; the new plan won't include this rig.
		if s.relays != nil {
			for _, a := range s.relays.drainForAgent(ac.agentID) {
				s.recordRelayFailure(a.AgentID)
				log.Printf("relay %s disconnected mid-session %s (left=%s right=%s) — marked failed",
					a.AgentID, a.SessionID, a.LeftPeer, a.RightPeer)
			}
		}
	}()

	// Tell the browser(s) the rig came online.  is_first_pair lets the UI
	// flip the install card to a ✅ ("rig paired!") without waiting for a
	// manual refresh — see the install picker status line.
	s.hub.broadcastToUser(uid, "rig_online", map[string]any{
		"agent_id":      hello.AgentID,
		"hostname":      hello.Hostname,
		"n_gpus":        hello.NGPUs,
		"vram_bytes":    hello.VRAMBytes,
		"is_first_pair": isFirstPair,
	})

	// Ping loop — catches half-open TCP connections (the peer NIC went away
	// but no RST arrived).  A failed Ping closes ac.closed which unblocks the
	// reader loop on the next Read.  Ping/pong is concurrency-safe on
	// coder/websocket, so we don't need to route it through the writer.
	go func() {
		t := time.NewTicker(20 * time.Second)
		defer t.Stop()
		for {
			select {
			case <-ac.closed:
				return
			case <-ctx.Done():
				return
			case <-t.C:
				// Ping timeout is generous (30s) because some agents reach the
				// control plane over high-latency or bandwidth-throttled paths
				// (reverse SSH tunnels, residential uplinks).  Half-open TCP is
				// still caught — just on a longer fuse.
				pctx, pc := context.WithTimeout(ctx, 30*time.Second)
				err := conn.Ping(pctx)
				pc()
				if err != nil {
					log.Printf("agent %s: ping failed, closing: %v", hello.AgentID, err)
					ac.close()
					_ = conn.Close(websocket.StatusGoingAway, "ping timeout")
					return
				}
			}
		}
	}()

	// Writer — serialises JSON control frames AND relay binary frames on the
	// single WS connection.  Concurrent writes on *websocket.Conn are unsafe,
	// so both kinds go through this goroutine.
	go func() {
		for {
			select {
			case <-ac.closed:
				return
			case <-ctx.Done():
				return
			case msg := <-ac.outCh:
				wctx, c := context.WithTimeout(ctx, 5*time.Second)
				err := codecWrite(wctx, conn, codec, msg)
				c()
				if err != nil {
					ac.close()
					return
				}
			case b := <-ac.binCh:
				wctx, c := context.WithTimeout(ctx, 5*time.Second)
				err := conn.Write(wctx, websocket.MessageBinary, b)
				c()
				if err != nil {
					ac.close()
					return
				}
			}
		}
	}()

	// Reader loop.  The codec decides what's a control frame vs a
	// relay passthrough — see ws_codec.go.  On the JSON wire, TEXT
	// frames are control and BINARY frames are relay; on the proto
	// wire control and relay are both BINARY, distinguished by whether
	// the bytes parse as a ClientFrame.
	for {
		mt, data, err := conn.Read(ctx)
		if err != nil {
			break
		}
		msg, isRelay, derr := codec.decodeClient(mt, data)
		if isRelay {
			// Route the frame: first chance to the inference peer (if any
			// request is in flight), fallback to the raw relay peer.
			_ = ac.dispatchBinaryFromAgent(data)
			continue
		}
		if derr != nil || msg == nil {
			continue
		}
		{
			kind, _ := msg["kind"].(string)
			if kind == "status" {
				// Hot path: coalesce into in-memory map, flushed every 1s.
				s.markLastSeen(uid, hello.AgentID)
				// Decode rich telemetry into ac.live.  We re-marshal so
				// json.Unmarshal handles type conversions; the wire copy
				// is small.
				var st agentStatus
				if data2, err := json.Marshal(msg); err == nil {
					if json.Unmarshal(data2, &st) == nil {
						validateAndClampStatus(&st, ac.remoteIP)
						ac.updateStatus(st)
						// Shard cache delta (rig just finished pulling a
						// new shard).  Status frames carry adds only;
						// removals only happen on disconnect.
						if len(st.CachedShards) > 0 {
							s.upsertRigShards(uid, hello.AgentID, st.CachedShards)
						}
					}
				}
			}
			if kind == "comfy_result" {
				if ac.deliverComfyResult(msg) {
					continue
				}
			}
			if kind == "comfy_meta_result" {
				if ac.deliverComfyMetaResult(msg) {
					continue
				}
			}
			if kind == "dpp_progress" {
				s.ingestDPPProgress(uid, hello.AgentID, msg)
				continue
			}
			if kind == "comfy_caps" {
				s.upsertComfyCaps(uid, hello.AgentID, msg)
				continue
			}
			if kind == "sglang_caps" {
				s.upsertSglangCaps(uid, hello.AgentID, msg)
				continue
			}
			if kind == "vllm_caps" {
				// Same shape (ok, base_url) — store minimally so the
				// control plane can route full-model requests.  No
				// per-prefix routing for vLLM; that's SGLang's edge.
				_, _ = s.dbExec(
					`INSERT INTO sglang_caps (user_id, agent_id, ok, base_url, prefix_cache, updated_at)
					 VALUES (?, ?, ?, ?, ?, ?)
					 ON CONFLICT(user_id, agent_id) DO UPDATE SET
					   ok = excluded.ok,
					   base_url = excluded.base_url,
					   updated_at = excluded.updated_at`,
					uid, hello.AgentID,
					func() int { if v, _ := msg["ok"].(bool); v { return 1 }; return 0 }(),
					func() string { v, _ := msg["base_url"].(string); return v }(),
					0,
					nowUnix(),
				)
				continue
			}
			if kind == "sdcpp_caps" {
				// CF12-W4: per-role sd.cpp capability claim.  Stored in
				// the sdcpp_caps table so the dispatcher can pick a
				// TE/UNet/VAE chain (or fall back to "full"-pipeline
				// routing on a single rig).
				s.upsertSdcppCaps(uid, hello.AgentID, msg)
				continue
			}
			if kind == "sdcpp_progress" || kind == "sdcpp_role_done" ||
				kind == "sdcpp_done" || kind == "sdcpp_error" ||
				kind == "sdcpp_need_denoise" {
				// Route the frame into the per-req_id channel owned by
				// the runSdcppComfyJob goroutine.  Stale frames (job
				// already errored / unsubscribed) are silently dropped.
				s.ingestSdcppFrame(hello.AgentID, kind, msg)
				continue
			}
			if kind == "spec_caps" {
				// P16: speculative-decoding capability claim.  Stored in
				// its own table; the dispatcher will look at it when
				// routing latency-sensitive requests.
				s.upsertSpecCaps(uid, hello.AgentID, msg)
				continue
			}
			if kind == "trtllm_caps" {
				// P14: TensorRT-LLM tier (via Triton).  Triton owns the
				// engine plan and per-model prefix-cache config; we treat
				// it like vLLM from the dispatcher's perspective (no
				// per-prefix cache_tokens signal) and store ok/base_url.
				_, _ = s.dbExec(
					`INSERT INTO sglang_caps (user_id, agent_id, ok, base_url, prefix_cache, updated_at)
					 VALUES (?, ?, ?, ?, ?, ?)
					 ON CONFLICT(user_id, agent_id) DO UPDATE SET
					   ok = excluded.ok,
					   base_url = excluded.base_url,
					   updated_at = excluded.updated_at`,
					uid, hello.AgentID,
					func() int { if v, _ := msg["ok"].(bool); v { return 1 }; return 0 }(),
					func() string { v, _ := msg["base_url"].(string); return v }(),
					0,
					nowUnix(),
				)
				continue
			}
			if kind == "relay_stats" {
				// Relay rig is reporting its byte counters from a finished
				// session.  We attribute the count to the assignment record
				// in-flight; the actual reputation row is updated when the
				// release path runs `recordRelaySuccess(sum)`.
				//
				// Two clamps are applied: (1) absolute per-session ceiling
				// (clampRelayBytes), (2) physically-plausible throughput
				// over the assignment's wall-clock window
				// (clampRelayBytesByElapsed).  Both must pass to credit
				// reputation; either being applied is suspicious enough to
				// log.
				if s.relays != nil {
					sid, _ := msg["session_id"].(string)
					var l2rRaw, r2lRaw int64
					var clampedL, clampedR bool
					if v, ok := msg["bytes_l2r"].(float64); ok {
						l2rRaw, clampedL = clampRelayBytes(int64(v))
					}
					if v, ok := msg["bytes_r2l"].(float64); ok {
						r2lRaw, clampedR = clampRelayBytes(int64(v))
					}
					s.relays.mu.Lock()
					a := s.relays.byKey[relayKey{sid, hello.AgentID}]
					if a != nil {
						now := nowUnix()
						l2r, eClampL := clampRelayBytesByElapsed(l2rRaw, a.StartedAt, now)
						r2l, eClampR := clampRelayBytesByElapsed(r2lRaw, a.StartedAt, now)
						a.BytesL2R = l2r
						a.BytesR2L = r2l
						if clampedL || clampedR || eClampL || eClampR {
							log.Printf("relay_stats: clamped bytes from agent %s session %s elapsed=%ds (raw l2r=%v r2l=%v -> %d/%d)",
								hello.AgentID, sid, now-a.StartedAt,
								msg["bytes_l2r"], msg["bytes_r2l"], l2r, r2l)
						}
					}
					s.relays.mu.Unlock()
				}
				continue
			}
			if deliverSignalAnswer(msg) {
				continue
			}
			if s.deliverP2PSignal(uid, hello.AgentID, msg) {
				continue
			}
			msg["agent_id"] = hello.AgentID
			s.hub.broadcastToUser(uid, "agent_message", msg)
		}
	}

	ac.close()
	_ = conn.Close(websocket.StatusNormalClosure, "")
	s.hub.broadcastToUser(uid, "rig_offline", map[string]any{
		"agent_id": hello.AgentID,
	})
}

// ─── json helpers ───────────────────────────────────────────────────────────
//
// `coder/websocket` has `wsjson`, but depending on it here would add another
// dep; inline tiny helpers.

func wsjsonWrite(ctx context.Context, c *websocket.Conn, v any) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	return c.Write(ctx, websocket.MessageText, b)
}

func wsjsonRead(ctx context.Context, c *websocket.Conn, v any) error {
	_, b, err := c.Read(ctx)
	if err != nil {
		return err
	}
	return json.Unmarshal(b, v)
}

// codecWrite sends a server-emitted control frame through the
// negotiated codec.  Behaves identically to wsjsonWrite when the
// codec is jsonCodec.
func codecWrite(ctx context.Context, c *websocket.Conn, codec wireCodec, msg any) error {
	mt, data, err := codec.encodeServer(msg)
	if err != nil {
		return err
	}
	return c.Write(ctx, mt, data)
}

// codecReadHello reads the first /ws/agent frame and unmarshals it
// into the typed agentHello struct.  Goes through the codec so a
// proto-negotiated rig can send its hello as a BINARY ClientFrame
// rather than as TEXT JSON.  Returns an error if the wire frame
// cannot be parsed.
func codecReadHello(ctx context.Context, c *websocket.Conn, codec wireCodec, out *agentHello) error {
	return codecReadInto(ctx, c, codec, out)
}

// codecReadInto reads one control frame through the codec and decodes
// it into a typed Go value via JSON tags.  We round-trip through JSON
// because (a) the codec normalizes to map[string]any anyway, (b) the
// destination struct's json tags handle the per-field type coercion
// (float64 → int, etc.).
func codecReadInto(ctx context.Context, c *websocket.Conn, codec wireCodec, out any) error {
	mt, data, err := c.Read(ctx)
	if err != nil {
		return err
	}
	msg, isRelay, err := codec.decodeClient(mt, data)
	if err != nil {
		return err
	}
	if isRelay || msg == nil {
		return errors.New("expected control frame, got relay or empty")
	}
	raw, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	return json.Unmarshal(raw, out)
}
