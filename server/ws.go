package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coder/websocket"
)

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
	peerMu    sync.RWMutex
	peer      *clientConn
	inferPeer *inferPeer // set while an /api/infer request is in flight
	ppPeer    *ppPeer    // set while this agent is a stage in an /api/infer_pp request
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
	a.once.Do(func() { close(a.closed) })
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
}

type agentStatus struct {
	Kind     string    `json:"kind"`
	TokensPS float64   `json:"tokens_sec"`
	GPUUtil  []float64 `json:"gpu_util"`
	NGPUs    int       `json:"n_gpus"`
}

func (s *server) handleAgentWS(w http.ResponseWriter, r *http.Request) {
	// The agent is not a browser — accept any origin.  Auth is via pair token
	// in the first message.
	conn, err := websocket.Accept(w, r, &websocket.AcceptOptions{
		InsecureSkipVerify:   true,
		OriginPatterns:       []string{"*"},
	})
	if err != nil {
		return
	}

	ctx := r.Context()
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	// First message must arrive within 10s.  Accept either hello (first-time
	// pair) or resume (every subsequent reconnect).
	readCtx, rc := context.WithTimeout(ctx, 10*time.Second)
	var hello agentHello
	err = wsjsonRead(readCtx, conn, &hello)
	rc()
	if err != nil || hello.AgentID == "" ||
		(hello.Kind != "hello" && hello.Kind != "resume") ||
		(hello.Kind == "hello" && hello.Token == "") ||
		(hello.Kind == "resume" && hello.AgentKey == "") {
		_ = wsjsonWrite(ctx, conn, map[string]any{
			"kind": "error", "message": "expected hello or resume frame",
		})
		_ = conn.Close(websocket.StatusPolicyViolation, "bad hello")
		return
	}

	var (
		uid           int64
		preferPoolID  int64
		agentKey      string // sent back in welcome on first pair only
		isFirstPair   bool
	)

	if hello.Kind == "hello" {
		var err2 error
		uid, preferPoolID, err2 = s.consumePairToken(hello.Token)
		if err2 != nil {
			_ = wsjsonWrite(ctx, conn, map[string]any{
				"kind": "error", "message": "bad pair token",
			})
			_ = conn.Close(websocket.StatusPolicyViolation, "bad token")
			return
		}
		// Mint a fresh agent_key that the rig persists for future reconnects.
		agentKey = newRandomToken(24)
		isFirstPair = true
	} else {
		// resume: look up (agent_id, agent_key) → user_id.  Constant-time-ish
		// match on a 48-char random token; not meaningfully attackable.
		err2 := s.db.QueryRow(
			`SELECT user_id FROM rigs WHERE agent_id = ? AND agent_key = ?`,
			hello.AgentID, hello.AgentKey,
		).Scan(&uid)
		if err2 != nil {
			_ = wsjsonWrite(ctx, conn, map[string]any{
				"kind": "error", "message": "bad agent_key — re-run the pair flow from the dashboard",
			})
			_ = conn.Close(websocket.StatusPolicyViolation, "bad agent_key")
			return
		}
	}

	// Upsert the rig.  On first pair, also persist the new agent_key; on
	// resume, leave it in place.
	if isFirstPair {
		_, err = s.db.Exec(`INSERT INTO rigs
			(user_id, agent_id, hostname, n_gpus, vram_bytes, last_seen, agent_key)
			VALUES (?, ?, ?, ?, ?, ?, ?)
			ON CONFLICT (user_id, agent_id)
			DO UPDATE SET hostname   = excluded.hostname,
			              n_gpus     = excluded.n_gpus,
			              vram_bytes = excluded.vram_bytes,
			              last_seen  = excluded.last_seen,
			              agent_key  = excluded.agent_key`,
			uid, hello.AgentID, hello.Hostname, hello.NGPUs, hello.VRAMBytes, nowUnix(), agentKey,
		)
	} else {
		_, err = s.db.Exec(`UPDATE rigs SET hostname = ?, n_gpus = ?, vram_bytes = ?, last_seen = ?
			WHERE user_id = ? AND agent_id = ?`,
			hello.Hostname, hello.NGPUs, hello.VRAMBytes, nowUnix(), uid, hello.AgentID,
		)
	}
	if err != nil {
		log.Printf("rig upsert: %v", err)
	}

	// Auto-attach this rig to the pool the browser pre-selected during
	// install.  Only runs on first pair — resume reconnects keep whatever
	// pool_rigs rows already exist.
	if isFirstPair && preferPoolID != 0 {
		var rigRowID int64
		if err := s.db.QueryRow(
			`SELECT id FROM rigs WHERE user_id = ? AND agent_id = ?`,
			uid, hello.AgentID,
		).Scan(&rigRowID); err == nil {
			if _, isMember := s.userIsMember(preferPoolID, uid); isMember {
				if _, err := s.db.Exec(
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
	_ = s.db.QueryRow(`SELECT display_name FROM users WHERE id = ?`, uid).
		Scan(&displayName)

	welcome := map[string]any{
		"kind":         "welcome",
		"user_id":      uid,
		"display_name": displayName,
	}
	if isFirstPair {
		// Only surface the key on the frame that mints it.  On resume we
		// don't re-emit — the rig already has it on disk.
		welcome["agent_key"] = agentKey
	}
	_ = wsjsonWrite(ctx, conn, welcome)

	ac := &agentConn{
		userID:   uid,
		agentID:  hello.AgentID,
		hostname: hello.Hostname,
		conn:     conn,
		outCh:    make(chan any, 64),
		binCh:    make(chan []byte, 64),
		closed:   make(chan struct{}),
	}
	s.hub.registerAgent(ac)
	defer s.hub.unregisterAgent(ac)

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
				pctx, pc := context.WithTimeout(ctx, 10*time.Second)
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
				err := wsjsonWrite(wctx, conn, msg)
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

	// Reader loop.  Text frames are JSON control messages (status, etc.).
	// Binary frames are forwarded to the bound relay client, if any.
	for {
		mt, data, err := conn.Read(ctx)
		if err != nil {
			break
		}
		switch mt {
		case websocket.MessageText:
			var msg map[string]any
			if err := json.Unmarshal(data, &msg); err != nil {
				continue
			}
			if kind, _ := msg["kind"].(string); kind == "status" {
				// Hot path: coalesce into in-memory map, flushed every 1s.
				s.markLastSeen(uid, hello.AgentID)
			}
			if deliverSignalAnswer(msg) {
				continue
			}
			msg["agent_id"] = hello.AgentID
			s.hub.broadcastToUser(uid, "agent_message", msg)
		case websocket.MessageBinary:
			// Route the frame: first chance to the inference peer (if any
			// request is in flight), fallback to the raw relay peer.
			_ = ac.dispatchBinaryFromAgent(data)
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
