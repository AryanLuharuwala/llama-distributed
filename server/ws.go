package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"sync"
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

func (b *browserConn) send(v any) {
	select {
	case b.outCh <- v:
	default:
		// drop — slow client
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

	// Relay binding: if non-nil, binary frames from this agent are forwarded
	// to the bound client.  Protected by peerMu.
	peerMu    sync.RWMutex
	peer      *clientConn
	inferPeer *inferPeer // set while an /api/infer request is in flight
	ppPeer    *ppPeer    // set while this agent is a stage in an /api/infer_pp request
}

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

type agentHello struct {
	Kind      string `json:"kind"`
	Token     string `json:"token"`
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

	// First message must be hello within 10s.
	readCtx, rc := context.WithTimeout(ctx, 10*time.Second)
	var hello agentHello
	err = wsjsonRead(readCtx, conn, &hello)
	rc()
	if err != nil || hello.Kind != "hello" || hello.Token == "" || hello.AgentID == "" {
		_ = wsjsonWrite(ctx, conn, map[string]any{
			"kind": "error", "message": "expected hello",
		})
		_ = conn.Close(websocket.StatusPolicyViolation, "bad hello")
		return
	}

	uid, err := s.consumePairToken(hello.Token)
	if err != nil {
		_ = wsjsonWrite(ctx, conn, map[string]any{
			"kind": "error", "message": "bad pair token",
		})
		_ = conn.Close(websocket.StatusPolicyViolation, "bad token")
		return
	}

	// Upsert the rig.
	_, err = s.db.Exec(`INSERT INTO rigs
		(user_id, agent_id, hostname, n_gpus, vram_bytes, last_seen)
		VALUES (?, ?, ?, ?, ?, ?)
		ON CONFLICT (user_id, agent_id)
		DO UPDATE SET hostname = excluded.hostname,
		              n_gpus = excluded.n_gpus,
		              vram_bytes = excluded.vram_bytes,
		              last_seen = excluded.last_seen`,
		uid, hello.AgentID, hello.Hostname, hello.NGPUs, hello.VRAMBytes, nowUnix(),
	)
	if err != nil {
		log.Printf("rig upsert: %v", err)
	}

	// Fetch display name for welcome.
	var displayName string
	_ = s.db.QueryRow(`SELECT display_name FROM users WHERE id = ?`, uid).
		Scan(&displayName)

	_ = wsjsonWrite(ctx, conn, map[string]any{
		"kind":         "welcome",
		"user_id":      uid,
		"display_name": displayName,
	})

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

	// Tell the browser(s) the rig came online.
	s.hub.broadcastToUser(uid, "rig_online", map[string]any{
		"agent_id":   hello.AgentID,
		"hostname":   hello.Hostname,
		"n_gpus":     hello.NGPUs,
		"vram_bytes": hello.VRAMBytes,
	})

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
				_, _ = s.db.Exec(`UPDATE rigs SET last_seen = ? WHERE user_id = ? AND agent_id = ?`,
					nowUnix(), uid, hello.AgentID)
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
