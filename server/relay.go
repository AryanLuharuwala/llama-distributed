package main

// Client relay (/ws/client).
//
// Browser (or dist-client --relay) opens this socket and sends one initial
// text frame:
//
//   {"kind":"relay_open", "pool_id": N}
//
// The server picks an online rig in that pool, binds the two sockets, and
// replies with:
//
//   {"kind":"relay_ready", "pool_id": N, "agent_id": "...", "hostname": "..."}
//
// From then on, every BINARY frame from the client is forwarded verbatim to
// the chosen agent, and every BINARY frame from that agent is forwarded
// back to the client.  When either side closes, the other is closed too.

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/coder/websocket"
)

type clientConn struct {
	userID int64
	poolID int64
	conn   *websocket.Conn
	outCh  chan any
	binCh  chan []byte
	closed chan struct{}
	once   sync.Once
}

func (c *clientConn) sendBin(b []byte) bool {
	select {
	case c.binCh <- b:
		return true
	default:
		return false
	}
}

func (c *clientConn) close() {
	c.once.Do(func() { close(c.closed) })
}

type relayOpen struct {
	Kind   string `json:"kind"`
	PoolID int64  `json:"pool_id"`
}

func (s *server) handleClientWS(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}

	conn, err := websocket.Accept(w, r, &websocket.AcceptOptions{
		InsecureSkipVerify: false,
	})
	if err != nil {
		return
	}

	ctx := r.Context()
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	// Expect relay_open within 5 s.
	openCtx, oc := context.WithTimeout(ctx, 5*time.Second)
	var open relayOpen
	err = wsjsonRead(openCtx, conn, &open)
	oc()
	if err != nil || open.Kind != "relay_open" || open.PoolID == 0 {
		_ = wsjsonWrite(ctx, conn, map[string]any{
			"kind": "error", "message": "expected relay_open",
		})
		_ = conn.Close(websocket.StatusPolicyViolation, "bad open")
		return
	}

	// Membership check — must be a member of the pool, OR the pool is public.
	vis, _, ok := s.poolVisibility(open.PoolID)
	if !ok {
		_ = wsjsonWrite(ctx, conn, map[string]any{"kind": "error", "message": "pool not found"})
		_ = conn.Close(websocket.StatusPolicyViolation, "no pool")
		return
	}
	if _, member := s.userIsMember(open.PoolID, u.ID); !member && vis != "public" {
		_ = wsjsonWrite(ctx, conn, map[string]any{"kind": "error", "message": "not a member"})
		_ = conn.Close(websocket.StatusPolicyViolation, "not member")
		return
	}

	// Pick any online rig in the pool.
	ac, ok := s.pickOnlineRigInPool(open.PoolID)
	if !ok {
		_ = wsjsonWrite(ctx, conn, map[string]any{
			"kind": "error", "message": "no online rigs in pool",
		})
		_ = conn.Close(websocket.StatusPolicyViolation, "no rigs")
		return
	}

	// Bind client to agent.  Refuse if the agent already has a peer.
	ac.peerMu.Lock()
	if ac.peer != nil {
		ac.peerMu.Unlock()
		_ = wsjsonWrite(ctx, conn, map[string]any{
			"kind": "error", "message": "agent busy",
		})
		_ = conn.Close(websocket.StatusTryAgainLater, "busy")
		return
	}
	cc := &clientConn{
		userID: u.ID,
		poolID: open.PoolID,
		conn:   conn,
		outCh:  make(chan any, 16),
		binCh:  make(chan []byte, 64),
		closed: make(chan struct{}),
	}
	ac.peer = cc
	ac.peerMu.Unlock()

	defer func() {
		ac.peerMu.Lock()
		if ac.peer == cc {
			ac.peer = nil
		}
		ac.peerMu.Unlock()
	}()

	_ = wsjsonWrite(ctx, conn, map[string]any{
		"kind":     "relay_ready",
		"pool_id":  open.PoolID,
		"agent_id": ac.agentID,
		"hostname": ac.hostname,
	})

	// Writer: JSON control + binary forwarded from agent.
	go func() {
		for {
			select {
			case <-cc.closed:
				return
			case <-ctx.Done():
				return
			case msg := <-cc.outCh:
				wctx, c := context.WithTimeout(ctx, 5*time.Second)
				err := wsjsonWrite(wctx, conn, msg)
				c()
				if err != nil {
					cc.close()
					return
				}
			case b := <-cc.binCh:
				wctx, c := context.WithTimeout(ctx, 5*time.Second)
				err := conn.Write(wctx, websocket.MessageBinary, b)
				c()
				if err != nil {
					cc.close()
					return
				}
			}
		}
	}()

	// Reader: text frames are treated as control (only close on unknown);
	// binary frames are forwarded to the agent.
	for {
		mt, data, err := conn.Read(ctx)
		if err != nil {
			break
		}
		switch mt {
		case websocket.MessageText:
			// Best-effort parse.  Unknown kinds are ignored.
			var msg map[string]any
			_ = json.Unmarshal(data, &msg)
			if k, _ := msg["kind"].(string); k == "close" {
				break
			}
		case websocket.MessageBinary:
			buf := make([]byte, len(data))
			copy(buf, data)
			if !ac.sendBin(buf) {
				log.Printf("relay: agent buffer full, dropping %d bytes", len(buf))
			}
		}
	}

	cc.close()
	_ = conn.Close(websocket.StatusNormalClosure, "")
}
