package main

// P2P signalling.
//
// The browser runs a WebRTC data channel to bypass the server relay whenever
// possible.  We don't terminate SDP ourselves — we just ferry offers/answers
// between the browser and the selected agent over the agent's control-plane
// WebSocket.
//
// POST /api/signal { pool_id, offer:{...} }
//   → { answer:{...}, ice_servers:[...] }       (2xx with SDP answer)
//   → { error:"...", ice_servers:[...] }        (4xx when peer unreachable —
//                                                 the client should fall back
//                                                 to the /ws/client relay)
//
// Agent JSON protocol additions (text frames):
//   server → agent:  {"kind":"signal", "req_id":N, "offer":..., "ice_servers":[...]}
//   agent  → server: {"kind":"signal_answer", "req_id":N, "answer":...}
//                    {"kind":"signal_error",  "req_id":N, "message":"..."}

import (
	"context"
	"encoding/json"
	"net/http"
	"sync"
	"sync/atomic"
	"time"
)

// Default ICE servers advertised to both ends.  Public STUN; users can add
// a TURN relay later via env/config.
func defaultICEServers() []map[string]any {
	return []map[string]any{
		{"urls": "stun:stun.l.google.com:19302"},
		{"urls": "stun:stun1.l.google.com:19302"},
	}
}

// Outstanding signal requests keyed by req_id.
type signalWaiter struct {
	answer chan json.RawMessage
	errMsg chan string
	once   sync.Once
}

func (w *signalWaiter) close() {
	w.once.Do(func() { close(w.answer); close(w.errMsg) })
}

var (
	signalMu      sync.Mutex
	signalWaiters = map[uint32]*signalWaiter{}
	signalReqCtr  uint32
)

func registerSignalWaiter() (uint32, *signalWaiter) {
	id := atomic.AddUint32(&signalReqCtr, 1)
	w := &signalWaiter{
		answer: make(chan json.RawMessage, 1),
		errMsg: make(chan string, 1),
	}
	signalMu.Lock()
	signalWaiters[id] = w
	signalMu.Unlock()
	return id, w
}

func takeSignalWaiter(id uint32) (*signalWaiter, bool) {
	signalMu.Lock()
	defer signalMu.Unlock()
	w, ok := signalWaiters[id]
	if ok {
		delete(signalWaiters, id)
	}
	return w, ok
}

// deliverSignalAnswer is called from the agent reader when a signal_answer or
// signal_error text frame arrives.  Returns true if the frame was consumed.
func deliverSignalAnswer(msg map[string]any) bool {
	kind, _ := msg["kind"].(string)
	if kind != "signal_answer" && kind != "signal_error" {
		return false
	}
	idf, ok := msg["req_id"].(float64)
	if !ok {
		return false
	}
	w, ok := takeSignalWaiter(uint32(idf))
	if !ok {
		return false
	}
	if kind == "signal_answer" {
		raw, _ := json.Marshal(msg["answer"])
		select {
		case w.answer <- raw:
		default:
		}
	} else {
		m, _ := msg["message"].(string)
		select {
		case w.errMsg <- m:
		default:
		}
	}
	return true
}

// ─── POST /api/signal ──────────────────────────────────────────────────────

type signalReq struct {
	PoolID int64           `json:"pool_id"`
	Offer  json.RawMessage `json:"offer"`
}

func (s *server) handleSignal(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	var body signalReq
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeErr(w, 400, "bad json")
		return
	}
	if body.PoolID == 0 || len(body.Offer) == 0 {
		writeErr(w, 400, "pool_id and offer required")
		return
	}
	if _, isMember := s.userIsMember(body.PoolID, u.ID); !isMember {
		writeErr(w, 403, "not a pool member")
		return
	}

	ice := defaultICEServers()

	ac, ok := s.pickOnlineRigInPool(body.PoolID)
	if !ok {
		writeJSON(w, 503, map[string]any{
			"error":       "no online rig in pool",
			"ice_servers": ice,
		})
		return
	}

	id, waiter := registerSignalWaiter()
	defer func() {
		if _, stillThere := takeSignalWaiter(id); stillThere {
			waiter.close()
		}
	}()

	ac.send(map[string]any{
		"kind":        "signal",
		"req_id":      id,
		"offer":       body.Offer,
		"ice_servers": ice,
	})

	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	select {
	case ans := <-waiter.answer:
		writeJSON(w, 200, map[string]any{
			"answer":      json.RawMessage(ans),
			"ice_servers": ice,
		})
	case msg := <-waiter.errMsg:
		writeJSON(w, 502, map[string]any{
			"error":       "agent: " + msg,
			"ice_servers": ice,
		})
	case <-ctx.Done():
		writeJSON(w, 504, map[string]any{
			"error":       "agent signalling timed out",
			"ice_servers": ice,
		})
	}
}
