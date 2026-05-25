package main

// Live per-stage progress for diffusion pipeline-parallel (CF12-F).
//
// Agents emit `dpp_progress` WS frames as they enter stages / advance
// denoise steps.  The server normalises, tags with the (user_id,
// agent_id) provenance, and fans out to all SSE subscribers for that
// user via /api/dpp/stream.
//
// The shape on the wire (agent -> server):
//
//   {"kind":"dpp_progress",
//    "req_id":  uint16,
//    "stage_idx": int,
//    "role":     string,           // "text_encoder"|"unet"|"vae"
//    "block_lo": int,              // -1 if N/A
//    "block_hi": int,              // -1 if N/A
//    "step_idx": int,              // 0..total_steps-1, -1 if N/A
//    "total_steps": int,           // 0 if N/A
//    "event":    string,           // "enter"|"step"|"exit"|"error"
//    "msg":      string}           // optional human-readable
//
// The SSE event re-encoded by the server adds:
//
//   {"agent_id": ..., "ts": unix_ms, ...above fields...}
//
// SSE retention: the broker keeps no history — late subscribers see
// only future events.  That matches /api/me/rigs/stream's behaviour.

import (
	"encoding/json"
	"net/http"
	"sync"
	"time"
)

type dppProgressEvent struct {
	AgentID    string `json:"agent_id"`
	TimestampMs int64 `json:"ts"`
	ReqID      uint16 `json:"req_id"`
	StageIdx   int    `json:"stage_idx"`
	Role       string `json:"role"`
	BlockLo    int    `json:"block_lo"`
	BlockHi    int    `json:"block_hi"`
	StepIdx    int    `json:"step_idx"`
	TotalSteps int    `json:"total_steps"`
	Event      string `json:"event"`
	Msg        string `json:"msg,omitempty"`
}

type dppProgressBroker struct {
	mu   sync.RWMutex
	subs map[int64]map[chan dppProgressEvent]struct{}
}

func newDPPProgressBroker() *dppProgressBroker {
	return &dppProgressBroker{
		subs: map[int64]map[chan dppProgressEvent]struct{}{},
	}
}

// Subscribe returns a buffered channel that receives all DPP events
// for the user.  Caller MUST call Unsubscribe with the same channel
// when done or the broker leaks.  Buffer size 32 — a stalled SSE
// connection can fall up to 32 events behind before we drop.
func (b *dppProgressBroker) Subscribe(userID int64) chan dppProgressEvent {
	ch := make(chan dppProgressEvent, 32)
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.subs[userID] == nil {
		b.subs[userID] = map[chan dppProgressEvent]struct{}{}
	}
	b.subs[userID][ch] = struct{}{}
	return ch
}

func (b *dppProgressBroker) Unsubscribe(userID int64, ch chan dppProgressEvent) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if set := b.subs[userID]; set != nil {
		delete(set, ch)
		if len(set) == 0 {
			delete(b.subs, userID)
		}
	}
	// Drain so any in-flight publish completes without blocking the
	// publisher on a closed channel.
	go func() {
		for range ch {
		}
	}()
	close(ch)
}

// Publish fans the event out to all subscribers of `userID`.  Slow
// subscribers are dropped from this event only (channel is buffered;
// non-blocking send).
func (b *dppProgressBroker) Publish(userID int64, ev dppProgressEvent) {
	b.mu.RLock()
	defer b.mu.RUnlock()
	for ch := range b.subs[userID] {
		select {
		case ch <- ev:
		default:
		}
	}
}

// ingestDPPProgress decodes a `dpp_progress` WS frame from an agent
// and republishes it on the broker.  Called from the WS reader.
func (s *server) ingestDPPProgress(userID int64, agentID string, msg map[string]any) {
	ev := dppProgressEvent{
		AgentID:    agentID,
		TimestampMs: time.Now().UnixMilli(),
		BlockLo:    -1,
		BlockHi:    -1,
		StepIdx:    -1,
	}
	if v, ok := msg["req_id"].(float64); ok {
		ev.ReqID = uint16(v)
	}
	if v, ok := msg["stage_idx"].(float64); ok {
		ev.StageIdx = int(v)
	}
	if v, ok := msg["role"].(string); ok {
		ev.Role = v
	}
	if v, ok := msg["block_lo"].(float64); ok {
		ev.BlockLo = int(v)
	}
	if v, ok := msg["block_hi"].(float64); ok {
		ev.BlockHi = int(v)
	}
	if v, ok := msg["step_idx"].(float64); ok {
		ev.StepIdx = int(v)
	}
	if v, ok := msg["total_steps"].(float64); ok {
		ev.TotalSteps = int(v)
	}
	if v, ok := msg["event"].(string); ok {
		ev.Event = v
	}
	if v, ok := msg["msg"].(string); ok {
		ev.Msg = v
	}
	s.dppProgress.Publish(userID, ev)
}

// handleDPPStream — GET /api/dpp/stream.  Per-user SSE channel.
func (s *server) handleDPPStream(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "auth required")
		return
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeErr(w, 500, "streaming unsupported")
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(200)
	flusher.Flush()

	ch := s.dppProgress.Subscribe(u.ID)
	defer s.dppProgress.Unsubscribe(u.ID, ch)

	// Heartbeat every 25s so proxies don't kill an idle stream.
	ticker := time.NewTicker(25 * time.Second)
	defer ticker.Stop()

	enc := json.NewEncoder(w)
	for {
		select {
		case <-r.Context().Done():
			return
		case ev, ok := <-ch:
			if !ok {
				return
			}
			if _, err := w.Write([]byte("data: ")); err != nil {
				return
			}
			if err := enc.Encode(ev); err != nil {
				return
			}
			flusher.Flush()
		case <-ticker.C:
			if _, err := w.Write([]byte(": ping\n\n")); err != nil {
				return
			}
			flusher.Flush()
		}
	}
}
