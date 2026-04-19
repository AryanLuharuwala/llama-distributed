package main

// Inference gateway.
//
// POST /api/infer  { pool_id, model, prompt, max_tokens?, temperature? }
//   → text/event-stream; events: {"token":"..."} ... {"done":true}
//
// Under the hood:
//   1. Rate-limit check.
//   2. Pick an online rig in the target pool (must be a member).
//   3. Bind this HTTP call to that agent via the same peer mechanism /ws/client
//      uses.  The server sends one inferenceRequest binary frame and then
//      drains agent binary responses until a terminator arrives.
//
// Wire format of the frames we put on the WS binary channel:
//
//   InferenceRequest:
//     magic  u32  "INFR"        // 0x494E4652 (big-endian)
//     ver    u8   0x01
//     type   u8   0x01
//     req_id u16  (client-chosen)
//     input_tokens u32 (best-effort; used for rate accounting)
//     payload_len  u32
//     payload      UTF-8 JSON { "model":..., "prompt":..., "max_tokens":N,
//                               "temperature":T }
//
//   InferenceChunk  (agent → server):
//     magic  "INFR", ver 0x01, type 0x02, req_id (echoed)
//     kind   u8   0 = token, 1 = done, 2 = error
//     tok_in   u32
//     tok_out  u32
//     payload_len u32
//     payload     UTF-8 (for kind=token: the delta text;
//                         for kind=error: the message;
//                         for kind=done: empty).

import (
	"context"
	"database/sql"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
	"sync/atomic"
	"time"
)

const (
	infMagic  = 0x494E4652 // "INFR"
	infVer    = 0x01
	typeReq   = 0x01
	typeChunk = 0x02

	chunkKindToken = 0
	chunkKindDone  = 1
	chunkKindError = 2
)

// ─── Frame encoding ────────────────────────────────────────────────────────

func encodeInferRequest(reqID uint16, inputTokens uint32, jsonPayload []byte) []byte {
	buf := make([]byte, 0, 16+len(jsonPayload))
	var hdr [16]byte
	binary.BigEndian.PutUint32(hdr[0:4], infMagic)
	hdr[4] = infVer
	hdr[5] = typeReq
	binary.BigEndian.PutUint16(hdr[6:8], reqID)
	binary.BigEndian.PutUint32(hdr[8:12], inputTokens)
	binary.BigEndian.PutUint32(hdr[12:16], uint32(len(jsonPayload)))
	buf = append(buf, hdr[:]...)
	buf = append(buf, jsonPayload...)
	return buf
}

type inferChunk struct {
	reqID   uint16
	kind    uint8
	tokIn   uint32
	tokOut  uint32
	payload []byte
}

// Chunk frame layout (agent → server, 24-byte header):
//
//   offset  field        size
//    0      magic        4    ("INFR")
//    4      ver          1
//    5      type         1    (0x02 = chunk)
//    6      req_id       2
//    8      kind         1    (0=token 1=done 2=error)
//    9..11  reserved     3
//   12      tok_in       4
//   16      tok_out      4
//   20      payload_len  4
//   24      payload      N

func encodeChunk(reqID uint16, kind uint8, tokIn, tokOut uint32, payload []byte) []byte {
	buf := make([]byte, 0, 24+len(payload))
	var hdr [24]byte
	binary.BigEndian.PutUint32(hdr[0:4], infMagic)
	hdr[4] = infVer
	hdr[5] = typeChunk
	binary.BigEndian.PutUint16(hdr[6:8], reqID)
	hdr[8] = kind
	// 9..11 reserved
	binary.BigEndian.PutUint32(hdr[12:16], tokIn)
	binary.BigEndian.PutUint32(hdr[16:20], tokOut)
	binary.BigEndian.PutUint32(hdr[20:24], uint32(len(payload)))
	buf = append(buf, hdr[:]...)
	buf = append(buf, payload...)
	return buf
}

func decodeChunk(b []byte) (*inferChunk, error) {
	if len(b) < 24 {
		return nil, fmt.Errorf("chunk too short: %d bytes", len(b))
	}
	if binary.BigEndian.Uint32(b[0:4]) != infMagic {
		return nil, fmt.Errorf("bad magic")
	}
	if b[4] != infVer {
		return nil, fmt.Errorf("bad version: %d", b[4])
	}
	if b[5] != typeChunk {
		return nil, fmt.Errorf("not a chunk: type=%d", b[5])
	}
	plen := binary.BigEndian.Uint32(b[20:24])
	if uint32(len(b)) < 24+plen {
		return nil, fmt.Errorf("truncated payload: got %d want %d", len(b)-24, plen)
	}
	return &inferChunk{
		reqID:   binary.BigEndian.Uint16(b[6:8]),
		kind:    b[8],
		tokIn:   binary.BigEndian.Uint32(b[12:16]),
		tokOut:  binary.BigEndian.Uint32(b[16:20]),
		payload: append([]byte(nil), b[24:24+plen]...),
	}, nil
}

// ─── Handler ───────────────────────────────────────────────────────────────

type inferRequestBody struct {
	PoolID      int64   `json:"pool_id"`
	Model       string  `json:"model,omitempty"`
	Prompt      string  `json:"prompt"`
	MaxTokens   int     `json:"max_tokens,omitempty"`
	Temperature float64 `json:"temperature,omitempty"`
	// NLayersOverride lets a caller drive the planner without a bound model
	// (useful for plumbing tests; ignored once a real model is attached).
	NLayersOverride int `json:"n_layers,omitempty"`
}

// Rough token estimator.  Real tokenisation happens on the agent; this is
// only for the rate limiter's pre-flight accounting.
func estimateTokens(s string) int {
	if s == "" {
		return 0
	}
	// Very crude: ~4 chars/token.
	return (len(s) + 3) / 4
}

var reqIDCtr uint32

func nextReqID() uint16 {
	return uint16(atomic.AddUint32(&reqIDCtr, 1) & 0xFFFF)
}

func (s *server) handleInfer(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	var body inferRequestBody
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeErr(w, 400, "bad json")
		return
	}
	if body.PoolID == 0 || body.Prompt == "" {
		writeErr(w, 400, "pool_id and prompt required")
		return
	}
	if body.MaxTokens == 0 {
		body.MaxTokens = 128
	}

	// Pool membership / visibility.
	vis, _, ok := s.poolVisibility(body.PoolID)
	if !ok {
		writeErr(w, 404, "pool not found")
		return
	}
	if _, member := s.userIsMember(body.PoolID, u.ID); !member && vis != "public" {
		writeErr(w, 403, "not a member of pool")
		return
	}

	// Rate limit pre-flight.
	ok, policy, snap := s.reserveRequestSlot(u.ID)
	if !ok {
		s.logInference(u.ID, body.PoolID, 0, "", 0, 0, "rate_limit")
		writeJSON(w, 429, map[string]any{
			"error":  "rate limit",
			"policy": policy,
			"usage":  snap,
		})
		return
	}

	// Pick an online rig in the pool.
	ac, ok := s.pickOnlineRigInPool(body.PoolID)
	if !ok {
		s.logInference(u.ID, body.PoolID, 0, "", 0, 0, "no_rig")
		writeErr(w, 503, "no online rigs in pool")
		return
	}

	// SSE response.
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	flusher, _ := w.(http.Flusher)

	// Attach a streaming peer to the agent, refusing if busy.
	ip := &inferPeer{
		reqID:  nextReqID(),
		incoming: make(chan *inferChunk, 32),
		closed:   make(chan struct{}),
	}
	ac.peerMu.Lock()
	if ac.peer != nil || ac.inferPeer != nil {
		ac.peerMu.Unlock()
		s.logInference(u.ID, body.PoolID, ac.userID, ac.agentID, 0, 0, "failed")
		writeErr(w, 503, "agent busy")
		return
	}
	ac.inferPeer = ip
	ac.peerMu.Unlock()

	defer func() {
		ac.peerMu.Lock()
		if ac.inferPeer == ip {
			ac.inferPeer = nil
		}
		ac.peerMu.Unlock()
	}()

	// Announce start in the server log.
	logID := s.logInference(u.ID, body.PoolID, ac.userID, ac.agentID, 0, 0, "running")

	// Build + send request frame.
	payloadJSON, _ := json.Marshal(map[string]any{
		"model":       body.Model,
		"prompt":      body.Prompt,
		"max_tokens":  body.MaxTokens,
		"temperature": body.Temperature,
	})
	reqFrame := encodeInferRequest(ip.reqID, uint32(estimateTokens(body.Prompt)), payloadJSON)
	if !ac.sendBin(reqFrame) {
		s.finishInference(logID, 0, 0, "failed")
		writeSSEErr(w, flusher, "agent buffer full")
		return
	}

	// Drain.  Budget: max_tokens + a slack factor, timeout 2 min overall.
	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Minute)
	defer cancel()

	var inTok, outTok uint32
	for {
		select {
		case c, ok := <-ip.incoming:
			if !ok {
				writeSSEEvent(w, flusher, map[string]any{"done": true})
				s.finishInference(logID, int(inTok), int(outTok), "ok")
				s.recordTokens(u.ID, int(inTok), int(outTok))
				return
			}
			inTok, outTok = c.tokIn, c.tokOut
			switch c.kind {
			case chunkKindToken:
				writeSSEEvent(w, flusher, map[string]any{"token": string(c.payload)})
			case chunkKindDone:
				writeSSEEvent(w, flusher, map[string]any{"done": true})
				s.finishInference(logID, int(inTok), int(outTok), "ok")
				s.recordTokens(u.ID, int(inTok), int(outTok))
				return
			case chunkKindError:
				writeSSEEvent(w, flusher, map[string]any{"error": string(c.payload)})
				s.finishInference(logID, int(inTok), int(outTok), "failed")
				return
			}
		case <-ctx.Done():
			writeSSEErr(w, flusher, "timeout")
			s.finishInference(logID, int(inTok), int(outTok), "failed")
			return
		}
	}
}

func writeSSEEvent(w http.ResponseWriter, fl http.Flusher, v any) {
	b, _ := json.Marshal(v)
	fmt.Fprintf(w, "data: %s\n\n", b)
	if fl != nil {
		fl.Flush()
	}
}

func writeSSEErr(w http.ResponseWriter, fl http.Flusher, msg string) {
	writeSSEEvent(w, fl, map[string]any{"error": msg})
}

// ─── Inference logging ─────────────────────────────────────────────────────

func (s *server) logInference(userID, poolID, agentUserID int64, agentID string,
	inTok, outTok int, status string) int64 {
	res, err := s.db.Exec(
		`INSERT INTO inference_log (user_id, pool_id, agent_user_id, agent_id,
		   input_tokens, output_tokens, started_at, status)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
		userID, poolID, agentUserID, agentID, inTok, outTok, nowUnix(), status,
	)
	if err != nil {
		log.Printf("inference_log insert: %v", err)
		return 0
	}
	id, _ := res.LastInsertId()
	return id
}

func (s *server) finishInference(id int64, inTok, outTok int, status string) {
	if id == 0 {
		return
	}
	_, _ = s.db.Exec(
		`UPDATE inference_log SET finished_at = ?, input_tokens = ?, output_tokens = ?, status = ?
		 WHERE id = ?`,
		nowUnix(), inTok, outTok, status, id,
	)
}

// ─── inferPeer plumbing on agentConn ───────────────────────────────────────
//
// We extend agentConn to carry an optional inference-stream peer (in addition
// to the optional raw relay peer used by /ws/client).  The agent's reader
// dispatches binary frames to whichever peer is present.

type inferPeer struct {
	reqID    uint16
	incoming chan *inferChunk
	closed   chan struct{}
	once     sync.Once
}

func (p *inferPeer) close() {
	p.once.Do(func() {
		close(p.closed)
		close(p.incoming)
	})
}

// dispatchBinaryFromAgent is called by the agent's WS reader on every binary
// frame.  Try INFR (single-agent inference) first, then ACTV (pipeline-stage
// activation), then fall through to the raw relay peer.  Returns true if the
// frame was consumed.
func (ac *agentConn) dispatchBinaryFromAgent(data []byte) bool {
	ac.peerMu.RLock()
	ip := ac.inferPeer
	rp := ac.peer
	ac.peerMu.RUnlock()

	if ip != nil && len(data) >= 4 && binary.BigEndian.Uint32(data[0:4]) == infMagic {
		c, err := decodeChunk(data)
		if err != nil {
			log.Printf("decode chunk: %v", err)
			return true
		}
		if c.reqID != ip.reqID {
			return true // stale
		}
		select {
		case ip.incoming <- c:
		default:
			log.Printf("inferPeer incoming full, dropping chunk")
		}
		if c.kind == chunkKindDone || c.kind == chunkKindError {
			ip.close()
		}
		return true
	}
	// Pipeline-parallel activation frame?
	if len(data) >= 4 && binary.BigEndian.Uint32(data[0:4]) == actvMagic {
		if ac.dispatchActvFromAgent(data) {
			return true
		}
	}
	if rp != nil {
		buf := make([]byte, len(data))
		copy(buf, data)
		if !rp.sendBin(buf) {
			log.Printf("relay: client buffer full, dropping %d bytes", len(buf))
		}
		return true
	}
	return false
}

// ─── GET /api/inference_log — caller's recent requests ────────────────────

func (s *server) handleInferenceLog(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	rows, err := s.db.Query(
		`SELECT id, pool_id, agent_id, input_tokens, output_tokens,
		        started_at, finished_at, status
		 FROM inference_log WHERE user_id = ?
		 ORDER BY started_at DESC LIMIT 50`, u.ID)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	defer rows.Close()

	type row struct {
		ID           int64   `json:"id"`
		PoolID       int64   `json:"pool_id"`
		AgentID      string  `json:"agent_id"`
		InputTokens  int     `json:"input_tokens"`
		OutputTokens int     `json:"output_tokens"`
		StartedAt    int64   `json:"started_at"`
		FinishedAt   *int64  `json:"finished_at,omitempty"`
		Status       string  `json:"status"`
	}
	var out []row
	for rows.Next() {
		var rw row
		var fin sql.NullInt64
		if err := rows.Scan(&rw.ID, &rw.PoolID, &rw.AgentID,
			&rw.InputTokens, &rw.OutputTokens,
			&rw.StartedAt, &fin, &rw.Status); err != nil {
			continue
		}
		if fin.Valid {
			rw.FinishedAt = &fin.Int64
		}
		out = append(out, rw)
	}
	writeJSON(w, 200, map[string]any{"log": out})
}

