package main

// Pipeline routing skeleton.
//
// POST /api/infer_pp
//   Same shape as /api/infer, but the coordinator:
//     1. Runs the planner to build a PP stage chain.
//     2. Sends each stage a `pp_route` JSON control message so the node knows
//        its layer range, tp_size, who to forward activations to, and how to
//        reach the requester at the terminal stage.
//     3. Kicks off stage 0 with a prompt ACTV frame.
//     4. Streams token frames from the terminal stage back to the browser via
//        SSE (same format as /api/infer: {"token":"..."}, {"done":true}).
//
// In this milestone there is no real llama_decode yet; the agents just pass
// activations through and the terminal stage synthesises tokens.  Plumbing
// only — the math goes in M5.

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"sync"
	"time"
)

// ppPeer is registered on every stage's agentConn while a request is in
// flight.  It catches ACTV frames from that stage and either forwards them to
// the next stage (intermediate) or drains tokens to the SSE writer (terminal).
type ppPeer struct {
	reqID      uint16
	stageIdx   uint16
	isTerminal bool

	// nextAgent is the agentConn for stage (stageIdx+1); nil for terminal.
	nextAgent *agentConn

	// For the terminal stage: tokens flow to the coordinator goroutine via
	// this channel.  Closed by the sender on done/error.
	tokens chan *ActvFrame
	closed chan struct{}
	once   sync.Once
}

func (p *ppPeer) close() {
	p.once.Do(func() {
		close(p.closed)
		if p.tokens != nil {
			close(p.tokens)
		}
	})
}

// dispatchActvFromAgent handles ACTV frames on an agent's WS.  Called from
// dispatchBinaryFromAgent (which tries INFR first).  Returns true if consumed.
func (ac *agentConn) dispatchActvFromAgent(data []byte) bool {
	ac.peerMu.RLock()
	pp := ac.ppPeer
	ac.peerMu.RUnlock()
	if pp == nil {
		return false
	}
	frame, err := DecodeActvFrame(data)
	if err != nil {
		log.Printf("actv decode: %v", err)
		return true
	}
	if frame == nil {
		return false
	}
	if frame.ReqID != pp.reqID {
		return true // stale / from a different request
	}
	switch frame.Type {
	case actvTypeAct:
		// Intermediate: re-label the stage and forward to next node.
		if pp.nextAgent == nil {
			// Shouldn't happen for a terminal stage — activations end here.
			log.Printf("pp: terminal stage got activation, dropping")
			return true
		}
		// Bump the stage idx so the receiver knows who produced it.
		fwd := &ActvFrame{
			Type:    frame.Type,
			ReqID:   frame.ReqID,
			Stage:   frame.Stage + 1,
			TokSeq:  frame.TokSeq,
			DType:   frame.DType,
			Flags:   frame.Flags,
			Dims:    frame.Dims,
			Payload: frame.Payload,
		}
		if !pp.nextAgent.sendBin(fwd.Encode()) {
			log.Printf("pp: next stage buffer full, dropping")
		}
	case actvTypeToken, actvTypeDone, actvTypeError:
		if !pp.isTerminal {
			// Non-terminal shouldn't be emitting tokens; but if it does,
			// still propagate it upstream so the coordinator can react.
			log.Printf("pp: non-terminal stage %d emitted type=%d", pp.stageIdx, frame.Type)
		}
		select {
		case pp.tokens <- frame:
		default:
			log.Printf("pp: token channel full, dropping")
		}
		if frame.Type == actvTypeDone || frame.Type == actvTypeError {
			pp.close()
		}
	}
	return true
}

// attachPPPeer installs a ppPeer on an agentConn.  Returns false if the agent
// is already busy (either raw relay, inference, or another PP request).
func (ac *agentConn) attachPPPeer(pp *ppPeer) bool {
	ac.peerMu.Lock()
	defer ac.peerMu.Unlock()
	if ac.peer != nil || ac.inferPeer != nil || ac.ppPeer != nil {
		return false
	}
	ac.ppPeer = pp
	return true
}

func (ac *agentConn) detachPPPeer(pp *ppPeer) {
	ac.peerMu.Lock()
	defer ac.peerMu.Unlock()
	if ac.ppPeer == pp {
		ac.ppPeer = nil
	}
}

// ─── Handler ───────────────────────────────────────────────────────────────

func (s *server) handleInferPP(w http.ResponseWriter, r *http.Request) {
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

	vis, _, ok := s.poolVisibility(body.PoolID)
	if !ok {
		writeErr(w, 404, "pool not found")
		return
	}
	if _, member := s.userIsMember(body.PoolID, u.ID); !member && vis != "public" {
		writeErr(w, 403, "not a member of pool")
		return
	}

	okRL, policy, snap := s.reserveRequestSlot(u.ID)
	if !okRL {
		s.logInference(u.ID, body.PoolID, 0, "", 0, 0, "rate_limit")
		writeJSON(w, 429, map[string]any{"error": "rate limit", "policy": policy, "usage": snap})
		return
	}

	reqID16 := nextReqID()
	plan, err := s.planPipeline(body.PoolID, uint32(reqID16), body.Model, body.NLayersOverride)
	if err != nil {
		s.logInference(u.ID, body.PoolID, 0, "", 0, 0, "no_rig")
		writeErr(w, 503, err.Error())
		return
	}

	// Resolve each stage to its live agentConn.
	agents := make([]*agentConn, len(plan.Stages))
	for i, st := range plan.Stages {
		ac, ok := s.hub.findAgent(st.UserID, st.AgentID)
		if !ok {
			writeErr(w, 503, "stage "+st.AgentID+" went offline")
			return
		}
		agents[i] = ac
	}

	// Attach a ppPeer to every stage.  If any stage is busy, abort and detach
	// what we've already installed.
	peers := make([]*ppPeer, len(plan.Stages))
	var attached []*agentConn
	cleanup := func() {
		for i, ac := range attached {
			ac.detachPPPeer(peers[i])
		}
	}
	for i := range plan.Stages {
		var next *agentConn
		if i+1 < len(plan.Stages) {
			next = agents[i+1]
		}
		pp := &ppPeer{
			reqID:      reqID16,
			stageIdx:   uint16(i),
			isTerminal: i == len(plan.Stages)-1,
			nextAgent:  next,
			closed:     make(chan struct{}),
		}
		if pp.isTerminal {
			pp.tokens = make(chan *ActvFrame, 64)
		}
		if !agents[i].attachPPPeer(pp) {
			cleanup()
			writeErr(w, 503, "stage "+plan.Stages[i].AgentID+" is busy")
			return
		}
		peers[i] = pp
		attached = append(attached, agents[i])
	}
	defer cleanup()

	// Stage 0 for token loopback: after terminal emits a token, re-kick stage
	// 0 with that token as a kv_append ACTV so it can extend its KV cache and
	// keep the sequence rolling.
	stage0 := agents[0]

	// Tell each stage its role.  The last stage doesn't need a "next".
	for i, st := range plan.Stages {
		msg := map[string]any{
			"kind":        "pp_route",
			"req_id":      reqID16,
			"stage_idx":   st.StageIdx,
			"stage_count": len(plan.Stages),
			"layer_lo":    st.LayerLo,
			"layer_hi":    st.LayerHi,
			"tp_size":     st.TPSize,
			"model":       plan.ModelName,
			"shard_url":   st.ShardURL,
			"shard_file":  st.ShardFile,
			"max_tokens":  body.MaxTokens,
			"is_first":    i == 0,
			"is_last":     i == len(plan.Stages)-1,
		}
		agents[i].send(msg)
	}

	// SSE response to the browser.
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	flusher, _ := w.(http.Flusher)

	logID := s.logInference(u.ID, body.PoolID, plan.Stages[0].UserID, plan.Stages[0].AgentID, 0, 0, "running")

	// Kick off stage 0 with the prompt as a byte-payload ACTV frame.
	// In M5 this will carry embedded token IDs instead of utf-8 bytes.
	kickoff := &ActvFrame{
		Type:    actvTypeAct,
		ReqID:   reqID16,
		Stage:   0,
		TokSeq:  0,
		DType:   actvDTypeBytes,
		Flags:   actvFlagIsPrompt | actvFlagEndOfPrompt,
		Payload: []byte(body.Prompt),
	}
	if !agents[0].sendBin(kickoff.Encode()) {
		s.finishInference(logID, 0, 0, "failed")
		writeSSEErr(w, flusher, "stage 0 buffer full")
		return
	}

	// Drain tokens from the terminal stage's ppPeer.
	termPeer := peers[len(peers)-1]
	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Minute)
	defer cancel()

	var inTok, outTok uint32
	for {
		select {
		case f, ok := <-termPeer.tokens:
			if !ok {
				writeSSEEvent(w, flusher, map[string]any{"done": true})
				s.finishInference(logID, int(inTok), int(outTok), "ok")
				s.recordTokens(u.ID, int(inTok), int(outTok))
				return
			}
			switch f.Type {
			case actvTypeToken:
				outTok = f.TokSeq
				writeSSEEvent(w, flusher, map[string]any{"token": string(f.Payload)})
				// Loopback: re-kick stage 0 with this token as a kv_append
				// activation so it can extend its KV cache.  The node runs one
				// decode step and fires the next round of activations down
				// the pipeline.  Terminal stage decides when to stop (EOS or
				// max_tokens), emitting actvTypeDone.
				if len(plan.Stages) > 0 && int(outTok) < body.MaxTokens {
					cont := &ActvFrame{
						Type:    actvTypeAct,
						ReqID:   reqID16,
						Stage:   0,
						TokSeq:  outTok,
						DType:   actvDTypeBytes,
						Flags:   actvFlagKVAppend,
						Payload: f.Payload,
					}
					if !stage0.sendBin(cont.Encode()) {
						log.Printf("pp: stage 0 loopback buffer full")
					}
				}
			case actvTypeDone:
				writeSSEEvent(w, flusher, map[string]any{"done": true})
				s.finishInference(logID, int(inTok), int(outTok), "ok")
				s.recordTokens(u.ID, int(inTok), int(outTok))
				return
			case actvTypeError:
				writeSSEEvent(w, flusher, map[string]any{"error": string(f.Payload)})
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
