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
	"io"
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
	frame, err := DecodeActvFrame(data)
	if err != nil {
		log.Printf("actv decode: %v", err)
		return true
	}
	if frame == nil {
		return false
	}
	// Hold peerMu.RLock across the token send so ac.close() (which
	// takes Lock and then closes pp.tokens) can't race with our send.
	ac.peerMu.RLock()
	pp := ac.ppPeer
	if pp == nil {
		ac.peerMu.RUnlock()
		return false
	}
	if frame.ReqID != pp.reqID {
		ac.peerMu.RUnlock()
		return true
	}
	var fwd *ActvFrame
	var doClose bool
	switch frame.Type {
	case actvTypeAct:
		if pp.nextAgent == nil {
			log.Printf("pp: terminal stage got activation, dropping")
			ac.peerMu.RUnlock()
			return true
		}
		// Forward to next stage; sendBin is safe to call under our
		// own peerMu.RLock (different agent's lock).
		fwd = &ActvFrame{
			Type:    frame.Type,
			ReqID:   frame.ReqID,
			Stage:   frame.Stage + 1,
			TokSeq:  frame.TokSeq,
			DType:   frame.DType,
			Flags:   frame.Flags,
			Dims:    frame.Dims,
			Payload: frame.Payload,
		}
	case actvTypeToken, actvTypeDone, actvTypeError:
		if !pp.isTerminal {
			log.Printf("pp: non-terminal stage %d emitted type=%d", pp.stageIdx, frame.Type)
		}
		// Probe pp.closed first so a peer already torn down by ac.close()
		// fan-out doesn't see a redundant send.
		select {
		case <-pp.closed:
			ac.peerMu.RUnlock()
			return true
		default:
		}
		select {
		case pp.tokens <- frame:
		default:
			log.Printf("pp: token channel full, dropping")
		}
		if frame.Type == actvTypeDone || frame.Type == actvTypeError {
			doClose = true
		}
	}
	ac.peerMu.RUnlock()
	if fwd != nil {
		if !pp.nextAgent.sendBin(fwd.Encode()) {
			log.Printf("pp: next stage buffer full, dropping")
		}
	}
	if doClose {
		pp.close()
	}
	return true
}

// attachPPPeer installs a ppPeer on an agentConn.  Returns false if the agent
// is already busy (either raw relay, inference, or another PP request).
func (ac *agentConn) attachPPPeer(pp *ppPeer) bool {
	ac.peerMu.Lock()
	defer ac.peerMu.Unlock()
	if ac.peer != nil || len(ac.inferPeers) > 0 || ac.ppPeer != nil {
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
	if err := json.NewDecoder(io.LimitReader(r.Body, 64<<10)).Decode(&body); err != nil {
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
	// Unconditional slot-refund guard — see inference.go for rationale.
	slotBilled := false
	defer func() {
		if !slotBilled {
			s.refundRequestSlot(u.ID)
		}
	}()

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

	// P2P signaling permissions: authorise each adjacent stage pair to
	// signal each other for the duration of the request, then tear down on
	// cleanup.  The session id ties any straggling ICE candidates to this
	// request so log lines correlate.
	sessionID := fmtSessionID("pp", reqID16)

	// Relay insertion: when an adjacent pair has one or both endpoints
	// behind a symmetric NAT, direct P2P will fail and the WS fallback
	// adds latency.  If a relay-capable peer (cone/open NAT) is in the
	// swarm, slot it between the two stages so frames hop A → B → C
	// over a pair of P2P data channels instead of the server.
	//
	// nextPeerOverride[i] is the agentID the planner will hand to stage i
	// as `next_peer_id` (defaults to plan.Stages[i+1]).  prevPeerOverride
	// is the same for the other direction.  When a relay is inserted
	// between i and i+1 both overrides for that pair point at the relay.
	nextPeerOverride := make([]string, len(plan.Stages))
	prevPeerOverride := make([]string, len(plan.Stages))
	type relaySlot struct {
		left  string
		right string
		agent *agentConn
	}
	var relays []relaySlot

	// Build an exclusion set so we don't pick a stage itself as a relay.
	excludeAsRelay := make(map[string]struct{}, len(plan.Stages))
	for _, st := range plan.Stages {
		excludeAsRelay[st.AgentID] = struct{}{}
	}

	for i := 0; i+1 < len(plan.Stages); i++ {
		left := plan.Stages[i].AgentID
		right := plan.Stages[i+1].AgentID
		nextPeerOverride[i] = right
		prevPeerOverride[i+1] = left

		// Decide whether this pair needs a relay.  We look at the live
		// status snapshot for each agent — if NAT type was reported as
		// symmetric or blocked we know direct P2P is unlikely.  Empty
		// or "unknown" NAT types are treated as "probably fine" so we
		// don't waste a relay slot speculatively.
		needs := func(ac *agentConn) bool {
			st := ac.snapshotStatus()
			return st.NATType == "symmetric" || st.NATType == "blocked"
		}
		if !needs(agents[i]) && !needs(agents[i+1]) {
			continue
		}
		relay := s.findRelayAgent(excludeAsRelay)
		if relay == nil {
			continue
		}
		// Make sure we don't pick the same rig as a relay for two pairs
		// in the same request (it'd have to manage too many forwards).
		excludeAsRelay[relay.agentID] = struct{}{}

		nextPeerOverride[i] = relay.agentID
		prevPeerOverride[i+1] = relay.agentID
		relays = append(relays, relaySlot{left: left, right: right, agent: relay})

		// Permission to signal both legs of the relay.
		allowP2PPair(left, relay.agentID, sessionID, 0)
		allowP2PPair(relay.agentID, right, sessionID, 0)

		// Tell the relay it has a job.  ICE servers reuse the public
		// STUN list — we explicitly don't recurse and offer the relay
		// another peer-relay (would loop forever).
		relayIce := defaultSTUNServers()
		if s.cfg.turnURL != "" {
			if user, pass := s.mintTURNCreds(relay.agentID); user != "" {
				relayIce = append(relayIce, map[string]any{
					"urls":       s.cfg.turnURL,
					"username":   user,
					"credential": pass,
				})
			}
		}
		relay.send(map[string]any{
			"kind":        "p2p_relay_assign",
			"session_id":  sessionID,
			"req_id":      reqID16,
			"left_peer":   left,
			"right_peer":  right,
			"ice_servers": relayIce,
		})
		// Track the assignment so unregisterAgent can mark it failed if
		// the relay rig disconnects mid-session, and so the release path
		// can credit byte counts to the right rig's reputation row.
		if s.relays != nil {
			s.relays.add(&relayAssignment{
				SessionID: sessionID,
				AgentID:   relay.agentID,
				LeftPeer:  left,
				RightPeer: right,
				StartedAt: nowUnix(),
			})
		}
	}

	// Permissions for direct stage↔stage signaling.  We allow them even
	// when a relay is in place — the planner doesn't force-pin the relay
	// path, so if direct P2P opportunistically works the rigs can use it.
	for i := 0; i+1 < len(plan.Stages); i++ {
		allowP2PPair(plan.Stages[i].AgentID, plan.Stages[i+1].AgentID, sessionID, 0)
	}
	prevCleanup := cleanup
	cleanup = func() {
		for i := 0; i+1 < len(plan.Stages); i++ {
			revokeP2PPair(plan.Stages[i].AgentID, plan.Stages[i+1].AgentID)
		}
		for _, r := range relays {
			revokeP2PPair(r.left, r.agent.agentID)
			revokeP2PPair(r.agent.agentID, r.right)
			r.agent.send(map[string]any{
				"kind":       "p2p_relay_release",
				"session_id": sessionID,
				"req_id":     reqID16,
			})
			// Credit successful relay session.  If the assignment is no
			// longer in the active map it means unregisterAgent already
			// drained it (relay rig disconnected) and counted it as a
			// failure — in that case there's nothing to credit here.
			if s.relays != nil {
				if a := s.relays.remove(sessionID, r.agent.agentID); a != nil {
					s.recordRelaySuccess(r.agent.agentID, a.BytesL2R+a.BytesR2L)
				}
			}
		}
		prevCleanup()
	}

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
			"session_id":  sessionID,
			"ice_servers": s.iceServersFor(st.AgentID),
		}
		// Inter-rig TP group context.  Only set when this stage belongs
		// to a multi-rig TP group (planner left TPGroupID=-1 otherwise).
		if st.TPGroupID >= 0 && len(st.TPPeers) > 0 {
			msg["tp_group_id"] = st.TPGroupID
			msg["tp_rank"] = st.TPRank
			msg["tp_peers"] = st.TPPeers
		}
		// The lower-indexed stage is the offerer (initiates the WebRTC
		// handshake); the higher-indexed stage answers.  Each stage only
		// needs to know who it should attempt P2P with — the other end
		// of the conversation.
		if i > 0 {
			if prevPeerOverride[i] != "" {
				msg["prev_peer_id"] = prevPeerOverride[i]
			} else {
				msg["prev_peer_id"] = plan.Stages[i-1].AgentID
			}
		}
		if i+1 < len(plan.Stages) {
			if nextPeerOverride[i] != "" {
				msg["next_peer_id"] = nextPeerOverride[i]
			} else {
				msg["next_peer_id"] = plan.Stages[i+1].AgentID
			}
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
	var bytesStreamed int
	// Attribute drift to the terminal stage since that's where the
	// tokens originate.  In the multi-stage case a malicious middle
	// node could only forge the count by colluding with the terminal —
	// detecting that requires per-stage attribution which we defer.
	termAgentID := plan.Stages[len(plan.Stages)-1].AgentID
	promptChars := len(body.Prompt)
	for {
		select {
		case f, ok := <-termPeer.tokens:
			if !ok {
				writeSSEEvent(w, flusher, map[string]any{"done": true})
				inSafe, outSafe := s.settleTokens(termAgentID, int(inTok), int(outTok), promptChars, bytesStreamed)
				s.finishInference(logID, inSafe, outSafe, "ok")
				s.recordTokens(u.ID, inSafe, outSafe)
				slotBilled = true
				return
			}
			switch f.Type {
			case actvTypeToken:
				outTok = f.TokSeq
				bytesStreamed += len(f.Payload)
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
				inSafe, outSafe := s.settleTokens(termAgentID, int(inTok), int(outTok), promptChars, bytesStreamed)
				s.finishInference(logID, inSafe, outSafe, "ok")
				s.recordTokens(u.ID, inSafe, outSafe)
				slotBilled = true
				return
			case actvTypeError:
				writeSSEEvent(w, flusher, map[string]any{"error": string(f.Payload)})
				inSafe, outSafe := s.settleTokens(termAgentID, int(inTok), int(outTok), promptChars, bytesStreamed)
				s.finishInference(logID, inSafe, outSafe, "failed")
				return
			}
		case <-ctx.Done():
			writeSSEErr(w, flusher, "timeout")
			inSafe, outSafe := s.settleTokens(termAgentID, int(inTok), int(outTok), promptChars, bytesStreamed)
			s.finishInference(logID, inSafe, outSafe, "failed")
			return
		}
	}
}
