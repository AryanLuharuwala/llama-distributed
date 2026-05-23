package main

// Diffusion pipeline-parallel (DPP) routing.
//
// Parallel to pp_route.go (LLM PP), but for image/video diffusion models.
// Each stage on the network is one WS endpoint to dist-server; what's
// inside the stage (single GPU today, NCCL-glued cluster later) is opaque
// to the rest of the system — that boundary is the place a future
// intra-cluster transport plugs in without touching this code.
//
// Stage roles for a typical SDXL/FLUX/SD3 split:
//
//   - "text_encoder"     — tokenise + run CLIP/T5 encoders → cond/uncond embeds
//   - "unet" (or "dit")  — owns one contiguous block range of the denoiser
//   - "vae"              — final-latent decode → PNG/JPEG bytes
//
// Wire frames reuse the ACTV envelope.  Every DPP frame carries one of
// the actvFlagDPP* bits in Flags so the dispatcher can route it.  Payload
// layout is just dtype × prod(Dims) bytes — no DPP-specific framing.
//
// Per-step traffic on the demo split (TE + UNet + VAE on 3 rigs, SDXL):
//   prompt utf8                                       ~tens of bytes
//   cond+uncond embeds fp16 [2,77,2048]               ~63 KiB
//   final latent fp16 [1,4,128,128]                   ~131 KiB
//   image bytes (PNG)                                 ~1-2 MiB
// Total <2 MiB per generation across the WAN.
//
// On the split-UNet stress case (laptop=TE+VAE, two rigs share UNet), every
// step ferries a UNet hidden state fp16 [1,Cmax,H,W] (~1-4 MiB) between
// the two UNet stages.  30 steps × 4 MiB = ~120 MiB per gen.  Acceptable.

import (
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"time"
)

// dppStage is the planner's per-stage record.  Built by planDPP() and
// materialised into a dpp_route control frame for each agent.
type dppStage struct {
	UserID   int64
	AgentID  string
	StageIdx int
	Role     string // "text_encoder" | "unet" | "unet_cond" | "unet_uncond" | "vae"
	// UNet block slice.  -1 means "all blocks this stage's role can hold".
	BlockLo int
	BlockHi int
	// Partner names the paired agent for CFG-split UNet halves.  An
	// `unet_cond` stage's Partner is the `unet_uncond` agent and vice
	// versa.  Empty on non-split stages.  The runtime uses this to know
	// where to ship its noise prediction for the cfg combine step.
	Partner string
}

type dppPlan struct {
	Model  string
	Stages []dppStage
	Config map[string]any
}

// dppPeer is registered on each stage's agentConn for the duration of a
// single DPP request.  Identical control-flow shape to ppPeer.
type dppPeer struct {
	reqID      uint16
	stageIdx   uint16
	isTerminal bool

	nextAgent *agentConn // stage (idx+1), or nil for the last stage

	// Terminal stage forwards image bytes (or error/done) here.
	tokens chan *ActvFrame
	closed chan struct{}
	once   sync.Once
}

func (p *dppPeer) close() {
	p.once.Do(func() {
		close(p.closed)
		if p.tokens != nil {
			close(p.tokens)
		}
	})
}

// dispatchDPPFromAgent — sibling of dispatchActvFromAgent for the diffusion
// path.  Called from dispatchBinaryFromAgent after the LLM-PP dispatcher
// passes.  Returns true if the frame was consumed.
func (ac *agentConn) dispatchDPPFromAgent(data []byte) bool {
	frame, err := DecodeActvFrame(data)
	if err != nil {
		log.Printf("dpp: actv decode: %v", err)
		return true
	}
	if frame == nil {
		return false
	}
	// Hold peerMu.RLock across the terminal-stage send so ac.close()
	// (which takes Lock then closes dpp.tokens) can't race with us.
	ac.peerMu.RLock()
	dpp := ac.dppPeer
	if dpp == nil {
		ac.peerMu.RUnlock()
		return false
	}
	if frame.ReqID != dpp.reqID {
		ac.peerMu.RUnlock()
		return true
	}
	// Only consume DPP-tagged frames here; bare ACTV without any DPP flag
	// belongs to the LLM pipeline path (which already passed).
	dppFlags := actvFlagDPPLatent | actvFlagDPPFinal | actvFlagDPPImage
	if frame.Type == actvTypeAct && (frame.Flags&dppFlags) == 0 {
		ac.peerMu.RUnlock()
		return false
	}

	var fwd *ActvFrame
	var nextAgent *agentConn
	var doClose bool

	switch frame.Type {
	case actvTypeAct:
		if dpp.nextAgent == nil {
			if frame.Flags&actvFlagDPPImage != 0 || frame.Flags&actvFlagDPPFinal != 0 {
				select {
				case <-dpp.closed:
				default:
					select {
					case dpp.tokens <- frame:
					default:
						log.Printf("dpp: terminal channel full, dropping")
					}
				}
				ac.peerMu.RUnlock()
				return true
			}
			log.Printf("dpp: terminal stage got non-final activation, dropping")
			ac.peerMu.RUnlock()
			return true
		}
		// Intermediate stage — forward to next agent.  The python worker
		// already bumps Stage on emit (worker.py: stage=frame.stage + 1)
		// so the incoming frame's Stage already names the destination
		// stage.  Bumping again here would shift it past the registered
		// runtime and trigger "no runtime for req=X" on the next agent.
		fwd = &ActvFrame{
			Type:    frame.Type,
			ReqID:   frame.ReqID,
			Stage:   frame.Stage,
			TokSeq:  frame.TokSeq,
			DType:   frame.DType,
			Flags:   frame.Flags,
			Dims:    frame.Dims,
			Payload: frame.Payload,
		}
		nextAgent = dpp.nextAgent
	case actvTypeToken, actvTypeDone, actvTypeError:
		select {
		case <-dpp.closed:
		default:
			select {
			case dpp.tokens <- frame:
			default:
				log.Printf("dpp: terminal channel full, dropping")
			}
		}
		if frame.Type == actvTypeDone || frame.Type == actvTypeError {
			doClose = true
		}
	}
	ac.peerMu.RUnlock()

	if fwd != nil && nextAgent != nil {
		if !nextAgent.sendBin(fwd.Encode()) {
			log.Printf("dpp: next stage buffer full, dropping")
		}
	}
	if doClose {
		dpp.close()
	}
	return true
}

func (ac *agentConn) attachDPPPeer(dp *dppPeer) bool {
	ac.peerMu.Lock()
	defer ac.peerMu.Unlock()
	if ac.peer != nil || len(ac.inferPeers) > 0 || ac.ppPeer != nil || ac.dppPeer != nil {
		return false
	}
	ac.dppPeer = dp
	return true
}

func (ac *agentConn) detachDPPPeer(dp *dppPeer) {
	ac.peerMu.Lock()
	defer ac.peerMu.Unlock()
	if ac.dppPeer == dp {
		ac.dppPeer = nil
	}
}

// ─── Planner ───────────────────────────────────────────────────────────────

// planDPP chooses the stage assignment for one diffusion request.  In this
// milestone we hardcode the 3-stage layout (TE → UNet → VAE) and let the
// caller declare which rig fills which role via the `roles` map.  A future
// pass should query rig caps (`dpp_caps`) and auto-assign.
//
// `roles` example: {"text_encoder": "rig-a", "unet": "rig-b", "vae": "rig-a"}.
// Stages with the same agentID share a process — the agent's runtime can
// keep multiple models resident concurrently if VRAM allows.
//
// If `config["cfg_split"]` is true, the UNet stage is replaced by two
// parallel halves — `unet_cond` then `unet_uncond` — that the runtime can
// dispatch in parallel.  The two halves are paired via dppStage.Partner so
// each side knows where to exchange its noise prediction for the combine
// step.  Two distinct rigs is the happy path; if only one UNet-capable rig
// exists the planner allows both halves on the same rig (degenerates to
// plain CFG but keeps the wire shape constant).
func (s *server) planDPP(poolID int64, model string, roles map[string]string, config map[string]any) (*dppPlan, error) {
	// Required role order for SDXL-class models.
	cfgSplit := false
	if v, ok := config["cfg_split"].(bool); ok && v {
		cfgSplit = true
	}
	order := []string{"text_encoder", "unet", "vae"}
	if cfgSplit {
		order = []string{"text_encoder", "unet_cond", "unet_uncond", "vae"}
	}
	plan := &dppPlan{Model: model, Config: config}

	// Caller-supplied role map wins — explicit beats heuristic.  We only
	// auto-plan the roles the caller left blank, so a partial map ("pin
	// unet on rig-b but auto-pick everything else") works.
	missing := false
	for _, role := range order {
		if roles == nil || roles[role] == "" {
			missing = true
			break
		}
	}
	if missing {
		costs, err := s.rigCostsForPool(poolID)
		if err != nil {
			return nil, err
		}
		if len(costs) == 0 {
			return nil, errMsg("dpp plan: no online rigs in pool")
		}
		if roles == nil {
			roles = map[string]string{}
		}
		taken := map[agentKey]bool{}
		// Mark caller-pinned rigs as taken so the auto-picker doesn't
		// duplicate them unless it has to.
		for _, role := range order {
			if ag := roles[role]; ag != "" {
				// We don't know the userID here; rely on agentID alone to
				// detect collisions across the rest of the planner pass.
				taken[agentKey{0, ag}] = true
			}
		}
		// Auto-assign roles in cost order: UNet halves first (they're the
		// bottleneck — and for cfg_split we want both on distinct rigs),
		// then text_encoder, then vae.  Order matters: assign unet_cond
		// before unet_uncond so the first/biggest rig grabs the cond half
		// and the second-largest claims uncond.
		autoOrder := []string{"unet", "text_encoder", "vae"}
		if cfgSplit {
			autoOrder = []string{"unet_cond", "unet_uncond", "text_encoder", "vae"}
		}
		for _, role := range autoOrder {
			if roles[role] != "" {
				continue
			}
			pick, ok := pickRigForRole(costs, role, taken)
			if !ok {
				return nil, errMsg("dpp plan: no rig available for role " + role)
			}
			roles[role] = pick.info.agentID
			taken[agentKey{pick.info.userID, pick.info.agentID}] = true
			// Also mark the userID=0 sentinel so subsequent roles see this
			// as pinned via the original loop above.
			taken[agentKey{0, pick.info.agentID}] = true
		}
	}

	for i, role := range order {
		ag, ok := roles[role]
		if !ok || ag == "" {
			return nil, errMsg("dpp plan: no rig assigned to role " + role)
		}
		// Resolve agent → (userID, agentID).  For now we accept "userID/agentID"
		// or just "agentID" (use the requester's own user).  The userID is
		// filled in by the handler before dispatch.
		var partner string
		if cfgSplit {
			switch role {
			case "unet_cond":
				partner = roles["unet_uncond"]
			case "unet_uncond":
				partner = roles["unet_cond"]
			}
		}
		plan.Stages = append(plan.Stages, dppStage{
			AgentID:  ag,
			StageIdx: i,
			Role:     role,
			BlockLo:  -1,
			BlockHi:  -1,
			Partner:  partner,
		})
	}
	return plan, nil
}

// ─── Handler ───────────────────────────────────────────────────────────────

type dppRequestBody struct {
	PoolID int64             `json:"pool_id"`
	Model  string            `json:"model"`
	Prompt string            `json:"prompt"`
	Roles  map[string]string `json:"roles"` // role -> agent_id
	Steps  int               `json:"steps"`
	CFG    float64           `json:"cfg_scale"`
	Width  int               `json:"width"`
	Height int               `json:"height"`
	Seed   int64             `json:"seed"`
	// CFGSplit asks the planner to run cond/uncond UNet halves on two
	// rigs in parallel (per denoise step).  Falls back to the same rig
	// for both halves when only one UNet-capable rig is available.
	CFGSplit bool `json:"cfg_split,omitempty"`
}

// POST /api/infer_dpp — kick off a diffusion-PP generation across N stages.
// Response is JSON (not SSE) — diffusion has no token stream; we either
// return the final image URL or an error.
func (s *server) handleInferDPP(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		tok := bearerFromRequest(r)
		if tok == "" {
			writeErr(w, 401, "auth required")
			return
		}
		var apiOK bool
		u, apiOK = s.userFromAPIKey(tok)
		if !apiOK {
			writeErr(w, 401, "bad api key")
			return
		}
	}
	var body dppRequestBody
	if err := json.NewDecoder(io.LimitReader(r.Body, 64<<10)).Decode(&body); err != nil {
		writeErr(w, 400, "bad json: "+err.Error())
		return
	}
	if body.PoolID == 0 || body.Prompt == "" {
		writeErr(w, 400, "pool_id and prompt required")
		return
	}
	if body.Steps == 0 {
		body.Steps = 30
	}
	if body.CFG == 0 {
		body.CFG = 7.5
	}
	if body.Width == 0 {
		body.Width = 1024
	}
	if body.Height == 0 {
		body.Height = 1024
	}
	if len(body.Roles) == 0 {
		writeErr(w, 400, "roles map required (e.g. {\"text_encoder\":\"rig-a\",\"unet\":\"rig-b\",\"vae\":\"rig-a\"})")
		return
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
		writeJSON(w, 429, map[string]any{"error": "rate limit", "policy": policy, "usage": snap})
		return
	}

	config := map[string]any{
		"steps":     body.Steps,
		"cfg_scale": body.CFG,
		"width":     body.Width,
		"height":    body.Height,
		"seed":      body.Seed,
		"prompt":    body.Prompt,
		"cfg_split": body.CFGSplit,
	}
	plan, err := s.planDPP(body.PoolID, body.Model, body.Roles, config)
	if err != nil {
		writeErr(w, 503, err.Error())
		return
	}

	// Resolve every stage's agentConn.  Same agentID across stages → same conn.
	type stageRef struct {
		stage dppStage
		conn  *agentConn
	}
	refs := make([]stageRef, len(plan.Stages))
	for i, st := range plan.Stages {
		ac, ok := s.hub.findAgent(u.ID, st.AgentID)
		if !ok {
			writeErr(w, 503, "stage "+st.AgentID+" offline")
			return
		}
		st.UserID = u.ID
		refs[i] = stageRef{stage: st, conn: ac}
	}

	reqID16 := nextReqID()

	// Attach a dppPeer to each *distinct* agentConn (dedupe; one agent can
	// own multiple roles).  Same-agent stages share the same peer because
	// the dispatch only checks ac.dppPeer.
	peers := make([]*dppPeer, len(refs))
	attached := make(map[*agentConn]*dppPeer)
	cleanup := func() {
		for ac, dp := range attached {
			ac.detachDPPPeer(dp)
		}
	}
	for i := range refs {
		ac := refs[i].conn
		var next *agentConn
		// Find the next *distinct* conn after i.
		for j := i + 1; j < len(refs); j++ {
			if refs[j].conn != ac {
				next = refs[j].conn
				break
			}
		}
		if existing, ok := attached[ac]; ok {
			// Same agent doing multiple roles — re-use the peer but flag it
			// as terminal if any of its stages is terminal.
			peers[i] = existing
			if i == len(refs)-1 {
				existing.isTerminal = true
			}
			continue
		}
		dp := &dppPeer{
			reqID:      reqID16,
			stageIdx:   uint16(i),
			isTerminal: i == len(refs)-1,
			nextAgent:  next,
			closed:     make(chan struct{}),
		}
		if i == len(refs)-1 {
			dp.tokens = make(chan *ActvFrame, 16)
		}
		if !ac.attachDPPPeer(dp) {
			cleanup()
			writeErr(w, 503, "stage "+refs[i].stage.AgentID+" is busy")
			return
		}
		peers[i] = dp
		attached[ac] = dp
	}
	defer cleanup()

	// Terminal peer is whichever peer wraps the last conn.  If the last
	// stage shares an agent with an earlier stage, the same peer covers it
	// and its tokens channel is already wired.
	termPeer := peers[len(peers)-1]
	if termPeer.tokens == nil {
		// We deduped earlier — promote it to terminal now.
		termPeer.tokens = make(chan *ActvFrame, 16)
		termPeer.isTerminal = true
	}

	// Send each stage its dpp_route control frame.  Same agent receiving
	// multiple control frames is fine — the runtime indexes by req_id +
	// stage_idx.
	for i, ref := range refs {
		var nextID, prevID string
		if i+1 < len(refs) {
			nextID = refs[i+1].stage.AgentID
		}
		if i > 0 {
			prevID = refs[i-1].stage.AgentID
		}
		msg := map[string]any{
			"kind":          "dpp_route",
			"req_id":        reqID16,
			"stage_idx":     ref.stage.StageIdx,
			"stage_count":   len(refs),
			"role":          ref.stage.Role,
			"block_lo":      ref.stage.BlockLo,
			"block_hi":      ref.stage.BlockHi,
			"model":         plan.Model,
			"is_first":      i == 0,
			"is_last":       i == len(refs)-1,
			"next_agent":    nextID,
			"prev_agent":    prevID,
			"partner_agent": ref.stage.Partner,
			"config":        plan.Config,
		}
		ref.conn.send(msg)
		log.Printf("dpp: sent dpp_route req=%d stage=%d role=%s to agent=%s", reqID16, ref.stage.StageIdx, ref.stage.Role, ref.stage.AgentID)
	}

	// Kick off stage 0 with the prompt as a utf-8 byte ACTV.  The stage's
	// runtime interprets it as the diffusion prompt (vs an LLM token stream)
	// because it just got a `dpp_route` control with role=text_encoder.
	kick := &ActvFrame{
		Type:    actvTypeAct,
		ReqID:   reqID16,
		Stage:   0,
		TokSeq:  0,
		DType:   actvDTypeBytes,
		Flags:   actvFlagIsPrompt | actvFlagEndOfPrompt | actvFlagDPPLatent,
		Payload: []byte(body.Prompt),
	}
	if !refs[0].conn.sendBin(kick.Encode()) {
		writeErr(w, 503, "stage 0 buffer full")
		return
	}

	// Wait for the terminal stage to emit either an image, a done, or an
	// error.  Diffusion takes a while — give it 5 minutes.
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Minute)
	defer cancel()

	for {
		select {
		case f, ok := <-termPeer.tokens:
			if !ok {
				writeErr(w, 500, "stage closed before image")
				return
			}
			switch f.Type {
			case actvTypeAct:
				// Final image bytes from the VAE stage.
				if f.Flags&actvFlagDPPImage == 0 {
					continue
				}
				outPath, err := s.persistDPPImage(u.ID, reqID16, f.Payload)
				if err != nil {
					writeErr(w, 500, "persist: "+err.Error())
					return
				}
				writeJSON(w, 200, map[string]any{
					"created": nowUnix(),
					"data": []map[string]string{
						{"url": outPath},
					},
				})
				return
			case actvTypeDone:
				writeJSON(w, 200, map[string]any{"done": true})
				return
			case actvTypeError:
				writeErr(w, 500, string(f.Payload))
				return
			}
		case <-ctx.Done():
			writeErr(w, 504, "dpp timeout")
			return
		}
	}
}

// persistDPPImage drops a generated image into the same comfy-out tree the
// non-sharded path uses, so the existing /comfy/out/{id}/{file} signed-URL
// machinery can serve it back to the client.
func (s *server) persistDPPImage(uid int64, reqID uint16, data []byte) (string, error) {
	res, err := s.db.Exec(
		`INSERT INTO comfy_jobs (user_id, prompt, params_json, status, created_at, updated_at)
		 VALUES (?, '', '{"dpp":true}', 'running', ?, ?)`,
		uid, nowUnix(), nowUnix(),
	)
	if err != nil {
		return "", err
	}
	jobID, _ := res.LastInsertId()
	file := "dpp-" + strconv.FormatUint(uint64(reqID), 10) + ".png"
	dir := filepath.Join(s.cfg.comfyOutDir, strconv.FormatInt(jobID, 10))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	if err := os.WriteFile(filepath.Join(dir, file), data, 0o644); err != nil {
		return "", err
	}
	outFilesJSON, _ := json.Marshal([]string{file})
	_, _ = s.db.Exec(
		`UPDATE comfy_jobs SET status='done', out_files=?, updated_at=? WHERE id=?`,
		string(outFilesJSON), nowUnix(), jobID,
	)
	return s.signComfyOutputURL(jobID, file, 1*time.Hour), nil
}

type dppErr string

func (e dppErr) Error() string { return string(e) }
func errMsg(s string) error    { return dppErr(s) }
