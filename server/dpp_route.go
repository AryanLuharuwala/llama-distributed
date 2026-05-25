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
//
// When one agent owns multiple non-adjacent stages (e.g. TE on stage 0 +
// VAE on stage 2 with someone else's UNet at stage 1), the same dppPeer is
// shared across both stages.  Dispatch must therefore decide routing from
// the *frame's* destination stage (worker.py emits with stage=frame.stage+1)
// rather than from a single "next agent" recorded on the peer — otherwise
// the VAE's image ACT would follow the TE→UNet next-hop and be lost.
type dppPeer struct {
	reqID      uint16
	isTerminal bool

	// chain[i] is the agentConn that hosts stage i in the plan; len(chain)
	// is the total stage count.  A frame whose destination stage equals
	// len(chain) (or higher) is terminal output for the server.  All peers
	// share the same chain slice.
	chain []*agentConn

	// Loopback target: if non-nil and the inbound frame has
	// actvFlagDPPLoop set, route to this agent instead of the per-stage
	// nextAgent.  Used for multi-step UNet where the last UNet stage's
	// per-step noise prediction goes back to the first UNet stage rather
	// than forward to VAE.  loopbackStage names the stage index to set on
	// the forwarded frame so the receiving agent's runtime resolves the
	// right per-req runtime slot.
	loopbackAgent *agentConn
	loopbackStage uint16

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
	dppFlags := actvFlagDPPLatent | actvFlagDPPFinal | actvFlagDPPImage | actvFlagDPPLoop
	if frame.Type == actvTypeAct && (frame.Flags&dppFlags) == 0 {
		ac.peerMu.RUnlock()
		return false
	}

	var fwd *ActvFrame
	var nextAgent *agentConn
	var doClose bool

	switch frame.Type {
	case actvTypeAct:
		// Loopback handling — last UNet stage emits this to send noise
		// prediction back to the first UNet stage for scheduler.step +
		// next-step kickoff.  Routes to loopbackAgent regardless of
		// where we are in the chain.  Override stage to the loopback
		// target's stage_idx so the receiving worker resolves the
		// correct per-req runtime.
		if frame.Flags&actvFlagDPPLoop != 0 && dpp.loopbackAgent != nil {
			fwd = &ActvFrame{
				Type:    frame.Type,
				ReqID:   frame.ReqID,
				Stage:   dpp.loopbackStage,
				TokSeq:  frame.TokSeq,
				DType:   frame.DType,
				Flags:   frame.Flags,
				Dims:    frame.Dims,
				Payload: frame.Payload,
			}
			nextAgent = dpp.loopbackAgent
			break
		}
		// Route by the frame's destination stage (worker.py emits with
		// stage=frame.stage+1).  Frames whose destination is past the
		// chain are terminal output for the server — even when this
		// agent also hosts an earlier stage in the chain.
		destStage := int(frame.Stage)
		if destStage >= len(dpp.chain) {
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
		// Intermediate destination — forward to the agent that hosts the
		// destination stage.  If that agent is this same agent the local
		// dpp_adapter already routed in-process; only frames crossing
		// agents reach us over the WS.
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
		nextAgent = dpp.chain[destStage]
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
	// unet_stages: number of pipeline-parallel UNet stages.  Default 1.
	// When > 1, the single logical "unet" role expands into N stages each
	// owning a contiguous block range of the linearised UNet.  Block
	// ranges are computed by partitionUNetBlocks() using the model's
	// known topology (knownUNetTopologies).  Mutually exclusive with
	// cfg_split — the wire shape would have ambiguous (cond, uncond) ×
	// (stage_i, stage_j) pairings.  An unknown model falls back to a
	// single stage so generation still succeeds.
	unetStages := 1
	if v, ok := config["unet_stages"].(int); ok && v > 1 {
		unetStages = v
	} else if vf, ok := config["unet_stages"].(float64); ok && int(vf) > 1 {
		unetStages = int(vf)
	}
	var unetBlocks []uNetBlockRange
	if unetStages > 1 {
		if cfgSplit {
			return nil, errMsg("dpp plan: unet_stages and cfg_split are mutually exclusive")
		}
		topo, ok := lookupUNetTopology(model)
		if !ok {
			// Unknown model — silently degrade to single UNet stage.
			unetStages = 1
		} else if unetStages > topo.TotalBlocks {
			return nil, errMsg("dpp plan: unet_stages exceeds model block count")
		} else {
			unetBlocks = partitionUNetBlocks(topo.TotalBlocks, unetStages)
			if unetBlocks == nil {
				return nil, errMsg("dpp plan: partition failed")
			}
		}
	}
	order := []string{"text_encoder", "unet", "vae"}
	if cfgSplit {
		order = []string{"text_encoder", "unet_cond", "unet_uncond", "vae"}
	} else if unetStages > 1 {
		// One logical "unet" role expands to N stages.  Role names are
		// "unet_0", "unet_1", ...; the runtime treats them identically
		// to plain "unet" but uses block_lo/block_hi from the control
		// frame to pick which torch modules to load.
		order = []string{"text_encoder"}
		for i := 0; i < unetStages; i++ {
			order = append(order, "unet_"+strconv.Itoa(i))
		}
		order = append(order, "vae")
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
		} else if unetStages > 1 {
			// Assign all UNet stages first (VRAM-led), then TE, then VAE.
			// Stage 0 ("unet_0") gets the biggest rig because it holds
			// conv_in + time_embedding in addition to its block share.
			autoOrder = nil
			for i := 0; i < unetStages; i++ {
				autoOrder = append(autoOrder, "unet_"+strconv.Itoa(i))
			}
			autoOrder = append(autoOrder, "text_encoder", "vae")
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
		blockLo, blockHi := -1, -1
		// Surface role: the runtime sees just "unet" + block range, so it
		// dispatches to the same code path regardless of how many UNet
		// stages there are.  unet_N is purely a planner-side label.
		emittedRole := role
		if unetStages > 1 && len(role) >= 5 && role[:5] == "unet_" {
			idx, err := strconv.Atoi(role[5:])
			if err != nil || idx < 0 || idx >= len(unetBlocks) {
				return nil, errMsg("dpp plan: invalid unet stage index " + role)
			}
			blockLo = unetBlocks[idx].Lo
			blockHi = unetBlocks[idx].Hi
			emittedRole = "unet"
		}
		plan.Stages = append(plan.Stages, dppStage{
			AgentID:  ag,
			StageIdx: i,
			Role:     emittedRole,
			BlockLo:  blockLo,
			BlockHi:  blockHi,
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
	// UnetStages > 1 splits the denoiser into N pipeline-parallel stages
	// each owning a contiguous block range of the linearised UNet.  Each
	// rig only loads its share of blocks (plus conv_in/time_embedding on
	// stage 0 and conv_norm_out/conv_out on the last stage), making it
	// possible to run models that don't fit on any single rig.  Mutually
	// exclusive with CFGSplit.
	UnetStages int `json:"unet_stages,omitempty"`
	// UnetAgents pins specific agents to the N UNet stages, in order
	// (stage 0 first).  Optional; if empty the planner auto-picks the N
	// most VRAM-rich rigs.  Length must match UnetStages when set.
	UnetAgents []string `json:"unet_agents,omitempty"`
	// InitImageURL is the URL of an existing image to seed an img2img run.
	// When set, the text_encoder stage fetches it and the UNet starts from
	// its VAE-encoded latent + diffusion noise per `strength` instead of
	// pure noise.
	InitImageURL string  `json:"init_image_url,omitempty"`
	Strength     float64 `json:"strength,omitempty"` // 0..1; default 0.6 for img2img
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
	if body.UnetStages < 0 {
		writeErr(w, 400, "unet_stages must be ≥ 0")
		return
	}
	if body.UnetStages > 1 && body.CFGSplit {
		writeErr(w, 400, "unet_stages and cfg_split are mutually exclusive")
		return
	}
	if body.UnetStages > 1 && len(body.UnetAgents) > 0 && len(body.UnetAgents) != body.UnetStages {
		writeErr(w, 400, "unet_agents length must match unet_stages")
		return
	}
	// Expand pinned unet_agents into role map slots "unet_0", "unet_1",
	// ... so the planner sees them as already-assigned and doesn't
	// auto-pick a different rig.
	if body.UnetStages > 1 && len(body.UnetAgents) > 0 {
		if body.Roles == nil {
			body.Roles = map[string]string{}
		}
		// If caller also supplied a legacy "unet" pin, treat it as the
		// first stage hint when no per-stage pins are given (already
		// handled above by len check — at this point both arrays exist
		// and are correctly sized).
		delete(body.Roles, "unet")
		for i, a := range body.UnetAgents {
			body.Roles["unet_"+strconv.Itoa(i)] = a
		}
	}
	// Empty roles map is fine — planDPP auto-picks by VRAM cost when slots
	// are blank.  We keep partial maps working too (caller pins one role,
	// auto-pick fills the rest).

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
	// Unconditional slot-refund guard — see inference.go for rationale.
	slotBilled := false
	defer func() {
		if !slotBilled {
			s.refundRequestSlot(u.ID)
		}
	}()

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Minute)
	defer cancel()

	img, err := s.executeDPPInference(ctx, u.ID, body)
	if err != nil {
		if err == errDPPTimeout {
			writeErr(w, 504, "dpp timeout")
			return
		}
		writeErr(w, 503, err.Error())
		return
	}
	reqID16 := nextReqID() // for the on-disk filename only; the inference path already minted its own
	outPath, perr := s.persistDPPImage(u.ID, reqID16, img)
	if perr != nil {
		writeErr(w, 500, "persist: "+perr.Error())
		return
	}
	writeJSON(w, 200, map[string]any{
		"created": nowUnix(),
		"data":    []map[string]string{{"url": outPath}},
	})
	slotBilled = true
	return
}

// errDPPTimeout signals the run exceeded its deadline.  The HTTP handler
// translates it to 504; the comfy-job goroutine translates it to status=failed.
var errDPPTimeout = errMsg("dpp timeout")

// executeDPPInference is the shared execution body used by both
// handleInferDPP and the comfy-routed DPP path.  It plans the stages,
// attaches dpp peers, sends dpp_route control frames, kicks off stage 0,
// and waits for the terminal stage to emit either a final image or an
// error.  Rate-limit + persistence are caller responsibilities.
func (s *server) executeDPPInference(ctx context.Context, userID int64, body dppRequestBody) ([]byte, error) {
	config := map[string]any{
		"steps":       body.Steps,
		"cfg_scale":   body.CFG,
		"width":       body.Width,
		"height":      body.Height,
		"seed":        body.Seed,
		"prompt":      body.Prompt,
		"cfg_split":   body.CFGSplit,
		"unet_stages": body.UnetStages,
	}
	if body.InitImageURL != "" {
		config["init_image_url"] = body.InitImageURL
	}
	if body.Strength > 0 {
		config["strength"] = body.Strength
	}
	plan, err := s.planDPP(body.PoolID, body.Model, body.Roles, config)
	if err != nil {
		return nil, err
	}

	// Resolve every stage's agentConn.  Same agentID across stages → same conn.
	type stageRef struct {
		stage dppStage
		conn  *agentConn
	}
	refs := make([]stageRef, len(plan.Stages))
	for i, st := range plan.Stages {
		ac, ok := s.hub.findAgent(userID, st.AgentID)
		if !ok {
			return nil, errMsg("stage " + st.AgentID + " offline")
		}
		st.UserID = userID
		refs[i] = stageRef{stage: st, conn: ac}
	}

	reqID16 := nextReqID()

	chain := make([]*agentConn, len(refs))
	for i := range refs {
		chain[i] = refs[i].conn
	}

	peers := make([]*dppPeer, len(refs))
	attached := make(map[*agentConn]*dppPeer)
	cleanup := func() {
		for ac, dp := range attached {
			ac.detachDPPPeer(dp)
		}
	}
	for i := range refs {
		ac := refs[i].conn
		if existing, ok := attached[ac]; ok {
			peers[i] = existing
			if i == len(refs)-1 {
				existing.isTerminal = true
			}
			continue
		}
		dp := &dppPeer{
			reqID:      reqID16,
			isTerminal: i == len(refs)-1,
			chain:      chain,
			closed:     make(chan struct{}),
		}
		if i == len(refs)-1 {
			dp.tokens = make(chan *ActvFrame, 16)
		}
		if !ac.attachDPPPeer(dp) {
			cleanup()
			return nil, errMsg("stage " + refs[i].stage.AgentID + " is busy")
		}
		peers[i] = dp
		attached[ac] = dp
	}
	defer cleanup()

	termPeer := peers[len(peers)-1]
	if termPeer.tokens == nil {
		termPeer.tokens = make(chan *ActvFrame, 16)
		termPeer.isTerminal = true
	}

	firstUnet, lastUnet := -1, -1
	for i, ref := range refs {
		if ref.stage.Role == "unet" && ref.stage.BlockLo >= 0 {
			if firstUnet < 0 {
				firstUnet = i
			}
			lastUnet = i
		}
	}
	if firstUnet >= 0 && lastUnet > firstUnet {
		peers[lastUnet].loopbackAgent = refs[firstUnet].conn
		peers[lastUnet].loopbackStage = uint16(refs[firstUnet].stage.StageIdx)
	}

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
		return nil, errMsg("stage 0 buffer full")
	}

	for {
		select {
		case f, ok := <-termPeer.tokens:
			if !ok {
				return nil, errMsg("stage closed before image")
			}
			switch f.Type {
			case actvTypeAct:
				if f.Flags&actvFlagDPPImage == 0 {
					continue
				}
				return f.Payload, nil
			case actvTypeDone:
				return nil, errMsg("dpp stage emitted done before image")
			case actvTypeError:
				return nil, errMsg(string(f.Payload))
			}
		case <-ctx.Done():
			return nil, errDPPTimeout
		}
	}
}

// persistDPPImage drops a generated image into the same comfy-out tree the
// non-sharded path uses, so the existing /comfy/out/{id}/{file} signed-URL
// machinery can serve it back to the client.
func (s *server) persistDPPImage(uid int64, reqID uint16, data []byte) (string, error) {
	res, err := s.dbExec(
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
	_, _ = s.dbExec(
		`UPDATE comfy_jobs SET status='done', out_files=?, updated_at=? WHERE id=?`,
		string(outFilesJSON), nowUnix(), jobID,
	)
	return s.signComfyOutputURL(uid, jobID, file, 1*time.Hour), nil
}

type dppErr string

func (e dppErr) Error() string { return string(e) }
func errMsg(s string) error    { return dppErr(s) }
