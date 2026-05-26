package main

// sdcpp_comfyjob.go — comfy-routed sd.cpp (stable-diffusion.cpp) execution.
//
// Parallel to runDPPComfyJob, but the underlying transport is the
// `sdcpp_role_route` (3-hop TE/UNet/VAE) or `sdcpp_route` (single rig
// "full" pipeline) frames defined in sdcpp_caps.go.  When the backend
// selector picks sd.cpp:
//
//   1. Try planSdcppRoleChain — three rigs covering TE/UNet/VAE.  If it
//      returns a full chain, fan the request out as three sequential
//      sdcpp_role_route hops, each waiting on the role's frame_b64 in
//      the previous step.
//   2. Otherwise try planSdcppFull — single rig with the "full" role.
//   3. Otherwise fail the job with no_rigs.
//
// Result PNG lands at <comfyOutDir>/<jobID>/sdcpp.png — mirroring the
// dpp.png / comfy default-result layout so the UI / signed-URL paths
// just work.

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"time"
)

// sdcppJobTimeout — sd.cpp is generally faster than DPP (no Python
// import overhead) but the role-chain is still bound by VAE decode +
// network hops.  10 minutes matches dppJobTimeout for parity.
const sdcppJobTimeout = 10 * time.Minute

// sdcppRequestBody is the comfy-routed request shape consumed by the
// job goroutine.  Mirrors dppRequestBody — only the fields sd.cpp
// actually uses are kept.
type sdcppRequestBody struct {
	PoolID    int64
	Model     string  // for routing decisions; not directly passed to the worker
	ModelPath string  // absolute path on each rig — required by sd.cpp adapter
	Prompt    string
	Negative  string
	Steps     int
	CFG       float64
	Width     int
	Height    int
	Seed      int64
	Sampler   string
	Scheduler string
	CFGSplit  bool
	ClipSkip  int
}

// runSdcppComfyJob owns the comfy_jobs row for an sd.cpp run.  Drives
// the role chain (or single "full" rig), waits on the result PNG,
// persists it, marks the row done/failed.
func (s *server) runSdcppComfyJob(ctx context.Context, userID, jobID int64, body sdcppRequestBody) {
	defer s.refundRequestSlot(userID)
	defer s.comfyJobs.finish(jobID)

	_, _ = s.dbExec(
		`UPDATE comfy_jobs SET status='running', updated_at=? WHERE id=?`,
		nowUnix(), jobID,
	)

	runCtx, cancel := context.WithTimeout(ctx, sdcppJobTimeout)
	defer cancel()

	img, err := s.executeSdcppInference(runCtx, userID, body)
	if err != nil {
		log.Printf("sdcpp-comfy job=%d failed: %v", jobID, err)
		s.failComfyJob(jobID, err.Error())
		return
	}
	if len(img) == 0 {
		s.failComfyJob(jobID, "empty image from sdcpp")
		return
	}

	dir := filepath.Join(s.cfg.comfyOutDir, strconv.FormatInt(jobID, 10))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		s.failComfyJob(jobID, "mkdir: "+err.Error())
		return
	}
	file := "sdcpp.png"
	if err := os.WriteFile(filepath.Join(dir, file), img, 0o644); err != nil {
		s.failComfyJob(jobID, "write: "+err.Error())
		return
	}
	outFiles, _ := json.Marshal([]string{file})
	_, _ = s.dbExec(
		`UPDATE comfy_jobs SET status='done', out_files=?, updated_at=? WHERE id=?`,
		string(outFiles), nowUnix(), jobID,
	)
}

// executeSdcppInference picks a plan, dispatches frames, blocks on the
// per-req_id result channel until a final PNG arrives or an error
// surfaces.  Returns the decoded PNG bytes.
//
// Model resolution: callers can either pass ModelPath (legacy — the
// path must exist on every rig in the chain) or pass body.Model to a
// file the rigs have advertised in sdcpp_models (preferred).  When
// body.Model is set we ask the planner to filter rigs by file
// availability and copy the rig-local path back onto the dispatched
// frame.
func (s *server) executeSdcppInference(ctx context.Context, userID int64, body sdcppRequestBody) ([]byte, error) {
	if body.ModelPath == "" && body.Model == "" {
		return nil, errors.New("either sdcpp_model_path or model required")
	}

	// Defaults — match the sd.cpp adapter's own fallbacks loosely so a
	// caller that only supplies a prompt still gets a reasonable image.
	if body.Steps == 0 {
		body.Steps = 20
	}
	if body.CFG == 0 {
		body.CFG = 7.5
	}
	if body.Width == 0 {
		body.Width = 512
	}
	if body.Height == 0 {
		body.Height = 512
	}

	reqID := nextReqID()
	ch := s.sdcppResults.Subscribe(reqID)
	defer s.sdcppResults.Unsubscribe(reqID)

	// When ModelPath was given explicitly we keep the legacy planner
	// path (no file filter) — the operator opted in to ensuring the
	// path exists on each rig.  Otherwise we filter by advertised name.
	plannerName := ""
	if body.ModelPath == "" {
		plannerName = body.Model
	}

	chain := s.planSdcppRoleChain(ctx, userID, plannerName)
	if chain != nil {
		return s.runSdcppRoleChain(ctx, reqID, ch, chain, body)
	}
	full, ok := s.planSdcppFull(ctx, userID, plannerName)
	if ok {
		return s.runSdcppFull(ctx, reqID, ch, full, body)
	}
	if plannerName != "" {
		return nil, errors.New("no sdcpp rig advertises model: " + plannerName)
	}
	return nil, errors.New("no sdcpp rig available (need 3-rig te/unet/vae chain or one 'full' rig)")
}

// runSdcppRoleChain fans the request out as TE → UNet → VAE.  Each hop
// waits for the prior role's `sdcpp_role_done`, copies the frame_b64
// into the next hop's input, and ships it.  VAE produces the final PNG
// via `sdcpp_done`.
func (s *server) runSdcppRoleChain(ctx context.Context, reqID uint16,
	ch chan sdcppResultMsg, chain []sdcppRoleAgent, body sdcppRequestBody) ([]byte, error) {
	if len(chain) < 3 {
		return nil, errors.New("incomplete role chain")
	}
	te, unet, vae := chain[0], chain[1], chain[2]

	// modelPathFor picks the rig-local advertised path when the planner
	// found one (CF12-A2); falls back to the body-supplied path so the
	// legacy caller still works.
	modelPathFor := func(a sdcppRoleAgent) string {
		if a.ModelPath != "" {
			return a.ModelPath
		}
		return body.ModelPath
	}

	// Hop 1: TE.
	if !s.dispatchSdcppRoleRoute(te, sdcppRoleRouteParams{
		ReqID:     reqID,
		ModelPath: modelPathFor(te),
		Prompt:    body.Prompt,
		Negative:  body.Negative,
		CFGSplit:  body.CFGSplit,
		ClipSkip:  body.ClipSkip,
	}) {
		return nil, errors.New("te rig offline at dispatch time")
	}

	sdcdB64, err := waitForRoleFrame(ctx, ch, "te")
	if err != nil {
		return nil, err
	}

	// Hop 2: UNet.
	if !s.dispatchSdcppRoleRoute(unet, sdcppRoleRouteParams{
		ReqID:     reqID,
		ModelPath: modelPathFor(unet),
		SDCDB64:   sdcdB64,
		Width:     body.Width,
		Height:    body.Height,
		Steps:     body.Steps,
		CFG:       body.CFG,
		Seed:      body.Seed,
		Sampler:   body.Sampler,
		Scheduler: body.Scheduler,
	}) {
		return nil, errors.New("unet rig offline at dispatch time")
	}
	sdtB64, err := waitForRoleFrame(ctx, ch, "unet")
	if err != nil {
		return nil, err
	}

	// Hop 3: VAE — produces the PNG via sdcpp_done.
	if !s.dispatchSdcppRoleRoute(vae, sdcppRoleRouteParams{
		ReqID:     reqID,
		ModelPath: modelPathFor(vae),
		SDTB64:    sdtB64,
	}) {
		return nil, errors.New("vae rig offline at dispatch time")
	}
	return waitForDonePNG(ctx, ch)
}

// runSdcppFull dispatches a single `sdcpp_route` frame to a rig that
// advertises the "full" role and waits for sdcpp_done.
//
// NOTE: the "full" path doesn't have a dedicated dispatch helper yet —
// sdcpp_caps.go only exports dispatchSdcppRoleRoute. We emit the route
// frame inline here. The rig's adapter accepts `sdcpp_route` (no role
// field) as the full-pipeline alias.
func (s *server) runSdcppFull(ctx context.Context, reqID uint16,
	ch chan sdcppResultMsg, target sdcppRoleAgent, body sdcppRequestBody) ([]byte, error) {
	ac, ok := s.hub.findAgent(target.UserID, target.AgentID)
	if !ok {
		return nil, errors.New("full rig offline at dispatch time")
	}
	modelPath := target.ModelPath
	if modelPath == "" {
		modelPath = body.ModelPath
	}
	msg := map[string]any{
		"kind":       "sdcpp_route",
		"req_id":     int(reqID),
		"model_path": modelPath,
		"prompt":     body.Prompt,
		"width":      body.Width,
		"height":     body.Height,
		"steps":      body.Steps,
		"cfg":        body.CFG,
		"seed":       body.Seed,
	}
	if body.Negative != "" {
		msg["negative_prompt"] = body.Negative
	}
	if body.Sampler != "" {
		msg["sampler"] = body.Sampler
	}
	if body.Scheduler != "" {
		msg["scheduler"] = body.Scheduler
	}
	if body.ClipSkip != 0 {
		msg["clip_skip"] = body.ClipSkip
	}
	ac.send(msg)
	return waitForDonePNG(ctx, ch)
}

// waitForRoleFrame blocks until the given role's sdcpp_role_done frame
// arrives (returning its frame_b64), an error frame arrives, or ctx
// expires.  Progress events are silently consumed.
func waitForRoleFrame(ctx context.Context, ch chan sdcppResultMsg, role string) (string, error) {
	for {
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case m, ok := <-ch:
			if !ok {
				return "", errors.New("sdcpp result channel closed")
			}
			switch m.Kind {
			case sdcppResultRoleDone:
				if m.Role == role {
					return m.FrameB64, nil
				}
			case sdcppResultError:
				return "", errors.New("sdcpp " + role + ": " + m.ErrMsg)
			case sdcppResultDone:
				// Unexpected — full done before VAE in a role chain.
				// Could happen if the rig collapses TE+UNet+VAE into one
				// pass behind the role labels; accept the PNG and stop.
				return "", errors.New("sdcpp: unexpected done frame before " + role)
			default:
				// progress — keep waiting
			}
		}
	}
}

// waitForDonePNG blocks until sdcpp_done arrives and decodes the
// png_b64 payload.  Returns an error on sdcpp_error / ctx expiry.
func waitForDonePNG(ctx context.Context, ch chan sdcppResultMsg) ([]byte, error) {
	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case m, ok := <-ch:
			if !ok {
				return nil, errors.New("sdcpp result channel closed")
			}
			switch m.Kind {
			case sdcppResultDone:
				if m.PNGB64 == "" {
					return nil, errors.New("sdcpp: done with empty png_b64")
				}
				png, err := base64.StdEncoding.DecodeString(m.PNGB64)
				if err != nil {
					return nil, errors.New("sdcpp: decode png_b64: " + err.Error())
				}
				return png, nil
			case sdcppResultError:
				return nil, errors.New("sdcpp: " + m.ErrMsg)
			default:
				// progress / role_done — keep waiting
			}
		}
	}
}

// applyComfyParamsToSdcppBody mirrors applyComfyParamsToDPPBody — pulls
// steps/cfg/width/height/seed/negative/sampler/scheduler/sdcpp_model_path
// out of the comfy params JSON onto the request body.
func applyComfyParamsToSdcppBody(params string, b *sdcppRequestBody) {
	if params == "" || params == "null" || params == "{}" {
		return
	}
	var m map[string]any
	if err := json.Unmarshal([]byte(params), &m); err != nil {
		return
	}
	if v, ok := numericInt(m["steps"]); ok && v > 0 {
		b.Steps = v
	}
	if v, ok := numericFloat(m["cfg_scale"]); ok && v >= 0 {
		b.CFG = v
	} else if v, ok := numericFloat(m["cfg"]); ok && v >= 0 {
		b.CFG = v
	}
	if v, ok := numericInt(m["width"]); ok && v > 0 {
		b.Width = v
	}
	if v, ok := numericInt(m["height"]); ok && v > 0 {
		b.Height = v
	}
	if v, ok := numericInt64(m["seed"]); ok {
		b.Seed = v
	}
	if v, ok := m["negative_prompt"].(string); ok {
		b.Negative = v
	}
	if v, ok := m["sampler"].(string); ok {
		b.Sampler = v
	}
	if v, ok := m["scheduler"].(string); ok {
		b.Scheduler = v
	}
	if v, ok := m["sdcpp_model_path"].(string); ok {
		b.ModelPath = v
	}
	if v, ok := numericInt(m["clip_skip"]); ok {
		b.ClipSkip = v
	}
	if v, ok := m["cfg_split"].(bool); ok {
		b.CFGSplit = v
	}
}
