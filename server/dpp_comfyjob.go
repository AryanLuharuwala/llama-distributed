package main

// dpp_comfyjob.go — comfy-routed DPP execution.
//
// When handleComfyGenerate decides to use the distributed-pipeline backend
// (model is dpp-eligible AND pool has ≥2 online rigs), it creates a normal
// comfy_jobs row and dispatches runDPPComfyJob in a goroutine.  The
// goroutine drives executeDPPInference, persists the resulting PNG into
// the same on-disk layout the comfy path uses, and updates the job row to
// "done" (or "failed") so the existing UI / signed-URL machinery serves
// it transparently.
//
// This keeps the Studio UI dumb: it always polls /api/comfy/jobs/{id},
// regardless of whether the work ran on a single comfy rig or as a 3-stage
// DPP fan-out.

import (
	"context"
	"encoding/json"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// dppJobTimeout — diffusion runs across 3+ rigs can take a while.
// 10 minutes is generous; the worker emits a per-step PROG event so a UI
// can show progress long before this deadline.
const dppJobTimeout = 10 * time.Minute

// runDPPComfyJob drives a comfy-routed DPP run.  Mirrors runComfyJob's
// outer shape: take ownership of the comfy_jobs row, run the work, update
// the row to done/failed, emit a final status_changed observation, refund
// the rate-limit slot via the caller's defer.
func (s *server) runDPPComfyJob(ctx context.Context, userID, jobID int64, body dppRequestBody) {
	defer s.refundRequestSlot(userID)
	defer s.comfyJobs.finish(jobID)

	// Flip the row to running so the UI shows movement.
	_, _ = s.dbExec(
		`UPDATE comfy_jobs SET status='running', updated_at=? WHERE id=?`,
		nowUnix(), jobID,
	)

	runCtx, cancel := context.WithTimeout(ctx, dppJobTimeout)
	defer cancel()

	img, err := s.executeDPPInference(runCtx, userID, body)
	if err != nil {
		log.Printf("dpp-comfy job=%d failed: %v", jobID, err)
		s.failComfyJob(jobID, err.Error())
		return
	}
	if len(img) == 0 {
		s.failComfyJob(jobID, "empty image from dpp")
		return
	}

	// Persist into the canonical comfy-out tree: <comfyOutDir>/<jobID>/dpp.png
	dir := filepath.Join(s.cfg.comfyOutDir, strconv.FormatInt(jobID, 10))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		s.failComfyJob(jobID, "mkdir: "+err.Error())
		return
	}
	file := "dpp.png"
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

// failComfyJob is a small DRY helper used by runDPPComfyJob's error paths.
func (s *server) failComfyJob(jobID int64, msg string) {
	_, _ = s.dbExec(
		`UPDATE comfy_jobs SET status='failed', error=?, updated_at=? WHERE id=?`,
		msg, nowUnix(), jobID,
	)
}

// stampParamsBackend embeds `_backend` into the params JSON so a later
// /api/comfy/jobs/{id} reader can tell DPP runs apart from comfy runs
// without inspecting workflow state.  Returns the original string if the
// input isn't a JSON object.
func stampParamsBackend(params, backend string) string {
	var m map[string]any
	if err := json.Unmarshal([]byte(params), &m); err != nil {
		return params
	}
	if m == nil {
		m = map[string]any{}
	}
	m["_backend"] = backend
	b, err := json.Marshal(m)
	if err != nil {
		return params
	}
	return string(b)
}

// applyComfyParamsToDPPBody reads steps/cfg/width/height/seed/negative
// from a comfy-style params JSON object and copies them onto the
// dppRequestBody so the same comfy generate call shape works for both
// backends.  Missing or malformed values fall through to dppRequestBody
// defaults (which executeDPPInference further defaults again).
func applyComfyParamsToDPPBody(params string, b *dppRequestBody) {
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
	if v, ok := m["init_image_url"].(string); ok && v != "" && b.InitImageURL == "" {
		b.InitImageURL = v
	}
	if v, ok := numericFloat(m["strength"]); ok && v > 0 && b.Strength == 0 {
		b.Strength = v
	}
}

func numericInt(v any) (int, bool) {
	switch x := v.(type) {
	case float64:
		return int(x), true
	case int:
		return x, true
	case int64:
		return int(x), true
	case string:
		n, err := strconv.Atoi(x)
		if err == nil {
			return n, true
		}
	}
	return 0, false
}

func numericInt64(v any) (int64, bool) {
	switch x := v.(type) {
	case float64:
		return int64(x), true
	case int:
		return int64(x), true
	case int64:
		return x, true
	case string:
		n, err := strconv.ParseInt(x, 10, 64)
		if err == nil {
			return n, true
		}
	}
	return 0, false
}

// parseOAISize accepts OpenAI's "WxH" or "WIDTHxHEIGHT" form and returns
// (w, h).  Returns (0, 0) for an unparseable input; the runner falls back
// to backbone defaults in that case.
func parseOAISize(s string) (int, int) {
	if s == "" {
		return 0, 0
	}
	parts := strings.SplitN(strings.ToLower(s), "x", 2)
	if len(parts) != 2 {
		return 0, 0
	}
	w, err := strconv.Atoi(strings.TrimSpace(parts[0]))
	if err != nil || w <= 0 {
		return 0, 0
	}
	h, err := strconv.Atoi(strings.TrimSpace(parts[1]))
	if err != nil || h <= 0 {
		return 0, 0
	}
	return w, h
}

func ifElseStr(cond bool, a, b string) string {
	if cond {
		return a
	}
	return b
}

func numericFloat(v any) (float64, bool) {
	switch x := v.(type) {
	case float64:
		return x, true
	case int:
		return float64(x), true
	case int64:
		return float64(x), true
	case string:
		n, err := strconv.ParseFloat(x, 64)
		if err == nil {
			return n, true
		}
	}
	return 0, false
}
