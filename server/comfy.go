package main

// ComfyUI integration — image/video gen models routed via ComfyUI workflows
// running on rigs.  Parallel to the LLM (llama.cpp) path; selection happens at
// the model layer (model_kind = 'llm' | 'image' | 'video').
//
// On the agent side, a rig advertises `comfy_ok=true` in its hello frame if
// its install brought up ComfyUI (third_party/ComfyUI submodule).  The server
// routes image/video infer requests only to comfy-capable rigs.
//
// Pipeline:
//   1. POST /api/comfy/workflows      — register a JSON workflow (or upload .json)
//   2. POST /api/comfy/models         — register a checkpoint by name + path on rig
//   3. POST /api/comfy/generate       — kick off a generation; returns job_id
//   4. GET  /api/comfy/jobs/{id}      — poll job status / fetch output URL
//   5. POST /v1/images/generations    — OpenAI-compatible image gen surface
//
// Multi-rig: workflows can declare `n_rigs` and the dispatcher fans the
// queue across that many comfy-capable rigs (one workflow per rig — true
// workflow sharding requires ComfyUI's split-comfy fork, out of scope here).

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// ─── DB migration ──────────────────────────────────────────────────────────

func migrateComfy(db *sql.DB, d sqlDialect) error {
	stmts := []string{
		// Workflow templates: parameterised ComfyUI graphs the user can call
		// by name from the dashboard / API.  The JSON is the raw ComfyUI
		// "API JSON" export (the one obtained via "Save (API Format)").
		`CREATE TABLE IF NOT EXISTS comfy_workflows (
			id          INTEGER PRIMARY KEY AUTOINCREMENT,
			user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
			name        TEXT NOT NULL,
			kind        TEXT NOT NULL DEFAULT 'image',  -- image | video
			graph_json  TEXT NOT NULL,
			n_rigs      INTEGER NOT NULL DEFAULT 1,
			created_at  INTEGER NOT NULL,
			UNIQUE (user_id, name)
		)`,
		`CREATE INDEX IF NOT EXISTS idx_comfy_wf_user ON comfy_workflows(user_id)`,

		// Comfy models: a checkpoint that one or more rigs know how to load.
		// Unlike llama models we don't split — comfy checkpoints live whole
		// on the rig's local disk and are referenced by name.
		//
		// NOTE on visibility: this table is intentionally a *shared catalog*
		// (no user_id column).  Any logged-in user can see and reference
		// every registered model — the assumption is that checkpoints are
		// non-secret artifacts (SDXL base, public LoRAs, etc.).  This is
		// inconsistent with comfy_workflows which is per-user, but matches
		// the typical operator-curated catalog mental model.  If you want
		// per-user scoping, add `user_id INTEGER REFERENCES users(id)` and
		// update handleComfyListModels + comfyModelRegistered to filter.
		`CREATE TABLE IF NOT EXISTS comfy_models (
			id          INTEGER PRIMARY KEY AUTOINCREMENT,
			name        TEXT UNIQUE NOT NULL,
			kind        TEXT NOT NULL DEFAULT 'image',  -- image | video
			family      TEXT NOT NULL DEFAULT '',       -- sdxl | sd15 | flux | svd | ...
			hf_repo     TEXT NOT NULL DEFAULT '',
			hf_file     TEXT NOT NULL DEFAULT '',
			created_at  INTEGER NOT NULL
		)`,

		// Per-rig advertised capabilities.  Populated from the agent hello
		// frame's `comfy` block.  Cleared on agent disconnect.
		`CREATE TABLE IF NOT EXISTS comfy_caps (
			user_id   INTEGER NOT NULL,
			agent_id  TEXT NOT NULL,
			ok        INTEGER NOT NULL DEFAULT 0,
			version   TEXT NOT NULL DEFAULT '',
			models    TEXT NOT NULL DEFAULT '[]',  -- JSON array of model names
			updated_at INTEGER NOT NULL,
			PRIMARY KEY (user_id, agent_id)
		)`,

		// Generation jobs.  Output stored under DIST_COMFY_OUT_DIR (default
		// ./comfy-out); served via /comfy/out/{id}/{file} (HMAC-signed).
		`CREATE TABLE IF NOT EXISTS comfy_jobs (
			id           INTEGER PRIMARY KEY AUTOINCREMENT,
			user_id      INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
			pool_id      INTEGER REFERENCES pools(id) ON DELETE SET NULL,
			workflow_id  INTEGER REFERENCES comfy_workflows(id) ON DELETE SET NULL,
			model_id     INTEGER REFERENCES comfy_models(id) ON DELETE SET NULL,
			prompt       TEXT NOT NULL DEFAULT '',
			params_json  TEXT NOT NULL DEFAULT '{}',
			status       TEXT NOT NULL,             -- queued | running | done | failed | cancelled
			out_files    TEXT NOT NULL DEFAULT '[]',-- JSON array of relative paths
			error        TEXT NOT NULL DEFAULT '',
			agent_user_id INTEGER,
			agent_id      TEXT,
			created_at   INTEGER NOT NULL,
			updated_at   INTEGER NOT NULL
		)`,
		`CREATE INDEX IF NOT EXISTS idx_comfy_jobs_user ON comfy_jobs(user_id, created_at)`,
	}
	for _, s := range stmts {
		if _, err := db.Exec(d.RewriteDDL(s)); err != nil {
			return fmt.Errorf("comfy migrate: %w", err)
		}
	}
	return nil
}

// ─── In-process job tracker ────────────────────────────────────────────────

type comfyJobs struct {
	mu       sync.Mutex
	inflight map[int64]*comfyJob  // job_id -> live state
}

type comfyJob struct {
	cancel  context.CancelFunc
	done    chan struct{}
	updated atomic.Int64
}

func newComfyJobs() *comfyJobs {
	return &comfyJobs{inflight: make(map[int64]*comfyJob)}
}

func (j *comfyJobs) register(id int64, cancel context.CancelFunc) *comfyJob {
	cj := &comfyJob{cancel: cancel, done: make(chan struct{})}
	j.mu.Lock()
	j.inflight[id] = cj
	j.mu.Unlock()
	return cj
}

func (j *comfyJobs) finish(id int64) {
	j.mu.Lock()
	cj, ok := j.inflight[id]
	if ok {
		delete(j.inflight, id)
	}
	j.mu.Unlock()
	if ok {
		close(cj.done)
	}
}

func (j *comfyJobs) cancel(id int64) bool {
	j.mu.Lock()
	cj, ok := j.inflight[id]
	j.mu.Unlock()
	if !ok {
		return false
	}
	cj.cancel()
	return true
}

// ─── Pre-flight helpers shared by handleComfyGenerate + handleOAIImageGen ─

// comfyInflightForUser returns the count of jobs currently in queued/
// running/streaming state for uid.  Used by the inflight cap.
func (s *server) comfyInflightForUser(uid int64) int {
	var n int
	_ = s.dbQueryRow(
		`SELECT COUNT(*) FROM comfy_jobs
		 WHERE user_id = ? AND status IN ('queued','running','streaming')`,
		uid).Scan(&n)
	return n
}

// comfyInflightCap is the per-user cap on simultaneous comfy jobs.  The
// number is small on purpose — image jobs are GPU-heavy and there is no
// admission queue today, so letting one user fan out 50 jobs would
// effectively DoS the comfy fleet for everyone else.  Tunable via
// DIST_COMFY_USER_INFLIGHT if we discover an operator workload that
// genuinely needs more — but see audit notes: bump this only after
// adding per-pool admission control (P21-C3).
const comfyInflightCap = 4

// autoRegisterComfyFromHF idempotently inserts a comfy_models row for an
// HF repo that we recognise as a DPP-eligible diffusion backbone. Called
// from the HF-import paths so a successful (or short-circuited) import
// also satisfies the single-rig ComfyUI gate without requiring the user
// to hit POST /api/comfy/models manually.
//
// Returns true when a new row was inserted (false if the row already
// existed or the repo isn't a recognised DPP backbone — both are no-ops).
// Errors other than UNIQUE conflicts are returned so the caller can log
// them; they're non-fatal for the import.
func (s *server) autoRegisterComfyFromHF(repoID, hfFile string) (bool, error) {
	family := dppFamilyForModel(repoID)
	if family == "" {
		return false, nil
	}
	kind := "image"
	if b := dppBackboneForID(repoID); b != nil {
		kind = b.Kind
	}
	_, err := s.dbExec(
		`INSERT INTO comfy_models (name, kind, family, hf_repo, hf_file, created_at)
		 VALUES (?, ?, ?, ?, ?, ?)`,
		repoID, kind, family, repoID, hfFile, nowUnix(),
	)
	if err != nil {
		msg := err.Error()
		if strings.Contains(msg, "UNIQUE") ||
			strings.Contains(strings.ToLower(msg), "duplicate") {
			return false, nil
		}
		return false, err
	}
	return true, nil
}

// comfyModelRegistered returns true if name is in the comfy_models
// catalog.  Empty name returns true (the default workflow handles it).
func (s *server) comfyModelRegistered(name string) bool {
	if name == "" {
		return true
	}
	var ok int
	_ = s.dbQueryRow(
		`SELECT 1 FROM comfy_models WHERE name = ?`, name,
	).Scan(&ok)
	return ok == 1
}

// ─── Authorization helpers ────────────────────────────────────────────────

// authorizePoolForComfy returns true if uid is allowed to dispatch a comfy
// job onto poolID.  poolID == 0 means "no pool — use my own rigs", which is
// always allowed.  Otherwise the user must be a pool member OR the pool
// must be public.  Returns false and writes the HTTP error if not allowed
// (404 pool not found, 403 not a member).
//
// Why this exists: handleComfyGenerate and handleOAIImageGen both accept
// pool_id from the request body / params.  Without this check, a logged-in
// user could submit jobs against any pool's rigs by guessing pool IDs — a
// straight IDOR — and the macaroon-signed output URLs would come back to
// them while strangers paid the GPU cycles.  Audited 2026-05-25.
func (s *server) authorizePoolForComfy(w http.ResponseWriter, uid, poolID int64) bool {
	if poolID == 0 {
		return true
	}
	vis, _, ok := s.poolVisibility(poolID)
	if !ok {
		// 404 — not 403 — so we don't enumerate the existence of pools the
		// user isn't a member of via a status-code oracle.  An IDOR scan
		// gets 404 for both "doesn't exist" and "not your pool."
		writeErr(w, 404, "pool not found")
		return false
	}
	if vis == "public" {
		return true
	}
	if _, member := s.userIsMember(poolID, uid); member {
		return true
	}
	writeErr(w, 404, "pool not found")
	return false
}

// ─── Handlers ──────────────────────────────────────────────────────────────

type comfyWorkflowReq struct {
	Name      string          `json:"name"`
	Kind      string          `json:"kind"`        // image | video
	Graph     json.RawMessage `json:"graph_json"`
	GraphText string          `json:"graph_text"`  // alternative: pasted JSON as string
	NRigs     int             `json:"n_rigs"`
}

// POST /api/comfy/workflows
func (s *server) handleComfyRegisterWorkflow(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	var body comfyWorkflowReq
	// Workflows can be sizeable JSON graphs but are bounded — 2MB is well
	// above anything the official ComfyUI examples produce.
	if err := json.NewDecoder(io.LimitReader(r.Body, 2<<20)).Decode(&body); err != nil {
		writeErr(w, 400, "bad json: "+err.Error())
		return
	}
	if body.Name == "" {
		writeErr(w, 400, "name required")
		return
	}
	if len(body.Name) > 128 {
		writeErr(w, 400, "name too long (max 128)")
		return
	}
	if body.Kind == "" {
		body.Kind = "image"
	}
	if body.Kind != "image" && body.Kind != "video" {
		writeErr(w, 400, "kind must be image or video")
		return
	}
	if body.NRigs < 1 {
		body.NRigs = 1
	}
	if body.NRigs > 16 {
		writeErr(w, 400, "n_rigs too large (max 16)")
		return
	}
	graph := string(body.Graph)
	if graph == "" || graph == "null" {
		graph = body.GraphText
	}
	if len(graph) > 2<<20 {
		writeErr(w, 400, "graph_json too large (>2MB)")
		return
	}
	if !json.Valid([]byte(graph)) {
		writeErr(w, 400, "graph_json must be valid JSON (ComfyUI API format)")
		return
	}
	res, err := s.dbExec(
		`INSERT INTO comfy_workflows (user_id, name, kind, graph_json, n_rigs, created_at)
		 VALUES (?, ?, ?, ?, ?, ?)`,
		u.ID, body.Name, body.Kind, graph, body.NRigs, nowUnix(),
	)
	if err != nil {
		if strings.Contains(err.Error(), "UNIQUE") {
			writeErr(w, 409, "workflow name exists")
			return
		}
		writeErr(w, 500, err.Error())
		return
	}
	id, _ := res.LastInsertId()
	writeJSON(w, 200, map[string]any{"id": id, "name": body.Name, "kind": body.Kind})
}

// GET /api/comfy/workflows
func (s *server) handleComfyListWorkflows(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	rows, err := s.dbQuery(
		`SELECT id, name, kind, n_rigs, created_at FROM comfy_workflows
		 WHERE user_id = ? ORDER BY id DESC`, u.ID)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	defer rows.Close()
	type wfRow struct {
		ID        int64  `json:"id"`
		Name      string `json:"name"`
		Kind      string `json:"kind"`
		NRigs     int    `json:"n_rigs"`
		CreatedAt int64  `json:"created_at"`
	}
	var out []wfRow
	for rows.Next() {
		var x wfRow
		_ = rows.Scan(&x.ID, &x.Name, &x.Kind, &x.NRigs, &x.CreatedAt)
		out = append(out, x)
	}
	writeJSON(w, 200, map[string]any{"workflows": out})
}

// POST /api/comfy/models
type comfyRegisterModelReq struct {
	Name   string `json:"name"`
	Kind   string `json:"kind"`   // image | video
	Family string `json:"family"` // sdxl | flux | svd | ...
	HFRepo string `json:"hf_repo"`
	HFFile string `json:"hf_file"`
}

func (s *server) handleComfyRegisterModel(w http.ResponseWriter, r *http.Request) {
	if _, ok := s.userFromRequest(r); !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	var body comfyRegisterModelReq
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeErr(w, 400, "bad json: "+err.Error())
		return
	}
	if body.Name == "" {
		writeErr(w, 400, "name required")
		return
	}
	if body.Kind == "" {
		body.Kind = "image"
	}
	res, err := s.dbExec(
		`INSERT INTO comfy_models (name, kind, family, hf_repo, hf_file, created_at)
		 VALUES (?, ?, ?, ?, ?, ?)`,
		body.Name, body.Kind, body.Family, body.HFRepo, body.HFFile, nowUnix(),
	)
	if err != nil {
		if strings.Contains(err.Error(), "UNIQUE") {
			writeErr(w, 409, "model name exists")
			return
		}
		writeErr(w, 500, err.Error())
		return
	}
	id, _ := res.LastInsertId()
	writeJSON(w, 200, map[string]any{"id": id, "name": body.Name})
}

// GET /api/comfy/models
func (s *server) handleComfyListModels(w http.ResponseWriter, r *http.Request) {
	if _, ok := s.userFromRequest(r); !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	rows, err := s.dbQuery(
		`SELECT id, name, kind, family, hf_repo, hf_file, created_at
		 FROM comfy_models ORDER BY id DESC`)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	defer rows.Close()
	type mRow struct {
		ID        int64  `json:"id"`
		Name      string `json:"name"`
		Kind      string `json:"kind"`
		Family    string `json:"family"`
		HFRepo    string `json:"hf_repo"`
		HFFile    string `json:"hf_file"`
		CreatedAt int64  `json:"created_at"`
	}
	var out []mRow
	for rows.Next() {
		var x mRow
		_ = rows.Scan(&x.ID, &x.Name, &x.Kind, &x.Family, &x.HFRepo, &x.HFFile, &x.CreatedAt)
		out = append(out, x)
	}
	writeJSON(w, 200, map[string]any{"models": out})
}

// POST /api/comfy/generate
type comfyGenerateReq struct {
	WorkflowID int64           `json:"workflow_id"`
	WorkflowName string        `json:"workflow"`     // alt: name lookup
	ModelName  string          `json:"model"`
	PoolID     int64           `json:"pool_id"`
	Prompt     string          `json:"prompt"`
	Params     json.RawMessage `json:"params"` // free-form overrides ({"seed": ..., "steps": ...})
	// Backend explicitly selects execution path.
	//   ""      — auto: route to DPP when model is dpp-eligible AND pool has
	//             ≥2 online rigs; otherwise fall through to single-rig comfy.
	//   "comfy" — force single-rig ComfyUI (legacy / safe path).
	//   "dpp"   — force distributed pipeline.  Errors out if the model isn't
	//             dpp-eligible or the pool lacks the rigs.
	Backend string `json:"backend,omitempty"`
	// Optional DPP-only hints — passed through to executeDPPInference.
	UnetStages   int               `json:"unet_stages,omitempty"`
	Roles        map[string]string `json:"roles,omitempty"`
	UnetAgents   []string          `json:"unet_agents,omitempty"`
	InitImageURL string            `json:"init_image_url,omitempty"`
	Strength     float64           `json:"strength,omitempty"`
	// sd.cpp-only: absolute path on each rig to the checkpoint file the
	// dist-sdcpp-worker should load.  Required until the rig-side file
	// advertisement (Phase A2) lands.
	SdcppModelPath string `json:"sdcpp_model_path,omitempty"`
}

func (s *server) handleComfyGenerate(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	var body comfyGenerateReq
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&body); err != nil {
		writeErr(w, 400, "bad json: "+err.Error())
		return
	}
	if body.Prompt == "" {
		writeErr(w, 400, "prompt required")
		return
	}
	if len(body.Prompt) > 8192 {
		writeErr(w, 400, "prompt too long (>8KB)")
		return
	}
	// IDOR guard: body.PoolID comes straight from JSON.  Without this
	// check, any logged-in user could pin a stranger's GPUs by guessing
	// pool IDs.  See authorizePoolForComfy.
	if !s.authorizePoolForComfy(w, u.ID, body.PoolID) {
		return
	}
	// Bound concurrency: each user gets up to comfyInflightCap in-flight
	// comfy jobs.  Past that we reject so a slow rig can't be hammered
	// into oblivion from one browser tab.  Shared with handleOAIImageGen
	// — the audit found that the OAI path was previously skipping this
	// gate entirely (P21-CF3).
	if s.comfyInflightForUser(u.ID) >= comfyInflightCap {
		writeErr(w, 429, fmt.Sprintf("too many in-flight comfy jobs (limit %d); wait for one to finish", comfyInflightCap))
		return
	}
	// Charge a request slot against the rolling-minute cap so a tight
	// loop of /api/comfy/generate calls is subject to the same global
	// admission control as /v1/chat and /api/infer.  If we never spawn
	// runComfyJob (graph build / workflow lookup failure), the deferred
	// refund returns the slot.
	admitted, policy, snap := s.reserveRequestSlot(u.ID)
	if !admitted {
		writeJSON(w, 429, map[string]any{
			"error":  "rate limit exceeded",
			"policy": policy,
			"usage":  snap,
		})
		return
	}
	jobAdmitted := false
	defer func() {
		if !jobAdmitted {
			s.refundRequestSlot(u.ID)
		}
	}()

	// Resolve workflow (id or name).
	wfID := body.WorkflowID
	var graph string
	var wfKind string
	var nRigs int = 1
	if wfID == 0 && body.WorkflowName != "" {
		err := s.dbQueryRow(
			`SELECT id, graph_json, kind, n_rigs FROM comfy_workflows
			 WHERE user_id = ? AND name = ?`, u.ID, body.WorkflowName,
		).Scan(&wfID, &graph, &wfKind, &nRigs)
		if err != nil {
			writeErr(w, 404, "workflow not found: "+body.WorkflowName)
			return
		}
	} else if wfID != 0 {
		err := s.dbQueryRow(
			`SELECT graph_json, kind, n_rigs FROM comfy_workflows
			 WHERE id = ? AND user_id = ?`, wfID, u.ID,
		).Scan(&graph, &wfKind, &nRigs)
		if err != nil {
			writeErr(w, 404, "workflow not found")
			return
		}
	}
	// No workflow → use the built-in default (text2image sdxl).
	if wfID == 0 {
		graph = defaultComfyWorkflowJSON
		wfKind = "image"
	}

	// ── Backend routing ────────────────────────────────────────────────
	//
	// Decide between single-rig ComfyUI (legacy), DPP (text_encoder →
	// UNet[stages] → VAE staged across rigs), and sd.cpp (C++ stable-
	// diffusion.cpp role chain or single "full" rig).  Decision tree:
	//
	//   backend=="comfy" → single-rig comfy
	//   backend=="dpp"   → DPP (error if not feasible)
	//   backend=="sdcpp" → sd.cpp (error if not feasible)
	//   backend==""      → DPP if model is dpp-eligible AND pool has
	//                      ≥2 online rigs; else sd.cpp if model is
	//                      sd.cpp-eligible AND user has ≥1 sd.cpp rig
	//                      online; else single-rig comfy.
	useDPP := false
	useSdcpp := false
	switch body.Backend {
	case "dpp":
		if !isDPPEligible(body.ModelName) {
			writeErr(w, 400, "backend=dpp requested but model is not DPP-eligible: "+body.ModelName)
			return
		}
		if n := s.countDPPRigsInPool(body.PoolID); n < 1 {
			writeErr(w, 503, "backend=dpp requested but no online rigs in pool")
			return
		}
		useDPP = true
	case "sdcpp":
		// Explicit sd.cpp requests can use either a heuristic-eligible HF
		// id (e.g. "stabilityai/stable-diffusion-xl-base-1.0") OR a
		// rig-advertised file name (e.g. "sdxl-base.safetensors").  We
		// trust the latter when the rig advertised the file in
		// sdcpp_models — the eligibility check is a UI hint, not a
		// security boundary.
		if !isSdcppEligible(body.ModelName) && !s.sdcppRigAdvertisesModel(u.ID, body.ModelName) {
			writeErr(w, 400, "backend=sdcpp requested but model is neither sd.cpp-eligible nor advertised by any rig: "+body.ModelName)
			return
		}
		if s.countSdcppRigsForUser(u.ID) < 1 {
			writeErr(w, 503, "backend=sdcpp requested but no sd.cpp-capable rigs online")
			return
		}
		useSdcpp = true
	case "", "auto":
		if isDPPEligible(body.ModelName) && s.countDPPRigsInPool(body.PoolID) >= 2 {
			useDPP = true
		} else if isSdcppEligible(body.ModelName) && s.countSdcppRigsForUser(u.ID) >= 1 {
			useSdcpp = true
		}
	case "comfy":
		useDPP = false
	default:
		writeErr(w, 400, "unknown backend: "+body.Backend)
		return
	}

	// Single-rig comfy path requires comfy-registry membership; DPP +
	// sd.cpp resolve the model through their own caches (HF / on-rig
	// path), so we only enforce registration once routing has actually
	// picked comfy.
	if !useDPP && !useSdcpp {
		if !s.comfyModelRegistered(body.ModelName) {
			writeErr(w, 404, "model not registered: "+body.ModelName)
			return
		}
	}

	params := string(body.Params)
	if params == "" || params == "null" {
		params = "{}"
	}

	// Stamp a marker into params_json so the UI / job-detail endpoint can
	// surface which path actually ran (handy for the live-pipeline graph).
	paramsForDB := params
	switch {
	case useDPP:
		paramsForDB = stampParamsBackend(params, "dpp")
	case useSdcpp:
		paramsForDB = stampParamsBackend(params, "sdcpp")
	default:
		paramsForDB = stampParamsBackend(params, "comfy")
	}

	res, err := s.dbExec(
		`INSERT INTO comfy_jobs
		 (user_id, pool_id, workflow_id, prompt, params_json, status, created_at, updated_at)
		 VALUES (?, ?, ?, ?, ?, 'queued', ?, ?)`,
		u.ID, sql.NullInt64{Int64: body.PoolID, Valid: body.PoolID != 0},
		sql.NullInt64{Int64: wfID, Valid: wfID != 0},
		body.Prompt, paramsForDB, nowUnix(), nowUnix(),
	)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	jobID, _ := res.LastInsertId()

	jobCtx, cancel := context.WithCancel(context.Background())
	s.comfyJobs.register(jobID, cancel)
	jobAdmitted = true
	if useDPP {
		dppBody := dppRequestBody{
			PoolID:       body.PoolID,
			Model:        body.ModelName,
			Prompt:       body.Prompt,
			Roles:        body.Roles,
			UnetStages:   body.UnetStages,
			UnetAgents:   body.UnetAgents,
			InitImageURL: body.InitImageURL,
			Strength:     body.Strength,
		}
		applyComfyParamsToDPPBody(params, &dppBody)
		go s.runDPPComfyJob(jobCtx, u.ID, jobID, dppBody)
		writeJSON(w, 200, map[string]any{"job_id": jobID, "status": "queued", "kind": "image", "backend": "dpp"})
		return
	}
	if useSdcpp {
		sdBody := sdcppRequestBody{
			PoolID:    body.PoolID,
			Model:     body.ModelName,
			ModelPath: body.SdcppModelPath,
			Prompt:    body.Prompt,
		}
		applyComfyParamsToSdcppBody(params, &sdBody)
		go s.runSdcppComfyJob(jobCtx, u.ID, jobID, sdBody)
		writeJSON(w, 200, map[string]any{"job_id": jobID, "status": "queued", "kind": "image", "backend": "sdcpp"})
		return
	}
	go s.runComfyJob(jobCtx, u.ID, jobID, body.PoolID, graph, wfKind, body.Prompt, params, body.ModelName, nRigs)

	writeJSON(w, 200, map[string]any{"job_id": jobID, "status": "queued", "kind": wfKind, "backend": "comfy"})
}

// GET /api/comfy/jobs/{id}
func (s *server) handleComfyJobDetail(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	id, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil {
		writeErr(w, 400, "bad id")
		return
	}
	var (
		uid       int64
		status    string
		outFiles  string
		errMsg    string
		prompt    string
		createdAt int64
		updatedAt int64
	)
	err = s.dbQueryRow(
		`SELECT user_id, status, out_files, error, prompt, created_at, updated_at
		 FROM comfy_jobs WHERE id = ?`, id,
	).Scan(&uid, &status, &outFiles, &errMsg, &prompt, &createdAt, &updatedAt)
	if err == sql.ErrNoRows {
		writeErr(w, 404, "no such job")
		return
	}
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	if uid != u.ID {
		writeErr(w, 403, "not your job")
		return
	}
	var files []string
	_ = json.Unmarshal([]byte(outFiles), &files)
	// Sign per-file URLs so they're shareable.
	signed := make([]string, 0, len(files))
	for _, f := range files {
		signed = append(signed, s.signComfyOutputURL(u.ID, id, f, 15*time.Minute))
	}
	writeJSON(w, 200, map[string]any{
		"id":         id,
		"status":     status,
		"error":      errMsg,
		"prompt":     prompt,
		"out_urls":   signed,
		"created_at": createdAt,
		"updated_at": updatedAt,
	})
}

// POST /api/comfy/jobs/{id}/cancel
func (s *server) handleComfyJobCancel(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	id, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil {
		writeErr(w, 400, "bad id")
		return
	}
	var uid int64
	if err := s.dbQueryRow(`SELECT user_id FROM comfy_jobs WHERE id = ?`, id).Scan(&uid); err != nil {
		writeErr(w, 404, "no such job")
		return
	}
	if uid != u.ID {
		writeErr(w, 403, "not your job")
		return
	}
	// Two-step cancel (audit P21-CF10c):
	//
	//   1. UPDATE the DB row so a runComfyJob goroutine that hasn't yet
	//      finished its register() — or is running on a different
	//      replica — still sees the cancellation when it checks the row.
	//   2. Call the in-process cancel so a live goroutine on this
	//      replica trips ctx.Done() immediately.
	//
	// We guard the UPDATE with the status filter so we don't downgrade a
	// 'done' or 'failed' row to 'cancelled' — that would corrupt the
	// audit trail.
	_, _ = s.dbExec(
		`UPDATE comfy_jobs
		   SET status = 'cancelled',
		       error  = 'cancelled by user',
		       updated_at = ?
		 WHERE id = ? AND status IN ('queued','running','streaming')`,
		nowUnix(), id,
	)
	s.comfyJobs.cancel(id)
	writeJSON(w, 200, map[string]any{"ok": true})
}

// GET /api/comfy/jobs
func (s *server) handleComfyListJobs(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	rows, err := s.dbQuery(
		`SELECT id, status, prompt, out_files, error, created_at, updated_at
		 FROM comfy_jobs WHERE user_id = ? ORDER BY id DESC LIMIT 200`, u.ID)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	defer rows.Close()
	type jobRow struct {
		ID         int64    `json:"id"`
		Status     string   `json:"status"`
		Prompt     string   `json:"prompt"`
		OutURLs    []string `json:"out_urls"`
		Error      string   `json:"error"`
		CreatedAt  int64    `json:"created_at"`
		UpdatedAt  int64    `json:"updated_at"`
	}
	var out []jobRow
	for rows.Next() {
		var j jobRow
		var outFiles string
		_ = rows.Scan(&j.ID, &j.Status, &j.Prompt, &outFiles, &j.Error, &j.CreatedAt, &j.UpdatedAt)
		var files []string
		_ = json.Unmarshal([]byte(outFiles), &files)
		for _, f := range files {
			j.OutURLs = append(j.OutURLs, s.signComfyOutputURL(u.ID, j.ID, f, 15*time.Minute))
		}
		out = append(out, j)
	}
	writeJSON(w, 200, map[string]any{"jobs": out})
}

// GET /comfy/out/{id}/{file}?cap=<macaroon>
//
// Capability-signed output retrieval.  Three URL formats are accepted:
//
//   P7 (current):  ?cap=<macaroon>   — caveats: path=comfy-out, uid=,
//                  job=, file=, exp=.  Still requires an authenticated
//                  session whose user_id matches the uid caveat in the
//                  cap — leaking the URL alone is not enough.
//
//   v2 (legacy):   ?v=2&uid=<uid>&exp=...&sig=hmac("comfyv2/<uid>/<id>/<file>@<exp>")
//                  Accepted within mintHMACGraceWindow of server start.
//
//   v1 (legacy):   ?exp=...&sig=hmac("comfy/<id>/<file>@<exp>")
//                  Accepted only within comfyV1GraceWindow of server
//                  startup, to let pre-existing dashboard tabs finish
//                  resolving any URLs minted before this code shipped.
//                  After the window: 401 (bad signature).
func (s *server) handleComfyOutput(w http.ResponseWriter, r *http.Request) {
	id, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil {
		writeErr(w, 400, "bad id")
		return
	}
	file := r.PathValue("file")
	if !isSafeOutputFile(file) {
		writeErr(w, 400, "bad file")
		return
	}
	q := r.URL.Query()

	// P7: macaroon-capability path.  When `cap=` is present we verify
	// the token (which carries uid + job + file + exp as caveats) and
	// then enforce the same session-uid binding that v2 HMAC did.
	if tok := q.Get("cap"); tok != "" {
		// We don't know which uid the cap was minted for until we parse
		// it; macaroon verify needs the expected uid up-front, so the
		// resolver hits the authenticated session first and matches the
		// uid into the cap check.
		u, ok := s.userFromRequest(r)
		if !ok {
			if bt := bearerFromRequest(r); bt != "" {
				if au, aok := s.userFromAPIKey(bt); aok {
					u, ok = au, true
				}
			}
		}
		if !ok {
			writeErr(w, 401, "auth required")
			return
		}
		if err := s.verifyComfyOutputCap(tok, u.ID, id, file); err != nil {
			writeErr(w, 403, "bad capability")
			return
		}
	} else {
		exp, _ := strconv.ParseInt(q.Get("exp"), 10, 64)
		sig := q.Get("sig")
		if exp == 0 || sig == "" {
			writeErr(w, 401, "missing signature")
			return
		}
		if time.Since(s.startedAt) > mintHMACGraceWindow {
			writeErr(w, 401, "legacy url no longer accepted; please refresh")
			return
		}
		if nowUnix() > exp {
			writeErr(w, 401, "url expired")
			return
		}

		version := q.Get("v")
		if version == "2" {
			uidStr := q.Get("uid")
			uid, err := strconv.ParseInt(uidStr, 10, 64)
			if err != nil || uid <= 0 {
				writeErr(w, 401, "bad uid")
				return
			}
			want := s.signComfyOutputV2(uid, id, file, exp)
			if !hmac.Equal([]byte(want), []byte(sig)) {
				writeErr(w, 401, "bad signature")
				return
			}
			u, ok := s.userFromRequest(r)
			if !ok {
				if bt := bearerFromRequest(r); bt != "" {
					if au, aok := s.userFromAPIKey(bt); aok {
						u, ok = au, true
					}
				}
			}
			if !ok || u.ID != uid {
				writeErr(w, 403, "url is bound to another user")
				return
			}
		} else {
			if time.Since(s.startedAt) > comfyV1GraceWindow {
				s.comfyLegacyURLRejected.Add(1)
				writeErr(w, 401, "legacy url no longer accepted; please refresh")
				return
			}
			want := s.signComfyOutputV1(id, file, exp)
			if !hmac.Equal([]byte(want), []byte(sig)) {
				writeErr(w, 401, "bad signature")
				return
			}
		}
	}
	// Resolve and confine the file path to the configured output root —
	// `isSafeOutputFile` already rejected slashes, but compute the clean
	// absolute paths and re-check the prefix as a defense-in-depth measure
	// in case the output root is itself a symlink.
	outRoot, _ := filepath.Abs(s.cfg.comfyOutDir)
	full, _ := filepath.Abs(filepath.Join(s.cfg.comfyOutDir, strconv.FormatInt(id, 10), file))
	if !strings.HasPrefix(full, outRoot+string(os.PathSeparator)) && full != outRoot {
		writeErr(w, 400, "bad path")
		return
	}
	if ct := mime.TypeByExtension(filepath.Ext(file)); ct != "" {
		w.Header().Set("Content-Type", ct)
	}
	http.ServeFile(w, r, full)
}

// POST /v1/images/generations   (OpenAI-compatible)
//
// Body: {prompt, n?, size?, model?}.  We map model→registered comfy_models
// row, run the default workflow with size+seed overrides, and return URLs.
func (s *server) handleOAIImageGen(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		// Allow API-key auth path used by /v1/*.
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
	var body struct {
		Prompt   string `json:"prompt"`
		Model    string `json:"model"`
		Size     string `json:"size"`
		N        int    `json:"n"`
		Quality  string `json:"quality,omitempty"`  // OAI-compat, unused
		Style    string `json:"style,omitempty"`    // OAI-compat, unused
		User     string `json:"user,omitempty"`     // OAI-compat, unused
		Response string `json:"response_format,omitempty"`
		// Extras beyond OpenAI's schema — passed through to the runner.
		Steps          int               `json:"steps,omitempty"`
		CFGScale       float64           `json:"cfg_scale,omitempty"`
		Seed           int64             `json:"seed,omitempty"`
		Negative       string            `json:"negative_prompt,omitempty"`
		PoolID         int64             `json:"pool_id,omitempty"`
		Backend        string            `json:"backend,omitempty"` // "", "comfy", "dpp", "auto"
		UnetStages     int               `json:"unet_stages,omitempty"`
		Roles          map[string]string `json:"roles,omitempty"`
		UnetAgents     []string          `json:"unet_agents,omitempty"`
		InitImageURL   string            `json:"init_image_url,omitempty"`
		Strength       float64           `json:"strength,omitempty"`
		SdcppModelPath string            `json:"sdcpp_model_path,omitempty"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&body); err != nil {
		writeErr(w, 400, "bad json: "+err.Error())
		return
	}
	if body.Prompt == "" {
		writeErr(w, 400, "prompt required")
		return
	}
	if len(body.Prompt) > 8192 {
		writeErr(w, 400, "prompt too long (>8KB)")
		return
	}
	if body.N > 8 {
		writeErr(w, 400, "n too large (max 8)")
		return
	}
	if body.N == 0 {
		body.N = 1
	}
	// IDOR + visibility guard.  Body.PoolID is a free-form JSON int; without
	// this an API-key caller could pin a stranger's GPUs by guessing pools.
	if body.PoolID != 0 {
		if !s.authorizePoolForComfy(w, u.ID, body.PoolID) {
			return
		}
	}
	// In-flight cap.  Previously this handler only enforced the rolling-
	// minute rate limit, which let a steady stream of API-key calls pin
	// every comfy rig (audit P21-CF3).  Cap on simultaneous jobs is the
	// real backpressure here.
	if s.comfyInflightForUser(u.ID) >= comfyInflightCap {
		oaiErr(w, 429, "rate_limit_exceeded",
			fmt.Sprintf("too many in-flight comfy jobs (limit %d); wait for one to finish", comfyInflightCap))
		return
	}
	// Backend routing — same rules as handleComfyGenerate.
	useDPP := false
	useSdcpp := false
	switch body.Backend {
	case "dpp":
		if !isDPPEligible(body.Model) {
			oaiErr(w, 400, "invalid_backend", "backend=dpp requires a dpp-eligible model")
			return
		}
		if n := s.countDPPRigsInPool(body.PoolID); n < 1 {
			oaiErr(w, 503, "no_rigs", "no online rigs in pool")
			return
		}
		useDPP = true
	case "sdcpp":
		if !isSdcppEligible(body.Model) && !s.sdcppRigAdvertisesModel(u.ID, body.Model) {
			oaiErr(w, 400, "invalid_backend", "backend=sdcpp requires an sd.cpp-eligible model or rig-advertised file name")
			return
		}
		if s.countSdcppRigsForUser(u.ID) < 1 {
			oaiErr(w, 503, "no_rigs", "no sd.cpp-capable rigs online")
			return
		}
		useSdcpp = true
	case "", "auto":
		if isDPPEligible(body.Model) && s.countDPPRigsInPool(body.PoolID) >= 2 {
			useDPP = true
		} else if isSdcppEligible(body.Model) && s.countSdcppRigsForUser(u.ID) >= 1 {
			useSdcpp = true
		}
	case "comfy":
		useDPP = false
	default:
		oaiErr(w, 400, "invalid_backend", "unknown backend: "+body.Backend)
		return
	}
	// Single-rig comfy path requires the model to be registered.  DPP +
	// sd.cpp resolve via their own caches, so we skip the registry check
	// for those routes.
	if !useDPP && !useSdcpp {
		if !s.comfyModelRegistered(body.Model) {
			oaiErr(w, 404, "model_not_found", "model not registered: "+body.Model)
			return
		}
	}
	// Same rolling-minute admission as /api/comfy/generate.  Without it,
	// API-key callers can pin every comfy rig with a tight curl loop.
	admitted, _, _ := s.reserveRequestSlot(u.ID)
	if !admitted {
		writeErr(w, 429, "rate limit exceeded")
		return
	}
	jobAdmitted := false
	defer func() {
		if !jobAdmitted {
			s.refundRequestSlot(u.ID)
		}
	}()
	// Parse OpenAI's "size":"WxH" into width/height — only used when caller
	// didn't supply explicit width/height extras.
	w_, h_ := parseOAISize(body.Size)
	params := map[string]any{
		"size":            body.Size,
		"n":               body.N,
		"steps":           body.Steps,
		"cfg_scale":       body.CFGScale,
		"seed":            body.Seed,
		"negative_prompt": body.Negative,
		"width":           w_,
		"height":          h_,
		"_backend": func() string {
			switch {
			case useDPP:
				return "dpp"
			case useSdcpp:
				return "sdcpp"
			default:
				return "comfy"
			}
		}(),
	}
	if body.InitImageURL != "" {
		params["init_image_url"] = body.InitImageURL
	}
	if body.Strength > 0 {
		params["strength"] = body.Strength
	}
	if body.SdcppModelPath != "" {
		params["sdcpp_model_path"] = body.SdcppModelPath
	}
	pj, _ := json.Marshal(params)
	res, err := s.dbExec(
		`INSERT INTO comfy_jobs
		 (user_id, pool_id, prompt, params_json, status, created_at, updated_at)
		 VALUES (?, ?, ?, ?, 'queued', ?, ?)`,
		u.ID, sql.NullInt64{Int64: body.PoolID, Valid: body.PoolID != 0},
		body.Prompt, string(pj), nowUnix(), nowUnix(),
	)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	jobID, _ := res.LastInsertId()
	jobCtx, cancel := context.WithCancel(context.Background())
	s.comfyJobs.register(jobID, cancel)
	jobAdmitted = true
	switch {
	case useDPP:
		dppBody := dppRequestBody{
			PoolID:       body.PoolID,
			Model:        body.Model,
			Prompt:       body.Prompt,
			Roles:        body.Roles,
			Steps:        body.Steps,
			CFG:          body.CFGScale,
			Width:        w_,
			Height:       h_,
			Seed:         body.Seed,
			UnetStages:   body.UnetStages,
			UnetAgents:   body.UnetAgents,
			InitImageURL: body.InitImageURL,
			Strength:     body.Strength,
		}
		go s.runDPPComfyJob(jobCtx, u.ID, jobID, dppBody)
	case useSdcpp:
		sdBody := sdcppRequestBody{
			PoolID:    body.PoolID,
			Model:     body.Model,
			ModelPath: body.SdcppModelPath,
			Prompt:    body.Prompt,
			Negative:  body.Negative,
			Steps:     body.Steps,
			CFG:       body.CFGScale,
			Width:     w_,
			Height:    h_,
			Seed:      body.Seed,
		}
		go s.runSdcppComfyJob(jobCtx, u.ID, jobID, sdBody)
	default:
		go s.runComfyJob(jobCtx, u.ID, jobID, body.PoolID, defaultComfyWorkflowJSON, "image", body.Prompt, string(pj), body.Model, 1)
	}

	// Block until the job completes or 5 min elapse — OpenAI clients
	// expect a synchronous response.  60s was the original bound and
	// almost every SDXL render at 20 steps blew past it (P21-CF10b).  For
	// renders that need longer than 5 min the dashboard's async API
	// (/api/comfy/jobs/{id}) is still the right path; the 504 body
	// includes the job_id so the SDK caller can poll.  Operator tunable
	// via DIST_COMFY_OAI_DEADLINE_SECS if 5 min is the wrong number for
	// a given fleet.
	oaiDeadline := 5 * time.Minute
	if s.cfg.comfyOAIDeadlineSecs > 0 {
		oaiDeadline = time.Duration(s.cfg.comfyOAIDeadlineSecs) * time.Second
	}
	deadline := time.After(oaiDeadline)
	tick := time.NewTicker(500 * time.Millisecond)
	defer tick.Stop()
	for {
		select {
		case <-deadline:
			writeJSON(w, 504, map[string]any{
				"error": "render timeout — poll /api/comfy/jobs/" + strconv.FormatInt(jobID, 10),
				"job_id": jobID,
			})
			return
		case <-tick.C:
			var status, outFiles, errMsg string
			err := s.dbQueryRow(
				`SELECT status, out_files, error FROM comfy_jobs WHERE id = ?`, jobID,
			).Scan(&status, &outFiles, &errMsg)
			if err != nil {
				writeErr(w, 500, err.Error())
				return
			}
			if status == "failed" {
				writeErr(w, 500, errMsg)
				return
			}
			if status == "done" {
				var files []string
				_ = json.Unmarshal([]byte(outFiles), &files)
				out := make([]map[string]string, 0, len(files))
				for _, f := range files {
					out = append(out, map[string]string{"url": s.signComfyOutputURL(u.ID, jobID, f, 1*time.Hour)})
				}
				writeJSON(w, 200, map[string]any{
					"created": nowUnix(),
					"data":    out,
				})
				return
			}
		}
	}
}

// ─── Runner ────────────────────────────────────────────────────────────────

// runComfyJob dispatches a job to a comfy-capable rig in the chosen pool
// (or any of the user's online rigs if no pool given), streams the workflow
// over the existing /ws/agent channel, waits for completion, and ingests
// the produced files back to the coordinator's comfy-out dir.
//
// If nRigs > 1 we fan out the same workflow to multiple rigs (each with a
// fresh seed) — used for batch image gen.
func (s *server) runComfyJob(
	ctx context.Context,
	uid, jobID, poolID int64,
	graphJSON, kind, prompt, paramsJSON, modelName string,
	nRigs int,
) {
	defer s.comfyJobs.finish(jobID)

	// Pre-check: did handleComfyJobCancel beat us to the start?  The
	// register-before-INSERT race is narrow (caller can't see job_id
	// until after this handler returns) but the row could already be
	// 'cancelled' if a different replica wrote it.  Bail out before we
	// touch any rigs.  Audit P21-CF10c.
	var preStatus string
	_ = s.dbQueryRow(`SELECT status FROM comfy_jobs WHERE id = ?`, jobID).Scan(&preStatus)
	if preStatus == "cancelled" {
		return
	}
	s.comfySetStatus(jobID, uid, "running", "")

	// Pick rig(s).
	rigs := s.pickComfyRigs(uid, poolID, nRigs)
	if len(rigs) == 0 {
		s.comfyFail(jobID, uid, "no comfy-capable rig online — install ComfyUI on at least one rig (DIST_WITH_COMFYUI=1)")
		return
	}

	// Output dir on the coordinator side — we collect renders here.
	outDir := filepath.Join(s.cfg.comfyOutDir, strconv.FormatInt(jobID, 10))
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		s.comfyFail(jobID, uid, "mkdir out: "+err.Error())
		return
	}

	// Resolve the base seed once so the user-supplied params.seed (if any)
	// is honoured, and so that fan-out is reproducible from (job_id, seed).
	// Each rig then gets a *distinct* derived seed — without this, a
	// user asking for n_rigs=4 would get four identical images back
	// (the original bug was that substituteComfyParams ran once before
	// the fan-out loop, baking the same seed into every rig's graph).
	baseSeed, seedSrc := resolveBaseSeed(paramsJSON, jobID)
	_ = seedSrc // reserved for future logging

	// Dispatch over WS to each chosen rig.
	type rigResult struct {
		agent  *agentConn
		files  []string
		err    error
	}
	resultsCh := make(chan rigResult, len(rigs))
	for i, ag := range rigs {
		ag := ag
		// Derive a per-rig seed.  XOR-mix the base seed with the rig
		// index so seeds are spread across the int63 space but stay
		// deterministic for a given (job, rig_idx).
		rigSeed := baseSeed ^ int64(uint64(i+1)*0x9E3779B97F4A7C15)
		if rigSeed < 0 {
			rigSeed = -rigSeed
		}
		rigGraph, err := substituteComfyParamsWithSeed(graphJSON, prompt, paramsJSON, modelName, rigSeed)
		if err != nil {
			// One rig's graph failed to materialise — drop the rig with a
			// synthetic result so the collector loop still terminates.
			resultsCh <- rigResult{agent: ag, err: fmt.Errorf("rig %d graph build: %w", i, err)}
			continue
		}
		go func() {
			files, err := s.dispatchComfyToAgent(ctx, ag, jobID, kind, rigGraph)
			resultsCh <- rigResult{agent: ag, files: files, err: err}
		}()
	}

	var allFiles []string
	var firstErr error
	collected := 0
	for collected < len(rigs) {
		select {
		case <-ctx.Done():
			s.comfySetStatus(jobID, uid, "cancelled", "cancelled by user")
			return
		case r := <-resultsCh:
			collected++
			if r.err != nil && firstErr == nil {
				firstErr = r.err
			}
			allFiles = append(allFiles, r.files...)
			// Record the agent that did the most work for credit attribution.
			if r.agent != nil {
				s.recordComfyAgent(jobID, r.agent)
			}
		}
	}
	if firstErr != nil && len(allFiles) == 0 {
		s.comfyFail(jobID, uid, firstErr.Error())
		return
	}

	// Persist file list.
	fb, _ := json.Marshal(allFiles)
	_, _ = s.dbExec(
		`UPDATE comfy_jobs SET status = 'done', out_files = ?, updated_at = ? WHERE id = ?`,
		string(fb), nowUnix(), jobID,
	)
	s.hub.broadcastToUser(uid, "comfy_progress", map[string]any{
		"job_id":   jobID,
		"status":   "done",
		"out_urls": s.signedURLsFor(uid, jobID, allFiles, 15*time.Minute),
	})
}

// dispatchComfyToAgent sends an `agent_message` of kind `comfy_run` over the
// rig's WS, then waits for `comfy_result` frames carrying base64-encoded
// outputs.  Files are written under outDir as the frames arrive.
//
// The agent-side implementation lives in src/node_agent.cpp (handled by the
// existing comfy adapter); when DIST_WITH_COMFYUI is off, the agent simply
// rejects the message and the dispatcher falls through to the next rig.
func (s *server) dispatchComfyToAgent(ctx context.Context, a *agentConn, jobID int64, kind, graph string) ([]string, error) {
	if a == nil {
		return nil, errors.New("nil agent")
	}
	// Build the request frame.
	req := map[string]any{
		"kind":     "comfy_run",
		"job_id":   jobID,
		"workflow": kind,    // image | video
		"graph":    graph,   // the JSON workflow to execute
	}
	// Subscribe to comfy_result frames for this job before sending so we
	// don't race the rig.
	resultCh := a.subscribeComfy(jobID)
	defer a.unsubscribeComfy(jobID)

	// Fail fast if the rig is gone or its outbound queue is wedged
	// (audit P21-CF8 — previously discarded send error meant the
	// dispatcher would sit on resultCh for the full 10-min deadline
	// before noticing).
	if !a.trySend(req) {
		return nil, errors.New("comfy: rig outbound queue full or closed; dropping job")
	}

	outDir := filepath.Join(s.cfg.comfyOutDir, strconv.FormatInt(jobID, 10))
	var written []string
	var bytesWritten int64
	// Size + count caps to keep a malicious or buggy rig from writing
	// the disk to death (audit P21-CF5).  Values come from config so
	// operators can override per fleet.  Sensible defaults: 64 MiB per
	// file, 1 GiB per job, 64 files per job.
	maxFile := s.cfg.comfyMaxFileBytes
	maxJob := s.cfg.comfyMaxJobBytes
	maxFiles := s.cfg.comfyMaxJobFiles
	deadline := time.NewTimer(10 * time.Minute)
	defer deadline.Stop()
	// On any non-completion exit, signal the rig so it can free GPU
	// memory instead of finishing a render the coordinator no longer
	// cares about.  Best-effort: a slammed outCh just drops the frame.
	sendAbort := func() {
		a.send(map[string]any{"kind": "comfy_cancel", "job_id": jobID})
	}
	for {
		select {
		case <-ctx.Done():
			sendAbort()
			return written, ctx.Err()
		case <-deadline.C:
			sendAbort()
			return written, errors.New("comfy: rig timed out (10m)")
		case msg, ok := <-resultCh:
			if !ok {
				return written, errors.New("rig disconnected")
			}
			if msg.Final {
				return written, msg.Err
			}
			name := msg.FileName
			if !isSafeOutputFile(name) {
				continue
			}
			if maxFile > 0 && int64(len(msg.Data)) > maxFile {
				sendAbort()
				return written, fmt.Errorf("comfy: rig sent oversize file %q (%d bytes, cap %d)", name, len(msg.Data), maxFile)
			}
			if maxJob > 0 && bytesWritten+int64(len(msg.Data)) > maxJob {
				sendAbort()
				return written, fmt.Errorf("comfy: job total size exceeded cap (%d bytes)", maxJob)
			}
			if maxFiles > 0 && len(written) >= maxFiles {
				sendAbort()
				return written, fmt.Errorf("comfy: job file count exceeded cap (%d files)", maxFiles)
			}
			dst := filepath.Join(outDir, name)
			if err := os.WriteFile(dst, msg.Data, 0o644); err != nil {
				return written, err
			}
			bytesWritten += int64(len(msg.Data))
			written = append(written, name)
			s.hub.broadcastToUser(a.userID, "comfy_progress", map[string]any{
				"job_id": jobID,
				"status": "streaming",
				"file":   name,
			})
		}
	}
}

// pickComfyRigs returns up to nRigs comfy-capable agents.  If poolID is set,
// scoped to that pool; otherwise across the user's own rigs.
//
// Single JOIN: the previous implementation did a full table scan +
// ORDER BY RANDOM() plus an N+1 SELECT against comfy_caps for every
// candidate.  We now filter cap-gated rigs in the same query and apply
// LIMIT so the engine stops once nRigs candidates are scanned.  The
// scheduler still randomises ordering (each call hashes against a
// per-request salt) but does so on a much smaller post-filter set —
// see comfy_pick_order_salt below.  Audited 2026-05-25 (P21-CF9).
func (s *server) pickComfyRigs(uid, poolID int64, nRigs int) []*agentConn {
	if nRigs < 1 {
		nRigs = 1
	}
	// Light over-fetch so we still hit nRigs picks even when some
	// candidates aren't in s.hub (rig connected to another replica) —
	// 3x is a generous-but-bounded margin.
	limit := nRigs * 3
	if limit < 8 {
		limit = 8
	}
	var rows *sql.Rows
	var err error
	if poolID > 0 {
		rows, err = s.dbQuery(`
			SELECT r.user_id, r.agent_id
			  FROM pool_rigs pr
			  JOIN rigs r ON r.id = pr.rig_id
			  JOIN comfy_caps cc
			    ON cc.user_id = r.user_id
			   AND cc.agent_id = r.agent_id
			   AND cc.ok = 1
			 WHERE pr.pool_id = ?
			 ORDER BY RANDOM()
			 LIMIT ?`, poolID, limit)
	} else {
		rows, err = s.dbQuery(`
			SELECT r.user_id, r.agent_id
			  FROM rigs r
			  JOIN comfy_caps cc
			    ON cc.user_id = r.user_id
			   AND cc.agent_id = r.agent_id
			   AND cc.ok = 1
			 WHERE r.user_id = ?
			 ORDER BY RANDOM()
			 LIMIT ?`, uid, limit)
	}
	if err != nil {
		return nil
	}
	defer rows.Close()
	var picks []*agentConn
	for rows.Next() && len(picks) < nRigs {
		var ru int64
		var aid string
		if err := rows.Scan(&ru, &aid); err != nil {
			continue
		}
		a, ok := s.hub.findAgent(ru, aid)
		if !ok {
			continue
		}
		picks = append(picks, a)
	}
	return picks
}

// recordComfyAgent stores (agent_user_id, agent_id) on the job so credit can
// be attributed at billing time.  Last-writer-wins is fine since the job is
// per-rig anyway.
func (s *server) recordComfyAgent(jobID int64, a *agentConn) {
	if a == nil {
		return
	}
	_, _ = s.dbExec(
		`UPDATE comfy_jobs SET agent_user_id = ?, agent_id = ?, updated_at = ?
		 WHERE id = ?`,
		a.userID, a.agentID, nowUnix(), jobID,
	)
}

// upsertComfyCaps records what an agent told us about its ComfyUI install.
// Called from the WS reader when a `comfy_caps` frame arrives.
func (s *server) upsertComfyCaps(uid int64, agentID string, msg map[string]any) {
	ok := 0
	if v, _ := msg["ok"].(bool); v {
		ok = 1
	}
	ver, _ := msg["version"].(string)
	var models string
	if mm, ok2 := msg["models"].([]any); ok2 {
		b, _ := json.Marshal(mm)
		models = string(b)
	} else {
		models = "[]"
	}
	_, _ = s.dbExec(
		`INSERT INTO comfy_caps (user_id, agent_id, ok, version, models, updated_at)
		 VALUES (?, ?, ?, ?, ?, ?)
		 ON CONFLICT(user_id, agent_id) DO UPDATE SET
		   ok = excluded.ok,
		   version = excluded.version,
		   models = excluded.models,
		   updated_at = excluded.updated_at`,
		uid, agentID, ok, ver, models, nowUnix(),
	)
}

// ─── Status + signing helpers ──────────────────────────────────────────────

func (s *server) comfySetStatus(jobID, uid int64, status, errMsg string) {
	_, _ = s.dbExec(
		`UPDATE comfy_jobs SET status = ?, error = ?, updated_at = ? WHERE id = ?`,
		status, errMsg, nowUnix(), jobID,
	)
	s.hub.broadcastToUser(uid, "comfy_progress", map[string]any{
		"job_id": jobID,
		"status": status,
		"error":  errMsg,
	})
}

func (s *server) comfyFail(jobID, uid int64, msg string) {
	s.comfySetStatus(jobID, uid, "failed", msg)
}

// reapStaleComfyJobs marks queued/running jobs whose updated_at has fallen
// behind maxIdleSec as failed.  Catches handler crashes, panics in
// runComfyJob, and rigs that vanished without their disconnect being
// observed (e.g., the server process was restarted while a job was live).
// Returns the number of rows updated.
//
// Why both queued and running?
//   - queued: handler never spawned runComfyJob (crash between INSERT and go)
//   - running: dispatch goroutine died without finalising
// Live jobs touch updated_at via comfySetStatus / recordComfyAgent, so any
// row with a stale timestamp is by definition orphaned.
func (s *server) reapStaleComfyJobs(maxIdleSec int64) (int64, error) {
	cutoff := nowUnix() - maxIdleSec

	// Step 1: find candidate IDs first so we can also cancel any
	// matching in-process job goroutines.  Without the in-memory cancel,
	// the runner would keep streaming and last-writer-wins on
	// comfySetStatus would silently overwrite our 'failed' verdict with
	// 'done' (audit P21-CF4).  Doing the SELECT before the UPDATE means
	// a job that finishes naturally in the window between the two
	// queries won't get cancelled — its row's updated_at advanced past
	// `cutoff` so it would no longer match anyway.
	rows, err := s.dbQuery(
		`SELECT id FROM comfy_jobs
		   WHERE status IN ('queued','running')
		     AND updated_at < ?`,
		cutoff,
	)
	if err != nil {
		return 0, err
	}
	var stale []int64
	for rows.Next() {
		var id int64
		if err := rows.Scan(&id); err == nil {
			stale = append(stale, id)
		}
	}
	rows.Close()
	if len(stale) == 0 {
		return 0, nil
	}
	// Step 2: cancel the in-process goroutines first so they don't write
	// 'done' after our UPDATE.  cancel() is a no-op if the job isn't
	// tracked on this replica (cross-replica reaping is fine — the DB
	// row update is still authoritative).
	for _, id := range stale {
		s.comfyJobs.cancel(id)
	}
	// Step 3: persist the verdict.
	res, err := s.dbExec(
		`UPDATE comfy_jobs
		   SET status = 'failed',
		       error  = 'reaped: job idle past timeout',
		       updated_at = ?
		 WHERE status IN ('queued','running')
		   AND updated_at < ?`,
		nowUnix(), cutoff,
	)
	if err != nil {
		return 0, err
	}
	n, _ := res.RowsAffected()
	return n, nil
}

// countComfyJobs returns (active, queued, total_failed) for /metrics.
// active = currently-running on this process (in-memory jobs map).
// queued = persistent rows in 'queued' state (not yet dispatched).
// total_failed = lifetime failed/cancelled counter (cheap COUNT).
func (s *server) countComfyJobs() (active, queued, totalFailed int64) {
	if s.comfyJobs != nil {
		s.comfyJobs.mu.Lock()
		active = int64(len(s.comfyJobs.inflight))
		s.comfyJobs.mu.Unlock()
	}
	if s.db != nil {
		_ = s.dbQueryRow(
			`SELECT COUNT(*) FROM comfy_jobs WHERE status = 'queued'`,
		).Scan(&queued)
		_ = s.dbQueryRow(
			`SELECT COUNT(*) FROM comfy_jobs WHERE status IN ('failed','cancelled')`,
		).Scan(&totalFailed)
	}
	return active, queued, totalFailed
}

// signComfyOutputV1 is the legacy URL signature scheme: HMAC over
// (id, file, exp).  Kept only for the 24h grace window after deploy —
// any URL minted before A11 shipped will still verify against this
// function for one day.  After that, handleComfyOutput refuses v1 sigs
// entirely.
func (s *server) signComfyOutputV1(id int64, file string, exp int64) string {
	h := hmac.New(sha256.New, []byte(s.cfg.sessionSecret))
	fmt.Fprintf(h, "comfy/%d/%s@%d", id, file, exp)
	return hex.EncodeToString(h.Sum(nil))
}

// signComfyOutputV2 binds the URL to the user_id that minted it.  Even
// if the URL leaks (image embedded in a public blog, copied into a
// chat log, indexed by a crawler), a third party can't open it: the
// HTTP handler additionally requires an authenticated session whose
// user_id matches the bound uid.
//
// Format:  HMAC-SHA256(secret, "comfyv2/<uid>/<id>/<file>@<exp>")
// URL:     /comfy/out/<id>/<file>?v=2&uid=<uid>&exp=<unix>&sig=<hex>
//
// The "comfyv2/" prefix prevents replay of v1 signatures at the v2
// verifier and vice versa.
func (s *server) signComfyOutputV2(uid, id int64, file string, exp int64) string {
	h := hmac.New(sha256.New, []byte(s.cfg.sessionSecret))
	fmt.Fprintf(h, "comfyv2/%d/%d/%s@%d", uid, id, file, exp)
	return hex.EncodeToString(h.Sum(nil))
}

func (s *server) signComfyOutputURL(uid, id int64, file string, ttl time.Duration) string {
	// P7: macaroon-capability URL.  Falls through to legacy v2 HMAC iff
	// the macaroon mint fails (which should not happen in practice).
	if u := s.mintComfyOutputURLCap(uid, id, file, ttl); u != "" {
		return u
	}
	exp := time.Now().Add(ttl).Unix()
	sig := s.signComfyOutputV2(uid, id, file, exp)
	return fmt.Sprintf("%s/comfy/out/%d/%s?v=2&uid=%d&exp=%d&sig=%s",
		strings.TrimRight(s.cfg.publicURL, "/"), id, file, uid, exp, sig)
}

func (s *server) signedURLsFor(uid, id int64, files []string, ttl time.Duration) []string {
	out := make([]string, 0, len(files))
	for _, f := range files {
		out = append(out, s.signComfyOutputURL(uid, id, f, ttl))
	}
	return out
}

// comfyV1GraceWindow is how long after server start we'll still accept
// v1 (uid-free) signatures.  Bounded so the audit answer to "can a
// leaked pre-A11 URL still be opened by anyone?" is "only for 24h
// after this server booted, then no."
const comfyV1GraceWindow = 24 * time.Hour

// isSafeOutputFile validates a filename a rig wants the coordinator to
// write under comfy-out/<job_id>/.  The coordinator serves these files
// via /comfy/out/<id>/<file>, so a hostile or buggy name here is both a
// disk-write hazard AND an open-redirect / file-disclosure hazard once
// the URL ships.  We require:
//
//   - non-empty, length ≤ 200 (well above any legitimate render name)
//   - no path separators or path-traversal pieces (no slash/backslash,
//     no leading dot — Linux hidden file, also blocks "." / "..")
//   - no NUL or other ASCII control characters
//   - no Windows-reserved basenames (CON/PRN/AUX/NUL/COM1-9/LPT1-9)
//     because operators on cross-platform setups (Docker on Windows
//     with bind mounts) would otherwise hit "filename or extension is
//     too long" errors that look like coordinator bugs
//   - no leading or trailing whitespace (rsync/tar surprises)
//   - allowed extension: png, jpg, jpeg, webp, gif, mp4, webm
func isSafeOutputFile(f string) bool {
	if f == "" || len(f) > 200 {
		return false
	}
	if strings.Contains(f, "/") || strings.Contains(f, "\\") {
		return false
	}
	if f[0] == '.' { // ".", "..", ".hidden"
		return false
	}
	if f != strings.TrimSpace(f) {
		return false
	}
	for _, r := range f {
		// All ASCII controls (including NUL) and DEL.  Unicode controls
		// pass — rendering libraries may legitimately use them in
		// generated names, and the path is already segmented above.
		if r < 0x20 || r == 0x7f {
			return false
		}
	}
	// Windows-reserved names (case-insensitive, basename only).
	base := strings.ToUpper(strings.TrimSuffix(f, filepath.Ext(f)))
	switch base {
	case "CON", "PRN", "AUX", "NUL",
		"COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
		"LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9":
		return false
	}
	switch strings.ToLower(filepath.Ext(f)) {
	case ".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".webm":
		return true
	}
	return false
}

// resolveBaseSeed picks the seed shared by all rigs in a fan-out.  If
// the user pinned one via params.seed we honour it; otherwise we derive
// one from (jobID, wall-clock-ns) so the chosen seed is captured in the
// job row and reruns are deterministic.
//
// Returns (seed, source) where source is "user" or "auto" — exposed
// for future logging.
func resolveBaseSeed(paramsJSON string, jobID int64) (int64, string) {
	var params map[string]any
	_ = json.Unmarshal([]byte(paramsJSON), &params)
	if sv, ok := params["seed"].(float64); ok {
		return int64(sv), "user"
	}
	// Mix jobID into the nanosecond clock so n_rigs simultaneous jobs
	// don't collide on the same UnixNano() bucket.
	base := time.Now().UnixNano() ^ (jobID * 0x100000001B3)
	if base < 0 {
		base = -base
	}
	return base & 0x7fffffff, "auto"
}

// substituteComfyParams is the legacy entry point — kept for tests that
// don't care about fan-out.  Forwards to the seed-aware variant.
func substituteComfyParams(graph, prompt, paramsJSON, modelName string) (string, error) {
	seed, _ := resolveBaseSeed(paramsJSON, 0)
	return substituteComfyParamsWithSeed(graph, prompt, paramsJSON, modelName, seed)
}

// substituteComfyParamsWithSeed rewrites the workflow JSON to insert
// the user's prompt and override params (seed, steps, size) into the
// right nodes.  Substitution is restricted to values inside "inputs"
// maps — anywhere else in the tree a literal "$PROMPT" / "$SEED" /
// etc. is left alone so a user-supplied prompt template can contain
// these strings without being silently clobbered into integers.
//
// Conventions (inside inputs maps only):
//   - Any string field equal to "$PROMPT" → the prompt.
//   - Any string field equal to "$MODEL"  → modelName.
//   - "$SEED" → the seed argument (per-rig in fan-out).
//   - "$WIDTH" / "$HEIGHT" → from params.size = "WxH" (defaults 1024x1024).
func substituteComfyParamsWithSeed(graph, prompt, paramsJSON, modelName string, seed int64) (string, error) {
	var node any
	if err := json.Unmarshal([]byte(graph), &node); err != nil {
		return "", fmt.Errorf("graph not valid JSON: %w", err)
	}
	var params map[string]any
	_ = json.Unmarshal([]byte(paramsJSON), &params)
	width, height := 1024, 1024
	if sz, _ := params["size"].(string); sz != "" {
		var wv, hv int
		if _, err := fmt.Sscanf(sz, "%dx%d", &wv, &hv); err == nil && wv > 0 && hv > 0 {
			width, height = wv, hv
		}
	}
	subst := func(s string) any {
		switch s {
		case "$PROMPT":
			return prompt
		case "$MODEL":
			return modelName
		case "$SEED":
			return seed
		case "$WIDTH":
			return width
		case "$HEIGHT":
			return height
		}
		return s
	}
	// substIn rewrites a subtree, but only descends into "inputs" maps
	// (and arrays/objects rooted inside them).  Top-level node maps —
	// the ComfyUI graph itself — keep their structure unchanged.
	var rewriteInputs func(v any) any
	rewriteInputs = func(v any) any {
		switch x := v.(type) {
		case map[string]any:
			for k, vv := range x {
				x[k] = rewriteInputs(vv)
			}
			return x
		case []any:
			for i, vv := range x {
				x[i] = rewriteInputs(vv)
			}
			return x
		case string:
			return subst(x)
		}
		return v
	}
	// Walk only the "inputs" sub-maps of each top-level node.  A
	// ComfyUI API-format graph looks like {"3":{"class_type":"...",
	// "inputs":{...}}, "4":{...}}; we substitute inside inputs only.
	if root, ok := node.(map[string]any); ok {
		for _, n := range root {
			if nm, ok := n.(map[string]any); ok {
				if in, ok := nm["inputs"]; ok {
					nm["inputs"] = rewriteInputs(in)
				}
			}
		}
	}
	out, _ := json.Marshal(node)
	return string(out), nil
}

// defaultComfyWorkflowJSON is a minimal SDXL text-to-image workflow used when
// the user runs /v1/images/generations without registering a workflow first.
// The agent-side ComfyUI install must have an SDXL checkpoint named
// "sd_xl_base_1.0.safetensors" (or whatever $MODEL is overridden to).
const defaultComfyWorkflowJSON = `{
  "3": {"class_type": "KSampler",
        "inputs": {"seed": "$SEED", "steps": 20, "cfg": 7,
                   "sampler_name": "euler", "scheduler": "normal", "denoise": 1,
                   "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0],
                   "latent_image": ["5", 0]}},
  "4": {"class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "$MODEL"}},
  "5": {"class_type": "EmptyLatentImage",
        "inputs": {"width": "$WIDTH", "height": "$HEIGHT", "batch_size": 1}},
  "6": {"class_type": "CLIPTextEncode",
        "inputs": {"text": "$PROMPT", "clip": ["4", 1]}},
  "7": {"class_type": "CLIPTextEncode",
        "inputs": {"text": "", "clip": ["4", 1]}},
  "8": {"class_type": "VAEDecode",
        "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
  "9": {"class_type": "SaveImage",
        "inputs": {"filename_prefix": "distpool", "images": ["8", 0]}}
}`

