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
	res, err := s.db.Exec(
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
	rows, err := s.db.Query(
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
	res, err := s.db.Exec(
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
	rows, err := s.db.Query(
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
	// Bound concurrency: each user gets up to 4 in-flight comfy jobs.  Past
	// that we reject so a slow rig can't be hammered into oblivion from one
	// browser tab.
	var inflight int
	_ = s.db.QueryRow(
		`SELECT COUNT(*) FROM comfy_jobs
		 WHERE user_id = ? AND status IN ('queued','running','streaming')`,
		u.ID).Scan(&inflight)
	if inflight >= 4 {
		writeErr(w, 429, "too many in-flight comfy jobs (limit 4); wait for one to finish")
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
		err := s.db.QueryRow(
			`SELECT id, graph_json, kind, n_rigs FROM comfy_workflows
			 WHERE user_id = ? AND name = ?`, u.ID, body.WorkflowName,
		).Scan(&wfID, &graph, &wfKind, &nRigs)
		if err != nil {
			writeErr(w, 404, "workflow not found: "+body.WorkflowName)
			return
		}
	} else if wfID != 0 {
		err := s.db.QueryRow(
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

	// Resolve model name if provided (informational; agent loads on demand).
	if body.ModelName != "" {
		var ok int
		_ = s.db.QueryRow(
			`SELECT 1 FROM comfy_models WHERE name = ?`, body.ModelName,
		).Scan(&ok)
		if ok == 0 {
			writeErr(w, 404, "model not registered: "+body.ModelName)
			return
		}
	}

	params := string(body.Params)
	if params == "" || params == "null" {
		params = "{}"
	}

	res, err := s.db.Exec(
		`INSERT INTO comfy_jobs
		 (user_id, pool_id, workflow_id, prompt, params_json, status, created_at, updated_at)
		 VALUES (?, ?, ?, ?, ?, 'queued', ?, ?)`,
		u.ID, sql.NullInt64{Int64: body.PoolID, Valid: body.PoolID != 0},
		sql.NullInt64{Int64: wfID, Valid: wfID != 0},
		body.Prompt, params, nowUnix(), nowUnix(),
	)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	jobID, _ := res.LastInsertId()

	jobCtx, cancel := context.WithCancel(context.Background())
	s.comfyJobs.register(jobID, cancel)
	jobAdmitted = true
	go s.runComfyJob(jobCtx, u.ID, jobID, body.PoolID, graph, wfKind, body.Prompt, params, body.ModelName, nRigs)

	writeJSON(w, 200, map[string]any{"job_id": jobID, "status": "queued", "kind": wfKind})
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
	err = s.db.QueryRow(
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
		signed = append(signed, s.signComfyOutputURL(id, f, 15*time.Minute))
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
	if err := s.db.QueryRow(`SELECT user_id FROM comfy_jobs WHERE id = ?`, id).Scan(&uid); err != nil {
		writeErr(w, 404, "no such job")
		return
	}
	if uid != u.ID {
		writeErr(w, 403, "not your job")
		return
	}
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
	rows, err := s.db.Query(
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
			j.OutURLs = append(j.OutURLs, s.signComfyOutputURL(j.ID, f, 15*time.Minute))
		}
		out = append(out, j)
	}
	writeJSON(w, 200, map[string]any{"jobs": out})
}

// GET /comfy/out/{id}/{file}?exp=...&sig=...
//
// HMAC-signed output retrieval — mirrors the shard-download URL signing
// scheme so the dashboard can serve images straight from disk.
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
	exp, _ := strconv.ParseInt(r.URL.Query().Get("exp"), 10, 64)
	sig := r.URL.Query().Get("sig")
	if exp == 0 || sig == "" {
		writeErr(w, 401, "missing signature")
		return
	}
	if nowUnix() > exp {
		writeErr(w, 401, "url expired")
		return
	}
	want := s.signComfyOutput(id, file, exp)
	if want != sig {
		writeErr(w, 401, "bad signature")
		return
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
		Prompt string `json:"prompt"`
		Model  string `json:"model"`
		Size   string `json:"size"`
		N      int    `json:"n"`
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
	params := map[string]any{
		"size": body.Size,
		"n":    body.N,
	}
	pj, _ := json.Marshal(params)
	res, err := s.db.Exec(
		`INSERT INTO comfy_jobs
		 (user_id, prompt, params_json, status, created_at, updated_at)
		 VALUES (?, ?, ?, 'queued', ?, ?)`,
		u.ID, body.Prompt, string(pj), nowUnix(), nowUnix(),
	)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	jobID, _ := res.LastInsertId()
	jobCtx, cancel := context.WithCancel(context.Background())
	s.comfyJobs.register(jobID, cancel)
	jobAdmitted = true
	go s.runComfyJob(jobCtx, u.ID, jobID, 0, defaultComfyWorkflowJSON, "image", body.Prompt, string(pj), body.Model, 1)

	// Block until the job completes or 60s elapse — OpenAI clients expect
	// synchronous response.  For longer renders the dashboard's async API
	// (/api/comfy/jobs/{id}) should be used instead.
	deadline := time.After(60 * time.Second)
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
			err := s.db.QueryRow(
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
					out = append(out, map[string]string{"url": s.signComfyOutputURL(jobID, f, 1*time.Hour)})
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

	// Materialise the workflow with the user's prompt + params substituted in.
	finalGraph, err := substituteComfyParams(graphJSON, prompt, paramsJSON, modelName)
	if err != nil {
		s.comfyFail(jobID, uid, "graph build: "+err.Error())
		return
	}

	// Dispatch over WS to each chosen rig.
	type rigResult struct {
		agent  *agentConn
		files  []string
		err    error
	}
	resultsCh := make(chan rigResult, len(rigs))
	for _, ag := range rigs {
		ag := ag
		go func() {
			files, err := s.dispatchComfyToAgent(ctx, ag, jobID, kind, finalGraph)
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
	_, _ = s.db.Exec(
		`UPDATE comfy_jobs SET status = 'done', out_files = ?, updated_at = ? WHERE id = ?`,
		string(fb), nowUnix(), jobID,
	)
	s.hub.broadcastToUser(uid, "comfy_progress", map[string]any{
		"job_id":   jobID,
		"status":   "done",
		"out_urls": s.signedURLsFor(jobID, allFiles, 15*time.Minute),
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

	a.send(req)

	outDir := filepath.Join(s.cfg.comfyOutDir, strconv.FormatInt(jobID, 10))
	var written []string
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
			// Write file from base64.
			name := msg.FileName
			if !isSafeOutputFile(name) {
				continue
			}
			dst := filepath.Join(outDir, name)
			if err := os.WriteFile(dst, msg.Data, 0o644); err != nil {
				return written, err
			}
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
func (s *server) pickComfyRigs(uid, poolID int64, nRigs int) []*agentConn {
	if nRigs < 1 {
		nRigs = 1
	}
	var rows *sql.Rows
	var err error
	if poolID > 0 {
		rows, err = s.db.Query(`
			SELECT r.user_id, r.agent_id FROM pool_rigs pr
			JOIN rigs r ON r.id = pr.rig_id
			WHERE pr.pool_id = ?
			ORDER BY RANDOM()`, poolID)
	} else {
		rows, err = s.db.Query(`
			SELECT user_id, agent_id FROM rigs WHERE user_id = ? ORDER BY RANDOM()`, uid)
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
		// Capability gate: rig must have advertised comfy_ok = 1.
		var ok2 int
		_ = s.db.QueryRow(
			`SELECT ok FROM comfy_caps WHERE user_id = ? AND agent_id = ?`,
			ru, aid,
		).Scan(&ok2)
		if ok2 != 1 {
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
	_, _ = s.db.Exec(
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
	_, _ = s.db.Exec(
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
	_, _ = s.db.Exec(
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
	res, err := s.db.Exec(
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
		_ = s.db.QueryRow(
			`SELECT COUNT(*) FROM comfy_jobs WHERE status = 'queued'`,
		).Scan(&queued)
		_ = s.db.QueryRow(
			`SELECT COUNT(*) FROM comfy_jobs WHERE status IN ('failed','cancelled')`,
		).Scan(&totalFailed)
	}
	return active, queued, totalFailed
}

func (s *server) signComfyOutput(id int64, file string, exp int64) string {
	// Reuse the shard-URL HMAC scheme with a distinct prefix so signed
	// shard URLs can't be replayed at the comfy endpoint.
	h := hmac.New(sha256.New, []byte(s.cfg.sessionSecret))
	fmt.Fprintf(h, "comfy/%d/%s@%d", id, file, exp)
	return hex.EncodeToString(h.Sum(nil))
}

func (s *server) signComfyOutputURL(id int64, file string, ttl time.Duration) string {
	exp := time.Now().Add(ttl).Unix()
	sig := s.signComfyOutput(id, file, exp)
	return fmt.Sprintf("%s/comfy/out/%d/%s?exp=%d&sig=%s",
		strings.TrimRight(s.cfg.publicURL, "/"), id, file, exp, sig)
}

func (s *server) signedURLsFor(id int64, files []string, ttl time.Duration) []string {
	out := make([]string, 0, len(files))
	for _, f := range files {
		out = append(out, s.signComfyOutputURL(id, f, ttl))
	}
	return out
}

func isSafeOutputFile(f string) bool {
	if f == "" || strings.Contains(f, "/") || strings.Contains(f, "\\") {
		return false
	}
	if f == "." || f == ".." {
		return false
	}
	switch strings.ToLower(filepath.Ext(f)) {
	case ".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".webm":
		return true
	}
	return false
}

// substituteComfyParams rewrites the workflow JSON to insert the user's
// prompt and override params (seed, steps, size) into the right nodes.
// Conventions:
//   - Any string field equal to "$PROMPT" is replaced with the prompt.
//   - Any string field equal to "$MODEL"  is replaced with modelName.
//   - "$SEED" → random int (or params.seed if provided).
//   - "$WIDTH" / "$HEIGHT" → from params.size = "WxH" (defaults 1024x1024).
func substituteComfyParams(graph, prompt, paramsJSON, modelName string) (string, error) {
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
	var seed int64
	if sv, ok := params["seed"].(float64); ok {
		seed = int64(sv)
	} else {
		seed = time.Now().UnixNano() & 0x7fffffff
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
	var walk func(v any) any
	walk = func(v any) any {
		switch x := v.(type) {
		case map[string]any:
			for k, vv := range x {
				x[k] = walk(vv)
			}
			return x
		case []any:
			for i, vv := range x {
				x[i] = walk(vv)
			}
			return x
		case string:
			return subst(x)
		}
		return v
	}
	walk(node)
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

