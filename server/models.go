package main

// Model registry + shard server.
//
// A model is stored on the coordinator as a directory containing stage-N.gguf
// shards (produced by `llama-split-gguf`) plus a manifest.json describing
// layer ranges per stage.  The coordinator exposes:
//
//   POST /api/models                 — register a model (split it if needed).
//   GET  /api/models                 — list known models.
//   GET  /models/:id/manifest.json   — public JSON manifest (session-gated).
//   GET  /models/:id/shards/:file    — binary shard download, HMAC-signed URL.
//
// Shard URLs are short-lived HMAC tokens so they can be safely handed to nodes
// (including those on other users' boxes) without exposing the raw shard path.

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// ─── Data types ────────────────────────────────────────────────────────────

type modelRow struct {
	ID         int64  `json:"id"`
	Name       string `json:"name"`
	NLayers    int    `json:"n_layers"`
	NShards    int    `json:"n_shards"`
	ShardsDir  string `json:"-"` // filesystem path, not exposed
	CreatedAt  int64  `json:"created_at"`
}

type shardManifestEntry struct {
	StageIdx int    `json:"stage_idx"`
	LayerLo  int    `json:"layer_lo"`
	LayerHi  int    `json:"layer_hi"`
	Tensors  int    `json:"tensors"`
	File     string `json:"file"`
}

type shardManifest struct {
	Arch     string               `json:"arch"`
	NBlocks  int                  `json:"n_blocks"`
	NStages  int                  `json:"n_stages"`
	Stages   []shardManifestEntry `json:"stages"`
}

// ─── Register handler ──────────────────────────────────────────────────────

type registerModelReq struct {
	Name      string `json:"name"`        // display name, unique
	SrcPath   string `json:"src_path"`    // absolute path on coordinator host
	NStages   int    `json:"n_stages"`    // how many pipeline stages to split into
	Splitter  string `json:"splitter"`    // optional override for llama-split-gguf binary
}

// POST /api/models
//
// In dev mode any logged-in user may register a model; in prod this would be
// admin-gated (out of scope here).  The source file must already exist on the
// coordinator's filesystem — we don't accept multi-GB uploads over HTTP.
func (s *server) handleRegisterModel(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	_ = u
	var body registerModelReq
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeErr(w, 400, "bad json: "+err.Error())
		return
	}
	if body.Name == "" || body.SrcPath == "" || body.NStages < 1 {
		writeErr(w, 400, "name, src_path, n_stages required")
		return
	}
	if _, err := os.Stat(body.SrcPath); err != nil {
		writeErr(w, 400, "src_path not readable: "+err.Error())
		return
	}

	// Reject duplicate name up-front with a clearer error than SQLite's
	// "UNIQUE constraint failed".
	var existing int64
	_ = s.db.QueryRow(`SELECT id FROM models WHERE name = ?`, body.Name).Scan(&existing)
	if existing != 0 {
		writeErr(w, 409, "model name already exists")
		return
	}

	shardsDir := filepath.Join(s.cfg.modelsDir, sanitizeModelName(body.Name))
	if err := os.MkdirAll(shardsDir, 0o755); err != nil {
		writeErr(w, 500, "mkdir: "+err.Error())
		return
	}

	splitter := body.Splitter
	if splitter == "" {
		splitter = s.cfg.splitterBin
	}
	cmd := exec.Command(splitter, body.SrcPath, shardsDir, strconv.Itoa(body.NStages))
	out, err := cmd.CombinedOutput()
	if err != nil {
		writeErr(w, 500, "splitter failed: "+err.Error()+": "+string(out))
		return
	}

	man, err := readShardManifest(shardsDir)
	if err != nil {
		writeErr(w, 500, "read manifest: "+err.Error())
		return
	}

	res, err := s.db.Exec(
		`INSERT INTO models (name, n_layers, n_shards, shards_dir, created_at) VALUES (?, ?, ?, ?, ?)`,
		body.Name, man.NBlocks, man.NStages, shardsDir, nowUnix(),
	)
	if err != nil {
		writeErr(w, 500, "db insert: "+err.Error())
		return
	}
	id, _ := res.LastInsertId()

	writeJSON(w, 200, map[string]any{
		"id":        id,
		"name":      body.Name,
		"n_layers":  man.NBlocks,
		"n_shards":  man.NStages,
		"arch":      man.Arch,
		"stages":    man.Stages,
	})
}

// ─── List handler ──────────────────────────────────────────────────────────

// GET /api/models
func (s *server) handleListModels(w http.ResponseWriter, r *http.Request) {
	if _, ok := s.userFromRequest(r); !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	rows, err := s.db.Query(
		`SELECT id, name, n_layers, n_shards, created_at FROM models ORDER BY id ASC`)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	defer rows.Close()
	var out []modelRow
	for rows.Next() {
		var m modelRow
		if err := rows.Scan(&m.ID, &m.Name, &m.NLayers, &m.NShards, &m.CreatedAt); err != nil {
			continue
		}
		out = append(out, m)
	}
	writeJSON(w, 200, map[string]any{"models": out})
}

// ─── Manifest handler ──────────────────────────────────────────────────────

// GET /models/:id/manifest.json
//
// Session-gated.  Returns the manifest as written by `llama-split-gguf`.
func (s *server) handleModelManifest(w http.ResponseWriter, r *http.Request) {
	if _, ok := s.userFromRequest(r); !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	id, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil {
		writeErr(w, 400, "bad id")
		return
	}
	dir, err := s.modelShardsDir(id)
	if err != nil {
		writeErr(w, 404, "no such model")
		return
	}
	http.ServeFile(w, r, filepath.Join(dir, "manifest.json"))
}

// ─── Shard download ────────────────────────────────────────────────────────

// GET /models/:id/shards/:file?exp=<unix>&sig=<hex>
//
// `exp` is a unix timestamp; `sig = HMAC(secret, "<id>/<file>@<exp>")` in hex.
// We do NOT require a session cookie here so that agents running off-box
// (and WebRTC data channels) can fetch shards with only the signed URL.
// Since the URL is short-lived and scoped to one file, leaking a URL leaks
// one shard's read access until `exp`, which is acceptable for M5.
func (s *server) handleShardDownload(w http.ResponseWriter, r *http.Request) {
	id, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil {
		writeErr(w, 400, "bad id")
		return
	}
	file := r.PathValue("file")
	if !isSafeShardFile(file) {
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
	want := s.signShardURL(id, file, exp)
	if !hmac.Equal([]byte(sig), []byte(want)) {
		writeErr(w, 401, "bad signature")
		return
	}

	dir, err := s.modelShardsDir(id)
	if err != nil {
		writeErr(w, 404, "no such model")
		return
	}
	http.ServeFile(w, r, filepath.Join(dir, file))
}

// ─── Helpers ───────────────────────────────────────────────────────────────

func (s *server) modelShardsDir(id int64) (string, error) {
	var dir string
	err := s.db.QueryRow(`SELECT shards_dir FROM models WHERE id = ?`, id).Scan(&dir)
	return dir, err
}

// signShardURL returns a hex-encoded HMAC-SHA256 over "<id>/<file>@<exp>".
func (s *server) signShardURL(id int64, file string, exp int64) string {
	h := hmac.New(sha256.New, []byte(s.cfg.sessionSecret))
	fmt.Fprintf(h, "%d/%s@%d", id, file, exp)
	return hex.EncodeToString(h.Sum(nil))
}

// mintShardURL produces a signed URL like
//   <publicURL>/models/42/shards/stage-0.gguf?exp=...&sig=...
// valid for `ttl`.
func (s *server) mintShardURL(id int64, file string, ttl time.Duration) string {
	exp := time.Now().Add(ttl).Unix()
	sig := s.signShardURL(id, file, exp)
	return fmt.Sprintf("%s/models/%d/shards/%s?exp=%d&sig=%s",
		strings.TrimRight(s.cfg.publicURL, "/"), id, file, exp, sig)
}

// modelInfoByName resolves a model by name; returns (0, "", err) if not found.
func (s *server) modelInfoByName(name string) (int64, string, int, error) {
	var id int64
	var dir string
	var nlay int
	err := s.db.QueryRow(
		`SELECT id, shards_dir, n_layers FROM models WHERE name = ?`, name,
	).Scan(&id, &dir, &nlay)
	return id, dir, nlay, err
}

// readShardManifest reads the manifest.json emitted by llama-split-gguf.
func readShardManifest(dir string) (*shardManifest, error) {
	b, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return nil, err
	}
	var m shardManifest
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, err
	}
	return &m, nil
}

// sanitizeModelName keeps only filename-safe chars; "/" becomes "_", etc.
func sanitizeModelName(n string) string {
	b := make([]byte, 0, len(n))
	for i := 0; i < len(n); i++ {
		c := n[i]
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9':
			b = append(b, c)
		case c == '-' || c == '_' || c == '.':
			b = append(b, c)
		default:
			b = append(b, '_')
		}
	}
	if len(b) == 0 {
		b = []byte("model")
	}
	return string(b)
}

// isSafeShardFile rejects paths that try to escape the shards dir.
func isSafeShardFile(f string) bool {
	if f == "" || strings.Contains(f, "/") || strings.Contains(f, "\\") {
		return false
	}
	if f == "." || f == ".." {
		return false
	}
	// Only permit .gguf files or the manifest.
	if !strings.HasSuffix(f, ".gguf") && f != "manifest.json" {
		return false
	}
	return true
}
