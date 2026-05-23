// hf_convert.go — fallback path for HF repos that don't ship GGUF.
//
// Flow:
//   1. User imports `org/model` via the same `/api/hf/import` UI but with
//      `convert: true` in the request body (or — when no .gguf files exist
//      in the repo at all — the importer auto-falls back here).
//   2. We list the "convertible" files in the repo: config.json,
//      tokenizer*, *.safetensors / *.bin (model weights), plus any
//      generation_config.json etc.
//   3. Download them into a staging dir.
//   4. Shell out to `python convert_hf_to_gguf.py <staging> --outtype <q8_0>`
//      (path configured via DIST_CONVERTER / DIST_PYTHON / DIST_CONVERT_QUANT
//      env vars; defaults assume the llama.cpp submodule).
//   5. Take the resulting GGUF and feed it into the existing splitter +
//      registration path — identical to the GGUF-direct branch.
//
// Why a separate file:
//   - The converter step is heavy (pip deps + minutes of compute) and we
//     want it cleanly separated so it can be re-tested / re-invoked on a
//     stuck import without churn in the GGUF-direct path.
//   - Reuses the downloader, status helpers, and progress broadcast from
//     hf_download.go.

package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync/atomic"
	"time"

	"encoding/json"
)

// hfListConvertible returns the set of files we need to materialize on disk
// for `convert_hf_to_gguf.py` to do its job.  HF repos with a transformers
// checkpoint usually have:
//
//   config.json
//   tokenizer.json | tokenizer.model | vocab.json + merges.txt
//   tokenizer_config.json (optional but common)
//   *.safetensors   ← weights (preferred)
//   pytorch_model*.bin  ← weights fallback
//   generation_config.json (optional)
//   chat_template.jinja (optional)
//
// We accept any of the above; if no weights file is present we bail.
func hfListConvertible(ctx context.Context, repoID, revision, token string) ([]hfFileInfo, error) {
	if revision == "" {
		revision = "main"
	}
	u := fmt.Sprintf("%s/api/models/%s/tree/%s?recursive=true",
		hfAPIBase, url.PathEscape(repoID), url.PathEscape(revision))
	req, err := http.NewRequestWithContext(ctx, "GET", u, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", hfUserAgent)
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := hfClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode == 401 || resp.StatusCode == 403 {
		return nil, fmt.Errorf("hf auth required for %s (status %d)", repoID, resp.StatusCode)
	}
	if resp.StatusCode == 404 {
		return nil, fmt.Errorf("hf repo not found: %s", repoID)
	}
	if resp.StatusCode/100 != 2 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return nil, fmt.Errorf("hf list failed: %d %s", resp.StatusCode, string(body))
	}
	var entries []hfTreeEntry
	if err := json.NewDecoder(resp.Body).Decode(&entries); err != nil {
		return nil, err
	}
	var out []hfFileInfo
	hasWeights := false
	for _, e := range entries {
		if e.Type != "file" {
			continue
		}
		p := strings.ToLower(e.Path)
		base := filepath.Base(p)
		// Skip giant binary blobs we don't need (tokenizer logs, onnx, tflite).
		if strings.HasSuffix(p, ".onnx") || strings.HasSuffix(p, ".tflite") ||
			strings.HasSuffix(p, ".msgpack") || strings.HasSuffix(p, ".h5") {
			continue
		}
		// Recognised file?
		keep := false
		switch {
		case strings.HasSuffix(p, ".safetensors"):
			keep = true
			hasWeights = true
		case strings.HasSuffix(p, ".bin") && strings.HasPrefix(base, "pytorch_model"):
			keep = true
			hasWeights = true
		case base == "config.json",
			base == "tokenizer.json",
			base == "tokenizer.model",
			base == "tokenizer_config.json",
			base == "vocab.json",
			base == "merges.txt",
			base == "generation_config.json",
			base == "special_tokens_map.json",
			base == "added_tokens.json",
			base == "preprocessor_config.json",
			strings.HasSuffix(base, "chat_template.jinja"):
			keep = true
		}
		if keep {
			fi := hfFileInfo{Path: e.Path, Size: e.Size}
			if e.Lfs != nil {
				fi.SHA256 = e.Lfs.OID
			}
			out = append(out, fi)
		}
	}
	if !hasWeights {
		return nil, errors.New("no weight files (.safetensors / pytorch_model*.bin) in repo")
	}
	return out, nil
}

// runConverter shells out to convert_hf_to_gguf.py.  Returns the absolute
// path to the produced .gguf, or an error.
//
//   srcDir   — directory containing the downloaded HF checkpoint
//   outDir   — directory where the .gguf should land (created if missing)
//   outType  — quantization type ("q8_0", "f16", "q4_k_m", ...)
func runConverter(pythonBin, converterPy, srcDir, outDir, outType string) (string, error) {
	if converterPy == "" {
		return "", errors.New("converter not configured (set DIST_CONVERTER)")
	}
	if _, err := os.Stat(converterPy); err != nil {
		return "", fmt.Errorf("converter not found at %s: %w", converterPy, err)
	}
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return "", err
	}
	outFile := filepath.Join(outDir, "model.gguf")
	args := []string{
		converterPy, srcDir,
		"--outfile", outFile,
		"--outtype", outType,
	}
	cmd := exec.Command(pythonBin, args...)
	// Plumb stderr to our own so a stuck converter is debuggable from the
	// server logs without surfacing 100MB of progress bars to the user.
	cmd.Stderr = os.Stderr
	cmd.Stdout = os.Stderr
	// Bound the conversion at 60 minutes — most ≤13B models convert in
	// under 20m on a CPU box; if it goes past 60m something is wrong.
	done := make(chan error, 1)
	if err := cmd.Start(); err != nil {
		return "", fmt.Errorf("spawn converter: %w", err)
	}
	go func() { done <- cmd.Wait() }()
	select {
	case err := <-done:
		if err != nil {
			return "", fmt.Errorf("converter exited: %w", err)
		}
	case <-time.After(60 * time.Minute):
		_ = cmd.Process.Kill()
		return "", errors.New("converter timed out (60m)")
	}
	if _, err := os.Stat(outFile); err != nil {
		return "", fmt.Errorf("converter produced no output at %s", outFile)
	}
	return outFile, nil
}

// runHFConvertImport mirrors runHFImport but routes through the converter.
// Stages a non-GGUF HF checkpoint, calls convert_hf_to_gguf.py, then hands
// the resulting GGUF off to the splitter + registration path.
func (s *server) runHFConvertImport(ctx context.Context, uid, jobID int64,
	repoID, revision, modelName string, files []hfFileInfo, nStages int,
	convertQuant string) {
	token := s.userHFToken(uid)
	if convertQuant == "" {
		convertQuant = s.cfg.convertQuant
	}

	s.hfSetStatus(jobID, uid, "downloading", "")

	stagingDir := filepath.Join(s.cfg.modelsDir, ".hf-staging", fmt.Sprintf("%d", jobID))
	if err := os.MkdirAll(stagingDir, 0o755); err != nil {
		s.hfFail(jobID, uid, "staging mkdir: "+err.Error())
		return
	}

	var totalDone atomic.Int64
	flushProgress := func(delta int64) {
		v := totalDone.Add(delta)
		// Throttle DB writes — every 4MB is plenty.
		if v%(4*1024*1024) < 1024*1024 {
			s.hfUpdateBytes(jobID, v)
		}
	}

	for i := range files {
		f := &files[i]
		if ctx.Err() != nil {
			s.hfSetStatus(jobID, uid, "cancelled", "cancelled by user")
			_ = os.RemoveAll(stagingDir)
			return
		}
		dst := filepath.Join(stagingDir, filepath.FromSlash(f.Path))
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			s.hfFail(jobID, uid, "mkdir: "+err.Error())
			return
		}
		if fi, err := os.Stat(dst); err == nil && fi.Size() == f.Size {
			flushProgress(f.Size)
			continue
		}
		url := hfResolveURL(repoID, revision, f.Path)
		err := downloadFile(ctx, url, token, dst, func(d int64) { flushProgress(d) })
		if err != nil {
			if errors.Is(err, context.Canceled) {
				s.hfSetStatus(jobID, uid, "cancelled", "cancelled by user")
				_ = os.RemoveAll(stagingDir)
				return
			}
			s.hfFail(jobID, uid, fmt.Sprintf("download %s: %v", f.Path, err))
			return
		}
	}
	s.hfUpdateBytes(jobID, totalDone.Load())

	// Run the converter.  This is the heavy step — minutes for small
	// models, much longer for 70B-class.
	s.hfSetStatus(jobID, uid, "converting", "")
	convertOut := filepath.Join(stagingDir, "_gguf")
	ggufPath, err := runConverter(
		s.cfg.pythonBin,
		s.cfg.converterPy,
		stagingDir,
		convertOut,
		convertQuant,
	)
	if err != nil {
		s.hfFail(jobID, uid, "convert: "+err.Error())
		return
	}

	// From here on, identical to the GGUF-direct branch: split + register.
	s.hfSetStatus(jobID, uid, "splitting", "")
	if nStages == 0 {
		nStages = 1
	}
	shardsDir := filepath.Join(s.cfg.modelsDir, sanitizeModelName(modelName))
	if err := os.MkdirAll(shardsDir, 0o755); err != nil {
		s.hfFail(jobID, uid, "mkdir shards: "+err.Error())
		return
	}
	if err := runSplitter(s.cfg.splitterBin, ggufPath, shardsDir, nStages); err != nil {
		s.hfFail(jobID, uid, "splitter: "+err.Error())
		return
	}
	man, err := readShardManifest(shardsDir)
	if err != nil {
		s.hfFail(jobID, uid, "read manifest: "+err.Error())
		return
	}
	res, err := s.db.Exec(
		`INSERT INTO models (name, n_layers, n_shards, shards_dir, created_at)
		 VALUES (?, ?, ?, ?, ?)`,
		modelName, man.NBlocks, man.NStages, shardsDir, nowUnix(),
	)
	if err != nil {
		s.hfFail(jobID, uid, "db insert: "+err.Error())
		return
	}
	modelID, _ := res.LastInsertId()

	// Wipe staging including the downloaded checkpoint + intermediate gguf;
	// the splitter owns the registered shards now.
	_ = os.RemoveAll(stagingDir)

	now := nowUnix()
	_, _ = s.db.Exec(
		`UPDATE hf_imports SET status = 'done', model_id = ?, updated_at = ?, error = ''
		 WHERE id = ?`, modelID, now, jobID)
	s.hub.broadcastToUser(uid, "hf_progress", map[string]any{
		"job_id":   jobID,
		"status":   "done",
		"model_id": modelID,
	})
}
