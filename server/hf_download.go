package main

// HuggingFace model import.
//
// Flow:
//   1. POST /api/hf/token              — user stores their HF token (encrypted).
//   2. POST /api/hf/resolve            — resolve repo_id → file listing (GGUF only).
//   3. POST /api/hf/import             — kick off download job, returns job_id.
//   4. GET  /api/hf/jobs               — list this user's import jobs.
//   5. GET  /api/hf/jobs/{id}          — poll a specific job; live progress via WS.
//
// A successful import auto-registers the resulting GGUF as a model (calling the
// same split pipeline used by /api/models).  Partial-shard mode lets a rig
// fetch only the GGUF files it needs (used when a model is already split into
// per-stage shards on the HF side; e.g. `model-00001-of-00008.gguf`).
//
// Encryption at rest: HF tokens are AES-GCM encrypted using a key derived from
// cfg.sessionSecret + a per-row nonce.  If the session secret rotates, tokens
// are invalidated and the user is asked to re-enter — by design.

import (
	"context"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// ─── HTTP client ──────────────────────────────────────────────────────────

// hfClient is shared by all import jobs.  Long timeout — model files can be
// tens of GB and we stream them.  Per-request contexts handle cancellation.
var hfClient = &http.Client{
	Timeout: 0, // streaming — rely on per-request context
	Transport: &http.Transport{
		MaxIdleConns:        16,
		MaxIdleConnsPerHost: 8,
		IdleConnTimeout:     90 * time.Second,
	},
}

// hfAPIBase is a var (not const) so tests can point it at httptest.Server.
var hfAPIBase = "https://huggingface.co"

// hfRetryDelay is a var so tests can compress the exponential-backoff schedule
// (otherwise 4 retries × ~2s × 2^attempt = ~30s per failure-mode test).
var hfRetryDelay = 2 * time.Second

const (
	hfMaxRetries = 4
	hfUserAgent  = "distpool-importer/1.0"
)

// ─── DB migration ─────────────────────────────────────────────────────────

// migrateHF adds the two tables this module owns.  Called from init().  Idempotent.
func migrateHF(db *sql.DB, d sqlDialect) error {
	stmts := []string{
		`CREATE TABLE IF NOT EXISTS hf_tokens (
			user_id     INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
			nonce       BLOB NOT NULL,
			ciphertext  BLOB NOT NULL,
			updated_at  INTEGER NOT NULL
		)`,
		`CREATE TABLE IF NOT EXISTS hf_imports (
			id            INTEGER PRIMARY KEY AUTOINCREMENT,
			user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
			repo_id       TEXT NOT NULL,
			revision      TEXT NOT NULL DEFAULT 'main',
			files_json    TEXT NOT NULL,           -- JSON array of {path, size, downloaded}
			n_stages      INTEGER NOT NULL DEFAULT 0,  -- 0 = single-file (don't split)
			status        TEXT NOT NULL,           -- queued | downloading | splitting | done | failed | cancelled
			bytes_total   INTEGER NOT NULL DEFAULT 0,
			bytes_done    INTEGER NOT NULL DEFAULT 0,
			error         TEXT NOT NULL DEFAULT '',
			model_id      INTEGER REFERENCES models(id) ON DELETE SET NULL,
			created_at    INTEGER NOT NULL,
			updated_at    INTEGER NOT NULL
		)`,
		`CREATE INDEX IF NOT EXISTS idx_hf_imports_user ON hf_imports(user_id, created_at)`,
	}
	for _, s := range stmts {
		if _, err := db.Exec(d.RewriteDDL(s)); err != nil {
			return fmt.Errorf("hf migrate: %w", err)
		}
	}
	return nil
}

// ─── Token encryption ─────────────────────────────────────────────────────

// hfTokenKey derives a 32-byte AES key from the session secret.  Stable
// across process restarts as long as DIST_SESSION_SECRET is stable.
func (s *server) hfTokenKey() []byte {
	h := sha256.Sum256([]byte("hf-token:" + s.cfg.sessionSecret))
	return h[:]
}

func (s *server) encryptHFToken(plain string) (nonce, ct []byte, err error) {
	block, err := aes.NewCipher(s.hfTokenKey())
	if err != nil {
		return nil, nil, err
	}
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, nil, err
	}
	nonce = make([]byte, gcm.NonceSize())
	if _, err := rand.Read(nonce); err != nil {
		return nil, nil, err
	}
	ct = gcm.Seal(nil, nonce, []byte(plain), nil)
	return nonce, ct, nil
}

func (s *server) decryptHFToken(nonce, ct []byte) (string, error) {
	block, err := aes.NewCipher(s.hfTokenKey())
	if err != nil {
		return "", err
	}
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return "", err
	}
	pt, err := gcm.Open(nil, nonce, ct, nil)
	if err != nil {
		return "", err
	}
	return string(pt), nil
}

// userHFToken returns the user's stored HF token, or "" if none / decrypt
// failed.  When decryption fails — usually because DIST_SESSION_SECRET was
// rotated since the token was saved — we log a warning AND delete the row
// so the user gets a clean "no token" state on next call (and the dashboard
// can prompt them to paste a fresh one) instead of silently 401-ing every
// HF download forever.
func (s *server) userHFToken(uid int64) string {
	var nonce, ct []byte
	err := s.db.QueryRow(
		`SELECT nonce, ciphertext FROM hf_tokens WHERE user_id = ?`, uid,
	).Scan(&nonce, &ct)
	if err != nil {
		return ""
	}
	tok, err := s.decryptHFToken(nonce, ct)
	if err != nil {
		log.Printf("[hf] decrypt token for user %d failed (%v) — clearing row; "+
			"user must re-paste their HF token", uid, err)
		_, _ = s.db.Exec(`DELETE FROM hf_tokens WHERE user_id = ?`, uid)
		return ""
	}
	return tok
}

// ─── HF API: list + resolve ───────────────────────────────────────────────

type hfFileInfo struct {
	Path string `json:"path"`
	Size int64  `json:"size"`
	// `downloaded` is updated as bytes stream in; reported back to the UI.
	Downloaded int64 `json:"downloaded"`
	// SHA256 from HF's LFS pointer when available.  Empty for non-LFS
	// blobs (config.json, tokenizer.json, etc.) — those we trust by
	// size only.
	SHA256 string `json:"sha256,omitempty"`
}

// hfModelTreeResp is the subset of /api/models/{repo}/tree/{rev} we use.
//
// LFS-backed files include an `lfs` object with the SHA-256 OID — we
// capture it so the downloader can verify the bytes on disk match what
// HF declared.  Tiny files (config.json, etc.) are stored as Git blobs
// and have no oid, so Lfs stays nil; we fall through to an unverified
// download for those (the size check is the only guard).
type hfTreeEntry struct {
	Type string `json:"type"` // "file" | "directory"
	Path string `json:"path"`
	Size int64  `json:"size"`
	Lfs  *hfLFS `json:"lfs,omitempty"`
}

type hfLFS struct {
	OID  string `json:"oid"`
	Size int64  `json:"size"`
}

// listGGUF returns all .gguf files in a repo, with sizes.  Empty token works
// for public models; gated models require a valid HF token.
func hfListGGUF(ctx context.Context, repoID, revision, token string) ([]hfFileInfo, error) {
	if revision == "" {
		revision = "main"
	}
	// HF tree API is paginated only at >1000 entries — we ignore that here.
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
		return nil, fmt.Errorf("hf auth required for %s (status %d) — set a HF token", repoID, resp.StatusCode)
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
	for _, e := range entries {
		if e.Type != "file" {
			continue
		}
		if !strings.HasSuffix(strings.ToLower(e.Path), ".gguf") {
			continue
		}
		fi := hfFileInfo{Path: e.Path, Size: e.Size}
		if e.Lfs != nil {
			fi.SHA256 = e.Lfs.OID
		}
		out = append(out, fi)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no .gguf files found in %s — this repo may not have a GGUF build", repoID)
	}
	return out, nil
}

// hfResolveURL returns the direct download URL for a single file.
func hfResolveURL(repoID, revision, path string) string {
	if revision == "" {
		revision = "main"
	}
	return fmt.Sprintf("%s/%s/resolve/%s/%s",
		hfAPIBase, repoID, revision, path)
}

// ─── Downloader ───────────────────────────────────────────────────────────

// downloadFile streams a single HF file to dst, with HTTP range resumes on
// transient failures and per-chunk progress callbacks.  Honors ctx cancel.
//
// progress is invoked with (bytesDelta) — the caller adds to its running
// total.  Called frequently; must not block.
func downloadFile(ctx context.Context, url, token, dst string, progress func(int64)) error {
	// Resume support: if the destination exists and is non-empty, ask for
	// the rest with a Range header.  We tolerate servers that ignore Range
	// (rare on HF, but possible) by falling back to a full restart if the
	// returned size doesn't match the expected remainder.
	var resumeFrom int64
	if fi, err := os.Stat(dst); err == nil && fi.Size() > 0 {
		resumeFrom = fi.Size()
	}

	var lastErr error
	for attempt := 0; attempt < hfMaxRetries; attempt++ {
		if attempt > 0 {
			delay := hfRetryDelay * time.Duration(1<<attempt)
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(delay):
			}
			// Re-check resume point in case the file grew between attempts.
			if fi, err := os.Stat(dst); err == nil {
				resumeFrom = fi.Size()
			}
		}
		req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
		if err != nil {
			return err
		}
		req.Header.Set("User-Agent", hfUserAgent)
		if token != "" {
			req.Header.Set("Authorization", "Bearer "+token)
		}
		if resumeFrom > 0 {
			req.Header.Set("Range", fmt.Sprintf("bytes=%d-", resumeFrom))
		}
		resp, err := hfClient.Do(req)
		if err != nil {
			lastErr = err
			continue
		}
		// HF redirects to CDN; client follows redirects by default.
		if resp.StatusCode == 416 {
			// Already complete from a prior attempt.
			resp.Body.Close()
			return nil
		}
		if resp.StatusCode == 401 || resp.StatusCode == 403 {
			resp.Body.Close()
			return fmt.Errorf("hf auth rejected (status %d) — check your HF token", resp.StatusCode)
		}
		if resp.StatusCode != 200 && resp.StatusCode != 206 {
			body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
			resp.Body.Close()
			lastErr = fmt.Errorf("hf download status %d: %s", resp.StatusCode, string(body))
			continue
		}

		// Open dst — append iff we got 206 Partial Content, otherwise truncate
		// (server didn't honor Range or we're starting fresh).
		flag := os.O_CREATE | os.O_WRONLY
		if resp.StatusCode == 206 && resumeFrom > 0 {
			flag |= os.O_APPEND
		} else {
			flag |= os.O_TRUNC
			resumeFrom = 0
		}
		f, err := os.OpenFile(dst, flag, 0o644)
		if err != nil {
			resp.Body.Close()
			return err
		}

		copied, err := copyWithProgress(ctx, f, resp.Body, progress)
		_ = f.Close()
		resp.Body.Close()
		if err == nil {
			_ = copied
			return nil
		}
		// Network error mid-stream — loop will retry with resume.
		lastErr = err
		// Update resumeFrom for next iteration.
		if fi, statErr := os.Stat(dst); statErr == nil {
			resumeFrom = fi.Size()
		}
	}
	if lastErr == nil {
		lastErr = errors.New("hf download: exhausted retries")
	}
	return lastErr
}

// verifyFileSHA256 computes the SHA-256 of dst and compares against want.
// Returns nil on match, an error on mismatch or any I/O failure.  The
// hash is streamed (no whole-file load) so this works on multi-GB files.
//
// HF stores LFS object IDs as the SHA-256 of the raw file bytes (hex,
// lowercase), so we compare hex-encoded.  Files HF serves as Git blobs
// (no LFS pointer) have no published hash and are skipped by the caller
// — that path is handled in downloadOneAndVerify, not here.
func verifyFileSHA256(dst, want string) error {
	if want == "" {
		return nil
	}
	f, err := os.Open(dst)
	if err != nil {
		return fmt.Errorf("open for verify: %w", err)
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return fmt.Errorf("read for verify: %w", err)
	}
	got := hex.EncodeToString(h.Sum(nil))
	if !strings.EqualFold(got, want) {
		return fmt.Errorf("sha256 mismatch: got %s want %s", got, want)
	}
	return nil
}

// downloadOneAndVerify wraps downloadFile + verifyFileSHA256 with one
// extra retry: if the bytes arrived but the digest is wrong, the file
// is truncated and re-fetched once.  A second mismatch is a hard fail
// — by then the issue is upstream corruption, not a flaky CDN.
func downloadOneAndVerify(ctx context.Context, url, token, dst, wantSHA string, progress func(int64)) error {
	if err := downloadFile(ctx, url, token, dst, progress); err != nil {
		return err
	}
	if wantSHA == "" {
		return nil
	}
	if err := verifyFileSHA256(dst, wantSHA); err != nil {
		// One re-fetch: nuke the file and start fresh.  We charge the
		// progress meter for negative delta so the UI doesn't think we
		// gained extra bytes (the second download will replay them).
		if fi, statErr := os.Stat(dst); statErr == nil {
			progress(-fi.Size())
		}
		_ = os.Remove(dst)
		if err2 := downloadFile(ctx, url, token, dst, progress); err2 != nil {
			return fmt.Errorf("re-fetch after sha mismatch: %w (original: %v)", err2, err)
		}
		if err2 := verifyFileSHA256(dst, wantSHA); err2 != nil {
			return fmt.Errorf("sha mismatch persists after re-fetch: %w", err2)
		}
	}
	return nil
}

// downloadFilesParallel fans `files` across `concurrency` workers.  Each
// worker pulls one file at a time via downloadOneAndVerify, so the per-file
// resume + sha-check guarantees still apply.  On any worker error we cancel
// the per-batch context so in-flight peers stop quickly.
//
// progress is the shared throttled flusher — safe to call from many
// goroutines (it uses atomic + mutex internally).
func (s *server) downloadFilesParallel(
	ctx context.Context,
	files []hfFileInfo,
	concurrency int,
	repoID, revision, token, stagingDir string,
	progress func(int64),
) error {
	if concurrency <= 0 {
		concurrency = 1
	}

	// Resolve stagingDir once so we can reject path-traversal attempts.
	// HF tree responses are not fully trusted: a malicious repo could
	// publish a file with a "../../escape" path, and without this check
	// the import would write outside the staging tree.
	absStaging, err := filepath.Abs(stagingDir)
	if err != nil {
		return fmt.Errorf("staging abs: %w", err)
	}

	// Filter out files that are already complete (resume-from-prior-attempt).
	// We do this up front so each surviving entry actually needs network.
	type job struct{ idx int }
	jobs := make([]job, 0, len(files))
	for i := range files {
		f := &files[i]
		if f.Size <= 0 {
			continue
		}
		dst := filepath.Join(stagingDir, filepath.FromSlash(f.Path))
		// Re-derive the safe destination and require it to live under
		// absStaging.  Reject absolute paths and any ".." escape.
		absDst, absErr := filepath.Abs(dst)
		if absErr != nil {
			return fmt.Errorf("hf path %q: %w", f.Path, absErr)
		}
		if rel, relErr := filepath.Rel(absStaging, absDst); relErr != nil ||
			rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
			return fmt.Errorf("hf path %q escapes staging dir", f.Path)
		}
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			return fmt.Errorf("mkdir: %w", err)
		}
		if fi, err := os.Stat(dst); err == nil && fi.Size() == f.Size {
			// Already on disk; if HF gave us a sha, double-check before
			// claiming success.  A corrupted leftover from a crashed
			// previous run would otherwise quietly poison the model.
			if f.SHA256 != "" {
				if err := verifyFileSHA256(dst, f.SHA256); err != nil {
					_ = os.Remove(dst)
					jobs = append(jobs, job{idx: i})
					continue
				}
			}
			progress(f.Size)
			f.Downloaded = f.Size
			continue
		}
		jobs = append(jobs, job{idx: i})
	}
	if len(jobs) == 0 {
		return nil
	}

	batchCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	jobCh := make(chan job, len(jobs))
	for _, j := range jobs {
		jobCh <- j
	}
	close(jobCh)

	var (
		wg      sync.WaitGroup
		errOnce sync.Once
		firstEr error
	)
	for w := 0; w < concurrency; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobCh {
				if batchCtx.Err() != nil {
					return
				}
				f := &files[j.idx]
				dst := filepath.Join(stagingDir, filepath.FromSlash(f.Path))
				url := hfResolveURL(repoID, revision, f.Path)
				if err := downloadOneAndVerify(batchCtx, url, token, dst, f.SHA256, progress); err != nil {
					errOnce.Do(func() {
						firstEr = fmt.Errorf("download %s: %w", f.Path, err)
						cancel()
					})
					return
				}
				f.Downloaded = f.Size
			}
		}()
	}
	wg.Wait()
	if firstEr != nil {
		return firstEr
	}
	// If the outer context was cancelled before we got anywhere, surface
	// it explicitly — workers that observed the cancel bail silently, so
	// otherwise we'd return nil and the caller would think the import
	// succeeded with zero files written.
	if err := ctx.Err(); err != nil {
		return err
	}
	return nil
}

// copyWithProgress copies src→dst, invoking progress(delta) ~every 256 KiB.
// Returns total bytes copied.  Cancels on ctx.
func copyWithProgress(ctx context.Context, dst io.Writer, src io.Reader, progress func(int64)) (int64, error) {
	buf := make([]byte, 64*1024)
	var total int64
	var sinceFlush int64
	const flushAt = 256 * 1024
	for {
		select {
		case <-ctx.Done():
			return total, ctx.Err()
		default:
		}
		n, err := src.Read(buf)
		if n > 0 {
			nw, werr := dst.Write(buf[:n])
			if werr != nil {
				return total, werr
			}
			if nw != n {
				return total, io.ErrShortWrite
			}
			total += int64(n)
			sinceFlush += int64(n)
			if sinceFlush >= flushAt {
				if progress != nil {
					progress(sinceFlush)
				}
				sinceFlush = 0
			}
		}
		if err != nil {
			if sinceFlush > 0 && progress != nil {
				progress(sinceFlush)
			}
			if err == io.EOF {
				return total, nil
			}
			return total, err
		}
	}
}

// ─── Import job manager ───────────────────────────────────────────────────

// importJobs tracks in-flight cancellations and progress aggregation so that
// status frames can be pushed to subscribed browsers in real time.  Job state
// is canonically in SQLite; this is the live-progress overlay.
type importJobs struct {
	mu     sync.Mutex
	cancel map[int64]context.CancelFunc
}

func newImportJobs() *importJobs {
	return &importJobs{cancel: make(map[int64]context.CancelFunc)}
}

func (j *importJobs) register(id int64, cancel context.CancelFunc) {
	j.mu.Lock()
	defer j.mu.Unlock()
	j.cancel[id] = cancel
}

func (j *importJobs) cancelJob(id int64) bool {
	j.mu.Lock()
	defer j.mu.Unlock()
	c, ok := j.cancel[id]
	if !ok {
		return false
	}
	c()
	delete(j.cancel, id)
	return true
}

func (j *importJobs) finished(id int64) {
	j.mu.Lock()
	defer j.mu.Unlock()
	delete(j.cancel, id)
}

// ─── Handlers ─────────────────────────────────────────────────────────────

// POST /api/hf/token  {token}
func (s *server) handleHFSetToken(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	var body struct {
		Token string `json:"token"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 4<<10)).Decode(&body); err != nil {
		writeErr(w, 400, "bad json: "+err.Error())
		return
	}
	body.Token = strings.TrimSpace(body.Token)
	if body.Token == "" {
		// Delete the token.
		_, _ = s.db.Exec(`DELETE FROM hf_tokens WHERE user_id = ?`, u.ID)
		writeJSON(w, 200, map[string]any{"ok": true, "set": false})
		return
	}
	// Light validation: HF tokens are typically "hf_..." but accept any
	// non-empty string; let HF reject bad ones at use time.
	if len(body.Token) > 200 {
		writeErr(w, 400, "token suspiciously long")
		return
	}
	nonce, ct, err := s.encryptHFToken(body.Token)
	if err != nil {
		writeErr(w, 500, "encrypt: "+err.Error())
		return
	}
	_, err = s.db.Exec(
		`INSERT INTO hf_tokens (user_id, nonce, ciphertext, updated_at)
		 VALUES (?, ?, ?, ?)
		 ON CONFLICT(user_id) DO UPDATE SET
		   nonce = excluded.nonce,
		   ciphertext = excluded.ciphertext,
		   updated_at = excluded.updated_at`,
		u.ID, nonce, ct, nowUnix(),
	)
	if err != nil {
		writeErr(w, 500, "db: "+err.Error())
		return
	}
	writeJSON(w, 200, map[string]any{"ok": true, "set": true})
}

// GET /api/hf/token  → {set: bool}
func (s *server) handleHFGetToken(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	var n int
	_ = s.db.QueryRow(`SELECT COUNT(*) FROM hf_tokens WHERE user_id = ?`, u.ID).Scan(&n)
	writeJSON(w, 200, map[string]any{"set": n > 0})
}

// POST /api/hf/resolve  {repo_id, revision?}
func (s *server) handleHFResolve(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	var body struct {
		RepoID   string `json:"repo_id"`
		Revision string `json:"revision"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 64<<10)).Decode(&body); err != nil {
		writeErr(w, 400, "bad json: "+err.Error())
		return
	}
	body.RepoID = normalizeRepoID(body.RepoID)
	if body.RepoID == "" {
		writeErr(w, 400, "repo_id required")
		return
	}
	if len(body.RepoID) > 256 || len(body.Revision) > 128 {
		writeErr(w, 400, "repo_id or revision too long")
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()
	token := s.userHFToken(u.ID)
	files, convertible, err := getHFCoalescer().resolve(ctx, body.RepoID, body.Revision, token,
		func(inner context.Context) ([]hfFileInfo, bool, error) {
			f, e := hfListGGUF(inner, body.RepoID, body.Revision, token)
			if e != nil && strings.Contains(e.Error(), "no .gguf files") {
				f2, e2 := hfListConvertible(inner, body.RepoID, body.Revision, token)
				if e2 != nil {
					return nil, false, fmt.Errorf("%s — convert fallback: %s", e.Error(), e2.Error())
				}
				return f2, true, nil
			}
			return f, false, e
		})
	if err != nil {
		writeErr(w, 502, err.Error())
		return
	}
	var total int64
	for _, f := range files {
		total += f.Size
	}
	writeJSON(w, 200, map[string]any{
		"repo_id":     body.RepoID,
		"revision":    body.Revision,
		"files":       files,
		"bytes_total": total,
		"convertible": convertible,
	})
}

// POST /api/hf/import  {repo_id, revision?, files?: [path,...], n_stages?, model_name?}
//
// If `files` is omitted, all .gguf files are downloaded.  If `n_stages > 0`,
// the largest downloaded file is fed into the splitter; otherwise the model
// is registered as a single-shard model (n_stages defaults to 1).
func (s *server) handleHFImport(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	var body struct {
		RepoID    string   `json:"repo_id"`
		Revision  string   `json:"revision"`
		Files     []string `json:"files"`
		NStages   int      `json:"n_stages"`
		ModelName string   `json:"model_name"`
		// Convert: when true, treat the repo as a raw HF checkpoint and run
		// convert_hf_to_gguf.py to produce a GGUF before splitting.  Auto-
		// enabled when the repo contains no .gguf files.
		Convert      bool   `json:"convert"`
		ConvertQuant string `json:"convert_quant"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&body); err != nil {
		writeErr(w, 400, "bad json: "+err.Error())
		return
	}
	body.RepoID = normalizeRepoID(body.RepoID)
	if body.RepoID == "" {
		writeErr(w, 400, "repo_id required")
		return
	}
	if len(body.RepoID) > 256 {
		writeErr(w, 400, "repo_id too long")
		return
	}
	if len(body.Files) > 4096 {
		writeErr(w, 400, "too many files (max 4096)")
		return
	}
	if len(body.ModelName) > 200 {
		writeErr(w, 400, "model_name too long (max 200)")
		return
	}
	if body.NStages < 0 || body.NStages > 256 {
		writeErr(w, 400, "n_stages must be 0..256")
		return
	}
	// Concurrent import limit per user — prevents one tab from clogging
	// disk and the converter queue.
	var inflight int
	_ = s.db.QueryRow(
		`SELECT COUNT(*) FROM hf_imports
		 WHERE user_id = ? AND status IN ('queued','downloading','converting','splitting')`,
		u.ID).Scan(&inflight)
	if inflight >= 3 {
		writeErr(w, 429, "too many in-flight HF imports (limit 3)")
		return
	}
	if body.ModelName == "" {
		body.ModelName = body.RepoID
	}

	// Validate model name uniqueness up-front so we can fail fast.
	var existing int64
	_ = s.db.QueryRow(`SELECT id FROM models WHERE name = ?`, body.ModelName).Scan(&existing)
	if existing != 0 {
		writeErr(w, 409, "model name already exists: "+body.ModelName)
		return
	}

	// List & filter files.  GGUF-direct first; on miss (or explicit Convert
	// flag), fall back to the convertible-checkpoint listing.
	ctx0, cancel0 := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel0()
	useConvert := body.Convert
	var allFiles []hfFileInfo
	var err error
	if !useConvert {
		allFiles, err = hfListGGUF(ctx0, body.RepoID, body.Revision, s.userHFToken(u.ID))
		if err != nil && strings.Contains(err.Error(), "no .gguf files") {
			// Auto-fallback: the repo exists but has no prebuilt GGUF.
			useConvert = true
		} else if err != nil {
			writeErr(w, 502, err.Error())
			return
		}
	}
	if useConvert {
		allFiles, err = hfListConvertible(ctx0, body.RepoID, body.Revision, s.userHFToken(u.ID))
		if err != nil {
			writeErr(w, 502, err.Error())
			return
		}
	}
	files := filterFiles(allFiles, body.Files)
	if len(files) == 0 {
		writeErr(w, 400, "no matching files in repo")
		return
	}
	var total int64
	for _, f := range files {
		total += f.Size
	}

	filesJSON, _ := json.Marshal(files)
	revision := body.Revision
	if revision == "" {
		revision = "main"
	}
	res, err := s.db.Exec(
		`INSERT INTO hf_imports
		 (user_id, repo_id, revision, files_json, n_stages, status,
		  bytes_total, bytes_done, created_at, updated_at)
		 VALUES (?, ?, ?, ?, ?, 'queued', ?, 0, ?, ?)`,
		u.ID, body.RepoID, revision, string(filesJSON),
		body.NStages, total, nowUnix(), nowUnix(),
	)
	if err != nil {
		writeErr(w, 500, "db: "+err.Error())
		return
	}
	jobID, _ := res.LastInsertId()

	// Kick off the actual download in a goroutine; ctx is detached from
	// the HTTP request so the user can navigate away.
	jobCtx, jobCancel := context.WithCancel(context.Background())
	s.hfJobs.register(jobID, jobCancel)
	if useConvert {
		quant := body.ConvertQuant
		if quant == "" {
			quant = s.cfg.convertQuant
		}
		go s.runHFConvertImport(jobCtx, u.ID, jobID, body.RepoID, revision, body.ModelName, files, body.NStages, quant)
	} else {
		go s.runHFImport(jobCtx, u.ID, jobID, body.RepoID, revision, body.ModelName, files, body.NStages)
	}

	writeJSON(w, 200, map[string]any{
		"job_id":      jobID,
		"repo_id":     body.RepoID,
		"bytes_total": total,
		"files":       files,
	})
}

// GET /api/hf/jobs
func (s *server) handleHFListJobs(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	rows, err := s.db.Query(
		`SELECT id, repo_id, revision, status, bytes_total, bytes_done,
		        error, model_id, created_at, updated_at
		 FROM hf_imports
		 WHERE user_id = ?
		 ORDER BY id DESC LIMIT 100`, u.ID)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	defer rows.Close()
	type jobRow struct {
		ID         int64  `json:"id"`
		RepoID     string `json:"repo_id"`
		Revision   string `json:"revision"`
		Status     string `json:"status"`
		BytesTotal int64  `json:"bytes_total"`
		BytesDone  int64  `json:"bytes_done"`
		Error      string `json:"error"`
		ModelID    *int64 `json:"model_id"`
		CreatedAt  int64  `json:"created_at"`
		UpdatedAt  int64  `json:"updated_at"`
	}
	var out []jobRow
	for rows.Next() {
		var j jobRow
		var mid sql.NullInt64
		if err := rows.Scan(&j.ID, &j.RepoID, &j.Revision, &j.Status,
			&j.BytesTotal, &j.BytesDone, &j.Error, &mid,
			&j.CreatedAt, &j.UpdatedAt); err != nil {
			continue
		}
		if mid.Valid {
			id := mid.Int64
			j.ModelID = &id
		}
		out = append(out, j)
	}
	writeJSON(w, 200, map[string]any{"jobs": out})
}

// GET /api/hf/jobs/{id}
func (s *server) handleHFJobDetail(w http.ResponseWriter, r *http.Request) {
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
		repoID    string
		rev       string
		status    string
		filesJSON string
		bt, bd    int64
		errMsg    string
		mid       sql.NullInt64
	)
	err = s.db.QueryRow(
		`SELECT user_id, repo_id, revision, status, files_json,
		        bytes_total, bytes_done, error, model_id
		 FROM hf_imports WHERE id = ?`, id,
	).Scan(&uid, &repoID, &rev, &status, &filesJSON, &bt, &bd, &errMsg, &mid)
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
	var files []hfFileInfo
	_ = json.Unmarshal([]byte(filesJSON), &files)
	resp := map[string]any{
		"id":          id,
		"repo_id":     repoID,
		"revision":    rev,
		"status":      status,
		"bytes_total": bt,
		"bytes_done":  bd,
		"error":       errMsg,
		"files":       files,
	}
	if mid.Valid {
		resp["model_id"] = mid.Int64
	}
	writeJSON(w, 200, resp)
}

// POST /api/hf/jobs/{id}/cancel
func (s *server) handleHFJobCancel(w http.ResponseWriter, r *http.Request) {
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
	if err := s.db.QueryRow(`SELECT user_id FROM hf_imports WHERE id = ?`, id).Scan(&uid); err != nil {
		writeErr(w, 404, "no such job")
		return
	}
	if uid != u.ID {
		writeErr(w, 403, "not your job")
		return
	}
	if !s.hfJobs.cancelJob(id) {
		// Already terminated.
		writeJSON(w, 200, map[string]any{"ok": true, "already_done": true})
		return
	}
	writeJSON(w, 200, map[string]any{"ok": true})
}

// ─── Worker ───────────────────────────────────────────────────────────────

// runHFImport streams files in sequence (avoids hammering HF + makes progress
// easy to reason about), then optionally splits and registers the model.
// Pushes 'hf_progress' events to the browser hub every ~256 KiB.
func (s *server) runHFImport(
	ctx context.Context,
	uid, jobID int64,
	repoID, revision, modelName string,
	files []hfFileInfo,
	nStages int,
) {
	defer s.hfJobs.finished(jobID)
	token := s.userHFToken(uid)

	stagingDir := filepath.Join(s.cfg.modelsDir, "_hf_staging",
		sanitizeModelName(repoID)+"-"+strconv.FormatInt(jobID, 10))
	if err := os.MkdirAll(stagingDir, 0o755); err != nil {
		s.hfFail(jobID, uid, "mkdir staging: "+err.Error())
		return
	}

	// On any error path past this point, drop the partial staging tree so a
	// failed import doesn't leak GB of partial bytes.  Success path zeroes
	// the flag after the explicit RemoveAll lower down (we wipe immediately
	// after splitter so disk pressure is released before model registration).
	cleanupStaging := true
	defer func() {
		if cleanupStaging {
			_ = os.RemoveAll(stagingDir)
		}
	}()

	s.hfSetStatus(jobID, uid, "downloading", "")

	// Aggregate progress across files.  totalDone is summed across all
	// concurrent workers; flushMu serialises the throttle check + the
	// downstream DB/broadcast write so two workers can't double-fire on
	// the same tick.
	var (
		totalDone atomic.Int64
		flushMu   sync.Mutex
		lastFlush = time.Now()
	)
	flushProgress := func(delta int64) {
		v := totalDone.Add(delta)
		flushMu.Lock()
		// Throttle DB writes + broadcasts to ~5 Hz.
		if time.Since(lastFlush) < 200*time.Millisecond {
			flushMu.Unlock()
			return
		}
		lastFlush = time.Now()
		flushMu.Unlock()
		s.hfUpdateBytes(jobID, v)
		s.hub.broadcastToUser(uid, "hf_progress", map[string]any{
			"job_id":     jobID,
			"bytes_done": v,
			"status":     "downloading",
		})
	}

	// Parallel worker pool.  HF's CDN handles concurrent connections from
	// the same client fine — they're the same fronting CloudFront that
	// serves browser downloads.  Concurrency=4 is the sweet spot in
	// practice: enough to hide per-file TLS handshake + slow-start, not
	// so many that we trigger rate limiting on gated repos.
	const hfConcurrency = 4
	if err := s.downloadFilesParallel(ctx, files, hfConcurrency, repoID, revision, token, stagingDir, flushProgress); err != nil {
		if errors.Is(err, context.Canceled) {
			s.hfSetStatus(jobID, uid, "cancelled", "cancelled by user")
			_ = os.RemoveAll(stagingDir)
			return
		}
		s.hfFail(jobID, uid, err.Error())
		return
	}

	// Final flush.
	s.hfUpdateBytes(jobID, totalDone.Load())

	// Pick the file we'll register.  Heuristic: the largest .gguf — for most
	// "single-file" GGUF repos this is the actual weights; for split repos
	// the first shard usually has the manifest header and works as the
	// llama-split-gguf input (since the splitter operates on a single file,
	// pre-split repos need their files concatenated or fed directly — we
	// register the first one and rely on the partial-layer-load patch).
	largest := pickLargest(files, stagingDir)
	if largest == "" {
		s.hfFail(jobID, uid, "no file landed on disk")
		return
	}

	// Move into the canonical models layout and register.
	s.hfSetStatus(jobID, uid, "splitting", "")
	if nStages == 0 {
		nStages = 1
	}

	shardsDir := filepath.Join(s.cfg.modelsDir, sanitizeModelName(modelName))
	if err := os.MkdirAll(shardsDir, 0o755); err != nil {
		s.hfFail(jobID, uid, "mkdir shards: "+err.Error())
		return
	}

	// Run splitter (same binary used by /api/models).
	if err := runSplitter(s.cfg.splitterBin, largest, shardsDir, nStages); err != nil {
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

	// Clean up staging (the source GGUF has been split — we don't need both copies).
	_ = os.RemoveAll(stagingDir)
	cleanupStaging = false

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

func (s *server) hfSetStatus(jobID, uid int64, status, errMsg string) {
	_, _ = s.db.Exec(
		`UPDATE hf_imports SET status = ?, error = ?, updated_at = ? WHERE id = ?`,
		status, errMsg, nowUnix(), jobID,
	)
	s.hub.broadcastToUser(uid, "hf_progress", map[string]any{
		"job_id": jobID,
		"status": status,
		"error":  errMsg,
	})
}

func (s *server) hfUpdateBytes(jobID, bytesDone int64) {
	_, _ = s.db.Exec(
		`UPDATE hf_imports SET bytes_done = ?, updated_at = ? WHERE id = ?`,
		bytesDone, nowUnix(), jobID,
	)
}

func (s *server) hfFail(jobID, uid int64, msg string) {
	_, _ = s.db.Exec(
		`UPDATE hf_imports SET status = 'failed', error = ?, updated_at = ? WHERE id = ?`,
		msg, nowUnix(), jobID,
	)
	s.hub.broadcastToUser(uid, "hf_progress", map[string]any{
		"job_id": jobID,
		"status": "failed",
		"error":  msg,
	})
}

// ─── Helpers ──────────────────────────────────────────────────────────────

// normalizeRepoID accepts either "org/name", a full HF URL, or a "hf:" prefix.
func normalizeRepoID(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	s = strings.TrimPrefix(s, "hf:")
	s = strings.TrimPrefix(s, "huggingface.co/")
	if strings.HasPrefix(s, "https://huggingface.co/") {
		s = strings.TrimPrefix(s, "https://huggingface.co/")
	}
	// Strip any trailing slash or /tree/... suffix.
	if i := strings.Index(s, "/tree/"); i >= 0 {
		s = s[:i]
	}
	if i := strings.Index(s, "/blob/"); i >= 0 {
		s = s[:i]
	}
	s = strings.TrimSuffix(s, "/")
	// repo IDs are "<org-or-user>/<name>"; reject anything else.
	parts := strings.Split(s, "/")
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		return ""
	}
	return s
}

// filterFiles returns the subset of `all` whose Path is in `selected`, in the
// order given by `all` (so behavior is deterministic).  Empty `selected`
// means "all of them".  Unknown paths in `selected` are silently dropped.
func filterFiles(all []hfFileInfo, selected []string) []hfFileInfo {
	if len(selected) == 0 {
		return all
	}
	set := make(map[string]struct{}, len(selected))
	for _, s := range selected {
		set[s] = struct{}{}
	}
	out := make([]hfFileInfo, 0, len(selected))
	for _, f := range all {
		if _, ok := set[f.Path]; ok {
			out = append(out, f)
		}
	}
	return out
}

// pickLargest returns the absolute path of the largest landed file in
// stagingDir, matching one of the files in `files` (so we don't pick up
// stray junk).
func pickLargest(files []hfFileInfo, stagingDir string) string {
	var best string
	var bestSize int64 = -1
	for _, f := range files {
		p := filepath.Join(stagingDir, filepath.FromSlash(f.Path))
		fi, err := os.Stat(p)
		if err != nil {
			continue
		}
		if fi.Size() > bestSize {
			bestSize = fi.Size()
			best = p
		}
	}
	return best
}

// runSplitter is a thin wrapper around llama-split-gguf so the HF importer
// and the manual /api/models registrar share one call site.
func runSplitter(splitter, src, dst string, n int) error {
	if splitter == "" {
		return errors.New("splitter binary not configured (DIST_SPLITTER)")
	}
	if _, err := os.Stat(splitter); err != nil {
		return fmt.Errorf("splitter binary not found at %s: %w", splitter, err)
	}
	out, err := exec.Command(splitter, src, dst, strconv.Itoa(n)).CombinedOutput()
	if err != nil {
		return fmt.Errorf("%w: %s", err, string(out))
	}
	return nil
}
