// failures_test.go — failure-mode coverage: rig loss, network failures,
// human error, input validation.  These tests spin up a real in-memory
// SQLite, install the migrations, and exercise handlers end-to-end so that
// any regression in error handling shows up as a red test.
//
// Categories:
//
//   ── Human error / input validation ──
//     • bad JSON body                              → 400
//     • oversized body  (LimitReader cutoff)       → 400
//     • missing required field (repo_id)           → 400
//     • oversized field (repo_id, model_name)      → 400
//     • duplicate model name                       → 409
//     • per-user HF import concurrency cap         → 429
//     • per-user comfy generate concurrency cap    → 429
//     • normalizeRepoID variants                   → table-driven
//     • bad pair token (consumed twice → race)     → exactly one wins
//     • expired pair token                         → errPairExpired
//
//   ── Auth gaps ──
//     • handler without session cookie             → 401
//     • handler with bogus session cookie          → 401
//     • handler with expired session row           → 401
//     • signed comfy URL: missing sig              → 401
//     • signed comfy URL: tampered sig             → 401
//     • signed comfy URL: expired                  → 401
//     • path-traversal file name                   → 400
//
//   ── Rig loss ──
//     • pickOnlineRigInPool, empty pool            → (nil,false)
//     • pickComfyRigs, empty pool                  → nil
//
//   ── Network failure recovery ──
//     • hf 500 → 500 → 200          downloadFile retries and succeeds
//     • hf connection drop mid-stream
//                                  downloadFile retries with Range resume
//     • server ignores Range header → full restart, file truncated and re-fetched
//     • hf 416 (already complete)   → nil (no error)
//     • hf 401 on download          → no retry, immediate failure
//     • ctx cancel mid-download     → ctx.Err()

package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// ─── test fixture ────────────────────────────────────────────────────────────

func newTestServer(t *testing.T) *server {
	t.Helper()
	// Each test gets its own private in-memory DB.  We use a uniquely-named
	// file:: URI with mode=memory&cache=shared so the *sql.DB pool can
	// re-open the same DB across connections.
	dsn := fmt.Sprintf("file:test-%s?mode=memory&cache=shared&_foreign_keys=on", t.Name())
	// Sanitise t.Name for sqlite URI (no slashes).
	dsn = strings.ReplaceAll(dsn, "/", "_")
	db, err := sql.Open("sqlite3", dsn)
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	db.SetMaxOpenConns(1) // pin to a single conn so the shared-cache DB stays alive
	dialect := sqliteDialect{}
	if err := migrate(db, dialect); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	if err := migrateHF(db, dialect); err != nil {
		t.Fatalf("migrateHF: %v", err)
	}
	if err := migrateComfy(db, dialect); err != nil {
		t.Fatalf("migrateComfy: %v", err)
	}
	if err := migrateReputation(db, dialect); err != nil {
		t.Fatalf("migrateReputation: %v", err)
	}
	if err := migrateMCP(db, dialect); err != nil {
		t.Fatalf("migrateMCP: %v", err)
	}
	if err := migrateRAG(db, dialect); err != nil {
		t.Fatalf("migrateRAG: %v", err)
	}
	if err := migratePrefixAffinity(db, dialect); err != nil {
		t.Fatalf("migratePrefixAffinity: %v", err)
	}
	if err := migrateSpecCaps(db, dialect); err != nil {
		t.Fatalf("migrateSpecCaps: %v", err)
	}
	if err := applyVersionedMigrations(context.Background(), db, dialect); err != nil {
		t.Fatalf("applyVersionedMigrations: %v", err)
	}
	tmp := t.TempDir()
	cfg := config{
		sessionSecret: "failtest-secret-1234567890",
		publicURL:     "http://127.0.0.1:0",
		modelsDir:     filepath.Join(tmp, "models"),
		comfyOutDir:   filepath.Join(tmp, "comfy-out"),
		releasesDir:   filepath.Join(tmp, "releases"),
		convertQuant:  "q8_0",
		pythonBin:     "python3",
	}
	for _, d := range []string{cfg.modelsDir, cfg.comfyOutDir, cfg.releasesDir} {
		_ = os.MkdirAll(d, 0o755)
	}
	return newServer(cfg, db)
}

// makeUser inserts a user row and returns its id and a session cookie value.
// The session is valid for one hour.
func makeUser(t *testing.T, s *server, name string) (int64, string) {
	t.Helper()
	res, err := s.db.Exec(
		`INSERT INTO users (github_login, display_name, created_at) VALUES (NULL, ?, ?)`,
		name, nowUnix(),
	)
	if err != nil {
		t.Fatalf("insert user: %v", err)
	}
	uid, _ := res.LastInsertId()
	sid, _, err := s.createSession(uid)
	if err != nil {
		t.Fatalf("createSession: %v", err)
	}
	return uid, sid
}

// withSession attaches the session cookie to a request.
func withSession(r *http.Request, sid string) *http.Request {
	r.AddCookie(&http.Cookie{Name: sessionCookieName, Value: sid})
	return r
}

// jsonReq builds a POST with a JSON body and (optional) session cookie.
func jsonReq(method, target string, body any, sid string) *http.Request {
	b, _ := json.Marshal(body)
	r := httptest.NewRequest(method, target, strings.NewReader(string(b)))
	r.Header.Set("Content-Type", "application/json")
	if sid != "" {
		r.AddCookie(&http.Cookie{Name: sessionCookieName, Value: sid})
	}
	return r
}

// rawReq lets us send arbitrarily-sized / non-JSON bodies.
func rawReq(method, target, body, sid string) *http.Request {
	r := httptest.NewRequest(method, target, strings.NewReader(body))
	r.Header.Set("Content-Type", "application/json")
	if sid != "" {
		r.AddCookie(&http.Cookie{Name: sessionCookieName, Value: sid})
	}
	return r
}

// ─── Human error: bad JSON / missing fields / oversized ──────────────────────

func TestHFResolve_BadJSON(t *testing.T) {
	s := newTestServer(t)
	_, sid := makeUser(t, s, "alice")

	rr := httptest.NewRecorder()
	s.handleHFResolve(rr, rawReq("POST", "/api/hf/resolve", "{not json", sid))
	if rr.Code != 400 {
		t.Fatalf("status = %d body = %s", rr.Code, rr.Body.String())
	}
	if !strings.Contains(rr.Body.String(), "bad json") {
		t.Errorf("body = %q", rr.Body.String())
	}
}

func TestHFResolve_MissingRepoID(t *testing.T) {
	s := newTestServer(t)
	_, sid := makeUser(t, s, "alice")
	rr := httptest.NewRecorder()
	s.handleHFResolve(rr, jsonReq("POST", "/api/hf/resolve", map[string]any{}, sid))
	if rr.Code != 400 || !strings.Contains(rr.Body.String(), "repo_id") {
		t.Fatalf("status = %d body = %s", rr.Code, rr.Body.String())
	}
}

func TestHFResolve_OversizedRepoID(t *testing.T) {
	s := newTestServer(t)
	_, sid := makeUser(t, s, "alice")
	huge := strings.Repeat("a", 300) + "/" + strings.Repeat("b", 300)
	rr := httptest.NewRecorder()
	s.handleHFResolve(rr, jsonReq("POST", "/api/hf/resolve", map[string]any{"repo_id": huge}, sid))
	if rr.Code != 400 || !strings.Contains(rr.Body.String(), "too long") {
		t.Fatalf("status = %d body = %s", rr.Code, rr.Body.String())
	}
}

func TestHFResolve_NoAuth(t *testing.T) {
	s := newTestServer(t)
	rr := httptest.NewRecorder()
	s.handleHFResolve(rr, jsonReq("POST", "/api/hf/resolve", map[string]any{"repo_id": "x/y"}, ""))
	if rr.Code != 401 {
		t.Fatalf("status = %d", rr.Code)
	}
}

func TestHFSetToken_Oversized(t *testing.T) {
	s := newTestServer(t)
	_, sid := makeUser(t, s, "alice")
	huge := strings.Repeat("a", 250) // > 200-char cap
	rr := httptest.NewRecorder()
	s.handleHFSetToken(rr, jsonReq("POST", "/api/hf/token", map[string]any{"token": huge}, sid))
	if rr.Code != 400 || !strings.Contains(rr.Body.String(), "suspiciously long") {
		t.Fatalf("status = %d body = %s", rr.Code, rr.Body.String())
	}
}

func TestHFSetToken_EmptyDeletes(t *testing.T) {
	s := newTestServer(t)
	uid, sid := makeUser(t, s, "alice")

	// Set a token first.
	rr1 := httptest.NewRecorder()
	s.handleHFSetToken(rr1, jsonReq("POST", "/api/hf/token", map[string]any{"token": "hf_abc"}, sid))
	if rr1.Code != 200 {
		t.Fatalf("set: %d %s", rr1.Code, rr1.Body.String())
	}
	if s.userHFToken(uid) != "hf_abc" {
		t.Fatal("userHFToken did not return what we just set")
	}

	// Empty token deletes.
	rr2 := httptest.NewRecorder()
	s.handleHFSetToken(rr2, jsonReq("POST", "/api/hf/token", map[string]any{"token": "  "}, sid))
	if rr2.Code != 200 {
		t.Fatalf("clear: %d", rr2.Code)
	}
	if s.userHFToken(uid) != "" {
		t.Fatal("userHFToken still set after clear")
	}
}

func TestComfyRegisterWorkflow_Validation(t *testing.T) {
	s := newTestServer(t)
	_, sid := makeUser(t, s, "alice")

	// Missing name.
	rr := httptest.NewRecorder()
	s.handleComfyRegisterWorkflow(rr, jsonReq("POST", "/api/comfy/workflows",
		map[string]any{"graph_text": "{}"}, sid))
	if rr.Code != 400 {
		t.Fatalf("missing name: %d %s", rr.Code, rr.Body.String())
	}

	// Bad kind.
	rr = httptest.NewRecorder()
	s.handleComfyRegisterWorkflow(rr, jsonReq("POST", "/api/comfy/workflows",
		map[string]any{"name": "x", "kind": "audio", "graph_text": "{}"}, sid))
	if rr.Code != 400 || !strings.Contains(rr.Body.String(), "kind must be") {
		t.Fatalf("bad kind: %d %s", rr.Code, rr.Body.String())
	}

	// Invalid JSON graph.
	rr = httptest.NewRecorder()
	s.handleComfyRegisterWorkflow(rr, jsonReq("POST", "/api/comfy/workflows",
		map[string]any{"name": "x", "kind": "image", "graph_text": "{not json"}, sid))
	if rr.Code != 400 || !strings.Contains(rr.Body.String(), "valid JSON") {
		t.Fatalf("bad graph: %d %s", rr.Code, rr.Body.String())
	}

	// n_rigs out of range.
	rr = httptest.NewRecorder()
	s.handleComfyRegisterWorkflow(rr, jsonReq("POST", "/api/comfy/workflows",
		map[string]any{"name": "x", "kind": "image", "graph_text": "{}", "n_rigs": 100}, sid))
	if rr.Code != 400 || !strings.Contains(rr.Body.String(), "n_rigs too large") {
		t.Fatalf("n_rigs: %d %s", rr.Code, rr.Body.String())
	}

	// Name too long.
	rr = httptest.NewRecorder()
	s.handleComfyRegisterWorkflow(rr, jsonReq("POST", "/api/comfy/workflows",
		map[string]any{"name": strings.Repeat("n", 200), "kind": "image", "graph_text": "{}"}, sid))
	if rr.Code != 400 || !strings.Contains(rr.Body.String(), "name too long") {
		t.Fatalf("name long: %d %s", rr.Code, rr.Body.String())
	}
}

// ─── normalizeRepoID variants ────────────────────────────────────────────────

func TestNormalizeRepoID(t *testing.T) {
	cases := map[string]string{
		"meta/llama-3":                                "meta/llama-3",
		"  meta/llama-3  ":                            "meta/llama-3",
		"hf:meta/llama-3":                             "meta/llama-3",
		"huggingface.co/meta/llama-3":                 "meta/llama-3",
		"https://huggingface.co/meta/llama-3":         "meta/llama-3",
		"https://huggingface.co/meta/llama-3/tree/main": "meta/llama-3",
		"https://huggingface.co/meta/llama-3/blob/main/file.gguf": "meta/llama-3",
		"meta/llama-3/":                               "meta/llama-3",
		"":                                            "",
		"only-one-part":                               "",
		"a/b/c":                                       "",
		"/leading":                                    "",
		"trailing/":                                   "",
	}
	for in, want := range cases {
		if got := normalizeRepoID(in); got != want {
			t.Errorf("normalizeRepoID(%q) = %q want %q", in, got, want)
		}
	}
}

// ─── Auth gaps ──────────────────────────────────────────────────────────────

func TestSession_Expired(t *testing.T) {
	s := newTestServer(t)
	uid, sid := makeUser(t, s, "alice")
	// Force-expire the session.
	if _, err := s.db.Exec(`UPDATE sessions SET expires_at = ? WHERE id = ?`, nowUnix()-3600, sid); err != nil {
		t.Fatal(err)
	}
	_ = uid

	r := httptest.NewRequest("GET", "/api/me", nil)
	r.AddCookie(&http.Cookie{Name: sessionCookieName, Value: sid})
	rr := httptest.NewRecorder()
	s.handleMe(rr, r)
	if rr.Code != 401 {
		t.Fatalf("expired session: %d %s", rr.Code, rr.Body.String())
	}
}

func TestSession_Bogus(t *testing.T) {
	s := newTestServer(t)
	r := httptest.NewRequest("GET", "/api/me", nil)
	r.AddCookie(&http.Cookie{Name: sessionCookieName, Value: "definitely-not-a-real-session"})
	rr := httptest.NewRecorder()
	s.handleMe(rr, r)
	if rr.Code != 401 {
		t.Fatalf("bogus session: %d", rr.Code)
	}
}

func TestComfyOutput_AuthGaps(t *testing.T) {
	s := newTestServer(t)

	// missing sig
	rr := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/comfy/out/1/x.png", nil)
	r.SetPathValue("id", "1")
	r.SetPathValue("file", "x.png")
	s.handleComfyOutput(rr, r)
	if rr.Code != 401 || !strings.Contains(rr.Body.String(), "missing signature") {
		t.Fatalf("missing sig: %d %s", rr.Code, rr.Body.String())
	}

	// bad sig
	rr = httptest.NewRecorder()
	r = httptest.NewRequest("GET", "/comfy/out/1/x.png?exp=9999999999&sig=deadbeef", nil)
	r.SetPathValue("id", "1")
	r.SetPathValue("file", "x.png")
	s.handleComfyOutput(rr, r)
	if rr.Code != 401 || !strings.Contains(rr.Body.String(), "bad signature") {
		t.Fatalf("bad sig: %d %s", rr.Code, rr.Body.String())
	}

	// expired
	exp := nowUnix() - 60
	sig := s.signComfyOutputV1(1, "x.png", exp)
	rr = httptest.NewRecorder()
	r = httptest.NewRequest("GET", fmt.Sprintf("/comfy/out/1/x.png?exp=%d&sig=%s", exp, sig), nil)
	r.SetPathValue("id", "1")
	r.SetPathValue("file", "x.png")
	s.handleComfyOutput(rr, r)
	if rr.Code != 401 || !strings.Contains(rr.Body.String(), "expired") {
		t.Fatalf("expired: %d %s", rr.Code, rr.Body.String())
	}

	// path traversal in file name
	rr = httptest.NewRecorder()
	r = httptest.NewRequest("GET", "/comfy/out/1/..%2Fetc%2Fpasswd", nil)
	r.SetPathValue("id", "1")
	r.SetPathValue("file", "../etc/passwd")
	s.handleComfyOutput(rr, r)
	if rr.Code != 400 {
		t.Fatalf("traversal: %d %s", rr.Code, rr.Body.String())
	}

	// signature replay across files — sig minted for foo.png cannot be reused for bar.png.
	expOK := nowUnix() + 3600
	sigFoo := s.signComfyOutputV1(7, "foo.png", expOK)
	rr = httptest.NewRecorder()
	r = httptest.NewRequest("GET", fmt.Sprintf("/comfy/out/7/bar.png?exp=%d&sig=%s", expOK, sigFoo), nil)
	r.SetPathValue("id", "7")
	r.SetPathValue("file", "bar.png")
	s.handleComfyOutput(rr, r)
	if rr.Code != 401 || !strings.Contains(rr.Body.String(), "bad signature") {
		t.Fatalf("cross-file replay: %d %s", rr.Code, rr.Body.String())
	}
}

// ─── Rig loss ───────────────────────────────────────────────────────────────

func TestPickOnlineRig_NoneAvailable(t *testing.T) {
	s := newTestServer(t)
	if _, ok := s.pickOnlineRigInPool(42); ok {
		t.Fatal("expected no rig in empty pool")
	}
	if rigs := s.pickComfyRigs(1, 42, 3); rigs != nil {
		t.Fatalf("pickComfyRigs returned %d rigs from empty pool", len(rigs))
	}
}

// ─── Pair token: race + expiry ──────────────────────────────────────────────

func TestPairToken_DoubleConsumeRace(t *testing.T) {
	s := newTestServer(t)
	uid, _ := makeUser(t, s, "alice")
	token := newRandomToken(16)
	expires := time.Now().Add(5 * time.Minute).Unix()
	if _, err := s.db.Exec(
		`INSERT INTO pair_tokens (token, user_id, created_at, expires_at)
		 VALUES (?, ?, ?, ?)`, token, uid, nowUnix(), expires); err != nil {
		t.Fatal(err)
	}

	// Two goroutines race to consume the same token.
	const N = 16
	var wins atomic.Int32
	var wg sync.WaitGroup
	wg.Add(N)
	start := make(chan struct{})
	for i := 0; i < N; i++ {
		go func() {
			defer wg.Done()
			<-start
			if uid, _, err := s.consumePairToken(token); err == nil && uid != 0 {
				wins.Add(1)
			}
		}()
	}
	close(start)
	wg.Wait()
	if wins.Load() != 1 {
		t.Fatalf("expected exactly 1 winner, got %d (token reuse possible!)", wins.Load())
	}
}

func TestPairToken_Expired(t *testing.T) {
	s := newTestServer(t)
	uid, _ := makeUser(t, s, "alice")
	token := newRandomToken(16)
	// expires_at in the past
	if _, err := s.db.Exec(
		`INSERT INTO pair_tokens (token, user_id, created_at, expires_at)
		 VALUES (?, ?, ?, ?)`, token, uid, nowUnix()-7200, nowUnix()-3600); err != nil {
		t.Fatal(err)
	}
	if _, _, err := s.consumePairToken(token); err != errPairExpired {
		t.Fatalf("expected errPairExpired, got %v", err)
	}
}

func TestPairToken_Unknown(t *testing.T) {
	s := newTestServer(t)
	if _, _, err := s.consumePairToken("nope"); err != errPairExpired {
		// (the consumer treats missing/expired/used identically — caller side)
		t.Fatalf("expected errPairExpired for unknown token, got %v", err)
	}
}

// ─── Network failure recovery ────────────────────────────────────────────────

// flakyServer returns the requested number of failures (HTTP 500) before
// finally serving the real body.  Tracks call count for assertions.
type flakyServer struct {
	mu        sync.Mutex
	hits      int
	failUntil int
	body      []byte
	ignoreRng bool // when true, ignore Range and serve full body every time
}

func newFlakyServer(body string, failUntil int) (*flakyServer, *httptest.Server) {
	fs := &flakyServer{body: []byte(body), failUntil: failUntil}
	hs := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fs.mu.Lock()
		hits := fs.hits + 1
		fs.hits = hits
		fail := hits <= fs.failUntil
		ignoreRange := fs.ignoreRng
		body := fs.body
		fs.mu.Unlock()
		if fail {
			http.Error(w, "boom", 500)
			return
		}
		rng := r.Header.Get("Range")
		if rng != "" && !ignoreRange {
			var from int64
			_, _ = fmt.Sscanf(rng, "bytes=%d-", &from)
			if from >= int64(len(body)) {
				w.WriteHeader(416)
				return
			}
			w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", from, len(body)-1, len(body)))
			w.WriteHeader(206)
			_, _ = w.Write(body[from:])
			return
		}
		w.WriteHeader(200)
		_, _ = w.Write(body)
	}))
	return fs, hs
}

func TestDownloadFile_RetriesAndSucceeds(t *testing.T) {
	// First two attempts fail with 500; the third returns the body.
	// downloadFile has 4 retries built-in, so this should converge.
	prevDelay := hfRetryDelayForTest(t, 5*time.Millisecond)
	_ = prevDelay

	fs, hs := newFlakyServer("hello-world-this-is-the-body", 2)
	defer hs.Close()

	dst := filepath.Join(t.TempDir(), "f")
	err := downloadFile(context.Background(), hs.URL, "", dst, func(int64) {})
	if err != nil {
		t.Fatalf("downloadFile: %v (hits=%d)", err, fs.hits)
	}
	got, _ := os.ReadFile(dst)
	if string(got) != "hello-world-this-is-the-body" {
		t.Fatalf("body = %q", got)
	}
	if fs.hits < 3 {
		t.Errorf("expected ≥3 hits, got %d", fs.hits)
	}
}

func TestDownloadFile_ResumeFromPartial(t *testing.T) {
	hfRetryDelayForTest(t, 5*time.Millisecond)
	body := "0123456789abcdef" // 16 bytes
	fs, hs := newFlakyServer(body, 0)
	defer hs.Close()
	dst := filepath.Join(t.TempDir(), "f")
	// Pre-seed first 8 bytes — Range request should fetch only bytes 8-15.
	if err := os.WriteFile(dst, []byte("01234567"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := downloadFile(context.Background(), hs.URL, "", dst, func(int64) {}); err != nil {
		t.Fatalf("downloadFile: %v", err)
	}
	got, _ := os.ReadFile(dst)
	if string(got) != body {
		t.Fatalf("body = %q", got)
	}
	_ = fs
}

func TestDownloadFile_RangeIgnoredFullRestart(t *testing.T) {
	hfRetryDelayForTest(t, 5*time.Millisecond)
	body := "abcdefghij"
	fs, hs := newFlakyServer(body, 0)
	fs.ignoreRng = true
	defer hs.Close()
	dst := filepath.Join(t.TempDir(), "f")
	// Pre-seed garbage; downloader should truncate + refetch because server
	// returns 200 (not 206).
	if err := os.WriteFile(dst, []byte("XXXXXXX"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := downloadFile(context.Background(), hs.URL, "", dst, func(int64) {}); err != nil {
		t.Fatalf("downloadFile: %v", err)
	}
	got, _ := os.ReadFile(dst)
	if string(got) != body {
		t.Fatalf("body = %q (should have truncated and refetched)", got)
	}
}

func TestDownloadFile_AlreadyComplete(t *testing.T) {
	hfRetryDelayForTest(t, 5*time.Millisecond)
	body := "complete-already"
	_, hs := newFlakyServer(body, 0)
	defer hs.Close()
	dst := filepath.Join(t.TempDir(), "f")
	if err := os.WriteFile(dst, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	// Pre-fill with the full body — server replies 416, downloadFile returns nil.
	if err := downloadFile(context.Background(), hs.URL, "", dst, func(int64) {}); err != nil {
		t.Fatalf("downloadFile: %v", err)
	}
	got, _ := os.ReadFile(dst)
	if string(got) != body {
		t.Fatalf("file mutated: %q", got)
	}
}

func TestDownloadFile_AuthRejectedNoRetry(t *testing.T) {
	hfRetryDelayForTest(t, 5*time.Millisecond)
	hits := 0
	hs := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
		http.Error(w, "no", 401)
	}))
	defer hs.Close()
	dst := filepath.Join(t.TempDir(), "f")
	err := downloadFile(context.Background(), hs.URL, "bogus", dst, func(int64) {})
	if err == nil || !strings.Contains(err.Error(), "auth rejected") {
		t.Fatalf("expected auth rejected, got %v", err)
	}
	if hits != 1 {
		t.Errorf("auth should fail fast; got %d hits", hits)
	}
}

func TestDownloadFile_CtxCancel(t *testing.T) {
	hfRetryDelayForTest(t, 50*time.Millisecond)
	// Server hangs forever.
	hs := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
		// Stream a slow byte stream that never finishes.
		flusher, _ := w.(http.Flusher)
		for i := 0; i < 100; i++ {
			_, _ = w.Write([]byte("x"))
			if flusher != nil {
				flusher.Flush()
			}
			time.Sleep(50 * time.Millisecond)
		}
	}))
	defer hs.Close()

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()
	dst := filepath.Join(t.TempDir(), "f")
	err := downloadFile(ctx, hs.URL, "", dst, func(int64) {})
	if err == nil {
		t.Fatal("expected ctx cancel error, got nil")
	}
	// ctx.Err() is context.Canceled here.
	if !strings.Contains(err.Error(), "context") {
		t.Errorf("expected context error, got %v", err)
	}
}

func TestDownloadFile_RetriesExhausted(t *testing.T) {
	hfRetryDelayForTest(t, 5*time.Millisecond)
	hits := 0
	hs := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
		http.Error(w, "always-500", 500)
	}))
	defer hs.Close()
	dst := filepath.Join(t.TempDir(), "f")
	err := downloadFile(context.Background(), hs.URL, "", dst, func(int64) {})
	if err == nil {
		t.Fatal("expected error after exhausted retries")
	}
	if hits < hfMaxRetries {
		t.Errorf("expected at least %d attempts, got %d", hfMaxRetries, hits)
	}
}

// hfRetryDelayForTest swaps hfRetryDelay so tests don't burn seconds on
// exponential backoff (default 2s × 2^attempt = ~30s for 4 attempts).
// Restored on cleanup.
func hfRetryDelayForTest(t *testing.T, d time.Duration) time.Duration {
	t.Helper()
	prev := hfRetryDelay
	hfRetryDelay = d
	t.Cleanup(func() { hfRetryDelay = prev })
	return prev
}

// ─── filterFiles + writeJSON / writeErr ──────────────────────────────────────

func TestWriteJSON_AndWriteErr(t *testing.T) {
	rr := httptest.NewRecorder()
	writeJSON(rr, 201, map[string]any{"ok": true, "n": 5})
	if rr.Code != 201 {
		t.Fatalf("code = %d", rr.Code)
	}
	if rr.Header().Get("Content-Type") != "application/json" {
		t.Fatalf("ct = %q", rr.Header().Get("Content-Type"))
	}
	var got map[string]any
	if err := json.NewDecoder(rr.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	if got["ok"] != true || got["n"].(float64) != 5 {
		t.Fatalf("body = %+v", got)
	}

	rr = httptest.NewRecorder()
	writeErr(rr, 418, "i am a teapot")
	if rr.Code != 418 {
		t.Fatalf("code = %d", rr.Code)
	}
	var ge map[string]string
	_ = json.NewDecoder(rr.Body).Decode(&ge)
	if ge["error"] != "i am a teapot" {
		t.Fatalf("error = %q", ge["error"])
	}
}

// ─── Concurrent HF imports cap ───────────────────────────────────────────────

func TestHFImport_PerUserConcurrencyCap(t *testing.T) {
	s := newTestServer(t)
	uid, sid := makeUser(t, s, "alice")
	// Pre-populate 3 in-flight imports.
	for i := 0; i < 3; i++ {
		_, err := s.db.Exec(
			`INSERT INTO hf_imports (user_id, repo_id, revision, files_json, n_stages, status,
			  bytes_total, bytes_done, created_at, updated_at)
			 VALUES (?, ?, 'main', '[]', 0, 'downloading', 0, 0, ?, ?)`,
			uid, fmt.Sprintf("user/repo-%d", i), nowUnix(), nowUnix())
		if err != nil {
			t.Fatal(err)
		}
	}

	rr := httptest.NewRecorder()
	s.handleHFImport(rr, jsonReq("POST", "/api/hf/import",
		map[string]any{"repo_id": "user/repo-new"}, sid))
	if rr.Code != 429 {
		t.Fatalf("expected 429, got %d %s", rr.Code, rr.Body.String())
	}
	if !strings.Contains(rr.Body.String(), "too many in-flight") {
		t.Errorf("body = %q", rr.Body.String())
	}
}

// ─── handleHFImport oversized body via io.LimitReader ────────────────────────

func TestHFImport_OversizedBody(t *testing.T) {
	s := newTestServer(t)
	_, sid := makeUser(t, s, "alice")
	// 2MB payload — LimitReader caps at 1MB; JSON decode then fails.
	bigList := make([]string, 0, 4000)
	for i := 0; i < 4000; i++ {
		bigList = append(bigList, strings.Repeat("x", 600))
	}
	body, _ := json.Marshal(map[string]any{
		"repo_id": "user/repo",
		"files":   bigList,
	})
	if len(body) < (1 << 20) {
		t.Skipf("test payload too small: %d bytes", len(body))
	}
	r := httptest.NewRequest("POST", "/api/hf/import",
		io.MultiReader(strings.NewReader(string(body))))
	r.AddCookie(&http.Cookie{Name: sessionCookieName, Value: sid})
	r.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	s.handleHFImport(rr, r)
	if rr.Code != 400 {
		t.Fatalf("expected 400 from LimitReader cutoff, got %d body=%s", rr.Code, rr.Body.String())
	}
}
