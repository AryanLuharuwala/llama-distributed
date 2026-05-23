// helpers_test.go — unit + A/B failure-mode coverage for the helpers most
// likely to silently break.  Intentionally avoids requiring a real DB; the
// pieces that touch SQL are exercised by integration test runs against a
// disposable distpool.sqlite (see Makefile target `test-integration`).
//
// Coverage map:
//
//   parseShortTarget          install.go              — installer URL parsing
//   isSafeOutputFile          comfy.go                — path-traversal defense
//   signComfyOutput / URL     comfy.go                — HMAC sign + verify + tamper
//   substituteComfyParams     comfy.go                — $PROMPT/$MODEL/$SEED/$W/$H templating
//   hfListGGUF                hf_download.go          — happy path + 401 + 404 + empty
//   hfListConvertible         hf_convert.go           — keep list + reject .onnx etc
//   encrypt/decrypt HFToken   hf_download.go          — AES-GCM roundtrip + tamper
//   comfyJobs                 comfy.go                — register/cancel/finish lifecycle
//   importJobs                hf_download.go          — register/cancel/finish lifecycle
//   runConverter              hf_convert.go           — missing-converter error
//   httpToWS                  server.go               — protocol upgrade mapping
//
// Run with:  go test ./...

package main

import (
	"context"
	"crypto/hmac"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// ─── parseShortTarget ───────────────────────────────────────────────────────

func TestParseShortTarget(t *testing.T) {
	cases := []struct {
		in   string
		os_  string
		arch string
		acc  string
		ext  string
		ok   bool
	}{
		{"linux-x86_64-cpu.tar.gz", "linux", "x86_64", "cpu", ".tar.gz", true},
		{"linux-x86_64-cuda.tar.gz", "linux", "x86_64", "cuda", ".tar.gz", true},
		{"macos-arm64-metal.tar.gz", "macos", "arm64", "metal", ".tar.gz", true},
		{"windows-x86_64-cuda.zip", "windows", "x86_64", "cuda", ".zip", true},
		// failures:
		{"linux-x86_64.tar.gz", "", "", "", "", false},                  // missing accel
		{"linux-x86_64-cpu-extra.tar.gz", "", "", "", "", false},        // too many parts
		{"linux-x86_64-cpu.rar", "", "", "", "", false},                 // bad ext
		{"", "", "", "", "", false},
	}
	for _, c := range cases {
		os_, arch, acc, ext, ok := parseShortTarget(c.in)
		if ok != c.ok || os_ != c.os_ || arch != c.arch || acc != c.acc || ext != c.ext {
			t.Errorf("parseShortTarget(%q) = (%q,%q,%q,%q,%v) want (%q,%q,%q,%q,%v)",
				c.in, os_, arch, acc, ext, ok, c.os_, c.arch, c.acc, c.ext, c.ok)
		}
	}
}

// ─── isSafeOutputFile ───────────────────────────────────────────────────────

func TestIsSafeOutputFile(t *testing.T) {
	good := []string{"a.png", "out.jpg", "v.mp4", "frame.webp", "clip.webm", "anim.gif"}
	bad := []string{
		"",
		".",
		"..",
		"../etc/passwd",
		"a/b.png",
		"a\\b.png",
		"file.txt",      // disallowed ext
		"file.exe",      // disallowed ext
		"file",          // no ext
		"a.png/",        // trailing slash
	}
	for _, f := range good {
		if !isSafeOutputFile(f) {
			t.Errorf("isSafeOutputFile(%q) = false, want true", f)
		}
	}
	for _, f := range bad {
		if isSafeOutputFile(f) {
			t.Errorf("isSafeOutputFile(%q) = true, want false", f)
		}
	}
}

// ─── signComfyOutput sign + verify + tamper ─────────────────────────────────

func TestSignComfyOutputRoundtripAndTamper(t *testing.T) {
	s := &server{cfg: config{sessionSecret: "test-secret-1234", publicURL: "https://x.example.com"}}

	// Sign across distinct (uid, id, file, exp) tuples — every change MUST
	// flip the signature.  Mirrors the four collision axes for v2.
	exp := time.Now().Add(time.Hour).Unix()
	sig1 := s.signComfyOutputV2(7, 42, "out.png", exp)
	sig2 := s.signComfyOutputV2(7, 42, "out.png", exp+1)
	sig3 := s.signComfyOutputV2(7, 43, "out.png", exp)
	sig4 := s.signComfyOutputV2(7, 42, "OUT.png", exp)
	sig5 := s.signComfyOutputV2(8, 42, "out.png", exp)
	if sig1 == sig2 || sig1 == sig3 || sig1 == sig4 || sig1 == sig5 {
		t.Fatal("HMAC collision across distinct (uid,id,file,exp) tuples")
	}

	if got := s.signComfyOutputV2(7, 42, "out.png", exp); got != sig1 {
		t.Fatalf("signComfyOutputV2 non-deterministic: got %q want %q", got, sig1)
	}

	// Changing the secret invalidates the signature.
	s2 := &server{cfg: config{sessionSecret: "different-secret", publicURL: "https://x.example.com"}}
	if s2.signComfyOutputV2(7, 42, "out.png", exp) == sig1 {
		t.Fatal("signature did not change when secret rotated")
	}

	// v1 must produce a different signature than v2 for the same (id,file,exp)
	// — the "comfyv2/" prefix is what prevents cross-version replay.
	if s.signComfyOutputV1(42, "out.png", exp) == s.signComfyOutputV2(7, 42, "out.png", exp) {
		t.Fatal("v1 and v2 signatures collide; cross-version replay is possible")
	}

	// signComfyOutputURL wraps it in a v2 URL with v/uid/exp/sig query params.
	u := s.signComfyOutputURL(7, 42, "out.png", time.Hour)
	parsed, err := url.Parse(u)
	if err != nil {
		t.Fatalf("signComfyOutputURL produced invalid URL: %v", err)
	}
	if parsed.Path != "/comfy/out/42/out.png" {
		t.Errorf("URL path = %q", parsed.Path)
	}
	q := parsed.Query()
	if q.Get("v") != "2" {
		t.Errorf("URL is not v2: v=%q", q.Get("v"))
	}
	if q.Get("uid") != "7" {
		t.Errorf("URL uid = %q want 7", q.Get("uid"))
	}
	if q.Get("exp") == "" || q.Get("sig") == "" {
		t.Error("URL missing exp/sig query params")
	}
}

// ─── substituteComfyParams ──────────────────────────────────────────────────

func TestSubstituteComfyParams(t *testing.T) {
	graph := `{
	  "3": {"inputs": {"text": "$PROMPT", "model": "$MODEL", "seed": "$SEED",
	                    "w": "$WIDTH", "h": "$HEIGHT"}}
	}`
	out, err := substituteComfyParams(graph, "a cat on a couch", `{"size":"512x768","seed":1234}`, "sdxl.safetensors")
	if err != nil {
		t.Fatalf("substituteComfyParams: %v", err)
	}
	var parsed map[string]any
	if err := json.Unmarshal([]byte(out), &parsed); err != nil {
		t.Fatalf("output not valid JSON: %v\n%s", err, out)
	}
	n3, _ := parsed["3"].(map[string]any)
	in, _ := n3["inputs"].(map[string]any)
	if got, _ := in["text"].(string); got != "a cat on a couch" {
		t.Errorf("text = %q", got)
	}
	if got, _ := in["model"].(string); got != "sdxl.safetensors" {
		t.Errorf("model = %q", got)
	}
	// seed comes through as a JSON number → float64 after json.Unmarshal.
	if got, _ := in["seed"].(float64); int64(got) != 1234 {
		t.Errorf("seed = %v want 1234", got)
	}
	if got, _ := in["w"].(float64); int(got) != 512 {
		t.Errorf("w = %v want 512", got)
	}
	if got, _ := in["h"].(float64); int(got) != 768 {
		t.Errorf("h = %v want 768", got)
	}
}

func TestSubstituteComfyParamsBadGraph(t *testing.T) {
	_, err := substituteComfyParams("{not json", "p", "{}", "m")
	if err == nil {
		t.Fatal("expected error on malformed graph JSON")
	}
}

// ─── hfListGGUF / hfListConvertible against httptest server ─────────────────

// fakeHFServer returns a stub HF API serving a fixed tree at /api/models/.../tree/main.
func fakeHFServer(t *testing.T, entries []hfTreeEntry, statusCode int) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if statusCode != 0 && statusCode != 200 {
			w.WriteHeader(statusCode)
			return
		}
		if !strings.HasPrefix(r.URL.Path, "/api/models/") {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(entries)
	}))
	return srv
}

func withHFBase(t *testing.T, base string, fn func()) {
	t.Helper()
	prev := hfAPIBase
	hfAPIBase = base
	defer func() { hfAPIBase = prev }()
	fn()
}

func TestHFListGGUF_HappyPath(t *testing.T) {
	srv := fakeHFServer(t, []hfTreeEntry{
		{Type: "file", Path: "README.md", Size: 100},
		{Type: "file", Path: "model.gguf", Size: 5000},
		{Type: "file", Path: "split.q4_0.gguf", Size: 2500},
		{Type: "directory", Path: "weights", Size: 0},
	}, 200)
	defer srv.Close()
	withHFBase(t, srv.URL, func() {
		files, err := hfListGGUF(context.Background(), "org/model", "", "")
		if err != nil {
			t.Fatalf("hfListGGUF: %v", err)
		}
		if len(files) != 2 {
			t.Fatalf("expected 2 .gguf files, got %d: %+v", len(files), files)
		}
	})
}

func TestHFListGGUF_AuthRequired(t *testing.T) {
	srv := fakeHFServer(t, nil, 401)
	defer srv.Close()
	withHFBase(t, srv.URL, func() {
		_, err := hfListGGUF(context.Background(), "org/gated", "", "")
		if err == nil || !strings.Contains(err.Error(), "auth required") {
			t.Fatalf("expected auth required, got %v", err)
		}
	})
}

func TestHFListGGUF_NotFound(t *testing.T) {
	srv := fakeHFServer(t, nil, 404)
	defer srv.Close()
	withHFBase(t, srv.URL, func() {
		_, err := hfListGGUF(context.Background(), "org/missing", "", "")
		if err == nil || !strings.Contains(err.Error(), "not found") {
			t.Fatalf("expected not found, got %v", err)
		}
	})
}

func TestHFListGGUF_NoGGUF(t *testing.T) {
	srv := fakeHFServer(t, []hfTreeEntry{
		{Type: "file", Path: "model.safetensors", Size: 1000},
		{Type: "file", Path: "config.json", Size: 1},
	}, 200)
	defer srv.Close()
	withHFBase(t, srv.URL, func() {
		_, err := hfListGGUF(context.Background(), "org/transformers", "", "")
		if err == nil || !strings.Contains(err.Error(), "no .gguf") {
			t.Fatalf("expected no-gguf error, got %v", err)
		}
	})
}

func TestHFListConvertible(t *testing.T) {
	srv := fakeHFServer(t, []hfTreeEntry{
		{Type: "file", Path: "config.json", Size: 1},
		{Type: "file", Path: "tokenizer.json", Size: 100},
		{Type: "file", Path: "tokenizer_config.json", Size: 200},
		{Type: "file", Path: "model.safetensors", Size: 1 << 20},
		{Type: "file", Path: "pytorch_model.bin", Size: 1 << 20},
		{Type: "file", Path: "model.onnx", Size: 1 << 20},                 // rejected
		{Type: "file", Path: "model.tflite", Size: 1 << 20},               // rejected
		{Type: "file", Path: "tokenizer.bin", Size: 100},                  // not pytorch_model — skipped as not recognised
		{Type: "file", Path: "added_tokens.json", Size: 50},
		{Type: "file", Path: "chat_template.jinja", Size: 80},
	}, 200)
	defer srv.Close()
	withHFBase(t, srv.URL, func() {
		files, err := hfListConvertible(context.Background(), "org/m", "", "")
		if err != nil {
			t.Fatalf("hfListConvertible: %v", err)
		}
		got := map[string]bool{}
		for _, f := range files {
			got[f.Path] = true
		}
		mustHave := []string{"config.json", "tokenizer.json", "tokenizer_config.json",
			"model.safetensors", "pytorch_model.bin", "added_tokens.json", "chat_template.jinja"}
		for _, p := range mustHave {
			if !got[p] {
				t.Errorf("missing kept file %q", p)
			}
		}
		mustNot := []string{"model.onnx", "model.tflite", "tokenizer.bin"}
		for _, p := range mustNot {
			if got[p] {
				t.Errorf("did not expect to keep %q", p)
			}
		}
	})
}

func TestHFListConvertible_NoWeights(t *testing.T) {
	srv := fakeHFServer(t, []hfTreeEntry{
		{Type: "file", Path: "config.json", Size: 1},
		{Type: "file", Path: "tokenizer.json", Size: 1},
	}, 200)
	defer srv.Close()
	withHFBase(t, srv.URL, func() {
		_, err := hfListConvertible(context.Background(), "org/m", "", "")
		if err == nil || !strings.Contains(err.Error(), "no weight files") {
			t.Fatalf("expected no-weights error, got %v", err)
		}
	})
}

// ─── encrypt/decrypt HF token ───────────────────────────────────────────────

func TestHFTokenEncryptRoundtrip(t *testing.T) {
	s := &server{cfg: config{sessionSecret: "rotation-test-secret"}}
	plain := "hf_AbCdEfGh_1234567890_xyz"

	nonce, ct, err := s.encryptHFToken(plain)
	if err != nil {
		t.Fatalf("encrypt: %v", err)
	}
	if len(nonce) == 0 || len(ct) == 0 {
		t.Fatal("empty nonce or ciphertext")
	}
	if strings.Contains(string(ct), plain) {
		t.Fatal("ciphertext leaks plaintext")
	}
	got, err := s.decryptHFToken(nonce, ct)
	if err != nil {
		t.Fatalf("decrypt: %v", err)
	}
	if got != plain {
		t.Fatalf("decrypted = %q want %q", got, plain)
	}

	// Tamper detection: flipping a bit in the ciphertext must fail GCM auth.
	bad := append([]byte(nil), ct...)
	bad[0] ^= 0x01
	if _, err := s.decryptHFToken(nonce, bad); err == nil {
		t.Fatal("decrypt accepted tampered ciphertext")
	}

	// Rotating the secret must invalidate the stored token (graceful degrade).
	s2 := &server{cfg: config{sessionSecret: "different-secret"}}
	if _, err := s2.decryptHFToken(nonce, ct); err == nil {
		t.Fatal("decrypt succeeded with rotated secret — expected failure")
	}
}

// ─── comfyJobs lifecycle ────────────────────────────────────────────────────

func TestComfyJobsLifecycle(t *testing.T) {
	j := newComfyJobs()
	var cancelled atomic.Int32
	_, cancel := context.WithCancel(context.Background())
	wrap := func() { cancelled.Add(1); cancel() }

	cj := j.register(1, wrap)
	if cj == nil {
		t.Fatal("register returned nil")
	}

	// cancel(unknown) is a no-op.
	if j.cancel(99) {
		t.Fatal("cancel(unknown) should return false")
	}
	// cancel(known) runs the cancel func.
	if !j.cancel(1) {
		t.Fatal("cancel(1) should return true")
	}
	if cancelled.Load() != 1 {
		t.Fatal("cancel func not invoked")
	}
	// finish closes the done channel exactly once.
	j.finish(1)
	select {
	case <-cj.done:
	case <-time.After(time.Second):
		t.Fatal("finish() did not close done channel")
	}
	// finish on an already-finished/unknown id is a no-op (no panic).
	j.finish(1)
	j.finish(999)
}

// ─── importJobs lifecycle ───────────────────────────────────────────────────

func TestImportJobsLifecycle(t *testing.T) {
	j := newImportJobs()
	var called atomic.Int32
	_, cancel := context.WithCancel(context.Background())
	wrap := func() { called.Add(1); cancel() }

	j.register(7, wrap)
	if !j.cancelJob(7) {
		t.Fatal("cancelJob(7) should return true")
	}
	if called.Load() != 1 {
		t.Fatal("cancel not invoked")
	}
	j.finished(7)
	if j.cancelJob(7) {
		t.Fatal("after finished, cancelJob should return false")
	}
}

// ─── runConverter missing-converter error ───────────────────────────────────

func TestRunConverterMissing(t *testing.T) {
	_, err := runConverter("python3", "", t.TempDir(), t.TempDir(), "q8_0")
	if err == nil || !strings.Contains(err.Error(), "not configured") {
		t.Fatalf("expected not-configured error, got %v", err)
	}
	_, err = runConverter("python3", "/nonexistent/path/to/convert.py", t.TempDir(), t.TempDir(), "q8_0")
	if err == nil || !strings.Contains(err.Error(), "not found") {
		t.Fatalf("expected not-found error, got %v", err)
	}
}

// ─── httpToWS protocol mapping ──────────────────────────────────────────────

func TestHttpToWS(t *testing.T) {
	cases := map[string]string{
		"http://localhost:8080":         "ws://localhost:8080",
		"https://pool.example.com":      "wss://pool.example.com",
		"https://x.y.z:443/path":        "wss://x.y.z:443/path",
		"wss://already.ws":              "wss://already.ws",
		"":                              "",
	}
	for in, want := range cases {
		if got := httpToWS(in); got != want {
			t.Errorf("httpToWS(%q) = %q want %q", in, got, want)
		}
	}
}

// ─── TURN per-rig credential derivation ─────────────────────────────────────
//
// dist-turn (Go sidecar) validates TURN long-term creds with the rig-derived
// secret, not the master secret.  The server must produce the *same*
// derivation when minting creds for a peer that targets that rig's TURN.
// If these drift we get silent unauthorized errors on real allocations.

func TestRigTURNSecretAndMint(t *testing.T) {
	s := &server{}
	s.cfg.turnSecret = "master-secret"
	s.cfg.turnURL    = "" // intentionally unset — mintRigTURNCreds must still mint

	// rigTURNSecret should be deterministic and depend on agent_id.
	a := s.rigTURNSecret("rig-A")
	b := s.rigTURNSecret("rig-B")
	if a == "" || b == "" {
		t.Fatalf("rigTURNSecret returned empty: a=%q b=%q", a, b)
	}
	if a == b {
		t.Fatalf("rigTURNSecret must differ per agent; got identical %q", a)
	}
	// Recomputation matches the documented HMAC-SHA256 hex format.
	mac := hmac.New(sha256.New, []byte(s.cfg.turnSecret))
	mac.Write([]byte("dist-turn-rig|"))
	mac.Write([]byte("rig-A"))
	want := hex.EncodeToString(mac.Sum(nil))
	if a != want {
		t.Errorf("rigTURNSecret(rig-A) = %q, want %q", a, want)
	}

	// Empty master / empty agent_id → empty rig secret.
	if s2 := (&server{}); s2.rigTURNSecret("rig-A") != "" {
		t.Errorf("empty master secret should produce empty rig secret")
	}
	if s.rigTURNSecret("") != "" {
		t.Errorf("empty agent_id should produce empty rig secret")
	}

	// mintRigTURNCreds produces a coturn-REST username + HMAC-SHA1 cred.
	user, cred := s.mintRigTURNCreds("rig-A", "audience-X")
	if user == "" || cred == "" {
		t.Fatalf("mintRigTURNCreds returned empties: user=%q cred=%q", user, cred)
	}
	// Username format: "<unix-exp>:<audience>"
	parts := strings.SplitN(user, ":", 2)
	if len(parts) != 2 || parts[1] != "audience-X" {
		t.Fatalf("bad username format: %q", user)
	}
	// Validate by recomputing HMAC-SHA1 with the rig secret.
	mac2 := hmac.New(sha1.New, []byte(a))
	mac2.Write([]byte(user))
	expectCred := base64.StdEncoding.EncodeToString(mac2.Sum(nil))
	if cred != expectCred {
		t.Errorf("mintRigTURNCreds cred mismatch:\n got  %q\n want %q", cred, expectCred)
	}

	// Compromised-rig attack: cred minted with rig-A's secret should fail
	// validation against rig-B's HMAC.  We re-derive rig-B's secret and
	// confirm the HMAC over the same username doesn't match.
	macB := hmac.New(sha1.New, []byte(b))
	macB.Write([]byte(user))
	credBVer := base64.StdEncoding.EncodeToString(macB.Sum(nil))
	if credBVer == cred {
		t.Errorf("rig-A cred verifies against rig-B secret — derivation is leaking")
	}
}
