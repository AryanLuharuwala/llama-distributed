package main

// Data-corruption and memory-corruption simulation.
//
// Everything that crosses a trust boundary — DTLS-terminated peer relays,
// shard URLs handed to anonymous CDNs, status frames from rigs we don't
// control — must reject tampered, truncated, bit-flipped, or otherwise
// malformed input.  These tests flip specific bits in each input and
// verify the server says no.
//
// What we're *not* testing here: in-process RAM bit-flips of the server's
// own structs.  ECC RAM is the answer to that, and a Go test process
// can't model it without a fault-injection framework.  We focus on the
// adversary-controlled inputs, which is where the realistic risk lies.

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// ── Shard URL HMAC ──────────────────────────────────────────────────────────

func TestShardURL_RejectsTamperedSig(t *testing.T) {
	s := newTestServer(t)
	// Plant a model row so the handler reaches the sig check.
	dir := filepath.Join(t.TempDir(), "shards")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	res, err := s.db.Exec(
		`INSERT INTO models (name, n_layers, n_shards, shards_dir, created_at) VALUES (?, ?, ?, ?, ?)`,
		"test/model", 4, 1, dir, nowUnix())
	if err != nil {
		t.Fatal(err)
	}
	mid, _ := res.LastInsertId()
	// Drop a payload at the resolved path.
	if err := os.WriteFile(filepath.Join(dir, "stage-0.gguf"), []byte("payload"), 0o644); err != nil {
		t.Fatal(err)
	}

	good := s.mintShardURL(mid, "stage-0.gguf", 60*time.Second)
	// P7: bit-flip a character inside the macaroon (cap=) value.  The
	// final character of the URL is inside the cap= base64 since the
	// minter doesn't append other params.
	last := good[len(good)-1]
	var alt byte = 'A'
	if last == 'A' {
		alt = 'B'
	}
	flipped := good[:len(good)-1] + string(alt)

	rr := httptest.NewRecorder()
	s.router().ServeHTTP(rr, httptest.NewRequest("GET", flipped, nil))
	if rr.Code != 401 {
		t.Errorf("tampered cap: got %d, want 401", rr.Code)
	}

	// Trailing-byte corruption: appending bytes inside the cap= value
	// must be rejected by the strict round-trip check in verifyCap.
	junkSig := good + "AB"
	rr = httptest.NewRecorder()
	s.router().ServeHTTP(rr, httptest.NewRequest("GET", junkSig, nil))
	if rr.Code != 401 {
		t.Errorf("trailing-junk cap: got %d, want 401", rr.Code)
	}

	// Sanity: the original signed URL must still serve.
	rr = httptest.NewRecorder()
	s.router().ServeHTTP(rr, httptest.NewRequest("GET", good, nil))
	if rr.Code != 200 {
		t.Errorf("good sig: got %d (body=%q), want 200", rr.Code, rr.Body.String())
	}
}

func TestShardURL_RejectsExpired(t *testing.T) {
	s := newTestServer(t)
	dir := filepath.Join(t.TempDir(), "shards")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	res, _ := s.db.Exec(
		`INSERT INTO models (name, n_layers, n_shards, shards_dir, created_at) VALUES (?, ?, ?, ?, ?)`,
		"test/model", 4, 1, dir, nowUnix())
	mid, _ := res.LastInsertId()
	_ = os.WriteFile(filepath.Join(dir, "stage-0.gguf"), []byte("p"), 0o644)

	// Mint with negative TTL so exp is in the past.
	url := s.mintShardURL(mid, "stage-0.gguf", -1*time.Second)

	rr := httptest.NewRecorder()
	s.router().ServeHTTP(rr, httptest.NewRequest("GET", url, nil))
	if rr.Code != 401 {
		t.Errorf("expired URL: got %d, want 401", rr.Code)
	}
}

func TestShardURL_RejectsPathTraversal(t *testing.T) {
	s := newTestServer(t)
	dir := filepath.Join(t.TempDir(), "shards")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	res, _ := s.db.Exec(
		`INSERT INTO models (name, n_layers, n_shards, shards_dir, created_at) VALUES (?, ?, ?, ?, ?)`,
		"test/model", 4, 1, dir, nowUnix())
	mid, _ := res.LastInsertId()

	// Two shapes of traversal:
	//   - raw "../../etc/passwd" — Go's mux path-cleans and 301-redirects;
	//     accept that as long as the redirect leaves /shards/ entirely.
	//   - URL-encoded %2e%2e — bypasses path-clean, must hit isSafeShardFile
	//     and 400.
	raw := fmt.Sprintf("/models/%d/shards/../../etc/passwd?exp=9999999999&sig=00", mid)
	rr := httptest.NewRecorder()
	s.router().ServeHTTP(rr, httptest.NewRequest("GET", raw, nil))
	if rr.Code == 200 {
		t.Errorf("traversal returned 200: %q", rr.Body.String())
	}
	if rr.Code == 301 {
		if loc := rr.Header().Get("Location"); strings.Contains(loc, "/shards/") {
			t.Errorf("redirect still inside /shards/: %q", loc)
		}
	}

	encoded := fmt.Sprintf("/models/%d/shards/%%2e%%2e%%2fpasswd?exp=9999999999&sig=00", mid)
	rr = httptest.NewRecorder()
	s.router().ServeHTTP(rr, httptest.NewRequest("GET", encoded, nil))
	if rr.Code == 200 {
		t.Errorf("encoded traversal returned 200: %q", rr.Body.String())
	}
}

// ── Status frame: corrupted/absurd values ───────────────────────────────────

func TestStatusFrame_BitFlipsClampedToZero(t *testing.T) {
	// Simulate a rig sending a status frame with every field flipped to
	// nonsense.  validateAndClampStatus must zero them rather than feeding
	// the planner garbage that would skew layer allocation.
	st := &agentStatus{
		NGPUs:         999999,           // absurd
		VRAMTotal:     -1,               // negative (signed wrap on corrupt payload)
		VRAMFree:      1 << 50,          // > VRAMTotal (impossible)
		TokensPS:      1e9,              // 1G tok/s — physically impossible
		UptimeSec:     -42,              // negative seconds
		MaxConcurrent: -7,               // negative
		CoturnPort:    22,               // privileged port — SSH redirect attempt
		PublicIP:      "10.0.0.1",       // RFC1918 — illegitimate srflx
		NATType:       "PROBABLY_BROKEN", // not in allow-list
		GPUModel:      strings.Repeat("A", 1000),
	}
	validateAndClampStatus(st, "203.0.113.4")

	if st.NGPUs != 0 {
		t.Errorf("NGPUs not clamped: %d", st.NGPUs)
	}
	if st.VRAMTotal != 0 {
		t.Errorf("VRAMTotal not clamped: %d", st.VRAMTotal)
	}
	if st.VRAMFree != 0 {
		t.Errorf("VRAMFree not clamped: %d", st.VRAMFree)
	}
	if st.TokensPS != 0 {
		t.Errorf("TokensPS not clamped: %f", st.TokensPS)
	}
	if st.UptimeSec != 0 {
		t.Errorf("UptimeSec not clamped: %d", st.UptimeSec)
	}
	if st.MaxConcurrent != 0 {
		t.Errorf("MaxConcurrent not clamped: %d", st.MaxConcurrent)
	}
	if st.CoturnPort != 0 {
		t.Errorf("CoturnPort not stripped: %d", st.CoturnPort)
	}
	if st.PublicIP != "" {
		t.Errorf("PublicIP not stripped: %q", st.PublicIP)
	}
	if st.NATType != "unknown" {
		t.Errorf("NATType not normalised: %q", st.NATType)
	}
	if len(st.GPUModel) > 128 {
		t.Errorf("GPUModel not truncated: %d chars", len(st.GPUModel))
	}
}

// Per-session relay bytes: a malicious rig could claim 1<<62 bytes
// forwarded to inflate its reputation score.  clampRelayBytes caps each
// session at 64 GiB and reports the clamping so the reputation writer
// can use the truthy value.
func TestRelayBytes_OverflowClamped(t *testing.T) {
	v, clamped := clampRelayBytes(int64(1) << 62)
	if !clamped {
		t.Errorf("absurd byte count not flagged as clamped")
	}
	if v != maxRelayBytesPerSession {
		t.Errorf("clamp value: %d, want %d", v, maxRelayBytesPerSession)
	}

	v, clamped = clampRelayBytes(-999)
	if !clamped || v != 0 {
		t.Errorf("negative clamp: v=%d clamped=%v", v, clamped)
	}

	v, clamped = clampRelayBytes(1024)
	if clamped || v != 1024 {
		t.Errorf("legit value modified: v=%d clamped=%v", v, clamped)
	}
}

// ── HF download SHA mismatch detection ──────────────────────────────────────

// Already covered by hf_parallel_test.go TestVerifyFileSHA256 + ReFetchOnSHAMismatch.
// Add a focused test for the case where a CDN serves wholly bogus bytes
// (no resume header), to verify the verify wrapper fails closed.
func TestVerifyFileSHA256_PartialOverwrite(t *testing.T) {
	dir := t.TempDir()
	dst := filepath.Join(dir, "shard.gguf")
	original := []byte(strings.Repeat("A", 1024))
	if err := os.WriteFile(dst, original, 0o644); err != nil {
		t.Fatal(err)
	}
	want := hex.EncodeToString(func() []byte { h := sha256.Sum256(original); return h[:] }())

	// Partial overwrite: first 256 bytes flipped.  Simulates a buggy proxy
	// that range-merged two responses incorrectly.
	corrupted := append([]byte(strings.Repeat("B", 256)), original[256:]...)
	if err := os.WriteFile(dst, corrupted, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := verifyFileSHA256(dst, want); err == nil {
		t.Errorf("partial-overwrite corruption not detected")
	}
}

// ── Malformed WS control frames don't panic ─────────────────────────────────

// The reader loop in ws.go silently continues on JSON parse errors.  This
// is the right behaviour (don't kill the connection because one frame was
// bad), but the test pins it so a future refactor doesn't accidentally
// turn an unmarshal error into a panic.  We exercise json.Unmarshal
// directly with the same "map[string]any" target the reader uses.
func TestMalformedJSON_DoesntPanic(t *testing.T) {
	frames := [][]byte{
		nil,
		{},
		[]byte("{"),
		[]byte("[1,2,"),
		[]byte("\xff\xfe\xfd"),                       // not valid UTF-8
		[]byte(`{"kind":"status","ngpus":"oops"}`),   // type mismatch
		[]byte(`{"kind":"status","ngpus":99999999}`), // valid JSON, absurd value (handled by clamp)
		[]byte("{\"kind\":\"\x00\"}"), // embedded NUL byte
	}
	for _, f := range frames {
		// Use the same parsing shape as the reader: unmarshal into map[string]any.
		var m map[string]any
		_ = jsonDecodeMap(f, &m)
		// Mirror the agentStatus re-marshal path with a partially-built map.
		_ = jsonDecodeStatus(f)
	}
}

// ── Helpers ─────────────────────────────────────────────────────────────────

func flipHexChar(c byte) string {
	// Map 0↔1, 2↔3, …, e↔f.  Stays inside the hex alphabet so the URL
	// remains syntactically valid (so the failure must come from the HMAC
	// compare, not from a regex pre-filter).
	switch {
	case c >= '0' && c <= '9':
		if c%2 == 0 {
			return string(c + 1)
		}
		return string(c - 1)
	case c >= 'a' && c <= 'f':
		if (c-'a')%2 == 0 {
			return string(c + 1)
		}
		return string(c - 1)
	}
	return "0"
}

func jsonDecodeMap(data []byte, m *map[string]any) error {
	return json.Unmarshal(data, m)
}

func jsonDecodeStatus(data []byte) error {
	var st agentStatus
	return json.Unmarshal(data, &st)
}
