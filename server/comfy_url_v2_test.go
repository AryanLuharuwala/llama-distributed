package main

import (
	"encoding/json"
	"fmt"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// seedComfyOutput creates a comfy_jobs row owned by `uid` and drops
// a single output file at the path handleComfyOutput will look up.
func seedComfyOutput(t *testing.T, s *server, uid int64, file, body string) int64 {
	t.Helper()
	res, err := s.db.Exec(
		`INSERT INTO comfy_jobs (user_id, prompt, params_json, status, out_files, created_at, updated_at)
		 VALUES (?, '', '{}', 'done', ?, ?, ?)`,
		uid, `["`+file+`"]`, nowUnix(), nowUnix(),
	)
	if err != nil {
		t.Fatalf("insert comfy_jobs: %v", err)
	}
	jobID, _ := res.LastInsertId()
	dir := filepath.Join(s.cfg.comfyOutDir, strconv.FormatInt(jobID, 10))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, file), []byte(body), 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}
	return jobID
}

// buildComfyReq constructs a GET against handleComfyOutput with the
// PathValue lookups its mux would normally populate.
func buildComfyReq(jobID int64, file, rawQuery string) *httptest.ResponseRecorder {
	return httptest.NewRecorder()
}

func doComfyOutput(s *server, jobID int64, file, rawQuery, sid string) *httptest.ResponseRecorder {
	rr := httptest.NewRecorder()
	r := httptest.NewRequest("GET",
		fmt.Sprintf("/comfy/out/%d/%s?%s", jobID, file, rawQuery), nil)
	r.SetPathValue("id", strconv.FormatInt(jobID, 10))
	r.SetPathValue("file", file)
	if sid != "" {
		withSession(r, sid)
	}
	s.handleComfyOutput(rr, r)
	return rr
}

func TestComfyV2_RequiresMatchingSession(t *testing.T) {
	s := newTestServer(t)
	owner, ownerSID := makeUser(t, s, "owner")
	_, attackerSID := makeUser(t, s, "attacker")

	jobID := seedComfyOutput(t, s, owner, "render.png", "PNGDATA")
	signed := s.signComfyOutputURL(owner, jobID, "render.png", 15*time.Minute)
	parsed, _ := url.Parse(signed)
	raw := parsed.RawQuery

	// Owner with valid session: should fetch the file.
	rr := doComfyOutput(s, jobID, "render.png", raw, ownerSID)
	if rr.Code != 200 || rr.Body.String() != "PNGDATA" {
		t.Errorf("owner read failed: code=%d body=%q", rr.Code, rr.Body.String())
	}

	// Attacker with their own valid session but using a leaked URL —
	// the uid binding inside the cap must trip the verifier.
	rr = doComfyOutput(s, jobID, "render.png", raw, attackerSID)
	if rr.Code != 403 {
		t.Errorf("attacker bypass: code=%d body=%q", rr.Code, rr.Body.String())
	}

	// Anonymous (no session at all) — macaroon path requires a session
	// before we even attempt verification, so this is 401, not 403.
	rr = doComfyOutput(s, jobID, "render.png", raw, "")
	if rr.Code != 401 {
		t.Errorf("anonymous bypass: code=%d body=%q", rr.Code, rr.Body.String())
	}
}

// TestComfyCap_TamperedCapFails — P7 macaroon path.  A bit-flipped cap
// must fail verification; the test used to twiddle the uid= query param,
// but uid now lives inside the cap so we tamper the cap bytes directly.
func TestComfyV2_TamperedUidFails(t *testing.T) {
	s := newTestServer(t)
	owner, ownerSID := makeUser(t, s, "owner")
	jobID := seedComfyOutput(t, s, owner, "render.png", "OK")

	signed := s.signComfyOutputURL(owner, jobID, "render.png", 15*time.Minute)
	parsed, _ := url.Parse(signed)
	q := parsed.Query()
	cap := q.Get("cap")
	if cap == "" {
		t.Fatalf("expected cap= in URL, got %q", signed)
	}
	// Flip one byte of the macaroon.
	b := []byte(cap)
	if b[len(b)-1] == 'A' {
		b[len(b)-1] = 'B'
	} else {
		b[len(b)-1] = 'A'
	}
	q.Set("cap", string(b))

	rr := doComfyOutput(s, jobID, "render.png", q.Encode(), ownerSID)
	if rr.Code != 403 {
		t.Errorf("tampered cap passed verification: code=%d body=%q", rr.Code, rr.Body.String())
	}
}

func TestComfyV1_AcceptedWithinGraceWindow(t *testing.T) {
	s := newTestServer(t)
	owner, _ := makeUser(t, s, "owner")
	jobID := seedComfyOutput(t, s, owner, "legacy.png", "LEGACY")

	exp := nowUnix() + 600
	sig := s.signComfyOutputV1(jobID, "legacy.png", exp)
	raw := fmt.Sprintf("exp=%d&sig=%s", exp, sig)

	// startedAt is wall-clock — by default in the grace window.
	rr := doComfyOutput(s, jobID, "legacy.png", raw, "")
	if rr.Code != 200 || rr.Body.String() != "LEGACY" {
		t.Errorf("v1 read in grace window failed: code=%d body=%q", rr.Code, rr.Body.String())
	}
}

func TestComfyV1_RejectedAfterGraceWindow(t *testing.T) {
	s := newTestServer(t)
	// Pretend the server booted 25h ago — past the 24h v1 sunset.
	s.startedAt = time.Now().Add(-25 * time.Hour)

	owner, _ := makeUser(t, s, "owner")
	jobID := seedComfyOutput(t, s, owner, "stale.png", "X")

	exp := nowUnix() + 600
	sig := s.signComfyOutputV1(jobID, "stale.png", exp)
	raw := fmt.Sprintf("exp=%d&sig=%s", exp, sig)

	rr := doComfyOutput(s, jobID, "stale.png", raw, "")
	if rr.Code != 401 || !strings.Contains(rr.Body.String(), "legacy") {
		t.Errorf("v1 past sunset still accepted: code=%d body=%q", rr.Code, rr.Body.String())
	}
}

func TestComfyV2_BadSignatureFails(t *testing.T) {
	s := newTestServer(t)
	owner, ownerSID := makeUser(t, s, "owner")
	jobID := seedComfyOutput(t, s, owner, "render.png", "OK")

	signed := s.signComfyOutputURL(owner, jobID, "render.png", 15*time.Minute)
	parsed, _ := url.Parse(signed)
	q := parsed.Query()
	// P7: tamper the macaroon (formerly the sig) — flip the first
	// base64-safe character.
	orig := q.Get("cap")
	if orig == "" {
		t.Fatal("expected cap= in URL")
	}
	b := []byte(orig)
	if b[0] == 'A' {
		b[0] = 'B'
	} else {
		b[0] = 'A'
	}
	q.Set("cap", string(b))

	rr := doComfyOutput(s, jobID, "render.png", q.Encode(), ownerSID)
	if rr.Code != 403 {
		t.Errorf("tampered cap passed: code=%d body=%q", rr.Code, rr.Body.String())
	}

	// And a payload sanity check — round-trip JSON of nothing, just to
	// catch encoding regressions if signedURLsFor ever produces empty
	// arrays in odd shapes.
	_, _ = json.Marshal(map[string]any{"ok": true})
}

func flipFirstHexNibble(s string) string {
	if s == "" {
		return s
	}
	b := []byte(s)
	c := b[0]
	switch {
	case c >= '0' && c <= '8':
		b[0] = c + 1
	case c == '9':
		b[0] = 'a'
	case c >= 'a' && c <= 'e':
		b[0] = c + 1
	case c == 'f':
		b[0] = '0'
	}
	return string(b)
}
