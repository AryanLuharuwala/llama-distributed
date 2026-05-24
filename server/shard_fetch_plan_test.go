package main

// Tests for the multi-source shard fetch planner:
//   - /api/models/{name}/fetch-plan returns peer URLs first, origin last.
//   - Plan ttl + chunk_size are present and sensible.
//   - Origin shard download honors Range-GET (http.ServeFile already
//     does, but this is the contract a parallel fetcher depends on, so
//     we lock it in with a regression test).

import (
	"encoding/json"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// registerTestModel inserts a `models` row pointing at a temp shard dir
// with `file` of `size` bytes.  Returns the model id.
func registerTestModel(t *testing.T, s *server, name, file string, size int) int64 {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, file), make([]byte, size), 0o644); err != nil {
		t.Fatalf("write shard: %v", err)
	}
	res, err := s.db.Exec(
		`INSERT INTO models (name, n_layers, n_shards, shards_dir, created_at)
		 VALUES (?, 1, 1, ?, ?)`,
		name, dir, nowUnix(),
	)
	if err != nil {
		t.Fatalf("insert model: %v", err)
	}
	id, _ := res.LastInsertId()
	return id
}

func TestFetchPlan_OriginOnly(t *testing.T) {
	s := newTestServer(t)
	_, sid := makeUser(t, s, "fetch-user")
	registerTestModel(t, s, "tinymodel", "stage-0.gguf", 1024)

	rr := httptest.NewRecorder()
	req := withSession(httptest.NewRequest(
		"GET", "/api/models/tinymodel/fetch-plan?file=stage-0.gguf", nil), sid)
	s.router().ServeHTTP(rr, req)
	if rr.Code != 200 {
		t.Fatalf("got %d: %s", rr.Code, rr.Body.String())
	}
	var plan shardFetchPlan
	if err := json.Unmarshal(rr.Body.Bytes(), &plan); err != nil {
		t.Fatalf("parse: %v", err)
	}
	if plan.Model != "tinymodel" || plan.File != "stage-0.gguf" {
		t.Errorf("plan identity: %+v", plan)
	}
	if plan.SizeBytes != 1024 {
		t.Errorf("size: got %d, want 1024", plan.SizeBytes)
	}
	if plan.ChunkSize <= 0 {
		t.Errorf("chunk size must be positive; got %d", plan.ChunkSize)
	}
	if len(plan.Sources) != 1 || plan.Sources[0].Kind != "origin" {
		t.Errorf("expected one origin source; got %+v", plan.Sources)
	}
	if !strings.Contains(plan.Sources[0].URL, "cap=") {
		t.Errorf("origin URL missing macaroon: %q", plan.Sources[0].URL)
	}
}

func TestFetchPlan_PeersBeforeOrigin(t *testing.T) {
	s := newTestServer(t)
	uid, sid := makeUser(t, s, "fp-peers")
	registerTestModel(t, s, "m1", "s.gguf", 4096)

	// Two peers claim the shard, both online.
	for _, agent := range []string{"peer-A", "peer-B"} {
		s.upsertRigShards(uid, agent, []cachedShardEntry{
			{ModelName: "m1", File: "s.gguf"},
		})
		registerStubAgent(s, uid, agent, agent+"-host")
	}

	rr := httptest.NewRecorder()
	req := withSession(httptest.NewRequest("GET", "/api/models/m1/fetch-plan?file=s.gguf", nil), sid)
	s.router().ServeHTTP(rr, req)
	if rr.Code != 200 {
		t.Fatalf("got %d: %s", rr.Code, rr.Body.String())
	}
	var plan shardFetchPlan
	_ = json.Unmarshal(rr.Body.Bytes(), &plan)
	if len(plan.Sources) != 3 {
		t.Fatalf("expected 3 sources (2 peers + origin); got %d (%+v)", len(plan.Sources), plan.Sources)
	}
	// Peers must come first.
	if plan.Sources[0].Kind != "peer" || plan.Sources[1].Kind != "peer" {
		t.Errorf("peers should be first: %+v", plan.Sources)
	}
	if plan.Sources[2].Kind != "origin" {
		t.Errorf("origin must be last: %+v", plan.Sources)
	}
	// Each peer URL must carry its agent tag.
	for _, src := range plan.Sources[:2] {
		if !strings.Contains(src.URL, "via=peer-") {
			t.Errorf("peer URL missing via= tag: %q", src.URL)
		}
		if src.AgentID == "" {
			t.Errorf("peer AgentID empty: %+v", src)
		}
	}
}

func TestFetchPlan_AuthRequired(t *testing.T) {
	s := newTestServer(t)
	registerTestModel(t, s, "m2", "s.gguf", 1)
	rr := httptest.NewRecorder()
	s.router().ServeHTTP(rr, httptest.NewRequest("GET", "/api/models/m2/fetch-plan?file=s.gguf", nil))
	if rr.Code != 401 {
		t.Errorf("expected 401 without session; got %d", rr.Code)
	}
}

func TestFetchPlan_404OnUnknown(t *testing.T) {
	s := newTestServer(t)
	_, sid := makeUser(t, s, "fp-unknown")
	rr := httptest.NewRecorder()
	req := withSession(httptest.NewRequest("GET", "/api/models/nope/fetch-plan?file=stage-0.gguf", nil), sid)
	s.router().ServeHTTP(rr, req)
	if rr.Code != 404 {
		t.Errorf("expected 404 for unknown model; got %d", rr.Code)
	}
}

// http.ServeFile already supports Range; this test locks that contract
// in so a future refactor (e.g. switching to a streaming reader) doesn't
// silently break the parallel-fetch primitive.
func TestShardDownload_HonorsRange(t *testing.T) {
	s := newTestServer(t)
	id := registerTestModel(t, s, "rangem", "stage-0.gguf", 4096)

	// Fill with a deterministic pattern so byte ranges are checkable.
	dir := ""
	_ = s.db.QueryRow(`SELECT shards_dir FROM models WHERE id = ?`, id).Scan(&dir)
	pattern := make([]byte, 4096)
	for i := range pattern {
		pattern[i] = byte(i & 0xFF)
	}
	if err := os.WriteFile(filepath.Join(dir, "stage-0.gguf"), pattern, 0o644); err != nil {
		t.Fatal(err)
	}

	url := s.mintShardURL(id, "stage-0.gguf", time.Hour)
	parsed := strings.TrimPrefix(url, s.cfg.publicURL)
	rr := httptest.NewRecorder()
	req := httptest.NewRequest("GET", parsed, nil)
	req.Header.Set("Range", "bytes=10-19")
	s.router().ServeHTTP(rr, req)
	if rr.Code != 206 {
		t.Fatalf("expected 206 Partial Content; got %d body=%q", rr.Code, rr.Body.String())
	}
	body := rr.Body.Bytes()
	if len(body) != 10 {
		t.Fatalf("expected 10 bytes; got %d", len(body))
	}
	for i := 0; i < 10; i++ {
		if body[i] != pattern[10+i] {
			t.Errorf("byte %d: got %d want %d", i, body[i], pattern[10+i])
		}
	}
}
