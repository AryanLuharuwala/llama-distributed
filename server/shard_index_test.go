package main

// Tests for the shard cache index:
//   - upsertRigShards persists what a rig reports, idempotently.
//   - clearRigShards drops only the named rig's rows.
//   - peersForShard returns rigs that are both indexed AND online.
//   - GET /api/models/{name}/peers is auth-gated and rejects path-escape.

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

// helper: stub a fully-formed agentConn into the hub map, so peersForShard
// sees this rig as online without needing a real WebSocket.
func registerStubAgent(s *server, uid int64, agentID, hostname string) *agentConn {
	ac := &agentConn{
		userID:   uid,
		agentID:  agentID,
		hostname: hostname,
		closed:   make(chan struct{}),
		outCh:    make(chan any, 1),
		binCh:    make(chan []byte, 1),
	}
	s.hub.registerAgent(ac)
	return ac
}

// Insert a rig row + upsert claims; verify SELECT count.
func TestUpsertRigShards_PersistsRows(t *testing.T) {
	s := newTestServer(t)
	uid, _ := makeUser(t, s, "shards-user")

	shards := []cachedShardEntry{
		{ModelName: "meta-llama/Llama-3-8B", File: "model-00001-of-00004.safetensors", SizeBytes: 1 << 30},
		{ModelName: "meta-llama/Llama-3-8B", File: "model-00002-of-00004.safetensors", SizeBytes: 1 << 30},
		{ModelName: "stabilityai/stable-diffusion-xl-base-1.0", File: "model.fp16.safetensors", SizeBytes: 7 << 30},
	}
	s.upsertRigShards(uid, "rig-A", shards)

	// Second upsert: same set — must not duplicate rows.
	s.upsertRigShards(uid, "rig-A", shards)

	var n int
	if err := s.db.QueryRow(
		`SELECT COUNT(*) FROM rig_shards WHERE user_id = ? AND agent_id = ?`,
		uid, "rig-A",
	).Scan(&n); err != nil {
		t.Fatalf("count: %v", err)
	}
	if n != 3 {
		t.Errorf("expected 3 rows after idempotent upsert; got %d", n)
	}
}

// A malicious rig that floods hello/status with tens of thousands of
// fake shard claims must be capped — otherwise it bloats rig_shards
// and slows every peer lookup.
func TestUpsertRigShards_CapsListSize(t *testing.T) {
	s := newTestServer(t)
	uid, _ := makeUser(t, s, "flooder")

	// 5× the cap.
	const N = maxShardsPerRig * 5
	shards := make([]cachedShardEntry, 0, N)
	for i := 0; i < N; i++ {
		shards = append(shards, cachedShardEntry{
			ModelName: "spam-model",
			File:      "spam-" + strings.Repeat("a", 1) + "-" + itoa(i) + ".bin",
		})
	}
	s.upsertRigShards(uid, "flood-rig", shards)

	var n int
	if err := s.db.QueryRow(
		`SELECT COUNT(*) FROM rig_shards WHERE user_id = ? AND agent_id = ?`,
		uid, "flood-rig",
	).Scan(&n); err != nil {
		t.Fatalf("count: %v", err)
	}
	if n > maxShardsPerRig {
		t.Errorf("flooded %d rows past the cap; want <= %d", n, maxShardsPerRig)
	}
	if n < 1 {
		t.Errorf("cap dropped everything; want some rows persisted")
	}
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var buf [20]byte
	pos := len(buf)
	for i > 0 {
		pos--
		buf[pos] = byte('0' + i%10)
		i /= 10
	}
	return string(buf[pos:])
}

// upsertRigShards must drop rig-supplied rows that try a path-escape.
func TestUpsertRigShards_RejectsBadInput(t *testing.T) {
	s := newTestServer(t)
	uid, _ := makeUser(t, s, "bad-input-user")

	shards := []cachedShardEntry{
		{ModelName: "ok", File: "good.safetensors"},
		{ModelName: "", File: "no-model.gguf"},                    // empty model
		{ModelName: "model", File: ""},                            // empty file
		{ModelName: "x\x00y", File: "f.gguf"},                     // NUL in model
		{ModelName: "m", File: "\n"},                              // newline in file
		{ModelName: strings.Repeat("a", 300), File: "long.gguf"},  // model too long
	}
	s.upsertRigShards(uid, "rig-B", shards)

	var n int
	_ = s.db.QueryRow(`SELECT COUNT(*) FROM rig_shards WHERE agent_id = ?`, "rig-B").Scan(&n)
	if n != 1 {
		t.Errorf("expected only 1 row to be persisted (the well-formed one); got %d", n)
	}
}

// clearRigShards must only drop rows for the named rig.
func TestClearRigShards_ScopedToAgent(t *testing.T) {
	s := newTestServer(t)
	uid, _ := makeUser(t, s, "clear-user")
	s.upsertRigShards(uid, "rig-A", []cachedShardEntry{{ModelName: "m", File: "a.gguf"}})
	s.upsertRigShards(uid, "rig-B", []cachedShardEntry{{ModelName: "m", File: "b.gguf"}})

	s.clearRigShards(uid, "rig-A")

	var nA, nB int
	_ = s.db.QueryRow(`SELECT COUNT(*) FROM rig_shards WHERE agent_id = ?`, "rig-A").Scan(&nA)
	_ = s.db.QueryRow(`SELECT COUNT(*) FROM rig_shards WHERE agent_id = ?`, "rig-B").Scan(&nB)
	if nA != 0 {
		t.Errorf("rig-A rows after clear: got %d, want 0", nA)
	}
	if nB != 1 {
		t.Errorf("rig-B rows: got %d, want 1 (clear must be scoped)", nB)
	}
}

// peersForShard returns only rigs that are online AND have the shard.
func TestPeersForShard_OnlineFilter(t *testing.T) {
	s := newTestServer(t)
	uid, _ := makeUser(t, s, "peer-user")

	// Three rigs claim the same shard; only two are online.
	for _, agent := range []string{"online-1", "online-2", "offline-3"} {
		s.upsertRigShards(uid, agent, []cachedShardEntry{
			{ModelName: "org/m", File: "shard.safetensors", SizeBytes: 100},
		})
	}
	registerStubAgent(s, uid, "online-1", "host-1")
	registerStubAgent(s, uid, "online-2", "host-2")
	// offline-3 deliberately not registered.

	peers := s.peersForShard(uid, "org/m", "shard.safetensors")
	if len(peers) != 2 {
		t.Fatalf("expected 2 online peers; got %d (%+v)", len(peers), peers)
	}
	got := map[string]bool{peers[0].AgentID: true, peers[1].AgentID: true}
	if !got["online-1"] || !got["online-2"] {
		t.Errorf("missing expected agent in peers: %+v", peers)
	}
	if got["offline-3"] {
		t.Errorf("offline-3 should have been filtered out")
	}
}

// Cross-user isolation: peer lookups under user A must not return user B's rigs.
func TestPeersForShard_UserIsolation(t *testing.T) {
	s := newTestServer(t)
	uidA, _ := makeUser(t, s, "user-A")
	uidB, _ := makeUser(t, s, "user-B")

	s.upsertRigShards(uidB, "rig-of-B", []cachedShardEntry{
		{ModelName: "org/m", File: "f.safetensors"},
	})
	registerStubAgent(s, uidB, "rig-of-B", "host-B")

	peers := s.peersForShard(uidA, "org/m", "f.safetensors")
	if len(peers) != 0 {
		t.Errorf("user-A must not see user-B's rigs; got %+v", peers)
	}
}

// GET /api/models/{name}/peers — auth required, bad file rejected, happy path.
func TestHandleShardPeers(t *testing.T) {
	s := newTestServer(t)
	uid, sid := makeUser(t, s, "http-user")
	s.upsertRigShards(uid, "rig-X", []cachedShardEntry{
		{ModelName: "org/repo", File: "weights.safetensors", SizeBytes: 4096},
	})
	registerStubAgent(s, uid, "rig-X", "host-X")

	// No session → 401.
	rr := httptest.NewRecorder()
	s.router().ServeHTTP(rr, httptest.NewRequest("GET", "/api/models/org%2Frepo/peers?file=weights.safetensors", nil))
	if rr.Code != 401 {
		t.Errorf("no-auth: got %d, want 401", rr.Code)
	}

	// Bad file (path traversal) → 400.
	rr = httptest.NewRecorder()
	req := withSession(httptest.NewRequest("GET", "/api/models/org%2Frepo/peers?file=..%2Fetc%2Fpasswd", nil), sid)
	s.router().ServeHTTP(rr, req)
	if rr.Code != 400 {
		t.Errorf("path-traversal: got %d, want 400", rr.Code)
	}

	// Happy path: real file, authed.
	rr = httptest.NewRecorder()
	req = withSession(httptest.NewRequest("GET", "/api/models/org%2Frepo/peers?file=weights.safetensors", nil), sid)
	s.router().ServeHTTP(rr, req)
	if rr.Code != 200 {
		t.Fatalf("happy-path: got %d, want 200; body=%s", rr.Code, rr.Body.String())
	}
	var resp struct {
		Model string      `json:"model"`
		File  string      `json:"file"`
		Peers []shardPeer `json:"peers"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("parse: %v", err)
	}
	if resp.Model != "org/repo" || resp.File != "weights.safetensors" {
		t.Errorf("response identity wrong: %+v", resp)
	}
	if len(resp.Peers) != 1 || resp.Peers[0].AgentID != "rig-X" {
		t.Fatalf("peers wrong: %+v", resp.Peers)
	}
	if resp.Peers[0].SizeBytes != 4096 {
		t.Errorf("size: got %d, want 4096", resp.Peers[0].SizeBytes)
	}
}

// Concurrent upserts must not corrupt the table (sanity: 16 goroutines * 4 shards).
func TestUpsertRigShards_RaceFree(t *testing.T) {
	s := newTestServer(t)
	uid, _ := makeUser(t, s, "race-user")

	var wg sync.WaitGroup
	for g := 0; g < 16; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			s.upsertRigShards(uid, "rig-R", []cachedShardEntry{
				{ModelName: "m", File: "a.gguf"},
				{ModelName: "m", File: "b.gguf"},
				{ModelName: "m", File: "c.gguf"},
				{ModelName: "m", File: "d.gguf"},
			})
		}(g)
	}
	wg.Wait()
	var n int
	_ = s.db.QueryRow(`SELECT COUNT(*) FROM rig_shards WHERE agent_id = ?`, "rig-R").Scan(&n)
	if n != 4 {
		t.Errorf("expected 4 distinct rows; got %d", n)
	}
}

// isSafeIndexedFile boundary cases.
func TestIsSafeIndexedFile(t *testing.T) {
	good := []string{
		"model.safetensors", "weights-00001-of-00004.bin", "config.json",
		"tokenizer.json", "Llama-3-8B.q4_0.gguf",
	}
	bad := []string{
		"", ".", "..", "../etc/passwd", "a/b.gguf", "a\\b.gguf",
		"with\nnewline.bin", "nul\x00.bin", "trailing/", "foo..bar",
		strings.Repeat("a", 300),
	}
	for _, f := range good {
		if !isSafeIndexedFile(f) {
			t.Errorf("isSafeIndexedFile(%q) = false, want true", f)
		}
	}
	for _, f := range bad {
		if isSafeIndexedFile(f) {
			t.Errorf("isSafeIndexedFile(%q) = true, want false", f)
		}
	}
}
