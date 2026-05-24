package main

// Tests for prefix-affinity routing.  We exercise the data layer
// directly — the dispatcher hook lives in inference.go and is covered
// by the integration tests there.

import (
	"context"
	"strings"
	"testing"
)

func TestPromptPrefixHash_Stable(t *testing.T) {
	a := promptPrefixHash("system: you are a helpful assistant.\nuser: hi")
	b := promptPrefixHash("system: you are a helpful assistant.\nuser: hi")
	if a != b {
		t.Errorf("hash not stable: %q vs %q", a, b)
	}
	if len(a) != 64 {
		t.Errorf("hash len=%d, want 64 hex chars", len(a))
	}
	if promptPrefixHash("") != "" {
		t.Error("empty prompt should hash to empty string")
	}
}

func TestPromptPrefixHash_DiffersOnSystemChange(t *testing.T) {
	a := promptPrefixHash("system: A\nuser: hi")
	b := promptPrefixHash("system: B\nuser: hi")
	if a == b {
		t.Error("different system prompts must hash apart")
	}
}

func TestPromptPrefixHash_StableUnderTailGrowth(t *testing.T) {
	// The hash uses the first prefixHashBytes bytes of the prompt;
	// extending the user input past that window must NOT change the hash.
	head := strings.Repeat("a", prefixHashBytes)
	a := promptPrefixHash(head)
	b := promptPrefixHash(head + " ... more user text ...")
	if a != b {
		t.Errorf("hash should be stable past byte %d: a=%s b=%s", prefixHashBytes, a, b)
	}
}

func TestRecordPrefixAffinity_RoundTrip(t *testing.T) {
	srv := newTestServer(t)
	uid, _ := makeUser(t, srv, "pa-user")
	rigUID, _ := makeUser(t, srv, "pa-rig")

	prefix := promptPrefixHash("system: greet politely\nuser: hello")
	ctx := context.Background()

	if err := srv.recordPrefixAffinity(ctx, uid, prefix, rigUID, "rig-1", 128); err != nil {
		t.Fatalf("record 1: %v", err)
	}
	if err := srv.recordPrefixAffinity(ctx, uid, prefix, rigUID, "rig-1", 256); err != nil {
		t.Fatalf("record 2: %v", err)
	}

	auid, aid, cached, err := srv.prefixAffinityRig(ctx, uid, prefix)
	if err != nil {
		t.Fatalf("query: %v", err)
	}
	if auid != rigUID || aid != "rig-1" {
		t.Errorf("got rig (%d, %q), want (%d, %q)", auid, aid, rigUID, "rig-1")
	}
	if cached != 256 {
		t.Errorf("cached_tokens=%d, want 256 (the upsert keeps the max)", cached)
	}
}

func TestPrefixAffinity_LRUPrune(t *testing.T) {
	srv := newTestServer(t)
	uid, _ := makeUser(t, srv, "pa-prune")
	rigUID, _ := makeUser(t, srv, "pa-prune-rig")
	ctx := context.Background()

	// Insert more rows than the per-user cap.
	const extra = 5
	total := prefixAffinityCapPerUser + extra
	for i := 0; i < total; i++ {
		hash := promptPrefixHash("prompt-" + string(rune('A'+i)))
		if hash == "" {
			t.Fatalf("hash empty for i=%d", i)
		}
		if err := srv.recordPrefixAffinity(ctx, uid, hash, rigUID, "rig-1", int64(i)); err != nil {
			t.Fatalf("record %d: %v", i, err)
		}
	}

	row := srv.dbQueryRow(`SELECT COUNT(*) FROM prefix_affinity WHERE user_id = ?`, uid)
	var n int
	if err := row.Scan(&n); err != nil {
		t.Fatalf("count: %v", err)
	}
	if n != prefixAffinityCapPerUser {
		t.Errorf("rows after prune=%d, want %d", n, prefixAffinityCapPerUser)
	}
}

func TestPrefixAffinityRig_MissReturnsZero(t *testing.T) {
	srv := newTestServer(t)
	uid, _ := makeUser(t, srv, "pa-miss")
	auid, aid, ct, err := srv.prefixAffinityRig(context.Background(), uid, promptPrefixHash("never-seen"))
	if err != nil {
		t.Fatalf("query: %v", err)
	}
	if auid != 0 || aid != "" || ct != 0 {
		t.Errorf("miss should return zeros, got (%d, %q, %d)", auid, aid, ct)
	}
}

func TestUpsertSglangCaps_StoresFields(t *testing.T) {
	srv := newTestServer(t)
	uid, _ := makeUser(t, srv, "caps-user")
	srv.upsertSglangCaps(uid, "rig-x", map[string]any{
		"ok":           true,
		"base_url":     "http://10.0.0.1:30000",
		"prefix_cache": true,
	})
	var ok, pc int
	var base string
	if err := srv.dbQueryRow(
		`SELECT ok, base_url, prefix_cache FROM sglang_caps WHERE user_id = ? AND agent_id = ?`,
		uid, "rig-x",
	).Scan(&ok, &base, &pc); err != nil {
		t.Fatalf("select: %v", err)
	}
	if ok != 1 || base != "http://10.0.0.1:30000" || pc != 1 {
		t.Errorf("got (%d, %q, %d), want (1, http://10.0.0.1:30000, 1)", ok, base, pc)
	}
}
