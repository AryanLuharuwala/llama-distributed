package main

import (
	"database/sql"
	"strings"
	"testing"
)

func TestHashAgentKey(t *testing.T) {
	// Determinism.
	a := hashAgentKey("ak-1234567890abcdef")
	b := hashAgentKey("ak-1234567890abcdef")
	if a != b {
		t.Fatalf("hashAgentKey not deterministic: %q vs %q", a, b)
	}
	// Empty in → empty out.
	if got := hashAgentKey(""); got != "" {
		t.Errorf("hashAgentKey(empty) = %q, want empty", got)
	}
	// Hex sha256 length is 64.
	if len(a) != 64 {
		t.Errorf("hash length = %d, want 64", len(a))
	}
	// Distinct inputs → distinct hashes.
	if hashAgentKey("ak-1") == hashAgentKey("ak-2") {
		t.Errorf("hash collision on different inputs")
	}
}

func TestVerifyAgentKey(t *testing.T) {
	stored := hashAgentKey("ak-correct-token")
	if !verifyAgentKey(stored, "ak-correct-token") {
		t.Errorf("verifyAgentKey should return true on match")
	}
	if verifyAgentKey(stored, "ak-wrong-token") {
		t.Errorf("verifyAgentKey should return false on mismatch")
	}
	if verifyAgentKey("", "ak-correct-token") {
		t.Errorf("empty stored hash must never authenticate")
	}
	if verifyAgentKey(stored, "") {
		t.Errorf("empty presented key must never authenticate")
	}
	// Different-length stored hash (corrupted row) must fail, not panic.
	if verifyAgentKey("abc", "ak-correct-token") {
		t.Errorf("short stored hash must fail")
	}
}

func TestVerifyAgentKeyWithFallback(t *testing.T) {
	stored := hashAgentKey("ak-correct")
	// Hash present → uses hash path.
	if !verifyAgentKeyWithFallback(
		sql.NullString{String: stored, Valid: true},
		sql.NullString{},
		"ak-correct",
	) {
		t.Errorf("hash-path verify should succeed")
	}
	// Hash absent, plaintext present → falls back to plaintext compare.
	if !verifyAgentKeyWithFallback(
		sql.NullString{},
		sql.NullString{String: "ak-correct", Valid: true},
		"ak-correct",
	) {
		t.Errorf("legacy plaintext fallback should succeed")
	}
	// Hash absent, plaintext absent → fail.
	if verifyAgentKeyWithFallback(
		sql.NullString{},
		sql.NullString{},
		"ak-correct",
	) {
		t.Errorf("no stored credential must never authenticate")
	}
	// Hash present, wrong key, plaintext also wrong → fail even with fallback.
	if verifyAgentKeyWithFallback(
		sql.NullString{String: stored, Valid: true},
		sql.NullString{String: "ak-correct", Valid: true},
		"ak-wrong",
	) {
		t.Errorf("wrong key must fail under both paths")
	}
}

func TestBackfillAgentKeyHashes(t *testing.T) {
	// Reuse openMCPTestDB to get an in-memory SQLite with the dialect set;
	// we apply the main `migrate` on top to get the rigs table.
	s, db := openMCPTestDB(t)
	if err := migrate(db, s.dialect); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	// Seed one row with plaintext only, one with hash already set.
	if _, err := db.Exec(`INSERT INTO users (display_name, created_at) VALUES ('u1', 0)`); err != nil {
		t.Fatalf("seed user: %v", err)
	}
	var uid int64
	if err := db.QueryRow(`SELECT id FROM users WHERE display_name='u1'`).Scan(&uid); err != nil {
		t.Fatalf("read user: %v", err)
	}
	if _, err := db.Exec(
		`INSERT INTO rigs (user_id, agent_id, hostname, n_gpus, vram_bytes, last_seen, agent_key, agent_key_hash)
		 VALUES (?, ?, '', 0, 0, 0, ?, NULL)`,
		uid, "rig-legacy", "ak-legacy-plain"); err != nil {
		t.Fatalf("seed legacy rig: %v", err)
	}
	preHash := hashAgentKey("ak-already-hashed")
	if _, err := db.Exec(
		`INSERT INTO rigs (user_id, agent_id, hostname, n_gpus, vram_bytes, last_seen, agent_key, agent_key_hash)
		 VALUES (?, ?, '', 0, 0, 0, '', ?)`,
		uid, "rig-modern", preHash); err != nil {
		t.Fatalf("seed modern rig: %v", err)
	}

	if err := s.backfillAgentKeyHashes(); err != nil {
		t.Fatalf("backfillAgentKeyHashes: %v", err)
	}

	// Legacy row should now have agent_key_hash populated.
	var legacyHash, modernHash string
	if err := db.QueryRow(`SELECT agent_key_hash FROM rigs WHERE agent_id='rig-legacy'`).
		Scan(&legacyHash); err != nil {
		t.Fatalf("read legacy hash: %v", err)
	}
	if legacyHash != hashAgentKey("ak-legacy-plain") {
		t.Errorf("legacy hash = %q, want %q", legacyHash, hashAgentKey("ak-legacy-plain"))
	}
	// Modern row should be untouched.
	if err := db.QueryRow(`SELECT agent_key_hash FROM rigs WHERE agent_id='rig-modern'`).
		Scan(&modernHash); err != nil {
		t.Fatalf("read modern hash: %v", err)
	}
	if modernHash != preHash {
		t.Errorf("modern hash got rewritten: %q want %q", modernHash, preHash)
	}

	// Idempotent: a second pass is a no-op.
	if err := s.backfillAgentKeyHashes(); err != nil {
		t.Fatalf("second backfill: %v", err)
	}
}

func TestAgentKeyHashStoredOnFirstPair(t *testing.T) {
	// Smoke check: verify the schema actually has agent_key_hash so the
	// inserts in ws.go won't silently fail in a fresh deployment.
	s, _ := openMCPTestDB(t)
	if err := migrate(s.db, s.dialect); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	var name string
	err := s.db.QueryRow(
		`SELECT name FROM pragma_table_info('rigs') WHERE name='agent_key_hash'`,
	).Scan(&name)
	if err != nil || !strings.EqualFold(name, "agent_key_hash") {
		t.Fatalf("agent_key_hash column missing: err=%v name=%q", err, name)
	}
}
