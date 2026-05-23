package main

// Agent-key hashing and constant-time verification.
//
// Background: agent_key is the long-lived bearer token a rig replays on
// every reconnect.  It used to be stored as plaintext in `rigs.agent_key`
// (and looked up with a SQL `WHERE agent_key = ?`), which meant:
//
//   1. A DB dump leaked every rig's auth credential.
//   2. The SQL `=` compare opens a (tiny but non-zero) timing side-channel:
//      MySQL/SQLite stop comparing at the first mismatched byte, which is
//      irrelevant for plaintext lookups but relevant if anyone ever ports
//      the table to a backend that returns a hit-count timing differential.
//
// Fix: store sha256(agent_key) in `rigs.agent_key_hash` and resolve rigs
// by `agent_id` alone, then constant-time compare hashes.  Plaintext is
// returned exactly once (on first pair / device-code approval) and is
// never persisted to the rigs table after that.
//
// The transition is handled at boot by backfillAgentKeyHashes, which
// computes the hash for any legacy row that has a plaintext agent_key
// but no hash yet.  The plaintext column survives — we don't drop it,
// because some older callers (e.g. emergency manual recovery scripts)
// may still reference it — but the resume + bearer-lookup paths read
// only the hash column.

import (
	"crypto/sha256"
	"crypto/subtle"
	"database/sql"
	"encoding/hex"
	"log"
)

// hashAgentKey returns the hex sha256 of a plaintext agent_key.  Returns
// the empty string on empty input so callers can write the result back
// to a TEXT NOT NULL column without ceremony — the verify path treats
// "" as "no key set".
func hashAgentKey(plain string) string {
	if plain == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(plain))
	return hex.EncodeToString(sum[:])
}

// verifyAgentKey returns true iff hashAgentKey(presented) == storedHash
// under a constant-time byte compare.  Empty storedHash always returns
// false so an un-backfilled row can't authenticate by accident.
func verifyAgentKey(storedHash, presented string) bool {
	if storedHash == "" || presented == "" {
		return false
	}
	got := hashAgentKey(presented)
	if len(got) != len(storedHash) {
		return false
	}
	return subtle.ConstantTimeCompare([]byte(got), []byte(storedHash)) == 1
}

// verifyAgentKeyWithFallback verifies a presented key against a stored
// hash, with a one-shot fallback to a legacy plaintext column for rows
// where backfill hasn't run yet.  The fallback exists because the
// backfill itself races against in-flight reconnects on a freshly-
// migrated DB — if a rig reconnects in the 50 ms between the migration
// and the backfill loop, we don't want to bounce it.
//
// In normal operation (post-backfill) storedHash is set, the fast path
// returns true on match, and the plaintext leg is never consulted.
func verifyAgentKeyWithFallback(storedHash, legacyPlain sql.NullString, presented string) bool {
	if presented == "" {
		return false
	}
	if storedHash.Valid && storedHash.String != "" {
		return verifyAgentKey(storedHash.String, presented)
	}
	// No hash on file — fall back to constant-time compare against the
	// legacy plaintext column.  We're still in the migration window.
	if legacyPlain.Valid && legacyPlain.String != "" {
		if len(legacyPlain.String) != len(presented) {
			return false
		}
		return subtle.ConstantTimeCompare([]byte(legacyPlain.String), []byte(presented)) == 1
	}
	return false
}

// backfillAgentKeyHashes computes agent_key_hash for any rigs row that
// still has a plaintext agent_key but no hash yet.  Idempotent — rows
// where hash is already set are left alone.  Runs once at boot from
// main() after the schema migration.
//
// We intentionally do NOT clear the plaintext column here; the operator
// can run a separate clean-up after they've confirmed the hash path is
// working in production.  The resume + bearer paths read only the hash
// column, so leaving the plaintext in place is a defense-in-depth
// concern, not a functional one.
func (s *server) backfillAgentKeyHashes() error {
	if s == nil || s.db == nil {
		return nil
	}
	q := s.dialect.RewriteQuery(`
		SELECT id, agent_key
		FROM rigs
		WHERE (agent_key_hash IS NULL OR agent_key_hash = '')
		  AND agent_key IS NOT NULL
		  AND agent_key <> ''
	`)
	rows, err := s.db.Query(q)
	if err != nil {
		return err
	}
	type pending struct {
		id   int64
		hash string
	}
	var todo []pending
	for rows.Next() {
		var id int64
		var key string
		if err := rows.Scan(&id, &key); err != nil {
			rows.Close()
			return err
		}
		todo = append(todo, pending{id: id, hash: hashAgentKey(key)})
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		return err
	}
	rows.Close()
	if len(todo) == 0 {
		return nil
	}
	u := s.dialect.RewriteQuery(`UPDATE rigs SET agent_key_hash = ? WHERE id = ?`)
	for _, p := range todo {
		if _, err := s.db.Exec(u, p.hash, p.id); err != nil {
			return err
		}
	}
	log.Printf("agent_key backfill: hashed %d legacy rows", len(todo))
	return nil
}
