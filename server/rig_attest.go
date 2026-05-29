package main

// rig_attest.go — defenses against a rig that returns garbage / random
// tokens.  Two complementary checks:
//
//   1. Build attestation (binary identity): the rig publishes the
//      sha256 of the gpunet-node binary it's running, plus a build_id
//      string the operator embeds via -ldflags="-X main.buildID=…".
//      We pin known-good hashes per (os, arch) and refuse rigs whose
//      binary doesn't match a release the operator has whitelisted.
//      This stops the simplest attack — a hand-rolled binary that
//      streams `printf "%d\n" $RANDOM` and collects compute credit.
//
//   2. Canary correctness: for diffusion (DPP) we can replay a fixed
//      (prompt, model, seed, steps, cfg, w, h) and expect a bit-exact
//      latent / image — diffusion is deterministic given a seed.  We
//      cache the SHA256 of the trusted-rig output once, then any
//      rig that returns a different SHA loses trust.  For LLMs the
//      same trick works at temperature=0 (greedy decode is
//      deterministic for a given model + prompt + max_tokens).
//
// Neither catches a sophisticated attacker who runs the *real* model
// to compute the canary but drops noise into real user requests, but
// it raises the bar from "trivial" to "actually run the model" — at
// which point honest compute is cheaper than the rig fee they earn.
//
// This file only models the data + verification primitives.  Wiring
// canaries into a periodic dispatcher and gating planners on
// failing-rig lists is left to a later patch — but the gating helper
// `rigIsTrusted` is plumbed and is consulted by countDPPRigsInPool /
// pickStagesByScore in subsequent patches.

import (
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"errors"
	"net/http"
	"os"
)

// migrateRigAttest installs the two tables.  Idempotent.
func migrateRigAttest(db *sql.DB, d sqlDialect) error {
	if _, err := db.Exec(d.RewriteDDL(`
		CREATE TABLE IF NOT EXISTS rig_attest (
			user_id           INTEGER NOT NULL,
			agent_id          TEXT    NOT NULL,
			binary_sha        TEXT    NOT NULL DEFAULT '',
			build_id          TEXT    NOT NULL DEFAULT '',
			last_canary_at    INTEGER NOT NULL DEFAULT 0,
			last_canary_ok    INTEGER NOT NULL DEFAULT 0,
			canary_fail_count INTEGER NOT NULL DEFAULT 0,
			trust_score       INTEGER NOT NULL DEFAULT 100,
			updated_at        INTEGER NOT NULL DEFAULT 0,
			PRIMARY KEY (user_id, agent_id)
		)
	`)); err != nil {
		return err
	}
	// Canary ground-truth: one row per (model, prompt_hash, seed, w, h,
	// steps, cfg_x100).  expected_sha is the SHA256 of the canonical
	// trusted-rig output (PNG bytes for DPP, raw token-string for LLM).
	if _, err := db.Exec(d.RewriteDDL(`
		CREATE TABLE IF NOT EXISTS rig_canary_truth (
			kind         TEXT    NOT NULL,         -- "dpp" | "llm"
			model        TEXT    NOT NULL,
			prompt_hash  TEXT    NOT NULL,
			seed         INTEGER NOT NULL,
			width        INTEGER NOT NULL DEFAULT 0,
			height       INTEGER NOT NULL DEFAULT 0,
			steps        INTEGER NOT NULL DEFAULT 0,
			cfg_x100     INTEGER NOT NULL DEFAULT 0,
			expected_sha TEXT    NOT NULL,
			updated_at   INTEGER NOT NULL DEFAULT 0,
			PRIMARY KEY (kind, model, prompt_hash, seed, width, height, steps, cfg_x100)
		)
	`)); err != nil {
		return err
	}
	return nil
}

// recordBinaryAttest stores the (binary_sha, build_id) a rig reports at
// hello.  Called from the hub on every (re)connect.  Zero-value
// inputs are tolerated for older nodes that don't ship the metadata.
func (s *server) recordBinaryAttest(userID int64, agentID, binarySHA, buildID string) {
	if s == nil || s.db == nil || agentID == "" {
		return
	}
	_, _ = s.dbExec(`
		INSERT INTO rig_attest (user_id, agent_id, binary_sha, build_id, updated_at)
		VALUES (?, ?, ?, ?, ?)
		ON CONFLICT(user_id, agent_id) DO UPDATE SET
		    binary_sha = excluded.binary_sha,
		    build_id   = excluded.build_id,
		    updated_at = excluded.updated_at
	`, userID, agentID, binarySHA, buildID, nowUnix())
}

// recordCanary stores the result of a canary check.  ok=false bumps
// canary_fail_count and decays trust_score by 25 (so 4 consecutive
// failures push a rig below the default trust threshold of 50).
func (s *server) recordCanary(userID int64, agentID string, ok bool) {
	if s == nil || s.db == nil || agentID == "" {
		return
	}
	delta := 0
	failBump := 0
	if ok {
		delta = +5
	} else {
		delta = -25
		failBump = 1
	}
	_, _ = s.dbExec(`
		INSERT INTO rig_attest (user_id, agent_id, last_canary_at, last_canary_ok,
		                        canary_fail_count, trust_score, updated_at)
		VALUES (?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(user_id, agent_id) DO UPDATE SET
		    last_canary_at    = excluded.last_canary_at,
		    last_canary_ok    = excluded.last_canary_ok,
		    canary_fail_count = canary_fail_count + ?,
		    trust_score       = MAX(0, MIN(100, trust_score + ?)),
		    updated_at        = excluded.updated_at
	`, userID, agentID, nowUnix(), boolInt(ok), failBump,
		clampInt(100+delta, 0, 100), nowUnix(), failBump, delta)
}

// rigIsTrusted returns true if the rig has not failed the most-recent
// canary AND its trust_score is above the threshold.  Unknown rigs
// (never canaried) are trusted by default — a "presumption of
// innocence" — so a fresh deploy isn't blocked from running anything
// until at least one canary has fired.  Operators can flip this to
// strict mode by setting RIG_TRUST_STRICT=1.
func (s *server) rigIsTrusted(userID int64, agentID string) bool {
	if s == nil || s.db == nil || agentID == "" {
		return true
	}
	var lastOK, score int
	var lastAt int64
	err := s.dbQueryRow(`
		SELECT last_canary_at, last_canary_ok, trust_score
		FROM rig_attest WHERE user_id = ? AND agent_id = ?`, userID, agentID).
		Scan(&lastAt, &lastOK, &score)
	if errors.Is(err, sql.ErrNoRows) {
		return os.Getenv("RIG_TRUST_STRICT") == ""
	}
	if err != nil {
		return true
	}
	if lastAt == 0 {
		return os.Getenv("RIG_TRUST_STRICT") == ""
	}
	if lastOK == 0 {
		return false
	}
	return score >= 50
}

// sha256Hex is the canonical helper used by canary truth lookups.
func sha256Hex(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

// loadCanaryTruth returns the expected SHA for a (kind, model, prompt,
// seed, w, h, steps, cfg) tuple, or "" if no truth has been recorded.
func (s *server) loadCanaryTruth(kind, model, prompt string, seed int64,
	w, h, steps int, cfg float64) string {
	if s == nil || s.db == nil {
		return ""
	}
	var sha string
	_ = s.dbQueryRow(`
		SELECT expected_sha FROM rig_canary_truth
		WHERE kind=? AND model=? AND prompt_hash=? AND seed=?
		      AND width=? AND height=? AND steps=? AND cfg_x100=?`,
		kind, model, sha256Hex([]byte(prompt)), seed,
		w, h, steps, int(cfg*100)).Scan(&sha)
	return sha
}

// recordCanaryTruth is called by an admin tool ("trust this rig as the
// source of truth") to install the expected SHA for a canary.
func (s *server) recordCanaryTruth(kind, model, prompt string, seed int64,
	w, h, steps int, cfg float64, expectedSHA string) {
	if s == nil || s.db == nil || expectedSHA == "" {
		return
	}
	_, _ = s.dbExec(`
		INSERT INTO rig_canary_truth (kind, model, prompt_hash, seed,
		    width, height, steps, cfg_x100, expected_sha, updated_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(kind, model, prompt_hash, seed, width, height, steps, cfg_x100)
		DO UPDATE SET expected_sha = excluded.expected_sha,
		              updated_at   = excluded.updated_at
	`, kind, model, sha256Hex([]byte(prompt)), seed, w, h, steps, int(cfg*100),
		expectedSHA, nowUnix())
}

// handleRigAttest — GET /api/admin/rig_attest.  Lists the current
// trust state per rig the caller owns (or all rigs when caller is
// admin).  Wired here as a stub so the UI can render a "rig trust"
// dashboard once the admin role lands (P20-D1).
func (s *server) handleRigAttest(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	rows, err := s.dbQuery(`
		SELECT agent_id, binary_sha, build_id, last_canary_at,
		       last_canary_ok, canary_fail_count, trust_score
		FROM rig_attest WHERE user_id = ?
		ORDER BY trust_score ASC, canary_fail_count DESC`, u.ID)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	defer rows.Close()
	type row struct {
		AgentID         string `json:"agent_id"`
		BinarySHA       string `json:"binary_sha,omitempty"`
		BuildID         string `json:"build_id,omitempty"`
		LastCanaryAt    int64  `json:"last_canary_at"`
		LastCanaryOK    int    `json:"last_canary_ok"`
		CanaryFailCount int    `json:"canary_fail_count"`
		TrustScore      int    `json:"trust_score"`
	}
	out := []row{}
	for rows.Next() {
		var rr row
		if err := rows.Scan(&rr.AgentID, &rr.BinarySHA, &rr.BuildID,
			&rr.LastCanaryAt, &rr.LastCanaryOK, &rr.CanaryFailCount,
			&rr.TrustScore); err != nil {
			continue
		}
		out = append(out, rr)
	}
	writeJSON(w, 200, map[string]any{"rigs": out})
}

func boolInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
