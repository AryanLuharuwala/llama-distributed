package main

// prefix_affinity.go — P13 prefix-cache-aware routing for SGLang rigs.
//
// SGLang surfaces `meta_info.cached_tokens` on every /generate response.
// When a chat request reuses a long shared prefix (system prompt + chat
// history), the rig that already holds that prefix in its radix-tree
// cache can decode the next turn with near-zero prefill cost.  We
// exploit this by remembering "rig R has recently served prefix P" and
// scoring rigs higher for requests whose prefix matches.
//
// Storage shape:
//
//   - prefix_affinity table: rolling cache of recent (prefix_hash, rig_id)
//     observations with the last observed cached_tokens count.  We bound
//     the table by row age and an LRU cap.  Pure SQLite for now; the
//     same table works on Postgres/Spanner via the dialect wrapper.
//   - sglang_caps table: which rigs advertise prefix-cache capability,
//     same shape as comfy_caps.
//
// Hash strategy: SHA-256 of the *first prefixHashBytes bytes* of the
// fully-rendered prompt.  Chats with shared system prompts collide on
// the hash, which is exactly what we want.  Different system prompts
// hash apart, so cross-user leakage isn't possible — affinity records
// are scoped to user_id anyway.

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
)

const (
	// Number of leading bytes of the prompt that go into the prefix
	// hash.  Long enough that a typical system prompt + first user
	// message collides reliably, short enough that the second turn's
	// added user input doesn't change the hash.
	prefixHashBytes = 512

	// Soft cap on rows per user.  Pruned to this on each upsert via a
	// trailing DELETE.  Beyond ~32 distinct prefixes per user, the LRU
	// window holds nothing of value — they've drifted off the rig's
	// radix cache by then.
	prefixAffinityCapPerUser = 32
)

// promptPrefixHash returns the hex SHA-256 of the prompt's first
// prefixHashBytes bytes.  Empty string for an empty prompt — callers
// should treat that as "no affinity".
func promptPrefixHash(prompt string) string {
	if prompt == "" {
		return ""
	}
	b := []byte(prompt)
	if len(b) > prefixHashBytes {
		b = b[:prefixHashBytes]
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func migratePrefixAffinity(db *sql.DB, d sqlDialect) error {
	if d == nil {
		d = sqliteDialect{}
	}
	stmts := []string{
		`CREATE TABLE IF NOT EXISTS prefix_affinity (
			user_id        INTEGER NOT NULL,
			prefix_hash    TEXT NOT NULL,
			agent_user_id  INTEGER NOT NULL,
			agent_id       TEXT NOT NULL,
			cached_tokens  INTEGER NOT NULL DEFAULT 0,
			updated_at     INTEGER NOT NULL,
			PRIMARY KEY (user_id, prefix_hash, agent_user_id, agent_id)
		)`,
		`CREATE INDEX IF NOT EXISTS idx_prefix_affinity_user_time
			ON prefix_affinity(user_id, updated_at)`,
		`CREATE TABLE IF NOT EXISTS sglang_caps (
			user_id        INTEGER NOT NULL,
			agent_id       TEXT NOT NULL,
			ok             INTEGER NOT NULL DEFAULT 0,
			base_url       TEXT NOT NULL DEFAULT '',
			prefix_cache   INTEGER NOT NULL DEFAULT 0,
			updated_at     INTEGER NOT NULL,
			PRIMARY KEY (user_id, agent_id)
		)`,
	}
	for _, q := range stmts {
		if _, err := db.Exec(d.RewriteDDL(q)); err != nil {
			return err
		}
	}
	return nil
}

// upsertSglangCaps records the sglang_caps frame.  Called from the WS
// reader, parallel to upsertComfyCaps.
func (s *server) upsertSglangCaps(uid int64, agentID string, msg map[string]any) {
	ok := 0
	if v, _ := msg["ok"].(bool); v {
		ok = 1
	}
	baseURL, _ := msg["base_url"].(string)
	pc := 0
	if v, _ := msg["prefix_cache"].(bool); v {
		pc = 1
	}
	_, _ = s.dbExec(
		`INSERT INTO sglang_caps (user_id, agent_id, ok, base_url, prefix_cache, updated_at)
		 VALUES (?, ?, ?, ?, ?, ?)
		 ON CONFLICT(user_id, agent_id) DO UPDATE SET
		   ok = excluded.ok,
		   base_url = excluded.base_url,
		   prefix_cache = excluded.prefix_cache,
		   updated_at = excluded.updated_at`,
		uid, agentID, ok, baseURL, pc, nowUnix(),
	)
}

// recordPrefixAffinity is called after an SGLang inference finishes and
// reported a non-zero cached_tokens (or a fresh prefix that may be worth
// remembering).  Upserts the row and prunes the user's tail beyond
// prefixAffinityCapPerUser.
func (s *server) recordPrefixAffinity(ctx context.Context, ownerUID int64, prefixHash string,
	agentUID int64, agentID string, cachedTokens int64) error {
	if prefixHash == "" {
		return nil
	}
	now := nowUnix()
	if _, err := s.dbExecCtx(ctx,
		`INSERT INTO prefix_affinity (user_id, prefix_hash, agent_user_id, agent_id, cached_tokens, updated_at)
		 VALUES (?, ?, ?, ?, ?, ?)
		 ON CONFLICT(user_id, prefix_hash, agent_user_id, agent_id) DO UPDATE SET
		   cached_tokens = MAX(prefix_affinity.cached_tokens, excluded.cached_tokens),
		   updated_at    = excluded.updated_at`,
		ownerUID, prefixHash, agentUID, agentID, cachedTokens, now,
	); err != nil {
		return err
	}

	// Prune.  Keep the N most recently updated rows for this owner.
	if _, err := s.dbExecCtx(ctx,
		`DELETE FROM prefix_affinity
		   WHERE user_id = ?
		     AND rowid NOT IN (
		       SELECT rowid FROM prefix_affinity
		         WHERE user_id = ?
		         ORDER BY updated_at DESC
		         LIMIT ?
		     )`,
		ownerUID, ownerUID, prefixAffinityCapPerUser,
	); err != nil {
		// Pruning is best-effort.  The cap is soft — leaving extra rows
		// in a transient SQL error is fine.
		return nil
	}
	return nil
}

// prefixAffinityRig returns (agent_user_id, agent_id, cached_tokens) of
// the most-recently-updated affinity row for (owner, prefix_hash), or
// zero values if none.  This is the routing hint the dispatcher uses.
func (s *server) prefixAffinityRig(ctx context.Context, ownerUID int64, prefixHash string) (int64, string, int64, error) {
	if prefixHash == "" {
		return 0, "", 0, nil
	}
	row := s.dbQueryRowCtx(ctx,
		`SELECT agent_user_id, agent_id, cached_tokens
		   FROM prefix_affinity
		   WHERE user_id = ? AND prefix_hash = ?
		   ORDER BY updated_at DESC
		   LIMIT 1`,
		ownerUID, prefixHash,
	)
	var auid, ct int64
	var aid string
	if err := row.Scan(&auid, &aid, &ct); err != nil {
		if err == sql.ErrNoRows {
			return 0, "", 0, nil
		}
		return 0, "", 0, err
	}
	return auid, aid, ct, nil
}

// ─── JSON shape for /api/console/prefix-affinity (debug) ──────────────

type prefixAffinityRow struct {
	PrefixHash    string `json:"prefix_hash"`
	AgentUserID   int64  `json:"agent_user_id"`
	AgentID       string `json:"agent_id"`
	CachedTokens  int64  `json:"cached_tokens"`
	UpdatedAt     int64  `json:"updated_at"`
}

func (s *server) listPrefixAffinity(ctx context.Context, ownerUID int64) ([]prefixAffinityRow, error) {
	rows, err := s.dbQueryCtx(ctx,
		`SELECT prefix_hash, agent_user_id, agent_id, cached_tokens, updated_at
		   FROM prefix_affinity
		   WHERE user_id = ?
		   ORDER BY updated_at DESC
		   LIMIT 64`,
		ownerUID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []prefixAffinityRow
	for rows.Next() {
		var r prefixAffinityRow
		if err := rows.Scan(&r.PrefixHash, &r.AgentUserID, &r.AgentID, &r.CachedTokens, &r.UpdatedAt); err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

// jsonPrefixAffinityList marshals to a compact JSON array.  Used by the
// console endpoint; kept separate so tests can assert the wire shape.
func jsonPrefixAffinityList(rows []prefixAffinityRow) []byte {
	if rows == nil {
		rows = []prefixAffinityRow{}
	}
	b, _ := json.Marshal(rows)
	return b
}
