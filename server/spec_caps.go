package main

// P16: speculative-decoding capability advertisement.
//
// Speculative decoding (Medusa heads / Eagle / draft-model) lets a rig
// emit K candidate tokens per target-model forward pass.  Acceptance
// rates of 1.5–3× are common, which is the difference between
// "interactive" and "fast" for long completions.  The actual heads
// live in the backing runtime (vLLM, SGLang, TRT-LLM) or in a
// llama.cpp draft-model pair; this control-plane table just records
// which rigs claim the capability and what they expect to deliver, so
// the dispatcher can prefer them for latency-sensitive endpoints
// (/v1/chat/completions with stream:true, small max_tokens).
//
// Wire frame (from agent):
//
//   {
//     "kind": "spec_caps",
//     "ok": true,
//     "method": "medusa" | "eagle" | "draft_model" | "none",
//     "draft_tokens": 4,           // K = number of speculated tokens per step
//     "accept_rate_hint": 0.62     // optional, server averages over time
//   }
//
// Storage: `spec_caps` table keyed by (user_id, agent_id).  Routing
// integration is deferred to a follow-up — this PR establishes the
// capability surface end-to-end.

import (
	"database/sql"
)

func migrateSpecCaps(db *sql.DB, d sqlDialect) error {
	if d == nil {
		d = sqliteDialect{}
	}
	stmts := []string{
		`CREATE TABLE IF NOT EXISTS spec_caps (
			user_id          INTEGER NOT NULL,
			agent_id         TEXT NOT NULL,
			ok               INTEGER NOT NULL DEFAULT 0,
			method           TEXT NOT NULL DEFAULT '',
			draft_tokens     INTEGER NOT NULL DEFAULT 0,
			accept_rate_hint REAL NOT NULL DEFAULT 0,
			updated_at       INTEGER NOT NULL,
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

// upsertSpecCaps stores a spec_caps frame from the agent.
func (s *server) upsertSpecCaps(uid int64, agentID string, msg map[string]any) {
	ok := 0
	if v, _ := msg["ok"].(bool); v {
		ok = 1
	}
	method, _ := msg["method"].(string)
	if method == "" {
		method = "none"
	}
	// Whitelist methods so a bogus agent string can't bloat the column.
	switch method {
	case "medusa", "eagle", "draft_model", "lookahead", "none":
	default:
		method = "none"
	}
	var draftTokens int64
	switch v := msg["draft_tokens"].(type) {
	case float64:
		draftTokens = int64(v)
	case int64:
		draftTokens = v
	}
	if draftTokens < 0 {
		draftTokens = 0
	}
	if draftTokens > 32 {
		draftTokens = 32 // sane upper bound; nobody runs >32 heads
	}
	var acceptHint float64
	if v, ok2 := msg["accept_rate_hint"].(float64); ok2 {
		acceptHint = v
	}
	if acceptHint < 0 {
		acceptHint = 0
	}
	if acceptHint > 1 {
		acceptHint = 1
	}
	_, _ = s.dbExec(
		`INSERT INTO spec_caps (user_id, agent_id, ok, method, draft_tokens, accept_rate_hint, updated_at)
		 VALUES (?, ?, ?, ?, ?, ?, ?)
		 ON CONFLICT(user_id, agent_id) DO UPDATE SET
		   ok = excluded.ok,
		   method = excluded.method,
		   draft_tokens = excluded.draft_tokens,
		   accept_rate_hint = excluded.accept_rate_hint,
		   updated_at = excluded.updated_at`,
		uid, agentID, ok, method, draftTokens, acceptHint, nowUnix(),
	)
}
