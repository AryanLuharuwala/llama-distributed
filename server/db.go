package main

import (
	"database/sql"
	"strings"
)

// addColumnIfMissing runs an ALTER TABLE ADD COLUMN, silently ignoring
// "duplicate column" errors so migrations are idempotent.
func addColumnIfMissing(db *sql.DB, stmt string) error {
	_, err := db.Exec(stmt)
	if err == nil {
		return nil
	}
	if strings.Contains(err.Error(), "duplicate column name") {
		return nil
	}
	return err
}

func migrate(db *sql.DB) error {
	stmts := []string{
		`CREATE TABLE IF NOT EXISTS users (
			id            INTEGER PRIMARY KEY AUTOINCREMENT,
			github_id     INTEGER UNIQUE,
			github_login  TEXT,
			display_name  TEXT NOT NULL,
			created_at    INTEGER NOT NULL
		)`,
		`CREATE TABLE IF NOT EXISTS sessions (
			id          TEXT PRIMARY KEY,
			user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
			created_at  INTEGER NOT NULL,
			expires_at  INTEGER NOT NULL
		)`,
		`CREATE TABLE IF NOT EXISTS pair_tokens (
			token       TEXT PRIMARY KEY,
			user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
			created_at  INTEGER NOT NULL,
			expires_at  INTEGER NOT NULL,
			used_at     INTEGER
		)`,
		`CREATE TABLE IF NOT EXISTS rigs (
			id          INTEGER PRIMARY KEY AUTOINCREMENT,
			user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
			agent_id    TEXT NOT NULL,
			hostname    TEXT NOT NULL,
			n_gpus      INTEGER NOT NULL DEFAULT 0,
			vram_bytes  INTEGER NOT NULL DEFAULT 0,
			last_seen   INTEGER NOT NULL,
			UNIQUE (user_id, agent_id)
		)`,
		`CREATE INDEX IF NOT EXISTS idx_rigs_user ON rigs(user_id)`,
		`CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at)`,

		// Pools: a named bag of rigs with membership.  visibility is one of
		// 'private' (owner only), 'invite' (owner + invited users), or 'public'
		// (anyone logged in can join).
		`CREATE TABLE IF NOT EXISTS pools (
			id          INTEGER PRIMARY KEY AUTOINCREMENT,
			owner_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
			name        TEXT NOT NULL,
			visibility  TEXT NOT NULL CHECK (visibility IN ('private','invite','public')),
			created_at  INTEGER NOT NULL
		)`,
		`CREATE INDEX IF NOT EXISTS idx_pools_owner      ON pools(owner_id)`,
		`CREATE INDEX IF NOT EXISTS idx_pools_visibility ON pools(visibility)`,

		`CREATE TABLE IF NOT EXISTS pool_members (
			pool_id    INTEGER NOT NULL REFERENCES pools(id) ON DELETE CASCADE,
			user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
			role       TEXT NOT NULL CHECK (role IN ('owner','member')),
			joined_at  INTEGER NOT NULL,
			PRIMARY KEY (pool_id, user_id)
		)`,
		`CREATE INDEX IF NOT EXISTS idx_members_user ON pool_members(user_id)`,

		`CREATE TABLE IF NOT EXISTS pool_rigs (
			pool_id   INTEGER NOT NULL REFERENCES pools(id) ON DELETE CASCADE,
			rig_id    INTEGER NOT NULL REFERENCES rigs(id)  ON DELETE CASCADE,
			added_at  INTEGER NOT NULL,
			PRIMARY KEY (pool_id, rig_id)
		)`,
		`CREATE INDEX IF NOT EXISTS idx_pool_rigs_rig ON pool_rigs(rig_id)`,

		`CREATE TABLE IF NOT EXISTS pool_invites (
			token       TEXT PRIMARY KEY,
			pool_id     INTEGER NOT NULL REFERENCES pools(id) ON DELETE CASCADE,
			created_by  INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
			created_at  INTEGER NOT NULL,
			expires_at  INTEGER NOT NULL,
			used_at     INTEGER
		)`,
		`CREATE INDEX IF NOT EXISTS idx_invites_pool ON pool_invites(pool_id)`,

		// Per-user rate-limit policy.  req_per_min is a short-window cap
		// (sliding minute); tokens_per_month is a soft budget reset via
		// period_start.  All counts accrue in usage_counters.
		`CREATE TABLE IF NOT EXISTS rate_limits (
			user_id           INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
			req_per_min       INTEGER NOT NULL DEFAULT 30,
			tokens_per_month  INTEGER NOT NULL DEFAULT 1000000,
			updated_at        INTEGER NOT NULL
		)`,

		// Usage counters.  One row per user per calendar-month and per
		// rolling-minute bucket.  We don't bother with hot-path writes on
		// every request; /api/infer flushes asynchronously.
		`CREATE TABLE IF NOT EXISTS usage_counters (
			user_id      INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
			period       TEXT    NOT NULL,    -- 'YYYY-MM' or 'minute:<unix_minute>'
			requests     INTEGER NOT NULL DEFAULT 0,
			input_tokens INTEGER NOT NULL DEFAULT 0,
			output_tokens INTEGER NOT NULL DEFAULT 0,
			PRIMARY KEY (user_id, period)
		)`,
		`CREATE INDEX IF NOT EXISTS idx_usage_period ON usage_counters(period)`,

		// Inference request log.  Short-lived (purged after N days); used for
		// auditing, receipts, and to credit pool contributors.
		`CREATE TABLE IF NOT EXISTS inference_log (
			id            INTEGER PRIMARY KEY AUTOINCREMENT,
			user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
			pool_id       INTEGER NOT NULL REFERENCES pools(id) ON DELETE CASCADE,
			agent_user_id INTEGER NOT NULL,
			agent_id      TEXT    NOT NULL,
			input_tokens  INTEGER NOT NULL DEFAULT 0,
			output_tokens INTEGER NOT NULL DEFAULT 0,
			started_at    INTEGER NOT NULL,
			finished_at   INTEGER,
			status        TEXT NOT NULL         -- 'ok' | 'rate_limit' | 'failed' | 'no_rig'
		)`,
		`CREATE INDEX IF NOT EXISTS idx_inflog_user ON inference_log(user_id, started_at)`,
		`CREATE INDEX IF NOT EXISTS idx_inflog_agent ON inference_log(agent_user_id, agent_id, started_at)`,

		// Models known to the coordinator.  A model is pre-split into N shard
		// files (one per pipeline stage); the planner maps stages → rigs.
		`CREATE TABLE IF NOT EXISTS models (
			id          INTEGER PRIMARY KEY AUTOINCREMENT,
			name        TEXT UNIQUE NOT NULL,
			n_layers    INTEGER NOT NULL,
			n_shards    INTEGER NOT NULL,
			shards_dir  TEXT NOT NULL,
			created_at  INTEGER NOT NULL
		)`,

	}
	for _, s := range stmts {
		if _, err := db.Exec(s); err != nil {
			return err
		}
	}

	// Idempotent column additions (SQLite has no ALTER TABLE IF NOT EXISTS).
	alters := []string{
		// Parallelism config on pools.  parallelism is one of 'pp', 'tp', 'pp+tp'.
		`ALTER TABLE pools ADD COLUMN parallelism TEXT NOT NULL DEFAULT 'pp'`,
		`ALTER TABLE pools ADD COLUMN model_id    INTEGER REFERENCES models(id)`,
		`ALTER TABLE pools ADD COLUMN pp_stages   INTEGER NOT NULL DEFAULT 1`,
		`ALTER TABLE pools ADD COLUMN tp_size     INTEGER NOT NULL DEFAULT 1`,
		// Rig capability fields needed by the planner.
		`ALTER TABLE rigs  ADD COLUMN n_gpus_available INTEGER NOT NULL DEFAULT 0`,
		// Pool slug — the <slug>.<apex> subdomain that serves the
		// OpenAI-compatible endpoint for this pool.  Backfilled on start
		// for any pool that doesn't have one yet.
		`ALTER TABLE pools ADD COLUMN slug TEXT`,
		// Optional pool the agent should auto-join when it pairs.  Set
		// when the browser chose a pool in the install picker; consumed
		// atomically with the pair token in the WS handshake.
		`ALTER TABLE pair_tokens ADD COLUMN pool_id INTEGER REFERENCES pools(id) ON DELETE SET NULL`,
	}
	for _, s := range alters {
		if err := addColumnIfMissing(db, s); err != nil {
			return err
		}
	}

	// New tables that live outside the initial CREATE block.
	more := []string{
		// API keys for programmatic access via /v1/* on pool subdomains.
		// We store only the prefix (for display, "sk-abcd…") and a sha256
		// of the full key; the plaintext key is returned once at creation.
		`CREATE TABLE IF NOT EXISTS api_keys (
			id           INTEGER PRIMARY KEY AUTOINCREMENT,
			user_id      INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
			label        TEXT NOT NULL DEFAULT '',
			prefix       TEXT NOT NULL,
			hash         TEXT NOT NULL UNIQUE,
			created_at   INTEGER NOT NULL,
			last_used_at INTEGER
		)`,
		`CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys(user_id)`,

		// Unique index on pools.slug.  NULLs are allowed (pre-migration
		// pools) — SQLite ignores NULLs in UNIQUE indexes by default.
		`CREATE UNIQUE INDEX IF NOT EXISTS idx_pools_slug ON pools(slug) WHERE slug IS NOT NULL`,
	}
	for _, s := range more {
		if _, err := db.Exec(s); err != nil {
			return err
		}
	}
	return nil
}
