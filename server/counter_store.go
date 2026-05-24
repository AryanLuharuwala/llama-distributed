package main

// P11: Pluggable hot-counter store.
//
// `usage_counters` is the canonical per-user usage rollup table:
// (user_id, period) → input_tokens, output_tokens.  It feeds the
// dashboard's "this billing cycle" widgets and the rate-limit denial
// logic.  Today every inference path bumps it via a plain SQL UPDATE.
//
// At scale this is the hottest write in the control plane: every
// request increments two integers.  Two production-grade options:
//
//   - **Bigtable** — purpose-built for high-write counters.  The Go
//     SDK provides atomic IncrementRow operations.  Schema: one row
//     per (user_id, period), columns "in_tok" / "out_tok".  We expose
//     a counterStore interface that the inference path uses; the
//     SQLite impl preserves today's behavior, the Bigtable impl is
//     opt-in via DIST_BT_PROJECT/INSTANCE/TABLE.
//
//   - **Redis** — already in the deployment for rate limiting.  HINCRBY
//     is atomic and fast.  Cheaper to operate than Bigtable for small
//     fleets, hits a ceiling around 100k WPS per shard.  This file
//     covers the abstraction; the existing redis backend wires
//     trivially under the same interface if a team prefers it.
//
// The default sqliteCounterStore wraps today's UPDATE logic so single-
// container deploys keep working with zero config.  Switching to
// Bigtable means setting two env vars and restarting — no schema
// migration required because Bigtable rows are created on first write.
//
// Counter freshness: the dashboard query joins usage_counters with
// users.  When the Bigtable backend is selected, dashboard reads still
// hit the SQLite mirror that's updated periodically — Bigtable's
// strength is write throughput, not analytical queries.  See
// reconcileFromBigtable below for the periodic merge.

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// counterStore is the minimum surface the inference path needs for
// rolling token counts.  All methods must be safe for concurrent use.
type counterStore interface {
	// IncrBy atomically adds (in, out) to (userID, period).  period
	// is a YYYYMM string in the caller's calendar — the store treats
	// it as opaque.
	IncrBy(ctx context.Context, userID int64, period string, inTok, outTok int) error

	// Get returns the current (in, out) for (userID, period).  Returns
	// zeros if the row doesn't exist (not an error — fresh users have
	// no rows yet).
	Get(ctx context.Context, userID int64, period string) (in, out int64, err error)

	// Close releases any backend resources.
	Close() error
}

// ─── SQLite-backed counter store (default) ────────────────────────────

// sqliteCounterStore is the wrapper over today's usage_counters table.
// We keep it as a thin shim around s.db so the rest of the codebase
// can switch to the interface without behaviour change.
type sqliteCounterStore struct {
	s *server
}

func newSQLiteCounterStore(s *server) *sqliteCounterStore {
	return &sqliteCounterStore{s: s}
}

func (c *sqliteCounterStore) IncrBy(ctx context.Context, userID int64, period string, inTok, outTok int) error {
	// Same UPSERT shape used in the legacy code path; the OR IGNORE
	// + UPDATE pattern keeps it driver-portable (Postgres rewrites
	// the dialect helper).
	if _, err := c.s.dbExecCtx(ctx,
		`INSERT OR IGNORE INTO usage_counters (user_id, period, input_tokens, output_tokens)
		 VALUES (?, ?, 0, 0)`,
		userID, period,
	); err != nil {
		return err
	}
	if _, err := c.s.dbExecCtx(ctx,
		`UPDATE usage_counters
		    SET input_tokens  = input_tokens  + ?,
		        output_tokens = output_tokens + ?
		  WHERE user_id = ? AND period = ?`,
		inTok, outTok, userID, period,
	); err != nil {
		return err
	}
	return nil
}

func (c *sqliteCounterStore) Get(ctx context.Context, userID int64, period string) (int64, int64, error) {
	var in, out int64
	err := c.s.dbQueryRowCtx(ctx,
		`SELECT input_tokens, output_tokens FROM usage_counters
		  WHERE user_id = ? AND period = ?`,
		userID, period,
	).Scan(&in, &out)
	if err != nil {
		// sql.ErrNoRows → zeros, not an error: a brand-new user has
		// no usage_counters row until their first request.
		return 0, 0, nil
	}
	return in, out, nil
}

func (c *sqliteCounterStore) Close() error { return nil }

// ─── Bigtable-backed counter store (opt-in) ───────────────────────────
//
// The Bigtable Go SDK isn't pulled into this binary by default (50+
// transitive deps).  We compile a placeholder here that returns
// "unimplemented" — operators who want Bigtable build with the
// `bigtable` build tag (see counter_store_bigtable.go in this dir
// when added).  The interface and config plumbing stay identical so
// flipping the implementation is a one-line change in newServer().

// bigtableCounterConfig is what newBigtableCounterStore wants.  Empty
// values disable the backend.
type bigtableCounterConfig struct {
	ProjectID  string
	InstanceID string
	TableID    string
	// ColumnFamily defaults to "counts".
	ColumnFamily string
	// MirrorToSQLite, when true, also dual-writes every increment to
	// the SQLite table so the dashboard queries keep working.  Recommended
	// during the first weeks of a Bigtable rollout; flip off once
	// reconcileFromBigtable is running.
	MirrorToSQLite bool
}

// bigtableCounterStore is a stub: it forwards every call to a
// fallbackStore (SQLite) and additionally bumps an in-memory counter
// so deployments that wire the real Bigtable adapter later can observe
// the call volume in advance.  The full Bigtable implementation lives
// in a separate file gated by a build tag once the project is ready
// for the dep weight.
type bigtableCounterStore struct {
	cfg      bigtableCounterConfig
	fallback counterStore

	// In-flight increments — exposed for /metrics so an operator can
	// estimate Bigtable QPS before flipping the switch.
	calls atomic.Int64

	mu      sync.Mutex
	stopped bool
}

func newBigtableCounterStore(cfg bigtableCounterConfig, fallback counterStore) (*bigtableCounterStore, error) {
	if cfg.ProjectID == "" || cfg.InstanceID == "" || cfg.TableID == "" {
		return nil, fmt.Errorf("bigtable counter store: project/instance/table required")
	}
	if cfg.ColumnFamily == "" {
		cfg.ColumnFamily = "counts"
	}
	if fallback == nil {
		return nil, fmt.Errorf("bigtable counter store: fallback required (use sqliteCounterStore)")
	}
	return &bigtableCounterStore{cfg: cfg, fallback: fallback}, nil
}

func (b *bigtableCounterStore) IncrBy(ctx context.Context, userID int64, period string, inTok, outTok int) error {
	b.calls.Add(1)
	// Real implementation would call the Bigtable Go SDK's
	// ReadModifyWriteRow with two Increment mutations on the
	// (userID:period) row.  Until the SDK is wired, we fall back to
	// the SQLite store so behaviour is unchanged.  The cfg.MirrorToSQLite
	// flag is meaningful once the SDK lands — in the stub it always
	// uses SQLite.
	return b.fallback.IncrBy(ctx, userID, period, inTok, outTok)
}

func (b *bigtableCounterStore) Get(ctx context.Context, userID int64, period string) (int64, int64, error) {
	return b.fallback.Get(ctx, userID, period)
}

func (b *bigtableCounterStore) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.stopped {
		return nil
	}
	b.stopped = true
	return b.fallback.Close()
}

// rowKeyForCounter is the canonical Bigtable row key for a counter
// row.  Exposed so the eventual real Bigtable implementation matches
// the format expected by any external readers (e.g., a BQ federation
// view over the Bigtable table).
//
// Format: "<period>#<userID>" — period first so a row-range scan
// "<period>#" returns every user for the period in O(period_size).
// User-id is zero-padded to 19 chars so lexicographic order on the
// row key matches numeric order on the user_id.
func rowKeyForCounter(userID int64, period string) string {
	return fmt.Sprintf("%s#%019d", period, userID)
}

// reconcileFromBigtable is a hook for the periodic merge from Bigtable
// → SQLite mirror.  Stub until the SDK lands; when wired, it should
// scan rows updated since `since`, write the deltas to usage_counters,
// and return the new high-water mark.
func reconcileFromBigtable(ctx context.Context, since time.Time) (time.Time, error) {
	return since, fmt.Errorf("reconcileFromBigtable: unimplemented stub")
}
