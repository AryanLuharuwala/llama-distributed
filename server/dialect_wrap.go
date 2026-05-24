package main

// P10: Query-time placeholder rewrite wrappers.
//
// The bulk of the codebase uses SQLite `?` placeholders, but on Postgres
// (and CockroachDB / Spanner-via-PGAdapter) those must become `$1, $2…`.
// dialect.go provides RewriteQuery; this file exposes thin server-method
// wrappers that auto-apply it at every call site, so a Postgres deploy
// "just works" without each caller remembering to wrap.
//
// The wrappers are idempotent for queries that don't contain `?`
// (RewriteQuery is a no-op then), so it's safe to call them even from
// code paths that already had RewriteQuery applied — handy during the
// transitional rollout.
//
// We also wrap *sql.Tx for the transactional callsites; the tx pointer
// is the second argument so the wrapper reads naturally
// (`s.txExec(tx, "INSERT INTO …", args...)`).
//
// Why methods on *server (not free functions taking a dialect)?
// Convenience: every caller of these is already on a *server receiver,
// so this lets the call site drop from `s.db.Exec(s.dialect.RewriteQuery(q), …)`
// to `s.dbExec(q, …)`.

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strings"
	"time"
)

// ─── *sql.DB wrappers ──────────────────────────────────────────────────────

func (s *server) dbExec(query string, args ...any) (sql.Result, error) {
	return s.db.Exec(s.dialect.RewriteQuery(query), args...)
}

func (s *server) dbExecCtx(ctx context.Context, query string, args ...any) (sql.Result, error) {
	return s.db.ExecContext(ctx, s.dialect.RewriteQuery(query), args...)
}

func (s *server) dbQuery(query string, args ...any) (*sql.Rows, error) {
	return s.db.Query(s.dialect.RewriteQuery(query), args...)
}

func (s *server) dbQueryCtx(ctx context.Context, query string, args ...any) (*sql.Rows, error) {
	return s.db.QueryContext(ctx, s.dialect.RewriteQuery(query), args...)
}

func (s *server) dbQueryRow(query string, args ...any) *sql.Row {
	return s.db.QueryRow(s.dialect.RewriteQuery(query), args...)
}

func (s *server) dbQueryRowCtx(ctx context.Context, query string, args ...any) *sql.Row {
	return s.db.QueryRowContext(ctx, s.dialect.RewriteQuery(query), args...)
}

// ─── *sql.Tx wrappers ──────────────────────────────────────────────────────

func (s *server) txExec(tx *sql.Tx, query string, args ...any) (sql.Result, error) {
	return tx.Exec(s.dialect.RewriteQuery(query), args...)
}

func (s *server) txExecCtx(ctx context.Context, tx *sql.Tx, query string, args ...any) (sql.Result, error) {
	return tx.ExecContext(ctx, s.dialect.RewriteQuery(query), args...)
}

func (s *server) txQueryRow(tx *sql.Tx, query string, args ...any) *sql.Row {
	return tx.QueryRow(s.dialect.RewriteQuery(query), args...)
}

// ─── Retry: serializable-isolation aware ───────────────────────────────────

// retryableSerializationErr returns true for Postgres SQLSTATE 40001
// (serialization failure) and the equivalent CockroachDB / Spanner
// retryable txn errors.  Used by dbDoTx to wrap a transaction in a
// bounded retry loop — Cockroach in particular needs this because every
// transaction can fail with 40001 under contention.
func retryableSerializationErr(err error) bool {
	if err == nil {
		return false
	}
	msg := err.Error()
	if strings.Contains(msg, "40001") || strings.Contains(msg, "SQLSTATE 40001") {
		return true
	}
	// CockroachDB string form (older versions): "restart transaction:".
	if strings.Contains(msg, "restart transaction") {
		return true
	}
	// Spanner ABORTED.
	if strings.Contains(strings.ToLower(msg), "transaction aborted") {
		return true
	}
	return false
}

// dbDoTx runs `body` inside a transaction with bounded retries on
// serializable-isolation conflicts.  SQLite never hits 40001 so the
// retry loop fires only on Postgres-family backends; this means
// SQLite-only deployments pay one branch per Begin and nothing else.
//
// The closure receives `*sql.Tx`; commit is handled here.  If body
// returns a retryable error, we roll back and try again (up to
// maxRetries times, with a 50ms × n backoff).
func (s *server) dbDoTx(ctx context.Context, body func(tx *sql.Tx) error) error {
	const maxRetries = 4
	var lastErr error
	for attempt := 0; attempt < maxRetries; attempt++ {
		tx, err := s.db.BeginTx(ctx, nil)
		if err != nil {
			return fmt.Errorf("begin tx: %w", err)
		}
		err = body(tx)
		if err != nil {
			_ = tx.Rollback()
			if retryableSerializationErr(err) {
				lastErr = err
				time.Sleep(time.Duration(50*(attempt+1)) * time.Millisecond)
				continue
			}
			return err
		}
		if err := tx.Commit(); err != nil {
			if retryableSerializationErr(err) {
				lastErr = err
				time.Sleep(time.Duration(50*(attempt+1)) * time.Millisecond)
				continue
			}
			return fmt.Errorf("commit: %w", err)
		}
		return nil
	}
	if lastErr == nil {
		lastErr = errors.New("transaction failed without specific error")
	}
	return fmt.Errorf("retries exhausted: %w", lastErr)
}
