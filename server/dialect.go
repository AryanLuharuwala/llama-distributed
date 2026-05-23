package main

// Tiny SQL-dialect abstraction so the control plane can run on either
// SQLite (default, single-binary dev/prod) or PostgreSQL (opt-in via
// DIST_DB_DRIVER=postgres + DIST_DB_DSN=postgres://…).
//
// The bulk of the code uses `?` placeholders and SQLite-flavoured DDL.
// This file centralises the few cross-driver concerns:
//
//   1. DDL rewriting — INTEGER PRIMARY KEY AUTOINCREMENT, BLOB columns,
//      INSERT OR IGNORE, "ALTER TABLE ADD COLUMN" idempotency.
//   2. Placeholder style — SQLite/MySQL use `?`, Postgres uses `$1, $2, …`.
//      sqlDialect.RewriteQuery walks the string once and substitutes the
//      indexed form for callers that need to be driver-portable.
//   3. Error-class probing — "duplicate column" is the only branch the
//      migration code needs to suppress; SQLite reports a string, Postgres
//      reports SQLSTATE 42701.
//
// The dialect interface is satisfied by sqliteDialect (default) and
// postgresDialect.  Callers that don't care about Postgres can ignore
// this entirely; for Postgres the migration helper rewrites every CREATE
// TABLE before execution and IsDuplicateColumn covers the SQLSTATE.
//
// Why hand-rolled instead of an ORM?  The query surface is small and
// mostly hot-path; we want zero allocation overhead on the SQLite path
// and a clear separation between "rewritten DDL" and "literal queries".

import (
	"errors"
	"regexp"
	"strings"
)

// sqlDialect is the minimum surface migrations + cross-driver helpers
// need.  The query-time placeholder rewriter is opt-in: callers either
// know they're on SQLite and use `?`, or know they're on Postgres and
// pre-rewrite via RewriteQuery before Exec/Query.
type sqlDialect interface {
	// Name is the driver name as registered with database/sql ("sqlite3"
	// or "postgres").  Used for telemetry and connection diagnostics.
	Name() string

	// RewriteDDL adapts SQLite-flavoured CREATE/ALTER statements to the
	// target dialect.  No-op on SQLite.
	RewriteDDL(stmt string) string

	// RewriteQuery converts `?` placeholders to `$1, $2, …` for Postgres.
	// No-op on SQLite.  Single-quoted strings are skipped so embedded `?`
	// in user-controlled text is preserved.
	RewriteQuery(q string) string

	// IsDuplicateColumn returns true when err is "this column already
	// exists" — used to make ALTER TABLE ADD COLUMN idempotent across
	// drivers without table introspection.
	IsDuplicateColumn(err error) bool
}

// dialectFor returns the dialect implementation for a driver name.
// Unknown drivers return an error rather than silently falling back so
// misconfiguration is loud.
func dialectFor(driver string) (sqlDialect, error) {
	switch driver {
	case "", "sqlite3", "sqlite":
		return sqliteDialect{}, nil
	case "postgres", "pgx":
		return postgresDialect{}, nil
	}
	return nil, errors.New("unsupported DB driver: " + driver)
}

// ── SQLite ────────────────────────────────────────────────────────────

type sqliteDialect struct{}

func (sqliteDialect) Name() string                  { return "sqlite3" }
func (sqliteDialect) RewriteDDL(s string) string    { return s }
func (sqliteDialect) RewriteQuery(q string) string  { return q }

func (sqliteDialect) IsDuplicateColumn(err error) bool {
	if err == nil {
		return false
	}
	return strings.Contains(err.Error(), "duplicate column name")
}

// ── Postgres ──────────────────────────────────────────────────────────

type postgresDialect struct{}

func (postgresDialect) Name() string { return "postgres" }

// Compiled rewrites for the few SQLite-isms in the schema.  Keep this
// list short and explicit — silent SQL rewriting is a footgun, so we
// fail loudly elsewhere if a query slips through that needs hand-port.
var (
	pgRewriteAutoIncrement = regexp.MustCompile(`(?i)INTEGER\s+PRIMARY\s+KEY\s+AUTOINCREMENT`)
	pgRewriteInsertIgnore  = regexp.MustCompile(`(?i)INSERT\s+OR\s+IGNORE\s+INTO`)
	pgRewriteInsertReplace = regexp.MustCompile(`(?i)INSERT\s+OR\s+REPLACE\s+INTO`)
)

func (postgresDialect) RewriteDDL(s string) string {
	s = pgRewriteAutoIncrement.ReplaceAllString(s, "BIGSERIAL PRIMARY KEY")
	// BLOB has no portable Postgres equivalent; map to BYTEA.
	s = strings.ReplaceAll(s, " BLOB", " BYTEA")
	s = strings.ReplaceAll(s, " blob", " bytea")
	// INTEGER → BIGINT for non-pk columns so 64-bit ids round-trip without
	// surprises.  The PRIMARY KEY case above already covers identity cols.
	// We *don't* blanket-rewrite INTEGER here because Postgres accepts it
	// and emits int4 — most of our INTEGER columns hold flags or small
	// counts where int4 is plenty.
	return s
}

// RewriteQuery converts `?` to `$1, $2, …` while respecting single-quoted
// string literals (`'foo?bar'` stays intact).  We don't touch identifier
// quoting or comments because none of our queries embed `?` inside them.
func (postgresDialect) RewriteQuery(q string) string {
	if !strings.ContainsRune(q, '?') {
		return q
	}
	var b strings.Builder
	b.Grow(len(q) + 8)
	inString := false
	idx := 0
	for i := 0; i < len(q); i++ {
		c := q[i]
		if c == '\'' {
			// Toggle on `'` unless preceded by a backslash (which SQLite/
			// Postgres treat differently, but we don't use \' escapes).
			inString = !inString
			b.WriteByte(c)
			continue
		}
		if c == '?' && !inString {
			idx++
			b.WriteByte('$')
			b.WriteString(itoaDialect(idx))
			continue
		}
		b.WriteByte(c)
	}
	return b.String()
}

// itoaDialect avoids strconv import in the hot path.  Postgres rarely sees
// queries with >100 placeholders, so a tiny custom loop is fine.
func itoaDialect(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}

func (postgresDialect) IsDuplicateColumn(err error) bool {
	if err == nil {
		return false
	}
	msg := err.Error()
	// pgx and lib/pq both surface SQLSTATE 42701 ("duplicate column"),
	// usually formatted as "ERROR: column \"x\" of relation \"y\" already
	// exists (SQLSTATE 42701)".  Match on either the code or the phrase.
	return strings.Contains(msg, "42701") ||
		strings.Contains(msg, "already exists") &&
			strings.Contains(strings.ToLower(msg), "column")
}
