package main

// Versioned-migration applier.
//
// Why this exists alongside the imperative CREATE TABLE IF NOT EXISTS
// migrations in db.go / *_migrate.go: those handle the legacy baseline
// (idempotent, work on every boot, are what existing deploys already
// ran).  This file handles *new* schema changes — ALTER TABLE, new
// indexes, new tables — as discrete files an operator can review,
// `atlas migrate diff`, and replay in a known order.
//
// Storage layout:
//
//   server/migrations/
//     20260524000001_baseline.sql       — empty marker, see comment in file
//     20260524000002_<name>.sql         — first real change
//     20260524000003_<name>.sql         — second, etc.
//
// Naming follows atlas's convention (YYYYMMDDHHMMSS_<slug>.sql) so the
// operator can run `atlas migrate diff` against the production DB and
// drop the resulting file straight into this directory.  We don't ship
// atlas as a runtime dep — the applier here is ~60 lines of SQL — but
// the format is interchangeable.
//
// Applied versions are tracked in `schema_migrations` (single row per
// version, with `applied_at` for the audit trail).  Each migration
// runs in its own transaction; a partial application is rolled back
// and the boot fails fast rather than leaving the DB in an undefined
// state.
//
// Concurrency: when N replicas boot at once they race to claim each
// migration.  We rely on the UNIQUE constraint on `version` to break
// the race — losers get a constraint violation, see the version is
// already applied, and skip.  Postgres-safe; SQLite is single-writer
// so the race is moot.

import (
	"context"
	"database/sql"
	"embed"
	"errors"
	"fmt"
	"io/fs"
	"log"
	"sort"
	"strings"
	"time"
)

//go:embed migrations/*.sql
var migrationFS embed.FS

// applyVersionedMigrations applies every .sql file under migrations/
// that hasn't been recorded in schema_migrations yet, in lexical order.
// Safe to call repeatedly; safe to call concurrently from multiple
// replicas (the UNIQUE(version) on schema_migrations breaks the race).
//
// dialect controls SQL idiosyncrasies — Postgres wants different
// syntax for the tracking table, but we keep it dialect-neutral by
// using only SQL-standard features (INTEGER PRIMARY KEY, TEXT UNIQUE,
// INTEGER for timestamps).
func applyVersionedMigrations(ctx context.Context, db *sql.DB, d sqlDialect) error {
	if err := ensureMigrationsTable(ctx, db); err != nil {
		return fmt.Errorf("ensure schema_migrations: %w", err)
	}

	files, err := listMigrationFiles()
	if err != nil {
		return fmt.Errorf("list migrations: %w", err)
	}

	applied, err := loadAppliedVersions(ctx, db)
	if err != nil {
		return fmt.Errorf("load applied: %w", err)
	}

	for _, f := range files {
		version := versionFromFilename(f.name)
		if version == "" {
			log.Printf("schema_migrations: skipping non-conforming file %q", f.name)
			continue
		}
		if applied[version] {
			continue
		}
		if err := applyOne(ctx, db, version, f.name, f.body); err != nil {
			return err
		}
	}
	return nil
}

func ensureMigrationsTable(ctx context.Context, db *sql.DB) error {
	// One row per applied migration.  applied_at is unix-seconds — same
	// shape as every other timestamp column in this codebase, so an
	// operator scripting `SELECT version FROM schema_migrations` gets
	// uniform data.
	_, err := db.ExecContext(ctx, `
		CREATE TABLE IF NOT EXISTS schema_migrations (
			version    TEXT NOT NULL UNIQUE,
			applied_at INTEGER NOT NULL,
			PRIMARY KEY (version)
		)`)
	return err
}

func loadAppliedVersions(ctx context.Context, db *sql.DB) (map[string]bool, error) {
	rows, err := db.QueryContext(ctx, `SELECT version FROM schema_migrations`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := map[string]bool{}
	for rows.Next() {
		var v string
		if err := rows.Scan(&v); err != nil {
			return nil, err
		}
		out[v] = true
	}
	return out, rows.Err()
}

type migrationFile struct {
	name string
	body string
}

func listMigrationFiles() ([]migrationFile, error) {
	entries, err := fs.ReadDir(migrationFS, "migrations")
	if err != nil {
		// An empty embed (no files matched) is fine — means we have
		// nothing to apply yet.
		var pathErr *fs.PathError
		if errors.As(err, &pathErr) {
			return nil, nil
		}
		return nil, err
	}
	names := make([]string, 0, len(entries))
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".sql") {
			continue
		}
		names = append(names, e.Name())
	}
	sort.Strings(names)
	out := make([]migrationFile, 0, len(names))
	for _, n := range names {
		body, err := fs.ReadFile(migrationFS, "migrations/"+n)
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", n, err)
		}
		out = append(out, migrationFile{name: n, body: string(body)})
	}
	return out, nil
}

// versionFromFilename pulls the leading numeric timestamp out of an
// atlas-style filename: "20260524000001_baseline.sql" → "20260524000001".
// Returns "" for non-conforming names so the caller can skip them.
func versionFromFilename(name string) string {
	i := strings.IndexByte(name, '_')
	if i <= 0 {
		return ""
	}
	v := name[:i]
	for _, c := range v {
		if c < '0' || c > '9' {
			return ""
		}
	}
	return v
}

func applyOne(ctx context.Context, db *sql.DB, version, name, body string) error {
	body = strings.TrimSpace(body)
	// Empty migrations are legal — useful for "baseline" markers that
	// only exist to set the starting version after a legacy schema.
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("begin %s: %w", name, err)
	}
	defer tx.Rollback() //nolint:errcheck — superseded by Commit on success.

	if body != "" {
		// Apply the whole file as one Exec.  Most drivers (sqlite3,
		// lib/pq, pgx) accept multi-statement payloads in a single
		// ExecContext; if a future migration needs per-statement
		// handling, split it into multiple files.
		if _, err := tx.ExecContext(ctx, body); err != nil {
			return fmt.Errorf("apply %s: %w", name, err)
		}
	}

	if _, err := tx.ExecContext(ctx,
		`INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)`,
		version, time.Now().Unix(),
	); err != nil {
		// Lost the race with another replica — they applied it first.
		// Treat as success: our transaction will rollback the SQL we
		// just ran, but since their version of the same SQL already
		// landed, the end state is identical.
		if isUniqueConstraintErr(err) {
			log.Printf("schema_migrations: %s already applied by another replica", version)
			return nil
		}
		return fmt.Errorf("record %s: %w", name, err)
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit %s: %w", name, err)
	}
	log.Printf("schema_migrations: applied %s", name)
	return nil
}

// isUniqueConstraintErr detects "duplicate key" errors across the
// three drivers we support.  String matching rather than typed errors
// because lib/pq returns a typed error, pgx returns its own, and
// mattn/sqlite3 returns a generic Error — string match is the only
// portable lowest common denominator.
func isUniqueConstraintErr(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "unique") || strings.Contains(msg, "duplicate")
}
