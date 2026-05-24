package main

import (
	"context"
	"database/sql"
	"testing"

	_ "github.com/mattn/go-sqlite3"
)

// memDB spins up an in-memory SQLite for one test.  Each test gets its
// own DB so they can run in parallel without state leaks.
func memDB(t *testing.T) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("open mem db: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	return db
}

func TestEnsureMigrationsTable(t *testing.T) {
	db := memDB(t)
	ctx := context.Background()
	if err := ensureMigrationsTable(ctx, db); err != nil {
		t.Fatalf("first create: %v", err)
	}
	// Idempotent — second call must not error.
	if err := ensureMigrationsTable(ctx, db); err != nil {
		t.Fatalf("second create: %v", err)
	}
	var n int
	if err := db.QueryRow(`SELECT COUNT(*) FROM schema_migrations`).Scan(&n); err != nil {
		t.Fatalf("count: %v", err)
	}
	if n != 0 {
		t.Errorf("fresh table should be empty, got %d", n)
	}
}

func TestVersionFromFilename(t *testing.T) {
	cases := []struct {
		name string
		want string
	}{
		{"20260524000001_baseline.sql", "20260524000001"},
		{"20260524000002_add_index.sql", "20260524000002"},
		{"baseline.sql", ""},                 // no prefix
		{"abc_baseline.sql", ""},             // non-numeric prefix
		{"20260524000001-baseline.sql", ""},  // wrong separator
		{"_baseline.sql", ""},                // empty prefix
	}
	for _, c := range cases {
		if got := versionFromFilename(c.name); got != c.want {
			t.Errorf("versionFromFilename(%q) = %q, want %q", c.name, got, c.want)
		}
	}
}

func TestApplyVersionedMigrationsAppliesBaseline(t *testing.T) {
	db := memDB(t)
	ctx := context.Background()
	// The embedded baseline file is empty but its version must be
	// recorded so subsequent migrations build on top of it.
	if err := applyVersionedMigrations(ctx, db, sqliteDialect{}); err != nil {
		t.Fatalf("first apply: %v", err)
	}
	var v string
	err := db.QueryRow(`SELECT version FROM schema_migrations ORDER BY version LIMIT 1`).Scan(&v)
	if err != nil {
		t.Fatalf("read version: %v", err)
	}
	if v != "20260524000001" {
		t.Errorf("expected baseline version recorded, got %q", v)
	}
	// Re-applying must be a no-op.
	if err := applyVersionedMigrations(ctx, db, sqliteDialect{}); err != nil {
		t.Fatalf("second apply: %v", err)
	}
	var n int
	_ = db.QueryRow(`SELECT COUNT(*) FROM schema_migrations`).Scan(&n)
	if n != 1 {
		t.Errorf("second apply should not duplicate rows, got %d", n)
	}
}

func TestApplyOneRecordsVersion(t *testing.T) {
	db := memDB(t)
	ctx := context.Background()
	if err := ensureMigrationsTable(ctx, db); err != nil {
		t.Fatalf("ensure: %v", err)
	}
	// Apply a tiny test migration that actually creates a table —
	// verifies both the SQL execution and the version recording paths.
	body := `CREATE TABLE _test_table (id INTEGER PRIMARY KEY)`
	if err := applyOne(ctx, db, "20990101000001", "test.sql", body); err != nil {
		t.Fatalf("applyOne: %v", err)
	}
	// Table exists.
	var name string
	err := db.QueryRow(`SELECT name FROM sqlite_master WHERE type='table' AND name='_test_table'`).Scan(&name)
	if err != nil {
		t.Errorf("table not created: %v", err)
	}
	// Version recorded.
	var v string
	err = db.QueryRow(`SELECT version FROM schema_migrations WHERE version=?`, "20990101000001").Scan(&v)
	if err != nil {
		t.Errorf("version not recorded: %v", err)
	}
}

func TestApplyOneRollsBackOnSQLError(t *testing.T) {
	db := memDB(t)
	ctx := context.Background()
	if err := ensureMigrationsTable(ctx, db); err != nil {
		t.Fatalf("ensure: %v", err)
	}
	// Syntactically broken SQL.  applyOne must surface the error and
	// must NOT record the version — otherwise a busted migration would
	// be skipped on the next boot, leaving the DB in an indeterminate
	// state.
	body := `CRATE TABLE this_is_not_sql`
	if err := applyOne(ctx, db, "20990101000002", "broken.sql", body); err == nil {
		t.Error("expected error from broken SQL, got nil")
	}
	var n int
	_ = db.QueryRow(`SELECT COUNT(*) FROM schema_migrations WHERE version=?`,
		"20990101000002").Scan(&n)
	if n != 0 {
		t.Errorf("failed migration should not record version, got count=%d", n)
	}
}

func TestIsUniqueConstraintErr(t *testing.T) {
	if !isUniqueConstraintErr(errString("UNIQUE constraint failed: schema_migrations.version")) {
		t.Error("sqlite3 unique error not detected")
	}
	if !isUniqueConstraintErr(errString("pq: duplicate key value violates unique constraint")) {
		t.Error("postgres duplicate error not detected")
	}
	if isUniqueConstraintErr(errString("some unrelated db error")) {
		t.Error("false positive on unrelated error")
	}
	if isUniqueConstraintErr(nil) {
		t.Error("nil must not match")
	}
}

type errString string

func (e errString) Error() string { return string(e) }
