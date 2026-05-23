package main

import (
	"errors"
	"strings"
	"testing"
)

func TestDialectFor(t *testing.T) {
	cases := []struct {
		driver string
		want   string
		err    bool
	}{
		{"", "sqlite3", false},
		{"sqlite", "sqlite3", false},
		{"sqlite3", "sqlite3", false},
		{"postgres", "postgres", false},
		{"pgx", "postgres", false},
		{"mysql", "", true},
		{"junk", "", true},
	}
	for _, c := range cases {
		d, err := dialectFor(c.driver)
		if (err != nil) != c.err {
			t.Errorf("dialectFor(%q) err=%v want_err=%v", c.driver, err, c.err)
			continue
		}
		if err == nil && d.Name() != c.want {
			t.Errorf("dialectFor(%q) name=%q want=%q", c.driver, d.Name(), c.want)
		}
	}
}

func TestSQLiteDialect_NoOpRewrites(t *testing.T) {
	d := sqliteDialect{}
	in := `CREATE TABLE x (id INTEGER PRIMARY KEY AUTOINCREMENT, blob BLOB)`
	if got := d.RewriteDDL(in); got != in {
		t.Errorf("sqlite should not rewrite DDL: %q -> %q", in, got)
	}
	q := `INSERT INTO x (a, b) VALUES (?, ?)`
	if got := d.RewriteQuery(q); got != q {
		t.Errorf("sqlite should not rewrite query: %q -> %q", q, got)
	}
}

func TestPostgresDialect_DDLRewrite(t *testing.T) {
	d := postgresDialect{}
	ddl := `CREATE TABLE x (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		nonce BLOB NOT NULL
	)`
	got := d.RewriteDDL(ddl)
	if !strings.Contains(got, "BIGSERIAL PRIMARY KEY") {
		t.Errorf("did not rewrite AUTOINCREMENT: %s", got)
	}
	if strings.Contains(got, "BLOB") {
		t.Errorf("did not rewrite BLOB to BYTEA: %s", got)
	}
	if !strings.Contains(got, "BYTEA") {
		t.Errorf("BYTEA missing after rewrite: %s", got)
	}
}

func TestPostgresDialect_QueryPlaceholders(t *testing.T) {
	d := postgresDialect{}
	cases := []struct {
		in, want string
	}{
		{"SELECT 1", "SELECT 1"},
		{"INSERT INTO t (a, b, c) VALUES (?, ?, ?)",
			"INSERT INTO t (a, b, c) VALUES ($1, $2, $3)"},
		{"SELECT * FROM t WHERE a = ? AND name = 'has?question'",
			"SELECT * FROM t WHERE a = $1 AND name = 'has?question'"},
		{"UPDATE t SET a = ?, b = ? WHERE id = ?",
			"UPDATE t SET a = $1, b = $2 WHERE id = $3"},
	}
	for _, c := range cases {
		got := d.RewriteQuery(c.in)
		if got != c.want {
			t.Errorf("RewriteQuery(%q)\n  got  %q\n  want %q", c.in, got, c.want)
		}
	}
}

func TestPostgresDialect_PlaceholderCounting(t *testing.T) {
	d := postgresDialect{}
	// 12 placeholders — make sure two-digit indices format correctly.
	in := strings.Repeat("?,", 12)
	in = "(" + strings.TrimRight(in, ",") + ")"
	got := d.RewriteQuery(in)
	if !strings.Contains(got, "$12)") {
		t.Errorf("expected $12 in output: %s", got)
	}
	if !strings.Contains(got, "$1,") {
		t.Errorf("expected $1, in output: %s", got)
	}
}

func TestIsDuplicateColumn(t *testing.T) {
	sq := sqliteDialect{}
	pg := postgresDialect{}

	if !sq.IsDuplicateColumn(errors.New(`duplicate column name: foo`)) {
		t.Errorf("sqlite should detect 'duplicate column name'")
	}
	if sq.IsDuplicateColumn(errors.New("no such table")) {
		t.Errorf("sqlite false positive on unrelated error")
	}
	if sq.IsDuplicateColumn(nil) {
		t.Errorf("nil error must return false")
	}

	if !pg.IsDuplicateColumn(errors.New(`pq: column "foo" of relation "bar" already exists (SQLSTATE 42701)`)) {
		t.Errorf("postgres should detect SQLSTATE 42701")
	}
	if !pg.IsDuplicateColumn(errors.New(`column "x" of relation "y" already exists`)) {
		t.Errorf("postgres should detect 'already exists' phrasing")
	}
	if pg.IsDuplicateColumn(errors.New("connection refused")) {
		t.Errorf("postgres false positive on unrelated error")
	}
}

func TestItoaDialect(t *testing.T) {
	cases := []struct {
		in   int
		want string
	}{
		{0, "0"}, {1, "1"}, {9, "9"}, {10, "10"}, {99, "99"},
		{123, "123"}, {1000, "1000"},
	}
	for _, c := range cases {
		if got := itoaDialect(c.in); got != c.want {
			t.Errorf("itoaDialect(%d)=%q want %q", c.in, got, c.want)
		}
	}
}
