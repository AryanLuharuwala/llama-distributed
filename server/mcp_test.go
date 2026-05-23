package main

import (
	"database/sql"
	"errors"
	"fmt"
	"strings"
	"testing"

	_ "github.com/mattn/go-sqlite3"
)

// openMCPTestDB opens a private in-memory SQLite with mcp + rag migrations
// applied.  Each test gets its own uniquely-named DB so we can run them in
// parallel without cross-contamination.
func openMCPTestDB(t *testing.T) (*server, *sql.DB) {
	t.Helper()
	dsn := fmt.Sprintf("file:mcptest-%s?mode=memory&cache=shared", strings.ReplaceAll(t.Name(), "/", "_"))
	db, err := sql.Open("sqlite3", dsn)
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	d := sqliteDialect{}
	if err := migrateMCP(db, d); err != nil {
		t.Fatalf("migrateMCP: %v", err)
	}
	if err := migrateRAG(db, d); err != nil {
		t.Fatalf("migrateRAG: %v", err)
	}
	if err := migrateConvMemory(db, d); err != nil {
		t.Fatalf("migrateConvMemory: %v", err)
	}
	s := &server{db: db, dialect: d}
	return s, db
}

func TestValidateMCPServer(t *testing.T) {
	good := []mcpServer{
		{Name: "fs-local", Transport: mcpTransportStdio, Endpoint: "/usr/local/bin/mcp-fs --root /home/u"},
		{Name: "github", Transport: mcpTransportHTTP, Endpoint: "https://api.example.com/mcp"},
		{Name: "ws_one", Transport: mcpTransportWS, Endpoint: "wss://x.example.com/mcp"},
		{Name: "a", Transport: mcpTransportStdio, Endpoint: "/bin/ls", Scopes: []string{"read", "list"}},
	}
	for _, m := range good {
		if err := validateMCPServer(&m); err != nil {
			t.Errorf("validateMCPServer(%+v) = %v, want nil", m, err)
		}
	}

	bad := []struct {
		m    mcpServer
		want string
	}{
		{mcpServer{Name: "Bad-Caps", Transport: mcpTransportStdio, Endpoint: "/bin/ls"}, "invalid name"},
		{mcpServer{Name: "ok", Transport: "junk", Endpoint: "/bin/ls"}, "invalid transport"},
		{mcpServer{Name: "ok", Transport: mcpTransportStdio, Endpoint: ""}, "endpoint length"},
		{mcpServer{Name: "ok", Transport: mcpTransportStdio, Endpoint: "relative/bin"}, "absolute path"},
		{mcpServer{Name: "ok", Transport: mcpTransportHTTP, Endpoint: "ftp://nope"}, "must be http"},
		{mcpServer{Name: "ok", Transport: mcpTransportWS, Endpoint: "https://nope"}, "must be ws"},
	}
	for _, c := range bad {
		err := validateMCPServer(&c.m)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("validateMCPServer(%+v) err=%v, want substring %q", c.m, err, c.want)
		}
	}
}

func TestMCPCRUD(t *testing.T) {
	s, _ := openMCPTestDB(t)

	m := &mcpServer{
		UserID:    42,
		Name:      "fs-local",
		Transport: mcpTransportStdio,
		Endpoint:  "/usr/local/bin/mcp-fs",
		Scopes:    []string{"read", "list"},
		SecretRef: "env:MCP_FS_TOKEN",
		Enabled:   true,
	}
	id, err := s.addMCPServer(m)
	if err != nil {
		t.Fatalf("addMCPServer: %v", err)
	}
	if id == 0 {
		t.Fatal("expected non-zero ID")
	}

	got, err := s.getMCPServer(42, id)
	if err != nil {
		t.Fatalf("getMCPServer: %v", err)
	}
	if got.Name != "fs-local" || got.Transport != mcpTransportStdio || !got.Enabled {
		t.Errorf("getMCPServer returned %+v", got)
	}
	if len(got.Scopes) != 2 || got.Scopes[0] != "read" {
		t.Errorf("scopes round-trip failed: %v", got.Scopes)
	}

	// Other-user isolation: a different uid must not see the row.
	if _, err := s.getMCPServer(43, id); !errors.Is(err, sql.ErrNoRows) {
		t.Errorf("expected ErrNoRows for foreign uid, got %v", err)
	}

	// Update.
	got.Endpoint = "/usr/local/bin/mcp-fs --read-only"
	got.Enabled = false
	if err := s.updateMCPServer(42, got); err != nil {
		t.Fatalf("updateMCPServer: %v", err)
	}
	again, err := s.getMCPServer(42, id)
	if err != nil {
		t.Fatalf("getMCPServer after update: %v", err)
	}
	if again.Enabled || !strings.Contains(again.Endpoint, "--read-only") {
		t.Errorf("update didn't persist: %+v", again)
	}

	// Update on foreign uid must be a no-op rejected with ErrNoRows.
	if err := s.updateMCPServer(99, again); !errors.Is(err, sql.ErrNoRows) {
		t.Errorf("foreign-uid update should fail with ErrNoRows, got %v", err)
	}

	// Duplicate name on the same uid must fail.
	if _, err := s.addMCPServer(&mcpServer{
		UserID:    42,
		Name:      "fs-local",
		Transport: mcpTransportStdio,
		Endpoint:  "/bin/ls",
	}); err == nil {
		t.Error("expected duplicate name to fail")
	}

	// list returns only this user's row.
	rows, err := s.listMCPServers(42)
	if err != nil {
		t.Fatalf("listMCPServers: %v", err)
	}
	if len(rows) != 1 {
		t.Fatalf("expected 1 row, got %d", len(rows))
	}

	// Delete + verify.
	if err := s.deleteMCPServer(42, id); err != nil {
		t.Fatalf("deleteMCPServer: %v", err)
	}
	if _, err := s.getMCPServer(42, id); !errors.Is(err, sql.ErrNoRows) {
		t.Errorf("expected ErrNoRows after delete, got %v", err)
	}
}

func TestMCPCallAudit(t *testing.T) {
	s, _ := openMCPTestDB(t)
	rec := mcpCallRecord{
		UserID:     7,
		ServerID:   1,
		Tool:       "read_file",
		ArgsSHA256: hashMCPArgs([]byte(`{"path":"/etc/hostname"}`)),
		SizeBytes:  23,
		Success:    true,
		LatencyMS:  12,
		CalledAt:   1700000000,
	}
	if err := s.recordMCPCall(rec); err != nil {
		t.Fatalf("recordMCPCall: %v", err)
	}
	// Verify the hash is hex sha256 (64 chars).
	if len(rec.ArgsSHA256) != 64 {
		t.Errorf("expected 64-char hex sha256, got %d: %q", len(rec.ArgsSHA256), rec.ArgsSHA256)
	}

	var count int
	if err := s.db.QueryRow(`SELECT COUNT(*) FROM mcp_calls WHERE user_id = ?`, 7).Scan(&count); err != nil {
		t.Fatalf("count: %v", err)
	}
	if count != 1 {
		t.Errorf("expected 1 call row, got %d", count)
	}
}

func TestMCPHealthRoundtrip(t *testing.T) {
	s, _ := openMCPTestDB(t)
	id, err := s.addMCPServer(&mcpServer{
		UserID: 1, Name: "x", Transport: mcpTransportHTTP,
		Endpoint: "https://x.example.com/mcp",
	})
	if err != nil {
		t.Fatalf("addMCPServer: %v", err)
	}
	if err := s.markMCPHealth(id, true, 1700000000); err != nil {
		t.Fatalf("markMCPHealth: %v", err)
	}
	m, err := s.getMCPServer(1, id)
	if err != nil {
		t.Fatalf("getMCPServer: %v", err)
	}
	if !m.LastHealthOK || m.LastHealthAt != 1700000000 {
		t.Errorf("health didn't persist: %+v", m)
	}
}
