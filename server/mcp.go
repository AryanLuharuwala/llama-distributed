package main

// MCP (Model Context Protocol) server registry.
//
// Users register external tool/resource servers (filesystem, github, web,
// custom) that the inference path can call when a model emits a tool_use
// frame.  This file owns the control-plane state: schema, CRUD, and the
// validation rules that govern what a user can register.
//
// Wire layout
// ───────────
// mcp_servers  — one row per (user_id, name).  Holds transport + endpoint
//                + scopes + enabled flag + opaque secret reference.
// mcp_calls    — append-only audit log of every tool invocation.  Carries
//                sha256(args) only (no plaintext args — those can contain
//                PII or auth tokens for downstream services).
//
// The broker in mcp_broker.go reads this table, holds a per-(user,server)
// connection, and proxies calls from the inference path through it.  The
// broker enforces per-server scopes, rate-limit, and payload caps; this
// file is just the persistent registry.
//
// Why two tables instead of folding into rigs/users?  MCP servers are a
// per-user resource that can outlive any single inference job, and they
// have their own lifecycle (enable/disable/rotate-secret).  Keeping them
// in their own namespace also lets us add admin endpoints later (org-wide
// shared servers, marketplace integrations) without schema churn.

import (
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"regexp"
	"strings"
	"time"
)

// mcpTransport enumerates the wire transports the broker knows how to
// speak.  stdio runs a subprocess sandboxed under bwrap/landlock; the
// http+sse and ws variants reach out to a network endpoint.
type mcpTransport string

const (
	mcpTransportStdio mcpTransport = "stdio"
	mcpTransportHTTP  mcpTransport = "http+sse"
	mcpTransportWS    mcpTransport = "ws"
)

func (t mcpTransport) valid() bool {
	switch t {
	case mcpTransportStdio, mcpTransportHTTP, mcpTransportWS:
		return true
	}
	return false
}

// mcpServer is the persisted shape of a registered MCP server.
type mcpServer struct {
	ID          int64
	UserID      int64
	Name        string       // user-chosen, unique per user
	Transport   mcpTransport
	Endpoint    string       // command line for stdio; URL for http+sse/ws
	Scopes      []string     // allow-list of tool names; empty == all
	SecretRef   string       // reference into the secret store (env var name, vault path); never the raw secret
	Enabled     bool
	LastHealthAt int64       // unix seconds; 0 if never probed
	LastHealthOK bool
	CreatedAt   int64
	UpdatedAt   int64
}

// mcpCallRecord is one entry in the audit log.
type mcpCallRecord struct {
	ID         int64
	UserID     int64
	ServerID   int64
	Tool       string
	ArgsSHA256 string // hex
	SizeBytes  int64
	Success    bool
	ErrorClass string // "timeout", "denied", "transport", "" on success
	LatencyMS  int64
	CalledAt   int64
}

// migrateMCP installs the two MCP tables.  Idempotent across SQLite +
// Postgres via the dialect helper.
func migrateMCP(db *sql.DB, d sqlDialect) error {
	if _, err := db.Exec(d.RewriteDDL(`
		CREATE TABLE IF NOT EXISTS mcp_servers (
			id              INTEGER PRIMARY KEY AUTOINCREMENT,
			user_id         INTEGER NOT NULL,
			name            TEXT    NOT NULL,
			transport       TEXT    NOT NULL,
			endpoint        TEXT    NOT NULL,
			scopes_json     TEXT    NOT NULL DEFAULT '[]',
			secret_ref      TEXT    NOT NULL DEFAULT '',
			enabled         INTEGER NOT NULL DEFAULT 1,
			last_health_at  INTEGER NOT NULL DEFAULT 0,
			last_health_ok  INTEGER NOT NULL DEFAULT 0,
			created_at      INTEGER NOT NULL,
			updated_at      INTEGER NOT NULL,
			UNIQUE(user_id, name)
		)
	`)); err != nil {
		return fmt.Errorf("create mcp_servers: %w", err)
	}
	if _, err := db.Exec(d.RewriteDDL(`
		CREATE TABLE IF NOT EXISTS mcp_calls (
			id            INTEGER PRIMARY KEY AUTOINCREMENT,
			user_id       INTEGER NOT NULL,
			server_id     INTEGER NOT NULL,
			tool          TEXT    NOT NULL,
			args_sha256   TEXT    NOT NULL,
			size_bytes    INTEGER NOT NULL DEFAULT 0,
			success       INTEGER NOT NULL DEFAULT 0,
			error_class   TEXT    NOT NULL DEFAULT '',
			latency_ms    INTEGER NOT NULL DEFAULT 0,
			called_at     INTEGER NOT NULL
		)
	`)); err != nil {
		return fmt.Errorf("create mcp_calls: %w", err)
	}
	// Index supports the common "show me my call history for this server"
	// query and the per-user rate-limit scan.
	if _, err := db.Exec(d.RewriteDDL(`
		CREATE INDEX IF NOT EXISTS idx_mcp_calls_user_called
			ON mcp_calls (user_id, called_at)
	`)); err != nil {
		return fmt.Errorf("create idx_mcp_calls_user_called: %w", err)
	}
	return nil
}

// ── Validation ────────────────────────────────────────────────────────

// mcpNameRE constrains user-chosen names: lowercase, digits, dashes,
// underscores.  Keeps the name safe to surface in CLI prompts and
// telemetry without escaping.
var mcpNameRE = regexp.MustCompile(`^[a-z0-9][a-z0-9_-]{0,62}$`)

const (
	mcpMaxScopes        = 64
	mcpMaxScopeLen      = 64
	mcpMaxEndpointLen   = 2048
	mcpMaxSecretRefLen  = 128
)

// validateMCPServer applies the registration-time rules.  Run on add and
// on update so we don't accept malformed updates that bypass create-time
// checks.
func validateMCPServer(m *mcpServer) error {
	if m == nil {
		return errors.New("mcpServer is nil")
	}
	if !mcpNameRE.MatchString(m.Name) {
		return errors.New("invalid name: must match [a-z0-9][a-z0-9_-]{0,62}")
	}
	if !m.Transport.valid() {
		return fmt.Errorf("invalid transport %q", m.Transport)
	}
	if len(m.Endpoint) == 0 || len(m.Endpoint) > mcpMaxEndpointLen {
		return fmt.Errorf("endpoint length must be 1..%d", mcpMaxEndpointLen)
	}
	if len(m.SecretRef) > mcpMaxSecretRefLen {
		return fmt.Errorf("secret_ref too long (max %d)", mcpMaxSecretRefLen)
	}
	if len(m.Scopes) > mcpMaxScopes {
		return fmt.Errorf("too many scopes (max %d)", mcpMaxScopes)
	}
	for _, s := range m.Scopes {
		if len(s) == 0 || len(s) > mcpMaxScopeLen {
			return fmt.Errorf("scope length must be 1..%d", mcpMaxScopeLen)
		}
	}
	switch m.Transport {
	case mcpTransportStdio:
		// stdio endpoints are command lines.  Require an absolute path
		// for the program so we don't pick up arbitrary $PATH binaries.
		f := strings.Fields(m.Endpoint)
		if len(f) == 0 || !strings.HasPrefix(f[0], "/") {
			return errors.New("stdio endpoint must start with an absolute path")
		}
	case mcpTransportHTTP, mcpTransportWS:
		u, err := url.Parse(m.Endpoint)
		if err != nil {
			return fmt.Errorf("endpoint url: %w", err)
		}
		switch m.Transport {
		case mcpTransportHTTP:
			if u.Scheme != "https" && u.Scheme != "http" {
				return errors.New("http+sse endpoint must be http:// or https://")
			}
		case mcpTransportWS:
			if u.Scheme != "wss" && u.Scheme != "ws" {
				return errors.New("ws endpoint must be ws:// or wss://")
			}
		}
		// Host-side validation (reject loopback, link-local, RFC1918 from
		// untrusted users) is enforced by isAllowedPublicIP in the broker
		// before dialling — kept there so a single allow-list governs all
		// outbound connections, not just MCP.
	}
	return nil
}

// ── CRUD ──────────────────────────────────────────────────────────────

// addMCPServer inserts a new server row.  Returns the assigned ID.
func (s *server) addMCPServer(m *mcpServer) (int64, error) {
	if err := validateMCPServer(m); err != nil {
		return 0, err
	}
	scopes, err := json.Marshal(m.Scopes)
	if err != nil {
		return 0, fmt.Errorf("marshal scopes: %w", err)
	}
	now := time.Now().Unix()
	m.CreatedAt = now
	m.UpdatedAt = now
	q := s.dialect.RewriteQuery(`
		INSERT INTO mcp_servers
			(user_id, name, transport, endpoint, scopes_json, secret_ref,
			 enabled, last_health_at, last_health_ok, created_at, updated_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?)
	`)
	res, err := s.dbExec(q,
		m.UserID, m.Name, string(m.Transport), m.Endpoint,
		string(scopes), m.SecretRef, boolToInt(m.Enabled),
		now, now)
	if err != nil {
		return 0, err
	}
	id, err := res.LastInsertId()
	if err != nil {
		return 0, err
	}
	m.ID = id
	return id, nil
}

// listMCPServers returns all servers owned by uid, ordered by name.
func (s *server) listMCPServers(uid int64) ([]mcpServer, error) {
	q := s.dialect.RewriteQuery(`
		SELECT id, user_id, name, transport, endpoint, scopes_json,
		       secret_ref, enabled, last_health_at, last_health_ok,
		       created_at, updated_at
		FROM mcp_servers
		WHERE user_id = ?
		ORDER BY name ASC
	`)
	rows, err := s.dbQuery(q, uid)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []mcpServer
	for rows.Next() {
		var m mcpServer
		var scopes string
		var transport string
		var enabled, healthOK int
		if err := rows.Scan(&m.ID, &m.UserID, &m.Name, &transport, &m.Endpoint,
			&scopes, &m.SecretRef, &enabled, &m.LastHealthAt, &healthOK,
			&m.CreatedAt, &m.UpdatedAt); err != nil {
			return nil, err
		}
		m.Transport = mcpTransport(transport)
		m.Enabled = enabled != 0
		m.LastHealthOK = healthOK != 0
		if scopes != "" {
			_ = json.Unmarshal([]byte(scopes), &m.Scopes)
		}
		out = append(out, m)
	}
	return out, rows.Err()
}

// getMCPServer fetches a single row owned by uid.  Returns sql.ErrNoRows
// if the server doesn't exist OR isn't owned by uid — same shape so we
// don't leak existence of other users' servers.
func (s *server) getMCPServer(uid, id int64) (*mcpServer, error) {
	q := s.dialect.RewriteQuery(`
		SELECT id, user_id, name, transport, endpoint, scopes_json,
		       secret_ref, enabled, last_health_at, last_health_ok,
		       created_at, updated_at
		FROM mcp_servers
		WHERE user_id = ? AND id = ?
	`)
	row := s.dbQueryRow(q, uid, id)
	var m mcpServer
	var scopes, transport string
	var enabled, healthOK int
	if err := row.Scan(&m.ID, &m.UserID, &m.Name, &transport, &m.Endpoint,
		&scopes, &m.SecretRef, &enabled, &m.LastHealthAt, &healthOK,
		&m.CreatedAt, &m.UpdatedAt); err != nil {
		return nil, err
	}
	m.Transport = mcpTransport(transport)
	m.Enabled = enabled != 0
	m.LastHealthOK = healthOK != 0
	if scopes != "" {
		_ = json.Unmarshal([]byte(scopes), &m.Scopes)
	}
	return &m, nil
}

// getMCPServerByName looks up an MCP server row by its (user, name)
// composite key.  Used by callers that reference servers by name in
// API payloads (e.g. the chat-completion tool-call path) rather than
// by numeric ID.
func (s *server) getMCPServerByName(uid int64, name string) (*mcpServer, error) {
	q := s.dialect.RewriteQuery(`
		SELECT id, user_id, name, transport, endpoint, scopes_json,
		       secret_ref, enabled, last_health_at, last_health_ok,
		       created_at, updated_at
		FROM mcp_servers
		WHERE user_id = ? AND name = ?
	`)
	row := s.dbQueryRow(q, uid, name)
	var m mcpServer
	var scopes, transport string
	var enabled, healthOK int
	if err := row.Scan(&m.ID, &m.UserID, &m.Name, &transport, &m.Endpoint,
		&scopes, &m.SecretRef, &enabled, &m.LastHealthAt, &healthOK,
		&m.CreatedAt, &m.UpdatedAt); err != nil {
		return nil, err
	}
	m.Transport = mcpTransport(transport)
	m.Enabled = enabled != 0
	m.LastHealthOK = healthOK != 0
	if scopes != "" {
		_ = json.Unmarshal([]byte(scopes), &m.Scopes)
	}
	return &m, nil
}

// updateMCPServer replaces the mutable fields (transport, endpoint,
// scopes, secret_ref, enabled).  Name is immutable — callers delete +
// re-add to rename, which keeps the audit log keyed cleanly.
func (s *server) updateMCPServer(uid int64, m *mcpServer) error {
	if err := validateMCPServer(m); err != nil {
		return err
	}
	scopes, err := json.Marshal(m.Scopes)
	if err != nil {
		return fmt.Errorf("marshal scopes: %w", err)
	}
	m.UpdatedAt = time.Now().Unix()
	q := s.dialect.RewriteQuery(`
		UPDATE mcp_servers
		SET transport = ?, endpoint = ?, scopes_json = ?, secret_ref = ?,
		    enabled = ?, updated_at = ?
		WHERE user_id = ? AND id = ?
	`)
	res, err := s.dbExec(q,
		string(m.Transport), m.Endpoint, string(scopes), m.SecretRef,
		boolToInt(m.Enabled), m.UpdatedAt, uid, m.ID)
	if err != nil {
		return err
	}
	n, err := res.RowsAffected()
	if err != nil {
		return err
	}
	if n == 0 {
		return sql.ErrNoRows
	}
	return nil
}

// deleteMCPServer removes the row.  Audit-log entries in mcp_calls are
// retained — they reference server_id by value, and we want history to
// survive deletes for billing/forensics.
func (s *server) deleteMCPServer(uid, id int64) error {
	q := s.dialect.RewriteQuery(`DELETE FROM mcp_servers WHERE user_id = ? AND id = ?`)
	res, err := s.dbExec(q, uid, id)
	if err != nil {
		return err
	}
	n, err := res.RowsAffected()
	if err != nil {
		return err
	}
	if n == 0 {
		return sql.ErrNoRows
	}
	return nil
}

// markMCPHealth records the most recent health-probe outcome.  Probes
// run from the broker on enable, on first use after a cold start, and
// on a slow background timer; this is the persisted result.
func (s *server) markMCPHealth(id int64, ok bool, at int64) error {
	q := s.dialect.RewriteQuery(`
		UPDATE mcp_servers
		SET last_health_at = ?, last_health_ok = ?, updated_at = ?
		WHERE id = ?
	`)
	_, err := s.dbExec(q, at, boolToInt(ok), at, id)
	return err
}

// recordMCPCall appends one row to mcp_calls.  Best-effort — a logging
// failure must never break an in-flight inference request, so callers
// log the error and move on.
func (s *server) recordMCPCall(rec mcpCallRecord) error {
	q := s.dialect.RewriteQuery(`
		INSERT INTO mcp_calls
			(user_id, server_id, tool, args_sha256, size_bytes,
			 success, error_class, latency_ms, called_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
	`)
	_, err := s.dbExec(q,
		rec.UserID, rec.ServerID, rec.Tool, rec.ArgsSHA256, rec.SizeBytes,
		boolToInt(rec.Success), rec.ErrorClass, rec.LatencyMS, rec.CalledAt)
	return err
}

// hashMCPArgs returns the audit-log hash of a tool-call payload.  We log
// only the hash to avoid persisting PII or secrets the user piped into
// the model (paths, query strings, oauth tokens for downstream APIs).
func hashMCPArgs(args []byte) string {
	sum := sha256.Sum256(args)
	return hex.EncodeToString(sum[:])
}

// boolToInt is the local sqlite3-flavoured bool encoder; both drivers
// accept 0/1 for the INTEGER column we declared, so this stays a tiny
// helper rather than dragging in a dialect method.
func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}
