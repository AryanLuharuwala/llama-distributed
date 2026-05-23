package main

// HTTP surface for the MCP registry + broker.
//
// Registry (CRUD over user-owned mcp_servers rows):
//   GET    /api/mcp/servers                 → list
//   POST   /api/mcp/servers                 → create
//   GET    /api/mcp/servers/{id}            → fetch
//   PUT    /api/mcp/servers/{id}            → update mutable fields
//   DELETE /api/mcp/servers/{id}            → delete (audit log retained)
//
// Broker (proxied JSON-RPC tool call):
//   POST   /api/mcp/{server}/call           → invoke tool on a server (by name)
//     body: { "method": "tools/call", "params": { ... } }
//     →     { "result": <json> }            on success
//     →     { "error":  "<msg>"  }          on RPC failure (HTTP 200)
//     →     HTTP 4xx/5xx                    on transport / auth failure
//
// Why 200 + error body for RPC failures?  Tool calls regularly "fail" in
// expected ways (the model asked for a nonexistent file, the upstream
// service returned a 4xx).  We want the dashboard's tool-builder to see
// those as structured errors, not exception-path 5xx that trigger retries
// at the HTTP layer.  Transport failures (broker can't reach the server)
// stay as 5xx so monitoring catches them.

import (
	"database/sql"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strconv"
	"strings"
)

const mcpMaxCallBodyBytes = 256 << 10 // 256 KiB — generous for tool args

// ── Registry CRUD ───────────────────────────────────────────────────────

// GET /api/mcp/servers
func (s *server) handleMCPListServers(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	rows, err := s.listMCPServers(u.ID)
	if err != nil {
		writeErr(w, 500, "list mcp servers")
		return
	}
	out := make([]map[string]any, 0, len(rows))
	for _, m := range rows {
		out = append(out, mcpServerToMap(&m))
	}
	writeJSON(w, 200, map[string]any{"servers": out})
}

// POST /api/mcp/servers
func (s *server) handleMCPCreateServer(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	var body mcpServerInput
	if err := json.NewDecoder(io.LimitReader(r.Body, 32<<10)).Decode(&body); err != nil {
		writeErr(w, 400, "bad json")
		return
	}
	m := body.toModel(u.ID)
	id, err := s.addMCPServer(m)
	if err != nil {
		if isUniqueConstraint(err) {
			writeErr(w, 409, "mcp server name already exists")
			return
		}
		writeErr(w, 400, err.Error())
		return
	}
	m.ID = id
	writeJSON(w, 201, mcpServerToMap(m))
}

// GET /api/mcp/servers/{id}
func (s *server) handleMCPGetServer(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	id, ok := parseIDPath(r, "id")
	if !ok {
		writeErr(w, 400, "bad id")
		return
	}
	m, err := s.getMCPServer(u.ID, id)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			writeErr(w, 404, "not found")
			return
		}
		writeErr(w, 500, "fetch mcp server")
		return
	}
	writeJSON(w, 200, mcpServerToMap(m))
}

// PUT /api/mcp/servers/{id}
func (s *server) handleMCPUpdateServer(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	id, ok := parseIDPath(r, "id")
	if !ok {
		writeErr(w, 400, "bad id")
		return
	}
	// Load existing so we preserve immutable fields (name) and only
	// touch what the caller supplied.
	existing, err := s.getMCPServer(u.ID, id)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			writeErr(w, 404, "not found")
			return
		}
		writeErr(w, 500, "fetch mcp server")
		return
	}
	var body mcpServerInput
	if err := json.NewDecoder(io.LimitReader(r.Body, 32<<10)).Decode(&body); err != nil {
		writeErr(w, 400, "bad json")
		return
	}
	// Name is immutable on update.  Apply only mutable fields.
	if body.Transport != "" {
		existing.Transport = mcpTransport(body.Transport)
	}
	if body.Endpoint != "" {
		existing.Endpoint = body.Endpoint
	}
	if body.Scopes != nil {
		existing.Scopes = body.Scopes
	}
	if body.SecretRef != "" {
		existing.SecretRef = body.SecretRef
	}
	existing.Enabled = body.Enabled
	if err := s.updateMCPServer(u.ID, existing); err != nil {
		writeErr(w, 400, err.Error())
		return
	}
	writeJSON(w, 200, mcpServerToMap(existing))
}

// DELETE /api/mcp/servers/{id}
func (s *server) handleMCPDeleteServer(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	id, ok := parseIDPath(r, "id")
	if !ok {
		writeErr(w, 400, "bad id")
		return
	}
	if err := s.deleteMCPServer(u.ID, id); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			writeErr(w, 404, "not found")
			return
		}
		writeErr(w, 500, "delete mcp server")
		return
	}
	writeJSON(w, 200, map[string]any{"deleted": id})
}

// ── Broker call ─────────────────────────────────────────────────────────

// POST /api/mcp/{server}/call
func (s *server) handleMCPCall(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	name := r.PathValue("server")
	if name == "" {
		writeErr(w, 400, "server name required")
		return
	}
	var body struct {
		Method string          `json:"method"`
		Params json.RawMessage `json:"params"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, mcpMaxCallBodyBytes)).Decode(&body); err != nil {
		writeErr(w, 400, "bad json")
		return
	}
	if strings.TrimSpace(body.Method) == "" {
		writeErr(w, 400, "method required")
		return
	}
	res, err := s.callMCP(r.Context(), u.ID, name, body.Method, body.Params)
	if err != nil {
		// 200 with error string — clients should distinguish HTTP-level
		// failures (5xx) from per-call RPC failures (200 + error body).
		writeJSON(w, 200, map[string]any{"error": err.Error()})
		return
	}
	writeJSON(w, 200, map[string]any{"result": json.RawMessage(res)})
}

// ── helpers ─────────────────────────────────────────────────────────────

type mcpServerInput struct {
	Name      string   `json:"name"`
	Transport string   `json:"transport"`
	Endpoint  string   `json:"endpoint"`
	Scopes    []string `json:"scopes"`
	SecretRef string   `json:"secret_ref"`
	Enabled   bool     `json:"enabled"`
}

func (in *mcpServerInput) toModel(uid int64) *mcpServer {
	return &mcpServer{
		UserID:    uid,
		Name:      strings.TrimSpace(in.Name),
		Transport: mcpTransport(in.Transport),
		Endpoint:  strings.TrimSpace(in.Endpoint),
		Scopes:    in.Scopes,
		SecretRef: strings.TrimSpace(in.SecretRef),
		Enabled:   in.Enabled,
	}
}

func mcpServerToMap(m *mcpServer) map[string]any {
	return map[string]any{
		"id":             m.ID,
		"name":           m.Name,
		"transport":      string(m.Transport),
		"endpoint":       m.Endpoint,
		"scopes":         m.Scopes,
		"secret_ref":     m.SecretRef,
		"enabled":        m.Enabled,
		"last_health_at": m.LastHealthAt,
		"last_health_ok": m.LastHealthOK,
		"created_at":     m.CreatedAt,
		"updated_at":     m.UpdatedAt,
	}
}

func parseIDPath(r *http.Request, name string) (int64, bool) {
	v := r.PathValue(name)
	if v == "" {
		return 0, false
	}
	n, err := strconv.ParseInt(v, 10, 64)
	if err != nil || n <= 0 {
		return 0, false
	}
	return n, true
}
