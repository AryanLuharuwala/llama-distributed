package main

// MCP broker: holds per-(user, server) connections to user-registered
// MCP servers and proxies JSON-RPC 2.0 calls.  The registry (mcp.go)
// is the persisted source of truth; the broker is the live runtime
// view that the chat / tool-call paths interact with.
//
// Transports:
//   • http  — JSON-RPC over POST.  Implemented here.
//   • ws    — JSON-RPC over WebSocket.  Skeleton; not active until
//             the upstream WS path lands on the same prod branch as
//             agent-side embed support.
//   • stdio — subprocess + JSON-RPC on stdin/stdout.  Skeleton.
//             Stdio requires sandboxing (cgroups + seccomp) to be
//             safe to expose multi-tenant, which is on the prod
//             roadmap, not the single-container scope.
//
// Connection model: each broker entry is reused across calls and held
// open for `mcpConnTTL` after the last use.  The janitor closes idle
// connections so a forgotten MCP server doesn't pin a file descriptor.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// osGetenv is a thin wrapper so tests can override env lookups.
var osGetenv = os.Getenv

const (
	mcpConnTTL     = 5 * time.Minute
	mcpCallTimeout = 30 * time.Second
)

// brokerConn is the runtime handle for one (user, server) MCP session.
type brokerConn struct {
	server  mcpServer
	t       mcpWireTransport
	lastUse atomic.Int64
	mu      sync.Mutex // serialises in-flight calls per connection
}

// mcpTransport abstracts the on-the-wire details so the rest of the
// broker doesn't care whether it's talking HTTP, WS, or stdio.
type mcpWireTransport interface {
	Call(ctx context.Context, method string, params any) (json.RawMessage, error)
	Close() error
}

// brokerRegistry holds live brokers keyed by (uid, server_id).  Calls
// land here; a janitor goroutine closes entries idle past mcpConnTTL.
type brokerRegistry struct {
	mu    sync.Mutex
	conns map[brokerKey]*brokerConn
}

type brokerKey struct {
	uid    int64
	server int64
}

func newBrokerRegistry() *brokerRegistry {
	r := &brokerRegistry{conns: make(map[brokerKey]*brokerConn)}
	go r.janitor()
	return r
}

// janitor closes connections whose last_use is older than mcpConnTTL.
func (r *brokerRegistry) janitor() {
	t := time.NewTicker(time.Minute)
	defer t.Stop()
	for range t.C {
		now := time.Now().Unix()
		r.mu.Lock()
		for k, c := range r.conns {
			if now-c.lastUse.Load() > int64(mcpConnTTL/time.Second) {
				_ = c.t.Close()
				delete(r.conns, k)
			}
		}
		r.mu.Unlock()
	}
}

// get returns (and lazily opens) the broker connection for (uid, server).
// The caller has already validated that server belongs to uid via the
// MCP registry — passing a stale server here is a bug.
func (s *server) brokerFor(ctx context.Context, srv *mcpServer) (*brokerConn, error) {
	if s.brokers == nil {
		// Lazy init — main() can also wire this explicitly, but tests
		// that hit the broker via a handler need the registry alive.
		s.brokers = newBrokerRegistry()
	}
	k := brokerKey{uid: srv.UserID, server: srv.ID}
	s.brokers.mu.Lock()
	if c, ok := s.brokers.conns[k]; ok {
		c.lastUse.Store(time.Now().Unix())
		s.brokers.mu.Unlock()
		return c, nil
	}
	s.brokers.mu.Unlock()

	var t mcpWireTransport
	var err error
	switch srv.Transport {
	case mcpTransportHTTP:
		t, err = newHTTPTransport(srv)
	case mcpTransportWS:
		err = errors.New("mcp: ws transport not yet implemented")
	case mcpTransportStdio:
		err = errors.New("mcp: stdio transport not yet implemented (requires sandbox)")
	default:
		err = fmt.Errorf("mcp: unknown transport %q", srv.Transport)
	}
	if err != nil {
		return nil, err
	}
	c := &brokerConn{server: *srv, t: t}
	c.lastUse.Store(time.Now().Unix())

	s.brokers.mu.Lock()
	// Double-check: another goroutine may have raced us.
	if existing, ok := s.brokers.conns[k]; ok {
		_ = t.Close()
		s.brokers.mu.Unlock()
		existing.lastUse.Store(time.Now().Unix())
		return existing, nil
	}
	s.brokers.conns[k] = c
	s.brokers.mu.Unlock()
	return c, nil
}

// CallMCP is the public entry point.  It looks up the MCP server row,
// gets/opens a broker, and invokes the JSON-RPC method.  Records the
// call in mcp_calls for audit (success or failure).
func (s *server) callMCP(ctx context.Context, uid int64, serverName, method string, params any) (json.RawMessage, error) {
	srv, err := s.getMCPServerByName(uid, serverName)
	if err != nil {
		return nil, err
	}
	if !srv.Enabled {
		return nil, fmt.Errorf("mcp: server %q is disabled", serverName)
	}
	c, err := s.brokerFor(ctx, srv)
	if err != nil {
		_ = s.markMCPHealth(srv.ID, false, time.Now().Unix())
		return nil, err
	}
	subCtx, cancel := context.WithTimeout(ctx, mcpCallTimeout)
	defer cancel()

	start := time.Now()
	c.mu.Lock()
	res, err := c.t.Call(subCtx, method, params)
	c.lastUse.Store(time.Now().Unix())
	c.mu.Unlock()
	latency := time.Since(start).Milliseconds()

	paramsBytes, _ := json.Marshal(params)
	argsHash := hashMCPArgs(paramsBytes)
	rec := mcpCallRecord{
		UserID:     uid,
		ServerID:   srv.ID,
		Tool:       method,
		ArgsSHA256: argsHash,
		SizeBytes:  int64(len(paramsBytes)),
		Success:    err == nil,
		LatencyMS:  latency,
		CalledAt:   time.Now().Unix(),
	}
	if err != nil {
		rec.ErrorClass = classifyMCPError(err)
		_ = s.recordMCPCall(rec)
		_ = s.markMCPHealth(srv.ID, false, time.Now().Unix())
		return nil, err
	}
	_ = s.recordMCPCall(rec)
	_ = s.markMCPHealth(srv.ID, true, time.Now().Unix())
	return res, nil
}

// classifyMCPError buckets the failure mode for the audit log so
// dashboards can chart "transport flakes" separately from "model
// denied us" errors without grepping free-form strings.
func classifyMCPError(err error) string {
	if err == nil {
		return ""
	}
	s := err.Error()
	switch {
	case strings.Contains(s, "context deadline") || strings.Contains(s, "timeout"):
		return "timeout"
	case strings.Contains(s, "rpc err"):
		return "rpc"
	case strings.Contains(s, "HTTP 4") || strings.Contains(s, "denied"):
		return "denied"
	case strings.Contains(s, "transport") || strings.Contains(s, "not yet implemented"):
		return "transport"
	default:
		return "other"
	}
}

// ─── HTTP transport ─────────────────────────────────────────────────

type httpTransport struct {
	endpoint string
	client   *http.Client
	auth     string // pre-resolved "env:VAR" → value, or literal
}

func newHTTPTransport(srv *mcpServer) (*httpTransport, error) {
	auth, err := resolveSecretRef(srv.SecretRef)
	if err != nil {
		return nil, err
	}
	return &httpTransport{
		endpoint: srv.Endpoint,
		client: &http.Client{
			Timeout: mcpCallTimeout,
		},
		auth: auth,
	}, nil
}

// Call posts a JSON-RPC 2.0 envelope.  We don't reuse a single id
// counter across goroutines because each Call is synchronous on the
// wire — id can be a fresh random/sequential value per request.
func (h *httpTransport) Call(ctx context.Context, method string, params any) (json.RawMessage, error) {
	req := struct {
		Jsonrpc string `json:"jsonrpc"`
		ID      int    `json:"id"`
		Method  string `json:"method"`
		Params  any    `json:"params,omitempty"`
	}{
		Jsonrpc: "2.0",
		ID:      1,
		Method:  method,
		Params:  params,
	}
	buf, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("mcp/http: marshal: %w", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, "POST", h.endpoint, bytes.NewReader(buf))
	if err != nil {
		return nil, fmt.Errorf("mcp/http: new req: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json")
	if h.auth != "" {
		httpReq.Header.Set("Authorization", "Bearer "+h.auth)
	}
	resp, err := h.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("mcp/http: do: %w", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil {
		return nil, fmt.Errorf("mcp/http: read: %w", err)
	}
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("mcp/http: HTTP %d: %s", resp.StatusCode, truncateMCP(string(body), 256))
	}
	var reply struct {
		Jsonrpc string          `json:"jsonrpc"`
		ID      json.RawMessage `json:"id"`
		Result  json.RawMessage `json:"result"`
		Error   *struct {
			Code    int             `json:"code"`
			Message string          `json:"message"`
			Data    json.RawMessage `json:"data"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &reply); err != nil {
		return nil, fmt.Errorf("mcp/http: parse: %w (body=%s)", err, truncateMCP(string(body), 256))
	}
	if reply.Error != nil {
		return nil, fmt.Errorf("mcp/http: rpc err %d: %s", reply.Error.Code, reply.Error.Message)
	}
	return reply.Result, nil
}

func (h *httpTransport) Close() error { return nil }

// resolveSecretRef expands "env:VAR" into the env var's value.  Any
// other prefix is treated as a literal — useful in tests, dangerous
// in production so we keep the literal path discoverable and audited
// (the registry validation refuses "literal:" without explicit opt-in).
func resolveSecretRef(ref string) (string, error) {
	ref = strings.TrimSpace(ref)
	if ref == "" {
		return "", nil
	}
	if strings.HasPrefix(ref, "env:") {
		key := strings.TrimPrefix(ref, "env:")
		if key == "" {
			return "", errors.New("mcp: secret_ref env: with empty key")
		}
		// Lookup deferred to the caller's environment.
		return getEnvOrEmpty(key), nil
	}
	if strings.HasPrefix(ref, "literal:") {
		return strings.TrimPrefix(ref, "literal:"), nil
	}
	return "", fmt.Errorf("mcp: secret_ref must be env:<VAR> or literal:<value>, got %q", ref)
}

// getEnvOrEmpty wraps os.Getenv so it's mockable in tests.
var getEnvOrEmpty = func(key string) string {
	return osGetenv(key)
}

func truncateMCP(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
