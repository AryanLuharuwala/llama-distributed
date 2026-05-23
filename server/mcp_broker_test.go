package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// TestHTTPTransportRoundtrip stands up an httptest server that speaks
// JSON-RPC 2.0 and asserts the broker's httpTransport marshals/unmarshals
// the envelope, propagates the Authorization header, and routes the
// reply.result back to the caller.
func TestHTTPTransportRoundtrip(t *testing.T) {
	var seen struct {
		method string
		params json.RawMessage
		auth   string
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var env struct {
			Jsonrpc string          `json:"jsonrpc"`
			ID      int             `json:"id"`
			Method  string          `json:"method"`
			Params  json.RawMessage `json:"params"`
		}
		if err := json.Unmarshal(body, &env); err != nil {
			http.Error(w, "bad json", 400)
			return
		}
		seen.method = env.Method
		seen.params = env.Params
		seen.auth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"jsonrpc":"2.0","id":%d,"result":{"echo":%s}}`, env.ID, string(env.Params))
	}))
	defer srv.Close()

	tr := &httpTransport{
		endpoint: srv.URL,
		client:   srv.Client(),
		auth:     "secret-token",
	}
	res, err := tr.Call(context.Background(), "tools/call", map[string]any{"name": "hello"})
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if seen.method != "tools/call" {
		t.Errorf("method = %q, want tools/call", seen.method)
	}
	if seen.auth != "Bearer secret-token" {
		t.Errorf("auth = %q, want Bearer secret-token", seen.auth)
	}
	if !strings.Contains(string(res), `"hello"`) {
		t.Errorf("result didn't echo: %s", res)
	}
}

func TestHTTPTransportRPCError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"jsonrpc":"2.0","id":1,"error":{"code":-32601,"message":"method not found"}}`)
	}))
	defer srv.Close()
	tr := &httpTransport{endpoint: srv.URL, client: srv.Client()}
	_, err := tr.Call(context.Background(), "tools/missing", nil)
	if err == nil || !strings.Contains(err.Error(), "rpc err -32601") {
		t.Errorf("expected rpc err, got %v", err)
	}
}

func TestHTTPTransportHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "forbidden", 403)
	}))
	defer srv.Close()
	tr := &httpTransport{endpoint: srv.URL, client: srv.Client()}
	_, err := tr.Call(context.Background(), "x", nil)
	if err == nil || !strings.Contains(err.Error(), "HTTP 403") {
		t.Errorf("expected HTTP 403, got %v", err)
	}
}

func TestClassifyMCPError(t *testing.T) {
	cases := []struct {
		in   string
		want string
	}{
		{"context deadline exceeded", "timeout"},
		{"i/o timeout", "timeout"},
		{"mcp/http: rpc err -32601: ...", "rpc"},
		{"mcp/http: HTTP 401: bad creds", "denied"},
		{"mcp: ws transport not yet implemented", "transport"},
		{"some other thing", "other"},
	}
	for _, c := range cases {
		got := classifyMCPError(fmt.Errorf("%s", c.in))
		if got != c.want {
			t.Errorf("classifyMCPError(%q) = %q, want %q", c.in, got, c.want)
		}
	}
	if got := classifyMCPError(nil); got != "" {
		t.Errorf("nil error → %q, want \"\"", got)
	}
}

func TestResolveSecretRef(t *testing.T) {
	prev := osGetenv
	defer func() { osGetenv = prev }()
	osGetenv = func(k string) string {
		if k == "MCP_TOK" {
			return "from-env"
		}
		return ""
	}

	cases := []struct {
		in      string
		want    string
		wantErr bool
	}{
		{"", "", false},
		{"env:MCP_TOK", "from-env", false},
		{"env:UNSET", "", false},
		{"env:", "", true},
		{"literal:abc123", "abc123", false},
		{"raw-string", "", true},
	}
	for _, c := range cases {
		got, err := resolveSecretRef(c.in)
		if (err != nil) != c.wantErr {
			t.Errorf("resolveSecretRef(%q) err=%v wantErr=%v", c.in, err, c.wantErr)
			continue
		}
		if got != c.want {
			t.Errorf("resolveSecretRef(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestBrokerRegistryReuseAndJanitor(t *testing.T) {
	r := &brokerRegistry{conns: make(map[brokerKey]*brokerConn)}
	k := brokerKey{uid: 1, server: 2}
	tr := &fakeWireTransport{}
	r.conns[k] = &brokerConn{t: tr}
	r.conns[k].lastUse.Store(time.Now().Add(-10 * time.Minute).Unix())

	// Inline the janitor sweep one tick.
	now := time.Now().Unix()
	r.mu.Lock()
	for k, c := range r.conns {
		if now-c.lastUse.Load() > int64(mcpConnTTL/time.Second) {
			_ = c.t.Close()
			delete(r.conns, k)
		}
	}
	r.mu.Unlock()
	if _, ok := r.conns[k]; ok {
		t.Errorf("expected idle entry to be evicted")
	}
	if !tr.closed.Load() {
		t.Errorf("expected Close() to be invoked on eviction")
	}
}

func TestCallMCPRecordsAuditOnSuccessAndFailure(t *testing.T) {
	s, _ := openMCPTestDB(t)
	// Stand up an MCP server that returns either result or error based on method.
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var env struct {
			ID     int    `json:"id"`
			Method string `json:"method"`
		}
		_ = json.Unmarshal(body, &env)
		if env.Method == "ok" {
			fmt.Fprintf(w, `{"jsonrpc":"2.0","id":%d,"result":{"ok":true}}`, env.ID)
			return
		}
		fmt.Fprintf(w, `{"jsonrpc":"2.0","id":%d,"error":{"code":-1,"message":"bad"}}`, env.ID)
	}))
	defer upstream.Close()

	id, err := s.addMCPServer(&mcpServer{
		UserID: 99, Name: "fixture", Transport: mcpTransportHTTP,
		Endpoint: upstream.URL, Enabled: true,
	})
	if err != nil {
		t.Fatalf("addMCPServer: %v", err)
	}
	_ = id

	_, err = s.callMCP(context.Background(), 99, "fixture", "ok", map[string]any{"x": 1})
	if err != nil {
		t.Fatalf("callMCP(ok): %v", err)
	}
	_, err = s.callMCP(context.Background(), 99, "fixture", "fail", nil)
	if err == nil {
		t.Fatalf("callMCP(fail): expected rpc err")
	}

	var ok, fail int
	if err := s.db.QueryRow(`SELECT COUNT(*) FROM mcp_calls WHERE success = 1`).Scan(&ok); err != nil {
		t.Fatalf("count ok: %v", err)
	}
	if err := s.db.QueryRow(`SELECT COUNT(*) FROM mcp_calls WHERE success = 0`).Scan(&fail); err != nil {
		t.Fatalf("count fail: %v", err)
	}
	if ok != 1 || fail != 1 {
		t.Errorf("ok=%d fail=%d, want 1/1", ok, fail)
	}

	// Disabled server should refuse.
	_ = s.updateMCPServer(99, &mcpServer{
		ID: id, UserID: 99, Name: "fixture", Transport: mcpTransportHTTP,
		Endpoint: upstream.URL, Enabled: false,
	})
	_, err = s.callMCP(context.Background(), 99, "fixture", "ok", nil)
	if err == nil || !strings.Contains(err.Error(), "disabled") {
		t.Errorf("expected disabled error, got %v", err)
	}
}

func TestCallMCPDeniesUnknownServer(t *testing.T) {
	s, _ := openMCPTestDB(t)
	_, err := s.callMCP(context.Background(), 1, "nonexistent", "x", nil)
	if err == nil {
		t.Errorf("expected error for unknown server")
	}
}

// fakeWireTransport satisfies mcpWireTransport for tests that need a
// stub without standing up a real HTTP server.
type fakeWireTransport struct {
	calls   atomic.Int64
	closed  atomic.Bool
	reply   json.RawMessage
	err     error
	mu      sync.Mutex
}

func (f *fakeWireTransport) Call(ctx context.Context, method string, params any) (json.RawMessage, error) {
	f.calls.Add(1)
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.err != nil {
		return nil, f.err
	}
	if f.reply != nil {
		return f.reply, nil
	}
	return json.RawMessage(`{}`), nil
}

func (f *fakeWireTransport) Close() error {
	f.closed.Store(true)
	return nil
}
