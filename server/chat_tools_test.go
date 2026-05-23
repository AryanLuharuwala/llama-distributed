package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestExtractToolCalls(t *testing.T) {
	in := `Sure, let me look that up.
<tool_call>{"server":"fs","method":"read_file","params":{"path":"/etc/hostname"}}</tool_call>
And another:
<tool_call>{"server":"github","method":"list_issues","params":{"repo":"acme/x"}}</tool_call>
Done.`
	calls := extractToolCalls(in)
	if len(calls) != 2 {
		t.Fatalf("want 2 calls, got %d", len(calls))
	}
	if calls[0].Server != "fs" || calls[0].Method != "read_file" {
		t.Errorf("call[0] = %+v", calls[0])
	}
	if calls[1].Server != "github" || calls[1].Method != "list_issues" {
		t.Errorf("call[1] = %+v", calls[1])
	}
}

func TestExtractToolCallsSkipsMalformed(t *testing.T) {
	in := `<tool_call>{not json}</tool_call>
<tool_call>{"server":"fs"}</tool_call>
<tool_call>{"server":"fs","method":"ok","params":{}}</tool_call>`
	calls := extractToolCalls(in)
	if len(calls) != 1 {
		t.Fatalf("want 1 valid call, got %d (%+v)", len(calls), calls)
	}
}

func TestStripToolCalls(t *testing.T) {
	in := `Here is the result:
<tool_call>{"server":"fs","method":"read_file","params":{}}</tool_call>
The file contains "ok".`
	out := stripToolCalls(in)
	if strings.Contains(out, "tool_call") {
		t.Errorf("envelope leaked: %q", out)
	}
	if !strings.Contains(out, "Here is the result") || !strings.Contains(out, `contains "ok"`) {
		t.Errorf("expected narrative kept, got %q", out)
	}
}

func TestChatToolsOptsCap(t *testing.T) {
	var nilOpts *chatToolsOpts
	if got := nilOpts.cap(); got != chatToolsDefaultMaxCalls {
		t.Errorf("nil cap = %d, want %d", got, chatToolsDefaultMaxCalls)
	}
	if got := (&chatToolsOpts{}).cap(); got != chatToolsDefaultMaxCalls {
		t.Errorf("zero cap = %d, want %d", got, chatToolsDefaultMaxCalls)
	}
	if got := (&chatToolsOpts{MaxCalls: 99}).cap(); got != chatToolsHardMaxCalls {
		t.Errorf("over-cap = %d, want %d", got, chatToolsHardMaxCalls)
	}
	if got := (&chatToolsOpts{MaxCalls: 3}).cap(); got != 3 {
		t.Errorf("explicit cap = %d, want 3", got)
	}
}

func TestBuildToolsManifest(t *testing.T) {
	s, _ := openMCPTestDB(t)
	_, err := s.addMCPServer(&mcpServer{
		UserID: 1, Name: "fs-local", Transport: mcpTransportStdio,
		Endpoint: "/bin/ls", Enabled: true,
		Scopes: []string{"read", "list"},
	})
	if err != nil {
		t.Fatalf("add: %v", err)
	}
	_, _ = s.addMCPServer(&mcpServer{
		UserID: 1, Name: "disabled", Transport: mcpTransportHTTP,
		Endpoint: "https://x.example.com/mcp", Enabled: false,
	})
	_, _ = s.addMCPServer(&mcpServer{
		UserID: 1, Name: "other", Transport: mcpTransportHTTP,
		Endpoint: "https://y.example.com/mcp", Enabled: true,
	})

	manifest, err := s.buildToolsManifest(1, &chatToolsOpts{MCPServers: []string{"fs-local", "disabled"}})
	if err != nil {
		t.Fatalf("buildToolsManifest: %v", err)
	}
	if !strings.Contains(manifest, "fs-local") {
		t.Errorf("manifest missing fs-local: %q", manifest)
	}
	if strings.Contains(manifest, "disabled") {
		t.Errorf("disabled server leaked into manifest: %q", manifest)
	}
	if strings.Contains(manifest, "other") {
		t.Errorf("unselected server leaked into manifest: %q", manifest)
	}
	if !strings.Contains(manifest, "<tool_call>") {
		t.Errorf("manifest missing call-syntax hint: %q", manifest)
	}
}

func TestBuildToolsManifestEmptyWhenNoMatches(t *testing.T) {
	s, _ := openMCPTestDB(t)
	out, err := s.buildToolsManifest(1, &chatToolsOpts{MCPServers: []string{"nonexistent"}})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if out != "" {
		t.Errorf("want empty manifest, got %q", out)
	}
}

func TestExecuteToolCallsRoutesThroughMCP(t *testing.T) {
	s, _ := openMCPTestDB(t)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var env struct {
			ID     int             `json:"id"`
			Method string          `json:"method"`
			Params json.RawMessage `json:"params"`
		}
		_ = json.Unmarshal(body, &env)
		fmt.Fprintf(w, `{"jsonrpc":"2.0","id":%d,"result":{"got":%s,"method":%q}}`, env.ID, string(env.Params), env.Method)
	}))
	defer upstream.Close()

	_, err := s.addMCPServer(&mcpServer{
		UserID: 7, Name: "fixture", Transport: mcpTransportHTTP,
		Endpoint: upstream.URL, Enabled: true,
	})
	if err != nil {
		t.Fatalf("add: %v", err)
	}

	calls := []chatToolCall{
		{Server: "fixture", Method: "alpha", Params: json.RawMessage(`{"x":1}`)},
		{Server: "fixture", Method: "beta", Params: json.RawMessage(`{"y":2}`)},
	}
	results := s.executeToolCalls(context.Background(), 7, &chatToolsOpts{MaxCalls: 5}, calls)
	if len(results) != 2 {
		t.Fatalf("want 2 results, got %d", len(results))
	}
	for i, r := range results {
		if r.Error != "" {
			t.Errorf("results[%d].Error = %q", i, r.Error)
		}
		if !strings.Contains(string(r.Result), `"method"`) {
			t.Errorf("results[%d].Result = %s", i, r.Result)
		}
	}
}

func TestExecuteToolCallsEnforcesCap(t *testing.T) {
	s, _ := openMCPTestDB(t)
	calls := []chatToolCall{
		{Server: "missing", Method: "x"},
		{Server: "missing", Method: "y"},
		{Server: "missing", Method: "z"},
	}
	// Cap to 1; the second and third must be flagged as dropped, regardless
	// of what would happen on the wire.
	results := s.executeToolCalls(context.Background(), 1, &chatToolsOpts{MaxCalls: 1}, calls)
	if len(results) != 3 {
		t.Fatalf("want 3 results (1 attempted + 2 dropped), got %d", len(results))
	}
	if results[0].Error == "" {
		t.Errorf("expected attempted call to error (no server), got success")
	}
	for i := 1; i < 3; i++ {
		if !strings.Contains(results[i].Error, "exceeded max_calls") {
			t.Errorf("results[%d].Error = %q, want dropped marker", i, results[i].Error)
		}
	}
}
