package main

// Tool-call channel on the chat-completion path.
//
// Most self-hosted gguf models we serve don't speak the OpenAI native
// "tool_calls" frame.  Rather than wait for runtime-level support (vLLM
// / SGLang / TensorRT-LLM all surface it differently — task #115 calls
// out the full protobuf-typed tool channel as prod-branch scope), we
// take a JSON-envelope approach that works for any model:
//
//   1. Caller supplies `"tools": { "mcp_servers": ["fs", "github"],
//      "auto_execute": true, "max_calls": 4 }` in the request.
//
//   2. We prepend a system message listing the available tools and the
//      exact sentinel syntax the model must emit to invoke one:
//
//        <tool_call>{"server":"fs","method":"read_file",
//                    "params":{"path":"..."}}</tool_call>
//
//   3. After the model finishes, we scan the response for tool_call
//      sentinels.  If auto_execute is set, each call is routed through
//      callMCP (audited like any other MCP call).  Results are returned
//      to the client as a `tool_calls` array alongside the assistant
//      message — the client can choose to feed them back into the next
//      turn, or render them inline.
//
// Why server-side execution instead of returning tool calls and waiting
// for the client to invoke them?  The MCP credentials (env:VAR pointers)
// live on the server.  Round-tripping through the client would require
// shipping the secret, which defeats the registry's purpose.  Clients
// that want manual invocation can still get there with auto_execute=false
// — they receive the parsed calls and can call POST /api/mcp/{server}/call
// directly.
//
// Bounded: max_calls caps how many tools a single response can invoke,
// so a runaway model can't burn through a user's MCP budget.  Default
// 4, hard ceiling 16.

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

const (
	chatToolsDefaultMaxCalls = 4
	chatToolsHardMaxCalls    = 16
)

type chatToolsOpts struct {
	MCPServers  []string `json:"mcp_servers"`
	AutoExecute bool     `json:"auto_execute"`
	MaxCalls    int      `json:"max_calls"`
}

func (o *chatToolsOpts) isEmpty() bool {
	return o == nil || len(o.MCPServers) == 0
}

func (o *chatToolsOpts) cap() int {
	if o == nil || o.MaxCalls <= 0 {
		return chatToolsDefaultMaxCalls
	}
	if o.MaxCalls > chatToolsHardMaxCalls {
		return chatToolsHardMaxCalls
	}
	return o.MaxCalls
}

// chatToolCall is the parsed shape of a single <tool_call>…</tool_call>
// envelope detected in model output.
type chatToolCall struct {
	Server string          `json:"server"`
	Method string          `json:"method"`
	Params json.RawMessage `json:"params"`
}

// chatToolResult is what we return to the client per call: the original
// invocation plus the broker's response.  Result is either the raw RPC
// result or — if the broker returned an error — an error string.
type chatToolResult struct {
	Call   chatToolCall    `json:"call"`
	Result json.RawMessage `json:"result,omitempty"`
	Error  string          `json:"error,omitempty"`
}

// buildToolsManifest renders the available tools into a system message
// the model can consult.  We list the user's registered MCP servers
// (filtered by opts.MCPServers if non-empty) along with their scopes.
// If the user has no servers, returns "" so we don't bloat the prompt.
func (s *server) buildToolsManifest(uid int64, opts *chatToolsOpts) (string, error) {
	if opts.isEmpty() {
		return "", nil
	}
	all, err := s.listMCPServers(uid)
	if err != nil {
		return "", err
	}
	want := make(map[string]bool, len(opts.MCPServers))
	for _, n := range opts.MCPServers {
		want[strings.TrimSpace(n)] = true
	}
	var avail []mcpServer
	for _, m := range all {
		if !m.Enabled {
			continue
		}
		if !want[m.Name] {
			continue
		}
		avail = append(avail, m)
	}
	if len(avail) == 0 {
		return "", nil
	}

	var b strings.Builder
	b.WriteString("[Tools: you may call any of the following MCP servers.\n")
	b.WriteString("To invoke a tool, emit a single line of the form:\n")
	b.WriteString(`  <tool_call>{"server":"<name>","method":"<method>","params":{…}}</tool_call>` + "\n")
	b.WriteString("Emit at most ")
	b.WriteString(itoaTools(opts.cap()))
	b.WriteString(" tool calls per response. After tool calls finish, ")
	b.WriteString("continue your answer using the results.]\n\n")
	for _, m := range avail {
		fmt.Fprintf(&b, "- server=%q transport=%s", m.Name, string(m.Transport))
		if len(m.Scopes) > 0 {
			fmt.Fprintf(&b, " scopes=%v", m.Scopes)
		}
		b.WriteString("\n")
	}
	b.WriteString("[end tools]")
	return b.String(), nil
}

// toolCallRE matches one <tool_call>…</tool_call> envelope.  The body
// is captured non-greedily so adjacent envelopes don't collapse into one.
var toolCallRE = regexp.MustCompile(`(?s)<tool_call>\s*(\{.*?\})\s*</tool_call>`)

// extractToolCalls finds every <tool_call> envelope in `text`, parses
// the JSON body, and returns the resulting calls.  Malformed envelopes
// are skipped — we don't want one bad payload to drop sibling calls.
func extractToolCalls(text string) []chatToolCall {
	matches := toolCallRE.FindAllStringSubmatch(text, -1)
	if len(matches) == 0 {
		return nil
	}
	out := make([]chatToolCall, 0, len(matches))
	for _, m := range matches {
		var c chatToolCall
		if err := json.Unmarshal([]byte(m[1]), &c); err != nil {
			continue
		}
		if c.Server == "" || c.Method == "" {
			continue
		}
		out = append(out, c)
	}
	return out
}

// executeToolCalls invokes each call through the MCP broker, capping the
// total at opts.cap().  Calls beyond the cap are recorded as errors so
// the client knows they were dropped intentionally.  Each call has its
// own audit row written by callMCP.
func (s *server) executeToolCalls(ctx context.Context, uid int64, opts *chatToolsOpts, calls []chatToolCall) []chatToolResult {
	limit := opts.cap()
	out := make([]chatToolResult, 0, len(calls))
	for i, c := range calls {
		if i >= limit {
			out = append(out, chatToolResult{
				Call:  c,
				Error: fmt.Sprintf("tool call dropped: exceeded max_calls=%d", limit),
			})
			continue
		}
		res, err := s.callMCP(ctx, uid, c.Server, c.Method, c.Params)
		if err != nil {
			out = append(out, chatToolResult{Call: c, Error: err.Error()})
			continue
		}
		out = append(out, chatToolResult{Call: c, Result: res})
	}
	return out
}

// stripToolCalls removes <tool_call>…</tool_call> envelopes from text
// so the assistant message we return to the client contains only the
// natural-language portion of the model output.  The structured calls
// live on the separate `tool_calls` field.
func stripToolCalls(text string) string {
	return strings.TrimSpace(toolCallRE.ReplaceAllString(text, ""))
}

// itoaTools is a tiny non-allocating itoa for the prompt builder.  We
// avoid pulling strconv into a hot path that's already touched twice.
func itoaTools(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	neg := false
	if n < 0 {
		neg = true
		n = -n
	}
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}
