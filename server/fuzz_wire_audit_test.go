package main

// Wire-format fuzz harnesses added for the audit at audit/wire.md.
//
// These don't ship a corpus and are intentionally cheap so they can be
// kicked off ad-hoc with `go test -run='^$' -fuzz=FuzzX -fuzztime=60s`.
//
// Each target lifts an attacker-controlled JSON unmarshal + validation
// flow out of ws.go and runs it as a pure function.  We don't try to
// stand up an *server; the parsers above the DB layer are pure data.

import (
	"encoding/json"
	"testing"
)

// FuzzAgentStatusUnmarshal exercises the JSON → agentStatus → clamp
// flow that runs for every "status" frame an agent sends.
func FuzzAgentStatusUnmarshal(f *testing.F) {
	// Realistic seeds first.
	f.Add([]byte(`{"kind":"status","n_gpus":1,"tokens_sec":42}`))
	f.Add([]byte(`{"kind":"status","vram_total":-1}`))
	f.Add([]byte(`{"kind":"status","models":["a","b","c"]}`))
	f.Add([]byte(`{"kind":"status","public_ip":"1.2.3.4"}`))
	// Adversarial seeds.
	f.Add([]byte(`{"kind":"status","n_gpus":2147483647}`))
	f.Add([]byte(`{"kind":"status","tokens_sec":1e308}`))
	f.Add([]byte(`{"kind":"status","gpu_model":"` + string(make([]byte, 1<<20)) + `"}`))
	f.Add([]byte(`{"kind":"status","models":` + bigArray(10000) + `}`))

	f.Fuzz(func(t *testing.T, data []byte) {
		var st agentStatus
		if err := json.Unmarshal(data, &st); err != nil {
			return
		}
		validateAndClampStatus(&st, "1.2.3.4")
	})
}

// FuzzMacaroonVerify hits the cap verification flow with arbitrary
// base64 tokens.  Trailing-junk attacks, malformed binary macaroons,
// and base64 with embedded NULs all funnel through here.
func FuzzMacaroonVerify(f *testing.F) {
	f.Add("")
	f.Add("not base64!")
	f.Add("AAAA")
	f.Add("invalidmacaroonbinarygarbage")

	f.Fuzz(func(t *testing.T, token string) {
		// Stand up a minimal server with a deterministic session secret so
		// the root key is stable across fuzz runs.
		s := &server{}
		s.cfg.sessionSecret = "fuzz-secret"
		// We don't care about the return error — only that it doesn't panic
		// or hang the goroutine on malformed input.
		_ = s.verifyCap(token, func(string) error { return nil }, nil)
	})
}

func bigArray(n int) string {
	out := "["
	for i := 0; i < n; i++ {
		if i > 0 {
			out += ","
		}
		out += `"x"`
	}
	out += "]"
	return out
}
