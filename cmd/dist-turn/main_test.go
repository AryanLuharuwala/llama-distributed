package main

import (
	"net"
	"strings"
	"testing"
	"time"
)

// TestHTTPToWS pins the scheme rewrite that runStandalone relies on
// when computing /api/agent.  http→ws, https→wss; ws/wss pass through;
// anything else is an error.
func TestHTTPToWS(t *testing.T) {
	cases := []struct {
		in, want string
		wantErr  bool
	}{
		{"http://pool.example.com", "ws://pool.example.com", false},
		{"https://pool.example.com/", "wss://pool.example.com/", false},
		{"http://pool.example.com:8080/path", "ws://pool.example.com:8080/path", false},
		{"ws://already", "ws://already", false},
		{"wss://already/x", "wss://already/x", false},
		{"ftp://nope", "", true},
		{"::not-a-url", "", true},
	}
	for _, c := range cases {
		got, err := httpToWS(c.in)
		if c.wantErr {
			if err == nil {
				t.Errorf("httpToWS(%q): expected error, got %q", c.in, got)
			}
			continue
		}
		if err != nil {
			t.Errorf("httpToWS(%q): %v", c.in, err)
			continue
		}
		if got != c.want {
			t.Errorf("httpToWS(%q): got %q want %q", c.in, got, c.want)
		}
	}
}

// TestBuildHello asserts the hello/resume distinction the standalone
// pair loop relies on: an empty agent_key sends "hello" + token; a
// populated one sends "resume" + agent_key (token is dropped — the
// server side ignores it but we don't want stale pair tokens leaking
// out in reconnect frames either).
func TestBuildHello(t *testing.T) {
	fresh := buildHello(pairConfig{
		AgentID:  "relay-aa",
		Token:    "pair-tok-xyz",
		Hostname: "h",
	})
	if fresh["kind"] != "hello" {
		t.Errorf("fresh kind = %v, want hello", fresh["kind"])
	}
	if fresh["token"] != "pair-tok-xyz" {
		t.Errorf("fresh token missing")
	}
	if _, ok := fresh["agent_key"]; ok {
		t.Errorf("fresh hello must not carry agent_key")
	}

	resumed := buildHello(pairConfig{
		AgentID:  "relay-aa",
		Token:    "ignored",
		AgentKey: "kkk",
		Hostname: "h",
	})
	if resumed["kind"] != "resume" {
		t.Errorf("resume kind = %v, want resume", resumed["kind"])
	}
	if resumed["agent_key"] != "kkk" {
		t.Errorf("resume must carry agent_key")
	}
	if _, ok := resumed["token"]; ok {
		t.Errorf("resume frame must not include pair token")
	}
}

// TestBuildStatus pins the relay's status frame: relay_capable=true,
// roles includes turn_relay, and coturn_port reflects the bound port.
// The planner uses these fields to route symmetric-NAT pairs onto the
// relay.
func TestBuildStatus(t *testing.T) {
	st := buildStatus(pairConfig{ExtIP: "1.2.3.4"}, 3478)
	if st["relay_capable"] != true {
		t.Errorf("relay_capable should be true: %v", st["relay_capable"])
	}
	if st["coturn_port"] != 3478 {
		t.Errorf("coturn_port = %v, want 3478", st["coturn_port"])
	}
	if st["public_ip"] != "1.2.3.4" {
		t.Errorf("public_ip = %v, want 1.2.3.4", st["public_ip"])
	}
	roles, ok := st["roles"].([]string)
	if !ok || len(roles) == 0 || roles[0] != "turn_relay" {
		t.Errorf("roles missing or wrong: %v", st["roles"])
	}
	if st["nat_type"] != "open" {
		t.Errorf("nat_type = %v, want open (relay nodes are by definition reachable)", st["nat_type"])
	}
}

// TestTruncate exercises the WS-error log helper.  Boundary cases:
// empty input, exact match, oversize, multibyte safety.
func TestTruncate(t *testing.T) {
	if got := truncate(nil, 10); got != "" {
		t.Errorf("nil → %q", got)
	}
	if got := truncate([]byte("short"), 10); got != "short" {
		t.Errorf("short → %q", got)
	}
	if got := truncate([]byte("0123456789"), 10); got != "0123456789" {
		t.Errorf("exact → %q", got)
	}
	if got := truncate([]byte("0123456789abc"), 5); got != "01234…" {
		t.Errorf("oversize → %q", got)
	}
}

// TestStartTurnRejectsBadAddr asserts the listenAddr validator: a port
// that fails strconv.Atoi or net.SplitHostPort must error before we
// open any UDP socket.  Catches the failure mode where an operator
// passes "--listen=:" or "--listen=foo".
func TestStartTurnRejectsBadAddr(t *testing.T) {
	cases := []string{
		"",
		"not-a-host-port",
		"0.0.0.0:not-a-port",
		"0.0.0.0:0",
		"0.0.0.0:99999",
	}
	for _, c := range cases {
		_, _, err := startTurn(c, "dist", "secret", "")
		if err == nil {
			t.Errorf("startTurn(%q): expected error, got nil", c)
		}
	}
}

// TestStartTurnHonorsExternalIP — when --external-ip is set to an
// invalid string, startTurn must refuse to start rather than silently
// fall back to RelayAddressGeneratorNone.  This is a security
// property: the operator's intent ("advertise this address") must be
// honored or rejected, never quietly downgraded.
func TestStartTurnHonorsExternalIP(t *testing.T) {
	port := pickEphemeralUDPPort(t)
	addr := net.JoinHostPort("127.0.0.1", port)
	_, _, err := startTurn(addr, "dist", "secret", "not.an.ip.address")
	if err == nil || !strings.Contains(err.Error(), "external-ip") {
		t.Errorf("expected external-ip error, got: %v", err)
	}
}

// TestStartTurnBindsAndCloses — happy path bind: opens a listener,
// returns a *turn.Server, and the conn list can be closed without
// panicking.  This is the smoke that catches API drift in pion/turn.
func TestStartTurnBindsAndCloses(t *testing.T) {
	port := pickEphemeralUDPPort(t)
	addr := net.JoinHostPort("127.0.0.1", port)
	srv, conns, err := startTurn(addr, "dist", "secret", "")
	if err != nil {
		t.Fatalf("startTurn: %v", err)
	}
	if srv == nil {
		t.Fatal("nil server")
	}
	if len(conns) == 0 {
		t.Fatal("no packet conns returned")
	}
	if err := srv.Close(); err != nil {
		t.Errorf("close: %v", err)
	}
	// Give the OS a beat to release the port before the test runner
	// moves on; not strictly required, but keeps -count=N runs sane.
	time.Sleep(20 * time.Millisecond)
}

// pickEphemeralUDPPort grabs and immediately releases a random UDP
// port so the next caller (startTurn) can claim it without the race
// of asking the kernel for "port 0" through pion/turn's address
// parser (which only takes explicit ports).
func pickEphemeralUDPPort(t *testing.T) string {
	t.Helper()
	pc, err := net.ListenPacket("udp4", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("probe listen: %v", err)
	}
	_, p, _ := net.SplitHostPort(pc.LocalAddr().String())
	_ = pc.Close()
	return p
}
