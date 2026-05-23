package main

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// /healthz, /readyz, /metrics — make sure each returns the right
// content-type and surfaces the data scrapers expect.

func TestHealthzAlwaysOK(t *testing.T) {
	s := newTestServer(t)
	rr := httptest.NewRecorder()
	s.router().ServeHTTP(rr, httptest.NewRequest("GET", "/healthz", nil))
	if rr.Code != 200 || strings.TrimSpace(rr.Body.String()) != "ok" {
		t.Errorf("/healthz: code=%d body=%q", rr.Code, rr.Body.String())
	}
}

func TestReadyzReady(t *testing.T) {
	s := newTestServer(t)
	rr := httptest.NewRecorder()
	s.router().ServeHTTP(rr, httptest.NewRequest("GET", "/readyz", nil))
	if rr.Code != 200 {
		t.Errorf("/readyz: code=%d body=%q", rr.Code, rr.Body.String())
	}
}

func TestReadyz503DuringShutdown(t *testing.T) {
	s := newTestServer(t)
	s.shuttingDown.Store(true)
	rr := httptest.NewRecorder()
	s.router().ServeHTTP(rr, httptest.NewRequest("GET", "/readyz", nil))
	if rr.Code != http.StatusServiceUnavailable {
		t.Errorf("/readyz during shutdown should be 503; got %d", rr.Code)
	}
}

func TestMetricsExposition(t *testing.T) {
	s := newTestServer(t)
	// Seed one reputation row so the COALESCE branch is exercised.
	s.recordRelaySuccess("rig-metrics", 4096)

	rr := httptest.NewRecorder()
	s.router().ServeHTTP(rr, httptest.NewRequest("GET", "/metrics", nil))
	if rr.Code != 200 {
		t.Fatalf("/metrics: code=%d", rr.Code)
	}
	ct := rr.Header().Get("Content-Type")
	if !strings.HasPrefix(ct, "text/plain") {
		t.Errorf("/metrics content-type %q should start with text/plain", ct)
	}
	body, _ := io.ReadAll(rr.Body)
	want := []string{
		"distinf_uptime_seconds",
		"distinf_agents_online",
		"distinf_reputation_rows",
		"distinf_relay_sessions_total",
		"distinf_relay_bytes_forwarded_total",
		"distinf_inference_slots_used",
		"distinf_inference_slots_capacity",
		"distinf_comfy_jobs_active",
		"distinf_comfy_jobs_queued",
		"distinf_comfy_jobs_failed_total",
		"go_goroutines",
		"# HELP",
		"# TYPE",
	}
	for _, w := range want {
		if !strings.Contains(string(body), w) {
			t.Errorf("/metrics missing %q\nbody:\n%s", w, body)
		}
	}
	// We seeded 4096 forwarded bytes — make sure the counter actually reflects it.
	if !strings.Contains(string(body), "distinf_relay_bytes_forwarded_total 4096") {
		t.Errorf("/metrics: byte counter should report 4096\nbody:\n%s", body)
	}
}

func TestNegotiateProtocol(t *testing.T) {
	cases := []struct {
		in        int
		wantVer   int
		wantOK    bool
	}{
		{0, 1, true},                  // legacy → v1
		{1, 1, true},                  // exact
		{serverProtocolMax + 5, serverProtocolMax, true}, // future rig clamped down
	}
	for _, c := range cases {
		got, ok := negotiateProtocol(c.in)
		if got != c.wantVer || ok != c.wantOK {
			t.Errorf("negotiateProtocol(%d) = (%d, %v); want (%d, %v)",
				c.in, got, ok, c.wantVer, c.wantOK)
		}
	}
	// Only meaningful when min > 1 — assert at least that the function
	// rejects below-min if we ever bump it.  Today this is a no-op.
	if serverProtocolMin > 1 {
		if _, ok := negotiateProtocol(serverProtocolMin - 1); ok {
			t.Errorf("expected reject for too-old protocol version")
		}
	}
}
