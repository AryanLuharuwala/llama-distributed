package main

import (
	"testing"
	"time"

	"golang.org/x/time/rate"
)

// TestHelloFailBucketPeek validates that the helloFail bucket is queried
// without consumption via peek, but allow() does consume.  This is the
// contract that handleAgentWS relies on for "check before, charge on
// failure" semantics.
func TestHelloFailBucketPeek(t *testing.T) {
	b := newIPRateBucket(rate.Every(time.Hour), 1)
	ip := "192.0.2.5"

	if !b.peek(ip) {
		t.Fatalf("peek must be true with full bucket")
	}
	if !b.peek(ip) {
		t.Fatalf("peek must remain true (does not consume)")
	}
	if !b.allow(ip) {
		t.Fatalf("allow must succeed and consume the one token")
	}
	if b.peek(ip) {
		t.Errorf("peek must be false after the token is consumed")
	}
	if b.allow(ip) {
		t.Errorf("allow must be false after exhaustion")
	}
}

// TestAuthFailJitterRuns just exercises the jitter helper to make sure
// it doesn't panic and returns within a sensible window.  The actual
// timing is non-deterministic but bounded to 75ms..250ms.
func TestAuthFailJitterRuns(t *testing.T) {
	start := time.Now()
	authFailJitter()
	d := time.Since(start)
	if d < 50*time.Millisecond || d > 400*time.Millisecond {
		t.Errorf("authFailJitter took %v, want 75-250ms", d)
	}
}
