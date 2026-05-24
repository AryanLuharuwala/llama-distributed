package main

import (
	"context"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"golang.org/x/time/rate"
)

// newTestRedisBackend spins up an in-process Redis substitute and dials
// a backend against it.  Used by the GCRA + wiring tests below; the
// caller owns the cleanup.
func newTestRedisBackend(t *testing.T) (*redisRateBackend, *miniredis.Miniredis) {
	t.Helper()
	mr := miniredis.RunT(t)
	be, err := newRedisRateBackend(context.Background(), "redis://"+mr.Addr())
	if err != nil {
		t.Fatalf("dial miniredis: %v", err)
	}
	t.Cleanup(func() { _ = be.close() })
	return be, mr
}

// TestRedisRateBackendGCRABurst — the bucket should allow up to `burst`
// requests immediately and then deny.  This is the core contract every
// caller of allow() depends on; if it ever regresses, every per-IP
// throttle silently stops working in multi-replica deploys.
func TestRedisRateBackendGCRABurst(t *testing.T) {
	be, _ := newTestRedisBackend(t)
	ctx := context.Background()
	const burst = 3
	for i := 0; i < burst; i++ {
		ok, err := be.allow(ctx, "k", 0.0001, burst) // tiny refill rate
		if err != nil {
			t.Fatalf("call %d: %v", i, err)
		}
		if !ok {
			t.Fatalf("call %d should be allowed inside burst", i+1)
		}
	}
	ok, err := be.allow(ctx, "k", 0.0001, burst)
	if err != nil {
		t.Fatalf("denied call err: %v", err)
	}
	if ok {
		t.Errorf("burst+1 should be denied")
	}
}

// Distinct keys must not share budget — the whole point of bucketing by
// (endpoint, IP) is that one IP's noise on /api/device/approve doesn't
// eat another IP's budget on the same endpoint or the same IP's budget
// on /api/device/token.
func TestRedisRateBackendKeysIndependent(t *testing.T) {
	be, _ := newTestRedisBackend(t)
	ctx := context.Background()
	if ok, _ := be.allow(ctx, "ip-a", 0.0001, 1); !ok {
		t.Fatalf("ip-a first call should be allowed")
	}
	if ok, _ := be.allow(ctx, "ip-a", 0.0001, 1); ok {
		t.Errorf("ip-a second call should be denied — burst exhausted")
	}
	if ok, _ := be.allow(ctx, "ip-b", 0.0001, 1); !ok {
		t.Errorf("ip-b should have its own bucket")
	}
}

// Refill: after waiting > 1/rate, a previously-empty bucket should
// recover one token.  Use miniredis.FastForward to skip real wall time.
func TestRedisRateBackendRefill(t *testing.T) {
	be, mr := newTestRedisBackend(t)
	ctx := context.Background()
	// Burst 1, refill 1/sec.
	if ok, _ := be.allow(ctx, "k", 1, 1); !ok {
		t.Fatalf("first call must be allowed")
	}
	if ok, _ := be.allow(ctx, "k", 1, 1); ok {
		t.Fatalf("second call must be denied (burst exhausted)")
	}
	// Advance miniredis's clock by 2s — the script reads time.Now() on
	// the Go side, so we also wait briefly.  Sleep is cheap here; 1.2s
	// keeps the test under 2s total.
	mr.FastForward(2 * time.Second)
	time.Sleep(1100 * time.Millisecond)
	if ok, _ := be.allow(ctx, "k", 1, 1); !ok {
		t.Errorf("after 1s+ wait, bucket should have refilled at least 1 token")
	}
}

// Empty key fails open (matches the in-process bucket's contract).
func TestRedisRateBackendEmptyKey(t *testing.T) {
	be, _ := newTestRedisBackend(t)
	ok, err := be.allow(context.Background(), "", 1, 0)
	if err != nil {
		t.Fatalf("empty key err: %v", err)
	}
	if !ok {
		t.Errorf("empty key must fail open — caller owns IP extraction")
	}
}

// When the backing Redis is gone we must fail open and not block the
// request path.  Simulated by closing the client; subsequent allow()
// calls should return (true, err) so callers honor the bool.
func TestRedisRateBackendFailsOpenOnError(t *testing.T) {
	be, mr := newTestRedisBackend(t)
	mr.Close() // pull the rug
	ok, err := be.allow(context.Background(), "k", 1, 1)
	if err == nil {
		t.Errorf("expected error from dead redis, got nil")
	}
	if !ok {
		t.Errorf("must fail open when redis is unreachable (got denied)")
	}
}

// End-to-end: ipRateBucket with a backend wired should consult the
// backend instead of the local sync.Map.  We verify this by giving the
// bucket a tight burst and observing that the *second* IP call still
// gets through (proves keys are scoped by name+ip).
func TestIPRateBucketWithRedisBackend(t *testing.T) {
	be, _ := newTestRedisBackend(t)
	b := newIPRateBucket(rate.Every(time.Hour), 1).withBackend("test", be)
	if !b.allow("203.0.113.1") {
		t.Fatalf("first call from ip1 should be allowed")
	}
	if b.allow("203.0.113.1") {
		t.Errorf("second call from ip1 should be denied (burst 1)")
	}
	if !b.allow("203.0.113.2") {
		t.Errorf("first call from ip2 should be allowed (independent bucket)")
	}
}
