package main

import (
	"net/http/httptest"
	"testing"
	"time"

	"golang.org/x/time/rate"
)

func TestIPRateBucketBurstThenDeny(t *testing.T) {
	b := newIPRateBucket(rate.Every(time.Hour), 3) // burst of 3, very slow refill
	ip := "203.0.113.7"
	for i := 0; i < 3; i++ {
		if !b.allow(ip) {
			t.Fatalf("call %d should be allowed inside burst", i+1)
		}
	}
	if b.allow(ip) {
		t.Errorf("4th call should be denied — burst exhausted")
	}
}

func TestIPRateBucketsAreIndependent(t *testing.T) {
	// Two distinct buckets so noise in one doesn't eat budget in the other.
	approve := newIPRateBucket(rate.Every(time.Hour), 1)
	poll := newIPRateBucket(rate.Every(time.Hour), 1)
	ip := "198.51.100.4"
	if !approve.allow(ip) {
		t.Fatalf("first approve must pass")
	}
	if approve.allow(ip) {
		t.Errorf("second approve must fail — burst exhausted")
	}
	if !poll.allow(ip) {
		t.Errorf("poll bucket should not have been touched by approve traffic")
	}
}

func TestIPRateBucketDifferentIPsDontShareBudget(t *testing.T) {
	b := newIPRateBucket(rate.Every(time.Hour), 1)
	if !b.allow("203.0.113.1") {
		t.Fatalf("first IP must pass")
	}
	if !b.allow("203.0.113.2") {
		t.Errorf("second IP should have its own bucket")
	}
}

func TestIPRateBucketEmptyIPFailsOpen(t *testing.T) {
	b := newIPRateBucket(rate.Every(time.Hour), 0) // burst 0 — would deny everything
	if !b.allow("") {
		t.Errorf("empty IP must fail open (caller's responsibility to extract IP)")
	}
}

func TestIPRateBucketJanitorPrunesStale(t *testing.T) {
	b := newIPRateBucket(rate.Every(time.Hour), 1)
	b.allow("203.0.113.99")
	// Force the bucket's lastSeen far into the past so the TTL we pass below is "old enough."
	b.buckets.Range(func(_, v any) bool {
		v.(*ipLimBucket).lastSeen.set(time.Now().Add(-2 * time.Hour).Unix())
		return true
	})
	b.janitor(time.Minute)
	count := 0
	b.buckets.Range(func(_, _ any) bool { count++; return true })
	if count != 0 {
		t.Errorf("janitor should have pruned the stale entry, got %d remaining", count)
	}
}

func TestRemoteIPForRateLimitXFF(t *testing.T) {
	cases := []struct {
		name string
		xff  string
		addr string
		want string
	}{
		{"single xff", "203.0.113.7", "127.0.0.1:1234", "203.0.113.7"},
		{"xff list takes first", "203.0.113.7, 10.0.0.1", "127.0.0.1:1234", "203.0.113.7"},
		{"no xff falls back to remoteaddr", "", "203.0.113.99:51234", "203.0.113.99"},
		{"remoteaddr unparseable returns raw", "", "weird", "weird"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := httptest.NewRequest("GET", "/", nil)
			r.RemoteAddr = c.addr
			if c.xff != "" {
				r.Header.Set("X-Forwarded-For", c.xff)
			}
			if got := remoteIPForRateLimit(r); got != c.want {
				t.Errorf("got %q want %q", got, c.want)
			}
		})
	}
}
