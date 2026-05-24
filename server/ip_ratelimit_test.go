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

// remoteIPForRateLimit on the prod branch consults the trusted-proxy
// resolver in trusted_proxy.go: XFF is honored only when the TCP peer
// is a trusted hop.  These cases verify the contract callers depend on
// (rate-limiter keys, device-code ratchet, audit trail).
func TestRemoteIPForRateLimitXFF(t *testing.T) {
	cases := []struct {
		name  string
		trust string
		xff   string
		addr  string
		want  string
	}{
		// Empty trust set ⇒ XFF ignored.  The historical "trust XFF
		// blindly" behavior is gone; without configuring trust, the
		// only thing we attribute traffic to is the TCP peer.
		{"trust empty: ignores xff",
			"", "203.0.113.7", "198.51.100.5:1234", "198.51.100.5"},
		// Loopback proxy, single client hop — the most common
		// envoy-on-the-same-host topology.  The peer (127.0.0.1) is
		// trusted, so XFF is consulted; the only entry is the real
		// client.
		{"trusted single hop",
			"127.0.0.1/32", "203.0.113.7", "127.0.0.1:1234", "203.0.113.7"},
		// Multi-hop chain where every intermediate proxy is trusted.
		// Walking right-to-left, we skip 10.0.0.1 (trusted) and stop
		// at 203.0.113.7 — the actual client.
		{"trusted chain walks right-to-left to client",
			"127.0.0.1/32,10.0.0.0/8",
			"203.0.113.7, 10.0.0.1", "127.0.0.1:1234", "203.0.113.7"},
		// Untrusted peer with an XFF header — the classic spoof.
		// We must refuse to honor the header and report the TCP peer.
		{"untrusted peer ignores xff (spoof defense)",
			"10.0.0.0/8", "203.0.113.7", "198.51.100.5:1234", "198.51.100.5"},
		// Trusted peer, no XFF — misconfigured proxy.  Best we can do
		// is surface the peer; we never invent a "client" IP.
		{"trusted peer no xff",
			"127.0.0.1/32", "", "127.0.0.1:1234", "127.0.0.1"},
		// Garbage in XFF mid-chain — at that point the chain integrity
		// is broken and we surface the garbage rather than scanning
		// past it (an attacker controlling one upstream-of-garbage hop
		// could otherwise inject a fake "earlier" entry).
		{"trusted chain with malformed entry",
			"127.0.0.1/32",
			"not-an-ip, 127.0.0.1", "127.0.0.1:1234", "not-an-ip"},
		// IPv6 trusted hop + IPv4 client — confirms the matcher works
		// across families.
		{"ipv6 trusted hop, ipv4 client",
			"::1/128", "203.0.113.7", "[::1]:1234", "203.0.113.7"},
		// RemoteAddr unparseable, empty trust — preserve raw RemoteAddr
		// so we still bucket on *something*.
		{"unparseable remoteaddr",
			"", "", "weird", "weird"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			tp, bad := parseTrustedProxies(c.trust)
			if len(bad) > 0 {
				t.Fatalf("test trust spec %q invalid: %v", c.trust, bad)
			}
			s := &server{cfg: config{trustedProxies: tp}}
			r := httptest.NewRequest("GET", "/", nil)
			r.RemoteAddr = c.addr
			if c.xff != "" {
				r.Header.Set("X-Forwarded-For", c.xff)
			}
			if got := s.remoteIPForRateLimit(r); got != c.want {
				t.Errorf("got %q want %q", got, c.want)
			}
		})
	}
}

// parseTrustedProxies rejects garbage but accepts every reasonable
// CIDR/IP form.  Verifying this matters: a typo in
// DIST_TRUSTED_PROXIES silently drops the entry, and we want operators
// to discover that at boot (warn log) — but it also can't crash startup.
func TestParseTrustedProxies(t *testing.T) {
	cases := []struct {
		name    string
		spec    string
		want    int  // # of accepted nets
		wantBad int  // # of rejected entries
		empty   bool // expected empty() result
	}{
		{"empty string", "", 0, 0, true},
		{"single cidr", "10.0.0.0/8", 1, 0, false},
		{"bare ipv4 promoted to /32", "127.0.0.1", 1, 0, false},
		{"bare ipv6 promoted to /128", "::1", 1, 0, false},
		{"mixed list", "127.0.0.1, 10.0.0.0/8, ::1/128", 3, 0, false},
		{"trailing comma + spaces tolerated", " 10.0.0.0/8 , ", 1, 0, false},
		{"garbage entries flagged", "10.0.0.0/8,nope,/24", 1, 2, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			tp, bad := parseTrustedProxies(c.spec)
			if got := len(tp.nets); got != c.want {
				t.Errorf("nets: got %d want %d", got, c.want)
			}
			if got := len(bad); got != c.wantBad {
				t.Errorf("bad: got %d want %d (%v)", got, c.wantBad, bad)
			}
			if got := tp.empty(); got != c.empty {
				t.Errorf("empty: got %v want %v", got, c.empty)
			}
		})
	}
}
