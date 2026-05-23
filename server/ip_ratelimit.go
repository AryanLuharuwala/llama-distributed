package main

// Per-IP token-bucket limiter for endpoints that are reached before any
// user authentication has happened — login pages, device-code approve/
// poll, OAuth redirect, /api/agent/auth/* bearer-trade endpoints, WS
// hello/resume failures.
//
// Implementation: golang.org/x/time/rate.Limiter keyed by client IP,
// stored in a sync.Map with a periodic janitor that prunes IPs with no
// recent activity.  This is in-process — fine for the current single-
// container deployment; the moment we run >1 instance behind a load
// balancer we swap the backing store for Redis (P9, deferred per the
// single-container scope).
//
// Each named bucket (deviceApprove, devicePoll, helloFail, …) has its
// own per-IP limiter so a flood on one endpoint doesn't accidentally
// throttle another.  Buckets are configured in newIPRateLimiterSet
// with sensible defaults; operator overrides go through env vars
// surfaced in config.go.

import (
	"net"
	"net/http"
	"strings"
	"sync"
	"time"

	"golang.org/x/time/rate"
)

// ipLimBucket holds one rate.Limiter plus a last-seen timestamp for the
// janitor.  rate.Limiter is internally thread-safe; lastSeen uses
// atomic.Int64 to avoid a per-IP mutex on the hot path.
type ipLimBucket struct {
	lim      *rate.Limiter
	lastSeen atomicInt64
}

// atomicInt64 is a minimal wrapper so we don't have to import sync/atomic
// at every call site.  Holds unix-seconds.
type atomicInt64 struct {
	v sync.Mutex
	t int64
}

func (a *atomicInt64) set(t int64) { a.v.Lock(); a.t = t; a.v.Unlock() }
func (a *atomicInt64) get() int64  { a.v.Lock(); t := a.t; a.v.Unlock(); return t }

// ipRateBucket is one named per-IP rate limiter.  Each protected
// endpoint takes its own bucket so noise on one doesn't cost budget on
// another.
type ipRateBucket struct {
	r       rate.Limit
	b       int
	buckets sync.Map // map[string]*ipLimBucket
}

func newIPRateBucket(perSec rate.Limit, burst int) *ipRateBucket {
	return &ipRateBucket{r: perSec, b: burst}
}

// peek returns true if the limiter would currently allow a token, without
// consuming one.  Useful before doing work that might fail: peek first,
// do work, consume on failure via allow().  Returns true for the empty
// IP (caller responsible for extraction).
func (b *ipRateBucket) peek(ip string) bool {
	if ip == "" {
		return true
	}
	v, ok := b.buckets.Load(ip)
	if !ok {
		return true // no record yet — full bucket implied
	}
	bk := v.(*ipLimBucket)
	return bk.lim.Tokens() >= 1.0
}

func (b *ipRateBucket) allow(ip string) bool {
	if ip == "" {
		// No identifiable IP — fail open rather than wedge the request,
		// but the caller is responsible for ensuring the IP extraction
		// path is correct.  Tests cover the happy path.
		return true
	}
	now := time.Now().Unix()
	v, ok := b.buckets.Load(ip)
	if !ok {
		fresh := &ipLimBucket{lim: rate.NewLimiter(b.r, b.b)}
		fresh.lastSeen.set(now)
		actual, _ := b.buckets.LoadOrStore(ip, fresh)
		v = actual
	}
	bk := v.(*ipLimBucket)
	bk.lastSeen.set(now)
	return bk.lim.Allow()
}

// janitor prunes IPs that haven't been seen for `ttl`.  Run from a
// background goroutine started in newIPRateLimiterSet.
func (b *ipRateBucket) janitor(ttl time.Duration) {
	cutoff := time.Now().Add(-ttl).Unix()
	b.buckets.Range(func(k, v any) bool {
		if v.(*ipLimBucket).lastSeen.get() < cutoff {
			b.buckets.Delete(k)
		}
		return true
	})
}

// ipRateLimiterSet is the bag of all per-IP buckets the server uses.
// One instance is constructed in main and lives on `server`.
type ipRateLimiterSet struct {
	deviceApprove *ipRateBucket
	devicePoll    *ipRateBucket
	helloFail     *ipRateBucket
	oauthStart    *ipRateBucket
}

// newIPRateLimiterSet wires up the default policy and starts the
// janitor.  These rates are tuned for the single-container shape:
//   - device approve: a logged-in human approves; 10/min is plenty.
//   - device poll:    rigs poll every 2s; allow ~30/min headroom.
//   - hello fail:     bad auth retries; 5/min, burst 3 — slows
//                     credential-stuffing without locking out genuine
//                     bad-network reconnect storms.
//   - oauth start:    1/sec per IP; burst 3.
func newIPRateLimiterSet() *ipRateLimiterSet {
	set := &ipRateLimiterSet{
		deviceApprove: newIPRateBucket(rate.Every(6*time.Second), 3),
		devicePoll:    newIPRateBucket(rate.Every(2*time.Second), 10),
		helloFail:     newIPRateBucket(rate.Every(12*time.Second), 3),
		oauthStart:    newIPRateBucket(rate.Limit(1), 3),
	}
	go func() {
		t := time.NewTicker(5 * time.Minute)
		defer t.Stop()
		for range t.C {
			set.deviceApprove.janitor(30 * time.Minute)
			set.devicePoll.janitor(30 * time.Minute)
			set.helloFail.janitor(30 * time.Minute)
			set.oauthStart.janitor(30 * time.Minute)
		}
	}()
	return set
}

// remoteIPForRateLimit extracts a stable IP key from an HTTP request.
// Honors X-Forwarded-For when set (matches the existing clientIP
// helper in ws.go).  The trust-XFF caveat is documented in
// [[xff_trust_known_gap]] — safe behind a real reverse proxy, spoofable
// when the server is exposed directly to the public internet.
func remoteIPForRateLimit(r *http.Request) string {
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		if i := strings.IndexByte(xff, ','); i > 0 {
			return strings.TrimSpace(xff[:i])
		}
		return strings.TrimSpace(xff)
	}
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return host
}
