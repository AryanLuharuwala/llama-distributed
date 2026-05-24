package main

// Cross-instance rate-limit backend.
//
// Why: every dist-server replica behind envoy has its own in-process
// sync.Map of per-IP token-bucket limiters.  Across N replicas an
// attacker effectively gets N× the configured budget — the limiter is
// only as protective as the smallest replica's view.  The fix is a
// shared backend; this file is that backend.
//
// Algorithm: GCRA (Generic Cell Rate Algorithm), the same shape
// rate.Limiter uses locally, implemented as a Lua script that runs
// atomically on the Redis side.  One round-trip per allow() call, with
// the bucket state encoded as (tokens, last-refill-ts) in a Redis hash.
//
// Behavior contract:
//   - Empty key ⇒ fail open (matches ipRateBucket.allow with empty IP).
//   - Redis error ⇒ fail open, with a rate-limited error log.  We never
//     wedge real traffic because the side-channel is sick; an attacker
//     killing Redis would otherwise be a DoS amplifier.
//   - Key prefix is "rl:" + bucket name + ":" + ip so the same physical
//     Redis can host multiple environments without collision (pair with
//     go-redis DB index for stronger isolation).
//
// Fallback: if DIST_REDIS_URL is unset the backend is nil and
// ipRateBucket uses its local sync.Map — single-container behavior is
// unchanged.

import (
	"context"
	"errors"
	"fmt"
	"log"
	"sync/atomic"
	"time"

	"github.com/redis/go-redis/v9"
)

// rateBackend is the optional cross-instance store consulted by
// ipRateBucket.allow().  Implementations must be safe for concurrent
// use.  Returning (false, nil) means "rate-limited"; (true, nil) means
// "allowed"; any non-nil error makes the caller fail open.
type rateBackend interface {
	allow(ctx context.Context, key string, perSec float64, burst int) (bool, error)
	close() error
}

// gcraScript is the atomic check-and-decrement.  Inputs:
//
//	KEYS[1]   = bucket key (rl:<name>:<ip>)
//	ARGV[1]   = rate (tokens per second; may be fractional)
//	ARGV[2]   = burst (max tokens; integer)
//	ARGV[3]   = now (unix seconds, may be fractional)
//	ARGV[4]   = cost (1.0 typical)
//
// Returns 1 if the request was allowed, 0 if denied.
//
// State is stored as a hash with fields {tokens, ts}.  We compute the
// refilled token count from (now-ts)*rate, cap at burst, and either
// decrement by cost or refuse.  EXPIRE is set to ⌈burst/rate⌉+1 so an
// inactive key drops out of memory on its own.
const gcraScript = `
local key = KEYS[1]
local rate = tonumber(ARGV[1])
local burst = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])

local data = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(data[1])
local ts = tonumber(data[2])

if tokens == nil then
    tokens = burst
    ts = now
end

local delta = now - ts
if delta < 0 then delta = 0 end
tokens = tokens + delta * rate
if tokens > burst then tokens = burst end

local allowed = 0
if tokens >= cost then
    tokens = tokens - cost
    allowed = 1
end

redis.call('HMSET', key, 'tokens', tokens, 'ts', now)

local ttl
if rate > 0 then
    ttl = math.ceil(burst / rate) + 1
else
    ttl = 3600
end
redis.call('EXPIRE', key, ttl)

return allowed
`

// redisRateBackend implements rateBackend against a single go-redis
// client.  EvalSha is preferred — the script SHA is cached at
// construction; on NOSCRIPT we transparently re-EVAL and refresh.
type redisRateBackend struct {
	rdb    *redis.Client
	script *redis.Script

	// Atomic counters surfaced via /metrics so an operator can see
	// fail-open events without having to grep logs.
	denies   atomic.Int64
	allows   atomic.Int64
	errors   atomic.Int64
	lastErrT atomic.Int64 // unix-seconds; throttles error logging
}

// newRedisRateBackend dials the URL (redis://, rediss://, or unix://)
// and verifies the connection with a PING.  Returns an error if the
// initial probe fails — the operator should see that at boot rather
// than discover later that every limiter is silently fail-open.
func newRedisRateBackend(ctx context.Context, url string) (*redisRateBackend, error) {
	if url == "" {
		return nil, errors.New("empty redis URL")
	}
	opts, err := redis.ParseURL(url)
	if err != nil {
		return nil, fmt.Errorf("parse %q: %w", url, err)
	}
	// Tight timeouts: rate-limit checks live on the request hot path; we
	// would rather fail open than queue requests behind a slow Redis.
	if opts.DialTimeout == 0 {
		opts.DialTimeout = 500 * time.Millisecond
	}
	if opts.ReadTimeout == 0 {
		opts.ReadTimeout = 250 * time.Millisecond
	}
	if opts.WriteTimeout == 0 {
		opts.WriteTimeout = 250 * time.Millisecond
	}
	rdb := redis.NewClient(opts)
	pingCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	if err := rdb.Ping(pingCtx).Err(); err != nil {
		_ = rdb.Close()
		return nil, fmt.Errorf("ping: %w", err)
	}
	return &redisRateBackend{
		rdb:    rdb,
		script: redis.NewScript(gcraScript),
	}, nil
}

func (b *redisRateBackend) allow(ctx context.Context, key string, perSec float64, burst int) (bool, error) {
	if b == nil || b.rdb == nil {
		return true, nil
	}
	if key == "" {
		return true, nil
	}
	now := float64(time.Now().UnixNano()) / 1e9
	res, err := b.script.Run(ctx, b.rdb,
		[]string{"rl:" + key},
		perSec, burst, now, 1.0,
	).Int64()
	if err != nil {
		b.errors.Add(1)
		// Log at most once per second so a Redis outage doesn't fill
		// the log with millions of identical lines.
		now := time.Now().Unix()
		last := b.lastErrT.Load()
		if now-last >= 1 && b.lastErrT.CompareAndSwap(last, now) {
			log.Printf("redis ratelimit: %v (failing open)", err)
		}
		return true, err
	}
	if res == 1 {
		b.allows.Add(1)
		return true, nil
	}
	b.denies.Add(1)
	return false, nil
}

func (b *redisRateBackend) close() error {
	if b == nil || b.rdb == nil {
		return nil
	}
	return b.rdb.Close()
}
