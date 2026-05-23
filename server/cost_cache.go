package main

// Per-pool rig-cost cache.
//
// Background: every chat completion and image gen calls planPipeline /
// planDPP, which in turn calls rigCostsForPool — a JOIN across pools,
// pool_rigs, and rigs, followed by N hub lookups and per-conn status
// snapshots.  Under sustained load (e.g. 100 RPS into a 50-rig pool)
// that's hundreds of identical queries per second pounding the single
// SQLite writer.  The list changes on rig connect/disconnect, which is
// orders of magnitude slower than request RPS.
//
// Trade-off: a TTL of 2s means a rig that disconnected can linger in
// the plan for up to 2s.  The PP/DPP handler already double-checks
// each stage with hub.findAgent before dispatch and returns 503 if the
// rig is gone — so the worst case is one user retry, not a wedged
// request.  We do NOT eagerly invalidate on rig events because the
// hub register/unregister paths are themselves hot, and a 2s drift is
// well within the noise of the live status snapshot they observe.
//
// Singleflight semantics: when many requests miss the cache at the
// same instant, only one of them runs the fill function; the rest
// block on a channel and share the result.  Without this, the cache
// itself would amplify the thundering herd it's trying to prevent.

import (
	"sync"
	"time"
)

// rigCostCache holds at most one entry per pool.  Entries are valid for
// `ttl` after their `at` timestamp; stale entries are replaced under the
// mutex, so only one filler runs per stale window.
type rigCostCache struct {
	mu  sync.Mutex
	m   map[int64]*rigCostCacheEntry
	ttl time.Duration
	now func() time.Time // injectable for tests
}

type rigCostCacheEntry struct {
	at    time.Time
	costs []rigCost
	err   error
	ready chan struct{} // closed by the filler once at/costs/err are settled
}

func newRigCostCache(ttl time.Duration) *rigCostCache {
	return &rigCostCache{
		m:   make(map[int64]*rigCostCacheEntry),
		ttl: ttl,
		now: time.Now,
	}
}

// getOrFill returns a cached value if fresh, otherwise calls fill exactly
// once across concurrent callers and caches the result.  An error in
// fill is also cached (briefly) so a flapping DB doesn't get hammered.
func (c *rigCostCache) getOrFill(poolID int64, fill func() ([]rigCost, error)) ([]rigCost, error) {
	c.mu.Lock()
	if e, ok := c.m[poolID]; ok {
		select {
		case <-e.ready:
			// Settled — check freshness.
			if c.now().Sub(e.at) <= c.ttl {
				c.mu.Unlock()
				return e.costs, e.err
			}
			// Stale — fall through to replace under the same lock.
		default:
			// Fill in progress; join the wait outside the lock.
			c.mu.Unlock()
			<-e.ready
			return e.costs, e.err
		}
	}
	e := &rigCostCacheEntry{ready: make(chan struct{})}
	c.m[poolID] = e
	c.mu.Unlock()

	costs, err := fill()

	c.mu.Lock()
	e.at = c.now()
	e.costs = costs
	e.err = err
	c.mu.Unlock()
	close(e.ready)

	return costs, err
}

// invalidate drops the cached entry for poolID, forcing the next call to
// refill from the DB.  Safe to call concurrently with getOrFill — a fill
// already in flight will still settle and serve any waiters, but the
// next caller will rebuild.
func (c *rigCostCache) invalidate(poolID int64) {
	c.mu.Lock()
	delete(c.m, poolID)
	c.mu.Unlock()
}

// invalidateAll clears every pool entry.  Cheap (just resets the map)
// and intended for "something big changed" signals — schema migrate,
// shutdown drain, etc.  Routine rig disconnects do not call this.
func (c *rigCostCache) invalidateAll() {
	c.mu.Lock()
	c.m = make(map[int64]*rigCostCacheEntry)
	c.mu.Unlock()
}
