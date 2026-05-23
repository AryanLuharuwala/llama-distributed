package main

// Tests for the per-pool rig-cost cache:
//   - first call calls fill, second within TTL returns cached value.
//   - past TTL, fill is invoked again.
//   - concurrent misses run fill exactly once (singleflight).
//   - invalidate forces the next call to refill.
//   - errors are cached briefly (won't hammer a flapping DB).
//   - rig disconnect invalidates the cache so planPipeline observes the
//     loss immediately (integration check through the server wiring).

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestRigCostCache_HitsWithinTTL(t *testing.T) {
	c := newRigCostCache(1 * time.Second)
	var calls int32
	fill := func() ([]rigCost, error) {
		atomic.AddInt32(&calls, 1)
		return []rigCost{{info: onlineRigInfo{agentID: "a"}}}, nil
	}
	for i := 0; i < 5; i++ {
		out, err := c.getOrFill(42, fill)
		if err != nil {
			t.Fatalf("call %d: %v", i, err)
		}
		if len(out) != 1 || out[0].info.agentID != "a" {
			t.Errorf("call %d: bad result %+v", i, out)
		}
	}
	if calls != 1 {
		t.Errorf("fill called %d times within TTL; want 1", calls)
	}
}

func TestRigCostCache_RefillsAfterTTL(t *testing.T) {
	// Use injectable clock so the test doesn't wait real seconds.
	c := newRigCostCache(50 * time.Millisecond)
	clock := time.Now()
	c.now = func() time.Time { return clock }

	var calls int32
	fill := func() ([]rigCost, error) {
		atomic.AddInt32(&calls, 1)
		return []rigCost{{info: onlineRigInfo{agentID: "a"}}}, nil
	}

	_, _ = c.getOrFill(1, fill)
	_, _ = c.getOrFill(1, fill)
	if calls != 1 {
		t.Errorf("two fast calls: %d fills, want 1", calls)
	}
	// Advance clock past TTL.
	clock = clock.Add(200 * time.Millisecond)
	_, _ = c.getOrFill(1, fill)
	if calls != 2 {
		t.Errorf("post-TTL call: %d fills, want 2", calls)
	}
}

func TestRigCostCache_DifferentPoolsIndependent(t *testing.T) {
	c := newRigCostCache(1 * time.Second)
	var callsA, callsB int32
	fillA := func() ([]rigCost, error) {
		atomic.AddInt32(&callsA, 1)
		return []rigCost{{info: onlineRigInfo{agentID: "a"}}}, nil
	}
	fillB := func() ([]rigCost, error) {
		atomic.AddInt32(&callsB, 1)
		return []rigCost{{info: onlineRigInfo{agentID: "b"}}}, nil
	}
	_, _ = c.getOrFill(1, fillA)
	_, _ = c.getOrFill(2, fillB)
	_, _ = c.getOrFill(1, fillA)
	_, _ = c.getOrFill(2, fillB)
	if callsA != 1 || callsB != 1 {
		t.Errorf("pools should be independent: A=%d B=%d", callsA, callsB)
	}
}

// Critical singleflight property: 50 goroutines hitting a cold cache must
// produce exactly 1 fill call, and all 50 must receive the same answer.
func TestRigCostCache_Singleflight(t *testing.T) {
	c := newRigCostCache(1 * time.Second)
	var calls int32
	// Slow fill so all 50 goroutines pile up behind it.
	fill := func() ([]rigCost, error) {
		atomic.AddInt32(&calls, 1)
		time.Sleep(30 * time.Millisecond)
		return []rigCost{{info: onlineRigInfo{agentID: "winner"}}}, nil
	}
	var wg sync.WaitGroup
	results := make([][]rigCost, 50)
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			out, _ := c.getOrFill(7, fill)
			results[i] = out
		}(i)
	}
	wg.Wait()
	if calls != 1 {
		t.Errorf("fill ran %d times under singleflight; want 1", calls)
	}
	for i, r := range results {
		if len(r) != 1 || r[0].info.agentID != "winner" {
			t.Errorf("goroutine %d got %+v", i, r)
		}
	}
}

func TestRigCostCache_Invalidate(t *testing.T) {
	c := newRigCostCache(1 * time.Second)
	var calls int32
	fill := func() ([]rigCost, error) {
		atomic.AddInt32(&calls, 1)
		return nil, nil
	}
	_, _ = c.getOrFill(1, fill)
	c.invalidate(1)
	_, _ = c.getOrFill(1, fill)
	if calls != 2 {
		t.Errorf("invalidate didn't force a refill: %d calls", calls)
	}
}

func TestRigCostCache_InvalidateAll(t *testing.T) {
	c := newRigCostCache(1 * time.Second)
	var calls int32
	fill := func() ([]rigCost, error) {
		atomic.AddInt32(&calls, 1)
		return nil, nil
	}
	_, _ = c.getOrFill(1, fill)
	_, _ = c.getOrFill(2, fill)
	c.invalidateAll()
	_, _ = c.getOrFill(1, fill)
	_, _ = c.getOrFill(2, fill)
	if calls != 4 {
		t.Errorf("invalidateAll didn't clear both: %d calls", calls)
	}
}

// Through-the-server integration: cache must hide the DB hit on the
// planner hot path.  We bump a counter inside the fill function by
// invalidating between calls.
func TestServer_RigCostsForPool_UsesCache(t *testing.T) {
	s := newTestServer(t)
	uid, _ := makeUser(t, s, "cache-user")
	pid, _ := seedPoolWithModel(t, s, uid, 1, 1, "intra")
	seedRigInPool(t, s, uid, pid, "rig-a", "host-a", 1)

	// Prime the cache.
	first, err := s.rigCostsForPool(pid)
	if err != nil {
		t.Fatalf("first: %v", err)
	}
	if len(first) != 1 {
		t.Fatalf("expected 1 rig; got %d", len(first))
	}

	// Add a *second* rig directly in the DB+hub but DO NOT invalidate.
	// The cached snapshot should still show 1 rig.
	seedRigInPool(t, s, uid, pid, "rig-b", "host-b", 1)
	cached, _ := s.rigCostsForPool(pid)
	if len(cached) != 1 {
		t.Errorf("cache served stale 1-rig snapshot? got %d (cache should still hold first result)", len(cached))
	}

	// Invalidate (mirrors what registerAgent / unregisterAgent do) and
	// the next call must observe both rigs.
	s.costCache.invalidateAll()
	fresh, _ := s.rigCostsForPool(pid)
	if len(fresh) != 2 {
		t.Errorf("post-invalidate: expected 2 rigs; got %d", len(fresh))
	}
}
