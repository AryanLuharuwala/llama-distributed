package main

// Race / concurrency tests for the system's "exactly-once" or
// "atomic-counter" paths.  These are the spots where a wrong locking
// choice would only surface under real production load; we exercise
// them with N concurrent goroutines and assert on the post-conditions.
//
// Run with `go test -race` to also catch unsynchronised writes.

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
)

// consumePairToken must be exactly-once even under concurrent consumers.
// SQLite's UPDATE … RETURNING is the only thing standing between this
// and a double-pair bug, so cover it with a worst-case race.
func TestPairTokenConcurrentConsume(t *testing.T) {
	s := newTestServer(t)
	uid, _ := makeUser(t, s, "race-user")

	// Seed one pair token directly so we don't need the auth handler.
	const token = "race-token-1234567890"
	_, err := s.db.Exec(
		`INSERT INTO pair_tokens (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)`,
		token, uid, nowUnix(), nowUnix()+300,
	)
	if err != nil {
		t.Fatalf("seed token: %v", err)
	}

	const N = 32
	var (
		wins int32
		fail int32
		wg   sync.WaitGroup
	)
	start := make(chan struct{})
	for i := 0; i < N; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			if _, _, err := s.consumePairToken(token); err == nil {
				atomic.AddInt32(&wins, 1)
			} else {
				atomic.AddInt32(&fail, 1)
			}
		}()
	}
	close(start)
	wg.Wait()

	if wins != 1 {
		t.Errorf("exactly one consumer must win; got %d (fail=%d)", wins, fail)
	}
	if int(wins)+int(fail) != N {
		t.Errorf("accounting: wins=%d fail=%d total=%d (want %d)", wins, fail, wins+fail, N)
	}
}

// recordRelaySuccess uses ON CONFLICT DO UPDATE; under contention the
// final counters must equal the number of inserts.  This catches any
// future regression where a refactor accidentally drops the atomicity
// (e.g., switching to a read-modify-write pattern).
func TestReputationConcurrentInserts(t *testing.T) {
	s := newTestServer(t)

	const (
		N        = 50
		bytesPer = int64(7)
	)
	var wg sync.WaitGroup
	start := make(chan struct{})
	for i := 0; i < N; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			s.recordRelaySuccess("rig-race", bytesPer)
		}()
	}
	close(start)
	wg.Wait()

	r := s.loadReputation("rig-race")
	if r.RelaySessionsTotal != int64(N) {
		t.Errorf("RelaySessionsTotal: got %d, want %d", r.RelaySessionsTotal, N)
	}
	if r.RelaySessionsSuccess != int64(N) {
		t.Errorf("RelaySessionsSuccess: got %d, want %d", r.RelaySessionsSuccess, N)
	}
	if r.RelayBytesForwarded != int64(N)*bytesPer {
		t.Errorf("RelayBytesForwarded: got %d, want %d", r.RelayBytesForwarded, int64(N)*bytesPer)
	}
}

// activeRelays.add + remove + drainForAgent under concurrent producers
// and consumers — verifies both byKey and byAgent stay consistent.
//
// We pump N producers (each adds an assignment), N/2 removers (each
// removes one), and one drainer at the end.  Post-conditions:
//   - producers + removers + drainer must account for exactly N adds
//   - no double-remove succeeds (remove returns nil on miss)
func TestActiveRelaysConcurrentAddRemove(t *testing.T) {
	ar := newActiveRelays()
	const N = 64

	assignments := make([]*relayAssignment, N)
	for i := 0; i < N; i++ {
		assignments[i] = &relayAssignment{
			SessionID: fmt.Sprintf("s-%d", i),
			AgentID:   "rig-race",
			StartedAt: 100,
		}
	}

	var wg sync.WaitGroup
	start := make(chan struct{})

	// Producers — all add.
	for i := 0; i < N; i++ {
		wg.Add(1)
		a := assignments[i]
		go func() {
			defer wg.Done()
			<-start
			ar.add(a)
		}()
	}

	// Removers — each tries to remove a specific assignment.  Half of
	// these will race with the producer; the remove must be a no-op
	// when called before add, then succeed once add lands.
	var removeWins int32
	for i := 0; i < N/2; i++ {
		wg.Add(1)
		a := assignments[i]
		go func() {
			defer wg.Done()
			<-start
			// Retry-until-success — if remove races ahead of add, it'll
			// return nil and we just try again.  Bounded retries so the
			// test can't hang on a real bug.  runtime.Gosched between
			// attempts so the producer goroutine actually gets a turn;
			// without it a tight spin can starve the scheduler on a
			// loaded machine and the producer never runs in time.
			for j := 0; j < 100000; j++ {
				if got := ar.remove(a.SessionID, a.AgentID); got == a {
					atomic.AddInt32(&removeWins, 1)
					return
				}
				runtime.Gosched()
			}
			t.Errorf("remove never succeeded for %s after producer", a.SessionID)
		}()
	}

	close(start)
	wg.Wait()

	if removeWins != N/2 {
		t.Errorf("removers: want %d wins, got %d", N/2, removeWins)
	}
	// Drainer pulls the remaining N/2.
	drained := ar.drainForAgent("rig-race")
	if len(drained) != N/2 {
		t.Errorf("drainForAgent: want %d remaining, got %d", N/2, len(drained))
	}
	// Double-drain is a no-op.
	if d := ar.drainForAgent("rig-race"); len(d) != 0 {
		t.Errorf("double-drain should be empty, got %d", len(d))
	}
}

// reserveRequestSlot is the hot-path gate that decides whether an
// inference call enters the system.  Under K concurrent racers with a
// per-minute cap of C, *exactly* C must win; anything else is either a
// race-condition admit (lets traffic past the cap) or a lost wakeup (a
// caller blocks forever).
func TestReserveRequestSlot_RespectsCapUnderConcurrency(t *testing.T) {
	s := newTestServer(t)
	uid, _ := makeUser(t, s, "rate-race-user")

	// Tighten the per-minute cap to a known number so we can assert on it.
	const cap = 5
	if _, err := s.db.Exec(
		`INSERT INTO rate_limits (user_id, req_per_min, tokens_per_month, updated_at)
		 VALUES (?, ?, ?, ?)
		 ON CONFLICT(user_id) DO UPDATE SET req_per_min=excluded.req_per_min`,
		uid, cap, 1_000_000_000, nowUnix(),
	); err != nil {
		t.Fatalf("seed rate_limits: %v", err)
	}

	const N = 64
	var (
		wins int32
		wg   sync.WaitGroup
	)
	start := make(chan struct{})
	for i := 0; i < N; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			if ok, _, _ := s.reserveRequestSlot(uid); ok {
				atomic.AddInt32(&wins, 1)
			}
		}()
	}
	close(start)
	wg.Wait()

	if wins != cap {
		t.Errorf("with cap=%d and %d racers, want exactly %d wins; got %d",
			cap, N, cap, wins)
	}
}

// refundRequestSlot composed with reserve must net out — if every winner
// also refunds, the counter returns to zero and a *fresh* burst of cap
// reservations must once again all succeed.
func TestReserveAndRefund_RoundTrip(t *testing.T) {
	s := newTestServer(t)
	uid, _ := makeUser(t, s, "rate-refund-user")
	const cap = 4
	_, _ = s.db.Exec(
		`INSERT INTO rate_limits (user_id, req_per_min, tokens_per_month, updated_at)
		 VALUES (?, ?, ?, ?)
		 ON CONFLICT(user_id) DO UPDATE SET req_per_min=excluded.req_per_min`,
		uid, cap, 1_000_000_000, nowUnix(),
	)
	// First wave: fill the bucket.
	for i := 0; i < cap; i++ {
		if ok, _, _ := s.reserveRequestSlot(uid); !ok {
			t.Fatalf("wave 1, req %d: want admit", i)
		}
	}
	// Next attempt must be denied.
	if ok, _, _ := s.reserveRequestSlot(uid); ok {
		t.Fatalf("post-fill admit must be denied")
	}
	// Refund all, then a new wave must again all succeed.
	for i := 0; i < cap; i++ {
		s.refundRequestSlot(uid)
	}
	for i := 0; i < cap; i++ {
		if ok, _, _ := s.reserveRequestSlot(uid); !ok {
			t.Fatalf("wave 2, req %d: want admit after refunds", i)
		}
	}
}

// comfyJobs.finish must be idempotent — calling it twice on the same id
// after register must not double-close the done channel (which would
// panic), and must not interfere with a parallel cancel().  This is the
// surface that proxies handler defers and websocket close callbacks.
func TestComfyJobs_FinishIsIdempotent(t *testing.T) {
	jobs := newComfyJobs()
	for i := int64(0); i < 32; i++ {
		_, cancel := context.WithCancel(context.Background())
		jobs.register(i, cancel)
	}

	// 32 ids × 4 concurrent finishers each — only the first finisher per id
	// should observe the in-flight job; the rest must be no-ops.  Any
	// panic (double-close) flunks the test.
	const finishersPerID = 4
	var wg sync.WaitGroup
	start := make(chan struct{})
	for id := int64(0); id < 32; id++ {
		for k := 0; k < finishersPerID; k++ {
			wg.Add(1)
			go func(jobID int64) {
				defer wg.Done()
				<-start
				jobs.finish(jobID)
			}(id)
		}
	}
	// Add a parallel cancel storm — must coexist with finish without
	// touching a deleted entry.
	for id := int64(0); id < 32; id++ {
		wg.Add(1)
		go func(jobID int64) {
			defer wg.Done()
			<-start
			_ = jobs.cancel(jobID)
		}(id)
	}
	close(start)
	wg.Wait()

	jobs.mu.Lock()
	defer jobs.mu.Unlock()
	if len(jobs.inflight) != 0 {
		t.Errorf("after finish-all, inflight should be empty; got %d", len(jobs.inflight))
	}
}

// reapStale must be safe to call while producers add new assignments —
// fresh ones must survive, stale ones must be reaped, regardless of
// interleaving.
func TestActiveRelaysConcurrentReapStale(t *testing.T) {
	ar := newActiveRelays()
	now := int64(1_000_000)
	const N = 50

	var wg sync.WaitGroup
	start := make(chan struct{})

	// Half are pre-seeded as stale, half added concurrently as fresh.
	for i := 0; i < N/2; i++ {
		ar.add(&relayAssignment{
			SessionID: fmt.Sprintf("stale-%d", i),
			AgentID:   fmt.Sprintf("rig-%d", i),
			StartedAt: now - 10_000,
		})
	}
	for i := 0; i < N/2; i++ {
		wg.Add(1)
		idx := i
		go func() {
			defer wg.Done()
			<-start
			ar.add(&relayAssignment{
				SessionID: fmt.Sprintf("fresh-%d", idx),
				AgentID:   fmt.Sprintf("rig-fresh-%d", idx),
				StartedAt: now - 10, // very recent
			})
		}()
	}

	// Reaper races with the producers.
	var reaped int32
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			n := len(ar.reapStale(now, 3600))
			atomic.AddInt32(&reaped, int32(n))
		}()
	}

	close(start)
	wg.Wait()

	// All stale must be gone.  Fresh must all survive.
	if int(reaped) != N/2 {
		t.Errorf("reaped: want %d, got %d", N/2, reaped)
	}
	// Verify no fresh assignment was touched.
	ar.mu.Lock()
	defer ar.mu.Unlock()
	if len(ar.byKey) != N/2 {
		t.Errorf("survivors: want %d fresh, got %d", N/2, len(ar.byKey))
	}
	for _, a := range ar.byKey {
		if now-a.StartedAt > 3600 {
			t.Errorf("stale survivor: %+v", a)
		}
	}
}
