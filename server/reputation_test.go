package main

import (
	"testing"
)

// relayScore math properties — keeps the scoring formula honest so we
// don't accidentally regress on (a) Bayesian smoothing penalising new
// rigs too hard, (b) recent failure decay being aggressive enough, and
// (c) a perfect track record outranking an unproven one.
func TestRelayScore_Math(t *testing.T) {
	now := int64(1_000_000)

	// Brand-new rig should sit at ~0.4 (high enough to get its first try,
	// below an established performer).
	newcomer := relayScore(rigReputation{}, now)
	if newcomer < 0.35 || newcomer > 0.45 {
		t.Errorf("newcomer score should be ~0.4, got %.3f", newcomer)
	}

	// Established perfect rig should be ranked higher than newcomer.
	veteran := relayScore(rigReputation{
		RelaySessionsTotal:   50,
		RelaySessionsSuccess: 50,
	}, now)
	if veteran <= newcomer {
		t.Errorf("veteran (%.3f) should outrank newcomer (%.3f)", veteran, newcomer)
	}

	// Rig with a recent failure (10s ago) should be penalised hard
	// regardless of lifetime success rate.
	justFailed := relayScore(rigReputation{
		RelaySessionsTotal:   50,
		RelaySessionsSuccess: 49,
		RelaySessionsFailed:  1,
		LastFailureAt:        now - 10,
	}, now)
	if justFailed > veteran*0.3 {
		t.Errorf("recent failure should heavily penalise; veteran=%.3f justFailed=%.3f", veteran, justFailed)
	}

	// 6 minutes after the failure, recency should be back to ~full.
	pastFailure := relayScore(rigReputation{
		RelaySessionsTotal:   50,
		RelaySessionsSuccess: 49,
		RelaySessionsFailed:  1,
		LastFailureAt:        now - 6*60,
	}, now)
	if pastFailure < veteran*0.9 {
		t.Errorf("6min-old failure should mostly decay; veteran=%.3f past=%.3f", veteran, pastFailure)
	}

	// A rig that's failed everything should be near zero.
	allFail := relayScore(rigReputation{
		RelaySessionsTotal:  20,
		RelaySessionsFailed: 20,
		LastFailureAt:       now - 6*60, // outside the recency window
	}, now)
	if allFail > 0.1 {
		t.Errorf("all-fail rig should score near zero, got %.3f", allFail)
	}
}

// recordRelaySuccess + recordRelayFailure persist counters that loadReputation
// can read back.  Exercises the SQLite UPSERT path.
func TestReputationRoundTrip(t *testing.T) {
	s := newTestServer(t)

	const agent = "rig-alpha"

	// Initial load — zero row.
	r := s.loadReputation(agent)
	if r.RelaySessionsTotal != 0 || r.RelaySessionsSuccess != 0 || r.RelaySessionsFailed != 0 {
		t.Fatalf("fresh rig should have zeroed counters, got %+v", r)
	}

	// Two successes + one failure → 3 total, 2 success, 1 failure, bytes=300.
	s.recordRelaySuccess(agent, 100)
	s.recordRelaySuccess(agent, 200)
	s.recordRelayFailure(agent)
	r = s.loadReputation(agent)
	if r.RelaySessionsTotal != 3 {
		t.Errorf("total: want 3, got %d", r.RelaySessionsTotal)
	}
	if r.RelaySessionsSuccess != 2 {
		t.Errorf("success: want 2, got %d", r.RelaySessionsSuccess)
	}
	if r.RelaySessionsFailed != 1 {
		t.Errorf("failed: want 1, got %d", r.RelaySessionsFailed)
	}
	if r.RelayBytesForwarded != 300 {
		t.Errorf("bytes: want 300, got %d", r.RelayBytesForwarded)
	}
	if r.LastSuccessAt == 0 {
		t.Error("LastSuccessAt should be set")
	}
	if r.LastFailureAt == 0 {
		t.Error("LastFailureAt should be set")
	}
}

// allReputations bulk-loads correctly; unknown agents are absent (not zero
// rows) so the caller can distinguish "never scored" from "all zeros".
func TestAllReputationsBulk(t *testing.T) {
	s := newTestServer(t)

	s.recordRelaySuccess("rig-a", 1000)
	s.recordRelayFailure("rig-b")
	// rig-c is unmentioned

	reps := s.allReputations([]string{"rig-a", "rig-b", "rig-c"})
	if _, ok := reps["rig-a"]; !ok {
		t.Error("rig-a should be present")
	}
	if _, ok := reps["rig-b"]; !ok {
		t.Error("rig-b should be present")
	}
	if _, ok := reps["rig-c"]; ok {
		t.Error("rig-c was never recorded — should be absent from bulk load")
	}
	if reps["rig-a"].RelaySessionsSuccess != 1 {
		t.Errorf("rig-a success: want 1, got %d", reps["rig-a"].RelaySessionsSuccess)
	}
	if reps["rig-b"].RelaySessionsFailed != 1 {
		t.Errorf("rig-b failed: want 1, got %d", reps["rig-b"].RelaySessionsFailed)
	}
}

// activeRelays add → remove → drainForAgent lifecycle.
func TestActiveRelaysLifecycle(t *testing.T) {
	ar := newActiveRelays()

	a1 := &relayAssignment{SessionID: "s1", AgentID: "rig-a", LeftPeer: "x", RightPeer: "y"}
	a2 := &relayAssignment{SessionID: "s2", AgentID: "rig-a", LeftPeer: "x", RightPeer: "z"}
	a3 := &relayAssignment{SessionID: "s3", AgentID: "rig-b", LeftPeer: "p", RightPeer: "q"}

	ar.add(a1)
	ar.add(a2)
	ar.add(a3)

	// Targeted remove pops one but leaves the rest.
	if got := ar.remove("s1", "rig-a"); got != a1 {
		t.Fatalf("remove s1/rig-a returned %v", got)
	}
	if got := ar.remove("s1", "rig-a"); got != nil {
		t.Errorf("second remove should be nil, got %v", got)
	}

	// drainForAgent pulls remaining rig-a sessions, leaves rig-b alone.
	drained := ar.drainForAgent("rig-a")
	if len(drained) != 1 || drained[0] != a2 {
		t.Errorf("drainForAgent(rig-a): want [a2], got %v", drained)
	}

	// rig-b still there.
	drained = ar.drainForAgent("rig-b")
	if len(drained) != 1 || drained[0] != a3 {
		t.Errorf("drainForAgent(rig-b): want [a3], got %v", drained)
	}

	// Double-drain is a safe no-op.
	if drained = ar.drainForAgent("rig-a"); len(drained) != 0 {
		t.Errorf("double-drain should be empty, got %v", drained)
	}
}

// reapStale evicts assignments older than the threshold and leaves fresh
// ones alone.  Both map sides (byKey, byAgent) must stay consistent.
func TestActiveRelaysReapStale(t *testing.T) {
	ar := newActiveRelays()
	now := int64(1_000_000)

	stale := &relayAssignment{SessionID: "s-old", AgentID: "rig-a", StartedAt: now - 4000}
	fresh := &relayAssignment{SessionID: "s-new", AgentID: "rig-a", StartedAt: now - 60}
	other := &relayAssignment{SessionID: "s-b", AgentID: "rig-b", StartedAt: now - 9999}

	ar.add(stale)
	ar.add(fresh)
	ar.add(other)

	got := ar.reapStale(now, 3600)
	if len(got) != 2 {
		t.Fatalf("expected 2 stale, got %d: %+v", len(got), got)
	}
	// Fresh assignment must still be there.
	if d := ar.drainForAgent("rig-a"); len(d) != 1 || d[0] != fresh {
		t.Errorf("fresh assignment lost; drained = %v", d)
	}
	// rig-b should be fully gone (both sides of the map).
	if d := ar.drainForAgent("rig-b"); len(d) != 0 {
		t.Errorf("rig-b should be empty after reap; got %v", d)
	}
}

// pruneIdleReputation deletes rows older than the cutoff but keeps fresh ones.
func TestPruneIdleReputation(t *testing.T) {
	s := newTestServer(t)

	// Fresh-touched row (just now).
	s.recordRelaySuccess("rig-fresh", 100)

	// Stale row: force updated_at into the past.
	s.recordRelaySuccess("rig-stale", 100)
	if _, err := s.db.Exec(`UPDATE rig_reputation SET updated_at = ? WHERE agent_id = 'rig-stale'`,
		nowUnix()-int64(40*24*3600)); err != nil {
		t.Fatalf("seed stale row: %v", err)
	}

	n, err := s.pruneIdleReputation(int64(30 * 24 * 3600))
	if err != nil {
		t.Fatalf("prune: %v", err)
	}
	if n != 1 {
		t.Errorf("expected 1 pruned row, got %d", n)
	}
	// Fresh row should survive.
	if r := s.loadReputation("rig-fresh"); r.RelaySessionsSuccess != 1 {
		t.Errorf("fresh row should survive prune; got %+v", r)
	}
	// Stale row should be gone.
	if r := s.loadReputation("rig-stale"); r.RelaySessionsSuccess != 0 {
		t.Errorf("stale row should be deleted; got %+v", r)
	}
}

// recordRelaySuccess increments byte counter cumulatively across calls.
func TestReputationByteAccumulation(t *testing.T) {
	s := newTestServer(t)

	s.recordRelaySuccess("rig-x", 1024)
	s.recordRelaySuccess("rig-x", 2048)
	s.recordRelaySuccess("rig-x", 0) // zero-byte session is still a session

	r := s.loadReputation("rig-x")
	if r.RelayBytesForwarded != 3072 {
		t.Errorf("bytes: want 3072, got %d", r.RelayBytesForwarded)
	}
	if r.RelaySessionsTotal != 3 || r.RelaySessionsSuccess != 3 {
		t.Errorf("counters: want total=3 success=3, got total=%d success=%d",
			r.RelaySessionsTotal, r.RelaySessionsSuccess)
	}
}
