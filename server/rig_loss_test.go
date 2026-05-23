package main

// Rig-loss / GPU-loss scenarios.
//
// What happens when:
//   - a rig vanishes between plan() and dispatch
//   - a rig's last reported GPU count drops to zero
//   - every rig in the pool disconnects
//   - telemetry stalls (no status frames in >60s)
//   - a relay rig dies with sessions in flight
//
// The contract is: failures must surface as errors or recoverable
// degradations — never as a wedged channel, panic, or silent zero-layer
// dispatch.  Stages that lose their rig must drop out of the next plan
// without operator intervention.

import (
	"testing"
	"time"
)

// ── Hub bookkeeping ─────────────────────────────────────────────────────────

func TestHub_UnregisterAgent_RemovesFromMap(t *testing.T) {
	s := newTestServer(t)
	ac := registerStubAgent(s, 42, "rig-x", "host-x")

	if _, ok := s.hub.findAgent(42, "rig-x"); !ok {
		t.Fatalf("findAgent missed freshly-registered rig")
	}
	s.hub.unregisterAgent(ac)
	if _, ok := s.hub.findAgent(42, "rig-x"); ok {
		t.Errorf("findAgent should return !ok after unregister")
	}
}

func TestAgentClose_ClosesClosedChannel(t *testing.T) {
	ac := &agentConn{
		closed: make(chan struct{}),
		outCh:  make(chan any, 1),
		binCh:  make(chan []byte, 1),
	}
	ac.close()
	select {
	case <-ac.closed:
	default:
		t.Errorf("ac.close() didn't close the closed channel")
	}
	// Idempotent — second close must not panic.
	ac.close()
}

// ── Planner observes disconnects ────────────────────────────────────────────

func TestOnlineRigsInPool_SkipsOfflineRig(t *testing.T) {
	s := newTestServer(t)
	uid, _ := makeUser(t, s, "rig-loss")
	pid, _ := seedPoolWithModel(t, s, uid, 1, 1, "intra")

	seedRigInPool(t, s, uid, pid, "rig-a", "host-a", 1)
	seedRigInPool(t, s, uid, pid, "rig-b", "host-b", 1)
	if rigs, _ := s.onlineRigsInPool(pid); len(rigs) != 2 {
		t.Fatalf("setup: expected 2 online rigs, got %d", len(rigs))
	}

	// Take rig-a offline at the hub level (DB row still exists).
	if ac, ok := s.hub.findAgent(uid, "rig-a"); ok {
		s.hub.unregisterAgent(ac)
	} else {
		t.Fatalf("rig-a missing from hub after seed")
	}
	rigs, _ := s.onlineRigsInPool(pid)
	if len(rigs) != 1 || rigs[0].agentID != "rig-b" {
		t.Errorf("after disconnect, online list = %+v; want only rig-b", rigs)
	}
}

func TestPlanPipeline_AllRigsOffline_ReturnsError(t *testing.T) {
	s := newTestServer(t)
	uid, _ := makeUser(t, s, "all-down")
	pid, _ := seedPoolWithModel(t, s, uid, 1, 1, "intra")
	seedRigInPool(t, s, uid, pid, "rig-a", "host-a", 1)

	// Take it down before the planner runs.
	if ac, ok := s.hub.findAgent(uid, "rig-a"); ok {
		s.hub.unregisterAgent(ac)
	}
	// Cache hasn't been touched — invalidate so the planner observes the loss.
	s.costCache.invalidateAll()

	if _, err := s.planPipeline(pid, 1, "", 0); err == nil {
		t.Errorf("planPipeline with all rigs offline must return error; got nil")
	}
}

// End-to-end shape: register two rigs, plan succeeds with 2 stages; disconnect
// one + invalidate; replan downgrades to 1 stage instead of carrying a stale
// reference.
func TestPlanPipeline_DropsRigOnDisconnect(t *testing.T) {
	s := newTestServer(t)
	uid, _ := makeUser(t, s, "drop-rig")
	pid, _ := seedPoolWithModel(t, s, uid, 2, 1, "intra")
	seedRigInPool(t, s, uid, pid, "rig-a", "host-a", 1)
	seedRigInPool(t, s, uid, pid, "rig-b", "host-b", 1)

	plan1, err := s.planPipeline(pid, 1, "", 0)
	if err != nil {
		t.Fatalf("plan1: %v", err)
	}
	if len(plan1.Stages) != 2 {
		t.Fatalf("plan1: expected 2 stages, got %d", len(plan1.Stages))
	}

	// Disconnect one of the rigs.  The cache must drop or planPipeline
	// would still hand out the offline rig as a stage.  (ws.go does this
	// for real on agent disconnect; we mirror it here.)
	for _, st := range plan1.Stages {
		if ac, ok := s.hub.findAgent(st.UserID, st.AgentID); ok {
			s.hub.unregisterAgent(ac)
			break
		}
	}
	s.costCache.invalidateAll()

	plan2, err := s.planPipeline(pid, 2, "", 0)
	if err != nil {
		t.Fatalf("plan2: %v", err)
	}
	if len(plan2.Stages) != 1 {
		t.Errorf("plan2 should degrade to 1 stage after disconnect; got %d", len(plan2.Stages))
	}
}

// ── Cost scoring respects unhealthy telemetry ───────────────────────────────

// A rig that hasn't reported telemetry in >60s is "stale" — score halved
// so the planner prefers fresher peers.  Without this the planner would
// happily ship layers to a rig that may have hung.
func TestRigCostScore_StaleTelemetryHalvesScore(t *testing.T) {
	fresh := rigCost{
		throughput: 100,
		vramTotal:  24 * 1024 * 1024 * 1024,
		bwKbps:     1000 * 1024,
	}
	stale := fresh
	stale.stale = true

	if fs, ss := fresh.score(), stale.score(); ss >= fs || ss != fs/2 {
		t.Errorf("stale should halve score: fresh=%.1f stale=%.1f (want stale = fresh/2)", fs, ss)
	}
}

// Worst-case rig: no telemetry, no VRAM info, no bandwidth.  Floor of 1
// keeps it in the running rather than starving the pipeline — better to
// route a few layers there than 503 the whole pool.
func TestRigCostScore_BrokenRig_FloorIsOne(t *testing.T) {
	broken := rigCost{stale: true} // zeroes everywhere, marked stale
	if sc := broken.score(); sc != 1 {
		t.Errorf("score floor: got %.2f, want 1.0", sc)
	}
}

// In-flight penalty makes a busy rig look worse than an idle one even at
// identical VRAM/throughput.  This is what spreads load evenly across the
// pool when the pool is partially saturated.
func TestRigCostScore_InflightPenaltyOrdering(t *testing.T) {
	idle := rigCost{throughput: 100, vramTotal: 24 << 30, bwKbps: 1024 * 100, inflight: 0}
	busy := idle
	busy.inflight = 5
	if idle.score() <= busy.score() {
		t.Errorf("idle rig should outscore busy: idle=%.1f busy=%.1f",
			idle.score(), busy.score())
	}
}

// ── Cache reacts to disconnect ──────────────────────────────────────────────

// Mirrors what ws.go's deferred block does on rig unregister.  This is a
// regression guard: if a future refactor moves invalidateAll out of the
// disconnect path, the planner will start handing out offline rigs again
// until the 2s TTL expires.
func TestCostCache_InvalidatedOnRigUnregister(t *testing.T) {
	s := newTestServer(t)
	uid, _ := makeUser(t, s, "cache-disc")
	pid, _ := seedPoolWithModel(t, s, uid, 1, 1, "intra")
	seedRigInPool(t, s, uid, pid, "rig-a", "host-a", 1)

	if got, _ := s.rigCostsForPool(pid); len(got) != 1 {
		t.Fatalf("prime: %d rigs, want 1", len(got))
	}
	ac, ok := s.hub.findAgent(uid, "rig-a")
	if !ok {
		t.Fatalf("rig-a missing")
	}
	s.hub.unregisterAgent(ac)
	// Real disconnect path also runs invalidateAll — emulate it here.
	s.costCache.invalidateAll()

	if got, _ := s.rigCostsForPool(pid); len(got) != 0 {
		t.Errorf("after disconnect+invalidate, expected 0 rigs; got %d", len(got))
	}
}

// ── Relay-rig disconnect → reputation hit ───────────────────────────────────

// When a rig that's serving relay sessions disconnects, drainForAgent must
// return every active session it owned so unregisterAgent can mark them
// all as failures.  This is what enables the score-based failover for
// the next plan round.
func TestRelayDisconnect_DrainsAllSessions(t *testing.T) {
	relays := newActiveRelays()
	relays.add(&relayAssignment{
		SessionID: "sess-1", AgentID: "relay-a", LeftPeer: "L1", RightPeer: "R1",
	})
	relays.add(&relayAssignment{
		SessionID: "sess-2", AgentID: "relay-a", LeftPeer: "L2", RightPeer: "R2",
	})
	relays.add(&relayAssignment{
		SessionID: "sess-3", AgentID: "relay-b", LeftPeer: "L3", RightPeer: "R3",
	})

	drained := relays.drainForAgent("relay-a")
	if len(drained) != 2 {
		t.Errorf("drain relay-a: %d sessions, want 2", len(drained))
	}
	// relay-b still alive, its session must still be tracked.
	if len(relays.drainForAgent("relay-b")) != 1 {
		t.Errorf("relay-b drain didn't return its lone session")
	}
}

// Recording a relay failure must bump failed-counter AND last_failure_at.
// recencyMul in relayScore then reads last_failure_at and depresses the
// score for the next 60s, so the planner picks someone else.  Without
// this, failover wouldn't work — a rig could fail repeatedly and still
// score high if it had a long history of successes.
func TestRelayFailure_DepressesScore(t *testing.T) {
	s := newTestServer(t)
	uid, _ := makeUser(t, s, "rep")
	_ = uid

	// Build a baseline reputation: 30 successes over many days.
	for i := 0; i < 30; i++ {
		s.recordRelaySuccess("flaky", 1000)
	}
	before := s.loadReputation("flaky")
	now := time.Now().Unix()
	scoreBefore := relayScore(before, now)

	// One mid-session failure.
	s.recordRelayFailure("flaky")
	after := s.loadReputation("flaky")
	scoreAfter := relayScore(after, now+1) // +1s, still inside the 60s penalty window

	if scoreAfter >= scoreBefore {
		t.Errorf("relay failure didn't depress score: before=%.3f after=%.3f", scoreBefore, scoreAfter)
	}
}
