package main

import (
	"testing"
)

// TestRelayBytesAggregateCap covers A12: a malicious rig that loops
// reporting per-session-capped byte totals must not be able to drive
// its lifetime relay_bytes_forwarded past maxAggregateRelayBytes.
func TestRelayBytesAggregateCap(t *testing.T) {
	s := newTestServer(t)
	const agentID = "rig-spammer"

	// Half-petabyte chunks would normally accumulate past the cap
	// after three iterations.  After the cap, additional updates
	// must keep the column pinned at maxAggregateRelayBytes.
	chunk := maxAggregateRelayBytes / 2 // 0.5 PiB
	for i := 0; i < 6; i++ {
		s.recordRelaySuccess(agentID, chunk)
	}

	var got int64
	if err := s.db.QueryRow(
		`SELECT relay_bytes_forwarded FROM rig_reputation WHERE agent_id = ?`,
		agentID,
	).Scan(&got); err != nil {
		t.Fatalf("read row: %v", err)
	}
	if got != maxAggregateRelayBytes {
		t.Errorf("relay_bytes_forwarded = %d, want exactly %d (cap)", got, maxAggregateRelayBytes)
	}

	// Another success report — should stay pinned, not roll over.
	s.recordRelaySuccess(agentID, 1<<30)
	if err := s.db.QueryRow(
		`SELECT relay_bytes_forwarded FROM rig_reputation WHERE agent_id = ?`,
		agentID,
	).Scan(&got); err != nil {
		t.Fatalf("read row 2: %v", err)
	}
	if got != maxAggregateRelayBytes {
		t.Errorf("after additional report relay_bytes_forwarded = %d, want %d (still capped)",
			got, maxAggregateRelayBytes)
	}

	// Session counter must keep climbing even though the byte counter is pinned.
	var sessions int64
	if err := s.db.QueryRow(
		`SELECT relay_sessions_success FROM rig_reputation WHERE agent_id = ?`,
		agentID,
	).Scan(&sessions); err != nil {
		t.Fatalf("read sessions: %v", err)
	}
	if sessions != 7 {
		t.Errorf("relay_sessions_success = %d, want 7", sessions)
	}
}

// TestRelayBytesNegativeRejected makes sure a rig reporting a negative
// or extreme value can't underflow / overflow the column.  This is the
// belt half of belt-and-suspenders alongside the A4 session clamp.
func TestRelayBytesNegativeRejected(t *testing.T) {
	s := newTestServer(t)
	const agentID = "rig-malicious"

	// Establish a baseline of 100 bytes.
	s.recordRelaySuccess(agentID, 100)

	// Now report -10**18 — must be treated as 0, leaving the aggregate at 100.
	s.recordRelaySuccess(agentID, -(1 << 60))

	var got int64
	_ = s.db.QueryRow(
		`SELECT relay_bytes_forwarded FROM rig_reputation WHERE agent_id = ?`,
		agentID,
	).Scan(&got)
	if got != 100 {
		t.Errorf("negative report leaked through: got %d, want 100", got)
	}

	// And a wildly-overlarge report — must be normalized to the cap, not used as-is.
	s.recordRelaySuccess(agentID, 1<<60)
	_ = s.db.QueryRow(
		`SELECT relay_bytes_forwarded FROM rig_reputation WHERE agent_id = ?`,
		agentID,
	).Scan(&got)
	if got != maxAggregateRelayBytes {
		t.Errorf("over-cap report not normalized: got %d, want %d", got, maxAggregateRelayBytes)
	}
}
