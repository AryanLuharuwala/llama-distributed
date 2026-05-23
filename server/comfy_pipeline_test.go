package main

// Tests covering the image-gen pipeline hardening:
//   - reapStaleComfyJobs: queued/running rows past the idle window get failed.
//   - countComfyJobs: gauges line up with table state.
//   - agentConn.close() fan-out: comfy subscribers see channel-closed, not hang.

import (
	"testing"
	"time"
)

// seedComfyRow inserts a comfy_jobs row at the given status / updated_at.
// Returns the inserted row id.
func seedComfyRow(t *testing.T, s *server, userID int64, status string, updatedAt int64) int64 {
	t.Helper()
	res, err := s.db.Exec(
		`INSERT INTO comfy_jobs (user_id, prompt, params_json, status, created_at, updated_at)
		 VALUES (?, '', '{}', ?, ?, ?)`,
		userID, status, updatedAt, updatedAt,
	)
	if err != nil {
		t.Fatalf("seed comfy_jobs: %v", err)
	}
	id, _ := res.LastInsertId()
	return id
}

// Stale queued + running rows must be reaped; fresh rows and terminal rows
// must survive untouched.
func TestReapStaleComfyJobs(t *testing.T) {
	s := newTestServer(t)
	uid, _ := makeUser(t, s, "reap-user")

	now := nowUnix()
	stale := now - 30*60 // 30 minutes ago
	fresh := now - 10    // 10 seconds ago

	idStaleQueued  := seedComfyRow(t, s, uid, "queued",    stale)
	idStaleRunning := seedComfyRow(t, s, uid, "running",   stale)
	idFreshQueued  := seedComfyRow(t, s, uid, "queued",    fresh)
	idDone         := seedComfyRow(t, s, uid, "done",      stale) // terminal, must not change
	idCancelled    := seedComfyRow(t, s, uid, "cancelled", stale) // terminal, must not change

	n, err := s.reapStaleComfyJobs(15 * 60) // 15-minute idle window
	if err != nil {
		t.Fatalf("reap: %v", err)
	}
	if n != 2 {
		t.Errorf("reaped: got %d, want 2 (stale queued + stale running)", n)
	}

	statusOf := func(id int64) string {
		var s2 string
		_ = s.db.QueryRow(`SELECT status FROM comfy_jobs WHERE id = ?`, id).Scan(&s2)
		return s2
	}
	if got := statusOf(idStaleQueued); got != "failed" {
		t.Errorf("stale queued: got status=%q, want failed", got)
	}
	if got := statusOf(idStaleRunning); got != "failed" {
		t.Errorf("stale running: got status=%q, want failed", got)
	}
	if got := statusOf(idFreshQueued); got != "queued" {
		t.Errorf("fresh queued: got status=%q, want queued (must survive)", got)
	}
	if got := statusOf(idDone); got != "done" {
		t.Errorf("done row: got status=%q, want done (terminal — must not change)", got)
	}
	if got := statusOf(idCancelled); got != "cancelled" {
		t.Errorf("cancelled row: got status=%q, want cancelled (terminal — must not change)", got)
	}
}

// countComfyJobs must reflect what /metrics will emit.
func TestCountComfyJobs(t *testing.T) {
	s := newTestServer(t)
	uid, _ := makeUser(t, s, "count-user")
	now := nowUnix()

	seedComfyRow(t, s, uid, "queued",    now)
	seedComfyRow(t, s, uid, "queued",    now)
	seedComfyRow(t, s, uid, "running",   now)
	seedComfyRow(t, s, uid, "done",      now)
	seedComfyRow(t, s, uid, "failed",    now)
	seedComfyRow(t, s, uid, "cancelled", now)

	// Active is the in-process inflight map; not touched by the seed.
	active, queued, totalFailed := s.countComfyJobs()
	if active != 0 {
		t.Errorf("active: got %d, want 0 (no in-process jobs registered)", active)
	}
	if queued != 2 {
		t.Errorf("queued: got %d, want 2", queued)
	}
	if totalFailed != 2 { // failed + cancelled
		t.Errorf("totalFailed: got %d, want 2 (failed + cancelled)", totalFailed)
	}
}

// On agent disconnect, every comfy result channel must observe the close
// signal so dispatchComfyToAgent returns "rig disconnected" instead of
// blocking on the 10-minute timer.  This is the symmetric fix to the
// inferPeers fan-out for batched LLM inference.
func TestAgentClose_FansOutToComfyResults(t *testing.T) {
	ac := newSlotTestConn()
	ac.closed = make(chan struct{})

	// Subscribe a few comfy jobs.
	ch1 := ac.subscribeComfy(101)
	ch2 := ac.subscribeComfy(102)
	ch3 := ac.subscribeComfy(103)

	ac.close()

	// All three subscribers must observe channel-closed; the deliverComfyResult
	// path that produces these never sends on a closed channel because it
	// holds comfyMu around its send (see ws.go).
	for jobID, ch := range map[int64]chan comfyResultMsg{101: ch1, 102: ch2, 103: ch3} {
		select {
		case _, ok := <-ch:
			if ok {
				t.Errorf("job %d: chan returned a value instead of close", jobID)
			}
		case <-time.After(time.Second):
			t.Errorf("job %d: chan never closed after ac.close()", jobID)
		}
	}
	if len(ac.comfyResults) != 0 {
		t.Errorf("comfyResults map should be empty after close; got %d", len(ac.comfyResults))
	}
}
