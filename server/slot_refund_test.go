package main

import (
	"sync"
	"testing"
)

// TestSlotRefundDeferOnAbnormalReturn exercises the "consumed bool +
// defer" pattern used in handleInfer / handleOAIChat / handleInferPP /
// handleDPP to ensure abnormal returns (any path that fails to flip
// slotBilled = true) restore the reservation.
//
// Stress shape: 10k loops of reserve-then-abandon must net out to zero
// reservations — proving an attacker who repeatedly opens connections
// and drops them mid-stream cannot lock a user out of their quota.
func TestSlotRefundDeferOnAbnormalReturn(t *testing.T) {
	s := newTestServer(t)
	uid, _ := makeUser(t, s, "drop-abuser")

	const N = 10000
	var wg sync.WaitGroup
	for i := 0; i < N; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ok, _, _ := s.reserveRequestSlot(uid)
			if !ok {
				return
			}
			// Mimic the new handler shape: defer-refund unless billed.
			slotBilled := false
			defer func() {
				if !slotBilled {
					s.refundRequestSlot(uid)
				}
			}()
			// "Connection drops here" — no billing happens.
			_ = slotBilled
		}()
	}
	wg.Wait()

	policy := s.loadRateLimit(uid)
	snap := s.usageSnapshot(uid)
	if snap.ReqThisMinute > policy.ReqPerMin {
		t.Errorf("after %d connect-drop loops, ReqThisMinute=%d exceeds policy.ReqPerMin=%d "+
			"— refunds aren't firing on abnormal return", N, snap.ReqThisMinute, policy.ReqPerMin)
	}

	// A fresh legit request should still succeed (refunds netted out).
	if ok, _, _ := s.reserveRequestSlot(uid); !ok {
		t.Errorf("user got locked out by connect-drop loops; defer-refund isn't covering all paths")
	}
}

// TestSlotRefundConsumedWhenBilled mirrors the happy path: slotBilled
// = true must suppress the refund so completed requests get correctly
// debited from the rolling minute counter.
func TestSlotRefundConsumedWhenBilled(t *testing.T) {
	s := newTestServer(t)
	uid, _ := makeUser(t, s, "honest-user")

	// Bump policy so we can drive N billed requests in a row without
	// hitting the default 30 req/min cap.
	const N = 50
	if _, err := s.db.Exec(
		`INSERT INTO rate_limits (user_id, req_per_min, tokens_per_month, updated_at) VALUES (?, ?, ?, ?)`,
		uid, N+10, 1_000_000, nowUnix(),
	); err != nil {
		t.Fatalf("seed rate_limits: %v", err)
	}

	for i := 0; i < N; i++ {
		ok, _, _ := s.reserveRequestSlot(uid)
		if !ok {
			t.Fatalf("ran out of budget at i=%d before policy cap", i)
		}
		func() {
			slotBilled := false
			defer func() {
				if !slotBilled {
					s.refundRequestSlot(uid)
				}
			}()
			slotBilled = true
		}()
	}
	snap := s.usageSnapshot(uid)
	if snap.ReqThisMinute != N {
		t.Errorf("ReqThisMinute=%d after %d billed requests; expected %d (defer is over-refunding)",
			snap.ReqThisMinute, N, N)
	}
}
