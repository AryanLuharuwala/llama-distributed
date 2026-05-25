package main

// Tests for the dppProgressBroker — verifies fan-out, isolation across
// users, slow-subscriber non-blocking semantics, and clean unsubscribe.

import (
	"testing"
	"time"
)

func TestDPPProgressBroker_FanOut(t *testing.T) {
	b := newDPPProgressBroker()
	ch1 := b.Subscribe(42)
	ch2 := b.Subscribe(42)
	defer b.Unsubscribe(42, ch1)
	defer b.Unsubscribe(42, ch2)

	b.Publish(42, dppProgressEvent{ReqID: 7, Event: "enter", Role: "unet"})

	for i, ch := range []chan dppProgressEvent{ch1, ch2} {
		select {
		case ev := <-ch:
			if ev.ReqID != 7 || ev.Event != "enter" {
				t.Errorf("sub %d: got %+v", i, ev)
			}
		case <-time.After(time.Second):
			t.Errorf("sub %d: timeout", i)
		}
	}
}

func TestDPPProgressBroker_UserIsolation(t *testing.T) {
	b := newDPPProgressBroker()
	chA := b.Subscribe(1)
	chB := b.Subscribe(2)
	defer b.Unsubscribe(1, chA)
	defer b.Unsubscribe(2, chB)

	b.Publish(1, dppProgressEvent{ReqID: 100})
	select {
	case ev := <-chA:
		if ev.ReqID != 100 {
			t.Errorf("user-1 got %+v", ev)
		}
	case <-time.After(time.Second):
		t.Fatal("user-1 timeout")
	}
	// user 2 must NOT receive user 1's event.
	select {
	case ev := <-chB:
		t.Errorf("user-2 leaked event: %+v", ev)
	case <-time.After(50 * time.Millisecond):
	}
}

func TestDPPProgressBroker_SlowSubscriberNonBlocking(t *testing.T) {
	b := newDPPProgressBroker()
	ch := b.Subscribe(9)
	defer b.Unsubscribe(9, ch)

	// Fill the channel buffer (cap 32) plus one extra — extra must be
	// dropped, not block.  Run in a goroutine so the test fails the
	// timeout instead of hanging the harness if Publish does block.
	done := make(chan struct{})
	go func() {
		for i := 0; i < 64; i++ {
			b.Publish(9, dppProgressEvent{ReqID: uint16(i)})
		}
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("Publish blocked on full subscriber channel")
	}
	// Drain — should have exactly 32 events.
	n := 0
	for {
		select {
		case <-ch:
			n++
		default:
			goto done
		}
	}
done:
	if n != 32 {
		t.Errorf("drained %d events, want 32 (buffer cap)", n)
	}
}

func TestIngestDPPProgress_FieldMapping(t *testing.T) {
	s := &server{dppProgress: newDPPProgressBroker()}
	ch := s.dppProgress.Subscribe(1)
	defer s.dppProgress.Unsubscribe(1, ch)

	s.ingestDPPProgress(1, "rig-x", map[string]any{
		"req_id":      float64(42),
		"stage_idx":   float64(2),
		"role":        "unet",
		"block_lo":    float64(3),
		"block_hi":    float64(5),
		"step_idx":    float64(7),
		"total_steps": float64(30),
		"event":       "step",
		"msg":         "denoise",
	})
	select {
	case ev := <-ch:
		if ev.AgentID != "rig-x" || ev.ReqID != 42 || ev.StageIdx != 2 ||
			ev.Role != "unet" || ev.BlockLo != 3 || ev.BlockHi != 5 ||
			ev.StepIdx != 7 || ev.TotalSteps != 30 ||
			ev.Event != "step" || ev.Msg != "denoise" {
			t.Errorf("field mapping wrong: %+v", ev)
		}
		if ev.TimestampMs <= 0 {
			t.Errorf("ts not set: %d", ev.TimestampMs)
		}
	case <-time.After(time.Second):
		t.Fatal("timeout")
	}
}

func TestIngestDPPProgress_OptionalFieldsDefault(t *testing.T) {
	// A minimal frame from the agent (just kind + event) must still
	// produce a valid event with sentinel defaults for missing fields.
	s := &server{dppProgress: newDPPProgressBroker()}
	ch := s.dppProgress.Subscribe(1)
	defer s.dppProgress.Unsubscribe(1, ch)
	s.ingestDPPProgress(1, "rig-x", map[string]any{"event": "enter"})
	select {
	case ev := <-ch:
		if ev.Event != "enter" {
			t.Errorf("event=%q want enter", ev.Event)
		}
		if ev.BlockLo != -1 || ev.BlockHi != -1 || ev.StepIdx != -1 {
			t.Errorf("missing fields not defaulted: %+v", ev)
		}
	case <-time.After(time.Second):
		t.Fatal("timeout")
	}
}
