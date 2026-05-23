package main

// Tests for the multi-slot inferPeer dispatch — the new code path that
// enables request batching on rigs that advertise MaxConcurrent > 1.

import (
	"sync"
	"sync/atomic"
	"testing"
)

// helper — fresh agentConn with the bare minimum to exercise slot logic.
func newSlotTestConn() *agentConn {
	return &agentConn{
		inferPeers: make(map[uint16]*inferPeer),
	}
}

// Default (legacy) capacity is 1.  Two acquires must serialise: the
// second fails until the first releases.
func TestAcquireInferSlot_LegacySingleSlot(t *testing.T) {
	ac := newSlotTestConn()
	ip1 := &inferPeer{reqID: 1, incoming: make(chan *inferChunk, 1), closed: make(chan struct{})}
	ip2 := &inferPeer{reqID: 2, incoming: make(chan *inferChunk, 1), closed: make(chan struct{})}

	if !ac.acquireInferSlot(ip1, 0) {
		t.Fatal("first acquire (cap=0 → 1) should succeed")
	}
	if ac.acquireInferSlot(ip2, 0) {
		t.Error("second acquire on legacy single-slot rig must fail")
	}
	ac.releaseInferSlot(ip1)
	if !ac.acquireInferSlot(ip2, 0) {
		t.Error("acquire after release should succeed")
	}
}

// MaxConcurrent > 1 admits multiple peers up to the cap.
func TestAcquireInferSlot_MultiSlot(t *testing.T) {
	ac := newSlotTestConn()
	peers := []*inferPeer{
		{reqID: 1, incoming: make(chan *inferChunk, 1), closed: make(chan struct{})},
		{reqID: 2, incoming: make(chan *inferChunk, 1), closed: make(chan struct{})},
		{reqID: 3, incoming: make(chan *inferChunk, 1), closed: make(chan struct{})},
		{reqID: 4, incoming: make(chan *inferChunk, 1), closed: make(chan struct{})},
	}
	for i := 0; i < 3; i++ {
		if !ac.acquireInferSlot(peers[i], 3) {
			t.Errorf("acquire %d should succeed within cap=3", i)
		}
	}
	if ac.acquireInferSlot(peers[3], 3) {
		t.Error("acquire beyond cap must fail")
	}
	// Releasing one frees room for another.
	ac.releaseInferSlot(peers[0])
	if !ac.acquireInferSlot(peers[3], 3) {
		t.Error("acquire after release should succeed")
	}
}

// Reusing the same reqID is rejected — a collision would silently lose
// chunks from the prior request because dispatch routes by reqID.
func TestAcquireInferSlot_RejectsDuplicateReqID(t *testing.T) {
	ac := newSlotTestConn()
	ip1 := &inferPeer{reqID: 42, incoming: make(chan *inferChunk, 1), closed: make(chan struct{})}
	ip2 := &inferPeer{reqID: 42, incoming: make(chan *inferChunk, 1), closed: make(chan struct{})}
	if !ac.acquireInferSlot(ip1, 4) {
		t.Fatal("first acquire should succeed")
	}
	if ac.acquireInferSlot(ip2, 4) {
		t.Error("duplicate reqID must be rejected")
	}
}

// A rig already bound as a relay/pp/dpp peer cannot accept inference.
func TestAcquireInferSlot_RejectedWhenRelayBound(t *testing.T) {
	ac := newSlotTestConn()
	ac.peer = &clientConn{} // pretend it's relaying
	ip := &inferPeer{reqID: 1, incoming: make(chan *inferChunk, 1), closed: make(chan struct{})}
	if ac.acquireInferSlot(ip, 4) {
		t.Error("relay-bound rig must refuse inference")
	}
}

// releaseInferSlot is idempotent — safe to call from a defer even after
// the dispatcher already removed the peer on a done chunk.
func TestReleaseInferSlot_Idempotent(t *testing.T) {
	ac := newSlotTestConn()
	ip := &inferPeer{reqID: 7, incoming: make(chan *inferChunk, 1), closed: make(chan struct{})}
	ac.acquireInferSlot(ip, 4)
	ac.releaseInferSlot(ip)
	ac.releaseInferSlot(ip) // no panic, no underflow
	if len(ac.inferPeers) != 0 {
		t.Errorf("map should be empty; got %d", len(ac.inferPeers))
	}
}

// Concurrent acquires from N goroutines against a cap of K must yield
// exactly min(N, K) winners and the rest must fail cleanly.
func TestAcquireInferSlot_ConcurrentAdmission(t *testing.T) {
	const (
		N = 64
		K = 8
	)
	ac := newSlotTestConn()
	var wins int32
	var wg sync.WaitGroup
	start := make(chan struct{})
	for i := 0; i < N; i++ {
		wg.Add(1)
		go func(id uint16) {
			defer wg.Done()
			ip := &inferPeer{reqID: id, incoming: make(chan *inferChunk, 1), closed: make(chan struct{})}
			<-start
			if ac.acquireInferSlot(ip, K) {
				atomic.AddInt32(&wins, 1)
			}
		}(uint16(i + 1))
	}
	close(start)
	wg.Wait()
	if wins != K {
		t.Errorf("with cap=%d and %d racers, want %d winners; got %d", K, N, K, wins)
	}
	if len(ac.inferPeers) != K {
		t.Errorf("map should hold %d peers; got %d", K, len(ac.inferPeers))
	}
}

// hasAnyInferPeer mirrors len(inferPeers) > 0 and is used by pp/dpp to
// detect a rig that's currently busy with /v1/chat traffic.
func TestHasAnyInferPeer(t *testing.T) {
	ac := newSlotTestConn()
	if ac.hasAnyInferPeer() {
		t.Error("fresh conn must not report any peer")
	}
	ip := &inferPeer{reqID: 1, incoming: make(chan *inferChunk, 1), closed: make(chan struct{})}
	ac.acquireInferSlot(ip, 2)
	if !ac.hasAnyInferPeer() {
		t.Error("after acquire, must report a peer")
	}
	ac.releaseInferSlot(ip)
	if ac.hasAnyInferPeer() {
		t.Error("after release, must be empty again")
	}
}

// On agent disconnect, every in-flight inferPeer must observe the close
// signal so its handler can fail fast instead of waiting on the 2-minute
// ctx timeout.  This regression-tests the fan-out we wired into
// agentConn.close().
func TestAgentClose_FansOutToInferPeers(t *testing.T) {
	ac := newSlotTestConn()
	ac.closed = make(chan struct{})

	peers := []*inferPeer{
		{reqID: 11, incoming: make(chan *inferChunk, 1), closed: make(chan struct{})},
		{reqID: 12, incoming: make(chan *inferChunk, 1), closed: make(chan struct{})},
		{reqID: 13, incoming: make(chan *inferChunk, 1), closed: make(chan struct{})},
	}
	for _, ip := range peers {
		if !ac.acquireInferSlot(ip, 4) {
			t.Fatalf("setup: acquire reqID=%d should succeed", ip.reqID)
		}
	}

	ac.close()

	// Map must be drained.
	if got := len(ac.inferPeers); got != 0 {
		t.Errorf("inferPeers should be empty after close; got %d", got)
	}
	// Every peer's incoming + closed channels must have been closed.  Reading
	// from a closed channel returns the zero value immediately; if any are
	// still open this select will block until the test deadline.
	for _, ip := range peers {
		select {
		case <-ip.closed:
		default:
			t.Errorf("reqID=%d: closed chan not signalled", ip.reqID)
		}
		select {
		case _, ok := <-ip.incoming:
			if ok {
				t.Errorf("reqID=%d: incoming returned a value instead of close", ip.reqID)
			}
		default:
			t.Errorf("reqID=%d: incoming chan still open after close", ip.reqID)
		}
	}
}

// MaxConcurrent gets clamped at the validation layer — make sure absurd
// rig-advertised values can't blow up the slot map.
func TestValidation_MaxConcurrentClamped(t *testing.T) {
	cases := []struct {
		in, want int
	}{
		{0, 0},
		{1, 1},
		{8, 8},
		{maxAdvertisedConcurrent, maxAdvertisedConcurrent},
		{maxAdvertisedConcurrent + 1, 0}, // out of range → reset to 0 (legacy)
		{1_000_000, 0},
		{-1, 0},
	}
	for _, c := range cases {
		st := &agentStatus{MaxConcurrent: c.in}
		validateAndClampStatus(st, "1.2.3.4")
		if st.MaxConcurrent != c.want {
			t.Errorf("MaxConcurrent %d → %d; want %d", c.in, st.MaxConcurrent, c.want)
		}
	}
}
