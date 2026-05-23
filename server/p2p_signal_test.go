package main

import (
	"testing"
	"time"
)

func TestP2PSignal_AllowAndRevoke(t *testing.T) {
	allowP2PPair("rigA", "rigB", "sess-1", 0)

	if _, ok := p2pPermissionFor("rigA", "rigB"); !ok {
		t.Fatal("expected A->B allowed")
	}
	if _, ok := p2pPermissionFor("rigB", "rigA"); !ok {
		t.Fatal("expected B->A allowed (bidirectional)")
	}

	revokeP2PPair("rigA", "rigB")
	if _, ok := p2pPermissionFor("rigA", "rigB"); ok {
		t.Fatal("expected A->B revoked")
	}
	if _, ok := p2pPermissionFor("rigB", "rigA"); ok {
		t.Fatal("expected B->A revoked")
	}
}

func TestP2PSignal_TTLExpiry(t *testing.T) {
	allowP2PPair("ttl-A", "ttl-B", "s", 10*time.Millisecond)
	if _, ok := p2pPermissionFor("ttl-A", "ttl-B"); !ok {
		t.Fatal("expected fresh permission")
	}
	time.Sleep(25 * time.Millisecond)
	if _, ok := p2pPermissionFor("ttl-A", "ttl-B"); ok {
		t.Fatal("expected expired permission to be denied")
	}
	// Cleanup: revoke just in case.
	revokeP2PPair("ttl-A", "ttl-B")
}

func TestP2PSignal_DropsUnauthorised(t *testing.T) {
	// Ensure no permission exists.
	revokeP2PPair("unauth-A", "unauth-B")

	s := &server{hub: newHub()}
	msg := map[string]any{
		"kind":       "p2p_offer",
		"to":         "unauth-B",
		"session_id": "x",
		"sdp":        "v=0\r\n",
	}
	if !s.deliverP2PSignal(1, "unauth-A", msg) {
		t.Fatal("p2p_* frames should always be considered consumed")
	}
}

func TestP2PSignal_PassesThroughNonP2PKinds(t *testing.T) {
	s := &server{hub: newHub()}
	if s.deliverP2PSignal(1, "rigX", map[string]any{"kind": "status"}) {
		t.Fatal("non-p2p frames must not be consumed")
	}
	if s.deliverP2PSignal(1, "rigX", map[string]any{"kind": "comfy_result"}) {
		t.Fatal("non-p2p frames must not be consumed")
	}
}

func TestP2PSignal_RoutesToPeer(t *testing.T) {
	s := &server{hub: newHub()}

	// Register a fake peer in the hub.  agentConn.send is non-blocking on
	// a buffered outCh, so we can just inspect what landed there.
	peer := &agentConn{
		userID:  42,
		agentID: "rigDest",
		outCh:   make(chan any, 4),
		closed:  make(chan struct{}),
	}
	s.hub.mu.Lock()
	s.hub.agents = map[agentKey]*agentConn{
		{userID: 42, agentID: "rigDest"}: peer,
	}
	s.hub.mu.Unlock()

	allowP2PPair("rigSrc", "rigDest", "sess-x", 0)
	defer revokeP2PPair("rigSrc", "rigDest")

	msg := map[string]any{
		"kind":       "p2p_offer",
		"to":         "rigDest",
		"session_id": "sess-x",
		"sdp":        "v=0\r\n",
	}
	if !s.deliverP2PSignal(7, "rigSrc", msg) {
		t.Fatal("expected frame to be consumed")
	}

	select {
	case got := <-peer.outCh:
		m, ok := got.(map[string]any)
		if !ok {
			t.Fatalf("expected map, got %T", got)
		}
		if m["kind"] != "p2p_offer" {
			t.Errorf("kind: got %v", m["kind"])
		}
		if m["from"] != "rigSrc" {
			t.Errorf("from: got %v", m["from"])
		}
		if m["from_user"] != int64(7) {
			t.Errorf("from_user: got %v", m["from_user"])
		}
		if _, hasTo := m["to"]; hasTo {
			t.Errorf("'to' field should be stripped on the outgoing frame")
		}
		if m["sdp"] != "v=0\r\n" {
			t.Errorf("sdp body must pass through unchanged: got %v", m["sdp"])
		}
	case <-time.After(100 * time.Millisecond):
		t.Fatal("peer never received the relayed frame")
	}
}

func TestP2PSignal_DropsSelfTarget(t *testing.T) {
	s := &server{hub: newHub()}
	allowP2PPair("loop", "loop", "x", 0) // even if pathologically allowed
	defer revokeP2PPair("loop", "loop")

	if !s.deliverP2PSignal(1, "loop", map[string]any{
		"kind": "p2p_offer", "to": "loop",
	}) {
		t.Fatal("frame should still be consumed (and dropped)")
	}
}
