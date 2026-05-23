package main

import (
	"context"
	"testing"
)

func TestRoleHeld(t *testing.T) {
	if !roleHeld([]string{"text_encoder", "embed", "unet"}, "embed") {
		t.Error("roleHeld should match 'embed'")
	}
	if !roleHeld([]string{"EMBED"}, "embed") {
		t.Error("roleHeld should be case-insensitive")
	}
	if roleHeld(nil, "embed") {
		t.Error("nil roles should not match")
	}
	if roleHeld([]string{"unet"}, "embed") {
		t.Error("unrelated roles should not match")
	}
}

func TestModelHeld(t *testing.T) {
	if !modelHeld([]string{"BAAI/bge-large-en-v1.5", "other"}, "baai/bge-large-en-v1.5") {
		t.Error("modelHeld should be case-insensitive")
	}
	if modelHeld([]string{"openai/text-embedding-3"}, "BAAI/bge-large-en-v1.5") {
		t.Error("modelHeld should not false-positive")
	}
}

func TestRigEmbedFleetFallsBackWhenNoRig(t *testing.T) {
	s, _ := openMCPTestDB(t)
	s.hub = newHub()

	hash := newHashEmbedder(96)
	fleet := newRigEmbedFleet(s, "BAAI/bge-large-en-v1.5", 96, hash)

	if fleet.Dim() != 96 {
		t.Errorf("Dim=%d want 96", fleet.Dim())
	}
	if fleet.ModelID() != "BAAI/bge-large-en-v1.5" {
		t.Errorf("ModelID=%q want BAAI/...", fleet.ModelID())
	}

	vecs, err := fleet.Embed(context.Background(), []string{"hello"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vecs) != 1 || len(vecs[0]) != 96 {
		t.Fatalf("shape: %d × %d", len(vecs), len(vecs[0]))
	}

	stats := fleet.Stats()
	if stats.RigsAdvertised != 0 || stats.Fallbacks != 1 || stats.Requests != 1 {
		t.Errorf("stats=%+v want Requests=1 Fallbacks=1 Rigs=0", stats)
	}
}

func TestRigEmbedFleetStatsCountsRigs(t *testing.T) {
	s, _ := openMCPTestDB(t)
	s.hub = newHub()

	// Register two agents — one embed-capable, one not.
	a1 := &agentConn{agentID: "rig-1", userID: 1}
	a1.live = liveStatus{RolesHeld: []string{"embed", "unet"}, UpdatedAt: 100}
	a2 := &agentConn{agentID: "rig-2", userID: 1}
	a2.live = liveStatus{RolesHeld: []string{"unet"}, UpdatedAt: 50}
	s.hub.registerAgent(a1)
	s.hub.registerAgent(a2)

	fleet := newRigEmbedFleet(s, "", 0, newHashEmbedder(32))
	stats := fleet.Stats()
	if stats.RigsAdvertised != 1 {
		t.Errorf("expected 1 embed-capable rig, got %d", stats.RigsAdvertised)
	}

	id, ok := fleet.pickRig()
	if !ok || id != "rig-1" {
		t.Errorf("pickRig: id=%q ok=%v want rig-1, true", id, ok)
	}
}
