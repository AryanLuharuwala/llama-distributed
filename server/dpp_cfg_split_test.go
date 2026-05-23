package main

// Tests for the CFG-split DPP planner:
//   - cfg_split=false produces the classic 3-stage TE→UNet→VAE layout.
//   - cfg_split=true expands UNet into unet_cond / unet_uncond halves.
//   - Each half names its sibling in Partner so the runtime can ship
//     intermediate noise between them.
//   - Partner is empty on non-split stages.
//   - Auto-pick (pickRigForRole) treats unet_cond/unet_uncond like unet
//     and prefers VRAM-rich rigs over throughput-rich ones.

import (
	"testing"
)

func TestPlanDPP_CFGSplit_ExplicitRoles(t *testing.T) {
	s := &server{}
	roles := map[string]string{
		"text_encoder": "rig-te",
		"unet_cond":    "rig-A",
		"unet_uncond":  "rig-B",
		"vae":          "rig-vae",
	}
	plan, err := s.planDPP(0, "sdxl", roles, map[string]any{"cfg_split": true})
	if err != nil {
		t.Fatalf("planDPP: %v", err)
	}
	if len(plan.Stages) != 4 {
		t.Fatalf("expected 4 stages with cfg_split, got %d", len(plan.Stages))
	}
	wantRoles := []string{"text_encoder", "unet_cond", "unet_uncond", "vae"}
	for i, st := range plan.Stages {
		if st.Role != wantRoles[i] {
			t.Errorf("stage %d: role=%q want %q", i, st.Role, wantRoles[i])
		}
	}
	// Partner pairing: unet_cond <-> unet_uncond.
	if plan.Stages[1].Partner != "rig-B" {
		t.Errorf("unet_cond.Partner=%q want rig-B", plan.Stages[1].Partner)
	}
	if plan.Stages[2].Partner != "rig-A" {
		t.Errorf("unet_uncond.Partner=%q want rig-A", plan.Stages[2].Partner)
	}
	// Non-UNet stages have no partner.
	if plan.Stages[0].Partner != "" || plan.Stages[3].Partner != "" {
		t.Errorf("non-UNet stages should have empty Partner: %+v", plan.Stages)
	}
}

func TestPlanDPP_NoSplit_ClassicLayout(t *testing.T) {
	s := &server{}
	roles := map[string]string{
		"text_encoder": "rig-te",
		"unet":         "rig-u",
		"vae":          "rig-vae",
	}
	plan, err := s.planDPP(0, "sdxl", roles, map[string]any{})
	if err != nil {
		t.Fatalf("planDPP: %v", err)
	}
	if len(plan.Stages) != 3 {
		t.Fatalf("expected 3 stages without cfg_split, got %d", len(plan.Stages))
	}
	for _, st := range plan.Stages {
		if st.Partner != "" {
			t.Errorf("non-split stage has Partner set: %+v", st)
		}
	}
}

func TestPlanDPP_CFGSplit_MissingRoleErrors(t *testing.T) {
	s := newTestServer(t)
	// roles missing unet_uncond and pool 0 has no rigs, so auto-pick fails.
	roles := map[string]string{
		"text_encoder": "rig-te",
		"unet_cond":    "rig-A",
		"vae":          "rig-vae",
	}
	_, err := s.planDPP(0, "sdxl", roles, map[string]any{"cfg_split": true})
	if err == nil {
		t.Fatal("expected error when unet_uncond is missing and no rigs are available")
	}
}

func TestPickRigForRole_CFGSplitHalvesPreferVRAM(t *testing.T) {
	rigs := []rigCost{
		{info: onlineRigInfo{agentID: "fast-small"}, vramTotal: 4 << 30, throughput: 200},
		{info: onlineRigInfo{agentID: "slow-large"}, vramTotal: 24 << 30, throughput: 10},
	}
	// Both halves should pick the VRAM-rich rig when the other half hasn't
	// been claimed yet — same scoring as plain "unet".
	for _, role := range []string{"unet_cond", "unet_uncond"} {
		pick, ok := pickRigForRole(rigs, role, map[agentKey]bool{})
		if !ok {
			t.Fatalf("%s: no pick", role)
		}
		if pick.info.agentID != "slow-large" {
			t.Errorf("%s should prefer VRAM; got %q", role, pick.info.agentID)
		}
	}
}

func TestPickRigForRole_CFGSplitHalvesPickDistinctRigs(t *testing.T) {
	// With two UNet-capable rigs and unet_cond already taken, unet_uncond
	// must pick the other rig.
	rigs := []rigCost{
		{info: onlineRigInfo{agentID: "A"}, vramTotal: 24 << 30, throughput: 50},
		{info: onlineRigInfo{agentID: "B"}, vramTotal: 16 << 30, throughput: 50},
	}
	taken := map[agentKey]bool{{0, "A"}: true}
	pick, ok := pickRigForRole(rigs, "unet_uncond", taken)
	if !ok {
		t.Fatal("expected a pick for unet_uncond")
	}
	if pick.info.agentID != "B" {
		t.Errorf("unet_uncond should land on the free rig 'B'; got %q", pick.info.agentID)
	}
}
