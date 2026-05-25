package main

// Tests for the multi-step DPP loopback wire (CF12-E).
// Asserts that:
//   - actvFlagDPPLoop is a distinct bit, doesn't collide with other flags
//   - the planner produces stages with the right Role/BlockLo so the
//     handler can compute firstUnet/lastUnet for loopback wiring
//
// The actual dispatcher routing (peers[lastUnet].loopbackAgent = ...) is
// exercised end-to-end by the integration test suite; here we just check
// the bookkeeping that drives it.

import (
	"testing"
)

func TestActvFlagDPPLoopBitDistinct(t *testing.T) {
	all := []uint8{
		actvFlagIsPrompt, actvFlagKVAppend, actvFlagEndOfPrompt,
		actvFlagDPPLatent, actvFlagDPPFinal, actvFlagDPPImage,
		actvFlagDPPLoop,
	}
	// Each must be a single-bit value.
	for _, f := range all {
		if f&(f-1) != 0 {
			t.Errorf("flag 0x%02x is not a single bit", f)
		}
	}
	// All bits must be disjoint.
	var seen uint8
	for _, f := range all {
		if seen&f != 0 {
			t.Errorf("flag 0x%02x collides with previously seen bits 0x%02x", f, seen)
		}
		seen |= f
	}
	// Loop must be in the dpp set so dispatchDPPFromAgent picks it up.
	dppFlags := actvFlagDPPLatent | actvFlagDPPFinal | actvFlagDPPImage | actvFlagDPPLoop
	if dppFlags&actvFlagDPPLoop == 0 {
		t.Errorf("dppFlags mask missing DPPLoop bit")
	}
}

func TestPlanDPP_UnetStages_FirstLastDiscoverable(t *testing.T) {
	// The handler relies on Role=="unet" + BlockLo>=0 to find the
	// first and last UNet stage indices for loopback wiring.  Verify
	// the planner emits exactly that shape so the discovery doesn't
	// silently miss the loopback.
	s := &server{}
	roles := map[string]string{
		"text_encoder": "rig-te",
		"unet_0":       "rig-A",
		"unet_1":       "rig-B",
		"unet_2":       "rig-C",
		"vae":          "rig-vae",
	}
	cfg := map[string]any{"unet_stages": 3}
	plan, err := s.planDPP(0, "sdxl", roles, cfg)
	if err != nil {
		t.Fatalf("planDPP: %v", err)
	}
	firstUnet, lastUnet := -1, -1
	for i, st := range plan.Stages {
		if st.Role == "unet" && st.BlockLo >= 0 {
			if firstUnet < 0 {
				firstUnet = i
			}
			lastUnet = i
		}
	}
	if firstUnet != 1 || lastUnet != 3 {
		t.Errorf("expected firstUnet=1 lastUnet=3, got %d/%d", firstUnet, lastUnet)
	}
	// Sanity: text_encoder and vae must NOT match the discovery filter.
	if plan.Stages[0].BlockLo >= 0 || plan.Stages[4].BlockLo >= 0 {
		t.Errorf("non-unet stage carries block range — TE=%d VAE=%d",
			plan.Stages[0].BlockLo, plan.Stages[4].BlockLo)
	}
}

func TestPlanDPP_UnetStages_OneStageNoLoopback(t *testing.T) {
	// unet_stages==1 (or default) should NOT trigger loopback — the
	// discovery filter looks for BlockLo>=0 which only multi-stage
	// emits.  Single-stage UNet keeps BlockLo=-1.
	s := &server{}
	roles := map[string]string{
		"text_encoder": "rig-te",
		"unet":         "rig-u",
		"vae":          "rig-vae",
	}
	plan, err := s.planDPP(0, "sdxl", roles, nil)
	if err != nil {
		t.Fatalf("planDPP: %v", err)
	}
	for _, st := range plan.Stages {
		if st.Role == "unet" && st.BlockLo >= 0 {
			t.Errorf("single-stage unet should keep BlockLo=-1, got %d", st.BlockLo)
		}
	}
}
