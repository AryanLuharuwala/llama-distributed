package main

// Planner-level tests for the multi-stage UNet split (CF12-A).
// Exercises planDPP's expansion of one logical "unet" role into N
// stages with disjoint block_lo/block_hi ranges drawn from the model
// family's known topology.

import (
	"strconv"
	"testing"
)

func TestPlanDPP_UnetStages_SDXL(t *testing.T) {
	s := &server{}
	roles := map[string]string{
		"text_encoder": "rig-te",
		"unet_0":       "rig-A",
		"unet_1":       "rig-B",
		"unet_2":       "rig-C",
		"vae":          "rig-vae",
	}
	cfg := map[string]any{"unet_stages": 3}
	plan, err := s.planDPP(0, "stabilityai/stable-diffusion-xl-base-1.0", roles, cfg)
	if err != nil {
		t.Fatalf("planDPP: %v", err)
	}
	// 1 text_encoder + 3 unet + 1 vae = 5 stages.
	if len(plan.Stages) != 5 {
		t.Fatalf("expected 5 stages, got %d", len(plan.Stages))
	}
	if plan.Stages[0].Role != "text_encoder" {
		t.Errorf("stage 0 role=%q want text_encoder", plan.Stages[0].Role)
	}
	if plan.Stages[4].Role != "vae" {
		t.Errorf("stage 4 role=%q want vae", plan.Stages[4].Role)
	}
	// All UNet stages surface as plain "unet" — runtime dispatches by
	// block_lo/block_hi, not by suffix.
	for i := 1; i <= 3; i++ {
		if plan.Stages[i].Role != "unet" {
			t.Errorf("stage %d role=%q want unet", i, plan.Stages[i].Role)
		}
	}
	// SDXL = 7 blocks → 3 stages → [3, 2, 2].
	wantRanges := [][2]int{{0, 3}, {3, 5}, {5, 7}}
	for i, want := range wantRanges {
		st := plan.Stages[i+1]
		if st.BlockLo != want[0] || st.BlockHi != want[1] {
			t.Errorf("unet stage %d: [%d,%d) want [%d,%d)",
				i, st.BlockLo, st.BlockHi, want[0], want[1])
		}
	}
}

func TestPlanDPP_UnetStages_Flux(t *testing.T) {
	s := &server{}
	roles := map[string]string{
		"text_encoder": "rig-te",
		"unet_0":       "rig-A",
		"unet_1":       "rig-B",
		"vae":          "rig-vae",
	}
	cfg := map[string]any{"unet_stages": 2}
	plan, err := s.planDPP(0, "black-forest-labs/FLUX.1-dev", roles, cfg)
	if err != nil {
		t.Fatalf("planDPP: %v", err)
	}
	if len(plan.Stages) != 4 {
		t.Fatalf("expected 4 stages, got %d", len(plan.Stages))
	}
	// Flux total = 57 blocks → 2 stages → [29, 28].
	if plan.Stages[1].BlockLo != 0 || plan.Stages[1].BlockHi != 29 {
		t.Errorf("flux stage 0: [%d,%d) want [0,29)",
			plan.Stages[1].BlockLo, plan.Stages[1].BlockHi)
	}
	if plan.Stages[2].BlockLo != 29 || plan.Stages[2].BlockHi != 57 {
		t.Errorf("flux stage 1: [%d,%d) want [29,57)",
			plan.Stages[2].BlockLo, plan.Stages[2].BlockHi)
	}
}

func TestPlanDPP_UnetStages_UnknownModelFallsBackToOne(t *testing.T) {
	s := &server{}
	// Unknown model — planner should silently degrade to a single-stage
	// UNet rather than refuse the request.
	roles := map[string]string{
		"text_encoder": "rig-te",
		"unet":         "rig-u",
		"vae":          "rig-vae",
	}
	cfg := map[string]any{"unet_stages": 4}
	plan, err := s.planDPP(0, "some-random-vendor/wild-diffuser", roles, cfg)
	if err != nil {
		t.Fatalf("planDPP: %v", err)
	}
	if len(plan.Stages) != 3 {
		t.Fatalf("expected 3 stages on fallback, got %d", len(plan.Stages))
	}
	if plan.Stages[1].BlockLo != -1 || plan.Stages[1].BlockHi != -1 {
		t.Errorf("fallback unet: block range should be (-1,-1), got [%d,%d)",
			plan.Stages[1].BlockLo, plan.Stages[1].BlockHi)
	}
}

func TestPlanDPP_UnetStages_RejectsCfgSplitMix(t *testing.T) {
	s := &server{}
	roles := map[string]string{
		"text_encoder": "rig-te",
		"unet_0":       "rig-A",
		"unet_1":       "rig-B",
		"vae":          "rig-vae",
	}
	cfg := map[string]any{"unet_stages": 2, "cfg_split": true}
	if _, err := s.planDPP(0, "sdxl", roles, cfg); err == nil {
		t.Errorf("expected error when mixing unet_stages with cfg_split")
	}
}

func TestPlanDPP_UnetStages_RejectsTooManyStages(t *testing.T) {
	s := &server{}
	roles := map[string]string{
		"text_encoder": "rig-te",
	}
	// SDXL has 7 blocks — asking for 8 stages must error.
	for i := 0; i < 8; i++ {
		roles["unet_"+strconv.Itoa(i)] = "rig-" + strconv.Itoa(i)
	}
	roles["vae"] = "rig-vae"
	cfg := map[string]any{"unet_stages": 8}
	if _, err := s.planDPP(0, "sdxl", roles, cfg); err == nil {
		t.Errorf("expected error when unet_stages exceeds block count")
	}
}

func TestPlanDPP_UnetStages_PartnerEmpty(t *testing.T) {
	// Multi-stage UNet should NOT set Partner on any stage — that's a
	// cfg_split concept only.
	s := &server{}
	roles := map[string]string{
		"text_encoder": "rig-te",
		"unet_0":       "rig-A",
		"unet_1":       "rig-B",
		"vae":          "rig-vae",
	}
	cfg := map[string]any{"unet_stages": 2}
	plan, err := s.planDPP(0, "sdxl", roles, cfg)
	if err != nil {
		t.Fatalf("planDPP: %v", err)
	}
	for i, st := range plan.Stages {
		if st.Partner != "" {
			t.Errorf("stage %d (%s): Partner=%q want empty", i, st.Role, st.Partner)
		}
	}
}
