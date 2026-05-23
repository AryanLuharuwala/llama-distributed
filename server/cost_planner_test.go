package main

import "testing"

func TestPickStagesByScore_ProportionalAllocation(t *testing.T) {
	// Three rigs with deliberately skewed VRAM — the 24G rig should take
	// more layers than the 8G rig.
	rigs := []rigCost{
		{info: onlineRigInfo{agentID: "small"},  vramTotal: 8 << 30,  throughput: 30},
		{info: onlineRigInfo{agentID: "medium"}, vramTotal: 16 << 30, throughput: 60},
		{info: onlineRigInfo{agentID: "large"},  vramTotal: 24 << 30, throughput: 100},
	}
	picked, counts := pickStagesByScore(rigs, 3, 32)
	if len(picked) != 3 || len(counts) != 3 {
		t.Fatalf("expected 3 stages, got %d/%d", len(picked), len(counts))
	}
	sum := 0
	for _, c := range counts {
		sum += c
		if c < 1 {
			t.Fatalf("every stage must get at least 1 layer, got %v", counts)
		}
	}
	if sum != 32 {
		t.Fatalf("layer counts must sum to nLayers, got %d (counts=%v)", sum, counts)
	}
	// Highest-score rig is picked first.
	if picked[0].info.agentID != "large" {
		t.Fatalf("expected 'large' rig first, got %q", picked[0].info.agentID)
	}
	// And it should get more layers than 'small'.
	smallIdx := -1
	for i, p := range picked {
		if p.info.agentID == "small" {
			smallIdx = i
			break
		}
	}
	if smallIdx < 0 {
		t.Fatalf("'small' rig not in picked set")
	}
	if counts[0] <= counts[smallIdx] {
		t.Fatalf("large rig should get more layers than small (large=%d small=%d)",
			counts[0], counts[smallIdx])
	}
}

func TestPickStagesByScore_ColdPoolEqualSlabs(t *testing.T) {
	// Zero telemetry — every rig has score 1 (the floor).  Layers should
	// split evenly with the last stage absorbing remainder.
	rigs := []rigCost{
		{info: onlineRigInfo{agentID: "a"}},
		{info: onlineRigInfo{agentID: "b"}},
		{info: onlineRigInfo{agentID: "c"}},
	}
	_, counts := pickStagesByScore(rigs, 3, 32)
	sum := 0
	for _, c := range counts {
		sum += c
	}
	if sum != 32 {
		t.Fatalf("layer counts must sum to nLayers, got %d (%v)", sum, counts)
	}
}

func TestPickStagesByScore_StagesGreaterThanRigs(t *testing.T) {
	// Two rigs, three stages requested — should silently degrade to 2.
	rigs := []rigCost{
		{info: onlineRigInfo{agentID: "a"}, vramTotal: 8 << 30},
		{info: onlineRigInfo{agentID: "b"}, vramTotal: 16 << 30},
	}
	picked, counts := pickStagesByScore(rigs, 3, 16)
	if len(picked) != 2 || len(counts) != 2 {
		t.Fatalf("expected 2 stages after degrade, got %d/%d", len(picked), len(counts))
	}
}

func TestPickRigForRole_UnetPicksBiggestVRAM(t *testing.T) {
	rigs := []rigCost{
		{info: onlineRigInfo{agentID: "fast-small"}, vramTotal: 4 << 30,  throughput: 200},
		{info: onlineRigInfo{agentID: "slow-large"}, vramTotal: 24 << 30, throughput: 10},
	}
	pick, ok := pickRigForRole(rigs, "unet", map[agentKey]bool{})
	if !ok {
		t.Fatal("expected a pick")
	}
	if pick.info.agentID != "slow-large" {
		t.Fatalf("unet should prefer VRAM; got %q", pick.info.agentID)
	}
}

func TestPickRigForRole_VAEPicksThroughput(t *testing.T) {
	rigs := []rigCost{
		{info: onlineRigInfo{agentID: "fast-small"}, vramTotal: 4 << 30,  throughput: 200},
		{info: onlineRigInfo{agentID: "slow-large"}, vramTotal: 24 << 30, throughput: 5},
	}
	pick, ok := pickRigForRole(rigs, "vae", map[agentKey]bool{})
	if !ok {
		t.Fatal("expected a pick")
	}
	if pick.info.agentID != "fast-small" {
		t.Fatalf("vae should prefer throughput; got %q", pick.info.agentID)
	}
}

func TestPickRigForRole_RespectsTaken(t *testing.T) {
	rigs := []rigCost{
		{info: onlineRigInfo{agentID: "a"}, vramTotal: 24 << 30},
		{info: onlineRigInfo{agentID: "b"}, vramTotal: 8 << 30},
	}
	taken := map[agentKey]bool{{0, "a"}: true}
	pick, ok := pickRigForRole(rigs, "unet", taken)
	if !ok || pick.info.agentID != "b" {
		t.Fatalf("expected fallback to 'b' when 'a' taken, got ok=%v agent=%q", ok, pick.info.agentID)
	}
}

func TestPickRigForRole_FallsBackWhenAllTaken(t *testing.T) {
	rigs := []rigCost{
		{info: onlineRigInfo{agentID: "a"}, vramTotal: 24 << 30},
	}
	taken := map[agentKey]bool{{0, "a"}: true}
	pick, ok := pickRigForRole(rigs, "unet", taken)
	if !ok {
		t.Fatal("expected fallback pick when all rigs are taken")
	}
	if pick.info.agentID != "a" {
		t.Fatalf("expected to reuse 'a', got %q", pick.info.agentID)
	}
}
