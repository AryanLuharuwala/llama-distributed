package main

import "testing"

func TestPartitionUNetBlocks_Balanced(t *testing.T) {
	cases := []struct {
		total, stages int
		want          []uNetBlockRange
	}{
		// Exact division — every stage same size.
		{6, 2, []uNetBlockRange{{0, 3}, {3, 6}}},
		{6, 3, []uNetBlockRange{{0, 2}, {2, 4}, {4, 6}}},
		{24, 4, []uNetBlockRange{{0, 6}, {6, 12}, {12, 18}, {18, 24}}},
		// Remainder front-loaded.  7 / 3 = [3, 2, 2].
		{7, 3, []uNetBlockRange{{0, 3}, {3, 5}, {5, 7}}},
		// SDXL 7 blocks / 4 stages = [2, 2, 2, 1].
		{7, 4, []uNetBlockRange{{0, 2}, {2, 4}, {4, 6}, {6, 7}}},
		// Single-stage degenerate.
		{12, 1, []uNetBlockRange{{0, 12}}},
		// Edge: total == stages, each owns one block.
		{4, 4, []uNetBlockRange{{0, 1}, {1, 2}, {2, 3}, {3, 4}}},
	}
	for _, tc := range cases {
		got := partitionUNetBlocks(tc.total, tc.stages)
		if !equalRanges(got, tc.want) {
			t.Errorf("partition(%d, %d) = %v, want %v", tc.total, tc.stages, got, tc.want)
		}
	}
}

func TestPartitionUNetBlocks_Invariants(t *testing.T) {
	// Property test: across a grid of inputs, the output partition is
	// (a) contiguous, (b) covers [0, total), (c) balanced to within 1.
	for total := 1; total <= 64; total++ {
		for stages := 1; stages <= total; stages++ {
			r := partitionUNetBlocks(total, stages)
			if len(r) != stages {
				t.Fatalf("total=%d stages=%d: len=%d want=%d", total, stages, len(r), stages)
			}
			if r[0].Lo != 0 {
				t.Fatalf("total=%d stages=%d: first.Lo=%d want=0", total, stages, r[0].Lo)
			}
			if r[len(r)-1].Hi != total {
				t.Fatalf("total=%d stages=%d: last.Hi=%d want=%d", total, stages, r[len(r)-1].Hi, total)
			}
			minS, maxS := total, 0
			for i, rng := range r {
				span := rng.Hi - rng.Lo
				if span < 1 {
					t.Fatalf("total=%d stages=%d idx=%d: span=%d (empty stage)", total, stages, i, span)
				}
				if i > 0 && rng.Lo != r[i-1].Hi {
					t.Fatalf("total=%d stages=%d idx=%d: gap or overlap (prev.Hi=%d, lo=%d)",
						total, stages, i, r[i-1].Hi, rng.Lo)
				}
				if span < minS {
					minS = span
				}
				if span > maxS {
					maxS = span
				}
			}
			if maxS-minS > 1 {
				t.Fatalf("total=%d stages=%d: imbalance max=%d min=%d", total, stages, maxS, minS)
			}
		}
	}
}

func TestPartitionUNetBlocks_Degenerate(t *testing.T) {
	if r := partitionUNetBlocks(0, 1); r != nil {
		t.Errorf("partition(0,1) = %v, want nil", r)
	}
	if r := partitionUNetBlocks(3, 4); r != nil {
		t.Errorf("partition(3,4) = %v, want nil (more stages than blocks)", r)
	}
	if r := partitionUNetBlocks(10, 0); r != nil {
		t.Errorf("partition(10,0) = %v, want nil", r)
	}
}

func TestLookupUNetTopology(t *testing.T) {
	cases := []struct {
		name      string
		wantOK    bool
		wantTotal int
		wantFam   string
	}{
		{"stabilityai/stable-diffusion-xl-base-1.0", true, 7, "unet-2d-cond"},
		{"playgroundai/playground-v2.5-1024px", false, 0, ""}, // not in family table — unknown is fine
		{"stabilityai/sd3.5-large", true, 38, "dit"},
		{"stabilityai/sd3-medium", true, 24, "dit"},
		{"black-forest-labs/FLUX.1-dev", true, 57, "flux"},
		{"SDXL-Turbo", true, 7, "unet-2d-cond"},
		{"", false, 0, ""},
		{"some-random-llama-merge", false, 0, ""},
	}
	for _, tc := range cases {
		got, ok := lookupUNetTopology(tc.name)
		if ok != tc.wantOK {
			t.Errorf("lookup(%q) ok=%v want=%v", tc.name, ok, tc.wantOK)
			continue
		}
		if !ok {
			continue
		}
		if got.TotalBlocks != tc.wantTotal {
			t.Errorf("lookup(%q) total=%d want=%d", tc.name, got.TotalBlocks, tc.wantTotal)
		}
		if got.Family != tc.wantFam {
			t.Errorf("lookup(%q) fam=%q want=%q", tc.name, got.Family, tc.wantFam)
		}
	}
}

func equalRanges(a, b []uNetBlockRange) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
