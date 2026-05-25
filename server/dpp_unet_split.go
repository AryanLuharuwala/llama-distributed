package main

// CF12-A: pipeline-parallel UNet across N rigs.
//
// Linearised block index for an SDXL/SD3/Flux-class UNet:
//
//   indices 0      ..  D-1      → down_blocks[0..D)
//   index   D                   → mid_block
//   indices D+1    ..  D+1+U-1  → up_blocks[0..U)
//
// Total = D + 1 + U sub-blocks (SDXL is 3+1+3 = 7; SD3 / Flux are
// transformer-block counts in the 24–40 range with no mid_block — we
// treat those as D=0, U=blocks).  The partitioner doesn't need to know
// the exact (D, U) decomposition: it just splits the linearised range
// roughly evenly across N stages and lets the worker translate block
// indices back to the underlying torch modules.
//
// Skip-connection traffic:
//
//   - Each down sub-block emits a residual tensor.  We pack residuals
//     into the inter-stage payload alongside the running sample so any
//     up-side stage can pop the one it needs.
//   - mid_block is pass-through w.r.t. residuals (it transforms the
//     sample only).
//   - Each up sub-block consumes (and removes) one residual from the
//     stack.
//
// Why partition the linearised range instead of stages-by-block:
//
//   A. It composes — a 2-rig SDXL setup can become 4-rig by re-running
//      the partitioner with N=4.  Block ranges shrink, no other code
//      changes.
//   B. The worker honors block_lo/block_hi as a contiguous slice; it
//      doesn't need to special-case "this stage holds mid_block plus
//      half of up_blocks."  Half of up_blocks IS the partition.
//   C. It mirrors the LLM pipeline-parallel code (server/pp_route.go)
//      almost line-for-line — a stage owns layers [lo, hi).

// uNetBlockRange describes one PP stage's slice of the linearised
// UNet.  Half-open [lo, hi).  The first stage owns conv_in and
// time_embedding; the last stage owns conv_norm_out and conv_out.
// Those tails attach to the boundary stages automatically — the
// worker checks `is_first` / `is_last` (set by dpp_route) instead of
// gluing them to a virtual extra block index.  The boundary handling
// thus stays out of the partitioner: it only concerns itself with the
// interior block count.
type uNetBlockRange struct {
	Lo int
	Hi int
}

// partitionUNetBlocks splits a linearised UNet of `totalBlocks`
// sub-blocks into `nStages` contiguous ranges.  Remainder blocks are
// distributed to the earliest stages (so e.g. 7 blocks / 3 stages
// yields [3, 2, 2] rather than [2, 2, 3]) — front-loading is the
// right call because the down-side is generally larger than the
// up-side in tensor count and the front rig is usually the most
// VRAM-rich (it also holds conv_in + time_embedding).
//
// totalBlocks must be ≥ nStages.  Returns nil on a degenerate input
// rather than panicking; callers fall back to single-stage UNet.
//
// Invariants checked by TestPartitionUNetBlocks:
//   - len(out) == nStages
//   - out[0].Lo == 0 and out[-1].Hi == totalBlocks
//   - out[i+1].Lo == out[i].Hi (no gaps, no overlap)
//   - max(hi-lo) - min(hi-lo) ≤ 1 (within-one-block balance)
func partitionUNetBlocks(totalBlocks, nStages int) []uNetBlockRange {
	if nStages < 1 || totalBlocks < nStages {
		return nil
	}
	if nStages == 1 {
		return []uNetBlockRange{{0, totalBlocks}}
	}
	base := totalBlocks / nStages
	extra := totalBlocks % nStages
	out := make([]uNetBlockRange, 0, nStages)
	cursor := 0
	for i := 0; i < nStages; i++ {
		span := base
		if i < extra {
			span++
		}
		out = append(out, uNetBlockRange{Lo: cursor, Hi: cursor + span})
		cursor += span
	}
	return out
}

// uNetModelTopology describes a model family's linearised block
// count.  Hardcoded for the families our dpp_runtime actually loads
// today; future families plug in by extending the table.  Values are
// what diffusers reports for the canonical hub checkpoints.
//
// The worker rejects unknown models rather than guessing — partitioning
// against the wrong block count would silently produce broken images.
type uNetModelTopology struct {
	// Linearised down + mid + up sub-block count.  For DiT/Flux
	// transformer stacks where there's no mid_block, the value is the
	// total transformer block count and the worker treats half as
	// "down" half as "up" with no residuals (set by HasResiduals=false).
	TotalBlocks   int
	HasResiduals  bool
	Family        string // "unet-2d-cond" | "dit" | "flux"
}

// knownUNetTopologies is intentionally short.  We only enumerate the
// shapes the worker has been validated against; an unknown model gets
// nStages=1 (single-rig UNet) so the user still gets an image
// instead of a hang.
//
// Keys are lowercased substrings; lookup is "first match wins".
var knownUNetTopologies = map[string]uNetModelTopology{
	"stable-diffusion-xl": {TotalBlocks: 7, HasResiduals: true, Family: "unet-2d-cond"},
	"sdxl":                {TotalBlocks: 7, HasResiduals: true, Family: "unet-2d-cond"},
	"sd3":                 {TotalBlocks: 24, HasResiduals: false, Family: "dit"},
	"sd3.5":               {TotalBlocks: 38, HasResiduals: false, Family: "dit"},
	"flux":                {TotalBlocks: 19 + 38, HasResiduals: false, Family: "flux"}, // double + single transformer stacks
}

// lookupUNetTopology resolves a model name → topology.  Case-folded
// substring match so "stabilityai/stable-diffusion-xl-base-1.0" and
// "playgroundai/playground-v2.5-1024px-aesthetic" both hit the SDXL
// entry.  Returns ok=false if the family is unknown.
func lookupUNetTopology(modelName string) (uNetModelTopology, bool) {
	lc := lowercaseASCII(modelName)
	// Try matches in length order — "sd3.5" before "sd3" so the
	// longer key wins when both substrings are present.
	keys := []string{"stable-diffusion-xl", "sdxl", "sd3.5", "sd3", "flux"}
	for _, k := range keys {
		if containsLower(lc, k) {
			return knownUNetTopologies[k], true
		}
	}
	return uNetModelTopology{}, false
}

// lowercaseASCII / containsLower keep the dependency footprint zero —
// we don't need strings.ToLower's full Unicode treatment for model
// names that are ASCII by convention.
func lowercaseASCII(s string) string {
	b := make([]byte, len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= 'A' && c <= 'Z' {
			c += 'a' - 'A'
		}
		b[i] = c
	}
	return string(b)
}

func containsLower(haystack, needle string) bool {
	if len(needle) > len(haystack) {
		return false
	}
	for i := 0; i+len(needle) <= len(haystack); i++ {
		if haystack[i:i+len(needle)] == needle {
			return true
		}
	}
	return false
}
