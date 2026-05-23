package main

import (
	"sort"
)

// Cost-based stage planner.
//
// The naïve planner (planner.go) split layers into equal-size slabs and
// picked rigs by sort order.  That works on a hand-curated homogeneous pool
// but collapses on a real swarm where rigs have wildly different VRAM,
// throughput, and bandwidth — a 3090 paired with an M1 Air will see the
// M1 bottleneck the whole pipeline.
//
// This file computes a per-rig "cost score" from the live telemetry the
// rigs already push every 5s (see ws.go agentStatus → agentConn.live):
//
//   score = throughput_weight + vram_weight + bandwidth_weight − inflight_penalty
//
// The score is used in two ways:
//   1. planPipeline allocates layers proportional to each chosen rig's score.
//   2. planDPP picks a rig per role based on the per-role bottleneck:
//        text_encoder → modest VRAM, throughput matters
//        unet         → biggest VRAM wins (the denoiser is VRAM-bound)
//        vae          → tiny model, throughput matters
//
// A rig with stale telemetry (UpdatedAt older than 60s) drops to a default
// score so it still participates but isn't preferred over a rig the planner
// knows is healthy.

// rigCost is the scoring view of an online rig.  Built from the live
// telemetry snapshot, plus the static info we already have.
type rigCost struct {
	info onlineRigInfo
	live liveStatus

	throughput float64 // tok/s; 0 if unknown
	vramTotal  int64
	vramFree   int64
	bwKbps     int64 // min(up, down); 0 if unknown
	inflight   int
	stale      bool

	// hasModel is true when the rig has reported the target model in its
	// ModelsHeld set — meaning the slab is already on disk, no download
	// required to start serving.  Set per-request by the caller; pickers
	// that don't know the model leave it false.
	hasModel bool
}

// score returns a single number that ranks this rig for general-purpose
// layer hosting.  Higher is better.  All terms are non-negative and the
// final value is clamped to [1, +inf) so a "broken" rig (no telemetry yet,
// no GPU info) still gets some share — better to use it than starve the
// pipeline.
func (r *rigCost) score() float64 {
	// Throughput term: tokens/sec, capped at 200 so a single hot rig
	// doesn't take 90% of layers (the warm-up tail of the first request
	// would then time out on it).
	tpt := r.throughput
	if tpt > 200 {
		tpt = 200
	}

	// VRAM term: GiB, capped at 80 (an H100).  We log-scale lightly so
	// a 24G rig isn't 3× a 8G rig — the 24G rig usually can't process 3×
	// as many layers per sec.
	vramGiB := float64(r.vramTotal) / (1024 * 1024 * 1024)
	if vramGiB > 80 {
		vramGiB = 80
	}

	// Bandwidth term: Mbps, capped at 10 Gbps (1.25 GB/s).  Pipelines
	// stall on the slowest link; bandwidth gates how many layers a rig can
	// usefully host before its uplink becomes the bottleneck.
	bwMbps := float64(r.bwKbps) / 1024.0
	if bwMbps > 10000 {
		bwMbps = 10000
	}

	// In-flight penalty: each ongoing request subtracts 5 points so an
	// idle rig is preferred over a busy one when capacity is otherwise
	// equal.  Capped so a runaway counter doesn't drive the score below 1.
	penalty := float64(r.inflight) * 5
	if penalty > 50 {
		penalty = 50
	}

	// Stale telemetry: if we haven't heard from this rig in 60s we trust
	// only the hello-frame VRAM figure; halve the score to discourage
	// preferring it.
	staleMul := 1.0
	if r.stale {
		staleMul = 0.5
	}

	// Cached-model bonus: a rig that already has the slab on disk saves
	// a several-hundred-MB download on the first request.  Weight chosen
	// so the bonus dominates a small VRAM/throughput differential but a
	// truly faster rig (e.g. 4× tokens/sec) can still win.
	cacheBonus := 0.0
	if r.hasModel {
		cacheBonus = 40.0
	}

	s := (tpt*1.0 + vramGiB*4.0 + bwMbps*0.05 + cacheBonus - penalty) * staleMul
	if s < 1 {
		s = 1
	}
	return s
}

// applyModelAffinity flips hasModel on each rig whose ModelsHeld contains
// modelName.  Called by planner.go right before pickStagesByScore so the
// scorer can give cached rigs a leg up.  Cheap: O(N rigs × M models held)
// with N ≪ 50 and M ≪ 10 in practice.
func applyModelAffinity(rigs []rigCost, modelName string) {
	if modelName == "" {
		return
	}
	for i := range rigs {
		for _, m := range rigs[i].live.ModelsHeld {
			if m == modelName {
				rigs[i].hasModel = true
				break
			}
		}
	}
}

// rigCostsForPool is the cached entry point used by the planner hot path.
// Pulls from a per-pool snapshot (rebuilt at most every costCache.ttl) so
// a 100 RPS chat-completion burst doesn't rebuild the same JOIN 100×/s.
// Direct callers that need a non-cached read (tests, admin tools) should
// call rigCostsForPoolUncached.
func (s *server) rigCostsForPool(poolID int64) ([]rigCost, error) {
	if s.costCache == nil {
		return s.rigCostsForPoolUncached(poolID)
	}
	return s.costCache.getOrFill(poolID, func() ([]rigCost, error) {
		return s.rigCostsForPoolUncached(poolID)
	})
}

// rigCostsForPoolUncached is the raw DB-touching implementation.  Builds
// a rigCost for every online rig in the pool.  Telemetry-less rigs still
// appear; their score is the floor.
func (s *server) rigCostsForPoolUncached(poolID int64) ([]rigCost, error) {
	rigs, err := s.onlineRigsInPool(poolID)
	if err != nil {
		return nil, err
	}
	out := make([]rigCost, 0, len(rigs))
	now := nowUnix()
	for _, info := range rigs {
		ac, ok := s.hub.findAgent(info.userID, info.agentID)
		if !ok {
			continue
		}
		live := ac.snapshotStatus()
		bw := live.BWDnKbps
		if live.BWUpKbps > 0 && (bw == 0 || live.BWUpKbps < bw) {
			bw = live.BWUpKbps
		}
		c := rigCost{
			info:       info,
			live:       live,
			throughput: live.TokensPS,
			vramTotal:  live.VRAMTotal,
			vramFree:   live.VRAMFree,
			bwKbps:     bw,
			inflight:   live.Inflight,
			stale:      live.UpdatedAt > 0 && (now-live.UpdatedAt) > 60,
		}
		// If telemetry never landed, fall back to the hello-time vram_bytes
		// recorded on the rigs row.  We can't read it here without another
		// query; the agentConn doesn't currently carry it.  Score handles
		// zero gracefully (just lower weight).
		out = append(out, c)
	}
	return out, nil
}

// pickStagesByScore returns the N highest-scoring rigs along with each
// rig's relative share (sum to 1.0) and layer counts that sum to nLayers.
//
// If len(rigs) < N, every rig is included (the caller decides whether to
// 503 or downgrade).
func pickStagesByScore(rigs []rigCost, nStages, nLayers int) ([]rigCost, []int) {
	sorted := make([]rigCost, len(rigs))
	copy(sorted, rigs)
	sort.SliceStable(sorted, func(i, j int) bool {
		return sorted[i].score() > sorted[j].score()
	})
	if nStages > len(sorted) {
		nStages = len(sorted)
	}
	picked := sorted[:nStages]

	// Proportional layer allocation.  Compute floor counts first; spread
	// remainder one layer at a time, highest-fractional-residue first.
	totalScore := 0.0
	for _, r := range picked {
		totalScore += r.score()
	}
	if totalScore <= 0 || nLayers <= 0 || nStages == 0 {
		// Falls back to equal slabs — keeps the planner honest if every
		// score is zero (cold pool with no telemetry yet).
		out := make([]int, nStages)
		base := nLayers / nStages
		rem := nLayers % nStages
		for i := range out {
			out[i] = base
		}
		if nStages > 0 {
			out[nStages-1] += rem
		}
		return picked, out
	}

	floors := make([]int, nStages)
	residues := make([]float64, nStages)
	assigned := 0
	for i, r := range picked {
		share := r.score() / totalScore
		ideal := share * float64(nLayers)
		floors[i] = int(ideal)
		residues[i] = ideal - float64(floors[i])
		assigned += floors[i]
	}
	left := nLayers - assigned
	// Distribute leftover layers to the highest residues.
	idx := make([]int, nStages)
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool {
		return residues[idx[a]] > residues[idx[b]]
	})
	for k := 0; k < left && k < nStages; k++ {
		floors[idx[k]]++
	}
	// Guard: every stage must get at least 1 layer.  Steal from the
	// largest stage if needed.
	for i := range floors {
		if floors[i] < 1 {
			// Find the largest-floor stage and decrement it.
			maxIdx := 0
			for j := range floors {
				if floors[j] > floors[maxIdx] {
					maxIdx = j
				}
			}
			if floors[maxIdx] > 1 {
				floors[maxIdx]--
				floors[i] = 1
			}
		}
	}
	return picked, floors
}

// pickRigForRole returns the rig in `rigs` best suited for the named DPP
// role, or false if `rigs` is empty.  The selector hints what kind of
// capacity dominates the role:
//   - "unet"       → VRAM-dominant
//   - "text_encoder" → throughput-dominant
//   - "vae"        → throughput-dominant, low VRAM
//   - default      → general score
//
// `taken` is a set of (userID, agentID) already used in this plan.  Same
// rig may host multiple roles (returned only if no fresh rig fits), so we
// retry without the filter if every rig is taken.
func pickRigForRole(rigs []rigCost, role string, taken map[agentKey]bool) (rigCost, bool) {
	if len(rigs) == 0 {
		return rigCost{}, false
	}
	scoreFn := func(r rigCost) float64 {
		switch role {
		case "unet", "dit", "unet_cond", "unet_uncond":
			// VRAM weight 8×, throughput weight 0.5×.
			// CFG-split halves (unet_cond / unet_uncond) score identically
			// to a plain unet stage — each runs the full denoiser on its
			// half of the conditioning.
			return float64(r.vramTotal)/(1024*1024*1024)*8 + r.throughput*0.5
		case "text_encoder", "vae":
			// Throughput-led with a small VRAM floor.
			return r.throughput*2 + float64(r.vramTotal)/(1024*1024*1024)
		default:
			return r.score()
		}
	}
	// First pass: untaken rigs only.
	var best *rigCost
	bestScore := -1.0
	for i := range rigs {
		k := agentKey{rigs[i].info.userID, rigs[i].info.agentID}
		if taken[k] {
			continue
		}
		if sc := scoreFn(rigs[i]); sc > bestScore {
			bestScore = sc
			best = &rigs[i]
		}
	}
	if best != nil {
		return *best, true
	}
	// Second pass: every rig is taken — pick the highest score anyway.
	// (Single-rig pool case, or DPP into a 2-rig pool.)
	bestScore = -1.0
	for i := range rigs {
		if sc := scoreFn(rigs[i]); sc > bestScore {
			bestScore = sc
			best = &rigs[i]
		}
	}
	if best == nil {
		return rigCost{}, false
	}
	return *best, true
}
