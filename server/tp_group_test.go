package main

// Tests for inter-rig TP group scaffolding in planPipeline:
//   - tp_mode=intra (default) keeps the classic 1-rig-per-stage layout.
//   - tp_mode=inter, tp_size=2 produces 2 StageAssignments per pp_stage
//     sharing TPGroupID + StageIdx but with distinct TPRank and AgentIDs.
//   - Each rank lists its sibling ranks in TPPeers (excluding self).
//   - Insufficient rigs → graceful degrade (pp_stages shrinks).

import (
	"testing"
)

// seedRigInPool inserts a rigs row + pool_rigs link + a hub stub for one
// online rig.  Returns the rigID.
func seedRigInPool(t *testing.T, s *server, uid, poolID int64, agentID, hostname string, nGPUs int) int64 {
	t.Helper()
	res, err := s.db.Exec(
		`INSERT INTO rigs (user_id, agent_id, hostname, n_gpus, last_seen, n_gpus_available)
		 VALUES (?, ?, ?, ?, ?, ?)`,
		uid, agentID, hostname, nGPUs, nowUnix(), nGPUs,
	)
	if err != nil {
		t.Fatalf("insert rig: %v", err)
	}
	rigID, _ := res.LastInsertId()
	if _, err := s.db.Exec(
		`INSERT INTO pool_rigs (pool_id, rig_id, added_at) VALUES (?, ?, ?)`,
		poolID, rigID, nowUnix(),
	); err != nil {
		t.Fatalf("insert pool_rig: %v", err)
	}
	registerStubAgent(s, uid, agentID, hostname)
	return rigID
}

// seedPoolWithModel inserts a model + a pool with the given parallelism
// config bound to that model.  Returns (poolID, modelID).
func seedPoolWithModel(t *testing.T, s *server, uid int64, ppStages, tpSize int, tpMode string) (int64, int64) {
	t.Helper()
	mres, err := s.db.Exec(
		`INSERT INTO models (name, n_layers, n_shards, shards_dir, created_at)
		 VALUES (?, ?, ?, ?, ?)`,
		"tp-test-model", 8, 1, t.TempDir(), nowUnix(),
	)
	if err != nil {
		t.Fatalf("insert model: %v", err)
	}
	mid, _ := mres.LastInsertId()
	pres, err := s.db.Exec(
		`INSERT INTO pools
		 (owner_id, name, visibility, created_at, parallelism, pp_stages, tp_size, tp_mode, slug, model_id)
		 VALUES (?, ?, 'private', ?, 'pp+tp', ?, ?, ?, ?, ?)`,
		uid, "tp-pool", nowUnix(), ppStages, tpSize, tpMode, "tp-pool", mid,
	)
	if err != nil {
		t.Fatalf("insert pool: %v", err)
	}
	pid, _ := pres.LastInsertId()
	if _, err := s.db.Exec(
		`INSERT INTO pool_members (pool_id, user_id, role, joined_at) VALUES (?, ?, 'owner', ?)`,
		pid, uid, nowUnix(),
	); err != nil {
		t.Fatalf("insert pool_member: %v", err)
	}
	return pid, mid
}

func TestPlanPipeline_TPMode_Intra_OneRigPerStage(t *testing.T) {
	s := newTestServer(t)
	uid, _ := makeUser(t, s, "tp-user")
	poolID, _ := seedPoolWithModel(t, s, uid, 2, 2, "intra")
	// 2 rigs, each claiming 2 GPUs (intra-rig TP).
	seedRigInPool(t, s, uid, poolID, "rig-A", "host-A", 2)
	seedRigInPool(t, s, uid, poolID, "rig-B", "host-B", 2)

	plan, err := s.planPipeline(poolID, 1, "", 0)
	if err != nil {
		t.Fatalf("planPipeline: %v", err)
	}
	if len(plan.Stages) != 2 {
		t.Fatalf("intra mode: expected 2 stages, got %d", len(plan.Stages))
	}
	for _, st := range plan.Stages {
		if st.TPGroupID != 0 && st.TPGroupID != -1 {
			// Intra mode leaves TPGroupID unset (defaults to 0 because of the
			// zero value).  Either 0 or -1 is acceptable for "no group" — but
			// TPPeers MUST be empty.
		}
		if len(st.TPPeers) != 0 {
			t.Errorf("intra stage has TPPeers set: %+v", st)
		}
		if st.TPSize != 2 {
			t.Errorf("intra TPSize should reflect requested width; got %d", st.TPSize)
		}
	}
}

func TestPlanPipeline_TPMode_Inter_ExpandsToGroups(t *testing.T) {
	s := newTestServer(t)
	uid, _ := makeUser(t, s, "tp-inter")
	poolID, _ := seedPoolWithModel(t, s, uid, 2, 2, "inter")
	// 4 rigs — pp_stages=2, tp_size=2 → 2*2 = 4 distinct rigs.
	for i, name := range []string{"r0", "r1", "r2", "r3"} {
		seedRigInPool(t, s, uid, poolID, name, "host-"+name, 1)
		_ = i
	}

	plan, err := s.planPipeline(poolID, 1, "", 0)
	if err != nil {
		t.Fatalf("planPipeline: %v", err)
	}
	if len(plan.Stages) != 4 {
		t.Fatalf("inter mode 2×2 expected 4 stage entries, got %d", len(plan.Stages))
	}

	// Group by StageIdx — each group must have 2 ranks with distinct AgentIDs.
	groups := map[int][]StageAssignment{}
	for _, st := range plan.Stages {
		groups[st.StageIdx] = append(groups[st.StageIdx], st)
	}
	if len(groups) != 2 {
		t.Fatalf("expected 2 pipeline stages; got %d", len(groups))
	}
	for stageIdx, group := range groups {
		if len(group) != 2 {
			t.Errorf("stage %d: expected 2 ranks, got %d", stageIdx, len(group))
		}
		seen := map[string]bool{}
		ranks := map[int]bool{}
		for _, m := range group {
			if seen[m.AgentID] {
				t.Errorf("stage %d: duplicate AgentID %q in group", stageIdx, m.AgentID)
			}
			seen[m.AgentID] = true
			ranks[m.TPRank] = true
			if m.TPGroupID < 0 {
				t.Errorf("stage %d rank %d: TPGroupID not set (got %d)", stageIdx, m.TPRank, m.TPGroupID)
			}
			if len(m.TPPeers) != 1 {
				t.Errorf("stage %d rank %d: expected 1 peer, got %v", stageIdx, m.TPRank, m.TPPeers)
			} else if m.TPPeers[0] == m.AgentID {
				t.Errorf("stage %d rank %d: peer list includes self", stageIdx, m.TPRank)
			}
			if m.LayerLo == m.LayerHi {
				t.Errorf("stage %d rank %d: empty layer slab", stageIdx, m.TPRank)
			}
		}
		// Ranks 0 and 1 must both appear.
		if !ranks[0] || !ranks[1] {
			t.Errorf("stage %d: missing ranks (got %+v)", stageIdx, ranks)
		}
	}

	// Across all stages, every rig is used exactly once (no rig in two groups).
	usedAgents := map[string]int{}
	for _, st := range plan.Stages {
		usedAgents[st.AgentID]++
	}
	if len(usedAgents) != 4 {
		t.Errorf("expected 4 distinct rigs across plan; got %d", len(usedAgents))
	}
	for ag, c := range usedAgents {
		if c != 1 {
			t.Errorf("rig %q appears %d times; expected once", ag, c)
		}
	}
}

func TestPlanPipeline_TPMode_Inter_DegradesWhenShort(t *testing.T) {
	s := newTestServer(t)
	uid, _ := makeUser(t, s, "tp-short")
	// Ask for 2 pp_stages × 2 tp_size = 4 rigs, but only seed 3.
	poolID, _ := seedPoolWithModel(t, s, uid, 2, 2, "inter")
	seedRigInPool(t, s, uid, poolID, "r0", "h0", 1)
	seedRigInPool(t, s, uid, poolID, "r1", "h1", 1)
	seedRigInPool(t, s, uid, poolID, "r2", "h2", 1)

	plan, err := s.planPipeline(poolID, 1, "", 0)
	if err != nil {
		t.Fatalf("planPipeline (should degrade, not error): %v", err)
	}
	// Degraded: 3 rigs / 2 tp_width = 1 pp_stage × 2 rigs = 2 stage entries.
	// (The third rig is unused.)
	if len(plan.Stages) != 2 {
		t.Fatalf("degraded inter: expected 2 stage entries (1 stage × 2 ranks), got %d", len(plan.Stages))
	}
	for _, st := range plan.Stages {
		if st.TPGroupID < 0 {
			t.Errorf("degraded stage missing TPGroupID: %+v", st)
		}
	}
}
