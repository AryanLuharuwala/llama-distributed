package main

// Pipeline / tensor-parallel planner.
//
// Given a pool and a request, the planner picks an ordered chain of online
// rigs that together cover all transformer layers of the target model.
//
//   PP only:     [stage0=rigA, stage1=rigB, stage2=rigC, ...]
//   PP+TP:       each stage is a rig that runs with tp_size > 1 across its
//                local GPUs; the *inter-node* layout is still pipeline.
//
// The planner only owns layer-to-stage assignment.  Model download and the
// actual forward pass are the agent's job.

import (
	"fmt"
	"time"
)

// StageAssignment is one node in the pipeline chain.
type StageAssignment struct {
	StageIdx   int    `json:"stage_idx"`
	LayerLo    int    `json:"layer_lo"`     // inclusive
	LayerHi    int    `json:"layer_hi"`     // exclusive
	UserID     int64  `json:"user_id"`
	AgentID    string `json:"agent_id"`
	Hostname   string `json:"hostname"`
	TPSize     int    `json:"tp_size"`      // intra-node GPUs (intra mode) OR group cardinality (inter)
	TransportHint string `json:"transport"` // "ws" for now; "rtc" later

	// Model-shard download pointer.  Populated by the coordinator after the
	// planner picks stages so the node can fetch just its slab.
	ShardURL   string `json:"shard_url,omitempty"`
	ShardFile  string `json:"shard_file,omitempty"`

	// Inter-rig TP group fields.  In tp_mode=intra these are zero/empty
	// (TPGroupID stays -1) and the rig handles its own NCCL.  In
	// tp_mode=inter, the planner emits TPSize StageAssignments per pipeline
	// stage, all sharing TPGroupID and StageIdx but differing in TPRank.
	// TPPeers lists peer agent_ids in the same group (excluding self),
	// in rank order; the runtime uses this to set up cross-rig allreduce.
	TPGroupID int      `json:"tp_group_id,omitempty"` // -1 = no group
	TPRank    int      `json:"tp_rank,omitempty"`     // 0..TPSize-1
	TPPeers   []string `json:"tp_peers,omitempty"`
}

// PipelinePlan is what we send to the first node (stage 0) and each next
// stage's "next" pointer so they can daisy-chain directly.
type PipelinePlan struct {
	ReqID       uint32            `json:"req_id"`
	PoolID      int64             `json:"pool_id"`
	ModelName   string            `json:"model"`
	NLayers     int               `json:"n_layers"`
	Parallelism string            `json:"parallelism"`
	Stages      []StageAssignment `json:"stages"`
}

// poolParallelism captures the pool config the planner needs.
type poolParallelism struct {
	Parallelism string
	ModelID     int64
	ModelName   string
	NLayers     int
	PPStages    int
	TPSize      int
	TPMode      string // "intra" (default) or "inter"
}

// loadPoolParallelism reads the parallelism config + (optional) bound model
// metadata.  Returns zeros if the pool has no model bound; the caller can
// still plan if the request itself carries an explicit model.
func (s *server) loadPoolParallelism(poolID int64) (poolParallelism, error) {
	var p poolParallelism
	var modelID *int64
	err := s.db.QueryRow(
		`SELECT parallelism, model_id, pp_stages, tp_size, COALESCE(tp_mode, 'intra')
		 FROM pools WHERE id = ?`,
		poolID,
	).Scan(&p.Parallelism, &modelID, &p.PPStages, &p.TPSize, &p.TPMode)
	if err != nil {
		return p, err
	}
	if p.TPMode != "inter" {
		p.TPMode = "intra"
	}
	if modelID != nil && *modelID > 0 {
		p.ModelID = *modelID
		_ = s.db.QueryRow(
			`SELECT name, n_layers FROM models WHERE id = ?`, *modelID,
		).Scan(&p.ModelName, &p.NLayers)
	}
	if p.PPStages < 1 {
		p.PPStages = 1
	}
	if p.TPSize < 1 {
		p.TPSize = 1
	}
	if p.Parallelism == "" {
		p.Parallelism = "pp"
	}
	return p, nil
}

// onlineRigInfo is the minimal capability view the planner needs.
type onlineRigInfo struct {
	userID   int64
	agentID  string
	hostname string
	nGPUs    int
}

// onlineRigsInPool returns all live, in-pool rigs in deterministic order
// (by rig.id ascending).
func (s *server) onlineRigsInPool(poolID int64) ([]onlineRigInfo, error) {
	rows, err := s.db.Query(`
		SELECT r.id, r.user_id, r.agent_id, r.hostname, r.n_gpus_available
		FROM pool_rigs pr
		JOIN rigs r ON r.id = pr.rig_id
		WHERE pr.pool_id = ?
		ORDER BY r.id ASC`, poolID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []onlineRigInfo
	for rows.Next() {
		var rigID int64
		var info onlineRigInfo
		if err := rows.Scan(&rigID, &info.userID, &info.agentID, &info.hostname, &info.nGPUs); err != nil {
			continue
		}
		if _, ok := s.hub.findAgent(info.userID, info.agentID); !ok {
			continue // skip offline
		}
		out = append(out, info)
	}
	return out, nil
}

// planPipeline builds a PP chain of length pool.PPStages over the online rigs.
// Layers are split into contiguous slabs of (nLayers / ppStages), remainder on
// the last stage.  If there aren't enough online rigs, returns an error so the
// caller can 503.
func (s *server) planPipeline(poolID int64, reqID uint32, modelOverride string, nLayersOverride int) (*PipelinePlan, error) {
	cfg, err := s.loadPoolParallelism(poolID)
	if err != nil {
		return nil, fmt.Errorf("load pool config: %w", err)
	}

	modelName := cfg.ModelName
	nLayers := cfg.NLayers
	if modelOverride != "" {
		modelName = modelOverride
	}
	if nLayersOverride > 0 {
		nLayers = nLayersOverride
	}
	if nLayers <= 0 {
		return nil, fmt.Errorf("pool has no model bound and request did not supply n_layers")
	}

	// Cost-based picker: scores rigs by throughput + VRAM + bandwidth and
	// allocates layer counts proportional to score.  Falls back to equal
	// slabs when no telemetry has arrived yet (cold pool).
	costs, err := s.rigCostsForPool(poolID)
	if err != nil {
		return nil, err
	}
	if len(costs) == 0 {
		return nil, fmt.Errorf("no online rigs in pool")
	}

	stages := cfg.PPStages

	// Inter-rig TP groups: each pipeline stage needs TPSize distinct rigs.
	// Total rigs required = stages * TPSize.  Degrade to what's available.
	tpWidth := 1
	if cfg.TPMode == "inter" && cfg.TPSize > 1 {
		tpWidth = cfg.TPSize
	}
	if stages*tpWidth > len(costs) {
		// Degrade: shrink the pipeline (keep TP group width if possible).
		stages = len(costs) / tpWidth
		if stages < 1 {
			stages = 1
			if tpWidth > len(costs) {
				tpWidth = len(costs)
			}
		}
	}

	picked, layerCounts := pickStagesByScore(costs, stages, nLayers)

	plan := &PipelinePlan{
		ReqID:       reqID,
		PoolID:      poolID,
		ModelName:   modelName,
		NLayers:     nLayers,
		Parallelism: cfg.Parallelism,
	}

	// Track rigs already pinned to a pipeline stage so the inter-rig TP
	// group expander doesn't re-use them for ranks > 0.
	taken := map[agentKey]bool{}
	for i := 0; i < stages; i++ {
		taken[agentKey{picked[i].info.userID, picked[i].info.agentID}] = true
	}

	lo := 0
	for i := 0; i < stages; i++ {
		hi := lo + layerCounts[i]
		rig := picked[i].info

		// Honour requested TP width, but clamp to what the rig actually has
		// in intra mode.  In inter mode the per-rank rig is its own host;
		// the TP width is the *group cardinality*.
		tp := cfg.TPSize
		if cfg.TPMode == "intra" {
			if rig.nGPUs > 0 && tp > rig.nGPUs {
				tp = rig.nGPUs
			}
		} else {
			tp = tpWidth
		}
		if tp < 1 {
			tp = 1
		}

		// Rank 0 — the picked rig for this pipeline stage.
		groupID := -1
		var peerIDs []string
		if cfg.TPMode == "inter" && tp > 1 {
			groupID = i // stable per pipeline stage
		}
		plan.Stages = append(plan.Stages, StageAssignment{
			StageIdx:      i,
			LayerLo:       lo,
			LayerHi:       hi,
			UserID:        rig.userID,
			AgentID:       rig.agentID,
			Hostname:      rig.hostname,
			TPSize:        tp,
			TransportHint: "ws",
			TPGroupID:     groupID,
			TPRank:        0,
			TPPeers:       peerIDs,
		})

		// Ranks 1..tp-1 — pick additional rigs for the TP group.  These
		// share the layer slab with rank 0 and run AllReduce across the
		// group every layer.
		if cfg.TPMode == "inter" && tp > 1 {
			groupMembers := []onlineRigInfo{rig}
			for rank := 1; rank < tp; rank++ {
				pick, ok := pickRigForRole(costs, "tp", taken)
				if !ok {
					break
				}
				taken[agentKey{pick.info.userID, pick.info.agentID}] = true
				groupMembers = append(groupMembers, pick.info)
				plan.Stages = append(plan.Stages, StageAssignment{
					StageIdx:      i,
					LayerLo:       lo,
					LayerHi:       hi,
					UserID:        pick.info.userID,
					AgentID:       pick.info.agentID,
					Hostname:      pick.info.hostname,
					TPSize:        tp,
					TransportHint: "ws",
					TPGroupID:     groupID,
					TPRank:        rank,
				})
			}
			// Backfill TPPeers on every member of this group (peers list
			// excludes self).
			for j := range plan.Stages {
				if plan.Stages[j].StageIdx != i || plan.Stages[j].TPGroupID != groupID {
					continue
				}
				peers := make([]string, 0, len(groupMembers)-1)
				for _, m := range groupMembers {
					if m.agentID != plan.Stages[j].AgentID {
						peers = append(peers, m.agentID)
					}
				}
				plan.Stages[j].TPPeers = peers
			}
		}
		lo = hi
	}

	// If the pool has a model bound, mint a per-stage signed shard URL so the
	// node can download just the slab it owns.  Leaves ShardURL empty if no
	// model is registered yet (M4 plumbing tests still work).
	if cfg.ModelID > 0 {
		for i := range plan.Stages {
			file := fmt.Sprintf("stage-%d.gguf", plan.Stages[i].StageIdx)
			plan.Stages[i].ShardFile = file
			plan.Stages[i].ShardURL = s.mintShardURL(cfg.ModelID, file, 15*time.Minute)
		}
	}
	return plan, nil
}
