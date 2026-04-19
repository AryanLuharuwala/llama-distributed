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
	TPSize     int    `json:"tp_size"`      // intra-node GPUs
	TransportHint string `json:"transport"` // "ws" for now; "rtc" later

	// Model-shard download pointer.  Populated by the coordinator after the
	// planner picks stages so the node can fetch just its slab.
	ShardURL   string `json:"shard_url,omitempty"`
	ShardFile  string `json:"shard_file,omitempty"`
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
}

// loadPoolParallelism reads the parallelism config + (optional) bound model
// metadata.  Returns zeros if the pool has no model bound; the caller can
// still plan if the request itself carries an explicit model.
func (s *server) loadPoolParallelism(poolID int64) (poolParallelism, error) {
	var p poolParallelism
	var modelID *int64
	err := s.db.QueryRow(
		`SELECT parallelism, model_id, pp_stages, tp_size FROM pools WHERE id = ?`,
		poolID,
	).Scan(&p.Parallelism, &modelID, &p.PPStages, &p.TPSize)
	if err != nil {
		return p, err
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

	rigs, err := s.onlineRigsInPool(poolID)
	if err != nil {
		return nil, err
	}
	if len(rigs) == 0 {
		return nil, fmt.Errorf("no online rigs in pool")
	}

	stages := cfg.PPStages
	if stages > len(rigs) {
		// Degrade: use as many stages as we have rigs.  The caller still gets
		// a plan; we just don't split as finely as the pool config asked for.
		stages = len(rigs)
	}

	// Contiguous layer slabs.  Last stage absorbs the remainder so every
	// layer is assigned exactly once.
	base := nLayers / stages
	rem := nLayers % stages

	plan := &PipelinePlan{
		ReqID:       reqID,
		PoolID:      poolID,
		ModelName:   modelName,
		NLayers:     nLayers,
		Parallelism: cfg.Parallelism,
	}

	lo := 0
	for i := 0; i < stages; i++ {
		hi := lo + base
		if i == stages-1 {
			hi += rem
		}
		rig := rigs[i%len(rigs)]

		// Honour requested TP width, but clamp to what the rig actually has.
		tp := cfg.TPSize
		if rig.nGPUs > 0 && tp > rig.nGPUs {
			tp = rig.nGPUs
		}
		if tp < 1 {
			tp = 1
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
		})
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
