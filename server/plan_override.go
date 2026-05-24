package main

// Manual layer-to-rig placement override.
//
// The cost-based planner (planner.go) decides which rigs run which layers on
// every dispatch.  That's the right default — telemetry-driven, self-healing
// — but operators sometimes need to pin a layout: stress-testing a particular
// rig, isolating a noisy peer, reproducing a bug.
//
// This file exposes per-pool plan overrides via three endpoints:
//
//   GET    /api/pools/{id}/plan   — read the saved override (any member)
//   PUT    /api/pools/{id}/plan   — save an override        (owner only)
//   DELETE /api/pools/{id}/plan   — clear the override       (owner only)
//
// The planner reads pools.plan_override at dispatch time.  If the column is
// non-NULL and every pinned agent_id is still a member + online, the planner
// builds StageAssignments directly from the override and skips the cost
// picker.  If any rig is missing the planner falls through to the cost
// picker so a stale override never takes the pool down — see
// applyPlanOverride() below.

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
)

// planOverride is the persisted shape.  We don't store TPSize, TPGroupID,
// etc. — those are runtime-derived from the pool's parallelism config so an
// override stays valid if the operator later flips tp_size.
type planOverrideStage struct {
	StageIdx int    `json:"stage_idx"`
	LayerLo  int    `json:"layer_lo"`
	LayerHi  int    `json:"layer_hi"` // inclusive (matches /topology)
	AgentID  string `json:"agent_id"`
}

type planOverride struct {
	Stages []planOverrideStage `json:"stages"`
}

// loadPlanOverride reads pools.plan_override.  Returns (nil, nil) if unset.
func (s *server) loadPlanOverride(poolID int64) (*planOverride, error) {
	var raw *string
	if err := s.dbQueryRow(`SELECT plan_override FROM pools WHERE id = ?`, poolID).
		Scan(&raw); err != nil {
		return nil, err
	}
	if raw == nil || *raw == "" {
		return nil, nil
	}
	var po planOverride
	if err := json.Unmarshal([]byte(*raw), &po); err != nil {
		return nil, fmt.Errorf("plan_override is malformed: %w", err)
	}
	return &po, nil
}

// validateOverride enforces:
//   - At least one stage.
//   - stage_idx values are 0..N-1 with no duplicates (after sort).
//   - Layer ranges cover [0, nLayers) contiguously with no gaps/overlap.
//   - Every agent_id is a current member of the pool.
//
// We do NOT require rigs to be online here — that's a transient state checked
// at dispatch time by applyPlanOverride.  Saving an override for an offline
// rig is fine; the planner will fall back until the rig comes back.
func (s *server) validateOverride(poolID int64, nLayers int, po *planOverride) error {
	if po == nil || len(po.Stages) == 0 {
		return errors.New("plan must contain at least one stage")
	}
	if nLayers <= 0 {
		return errors.New("pool model has unknown n_layers — bind a model first")
	}

	// Stage-idx contiguity from 0.
	stages := make([]planOverrideStage, len(po.Stages))
	copy(stages, po.Stages)
	seen := map[int]bool{}
	for _, st := range stages {
		if seen[st.StageIdx] {
			return fmt.Errorf("duplicate stage_idx %d", st.StageIdx)
		}
		seen[st.StageIdx] = true
	}
	for i := 0; i < len(stages); i++ {
		if !seen[i] {
			return fmt.Errorf("missing stage_idx %d (stages must be 0..%d contiguous)",
				i, len(stages)-1)
		}
	}

	// Sort by stage_idx and check layer contiguity.
	byIdx := make([]planOverrideStage, len(stages))
	for _, st := range stages {
		byIdx[st.StageIdx] = st
	}
	if byIdx[0].LayerLo != 0 {
		return fmt.Errorf("stage 0 must start at layer 0 (got %d)", byIdx[0].LayerLo)
	}
	for i := 0; i < len(byIdx); i++ {
		st := byIdx[i]
		if st.LayerLo > st.LayerHi {
			return fmt.Errorf("stage %d: layer_lo (%d) > layer_hi (%d)",
				i, st.LayerLo, st.LayerHi)
		}
		if i > 0 && st.LayerLo != byIdx[i-1].LayerHi+1 {
			return fmt.Errorf("stage %d starts at layer %d; expected %d (no gaps/overlap)",
				i, st.LayerLo, byIdx[i-1].LayerHi+1)
		}
	}
	last := byIdx[len(byIdx)-1]
	if last.LayerHi != nLayers-1 {
		return fmt.Errorf("last stage must end at layer %d (got %d)",
			nLayers-1, last.LayerHi)
	}

	// Agent membership: every pinned agent_id must currently be a rig in the
	// pool owned by some user.  We don't lock the lookup — by the time the
	// planner runs the rig may have left, and applyPlanOverride handles that.
	for _, st := range byIdx {
		if st.AgentID == "" {
			return fmt.Errorf("stage %d has empty agent_id", st.StageIdx)
		}
		var n int
		err := s.dbQueryRow(`
			SELECT COUNT(*) FROM pool_rigs pr
			JOIN rigs r ON r.id = pr.rig_id
			WHERE pr.pool_id = ? AND r.agent_id = ?`,
			poolID, st.AgentID).Scan(&n)
		if err != nil {
			return err
		}
		if n == 0 {
			return fmt.Errorf("agent %q is not a member of pool %d", st.AgentID, poolID)
		}
	}
	return nil
}

// applyPlanOverride converts a saved override into a PipelinePlan when every
// pinned rig is online.  Returns nil (no error) if the override is unusable
// right now so the caller falls back to the cost picker — this is a feature,
// not a bug: an override should degrade, not fail closed.
func (s *server) applyPlanOverride(poolID int64, reqID uint32, cfg poolParallelism,
	po *planOverride) *PipelinePlan {

	type rigRec struct {
		userID   int64
		hostname string
	}
	rigs := map[string]rigRec{}
	rows, err := s.dbQuery(`
		SELECT r.user_id, r.agent_id, r.hostname FROM pool_rigs pr
		JOIN rigs r ON r.id = pr.rig_id WHERE pr.pool_id = ?`, poolID)
	if err != nil {
		return nil
	}
	for rows.Next() {
		var uid int64
		var aid, host string
		if err := rows.Scan(&uid, &aid, &host); err != nil {
			continue
		}
		rigs[aid] = rigRec{uid, host}
	}
	rows.Close()

	// All pinned rigs must be (a) members and (b) currently online.
	for _, st := range po.Stages {
		r, ok := rigs[st.AgentID]
		if !ok {
			return nil
		}
		if _, online := s.hub.findAgent(r.userID, st.AgentID); !online {
			return nil
		}
	}

	plan := &PipelinePlan{
		ReqID:       reqID,
		PoolID:      poolID,
		ModelName:   cfg.ModelName,
		NLayers:     cfg.NLayers,
		Parallelism: cfg.Parallelism,
	}
	for _, st := range po.Stages {
		r := rigs[st.AgentID]
		plan.Stages = append(plan.Stages, StageAssignment{
			StageIdx:      st.StageIdx,
			LayerLo:       st.LayerLo,
			LayerHi:       st.LayerHi + 1, // override is inclusive; StageAssignment is exclusive
			UserID:        r.userID,
			AgentID:       st.AgentID,
			Hostname:      r.hostname,
			TPSize:        1, // overrides don't currently express TP groups
			TransportHint: "ws",
			TPGroupID:     -1,
		})
	}
	return plan
}

// ─── HTTP handlers ───────────────────────────────────────────────────────

func (s *server) handlePoolPlanGet(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	pid, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil {
		writeErr(w, 400, "bad id")
		return
	}
	vis, _, ok := s.poolVisibility(pid)
	if !ok {
		writeErr(w, 404, "pool not found")
		return
	}
	if _, isMember := s.userIsMember(pid, u.ID); !isMember && vis != "public" {
		writeErr(w, 403, "not a member")
		return
	}
	po, err := s.loadPlanOverride(pid)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	if po == nil {
		writeJSON(w, 200, map[string]any{"pool_id": pid, "override": nil})
		return
	}
	writeJSON(w, 200, map[string]any{"pool_id": pid, "override": po})
}

func (s *server) handlePoolPlanPut(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	pid, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil {
		writeErr(w, 400, "bad id")
		return
	}
	role, isMember := s.userIsMember(pid, u.ID)
	if !isMember || role != "owner" {
		writeErr(w, 403, "only the pool owner can set the plan")
		return
	}

	body, err := io.ReadAll(io.LimitReader(r.Body, 64<<10))
	if err != nil {
		writeErr(w, 400, "read body: "+err.Error())
		return
	}
	var po planOverride
	if err := json.Unmarshal(body, &po); err != nil {
		writeErr(w, 400, "bad json: "+err.Error())
		return
	}

	cfg, err := s.loadPoolParallelism(pid)
	if err != nil {
		writeErr(w, 500, "load pool: "+err.Error())
		return
	}
	if cfg.NLayers <= 0 {
		writeErr(w, 400, "pool has no model bound; bind one before pinning a plan")
		return
	}
	if err := s.validateOverride(pid, cfg.NLayers, &po); err != nil {
		writeErr(w, 400, err.Error())
		return
	}

	canonical, _ := json.Marshal(&po) // re-marshal so we store a canonical form
	if _, err := s.dbExec(`UPDATE pools SET plan_override = ? WHERE id = ?`,
		string(canonical), pid); err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	writeJSON(w, 200, map[string]any{"pool_id": pid, "override": po})
}

func (s *server) handlePoolPlanDelete(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	pid, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil {
		writeErr(w, 400, "bad id")
		return
	}
	role, isMember := s.userIsMember(pid, u.ID)
	if !isMember || role != "owner" {
		writeErr(w, 403, "only the pool owner can clear the plan")
		return
	}
	if _, err := s.dbExec(`UPDATE pools SET plan_override = NULL WHERE id = ?`, pid); err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	writeJSON(w, 200, map[string]any{"pool_id": pid, "override": nil})
}
