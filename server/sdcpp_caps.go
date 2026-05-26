package main

// sdcpp_caps.go — CF12-W4 capability storage + role planner for the C++
// stable-diffusion.cpp backend.
//
// Each agent that ships dist-sdcpp-worker emits an `sdcpp_caps` frame at
// hello time:
//
//   {
//     "kind":  "sdcpp_caps",
//     "ok":    true,
//     "roles": ["full","te","unet","vae"],
//     "worker": "/abs/path/to/dist-sdcpp-worker",
//     "backend":"vulkan:1 cuda:0 cpu:1",
//     "error": ""
//   }
//
// The `roles` field is the per-role split CF12-W3 introduced: a rig that
// reports "te" can run the text-encoder; "unet" can sample; "vae" can
// decode latents.  "full" is the pre-existing whole-pipeline path
// (`sdcpp_route`).  A single rig may advertise multiple roles — the
// resident daemon multiplexes them.
//
// Planner: planSdcppRoleChain picks one rig per role from the pool.  The
// caller (an OpenAI-compat image endpoint, or the dispatcher) decides
// whether to issue a single `sdcpp_route` to a "full" rig or a 3-hop
// `sdcpp_role_route` chain to three different rigs.

import (
	"context"
	"database/sql"
	"encoding/json"
	"net/http"
	"strings"
)

// sdcppRoleTE / UNet / VAE / Full mirror the four canonical roles the
// adapter accepts.  We use strings on the wire to stay forward-compatible
// with future role names (e.g. "controlnet", "vae_encode").
const (
	sdcppRoleTE         = "te"
	sdcppRoleUNet       = "unet"
	sdcppRoleUNetBlocks = "unet_blocks"
	sdcppRoleVAE        = "vae"
	sdcppRoleFull       = "full"
)

// validSdcppRoles is the closed set the planner is willing to consider.
// "unet_blocks" is wire-defined but ungated upstream — a rig advertising
// it today still gets stored; the role-chain planner doesn't pick it
// for TE/UNet/VAE rotations (those use the plain "unet" role), but the
// block-split planner does (see planSdcppUnetSplit).
var validSdcppRoles = map[string]struct{}{
	sdcppRoleTE:         {},
	sdcppRoleUNet:       {},
	sdcppRoleUNetBlocks: {},
	sdcppRoleVAE:        {},
	sdcppRoleFull:       {},
}

func migrateSdcppCaps(db *sql.DB, d sqlDialect) error {
	if d == nil {
		d = sqliteDialect{}
	}
	stmts := []string{
		// Per (user_id, agent_id, role) row.  The "role" column doubles
		// as the discriminator, so a rig that advertises three roles
		// produces three rows.  Lookups by role become a single index
		// scan instead of an N+1 SELECT on a packed-list column.
		`CREATE TABLE IF NOT EXISTS sdcpp_caps (
			user_id     INTEGER NOT NULL,
			agent_id    TEXT    NOT NULL,
			role        TEXT    NOT NULL,
			ok          INTEGER NOT NULL DEFAULT 0,
			worker      TEXT    NOT NULL DEFAULT '',
			backend     TEXT    NOT NULL DEFAULT '',
			updated_at  INTEGER NOT NULL,
			PRIMARY KEY (user_id, agent_id, role)
		)`,
		`CREATE INDEX IF NOT EXISTS idx_sdcpp_caps_role_ok
			ON sdcpp_caps(role, ok, updated_at)`,
		// CF12-A2: sibling table populated from the `models` array each
		// rig ships in its sdcpp_caps frame.  One row per (user_id,
		// agent_id, name) — `path` is the absolute path on that rig, so
		// the dispatcher can pin the chain to rigs that actually have
		// the requested checkpoint.  Lifecycle is identical to sdcpp_caps:
		// blown away + re-inserted on every hello frame.
		`CREATE TABLE IF NOT EXISTS sdcpp_models (
			user_id     INTEGER NOT NULL,
			agent_id    TEXT    NOT NULL,
			name        TEXT    NOT NULL,
			path        TEXT    NOT NULL,
			size        INTEGER NOT NULL DEFAULT 0,
			updated_at  INTEGER NOT NULL,
			PRIMARY KEY (user_id, agent_id, name)
		)`,
		`CREATE INDEX IF NOT EXISTS idx_sdcpp_models_name
			ON sdcpp_models(name, updated_at)`,
	}
	for _, q := range stmts {
		if _, err := db.Exec(d.RewriteDDL(q)); err != nil {
			return err
		}
	}
	return nil
}

// upsertSdcppCaps replaces the (user_id, agent_id) row-set for this rig
// with one row per advertised role.  Idempotent — re-hello'ing a rig
// just bumps updated_at.  A rig that downgrades (drops a role) sees the
// old rows removed in the same transaction.
func (s *server) upsertSdcppCaps(uid int64, agentID string, msg map[string]any) {
	ok := 0
	if v, _ := msg["ok"].(bool); v {
		ok = 1
	}
	worker, _ := msg["worker"].(string)
	backend, _ := msg["backend"].(string)

	roles := []string{}
	if rs, _ := msg["roles"].([]any); rs != nil {
		for _, r := range rs {
			if str, _ := r.(string); str != "" {
				str = strings.ToLower(strings.TrimSpace(str))
				if _, valid := validSdcppRoles[str]; valid {
					roles = append(roles, str)
				}
			}
		}
	}
	// A rig that says ok=false still gets its rows blown away; otherwise
	// stale "ok=1 from last hello" rows would linger after a downgrade.
	if ok == 0 {
		roles = nil
	}

	// Parse the `models` array (CF12-A2).  Each entry is
	// {"name":..., "path":..., "size":...}.  The rig has already filtered
	// to legitimate extensions and a minimum size; we just persist.
	type sdcppModel struct{ Name, Path string; Size int64 }
	var models []sdcppModel
	if rawList, _ := msg["models"].([]any); rawList != nil {
		for _, raw := range rawList {
			entry, _ := raw.(map[string]any)
			if entry == nil {
				continue
			}
			name, _ := entry["name"].(string)
			path, _ := entry["path"].(string)
			if name == "" || path == "" {
				continue
			}
			var size int64
			if v, ok := entry["size"].(float64); ok {
				size = int64(v)
			}
			models = append(models, sdcppModel{Name: name, Path: path, Size: size})
		}
	}

	now := nowUnix()
	// Best-effort transaction: delete then re-insert.  Sqlite doesn't
	// give us a clean ON CONFLICT path when the *role set* changes.
	_, _ = s.dbExec(`DELETE FROM sdcpp_caps WHERE user_id = ? AND agent_id = ?`,
		uid, agentID)
	for _, role := range roles {
		_, _ = s.dbExec(
			`INSERT INTO sdcpp_caps (user_id, agent_id, role, ok, worker, backend, updated_at)
			 VALUES (?, ?, ?, ?, ?, ?, ?)`,
			uid, agentID, role, ok, worker, backend, now,
		)
	}

	// Same delete+re-insert lifecycle for the file advertisement.  A rig
	// that removed a checkpoint between hellos sees its old row drop.
	_, _ = s.dbExec(`DELETE FROM sdcpp_models WHERE user_id = ? AND agent_id = ?`,
		uid, agentID)
	for _, m := range models {
		_, _ = s.dbExec(
			`INSERT INTO sdcpp_models (user_id, agent_id, name, path, size, updated_at)
			 VALUES (?, ?, ?, ?, ?, ?)`,
			uid, agentID, m.Name, m.Path, m.Size, now,
		)
	}
}

// sdcppRoleAgent identifies a rig that has advertised support for one
// role.  Returned by planSdcppRoleChain; the dispatcher uses AgentID to
// look up the live WS conn.
type sdcppRoleAgent struct {
	UserID  int64
	AgentID string
	Role    string
	Worker  string
	Backend string
	// ModelPath is the rig-local absolute path to the checkpoint, taken
	// from sdcpp_models when the planner was called with a non-empty
	// modelName.  Empty when the caller is using the legacy path-from-
	// request flow.
	ModelPath string
}

// planSdcppRoleChain returns up to three rigs covering te/unet/vae for
// the given owner.  Order: TE first, then UNet, then VAE.  If a single
// rig covers two roles it may appear twice in the slice — the caller
// dedups by AgentID if it cares.  Returns nil when any role can't be
// filled (caller falls back to dpp_route or sdcpp_route "full").
//
// When modelName is non-empty, only rigs that have advertised that
// file in sdcpp_models are considered, and the picked agent's
// ModelPath is populated with the rig-local absolute path the file
// scanner reported (CF12-A2).  When modelName == "" the legacy
// caller-supplies-path behaviour is preserved.
//
// We deliberately don't require the same backbone-family at this
// layer: the same dist-sdcpp-worker can serve any sd.cpp-supported
// model, and model_path is part of the per-request `sdcpp_role_route`.
func (s *server) planSdcppRoleChain(ctx context.Context, ownerUID int64, modelName string) []sdcppRoleAgent {
	pick := func(role string) (sdcppRoleAgent, bool) {
		var rows *sql.Rows
		var err error
		if modelName == "" {
			rows, err = s.dbQueryCtx(ctx,
				`SELECT c.user_id, c.agent_id, c.worker, c.backend, ''
				   FROM sdcpp_caps c
				  WHERE c.role = ? AND c.ok = 1
				  ORDER BY (c.user_id = ?) DESC, c.updated_at DESC
				  LIMIT 1`,
				role, ownerUID)
		} else {
			rows, err = s.dbQueryCtx(ctx,
				`SELECT c.user_id, c.agent_id, c.worker, c.backend, m.path
				   FROM sdcpp_caps c
				   JOIN sdcpp_models m
				     ON m.user_id = c.user_id AND m.agent_id = c.agent_id
				  WHERE c.role = ? AND c.ok = 1 AND m.name = ?
				  ORDER BY (c.user_id = ?) DESC, c.updated_at DESC
				  LIMIT 1`,
				role, modelName, ownerUID)
		}
		if err != nil {
			return sdcppRoleAgent{}, false
		}
		defer rows.Close()
		if !rows.Next() {
			return sdcppRoleAgent{}, false
		}
		var a sdcppRoleAgent
		a.Role = role
		if err := rows.Scan(&a.UserID, &a.AgentID, &a.Worker, &a.Backend, &a.ModelPath); err != nil {
			return sdcppRoleAgent{}, false
		}
		return a, true
	}

	te, ok := pick(sdcppRoleTE)
	if !ok {
		return nil
	}
	unet, ok := pick(sdcppRoleUNet)
	if !ok {
		return nil
	}
	vae, ok := pick(sdcppRoleVAE)
	if !ok {
		return nil
	}
	return []sdcppRoleAgent{te, unet, vae}
}

// sdcppRoleRouteParams is the per-request payload shared by the three
// hops (TE → UNet → VAE).  Binary inputs (SDCD cond from TE, latent
// SDT from UNet) are base64 strings — the adapter passes them through
// to the resident sdcpp daemon unchanged.
type sdcppRoleRouteParams struct {
	ReqID      uint16
	ModelPath  string
	Prompt     string  // TE only
	Negative   string  // TE only
	CFGSplit   bool    // TE only
	ClipSkip   int     // TE only (-1 = model default)
	SDCDB64    string  // UNet input (output of TE)
	SDTB64     string  // VAE input (output of UNet)
	Width      int
	Height     int
	Steps      int
	CFG        float64
	Seed       int64
	Sampler    string  // "" → adapter default
	Scheduler  string  // "" → adapter default
	Threads    int     // 0 → worker default
}

// dispatchSdcppRoleRoute composes the `sdcpp_role_route` control frame
// for one hop in the chain and ships it to the agent.  Returns false if
// the agent is offline (the dispatcher should retry-plan with a fresh
// pick).
func (s *server) dispatchSdcppRoleRoute(target sdcppRoleAgent, p sdcppRoleRouteParams) bool {
	ac, ok := s.hub.findAgent(target.UserID, target.AgentID)
	if !ok {
		return false
	}
	msg := map[string]any{
		"kind":       "sdcpp_role_route",
		"req_id":     int(p.ReqID),
		"role":       target.Role,
		"model_path": p.ModelPath,
	}
	if p.Threads > 0 {
		msg["threads"] = p.Threads
	}
	switch target.Role {
	case sdcppRoleTE:
		msg["prompt"] = p.Prompt
		if p.Negative != "" {
			msg["negative_prompt"] = p.Negative
		}
		if p.CFGSplit {
			msg["cfg_split"] = 1
		}
		if p.ClipSkip != 0 {
			msg["clip_skip"] = p.ClipSkip
		}
	case sdcppRoleUNet:
		msg["sdcd_b64"] = p.SDCDB64
		msg["width"] = p.Width
		msg["height"] = p.Height
		msg["steps"] = p.Steps
		msg["cfg"] = p.CFG
		msg["seed"] = p.Seed
		if p.Sampler != "" {
			msg["sampler"] = p.Sampler
		}
		if p.Scheduler != "" {
			msg["scheduler"] = p.Scheduler
		}
	case sdcppRoleVAE:
		msg["sdt_b64"] = p.SDTB64
	}
	ac.send(msg)
	return true
}

// sdcppUnetBlockRange is one stage of a within-UNet split: a contiguous
// transformer-block window [BlockLo, BlockHi) served by one rig.  N such
// ranges sum to [0, BlockTotal) with no gaps.
type sdcppUnetBlockRange struct {
	Agent      sdcppRoleAgent
	BlockLo    int
	BlockHi    int
	BlockTotal int
}

// planSdcppUnetSplit picks N rigs to share the UNet transformer-block
// range and partitions the blocks evenly between them.  Returns nil when:
//   - fewer than 1 unet rig has advertised the "unet_blocks" role
//     (this role is gated behind the W6a upstream patch — until then
//     the planner is a no-op and the dispatcher uses the single-UNet
//     role chain instead)
//   - totalBlocks <= 0 or stages <= 0
//
// Today no rig advertises "unet_blocks": the gating ensures the wire
// path is exercised end-to-end (planner inputs → range partitioning →
// adapter cmd → ENOTIMPL fallback) without lying about backbone support.
// When CF12-W6a lands the planner picks up the new caps with no other
// code-path changes.
func (s *server) planSdcppUnetSplit(ctx context.Context, ownerUID int64,
	totalBlocks, stages int) []sdcppUnetBlockRange {
	if totalBlocks <= 0 || stages <= 0 {
		return nil
	}
	rows, err := s.dbQueryCtx(ctx,
		`SELECT user_id, agent_id, worker, backend
		   FROM sdcpp_caps
		  WHERE role = 'unet_blocks' AND ok = 1
		  ORDER BY (user_id = ?) DESC, updated_at DESC
		  LIMIT ?`,
		ownerUID, stages)
	if err != nil {
		return nil
	}
	defer rows.Close()
	var picked []sdcppRoleAgent
	for rows.Next() {
		var a sdcppRoleAgent
		a.Role = "unet_blocks"
		if err := rows.Scan(&a.UserID, &a.AgentID, &a.Worker, &a.Backend); err != nil {
			return nil
		}
		picked = append(picked, a)
	}
	if len(picked) < stages {
		return nil
	}

	// Partition `totalBlocks` blocks across `stages` rigs.  Same heuristic
	// as partitionUNetBlocks for python (server/dpp_unet_split.go) — try
	// to keep stage sizes even and front-load the remainder so the first
	// stage owns the input conv block.
	out := make([]sdcppUnetBlockRange, stages)
	base := totalBlocks / stages
	extra := totalBlocks % stages
	cursor := 0
	for i := 0; i < stages; i++ {
		size := base
		if i < extra {
			size++
		}
		out[i] = sdcppUnetBlockRange{
			Agent:      picked[i],
			BlockLo:    cursor,
			BlockHi:    cursor + size,
			BlockTotal: totalBlocks,
		}
		cursor += size
	}
	return out
}

// sdcppRigAdvertisesModel reports whether any of the user's rigs has
// the given model file in sdcpp_models.  Used by the routing layer to
// accept rig-advertised file names (like "sdxl-base.safetensors") as
// valid sd.cpp model identifiers in addition to the heuristic eligible
// HF ids.
func (s *server) sdcppRigAdvertisesModel(ownerUID int64, name string) bool {
	if name == "" {
		return false
	}
	var n int
	_ = s.dbQueryRow(
		`SELECT COUNT(*) FROM sdcpp_models WHERE user_id = ? AND name = ?`,
		ownerUID, name).Scan(&n)
	return n > 0
}

// countSdcppRigsForUser returns how many distinct online rigs the user
// has with at least one sd.cpp role advertised (ok=1).  Used by the
// comfy auto-router to decide whether the sd.cpp backend is feasible.
func (s *server) countSdcppRigsForUser(ownerUID int64) int {
	rows, err := s.dbQuery(
		`SELECT DISTINCT agent_id, user_id FROM sdcpp_caps
		  WHERE ok = 1 AND user_id = ?`, ownerUID)
	if err != nil {
		return 0
	}
	defer rows.Close()
	n := 0
	for rows.Next() {
		var aid string
		var uid int64
		if err := rows.Scan(&aid, &uid); err != nil {
			continue
		}
		if s.hub.agentOnline(uid, aid) {
			n++
		}
	}
	return n
}

// planSdcppFull picks a single rig that can run the whole sd.cpp
// pipeline.  Used when the request is small enough to fit one rig — or
// when the role chain can't be filled.  When modelName is non-empty
// the picked rig is required to have advertised the file in
// sdcpp_models; ModelPath on the returned agent is the absolute path
// on that rig.
func (s *server) planSdcppFull(ctx context.Context, ownerUID int64, modelName string) (sdcppRoleAgent, bool) {
	var rows *sql.Rows
	var err error
	if modelName == "" {
		rows, err = s.dbQueryCtx(ctx,
			`SELECT c.user_id, c.agent_id, c.worker, c.backend, ''
			   FROM sdcpp_caps c
			  WHERE c.role = ? AND c.ok = 1
			  ORDER BY (c.user_id = ?) DESC, c.updated_at DESC
			  LIMIT 1`,
			sdcppRoleFull, ownerUID)
	} else {
		rows, err = s.dbQueryCtx(ctx,
			`SELECT c.user_id, c.agent_id, c.worker, c.backend, m.path
			   FROM sdcpp_caps c
			   JOIN sdcpp_models m
			     ON m.user_id = c.user_id AND m.agent_id = c.agent_id
			  WHERE c.role = ? AND c.ok = 1 AND m.name = ?
			  ORDER BY (c.user_id = ?) DESC, c.updated_at DESC
			  LIMIT 1`,
			sdcppRoleFull, modelName, ownerUID)
	}
	if err != nil {
		return sdcppRoleAgent{}, false
	}
	defer rows.Close()
	if !rows.Next() {
		return sdcppRoleAgent{}, false
	}
	var a sdcppRoleAgent
	a.Role = sdcppRoleFull
	if err := rows.Scan(&a.UserID, &a.AgentID, &a.Worker, &a.Backend, &a.ModelPath); err != nil {
		return sdcppRoleAgent{}, false
	}
	return a, true
}

// sdcppModelInfo is one row in /api/sdcpp/models.
type sdcppModelInfo struct {
	Name   string   `json:"name"`
	OnRigs []string `json:"on_rigs"` // agent IDs that have advertised the file
	Size   int64    `json:"size"`    // largest size seen (rigs may have re-quantised copies)
}

// handleSdcppModels — GET /api/sdcpp/models.
//
// Returns the union of files advertised in sdcpp_models for this user's
// rigs.  Only currently-online rigs count toward `on_rigs`: a file on a
// disconnected rig is unreachable until it reconnects, and surfacing it
// in the UI would invite a user to pick a model that immediately fails
// to dispatch.
func (s *server) handleSdcppModels(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "auth required")
		return
	}
	rows, err := s.dbQuery(
		`SELECT name, agent_id, user_id, size FROM sdcpp_models
		  WHERE user_id = ?
		  ORDER BY name ASC`, u.ID)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	defer rows.Close()
	idx := map[string]*sdcppModelInfo{}
	var ordered []string
	for rows.Next() {
		var name, agentID string
		var uid, size int64
		if err := rows.Scan(&name, &agentID, &uid, &size); err != nil {
			continue
		}
		if !s.hub.agentOnline(uid, agentID) {
			continue
		}
		entry, ok := idx[name]
		if !ok {
			entry = &sdcppModelInfo{Name: name}
			idx[name] = entry
			ordered = append(ordered, name)
		}
		entry.OnRigs = append(entry.OnRigs, agentID)
		if size > entry.Size {
			entry.Size = size
		}
	}
	out := make([]sdcppModelInfo, 0, len(ordered))
	for _, name := range ordered {
		out = append(out, *idx[name])
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"models": out})
}
