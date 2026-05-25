package main

// dpp_rigs.go — GET /api/pools/{id}/rigs/dpp.
//
// Returns the per-rig view the Studio "distributed pipeline" picker needs
// to let a user assign roles (text_encoder / unet / vae) to specific rigs
// in a pool, with enough live-telemetry context (VRAM free, loaded models,
// comfy capability) to make an informed choice.
//
// DPP-eligibility here is intentionally permissive: any online pool rig
// with a positive GPU count is eligible.  The planner will fall back to
// CPU on rigs that report n_gpus_available==0, but the UI should still
// show them so the operator can choose deliberately.

import (
	"encoding/json"
	"net/http"
	"strconv"
)

// dppRigOut is one row in /api/pools/{id}/rigs/dpp.
type dppRigOut struct {
	RigID           int64    `json:"rig_id"`
	OwnerID         int64    `json:"owner_id"`
	AgentID         string   `json:"agent_id"`
	Hostname        string   `json:"hostname"`
	Online          bool     `json:"online"`
	NGPUs           int      `json:"n_gpus"`
	NGPUsAvailable  int      `json:"n_gpus_available"`
	VRAMTotal       int64    `json:"vram_total"`
	VRAMFree        int64    `json:"vram_free"`
	GPUModel        string   `json:"gpu_model,omitempty"`
	TokensPS        float64  `json:"tokens_ps,omitempty"`
	Inflight        int      `json:"inflight"`
	RolesHeld       []string `json:"roles_held,omitempty"`
	ModelsHeldLLM   []string `json:"models_loaded_llm,omitempty"`
	ComfyOK         bool     `json:"comfy_ok"`
	ComfyVersion    string   `json:"comfy_version,omitempty"`
	ComfyModels     []string `json:"comfy_models,omitempty"`
	BWUpKbps        int64    `json:"bw_up_kbps,omitempty"`
	BWDnKbps        int64    `json:"bw_dn_kbps,omitempty"`
	DPPEligible     bool     `json:"dpp_eligible"`
	DPPEligibleNote string   `json:"dpp_eligible_note,omitempty"`
}

// handleDPPRigsForPool — GET /api/pools/{id}/rigs/dpp.
//
// Visibility: pool members + public pools, mirroring handlePoolDetail.
func (s *server) handleDPPRigsForPool(w http.ResponseWriter, r *http.Request) {
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

	// Base query — every pool rig with its at-hello metadata.  comfy_caps
	// joined LEFT so non-comfy rigs still appear (DPP works without comfy).
	rows, err := s.dbQuery(`
		SELECT r.id, r.user_id, r.agent_id, r.hostname,
		       COALESCE(r.n_gpus, 0), COALESCE(r.n_gpus_available, 0),
		       COALESCE(r.vram_bytes, 0),
		       COALESCE(cc.ok, 0), COALESCE(cc.version, ''), COALESCE(cc.models, '')
		FROM pool_rigs pr
		JOIN rigs r       ON r.id = pr.rig_id
		LEFT JOIN comfy_caps cc
		       ON cc.user_id = r.user_id AND cc.agent_id = r.agent_id
		WHERE pr.pool_id = ?
		ORDER BY r.id ASC`, pid)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	defer rows.Close()

	out := make([]dppRigOut, 0)
	for rows.Next() {
		var (
			ro            dppRigOut
			vramHello     int64
			comfyOKInt    int
			comfyModelsJS string
		)
		if err := rows.Scan(
			&ro.RigID, &ro.OwnerID, &ro.AgentID, &ro.Hostname,
			&ro.NGPUs, &ro.NGPUsAvailable, &vramHello,
			&comfyOKInt, &ro.ComfyVersion, &comfyModelsJS,
		); err != nil {
			continue
		}
		ro.VRAMTotal = vramHello
		ro.ComfyOK = comfyOKInt == 1
		if comfyModelsJS != "" {
			var ms []string
			if err := json.Unmarshal([]byte(comfyModelsJS), &ms); err == nil {
				ro.ComfyModels = ms
			}
		}

		// Live telemetry: prefer the agent's most recent status frame if
		// the rig is online — falls back to hello-time vram_bytes if not.
		if a, isOn := s.hub.findAgent(ro.OwnerID, ro.AgentID); isOn {
			ro.Online = true
			st := a.snapshotStatus()
			if st.VRAMTotal > 0 {
				ro.VRAMTotal = st.VRAMTotal
			}
			ro.VRAMFree = st.VRAMFree
			ro.GPUModel = st.GPUModel
			ro.TokensPS = st.TokensPS
			ro.Inflight = st.Inflight
			ro.RolesHeld = append([]string(nil), st.RolesHeld...)
			ro.ModelsHeldLLM = append([]string(nil), st.ModelsHeld...)
			ro.BWUpKbps = st.BWUpKbps
			ro.BWDnKbps = st.BWDnKbps
			if st.NGPUs > 0 {
				ro.NGPUs = st.NGPUs
			}
		}

		// Eligibility — keep the rule transparent so the UI can render a
		// helpful tooltip explaining why a rig is greyed out.
		switch {
		case !ro.Online:
			ro.DPPEligible = false
			ro.DPPEligibleNote = "rig offline"
		case ro.NGPUs == 0:
			// CPU-only — eligible but slow; surface a note rather than
			// hiding so the operator can deliberately place TE/VAE there.
			ro.DPPEligible = true
			ro.DPPEligibleNote = "CPU-only (slow)"
		case ro.VRAMTotal > 0 && ro.VRAMTotal < (4<<30):
			ro.DPPEligible = true
			ro.DPPEligibleNote = "low VRAM (<4 GiB)"
		default:
			ro.DPPEligible = true
		}
		out = append(out, ro)
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"pool_id": pid,
		"rigs":    out,
	})
}

// countDPPRigsInPool returns how many online rigs sit in the given pool.
// Used by the comfy auto-router to decide whether DPP is feasible.  We
// don't require a separate dpp_caps table yet — any online pool rig is
// considered DPP-capable (the runtime probes its own GPU at hello time).
func (s *server) countDPPRigsInPool(poolID int64) int {
	if poolID == 0 {
		return 0
	}
	rows, err := s.dbQuery(`
		SELECT r.user_id, r.agent_id
		FROM pool_rigs pr JOIN rigs r ON r.id = pr.rig_id
		WHERE pr.pool_id = ?`, poolID)
	if err != nil {
		return 0
	}
	defer rows.Close()
	n := 0
	for rows.Next() {
		var uid int64
		var aid string
		if err := rows.Scan(&uid, &aid); err != nil {
			continue
		}
		if s.hub.agentOnline(uid, aid) {
			n++
		}
	}
	return n
}
