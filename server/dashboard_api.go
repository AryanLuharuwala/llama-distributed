package main

// Dashboard-facing read-only APIs.
//
// Feeds the futuristic /console UI (model loading, rig popups, network
// graph, NAT/firewall overlay, traffic routing).  Everything here is a
// thin projection over hub + db state — no new persistence.
//
// Endpoints:
//   GET /api/me/rigs              — list rigs owned by current user + live telemetry
//   GET /api/me/rigs/stream       — SSE stream of /api/me/rigs deltas
//   GET /api/me/earnings          — daily/weekly/monthly token contribution per rig
//   GET /api/pools/{id}/topology  — current pipeline plan (stages, layers, peers)
//   GET /api/pools/{id}/sessions  — live inference slots per rig in the pool
//   GET /api/console/network      — graph view: nodes + edges with NAT + relay info

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// ─── /api/me/rigs (rich variant of /api/rigs) ─────────────────────────────

type meRig struct {
	AgentID   string   `json:"agent_id"`
	Hostname  string   `json:"hostname"`
	NGPUs     int      `json:"n_gpus"`
	GPUModel  string   `json:"gpu_model,omitempty"`
	VRAMTotal int64    `json:"vram_total"`
	VRAMFree  int64    `json:"vram_free,omitempty"`
	TokensPS  float64  `json:"tokens_sec"`
	UptimeSec int64    `json:"uptime_sec"`
	BWUpKbps  int64    `json:"bw_up_kbps"`
	BWDnKbps  int64    `json:"bw_dn_kbps"`
	Inflight  int      `json:"inflight"`
	MaxConc   int      `json:"max_concurrent"`
	Roles     []string `json:"roles,omitempty"`
	Models    []string `json:"models,omitempty"`
	LastSeen  int64    `json:"last_seen"`
	Online    bool     `json:"online"`

	// Network / NAT health.  Drives the firewall + relay status pill in
	// each rig popup.
	NATType      string `json:"nat_type"`
	RelayCapable bool   `json:"relay_capable"`
	CoturnPort   int    `json:"coturn_port"`
	PublicIP     string `json:"public_ip,omitempty"`
	RemoteIP     string `json:"remote_ip,omitempty"`

	// Derived health classification: "ok" | "stale" | "saturated" | "offline".
	// The frontend uses this to pick a single colour for the rig tile.
	Health string `json:"health"`

	// First pool this rig is attached to (alphabetically by id).  Empty if the
	// rig isn't pooled yet.  The console table uses this to show *where* a rig
	// is wired up so users don't have to cross-check from /nexus.
	PoolID   int64  `json:"pool_id,omitempty"`
	PoolName string `json:"pool_name,omitempty"`
}

func (s *server) collectMeRigs(uid int64) []meRig {
	rows, err := s.db.Query(
		`SELECT agent_id, hostname, n_gpus, vram_bytes, last_seen
		 FROM rigs WHERE user_id = ? ORDER BY last_seen DESC`,
		uid,
	)
	if err != nil {
		return nil
	}
	defer rows.Close()
	out := []meRig{}
	for rows.Next() {
		var r meRig
		if err := rows.Scan(&r.AgentID, &r.Hostname, &r.NGPUs, &r.VRAMTotal, &r.LastSeen); err != nil {
			continue
		}
		if ac, ok := s.hub.findAgent(uid, r.AgentID); ok {
			r.Online = true
			st := ac.snapshotStatus()
			r.GPUModel = st.GPUModel
			if st.VRAMTotal > 0 {
				r.VRAMTotal = st.VRAMTotal
			}
			r.VRAMFree = st.VRAMFree
			r.TokensPS = st.TokensPS
			r.UptimeSec = st.UptimeSec
			r.BWUpKbps = st.BWUpKbps
			r.BWDnKbps = st.BWDnKbps
			r.Inflight = st.Inflight
			r.MaxConc = st.MaxConcurrent
			r.Roles = st.RolesHeld
			r.Models = st.ModelsHeld
			r.NATType = st.NATType
			r.RelayCapable = st.RelayCapable
			r.CoturnPort = st.CoturnPort
			r.PublicIP = st.PublicIP
			r.RemoteIP = ac.remoteIP
		}
		r.Health = classifyRigHealth(r)
		// First pool membership for this rig, if any.  Lets the console table
		// show a "Pool" column without a second round-trip.
		_ = s.db.QueryRow(`
			SELECT p.id, p.name
			FROM pool_rigs pr
			JOIN rigs r2 ON r2.id = pr.rig_id
			JOIN pools p ON p.id = pr.pool_id
			WHERE r2.user_id = ? AND r2.agent_id = ?
			ORDER BY p.id ASC LIMIT 1`,
			uid, r.AgentID,
		).Scan(&r.PoolID, &r.PoolName)
		out = append(out, r)
	}
	return out
}

// classifyRigHealth boils a rig's telemetry down to one of four buckets.
// Picks the *worst* signal so a saturated-but-stale rig still flags as stale.
func classifyRigHealth(r meRig) string {
	if !r.Online {
		return "offline"
	}
	// Stale telemetry: no status frame for >60s.
	if r.LastSeen > 0 && nowUnix()-r.LastSeen > 60 {
		return "stale"
	}
	// Saturated: inflight at or above advertised slot count.
	if r.MaxConc > 0 && r.Inflight >= r.MaxConc {
		return "saturated"
	}
	// VRAM <10% free is also a saturation indicator.
	if r.VRAMTotal > 0 && r.VRAMFree > 0 && r.VRAMFree*10 < r.VRAMTotal {
		return "saturated"
	}
	return "ok"
}

func (s *server) handleMeRigs(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	writeJSON(w, 200, map[string]any{
		"rigs":         s.collectMeRigs(u.ID),
		"generated_at": nowUnix(),
	})
}

// ─── /api/me/rigs/stream (SSE) ────────────────────────────────────────────
//
// Push the same payload every 2s (or on disconnect) without polling.
// Browsers reconnect automatically; we close cleanly on ctx cancel.

func (s *server) handleMeRigsStream(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	f, ok := w.(http.Flusher)
	if !ok {
		writeErr(w, 500, "streaming unsupported")
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no") // disable nginx buffering

	ctx := r.Context()
	tick := time.NewTicker(2 * time.Second)
	defer tick.Stop()

	send := func() {
		buf, err := json.Marshal(map[string]any{
			"rigs":         s.collectMeRigs(u.ID),
			"generated_at": nowUnix(),
		})
		if err != nil {
			return
		}
		fmt.Fprintf(w, "event: rigs\ndata: %s\n\n", buf)
		f.Flush()
	}
	send() // initial frame so the UI can render immediately
	for {
		select {
		case <-ctx.Done():
			return
		case <-tick.C:
			send()
		}
	}
}

// ─── /api/me/earnings ─────────────────────────────────────────────────────
//
// Aggregates inference_log rows where the *agent* belongs to the requesting
// user (agent_user_id = uid).  The compute-side of the economy: tokens you
// produced for other people's requests.  Self-served jobs don't count
// (would inflate earnings); we filter user_id != agent_user_id.

type earningsBucket struct {
	Period       string `json:"period"`
	Requests     int64  `json:"requests"`
	InputTokens  int64  `json:"input_tokens"`
	OutputTokens int64  `json:"output_tokens"`
}

func (s *server) handleMeEarnings(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	// We bucket the last 30 days by UTC date string and additionally roll up
	// per-rig totals.  Both are cheap; UI renders a sparkline + per-rig bars.
	since := time.Now().Add(-30 * 24 * time.Hour).Unix()
	rows, err := s.db.Query(`
		SELECT strftime('%Y-%m-%d', started_at, 'unixepoch') AS day,
		       COUNT(*), COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0)
		FROM inference_log
		WHERE agent_user_id = ? AND user_id != agent_user_id AND started_at >= ?
		GROUP BY day ORDER BY day ASC`, u.ID, since)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	defer rows.Close()
	var daily []earningsBucket
	var totalReq, totalIn, totalOut int64
	for rows.Next() {
		var b earningsBucket
		if err := rows.Scan(&b.Period, &b.Requests, &b.InputTokens, &b.OutputTokens); err == nil {
			daily = append(daily, b)
			totalReq += b.Requests
			totalIn += b.InputTokens
			totalOut += b.OutputTokens
		}
	}

	// Per-rig roll-up (last 30d), so the UI can show which of your rigs
	// did the heavy lifting.
	type rigEarn struct {
		AgentID      string `json:"agent_id"`
		Hostname     string `json:"hostname"`
		Requests     int64  `json:"requests"`
		InputTokens  int64  `json:"input_tokens"`
		OutputTokens int64  `json:"output_tokens"`
	}
	prows, err := s.db.Query(`
		SELECT il.agent_id, COALESCE(r.hostname, ''),
		       COUNT(*), COALESCE(SUM(il.input_tokens),0), COALESCE(SUM(il.output_tokens),0)
		FROM inference_log il
		LEFT JOIN rigs r ON r.user_id = il.agent_user_id AND r.agent_id = il.agent_id
		WHERE il.agent_user_id = ? AND il.user_id != il.agent_user_id AND il.started_at >= ?
		GROUP BY il.agent_id ORDER BY (COALESCE(SUM(il.input_tokens),0)+COALESCE(SUM(il.output_tokens),0)) DESC`,
		u.ID, since)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	defer prows.Close()
	var perRig []rigEarn
	for prows.Next() {
		var e rigEarn
		if err := prows.Scan(&e.AgentID, &e.Hostname, &e.Requests, &e.InputTokens, &e.OutputTokens); err == nil {
			perRig = append(perRig, e)
		}
	}

	// Per-pool roll-up.  Tells the operator which pools their rigs earned
	// from in the last 30d — useful when you contribute the same rig to
	// multiple pools and want to see which is generating real traffic.
	// Joined against pools so the UI can render the pool's display name
	// without a second round trip.
	type poolEarn struct {
		PoolID       int64  `json:"pool_id"`
		PoolName     string `json:"pool_name"`
		Requests     int64  `json:"requests"`
		InputTokens  int64  `json:"input_tokens"`
		OutputTokens int64  `json:"output_tokens"`
	}
	qrows, err := s.db.Query(`
		SELECT il.pool_id, COALESCE(p.name, ''),
		       COUNT(*), COALESCE(SUM(il.input_tokens),0), COALESCE(SUM(il.output_tokens),0)
		FROM inference_log il
		LEFT JOIN pools p ON p.id = il.pool_id
		WHERE il.agent_user_id = ? AND il.user_id != il.agent_user_id AND il.started_at >= ?
		GROUP BY il.pool_id
		ORDER BY (COALESCE(SUM(il.input_tokens),0)+COALESCE(SUM(il.output_tokens),0)) DESC`,
		u.ID, since)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	defer qrows.Close()
	var perPool []poolEarn
	for qrows.Next() {
		var e poolEarn
		if err := qrows.Scan(&e.PoolID, &e.PoolName, &e.Requests, &e.InputTokens, &e.OutputTokens); err == nil {
			perPool = append(perPool, e)
		}
	}

	writeJSON(w, 200, map[string]any{
		"window_days": 30,
		"totals": map[string]int64{
			"requests":      totalReq,
			"input_tokens":  totalIn,
			"output_tokens": totalOut,
		},
		"daily":    daily,
		"per_rig":  perRig,
		"per_pool": perPool,
	})
}

// ─── /api/pools/{id}/topology ─────────────────────────────────────────────
//
// Runs the planner *without* dispatching anything and returns the stage
// assignment + per-stage layer range + which rigs the planner would pick.
// Powers the "network compute path" visualization.

func (s *server) handlePoolTopology(w http.ResponseWriter, r *http.Request) {
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
	_, isMember := s.userIsMember(pid, u.ID)
	if !isMember && vis != "public" {
		writeErr(w, 403, "not a member")
		return
	}

	// Plan once.  reqID=0 — the planner uses it only for shard URL minting,
	// which we strip below so we don't leak signed URLs to the dashboard.
	plan, err := s.planPipeline(pid, 0, "", 0)
	if err != nil {
		// Pool may simply have no rigs yet — return an empty plan instead
		// of 500 so the dashboard can show a friendly "waiting for rigs"
		// state.
		writeJSON(w, 200, map[string]any{
			"pool_id": pid,
			"stages":  []any{},
			"note":    err.Error(),
		})
		return
	}
	// Sanitize: strip shard URLs (signed, short-lived, not the dashboard's
	// business) and enrich each stage with live status from the hub so the
	// graph can colour edges by current load.
	type stageOut struct {
		StageIdx     int      `json:"stage_idx"`
		LayerLo      int      `json:"layer_lo"`
		LayerHi      int      `json:"layer_hi"`
		AgentID      string   `json:"agent_id"`
		Hostname     string   `json:"hostname"`
		TPSize       int      `json:"tp_size"`
		TPGroupID    int      `json:"tp_group_id"`
		TPRank       int      `json:"tp_rank"`
		TPPeers      []string `json:"tp_peers,omitempty"`
		Online       bool     `json:"online"`
		Inflight     int      `json:"inflight"`
		TokensPS     float64  `json:"tokens_sec"`
		NATType      string   `json:"nat_type"`
		PublicIP     string   `json:"public_ip,omitempty"`
		RelayCapable bool     `json:"relay_capable"`
	}
	stages := make([]stageOut, 0, len(plan.Stages))
	for _, st := range plan.Stages {
		o := stageOut{
			StageIdx:  st.StageIdx,
			LayerLo:   st.LayerLo,
			LayerHi:   st.LayerHi,
			AgentID:   st.AgentID,
			Hostname:  st.Hostname,
			TPSize:    st.TPSize,
			TPGroupID: st.TPGroupID,
			TPRank:    st.TPRank,
			TPPeers:   st.TPPeers,
		}
		if ac, ok := s.hub.findAgent(st.UserID, st.AgentID); ok {
			o.Online = true
			ls := ac.snapshotStatus()
			o.Inflight = ls.Inflight
			o.TokensPS = ls.TokensPS
			o.NATType = ls.NATType
			o.PublicIP = ls.PublicIP
			o.RelayCapable = ls.RelayCapable
		}
		// Public-fingerprint the agent_id so an unrelated pool member can't
		// correlate it with a private rig list.
		o.AgentID = publicAgentID(0, o.AgentID)
		stages = append(stages, o)
	}
	writeJSON(w, 200, map[string]any{
		"pool_id":     pid,
		"model":       plan.ModelName,
		"n_layers":    plan.NLayers,
		"parallelism": plan.Parallelism,
		"stages":      stages,
	})
}

// ─── /api/pools/{id}/sessions ─────────────────────────────────────────────
//
// Live snapshot of every rig in this pool with current load.  Powers the
// "where is the compute running right now" view — bigger dots for busier
// rigs, edge thickness from BW.

type poolSession struct {
	AgentID    string   `json:"agent_id"`
	Hostname   string   `json:"hostname"`
	Online     bool     `json:"online"`
	NGPUs      int      `json:"n_gpus"`
	VRAMTotal  int64    `json:"vram_total"`
	VRAMFree   int64    `json:"vram_free"`
	Inflight   int      `json:"inflight"`
	MaxConc    int      `json:"max_concurrent"`
	TokensPS   float64  `json:"tokens_sec"`
	NATType    string   `json:"nat_type"`
	CoturnPort int      `json:"coturn_port"`
	BWUpKbps   int64    `json:"bw_up_kbps"`
	BWDnKbps   int64    `json:"bw_dn_kbps"`
	Roles      []string `json:"roles,omitempty"`
	Bottleneck string   `json:"bottleneck"`
}

// bottleneck classifies what's pinning a rig:
//
//	 "vram" / "compute" / "network" / "none".  Frontend uses this to mark
//	the rig tile with a red badge so users can see "ah this rig is slow
//	 because it's saturated on VRAM" at a glance.
func bottleneckFor(p poolSession) string {
	if !p.Online {
		return ""
	}
	if p.VRAMTotal > 0 && p.VRAMFree*10 < p.VRAMTotal { // <10% VRAM free
		return "vram"
	}
	if p.MaxConc > 0 && p.Inflight >= p.MaxConc {
		return "compute"
	}
	// "network" if the rig reports an upload BW < 1 Mbps under load.
	if p.Inflight > 0 && p.BWUpKbps > 0 && p.BWUpKbps < 1024 {
		return "network"
	}
	return "none"
}

func (s *server) handlePoolSessions(w http.ResponseWriter, r *http.Request) {
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
	_, isMember := s.userIsMember(pid, u.ID)
	if !isMember && vis != "public" {
		writeErr(w, 403, "not a member")
		return
	}

	rows, err := s.db.Query(`
		SELECT r.user_id, r.agent_id, r.hostname
		FROM pool_rigs pr JOIN rigs r ON r.id = pr.rig_id
		WHERE pr.pool_id = ?`, pid)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	defer rows.Close()
	var sessions []poolSession
	for rows.Next() {
		var (
			ownerID  int64
			agentID  string
			hostname string
		)
		if err := rows.Scan(&ownerID, &agentID, &hostname); err != nil {
			continue
		}
		ps := poolSession{
			AgentID:  publicAgentID(ownerID, agentID),
			Hostname: hostname,
		}
		if ac, ok := s.hub.findAgent(ownerID, agentID); ok {
			ps.Online = true
			st := ac.snapshotStatus()
			ps.NGPUs = st.NGPUs
			ps.VRAMTotal = st.VRAMTotal
			ps.VRAMFree = st.VRAMFree
			ps.Inflight = st.Inflight
			ps.MaxConc = st.MaxConcurrent
			ps.TokensPS = st.TokensPS
			ps.NATType = st.NATType
			ps.CoturnPort = st.CoturnPort
			ps.BWUpKbps = st.BWUpKbps
			ps.BWDnKbps = st.BWDnKbps
			ps.Roles = st.RolesHeld
		}
		ps.Bottleneck = bottleneckFor(ps)
		sessions = append(sessions, ps)
	}
	writeJSON(w, 200, map[string]any{
		"pool_id":      pid,
		"sessions":     sessions,
		"generated_at": nowUnix(),
	})
}

// ─── /api/console/network ─────────────────────────────────────────────────
//
// Graph-shape blob for the network visualization.  Returns nodes (rigs
// the user owns or shares pools with), edges between rigs that participate
// in the same pool, and a flag per edge for "relayed" (one of them is a
// relay-capable rig and at least one peer has symmetric NAT).

type netNode struct {
	ID           string  `json:"id"` // publicAgentID
	Hostname     string  `json:"hostname"`
	NATType      string  `json:"nat_type"`
	PublicIP     string  `json:"public_ip,omitempty"`
	CoturnPort   int     `json:"coturn_port"`
	RelayCapable bool    `json:"relay_capable"`
	Online       bool    `json:"online"`
	VRAMTotal    int64   `json:"vram_total"`
	TokensPS     float64 `json:"tokens_sec"`
	Inflight     int     `json:"inflight"`
	Owner        bool    `json:"owner"`
	Pools        []int64 `json:"pools,omitempty"`
}

type netEdge struct {
	A         string `json:"a"`
	B         string `json:"b"`
	PoolID    int64  `json:"pool_id"`
	Relayed   bool   `json:"relayed"`
	RelayHint string `json:"relay_hint,omitempty"` // "turn", "peer", ""
}

func (s *server) handleConsoleNetwork(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}

	// Pools this user can see (owned + member-of).  We use this as the
	// edge fabric — two rigs are connected iff they're in the same visible
	// pool.
	prows, err := s.db.Query(`
		SELECT DISTINCT p.id
		FROM pools p
		LEFT JOIN pool_members m ON m.pool_id = p.id
		WHERE p.owner_id = ? OR m.user_id = ?`, u.ID, u.ID)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	defer prows.Close()
	var poolIDs []int64
	for prows.Next() {
		var pid int64
		if err := prows.Scan(&pid); err == nil {
			poolIDs = append(poolIDs, pid)
		}
	}

	// Map: publicID -> netNode.  Build by walking pool_rigs for visible pools.
	nodes := map[string]*netNode{}
	rigPools := map[string]map[int64]bool{}
	for _, pid := range poolIDs {
		rrows, err := s.db.Query(`
			SELECT r.user_id, r.agent_id, r.hostname
			FROM pool_rigs pr JOIN rigs r ON r.id = pr.rig_id
			WHERE pr.pool_id = ?`, pid)
		if err != nil {
			continue
		}
		for rrows.Next() {
			var ownerID int64
			var agentID, hostname string
			if err := rrows.Scan(&ownerID, &agentID, &hostname); err != nil {
				continue
			}
			id := publicAgentID(ownerID, agentID)
			n, ok := nodes[id]
			if !ok {
				n = &netNode{ID: id, Hostname: hostname, Owner: ownerID == u.ID}
				if ac, ok := s.hub.findAgent(ownerID, agentID); ok {
					n.Online = true
					st := ac.snapshotStatus()
					n.NATType = st.NATType
					n.PublicIP = st.PublicIP
					n.CoturnPort = st.CoturnPort
					n.RelayCapable = st.RelayCapable
					n.VRAMTotal = st.VRAMTotal
					n.TokensPS = st.TokensPS
					n.Inflight = st.Inflight
				}
				nodes[id] = n
			}
			if rigPools[id] == nil {
				rigPools[id] = map[int64]bool{}
			}
			rigPools[id][pid] = true
		}
		rrows.Close()
	}

	// Edge generation: for each pool, fully-connect its rigs.  Mark edges
	// "relayed" when at least one endpoint reports symmetric NAT (needs
	// TURN) and at least one node in the graph is relay-capable.
	type pair struct{ a, b string }
	seen := map[pair]bool{}
	var edges []netEdge
	for _, pid := range poolIDs {
		// All rigs in this pool.
		ids := []string{}
		for id, ps := range rigPools {
			if ps[pid] {
				ids = append(ids, id)
			}
		}
		for i := 0; i < len(ids); i++ {
			for j := i + 1; j < len(ids); j++ {
				a, b := ids[i], ids[j]
				if a > b {
					a, b = b, a
				}
				key := pair{a, b}
				if seen[key] {
					continue
				}
				seen[key] = true
				na, nb := nodes[a], nodes[b]
				relayed := false
				hint := ""
				if na.NATType == "symmetric" || nb.NATType == "symmetric" {
					relayed = true
					hint = "turn"
				}
				edges = append(edges, netEdge{
					A: a, B: b, PoolID: pid, Relayed: relayed, RelayHint: hint,
				})
			}
		}
	}

	// Populate Pools on each node + flatten map → slice.
	var nodeList []netNode
	for id, n := range nodes {
		for pid := range rigPools[id] {
			n.Pools = append(n.Pools, pid)
		}
		nodeList = append(nodeList, *n)
	}

	writeJSON(w, 200, map[string]any{
		"nodes":        nodeList,
		"edges":        edges,
		"generated_at": nowUnix(),
	})
}

// ─── /console — futuristic dashboard HTML ─────────────────────────────────

//go:embed assets/console.html
var consoleHTML []byte

func (s *server) handleConsolePage(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write(consoleHTML)
}

//go:embed assets/observatory.html
var observatoryHTML []byte

// /observatory — tactical operations dashboard.  A distinct aesthetic
// from /console: amber/copper command-deck palette, layer-by-layer model
// visualizer, NAT/firewall matrix, route attribution, per-rig modal.
// Same data sources as /console — pure JS composition, no new server endpoints.
func (s *server) handleObservatoryPage(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write(observatoryHTML)
}

//go:embed assets/playground.html
var playgroundHTML []byte

// /playground — chat + image/video studio + HuggingFace discovery.
// Three tabs: Chat (LLM testing via /api/infer), Studio (image/video gen
// via /api/comfy/generate), Discover (HF model search w/ tag + category
// filters via /api/hf/search).  Same paper-white editorial aesthetic as
// /auth, /console, /observatory, /nexus.
func (s *server) handlePlaygroundPage(w http.ResponseWriter, r *http.Request) {
	if _, ok := s.userFromRequest(r); !ok {
		http.Redirect(w, r, "/auth?next=/playground", http.StatusFound)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	w.Write(playgroundHTML)
}

//go:embed assets/nexus.html
var nexusHTML []byte

// /nexus — third dashboard.  Indigo/mint quantum-lab palette with a
// force-directed network graph, model loader, layer-by-stage visualizer,
// rig fleet grid with per-rig modal, NAT/firewall posture matrix,
// bottleneck inspector, and live route attribution.  All data sourced
// from existing read-only APIs (/api/me, /api/me/rigs, /api/me/rigs/stream,
// /api/console/network, /api/pools/{id}/topology) — no new server endpoints.
// The model-load form posts to /api/models/load which can be wired by a
// future agent-side endpoint; the UI degrades cleanly when it's absent.
func (s *server) handleNexusPage(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write(nexusHTML)
}

// ─── /api/widget/state — desktop widget combined feed ─────────────────────
//
// One-shot endpoint the surd desktop widget polls every few seconds.
// Joins the things a tray UI needs:
//   - identity (display name, user id) so we can show "logged in as X"
//   - all rigs owned by the caller (online + offline) with health
//   - 24h tokens contributed (rough committed-work metric)
//
// Auth: accepts either the dashboard session cookie OR X-Agent-Key —
// the widget can authenticate using the same creds.json `surd login`
// produced, so it doesn't need a browser handshake.

func (s *server) handleWidgetState(w http.ResponseWriter, r *http.Request) {
	uid := int64(0)
	var displayName, agentID string
	if u, ok := s.userFromRequest(r); ok {
		uid = u.ID
		displayName = u.DisplayName
	} else if uidA, aid, ok := s.agentFromRequest(r); ok {
		uid = uidA
		agentID = aid
		_ = s.db.QueryRow(`SELECT display_name FROM users WHERE id = ?`, uid).Scan(&displayName)
	} else {
		writeErr(w, 401, "session cookie or X-Agent-Key required")
		return
	}

	rigs := s.collectMeRigs(uid)
	online := 0
	totTokPS := 0.0
	totInflight := 0
	for _, rig := range rigs {
		if rig.Online {
			online++
		}
		totTokPS += rig.TokensPS
		totInflight += rig.Inflight
	}

	// 24h tokens committed: sum input+output for finished jobs in window.
	var tokens24h int64
	_ = s.db.QueryRow(
		`SELECT COALESCE(SUM(input_tokens + output_tokens), 0)
		   FROM inference_log
		  WHERE agent_user_id = ?
		    AND started_at >= ?
		    AND status = 'ok'`,
		uid, nowUnix()-86400,
	).Scan(&tokens24h)

	writeJSON(w, 200, map[string]any{
		"user": map[string]any{
			"id":           uid,
			"display_name": displayName,
			"agent_id":     agentID, // present when authed via agent key
		},
		"rigs":         rigs,
		"online":       online,
		"total":        len(rigs),
		"tokens_sec":   totTokPS,
		"inflight":     totInflight,
		"tokens_24h":   tokens24h,
		"generated_at": nowUnix(),
	})
}

// ─── /api/install/oneliner ────────────────────────────────────────────────
//
// Personalised install snippet for the current user.  Used by the /install
// page on the dashboard so a logged-in user can paste a single command
// that ships with an embedded device-code (no manual code-typing required).

// shellQuote returns "" for empty input, otherwise a single-quoted POSIX
// shell literal safe to interpolate into a command line.  Conservative —
// we strip any chars we don't want in pool/invite anyway, so the only
// quoting we need is `'` → `'\”`.  This double-duties for PowerShell
// since we wrap it in `'…'` (PowerShell single-quotes are literal).
func shellQuote(s string) string {
	if s == "" {
		return ""
	}
	// Defence-in-depth: reject anything outside [A-Za-z0-9._-] entirely.
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z',
			r >= 'A' && r <= 'Z',
			r >= '0' && r <= '9',
			r == '.' || r == '_' || r == '-':
		default:
			return ""
		}
	}
	return "'" + s + "'"
}

func (s *server) handleInstallOneliner(w http.ResponseWriter, r *http.Request) {
	target := strings.ToLower(r.URL.Query().Get("os"))
	if target == "" {
		target = "linux"
	}
	pool := shellQuote(r.URL.Query().Get("pool"))
	invite := shellQuote(r.URL.Query().Get("invite"))
	host := strings.TrimRight(s.cfg.publicURL, "/")
	if host == "" {
		host = "http://" + r.Host
	}
	// Build forwarded args (POSIX) and env-var prefix (PowerShell).  We
	// route them via `sh -s -- --pool ... --invite ...` for sh/zsh and via
	// $env:SURD_POOL / $env:SURD_INVITE for PowerShell so they survive the
	// `iwr | iex` pipe (which can't take argv).
	shArgs := ""
	if pool != "" {
		shArgs += " --pool " + pool
	}
	if invite != "" {
		shArgs += " --invite " + invite
	}
	psPrefix := ""
	if pool != "" {
		psPrefix += "$env:SURD_POOL=" + pool + "; "
	}
	if invite != "" {
		psPrefix += "$env:SURD_INVITE=" + invite + "; "
	}
	var line, shell string
	switch target {
	case "windows", "win", "ps1", "powershell":
		shell = "powershell"
		line = fmt.Sprintf(`%siwr -useb %s/setup.ps1 | iex`, psPrefix, host)
	case "macos", "darwin", "mac":
		shell = "sh"
		line = fmt.Sprintf(`curl -fsSL %s/setup.sh | sh -s --%s`, host, shArgs)
	case "zsh":
		shell = "zsh"
		line = fmt.Sprintf(`curl -fsSL %s/setup.zsh | zsh -s --%s`, host, shArgs)
	default: // linux/bash use setup.sh
		shell = "sh"
		line = fmt.Sprintf(`curl -fsSL %s/setup.sh | sh -s --%s`, host, shArgs)
	}
	// Trim trailing " --" when no args were forwarded; otherwise sh sees an
	// empty `--` and warns.
	line = strings.TrimRight(line, " ")
	line = strings.TrimSuffix(line, " --")
	writeJSON(w, 200, map[string]any{
		"os":       target,
		"shell":    shell,
		"oneliner": line,
		"server":   host,
	})
}
