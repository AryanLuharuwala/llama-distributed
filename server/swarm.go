package main

import (
	"net/http"
	"sort"
	"strings"
)

// ─── /api/swarm ─────────────────────────────────────────────────────────────
//
// Public, unauthenticated endpoint that aggregates live telemetry across
// every connected rig.  Modelled after the Petals / Hivemind stats page:
// returns a JSON document with global counters and a per-node array.
//
// We deliberately omit anything that identifies a specific user.  Each
// node entry carries:
//   • a 12-char public ID derived from the agent_id (not the raw key)
//   • coarse fields: GPU model, VRAM, roles, models, throughput
//   • a coarse IP-bucket (first three octets) for the world-map view
//
// The response is recomputed on every request; the work is O(N) over
// connected agents and acquires only the hub RLock once.

type swarmNode struct {
	ID         string   `json:"id"`           // 12-char public id
	Hostname   string   `json:"hostname"`     // self-reported, not auth'd
	NGPUs      int      `json:"n_gpus"`
	GPUModel   string   `json:"gpu_model,omitempty"`
	GPUUtil    []float64 `json:"gpu_util,omitempty"`
	VRAMTotal  int64    `json:"vram_total"`
	VRAMFree   int64    `json:"vram_free,omitempty"`
	TokensPS   float64  `json:"tokens_sec"`
	Roles      []string `json:"roles,omitempty"`
	Models     []string `json:"models,omitempty"`
	Inflight   int      `json:"inflight"`
	UptimeSec  int64    `json:"uptime_sec,omitempty"`
	BWUpKbps   int64    `json:"bw_up_kbps,omitempty"`
	BWDnKbps   int64    `json:"bw_dn_kbps,omitempty"`
	IPBucket   string   `json:"ip_bucket,omitempty"` // coarse, for map
	UpdatedAt  int64    `json:"updated_at"`
}

type swarmStats struct {
	GeneratedAt int64                  `json:"generated_at"`
	NodeCount   int                    `json:"node_count"`
	TotalVRAM   int64                  `json:"total_vram"`
	TotalTokPS  float64                `json:"total_tokens_sec"`
	GPUCount    int                    `json:"total_gpus"`
	InflightAll int                    `json:"total_inflight"`
	ByModel     map[string]int         `json:"nodes_by_model"`
	ByRole      map[string]int         `json:"nodes_by_role"`
	ByGPU       map[string]int         `json:"nodes_by_gpu"`
	ByCountry   map[string]int         `json:"nodes_by_ip_bucket"`
	Nodes       []swarmNode            `json:"nodes"`
}

// publicAgentID returns a stable but non-reversible 12-char public id for
// the (userID, agentID) pair.  agentIDs are random 24-char tokens, so the
// first 12 hex chars are already non-guessable; we just truncate so an
// observer can correlate the same node across reloads without learning
// the full key.
func publicAgentID(userID int64, agentID string) string {
	if len(agentID) >= 12 {
		return agentID[:12]
	}
	return agentID
}

// ipBucket coarsens an IPv4 to first three octets ("203.0.113") or an
// IPv6 to its /48.  Empty input passes through.
func ipBucket(ip string) string {
	if ip == "" {
		return ""
	}
	if strings.Count(ip, ".") == 3 {
		// IPv4: keep first three octets.
		parts := strings.SplitN(ip, ".", 4)
		if len(parts) == 4 {
			return parts[0] + "." + parts[1] + "." + parts[2] + ".0/24"
		}
	}
	if strings.Contains(ip, ":") {
		// IPv6: keep first three hex groups.
		parts := strings.SplitN(ip, ":", 4)
		if len(parts) >= 3 {
			return parts[0] + ":" + parts[1] + ":" + parts[2] + "::/48"
		}
	}
	return ip
}

func (s *server) handleSwarmStats(w http.ResponseWriter, r *http.Request) {
	agents := s.hub.snapshotAgents()
	out := swarmStats{
		GeneratedAt: nowUnix(),
		NodeCount:   len(agents),
		ByModel:     make(map[string]int),
		ByRole:      make(map[string]int),
		ByGPU:       make(map[string]int),
		ByCountry:   make(map[string]int),
		Nodes:       make([]swarmNode, 0, len(agents)),
	}
	for _, a := range agents {
		st := a.snapshotStatus()
		n := swarmNode{
			ID:         publicAgentID(a.userID, a.agentID),
			Hostname:   a.hostname,
			NGPUs:      st.NGPUs,
			GPUModel:   st.GPUModel,
			GPUUtil:    st.GPUUtil,
			VRAMTotal:  st.VRAMTotal,
			VRAMFree:   st.VRAMFree,
			TokensPS:   st.TokensPS,
			Roles:      st.RolesHeld,
			Models:     st.ModelsHeld,
			Inflight:   st.Inflight,
			UptimeSec:  st.UptimeSec,
			BWUpKbps:   st.BWUpKbps,
			BWDnKbps:   st.BWDnKbps,
			IPBucket:   ipBucket(a.remoteIP),
			UpdatedAt:  st.UpdatedAt,
		}
		out.Nodes = append(out.Nodes, n)
		out.TotalVRAM += n.VRAMTotal
		out.TotalTokPS += n.TokensPS
		out.GPUCount += n.NGPUs
		out.InflightAll += n.Inflight
		if n.GPUModel != "" {
			out.ByGPU[n.GPUModel]++
		}
		if n.IPBucket != "" {
			out.ByCountry[n.IPBucket]++
		}
		for _, m := range n.Models {
			out.ByModel[m]++
		}
		for _, role := range n.Roles {
			out.ByRole[role]++
		}
	}
	// Stable order: hostname asc.  Lets the dashboard render deterministically
	// for diffing across reloads.
	sort.SliceStable(out.Nodes, func(i, j int) bool {
		if out.Nodes[i].Hostname == out.Nodes[j].Hostname {
			return out.Nodes[i].ID < out.Nodes[j].ID
		}
		return out.Nodes[i].Hostname < out.Nodes[j].Hostname
	})
	writeJSON(w, 200, out)
}

// /swarm — public HTML view of the swarm.  Vanilla JS + fetch on
// /api/swarm every 5s.  No frameworks, no auth required.
func (s *server) handleSwarmPage(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write([]byte(swarmHTML))
}

const swarmHTML = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Swarm — Distributed Inference</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { color-scheme: dark; }
  body { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
         background:#0b0d10; color:#e6edf3; margin:0; padding:1.5rem 2rem; }
  h1 { font-size:1.4rem; margin:0 0 1rem; letter-spacing:.02em; }
  .sub { color:#7d8590; font-size:.85rem; margin-bottom:1.5rem; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
          gap:.75rem; margin-bottom:1.5rem; }
  .stat { background:#161b22; border:1px solid #30363d; padding:.75rem 1rem;
          border-radius:6px; }
  .stat .v { font-size:1.4rem; font-weight:600; }
  .stat .k { color:#7d8590; font-size:.75rem; text-transform:uppercase;
             letter-spacing:.05em; margin-top:.25rem; }
  table { width:100%; border-collapse:collapse; font-size:.85rem;
          background:#161b22; border:1px solid #30363d; border-radius:6px;
          overflow:hidden; }
  th, td { padding:.5rem .75rem; text-align:left; border-bottom:1px solid #21262d; }
  th { color:#7d8590; font-weight:500; background:#0d1117; }
  tr:last-child td { border-bottom:none; }
  .pill { display:inline-block; padding:.1rem .5rem; border-radius:10px;
          background:#21262d; color:#7d8590; font-size:.7rem; margin-right:.25rem; }
  .ok { color:#3fb950; }
  .warn { color:#d29922; }
  .err { color:#f85149; }
  details { margin-top:1.5rem; }
  summary { cursor:pointer; color:#7d8590; padding:.5rem 0; }
  pre { background:#161b22; border:1px solid #30363d; padding:.75rem;
        border-radius:6px; font-size:.75rem; overflow:auto; }
  .footer { color:#7d8590; font-size:.75rem; margin-top:2rem; }
</style>
</head>
<body>
<h1>Swarm</h1>
<div class="sub">Live view of every rig connected to the control plane. Refreshes every 5s.</div>

<div class="grid" id="stats"></div>

<table>
  <thead>
    <tr>
      <th>Node</th>
      <th>GPU</th>
      <th>VRAM</th>
      <th>Roles</th>
      <th>Models</th>
      <th>Tok/s</th>
      <th>In-flight</th>
      <th>Region</th>
    </tr>
  </thead>
  <tbody id="nodes"></tbody>
</table>

<details>
  <summary>Raw /api/swarm response</summary>
  <pre id="raw"></pre>
</details>

<div class="footer">
  Source: <code>GET /api/swarm</code> · Updated <span id="age">—</span>
</div>

<script>
function bytes(n) {
  if (!n) return '—';
  const u = ['B','KB','MB','GB','TB'];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return n.toFixed(n < 10 ? 1 : 0) + ' ' + u[i];
}
function age(t) {
  if (!t) return '—';
  const s = Math.floor(Date.now()/1000 - t);
  if (s < 60) return s + 's ago';
  if (s < 3600) return Math.floor(s/60) + 'm ago';
  return Math.floor(s/3600) + 'h ago';
}
async function tick() {
  let d;
  try { d = await fetch('/api/swarm').then(r => r.json()); }
  catch (e) { return; }
  const stats = document.getElementById('stats');
  stats.innerHTML = '';
  function stat(label, value) {
    const el = document.createElement('div');
    el.className = 'stat';
    el.innerHTML = '<div class="v">' + value + '</div><div class="k">' + label + '</div>';
    stats.appendChild(el);
  }
  stat('Nodes',     d.node_count);
  stat('Total GPUs', d.total_gpus);
  stat('Total VRAM', bytes(d.total_vram));
  stat('Tokens/s',   d.total_tokens_sec.toFixed(1));
  stat('In-flight',  d.total_inflight);
  stat('Regions',    Object.keys(d.nodes_by_ip_bucket || {}).length);

  const tbody = document.getElementById('nodes');
  tbody.innerHTML = '';
  for (const n of d.nodes) {
    const tr = document.createElement('tr');
    function td(t) { const c = document.createElement('td'); c.innerHTML = t || '—'; tr.appendChild(c); }
    td('<code>' + n.id + '</code><br><span class="pill">' + (n.hostname || '?') + '</span>');
    td((n.gpu_model || '?') + (n.n_gpus > 1 ? ' ×' + n.n_gpus : ''));
    td(bytes(n.vram_total) + (n.vram_free ? '<br><span class="pill">' + bytes(n.vram_free) + ' free</span>' : ''));
    td((n.roles || []).map(r => '<span class="pill">' + r + '</span>').join(''));
    td((n.models || []).map(m => '<span class="pill">' + m + '</span>').join(''));
    td(n.tokens_sec ? n.tokens_sec.toFixed(1) : '—');
    td(n.inflight ? '<span class="warn">' + n.inflight + '</span>' : '<span class="ok">0</span>');
    td(n.ip_bucket || '—');
    tbody.appendChild(tr);
  }
  document.getElementById('raw').textContent = JSON.stringify(d, null, 2);
  document.getElementById('age').textContent = age(d.generated_at);
}
tick();
setInterval(tick, 5000);
</script>
</body>
</html>
`
