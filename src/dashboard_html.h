// dashboard_html.h — auto-generated, do not edit.
// The full dashboard SPA as a C++ string literal.
// Included only by dashboard_server.cpp.
#pragma once
#include <string>
namespace dist {
inline const std::string DASHBOARD_HTML = R"HTMLEOF(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>llama-distributed · cluster dashboard</title>
<style>
  :root {
    --bg:      #0d1117;
    --surface: #161b22;
    --border:  #30363d;
    --accent:  #58a6ff;
    --green:   #3fb950;
    --yellow:  #d29922;
    --red:     #f85149;
    --text:    #c9d1d9;
    --muted:   #8b949e;
    --radius:  8px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    font-size: 14px;
    min-height: 100vh;
  }

  /* ── Header ── */
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 24px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    position: sticky; top: 0; z-index: 10;
  }
  header .logo { font-size: 18px; font-weight: 700; color: var(--accent); letter-spacing: -0.5px; }
  header .logo span { color: var(--text); font-weight: 400; }
  header .status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--muted); display: inline-block; margin-right: 6px;
    transition: background 0.3s;
  }
  header .status-dot.live { background: var(--green); box-shadow: 0 0 6px var(--green); }
  header .model-badge {
    background: #1c2128; border: 1px solid var(--border);
    padding: 3px 10px; border-radius: 20px; font-size: 12px; color: var(--muted);
  }

  /* ── Layout ── */
  main { max-width: 1200px; margin: 0 auto; padding: 24px 20px; }

  /* ── KPI cards ── */
  .kpi-row {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 14px;
    margin-bottom: 24px;
  }
  .kpi-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 18px;
  }
  .kpi-card .label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .6px; margin-bottom: 6px; }
  .kpi-card .value { font-size: 28px; font-weight: 700; color: var(--text); line-height: 1; }
  .kpi-card .unit  { font-size: 12px; color: var(--muted); margin-top: 2px; }

  /* ── Join box ── */
  .join-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px 20px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 14px;
    flex-wrap: wrap;
  }
  .join-box .join-label { font-weight: 600; white-space: nowrap; color: var(--accent); }
  .join-box code {
    flex: 1;
    background: #0d1117;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 12px;
    font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
    font-size: 13px;
    color: var(--green);
    overflow-x: auto;
    white-space: nowrap;
  }
  .copy-btn {
    background: var(--accent);
    color: #000;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: opacity 0.15s;
    white-space: nowrap;
  }
  .copy-btn:hover { opacity: 0.85; }
  .copy-btn.copied { background: var(--green); }

  /* ── Section header ── */
  .section-hdr {
    font-size: 13px; font-weight: 600; color: var(--muted);
    text-transform: uppercase; letter-spacing: .7px;
    margin-bottom: 12px; padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
  }

  /* ── Node grid ── */
  .node-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
    gap: 16px;
    margin-bottom: 32px;
  }
  .node-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 18px;
    transition: border-color 0.2s;
  }
  .node-card.dead { opacity: 0.45; border-color: var(--red); }
  .node-card .node-hdr {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 12px;
  }
  .node-card .node-id { font-weight: 700; font-size: 14px; }
  .node-card .node-addr { font-size: 11px; color: var(--muted); margin-top: 2px; }
  .pill {
    font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 20px;
    text-transform: uppercase; letter-spacing: .4px;
  }
  .pill.alive    { background: rgba(63,185,80,.15); color: var(--green); border: 1px solid rgba(63,185,80,.3); }
  .pill.dead     { background: rgba(248,81,73,.15); color: var(--red);   border: 1px solid rgba(248,81,73,.3); }
  .pill.loading  { background: rgba(210,153,34,.15); color: var(--yellow); border: 1px solid rgba(210,153,34,.3); }

  /* Stat rows inside node card */
  .stat-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 4px 0; font-size: 12px;
  }
  .stat-row .sk { color: var(--muted); }
  .stat-row .sv { color: var(--text); font-variant-numeric: tabular-nums; }

  /* Progress bar */
  .prog-wrap { margin: 8px 0 4px; }
  .prog-label { display: flex; justify-content: space-between; font-size: 11px; color: var(--muted); margin-bottom: 4px; }
  .prog-bg {
    height: 6px; background: #1c2128; border-radius: 3px; overflow: hidden;
  }
  .prog-fill {
    height: 100%; border-radius: 3px;
    background: linear-gradient(90deg, var(--accent), var(--green));
    transition: width 0.4s ease;
  }
  .prog-fill.warn { background: linear-gradient(90deg, var(--yellow), #f0a500); }
  .prog-fill.crit { background: linear-gradient(90deg, var(--red),    #c0392b); }

  /* ── Contribution bar ── */
  .contrib-section { margin-bottom: 32px; }
  .contrib-bar {
    height: 28px; background: #1c2128; border-radius: var(--radius);
    overflow: hidden; display: flex; margin-top: 10px;
  }
  .contrib-seg {
    height: 100%; display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 700; color: #fff; transition: width 0.5s ease;
    overflow: hidden; white-space: nowrap; padding: 0 6px;
    cursor: default;
  }
  .contrib-legend {
    display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px;
  }
  .contrib-legend .leg {
    display: flex; align-items: center; gap: 5px; font-size: 12px;
  }
  .contrib-legend .leg-dot {
    width: 10px; height: 10px; border-radius: 2px;
  }

  /* ── Footer ── */
  footer {
    text-align: center; font-size: 11px; color: var(--muted);
    padding: 20px; border-top: 1px solid var(--border); margin-top: 8px;
  }

  /* ── Responsive ── */
  @media(max-width:600px) {
    .kpi-row { grid-template-columns: repeat(2,1fr); }
    .node-grid { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>

<header>
  <div>
    <div class="logo">llama<span>-distributed</span></div>
    <div class="model-badge" id="modelBadge">no model</div>
  </div>
  <div style="display:flex;align-items:center;gap:8px">
    <span class="status-dot" id="dot"></span>
    <span id="statusTxt" style="font-size:12px;color:var(--muted)">connecting…</span>
  </div>
</header>

<main>

  <!-- KPI cards -->
  <div class="kpi-row">
    <div class="kpi-card">
      <div class="label">Nodes online</div>
      <div class="value" id="kpiNodes">–</div>
      <div class="unit">connected</div>
    </div>
    <div class="kpi-card">
      <div class="label">Tokens / sec</div>
      <div class="value" id="kpiTps">–</div>
      <div class="unit">10-s rolling avg</div>
    </div>
    <div class="kpi-card">
      <div class="label">Tokens total</div>
      <div class="value" id="kpiTotal">–</div>
      <div class="unit">lifetime</div>
    </div>
    <div class="kpi-card">
      <div class="label">Active requests</div>
      <div class="value" id="kpiReqs">–</div>
      <div class="unit">in-flight</div>
    </div>
    <div class="kpi-card">
      <div class="label">Layers</div>
      <div class="value" id="kpiLayers">–</div>
      <div class="unit">total model layers</div>
    </div>
  </div>

  <!-- Join box -->
  <div class="join-box">
    <div class="join-label">⊕ Join this pool</div>
    <code id="joinCmd">loading…</code>
    <button class="copy-btn" id="copyBtn" onclick="copyJoin()">Copy</button>
  </div>

  <!-- Contribution bar -->
  <div class="contrib-section">
    <div class="section-hdr">Layer contribution</div>
    <div class="contrib-bar" id="contribBar"></div>
    <div class="contrib-legend" id="contribLegend"></div>
  </div>

  <!-- Node cards -->
  <div class="section-hdr">Nodes</div>
  <div class="node-grid" id="nodeGrid">
    <div style="color:var(--muted);padding:20px">Waiting for data…</div>
  </div>

</main>

<footer>llama-distributed · <span id="ts">–</span></footer>

<script>
// ── Colour palette for nodes ─────────────────────────────────────────────────
const PALETTE = [
  '#58a6ff','#3fb950','#d29922','#f78166','#a5d6ff',
  '#7ee787','#ffa657','#ff7b72','#d2a8ff','#79c0ff',
];
let nodeColours = {};
let colIdx = 0;
function colorFor(id) {
  if (!nodeColours[id]) nodeColours[id] = PALETTE[colIdx++ % PALETTE.length];
  return nodeColours[id];
}

// ── Formatters ────────────────────────────────────────────────────────────────
function fmtBytes(b) {
  if (b >= 1e12) return (b/1e12).toFixed(1)+'TB';
  if (b >= 1e9)  return (b/1e9).toFixed(1)+'GB';
  if (b >= 1e6)  return (b/1e6).toFixed(0)+'MB';
  if (b >= 1e3)  return (b/1e3).toFixed(0)+'KB';
  return b+'B';
}
function fmtNum(n) {
  if (n >= 1e9) return (n/1e9).toFixed(2)+'B';
  if (n >= 1e6) return (n/1e6).toFixed(2)+'M';
  if (n >= 1e3) return (n/1e3).toFixed(1)+'K';
  return String(n);
}
function pct(used, total) { return total > 0 ? Math.min(100, 100*used/total) : 0; }

// ── Progress bar HTML ─────────────────────────────────────────────────────────
function progBar(label, usedPct, leftTxt, rightTxt) {
  const cls = usedPct > 90 ? 'crit' : usedPct > 70 ? 'warn' : '';
  return `<div class="prog-wrap">
    <div class="prog-label"><span>${label}</span><span>${leftTxt} / ${rightTxt}</span></div>
    <div class="prog-bg"><div class="prog-fill ${cls}" style="width:${usedPct.toFixed(1)}%"></div></div>
  </div>`;
}

// ── Render ────────────────────────────────────────────────────────────────────
function render(data) {
  const nodes = data.nodes || [];
  const alive = nodes.filter(n => n.alive);

  // KPI
  document.getElementById('kpiNodes').textContent  = alive.length;
  document.getElementById('kpiTps').textContent    = data.tokens_per_second.toFixed(1);
  document.getElementById('kpiTotal').textContent  = fmtNum(data.tokens_total);
  document.getElementById('kpiReqs').textContent   = data.n_active_requests;
  document.getElementById('kpiLayers').textContent = data.n_layers || data.n_layers_total || '–';
  document.getElementById('modelBadge').textContent = data.model || '(no model)';
  document.getElementById('ts').textContent =
    new Date(data.timestamp_us/1000).toLocaleTimeString();

  // Contribution bar
  const bar    = document.getElementById('contribBar');
  const legend = document.getElementById('contribLegend');
  bar.innerHTML = '';
  legend.innerHTML = '';
  const total = data.n_layers || data.n_layers_total || 0;
  if (total > 0) {
    nodes.forEach(n => {
      if (!n.alive) return;
      const c = colorFor(n.id);
      const w = n.layer_pct.toFixed(1);
      const seg = document.createElement('div');
      seg.className = 'contrib-seg';
      seg.style.width = w + '%';
      seg.style.background = c;
      seg.title = `${n.id}: layers ${n.layer_first}–${n.layer_last} (${w}%)`;
      if (parseFloat(w) > 8) seg.textContent = w + '%';
      bar.appendChild(seg);

      legend.innerHTML += `<div class="leg">
        <div class="leg-dot" style="background:${c}"></div>
        <span>${n.id.replace(/:\d+$/,'')} (L${n.layer_first}–${n.layer_last})</span>
      </div>`;
    });
  }

  // Node cards
  const grid = document.getElementById('nodeGrid');
  grid.innerHTML = '';
  if (!nodes.length) {
    grid.innerHTML = '<div style="color:var(--muted);padding:20px">No nodes connected yet.</div>';
    return;
  }

  nodes.forEach(n => {
    const c      = colorFor(n.id);
    const vused  = n.vram_total - n.vram_free;
    const vp     = pct(vused, n.vram_total);
    const rused  = n.cpu_ram_total - n.cpu_ram_free;
    const rp     = pct(rused, n.cpu_ram_total);
    const pill   = n.alive
      ? (n.model_loaded ? '<span class="pill alive">online</span>'
                        : '<span class="pill loading">loading</span>')
      : '<span class="pill dead">offline</span>';

    const layers = n.layer_last >= n.layer_first
      ? `L${n.layer_first}–${n.layer_last} (${(n.layer_pct||0).toFixed(1)}%)`
      : '—';

    grid.innerHTML += `
      <div class="node-card ${n.alive?'':'dead'}">
        <div class="node-hdr">
          <div>
            <div class="node-id" style="color:${c}">${n.id}</div>
            <div class="node-addr">${n.addr}</div>
          </div>
          ${pill}
        </div>
        <div class="stat-row"><span class="sk">GPUs</span><span class="sv">${n.n_gpus}</span></div>
        <div class="stat-row"><span class="sk">Layers</span><span class="sv">${layers}</span></div>
        <div class="stat-row"><span class="sk">Tokens total</span><span class="sv">${fmtNum(n.tokens_total)}</span></div>
        <div class="stat-row"><span class="sk">TPS (10s avg)</span><span class="sv">${(n.tps||0).toFixed(1)}</span></div>
        <div class="stat-row"><span class="sk">Data sent</span><span class="sv">${fmtBytes(n.bytes_sent)}</span></div>
        ${n.vram_total > 0 ? progBar('VRAM', vp, fmtBytes(vused), fmtBytes(n.vram_total)) : ''}
        ${n.cpu_ram_total > 0 ? progBar('RAM', rp, fmtBytes(rused), fmtBytes(n.cpu_ram_total)) : ''}
        ${n.n_gpus > 0 ? progBar('GPU util', (n.gpu_util||0)*100, ((n.gpu_util||0)*100).toFixed(0)+'%', '100%') : ''}
      </div>`;
  });
}

// ── Join command ──────────────────────────────────────────────────────────────
function loadJoin() {
  fetch('/join').then(r=>r.text()).then(t=>{
    document.getElementById('joinCmd').textContent = t.trim();
  }).catch(()=>{});
}

function copyJoin() {
  const txt = document.getElementById('joinCmd').textContent;
  navigator.clipboard.writeText(txt).then(()=>{
    const btn = document.getElementById('copyBtn');
    btn.textContent = 'Copied!';
    btn.classList.add('copied');
    setTimeout(()=>{ btn.textContent='Copy'; btn.classList.remove('copied'); }, 2000);
  });
}

// ── SSE live updates ──────────────────────────────────────────────────────────
function connect() {
  const dot = document.getElementById('dot');
  const txt = document.getElementById('statusTxt');
  const es  = new EventSource('/events');

  es.onopen = () => {
    dot.classList.add('live');
    txt.textContent = 'live';
  };

  es.onmessage = e => {
    try { render(JSON.parse(e.data)); } catch(_) {}
  };

  es.onerror = () => {
    dot.classList.remove('live');
    txt.textContent = 'reconnecting…';
    es.close();
    setTimeout(connect, 3000);
  };
}

// ── Boot ──────────────────────────────────────────────────────────────────────
// Fetch initial state immediately, then hook up SSE
fetch('/stats').then(r=>r.json()).then(render).catch(()=>{});
loadJoin();
connect();
</script>
</body>
</html>
)HTMLEOF";
} // namespace dist
