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
<title>llama-distributed · cluster</title>
<style>
:root{
  --bg:#0a0d12;--surface:#0d1117;--card:#161b22;--elevated:#1c2128;
  --border:#30363d;--fg:#e6edf3;--muted:#8b949e;--dim:#484f58;
  --blue:#58a6ff;--green:#3fb950;--yellow:#d29922;--red:#f85149;
  --r:8px;--r2:4px;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  min-height:100vh;-webkit-font-smoothing:antialiased}
a{color:var(--blue);text-decoration:none}

/* header */
header{display:flex;align-items:center;justify-content:space-between;
  padding:14px 24px;background:var(--surface);border-bottom:1px solid var(--border);
  position:sticky;top:0;z-index:10}
.logo{font-size:17px;font-weight:800;letter-spacing:-.5px}
.logo b{color:var(--blue)}
.model-badge{background:var(--card);border:1px solid var(--border);
  padding:3px 10px;border-radius:100px;font-size:12px;color:var(--muted)}
.status-row{display:flex;align-items:center;gap:8px}
.sdot{width:8px;height:8px;border-radius:50%;background:var(--dim);transition:.3s}
.sdot.live{background:var(--green);box-shadow:0 0 6px var(--green)}
.stxt{font-size:12px;color:var(--muted)}

/* layout */
main{max-width:1200px;margin:0 auto;padding:24px 20px}

/* kpi row */
.kpi-row{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:12px;margin-bottom:24px}
.kpi{background:var(--card);border:1px solid var(--border);border-radius:var(--r);padding:16px 18px}
.kpi-lbl{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.6px;margin-bottom:4px}
.kpi-val{font-size:30px;font-weight:700;color:var(--fg);line-height:1}
.kpi-sub{font-size:11px;color:var(--dim);margin-top:3px}

/* join box */
.join-box{background:var(--card);border:1px solid var(--border);border-radius:var(--r);
  padding:16px 20px;margin-bottom:24px;display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.join-lbl{font-weight:700;color:var(--blue);white-space:nowrap;font-size:13px}
.join-code{flex:1;background:var(--elevated);border:1px solid var(--border);border-radius:var(--r2);
  padding:9px 13px;font-family:"SF Mono",Menlo,Consolas,monospace;font-size:12px;
  color:var(--green);overflow-x:auto;white-space:nowrap}
.copy-btn{background:var(--blue);color:#fff;border:none;border-radius:var(--r2);
  padding:8px 16px;font-size:13px;font-weight:600;cursor:pointer;transition:.15s;white-space:nowrap}
.copy-btn:hover{background:#79bcff}
.copy-btn.ok{background:var(--green)}

/* section */
.sec{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.7px;
  color:var(--muted);margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid var(--border)}

/* contrib bar */
.contrib{margin-bottom:28px}
.cbar{height:26px;background:var(--elevated);border-radius:var(--r);overflow:hidden;
  display:flex;margin-top:10px;border:1px solid var(--border)}
.cseg{height:100%;display:flex;align-items:center;justify-content:center;
  font-size:11px;font-weight:700;color:#fff;overflow:hidden;white-space:nowrap;
  padding:0 6px;transition:width .5s ease;cursor:default}
.cleg{display:flex;flex-wrap:wrap;gap:10px;margin-top:10px}
.cleg-item{display:flex;align-items:center;gap:5px;font-size:12px;color:var(--muted)}
.cleg-dot{width:10px;height:10px;border-radius:2px;flex-shrink:0}

/* node grid */
.node-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px;margin-bottom:32px}
.node-card{background:var(--card);border:1px solid var(--border);border-radius:var(--r);padding:16px 18px;transition:.2s}
.node-card:hover{border-color:var(--blue)}
.node-card.dead{opacity:.45;border-color:var(--red)}
.node-hd{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:12px}
.node-id{font-weight:700;font-size:14px}
.node-addr{font-size:11px;color:var(--muted);margin-top:2px}
.pill{font-size:10px;font-weight:700;padding:2px 8px;border-radius:100px;text-transform:uppercase;letter-spacing:.4px}
.p-alive{background:rgba(63,185,80,.15);color:var(--green);border:1px solid rgba(63,185,80,.3)}
.p-dead{background:rgba(248,81,73,.15);color:var(--red);border:1px solid rgba(248,81,73,.3)}
.p-load{background:rgba(210,153,34,.15);color:var(--yellow);border:1px solid rgba(210,153,34,.3)}

/* sparkline */
.sparkline-wrap{margin:10px 0 6px}
.sparkline-lbl{display:flex;justify-content:space-between;font-size:11px;color:var(--muted);margin-bottom:4px}
canvas.spark{display:block;width:100%;height:32px;border-radius:var(--r2)}

/* stat rows */
.stat-rows{display:grid;grid-template-columns:1fr 1fr;gap:4px 16px;margin-bottom:8px}
.sr{display:flex;justify-content:space-between;font-size:12px;padding:2px 0}
.sr-k{color:var(--muted)}.sr-v{color:var(--fg);font-variant-numeric:tabular-nums}

/* progress bar */
.prog{margin:4px 0}
.prog-lbl{display:flex;justify-content:space-between;font-size:11px;color:var(--muted);margin-bottom:3px}
.prog-bg{height:5px;background:var(--elevated);border-radius:2px;overflow:hidden}
.prog-fill{height:100%;border-radius:2px;transition:width .4s ease;
  background:linear-gradient(90deg,var(--blue),var(--green))}
.prog-fill.warn{background:linear-gradient(90deg,var(--yellow),#f0a500)}
.prog-fill.crit{background:linear-gradient(90deg,var(--red),#c0392b)}

/* footer */
footer{text-align:center;font-size:11px;color:var(--dim);
  padding:20px;border-top:1px solid var(--border);margin-top:8px}

@media(max-width:600px){.kpi-row{grid-template-columns:repeat(2,1fr)}.node-grid{grid-template-columns:1fr}}
</style>
</head>
<body>

<header>
  <div>
    <div class="logo">🦙 llama<b>-distributed</b></div>
    <div class="model-badge" id="modelBadge">no model</div>
  </div>
  <div class="status-row">
    <span class="sdot" id="dot"></span>
    <span class="stxt" id="statusTxt">connecting…</span>
  </div>
</header>

<main>
  <div class="kpi-row">
    <div class="kpi"><div class="kpi-lbl">Nodes online</div><div class="kpi-val" id="kpiNodes">–</div><div class="kpi-sub">connected</div></div>
    <div class="kpi"><div class="kpi-lbl">Tokens / sec</div><div class="kpi-val" id="kpiTps">–</div><div class="kpi-sub">10-s rolling avg</div></div>
    <div class="kpi"><div class="kpi-lbl">Tokens total</div><div class="kpi-val" id="kpiTotal">–</div><div class="kpi-sub">lifetime</div></div>
    <div class="kpi"><div class="kpi-lbl">Active requests</div><div class="kpi-val" id="kpiReqs">–</div><div class="kpi-sub">in-flight</div></div>
    <div class="kpi"><div class="kpi-lbl">Layers</div><div class="kpi-val" id="kpiLayers">–</div><div class="kpi-sub">model total</div></div>
  </div>

  <div class="join-box">
    <div class="join-lbl">⊕ Join command</div>
    <div class="join-code" id="joinCmd">loading…</div>
    <button class="copy-btn" onclick="copyJoin()">Copy</button>
  </div>

  <div class="contrib">
    <div class="sec">Layer contribution</div>
    <div class="cbar" id="cbar"></div>
    <div class="cleg" id="cleg"></div>
  </div>

  <div class="sec">Nodes</div>
  <div class="node-grid" id="nodeGrid">
    <div style="color:var(--muted);padding:20px">Waiting for data…</div>
  </div>
</main>

<footer>llama-distributed · <span id="ts">–</span></footer>

<script>
const PALETTE=['#58a6ff','#3fb950','#d29922','#f78166','#a5d6ff','#7ee787','#ffa657','#ff7b72','#d2a8ff','#79c0ff'];
const nodeColors={};let colIdx=0;
function colorFor(id){if(!nodeColors[id])nodeColors[id]=PALETTE[colIdx++%PALETTE.length];return nodeColors[id];}

// per-node TPS history for sparklines
const tpsHistory={};
const HIST=30;
function pushTps(id,v){if(!tpsHistory[id])tpsHistory[id]=[];const h=tpsHistory[id];h.push(v);if(h.length>HIST)h.shift();}

function fmtBytes(b){if(b>=1e12)return(b/1e12).toFixed(1)+'TB';if(b>=1e9)return(b/1e9).toFixed(1)+'GB';if(b>=1e6)return(b/1e6).toFixed(0)+'MB';if(b>=1e3)return(b/1e3).toFixed(0)+'KB';return b+'B';}
function fmtNum(n){if(n>=1e9)return(n/1e9).toFixed(1)+'B';if(n>=1e6)return(n/1e6).toFixed(1)+'M';if(n>=1e3)return(n/1e3).toFixed(1)+'K';return String(n);}
function pct(u,t){return t>0?Math.min(100,100*u/t):0;}

function drawSparkline(canvas,data,color){
  if(!canvas)return;
  const dpr=window.devicePixelRatio||1;
  const w=canvas.clientWidth||240,h=32;
  canvas.width=w*dpr;canvas.height=h*dpr;
  const ctx=canvas.getContext('2d');
  ctx.scale(dpr,dpr);
  if(!data||data.length<2){ctx.fillStyle='var(--elevated)';ctx.fillRect(0,0,w,h);return;}
  const max=Math.max(...data,0.01);
  // gradient fill
  const grad=ctx.createLinearGradient(0,0,0,h);
  grad.addColorStop(0,color+'55');
  grad.addColorStop(1,color+'00');
  ctx.fillStyle=grad;
  ctx.beginPath();ctx.moveTo(0,h);
  data.forEach((v,i)=>{
    const x=i*(w/(data.length-1));
    const y=h-(v/max)*(h-2)-1;
    i===0?ctx.lineTo(x,y):ctx.lineTo(x,y);
  });
  ctx.lineTo(w,h);ctx.closePath();ctx.fill();
  // line
  ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.lineJoin='round';
  ctx.beginPath();
  data.forEach((v,i)=>{const x=i*(w/(data.length-1));const y=h-(v/max)*(h-2)-1;i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);});
  ctx.stroke();
}

function progBar(label,usedPct,leftTxt,rightTxt){
  const cls=usedPct>90?'crit':usedPct>70?'warn':'';
  return `<div class="prog">
    <div class="prog-lbl"><span>${label}</span><span>${leftTxt} / ${rightTxt}</span></div>
    <div class="prog-bg"><div class="prog-fill ${cls}" style="width:${usedPct.toFixed(1)}%"></div></div>
  </div>`;
}

function render(data){
  const nodes=data.nodes||[];
  const alive=nodes.filter(n=>n.alive);

  document.getElementById('kpiNodes').textContent=alive.length;
  document.getElementById('kpiTps').textContent=(data.tokens_per_second||0).toFixed(1);
  document.getElementById('kpiTotal').textContent=fmtNum(data.tokens_total||0);
  document.getElementById('kpiReqs').textContent=data.n_active_requests||0;
  document.getElementById('kpiLayers').textContent=data.n_layers||data.n_layers_total||'–';
  document.getElementById('modelBadge').textContent=data.model||'(no model)';
  document.getElementById('ts').textContent=new Date(data.timestamp_us/1000).toLocaleTimeString();

  // contribution bar
  const bar=document.getElementById('cbar');
  const leg=document.getElementById('cleg');
  bar.innerHTML='';leg.innerHTML='';
  const total=data.n_layers||data.n_layers_total||0;
  if(total>0){
    nodes.forEach(n=>{
      if(!n.alive)return;
      const c=colorFor(n.id);
      const w=(n.layer_pct||0).toFixed(1);
      const seg=document.createElement('div');
      seg.className='cseg';
      seg.style.width=w+'%';
      seg.style.background=c;
      seg.title=`${n.id}: L${n.layer_first}–${n.layer_last} (${w}%)`;
      if(parseFloat(w)>7)seg.textContent=w+'%';
      bar.appendChild(seg);
      leg.innerHTML+=`<div class="cleg-item"><div class="cleg-dot" style="background:${c}"></div><span>${n.id.replace(/:\d+$/,'')} L${n.layer_first}–${n.layer_last}</span></div>`;
    });
  }

  // node cards
  const grid=document.getElementById('nodeGrid');
  grid.innerHTML='';
  if(!nodes.length){grid.innerHTML='<div style="color:var(--muted);padding:20px">No nodes connected yet.</div>';return;}

  nodes.forEach(n=>{
    const c=colorFor(n.id);
    const vused=n.vram_total-n.vram_free;
    const vp=pct(vused,n.vram_total);
    const rused=n.cpu_ram_total-n.cpu_ram_free;
    const rp=pct(rused,n.cpu_ram_total);
    const tps=n.tps||0;
    pushTps(n.id,tps);

    const pillCls=n.alive?(n.model_loaded?'p-alive':'p-load'):'p-dead';
    const pillTxt=n.alive?(n.model_loaded?'online':'loading'):'offline';
    const layers=n.layer_last>=n.layer_first?`L${n.layer_first}–${n.layer_last} (${(n.layer_pct||0).toFixed(1)}%)`:'—';

    const card=document.createElement('div');
    card.className='node-card'+(n.alive?'':' dead');
    card.innerHTML=`
      <div class="node-hd">
        <div>
          <div class="node-id" style="color:${c}">${n.id}</div>
          <div class="node-addr">${n.addr||''}</div>
        </div>
        <span class="pill ${pillCls}">${pillTxt}</span>
      </div>
      <div class="stat-rows">
        <div class="sr"><span class="sr-k">GPUs</span><span class="sr-v">${n.n_gpus||0}</span></div>
        <div class="sr"><span class="sr-k">Layers</span><span class="sr-v">${layers}</span></div>
        <div class="sr"><span class="sr-k">Tokens</span><span class="sr-v">${fmtNum(n.tokens_total||0)}</span></div>
        <div class="sr"><span class="sr-k">TPS (10s)</span><span class="sr-v">${tps.toFixed(1)}</span></div>
        <div class="sr"><span class="sr-k">Data sent</span><span class="sr-v">${fmtBytes(n.bytes_sent||0)}</span></div>
      </div>
      <div class="sparkline-wrap">
        <div class="sparkline-lbl"><span>Tokens/sec</span><span>${tps.toFixed(1)} t/s</span></div>
        <canvas class="spark" id="sp-${n.id.replace(/[^a-z0-9]/gi,'_')}"></canvas>
      </div>
      ${n.vram_total>0?progBar('VRAM',vp,fmtBytes(vused),fmtBytes(n.vram_total)):''}
      ${n.cpu_ram_total>0?progBar('RAM',rp,fmtBytes(rused),fmtBytes(n.cpu_ram_total)):''}
      ${n.n_gpus>0?progBar('GPU util',(n.gpu_util||0)*100,((n.gpu_util||0)*100).toFixed(0)+'%','100%'):''}
    `;
    grid.appendChild(card);

    // draw sparkline after card is in DOM
    requestAnimationFrame(()=>{
      const cv=document.getElementById('sp-'+n.id.replace(/[^a-z0-9]/gi,'_'));
      drawSparkline(cv,tpsHistory[n.id],c);
    });
  });
}

function loadJoin(){
  fetch('/join').then(r=>r.text()).then(t=>{document.getElementById('joinCmd').textContent=t.trim();}).catch(()=>{});
}

function copyJoin(){
  const txt=document.getElementById('joinCmd').textContent;
  navigator.clipboard.writeText(txt).then(()=>{
    const btn=document.querySelector('.copy-btn');
    const orig=btn.textContent;btn.textContent='Copied!';btn.classList.add('ok');
    setTimeout(()=>{btn.textContent=orig;btn.classList.remove('ok');},2000);
  }).catch(()=>{});
}

function connect(){
  const dot=document.getElementById('dot');
  const txt=document.getElementById('statusTxt');
  const es=new EventSource('/events');
  es.onopen=()=>{dot.classList.add('live');txt.textContent='live';};
  es.onmessage=e=>{try{render(JSON.parse(e.data));}catch(_){}};
  es.onerror=()=>{
    dot.classList.remove('live');txt.textContent='reconnecting…';
    es.close();setTimeout(connect,3000);
  };
}

fetch('/stats').then(r=>r.json()).then(render).catch(()=>{});
loadJoin();
connect();
</script>
</body>
</html>
)HTMLEOF";
} // namespace dist
