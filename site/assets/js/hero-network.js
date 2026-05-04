(function () {
  const root = document.querySelector('[data-hero-network]');
  if (!root) return;
  const SVGNS = 'http://www.w3.org/2000/svg';
  const W = 800, H = 560;
  const origin = { x: 92, y: 290 };

  // Deterministic pseudo-random so the network looks the same on every reload
  let seed = 1337;
  const rand = () => {
    seed = (seed * 9301 + 49297) % 233280;
    return seed / 233280;
  };
  const gauss = () => (rand() + rand() + rand() + rand() - 2) / 2;

  // ---------- destination cluster (the "pool of GPUs") ----------
  const N = 72;
  const dests = [];
  for (let i = 0; i < N; i++) {
    const x = 640 + rand() * 130;
    const y = Math.max(55, Math.min(515, 282 + gauss() * 175));
    const r = 1.6 + rand() * 2.5;
    dests.push({ x, y, r });
  }
  dests.sort((a, b) => a.y - b.y);

  // Pick the active node — the one closest to a target slightly above center
  let activeIdx = 0;
  let bestScore = Infinity;
  dests.forEach((d, i) => {
    const score = Math.abs(d.y - 248) + Math.abs(d.x - 705) * 0.4;
    if (score < bestScore) { bestScore = score; activeIdx = i; }
  });
  const active = dests[activeIdx];

  // ---------- helpers ----------
  const sel = (cls) => root.querySelector('.' + cls);
  const make = (tag, attrs, parent, cls) => {
    const el = document.createElementNS(SVGNS, tag);
    if (cls) el.setAttribute('class', cls);
    for (const k in attrs) el.setAttribute(k, attrs[k]);
    parent.appendChild(el);
    return el;
  };
  const cubicPath = (a, b) => {
    const dx = b.x - a.x;
    const cp1x = a.x + dx * 0.50;
    const cp2x = a.x + dx * 0.50;
    return `M${a.x},${a.y} C${cp1x},${a.y} ${cp2x},${b.y} ${b.x},${b.y}`;
  };

  // ---------- halos (the soft smudges behind everything) ----------
  const halos = sel('hero-network-halos');
  const haloSpec = [
    { x: 200, y: 200, r: 92,  id: 'haloOlive',  op: 0.65 },
    { x: 280, y: 415, r: 105, id: 'haloOlive',  op: 0.55 },
    { x: 135, y: 470, r: 78,  id: 'haloOlive',  op: 0.45 },
    { x: 595, y: 145, r: 72,  id: 'haloOlive',  op: 0.45 },
    { x: 700, y: 480, r: 90,  id: 'haloOlive',  op: 0.55 },
    { x: 720, y: 305, r: 70,  id: 'haloOlive',  op: 0.50 },
    { x: active.x - 18, y: active.y, r: 76, id: 'haloOrange', op: 0.95 },
  ];
  haloSpec.forEach(h => make('circle', {
    cx: h.x, cy: h.y, r: h.r, fill: `url(#${h.id})`, opacity: h.op,
  }, halos));

  // ---------- background fan-out paths ----------
  const pathsG = sel('hero-network-paths');
  dests.forEach((d, i) => {
    if (i === activeIdx) return;
    const cls = rand() < 0.22 ? 'faint' : (rand() < 0.18 ? 'bold' : '');
    make('path', { d: cubicPath(origin, d) }, pathsG, cls);
  });

  // ---------- destination cluster dots ----------
  const clusterG = sel('hero-network-cluster');
  dests.forEach((d, i) => {
    if (i === activeIdx) return;
    const roll = rand();
    let cls = '';
    if (roll < 0.18) cls = 'faint';
    else if (roll < 0.46) cls = 'muted';
    else if (roll < 0.55) cls = 'warm';
    make('circle', { cx: d.x, cy: d.y, r: d.r }, clusterG, cls);
  });

  // ---------- active path + endpoint ----------
  const activeG = sel('hero-network-active');
  make('path', { d: cubicPath(origin, active) }, activeG);
  const flow = make('path', { d: cubicPath(origin, active) }, activeG, 'flow');
  flow.setAttribute('stroke', '#FFEDD5');
  flow.setAttribute('stroke-width', '1.8');
  flow.setAttribute('opacity', '0.85');
  make('circle', { cx: active.x, cy: active.y, r: 5 }, activeG);

  // ---------- origin ("you") node ----------
  const originG = sel('hero-network-origin');
  make('circle', { cx: origin.x, cy: origin.y, r: 8 }, originG, 'ring');
  make('circle', { cx: origin.x, cy: origin.y, r: 6 }, originG, 'core');

  // ---------- labels ----------
  const labelsG = sel('hero-network-labels');
  const addLabel = (x, y, text, cls) => {
    const t = make('text', { x, y }, labelsG, cls);
    t.textContent = text;
  };
  addLabel(origin.x - 4, origin.y + 28, 'your prompt', 'origin');
  addLabel(origin.x - 4, origin.y + 42, 'OpenAI SDK · curl · /v1/chat', 'dim');
  // Active labels stacked above the dot, centered, so they never clip the right edge
  const topLabel = make('text', { x: active.x, y: active.y - 24, 'text-anchor': 'middle' }, labelsG);
  topLabel.textContent = 'layer-split target';
  const subLabel = make('text', { x: active.x, y: active.y - 10, 'text-anchor': 'middle' }, labelsG, 'dim');
  subLabel.textContent = 'CUDA · Metal · Vulkan · CPU';
})();
