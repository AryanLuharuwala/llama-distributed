(function () {
  const root = document.querySelector('[data-explorer]');
  if (!root) return;
  const SVGNS = 'http://www.w3.org/2000/svg';
  const svg = root.querySelector('[data-explorer-svg]');

  // ============================================================
  //  Component data — every node here maps to a real piece of the
  //  project (Go control plane, C++ workers, SQLite, etc.).
  // ============================================================
  const C = {
    client: {
      pos: { x: 75, y: 300 }, kind: 'client',
      label: 'your app', sub: 'OpenAI SDK · curl',
      role: 'Client',
      protocol: 'HTTPS · /v1/chat/completions',
      body: "Standard OpenAI client pointed at your pool's base_url, with an sk-dist-… bearer token. Sends Chat Completions requests; reads responses as Server-Sent Events or buffered JSON.",
      interesting: "No vendor lock-in. Any tool that speaks OpenAI's API works unchanged — Python SDK, Node SDK, curl, LangChain, Cursor, Continue, you name it.",
    },
    server: {
      pos: { x: 290, y: 300 }, kind: 'control',
      label: 'dist-server', sub: 'Go · :8080 · control plane',
      role: 'Control plane (Go)',
      protocol: 'HTTPS · WebSocket · :8080',
      body: "Validates API keys, resolves the pool, picks an online rig, brokers the inference request over WebSocket, and streams responses back to the client as SSE.",
      interesting: "One Go binary, SQLite-backed. All auth, routing, and quota enforcement happen here — no separate service mesh, no clustered DB. Per-pool subdomains route to the same process by slug.",
    },
    sqlite: {
      pos: { x: 290, y: 480 }, kind: 'storage',
      label: 'SQLite', sub: 'pools · keys · rigs',
      role: 'Persistent state',
      protocol: 'Local file · WAL mode',
      body: "Single source of truth for pool memberships, hashed API keys, rig identities, and user accounts. Lives next to dist-server on disk.",
      interesting: "WAL mode keeps concurrent readers honest; foreign keys enforce referential integrity. WebSocket fans out pool_rigs_changed to every browser tab in real time — no polling.",
    },
    oauth: {
      pos: { x: 290, y: 120 }, kind: 'auth',
      label: 'GitHub OAuth', sub: 'operator login',
      role: 'Identity provider',
      protocol: 'OAuth 2.0',
      body: "Operators sign in with GitHub. dist-server stores user records keyed by GitHub login and signs JWT session cookies for the dashboard.",
      interesting: "API keys (sk-dist-…) are minted by signed-in operators in the dashboard. They're shown once, hashed at rest, scoped to a single pool.",
    },
    rig0: {
      pos: { x: 530, y: 90 }, kind: 'rig', accel: 'CUDA',
      label: 'rig 0', sub: 'CUDA · L 0–19',
      role: 'Worker · entry node (C++)',
      protocol: 'WebSocket :8080 + TCP :7701',
      body: "Receives the tokenized prompt over WebSocket, runs llama_decode() on its layer slice, and streams 4 KB activation tensors to the next rig over TCP :7701.",
      interesting: "Zero modifications to llama.cpp — the rig is a pure library call. A 3-thread RX/compute/TX double-buffer keeps the GPU near 100% utilization under sustained load.",
    },
    rig1: {
      pos: { x: 660, y: 210 }, kind: 'rig', accel: 'Metal',
      label: 'rig 1', sub: 'Metal · L 20–39',
      role: 'Worker · pipeline middle',
      protocol: 'TCP :7701',
      body: "Pulls activations from rig 0, runs its layer slice on Apple Silicon, forwards activations to rig 2.",
      interesting: "Same C++ binary as the CUDA rig — Metal vs CUDA is a llama.cpp compile-time backend swap. The wire protocol does not care about the accelerator.",
    },
    rig2: {
      pos: { x: 780, y: 340 }, kind: 'rig', accel: 'Vulkan',
      label: 'rig 2', sub: 'Vulkan · L 40–59',
      role: 'Worker · pipeline middle',
      protocol: 'TCP :7701',
      body: "AMD or Intel GPU running through Vulkan compute shaders. Same activation streaming, same binary protocol.",
      interesting: "Mixing accelerator vendors used to mean rewriting kernels. Layer-parallel + activation streaming means each rig just runs llama.cpp's existing backend for its hardware.",
    },
    rig3: {
      pos: { x: 880, y: 470 }, kind: 'rig', accel: 'CPU',
      label: 'rig 3', sub: 'CPU · L 60–79',
      role: 'Worker · exit node',
      protocol: 'WebSocket back to dist-server',
      body: "Last node in the pipeline. Produces logits, samples a token, sends it back to dist-server over WebSocket. Repeats until end-of-stream.",
      interesting: "Even a CPU rig works. The VRAM-proportional planner gives slow nodes fewer layers, so the pipeline self-balances to its weakest link instead of stalling on it.",
    },
  };

  // ============================================================
  //  Edges — directed, with protocol plane and tour-step ownership.
  // ============================================================
  const E = [
    { id: 0, from: 'client',  to: 'server', plane: 'control', step: 0, label: 'POST /v1/chat · sk-dist-…', scene: ['flow','control'] },
    { id: 1, from: 'server',  to: 'sqlite', plane: 'control', step: 1, label: 'lookup', scene: ['flow','control'] },
    { id: 2, from: 'oauth',   to: 'server', plane: 'control',           label: 'OAuth 2.0', scene: ['control'] },
    { id: 3, from: 'server',  to: 'rig0',   plane: 'control', step: 2, label: 'WS · forwarded request', scene: ['flow','control','topology'] },
    { id: 4, from: 'rig0',    to: 'rig1',   plane: 'data',    step: 3, label: 'TCP :7701 · 4 KB frame', scene: ['flow','data','topology'] },
    { id: 5, from: 'rig1',    to: 'rig2',   plane: 'data',    step: 3, label: 'TCP :7701', scene: ['flow','data','topology'] },
    { id: 6, from: 'rig2',    to: 'rig3',   plane: 'data',    step: 3, label: 'TCP :7701', scene: ['flow','data','topology'] },
    { id: 7, from: 'rig3',    to: 'server', plane: 'control', step: 4, label: 'WS · token', scene: ['flow','control'], curve: 'bow-down' },
    { id: 8, from: 'server',  to: 'client', plane: 'control', step: 4, label: 'SSE · stream chunk', scene: ['flow','control'], curve: 'bow-down' },
  ];

  // ============================================================
  //  Tour steps — narrate the request lifecycle.
  // ============================================================
  const STEPS = [
    { idx: 0, title: 'Client request', desc: "Your app POSTs /v1/chat/completions to your pool's base_url. Auth is an sk-dist-… bearer token.", focus: ['client','server'], edges: [0] },
    { idx: 1, title: 'Auth & routing', desc: "dist-server hashes the bearer token, looks it up in SQLite, resolves the pool, and picks an online rig.", focus: ['server','sqlite'], edges: [1] },
    { idx: 2, title: 'Hand off to entry rig', desc: "The control plane streams the request over the rig's persistent WebSocket. The entry rig tokenizes the prompt and warms its layer slice.", focus: ['server','rig0'], edges: [3] },
    { idx: 3, title: 'Activations stream node-to-node', desc: "Each rig runs llama_decode() on its layer slice, then forwards a 4 KB activation tensor over TCP :7701. RX/compute/TX threads overlap so GPUs stay hot.", focus: ['rig0','rig1','rig2','rig3'], edges: [4,5,6] },
    { idx: 4, title: 'Tokens stream back', desc: "The exit rig samples a token and sends it back to dist-server. dist-server pushes the chunk to the client as Server-Sent Events. Repeat per token.", focus: ['rig3','server','client'], edges: [7,8] },
  ];

  // ============================================================
  //  Scenes — what each tab shows / hides.
  // ============================================================
  const SCENES = {
    flow:     { eyebrow: 'Request flow',     headline: 'A round-trip in five steps.',     blurb: "Click any node to read its role. Or run the tour and watch a single inference request walk through every component.", tour: true },
    topology: { eyebrow: 'Pool topology',    headline: 'Layer-split across whatever GPUs you have.',  blurb: "Heterogeneous workers sit in a pipeline. The planner gives more VRAM more layers, so the slowest node never stalls the rest.", tour: false },
    control:  { eyebrow: 'Control plane',    headline: 'Auth, routing, and SSE — all in Go.',  blurb: "dist-server is the only thing facing the public internet. It does login, key minting, pool routing, and response streaming. SQLite holds the state.", tour: false },
    data:     { eyebrow: 'Data plane',       headline: 'Activations stream over plain TCP.',  blurb: "No gRPC, no Protobuf — just 24-byte headers and raw tensor bytes on TCP :7701. One packet per token, often a single Ethernet frame.", tour: false },
  };

  // ============================================================
  //  Helpers
  // ============================================================
  const sel = (cls) => root.querySelector('.' + cls);
  const make = (tag, attrs, parent, cls) => {
    const el = document.createElementNS(SVGNS, tag);
    if (cls) el.setAttribute('class', cls);
    for (const k in attrs) el.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(el);
    return el;
  };
  const edgePath = (a, b, curve) => {
    if (curve === 'bow-down') {
      const mx = (a.x + b.x) / 2;
      const my = Math.max(a.y, b.y) + 140;
      return `M${a.x},${a.y + 14} Q${mx},${my} ${b.x},${b.y + 14}`;
    }
    if (curve === 'bow-up') {
      const mx = (a.x + b.x) / 2;
      const my = Math.min(a.y, b.y) - 80;
      return `M${a.x},${a.y - 14} Q${mx},${my} ${b.x},${b.y - 14}`;
    }
    return `M${a.x},${a.y} L${b.x},${b.y}`;
  };

  // ============================================================
  //  Background cluster + halos (decorative — preserves the
  //  fan-out network aesthetic from the hero, but quieter).
  // ============================================================
  function buildBackground() {
    const halos = sel('explorer-halos');
    const halo = (x, y, r, op, fill) => {
      const c = make('circle', { cx: x, cy: y, r, fill: `url(#${fill})`, opacity: op }, halos);
      return c;
    };
    halo(220, 170, 110, 0.55, 'explHaloOlive');
    halo(310, 480, 130, 0.50, 'explHaloOlive');
    halo(620, 130, 90, 0.50, 'explHaloOlive');
    halo(800, 460, 110, 0.55, 'explHaloOlive');
    halo(740, 290, 140, 0.45, 'explHaloOlive');

    // Background pseudo-cluster of muted dots, biased toward the worker side
    const cluster = sel('explorer-cluster');
    let seed = 4242;
    const rand = () => { seed = (seed * 9301 + 49297) % 233280; return seed / 233280; };
    for (let i = 0; i < 60; i++) {
      const x = 540 + rand() * 380;
      const y = 60 + rand() * 500;
      const r = 1.2 + rand() * 2.2;
      const op = 0.10 + rand() * 0.18;
      make('circle', { cx: x, cy: y, r, fill: '#3A4A1F', opacity: op }, cluster);
    }
  }

  // ============================================================
  //  Edges
  // ============================================================
  const edgeEls = {};
  function buildEdges() {
    const g = sel('explorer-edges');
    const labelG = sel('explorer-edge-labels');
    E.forEach(e => {
      const a = C[e.from].pos, b = C[e.to].pos;
      const path = make('path', {
        d: edgePath(a, b, e.curve),
        'data-edge': e.id,
        'data-plane': e.plane,
      }, g, 'explorer-edge ' + (e.plane === 'data' ? 'is-data' : 'is-control'));
      // Label
      const dx = b.x - a.x, dy = b.y - a.y;
      let lx = (a.x + b.x) / 2, ly = (a.y + b.y) / 2 - 6;
      if (e.curve === 'bow-down') { ly = Math.max(a.y, b.y) + 56; }
      if (e.curve === 'bow-up')   { ly = Math.min(a.y, b.y) - 48; }
      const t = make('text', { x: lx, y: ly, 'text-anchor': 'middle', 'data-edge-label': e.id }, labelG, 'explorer-edge-label');
      t.textContent = e.label;
      edgeEls[e.id] = { path, label: t, def: e };
    });
  }

  // ============================================================
  //  Major nodes
  // ============================================================
  const nodeEls = {};
  function buildNodes() {
    const g = sel('explorer-nodes');
    Object.keys(C).forEach(key => {
      const c = C[key];
      const ng = make('g', {
        'data-node': key,
        transform: `translate(${c.pos.x}, ${c.pos.y})`,
        tabindex: 0,
        role: 'button',
        'aria-label': c.label + ' — ' + c.role,
      }, g, 'explorer-node kind-' + c.kind);
      // hit area
      make('rect', { x: -50, y: -36, width: 100, height: 72, rx: 12, fill: 'transparent' }, ng, 'hit');
      // halo
      make('circle', { cx: 0, cy: 0, r: 22, fill: 'currentColor', opacity: 0 }, ng, 'halo');
      // ring (focus / selected)
      make('circle', { cx: 0, cy: 0, r: 13, fill: 'none', 'stroke-width': 2 }, ng, 'ring');
      // core dot
      make('circle', { cx: 0, cy: 0, r: 7 }, ng, 'core');
      // labels
      const t1 = make('text', { x: 0, y: 26, 'text-anchor': 'middle' }, ng, 'label');
      t1.textContent = c.label;
      const t2 = make('text', { x: 0, y: 39, 'text-anchor': 'middle' }, ng, 'sub');
      t2.textContent = c.sub;
      nodeEls[key] = ng;
    });
  }

  // ============================================================
  //  State
  // ============================================================
  const state = {
    scene: 'flow',
    selected: null,
    hovered: null,
    tourStep: -1,
    tourPlaying: false,
    tourTimer: null,
  };

  // ============================================================
  //  Highlight rules
  // ============================================================
  function applyHighlight() {
    const scene = SCENES[state.scene];
    const visibleNodes = new Set(getSceneNodes(state.scene));
    const visibleEdgeIds = new Set(E.filter(e => e.scene && e.scene.indexOf(state.scene) !== -1).map(e => e.id));

    // Node visibility / dim state
    Object.keys(nodeEls).forEach(key => {
      const el = nodeEls[key];
      el.classList.toggle('is-hidden', !visibleNodes.has(key));
      el.classList.remove('is-focus', 'is-dim', 'is-selected', 'is-hover-neighbor');
    });
    // Edges
    Object.keys(edgeEls).forEach(id => {
      const { path, label } = edgeEls[id];
      const visible = visibleEdgeIds.has(+id);
      path.classList.toggle('is-hidden', !visible);
      label.classList.toggle('is-hidden', !visible);
      path.classList.remove('is-active', 'is-dim', 'is-hover');
      label.classList.remove('is-active', 'is-dim', 'is-hover');
    });

    // Hover neighborhood
    if (state.hovered) {
      highlightNeighborhood(state.hovered, 'is-hover');
    }
    // Selected node
    if (state.selected && nodeEls[state.selected]) {
      nodeEls[state.selected].classList.add('is-selected');
    }
    // Tour step
    if (state.tourStep >= 0) {
      const step = STEPS[state.tourStep];
      step.focus.forEach(k => {
        if (nodeEls[k]) nodeEls[k].classList.add('is-focus');
      });
      Object.keys(nodeEls).forEach(k => {
        if (step.focus.indexOf(k) === -1 && !nodeEls[k].classList.contains('is-hidden')) {
          nodeEls[k].classList.add('is-dim');
        }
      });
      step.edges.forEach(id => {
        const { path, label } = edgeEls[id];
        path.classList.add('is-active');
        label.classList.add('is-active');
      });
      Object.keys(edgeEls).forEach(id => {
        if (step.edges.indexOf(+id) === -1 && !edgeEls[id].path.classList.contains('is-hidden')) {
          edgeEls[id].path.classList.add('is-dim');
          edgeEls[id].label.classList.add('is-dim');
        }
      });
    }
  }

  function highlightNeighborhood(nodeKey, cls) {
    const neighbors = new Set([nodeKey]);
    E.forEach(e => {
      if (e.scene && e.scene.indexOf(state.scene) === -1) return;
      if (e.from === nodeKey) neighbors.add(e.to);
      if (e.to === nodeKey) neighbors.add(e.from);
      if (e.from === nodeKey || e.to === nodeKey) {
        edgeEls[e.id].path.classList.add(cls);
        edgeEls[e.id].label.classList.add(cls);
      }
    });
    Object.keys(nodeEls).forEach(k => {
      if (nodeEls[k].classList.contains('is-hidden')) return;
      if (neighbors.has(k)) {
        nodeEls[k].classList.add(cls === 'is-hover' ? 'is-hover-neighbor' : 'is-focus');
      } else {
        nodeEls[k].classList.add('is-dim');
      }
    });
  }

  function getSceneNodes(sceneKey) {
    const set = new Set();
    E.forEach(e => {
      if (e.scene && e.scene.indexOf(sceneKey) !== -1) {
        set.add(e.from); set.add(e.to);
      }
    });
    // include all nodes even if no incident edge in flow scene to avoid orphans
    return Array.from(set);
  }

  // ============================================================
  //  Panel
  // ============================================================
  const panelEls = {
    eyebrow: root.querySelector('[data-panel-eyebrow]'),
    title: root.querySelector('[data-panel-title]'),
    sub: root.querySelector('[data-panel-sub]'),
    role: root.querySelector('[data-panel-role]'),
    protocol: root.querySelector('[data-panel-protocol]'),
    body: root.querySelector('[data-panel-body]'),
    interesting: root.querySelector('[data-panel-interesting]'),
    detail: root.querySelector('[data-panel-detail]'),
  };
  function paintPanelForScene() {
    const s = SCENES[state.scene];
    panelEls.eyebrow.textContent = s.eyebrow;
    panelEls.title.textContent = s.headline;
    panelEls.sub.textContent = s.blurb;
    panelEls.role.textContent = '—';
    panelEls.protocol.textContent = '—';
    panelEls.body.textContent = 'Click any node to pin its details here. Hover to highlight neighbours without selecting.';
    panelEls.detail.hidden = true;
  }
  function paintPanelForNode(key) {
    const c = C[key];
    panelEls.eyebrow.textContent = SCENES[state.scene].eyebrow + ' · ' + (c.kind === 'rig' ? 'worker' : c.kind);
    panelEls.title.textContent = c.label + (c.accel ? ' · ' + c.accel : '');
    panelEls.sub.textContent = c.sub;
    panelEls.role.textContent = c.role;
    panelEls.protocol.textContent = c.protocol;
    panelEls.body.textContent = c.body;
    panelEls.interesting.textContent = c.interesting;
    panelEls.detail.hidden = false;
  }
  function paintPanelForStep(step) {
    panelEls.eyebrow.textContent = 'Step ' + (step.idx + 1) + ' of ' + STEPS.length;
    panelEls.title.textContent = step.title;
    panelEls.sub.textContent = step.desc;
    panelEls.role.textContent = step.focus.map(k => C[k].label).join(' · ');
    const planes = new Set(step.edges.map(id => edgeEls[id].def.plane));
    panelEls.protocol.textContent = Array.from(planes).map(p => p === 'data' ? 'data plane (TCP :7701)' : 'control plane (HTTPS / WS / SSE)').join('  +  ');
    panelEls.body.textContent = step.edges.map(id => edgeEls[id].def.label).join('  →  ');
    panelEls.detail.hidden = true;
  }

  // ============================================================
  //  Tour controls
  // ============================================================
  const tourEls = {
    prev: root.querySelector('[data-tour-prev]'),
    next: root.querySelector('[data-tour-next]'),
    play: root.querySelector('[data-tour-play]'),
    playLabel: root.querySelector('[data-tour-play-label]'),
    counter: root.querySelector('[data-tour-counter]'),
    label: root.querySelector('[data-tour-label]'),
    container: root.querySelector('[data-tour]'),
  };
  function paintTour() {
    const playing = state.tourPlaying;
    tourEls.play.classList.toggle('is-playing', playing);
    tourEls.playLabel.textContent = playing ? 'Pause' : (state.tourStep >= 0 ? 'Resume' : 'Run the tour');
    if (state.tourStep < 0) {
      tourEls.counter.textContent = '—';
      tourEls.label.textContent = 'Walk through a single inference request, end-to-end.';
    } else {
      const step = STEPS[state.tourStep];
      tourEls.counter.textContent = (state.tourStep + 1) + ' / ' + STEPS.length;
      tourEls.label.textContent = step.title;
    }
  }
  function setTourStep(n, opts) {
    opts = opts || {};
    state.tourStep = n;
    if (n < 0 || n >= STEPS.length) {
      state.tourStep = -1;
      stopTour();
      paintPanelForScene();
    } else {
      paintPanelForStep(STEPS[n]);
    }
    applyHighlight();
    paintTour();
  }
  function startTour() {
    if (!SCENES[state.scene].tour) {
      // Switch to flow scene to play
      setScene('flow', { silent: true });
    }
    state.tourPlaying = true;
    if (state.tourStep < 0) state.tourStep = 0;
    setTourStep(state.tourStep);
    if (state.tourTimer) clearInterval(state.tourTimer);
    state.tourTimer = setInterval(() => {
      let next = state.tourStep + 1;
      if (next >= STEPS.length) next = 0;
      setTourStep(next);
    }, 4200);
  }
  function stopTour() {
    state.tourPlaying = false;
    if (state.tourTimer) clearInterval(state.tourTimer);
    state.tourTimer = null;
    paintTour();
  }
  function toggleTour() {
    if (state.tourPlaying) stopTour();
    else startTour();
  }

  // ============================================================
  //  Scenes / tabs
  // ============================================================
  function setScene(key, opts) {
    opts = opts || {};
    state.scene = key;
    state.selected = null;
    if (state.tourStep >= 0 && !SCENES[key].tour) {
      stopTour();
      state.tourStep = -1;
    }
    // tab UI
    root.querySelectorAll('[data-scene]').forEach(btn => {
      const isActive = btn.dataset.scene === key;
      btn.classList.toggle('is-active', isActive);
      btn.setAttribute('aria-selected', isActive ? 'true' : 'false');
    });
    paintPanelForScene();
    applyHighlight();
    paintTour();
  }

  // ============================================================
  //  Wiring
  // ============================================================
  function wire() {
    // Tabs
    root.querySelectorAll('[data-scene]').forEach(btn => {
      btn.addEventListener('click', () => setScene(btn.dataset.scene));
    });
    // Tour
    tourEls.play.addEventListener('click', toggleTour);
    tourEls.next.addEventListener('click', () => {
      stopTour();
      setTourStep(Math.min(STEPS.length - 1, (state.tourStep < 0 ? 0 : state.tourStep + 1)));
    });
    tourEls.prev.addEventListener('click', () => {
      stopTour();
      setTourStep(Math.max(0, (state.tourStep < 0 ? 0 : state.tourStep - 1)));
    });
    // Per-node listeners (mouseenter/mouseleave don't bubble — no flicker
    // when moving between a node's child SVG elements).
    Object.keys(nodeEls).forEach(key => {
      const el = nodeEls[key];
      const onEnter = () => {
        if (el.classList.contains('is-hidden')) return;
        state.hovered = key;
        applyHighlight();
      };
      const onLeave = () => {
        if (state.hovered === key) {
          state.hovered = null;
          applyHighlight();
        }
      };
      const onClick = () => {
        if (el.classList.contains('is-hidden')) return;
        stopTour();
        state.tourStep = -1;
        state.selected = key;
        paintPanelForNode(key);
        applyHighlight();
        paintTour();
      };
      el.addEventListener('mouseenter', onEnter);
      el.addEventListener('mouseleave', onLeave);
      el.addEventListener('focus', onEnter);
      el.addEventListener('blur', onLeave);
      el.addEventListener('click', onClick);
      el.addEventListener('keydown', (ev) => {
        if (ev.key === 'Enter' || ev.key === ' ') {
          ev.preventDefault();
          onClick();
        }
      });
    });
  }

  // ============================================================
  //  Live stats HUD (project-relevant fluctuating numbers — gives
  //  the diagram a feeling of "this is alive")
  // ============================================================
  const statEls = {
    tps: root.querySelector('[data-stat-tps]'),
    rigs: root.querySelector('[data-stat-rigs]'),
    p50: root.querySelector('[data-stat-p50]'),
    reqs: root.querySelector('[data-stat-reqs]'),
  };
  let reqCounter = 47392;
  function tickStats() {
    if (!statEls.tps) return;
    // Tokens/sec: drift around 140 with mild noise
    const tps = (132 + Math.random() * 26).toFixed(1);
    statEls.tps.textContent = tps;
    // p50 latency: 32-48ms band
    const p50 = Math.round(32 + Math.random() * 14);
    statEls.p50.innerHTML = p50 + '<span class="stat-unit">ms</span>';
    // active rigs reflects what's visible in the current scene
    const rigKeys = ['rig0','rig1','rig2','rig3'];
    const visibleRigs = rigKeys.filter(k => nodeEls[k] && !nodeEls[k].classList.contains('is-hidden')).length;
    statEls.rigs.textContent = visibleRigs;
    // request counter: increments by ~10–40 per tick
    reqCounter += 7 + Math.floor(Math.random() * 35);
    statEls.reqs.textContent = reqCounter.toLocaleString('en-US');
  }
  setInterval(tickStats, 1500);
  tickStats();

  // ============================================================
  //  Auto-start the tour when the explorer scrolls into view.
  //  Pauses again if the user scrolls away.
  // ============================================================
  let autoStarted = false;
  let userInteracted = false;
  ['mouseenter', 'click', 'keydown'].forEach(ev => {
    root.addEventListener(ev, () => { userInteracted = true; }, { once: true });
  });
  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting && entry.intersectionRatio > 0.35) {
          if (!autoStarted && !userInteracted) {
            autoStarted = true;
            startTour();
          }
        } else if (state.tourPlaying && autoStarted && !userInteracted) {
          // Pause when scrolled away to save cycles + battery
          stopTour();
          autoStarted = false;
        }
      });
    }, { threshold: [0, 0.35, 0.6] });
    io.observe(root);
  }

  // ============================================================
  //  Init
  // ============================================================
  buildBackground();
  buildEdges();
  buildNodes();
  paintPanelForScene();
  applyHighlight();
  paintTour();
  wire();
})();
