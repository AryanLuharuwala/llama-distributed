(function () {
  const root = document.querySelector('[data-primer]');
  if (!root) return;
  const SVGNS = 'http://www.w3.org/2000/svg';
  const XLINK = 'http://www.w3.org/1999/xlink';
  const edgesGroup = root.querySelector('[data-primer-edges]');
  const packetsGroup = root.querySelector('[data-primer-packets]');

  // Edges with anchor coords matching the inline static SVG layout above:
  //   chat:    translate(30,30)   w=220 h=330  → right edge x=250, request y=110, response y=200
  //   server:  translate(310,60)  w=200 h=110  → top y=60, mid y=115, right x=510, top-mid x=420, bottom-mid (415,170)
  //   sqlite:  translate(350,230) w=130 h=50   → top-mid x=415,y=230
  //   rig0:    translate(570,100) w=130 h=220  → left x=570, right x=700, mid y=210
  //   rig1:    translate(720,100) → right x=850
  //   rig2:    translate(870,100) → right x=1000
  //   rig3:    translate(1020,100) w=150       → left x=1020, top-mid x=1095,y=100, right x=1170
  const edges = [
    { id: 'e1', d: 'M250,110 L310,90',                           plane: 'control', label: 'POST /v1/chat',  labelPos: { x: 280, y: 88 },  packets: 1, dur: 1.8 },
    { id: 'e2', d: 'M415,170 L415,230',                          plane: 'control', label: 'auth · lookup',  labelPos: { x: 478, y: 205 }, packets: 0 },
    { id: 'e3', d: 'M510,115 L570,210',                          plane: 'control', label: 'WebSocket',      labelPos: { x: 545, y: 152 }, packets: 1, dur: 1.8, begin: 0.6 },
    { id: 'e4', d: 'M700,210 L720,210',                          plane: 'data',    label: 'TCP :7701',      labelPos: { x: 785, y: 198 }, packets: 2, dur: 0.7 },
    { id: 'e5', d: 'M850,210 L870,210',                          plane: 'data',    label: '',                                              packets: 2, dur: 0.7, begin: 0.23 },
    { id: 'e6', d: 'M1000,210 L1020,210',                        plane: 'data',    label: '',                                              packets: 2, dur: 0.7, begin: 0.46 },
    { id: 'e7', d: 'M1095,100 L1095,42 L410,42 L410,60',         plane: 'control', label: 'WS · token',     labelPos: { x: 760, y: 32 },  packets: 2, dur: 2.4 },
    { id: 'e8', d: 'M310,140 L250,200',                          plane: 'control', label: 'SSE stream',     labelPos: { x: 286, y: 158 }, packets: 2, dur: 1.6 },
  ];

  function svgEl(tag, attrs, parent) {
    const el = document.createElementNS(SVGNS, tag);
    if (attrs) for (const k in attrs) {
      if (k === 'href') {
        el.setAttribute('href', attrs[k]);
        el.setAttributeNS(XLINK, 'xlink:href', attrs[k]);
      } else {
        el.setAttribute(k, attrs[k]);
      }
    }
    if (parent) parent.appendChild(el);
    return el;
  }

  edges.forEach(e => {
    const pathId = 'primer-' + e.id;
    svgEl('path', {
      id: pathId,
      d: e.d,
      class: 'primer-edge primer-edge-' + e.plane,
    }, edgesGroup);

    if (e.label) {
      const t = svgEl('text', {
        x: e.labelPos.x,
        y: e.labelPos.y,
        class: 'primer-edge-label',
        'text-anchor': 'middle',
      }, edgesGroup);
      t.textContent = e.label;
    }

    if (e.packets > 0) {
      for (let i = 0; i < e.packets; i++) {
        const pkt = svgEl('circle', {
          r: 4,
          class: 'primer-packet primer-packet-' + e.plane,
        }, packetsGroup);
        const beginAt = (e.begin || 0) + (e.dur * i / e.packets);
        const anim = svgEl('animateMotion', {
          dur: e.dur + 's',
          repeatCount: 'indefinite',
          begin: beginAt + 's',
          rotate: 'auto',
        }, pkt);
        svgEl('mpath', { href: '#' + pathId }, anim);
      }
    }
  });
})();
