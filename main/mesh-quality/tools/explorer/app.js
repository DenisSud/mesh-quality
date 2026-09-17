// Mesh-quality data explorer. No external deps: everything renders on canvas.
//
// Data contract (injected as JSON in #payload):
//   DATA.meta    {n_train, n_test, split_note, labels:[...], features:[...]}
//   DATA.items   [{id, split, labels:[0/1 per DATA.meta.labels], quality, feats:{...},
//                  thumb: <key into DATA.thumbs>, mesh: <key into DATA.meshes>, open_path}]
//   DATA.stats   {prevalence:{label:pct}, cooc:[[...]], combos:[[combo,count]], ...}
//   DATA.auc     {labels:[...], features:[...], values:[[...]]}
//   DATA.baseline [{label, prec, rec, f1, pos}]
//   DATA.thumbs  {key: dataURL}
//   DATA.meshes  {key: {nv, nf, pos: b64(uint16, bbox-normalised), idx: b64(uint16|uint32), dtype}}
const DATA = JSON.parse(document.getElementById('payload').textContent);

const LABELS = DATA.meta.labels;
const FEATURES = DATA.meta.features;
const BY_ID = new Map(DATA.items.map(it => [it.id, it]));
const POS = '#ff7a45', NEG = '#46506b', ACCENT = '#63b3ed';

const b64ToBuf = s => {
  const bin = atob(s), buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return buf.buffer;
};

// ---------------------------------------------------------------- overview ---
function renderOverview() {
  const prev = DATA.stats.prevalence;
  const rows = [...LABELS].concat(['quality']).map(name => {
    const v = name === 'quality' ? DATA.stats.quality_pct : prev[name];
    return `<div class="bar-row"><span class="bar-label">${name}</span>
      <span class="bar-track"><span class="bar-fill" style="width:${v}%"></span></span>
      <span class="bar-val">${v.toFixed(1)}%</span></div>`;
  }).join('');
  document.getElementById('prevalence').innerHTML = rows;

  const labels = LABELS.concat(['quality']);
  const cooc = DATA.stats.cooc_with_quality;
  let html = '<table class="heat"><thead><tr><th></th>' +
    labels.map(l => `<th title="${l}">${l.slice(0, 4)}</th>`).join('') + '</tr></thead><tbody>';
  let max = 0;
  cooc.forEach(r => r.forEach(v => { max = Math.max(max, v); }));
  cooc.forEach((row, i) => {
    html += `<tr><th class="rowhead" title="${labels[i]}">${labels[i]}</th>`;
    row.forEach((v, j) => {
      const t = max ? Math.log1p(v) / Math.log1p(max) : 0;
      html += `<td style="background:rgba(255,122,69,${(t * 0.9).toFixed(3)})"
        title="${labels[i]} &amp; ${labels[j]}: ${v}">${v > 0 ? v : ''}</td>`;
    });
    html += '</tr>';
  });
  document.getElementById('cooc').innerHTML = html + '</tbody></table>';

  document.getElementById('combos').innerHTML = DATA.stats.combos.map(([combo, count]) =>
    `<div class="bar-row"><span class="bar-label wide">${combo || '(clean)'}</span>
     <span class="bar-track"><span class="bar-fill alt" style="width:${(count / DATA.stats.combos_max * 100).toFixed(1)}%"></span></span>
     <span class="bar-val">${count}</span></div>`).join('');

  document.getElementById('overview-notes').innerHTML = DATA.stats.notes.map(n => `<li>${n}</li>`).join('');
}

// ----------------------------------------------------------------- scatter ---
const S = {x: 'log_faces', y: 'bbox_flatness', color: 'quality', logx: true, logy: false, subset: 'all', items: []};

function applySubset() {
  S.items = DATA.items.filter(it => {
    if (S.subset === 'all') return true;
    if (S.subset === 'quality') return it.quality === 1;
    return it.labels[LABELS.indexOf(S.subset)] === 1;
  });
}

function drawScatter() {
  const cv = document.getElementById('scatter');
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w * dpr; cv.height = h * dpr;
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const m = {l: 62, r: 14, t: 12, b: 34};
  const pw = w - m.l - m.r, ph = h - m.t - m.b;

  // log_faces / faces_per_component are already log10 — never log them twice.
  const PRE_LOGGED = new Set(['log_faces', 'faces_per_component']);
  const useLogX = S.logx && !PRE_LOGGED.has(S.x);
  const useLogY = S.logy && !PRE_LOGGED.has(S.y);
  const pts = [];
  for (const it of S.items) {
    const rawX = it.feats[S.x], rawY = it.feats[S.y];
    if (!Number.isFinite(rawX) || !Number.isFinite(rawY)) continue;
    const x = useLogX ? Math.log10(Math.max(rawX, 1e-9)) : rawX;
    const y = useLogY ? Math.log10(Math.max(rawY, 1e-9)) : rawY;
    const positive = S.color === 'quality' ? it.quality === 1 : it.labels[LABELS.indexOf(S.color)] === 1;
    pts.push({it, x, y, rawX, rawY, positive});
  }
  if (!pts.length) return;
  const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
  let [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  let [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  const pad = v => { const d = (v[1] - v[0]) * 0.05 || 1e-3; return [v[0] - d, v[1] + d]; };
  [x0, x1] = pad([x0, x1]); [y0, y1] = pad([y0, y1]);
  S.view = {x0, x1, y0, y1, m, pw, ph};
  S.useLog = {x: useLogX, y: useLogY};
  const sx = v => m.l + (v - x0) / (x1 - x0) * pw;
  const sy = v => m.t + ph - (v - y0) / (y1 - y0) * ph;

  // grid + ticks
  ctx.strokeStyle = 'rgba(255,255,255,.08)'; ctx.fillStyle = '#8b95a7';
  ctx.font = '11px ui-monospace, monospace';
  for (let i = 0; i <= 5; i++) {
    const gx = x0 + (x1 - x0) * i / 5, gy = y0 + (y1 - y0) * i / 5;
    ctx.beginPath(); ctx.moveTo(sx(gx), m.t); ctx.lineTo(sx(gx), m.t + ph); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(m.l, sy(gy)); ctx.lineTo(m.l + pw, sy(gy)); ctx.stroke();
    const lx = useLogX ? Math.pow(10, gx) : gx, ly = useLogY ? Math.pow(10, gy) : gy;
    ctx.textAlign = 'center'; ctx.fillText(fmtTick(lx), sx(gx), m.t + ph + 16);
    ctx.textAlign = 'right'; ctx.fillText(fmtTick(ly), m.l - 6, sy(gy) + 4);
  }
  ctx.fillStyle = '#c8d0dc'; ctx.textAlign = 'center';
  ctx.fillText(S.x, m.l + pw / 2, h - 4);
  ctx.save(); ctx.translate(12, m.t + ph / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText(S.y, 0, 0); ctx.restore();

  S.screen = [];
  for (const p of pts) {
    const px = sx(p.x), py = sy(p.y);
    S.screen.push({px, py, p});
    ctx.beginPath(); ctx.arc(px, py, p.positive ? 3.6 : 2.8, 0, 7);
    ctx.fillStyle = p.positive ? POS : NEG;
    ctx.globalAlpha = p.positive ? 0.95 : 0.75;
    ctx.fill();
  }
  ctx.globalAlpha = 1;
  const npos = pts.filter(p => p.positive).length;
  document.getElementById('scatter-legend').innerHTML =
    `<span class="dot" style="background:${POS}"></span>${S.color}=1 (${npos}) &nbsp;` +
    `<span class="dot" style="background:${NEG}"></span>${S.color}=0 (${pts.length - npos})` +
    ` &nbsp;·&nbsp; showing ${pts.length}/${DATA.items.length} sampled items`;
  drawHist();
}

// Marginal distribution of the x feature for the two classes.
function drawHist() {
  const cv = document.getElementById('hist');
  if (!cv) return;
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w * dpr; cv.height = h * dpr;
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const PRE_LOGGED = new Set(['log_faces', 'faces_per_component']);
  const lg = S.logx && !PRE_LOGGED.has(S.x);
  const vals = [], cls = [];
  for (const it of S.items) {
    const v = it.feats[S.x];
    if (!Number.isFinite(v)) continue;
    vals.push(lg ? Math.log10(Math.max(v, 1e-9)) : v);
    cls.push((S.color === 'quality' ? it.quality === 1 : it.labels[LABELS.indexOf(S.color)] === 1) ? 1 : 0);
  }
  if (!vals.length) return;
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const bins = 44, span = (hi - lo) || 1;
  const counts = [new Float64Array(bins), new Float64Array(bins)];
  vals.forEach((v, i) => {
    const b = Math.min(bins - 1, Math.floor((v - lo) / span * bins));
    counts[cls[i]][b]++;
  });
  const n0 = counts[0].reduce((a, b) => a + b, 0) || 1, n1 = counts[1].reduce((a, b) => a + b, 0) || 1;
  const peak = Math.max(...counts[0].map(v => v / n0), ...counts[1].map(v => v / n1)) || 1;
  const m = {l: 10, r: 10, t: 8, b: 18}, pw = w - m.l - m.r, ph = h - m.t - m.b;
  const draw = (arr, n, colour) => {
    ctx.beginPath();
    for (let b = 0; b < bins; b++) {
      const x = m.l + b / bins * pw, y = m.t + ph - (arr[b] / n) / peak * ph;
      b ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    for (let b = bins - 1; b >= 0; b--) {
      const x = m.l + (b + 1) / bins * pw, y = m.t + ph - (arr[b] / n) / peak * ph;
      ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.globalAlpha = 0.45; ctx.fillStyle = colour; ctx.fill();
    ctx.globalAlpha = 1; ctx.strokeStyle = colour; ctx.lineWidth = 1.4; ctx.stroke();
  };
  draw(counts[1], n1, POS);
  draw(counts[0], n0, NEG);
  ctx.fillStyle = '#8b95a7'; ctx.font = '11px ui-monospace, monospace';
  ctx.textAlign = 'left'; ctx.fillText(fmtTick(lg ? Math.pow(10, lo) : lo), m.l, h - 4);
  ctx.textAlign = 'right'; ctx.fillText(fmtTick(lg ? Math.pow(10, hi) : hi), w - m.r, h - 4);
  ctx.textAlign = 'center';
  ctx.fillText(`${S.x} — density by ${S.color}`, w / 2, h - 4);
}

function fmtTick(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return '–';
  if (Math.abs(v) >= 1e5 || (Math.abs(v) < 1e-3 && v !== 0)) return v.toExponential(1);
  if (Math.abs(v) >= 1000) return v.toFixed(0);
  if (Math.abs(v) >= 1) return v.toFixed(Math.abs(v) < 10 ? 2 : 1);
  return v.toPrecision(2);
}

function hoverScatter(ev) {
  if (!S.screen) return;
  const r = document.getElementById('scatter').getBoundingClientRect();
  const mx = ev.clientX - r.left, my = ev.clientY - r.top;
  let best = null, bd = 14;
  for (const s of S.screen) {
    const d = Math.hypot(s.px - mx, s.py - my);
    if (d < bd) { bd = d; best = s; }
  }
  const tip = document.getElementById('tooltip');
  if (!best) { tip.style.display = 'none'; return; }
  const it = best.p.it;
  tip.style.display = 'block';
  tip.style.left = Math.min(ev.clientX + 14, window.innerWidth - 300) + 'px';
  tip.style.top = (ev.clientY + 14) + 'px';
  tip.innerHTML = `<div class="tip-id">${it.id.slice(0, 8)}</div>
    <img src="${DATA.thumbs[it.thumb] || ''}" class="${DATA.thumbs[it.thumb] ? '' : 'hidden'}"/>
    <div class="tips">${chipList(it)}</div>
    <div class="tip-feats">${FEATURES.filter(f => S.x === f || S.y === f).map(f =>
      `${f}: <b>${fmtTick(it.feats[f])}</b>`).join('<br>')}</div>`;
}

// --------------------------------------------------------------- auc table ---
function renderAuc() {
  const {labels, features, values} = DATA.auc;
  let html = '<table class="heat auc"><thead><tr><th></th>' +
    labels.map(l => `<th title="${l}">${l.slice(0, 5)}</th>`).join('') + '<th title="mean |AUC-0.5|">score</th></tr></thead><tbody>';
  features.forEach((f, i) => {
    const row = values[i];
    const score = row.reduce((a, v) => a + Math.abs(v - 0.5), 0) / row.length;
    html += `<tr><th class="rowhead" title="${f}">${f}</th>`;
    row.forEach((v, j) => {
      const d = Math.max(-1, Math.min(1, (v - 0.5) * 2));
      const col = d >= 0 ? `rgba(99,179,237,${(d * 0.85).toFixed(3)})` : `rgba(255,122,69,${(-d * 0.85).toFixed(3)})`;
      html += `<td style="background:${col}" title="${f} → ${labels[j]}: AUC ${v.toFixed(3)}">${v.toFixed(2)}</td>`;
    });
    html += `<td class="score">${score.toFixed(3)}</td></tr>`;
  });
  document.getElementById('auc').innerHTML = html + '</tbody></table>';
}

function renderBaseline() {
  const rows = DATA.baseline;
  const wf1 = rows.reduce((a, r) => a + r.f1 * r.pos, 0) / rows.reduce((a, r) => a + r.pos, 0);
  const q = rows.find(r => r.label === 'quality');
  const total = 10 * wf1 + 10 * (q ? q.f1 : 0);
  document.getElementById('baseline').innerHTML =
    '<table class="plain"><thead><tr><th>label</th><th>train support %</th><th>precision</th><th>recall</th><th>F1</th></tr></thead><tbody>' +
    rows.map(r => `<tr><td>${r.label}</td><td>${r.pos.toFixed(1)}</td><td>${r.prec.toFixed(3)}</td>
      <td>${r.rec.toFixed(3)}</td><td><b>${r.f1.toFixed(3)}</b></td></tr>`).join('') +
    '</tbody></table>' +
    `<p class="note">weighted artefact F1 = <b>${wf1.toFixed(3)}</b>, quality F1 = <b>${q ? q.f1.toFixed(3) : '—'}</b>
     → geometry-only metric ≈ <b>${total.toFixed(2)} / 20</b></p>` +
    `<p class="note">Labels geometry alone cannot separate (F1 &lt; 0.25 — the renders must carry them):
     <b>${rows.filter(r => r.f1 < 0.25 && r.label !== 'quality').map(r => r.label).join(', ') || 'none'}</b>.<br>
     Geometry carries the size/texture labels: ${rows.filter(r => r.f1 >= 0.5).sort((a, b) => b.f1 - a.f1)
       .map(r => `${r.label} (${r.f1.toFixed(2)})`).join(', ')}.</p>`;
}

// ----------------------------------------------------------------- gallery ---
const G = {filter: 'all', sort: 'n_faces_desc'};

function renderGallery() {
  let items = DATA.items.filter(it => it.mesh);
  if (G.filter === 'quality') items = items.filter(it => it.quality === 1);
  else if (G.filter !== 'all') items = items.filter(it => it.labels[LABELS.indexOf(G.filter)] === 1);
  const key = G.sort.replace(/_(asc|desc)$/, '');
  const dir = G.sort.endsWith('desc') ? -1 : 1;
  items.sort((a, b) => dir * ((a.feats[key] ?? -1) - (b.feats[key] ?? -1)));
  document.getElementById('gallery').innerHTML = items.map(it => `
    <div class="card" data-id="${it.id}">
      <img loading="lazy" src="${DATA.thumbs[it.thumb]}"/>
      <div class="card-body">
        <div class="tips">${chipList(it)}</div>
        <div class="tip-feats">faces ${it.feats.n_faces.toLocaleString()} · islands ${it.feats.n_components ?? '–'}
        · bnd ${fmtTick(it.feats.boundary_edge_frac)} · axis ${fmtTick(it.feats.axis_aligned_frac)}</div>
      </div>
    </div>`).join('');
  document.querySelectorAll('.card').forEach(c => c.onclick = () => openDetail(c.dataset.id));
}

function chipList(it) {
  const chips = LABELS.filter((_, i) => it.labels[i]).map(l => `<span class="chip d">${l}</span>`);
  if (it.quality === 1) chips.unshift('<span class="chip q">clean</span>');
  return chips.join('') || '<span class="chip none">no labels</span>';
}

// ------------------------------------------------------------ detail panel ---
function openDetail(id) {
  const it = BY_ID.get(id);
  const panel = document.getElementById('detail');
  panel.classList.add('open');
  const feats = FEATURES.map(f => `<tr><td>${f}</td><td>${fmtTick(it.feats[f])}</td></tr>`).join('');
  panel.innerHTML = `
    <div class="detail-head"><h3>${it.id}</h3><button onclick="document.getElementById('detail').classList.remove('open')">✕</button></div>
    <div class="detail-body">
      <div class="detail-col">
        <img class="views" src="${DATA.thumbs[it.thumb]}"/>
        <div class="tips">${chipList(it)}</div>
        ${it.open_path ? `<p class="note"><a href="${it.open_path}">open full-size render (file://)</a></p>` : ''}
      </div>
      <div class="detail-col">
        <canvas id="viewer" width="420" height="340"></canvas>
        <p class="note">drag to rotate · wheel to zoom · ${it.mesh ? 'decimated mesh, flat shading by face normal' : 'no mesh payload'}</p>
      </div>
      <div class="detail-col"><table class="plain kv">${feats}</table></div>
    </div>`;
  if (it.mesh && DATA.meshes[it.mesh]) initViewer(DATA.meshes[it.mesh]);
}

// -------------------------------------------------- tiny software 3D viewer ---
const V = {rot: {x: -0.35, y: 0.6}, zoom: 1, drag: null};

function initViewer(mesh) {
  const buf = b64ToBuf(mesh.pos);
  const pos = new Uint16Array(buf);
  const idx = mesh.dtype === 'u16' ? new Uint16Array(b64ToBuf(mesh.idx)) : new Uint32Array(b64ToBuf(mesh.idx));
  V.mesh = {pos, idx, nv: mesh.nv, nf: mesh.nf};
  const cv = document.getElementById('viewer');
  const dpr = window.devicePixelRatio || 1;
  cv.width = cv.clientWidth * dpr;
  cv.height = cv.clientHeight * dpr;
  drawViewer();
  cv.onmousedown = e => { V.drag = {x: e.clientX, y: e.clientY}; e.preventDefault(); };
  window.onmouseup = () => { V.drag = null; };
  cv.onmousemove = e => {
    if (!V.drag) return;
    V.rot.y += (e.clientX - V.drag.x) * 0.01;
    V.rot.x += (e.clientY - V.drag.y) * 0.01;
    V.drag = {x: e.clientX, y: e.clientY};
    drawViewer();
  };
  cv.onwheel = e => { V.zoom *= e.deltaY < 0 ? 1.08 : 0.93; drawViewer(); e.preventDefault(); };
}

function drawViewer() {
  const cv = document.getElementById('viewer');
  if (!cv || !V.mesh) return;
  const ctx = cv.getContext('2d'), w = cv.width, h = cv.height;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  const {pos, idx} = V.mesh;
  const nv = pos.length / 3;
  const P = new Float32Array(nv * 3);
  for (let i = 0; i < nv; i++) {
    P[3 * i] = pos[3 * i] / 65535 - 0.5;
    P[3 * i + 1] = pos[3 * i + 1] / 65535 - 0.5;
    P[3 * i + 2] = pos[3 * i + 2] / 65535 - 0.5;
  }
  const cy = Math.cos(V.rot.y), sy = Math.sin(V.rot.y);
  const cx = Math.cos(V.rot.x), sx = Math.sin(V.rot.x);
  const R = new Float32Array(nv * 3);
  for (let i = 0; i < nv; i++) {
    let x = P[3 * i], y = P[3 * i + 1], z = P[3 * i + 2];
    let x1 = cy * x + sy * z, z1 = -sy * x + cy * z;
    let y1 = cx * y - sx * z1, z2 = sx * y + cx * z1;
    R[3 * i] = x1; R[3 * i + 1] = y1; R[3 * i + 2] = z2;
  }
  const scale = Math.min(w, h) * 0.8 * V.zoom;
  const faces = [];
  for (let t = 0; t < idx.length; t += 3) {
    const a = idx[t], b = idx[t + 1], c = idx[t + 2];
    const ax = R[3 * a], ay = R[3 * a + 1], az = R[3 * a + 2];
    const bx = R[3 * b], by = R[3 * b + 1], bz = R[3 * b + 2];
    const cxx = R[3 * c], cyy = R[3 * c + 1], cz = R[3 * c + 2];
    const ux = bx - ax, uy = by - ay, uz = bz - az;
    const vx = cxx - ax, vy = cyy - ay, vz = cz - az;
    let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const nl = Math.hypot(nx, ny, nz) || 1;
    nx /= nl; ny /= nl; nz /= nl;
    const depth = (az + bz + cz) / 3;
    if (nz >= 0) continue; // backface cull (camera looks down -z)
    faces.push({a, b, c, depth, nx, ny, nz});
  }
  faces.sort((p, q) => p.depth - q.depth);
  ctx.fillStyle = '#12151c'; ctx.fillRect(0, 0, w, h);
  for (const f of faces) {
    // Match the provided renders: RGB encodes the normal direction.
    const col = `rgb(${Math.round(127.5 + 127.5 * f.nx)},${Math.round(127.5 + 127.5 * f.ny)},${Math.round(127.5 + 127.5 * f.nz)})`;
    ctx.beginPath();
    ctx.moveTo(w / 2 + R[3 * f.a] * scale, h / 2 - R[3 * f.a + 1] * scale);
    ctx.lineTo(w / 2 + R[3 * f.b] * scale, h / 2 - R[3 * f.b + 1] * scale);
    ctx.lineTo(w / 2 + R[3 * f.c] * scale, h / 2 - R[3 * f.c + 1] * scale);
    ctx.closePath();
    ctx.fillStyle = col; ctx.strokeStyle = col; ctx.lineWidth = 0.7; ctx.fill(); ctx.stroke();
  }
}

// ------------------------------------------------------------------ wiring ---
function boot() {
  applySubset();
  renderOverview(); renderAuc(); renderBaseline(); renderGallery(); drawScatter();
  const axisOpts = FEATURES.map(f => `<option value="${f}">${f}</option>`).join('');
  document.getElementById('x-axis').innerHTML = axisOpts;
  document.getElementById('y-axis').innerHTML = axisOpts;
  document.getElementById('x-axis').value = S.x;
  document.getElementById('y-axis').value = S.y;
  const colorOpts = ['quality'].concat(LABELS).map(l => `<option value="${l}">${l}</option>`).join('');
  document.getElementById('color-by').innerHTML = colorOpts;
  const filterOpts = ['all', 'quality'].concat(LABELS).map(l => `<option value="${l}">${l}</option>`).join('');
  document.getElementById('gallery-filter').innerHTML = filterOpts;
  document.getElementById('gallery-sort').innerHTML = ['n_faces', 'bbox_diag', 'n_components', 'boundary_edge_frac', 'axis_aligned_frac', 'normal_entropy', 'radial_frac']
    .flatMap(f => [`<option value="${f}_desc">${f} ↓</option>`, `<option value="${f}_asc">${f} ↑</option>`]).join('');

  const upd = () => {
    S.x = document.getElementById('x-axis').value;
    S.y = document.getElementById('y-axis').value;
    S.color = document.getElementById('color-by').value;
    S.logx = document.getElementById('logx').checked;
    S.logy = document.getElementById('logy').checked;
    S.subset = document.getElementById('subset').value;
    applySubset(); drawScatter();
  };
  ['x-axis', 'y-axis', 'color-by', 'logx', 'logy', 'subset'].forEach(id =>
    document.getElementById(id).onchange = upd);
  document.getElementById('scatter').onmousemove = hoverScatter;
  document.getElementById('scatter').onmouseleave = () => document.getElementById('tooltip').style.display = 'none';
  document.getElementById('scatter').onclick = ev => {
    if (!S.screen) return;
    const r = ev.target.getBoundingClientRect();
    const mx = ev.clientX - r.left, my = ev.clientY - r.top;
    let best = null, bd = 14;
    for (const s of S.screen) {
      const d = Math.hypot(s.px - mx, s.py - my);
      if (d < bd) { bd = d; best = s; }
    }
    if (best) openDetail(best.p.it.id);
  };
  const showPage = page => {
    document.querySelectorAll('.tab').forEach(o => o.classList.toggle('active', o.dataset.page === page));
    document.querySelectorAll('.page').forEach(p => p.classList.toggle('active', p.id === page));
    if (page === 'page-geometry') drawScatter();
  };
  document.querySelectorAll('.tab').forEach(t => t.onclick = () => {
    const page = t.dataset.page;
    showPage(page);
    if (location.hash !== '#' + page) history.replaceState(null, '', '#' + page);
  });
  // deep links: #page-geometry, #item=<id>
  const hash = decodeURIComponent(location.hash.slice(1));
  if (hash.startsWith('page-')) showPage(hash);
  else if (hash.startsWith('item=')) {
    const id = hash.slice(5);
    if (BY_ID.has(id)) openDetail(id);
  }
  document.getElementById('gallery-filter').onchange = e => { G.filter = e.target.value; renderGallery(); };
  document.getElementById('gallery-sort').onchange = e => { G.sort = e.target.value; renderGallery(); };
  window.addEventListener('resize', () => { drawScatter(); });
}
boot();
