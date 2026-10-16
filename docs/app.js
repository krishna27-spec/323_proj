/* Chatter Monitor: time signals, FFT, STFT waterfall, dominant frequencies and process state.
 *
 * Two data sources share one interface (fs, total, read(), labelAt()):
 *   DemoSource  plays the test recordings in data/ at the true rate, looping;
 *   LiveSource  polls scope.py (the DAQ server) and keeps the last BUF_S seconds.
 * Everything else (RMS, FFT, peaks, waterfall) is computed here from the raw samples,
 * so both sources look the same.
 */
'use strict';

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const BUF_S = 70;                       // live history kept in the page, s
const MAX_INPUTS = 4;                   // NI 9775: ai0..ai3
const STATE_NAMES = ['Air cutting', 'Stable', 'Transition', 'Chatter'];
const RULE_CODES = { 'no cutting': 0, 'stable': 1, 'chatter': 2, 'uncertain': -1 };
const MODEL_CODES = { 'no cutting': 0, 'stable': 1, 'chatter': 2 };
const DEMO_AXES = ['X · feed', 'Y · normal to feed', 'Z · axial'];
const STORE_KEY = 'chatter-monitor-inputs-v1';
const PARAMS = new URLSearchParams(location.search);   // ?run=60k_test&t=9 opens the demo at that moment
const SERVED_BY_SCOPE = location.protocol === 'http:' && location.pathname.startsWith('/app/');

const SEGS = {
  source: [['demo', 'Demo'], ['live', 'Live']],
  spanAll: [[0.1, '0.1 s'], [0.5, '0.5 s'], [1, '1 s'], [2, '2 s'], [5, '5 s']],
  spanOne: [[0.02, '20 ms'], [0.05, '50 ms'], [0.1, '100 ms'], [0.2, '200 ms'], [0.5, '0.5 s']],
  win: [[256, '256'], [512, '512'], [1024, '1024'], [2048, '2048'], [4096, '4096']],
  fftScale: [['db', 'dB'], ['lin', 'Linear']],
  hist: [[5, '5 s'], [10, '10 s'], [30, '30 s'], [60, '60 s']],
  overlap: [[0, '0 %'], [50, '50 %'], [75, '75 %']],
  dyn: [[40, '40 dB'], [60, '60 dB'], [80, '80 dB']],
  units: [['V', 'Volts'], ['g', 'g']],
};

const DEFAULTS = {
  source: SERVED_BY_SCOPE ? 'live' : 'demo',
  demoRun: '20k_test',
  liveUrl: SERVED_BY_SCOPE ? location.origin : 'http://localhost:8050',
  channels: [0, 1, 2, 3].map((p) => ({ phys: p, name: 'Channel ' + (p + 1), sens: 100 })),
  sel: 1,
  win: 1024, wfn: 'hann', overlap: 50, fmax: 0, hist: 10, dyn: 60,
  units: 'V', spanAll: 1, spanOne: 0.1, yScale: 'auto', dc: true, fftScale: 'db',
};

// ------------------------------------------------------------------ settings

let cfg = loadCfg();

function loadCfg() {
  let c = {};
  try { c = JSON.parse(localStorage.getItem(STORE_KEY) || '{}') || {}; } catch (e) { c = {}; }
  const out = { ...structuredClone(DEFAULTS), ...c };
  if (!Array.isArray(out.channels) || !out.channels.length) out.channels = structuredClone(DEFAULTS.channels);
  out.channels = out.channels
    .filter((ch) => ch && Number.isInteger(ch.phys) && ch.phys >= 0 && ch.phys < MAX_INPUTS)
    .map((ch) => ({ phys: ch.phys, name: String(ch.name || 'Channel ' + (ch.phys + 1)).slice(0, 40), sens: +ch.sens > 0 ? +ch.sens : 100 }));
  if (!out.channels.length) out.channels = structuredClone(DEFAULTS.channels);
  for (const k of Object.keys(SEGS)) if (!SEGS[k].some(([v]) => v === out[k])) out[k] = DEFAULTS[k];
  if (!out.channels.some((ch) => ch.phys === out.sel)) out.sel = out.channels[0].phys;
  if (SERVED_BY_SCOPE && !c.liveUrl) out.liveUrl = location.origin;
  if (PARAMS.get('run')) { out.source = 'demo'; out.demoRun = PARAMS.get('run'); }
  return out;
}
function saveCfg() {
  try { localStorage.setItem(STORE_KEY, JSON.stringify(cfg)); } catch (e) { /* private mode: settings last for this visit */ }
}
function setCfg(key, val) {
  cfg[key] = val;
  saveCfg();
  if (key === 'source' || key === 'demoRun' || key === 'liveUrl') switchSource();
  if (key === 'units') view.ranges = {};
  if (key === 'sel') renderChannels();
  syncControls();
}

// ------------------------------------------------------------------ sources

class DemoSource {
  constructor(run) {
    this.kind = 'demo'; this.run = run; this.ready = false; this.fs = 20000;
    this.pos = 0; this.last = null; this.playing = true; this.epoch = 0;
  }
  async load() {
    const [meta, buf] = await Promise.all([
      fetch(`data/${this.run}.json`).then((r) => { if (!r.ok) throw new Error(`data/${this.run}.json: HTTP ${r.status}`); return r.json(); }),
      fetch(`data/${this.run}.i16`).then((r) => { if (!r.ok) throw new Error(`data/${this.run}.i16: HTTP ${r.status}`); return r.arrayBuffer(); }),
    ]);
    const i16 = new Int16Array(buf), n = meta.n;
    this.x = meta.scale.map((s, c) => {
      const a = new Float32Array(n);
      for (let i = 0; i < n; i++) a[i] = i16[c * n + i] * s;
      return a;
    });
    this.n = n; this.fs = meta.fs; this.dur = n / meta.fs; this.rpm = meta.rpm; this.lab = meta.labels;
    const t = parseFloat(PARAMS.get('t'));
    this.pos = (isFinite(t) && t >= 0 ? t : 2) * this.fs;   // by default cutting starts about 2 s after the page opens
    this.ready = true;
  }
  tick(ms) {
    if (this.last != null && this.playing) this.pos += Math.min(ms - this.last, 250) / 1000 * this.fs;
    this.last = ms;
  }
  get total() { return Math.floor(this.pos); }
  get inputs() { return MAX_INPUTS; }
  has(p) { return this.ready && p < this.x.length; }
  describe(p) { return p < DEMO_AXES.length ? DEMO_AXES[p] : 'not in the recording'; }
  read(p, a, b) {
    if (!this.has(p)) return null;
    const out = new Float32Array(b - a), x = this.x[p], n = this.n;
    let k = ((a % n) + n) % n;
    for (let i = 0; i < out.length; i++) { out[i] = x[k]; if (++k === n) k = 0; }
    return out;
  }
  labelAt(sec) {
    if (!this.ready) return null;
    const s = ((sec % this.dur) + this.dur) % this.dur, t = this.lab.t;
    const k = lastAtOrBefore(t.length, (i) => t[i], s);
    return k < 0 ? null : { rule: this.lab.rule[k], model: this.lab.model[k] };
  }
  stop() {}
}

class LiveSource {
  constructor(url) {
    this.kind = 'live'; this.url = String(url || '').replace(/\/+$/, '');
    this.fs = 20000; this.nPhys = 0; this.total = 0; this.first = 0; this.ring = null; this.N = 0;
    this.labels = []; this.labSince = 0; this.info = null; this.err = ''; this.alive = true; this.epoch = 0;
    this.loop();
  }
  get ready() { return !!this.ring && this.total > 0; }
  get inputs() { return this.nPhys || MAX_INPUTS; }
  async loop() {
    while (this.alive) {
      await this.poll();
      await new Promise((r) => setTimeout(r, this.err ? 1500 : 80));
    }
  }
  async poll() {
    try {
      const since = this.ring ? this.total : -1;
      const r = await fetch(`${this.url}/poll?since=${since}&spec=0&lab=${this.labSince}`, { cache: 'no-store' });
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const d = await r.json();
      if (!this.alive) return;
      this.info = d; this.err = '';
      if (!this.ring || d.fs !== this.fs || d.data.length !== this.nPhys || d.total < this.total) {
        this.fs = d.fs; this.nPhys = d.data.length; this.N = Math.round(BUF_S * d.fs);
        this.ring = Array.from({ length: this.nPhys }, () => new Float32Array(this.N));
        this.labels = []; this.labSince = 0; this.first = d.start; this.total = d.start; this.epoch++;
      }
      if (d.start > this.total) this.first = d.start;        // the page slept: drop the gap
      d.data.forEach((s, c) => {
        const v = b64f32(s), ring = this.ring[c], N = this.N;
        for (let k = 0; k < v.length; k++) ring[(d.start + k) % N] = v[k];
      });
      this.total = d.total;
      for (const l of d.labels) {
        this.labels.push({
          t: l.t,
          rule: l.rule in RULE_CODES ? RULE_CODES[l.rule] : null,
          model: l.models && l.models['CNN + GRU'] in MODEL_CODES ? MODEL_CODES[l.models['CNN + GRU']] : null,
        });
        this.labSince = l.id + 1;
      }
      if (this.labels.length > 6000) this.labels.splice(0, 2000);
    } catch (e) {
      this.err = e && e.message ? e.message : String(e);
    }
  }
  has(p) { return !!this.ring && p < this.nPhys; }
  describe(p) { return this.ring && p >= this.nPhys ? 'not acquired by scope.py' : ''; }
  read(p, a, b) {
    if (!this.has(p)) return null;
    const out = new Float32Array(b - a), ring = this.ring[p], N = this.N;
    const lo = Math.max(this.first, this.total - N + 1);
    for (let i = 0; i < out.length; i++) { const k = a + i; out[i] = k >= lo && k < this.total ? ring[k % N] : 0; }
    return out;
  }
  labelAt(sec) {
    const L = this.labels;
    const k = lastAtOrBefore(L.length, (i) => L[i].t, sec);
    return k < 0 || sec - L[k].t > 1 ? null : L[k];
  }
  async post(path) {
    const r = await fetch(this.url + path, { method: 'POST' });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.error || 'HTTP ' + r.status);
    return j;
  }
  stop() { this.alive = false; }
}

function lastAtOrBefore(n, at, x) {
  let lo = 0, hi = n - 1, k = -1;
  while (lo <= hi) { const m = (lo + hi) >> 1; if (at(m) <= x) { k = m; lo = m + 1; } else hi = m - 1; }
  return k;
}
function b64f32(s) {
  const bin = atob(s), u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(u8.buffer);
}

/* 0 air cutting, 1 stable, 2 transition, 3 chatter.
 * The CNN + GRU model decides; the entropy rule is the cross-check. While the tool cuts,
 * an "uncertain" rule or a rule that says the other one of stable / chatter makes it a transition. */
function processState(l) {
  if (!l) return null;
  const { rule, model } = l;
  if (model == null) return rule == null ? null : rule === -1 ? 2 : rule === 2 ? 3 : rule;
  if (model === 0) return 0;
  if (rule === -1 || (rule != null && rule !== 0 && rule !== model)) return 2;
  return model === 1 ? 1 : 3;
}

// ------------------------------------------------------------------ signal maths

const windows = new Map();
function getWindow(N, kind) {
  const key = N + kind;
  if (windows.has(key)) return windows.get(key);
  const w = new Float64Array(N), T = 2 * Math.PI / (N - 1);
  for (let i = 0; i < N; i++) {
    const x = i * T;
    w[i] = kind === 'hann' ? 0.5 - 0.5 * Math.cos(x)
      : kind === 'hamming' ? 0.54 - 0.46 * Math.cos(x)
      : kind === 'flattop' ? 0.21557895 - 0.41663158 * Math.cos(x) + 0.277263158 * Math.cos(2 * x) - 0.083578947 * Math.cos(3 * x) + 0.006947368 * Math.cos(4 * x)
      : 1;
  }
  const sum = w.reduce((a, b) => a + b, 0);
  const out = { w, sum };
  windows.set(key, out);
  return out;
}
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang), half = len >> 1;
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < half; k++) {
        const a = i + k, b = a + half;
        const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
        const nc = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = nc;
      }
    }
  }
}
/** Amplitude spectrum (peak amplitude of a sine, in the input unit), bins 0..N/2. */
function amplitude(x, kind) {
  const N = x.length, { w, sum } = getWindow(N, kind);
  let mean = 0;
  for (let i = 0; i < N; i++) mean += x[i];
  mean /= N;
  const re = new Float64Array(N), im = new Float64Array(N);
  for (let i = 0; i < N; i++) re[i] = (x[i] - mean) * w[i];
  fft(re, im);
  const mag = new Float32Array(N / 2 + 1);
  for (let k = 0; k <= N / 2; k++) mag[k] = (k && k < N / 2 ? 2 : 1) * Math.hypot(re[k], im[k]) / sum;
  return mag;
}
function topPeaks(mag, df, kTop, count = 10) {
  const ks = [];
  for (let k = Math.max(1, Math.ceil(10 / df)); k < Math.min(kTop, mag.length - 1); k++) {
    if (mag[k] > mag[k - 1] && mag[k] >= mag[k + 1]) ks.push(k);
  }
  ks.sort((a, b) => mag[b] - mag[a]);
  return ks.slice(0, count).map((k) => {
    const a = Math.log(mag[k - 1] + 1e-30), b = Math.log(mag[k] + 1e-30), c = Math.log(mag[k + 1] + 1e-30);
    const den = a - 2 * b + c, p = den ? Math.max(-0.5, Math.min(0.5, 0.5 * (a - c) / den)) : 0;
    return { f: (k + p) * df, a: mag[k], k };
  });
}
function rmsOf(v) { let s = 0; for (let i = 0; i < v.length; i++) s += v[i] * v[i]; return Math.sqrt(s / v.length); }
function peakOf(v) { let m = 0; for (let i = 0; i < v.length; i++) { const a = Math.abs(v[i]); if (a > m) m = a; } return m; }
function niceUp(v) {
  if (!(v > 0)) return 0.001;
  const e = Math.pow(10, Math.floor(Math.log10(v))), m = v / e;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * e;
}
function fmt(v) {
  if (v == null || !isFinite(v)) return '–';
  const a = Math.abs(v);
  return a >= 100 ? v.toFixed(0) : a === 0 ? '0' : v.toPrecision(3);
}
function fmtHz(f) { return f >= 1000 ? Math.round(f).toLocaleString('en-US') : f >= 100 ? f.toFixed(0) : f.toFixed(1); }
function fmtK(f) { return f >= 1000 ? +(f / 1000).toFixed(f % 1000 ? 1 : 0) + 'k' : String(Math.round(f)); }
function fmtFs(f) { return f >= 1000 ? +(f / 1000).toFixed(1) + ' kHz' : Math.round(f) + ' Hz'; }
function fmtSpan(s) { return s >= 1 ? s + ' s' : Math.round(s * 1000) + ' ms'; }

// ------------------------------------------------------------------ colours and canvas

let C = {};
let FONT = 'system-ui, sans-serif';
let LUT = new Uint8ClampedArray(256 * 3);
function readColors() {
  const cs = getComputedStyle(document.documentElement);
  const get = (n) => cs.getPropertyValue(n).trim();
  C = { ink: get('--ink'), ink2: get('--ink2'), grid: get('--grid'), accent: get('--accent'), fill2: get('--fill2'),
        s: [0, 1, 2, 3].map((i) => get(`--s${i}-bar`)), ch: [0, 1, 2, 3].map((i) => get(`--ch${i}`)),
        wf: [0, 1, 2, 3].map((i) => hexRgb(get(`--wf${i}`))) };
  FONT = getComputedStyle(document.body).fontFamily;
  const stops = [0, 0.45, 0.78, 1];
  for (let i = 0; i < 256; i++) {
    const x = i / 255;
    let j = 0;
    while (j < 2 && x > stops[j + 1]) j++;
    const t = (x - stops[j]) / (stops[j + 1] - stops[j]);
    for (let c = 0; c < 3; c++) LUT[i * 3 + c] = C.wf[j][c] + (C.wf[j + 1][c] - C.wf[j][c]) * t;
  }
}
function hexRgb(h) {
  const m = /^#?([0-9a-f]{6})$/i.exec(h || '');
  const n = m ? parseInt(m[1], 16) : 0x888888;
  return [n >> 16, (n >> 8) & 255, n & 255];
}
function chColor(phys) { return C.ch[phys % C.ch.length]; }
function fit(cv) {
  const r = cv.getBoundingClientRect(), d = window.devicePixelRatio || 1;
  const W = Math.max(1, Math.round(r.width * d)), H = Math.max(1, Math.round(r.height * d));
  if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
  const g = cv.getContext('2d');
  g.setTransform(d, 0, 0, d, 0, 0);
  g.clearRect(0, 0, r.width, r.height);
  return { g, w: r.width, h: r.height };
}
function drawWave(cv, v, range, color, opts = {}) {
  const { g, w, h } = fit(cv);
  const mid = h / 2;
  g.strokeStyle = C.grid; g.lineWidth = 1; g.beginPath();
  g.moveTo(0, Math.round(mid) + 0.5); g.lineTo(w, Math.round(mid) + 0.5);
  if (opts.grid) for (const f of [0.25, 0.75]) { const y = Math.round(h * f) + 0.5; g.moveTo(0, y); g.lineTo(w, y); }
  g.stroke();
  g.font = '11px ' + FONT; g.fillStyle = C.ink2;
  if (!v) { g.textAlign = 'center'; g.fillText(opts.empty || 'No signal', w / 2, mid - 8); return; }
  const n = v.length, sy = (h / 2 - 2) / range, cols = Math.max(1, Math.floor(w));
  const Y = (s) => Math.max(1, Math.min(h - 1, mid - s * sy));
  g.strokeStyle = color; g.lineWidth = 1.25; g.lineJoin = 'round'; g.beginPath();
  if (n <= cols * 2) {
    for (let i = 0; i < n; i++) { const x = (i / (n - 1)) * w; i ? g.lineTo(x, Y(v[i])) : g.moveTo(x, Y(v[i])); }
  } else {
    for (let c = 0; c < cols; c++) {
      const a = Math.floor((c * n) / cols), b = Math.floor(((c + 1) * n) / cols);
      let lo = Infinity, hi = -Infinity;
      for (let i = a; i < b; i++) { const s = v[i]; if (s < lo) lo = s; if (s > hi) hi = s; }
      const x = c + 0.5;
      c ? g.lineTo(x, Y(hi)) : g.moveTo(x, Y(hi));
      g.lineTo(x, Y(lo) + 0.5);
    }
  }
  g.stroke();
  if (opts.label) { g.textAlign = 'left'; g.fillStyle = C.ink2; g.fillText(opts.label, 4, 12); }
}

// ------------------------------------------------------------------ state of the page

let src = null;
let paused = false, frozen = 0;
const view = { ranges: {}, fftTop: null, linTop: null, lastState: undefined, stateSince: 0, lastText: 0, sp: null };
const wf = { key: '', off: null, img: null, rows: 0, cols: 0, hop: 0, head: 0, lastEnd: 0, top: -Infinity, dirty: false };

function now() { return paused ? frozen : src ? src.total : 0; }
function unitName() { return cfg.units === 'g' ? 'g' : 'V'; }
function chByPhys(p) { return cfg.channels.find((c) => c.phys === p); }
function selCh() { return chByPhys(cfg.sel) || cfg.channels[0]; }
function readCh(ch, a, b) {
  const v = src && src.ready ? src.read(ch.phys, a, b) : null;
  if (!v) return null;
  const k = cfg.units === 'g' ? 1000 / ch.sens : 1;
  let mean = 0;
  if (cfg.dc) { for (let i = 0; i < v.length; i++) mean += v[i]; mean /= v.length || 1; }
  for (let i = 0; i < v.length; i++) v[i] = (v[i] - mean) * k;
  return v;
}
function rangeFor(id, v) {
  if (cfg.yScale !== 'auto') return +cfg.yScale;
  const need = v ? peakOf(v) * 1.1 : 0;
  const r = Math.max(need, (view.ranges[id] || 0) * 0.985);
  view.ranges[id] = r;
  return niceUp(r || 0.001);
}
function describe(p) {
  const d = src ? src.describe(p) : '';
  return 'ai' + p + (d ? ' · ' + d : '');
}

// ------------------------------------------------------------------ controls

function buildSegs() {
  for (const el of $$('.seg[data-key]')) {
    const key = el.dataset.key;
    el.setAttribute('role', 'group');
    el.innerHTML = '';
    for (const [v, label] of SEGS[key]) {
      const b = document.createElement('button');
      b.type = 'button'; b.textContent = label; b.dataset.v = JSON.stringify(v);
      b.addEventListener('click', () => setCfg(key, v));
      el.appendChild(b);
    }
  }
  for (const el of $$('select[data-key], input[data-key]')) {
    const key = el.dataset.key;
    el.addEventListener('change', () => {
      let v = el.type === 'checkbox' ? el.checked : el.value;
      if (key === 'sel' || key === 'fmax') v = +v;
      if (key === 'yScale' && v !== 'auto') v = +v;
      setCfg(key, v);
    });
  }
}
function syncControls() {
  for (const el of $$('.seg[data-key]')) {
    const cur = JSON.stringify(cfg[el.dataset.key]);
    for (const b of el.children) b.setAttribute('aria-pressed', String(b.dataset.v === cur));
  }
  for (const el of $$('select[data-key], input[data-key]')) {
    const v = cfg[el.dataset.key];
    if (el.type === 'checkbox') el.checked = !!v;
    else if (document.activeElement !== el) el.value = String(v);
  }
  const live = cfg.source === 'live';
  $('#rowDemo').hidden = live;
  $('#rowLive').hidden = !live;
  $('#recBtn').hidden = !live;
  const fsPick = $('#fsPick');
  fsPick.disabled = !live || !src || !src.ready;
  if (src) {
    if (![...fsPick.options].some((o) => +o.value === src.fs)) fsPick.add(new Option(fmtFs(src.fs), src.fs));
    fsPick.value = String(src.fs);
  }
  $('#fsSub').textContent = live ? 'Restarts the acquisition. The process state needs 20 kHz.' : 'Fixed by the recording';
  const fs = src ? src.fs : 20000, N = cfg.win;
  $('#winSub').textContent = `${N} samples = ${(N / fs * 1000).toFixed(1)} ms · Δf ${(fs / N).toFixed(2)} Hz`;
  $('#nChVal').textContent = cfg.channels.length;
  const inputs = src ? src.inputs : MAX_INPUTS;
  $('#chMinus').disabled = cfg.channels.length <= 1;
  $('#chPlus').disabled = $('#addCh').disabled = cfg.channels.length >= inputs;
  $('#nChSub').textContent = `${inputs} inputs available (ai0 to ai${inputs - 1})`;
  $$('.unitLbl').forEach((e) => { e.textContent = unitName(); });
  const pb = $('#pauseBtn');
  pb.querySelector('span').textContent = paused ? 'Resume' : 'Pause';
  pb.querySelector('svg').innerHTML = paused ? '<path d="M7 5l12 7-12 7z"/>' : '<path d="M9 5v14M15 5v14"/>';
  pb.setAttribute('aria-pressed', String(paused));
}

function addChannel() {
  const inputs = src ? src.inputs : MAX_INPUTS;
  const used = new Set(cfg.channels.map((c) => c.phys));
  let p = 0;
  while (used.has(p) && p < inputs) p++;
  if (p >= inputs) return;
  cfg.channels.push({ phys: p, name: 'Channel ' + (p + 1), sens: 100 });
  cfg.channels.sort((a, b) => a.phys - b.phys);
  saveCfg(); renderChannels(); syncControls();
}
function removeChannel(phys) {
  if (cfg.channels.length <= 1) return;
  cfg.channels = cfg.channels.filter((c) => c.phys !== phys);
  if (cfg.sel === phys) cfg.sel = cfg.channels[0].phys;
  saveCfg(); renderChannels(); syncControls();
}

const X_ICON = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg>';

function renderChannels() {
  // the stacked list
  const list = $('#chList');
  list.innerHTML = '';
  for (const ch of cfg.channels) {
    const li = document.createElement('li');
    li.className = 'ch-row' + (ch.phys === cfg.sel ? ' sel' : '');
    li.innerHTML = `
      <button class="ch-pick" type="button" aria-pressed="${ch.phys === cfg.sel}" title="Show in the real-time view">
        <span class="dot" style="background:${chColor(ch.phys)}"></span>
        <span style="min-width:0"><span class="nm"></span><span class="ds"></span><span class="rms" data-rms>–</span></span>
      </button>
      <canvas role="img"></canvas>
      <button class="icon-btn" type="button">${X_ICON}</button>`;
    li.querySelector('.nm').textContent = ch.name;
    li.querySelector('.ds').textContent = describe(ch.phys);
    li.querySelector('canvas').setAttribute('aria-label', 'Waveform of ' + ch.name);
    const rm = li.querySelector('.icon-btn');
    rm.setAttribute('aria-label', 'Remove ' + ch.name);
    rm.disabled = cfg.channels.length <= 1;
    rm.addEventListener('click', () => removeChannel(ch.phys));
    li.querySelector('.ch-pick').addEventListener('click', () => setCfg('sel', ch.phys));
    li.dataset.phys = ch.phys;
    list.appendChild(li);
  }
  // channel pickers
  for (const sel of [$('#selPick'), $('#selPick2')]) {
    sel.innerHTML = '';
    for (const ch of cfg.channels) sel.add(new Option(ch.name, ch.phys));
    sel.value = String(cfg.sel);
  }
  // the editor in the inputs sheet
  const ed = $('#chEdit');
  ed.innerHTML = '';
  const inputs = src ? src.inputs : MAX_INPUTS;
  cfg.channels.forEach((ch, i) => {
    const row = document.createElement('div');
    row.className = 'ch-edit';
    const physOpts = [];
    for (let p = 0; p < Math.max(inputs, ch.phys + 1); p++) {
      if (p === ch.phys || !cfg.channels.some((c) => c.phys === p)) physOpts.push(`<option value="${p}">ai${p}</option>`);
    }
    row.innerHTML = `
      <span class="dot" style="background:${chColor(ch.phys)}"></span>
      <input class="field name" type="text" maxlength="40" aria-label="Name of channel ${i + 1}">
      <span class="select"><select aria-label="Input of channel ${i + 1}">${physOpts.join('')}</select></span>
      <button class="icon-btn" type="button" aria-label="Remove channel ${i + 1}">${X_ICON}</button>
      <div class="meta">
        <label for="sens${i}">Scaling factor</label>
        <input class="field" id="sens${i}" type="number" min="0.001" step="any">
        <span>mV/g</span>
      </div>`;
    const name = row.querySelector('input.name'), phys = row.querySelector('select'), sens = row.querySelector('input[type=number]');
    name.value = ch.name; phys.value = String(ch.phys); sens.value = String(ch.sens);
    name.addEventListener('input', () => {
      ch.name = name.value || 'Channel ' + (ch.phys + 1); saveCfg();
      const r = $(`#chList li[data-phys="${ch.phys}"] .nm`); if (r) r.textContent = ch.name;
      for (const s of [$('#selPick'), $('#selPick2')]) { const o = [...s.options].find((o) => +o.value === ch.phys); if (o) o.text = ch.name; }
    });
    phys.addEventListener('change', () => {
      const was = ch.phys; ch.phys = +phys.value;
      if (cfg.sel === was) cfg.sel = ch.phys;
      cfg.channels.sort((a, b) => a.phys - b.phys);
      saveCfg(); renderChannels(); syncControls();
    });
    sens.addEventListener('change', () => { const v = +sens.value; if (v > 0) { ch.sens = v; saveCfg(); view.ranges = {}; } else sens.value = String(ch.sens); });
    const rm = row.querySelector('.icon-btn');
    rm.disabled = cfg.channels.length <= 1;
    rm.addEventListener('click', () => removeChannel(ch.phys));
    ed.appendChild(row);
  });
  const add = document.createElement('div');
  add.className = 'g-row';
  add.innerHTML = '<span class="g-sub">Accelerometers on unused inputs</span><button class="btn" type="button">Add accelerometer</button>';
  const addBtn = add.querySelector('button');
  addBtn.disabled = cfg.channels.length >= inputs;
  addBtn.addEventListener('click', addChannel);
  ed.appendChild(add);
}

// ------------------------------------------------------------------ sources: switch, status

async function loadDemoIndex() {
  try {
    const j = await fetch('data/index.json').then((r) => r.json());
    const pick = $('#demoPick');
    pick.innerHTML = '';
    for (const r of j.runs) pick.add(new Option(`${r.title} (${r.seconds} s)`, r.id));
    if (!j.runs.some((r) => r.id === cfg.demoRun)) cfg.demoRun = j.runs[0].id;
    pick.value = cfg.demoRun;
  } catch (e) { /* shown by the demo source itself */ }
}
async function switchSource() {
  if (src) src.stop();
  paused = false;
  view.ranges = {}; view.fftTop = view.linTop = null; view.lastState = undefined; wf.key = '';
  if (cfg.source === 'demo') {
    const s = new DemoSource(cfg.demoRun);
    src = s;
    try { await s.load(); } catch (e) { if (src === s) showBanner('The demo recording could not be loaded: ' + e.message); }
  } else {
    src = new LiveSource(cfg.liveUrl);
  }
  renderChannels();
  syncControls();
}
function showBanner(msg, info) {
  const b = $('#banner');
  b.hidden = !msg; b.textContent = msg || ''; b.classList.toggle('info', !!info);
}
function updateStatus() {
  const st = $('#status'), text = st.querySelector('span');
  let cls = 'wait', t = 'Loading…', banner = '', info = false;
  if (src && src.kind === 'demo') {
    if (src.ready) { cls = paused ? 'wait' : 'ok'; t = paused ? 'Paused' : `Demo · ${(src.rpm || 0).toLocaleString('en-US')} rpm · ${fmtFs(src.fs)}`; }
  } else if (src) {
    const d = src.info;
    if (src.err) {
      cls = 'bad'; t = 'Offline';
      banner = `No answer from ${src.url}. Start scope.py on the measuring PC (.venv/bin/python scope.py --rpm 20000 --models), `
        + `then allow this page to access your local network if the browser asks. (${src.err})`;
    } else if (!d) { t = 'Connecting…'; }
    else if (d.status === 'running') { cls = paused ? 'wait' : 'ok'; t = paused ? 'Paused' : `Live · ${fmtFs(d.fs)}`; }
    else if (d.status === 'stopped') { t = 'Acquisition stopped'; }
    else if (d.status === 'error') { cls = 'bad'; t = 'DAQ error'; banner = d.error; }
    else { t = d.status; }
    if (!banner && d && d.notice) { banner = d.notice; info = true; }
    if (!banner && d && d.status === 'running' && !d.models && !d.rule) {
      banner = 'No process state: scope.py runs without the models. Start it with --models (and --rpm for the rule).'; info = true;
    }
    const rec = $('#recBtn');
    const on = !!(d && d.record && d.record.on);
    rec.classList.toggle('on', on);
    rec.setAttribute('aria-pressed', String(on));
    rec.querySelector('span').textContent = on ? `Recording ${d.record.seconds.toFixed(0)} s` : 'Record';
    rec.disabled = !d || d.status !== 'running';
  }
  st.className = 'status ' + cls;
  text.textContent = t;
  if (src && src.kind === 'live') showBanner(banner, info);
  else if (src && src.ready) showBanner('');
}

// ------------------------------------------------------------------ drawing

function drawChannels(n) {
  const span = Math.round(cfg.spanAll * src.fs);
  for (const li of $$('#chList li')) {
    const ch = chByPhys(+li.dataset.phys);
    if (!ch) continue;
    const v = readCh(ch, n - span, n);
    drawWave(li.querySelector('canvas'), v, rangeFor('all' + ch.phys, v), chColor(ch.phys),
      { label: v ? '±' + fmt(rangeFor('all' + ch.phys, null)) + ' ' + unitName() : '', empty: 'No signal on ai' + ch.phys });
  }
  $('#spanAllLbl').textContent = '−' + fmtSpan(cfg.spanAll);
}
function drawSingle(n) {
  const ch = selCh();
  const span = Math.round(cfg.spanOne * src.fs);
  const v = readCh(ch, n - span, n);
  const r = rangeFor('one' + ch.phys, v);
  drawWave($('#cvSingle'), v, r, chColor(ch.phys), { grid: true, label: v ? '±' + fmt(r) + ' ' + unitName() : '', empty: 'No signal on ai' + ch.phys });
  $('#spanOneLbl').textContent = '−' + fmtSpan(cfg.spanOne);
  return v;
}
function spectrum(n) {
  const ch = selCh(), N = cfg.win, fs = src.fs, nyq = fs / 2, df = fs / N;
  const fTop = cfg.fmax && cfg.fmax < nyq ? cfg.fmax : nyq;
  const x = readCh(ch, n - N, n);
  if (!x) return { ch, N, df, fTop, nyq, mag: null, peaks: [], rms: NaN };
  const mag = amplitude(x, cfg.wfn);
  return { ch, N, df, fTop, nyq, mag, peaks: topPeaks(mag, df, Math.floor(fTop / df) + 1), rms: rmsOf(x), x };
}
function drawFFT(sp) {
  const { g, w, h } = fit($('#cvFFT'));
  const kTop = Math.min(sp.mag ? sp.mag.length - 1 : 0, Math.round(sp.fTop / sp.df));
  const X = (f) => (f / sp.fTop) * w;
  // grid
  g.strokeStyle = C.grid; g.lineWidth = 1; g.beginPath();
  for (let i = 1; i < 4; i++) { const y = Math.round((h * i) / 4) + 0.5; g.moveTo(0, y); g.lineTo(w, y); }
  for (let i = 1; i < 4; i++) { const x = Math.round((w * i) / 4) + 0.5; g.moveTo(x, 0); g.lineTo(x, h); }
  g.stroke();
  g.font = '11px ' + FONT; g.fillStyle = C.ink2;
  if (!sp.mag) { g.textAlign = 'center'; g.fillText('No signal on ai' + sp.ch.phys, w / 2, h / 2); return; }
  const db = cfg.fftScale === 'db';
  let maxV = 0;
  for (let k = 1; k <= kTop; k++) if (sp.mag[k] > maxV) maxV = sp.mag[k];
  let Y, labels;
  if (db) {
    const want = Math.ceil((20 * Math.log10(maxV + 1e-12) + 6) / 10) * 10;
    view.fftTop = view.fftTop == null ? want : Math.max(want, view.fftTop - 0.2);
    const top = Math.ceil(view.fftTop / 10) * 10, span = cfg.dyn + 20;
    Y = (m) => Math.min(h, Math.max(0, ((top - 20 * Math.log10(m + 1e-12)) / span) * h));
    labels = [0, 1, 2, 3].map((i) => (top - (span * i) / 4).toFixed(0) + ' dB');
  } else {
    view.linTop = view.linTop == null ? maxV * 1.1 : Math.max(maxV * 1.1, view.linTop * 0.98);
    const top = niceUp(view.linTop);
    Y = (m) => Math.max(0, h - (m / top) * (h - 2));
    labels = [0, 1, 2, 3].map((i) => fmt(top * (1 - i / 4)) + ' ' + unitName());
  }
  // area + line
  const cols = Math.max(1, Math.floor(w)), path = new Path2D();
  for (let c = 0; c <= cols; c++) {
    const k0 = Math.floor((c / cols) * kTop), k1 = Math.max(k0 + 1, Math.floor(((c + 1) / cols) * kTop));
    let m = 0;
    for (let k = k0; k < Math.min(k1, kTop + 1); k++) if (sp.mag[k] > m) m = sp.mag[k];
    c ? path.lineTo(c, Y(m)) : path.moveTo(0, Y(m));
  }
  const area = new Path2D(path);
  area.lineTo(w, h); area.lineTo(0, h); area.closePath();
  g.fillStyle = C.accent; g.globalAlpha = 0.12; g.fill(area); g.globalAlpha = 1;
  g.strokeStyle = C.accent; g.lineWidth = 1.25; g.stroke(path);
  // peaks
  sp.peaks.forEach((p, i) => {
    const x = X(p.f), y = Y(p.a);
    g.beginPath(); g.arc(x, y, i ? 2.5 : 4, 0, 2 * Math.PI);
    g.fillStyle = i ? C.ink2 : C.ink; g.fill();
  });
  if (sp.peaks.length) {
    const p = sp.peaks[0], x = X(p.f), y = Y(p.a), t = fmtHz(p.f) + ' Hz';
    g.font = '600 11px ' + FONT;
    const tw = g.measureText(t).width + 12, bx = Math.min(w - tw - 2, Math.max(2, x - tw / 2)), by = Math.max(2, y - 26);
    g.fillStyle = C.ink; g.beginPath();
    g.roundRect ? g.roundRect(bx, by, tw, 18, 9) : g.rect(bx, by, tw, 18);
    g.fill();
    g.fillStyle = getComputedStyle(document.body).backgroundColor; g.textAlign = 'left'; g.fillText(t, bx + 6, by + 13);
  }
  g.font = '11px ' + FONT; g.fillStyle = C.ink2; g.textAlign = 'right';
  labels.forEach((t, i) => { if (i) g.fillText(t, w - 4, (h * i) / 4 - 4); });
}
function freqAxis(el, fTop) {
  const html = [0, 0.25, 0.5, 0.75, 1].map((k, i) => `<span>${fmtK(k * fTop)}${i === 4 ? ' Hz' : ''}</span>`).join('');
  if (el.innerHTML !== html) el.innerHTML = html;
}

function updateWaterfall(n) {
  const ch = selCh(), N = cfg.win, fs = src.fs;
  let hop = Math.max(1, Math.round(N * (1 - cfg.overlap / 100)));
  let rows = Math.ceil((cfg.hist * fs) / hop);
  if (rows > 720) { rows = 720; hop = Math.ceil((cfg.hist * fs) / rows); }
  const cols = Math.min(N / 2, 512);
  const key = [src.kind, src.run || src.url, src.epoch, ch.phys, ch.sens, N, cfg.wfn, cfg.overlap, cfg.hist, cfg.dyn, cfg.units, cfg.dc, fs].join('|');
  if (key !== wf.key || n < wf.lastEnd || n - wf.lastEnd > rows * hop) {
    wf.key = key; wf.rows = rows; wf.cols = cols; wf.hop = hop; wf.head = 0; wf.top = -Infinity;
    wf.off = document.createElement('canvas'); wf.off.width = cols; wf.off.height = rows;
    wf.img = new ImageData(cols, rows);
    for (let i = 0; i < cols * rows; i++) { wf.img.data[i * 4] = C.wf[0][0]; wf.img.data[i * 4 + 1] = C.wf[0][1]; wf.img.data[i * 4 + 2] = C.wf[0][2]; wf.img.data[i * 4 + 3] = 255; }
    wf.lastEnd = Math.floor(n / hop) * hop - rows * hop;
    wf.dirty = true;
  }
  let steps = 0;
  while (wf.lastEnd + wf.hop <= n && steps++ <= wf.rows) {
    wf.lastEnd += wf.hop;
    wfRow(ch, wf.lastEnd);
  }
}
function wfRow(ch, end) {
  const N = cfg.win, x = readCh(ch, end - N, end);
  wf.head = (wf.head - 1 + wf.rows) % wf.rows;
  const d = wf.img.data, base = wf.head * wf.cols * 4;
  if (!x) {
    for (let c = 0; c < wf.cols; c++) { const o = base + c * 4; d[o] = C.wf[0][0]; d[o + 1] = C.wf[0][1]; d[o + 2] = C.wf[0][2]; }
    wf.dirty = true;
    return;
  }
  const mag = amplitude(x, cfg.wfn), per = N / 2 / wf.cols, row = new Float32Array(wf.cols);
  let rowMax = -Infinity;
  for (let c = 0; c < wf.cols; c++) {
    let m = 0;
    for (let k = Math.floor(c * per); k < Math.floor((c + 1) * per); k++) if (mag[k] > m) m = mag[k];
    row[c] = 20 * Math.log10(m + 1e-12);
    if (c > 0 && row[c] > rowMax) rowMax = row[c];
  }
  wf.top = Math.max(rowMax, wf.top - 0.01);
  const bottom = wf.top - cfg.dyn;
  for (let c = 0; c < wf.cols; c++) {
    const li = Math.max(0, Math.min(255, Math.round(((row[c] - bottom) / cfg.dyn) * 255))) * 3, o = base + c * 4;
    d[o] = LUT[li]; d[o + 1] = LUT[li + 1]; d[o + 2] = LUT[li + 2];
  }
  wf.dirty = true;
}
function drawWaterfall(fTop, nyq) {
  const { g, w, h } = fit($('#cvWF'));
  if (!wf.off) return;
  if (wf.dirty) { wf.off.getContext('2d').putImageData(wf.img, 0, 0); wf.dirty = false; }
  const sw = Math.max(1, Math.round((fTop / nyq) * wf.cols)), R = wf.rows, top = R - wf.head;
  g.imageSmoothingEnabled = false;
  g.drawImage(wf.off, 0, wf.head, sw, top, 0, 0, w, (top / R) * h);
  if (wf.head) g.drawImage(wf.off, 0, 0, sw, wf.head, 0, (top / R) * h, w, (wf.head / R) * h);
}

function drawTimeline(nSec) {
  const { g, w, h } = fit($('#cvTL'));
  const bars = 120, bw = w / bars;
  for (let i = 0; i < bars; i++) {
    const s = processState(src.labelAt(nSec - 30 + ((i + 1) * 30) / bars));
    g.fillStyle = s == null ? C.fill2 : C.s[s];
    g.fillRect(i * bw + 0.5, 0, Math.max(1, bw - 1.5), h);
  }
}

function updateOutputs(sp, nSec, single) {
  const s = processState(src.labelAt(nSec));
  if (s !== view.lastState) { view.lastState = s; view.stateSince = nSec; }
  const card = $('.kpi-state');
  if (s == null) {
    delete card.dataset.s;
    $('#stateCode').textContent = '–';
    $('#stateName').textContent = src.kind === 'live' && src.ready ? 'No label yet' : 'Waiting for data';
  } else {
    card.dataset.s = s;
    $('#stateCode').textContent = s;
    if ($('#stateName').textContent !== STATE_NAMES[s]) $('#stateName').textContent = STATE_NAMES[s];
  }
  $$('#stateSteps li').forEach((li) => li.classList.toggle('on', +li.dataset.s === s));
  $('#stateSub').textContent = s == null ? ' ' : `for ${Math.max(0, nSec - view.stateSince).toFixed(1)} s`;

  const unit = unitName(), ch = sp.ch;
  $$('.selName').forEach((e) => { e.textContent = ch.name; });
  $('#selDot').style.background = chColor(ch.phys);
  $('#selSub').textContent = describe(ch.phys);
  $('#rmsVal').textContent = fmt(sp.rms);
  $('#rmsSub').textContent = `Over the last ${sp.N} samples (${((sp.N / src.fs) * 1000).toFixed(1)} ms)`;
  const top = sp.peaks[0];
  $('#domVal').textContent = top ? fmtHz(top.f) : '–';
  $('#domSub').textContent = top ? `${fmt(top.a)} ${unit} · Δf ${sp.df.toFixed(2)} Hz` : ' ';
  $('#selPeak').textContent = single ? fmt(peakOf(single)) + ' ' + unit : '–';
  $('#selCrest').textContent = single && rmsOf(single) > 0 ? (peakOf(single) / rmsOf(single)).toFixed(2) : '–';
  $('#fftSub').textContent = `${ch.name} · N = ${sp.N} · Δf = ${sp.df.toFixed(2)} Hz · ${cfg.fftScale === 'db' ? 'dB re 1 ' + unit : unit}`;
  $('#wfSub').textContent = `${ch.name} · hop ${((wf.hop / src.fs) * 1000).toFixed(1)} ms · newest at the top`;
  $('#wfMid').textContent = '−' + cfg.hist / 2 + ' s';
  $('#wfOld').textContent = '−' + cfg.hist + ' s';
  $('#topSub').textContent = `Top 10 by magnitude · 0 to ${fmtFs(sp.fTop)}`;

  const amax = top ? top.a : 1;
  const html = sp.peaks.map((p, i) => `<li><span class="rk">${i + 1}</span><span class="fq">${fmtHz(p.f)} Hz</span>`
    + `<span class="mg">${fmt(p.a)} ${unit}</span><span class="bar"><i style="width:${Math.max(2, (p.a / amax) * 100).toFixed(1)}%"></i></span></li>`).join('');
  $('#peaks').innerHTML = html || '<li class="empty">No peaks</li>';

  for (const li of $$('#chList li')) {
    const c = chByPhys(+li.dataset.phys);
    const v = c ? readCh(c, Math.round(nSec * src.fs) - sp.N, Math.round(nSec * src.fs)) : null;
    li.querySelector('[data-rms]').textContent = v ? 'RMS ' + fmt(rmsOf(v)) + ' ' + unit : 'no signal';
  }
  $('#chCount').textContent = `${cfg.channels.length} ${cfg.channels.length === 1 ? 'channel' : 'channels'} · click one to show it on the right`;
}

// ------------------------------------------------------------------ main loop

let lastDraw = 0;
function frame(ms) {
  requestAnimationFrame(frame);
  if (src && src.tick) src.tick(ms);
  if (ms - lastDraw < 33) return;
  lastDraw = ms;
  const textNow = ms - view.lastText > 200;
  if (textNow) updateStatus();
  if (!src || !src.ready) return;
  const n = now(), nSec = n / src.fs;
  drawChannels(n);
  const single = drawSingle(n);
  const sp = spectrum(n);
  drawFFT(sp);
  freqAxis($('#fftAxis'), sp.fTop);
  freqAxis($('#wfAxis'), sp.fTop);
  updateWaterfall(n);
  drawWaterfall(sp.fTop, sp.nyq);
  drawTimeline(nSec);
  if (textNow) { view.lastText = ms; updateOutputs(sp, nSec, single); syncControls(); }
}

// ------------------------------------------------------------------ wiring

function openSheet(open) {
  const sheet = $('#sheet');
  sheet.classList.toggle('open', open);
  sheet.setAttribute('aria-hidden', String(!open));
  sheet.inert = !open;
  $('#scrim').hidden = !open;
  $('#inputsBtn').setAttribute('aria-expanded', String(open));
  if (open) $('#sheetClose').focus(); else $('#inputsBtn').focus();
}

function init() {
  readColors();
  buildSegs();
  $('#sheet').inert = true;
  $('#inputsBtn').addEventListener('click', () => openSheet(true));
  $('#sheetClose').addEventListener('click', () => openSheet(false));
  $('#scrim').addEventListener('click', () => openSheet(false));
  $('#sheet').addEventListener('keydown', (e) => { if (e.key === 'Escape') openSheet(false); });
  $('#addCh').addEventListener('click', addChannel);
  $('#chPlus').addEventListener('click', addChannel);
  $('#chMinus').addEventListener('click', () => removeChannel(cfg.channels[cfg.channels.length - 1].phys));
  $('#pauseBtn').addEventListener('click', () => {
    paused = !paused;
    frozen = src ? src.total : 0;
    if (src && src.kind === 'demo') src.playing = !paused;
    syncControls();
  });
  $('#recBtn').addEventListener('click', async () => {
    if (!src || src.kind !== 'live') return;
    const on = src.info && src.info.record && src.info.record.on;
    try { await src.post(on ? '/record/stop' : '/record/start'); } catch (e) { showBanner('Record failed: ' + e.message); }
  });
  $('#connectBtn').addEventListener('click', () => { cfg.liveUrl = $('#liveUrl').value.trim(); saveCfg(); switchSource(); });
  $('#fsPick').addEventListener('change', async (e) => {
    if (!src || src.kind !== 'live') return;
    try { await src.post('/config?fs=' + encodeURIComponent(e.target.value)); } catch (err) { showBanner('Could not change the sampling frequency: ' + err.message); }
  });
  $('#resetBtn').addEventListener('click', () => {
    try { localStorage.removeItem(STORE_KEY); } catch (e) { /* nothing stored */ }
    const keepSource = { source: cfg.source, demoRun: cfg.demoRun, liveUrl: cfg.liveUrl };
    cfg = { ...structuredClone(DEFAULTS), ...keepSource };
    saveCfg(); view.ranges = {}; renderChannels(); syncControls();
  });
  const mq = window.matchMedia('(prefers-color-scheme: dark)');
  (mq.addEventListener ? mq.addEventListener.bind(mq, 'change') : mq.addListener.bind(mq))(() => { readColors(); wf.key = ''; renderChannels(); });
  loadDemoIndex().then(switchSource);
  if (location.hash === '#inputs') openSheet(true);
  syncControls();
  requestAnimationFrame(frame);
}
init();
