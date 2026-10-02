/* Scenario forecast figure.
 *
 * Draws the posterior-predictive forecast computed in Python for each storm scenario: three
 * stacked panels (warehouse receipts, the cumulative gap against the baseline, and the
 * backlog of returns in transit) sharing one day cursor, two planning tiles (the storm
 * and the week after) and a summary sentence. The gap is a paired per-draw difference
 * computed in Python. Every scenario is already in the `data` trait and the buttons only
 * choose which one to draw, so the figure needs no Python once it is on the page (it works
 * in a static export). No numbers, dates or labels are defaulted here. */

const PANELS = [
  { key: 'receipt', title: 'Warehouse receipts', unit: 'forecast receipts / day', h: 76 },
  { key: 'gap', title: 'Receipts so far, against the baseline', unit: 'scenario minus storm as forecast', h: 60 },
  { key: 'open', title: 'Backlog', unit: 'returns in transit \u00b7 initiated, not yet received \u00b7 end of day', h: 56, axis: true },
];

const NARROW_PX = 600;
const STRIP_PX = 5;

const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const MINUS = '\u2212';
const body = (a) => a.toLocaleString('en-US', { maximumFractionDigits: a >= 100 ? 0 : a >= 10 ? 1 : 2 });
const num = (n) => n.toLocaleString('en-US', { maximumFractionDigits: Math.abs(n) >= 100 ? 0 : Math.abs(n) >= 10 ? 1 : 2 });
const sgn = (n) => {
  const b = body(Math.abs(n));
  return (n > 0 && b !== '0' ? '+' : n < 0 && b !== '0' ? MINUS : '\u00b1') + b;
};
const range = (lo, hi) => num(lo) + '\u2013' + num(hi);
const sgnRange = (lo, hi) => sgn(lo) + ' to ' + sgn(hi);
// Whole-unit versions for totals over several days, where decimals are noise.
const whole = (n) => num(Math.round(n));
const sgnWhole = (n) => sgn(Math.round(n));

const parseIso = (iso) => {
  const [y, m, d] = iso.slice(0, 10).split('-').map(Number);
  return new Date(Date.UTC(y, m - 1, d));
};
const fmtDate = (iso, opts) => new Intl.DateTimeFormat('en-US', { timeZone: 'UTC', ...opts }).format(parseIso(iso));
const dayShort = (iso) => fmtDate(iso, { month: 'short', day: 'numeric' });
const dayLong = (iso) => fmtDate(iso, { weekday: 'long', month: 'short', day: 'numeric' });
const dayFull = (iso) => fmtDate(iso, { month: 'short', day: 'numeric', year: 'numeric' });
const isWeekend = (iso) => [0, 6].includes(parseIso(iso).getUTCDay());
const weekdayInitial = (iso) => fmtDate(iso, { weekday: 'narrow' });

function niceMax(v) {
  if (!(v > 0)) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) if (m * p >= v * 1.04) return m * p;
  return 10 * p;
}

// Contiguous runs of true values, as [first, last] index pairs.
function spans(flags) {
  const out = [];
  let d = 0;
  while (d < flags.length) {
    if (flags[d]) {
      let e = d;
      while (e + 1 < flags.length && flags[e + 1]) e++;
      out.push([d, e]);
      d = e + 1;
    } else d++;
  }
  return out;
}

// One scenario as the figure draws it: the shared header joined to that scenario's series.
const viewOf = (data, key) => ({
  as_of: data.as_of,
  dates: data.dates,
  weather_labels: data.weather_labels,
  selection: { weather: key },
  ...data.scenarios[key],
});

function validate(data) {
  const bad = (msg) => { throw new Error('Scenario payload: ' + msg); };
  if (!data || typeof data !== 'object') bad('expected an object');
  if (!Array.isArray(data.order) || data.order.length === 0) bad('order is missing');
  if (!data.order.includes(data.default)) bad('default is not one of the scenarios');
  if (!data.scenarios || typeof data.scenarios !== 'object') bad('scenarios are missing');
  if (!data.weather_labels) bad('weather_labels are missing');
  for (const key of data.order) {
    if (typeof data.weather_labels[key] !== 'string') bad('weather_labels.' + key + ' is missing');
    if (!data.scenarios[key]) bad('scenarios.' + key + ' is missing');
    validateView(viewOf(data, key));
  }
  return data;
}

function validateView(p) {
  const bad = (msg) => { throw new Error('Scenario payload: ' + msg); };
  if (!p || typeof p !== 'object') bad('expected an object');
  if (typeof p.as_of !== 'string') bad('as_of is missing');
  if (!Array.isArray(p.dates) || p.dates.length === 0) bad('dates are missing');
  const n = p.dates.length;
  if (!p.selection || typeof p.selection.weather !== 'string') bad('selection is missing');
  const nums = (a, what) => { if (!Array.isArray(a) || a.length !== n || !a.every((v) => Number.isFinite(v))) bad(what + ' must be ' + n + ' finite numbers'); };
  const bools = (a, what) => { if (!Array.isArray(a) || a.length !== n) bad(what + ' must have ' + n + ' entries'); };
  const band = (s, what) => { if (!s) bad(what + ' is missing'); for (const f of ['mean', 'lo90', 'hi90', 'lo50', 'hi50']) nums(s[f], what + '.' + f); };
  const total = (t, what) => { if (!t || !['mean', 'lo90', 'hi90'].every((f) => Number.isFinite(t[f]))) bad(what + ' is incomplete'); };
  if (!p.storm) bad('storm is missing');
  bools(p.storm.baseline, 'storm.baseline');
  bools(p.storm.scenario, 'storm.scenario');
  for (const key of ['receipt', 'open']) for (const arm of ['baseline', 'scenario']) band(p.panels && p.panels[key] && p.panels[key][arm], 'panels.' + key + '.' + arm);
  band(p.gap, 'gap');
  for (const key of ['receipt', 'open_end']) for (const arm of ['baseline', 'scenario']) total(p.totals && p.totals[key] && p.totals[key][arm], 'totals.' + key + '.' + arm);
  if (!Array.isArray(p.windows) || p.windows.length === 0) bad('windows are missing');
  p.windows.forEach((w, i) => {
    if (!Number.isInteger(w.start) || !Number.isInteger(w.end) || w.start < 0 || w.end < w.start || w.end >= n) bad('windows[' + i + '] has a bad range');
    for (const k of ['baseline', 'scenario', 'diff']) total(w.receipt && w.receipt[k], 'windows[' + i + '].receipt.' + k);
  });
  return p;
}

const SKELETON = `
<section class="sc">
  <div class="sc-msg" data-part="error" role="alert" hidden></div>
  <div data-part="ready" hidden>
    <div class="sc-controls">
      <span class="sc-controls-label" id="sc-scenario-label">Storm scenario</span>
      <div class="sc-seg" role="group" aria-labelledby="sc-scenario-label" data-part="pills"></div>
      <p class="sc-controls-help">A scenario you choose, not a weather prediction. The fitted model is not refit.</p>
    </div>
    <div class="sc-status">
      <span class="sc-badge">Scenario forecast</span>
      <span class="sc-from" data-part="from"></span>
    </div>
    <div class="sc-tiles" data-part="tiles"></div>
    <div class="sc-body">
      <h3 class="sc-day-name" data-part="day-name"></h3>
      <div class="sc-plots" data-part="plots" tabindex="0" role="slider" aria-orientation="horizontal" aria-label="Forecast day cursor shared by all charts" aria-valuemin="1" aria-valuenow="1">
        ${PANELS.map((p) => `
        <div class="sc-panel" data-panelwrap="${p.key}">
          <div class="sc-ph">
            <h3>${p.title}<span>${p.unit}</span></h3>
            <div class="sc-readout" data-readout="${p.key}"></div>
          </div>
          <div data-svgwrap="${p.key}"></div>
        </div>`).join('')}
      </div>
      <div class="sc-legend" role="list" aria-label="Figure legend" data-part="legend"></div>
    </div>
    <div class="sc-summary-wrap">
      <p class="sc-summary" data-part="summary" role="status" aria-live="polite"></p>
      <details class="sc-notes">
        <summary>How to read this forecast</summary>
        <ul data-part="notes"></ul>
      </details>
    </div>
  </div>
</section>`;

export default {
  render({ model, el }) {
    const host = document.createElement('div');
    host.innerHTML = SKELETON;
    const root = host.firstElementChild;
    el.appendChild(root);

    const $ = (part) => root.querySelector('[data-part="' + part + '"]');
    const ui = {
      error: $('error'), ready: $('ready'), from: $('from'), dayName: $('day-name'), tiles: $('tiles'),
      plots: $('plots'), legend: $('legend'), summary: $('summary'), notes: $('notes'), pills: $('pills'),
    };

    const state = { data: null, key: null, payload: null, day: null, hover: null, width: 0, renderedWidth: 0, narrow: false };
    const listeners = [];
    let ro = null;
    let disposed = false;

    const activeDay = () => (state.hover === null ? state.day : state.hover);
    // Labels are written for the control pills; every use here is mid-sentence.
    const weatherLabel = (key) => {
      const label = (state.payload.weather_labels && state.payload.weather_labels[key]) || key;
      return label.charAt(0).toLowerCase() + label.slice(1);
    };
    const nDays = () => state.payload.dates.length;
    const isBaseline = () => state.payload.selection.weather === 'as_forecast';
    const stormsDiffer = () => state.payload.storm.baseline.some((v, i) => v !== state.payload.storm.scenario[i]);
    // With the storm as forecast the gap is zero by construction, so its panel is hidden.
    const visiblePanels = () => PANELS.filter((p) => p.key !== 'gap' || !isBaseline());

    /* ---------- geometry ---------- */
    function geom(p) {
      const mob = state.narrow;
      const ml = mob ? 38 : 44, mr = mob ? 6 : 12, mt = 12;
      const ph = Math.max(40, (p.h || 60) - (mob ? 12 : 0));
      const ax = p.axis ? 30 : 0;
      const pw = Math.max(120, state.width - ml - mr);
      return { ml, mr, mt, ph, pw, cw: pw / nDays(), H: mt + ph + 6 + ax };
    }

    // Receipts start at zero and the gap is symmetric about zero. The backlog sits in the
    // hundreds and moves by tens, so its axis spans only the data.
    function domain(p) {
      const pl = state.payload;
      if (p.key === 'gap') {
        const m = niceMax(Math.max(...pl.gap.hi90.map(Math.abs), ...pl.gap.lo90.map(Math.abs)));
        return { lo: -m, hi: m };
      }
      const pn = pl.panels[p.key];
      const hi = Math.max(...pn.scenario.hi90, ...pn.baseline.mean);
      if (p.key !== 'open') return { lo: 0, hi: niceMax(hi) };
      const lo = Math.min(...pn.scenario.lo90, ...pn.baseline.mean);
      const step = niceMax((hi - lo) / 2);
      return { lo: Math.floor(lo / step) * step, hi: Math.ceil(hi / step) * step };
    }

    const seriesOf = (p) => (p.key === 'gap'
      ? { s: state.payload.gap, b: null }
      : { s: state.payload.panels[p.key].scenario, b: state.payload.panels[p.key].baseline });

    /* ---------- charts ---------- */
    function chartSvg(p) {
      const pl = state.payload;
      const n = nDays();
      const { s, b } = seriesOf(p);
      const g = geom(p);
      const dom = domain(p);
      const X = (d) => g.ml + (d + 0.5) * g.cw;
      const Y = (v) => g.mt + g.ph * (1 - (v - dom.lo) / (dom.hi - dom.lo));
      const fs = 11;
      const pt = (d, v) => X(d).toFixed(1) + ',' + Y(v).toFixed(1);
      const line = (arr) => arr.map((v, d) => pt(d, v)).join(' ');
      const band = (lo, hi) => hi.map((v, d) => pt(d, v)).join(' ') + ' ' + lo.map((v, d) => pt(d, v)).reverse().join(' ');
      const x0 = (d) => (g.ml + d * g.cw).toFixed(1);
      const width = (d, e) => ((e - d + 1) * g.cw).toFixed(1);
      const full = p.key === 'receipt';
      let h = `<svg class="sc-svg" data-panel="${p.key}" viewBox="0 0 ${state.width} ${g.H}" width="${state.width}" height="${g.H}" aria-hidden="true" focusable="false">`;
      // Receipts are shaded through the storm; the other panels carry a thin strip.
      for (const [d, e] of spans(pl.storm.scenario)) {
        h += `<rect class="sc-storm" x="${x0(d)}" y="${g.mt}" width="${width(d, e)}" height="${full ? g.ph : STRIP_PX}"/>`;
        if (full && (e - d + 1) * g.cw > 44) h += `<text class="sc-storm-label" x="${(g.ml + ((d + e + 1) / 2) * g.cw).toFixed(1)}" y="${g.mt + 12}" text-anchor="middle" font-size="${fs}">STORM</text>`;
      }
      if (stormsDiffer()) {
        for (const [d, e] of spans(pl.storm.baseline)) {
          h += `<rect class="sc-storm-base" x="${x0(d)}" y="${g.mt}" width="${width(d, e)}" height="${full ? g.ph : STRIP_PX}"/>`;
        }
      }
      for (const t of [0, 0.5, 1]) {
        const v = dom.lo + (dom.hi - dom.lo) * t;
        const y = Y(v);
        const zero = p.key === 'gap' && t === 0.5;
        h += `<line class="${zero ? 'sc-zero' : 'sc-grid'}" x1="${g.ml}" x2="${g.ml + g.pw}" y1="${y}" y2="${y}"${zero || t === 0 ? '' : ' stroke-dasharray="2 4"'}/>`;
        h += `<text x="${g.ml - 6}" y="${y + 4}" text-anchor="end" font-size="${fs}">${p.key === 'gap' ? (v === 0 ? '0' : sgn(v)) : num(v)}</text>`;
      }
      h += `<polygon class="sc-b90" points="${band(s.lo90, s.hi90)}"/>`;
      h += `<polygon class="sc-b50" points="${band(s.lo50, s.hi50)}"/>`;
      if (b) h += `<polyline class="sc-base" points="${line(b.mean)}"/>`;
      h += `<polyline class="sc-scen" points="${line(s.mean)}"/>`;
      if (p.axis) {
        const base = g.mt + g.ph;
        const showLetters = g.cw >= 12;
        for (let d = 0; d < n; d++) {
          const wk = isWeekend(pl.dates[d]);
          if (showLetters) h += `<text class="${wk ? 'sc-wk' : ''}" x="${X(d).toFixed(1)}" y="${base + 17}" text-anchor="middle" font-size="${fs}">${weekdayInitial(pl.dates[d])}</text>`;
          const tick = d % 7 === 0 || (d === n - 1 && (n - 1) % 7 >= 4);
          if (tick) h += `<text class="sc-tick-ink" x="${X(d).toFixed(1)}" y="${base + 31}" text-anchor="middle" font-size="${fs}">${dayShort(pl.dates[d])}</text>`;
        }
      }
      h += `<g class="sc-cursor" pointer-events="none"></g>`;
      h += `<rect x="0" y="0" width="${state.width}" height="${g.H}" fill="transparent"/>`;
      return h + '</svg>';
    }

    function cursorSvg(p, d) {
      const { s, b } = seriesOf(p);
      const g = geom(p);
      const dom = domain(p);
      const x = (g.ml + (d + 0.5) * g.cw).toFixed(1);
      const Y = (v) => (g.mt + g.ph * (1 - (v - dom.lo) / (dom.hi - dom.lo))).toFixed(1);
      let h = `<line class="sc-cur-line" x1="${x}" x2="${x}" y1="${g.mt - 4}" y2="${g.mt + g.ph}"/>`;
      h += `<line class="sc-cur-50" x1="${x}" x2="${x}" y1="${Y(s.lo50[d])}" y2="${Y(s.hi50[d])}"/>`;
      h += `<line class="sc-cur-90" x1="${x}" x2="${x}" y1="${Y(s.lo90[d])}" y2="${Y(s.hi90[d])}"/>`;
      h += `<line class="sc-cur-90" x1="${+x - 4}" x2="${+x + 4}" y1="${Y(s.lo90[d])}" y2="${Y(s.lo90[d])}"/>`;
      h += `<line class="sc-cur-90" x1="${+x - 4}" x2="${+x + 4}" y1="${Y(s.hi90[d])}" y2="${Y(s.hi90[d])}"/>`;
      if (b) h += `<circle class="sc-cur-base" cx="${x}" cy="${Y(b.mean[d])}" r="3.8"/>`;
      h += `<circle class="sc-cur-mean" cx="${x}" cy="${Y(s.mean[d])}" r="4.2"/>`;
      return h;
    }

    function readoutHtml(p, d) {
      const pl = state.payload;
      const { s, b } = seriesOf(p);
      if (p.key === 'receipt' && isWeekend(pl.dates[d])) return 'warehouse closed';
      if (p.key === 'gap') {
        return `<b class="g">${sgnWhole(s.mean[d])}</b> receipts so far \u00b7 90% interval <b>${sgnWhole(s.lo90[d])} to ${sgnWhole(s.hi90[d])}</b>`;
      }
      const delta = Math.abs(s.mean[d] - b.mean[d]) < 1e-9 ? '' : ` \u00b7 vs baseline <b>${sgn(s.mean[d] - b.mean[d])}</b>`;
      return `mean <b class="g">${num(s.mean[d])}</b> \u00b7 90% interval <b>${range(s.lo90[d], s.hi90[d])}</b>${delta}`;
    }

    function renderCharts() {
      const shown = new Set(visiblePanels().map((p) => p.key));
      for (const p of PANELS) {
        const wrap = root.querySelector(`[data-panelwrap="${p.key}"]`);
        wrap.hidden = !shown.has(p.key);
        if (shown.has(p.key)) root.querySelector(`[data-svgwrap="${p.key}"]`).innerHTML = chartSvg(p);
      }
      state.renderedWidth = state.width;
      renderCursor();
    }

    function renderCursor() {
      const pl = state.payload;
      const d = activeDay();
      for (const p of visiblePanels()) {
        const g = root.querySelector(`[data-svgwrap="${p.key}"] .sc-cursor`);
        if (g) g.innerHTML = cursorSvg(p, d);
        root.querySelector(`[data-readout="${p.key}"]`).innerHTML = readoutHtml(p, d);
      }
      ui.dayName.textContent = dayLong(pl.dates[d]) + ' \u00b7 forecast day ' + (d + 1) + ' of ' + nDays();
      const k = state.day;
      const at = (key) => {
        const s = pl.panels[key].scenario;
        return `mean ${num(s.mean[k])}, 90% interval ${range(s.lo90[k], s.hi90[k])}`;
      };
      ui.plots.setAttribute('aria-valuemax', String(nDays()));
      ui.plots.setAttribute('aria-valuenow', String(k + 1));
      ui.plots.setAttribute('aria-valuetext', `${dayLong(pl.dates[k])}, forecast: receipts ${at('receipt')}; backlog ${at('open')}`);
    }

    /* ---------- legend, tiles, summary ---------- */
    function renderLegend() {
      const pl = state.payload;
      const sw = (inner, w) => `<svg width="${w || 30}" height="12" aria-hidden="true">${inner}</svg>`;
      ui.legend.innerHTML = [
        sw('<line x1="1" y1="6" x2="29" y2="6" class="sc-base"/>') + 'Baseline mean: storm as forecast',
        sw('<line x1="1" y1="6" x2="29" y2="6" class="sc-scen"/>') + 'Scenario mean',
        sw('<rect x="1" y="1" width="28" height="10" class="sc-b50"/>') + 'Scenario 50% predictive interval',
        sw('<rect x="1" y="1" width="28" height="10" class="sc-b90"/>') + 'Scenario 90% predictive interval',
        pl.storm.scenario.some(Boolean) ? sw('<rect x="1" y="1" width="28" height="10" class="sc-storm"/>') + 'Storm days in the scenario' : '',
        stormsDiffer() ? sw('<rect x="1" y="1" width="28" height="10" class="sc-storm-base"/>') + 'Storm days as forecast' : '',
      ].filter(Boolean).map((t) => `<span role="listitem">${t}</span>`).join('') +
        '<span class="sc-hint" role="listitem">hover, tap, or focus \u00b7 \u2190 \u2192 Home End</span>';
    }

    function renderTiles() {
      const pl = state.payload;
      const names = ['During the storm', 'The week after'];
      ui.tiles.innerHTML = pl.windows.map((w, i) => {
        const r = w.receipt;
        const a = pl.dates[w.start], z = pl.dates[w.end];
        const sameMonth = a.slice(0, 7) === z.slice(0, 7);
        const dates = w.start === w.end ? dayShort(a) : `${dayShort(a)}\u2013${sameMonth ? parseIso(z).getUTCDate() : dayShort(z)}`;
        const head = `<p class="sc-tile-label">${names[i] || 'Window'}<span>${dates}</span></p>`;
        if (isBaseline()) {
          return `<div class="sc-tile">${head}<p class="sc-tile-value">${whole(r.scenario.mean)}<small>receipts expected</small></p>` +
            `<p class="sc-tile-detail">90% interval ${whole(r.scenario.lo90)}\u2013${whole(r.scenario.hi90)}</p></div>`;
        }
        return `<div class="sc-tile">${head}<p class="sc-tile-value">${sgnWhole(r.diff.mean)}<small>receipts vs the storm as forecast</small></p>` +
          `<p class="sc-tile-detail">${whole(r.scenario.mean)} expected, against ${whole(r.baseline.mean)} \u00b7 90% interval on the difference ${sgnWhole(r.diff.lo90)} to ${sgnWhole(r.diff.hi90)}</p></div>`;
      }).join('');
    }

    function renderSummary() {
      const pl = state.payload;
      const n = nDays();
      const first = dayShort(pl.dates[0]), last = dayShort(pl.dates[n - 1]);
      const rt = pl.totals.receipt, ot = pl.totals.open_end;
      const rawLabel = (pl.weather_labels && pl.weather_labels[pl.selection.weather]) || pl.selection.weather;
      let t = isBaseline() ? `${esc(rawLabel)}: ` : `With <em>${esc(weatherLabel(pl.selection.weather))}</em>: `;
      t += `the model forecasts <em>${num(rt.scenario.mean)}</em> warehouse receipts over ${first}\u2013${last} (90% interval ${range(rt.scenario.lo90, rt.scenario.hi90)} for the ${n}-day total)`;
      if (isBaseline()) {
        t += `, and <em>${num(ot.scenario.mean)}</em> returns still in transit on ${last}.`;
      } else {
        const totalGap = rt.scenario.mean - rt.baseline.mean;
        const biggest = Math.max(...pl.windows.map((w) => Math.abs(w.receipt.diff.mean)));
        t += Math.abs(totalGap) < 0.5 ? ', the same total as with the storm as forecast' : `, ${sgn(totalGap)} versus the storm as forecast`;
        // A storm mostly moves receipts in time: the total barely changes next to the windows.
        t += Math.abs(totalGap) < 0.25 * biggest ? '. The storm moves receipts in time rather than removing them.' : '.';
      }
      ui.summary.innerHTML = t;

      ui.notes.innerHTML = [
        'These are posterior-predictive forecasts of counts, not observations. Each band is a predictive interval for that single day; the line is the predictive mean.',
        'The middle panel subtracts the baseline from the scenario draw by draw (both runs share a seed, so draws pair) and accumulates the difference. Its band is an interval on that difference, not on either level.',
        `The 90% interval on the ${n}-day receipt total is an interval for the sum over all ${n} days. It is not the sum of the daily interval ends, and it can be narrower or wider relative to its mean than a single day.`,
        `Weather is a scenario input you choose, not a weather prediction. The dashed baseline is always ${esc(weatherLabel('as_forecast'))}.`,
        'The fitted return stages are not refit: only the storm input to the receipt stage changes.',
        'The backlog (initiated, not yet received) grows while a storm slows receipts and drains over the open days after it clears.',
        'Whether a customer who has initiated a return will ever ship it is latent. A missing receipt alone does not establish abandonment.',
      ].map((x) => `<li>${x}</li>`).join('');
    }

    /* ---------- data -> DOM ---------- */
    const measureRoot = () => Math.floor(root.getBoundingClientRect().width);
    const measurePlots = () => Math.floor(ui.plots.getBoundingClientRect().width);

    function applyNarrow() {
      const outer = measureRoot();
      state.narrow = outer > 0 && outer <= NARROW_PX;
      if (state.narrow) root.setAttribute('data-narrow', ''); else root.removeAttribute('data-narrow');
    }

    // Open on the first open day after the forecast storm, where the drained backlog lands.
    function defaultDay(pl) {
      const n = pl.dates.length;
      const lastStorm = pl.storm.baseline.lastIndexOf(true);
      if (lastStorm >= 0) {
        for (let d = lastStorm + 1; d < n; d++) if (!isWeekend(pl.dates[d])) return d;
      }
      return Math.floor((n - 1) / 2);
    }

    // The scenario buttons: they only choose which precomputed scenario to draw.
    function renderPills() {
      const { data } = state;
      ui.pills.innerHTML = data.order
        .map((key) => `<button type="button" class="sc-pill" data-key="${esc(key)}" aria-pressed="false">${esc(data.weather_labels[key])}</button>`)
        .join('');
    }

    function syncPills() {
      for (const b of ui.pills.querySelectorAll('.sc-pill')) b.setAttribute('aria-pressed', String(b.dataset.key === state.key));
    }

    // Draw the chosen scenario. Cheap enough to redo in full: no Python involved.
    function show() {
      const payload = viewOf(state.data, state.key);
      state.payload = payload;
      const n = nDays();
      if (state.day === null || state.day >= n) state.day = defaultDay(payload);
      state.hover = state.hover !== null && state.hover < n ? state.hover : null;
      ui.from.textContent = `Forecast from ${dayFull(payload.as_of)} \u00b7 next ${n} days, ${dayShort(payload.dates[0])} to ${dayShort(payload.dates[n - 1])}`;
      applyNarrow();
      syncPills();
      renderTiles();
      state.width = Math.max(160, measurePlots() || state.width || 700);
      renderLegend();
      renderCharts();
      renderSummary();
    }

    function renderAll() {
      let data = null;
      try {
        data = validate(model.get('data'));
      } catch (err) {
        state.payload = null;
        ui.ready.hidden = true;
        ui.error.hidden = false;
        ui.error.innerHTML = '<b>Scenario forecast unavailable</b>' + esc(err.message);
        return;
      }
      state.data = data;
      if (!data.order.includes(state.key)) state.key = data.default;
      ui.error.hidden = true;
      ui.ready.hidden = false;
      renderPills();
      show();
    }

    /* ---------- interaction ---------- */
    const on = (node, ev, fn) => { node.addEventListener(ev, fn); listeners.push([node, ev, fn]); };
    on(ui.pills, 'click', (e) => {
      const b = e.target.closest && e.target.closest('.sc-pill');
      if (!b || !state.data || b.dataset.key === state.key) return;
      state.key = b.dataset.key;
      show();
    });
    on(ui.plots, 'keydown', (e) => {
      if (!state.payload || e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
      const d = activeDay();
      const next = e.key === 'ArrowLeft' ? d - 1 : e.key === 'ArrowRight' ? d + 1 : e.key === 'Home' ? 0 : e.key === 'End' ? nDays() - 1 : null;
      if (next === null) return;
      e.preventDefault();
      state.day = Math.max(0, Math.min(nDays() - 1, next));
      state.hover = null;
      renderCursor();
    });
    const dayFromEvent = (e) => {
      const svg = e.target.closest && e.target.closest('svg.sc-svg');
      if (!svg || !state.payload) return null;
      const r = svg.getBoundingClientRect();
      const g = geom({});
      const x = ((e.clientX - r.left) / (r.width || state.width)) * state.width;
      return Math.max(0, Math.min(nDays() - 1, Math.floor((x - g.ml) / g.cw)));
    };
    const clearHover = () => { if (state.hover !== null) { state.hover = null; renderCursor(); } };
    on(ui.plots, 'pointermove', (e) => {
      const d = dayFromEvent(e);
      if (d === null) return;
      if (e.pointerType === 'mouse' || e.pointerType === 'pen' || e.pointerType === '') {
        if (d !== state.hover) { state.hover = d; renderCursor(); }
      } else if (d !== state.day) { state.day = d; state.hover = null; renderCursor(); }
    });
    on(ui.plots, 'pointerleave', clearHover);
    on(ui.plots, 'pointercancel', clearHover);
    on(ui.plots, 'pointerdown', (e) => {
      const d = dayFromEvent(e);
      if (d === null) return;
      state.day = d;
      state.hover = e.pointerType === 'mouse' ? d : null;
      renderCursor();
    });

    if (typeof ResizeObserver !== 'undefined') {
      ro = new ResizeObserver(() => {
        if (disposed || !state.payload) return;
        const wasNarrow = state.narrow;
        applyNarrow();
        const w = Math.max(160, measurePlots());
        if (wasNarrow !== state.narrow || Math.abs(w - state.renderedWidth) >= 2) {
          state.width = Math.max(160, measurePlots());
          renderCharts();
        }
      });
      ro.observe(root);
    }

    model.on('change:data', renderAll);
    renderAll();

    return function cleanup() {
      if (disposed) return;
      disposed = true;
      model.off('change:data', renderAll);
      if (ro) ro.disconnect();
      for (const [node, ev, fn] of listeners) node.removeEventListener(ev, fn);
      listeners.length = 0;
      root.remove();
    };
  },
};
