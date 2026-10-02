/* "Follow one purchase": three real units from one simulated sale day, shown as the ledger looked on
   a selected day. Every date, id, storm interval and count arrives in the widget's `data` trait
   (`purchase_story` in examples/retail_returns_story.py); nothing below is a forecast or a fitted result.
   Leaf module: no imports, no network. Styles live in purchase.css. */

const WD = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
const WDL = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
const MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const MONL = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
const days = (n) => `${n} day${n === 1 ? '' : 's'}`;
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

/* ---------- payload validation: fail loudly instead of drawing a plausible-looking wrong chart ---------- */
const isInt = (n) => Number.isInteger(n);
function buildModel(data) {
  const bad = (why) => { throw new Error(`purchase payload is not usable: ${why}`); };
  if (!data || typeof data !== 'object') bad('expected an object');
  const iso = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(data.sale));
  if (!iso) bad('`sale` is not an ISO date');
  const WINDOW = data.window90, SNAP = data.snapshot100;
  if (!isInt(WINDOW) || !isInt(SNAP) || WINDOW < 1 || SNAP < WINDOW) bad('`window90`/`snapshot100` are not ordered integers');
  if (!isInt(data.initial_day) || data.initial_day < 0 || data.initial_day > SNAP) bad('`initial_day` is outside the snapshot');
  if (!Array.isArray(data.records) || !data.records.length) bad('`records` is empty');
  const ids = new Set();
  const records = data.records.map((r) => {
    const tag = `record ${r && r.id}`;
    if (!r || !r.id || ids.has(r.id)) bad('records need unique ids');
    ids.add(r.id);
    ['letter', 'order', 'item', 'title', 'blurb'].forEach((k) => { if (typeof r[k] !== 'string' || !r[k]) bad(`${tag} lacks ${k}`); });
    const init = r.init == null ? null : r.init, recv = r.recv == null ? null : r.recv;
    if (init !== null && (!isInt(init) || init < 0 || init > WINDOW)) bad(`${tag} initiation is outside days 0-${WINDOW}`);
    if (recv !== null && (!isInt(recv) || init === null || recv < init || recv > SNAP)) bad(`${tag} receipt is not between initiation and the snapshot`);
    const storm = r.storm == null ? null : r.storm;
    if (storm !== null && !(Array.isArray(storm) && storm.length === 2 && isInt(storm[0]) && isInt(storm[1]) && storm[0] >= 0 && storm[0] <= storm[1] && storm[1] <= SNAP)) bad(`${tag} storm interval is malformed`);
    return { id: String(r.id), letter: r.letter, order: r.order, item: r.item, title: r.title, blurb: r.blurb, init, recv, storm };
  });
  const SALE = Date.UTC(+iso[1], +iso[2] - 1, +iso[3]);
  /* dates in UTC so every viewer sees the same calendar */
  const info = (d) => { const t = new Date(SALE + d * 864e5); return { wd: t.getUTCDay(), m: t.getUTCMonth(), day: t.getUTCDate(), y: t.getUTCFullYear() }; };
  const cal = {
    info,
    short: (d) => { const o = info(d); return `${MON[o.m]} ${o.day}`; },
    dow: (d) => WD[info(d).wd],
    longDate: (d) => { const o = info(d); return `${WDL[o.wd]}, ${MONL[o.m]} ${o.day}, ${o.y}`; },
    isWeekend: (d) => { const w = info(d).wd; return w === 0 || w === 6; },
  };
  return {
    WINDOW, SNAP, cal, records, initial: data.initial_day,
    /* "weekend posts zero receipts" is only claimed when the ledger's own receipts show it */
    quietWeekends: data.weekend_receipts === 0,
  };
}

/* ---------- record state at a given day (uses only what has been observed by then) ---------- */
function view(M, r, d) {
  const { cal, WINDOW } = M;
  const inited = r.init != null && d >= r.init;
  const got = r.recv != null && d >= r.recv;
  const since = inited ? (got ? r.recv - r.init : d - r.init) : null;
  const cur = got ? 3 : inited ? (d === r.init ? 1 : 2) : 0;
  let label, tone;
  if (got) { label = `Received ${cal.dow(r.recv)} ${cal.short(r.recv)}`; tone = 'done'; }
  else if (inited) { label = d === r.init ? 'Return just started' : 'Return started, no receipt yet'; tone = 'open'; }
  else if (d < WINDOW) { label = 'No return started yet'; tone = 'quiet'; }
  else if (d === WINDOW) { label = 'Last day to start a return'; tone = 'quiet'; }
  else { label = 'Window closed, no return started'; tone = 'closed'; }
  return { inited, got, since, cur, label, tone };
}

function stages(M, r, d) {
  const { cal, WINDOW } = M, v = view(M, r, d), s = [];
  s.push({ name: 'Purchase', st: 'observed', tag: 'Observed', big: `${cal.dow(0)} ${cal.short(0)}`, sub: `Day 0 · ${cal.short(0)} sale` });
  if (v.inited) s.push({ name: 'Return started', st: 'observed', tag: 'Observed', big: `${cal.dow(r.init)} ${cal.short(r.init)}`, sub: `Day ${r.init} · initiation recorded` });
  else if (d < WINDOW) s.push({ name: 'Return started', st: 'pending', tag: 'Not seen yet', big: 'No initiation', sub: `${days(WINDOW - d)} left in the window` });
  else if (d === WINDOW) s.push({ name: 'Return started', st: 'pending', tag: 'Last day', big: 'No initiation', sub: `Window closes tonight, ${cal.short(WINDOW)}` });
  else s.push({ name: 'Return started', st: 'closed', tag: 'Window closed', big: 'None started', sub: `Deadline was ${cal.short(WINDOW)} (day ${WINDOW})` });
  if (!v.inited) s.push({ name: 'In transit', st: 'na', tag: 'Not entered', big: '—', sub: 'Nothing to ship yet' });
  else if (v.got) s.push({ name: 'In transit', st: 'observed', tag: 'Observed span', big: days(v.since), sub: 'Initiation to receipt' });
  else s.push({ name: 'In transit', st: 'open', tag: 'Clock running', big: d === r.init ? 'Day 0' : `Day ${v.since}`, sub: 'Open interval, unfinished' });
  if (v.got) s.push({ name: 'Received', st: 'observed', tag: 'Observed', big: `${cal.dow(r.recv)} ${cal.short(r.recv)}`, sub: `Day ${r.recv} · receipt recorded` });
  else if (v.inited) s.push({ name: 'Received', st: 'open', tag: 'No receipt yet', big: 'Not observed', sub: 'Absence is not abandonment' });
  else s.push({ name: 'Received', st: 'na', tag: 'Not applicable', big: '—', sub: d <= WINDOW ? 'Needs a return to start first' : 'Nothing was started' });
  return s;
}

/* one-line censoring context shown beside the status */
function ctx(M, r, d) {
  const { cal, WINDOW, SNAP } = M, v = view(M, r, d);
  if (!v.inited) {
    if (d === 0) return 'whether a return ever starts is not in the ledger yet';
    if (d < WINDOW) return 'silence is not an answer; a return can still start';
    if (d === WINDOW) return `the ${WINDOW}-day policy limits starting a return, not receiving one`;
    return 'no receipt to wait for; the deadline limited starting, not receipt';
  }
  if (v.got) return `second clock stopped after ${days(v.since)}${r.storm ? '; a storm fell inside the wait' : ''}`;
  if (d >= SNAP) return 'snapshot ends here; no receipt is not proof of abandonment';
  if (d === r.init) return 'second clock starts here; receipt is not observed';
  if (r.storm && d >= r.storm[0] && d <= r.storm[1]) return 'storm slowed carriers: a delay, not abandonment';
  if (M.quietWeekends && cal.isWeekend(d)) return 'no receipt posts on weekends, so a quiet ledger says little';
  if (d > WINDOW) return 'the deadline limited starting, not receipt; no receipt is not abandonment';
  return 'no receipt yet is not proof of abandonment';
}

/* ---------- little SVG pieces ---------- */
const P = (a) => a.map((p) => p.map((n) => +n.toFixed(2)).join(',')).join(' ');
function cube(cx, cy, s, o = {}) {
  const f = o.fill || '#fffef9', da = o.dash ? ` stroke-dasharray="${o.dash}"` : '';
  const top = [[cx, cy - .9 * s], [cx + s, cy - .4 * s], [cx, cy + .1 * s], [cx - s, cy - .4 * s]];
  const left = [[cx - s, cy - .4 * s], [cx, cy + .1 * s], [cx, cy + 1.1 * s], [cx - s, cy + .6 * s]];
  const right = [[cx, cy + .1 * s], [cx + s, cy - .4 * s], [cx + s, cy + .6 * s], [cx, cy + 1.1 * s]];
  const sw = o.sw || 1.5;
  return `<g stroke="${o.stroke || 'currentColor'}" stroke-width="${sw}" stroke-linejoin="round"${da}>
    <polygon points="${P(left)}" fill="${o.shade || f}"/><polygon points="${P(right)}" fill="${f}"/><polygon points="${P(top)}" fill="${o.top || f}"/>
    <line x1="${cx - s / 2}" y1="${cy - .65 * s}" x2="${cx + s / 2}" y2="${cy - .15 * s}" stroke-width="${sw * 3}" stroke-opacity=".3"/></g>`;
}
const ICONS = [
  () => `<svg viewBox="0 0 44 44" width="30" height="30" aria-hidden="true" focusable="false">${cube(22, 19, 11)}</svg>`,
  () => `<svg viewBox="0 0 44 44" width="30" height="30" aria-hidden="true" focusable="false">${cube(18, 24, 9)}<g fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M38 11A6.5 6.5 0 1 0 34 17.5"/><path d="M34 17.5l-3.6-.2M34 17.5l.6-3.7"/></g></svg>`,
  () => `<svg viewBox="0 0 44 44" width="30" height="30" aria-hidden="true" focusable="false"><path d="M4 35C13 12 26 40 39 12" fill="none" stroke="currentColor" stroke-width="1.8" stroke-dasharray="1 4.5" stroke-linecap="round"/>${cube(22, 23, 5.5, { sw: 1.3 })}<path d="M39 12l-4.5 1.2M39 12l-.5 4.6" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>`,
  () => `<svg viewBox="0 0 44 44" width="30" height="30" aria-hidden="true" focusable="false">${cube(19, 20, 10)}<circle cx="33" cy="32" r="8" fill="currentColor" stroke="none"/><path d="M29.2 32.2l2.7 2.7 5-5.4" fill="none" stroke="#fffef9" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
];

/* ---------- main wide timeline (rendered at real pixel width so labels stay legible) ---------- */
function axis(M, r, d, w, hind, uid) {
  const { cal, WINDOW, SNAP } = M;
  const nar = w < 520, L = nar ? 16 : 28, T = w - 2 * L, x = (v) => L + v * T / SNAP, dx = T / SNAP;
  const k = nar ? 16 : 0, PT = 26, BL = 138 + k, PH = BL - PT, H = BL + 93, R = (n) => BL + 43 + n * 14;
  const c2 = 90 + k, c3 = 116 + k;
  const v = view(M, r, d), step = nar ? 20 : 10;
  const o = [];
  /* estimated text width (sans ~.56em, mono ~.6em) so no label is clipped by the svg edge */
  const tw = (s, px, mono) => s.length * px * (mono ? .6 : .56);
  const place = (cx, text, px) => {
    const t = tw(text, px);
    if (cx + 6 + t <= w - 4) return { x: cx + 6, a: 'start' };
    if (cx - 6 - t >= 4) return { x: cx - 6, a: 'end' };
    return { x: Math.max(4, w - 4 - t), a: 'start' };
  };
  o.push(`<defs>
    <pattern id="${uid}-hs" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(60)"><rect width="5" height="5" fill="#e3e8ec"/><line x1="0" y1="0" x2="0" y2="5" stroke="#4c5a6b" stroke-opacity=".55" stroke-width="1.2"/></pattern>
  </defs>`);
  o.push(`<rect x="0" y="0" width="${w}" height="${H}" fill="#fffef9"/>`);
  const snapLab = `snapshot · ${cal.short(SNAP)}`, snapX1 = x(SNAP) - 6, snapX0 = snapX1 - tw(snapLab, 11, true);
  if (r.storm) {
    const a = x(r.storm[0]) - dx / 2, b = x(r.storm[1]) + dx / 2;
    o.push(`<rect x="${a.toFixed(2)}" y="${PT}" width="${(b - a).toFixed(2)}" height="${PH}" fill="url(#${uid}-hs)"/>`);
    /* label beside the band; flips left of it when it would collide with the snapshot label */
    const t = tw('storm', 11);
    if (b > snapX0 - t - 4) o.push(`<text x="${(a - 4).toFixed(1)}" y="37" text-anchor="end" font-size="11" fill="#4c5a6b" font-style="italic">storm</text>`);
    else o.push(`<text x="${b.toFixed(1)}" y="37" text-anchor="end" font-size="11" fill="#4c5a6b" font-style="italic">storm</text>`);
  }
  /* lane 1: the policy window */
  const dlLab = `deadline to start · day ${WINDOW}`;
  o.push(`<text x="${L}" y="51" font-size="11" fill="#62695d">${WINDOW}-day window to start a return</text>
    <rect x="${x(0)}" y="56" width="${(x(WINDOW) - x(0)).toFixed(2)}" height="8" fill="#e9eee5" stroke="#42644d" stroke-width="1"/>
    <rect x="${x(WINDOW).toFixed(2)}" y="56" width="${(x(SNAP) - x(WINDOW)).toFixed(2)}" height="8" fill="#efede2" stroke="#c4c8b8"/>
    <line x1="${x(WINDOW).toFixed(2)}" y1="49" x2="${x(WINDOW).toFixed(2)}" y2="68" stroke="#42644d" stroke-width="2"/>`);
  /* lane 2: clock 1 */
  o.push(`<text x="${L}" y="${79 + k}" font-size="11" fill="#62695d">Clock 1 · age since purchase</text>
    <line x1="${x(0)}" y1="${c2}" x2="${x(SNAP)}" y2="${c2}" stroke="#dcded1" stroke-width="2"/>
    <rect x="${x(0)}" y="${c2 - 5}" width="${Math.max(0, x(d) - x(0)).toFixed(2)}" height="10" fill="#4c5a6b"/>
    <circle cx="${x(d).toFixed(2)}" cy="${c2}" r="6" fill="#4c5a6b" stroke="#fffef9" stroke-width="2"/>`);
  /* lane 3: clock 2 */
  o.push(`<text x="${L}" y="${106 + k}" font-size="11" fill="#62695d">Clock 2 · age since initiation</text>`);
  if (v.inited) {
    const a = x(r.init), b = v.got ? x(r.recv) : x(d);
    if (v.got) {
      o.push(`<rect x="${a.toFixed(2)}" y="${c3 - 5}" width="${(b - a).toFixed(2)}" height="10" fill="#42644d"/><rect x="${(a - 2).toFixed(2)}" y="${c3 - 8}" width="4" height="16" fill="#42644d"/><rect x="${(b - 2).toFixed(2)}" y="${c3 - 8}" width="4" height="16" fill="#42644d"/>`);
    } else {
      o.push(`<rect x="${a.toFixed(2)}" y="${c3 - 4}" width="${Math.max(0, b - a).toFixed(2)}" height="8" fill="#f4ebd8" stroke="#956017" stroke-width="1.5" stroke-dasharray="5 4"/><rect x="${(a - 2).toFixed(2)}" y="${c3 - 8}" width="4" height="16" fill="#956017"/><circle cx="${b.toFixed(2)}" cy="${c3}" r="6.5" fill="#fffef9" stroke="#956017" stroke-width="2"/>`);
    }
  } else {
    o.push(`<line x1="${x(0)}" y1="${c3}" x2="${x(SNAP)}" y2="${c3}" stroke="#c4c8b8" stroke-width="2" stroke-dasharray="1 6" stroke-linecap="round"/><text x="${L}" y="${c3 + 16}" font-size="11" fill="#8c917f" font-style="italic">${d > WINDOW ? 'never started' : 'not started'}</text>`);
  }
  if (hind && r.init != null) {
    const from = Math.max(d, r.init), to = r.recv != null ? r.recv : SNAP;
    if (to > from) o.push(`<rect x="${x(from).toFixed(2)}" y="${c3 - 4}" width="${(x(to) - x(from)).toFixed(2)}" height="8" fill="none" stroke="${r.recv != null ? '#42644d' : '#956017'}" stroke-width="1.3" stroke-dasharray="1 4" stroke-linecap="round" opacity=".8"/>`);
  }
  /* baseline + ticks */
  o.push(`<line x1="${x(0)}" y1="${BL}" x2="${x(SNAP)}" y2="${BL}" stroke="#292d26" stroke-width="1"/>`);
  for (let t = 0; t <= SNAP; t += step) o.push(`<line x1="${x(t).toFixed(2)}" y1="${BL}" x2="${x(t).toFixed(2)}" y2="${BL + 5}" stroke="#292d26"/><text x="${x(t).toFixed(2)}" y="${BL + 17}" text-anchor="middle" font-size="11" fill="#62695d" paint-order="stroke" stroke="#fffef9" stroke-width="4">${t}</text>`);
  /* calendar labels: the sale day plus each month start that fits without crowding */
  let lastEnd = -1e9;
  for (let dd = 0; dd <= SNAP; dd++) {
    if (dd !== 0 && cal.info(dd).day !== 1) continue;
    const lab = cal.short(dd), x0 = x(dd) + 4, x1 = x0 + tw(lab, 11, true);
    if (dd !== 0 && (x0 < lastEnd + 8 || x1 > w - 2)) continue;
    lastEnd = x1;
    o.push(`<line x1="${x(dd).toFixed(2)}" y1="${BL + 5}" x2="${x(dd).toFixed(2)}" y2="${BL + 24}" stroke="#b9bdab"/><text x="${x0.toFixed(1)}" y="${BL + 31}" font-size="11" fill="#62695d" font-family="ui-monospace,monospace" paint-order="stroke" stroke="#fffef9" stroke-width="4">${lab}</text>`);
  }
  /* quiet tint over days not yet observed on the selected day */
  if (d < SNAP) o.push(`<rect x="${x(d).toFixed(2)}" y="${PT}" width="${(w - x(d)).toFixed(2)}" height="${PH}" fill="#f6f5ef" opacity="${hind ? .3 : .84}"/>`);
  if (!hind && !nar && T * (SNAP - d) / SNAP > 170) o.push(`<text x="${((x(d) + x(SNAP)) / 2).toFixed(1)}" y="76" text-anchor="middle" font-family="Georgia,serif" font-style="italic" font-size="14" fill="#62695d" paint-order="stroke" stroke="#fffef9" stroke-width="5">not observed yet on day ${d}</text>`);
  /* the deadline is policy, known from day 0: keep its label legible above the tint */
  o.push(`<text x="${Math.max(L + tw(dlLab, 11), x(WINDOW) - 6).toFixed(1)}" y="${nar ? 80 : 51}" text-anchor="end" font-size="11" fill="#42644d" paint-order="stroke" stroke="#fffef9" stroke-width="4">${dlLab}</text>`);
  /* snapshot boundary */
  o.push(`<rect x="${x(SNAP).toFixed(2)}" y="${PT}" width="${(w - x(SNAP)).toFixed(2)}" height="${PH}" fill="#efede2"/><line x1="${x(SNAP).toFixed(2)}" y1="${PT}" x2="${x(SNAP).toFixed(2)}" y2="${BL}" stroke="#292d26" stroke-width="1.5" stroke-dasharray="5 3"/>
    <text x="${snapX1.toFixed(1)}" y="37" text-anchor="end" font-size="11" fill="#292d26" font-family="ui-monospace,monospace">${snapLab}</text>`);
  /* events */
  const ev = [{ day: 0, row: 0, shape: 'c', label: `Purchase · ${cal.short(0)}` }];
  if (r.init != null) ev.push({ day: r.init, row: 1, shape: 'd', label: `Return started · ${cal.short(r.init)}` });
  if (r.recv != null) ev.push({ day: r.recv, row: 2, shape: 's', label: `Received · ${cal.short(r.recv)}` });
  ev.forEach((e) => {
    const seen = d >= e.day;
    if (!seen && !hind) return;
    const cx = x(e.day), ry = R(e.row), col = seen ? '#42644d' : '#8c917f';
    const lab = seen ? e.label : `${e.label} (later)`, p = place(cx, lab, 12);
    o.push(`<line x1="${cx.toFixed(2)}" y1="${BL + 8}" x2="${cx.toFixed(2)}" y2="${ry - 4}" stroke="${col}" stroke-width="1" ${seen ? '' : 'stroke-dasharray="2 3"'}/>`);
    o.push(`<text x="${p.x.toFixed(1)}" y="${ry}" text-anchor="${p.a}" font-size="12" fill="${seen ? '#292d26' : '#8c917f'}" ${seen ? '' : 'font-style="italic"'}>${lab}</text>`);
    const fill = seen ? col : '#fffef9', dash = seen ? '' : ' stroke-dasharray="2 2"';
    const m = e.shape === 'c' ? `<circle cx="${cx.toFixed(2)}" cy="${BL}" r="6" fill="${fill}" stroke="${col}" stroke-width="2"${dash}/>`
      : e.shape === 'd' ? `<polygon points="${cx},${BL - 7} ${cx + 7},${BL} ${cx},${BL + 7} ${cx - 7},${BL}" fill="${fill}" stroke="${col}" stroke-width="2"${dash}/>`
      : `<rect x="${(cx - 6).toFixed(2)}" y="${BL - 6}" width="12" height="12" fill="${fill}" stroke="${col}" stroke-width="2"${dash}/>`;
    o.push(m);
  });
  if (r.init == null && d > WINDOW) {
    const cx = x(WINDOW), ry = R(1), lab = 'Window closed · none started', p = place(cx, lab, 12);
    o.push(`<line x1="${cx.toFixed(2)}" y1="${BL + 8}" x2="${cx.toFixed(2)}" y2="${ry - 4}" stroke="#4c5a6b"/><text x="${p.x.toFixed(1)}" y="${ry}" text-anchor="${p.a}" font-size="12" fill="#292d26">${lab}</text>`);
  }
  if (v.inited && !v.got) {
    const cx = x(d), ry = R(hind && r.recv != null ? 3 : 2), lab = `No receipt yet as of day ${d}`, p = place(cx, lab, 12);
    o.push(`<line x1="${cx.toFixed(2)}" y1="${BL + 8}" x2="${cx.toFixed(2)}" y2="${ry - 4}" stroke="#956017" stroke-dasharray="3 3"/><text x="${p.x.toFixed(1)}" y="${ry}" text-anchor="${p.a}" font-size="12" fill="#956017">${lab}</text>`);
  }
  /* playhead */
  const lab = `Day ${d} · ${cal.short(d)}`, bw = lab.length * 6.6 + 20, bx = Math.max(2, Math.min(w - bw - 2, x(d) - bw / 2));
  o.push(`<line x1="${x(d).toFixed(2)}" y1="22" x2="${x(d).toFixed(2)}" y2="${BL}" stroke="#292d26" stroke-width="1.6"/>
    <rect x="${bx.toFixed(1)}" y="2" width="${bw.toFixed(1)}" height="20" rx="10" fill="#292d26"/>
    <text x="${(bx + bw / 2).toFixed(1)}" y="16" text-anchor="middle" font-size="12" fill="#fffef9" font-family="ui-monospace,monospace">${lab}</text>`);
  return { svg: o.join(''), L, T, H };
}

/* ---------- interactive view for one validated payload ---------- */
let instances = 0;

function mountView(root, M, prev) {
  const { cal, WINDOW, SNAP, records } = M;
  const byId = Object.fromEntries(records.map((r) => [r.id, r]));
  const uid = `pu${++instances}`;
  root.innerHTML = `<div class="pu">
  <p class="pu-label pu-kicker">Simulated ledger records · not a forecast</p>
  <div class="pu-fig">
    <div class="pu-recs" role="group" aria-label="Choose an example record" data-pu="recs">
      ${records.map((r) => `<button type="button" class="pu-rec" data-rec="${esc(r.id)}" aria-pressed="false"><b>${esc(r.title)}</b><span class="pu-label"><span>Record ${esc(r.letter)}</span><span>${esc(r.order)}</span></span><span class="d">${esc(r.blurb)}</span></button>`).join('')}
    </div>
    <ol class="pu-stages" aria-label="Stages of the selected purchase" data-pu="stages"></ol>

    <div class="pu-axisbox" data-pu="axisbox"><svg role="img" data-pu="axis" aria-label="Timeline"></svg></div>
    <div class="pu-legend" aria-label="Legend" data-pu="legend"></div>

    <div class="pu-scrub">
      <div class="pu-scrub-head">
        <label class="pu-day" for="${uid}-range" data-pu="dayread"></label>
        <div class="pu-status" data-pu="status"></div>
      </div>
      <input class="pu-range" id="${uid}-range" type="range" min="0" max="${SNAP}" step="1" value="${M.initial}" data-pu="range" aria-label="Day since purchase, 0 to ${SNAP}">
      <div class="pu-ctrls">
        <div class="pu-presets" role="group" aria-label="Jump to a day" data-pu="presets">
          <span class="pu-days" data-pu="days"></span>
          <span class="sep" aria-hidden="true"></span>
          <button type="button" class="pu-pill" data-step="-1" aria-label="Jump to previous event">&#8592; Previous event</button>
          <button type="button" class="pu-pill" data-step="1" aria-label="Jump to next event">Next event &#8594;</button>
        </div>
        <button type="button" class="pu-pill" data-pu="hind" aria-pressed="false">Show later events (hindsight)</button>
      </div>
    </div>
    <p class="pu-foot" data-pu="foot"></p>
    <div class="pu-sr" role="status" aria-live="polite" data-pu="live"></div>
  </div>
</div>`;

  const q = (n) => root.querySelector(`[data-pu="${n}"]`);
  const el = { recs: q('recs'), stages: q('stages'), axisbox: q('axisbox'), axis: q('axis'), legend: q('legend'), dayread: q('dayread'), status: q('status'), hind: q('hind'), range: q('range'), presets: q('presets'), days: q('days'), foot: q('foot'), live: q('live') };
  const keep = prev && byId[prev.rec];
  const state = { rec: keep ? prev.rec : records[0].id, day: keep ? prev.day : M.initial, hind: keep ? prev.hind : false };
  let geom = { L: 28, T: 800 }, liveTimer = 0, raf = 0, daysFor = null;

  const listeners = [];
  const on = (target, type, fn) => { target.addEventListener(type, fn); listeners.push([target, type, fn]); };

  function eventDays(r) { return [...new Set([0, r.init, r.recv, WINDOW, SNAP].filter((n) => n != null))].sort((a, b) => a - b); }

  function renderAxis() {
    const r = byId[state.rec], w = Math.max(280, Math.round(el.axisbox.clientWidth || 900));
    const a = axis(M, r, state.day, w, state.hind, uid);
    geom = a;
    el.axis.setAttribute('viewBox', `0 0 ${w} ${a.H}`);
    el.axis.setAttribute('width', w); el.axis.setAttribute('height', a.H);
    el.axis.innerHTML = a.svg;
    el.range.style.setProperty('--rin', `${a.L - 11}px`);
  }

  function render() {
    const r = byId[state.rec], d = state.day, v = view(M, r, d), st = stages(M, r, d);
    root.dataset.puRec = r.id; root.dataset.puDay = String(d);
    el.recs.querySelectorAll('[data-rec]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.rec === state.rec)));
    const here = (i) => (i === v.cur ? '<span class="here">Unit is here</span>' : '');
    el.stages.innerHTML = st.map((s, i) => {
      const next = st[i + 1];
      let link = 'dot';
      if (next) { if (s.st === 'observed' && next.st === 'observed') link = 'solid'; else if (s.st === 'observed' && next.st === 'open') link = 'dash'; else if (s.st === 'open' && next.st === 'open') link = 'dash'; }
      return `<li class="pu-st" data-st="${s.st}" data-link="${link}"${i === v.cur ? ' aria-current="step"' : ''}>
        <span class="pu-ic">${ICONS[i]()}</span>
        <span class="pu-tx"><span class="l1"><span class="nm">${s.name}</span><span class="tag">${s.tag}</span></span><span class="big">${s.big}</span><span class="sub">${s.sub}</span>${here(i)}</span></li>`;
    }).join('');
    if (daysFor !== r.id) {
      daysFor = r.id;
      el.days.innerHTML = eventDays(r).map((n) => `<button type="button" class="pu-pill" data-day="${n}" aria-pressed="false">Day ${n}${n === SNAP ? ' · snapshot' : ''}</button>`).join('');
    }
    el.legend.innerHTML = `<span><i style="background:#42644d"></i>observed event or stopped clock</span><span><i style="background:#f4ebd8;border:1.5px dashed #956017;height:10px"></i>clock still running, nothing observed</span>${r.storm ? '<span><i class="storm"></i>storm days</span>' : ''}`;
    el.dayread.innerHTML = `Day ${d}<small>${cal.longDate(d)}</small>`;
    el.range.value = String(d);
    el.range.setAttribute('aria-valuetext', `Day ${d}, ${cal.longDate(d)}. ${v.label}.`);
    el.presets.querySelectorAll('[data-day]').forEach((b) => b.setAttribute('aria-pressed', String(+b.dataset.day === d)));
    el.hind.setAttribute('aria-pressed', String(state.hind));
    el.axis.setAttribute('aria-label', `Timeline of days 0 to ${SNAP} for record ${r.letter}. Playhead at day ${d}. ${v.label}.`);
    el.status.innerHTML = `<span class="pu-label">Status</span><b class="pu-tone-${v.tone}">${v.label}</b><span class="pu-ctx">${ctx(M, r, d)}</span>`;
    el.foot.innerHTML = `<span class="syn">Simulated ledger</span><span>${esc(r.order)} · ${esc(r.item)}</span><span>As of ${cal.longDate(d)} (day ${d}) · snapshot ends day ${SNAP}, ${cal.short(SNAP)}; nothing after is claimed</span><span>A ledger example, not a forecast</span>`;

    renderAxis();
    clearTimeout(liveTimer);
    liveTimer = setTimeout(() => { el.live.textContent = `Record ${r.letter}, day ${d}, ${cal.longDate(d)}. ${v.label}. ${ctx(M, r, d)}.`; }, 250);
  }

  const setDay = (n) => { state.day = Math.max(0, Math.min(SNAP, Math.round(n))); render(); };
  /* choosing a record restarts at the story's opening day so the three records compare at one moment */
  on(el.recs, 'click', (e) => { const b = e.target.closest('[data-rec]'); if (b && byId[b.dataset.rec]) { state.rec = b.dataset.rec; state.day = M.initial; render(); } });
  on(el.range, 'input', () => setDay(+el.range.value));
  on(el.hind, 'click', () => { state.hind = !state.hind; render(); });
  on(el.presets, 'click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    if (b.dataset.day != null) setDay(+b.dataset.day);
    else if (b.dataset.step) {
      const ev = eventDays(byId[state.rec]);
      const t = b.dataset.step === '1' ? ev.find((n) => n > state.day) : [...ev].reverse().find((n) => n < state.day);
      if (t != null) setDay(t);
    }
  });

  /* drag/click on the timeline to scrub */
  const dayAt = (cx) => { const rect = el.axis.getBoundingClientRect(); return (cx - rect.left - geom.L) / geom.T * SNAP; };
  let drag = false;
  on(el.axis, 'pointerdown', (e) => { drag = true; try { el.axis.setPointerCapture(e.pointerId); } catch (_) { /* noop */ } setDay(dayAt(e.clientX)); });
  on(el.axis, 'pointermove', (e) => { if (drag) setDay(dayAt(e.clientX)); });
  const end = () => { drag = false; };
  on(el.axis, 'pointerup', end);
  on(el.axis, 'pointercancel', end);

  let ro = null;
  if (typeof ResizeObserver !== 'undefined') {
    let lastW = 0;
    ro = new ResizeObserver(() => { const w = el.axisbox.clientWidth; if (w && w !== lastW) { lastW = w; cancelAnimationFrame(raf); raf = requestAnimationFrame(renderAxis); } });
    ro.observe(el.axisbox);
  }

  render();

  return {
    state,
    cleanup() {
      if (ro) ro.disconnect();
      cancelAnimationFrame(raf); clearTimeout(liveTimer);
      listeners.forEach(([t, type, fn]) => t.removeEventListener(type, fn));
      listeners.length = 0;
      root.innerHTML = '';
    },
  };
}

export default {
  render({ model, el }) {
    const host = document.createElement('div');
    host.className = 'pu-host';
    el.appendChild(host);
    let live = null;

    const message = (text) => {
      if (live) { live.cleanup(); live = null; }
      host.innerHTML = `<div class="pu"><p class="pu-msg" role="alert">${esc(text)}</p></div>`;
    };
    const sync = () => {
      let M;
      try { M = buildModel(model.get('data')); } catch (err) { message(err.message); return; }
      const prev = live ? { ...live.state } : null;
      if (live) { live.cleanup(); live = null; }
      live = mountView(host, M, prev);
    };

    model.on('change:data', sync);
    sync();

    return () => {
      model.off('change:data', sync);
      if (live) { live.cleanup(); live = null; }
      host.remove();
    };
  },
};
