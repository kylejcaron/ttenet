// Chain stepper: Sales -> Return Initiation -> Receipt -> Forecast, one chapter at a time.
// Every number drawn here comes from the notebook's `data` value: timing curves replay the fitted
// stages' own timing laws, and the final panel is the forecast's own predictive summary. This file
// holds layout and wording only.

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const FIG_TAGS = ["forecast", "fitted timing \u00b7 hypothetical", "fitted timing \u00b7 hypothetical", "forecast"];
const NAVS = ["Sales", "Return Initiation", "Receipts", "Forecast"];
const HINT = "Select a node in the diagram to read what it does. Escape clears the selection.";

// Container width at which the text and figure sit side by side.
const WIDE_FROM = 760;
// Plot width under which the figure switches to the taller, larger-text geometry.
const TALL_BELOW = 600;

const XL = 58;
const XR = 626;
const W = XR - XL;
// Plot rows; set by buildSvg for the geometry being drawn.
const geo = { tall: false, PT: 226, PH: 106, PB: 332 };

let instances = 0;

/* -------------------------------------------------------------- helpers */
function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}
const f1 = (v) => v.toFixed(1);
const pct = (p, d) => (p * 100).toFixed(d == null ? 1 : d) + "%";
const pctTail = (p) => (p < 1e-4 ? "under 0.01%" : pct(p, p < 0.01 ? 2 : 1));
const pctRange = (b) => pct(b.mean) + " (90% band " + pct(b.lo90) + "\u2013" + pct(b.hi90) + ")";
const fmtCount = (v) => (Math.abs(v) >= 100 ? Math.round(v).toLocaleString("en-US") : v.toFixed(1));
const countRange = (b) => fmtCount(b.mean) + " (90% interval " + fmtCount(b.lo90) + "\u2013" + fmtCount(b.hi90) + ")";
const path = (pts) => "M" + pts.map((p) => f1(p[0]) + " " + f1(p[1])).join(" L");
function isoParts(iso) {
  const p = iso.split("-");
  return { y: +p[0], m: +p[1], d: +p[2] };
}
function dow(iso) {
  const p = isoParts(iso);
  return new Date(Date.UTC(p.y, p.m - 1, p.d)).getUTCDay();
}
function shortDate(iso) {
  const p = isoParts(iso);
  return MONTHS[p.m - 1] + " " + p.d;
}
const longDate = (iso) => WEEKDAYS[dow(iso)] + " " + shortDate(iso);
const maxOf = (a) => a.reduce((m, v) => (v > m ? v : m), -Infinity);
function argmax(a) {
  let k = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[k]) k = i;
  return k;
}

// y axis: first 1/2/5 x 10^k step that needs at most 6 intervals
function niceAxis(max) {
  const top = Math.max(max, 1e-9);
  for (let e = -4; ; e++) {
    for (const mult of [1, 2, 5]) {
      const step = mult * Math.pow(10, e);
      const n = Math.ceil(top / step - 1e-9);
      if (n <= 6) {
        const ticks = [];
        for (let i = 0; i <= n; i++) ticks.push(i * step);
        return { step, n, top: Math.max(n, 1) * step, ticks };
      }
    }
  }
}
const fmtTick = (v, step) => v.toFixed(step >= 1 ? 0 : Math.ceil(-Math.log10(step) - 1e-9));

/* ---------------------------------------------------------------- model */
function prepare(d) {
  const N = d.datesISO.length;
  const init = d.initiation;
  const recv = d.receipt;
  const policy = init.policy_days;
  const recvLast = recv.ages[recv.ages.length - 1];
  const stormDays = d.storm ? d.storm.reduce((n, s) => n + (s ? 1 : 0), 0) : 0;
  const weekendZero = d.weekend.every((w, i) => !w || d.receipt_forecast.hi90[i] === 0);
  const policyTxt = policy == null ? "with no deadline" : "only through day " + policy;
  const weibull = recv.family === "WeibullFamily";
  const m = { d, N, init, recv, policy, recvLast, stormDays };

  m.caps = [
    "Predictive daily units sold over the next " + N + " days: bars are the mean, the thick whisker the 50% interval and the thin whisker the 90% interval. Expected total " + countRange(d.totals.sales) + ". A forecast: no actual sales are drawn.",
    "Daily probability, out of all purchases; bars (posterior mean) sum to " + pctRange(init.total) + ". The rest never start: " + pctRange(init.never) + " are never-return units, and " +
      pctRange(init.beyond) + " are susceptible but " + (policy == null ? "still waiting past the plotted days" : "run out the " + policy + "-day window") + ". Hypothetical setting (product A, neutral weekday), not a calendar forecast and not conditioned on being susceptible.",
    "Daily probability, out of all initiated returns; bars (posterior mean) sum to " + pctRange(recv.total) + ". The other " + pctRange(recv.never) + " never arrive" +
      (recv.beyond.mean > 0 ? ", and " + pctTail(recv.beyond.mean) + " would land after day " + recvLast : "") + ". Hypothetical setting (clear weather, every day open); weekends and storms enter in step 4. No receipt deadline.",
    "Predictive daily receipts, " + N + " days from " + shortDate(d.datesISO[0]) + ": line is the mean, bands the 50% and 90% predictive intervals. Expected total " + countRange(d.totals.receipt_forecast) + ". " +
      (stormDays ? "Shading marks the supplied storm days (a weather input, not a forecast). " : "") +
      (weekendZero ? "Weekends are exactly zero in every draw." : ""),
  ];

  m.nodes = {
    sales: ["Sales", "The feed. Predictive daily units sold set how many units are in play. Sales are modelled and forecast as counts, so this layer carries its own uncertainty."],
    initiation: ["Return Initiation", "Susceptible units start a return some days after purchase, on a flexible age pattern fitted to the data, " + policyTxt + ". This is forecast 2."],
    never: ["Never-return units", "These units never start a return at any age. Recent data cannot tell them from units that simply have not returned yet, so their share and the timing are estimated together."],
    receipt: ["Receipt", "A second clock runs from initiation to receipt: " + (weibull ? "a Weibull delay" : "a fitted delay law") + ", open days only, slowed by storms, with no deadline. This is forecast 3."],
    latent: ["Never arrives", "Some started returns never arrive. That outcome is latent: a missing receipt is only evidence, never proof, because the parcel may still be on its way."],
  };

  m.beats = [
    ["Start with what was sold",
      ["Every return begins as a sale, so the model first forecasts daily sales. That says how many units are in play for the next stage."],
      "sales(t) \u2192 units at risk"],
    ["Not every unit ever returns",
      ["Each unit is either never-return or susceptible. Only susceptible units start a return, at some age after purchase" + (policy == null ? "." : ", and only through day " + policy + ".") + " Quiet so far looks the same for both, so share and timing are estimated together.",
        "Each bar is a probability out of all purchases, not a hazard: susceptible share \u00d7 still waiting \u00d7 that day\u2019s hazard. Bars cover " + pct(init.total.mean) + "; the rest never start or run out the window."],
      "P(start on day a) = susceptible \u00d7 still waiting \u00d7 hazard(a)" + (policy == null ? "" : ", a = 0\u2013" + policy)],
    ["Started is not received",
      ["A started return begins a second clock that ends when the parcel is received. Unlike initiation, it has no deadline.",
        "Each bar is the share of all started returns received that many days after initiation, " + pct(recv.total.mean) + " in total. The other " + pct(recv.never.mean) + " never arrive, and a missing receipt alone is weak evidence, not proof of abandonment."],
      "P(received d days after start) = receivable \u00d7 still waiting \u00d7 hazard(d)"],
    ["The forecast carries the whole chain",
      ["Sales feed initiation, initiation feeds receipt, and closed days carry no arrivals: the fitted receipt law gives weekends zero and storms a slower rate, and parcels not yet received stay in the pipeline for later open days.",
        "Each stage passes a distribution, not a number, to the next, so the result is one band of plausible daily receipts. The bands are the forecast\u2019s own predictive intervals."],
      "receipts(t) = sales \u2192 initiation timing \u2192 receipt timing, on the open calendar"],
  ];
  return m;
}

/* -------------------------------------------------------------- svg build */
function yAxis(ax, label) {
  const { PT, PB, PH, tall } = geo;
  return ax.ticks.map((v) => {
    const y = PB - (v / ax.top) * PH;
    return (v > 0 ? '<line class="cm-grid" x1="' + XL + '" x2="' + XR + '" y1="' + f1(y) + '" y2="' + f1(y) + '"/>' : "") +
      '<text class="t-xs mut" x="' + (XL - 6) + '" y="' + f1(y + 4) + '" text-anchor="end">' + fmtTick(v, ax.step) + "</text>";
  }).join("") + '<text class="t-xs mut" x="14" y="' + (PT - (tall ? 35 : 10)) + '">' + esc(label) + "</text>";
}
function xTick(x, s, cls) {
  return '<text class="t-xs ' + (cls || "mut") + '" x="' + f1(x) + '" y="' + (geo.PB + (geo.tall ? 36 : 17)) + '" text-anchor="middle">' + esc(s) + "</text>";
}
function xLabel(s) {
  return '<text class="t-xs mut" x="' + (XL + XR) / 2 + '" y="' + (geo.PB + (geo.tall ? 72 : 37)) + '" text-anchor="middle">' + esc(s) + "</text>";
}
function note(x, y, s, cls, anchor) {
  return '<text class="cm-note t-xs halo ' + (cls || "mut") + '" x="' + f1(x) + '" y="' + f1(y) + '" text-anchor="' + (anchor || "start") + '">' + esc(s) + "</text>";
}
function bars(vals, ax, x0, bw, peak, cls, tip) {
  const { PB, PH } = geo;
  return '<g class="bars ' + cls + '">' + vals.map((v, i) => {
    const h = (v / ax.top) * PH;
    return '<rect class="tip' + (i === peak ? " pk" : "") + '" x="' + f1(x0(i)) + '" y="' + f1(PB - h) + '" width="' + f1(bw) + '" height="' + f1(h) + '" rx="1"><title>' + esc(tip(i)) + "</title></rect>";
  }).join("") + "</g>";
}
function stepBand(lo, hi, ax, slot, cls) {
  const { PB, PH } = geo;
  const up = [];
  const down = [];
  for (let i = 0; i < lo.length; i++) {
    const yh = PB - (hi[i] / ax.top) * PH;
    up.push([slot(i), yh], [slot(i + 1), yh]);
  }
  for (let j = lo.length - 1; j >= 0; j--) {
    const yl = PB - (lo[j] / ax.top) * PH;
    down.push([slot(j + 1), yl], [slot(j), yl]);
  }
  return '<path class="band ' + cls + '" d="' + path(up.concat(down)) + ' Z"/>';
}
function legend(items) {
  return '<g class="lg" transform="translate(14 ' + (geo.tall ? 262 : 193) + ')">' + items.map((it) => '<g class="lg-item">' + it + "</g>").join("") + "</g>";
}
const LG = {
  bar: (label, color) => '<rect x="0" y="-9" width="18" height="10" fill="' + color + '" fill-opacity=".5"/><text class="t-xs mut" x="24" y="0">' + label + "</text>",
  band: (label, color, op) => '<rect x="0" y="-9" width="18" height="10" fill="' + color + '" fill-opacity="' + op + '"/><text class="t-xs mut" x="24" y="0">' + label + "</text>",
  line: (label) => '<line x1="0" x2="18" y1="-4" y2="-4" stroke="#4c5a6b" stroke-width="2.4"/><text class="t-xs mut" x="24" y="0">' + label + "</text>",
  w50: (label) => '<line x1="0" x2="18" y1="-4" y2="-4" stroke="#292d26" stroke-opacity=".6" stroke-width="4"/><text class="t-xs mut" x="24" y="0">' + label + "</text>",
  w90: (label) => '<line x1="0" x2="18" y1="-4" y2="-4" stroke="#292d26" stroke-opacity=".5" stroke-width="1.4"/><text class="t-xs mut" x="24" y="0">' + label + "</text>",
  dot: (label) => '<circle cx="6" cy="-4" r="3" fill="#f7f7ef" stroke="#4c5a6b" stroke-width="1.6"/><text class="t-xs mut" x="18" y="0">' + label + "</text>",
};
function plot(show, title, sub, body) {
  const { PT, PB, tall } = geo;
  return '<g class="cm-layer" data-plot="' + show + '" data-show="' + show + '"><text class="t-xs lbl" x="14" y="' + (tall ? 228 : 176) + '">' + esc(title) + "</text>" + sub +
    '<line class="axis" x1="' + XL + '" x2="' + XR + '" y1="' + PB + '" y2="' + PB + '"/><line class="axis" x1="' + XL + '" x2="' + XL + '" y1="' + PT + '" y2="' + PB + '"/>' + body + "</g>";
}
function nodeG(id, o) {
  return '<g class="cm-layer cm-node" data-node="' + id + '" data-show="' + o.show + '"' + (o.emph ? ' data-emph="' + o.emph + '"' : "") + ' role="button" tabindex="-1" aria-pressed="false" aria-label="' + o.label + '">' +
    '<rect class="hit" x="' + o.hit[0] + '" y="' + o.hit[1] + '" width="' + o.hit[2] + '" height="' + o.hit[3] + '" rx="10"/>' + o.body + "</g>";
}
function box(x, y, w, h, title, sub, big, cls) {
  const tall = geo.tall;
  const cx = x + w / 2;
  const lines = Array.isArray(title) ? title : [title];
  const t = lines.length === 1 ? lines[0] : lines.map((s, i) => '<tspan x="' + cx + '" dy="' + (i ? (tall ? 28 : 22) : 0) + '">' + s + (i < lines.length - 1 ? " " : "") + "</tspan>").join("");
  let ty = y + (big ? (lines.length > 1 ? 28 : 38) : 21);
  let sy = y + (big ? (lines.length > 1 ? 69 : 60) : 39);
  if (tall) {
    h = big ? 98 : 66;
    ty = y + (big ? (lines.length > 1 ? 28 : 36) : 27);
    sy = y + (big ? (lines.length > 1 ? 86 : 72) : 57);
  }
  return '<rect class="nb ' + (cls || "") + '" x="' + x + '" y="' + y + '" width="' + w + '" height="' + h + '" rx="7"/>' +
    '<text class="' + (big ? "t-l" : "t-m") + ' serif" x="' + cx + '" y="' + ty + '" text-anchor="middle">' + t + "</text>" +
    '<text class="t-ns mut" x="' + cx + '" y="' + sy + '" text-anchor="middle">' + sub + "</text>";
}

function dayTicks(m) {
  const dates = m.d.datesISO;
  const DW = W / m.N;
  const letters = dates.map((iso, i) => {
    const k = dow(iso);
    return xTick(XL + (i + 0.5) * DW, "SMTWTFS"[k], "cm-weekday " + (k === 0 || k === 6 ? "wk" : "mut"));
  }).join("");
  const labels = dates.map((iso, i) => {
    if (i !== 0 && dow(iso) !== 1) return "";
    const x = Math.min(Math.max(XL + (i + 0.5) * DW, XL + 18), XR - 18);
    return '<text class="t-xs mut" x="' + f1(x) + '" y="' + (geo.PB + 35) + '" text-anchor="middle">' + shortDate(iso) + "</text>";
  }).join("");
  return letters + labels;
}

function salesPlot(m) {
  const { PB, PH } = geo;
  const d = m.d;
  const s = d.sales;
  const N = m.N;
  const DW = W / N;
  const ax = niceAxis(maxOf(s.hi90));
  const yv = (v) => PB - (v / ax.top) * PH;
  const dayX = (i) => XL + (i + 0.5) * DW;
  const bw = Math.min(14, DW * 0.58);
  const whiskers = s.mean.map((_, i) => {
    const x = f1(dayX(i));
    return '<line class="w90" x1="' + x + '" x2="' + x + '" y1="' + f1(yv(s.lo90[i])) + '" y2="' + f1(yv(s.hi90[i])) + '"/>' +
      '<line class="w50" x1="' + x + '" x2="' + x + '" y1="' + f1(yv(s.lo50[i])) + '" y2="' + f1(yv(s.hi50[i])) + '"/>';
  }).join("");
  return plot("1", "Sales \u00b7 next " + N + " days \u00b7 forecast",
    legend([LG.bar("mean", "#42644d"), LG.w50("50%"), LG.w90("90%")]),
    yAxis(ax, "units sold per day") +
    bars(s.mean, ax, (i) => dayX(i) - bw / 2, bw, -1, "", (i) =>
      longDate(d.datesISO[i]) + ": mean " + fmtCount(s.mean[i]) + " units (50% " + fmtCount(s.lo50[i]) + "\u2013" + fmtCount(s.hi50[i]) + ", 90% " + fmtCount(s.lo90[i]) + "\u2013" + fmtCount(s.hi90[i]) + ")") +
    whiskers + dayTicks(m));
}

function timingPlot(show, title, st, color, cls, slot, ticks, xTitle, peakWord, extra) {
  const { PT, tall } = geo;
  const hi = st.prob.hi90.map((v) => v * 100);
  const lo = st.prob.lo90.map((v) => v * 100);
  const mean = st.prob.mean.map((v) => v * 100);
  const ax = niceAxis(maxOf(hi));
  const A = st.ages.length;
  const aw = W / A;
  const peak = argmax(mean);
  const peakX = slot(peak + 1) + 10;
  const late = peak > A * 0.4;
  const peakNote = note(late ? slot(peak) - 8 : peakX, PT + (tall ? 27 : 13), "most likely " + peakWord + ": day " + st.ages[peak] + " \u00b7 " + mean[peak].toFixed(2) + "%", "mut", late ? "end" : "start");
  const tickEls = ticks.map((a) => xTick(slot(st.ages.indexOf(a) + 0.5), a)).join("");
  return plot(show, title, legend([LG.bar("mean", color), LG.band("90% band", color, 0.16)]),
    yAxis(ax, "daily probability (%)") +
    extra.under +
    stepBand(lo, hi, ax, slot, cls) +
    bars(mean, ax, (i) => slot(i) + aw * 0.12, aw * 0.76, peak, cls, (i) =>
      xTitle.tip + " " + st.ages[i] + ": " + mean[i].toFixed(2) + "% (90% band " + lo[i].toFixed(2) + "\u2013" + hi[i].toFixed(2) + "%)") +
    extra.over + peakNote + tickEls + xLabel(xTitle.axis));
}

function initiationPlot(m) {
  const { PT, PB, PH, tall } = geo;
  const st = m.init;
  const A = st.ages.length;
  const aw = W / A;
  const slot = (a) => XL + a * aw;
  let under = "";
  let over = "";
  if (m.policy != null && m.policy + 1 < A) {
    const bx = slot(m.policy + 1);
    under = '<rect x="' + f1(bx) + '" y="' + PT + '" width="' + f1(XR - bx) + '" height="' + PH + '" fill="#956017" fill-opacity=".1"/>';
    over = '<line x1="' + f1(bx) + '" x2="' + f1(bx) + '" y1="' + PT + '" y2="' + PB + '" stroke="#956017" stroke-width="2" stroke-dasharray="5 3"/>' +
      note(bx - 6, PT + (tall ? 58 : 31), "no starts after day " + m.policy, "amber", "end");
  }
  const last = st.ages[A - 1];
  const ticks = [];
  for (let a = 0; a <= last; a += 15) ticks.push(a);
  return timingPlot("2", "Return Initiation \u00b7 all purchases", st, "#42644d", "", slot, ticks,
    { axis: "days since purchase", tip: "Starts on day" }, "start", { under, over });
}

function receiptPlot(m) {
  const st = m.recv;
  const A = st.ages.length;
  const aw = W / A;
  const slot = (a) => XL + a * aw;
  const last = st.ages[A - 1];
  const step = last <= 30 ? 5 : 10;
  const ticks = [];
  for (let a = 0; a <= last; a += step) ticks.push(a);
  const over = st.beyond.mean > 0 ? note(XR - 4, geo.PB - 8, "after day " + m.recvLast + ": " + pctTail(st.beyond.mean), "mut", "end") : "";
  return timingPlot("3", "Receipt \u00b7 all initiated returns", st, "#4c5a6b", "slate", slot, ticks,
    { axis: "days since return initiation", tip: "Received on day" }, "delay", { under: "", over });
}

function forecastPlot(m) {
  const { PT, PB, PH } = geo;
  const d = m.d;
  const r = d.receipt_forecast;
  const N = m.N;
  const DW = W / N;
  const ax = niceAxis(maxOf(r.hi90));
  const yv = (v) => PB - (v / ax.top) * PH;
  const dayX = (i) => XL + (i + 0.5) * DW;
  let storm = "";
  let run = -1;
  if (d.storm) {
    for (let i = 0; i <= N; i++) {
      const on = i < N && d.storm[i];
      if (on && run < 0) run = i;
      if (!on && run >= 0) {
        storm += '<rect x="' + f1(XL + run * DW) + '" y="' + PT + '" width="' + f1((i - run) * DW) + '" height="' + PH + '" fill="#956017" fill-opacity=".13"/>';
        run = -1;
      }
    }
  }
  const poly = (hi, lo) => path(hi.map((v, k) => [dayX(k), yv(v)]).concat(lo.map((v, k) => [dayX(k), yv(v)]).reverse())) + " Z";
  const dots = d.weekend.map((w, k) =>
    w ? '<circle cx="' + f1(dayX(k)) + '" cy="' + f1(yv(r.mean[k])) + '" r="3" fill="#f7f7ef" stroke="#4c5a6b" stroke-width="1.6"/>' : "").join("");
  const tips = d.datesISO.map((iso, k) => {
    const txt = longDate(iso) + (d.weekend[k] ? " (weekend)" : "") + ": mean " + fmtCount(r.mean[k]) + " received (50% " + fmtCount(r.lo50[k]) + "\u2013" + fmtCount(r.hi50[k]) + ", 90% " + fmtCount(r.lo90[k]) + "\u2013" + fmtCount(r.hi90[k]) + ")" + (d.storm && d.storm[k] ? ", storm day" : "");
    return '<rect class="tip" x="' + f1(XL + k * DW) + '" y="' + PT + '" width="' + f1(DW) + '" height="' + PH + '" fill="#fff" fill-opacity="0"><title>' + esc(txt) + "</title></rect>";
  }).join("");
  const items = [LG.line("mean"), LG.band("50%", "#4c5a6b", 0.3), LG.band("90%", "#4c5a6b", 0.16)];
  if (m.stormDays) items.push(LG.band("storm", "#956017", 0.18));
  if (d.weekend.some(Boolean)) items.push(LG.dot("weekend"));
  return plot("4", "Receipts \u00b7 next " + N + " days \u00b7 forecast", legend(items),
    yAxis(ax, "units received per day") + storm +
    '<path class="band slate" d="' + poly(r.hi90, r.lo90) + '"/>' +
    '<path d="' + poly(r.hi50, r.lo50) + '" fill="#4c5a6b" fill-opacity=".3"/>' +
    '<path class="draw" pathLength="1" d="' + path(r.mean.map((v, k) => [dayX(k), yv(v)])) + '" fill="none" stroke="#4c5a6b" stroke-width="2.4" stroke-linejoin="round"/>' +
    dots + tips + dayTicks(m));
}

function buildSvg(m, tall, uid) {
  geo.tall = tall;
  geo.PT = tall ? 326 : 226;
  geo.PH = tall ? 208 : 106;
  geo.PB = geo.PT + geo.PH;
  const hitH = tall ? 110 : 88;
  const lowY = tall ? 104 : 88;
  const lowTo = tall ? 120 : 108;
  const branchY = tall ? 116 : 104;
  const branchH = tall ? 78 : 54;
  const branchBoxY = tall ? 122 : 110;
  return '<svg class="cm-svg' + (tall ? " cm-tall" : "") + '" viewBox="0 0 640 ' + (tall ? 632 : 374) + '" role="group" aria-labelledby="' + uid + '-t ' + uid + '-d" preserveAspectRatio="xMidYMid meet">' +
    '<title id="' + uid + '-t">Sales to return initiation to receipt</title>' +
    '<desc id="' + uid + '-d">' + esc("Four chapters of the fitted chain: forecast daily sales; the daily-bin probability of starting a return" + (m.policy == null ? "" : " under a " + m.policy + "-day policy") +
      " (hypothetical setting); the daily-bin probability of receipt after initiation (hypothetical setting); and the " + m.N + "-day receipt forecast with 50% and 90% predictive bands.") + "</desc>" +
    '<defs><marker id="' + uid + '-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 1 L9 5 L0 9z" fill="#292d26" fill-opacity=".62"/></marker></defs>' +
    '<path class="conn" d="M182 46 H231" marker-end="url(#' + uid + '-arr)"/>' +
    '<path class="conn" d="M407 46 H456" marker-end="url(#' + uid + '-arr)"/>' +
    nodeG("sales", { show: "1,2,3,4", emph: "1,4", label: "Sales node", hit: [4, 2, 182, hitH], body: box(10, 6, 170, 80, "Sales", "1 \u00b7 units sold", true) }) +
    nodeG("initiation", { show: "1,2,3,4", emph: "2,4", label: "Return Initiation node", hit: [229, 2, 182, hitH], body: box(235, 6, 170, 80, ["Return", "Initiation"], "2 \u00b7 starts", true) }) +
    nodeG("receipt", { show: "1,2,3,4", emph: "3,4", label: "Receipt node", hit: [454, 2, 182, hitH], body: box(460, 6, 170, 80, "Receipt", "3 \u00b7 receipts", true) }) +
    '<g class="cm-layer" data-show="2"><path class="conn dash" d="M320 ' + lowY + " V" + lowTo + '" marker-end="url(#' + uid + '-arr)"/></g>' +
    '<g class="cm-layer" data-show="3"><path class="conn dash" d="M545 ' + lowY + " V" + lowTo + '" marker-end="url(#' + uid + '-arr)"/></g>' +
    nodeG("never", { show: "2", label: "Never-return units branch", hit: [229, branchY, 182, branchH], body: box(235, branchBoxY, 170, 46, "Never-return", "never starts", false, "branch") }) +
    nodeG("latent", { show: "3", label: "Never arrives branch", hit: [454, branchY, 182, branchH], body: box(460, branchBoxY, 170, 46, "Never arrives", "never received", false, "branch latent") }) +
    salesPlot(m) + initiationPlot(m) + receiptPlot(m) + forecastPlot(m) + "</svg>";
}

/* ------------------------------------------------------------ html build */
function buildChapters(m, uid) {
  return m.beats.map((b, i) =>
    '<article class="cm-ch" data-ch="' + (i + 1) + '" aria-labelledby="' + uid + "-h" + (i + 1) + '">' +
    '<p class="cm-num">0' + (i + 1) + " / 0" + m.beats.length + " \u00b7 " + esc(NAVS[i].toUpperCase()) + "</p>" +
    '<h3 class="cm-h" id="' + uid + "-h" + (i + 1) + '">' + esc(b[0]) + "</h3>" +
    b[1].map((p) => "<p>" + esc(p) + "</p>").join("") +
    '<p class="cm-mech">' + esc(b[2]) + "</p>" +
    '<p class="cm-cap">' + esc(m.caps[i]) + "</p></article>").join("");
}

function buildReadouts(m) {
  const hint = '<div class="cm-ro" data-ro=""><span class="cm-label cm-rk">Nodes</span><p class="cm-rx">' + esc(HINT) + "</p></div>";
  return hint + Object.keys(m.nodes).map((id) =>
    '<div class="cm-ro" data-ro="' + id + '"><span class="cm-label cm-rk">Node</span><h4 class="cm-rt">' + esc(m.nodes[id][0]) + '</h4><p class="cm-rx">' + esc(m.nodes[id][1]) + "</p></div>").join("");
}

const list = (s) => (s ? s.split(",").map(Number) : []);

/* ----------------------------------------------------------------- render */
function render({ model, el }) {
  const uid = "cm" + instances++;
  const disposers = [];
  const state = { step: 1, pinned: null, hover: null, focus: null, tall: false };
  let m = null;
  let frame = 0;
  let observer = null;
  let done = false;

  function listen(target, type, fn, opts) {
    target.addEventListener(type, fn, opts);
    disposers.push(() => target.removeEventListener(type, fn, opts));
  }

  el.classList.add("cm-root");
  el.innerHTML =
    '<div class="cm-nav">' +
    '<ol class="cm-chips" aria-label="Chapters">' +
    NAVS.map((nav, i) =>
      '<li><button type="button" class="cm-chip" data-chip="' + (i + 1) + '" aria-pressed="false" aria-label="Chapter ' + (i + 1) + ": " + nav + '"><b>0' + (i + 1) + "</b><span>" + nav + "</span></button></li>").join("") +
    "</ol>" +
    '<div class="cm-ctl"><span class="cm-stepno" aria-live="polite"></span>' +
    '<button type="button" class="cm-btn" data-dir="-1">Previous</button><button type="button" class="cm-btn" data-dir="1">Next</button></div>' +
    '<span class="cm-stagename"></span>' +
    "</div>" +
    '<div class="cm-body">' +
    '<div class="cm-text"></div>' +
    '<figure class="cm-fig"><div class="cm-figtop"><span class="cm-label">The chain</span><span class="cm-label cm-figtag"></span></div>' +
    '<div class="cm-plot"></div>' +
    '<div class="cm-readout" aria-live="polite"><div class="cm-ros"></div><div class="cm-r-foot"><button type="button" class="cm-clear" hidden>Clear node</button></div></div></figure>' +
    "</div>";

  const q = (s) => el.querySelector(s);
  const shell = {
    stepno: q(".cm-stepno"),
    stagename: q(".cm-stagename"),
    text: q(".cm-text"),
    figtag: q(".cm-figtag"),
    plot: q(".cm-plot"),
    ros: q(".cm-ros"),
    clear: q(".cm-clear"),
    chips: Array.from(el.querySelectorAll("[data-chip]")),
    dirs: Array.from(el.querySelectorAll("[data-dir]")),
    svg: null, layers: [], nodes: [], chapters: [], readouts: [],
  };

  function visibleNode(id, step) {
    const g = shell.svg.querySelector('[data-node="' + id + '"]');
    return !!g && list(g.dataset.show).indexOf(step) >= 0;
  }

  // The drawing is rebuilt on data or geometry changes; keep keyboard focus on the node's twin.
  function buildPlot() {
    const active = el.getRootNode().activeElement;
    const holder = active && el.contains(active) && active.closest ? active.closest("[data-node]") : null;
    const keep = holder ? holder.dataset.node : null;
    shell.plot.innerHTML = buildSvg(m, state.tall, uid);
    shell.svg = shell.plot.querySelector(".cm-svg");
    shell.layers = Array.from(shell.svg.querySelectorAll(".cm-layer"));
    shell.nodes = Array.from(shell.svg.querySelectorAll('.cm-node[role="button"]'));
    applyScale();
    paint();
    if (keep) {
      const twin = shell.svg.querySelector('[data-node="' + keep + '"]');
      if (twin) twin.focus({ preventScroll: true });
    }
  }

  // Tall geometry draws text in user units sized so it lands near 13.5px on screen at any plot width.
  function applyScale() {
    if (!state.tall) {
      shell.svg.style.removeProperty("--cm-fs");
    } else {
      const scale = shell.plot.clientWidth / 640;
      shell.svg.style.setProperty("--cm-fs", Math.min(26, Math.max(14, 13.5 / scale)).toFixed(1));
    }
    layoutLegends();
    fitNotes();
  }

  function buildData() {
    shell.text.innerHTML = buildChapters(m, uid);
    shell.ros.innerHTML = buildReadouts(m);
    shell.chapters = Array.from(shell.text.querySelectorAll(".cm-ch"));
    shell.readouts = Array.from(shell.ros.querySelectorAll(".cm-ro"));
    buildPlot();
  }

  // legend items sit side by side at the width their labels actually take
  function layoutLegends() {
    shell.svg.querySelectorAll(".lg").forEach((lg) => {
      let x = 0;
      lg.querySelectorAll(".lg-item").forEach((item) => {
        let w = 0;
        try { w = item.getBBox().width; } catch (err) { w = 0; }
        item.setAttribute("transform", "translate(" + f1(x) + " 0)");
        x += (w || 90) + 18;
      });
    });
  }

  // in-plot notes are measured, then nudged so none runs off the drawing
  function fitNotes() {
    shell.svg.querySelectorAll(".cm-note").forEach((t) => {
      let b;
      t.removeAttribute("transform");
      try { b = t.getBBox(); } catch (err) { return; }
      if (!b.width) return;
      let dx = 0;
      if (b.x + b.width > XR + 8) dx = XR + 8 - (b.x + b.width);
      if (b.x + dx < XL + 4) dx = XL + 4 - b.x;
      if (dx) t.setAttribute("transform", "translate(" + f1(dx) + " 0)");
    });
  }

  function paint() {
    if (!m || !shell.svg) return;
    const s = state.step;
    ["hover", "focus", "pinned"].forEach((k) => {
      if (state[k] && !visibleNode(state[k], s)) state[k] = null;
    });
    const shown = state.hover || state.focus || state.pinned || "";
    el.dataset.cmStep = String(s);
    el.dataset.cmNode = state.pinned || "";
    shell.stepno.innerHTML = "Step <b>" + s + "</b> of " + NAVS.length;
    shell.stagename.textContent = NAVS[s - 1];
    shell.figtag.textContent = FIG_TAGS[s - 1];
    shell.chips.forEach((b) => {
      const on = Number(b.dataset.chip) === s;
      b.setAttribute("aria-pressed", String(on));
      if (on) b.setAttribute("aria-current", "step"); else b.removeAttribute("aria-current");
    });
    shell.dirs.forEach((b) => {
      const to = s + Number(b.dataset.dir);
      b.setAttribute("aria-disabled", String(to < 1 || to > NAVS.length));
    });
    shell.chapters.forEach((c) => {
      const on = Number(c.dataset.ch) === s;
      c.classList.toggle("on", on);
      c.setAttribute("aria-hidden", String(!on));
      if (on) c.removeAttribute("inert"); else c.setAttribute("inert", "");
    });
    shell.layers.forEach((g) => {
      const show = list(g.dataset.show).indexOf(s) >= 0;
      g.classList.toggle("on", show);
      g.classList.toggle("emph", list(g.dataset.emph).indexOf(s) >= 0);
      g.setAttribute("aria-hidden", String(!show));
      if (g.matches('[role="button"]')) g.setAttribute("tabindex", show ? "0" : "-1");
    });
    shell.nodes.forEach((g) => {
      const sel = state.pinned === g.dataset.node;
      g.classList.toggle("sel", sel);
      g.setAttribute("aria-pressed", String(sel));
    });
    shell.readouts.forEach((r) => {
      const on = r.dataset.ro === shown;
      r.classList.toggle("on", on);
      if (on && shown) r.querySelector(".cm-rk").textContent = state.pinned === shown ? "Selected node" : "Node";
    });
    el.dataset.cmMode = shown ? "node" : "stage";
    shell.clear.hidden = !state.pinned;
    // a node that leaves the figure cannot keep keyboard focus
    const active = el.getRootNode().activeElement;
    if (active && el.contains(active) && active.matches('[data-node]') && !visibleNode(active.dataset.node, s)) {
      shell.chips[s - 1].focus({ preventScroll: true });
    }
  }

  function go(n) {
    const to = Math.min(Math.max(n, 1), NAVS.length);
    if (to === state.step) return;
    state.step = to;
    paint();
  }
  function pin(id) {
    state.pinned = id;
    paint();
  }

  const nodeOf = (t) => (t && t.closest ? t.closest("[data-node]") : null);
  const usable = (g) => !!g && g.classList.contains("on");

  shell.chips.forEach((b) => listen(b, "click", () => go(Number(b.dataset.chip))));
  shell.dirs.forEach((b) => listen(b, "click", () => go(state.step + Number(b.dataset.dir))));
  listen(shell.clear, "click", () => {
    pin(null);
    shell.chips[state.step - 1].focus({ preventScroll: true });
  });
  listen(el, "keydown", (e) => {
    if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      e.preventDefault();
      go(state.step + (e.key === "ArrowRight" ? 1 : -1));
    } else if (e.key === "Escape" && state.pinned) {
      pin(null);
    }
  });
  listen(shell.plot, "click", (e) => {
    const g = nodeOf(e.target);
    if (usable(g)) pin(state.pinned === g.dataset.node ? null : g.dataset.node);
  });
  listen(shell.plot, "keydown", (e) => {
    const g = e.target.closest && e.target.closest('[role="button"]');
    if (!g || !(e.key === "Enter" || e.key === " ")) return;
    e.preventDefault();
    pin(state.pinned === g.dataset.node ? null : g.dataset.node);
  });
  listen(shell.plot, "pointerover", (e) => {
    if (e.pointerType !== "mouse") return;
    const g = nodeOf(e.target);
    const id = usable(g) ? g.dataset.node : null;
    if (id !== state.hover) {
      state.hover = id;
      paint();
    }
  });
  listen(shell.plot, "pointerleave", () => {
    if (state.hover) {
      state.hover = null;
      paint();
    }
  });
  listen(shell.plot, "focusin", (e) => {
    const g = e.target.closest && e.target.closest('[role="button"]');
    // pointer clicks also focus the node; only keyboard focus drives the readout
    if (g && g.matches(":focus-visible")) {
      state.focus = g.dataset.node;
      paint();
    }
  });
  listen(shell.plot, "focusout", () => {
    if (state.focus) {
      state.focus = null;
      paint();
    }
  });

  // Layout and drawing geometry follow the widget's own width, not the window's.
  function measure() {
    frame = 0;
    if (done) return;
    el.classList.toggle("cm-wide", el.clientWidth >= WIDE_FROM);
    const tall = shell.plot.clientWidth < TALL_BELOW;
    if (tall !== state.tall) {
      state.tall = tall;
      buildPlot();
    } else if (tall) {
      applyScale();
    }
  }
  function schedule() {
    if (!frame && !done) frame = requestAnimationFrame(measure);
  }
  if (typeof ResizeObserver !== "undefined") {
    observer = new ResizeObserver(schedule);
    observer.observe(el);
  }
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(() => {
      if (!done && shell.svg) {
        layoutLegends();
        fitNotes();
      }
    });
  }

  function onData() {
    m = prepare(model.get("data"));
    buildData();
  }
  model.on("change:data", onData);
  onData();
  measure();

  return () => {
    if (done) return;
    done = true;
    model.off("change:data", onData);
    disposers.splice(0).forEach((fn) => fn());
    if (observer) observer.disconnect();
    if (frame) cancelAnimationFrame(frame);
    el.replaceChildren();
    el.classList.remove("cm-root", "cm-wide");
    delete el.dataset.cmStep;
    delete el.dataset.cmNode;
    delete el.dataset.cmMode;
  };
}

export default { render };
