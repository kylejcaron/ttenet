/* Model explanations and real source; every interaction is local, including in exports. */
const escape = (value) => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const tokens = /#[^\n]*|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|\b(?:def|class|return|if|else|elif|for|in|with|as|is|not|None|True|False|and|or|lambda)\b/g;

function highlight(line) {
  let html = '';
  let start = 0;
  for (const match of line.matchAll(tokens)) {
    html += escape(line.slice(start, match.index));
    const kind = match[0].startsWith('#') ? 'comment' : /^["']/.test(match[0]) ? 'string' : 'keyword';
    html += `<span class="${kind}">${escape(match[0])}</span>`;
    start = match.index + match[0].length;
  }
  return html + escape(line.slice(start));
}

function sourceCard(source) {
  const row = (index) => `<div class="source-line"><span class="line-number">${source.start + index}</span><span class="source-text">${highlight(source.lines[index]) || ' '}</span></div>`;
  let code = '';
  let index = 0;
  while (index < source.lines.length) {
    const fold = source.folds.find((item) => item.start === index);
    if (fold) {
      code += `<details class="source-fold"><summary>${escape(fold.label)} · ${fold.end - fold.start + 1} lines</summary>`;
      while (index <= fold.end) code += row(index++);
      code += '</details>';
    } else {
      code += row(index++);
    }
  }
  return `<article class="source-card">
    <div class="source-head"><strong>${escape(source.name)}</strong><span>${escape(source.file)}:${source.start}</span></div>
    <div class="source-code">${code}</div>
    <div class="source-path">Actual source · ${escape(source.file)}</div>
  </article>`;
}

// Schematic marks, not fitted values. The return stages emphasize their different clocks.
const icons = {
  sales: `<svg viewBox="0 0 260 81" aria-hidden="true">
    <g fill="var(--overview-blue)">
      <rect x="10" y="45" width="14" height="23" rx="2" opacity=".3"/>
      <rect x="36" y="38" width="14" height="30" rx="2" opacity=".3"/>
      <rect x="62" y="48" width="14" height="20" rx="2" opacity=".3"/>
      <rect x="88" y="30" width="14" height="38" rx="2" opacity=".4"/>
      <rect x="114" y="8" width="14" height="60" rx="2"/>
      <rect x="140" y="20" width="14" height="48" rx="2" opacity=".8"/>
      <rect x="166" y="39" width="14" height="29" rx="2" opacity=".4"/>
      <rect x="192" y="44" width="14" height="24" rx="2" opacity=".3"/>
      <rect x="218" y="49" width="14" height="19" rx="2" opacity=".3"/>
    </g><path d="M4 69H245" stroke="var(--overview-rule)"/>
    <text x="143" y="9" font-size="9" fill="var(--overview-blue)">promo</text>
  </svg>`,
  initiation: `<svg viewBox="0 0 260 81" aria-hidden="true">
    <circle cx="14" cy="40" r="4" fill="var(--overview-gold)"/>
    <path d="M18 40H67" fill="none" stroke="var(--overview-gold)" stroke-width="1.4"/>
    <path d="M67 40C80 40 80 10 97 10S112 24 125 21S144 32 160 38S188 40 226 40" fill="none" stroke="var(--overview-gold)" stroke-width="2"/>
    <path d="M18 40H30Q39 40 43 50L50 68H222" fill="none" stroke="var(--overview-muted)" stroke-opacity=".55" stroke-width="1.2"/>
    <circle cx="227" cy="68" r="4" fill="none" stroke="var(--overview-muted)" stroke-opacity=".55"/>
    <text x="160" y="18" fill="var(--overview-gold)" font-size="9">later</text>
    <text x="112" y="63" fill="var(--overview-muted)" font-size="9">or never</text>
  </svg>`,
  receipt: `<svg viewBox="0 0 260 81" aria-hidden="true">
    <g fill="var(--overview-muted)" opacity=".1"><rect x="158" y="6" width="24" height="54" rx="2"/><rect x="186" y="6" width="24" height="54" rx="2"/></g>
    <circle cx="14" cy="54" r="4" fill="var(--overview-green)"/>
    <path d="M21 54C36 54 44 18 64 18S99 27 121 37L149 48L162 54H200L219 52L246 54" fill="none" stroke="var(--overview-green)" stroke-width="2"/>
    <path d="M7 60H250" stroke="var(--overview-rule)"/>
    <text x="20" y="76" fill="var(--overview-muted)" font-size="9">started</text>
    <text x="161" y="75" fill="var(--overview-muted)" font-size="9">Sat</text>
    <text x="188" y="75" fill="var(--overview-muted)" font-size="9">Sun</text>
  </svg>`,
};

function convolutionDiagram(markerId) {
  const times = Array.from({ length: 81 }, (_, i) => i / 80);
  const terms = [[0, 20], [.18, 34], [.38, 26]].map(([origin, weight]) => times.map((time) => {
    const age = Math.max(time - origin, 0) / .13;
    return weight * age ** 2 * Math.exp(2 - 2 * age);
  }));
  const total = times.map((_, i) => terms.reduce((sum, term) => sum + term[i], 0));
  const scale = 68 / Math.max(...total);
  const path = (values) => values.map((value, i) => `${i ? 'L' : 'M'}${(168 + 157 * times[i]).toFixed(2)},${(96 - value * scale).toFixed(2)}`).join(' ');
  const curves = terms.map((values, i) => `<path d="${path(values)}" fill="none" stroke="var(--overview-gold)" stroke-opacity="${[.35, .65, .45][i]}" stroke-width="1.2"/>`).join('');
  return `<svg viewBox="0 0 570 120" aria-hidden="true">
    <defs><marker id="${markerId}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10" fill="var(--overview-muted)"/></marker></defs>
    <g fill="var(--overview-muted)" font-size="10" text-anchor="middle">
      <text x="62" y="15">Sales occur</text><text x="245" y="15">Shift, scale, sum</text><text x="470" y="15">Each path feeds the next clock</text>
    </g>
    <path d="M16 96H118" stroke="var(--overview-rule)"/>
    <path d="M32 96V71M64 96V40M96 96V59" stroke="var(--overview-blue)" stroke-width="8"/>
    <path d="M131 65H154" stroke="var(--overview-muted)" marker-end="url(#${markerId})"/>
    ${curves}<path d="${path(total)}" fill="none" stroke="var(--overview-gold)" stroke-width="2.2"/>
    <path d="M340 65H363" stroke="var(--overview-muted)" marker-end="url(#${markerId})"/>
    <g fill="none" stroke="var(--overview-green)">
      <path d="M380 96C403 96 407 64 431 61S468 25 487 41S517 86 557 96" stroke-width="1.5" opacity=".35"/>
      <path d="M380 96C404 96 421 48 440 51S470 33 489 46S530 93 557 96" stroke-width="2"/>
      <path d="M380 96C407 96 425 66 447 55S476 41 497 52S535 93 557 96" stroke-width="1.5" opacity=".5"/>
    </g>
  </svg>`;
}

export default {
  render({ model, el }) {
    const root = document.createElement('section');
    root.className = 'ttn-overview';
    const id = `overview-${crypto.randomUUID()}`;
    const detailId = `${id}-detail`;
    let selected = null;
    let data;
    el.append(root);
    const find = (selector) => root.querySelector(selector);

    function select(piece, scroll = true) {
      selected = piece;
      root.dataset.selected = piece || '';
      for (const button of root.querySelectorAll('[data-piece]')) {
        button.setAttribute('aria-expanded', String(button.dataset.piece === piece));
      }
      const panel = find('.overview-detail');
      panel.hidden = !piece;
      if (!piece) return;
      const content = data.pieces[piece];
      root.dataset.color = piece;
      find('.detail-kicker').textContent = content.kicker;
      find('.detail-title').textContent = content.title;
      find('.detail-copy').innerHTML = `<p class="main-point">${escape(content.main)}</p><p>${escape(content.body)}</p><p class="equation">${escape(content.equation)}</p>`;
      find('.detail-source').innerHTML = sourceCard(content.source);
      find('.deeper-label').textContent = piece === 'convolution' ? 'Censoring, calendar rules and propagation source' : 'Equations, priors and full source';
      find('.deeper-copy').innerHTML = `<h5>${escape(content.more_title)}</h5>${content.notes.map((note) => `<p${note.math ? ' class="math-line"' : ''}>${escape(note.text)}</p>`).join('')}`;
      find('.deeper-source').innerHTML = content.extra_sources.map((source) => `<details class="extra-source"><summary>${escape(source.name)}</summary>${sourceCard(source)}</details>`).join('');
      find('.overview-deeper').open = false;
      if (scroll) panel.scrollIntoView({ block: 'nearest', behavior: 'instant' });
    }

    function draw() {
      data = model.get('data');
      root.innerHTML = `<h3 class="overview-title">Three forecasts. One chain.</h3>
        <p class="overview-lede">A count model for sales. Two event-time models for returns. Each has a different job.</p>
        <div class="constructor-head"><code>RetailReturnModel(</code><span>Constructor · abbreviated</span></div>
        <div class="overview-stages" aria-label="Three model components">
          ${data.stages.map((stage) => `<button class="overview-stage" data-piece="${escape(stage.id)}" aria-expanded="false" aria-controls="${detailId}" aria-label="${escape(stage.name)}: ${escape(stage.question)} Show mechanism and source">
            <code class="code-field">${escape(stage.code)}</code><span class="stage-body">
              <span class="stage-title">${escape(stage.name)}</span><span class="question">${escape(stage.question)}</span>
              ${icons[stage.id]}<span class="takeaway">${escape(stage.takeaway)}</span>
              <span class="stage-foot"><span>${escape(stage.rule)}</span><strong>Mechanism + source ↘</strong></span>
            </span></button>`).join('')}
        </div><div class="constructor-close">)</div>
        <p class="overview-help">Select a stage for its mechanism and actual code. Open one detail at a time.</p>
        <button class="overview-shared" data-piece="convolution" aria-expanded="false" aria-controls="${detailId}">
          <span class="shared-copy"><span class="eyebrow">What connects the stages</span><span class="shared-title">Spread, add, pass on.</span>
          <span class="shared-description">Each cohort spreads through its timing law. Add the contributions; pass whole sampled paths to the next stage.</span>
          <span class="open-detail">The survival convolution · see inside ↘</span></span>
          <span class="shared-figure">${convolutionDiagram(`${id}-arrow`)}<small>Mechanism schematic · not forecast values</small></span>
        </button>
        <section class="overview-detail" id="${detailId}" hidden aria-labelledby="${id}-title">
          <div class="detail-top"><div><div class="eyebrow detail-kicker"></div><h4 class="detail-title" id="${id}-title"></h4></div><button class="close-detail" type="button">Close ×</button></div>
          <div class="detail-layout"><div class="detail-copy"></div><div class="detail-source"></div></div>
          <details class="overview-deeper"><summary class="deeper-label"></summary><div class="deeper-content"><div class="deeper-copy"></div><div class="deeper-source"></div></div></details>
        </section>`;
      if (selected && !data.pieces[selected]) selected = null;
      select(selected, false);
    }

    function click(event) {
      const button = event.target.closest('button');
      if (!button || !root.contains(button)) return;
      if (button.classList.contains('close-detail')) {
        const previous = selected;
        select(null);
        find(`[data-piece="${previous}"]`).focus({ preventScroll: true });
      } else if (button.dataset.piece) {
        select(selected === button.dataset.piece ? null : button.dataset.piece);
      }
    }

    root.addEventListener('click', click);
    model.on('change:data', draw);
    draw();
    return () => {
      root.removeEventListener('click', click);
      model.off('change:data', draw);
      root.remove();
    };
  },
};
