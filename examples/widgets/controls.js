/* The essay's three controls select precomputed data locally, including in static HTML. */
import embed from 'https://cdn.jsdelivr.net/npm/vega-embed@7.1.0/+esm';

const localDate = (iso) => {
  const [year, month, day] = iso.slice(0, 10).split('-').map(Number);
  return new Date(year, month - 1, day);
};
const offsetDate = (iso, offset) => {
  const date = localDate(iso);
  date.setDate(date.getDate() + offset);
  return date;
};
const isoDate = (date) => [date.getFullYear(), String(date.getMonth() + 1).padStart(2, '0'), String(date.getDate()).padStart(2, '0')].join('-');
const dateTime = (date) => ({ year: date.getFullYear(), month: date.getMonth() + 1, date: date.getDate(), hours: date.getHours() });
const shortDate = (date) => date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
const longDate = (date) => `${date.toLocaleDateString('en-US', { weekday: 'short' })} ${shortDate(date)}`;
const whole = (value) => value.toFixed(0);

// Only chart grammar is traversed; the immutable data tables live outside the specs.
function walkSpec(spec, visit) {
  visit(spec);
  for (const key of ['layer', 'vconcat', 'hconcat']) {
    for (const child of spec[key] || []) walkSpec(child, visit);
  }
}

function historyAxes(spec, first, last, width) {
  const lo = new Date(first);
  const hi = new Date(last);
  lo.setHours(-12);
  hi.setHours(12);
  const sundays = [];
  const day = new Date(first);
  day.setDate(day.getDate() + (7 - day.getDay()) % 7);
  while (day <= last) {
    sundays.push(dateTime(day));
    day.setDate(day.getDate() + 7);
  }
  const tickCount = Math.max(2, Math.min(7, Math.floor((width - 83) / 85)));
  const step = Math.max(1, Math.ceil(sundays.length / tickCount));
  const ticks = sundays.filter((_, i) => i % step === 0);
  walkSpec(spec, (node) => {
    const x = node.encoding?.x;
    if (x?.type !== 'temporal') return;
    x.scale = { ...x.scale, domain: [dateTime(lo), dateTime(hi)] };
    if (x.axis) x.axis.values = ticks;
  });
}

export default {
  render({ model, el }) {
    const root = document.createElement('section');
    root.className = 'ttn-controls';
    el.append(root);
    const listeners = [];
    let data;
    let selected;
    let windowDays;
    let views = [];
    let disposed = false;
    let revision = 0;
    let frame = 0;
    let renderedWidth = 0;
    let title;
    let plots;
    let caption;
    let output;
    let start;
    let end;
    let rangeTrack;
    let observer;

    const node = (tag, className, text) => {
      const result = document.createElement(tag);
      if (className) result.className = className;
      if (text !== undefined) result.textContent = text;
      return result;
    };
    const on = (target, event, fn) => {
      target.addEventListener(event, fn);
      listeners.push([target, event, fn]);
    };
    function clearViews() {
      for (const view of views) view.finalize();
      views = [];
    }
    function clearListeners() {
      for (const [target, event, fn] of listeners) target.removeEventListener(event, fn);
      listeners.length = 0;
    }
    function showCaption(rows) {
      caption.replaceChildren(...rows.map(([mark, text]) => {
        const row = node('li');
        const key = node('span', 'ttn-key-mark');
        key.innerHTML = mark;
        row.append(key, node('span', '', text));
        return row;
      }));
    }
    function stateText() {
      if (data.kind === 'history') return windowDays.join(',');
      return String(selected);
    }
    function showText() {
      if (data.kind === 'history') {
        const first = offsetDate(data.as_of, windowDays[0]);
        const last = offsetDate(data.as_of, windowDays[1]);
        output.textContent = `${shortDate(first)} – ${shortDate(last)}`;
        start.value = windowDays[0];
        end.value = windowDays[1];
        start.setAttribute('aria-valuetext', longDate(first));
        end.setAttribute('aria-valuetext', longDate(last));
        root.dataset.windowStart = isoDate(first);
        root.dataset.windowEnd = isoDate(last);
        const fraction = (day) => 100 * (day - data.min) / (data.max - data.min);
        rangeTrack.style.setProperty('--range-start', `${fraction(windowDays[0])}%`);
        rangeTrack.style.setProperty('--range-end', `${fraction(windowDays[1])}%`);
        // A collapsed range must leave the thumb that can move inward on top.
        rangeTrack.dataset.front = windowDays[0] > (data.min + data.max) / 2 ? 'start' : 'end';
      } else if (data.kind === 'sale') {
        const day = localDate(data.dates[selected]);
        output.textContent = longDate(day);
        start.setAttribute('aria-valuetext', longDate(day));
        root.dataset.selected = String(selected);
        root.dataset.selectedDate = data.dates[selected];
        showCaption([
          [data.caption[0][0], `${shortDate(day)}: ${whole(data.sales[selected])} units sold → ${whole(data.started[selected])} returns started → ${whole(data.received[selected])} received.`],
          ...data.caption.slice(1),
        ]);
      } else {
        const choice = data.choices[selected];
        title.textContent = choice.title;
        root.dataset.selected = choice.label;
        root.dataset.series = choice.series;
        showCaption([[choice.mark, choice.caption], ...data.caption]);
      }
    }
    async function draw() {
      frame = 0;
      const current = ++revision;
      const currentState = stateText();
      const width = Math.max(160, Math.floor(root.getBoundingClientRect().width));
      renderedWidth = width;
      root.dataset.rendering = 'true';
      const nextViews = [];
      const containers = data.panels.map(() => node('div', 'ttn-control-panel'));
      clearViews();
      plots.replaceChildren(...containers);
      try {
        for (let i = 0; i < data.panels.length; i++) {
          if (disposed || current !== revision) break;
          const panel = data.panels[i];
          const spec = structuredClone(panel.spec);
          spec.width = width;
          spec.datasets = Object.fromEntries(panel.datasets.map((name) => [name, structuredClone(data.datasets[name])]));
          // Flush edge labels inward so labelled and unlabelled panels keep equal widths.
          walkSpec(spec, (part) => {
            if (part.encoding?.x?.axis) part.encoding.x.axis.labelFlush = 45;
          });
          if (data.kind === 'history') {
            historyAxes(spec, offsetDate(data.as_of, windowDays[0]), offsetDate(data.as_of, windowDays[1]), width);
          } else if (data.kind === 'sale') {
            for (const name of data.highlight_datasets) {
              for (const row of spec.datasets[name] || []) row.picked = row.sale === selected;
            }
          } else {
            const choice = data.choices[selected];
            spec.datasets[data.line_dataset] = spec.datasets[data.line_dataset].filter((row) => row.series === data.model_series || row.series === choice.series);
            walkSpec(spec, (part) => {
              const color = part.encoding?.color;
              if (color?.field === 'series') {
                color.scale = { domain: [data.model_series, choice.series], range: [color.scale.range[0], choice.color] };
              }
            });
          }
          const result = await embed(containers[i], spec, { actions: false, renderer: 'svg' });
          if (disposed || current !== revision) {
            result.finalize();
            break;
          }
          containers[i].dataset.panel = String(i);
          containers[i].dataset.renderedState = currentState;
          nextViews.push(result);
        }
        if (disposed || current !== revision) {
          for (const view of nextViews) view.finalize();
          return;
        }
        views = nextViews;
        root.dataset.renderedState = currentState;
        root.dataset.rendering = 'false';
      } catch (error) {
        for (const view of nextViews) view.finalize();
        if (disposed || current !== revision) return;
        root.dataset.rendering = 'false';
        const message = node('p', 'ttn-control-error', `Unable to draw figure: ${error.message}`);
        message.setAttribute('role', 'alert');
        plots.replaceChildren(message);
      }
    }
    function scheduleDraw() {
      // Invalidate a pending embed before the next frame can apply an older selection.
      revision++;
      if (!frame) frame = requestAnimationFrame(draw);
    }
    function rangeInput(label, value, min, max) {
      const input = node('input');
      input.type = 'range';
      input.min = min;
      input.max = max;
      input.step = '1';
      input.value = value;
      input.setAttribute('aria-label', label);
      return input;
    }
    function controls() {
      const box = node('div', 'ttn-control-row');
      output = node('output', 'ttn-range-dates');
      output.setAttribute('aria-live', 'polite');
      if (data.kind === 'history') {
        box.setAttribute('role', 'group');
        box.setAttribute('aria-label', 'History shown');
        const label = node('span', 'ttn-control-label', 'History shown');
        rangeTrack = node('div', 'ttn-double-range');
        start = rangeInput('History start date', windowDays[0], data.min, data.max);
        end = rangeInput('History end date', windowDays[1], data.min, data.max);
        rangeTrack.append(start, end);
        on(start, 'input', () => {
          windowDays[0] = Math.min(Number(start.value), windowDays[1]);
          showText();
          scheduleDraw();
        });
        on(end, 'input', () => {
          windowDays[1] = Math.max(Number(end.value), windowDays[0]);
          showText();
          scheduleDraw();
        });
        box.append(label, rangeTrack, output);
      } else if (data.kind === 'sale') {
        start = rangeInput('Sale day', selected, 0, data.dates.length - 1);
        on(start, 'input', () => {
          selected = Number(start.value);
          showText();
          scheduleDraw();
        });
        const label = node('label', 'ttn-control-label', 'Sale day');
        label.append(start);
        box.append(label, output);
      } else {
        const fieldset = node('fieldset', 'ttn-baseline-options');
        fieldset.append(node('legend', 'ttn-control-label', 'Compare the chain with'));
        const name = `baseline-${crypto.randomUUID()}`;
        data.choices.forEach((choice, index) => {
          const label = node('label');
          const input = node('input');
          input.type = 'radio';
          input.name = name;
          input.value = String(index);
          input.checked = index === selected;
          on(input, 'change', () => {
            if (!input.checked) return;
            selected = index;
            showText();
            scheduleDraw();
          });
          label.append(input, document.createTextNode(choice.label));
          fieldset.append(label);
        });
        box.append(fieldset);
      }
      return box;
    }
    function renderAll() {
      revision++;
      clearViews();
      clearListeners();
      data = model.get('data');
      selected = data.default;
      windowDays = data.kind === 'history' ? [...data.default] : null;
      root.dataset.control = data.kind;
      title = node('p', 'ttn-figure-title', data.title);
      plots = node('div', 'ttn-control-plots');
      caption = node('ul', 'ttn-caption ttn-key');
      showCaption(data.caption);
      const inputs = controls();
      root.replaceChildren(title);
      if (data.description) root.append(node('p', 'ttn-control-description', data.description));
      if (data.kind === 'baseline') root.append(inputs, plots);
      else root.append(plots, inputs);
      root.append(caption);
      showText();
      scheduleDraw();
    }
    observer = new ResizeObserver(() => {
      if (!disposed && Math.abs(root.getBoundingClientRect().width - renderedWidth) >= 2) scheduleDraw();
    });
    observer.observe(root);
    model.on('change:data', renderAll);
    renderAll();
    return () => {
      disposed = true;
      revision++;
      cancelAnimationFrame(frame);
      observer.disconnect();
      model.off('change:data', renderAll);
      clearViews();
      clearListeners();
      root.remove();
    };
  },
};
