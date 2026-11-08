"""How the returns essay looks: palette, themes, page furniture and one chart per figure.

Nothing here fits, forecasts or transforms data. Each chart function takes the finished
arrays and frames from ``retail_returns_analysis`` and returns a drawable. The notebook
keeps the words (titles and captions) and the modelling.
"""

from __future__ import annotations

import dataclasses
import inspect
import textwrap
from html import escape

import altair as alt
import coeftable as ct
import marimo as mo
import numpy as np
import pandas as pd
from great_tables import html as gt_html
from great_tables import loc as gt_loc
from great_tables import style as gt_style
from retail_returns_analysis import fan

#: What the fitted chain is called in chart legends and tables.
MODEL_NAME = "Chained convolution model"

# ACCENT is the model's colour, the essay's link green, bright enough to stand out from
# the grey history and the dashed shortcuts.
INK, MUTED, ACCENT, AMBER, RED = "#292d26", "#62695d", "#0a8a45", "#d18f00", "#d03b2f"
RULE, GRID, PAPER, SLATE, STORM = "#dcded1", "#ecefe4", "#fffef9", "#4c5a6b", "#e9d6b0"
AMBER_LIGHT = "#edd295"  # the gold at 40% on paper: returns still to come
# The return-rate shortcuts: blues, clearly apart from the chain's green.
RATE, RATE_LIGHT = "#2f6db5", "#8db3e2"
VIOLET = "#7b52c7"  # the standard time-series baseline


@alt.theme.register("returns_paper", enable=True)
def _paper_theme():
    return alt.theme.ThemeConfig(
        {
            "config": {
                "background": PAPER,
                "font": "system-ui, -apple-system, 'Segoe UI', sans-serif",
                "view": {"stroke": None},
                "autosize": {"type": "fit-x", "contains": "padding"},
                "axis": {
                    "domainColor": RULE,
                    "tickColor": RULE,
                    "gridColor": GRID,
                    "labelColor": MUTED,
                    "titleColor": MUTED,
                    "labelFontSize": 11,
                    "titleFontSize": 11,
                    "titleFontWeight": 500,
                    "labelPadding": 6,
                    "titlePadding": 10,
                },
                "axisX": {"grid": False},
                "axisY": {"minExtent": 52},
                "legend": {
                    "orient": "top",
                    "direction": "horizontal",
                    "title": None,
                    "labelColor": INK,
                    "labelFontSize": 12,
                    "symbolStrokeWidth": 3,
                    "columnPadding": 18,
                },
                "title": {
                    "font": "Georgia, 'Iowan Old Style', serif",
                    "fontSize": 14,
                    "fontWeight": 500,
                    "color": INK,
                    "anchor": "start",
                    "offset": 10,
                },
            }
        }
    )


TABLE_THEME = ct.Theme(
    favorable="#386647",
    unfavorable="#A23C3C",
    inconclusive="#74816F",
    neutral=ACCENT,
    header_bg=PAPER,
    header_fg=INK,
    column_label_bg=GRID,
    band="#F5F5ED",
    surface=PAPER,
    rule=RULE,
    border_color=RULE,
    axis=MUTED,
    muted=MUTED,
    text=INK,
    value_size="14px",
    ci_size="12px",
    table_font_size="14px",
    border_style="minimal",
    series_palette=(MUTED, ACCENT),
)

# (anchor, part number, short label) for every opener, in reading order.
SECTIONS = (
    ("tldr", "", "TL;DR"),
    ("data", "1", "Data"),
    ("model", "2", "Model"),
    ("fit", "3", "Fit"),
    ("forecast", "4", "Does it work?"),
    ("scenarios", "5", "What if?"),
    ("wrap-up", "", "Wrap-up"),
)


# --- Page furniture -------------------------------------------------------------------


def nav_header(brand):
    """The sticky top bar: the brand and a link to every section."""
    links = "".join(
        f'<a href="#{key}">{label.replace("&", "&amp;")}</a>' for key, _, label in SECTIONS
    )
    return mo.Html(
        f'<header class="ttn-nav"><a class="ttn-brand" href="#top">{brand}</a>'
        f'<nav aria-label="Sections">{links}</nav></header>'
    )


def section(anchor, kicker, title, lede="", points=()):
    """A chapter opener: where you are in the essay, the title, and what this part shows."""
    here = [key for key, _, _ in SECTIONS].index(anchor)

    def _step(i, key, num, label):
        state = " is-here" if i == here else " is-done" if i < here else ""
        current = ' aria-current="location"' if i == here else ""
        number_html = f'<span class="ttn-map-num">{num}</span>' if num else ""
        return (
            f'<a class="ttn-map-step{state}" href="#{key}"{current}>'
            f'{number_html}<span class="ttn-map-label">{label}</span></a>'
        )

    steps = "".join(_step(i, *entry) for i, entry in enumerate(SECTIONS))
    number = SECTIONS[here][1]
    lede_html = f'<p class="ttn-section-lede">{lede}</p>' if lede else ""
    points_html = (
        '<ul class="ttn-section-points" aria-label="In this part">'
        + "".join(f"<li>{point}</li>" for point in points)
        + "</ul>"
        if points
        else ""
    )
    return mo.Html(
        f'<section id="{anchor}" class="ttn-section">'
        f'<nav class="ttn-map" aria-label="Essay sections">{steps}</nav>'
        f'<div class="ttn-section-head">'
        f'<p class="ttn-section-num" aria-hidden="true">{number}</p>'
        f'<div><p class="ttn-section-kicker">{kicker}</p>'
        f'<h2 class="ttn-section-title">{title}</h2>{lede_html}{points_html}</div>'
        f"</div></section>"
    )


def callout(label, body, tone="note", icon=""):
    """An aside with a small uppercase label; ``icon`` is optional inline SVG before it."""
    tone_class = " ttn-callout--warn" if tone == "warn" else ""
    mark = (
        f'<span style="display:inline-block;vertical-align:middle;margin:-4px 8px 0 0">{icon}</span>'
        if icon
        else ""
    )
    return mo.Html(
        f'<aside class="ttn-callout{tone_class}">'
        f'<p class="ttn-callout-label">{mark}{escape(label)}</p>{mo.md(body).text}</aside>'
    )


def convolution_icon(size=30, days=4, chosen=1, shift=6):
    """The survival-convolution mark: one time-to-event curve shifted by each sale day and
    stacked, so the outline is their sum. The chosen day is solid ink over a haloed dot;
    the others keep their own light-to-dark green as a translucent fill with an outline."""

    def mix(a, b, t):
        return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))

    xs = np.arange(0, 40.5, 0.5)
    curve = np.where(xs > 0, np.power(np.clip(xs, 0, None), 1.6) * np.exp(-xs / 4), 0.0)
    layers = np.stack([np.interp(xs - k * shift, xs, curve, left=0) for k in range(days)])
    tops = layers.cumsum(0)
    base, top, left = 34, 7, 5
    height = base - (tops / tops[-1].max()) * (base - top)
    shapes, dots = [], []
    for k in range(days):
        t = k / (days - 1)
        upper = [f"{left + x:.1f},{y:.1f}" for x, y in zip(xs, height[k])]
        lower = [
            f"{left + x:.1f},{y:.1f}"
            for x, y in zip(xs[::-1], (height[k - 1] if k else np.full_like(xs, base))[::-1])
        ]
        points = " ".join(upper + lower)
        cx = left + k * shift + 0.6
        if k == chosen:
            shapes.append(f'<polygon points="{points}" fill="{INK}"/>')
            dots.append(
                f'<circle cx="{cx}" cy="41.5" r="3.6" fill="none" stroke="{INK}" stroke-width="1"/>'
                f'<circle cx="{cx}" cy="41.5" r="2" fill="{INK}"/>'
            )
        else:
            fill = mix((207, 224, 210), (66, 100, 77), t)
            line = mix((150, 180, 156), (96, 130, 105), t)
            shapes.append(
                f'<polygon points="{points}" fill="{fill}" fill-opacity="0.35" stroke="{line}" '
                'stroke-width="0.8" stroke-linejoin="round"/>'
            )
            dots.append(
                f'<circle cx="{cx}" cy="41.5" r="1.4" fill="none" stroke="{line}" stroke-width="0.9"/>'
            )
    axis = f'<line x1="3" y1="{base + 0.5}" x2="46" y2="{base + 0.5}" stroke="{INK}" stroke-width="0.9"/>'
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 48 48" role="img" '
        f'aria-label="One sale day\'s spread among the others, adding up">'
        + "".join(shapes)
        + axis
        + "".join(dots)
        + "</svg>"
    )


def cards(items):
    """Summary cards: (label, value, detail, tone) with tone in accent/amber/red/None."""
    body = "".join(
        f'<div class="ttn-card{f" ttn-card--{tone}" if tone else ""}">'
        f'<p class="ttn-card-label">{label}</p><p class="ttn-card-value">{value}</p>'
        f'<p class="ttn-card-detail">{detail}</p></div>'
        for label, value, detail, tone in items
    )
    return mo.Html(f'<div class="ttn-cards" style="--ttn-card-count:{len(items)}">{body}</div>')


def interval_value(draws):
    """Card value: the mean with its 90% interval underneath."""
    return (
        f"{draws.mean():,.0f}"
        f"<small>[{np.quantile(draws, 0.05):,.0f}–{np.quantile(draws, 0.95):,.0f}]</small>"
    )


def figure(chart, title, caption):
    """A titled figure. ``caption`` is prose, or a list of ``(swatch, text)`` rows drawn as a
    key with every swatch in one left-hand column; use ``""`` as the swatch for a plain row."""
    if isinstance(caption, str):
        body = f'<p class="ttn-caption">{caption}</p>'
    else:
        rows = "".join(
            f'<li><span class="ttn-key-mark">{mark}</span><span>{text}</span></li>'
            for mark, text in caption
        )
        body = f'<ul class="ttn-caption ttn-key">{rows}</ul>'
    return mo.vstack(
        [
            mo.Html(f'<p class="ttn-figure-title">{title}</p>'),
            chart,
            mo.Html(body),
        ],
        gap=0.25,
    )


def swatch(color, dashed=False, block=False, opacity=0.35):
    """Inline legend key: a line, a dashed line, or a shaded block for bands."""
    if block:
        style = f"background: {color}; opacity: {opacity}; height: 11px; width: 13px"
    elif dashed:
        style = f"background: repeating-linear-gradient(90deg, {color} 0 5px, transparent 5px 8px)"
    else:
        style = f"background: {color}"
    return f'<span class="ttn-swatch" style="{style}"></span>'


def checks_list(checks):
    """A checklist of (label, passed, detail)."""
    return mo.Html(
        '<ul class="ttn-checks">'
        + "".join(
            f'<li class="{"" if ok else "ttn-check--fail"}">{"✓" if ok else "✗"} {label} '
            f"<code>{detail}</code></li>"
            for label, ok, detail in checks
        )
        + "</ul>"
    )


def source_md(*objects):
    """Markdown code blocks holding the actual source of the given functions."""
    return mo.md(
        "\n\n".join(
            f"```python\n{textwrap.dedent(inspect.getsource(obj)).rstrip()}\n```" for obj in objects
        )
    )


# --- Tables ---------------------------------------------------------------------------


def recovery_view(recovery):
    """Posterior against truth for every parameter, on one ratio axis."""
    return (
        ct.CoefTable(recovery, rows="Parameter", groups="stage")
        .estimate("Posterior", "estimate", ci=("lower", "upper"), fmt=ct.Number(decimals=2))
        .estimate("Truth", "truth", fmt=ct.Number(decimals=2))
        .estimate(
            "Posterior ÷ truth",
            "ratio",
            ci=("ratio_lower", "ratio_upper"),
            fmt=ct.Number(decimals=2),
        )
        .forest(
            "90% interval, as a multiple of the truth",
            of="Posterior ÷ truth",
            ref=1.0,
            scale="table",
            width=360,
            height=22,
        )
        .with_theme(
            # Sign is not good or bad here: one neutral colour for every interval.
            dataclasses.replace(TABLE_THEME, favorable=SLATE, unfavorable=SLATE, inconclusive=SLATE)
        )
    )


def scores_view(scores, actual_total, close=0.05):
    """Baselines against the model: daily error, CRPS, 28-day total (the actual total in
    small type under that column's header) and the total's signed error against the actual,
    green within ``close`` and red outside it. The model is the last row and is highlighted.
    """
    total, error = "28-day total", "Total error"
    misses = scores.total / actual_total - 1
    table = (
        ct.CoefTable(scores.assign(miss=misses), rows="Method")
        .estimate("Daily MAE", "mae", fmt=ct.Number(decimals=2))
        .estimate("Daily CRPS", "crps", fmt=ct.Number(decimals=2))
        .estimate(total, "total", ci=("total_lo", "total_hi"), fmt=ct.Number(decimals=0))
        .estimate(error, "miss", fmt=ct.Percent(decimals=0, scale=100))
        .with_theme(TABLE_THEME)
    )
    model = len(scores) - 1
    styled = (
        table.gt()
        .cols_label(
            cases={
                total: gt_html(
                    f'{total}<br><span style="font-size:0.78em;font-weight:400">'
                    f"actual {actual_total:,}</span>"
                )
            }
        )
        .tab_style(
            style=[gt_style.fill(color="#e6f4ea"), gt_style.text(weight="bold", color=INK)],
            locations=[gt_loc.body(rows=[model]), gt_loc.stub(rows=[model])],
        )
    )
    for row, miss in enumerate(misses):
        styled = styled.tab_style(
            style=gt_style.text(color=ACCENT if abs(miss) <= close else RED, weight="bold"),
            locations=gt_loc.body(columns=error, rows=[row]),
        )
    return styled


# --- Charts ---------------------------------------------------------------------------


def sale_shade(promos, first_day):
    """Slate bands for the sale spans that reach past ``first_day``."""
    return (
        alt.Chart(promos[promos.end > pd.Timestamp(first_day)])
        .mark_rect(color=SLATE, opacity=0.08)
        .encode(x="start:T", x2="end:T")
    )


def observed(recent, column, x="date:T", clip=False):
    """Recorded daily counts before the snapshot, in grey so the forecast stands out."""
    frame = recent[["date", column]].rename(columns={column: "value"})
    return (
        alt.Chart(frame)
        .mark_line(color=MUTED, strokeWidth=1.4, opacity=0.85, clip=clip)
        .encode(
            x=x,
            y="value:Q",
            tooltip=[
                alt.Tooltip("date:T", format="%a %b %-d"),
                alt.Tooltip("value:Q", title=column.lower()),
            ],
        )
    )


def snapshot_rule(recent, label=False, x="date:T"):
    """A dotted line between the last observed day and the first forecast day."""
    at = pd.DataFrame({"date": [recent.date.max() + pd.Timedelta(hours=12)]})
    rule = alt.Chart(at).mark_rule(color=MUTED, strokeDash=[3, 3], clip=True).encode(x=x)
    if not label:
        return rule
    return rule + alt.Chart(at).mark_text(
        align="right", baseline="top", dx=-4, dy=4, fontSize=11, color=MUTED, clip=True
    ).encode(x=x, y=alt.value(0), text=alt.value("today"))


def _local(day):
    """A Vega-Lite DateTime read in local time, like the dates in the data."""
    return alt.DateTime(year=day.year, month=day.month, date=day.day, hours=day.hour)


def window_x(first, last, show_axis=True, field="date"):
    """A shared date axis from day ``first`` to day ``last``, both inclusive.

    An explicit domain keeps every panel of a figure on the same scale. Ticks sit on
    Sundays, spaced so about seven fit, so no label lands near an edge and shrinks the panel
    that shows it.
    """
    lo = pd.Timestamp(first) - pd.Timedelta(hours=12)
    hi = pd.Timestamp(last) + pd.Timedelta(hours=12)
    sundays = pd.date_range(lo.ceil("D"), hi.floor("D"), freq="W-SUN")
    ticks = sundays[:: max(1, int(np.ceil(len(sundays) / 7)))]
    return alt.X(
        f"{field}:T",
        title=None,
        scale=alt.Scale(domain=[_local(lo), _local(hi)]),
        axis=alt.Axis(
            values=[_local(day) for day in ticks],
            format="%a %b %-d",
            labels=show_axis,
            ticks=show_axis,
            labelAngle=0,
        ),
    )


def tldr_panels(forecast, sales_draws, sales_actual, held_out, storms, promos, history, window):
    """Sales, starts and receipts as three shared-date Altair panels.

    ``history`` is the whole ledger as daily counts; ``window`` is the inclusive
    ``(first, last)`` pair. The browser can change that domain without recomputing data.
    """
    first, last = (pd.Timestamp(day) for day in window)
    snapshot = pd.Timestamp(history.date.max())

    def x(show_axis, field="date"):
        return window_x(first, last, show_axis, field)

    # One shaded rectangle per supplied storm day, each run labelled once.
    days = pd.DatetimeIndex(history.date).append(pd.DatetimeIndex(forecast.dates))
    stormy = storms(days.values) > 0
    storm = pd.DataFrame({"start": days[stormy], "end": days[stormy] + pd.Timedelta(days=1)})
    starts = stormy & ~pd.Series(stormy).shift(1, fill_value=False).to_numpy()
    storm_label = pd.DataFrame({"start": days[starts]})
    sales = promos.assign(label=lambda f: np.where(f.start > snapshot, "planned sale", "sale"))

    def sale_layers(show_axis, label):
        shade = (
            alt.Chart(sales)
            .mark_rect(color=SLATE, opacity=0.08, clip=True)
            .encode(x=x(show_axis, "start"), x2="end:T")
        )
        if not label:
            return [shade]
        text = (
            alt.Chart(sales)
            .mark_text(
                align="left", baseline="top", dx=5, dy=4, fontSize=11, color=SLATE, clip=True
            )
            .encode(x=x(show_axis, "start"), y=alt.value(0), text="label:N")
        )
        return [shade, text]

    def panel(draws, actual, color, title, column, show_axis, storm_shade=False, labels=False):
        frame = fan(forecast.dates, draws, title).assign(actual=actual)
        base = alt.Chart(frame).encode(x=x(show_axis))
        layers = [
            *sale_layers(show_axis, labels),
            snapshot_rule(history, label=column == "Returns initiated", x=x(show_axis)),
            observed(history, column, x=x(show_axis), clip=True),
            base.mark_area(color=color, opacity=0.14, clip=True).encode(y="lo90:Q", y2="hi90:Q"),
            base.mark_area(color=color, opacity=0.26, clip=True).encode(y="lo50:Q", y2="hi50:Q"),
            base.mark_line(color=color, strokeWidth=2, clip=True).encode(
                y=alt.Y("mean:Q", title=title)
            ),
            base.mark_circle(color=INK, size=26, opacity=0.85, clip=True).encode(y="actual:Q"),
        ]
        if storm_shade:
            layers = [
                alt.Chart(storm)
                .mark_rect(color=STORM, opacity=0.3, clip=True)
                .encode(x=x(show_axis, "start"), x2="end:T"),
                alt.Chart(storm_label)
                .mark_text(
                    align="left", baseline="top", dx=5, dy=4, fontSize=11, color=AMBER, clip=True
                )
                .encode(x=x(show_axis, "start"), y=alt.value(0), text=alt.value("storm")),
                *layers,
            ]
        return alt.layer(*layers).properties(height=95, width="container")

    return [
        panel(sales_draws, sales_actual, MUTED, "Units sold", "Sales", False, labels=True),
        panel(
            forecast.initiations,
            held_out.initiations,
            AMBER,
            "Returns started",
            "Returns initiated",
            False,
        ),
        panel(
            forecast.receipts,
            held_out.receipts,
            ACCENT,
            "Returns received",
            "Returns received",
            True,
            storm_shade=True,
        ),
    ]


def purchase_timeline(unit):
    """One unit's return, drawn to scale: purchased, return initiated, in transit, received.

    ``unit`` maps ``sold``, ``started`` and ``received`` to timestamps.
    """
    wait = (unit["started"] - unit["sold"]).days
    transit = (unit["received"] - unit["started"]).days
    left, right, line = 75, 605, 72

    def x(day):
        return left + day / (wait + transit) * (right - left)

    def stop(day, name, when, color):
        return (
            f'<circle cx="{x(day):.1f}" cy="{line}" r="7" fill="{PAPER}" stroke="{color}" '
            'stroke-width="2.5"/>'
            f'<text x="{x(day):.1f}" y="28" text-anchor="middle" font-size="14" '
            f'font-weight="600" fill="{INK}">{name}</text>'
            f'<text x="{x(day):.1f}" y="47" text-anchor="middle" font-size="12" '
            f'fill="{MUTED}">{when:%a %b} {when.day} · day {day}</text>'
        )

    def leg(start, end, color, text):
        return (
            f'<line x1="{x(start):.1f}" y1="{line}" x2="{x(end):.1f}" y2="{line}" '
            f'stroke="{color}" stroke-width="5" stroke-linecap="round"/>'
            f'<text x="{x((start + end) / 2):.1f}" y="{line + 30}" text-anchor="middle" '
            f'font-size="13" fill="{color}">{text}</text>'
        )

    return mo.Html(
        '<svg class="ttn-journey" viewBox="0 0 680 112" role="img" '
        'aria-label="One unit: purchased, return initiated, in transit, received" '
        'style="width: 100%; height: auto; font-family: inherit">'
        + leg(0, wait, AMBER, f"With the customer · {wait} days")
        + leg(wait, wait + transit, ACCENT, f"In transit · {transit} days")
        + stop(0, "Purchased", unit["sold"], INK)
        + stop(wait, "Return initiated", unit["started"], AMBER)
        + stop(wait + transit, "Received", unit["received"], ACCENT)
        + "</svg>"
    )


def convolution_panels(followed, sales, first_date, day):
    """One forecast sale day followed forward. The top row is units sold per day; below it,
    each day's returns started and returns received, stacked by the sale day they came from
    (later sale days darker), so a day's bar is the sum of its contributions. The chosen
    sale day is black in every panel, and the busiest receipt day is labelled.

    ``followed`` is ``follow_sales`` output (``started`` and ``received``, ``[S, days]``);
    ``sales`` the ``[S]`` expected units sold per day; ``day`` the highlighted sale day.
    """
    days = followed["started"].shape[1]
    dates = pd.date_range(str(first_date), periods=days)
    x_scale = alt.Scale(
        domain=[
            _local(dates[0] - pd.Timedelta(hours=12)),
            _local(dates[-1] + pd.Timedelta(days=1)),
        ]
    )

    def x(show_axis):
        return alt.X(
            "date:T",
            title=None,
            scale=x_scale,
            axis=alt.Axis(format="%b %-d", labels=show_axis, ticks=show_axis),
        )

    sold = pd.DataFrame({"date": dates[: len(sales)], "units": np.asarray(sales, dtype=float)})
    sold["end"] = sold.date + pd.Timedelta(hours=20)  # a small gap between days
    sold["sale"] = np.arange(len(sales))
    sold["picked"] = np.arange(len(sales)) == day
    top = (
        alt.Chart(sold)
        .mark_rect(clip=True)
        .encode(
            x=x(False),
            x2="end:T",
            y=alt.Y("units:Q", title="Units sold"),
            y2=alt.datum(0),
            color=alt.Color(
                "picked:N",
                scale=alt.Scale(domain=[False, True], range=["#d5d8cd", INK]),
                legend=None,
            ),
        )
        .properties(height=80, width="container")
    )

    def stage(rows, light, color, title, show_axis, label_peak):
        long = pd.DataFrame(
            [
                {
                    "date": dates[t],
                    "end": dates[t] + pd.Timedelta(hours=20),
                    "sale": s,
                    "sold": dates[s],
                    "units": rows[s, t],
                    "picked": s == day,
                }
                for s in range(rows.shape[0])
                for t in range(rows.shape[1])
                if rows[s, t] > 0.05
            ]
        )
        # Stack by hand: each segment sits on the sale days before it on the same day.
        long = long.sort_values(["date", "sale"])
        long["top"] = long.groupby("date").units.cumsum()
        long["bottom"] = long.top - long.units
        chart = (
            alt.Chart(long)
            .mark_rect(stroke=PAPER, strokeWidth=0.4, clip=True)
            .encode(
                x=x(show_axis),
                x2="end:T",
                y=alt.Y("top:Q", title=title),
                y2="bottom:Q",
                color=alt.condition(
                    alt.datum.picked,
                    alt.value(INK),
                    alt.Color("sale:Q", scale=alt.Scale(range=[light, color]), legend=None),
                ),
                order=alt.Order("sale:Q"),
                tooltip=[
                    alt.Tooltip("date:T", title="day", format="%a %b %-d"),
                    alt.Tooltip("sold:T", title="sold", format="%a %b %-d"),
                    alt.Tooltip("units:Q", title="returns", format=".1f"),
                ],
            )
        )
        if label_peak:
            totals = rows.sum(0)
            peak = int(np.argmax(totals))
            sources = int((rows[:, peak] > 0.05).sum())
            note = pd.DataFrame(
                {
                    "date": [dates[peak] + pd.Timedelta(hours=10)],
                    "units": [totals[peak]],
                    "text": [f"{dates[peak]:%b %-d}: {totals[peak]:.0f} from {sources} sale days"],
                }
            )
            chart += (
                alt.Chart(note)
                .mark_text(align="left", dx=6, dy=-2, fontSize=11, color=INK, baseline="bottom")
                .encode(x="date:T", y="units:Q", text="text:N")
            )
        return chart.properties(height=120, width="container")

    return [
        top,
        stage(followed["started"], AMBER_LIGHT, AMBER, "Returns started", False, False),
        stage(followed["received"], "#cde9d7", ACCENT, "Returns received", True, True),
    ]


def sale_echo(daily, sale, before=7, after=28):
    """One sale and the returns it brings: units sold, returns started and returns received
    per day, on one shared date axis, the sale shaded in every panel.

    Receipts are drawn on open (weekday) days only, so the line is not broken by the
    warehouse's closed weekends.
    """
    first = sale.start - pd.Timedelta(days=before)
    last = min(sale.end + pd.Timedelta(days=after), daily.date.max() + pd.Timedelta(days=1))
    frame = daily[(daily.date >= first) & (daily.date < last)]
    shade = (
        alt.Chart(pd.DataFrame([sale]))
        .mark_rect(color=SLATE, opacity=0.08)
        .encode(x="start:T", x2="end:T")
    )

    def panel(column, color, title, show_axis, rows=frame):
        return (
            shade
            + alt.Chart(rows)
            .mark_line(color=color, strokeWidth=2)
            .encode(
                x=alt.X(
                    "date:T",
                    title=None,
                    scale=alt.Scale(domain=[_local(first), _local(last - pd.Timedelta(days=1))]),
                    axis=alt.Axis(format="%b %-d", labels=show_axis, ticks=show_axis),
                ),
                y=alt.Y(f"{column}:Q", title=title),
            )
        ).properties(height=95, width="container")

    return mo.vstack(
        [
            panel("Sales", MUTED, "Units sold", False),
            panel("Returns initiated", AMBER, "Returns started", False),
            panel(
                "Returns received",
                ACCENT,
                "Returns received",
                True,
                rows=frame[frame.date.dt.dayofweek < 5],
            ),
        ],
        gap=0,
    )


def return_fates_chart(fates, marker=14):
    """Stacked shares of one batch of orders by age: returned so far, will return but
    hasn't yet, never returns. A dotted line at ``marker`` days shows how much of the
    silence at that age is still to come."""
    order = ["Returned so far", "Will return, not yet", "Never returns"]
    colors = [AMBER, AMBER_LIGHT, "#e3e5da"]
    long = fates.melt(id_vars="age", value_vars=order, var_name="fate", value_name="share")
    long["rank"] = long.fate.map(order.index)
    at = fates[fates.age == marker].iloc[0]
    rule = pd.DataFrame({"age": [marker]})
    rate = 1 - float(fates["Never returns"].iloc[0])
    labels = pd.DataFrame(
        {
            "age": [89, 60],
            "share": [rate, rate * 0.45],
            "text": [f"The return rate: {rate:.0%}", "Time to event (CDF)"],
            "color": [INK, PAPER],
            "baseline": ["bottom", "middle"],
        }
    )
    return (
        alt.Chart(long)
        .mark_area(opacity=1)
        .encode(
            x=alt.X("age:Q", title="Days since the order", scale=alt.Scale(domain=[0, 90])),
            y=alt.Y("share:Q", stack=True, title="Return rate", axis=alt.Axis(format="%")),
            color=alt.Color(
                "fate:N",
                scale=alt.Scale(domain=order, range=colors),
                legend=None,  # the figure's caption is the key
            ),
            order=alt.Order("rank:Q"),
            tooltip=[
                alt.Tooltip("age:Q", title="day"),
                alt.Tooltip("fate:N"),
                alt.Tooltip("share:Q", format=".0%"),
            ],
        )
        + alt.Chart(rule).mark_rule(color=INK, strokeDash=[3, 3]).encode(x="age:Q")
        + alt.Chart(rule)
        .mark_text(align="left", dx=5, dy=-4, fontSize=11, color=INK, baseline="bottom")
        .encode(
            x="age:Q",
            y=alt.value(14),
            text=alt.value(
                f"day {marker}: {at['Returned so far']:.0%} returned, "
                f"{at['Will return, not yet']:.0%} still to come"
            ),
        )
        + alt.Chart(pd.DataFrame({"share": [rate]}))
        .mark_rule(color=INK, strokeWidth=1.2)
        .encode(y="share:Q")
        + alt.Chart(labels[labels.baseline == "bottom"])
        .mark_text(align="right", dy=-5, fontSize=12, fontWeight=600, baseline="bottom")
        .encode(x="age:Q", y="share:Q", text="text:N", color=alt.Color("color:N", scale=None))
        + alt.Chart(labels[labels.baseline == "middle"])
        .mark_text(align="center", fontSize=12, fontWeight=600, baseline="middle")
        .encode(x="age:Q", y="share:Q", text="text:N", color=alt.Color("color:N", scale=None))
    ).properties(height=200, width="container")


def wait_charts(to_start, in_transit):
    """Histograms of two waits in days: sale to return started, and time in transit."""

    def panel(days, color, title, limit):
        counts = days.clip(upper=limit).value_counts().sort_index()
        frame = pd.DataFrame({"start": counts.index - 0.45, "end": counts.index + 0.45})
        frame["returns"] = counts.to_numpy()
        return (
            alt.Chart(frame)
            .mark_rect(color=color, opacity=0.85)
            .encode(
                x=alt.X(
                    "start:Q",
                    title=title,
                    scale=alt.Scale(domain=[-0.5, limit + 0.5], nice=False),
                    axis=alt.Axis(values=list(range(0, limit + 1, 10 if limit > 30 else 5))),
                ),
                x2="end:Q",
                y=alt.Y("returns:Q", title="Returns"),
                y2=alt.datum(0),
            )
            .properties(height=130, width="container")
        )

    # Stacked, not side by side: two container-width charts in a row overflow narrow pages.
    return mo.vstack(
        [
            panel(to_start, AMBER, "Days from sale to return", 90),
            panel(in_transit, ACCENT, "Days in transit", 30),
        ],
        gap=0.5,
    )


def cohort_chart(rates):
    """Return rate by weekly sale cohort: what the ledger shows, the truth, and the model."""
    window_span = pd.DataFrame(
        {
            "start": [rates.window_opens.iloc[0]],
            "end": [rates.week.max() + pd.Timedelta(days=6)],
        }
    )
    base = alt.Chart(rates).encode(
        x=alt.X("week:T", title="Week the unit was sold", axis=alt.Axis(format="%b %-d"))
    )
    window = (
        alt.Chart(window_span)
        .mark_rect(color="#ecefe4", opacity=0.7)
        .encode(x="start:T", x2="end:T")
    )
    label = (
        alt.Chart(window_span)
        .mark_text(align="left", dx=8, dy=12, color="#62695d", fontSize=11)
        .encode(x="start:T", y=alt.value(0), text=alt.value("Still inside the 90-day window"))
    )
    band = base.mark_area(color=SLATE, opacity=0.16).encode(y="model_lo:Q", y2="model_hi:Q")
    names = {
        "eventual": "Eventually returned (truth)",
        "observed": "Returned so far",
        "model": f"{MODEL_NAME} estimate",
    }
    series = list(names.values())
    lines = (
        alt.Chart(
            rates.melt(
                id_vars="week",
                value_vars=["eventual", "observed", "model"],
                var_name="series",
                value_name="rate",
            ).replace({"series": names})
        )
        .mark_line(strokeWidth=2.4, point=alt.OverlayMarkDef(size=26, filled=True))
        .encode(
            x="week:T",
            y=alt.Y(
                "rate:Q",
                title="Share of units with a return initiated",
                axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 0.5]),
            ),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=series, range=[INK, RED, SLATE]),
                legend=alt.Legend(orient="top", title=None, symbolType="stroke"),
            ),
            strokeDash=alt.StrokeDash(
                "series:N",
                scale=alt.Scale(domain=series, range=[[5, 4], [1, 0], [1, 0]]),
                legend=None,
            ),
        )
    )
    return (window + label + band + lines).properties(height=320, width="container")


def hazard_charts(initiation, receipt):
    """The two fitted timing curves with 90% bands, against the simulator's truth."""

    def panel(frame, color, title):
        base = alt.Chart(frame).encode(x=alt.X("age:Q", title=title))
        return (
            base.mark_area(color=color, opacity=0.18).encode(y="lo:Q", y2="hi:Q")
            + base.mark_line(color=color, strokeWidth=2.4).encode(
                y=alt.Y("mean:Q", title="Daily hazard", axis=alt.Axis(format="%"))
            )
            + base.mark_line(color=INK, strokeDash=[5, 4], strokeWidth=1.4).encode(y="truth:Q")
        ).properties(height=150, width="container")

    # Stacked, not side by side: two container-width charts in a row overflow narrow pages.
    return mo.vstack(
        [
            panel(initiation, AMBER, "Days since sale"),
            panel(receipt, ACCENT, "Days since return initiated"),
        ],
        gap=0.5,
    )


def forecast_chart(forecast, held_out, spans, promos, recent):
    """Returns initiated and received: the last weeks of the ledger, then the forecast against
    the held-out days; storm on receipts, sales on both."""

    def panel(draws, actual, color, title, show_axis, storm):
        frame = fan(forecast.dates, draws, title).assign(actual=actual)
        base = alt.Chart(frame).encode(x=window_x(recent.date.min(), forecast.dates[-1], show_axis))
        shade = (
            alt.Chart(spans[spans.end > recent.date.min()] if storm else spans[:0])
            .mark_rect(color=AMBER, opacity=0.07)
            .encode(x="start:T", x2="end:T")
        )
        return (
            sale_shade(promos, recent.date.min())
            + shade
            + snapshot_rule(recent, label=not show_axis)
            + observed(recent, title)
            + base.mark_area(color=color, opacity=0.14).encode(y="lo90:Q", y2="hi90:Q")
            + base.mark_area(color=color, opacity=0.26).encode(y="lo50:Q", y2="hi50:Q")
            + base.mark_line(color=color, strokeWidth=2).encode(y=alt.Y("mean:Q", title=title))
            + base.mark_circle(color=INK, size=34, opacity=0.9).encode(
                y="actual:Q",
                tooltip=[
                    alt.Tooltip("date:T", format="%a %b %-d"),
                    alt.Tooltip("actual:Q", title="held-out actual"),
                    alt.Tooltip("mean:Q", format=".1f", title="forecast mean"),
                ],
            )
        ).properties(height=150, width="container")

    return mo.vstack(
        [
            panel(
                forecast.initiations,
                held_out.initiations,
                AMBER,
                "Returns initiated",
                show_axis=False,
                storm=False,
            ),
            panel(
                forecast.receipts,
                held_out.receipts,
                ACCENT,
                "Returns received",
                show_axis=True,
                storm=True,
            ),
        ],
        gap=0,
    )


def owed_chart(forecast, eventual_truth):
    """Forest plot of the returns still owed by past sales: in transit, not yet started,
    and their total, each as a mean with its 90% interval against what actually arrived."""
    in_transit = np.asarray(forecast.expected_open_receipts)
    not_started = np.asarray(forecast.expected_uninitiated_receipts)
    rows = [
        ("In transit", in_transit, eventual_truth["open"], ACCENT),
        ("Not started yet", not_started, eventual_truth["uninitiated"], AMBER),
        ("Total still expected", in_transit + not_started, sum(eventual_truth.values()), INK),
    ]
    parts = pd.DataFrame(
        [
            {
                "component": label,
                "mean": float(np.mean(draws)),
                "lo": float(np.quantile(draws, 0.05)),
                "hi": float(np.quantile(draws, 0.95)),
                "truth": truth,
                "color": color,
            }
            for label, draws, truth, color in rows
        ]
    )
    parts["text"] = [
        f"{m:,.0f}  [{lo:,.0f}–{hi:,.0f}]" for m, lo, hi in zip(parts["mean"], parts.lo, parts.hi)
    ]
    base = alt.Chart(parts).encode(
        y=alt.Y(
            "component:N",
            title=None,
            sort=None,
            axis=alt.Axis(labelFontSize=12, labelFontWeight=600, ticks=False, domain=False),
        )
    )
    color = alt.Color("color:N", scale=None)
    return (
        base.mark_rule(strokeWidth=3, opacity=0.55).encode(
            x=alt.X("lo:Q", title="Units that will still come back", scale=alt.Scale(zero=True)),
            x2="hi:Q",
            color=color,
        )
        + base.mark_tick(color=INK, thickness=2, size=22).encode(x="truth:Q")
        + base.mark_circle(size=120, opacity=1).encode(
            x="mean:Q",
            color=color,
            tooltip=[
                alt.Tooltip("component:N", title="state"),
                alt.Tooltip("mean:Q", format=",.0f", title="model mean"),
                alt.Tooltip("lo:Q", format=",.0f", title="5%"),
                alt.Tooltip("hi:Q", format=",.0f", title="95%"),
                alt.Tooltip("truth:Q", title="actually arrived"),
            ],
        )
        + base.mark_text(align="left", dx=10, dy=-12, fontSize=11, color=MUTED).encode(
            x="hi:Q", text="text:N"
        )
    ).properties(height=150, width="container")


def drift_chart(forecast, drift_forecast, sales_draws, drift_sales_draws, sales_actual, held_out):
    """The Part 4 forecast against the drifting-level one, stage by stage."""

    def panel(current, drifting, actual, color, title, show_axis):
        now = fan(forecast.dates, current, "current")
        new = fan(forecast.dates, drifting, "drifting level").assign(actual=actual)
        x = alt.X(
            "date:T",
            title=None,
            axis=alt.Axis(format="%a %b %-d", labels=show_axis, ticks=show_axis, labelAngle=0),
        )
        return alt.layer(
            alt.Chart(now)
            .mark_area(color=color, opacity=0.14)
            .encode(x=x, y="lo90:Q", y2="hi90:Q"),
            alt.Chart(now)
            .mark_line(color=color, strokeWidth=2)
            .encode(x=x, y=alt.Y("mean:Q", title=title)),
            alt.Chart(new)
            .mark_area(color=SLATE, opacity=0.16)
            .encode(x=x, y="lo90:Q", y2="hi90:Q"),
            alt.Chart(new)
            .mark_line(color=SLATE, strokeWidth=2.2, strokeDash=[5, 3])
            .encode(x=x, y="mean:Q"),
            alt.Chart(new).mark_circle(color=INK, size=30, opacity=0.9).encode(x=x, y="actual:Q"),
        ).properties(height=130, width="container")

    return mo.vstack(
        [
            panel(sales_draws, drift_sales_draws, sales_actual, MUTED, "Units sold", False),
            panel(
                forecast.initiations,
                drift_forecast.initiations,
                held_out.initiations,
                AMBER,
                "Returns started",
                False,
            ),
            panel(
                forecast.receipts,
                drift_forecast.receipts,
                held_out.receipts,
                ACCENT,
                "Returns received",
                True,
            ),
        ],
        gap=0,
    )


def baseline_chart(forecast, held_out, baselines, spans, promos, recent, colors=None, y_max=None):
    """The last weeks of observed receipts, then the model mean and 90% band against each
    shortcut and the held-out actuals. Shortcut lines are dashed and partly transparent so
    the chain stays in front.

    ``baselines`` maps a legend label to a point forecast. ``colors`` gives one line colour
    per baseline; by default red, blue, light blue and violet in order. ``y_max`` fixes the
    top of the y-axis, so switching which shortcut is shown doesn't rescale the chart.
    """
    colors = list(colors or [RED, RATE, RATE_LIGHT, VIOLET])[: len(baselines)]
    frame = fan(forecast.dates, forecast.receipts, MODEL_NAME).assign(
        actual=held_out.receipts.to_numpy()
    )
    x = window_x(recent.date.min(), forecast.dates[-1])
    model = f"{MODEL_NAME} mean"
    lines = pd.concat(
        [frame[["date", "mean"]].rename(columns={"mean": "value"}).assign(series=model)]
        + [
            pd.DataFrame({"date": frame.date, "value": np.asarray(point), "series": label})
            for label, point in baselines.items()
        ]
    )
    return (
        sale_shade(promos, recent.date.min())
        + alt.Chart(spans[spans.end > recent.date.min()])
        .mark_rect(color=AMBER, opacity=0.07)
        .encode(x="start:T", x2="end:T")
        + snapshot_rule(recent, label=True)
        + observed(recent, "Returns received")
        + alt.Chart(frame)
        .mark_area(color=ACCENT, opacity=0.14)
        .encode(x=x, y="lo90:Q", y2="hi90:Q")
        + alt.Chart(lines)
        .mark_line()
        .encode(
            x=x,
            y=alt.Y(
                "value:Q",
                title="Returns received per day",
                scale=alt.Scale(domain=[0, y_max]) if y_max else alt.Undefined,
            ),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    domain=[model, *baselines],
                    range=[ACCENT, *colors],
                ),
                legend=alt.Legend(labelLimit=0, columns=1),
            ),
            strokeDash=alt.condition(
                alt.datum.series == model, alt.value([1, 0]), alt.value([6, 4])
            ),
            opacity=alt.condition(alt.datum.series == model, alt.value(1.0), alt.value(0.65)),
            strokeWidth=alt.condition(alt.datum.series == model, alt.value(2.6), alt.value(2.2)),
        )
        + alt.Chart(frame).mark_circle(color=INK, size=34).encode(x=x, y="actual:Q")
    ).properties(height=240, width="container")
