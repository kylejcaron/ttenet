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
from retail_returns_analysis import fan

#: What the fitted chain is called in chart legends and tables.
MODEL_NAME = "Chained model"

INK, MUTED, ACCENT, AMBER, RED = "#292d26", "#62695d", "#42644d", "#956017", "#a23c3c"
RULE, GRID, PAPER, SLATE, STORM = "#dcded1", "#ecefe4", "#fffef9", "#4c5a6b", "#e9d6b0"


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
    ("forecast", "4", "Forecast"),
    ("scenarios", "5", "What if?"),
    ("sales-model", "6", "Swap & compare"),
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


def callout(label, body, tone="note"):
    tone_class = " ttn-callout--warn" if tone == "warn" else ""
    return mo.Html(
        f'<aside class="ttn-callout{tone_class}">'
        f'<p class="ttn-callout-label">{escape(label)}</p>{mo.md(body).text}</aside>'
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
    return mo.vstack(
        [
            mo.Html(f'<p class="ttn-figure-title">{title}</p>'),
            chart,
            mo.Html(f'<p class="ttn-caption">{caption}</p>'),
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


def status_figure(status, abandoned, as_of):
    """The stacked bar of where every unit stands at the snapshot, with its legend."""
    counts = pd.Series(status).value_counts()
    snapshot = pd.Timestamp(str(as_of))
    total = len(status)
    # Settled outcomes at the two ends, the censored ones (clock still running) between.
    parts = [
        ("Received", "received", True, "Back on the shelf. The only fully observed outcome."),
        (
            "Return in transit",
            "transit",
            False,
            f"Started, not received. Secretly, <b>{abandoned:.0%}</b> of these are abandoned; "
            "the ledger can't say which.",
        ),
        ("Could still return", "eligible", False, "No return yet, still inside the 90-day window."),
        ("Window closed", "expired", True, "Kept for good: past the deadline with no return."),
    ]
    bar = "".join(
        f'<span class="ttn-status-seg ttn-status-seg--{key}" '
        f'style="flex-grow:{counts.get(label, 0)}" title="{label}: {counts.get(label, 0):,}">'
        "</span>"
        for label, key, _, _ in parts
    )
    legend = "".join(
        f'<li class="ttn-status-item ttn-status-item--{key}">'
        f'<span class="ttn-status-key"></span><b>{counts.get(label, 0):,}</b> {label}'
        f'<span class="ttn-status-state">{"settled" if settled else "clock running"}</span>'
        f"<p>{detail}</p></li>"
        for label, key, settled, detail in parts
    )
    return mo.Html(
        '<figure class="ttn-status">'
        f'<p class="ttn-figure-title">Where all {total:,} units sold so far stand at the '
        f"snapshot, {snapshot:%b} {snapshot.day}</p>"
        f'<div class="ttn-status-bar" role="img" aria-label="Status of every unit">{bar}</div>'
        f'<ul class="ttn-status-legend">{legend}</ul>'
        "</figure>"
    )


def stage_cards(family):
    """Side-by-side cards: what each return stage learns and what it is told."""
    share = 1 / (1 + np.exp(-float(family.susceptibility_logit_prior.loc)))

    def stage(name, code, happen, when, rules):
        facets = (
            ("Will it happen?", "learned", happen),
            ("When?", "learned", when),
            ("Rules", "declared", rules),
        )
        rows = "".join(
            f'<dt>{label}<span class="ttn-tag ttn-tag--{kind}">{kind}</span></dt><dd>{text}</dd>'
            for label, kind, text in facets
        )
        return f'<article class="ttn-stage"><h4>{name}</h4><code>{code}</code><dl>{rows}</dl></article>'

    return mo.Html(
        '<div class="ttn-stages">'
        + stage(
            "Return initiated",
            "EventProcess(age_bins=16, deadline_days=90)",
            "A susceptibility logit, intercept plus a product effect: "
            "<code>susceptibility_intercept ~ Normal(0, 2)</code>, "
            "<code>susceptibility_beta ~ Normal(0, 1)</code>.",
            "The default <b>random-walk</b> daily hazard over 16 age bins, plus effects of "
            "product and weekday.",
            "The 90-day policy, inclusive, measured from the sale. Open every day.",
        )
        + stage(
            "Return received",
            "EventProcess(family=WeibullFamily(...), allowed_weekdays=range(5))",
            "A susceptibility logit: <code>susceptibility_intercept ~ "
            f"Normal({float(family.susceptibility_logit_prior.loc):g}, "
            f"{float(family.susceptibility_logit_prior.scale):g})</code>, "
            f"a prior median of {share:.0%} of initiated returns being receivable.",
            "A <b>Weibull</b> delay: <code>scale ~ LogNormal("
            f"{float(family.scale_prior.loc):g}, {float(family.scale_prior.scale):g})</code> "
            f"(median {np.exp(float(family.scale_prior.loc)):.1f} days), "
            f"<code>shape ~ LogNormal({float(family.shape_prior.loc):g}, "
            f"{float(family.shape_prior.scale):g})</code> "
            f"(median {np.exp(float(family.shape_prior.loc)):.1f}). A storm rescales each "
            "day's hazard increment.",
            "Monday to Friday only. No receipt deadline.",
        )
        + "</div>"
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


def scores_view(scores, actual_total):
    """Baseline against model: daily error, CRPS, storm-fortnight error and 28-day total."""
    return (
        ct.CoefTable(scores, rows="Method")
        .estimate("Daily MAE", "mae", fmt=ct.Number(decimals=2))
        .estimate("Daily CRPS", "crps", fmt=ct.Number(decimals=2))
        .estimate("MAE, storm fortnight", "storm", fmt=ct.Number(decimals=2))
        .estimate(
            f"28-day total (actual {actual_total:,})",
            "total",
            ci=("total_lo", "total_hi"),
            fmt=ct.Number(decimals=0),
        )
        .with_theme(TABLE_THEME)
    )


# --- Charts ---------------------------------------------------------------------------


def tldr_chart(forecast, sales_draws, sales_actual, held_out, storms):
    """The opening figure: sales, returns started and returns received, storm shaded."""
    # One shaded rectangle per supplied storm day in the forecast window, labelled once.
    days = pd.DatetimeIndex(forecast.dates)
    stormy = storms(forecast.dates) > 0
    storm = pd.DataFrame({"start": days[stormy], "end": days[stormy] + pd.Timedelta(days=1)})
    first = stormy & ~pd.Series(stormy).shift(1, fill_value=False).to_numpy()
    storm_label = pd.DataFrame({"start": days[first]})

    def panel(draws, actual, color, title, show_axis, with_storm=False):
        frame = fan(forecast.dates, draws, title).assign(actual=actual)
        base = alt.Chart(frame).encode(
            x=alt.X(
                "date:T",
                title=None,
                axis=alt.Axis(format="%a %b %-d", labels=show_axis, ticks=show_axis, labelAngle=0),
            )
        )
        layers = [
            base.mark_area(color=color, opacity=0.14).encode(y="lo90:Q", y2="hi90:Q"),
            base.mark_area(color=color, opacity=0.26).encode(y="lo50:Q", y2="hi50:Q"),
            base.mark_line(color=color, strokeWidth=2).encode(y=alt.Y("mean:Q", title=title)),
            base.mark_circle(color=INK, size=26, opacity=0.85).encode(y="actual:Q"),
        ]
        if with_storm:
            layers = [
                alt.Chart(storm)
                .mark_rect(color=STORM, opacity=0.55)
                .encode(x="start:T", x2="end:T"),
                alt.Chart(storm_label)
                .mark_text(align="left", baseline="top", dx=5, dy=4, fontSize=11, color=AMBER)
                .encode(x="start:T", y=alt.value(0), text=alt.value("storm")),
                *layers,
            ]
        return alt.layer(*layers).properties(height=120, width="container")

    return mo.vstack(
        [
            panel(sales_draws, sales_actual, MUTED, "Units sold", False),
            panel(forecast.initiations, held_out.initiations, AMBER, "Returns started", False),
            panel(forecast.receipts, held_out.receipts, ACCENT, "Returns received", True, True),
        ],
        gap=0,
    )


def daily_history_chart(daily, spans, training_start):
    """Six months of daily counts, storms shaded, the sales training window marked."""
    colors = {"Sales": MUTED, "Returns initiated": AMBER, "Returns received": ACCENT}
    window = pd.DataFrame({"date": [pd.Timestamp(str(training_start))]})

    def panel(name, height, show_axis):
        line = (
            alt.Chart(daily)
            .mark_line(color=colors[name], strokeWidth=1.6)
            .encode(
                x=alt.X(
                    "date:T",
                    title=None,
                    axis=alt.Axis(
                        format="%b", tickCount="month", labels=show_axis, ticks=show_axis
                    ),
                ),
                y=alt.Y(f"{name}:Q", title=name),
                tooltip=[alt.Tooltip("date:T", format="%a %b %-d"), alt.Tooltip(f"{name}:Q")],
            )
        )
        storm = (
            alt.Chart(spans[spans.start <= daily.date.max()])
            .mark_rect(color=AMBER, opacity=0.14)
            .encode(x="start:T", x2="end:T")
        )
        rule = alt.Chart(window).mark_rule(color=MUTED, strokeDash=[3, 3]).encode(x="date:T")
        return (storm + rule + line).properties(height=height, width="container")

    return mo.vstack(
        [
            panel("Sales", 110, False),
            panel("Returns initiated", 110, False),
            panel("Returns received", 130, True),
        ],
        gap=0,
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
        ).properties(height=210, width="container")

    return mo.hstack(
        [
            panel(initiation, AMBER, "Days since sale"),
            panel(receipt, ACCENT, "Days since return initiated"),
        ],
        widths="equal",
        gap=2,
    )


def forecast_chart(forecast, held_out, spans):
    """Returns initiated and received against the held-out days, storm shaded on receipts."""

    def panel(draws, actual, color, title, show_axis, storm):
        frame = fan(forecast.dates, draws, title).assign(actual=actual)
        base = alt.Chart(frame).encode(
            x=alt.X(
                "date:T",
                title=None,
                axis=alt.Axis(format="%a %b %-d", labels=show_axis, ticks=show_axis, labelAngle=0),
            )
        )
        shade = (
            alt.Chart(spans[spans.end > frame.date.min()] if storm else spans[:0])
            .mark_rect(color=AMBER, opacity=0.14)
            .encode(x="start:T", x2="end:T")
        )
        return (
            shade
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
    """Where the returns still owed by past sales are: in transit or not yet started."""
    parts = pd.DataFrame(
        [
            {
                "component": label,
                "mean": float(np.mean(draws)),
                "lo": float(np.quantile(draws, 0.05)),
                "hi": float(np.quantile(draws, 0.95)),
                "truth": eventual_truth[key],
                "color": color,
            }
            for label, key, draws, color in [
                (
                    "Returns already in transit",
                    "open",
                    forecast.expected_open_receipts,
                    ACCENT,
                ),
                (
                    "Sales that haven't started a return",
                    "uninitiated",
                    forecast.expected_uninitiated_receipts,
                    AMBER,
                ),
            ]
        ]
    )
    base = alt.Chart(parts).encode(
        y=alt.Y(
            "component:N", title=None, sort=None, axis=alt.Axis(labelLimit=260, labelFontSize=12)
        )
    )
    return (
        base.mark_bar(height=26, opacity=0.85).encode(
            x=alt.X("mean:Q", title="Expected eventual receipts"),
            color=alt.Color("color:N", scale=None),
        )
        + base.mark_rule(color=INK, strokeWidth=1.5).encode(x="lo:Q", x2="hi:Q")
        + base.mark_tick(color=INK, thickness=3, size=34).encode(x="truth:Q")
    ).properties(height=110, width="container")


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


def baseline_chart(forecast, held_out, baseline, spans):
    """Model mean and 90% band against the weekday-average shortcut and the held-out actuals."""
    frame = fan(forecast.dates, forecast.receipts, MODEL_NAME).assign(
        weekday_avg=baseline.to_numpy(), actual=held_out.receipts.to_numpy()
    )
    x = alt.X("date:T", title=None, axis=alt.Axis(format="%a %b %-d", labelAngle=0))
    lines = pd.concat(
        [
            frame[["date", "mean"]]
            .rename(columns={"mean": "value"})
            .assign(series=f"{MODEL_NAME} mean"),
            frame[["date", "weekday_avg"]]
            .rename(columns={"weekday_avg": "value"})
            .assign(series="Weekday average"),
        ]
    )
    return (
        alt.Chart(spans[spans.end > frame.date.min()])
        .mark_rect(color=AMBER, opacity=0.14)
        .encode(x="start:T", x2="end:T")
        + alt.Chart(frame)
        .mark_area(color=ACCENT, opacity=0.14)
        .encode(x=x, y="lo90:Q", y2="hi90:Q")
        + alt.Chart(lines)
        .mark_line(strokeWidth=2.2)
        .encode(
            x=x,
            y=alt.Y("value:Q", title="Returns received per day"),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    domain=[f"{MODEL_NAME} mean", "Weekday average"], range=[ACCENT, RED]
                ),
            ),
        )
        + alt.Chart(frame).mark_circle(color=INK, size=34).encode(x=x, y="actual:Q")
    ).properties(height=240, width="container")
