# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#     "marimo>=0.23.15",
#     "altair>=5.5",
#     "coeftable>=0.12.1",
#     "numpy>=2.0",
#     "pandas>=2.2",
#     "jax>=0.10.0,<0.11.3",
#     "numpyro>=0.22.0",
#     "numpyro-forecast>=0.4.0,<0.5",
#     "ttenet @ git+https://github.com/kylejcaron/ttenet@84b67dd52a63eefe21b8b0841fcf9ef076c30770",
#     "anywidget>=0.9.18",
# ]
# [tool.marimo.display]
# theme = "light"
# ///
import marimo

__generated_with = "0.25.0"
app = marimo.App(
    width="full",
    css_file="retail_returns_blog.css",
    app_title="Returns you haven't seen yet",
)


@app.cell
def _():
    # Kept alone so the page styling below still renders if a heavier import fails.
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import dataclasses
    import inspect
    import textwrap
    from html import escape

    import altair as alt
    import coeftable as ct
    import jax
    import jax.numpy as jnp
    import numpy as np
    import numpyro
    import numpyro.distributions as dist
    import numpyro_forecast
    import pandas as pd
    from numpyro_forecast import Horizon, draw_posterior, innovations, predict

    import ttenet
    from ttenet import RetailData, SalesForecast, TimingInputs, date_grid

    return (
        Horizon,
        RetailData,
        SalesForecast,
        TimingInputs,
        alt,
        ct,
        dataclasses,
        date_grid,
        dist,
        draw_posterior,
        escape,
        innovations,
        inspect,
        jax,
        jnp,
        np,
        numpyro,
        numpyro_forecast,
        pd,
        predict,
        textwrap,
        ttenet,
    )


@app.cell(hide_code=True)
def _():
    # The interactive figures live beside this notebook: Python computes, they only draw.
    from retail_returns_widgets import (
        ChainStepper,
        PurchaseStory,
        ScenarioForecast,
    )

    return ChainStepper, PurchaseStory, ScenarioForecast


@app.cell(hide_code=True)
def _(Horizon, date_grid, dist, jnp, np, numpyro, pd, predict):
    # The synthetic world, copied from examples/retail_returns.py so this notebook is one
    # file. One difference: receipts here follow the same discretized Weibull law the
    # receipt stage fits, so every fitted receipt parameter has a true value to check.

    # The true receipt law. An initiated return is receivable with probability
    # RECEIPT_RECEIVABLE; a receivable one arrives at age a (days since initiation) with
    # cumulative-hazard increment ((a+1)/scale)^shape - (a/scale)^shape, scaled by
    # exp(RECEIPT_STORM) on storm days. Weekends accrue no hazard: the warehouse is closed.
    RECEIPT_SCALE, RECEIPT_SHAPE, RECEIPT_STORM, RECEIPT_RECEIVABLE = 6.0, 1.8, -1.5, 0.8

    def sigmoid(value):
        return 1 / (1 + np.exp(-value))

    def storms(days):
        """An explicitly supplied historical/future weather scenario, not a forecast."""
        age = (np.asarray(days, dtype="datetime64[D]") - np.datetime64("2026-01-01")).astype(int)
        return (
            ((age >= 45) & (age <= 49))
            | ((age >= 125) & (age <= 129))
            | ((age >= 185) & (age <= 189))
        ).astype(float)

    def simulate_retail(seed=21, training_days=180, horizon=28, volume=1.0):
        """Simulate unit histories; ``volume`` scales both products' daily sales rates."""
        rng = np.random.default_rng(seed)
        start = np.datetime64("2026-01-01")
        sales_days = date_grid(start, start + np.timedelta64(training_days + horizon - 1, "D"))
        weekday = pd.DatetimeIndex(sales_days).dayofweek.to_numpy()
        season = np.sin(2 * np.pi * weekday / 7)
        rates = volume * np.exp(np.log([1.2, 0.8]) + season[:, None] * np.array([0.3, -0.2]))
        daily_sales = rng.poisson(rates)
        rows = []
        for day_index, date in enumerate(sales_days):
            for product in range(2):
                for _ in range(daily_sales[day_index, product]):
                    initiated = np.datetime64("NaT", "D")
                    received = np.datetime64("NaT", "D")
                    susceptible = rng.random() < sigmoid(-0.8 + 0.45 * product)
                    if susceptible:
                        for age in range(91):
                            day = date + np.timedelta64(age, "D")
                            week = pd.Timestamp(day).dayofweek
                            hazard = sigmoid(
                                -2.5
                                + 0.35 * np.cos(min(age, 15) / 5)
                                - 0.25 * product
                                + 0.3 * np.sin(2 * np.pi * week / 7)
                            )
                            if rng.random() < hazard:
                                initiated = day
                                break
                    abandoned = False
                    if not np.isnat(initiated):
                        abandoned = rng.random() >= RECEIPT_RECEIVABLE
                        if not abandoned:
                            # Continue until receipt, not until the sale's policy expires.
                            age = 0
                            while np.isnat(received):
                                day = initiated + np.timedelta64(age, "D")
                                if pd.Timestamp(day).dayofweek < 5:
                                    increment = (
                                        ((age + 1) / RECEIPT_SCALE) ** RECEIPT_SHAPE
                                        - (age / RECEIPT_SCALE) ** RECEIPT_SHAPE
                                    ) * np.exp(RECEIPT_STORM * float(storms([day])[0]))
                                    if rng.random() < 1 - np.exp(-increment):
                                        received = day
                                age += 1
                    rows.append(
                        {
                            "item_id": f"sale_{len(rows)}",
                            "sale_date": date,
                            "initiation_date": initiated,
                            "receipt_date": received,
                            "product": float(product),
                            "abandoned_truth": abandoned,
                        }
                    )
        return pd.DataFrame(rows), daily_sales

    def initiation_covariates(frame, calendar):
        """Product attributes and calendar features with fixed meanings across windows."""
        product = frame["product"].to_numpy(float)
        weekday = pd.DatetimeIndex(calendar).dayofweek.to_numpy()
        features = np.empty((len(frame), len(calendar), 2))
        features[:, :, 0] = product[:, None]
        features[:, :, 1] = np.sin(2 * np.pi * weekday / 7)[None, :]
        return {"features": features, "susceptibility_features": product[:, None]}

    def receipt_covariates(frame, calendar):
        return {
            "features": np.broadcast_to(
                storms(calendar)[None, :, None],
                (len(frame), len(calendar), 1),
            ),
        }

    def sales_covariates(calendar):
        weekday = pd.DatetimeIndex(calendar).dayofweek.to_numpy()
        return jnp.asarray(np.sin(2 * np.pi * weekday / 7)[:, None])

    def sales_model(covariates, data=None):
        h = Horizon.from_data(covariates, data)
        log_rate = numpyro.sample("log_rate", dist.Normal(0, 0.6).expand([2]).to_event(1))
        weekday_effect = numpyro.sample(
            "weekday_effect", dist.Normal(0, 0.4).expand([2]).to_event(1)
        )
        eta = log_rate + covariates * weekday_effect
        predict(h, lambda value: dist.Poisson(jnp.exp(value)), eta)

    return (
        RECEIPT_RECEIVABLE,
        RECEIPT_SCALE,
        RECEIPT_SHAPE,
        RECEIPT_STORM,
        initiation_covariates,
        receipt_covariates,
        sales_covariates,
        sales_model,
        simulate_retail,
        storms,
    )


@app.cell(hide_code=True)
def _(alt, ct, escape, mo):
    INK, MUTED, ACCENT, AMBER, RED = "#292d26", "#62695d", "#42644d", "#956017", "#a23c3c"
    RULE, GRID, PAPER, SLATE, STORM = "#dcded1", "#ecefe4", "#fffef9", "#4c5a6b", "#e9d6b0"

    @alt.theme.register("ttenet_paper", enable=True)
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
            style = (
                f"background: repeating-linear-gradient(90deg, {color} 0 5px, transparent 5px 8px)"
            )
        else:
            style = f"background: {color}"
        return f'<span class="ttn-swatch" style="{style}"></span>'

    return (
        ACCENT,
        AMBER,
        INK,
        MUTED,
        RED,
        RULE,
        SLATE,
        STORM,
        TABLE_THEME,
        callout,
        cards,
        figure,
        section,
        swatch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.Html("""
        <header class="ttn-nav">
          <a class="ttn-brand" href="#top">ttenet</a>
          <nav aria-label="Sections">
            <a href="#tldr">TL;DR</a>
            <a href="#data">Data</a>
            <a href="#model">Model</a>
            <a href="#fit">Fit</a>
            <a href="#forecast">Forecast</a>
            <a href="#scenarios">What if?</a>
            <a href="#sales-model">Swap &amp; compare</a>
            <a href="#wrap-up">Wrap-up</a>
          </nav>
        </header>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html("""
        <div id="top" class="ttn-hero">
          <p class="ttn-kicker">Forecasting · Time-to-event · Supply chain · Bayesian modeling</p>
          <h1 class="ttn-title">Returns you haven't seen yet</h1>
          <p class="ttn-dek">
            Forecasting retail returns as a chain of forecasts. Sales feed return
            initiations, which feed warehouse receipts. Fit the chain once, then push any
            sales forecast through it, uncertainty and all.
          </p>
          <ul class="ttn-byline">
            <li><b>Kyle Caron</b></li>
            <li>September 2026</li>
            <li>Built with <b>ttenet</b>, NumPyro &amp; marimo</li>
          </ul>
        </div>
    """)
    return


@app.cell(hide_code=True)
def _(section):
    section(
        "tldr",
        "TL;DR",
        "Returns are a forecast of a forecast",
        "Next week's warehouse receipts depend on the returns customers start this week, "
        "which depend on what sold last month and on what sells this month. Forecast each "
        "step, chain them, and push a sales forecast all the way through.",
    )
    return


@app.cell(hide_code=True)
def _(
    ACCENT,
    AMBER,
    INK,
    MUTED,
    alt,
    fan,
    figure,
    forecast,
    held_out,
    mo,
    sales_actual,
    sales_draws,
    pd,
    STORM,
    storms,
    swatch,
):
    # One shaded rectangle per supplied storm day in the forecast window, labelled once.
    _days = pd.DatetimeIndex(forecast.dates)
    _stormy = storms(forecast.dates) > 0
    _storm = pd.DataFrame({"start": _days[_stormy], "end": _days[_stormy] + pd.Timedelta(days=1)})
    _first = _stormy & ~pd.Series(_stormy).shift(1, fill_value=False).to_numpy()
    _storm_label = pd.DataFrame({"start": _days[_first]})

    def _panel(draws, actual, color, title, show_axis, storm=False):
        _frame = fan(forecast.dates, draws, title).assign(actual=actual)
        _base = alt.Chart(_frame).encode(
            x=alt.X(
                "date:T",
                title=None,
                axis=alt.Axis(format="%a %b %-d", labels=show_axis, ticks=show_axis, labelAngle=0),
            )
        )
        _layers = [
            _base.mark_area(color=color, opacity=0.14).encode(y="lo90:Q", y2="hi90:Q"),
            _base.mark_area(color=color, opacity=0.26).encode(y="lo50:Q", y2="hi50:Q"),
            _base.mark_line(color=color, strokeWidth=2).encode(y=alt.Y("mean:Q", title=title)),
            _base.mark_circle(color=INK, size=26, opacity=0.85).encode(y="actual:Q"),
        ]
        if storm:
            _layers = [
                alt.Chart(_storm)
                .mark_rect(color=STORM, opacity=0.55)
                .encode(x="start:T", x2="end:T"),
                alt.Chart(_storm_label)
                .mark_text(align="left", baseline="top", dx=5, dy=4, fontSize=11, color=AMBER)
                .encode(x="start:T", y=alt.value(0), text=alt.value("storm")),
                *_layers,
            ]
        return alt.layer(*_layers).properties(height=120, width="container")

    figure(
        mo.vstack(
            [
                _panel(sales_draws, sales_actual, MUTED, "Units sold", False),
                _panel(forecast.initiations, held_out.initiations, AMBER, "Returns started", False),
                _panel(
                    forecast.receipts, held_out.receipts, ACCENT, "Returns received", True, True
                ),
            ],
            gap=0,
        ),
        "One sales forecast, propagated twice",
        f"{swatch(MUTED)}The sales forecast for the next four weeks (top). Every simulated "
        f"sale starts a return clock {swatch(AMBER)}(middle), and every started return, "
        "including those started before today, starts a shipping clock "
        f"{swatch(ACCENT)}(bottom). Bands are 50% and 90% intervals; {swatch(INK)}dots are "
        "what actually happened. Receipts are zero on weekends and slow to a trickle during "
        f"the {swatch(STORM, block=True, opacity=0.55)}early-July storm, a weather input the "
        "forecast is given.",
    )
    return


@app.cell(hide_code=True)
def _(cards, forecast, held_out, np, sales_actual, sales_draws):
    def _interval(draws):
        return (
            f"{draws.mean():,.0f}"
            f"<small>[{np.quantile(draws, 0.05):,.0f}–{np.quantile(draws, 0.95):,.0f}]</small>"
        )

    cards(
        [
            (
                "Units sold, next 28 days",
                _interval(sales_draws.sum(axis=1)),
                f"The sales forecast going in. Actual: <b>{sales_actual.sum():,}</b>.",
                None,
            ),
            (
                "Returns started",
                _interval(forecast.initiations.sum(axis=1)),
                f"Propagated once. Actual: <b>{held_out.initiations.sum():,}</b>.",
                "amber",
            ),
            (
                "Returns received",
                _interval(forecast.receipts.sum(axis=1)),
                f"Propagated twice. Actual: <b>{held_out.receipts.sum():,}</b>.",
                "accent",
            ),
            (
                "Still owed by past sales",
                f"{forecast.expected_existing_receipts.mean():,.0f}",
                "Eventual receipts from units already sold, whenever they arrive.",
                None,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The usual shortcut multiplies a sales forecast by a return rate and shifts it by a
    typical lag. It holds until something changes: a promo, a storm, a new policy, or a
    quarter whose recent sales haven't had time to come back yet. This post treats returns
    as a **chain of forecasts** instead, with [ttenet](https://github.com/kylejcaron/ttenet)
    doing the plumbing: one call fits the chain, one call pushes any sales forecast through
    it with uncertainty intact, and a storm scenario or a different sales model is one
    argument.
    """)
    return


@app.cell(hide_code=True)
def _(section):
    section(
        "data",
        "Part 1",
        "Looking at the data",
        "A retailer sells two products and offers a 90-day return window. The only thing "
        "the business actually records is a ledger: one row per unit sold, with the dates "
        "things happened to it.",
        points=(
            "Follow one purchase through both clocks",
            "What the ledger can and cannot see yet",
            "Why the return-rate shortcut breaks",
        ),
    )
    return


@app.cell
def _(np, simulate_retail):
    SEED = 21
    TRAINING_DAYS, HORIZON = 180, 28
    START = np.datetime64("2026-01-01")
    TRAINING_START = np.datetime64("2026-01-31")
    AS_OF = START + np.timedelta64(TRAINING_DAYS - 1, "D")

    # A simulated world: we get to see every unit's full future, the model never does.
    truth, _ = simulate_retail(SEED, TRAINING_DAYS, HORIZON, volume=10.0)
    ledger = truth.drop(columns="abandoned_truth")
    return AS_OF, HORIZON, SEED, START, TRAINING_START, ledger, truth


@app.cell
def _(AS_OF, RetailData, TRAINING_START, date_grid, ledger, mo):
    # The ledger of units becomes the model's input.
    data = RetailData.from_units(
        ledger,
        as_of=AS_OF,  # end-of-day snapshot; anything later is hidden
        calendar=date_grid(TRAINING_START, AS_OF),  # the sales observation window
        group_by=["product"],
    )
    mo.show_code(position="above")
    return (data,)


@app.cell(hide_code=True)
def _(AS_OF, data, np):
    # RetailData imposes no deadline; the 90-day policy belongs to the initiation process.
    _within_policy = (AS_OF - data.units.sale_date.to_numpy().astype("datetime64[D]")).astype(
        int
    ) < 90  # as_of is end-of-day and day 90 is inclusive: age 90 has no opportunity left
    unit_status = np.select(
        [
            data.units.receipt_date.notna(),
            data.units.initiation_date.notna(),
            _within_policy,
        ],
        ["Received", "Return in transit", "Could still return"],
        default="Window closed",
    )
    return (unit_status,)


@app.cell(hide_code=True)
def _(AS_OF, PurchaseStory, data, mo, pd, storms, truth):
    from retail_returns_story import purchase_story

    _story = purchase_story(truth, AS_OF, storms)
    _sold = pd.Timestamp(_story["sale"]).strftime("%A, %B %-d")
    mo.vstack(
        [
            mo.md(
                f"""
    ### Follow one purchase

    Each unit runs on two clocks. The first starts at the sale and can stop when the
    customer starts a return, which the policy allows up to day 90. The second starts at
    that initiation and stops when the parcel is scanned in at the warehouse; it has no
    deadline. {_story["cohort_units"]} units sold on {_sold}; three of them are below.
    Choose one and move through time. The figure shows only what the ledger had recorded
    by the chosen day.
    """
            ),
            mo.ui.anywidget(PurchaseStory(data=_story)),
            # A published page has no Python to page through data, so show a fixed sample.
            mo.accordion(
                {
                    f"A sample of the ledger: the first 25 of {len(data.units):,} units": mo.ui.table(
                        data.units[
                            ["item_id", "product", "sale_date", "initiation_date", "receipt_date"]
                        ].head(25),
                        page_size=25,
                        selection=None,
                    )
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(AS_OF, data, mo, pd, truth, unit_status):
    _counts = pd.Series(unit_status).value_counts()
    _open_ids = data.units.item_id[unit_status == "Return in transit"]
    _abandoned = truth.set_index("item_id").abandoned_truth.loc[_open_ids].mean()
    _total = len(unit_status)
    # Settled outcomes at the two ends, the censored ones (clock still running) between.
    _parts = [
        ("Received", "received", True, "Back on the shelf. The only fully observed outcome."),
        (
            "Return in transit",
            "transit",
            False,
            f"Started, not received. Secretly, <b>{_abandoned:.0%}</b> of these are abandoned; "
            "the ledger can't say which.",
        ),
        ("Could still return", "eligible", False, "No return yet, still inside the 90-day window."),
        ("Window closed", "expired", True, "Kept for good: past the deadline with no return."),
    ]
    _bar = "".join(
        f'<span class="ttn-status-seg ttn-status-seg--{key}" '
        f'style="flex-grow:{_counts.get(label, 0)}" title="{label}: {_counts.get(label, 0):,}">'
        "</span>"
        for label, key, _, _ in _parts
    )
    _legend = "".join(
        f'<li class="ttn-status-item ttn-status-item--{key}">'
        f'<span class="ttn-status-key"></span><b>{_counts.get(label, 0):,}</b> {label}'
        f'<span class="ttn-status-state">{"settled" if settled else "clock running"}</span>'
        f"<p>{detail}</p></li>"
        for label, key, settled, detail in _parts
    )
    mo.Html(
        '<figure class="ttn-status">'
        f'<p class="ttn-figure-title">Where all {_total:,} units sold so far stand at the '
        f"snapshot, {pd.Timestamp(str(AS_OF)):%b %-d}</p>"
        f'<div class="ttn-status-bar" role="img" aria-label="Status of every unit">{_bar}</div>'
        f'<ul class="ttn-status-legend">{_legend}</ul>'
        "</figure>"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The middle of the bar is **censored**: the clock is still running and the ledger
    cannot say where it will stop. The in-transit pile is the sneakiest. In this world
    about one in five initiated returns is never shipped back, and the ledger records no
    "abandoned" event, so the model has to learn that share from how long open returns
    stay open. The same events as daily time series:
    """)
    return


@app.cell(hide_code=True)
def _(AS_OF, START, data, np, pd, storms):
    _days = pd.date_range(str(START), str(AS_OF), freq="D")
    daily_history = pd.DataFrame(
        {
            "date": _days,
            "Sales": data.units.groupby("sale_date").size().reindex(_days, fill_value=0).values,
            "Returns initiated": data.units.groupby("initiation_date")
            .size()
            .reindex(_days, fill_value=0)
            .values,
            "Returns received": data.units.groupby("receipt_date")
            .size()
            .reindex(_days, fill_value=0)
            .values,
        }
    )

    # Contiguous storm spans over the full simulated period, for shading.
    _all_days = pd.date_range(str(START), str(AS_OF + np.timedelta64(28, "D")), freq="D")
    _flag = storms(_all_days.values.astype("datetime64[D]")) > 0
    _edges = np.flatnonzero(np.diff(np.r_[0, _flag.astype(int), 0]))
    storm_spans = pd.DataFrame(
        {
            "start": _all_days[_edges[::2]],
            "end": _all_days[_edges[1::2] - 1] + pd.Timedelta(days=1),
        }
    )
    return daily_history, storm_spans


@app.cell(hide_code=True)
def _(
    ACCENT,
    AMBER,
    MUTED,
    TRAINING_START,
    alt,
    daily_history,
    figure,
    mo,
    pd,
    storm_spans,
    swatch,
):
    _colors = {"Sales": MUTED, "Returns initiated": AMBER, "Returns received": ACCENT}
    _window = pd.DataFrame({"date": [pd.Timestamp(str(TRAINING_START))]})

    def _panel(name, height, show_axis):
        _line = (
            alt.Chart(daily_history)
            .mark_line(color=_colors[name], strokeWidth=1.6)
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
        _storm = (
            alt.Chart(storm_spans[storm_spans.start <= daily_history.date.max()])
            .mark_rect(color=AMBER, opacity=0.14)
            .encode(x="start:T", x2="end:T")
        )
        _rule = alt.Chart(_window).mark_rule(color=MUTED, strokeDash=[3, 3]).encode(x="date:T")
        return (_storm + _rule + _line).properties(height=height, width="container")

    figure(
        mo.vstack(
            [
                _panel("Sales", 110, False),
                _panel("Returns initiated", 110, False),
                _panel("Returns received", 130, True),
            ],
            gap=0,
        ),
        "Six months of the ledger, as daily counts",
        "The dotted line marks where the <b>sales</b> training window starts (Jan 31); units "
        "sold before it keep their full histories and clocks. Shaded bands "
        f"{swatch(AMBER, block=True)}are storms: receipts stall, then catch up. Receipts are "
        "exactly zero on weekends because the warehouse is closed.",
    )
    return


@app.cell(hide_code=True)
def _(cohort_rates, mo):
    _recent = cohort_rates.tail(4)
    mo.md(rf"""
    ### Why the return-rate shortcut breaks

    The shortcut needs a return rate, and the obvious estimate is returns so far divided
    by units sold. For the last four weeks of sales that comes out
    **{(_recent.eventual - _recent.observed).mean() * 100:.0f} points** below the rate those
    same units eventually reach. Recent units haven't had time: a unit sold three weeks ago
    with no return yet has only told us "not yet", and the average treats it as "never".
    """)
    return


@app.cell(hide_code=True)
def _(INK, RED, SLATE, alt, cohort_rates, figure, pd, swatch):
    _open = pd.DataFrame(
        {
            "start": [cohort_rates.window_opens.iloc[0]],
            "end": [cohort_rates.week.max() + pd.Timedelta(days=6)],
        }
    )
    _base = alt.Chart(cohort_rates).encode(
        x=alt.X("week:T", title="Week the unit was sold", axis=alt.Axis(format="%b %-d"))
    )
    _window = (
        alt.Chart(_open).mark_rect(color="#ecefe4", opacity=0.7).encode(x="start:T", x2="end:T")
    )
    _label = (
        alt.Chart(_open)
        .mark_text(align="left", dx=8, dy=12, color="#62695d", fontSize=11)
        .encode(x="start:T", y=alt.value(0), text=alt.value("Still inside the 90-day window"))
    )
    _band = _base.mark_area(color=SLATE, opacity=0.16).encode(y="model_lo:Q", y2="model_hi:Q")
    _lines = (
        alt.Chart(
            cohort_rates.melt(
                id_vars="week",
                value_vars=["eventual", "observed", "model"],
                var_name="series",
                value_name="rate",
            ).replace(
                {
                    "series": {
                        "eventual": "Eventually returned (truth)",
                        "observed": "Returned so far",
                        "model": "ttenet estimate",
                    }
                }
            )
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
                scale=alt.Scale(
                    domain=[
                        "Eventually returned (truth)",
                        "Returned so far",
                        "ttenet estimate",
                    ],
                    range=[INK, RED, SLATE],
                ),
                legend=alt.Legend(orient="top", title=None, symbolType="stroke"),
            ),
            strokeDash=alt.StrokeDash(
                "series:N",
                scale=alt.Scale(
                    domain=[
                        "Eventually returned (truth)",
                        "Returned so far",
                        "ttenet estimate",
                    ],
                    range=[[5, 4], [1, 0], [1, 0]],
                ),
                legend=None,
            ),
        )
    )
    figure(
        (_window + _label + _band + _lines).properties(height=320, width="container"),
        "Recent sales look like they never get returned",
        f"{swatch(RED)}<b>Returns so far</b> collapse for recent cohorts: those units simply "
        "haven't had time. The truth (dashed) doesn't fall at all. ttenet's estimate "
        f"{swatch(SLATE)} adds, for every still-eligible unit, the probability it starts a "
        "return before its deadline given that it hasn't yet. Band: 90% posterior interval.",
    )
    return


@app.cell(hide_code=True)
def _(section):
    section(
        "model",
        "Part 2",
        "Three forecasts, chained",
        "Each step gets its own model: a count forecast for sales, and a time-to-event "
        "forecast for each return stage. The output of one is the input of the next.",
        points=(
            "The model, in one cell",
            "What each stage learns and what it is told",
            "How censoring enters the likelihood",
        ),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The model, in one cell

    Sales are a NumPyro count model (`sales_model`, in the tabs below). Each return stage
    is an `EventProcess`: initiation takes the defaults, a flexible random-walk hazard
    with a 90-day deadline; receipt uses the packaged Weibull family on a warehouse that
    opens Monday to Friday. `dist` is `numpyro.distributions`.
    """)
    return


@app.cell
def _(dist, mo, sales_model):
    from ttenet import CountProcess, EventProcess, RetailReturnModel, WeibullFamily

    model = RetailReturnModel(
        sales=CountProcess(model=sales_model),
        # age_bins=16: one learned hazard level for each day of age 0-14, and one shared
        # level for every age from 15 on. It is the number of parameters, not a max delay.
        initiation=EventProcess(age_bins=16, deadline_days=90),
        receipt=EventProcess(
            family=WeibullFamily(
                scale_prior=dist.LogNormal(1.8, 0.3),
                shape_prior=dist.LogNormal(0.7, 0.2),
                susceptibility_logit_prior=dist.Normal(1.0, 1.0),
            ),
            allowed_weekdays=range(5),  # Mon-Fri only
        ),
    )
    mo.show_code(position="above")
    return WeibullFamily, model


@app.cell(hide_code=True)
def _(mo, model, np):
    _family = model.receipt.family
    _share = 1 / (1 + np.exp(-float(_family.susceptibility_logit_prior.loc)))

    def _stage(name, code, happen, when, rules):
        _facets = (
            ("Will it happen?", "learned", happen),
            ("When?", "learned", when),
            ("Rules", "declared", rules),
        )
        _rows = "".join(
            f'<dt>{label}<span class="ttn-tag ttn-tag--{kind}">{kind}</span></dt><dd>{text}</dd>'
            for label, kind, text in _facets
        )
        return (
            f'<article class="ttn-stage"><h4>{name}</h4><code>{code}</code>'
            f"<dl>{_rows}</dl></article>"
        )

    mo.Html(
        '<div class="ttn-stages">'
        + _stage(
            "Return initiated",
            "EventProcess(age_bins=16, deadline_days=90)",
            "A susceptibility logit, intercept plus a product effect: "
            "<code>susceptibility_intercept ~ Normal(0, 2)</code>, "
            "<code>susceptibility_beta ~ Normal(0, 1)</code>.",
            "The default <b>random-walk</b> daily hazard over 16 age bins, plus effects of "
            "product and weekday.",
            "The 90-day policy, inclusive, measured from the sale. Open every day.",
        )
        + _stage(
            "Return received",
            "EventProcess(family=WeibullFamily(...), allowed_weekdays=range(5))",
            "A susceptibility logit: <code>susceptibility_intercept ~ "
            f"Normal({float(_family.susceptibility_logit_prior.loc):g}, "
            f"{float(_family.susceptibility_logit_prior.scale):g})</code>, "
            f"a prior median of {_share:.0%} of initiated returns being receivable.",
            "A <b>Weibull</b> delay: <code>scale ~ LogNormal("
            f"{float(_family.scale_prior.loc):g}, {float(_family.scale_prior.scale):g})</code> "
            f"(median {np.exp(float(_family.scale_prior.loc)):.1f} days), "
            f"<code>shape ~ LogNormal({float(_family.shape_prior.loc):g}, "
            f"{float(_family.shape_prior.scale):g})</code> "
            f"(median {np.exp(float(_family.shape_prior.loc)):.1f}). A storm rescales each "
            "day's hazard increment.",
            "Monday to Friday only. No receipt deadline.",
        )
        + "</div>"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each stage separates what is **learned** from what is **declared**: whether a unit
    ever does it and when are estimated from the ledger; the deadline and the opening
    days are rules you state. For each unit, four things happen (or don't):

    1. **Sales.** Product $p$ sells $\text{sales}_{t,p} \sim \text{Poisson}\big(\exp(\alpha_p + \gamma_p \sin(2\pi\,\text{weekday}_t / 7))\big)$ units on day $t$.
    2. **Will this unit start a return?** A sale is *susceptible* with probability $\pi^{\text{init}}_p = \sigma(a + b\,x_p)$; the rest never start one, however long you wait. The ledger never labels which units are which.
    3. **When?** A susceptible unit of age $a$ starts a return on day $t$ with hazard $h(a, t) = \sigma\big(\ell_{\min(a,\,15)} + \beta^\top x_t\big)$, through day 90 of the policy. The $\ell$'s are a regularized random walk over age, so nothing forces a parametric shape.
    4. **Will it arrive, and when?** With probability $\pi^{\text{rec}} = \sigma(c)$ an initiated return is receivable. Its arrival follows a Weibull clock, $S(a) = \exp\!\big(-(a/\lambda)^k\big)$, discretized to days; a storm multiplies each day's hazard increment by $e^{\beta x_t}$, and weekends are exactly zero. No deadline.
    """)
    return


@app.cell(hide_code=True)
def _(callout):
    callout(
        "Key point",
        """Propagation happens per draw. A sale simulated for Jul 3 in posterior draw 17
        starts its own return clock in draw 17, with draw 17's parameters, and so on down
        the chain. That is why any sales forecast can be swapped in and pushed through with
        its uncertainty intact, and why every stage's counts add up exactly.""",
    )
    return


@app.cell(hide_code=True)
def _(callout):
    callout(
        "How censoring is handled",
        r"""A unit quiet for $a$ days contributes $(1-\pi) + \pi\,S(a)$ to the likelihood:
        either it is not susceptible, or it is and hasn't returned yet. Given the silence,
        the chance it is still susceptible is $\pi S(a) / \big((1-\pi) + \pi S(a)\big)$,
        which falls the longer the silence lasts but never reaches a verdict. That is what
        fixes the collapsing line in Part 1.""",
    )
    return


@app.cell(hide_code=True)
def _(inspect, mo, textwrap):
    def source_md(*objects):
        """Markdown code blocks holding the actual source of the given functions."""
        return mo.md(
            "\n\n".join(
                f"```python\n{textwrap.dedent(inspect.getsource(obj)).rstrip()}\n```"
                for obj in objects
            )
        )

    return (source_md,)


@app.cell(hide_code=True)
def _(
    WeibullFamily,
    initiation_covariates,
    mo,
    receipt_covariates,
    sales_model,
    source_md,
    ttenet,
):
    _convolution = mo.md(r"""
    A sale does not promise a return on one particular day. It spreads probability across
    the days ahead. A **survival kernel** captures **susceptible × still waiting × event today**:

    $$
    K(s,t) = \pi\,S(s,t^-)\,h(t-s,t).
    $$

    The mass left outside the forecast window means *not by then*, not necessarily
    *never*. Write the initiation kernel as $K_0$ and the receipt kernel as $K_1$.
    For a new, homogeneous cohort, with parameters fixed, their composition is a
    **survival convolution**:

    $$
    (K_0 K_1)(s,t) = \sum_u K_0(s,u)\,K_1(u,t).
    $$

    Ordinary convolution shifts one lag curve along time. Here, storms and weekends can
    change the curve for each source date, so the composition runs **within each posterior
    draw**, and simulated paths still allocate whole units and carry their actual parent
    dates forward. The interface does not prescribe a delay family: leaving `family` unset
    gives the default random-walk hazard (initiation here), `WeibullFamily` (receipt here)
    supplies the same kernel from a two-parameter curve, and a custom NumPyro model joins
    through an `EventFamily` wrapper. A family only states how the delay behaves; the
    shared core handles closed days, exposure, the mixture-cure marginalization, dated
    cohorts and integer counts. The wiring is in the *Internals* tab and the README.
    """)

    _sales = mo.vstack(
        [
            mo.md(r"""
    The sales process is an ordinary NumPyro model following the
    [`numpyro_forecast`](https://github.com/juanitorduz/numpyro_forecast) protocol: it
    receives covariates with time on axis `-2`, and `predict` handles the observed prefix
    versus the future suffix. The two functions after it are the event covariate providers
    passed to `fit`.
    """),
            source_md(sales_model, initiation_covariates, receipt_covariates),
            mo.md(r"""
    **The covariate contract.** A provider is called as `provider(frame, calendar)`, where
    `frame` holds the historical units followed by any new sales cohorts and `calendar` the
    days requested, and returns a mapping of up to three arrays:

    - `features`, shape `[unit, day, P]`: time-varying regressors, defined for every
      historical and future day. In the default family they shift the hazard logit; in
      `WeibullFamily` they multiply each day's cumulative-hazard increment by
      $e^{x_t^\top\beta}$. Same array, different effect, so the coefficients are not
      interchangeable between the two.
    - `susceptibility_features`, shape `[unit, Q]`: static attributes known when the
      source event happens. They enter the susceptibility logit, and for units that already
      exist they cannot change in a scenario.
    - `allowed`, an optional boolean mask over days (or units and days) that closes further
      days on top of the process's own `allowed_weekdays`.

    Each feature must keep one meaning across fitting and forecasting windows, which is why
    `product` and the weekday sine are defined by formula rather than by position.
    """),
        ]
    )

    _initiation = mo.vstack(
        [
            mo.md(r"""
    **Initiation** is the default `EventProcess`. It samples `age_scale ~ HalfNormal(1)`,
    `age_init ~ Normal(0, 2)` and standardized steps `age_steps ~ Normal(0, 1)` for the
    random-walk baseline, plus `beta ~ Normal(0, 1)` for the hazard regressors and
    `susceptibility_intercept ~ Normal(0, 2)` and `susceptibility_beta ~ Normal(0, 1)` for
    susceptibility. `default_timing` turns one draw into the law:
    """),
            source_md(ttenet.event_times.default_timing),
            mo.accordion(
                {
                    "The default random-walk prior, in full": source_md(
                        ttenet.models.sample_stage_parameters
                    )
                }
            ),
        ]
    )

    _receipt = mo.vstack(
        [
            mo.md(r"""
    **Receipt** is `WeibullFamily`. Scalar NumPyro priors are sampled at the sites `scale`,
    `shape` and `susceptibility_intercept`. A fixed positive scale or shape (say
    `scale_prior=6.0`) is used as given and creates no site; a fixed susceptibility logit
    can be any finite number. `susceptibility_logit_prior` is on log-odds, not probability.
    Regressors add `beta` for the cumulative-hazard multiplier, and static susceptibility
    features would add `susceptibility_beta`; neither exists unless supplied. This is the
    family's entire model:
    """),
            source_md(WeibullFamily.model),
        ]
    )

    _internals = mo.vstack(
        [
            mo.md(
                "One stage's model: the family's law, masked by the stage's exposure and "
                "closures, observed at the native `EventTime` site through "
                "`numpyro_forecast.predict`. These are private helpers and may change; they "
                "are shown so nothing is hidden."
            ),
            source_md(
                ttenet.network._event_model,
                ttenet.models.observation_kernel,
                ttenet.models._observe,
                ttenet.families._weibull_log_increment,
                ttenet.families._timing_from_log_increment,
            ),
        ]
    )

    mo.vstack(
        [
            mo.md(r"""
    ### Under the hood

    The same model in more detail. Each tab stands on its own; skip them on a first read
    and come back when you want the math or the code.
    """),
            mo.ui.tabs(
                {
                    "Survival convolution": _convolution,
                    "Sales and covariates": _sales,
                    "Initiation": _initiation,
                    "Receipt": _receipt,
                    "Internals": _internals,
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(section):
    section(
        "fit",
        "Part 3",
        "Fitting everything at once",
        "One call fits the sales process and both return stages as a single NumPyro model. "
        "Calendar covariates are passed as providers, so the same function describes history "
        "and the future.",
        points=(
            "One joint fit",
            "Recovered parameters against the simulator's truth",
            "Fitted timing curves",
        ),
    )
    return


@app.cell
def _(
    SEED,
    data,
    initiation_covariates,
    mo,
    model,
    receipt_covariates,
    sales_covariates,
):
    fitted = model.fit(
        data,
        covariates={
            "sales": sales_covariates(data.calendar),
            "initiations": initiation_covariates,
            "receipts": receipt_covariates,
        },
        mode="joint",
        num_steps=2000,
        num_samples=200,
        seed=SEED,
    )
    mo.show_code(position="above")
    return (fitted,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Because this world is simulated, every fitted parameter can be checked against its true
    value, including what the ledger never shows directly: the share of units susceptible
    to returning and the share of initiated returns that ever arrive. The bars put every
    row on one scale, the posterior divided by the truth: a bar that covers 1 contains the
    truth, and a narrow bar is a precise estimate.
    """)
    return


@app.cell(hide_code=True)
def _(
    RECEIPT_RECEIVABLE,
    RECEIPT_SCALE,
    RECEIPT_SHAPE,
    RECEIPT_STORM,
    SLATE,
    TABLE_THEME,
    ct,
    data,
    dataclasses,
    fitted,
    np,
    pd,
):
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    _ip, _rp = fitted.initiation_fit.parameters, fitted.receipt_fit.parameters
    _posterior = fitted.network.posterior
    _order = list(data.groups["product"].to_numpy(int))  # series order is explicit
    _a, _b = _order.index(0), _order.index(1)
    _si, _sb = np.asarray(_ip.susceptibility_intercept), np.asarray(_ip.susceptibility_beta)[:, 0]
    _beta_i = np.asarray(_ip.beta)
    _rate = np.exp(np.asarray(_posterior["sales/log_rate"]))
    _weekday = np.asarray(_posterior["sales/weekday_effect"])

    _rows = [
        ("Sales", "Daily sales rate, product A", _rate[:, _a], 12.0),
        ("Sales", "Daily sales rate, product B", _rate[:, _b], 8.0),
        ("Sales", "Weekday effect, product A", _weekday[:, _a], 0.3),
        ("Sales", "Weekday effect, product B", _weekday[:, _b], -0.2),
        ("Return initiation", "Susceptible, product A", _sigmoid(_si), _sigmoid(-0.8)),
        ("Return initiation", "Susceptible, product B", _sigmoid(_si + _sb), _sigmoid(-0.35)),
        ("Return initiation", "Hazard shift, product B", _beta_i[:, 0], -0.25),
        ("Return initiation", "Hazard shift, weekday", _beta_i[:, 1], 0.3),
        (
            "Return receipt",
            "Initiated return ever arrives",
            _sigmoid(np.asarray(_rp["susceptibility_intercept"])),
            RECEIPT_RECEIVABLE,
        ),
        ("Return receipt", "Weibull scale (days)", np.asarray(_rp["scale"]), RECEIPT_SCALE),
        ("Return receipt", "Weibull shape", np.asarray(_rp["shape"]), RECEIPT_SHAPE),
        (
            "Return receipt",
            "Storm effect (log-multiplier of each day's hazard increment)",
            np.asarray(_rp["beta"])[:, 0],
            RECEIPT_STORM,
        ),
    ]
    recovery = pd.DataFrame(
        [
            {
                "stage": stage,
                "Parameter": name,
                "estimate": float(np.mean(draws)),
                "lower": float(np.quantile(draws, 0.05)),
                "upper": float(np.quantile(draws, 0.95)),
                "truth": truth_value,
            }
            for stage, name, draws, truth_value in _rows
        ]
    )
    # Every truth here is nonzero, so the ratio puts all rows on one axis; dividing by a
    # negative truth flips the interval's ends, hence the min/max.
    _lo, _hi = recovery.lower / recovery.truth, recovery.upper / recovery.truth
    recovery = recovery.assign(
        ratio=recovery.estimate / recovery.truth,
        ratio_lower=np.minimum(_lo, _hi),
        ratio_upper=np.maximum(_lo, _hi),
    )
    (
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
    return (recovery,)


@app.cell(hide_code=True)
def _(
    ACCENT,
    AMBER,
    INK,
    RECEIPT_SCALE,
    RECEIPT_SHAPE,
    TimingInputs,
    alt,
    figure,
    fitted,
    jax,
    jnp,
    mo,
    np,
    pd,
    swatch,
):
    def _curve(fit, true_hazard, stage):
        # The fitted stage replays its own timing law, the random walk or the Weibull, at
        # every posterior draw: a neutral day, a clear day, product A.
        _named = fit.parameters._asdict() if fit.family is None else dict(fit.parameters)
        _ages = np.arange(0, 31)
        _inputs = TimingInputs(
            ages=jnp.asarray(_ages)[:, None],
            features=jnp.zeros((len(_ages), 1, _named["beta"].shape[-1])),
            susceptibility_features=jnp.zeros(
                (
                    1,
                    _named["susceptibility_beta"].shape[-1]
                    if "susceptibility_beta" in _named
                    else 0,
                )
            ),
        )
        _log_hazard = jax.vmap(lambda draw: fit.timing(_inputs, draw=draw).timing.log_hazard)(
            jnp.arange(fit.draws)
        )
        _hazard = np.exp(np.asarray(_log_hazard))[:, :, 0]  # [draw, age]
        _truth = true_hazard(_ages)
        return pd.DataFrame(
            {
                "age": _ages,
                "mean": _hazard.mean(0),
                "lo": np.quantile(_hazard, 0.05, axis=0),
                "hi": np.quantile(_hazard, 0.95, axis=0),
                "truth": _truth,
                "stage": stage,
            }
        )

    _init = _curve(
        fitted.initiation_fit,
        lambda a: 1 / (1 + np.exp(2.5 - 0.35 * np.cos(np.minimum(a, 15) / 5))),
        "Initiation: days since sale",
    )

    def _true_receipt(a):
        # The simulator's own Weibull bin [a, a + 1) on a clear, open day.
        _scaled = np.asarray(a, float) / RECEIPT_SCALE
        _increment = (_scaled + 1 / RECEIPT_SCALE) ** RECEIPT_SHAPE - _scaled**RECEIPT_SHAPE
        return 1 - np.exp(-_increment)

    _rec = _curve(fitted.receipt_fit, _true_receipt, "Receipt: days since initiation")

    def _panel(frame, color, title):
        _base = alt.Chart(frame).encode(x=alt.X("age:Q", title=title))
        return (
            _base.mark_area(color=color, opacity=0.18).encode(y="lo:Q", y2="hi:Q")
            + _base.mark_line(color=color, strokeWidth=2.4).encode(
                y=alt.Y("mean:Q", title="Daily hazard", axis=alt.Axis(format="%"))
            )
            + _base.mark_line(color=INK, strokeDash=[5, 4], strokeWidth=1.4).encode(y="truth:Q")
        ).properties(height=210, width="container")

    figure(
        mo.hstack(
            [
                _panel(_init, AMBER, "Days since sale"),
                _panel(_rec, ACCENT, "Days since return initiated"),
            ],
            widths="equal",
            gap=2,
        ),
        "The two clocks: one flexible, one Weibull",
        f"Daily hazard for a susceptible unit on an open day ({swatch(AMBER)}initiation: the "
        f"random-walk baseline, product A, neutral weekday; {swatch(ACCENT)}receipt: the "
        "Weibull family, clear weather) with 90% posterior bands, each replayed from the "
        f"fitted stage. {swatch(INK, dashed=True)}Dashed lines are the simulator's true "
        "hazards. Both initiation curves hold their last value from age 15 on; the receipt "
        "stage fits the same Weibull family the simulator draws from.",
    )
    return


@app.cell(hide_code=True)
def _(AS_OF, data, fitted, np, pd, truth):
    # The model's belief about each still-eligible unit: the probability it starts a return
    # before its deadline, given it has been quiet through the snapshot. Same formula as the
    # "How censoring is handled" note above, evaluated per unit and per posterior draw.
    _params = fitted.initiation_fit.parameters
    _age_logits = np.asarray(_params.age_logits)
    _beta = np.asarray(_params.beta)
    _susceptibility_intercept = np.asarray(_params.susceptibility_intercept)
    _susceptibility_beta = np.asarray(_params.susceptibility_beta)[:, 0]

    _eligible = data.units[
        data.units.initiation_date.isna()
        & ((AS_OF - data.units.sale_date.to_numpy().astype("datetime64[D]")).astype(int) < 90)
    ]
    _sale = _eligible.sale_date.to_numpy().astype("datetime64[D]")
    _product = _eligible["product"].to_numpy(float)
    _ages = np.arange(91)
    _days = _sale[:, None] + _ages[None, :].astype("timedelta64[D]")
    _weekday = ((_days.astype("datetime64[D]").astype(int) + 3) % 7).astype(float)  # Mon=0
    _season = np.sin(2 * np.pi * _weekday / 7)
    _quiet = _ages[None, :] <= (AS_OF - _sale).astype(int)[:, None]
    _age_index = np.minimum(_ages, _age_logits.shape[1] - 1)

    _p_future = np.empty((len(_susceptibility_intercept), len(_eligible)))
    for _d in range(len(_susceptibility_intercept)):
        _logit = (
            _age_logits[_d, _age_index][None, :]
            + _beta[_d, 0] * _product[:, None]
            + _beta[_d, 1] * _season
        )
        _log_survive = -np.logaddexp(0.0, _logit)  # log(1 - hazard)
        _log_s_quiet = np.where(_quiet, _log_survive, 0.0).sum(1)
        _log_s_ahead = np.where(_quiet, 0.0, _log_survive).sum(1)
        _still_susceptible = 1 / (
            1
            + np.exp(
                -(
                    _susceptibility_intercept[_d]
                    + _susceptibility_beta[_d] * _product
                    + _log_s_quiet
                )
            )
        )
        _p_future[_d] = _still_susceptible * (1 - np.exp(_log_s_ahead))

    _sold = truth[truth.sale_date <= pd.Timestamp(str(AS_OF))].copy()
    _sold["week"] = pd.to_datetime(_sold.sale_date).dt.to_period("W-SUN").dt.start_time
    _sold["observed"] = _sold.initiation_date <= pd.Timestamp(str(AS_OF))
    _sold["eventual"] = _sold.initiation_date.notna()
    _grouped = _sold.groupby("week")
    _weeks = _grouped.size().index

    _unit_week = _sold.set_index("item_id").week
    _future_by_week = (
        pd.DataFrame(_p_future.T, index=_unit_week.loc[_eligible.item_id].values)
        .groupby(level=0)
        .sum()
        .reindex(_weeks, fill_value=0.0)
    )
    _model_rate = (
        _grouped["observed"].sum().to_numpy()[:, None] + _future_by_week.to_numpy()
    ) / _grouped.size().to_numpy()[:, None]

    cohort_rates = pd.DataFrame(
        {
            "week": _weeks,
            "units": _grouped.size().to_numpy(),
            "observed": _grouped["observed"].mean().to_numpy(),
            "eventual": _grouped["eventual"].mean().to_numpy(),
            "model": _model_rate.mean(1),
            "model_lo": np.quantile(_model_rate, 0.05, axis=1),
            "model_hi": np.quantile(_model_rate, 0.95, axis=1),
            "window_opens": pd.Timestamp(str(AS_OF - np.timedelta64(89, "D"))),
        }
    )
    # The partial first and last calendar weeks hold only a few days of sales.
    cohort_rates = cohort_rates[cohort_rates.units >= 60].reset_index(drop=True)
    return (cohort_rates,)


@app.cell(hide_code=True)
def _(section):
    section(
        "forecast",
        "Part 4",
        "Forecasting the next four weeks",
        "The fitted network simulates new sales, pushes every cohort (old and new) through "
        "both clocks, and keeps integer counts consistent in every posterior draw.",
        points=(
            "The fitted chain, one stage at a time",
            "Receipts against held-out days",
            "Returns still owed",
        ),
    )
    return


@app.cell(hide_code=True)
def _(AS_OF, HORIZON, date_grid, np):
    forecast_dates = date_grid(AS_OF + np.timedelta64(1, "D"), AS_OF + np.timedelta64(HORIZON, "D"))
    return (forecast_dates,)


@app.cell
def _(HORIZON, SEED, fitted, forecast_dates, mo, sales_covariates):
    # Simulate sales, then push every cohort through both clocks: one call.
    forecast = fitted.forecast(
        horizon=HORIZON,
        covariates={"sales": sales_covariates(forecast_dates)},  # the weather is a known input
        seed=SEED + 5,
    )
    mo.show_code(position="above")
    return (forecast,)


@app.cell(hide_code=True)
def _(AS_OF, forecast, pd, truth):
    _dates = pd.DatetimeIndex(forecast.dates)
    held_out = pd.DataFrame(
        {
            "date": _dates,
            "initiations": truth.groupby("initiation_date")
            .size()
            .reindex(_dates, fill_value=0)
            .values,
            "receipts": truth.groupby("receipt_date").size().reindex(_dates, fill_value=0).values,
        }
    )
    _as_of = pd.Timestamp(str(AS_OF))
    _past = truth[truth.sale_date <= _as_of]
    _later = _past.receipt_date > _as_of
    eventual_truth = {
        "open": int((_later & (_past.initiation_date <= _as_of)).sum()),
        "uninitiated": int((_later & (_past.initiation_date > _as_of)).sum()),
    }
    return eventual_truth, held_out


@app.cell(hide_code=True)
def _(daily_totals, forecast, pd, truth):
    # Daily totals of the sales forecast that feeds the chain, and what actually sold.
    sales_draws = daily_totals(forecast.sales, forecast.dates)
    sales_actual = (
        truth.groupby("sale_date")
        .size()
        .reindex(pd.DatetimeIndex(forecast.dates), fill_value=0)
        .to_numpy()
    )
    return sales_actual, sales_draws


@app.cell(hide_code=True)
def _(ChainStepper, fitted, forecast, mo, sales_draws, storms):
    from retail_returns_chain import chain_data

    mo.vstack(
        [
            mo.md(r"""
    ### How the forecast is assembled

    The call above ran the whole chain. Step through it one stage at a time: the sales
    forecast, when the fitted model expects returns to start, when started returns arrive,
    and the receipt forecast that results. The timing bars are probabilities out of all
    units entering the stage, not hazards.
    """),
            mo.ui.anywidget(
                ChainStepper(
                    data=chain_data(fitted, forecast, sales_draws, storm=storms(forecast.dates) > 0)
                )
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(np, pd):
    def fan(dates, draws, label):
        """Long frame of mean and 50%/90% intervals for a [draw, day] array."""
        draws = np.asarray(draws)
        return pd.DataFrame(
            {
                "date": pd.DatetimeIndex(dates),
                "mean": draws.mean(0),
                "lo90": np.quantile(draws, 0.05, axis=0),
                "hi90": np.quantile(draws, 0.95, axis=0),
                "lo50": np.quantile(draws, 0.25, axis=0),
                "hi50": np.quantile(draws, 0.75, axis=0),
                "series": label,
            }
        )

    def crps(draws, actual):
        """Empirical CRPS per day, averaged over the horizon."""
        draws = np.asarray(draws, dtype=float)
        spread = np.abs(draws[:, None, :] - draws[None, :, :]).mean(axis=(0, 1))
        return float((np.abs(draws - actual).mean(axis=0) - 0.5 * spread).mean())

    def daily_totals(sales, dates):
        """[draw, day] unit sales of a ``SalesForecast``, summed over its product cohorts."""
        days = pd.DatetimeIndex(dates)
        slot = days.get_indexer(pd.to_datetime(sales.cohorts.sale_date))
        counts = np.asarray(sales.counts)  # [draw, cohort]
        totals = np.zeros((counts.shape[0], len(days)))
        np.add.at(totals.T, slot, counts.T)
        return totals

    return crps, daily_totals, fan


@app.cell(hide_code=True)
def _(forecast, held_out, np, pd, storm_spans):
    # The first open day after the storm, where the drained backlog is expected to land.
    _dates = pd.DatetimeIndex(forecast.dates)
    _after = storm_spans[storm_spans.end > _dates.min()].iloc[0].end
    while _after.dayofweek >= 5:  # the warehouse is closed on weekends
        _after += pd.Timedelta(days=1)
    _i = int(_dates.get_loc(_after))
    _lo, _hi = np.quantile(forecast.receipts[:, _i], [0.05, 0.95])
    receipt_check = {
        "date": _after,
        "actual": int(held_out.receipts.to_numpy()[_i]),
        "mean": float(forecast.receipts[:, _i].mean()),
        "lo": float(_lo),
        "hi": float(_hi),
    }
    return (receipt_check,)


@app.cell(hide_code=True)
def _(
    ACCENT,
    AMBER,
    INK,
    alt,
    fan,
    figure,
    forecast,
    held_out,
    mo,
    receipt_check,
    storm_spans,
    swatch,
):
    def _panel(draws, actual, color, title, show_axis, storm):
        _frame = fan(forecast.dates, draws, title).assign(actual=actual)
        _base = alt.Chart(_frame).encode(
            x=alt.X(
                "date:T",
                title=None,
                axis=alt.Axis(format="%a %b %-d", labels=show_axis, ticks=show_axis, labelAngle=0),
            )
        )
        _storm = (
            alt.Chart(
                storm_spans[storm_spans.end > _frame.date.min()] if storm else storm_spans[:0]
            )
            .mark_rect(color=AMBER, opacity=0.14)
            .encode(x="start:T", x2="end:T")
        )
        return (
            _storm
            + _base.mark_area(color=color, opacity=0.14).encode(y="lo90:Q", y2="hi90:Q")
            + _base.mark_area(color=color, opacity=0.26).encode(y="lo50:Q", y2="hi50:Q")
            + _base.mark_line(color=color, strokeWidth=2).encode(y=alt.Y("mean:Q", title=title))
            + _base.mark_circle(color=INK, size=34, opacity=0.9).encode(
                y="actual:Q",
                tooltip=[
                    alt.Tooltip("date:T", format="%a %b %-d"),
                    alt.Tooltip("actual:Q", title="held-out actual"),
                    alt.Tooltip("mean:Q", format=".1f", title="forecast mean"),
                ],
            )
        ).properties(height=150, width="container")

    figure(
        mo.vstack(
            [
                _panel(
                    forecast.initiations,
                    held_out.initiations,
                    AMBER,
                    "Returns initiated",
                    show_axis=False,
                    storm=False,
                ),
                _panel(
                    forecast.receipts,
                    held_out.receipts,
                    ACCENT,
                    "Returns received",
                    show_axis=True,
                    storm=True,
                ),
            ],
            gap=0,
        ),
        "Daily forecasts against what actually happened",
        f"Bands are 50% and 90% predictive intervals; {swatch(INK)}dots are held-out actuals "
        "the model never saw. Receipts slow during the storm, then the backlog drains over "
        f"the open days after it: on {receipt_check['date']:%a %b} {receipt_check['date'].day} "
        f"the forecast mean is {receipt_check['mean']:.1f} (90% interval "
        f"{receipt_check['lo']:.0f}–{receipt_check['hi']:.0f}) against "
        f"{receipt_check['actual']} observed.",
    )
    return


@app.cell(hide_code=True)
def _(cards, crps, forecast, held_out, np):
    _total = forecast.receipts.sum(axis=1)
    _lo, _hi = np.quantile(_total, [0.05, 0.95])
    _inside = np.mean(
        (held_out.receipts.to_numpy() >= np.quantile(forecast.receipts, 0.05, axis=0))
        & (held_out.receipts.to_numpy() <= np.quantile(forecast.receipts, 0.95, axis=0))
    )
    cards(
        [
            (
                "28-day receipts",
                f"{_total.mean():,.0f}<small>[{_lo:,.0f}–{_hi:,.0f}]</small>",
                f"Held-out actual: <b>{held_out.receipts.sum():,}</b>.",
                "accent",
            ),
            (
                "Daily coverage",
                f"{_inside:.0%}",
                "Share of held-out days inside the 90% daily interval.",
                None,
            ),
            (
                "Daily receipt CRPS",
                f"{crps(forecast.receipts, held_out.receipts.to_numpy()):.2f}",
                "Units per day; lower is better. Scores the whole predictive distribution.",
                None,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Returns still owed

    Planning often needs a number no finite horizon gives you: how many units **already
    sold** will eventually come back? ttenet computes it per posterior draw and splits it
    by where each unit is today.
    """)
    return


@app.cell(hide_code=True)
def _(
    ACCENT,
    AMBER,
    INK,
    alt,
    eventual_truth,
    figure,
    forecast,
    np,
    pd,
    swatch,
):
    _parts = pd.DataFrame(
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
    _base = alt.Chart(_parts).encode(
        y=alt.Y(
            "component:N", title=None, sort=None, axis=alt.Axis(labelLimit=260, labelFontSize=12)
        )
    )
    _chart = (
        _base.mark_bar(height=26, opacity=0.85).encode(
            x=alt.X("mean:Q", title="Expected eventual receipts"),
            color=alt.Color("color:N", scale=None),
        )
        + _base.mark_rule(color=INK, strokeWidth=1.5).encode(x="lo:Q", x2="hi:Q")
        + _base.mark_tick(color=INK, thickness=3, size=34).encode(x="truth:Q")
    ).properties(height=110, width="container")
    figure(
        _chart,
        "Where the owed returns are hiding",
        "Bars are posterior means with 90% intervals; the black tick is how many actually "
        f"arrived in the simulated future. Most owed returns sit with customers who haven't "
        f'clicked "return" yet {swatch(AMBER)}, the population a return-rate shortcut '
        "undercounts. Eventual counts assume the warehouse keeps opening on weekdays.",
    )
    return


@app.cell(hide_code=True)
def _(data, forecast, mo, np):
    _open_now = int((data.units.initiation_date.notna() & data.units.receipt_date.isna()).sum())
    _closed = (np.asarray(forecast.dates, "datetime64[D]").astype(int) + 3) % 7 >= 5  # Sat, Sun
    _checks = [
        (
            "Every return is counted exactly once",
            np.array_equal(
                forecast.open_returns + forecast.receipts.cumsum(axis=1),
                _open_now + forecast.initiations.cumsum(axis=1),
            ),
            "open + cumulative receipts = open today + cumulative initiations, per draw",
        ),
        (
            "No receipts on a closed warehouse day",
            bool((forecast.receipts[:, _closed] == 0).all()),
            "Saturday and Sunday receipts are exactly zero in every draw",
        ),
        (
            "Owed returns decompose exactly",
            np.allclose(
                forecast.expected_existing_receipts,
                forecast.expected_open_receipts + forecast.expected_uninitiated_receipts,
            ),
            "owed = in transit + not yet initiated, per draw",
        ),
    ]
    mo.Html(
        '<ul class="ttn-checks">'
        + "".join(
            f'<li class="{"" if ok else "ttn-check--fail"}">{"✓" if ok else "✗"} {label} '
            f"<code>{detail}</code></li>"
            for label, ok, detail in _checks
        )
        + "</ul>"
    )
    return


@app.cell(hide_code=True)
def _(section):
    section(
        "scenarios",
        "Part 5",
        "Asking “what if?”",
        "This is where the chain pays off. Weather enters the receipt stage as an input, so "
        "you can ask how next month's receipts change if the storm forecast is wrong: no "
        "storm at all, or one that lingers. Nothing is refit; the fitted stages stay exactly "
        "as they are and only the storm input changes.",
        points=(
            "Choose a storm scenario",
            "Receipts and returns in transit against the forecast storm",
        ),
    )
    return


@app.cell(hide_code=True)
def _(AS_OF, np, pd, storm_spans):
    _next = storm_spans[storm_spans.start > str(AS_OF)].iloc[0]
    storm_start = np.datetime64(_next.start.date())
    _first = _next.start
    _last = _next.end - pd.Timedelta(days=1)
    _lingers_to = _first + pd.Timedelta(days=9)
    _baseline_storm = f"{_first:%b} {_first.day}–{_last.day}"
    _lingering_storm = f"{_first:%b} {_first.day}–{_lingers_to.day}"
    weather_labels = {
        "clear": "No storm",
        "as_forecast": f"Storm as forecast ({_baseline_storm})",
        "long": f"Storm lingers ten days ({_lingering_storm})",
    }
    return storm_start, weather_labels


@app.cell(hide_code=True)
def _(np, storm_start, storms):
    def storm_scenario(weather):
        """Receipt covariates for one storm scenario; history keeps its cached values."""

        def receipts(frame, calendar):
            days = np.asarray(calendar, dtype="datetime64[D]")
            storm = storms(days)
            if weather == "clear":
                storm = np.zeros_like(storm)
            elif weather == "long":
                lingering = (days >= storm_start) & (days < storm_start + np.timedelta64(10, "D"))
                storm = np.maximum(storm, lingering.astype(float))
            return {"features": np.broadcast_to(storm[None, :, None], (len(frame), len(days), 1))}

        return receipts

    return (storm_scenario,)


@app.cell(hide_code=True)
def _(HORIZON, SEED, fitted, forecast, forecast_dates, sales_covariates, storm_scenario):
    # Same fitted model, no refit: only the storm input to the receipt stage changes. All
    # three runs are computed here, so the figure below needs no Python once it is drawn
    # and its buttons work on a published page. The forecast's own storm is the baseline.
    scenario_runs = {"as_forecast": forecast} | {
        weather: fitted.forecast(
            horizon=HORIZON,
            covariates={
                "sales": sales_covariates(forecast_dates),
                "receipts": storm_scenario(weather),
            },
            seed=SEED + 5,
        )
        for weather in ("clear", "long")
    }
    return (scenario_runs,)


@app.cell(hide_code=True)
def _(
    AS_OF,
    ScenarioForecast,
    forecast,
    mo,
    np,
    pd,
    scenario_runs,
    storm_scenario,
    storms,
    weather_labels,
):
    def _summary(draws):
        values = np.asarray(draws)
        lo90, lo50, hi50, hi90 = np.quantile(values, [0.05, 0.25, 0.75, 0.95], axis=0)
        return {
            "mean": values.mean(axis=0).tolist(),
            "lo90": lo90.tolist(),
            "hi90": hi90.tolist(),
            "lo50": lo50.tolist(),
            "hi50": hi50.tolist(),
        }

    def _total(draws):
        # An interval for a per-draw quantity (a 28-day sum, a final-day count).
        values = np.asarray(draws)
        lo, hi = np.quantile(values, [0.05, 0.95])
        return {"mean": float(values.mean()), "lo90": float(lo), "hi90": float(hi)}

    _storm_baseline = storms(forecast.dates) > 0

    def _payload_for(weather, scenario):
        # The runs share a seed and the same sales draws, so draw i of a scenario pairs with
        # draw i of the baseline and their difference is a real per-draw quantity.
        assert np.array_equal(scenario.sales.counts, forecast.sales.counts), "draws do not pair"
        arms = {"baseline": forecast, "scenario": scenario}
        storm_scenario_days = (
            storm_scenario(weather)(pd.DataFrame(index=[0]), forecast.dates)["features"][0, :, 0]
            > 0
        )
        difference = scenario.receipts - forecast.receipts

        # Planning windows: the days either run calls stormy, then the week that follows.
        stormy = np.flatnonzero(_storm_baseline | storm_scenario_days)
        first, last = int(stormy[0]), int(stormy[-1])
        spans = [(first, last), (last + 1, min(last + 7, len(_storm_baseline) - 1))]
        return {
            "storm": {
                "baseline": _storm_baseline.tolist(),
                "scenario": storm_scenario_days.tolist(),
            },
            "panels": {
                "receipt": {arm: _summary(run.receipts) for arm, run in arms.items()},
                "open": {arm: _summary(run.open_returns) for arm, run in arms.items()},
            },
            # Receipts so far, scenario minus baseline, summarized over paired draws.
            "gap": _summary(np.cumsum(difference, axis=1)),
            "windows": [
                {
                    "start": start,
                    "end": end,
                    "receipt": {
                        **{
                            arm: _total(run.receipts[:, start : end + 1].sum(axis=1))
                            for arm, run in arms.items()
                        },
                        "diff": _total(difference[:, start : end + 1].sum(axis=1)),
                    },
                }
                for start, end in spans
                if start <= end
            ],
            "totals": {
                "receipt": {arm: _total(run.receipts.sum(axis=1)) for arm, run in arms.items()},
                "open_end": {arm: _total(run.open_returns[:, -1]) for arm, run in arms.items()},
            },
        }

    mo.ui.anywidget(
        ScenarioForecast(
            data={
                "as_of": str(AS_OF),
                "dates": np.asarray(forecast.dates, dtype="datetime64[D]").astype(str).tolist(),
                "weather_labels": weather_labels,
                "order": ["clear", "as_forecast", "long"],
                "default": "as_forecast",
                "scenarios": {
                    weather: _payload_for(weather, run) for weather, run in scenario_runs.items()
                },
            }
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, source_md, storm_scenario):
    mo.accordion(
        {
            "How the scenarios are computed": mo.vstack(
                [
                    mo.md(
                        "One function rewrites the receipt stage's storm covariate for each "
                        "scenario. The same fitted model forecasts again with the same seed, so "
                        "each draw differs from the baseline only through the storm. All three "
                        "runs are computed up front; the buttons only switch between them."
                    ),
                    source_md(storm_scenario),
                    mo.md(
                        "```python\nfitted.forecast(\n    horizon=HORIZON,\n"
                        '    covariates={"sales": sales_covariates(forecast_dates),\n'
                        '                "receipts": storm_scenario("long")},\n'
                        "    seed=SEED + 5,\n)\n```"
                    ),
                ]
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(section):
    section(
        "sales-model",
        "Part 6",
        "Swap the sales model, then beat a baseline",
        "Two more things the chain buys you. A different sales forecaster can feed the same "
        "fitted return stages without a refit. And against the usual warehouse shortcut, the "
        "chain wins on timing, which is what a warehouse staffs for.",
        points=(
            "A drifting sales model through the same stages",
            "The weekday-average shortcut, scored on held-out days",
        ),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A second sales model

    The alternative is a **local level**: each product's log sales rate takes a small
    random-walk step every day, so the forecast can follow a drifted level instead of
    returning to a fixed mean. It is ordinary NumPyro, fitted on the training sales alone,
    and its 28-day draws go through the return stages fitted in Part 3 by row: sales draw
    $i$ runs through return draw $i$. Uncertainty flows down the chain, but neither side
    was conditioned on the other.
    """)
    return


@app.cell
def _(Horizon, dist, innovations, jnp, mo, numpyro, predict):
    def drifting_sales_model(covariates, data=None):
        """Poisson sales whose log rate follows a random walk (a drifting local level)."""
        h = Horizon.from_data(covariates, data)
        level0 = numpyro.sample("level0", dist.Normal(2.0, 1.0).expand([2]).to_event(1))
        drift_scale = numpyro.sample("drift_scale", dist.HalfNormal(0.05).expand([2]).to_event(1))
        weekday_effect = numpyro.sample(
            "weekday_effect", dist.Normal(0, 0.4).expand([2]).to_event(1)
        )
        with numpyro.plate("product", 2, dim=-1):  # one random walk per product
            steps = innovations(h, "steps", dist.Normal(0.0, 1.0))
        level = level0 + jnp.cumsum(steps * drift_scale, axis=-2)
        predict(h, lambda value: dist.Poisson(jnp.exp(value)), level + covariates * weekday_effect)

    mo.show_code(position="above")
    return (drifting_sales_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Fitting it and handing it to the chain": mo.md(r"""
    `numpyro_forecast.forecast` replays the training prefix from the fitted posterior and
    samples the next 28 days; `SalesForecast.from_numpyro_forecast` binds those
    `[draw, day, product]` counts to dates and products; `fitted.forecast(future_sales=...)`
    pushes them through the return stages. Three ways to change the sales side, and what
    each refits:

    | What you change | What is refit | How sales draws meet return draws |
    | --- | --- | --- |
    | Hand in a future sales path (a plan, a promo) as `future_sales=` | Nothing | Paths from this fit keep their draw labels; a table from elsewhere is paired by row. |
    | Fit another sales model separately and pass its forecast (this part) | Only the sales model | By row: sales draw $i$ through return draw $i$. Propagation, not joint conditioning. |
    | Replace the model inside `RetailReturnModel(sales=CountProcess(model=...))`, then fit | Everything, jointly | A new joint posterior; not run in this post. |
    """)
        }
    )
    return


@app.cell(hide_code=True)
def _(
    HORIZON,
    SEED,
    SalesForecast,
    data,
    draw_posterior,
    drifting_sales_model,
    fitted,
    forecast_dates,
    jax,
    jnp,
    np,
    numpyro,
    numpyro_forecast,
    sales_covariates,
):
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoNormal

    # Fit on the training window only: data.sales stops at the as_of snapshot.
    training_sales = jnp.asarray(data.sales)  # [day, product] counts
    guide = AutoNormal(drifting_sales_model)
    svi = SVI(drifting_sales_model, guide, numpyro.optim.Adam(0.02), Trace_ELBO())
    sales_fit = svi.run(
        jax.random.PRNGKey(SEED),
        2000,
        sales_covariates(data.calendar),
        training_sales,
        progress_bar=False,
    )
    sales_posterior = draw_posterior(
        jax.random.PRNGKey(SEED + 1), guide, sales_fit.params, fitted.network.num_samples
    )

    # Training prefix from the posterior, then the next 28 days sampled: [draw, day, product].
    all_days = np.concatenate([data.calendar, forecast_dates])
    drift_draws = numpyro_forecast.forecast(
        jax.random.PRNGKey(SEED + 2),
        drifting_sales_model,
        sales_posterior,
        training_sales,
        sales_covariates(all_days),
    )
    drift_sales = SalesForecast.from_numpyro_forecast(
        drift_draws, forecast_dates, groups=data.groups, prefix="drift"
    )

    # The same fitted return stages, no refit: only the sales source changes.
    drift_forecast = fitted.forecast(horizon=HORIZON, future_sales=drift_sales, seed=SEED + 5)
    return drift_forecast, drift_sales, sales_fit, sales_posterior


@app.cell(hide_code=True)
def _(daily_totals, drift_forecast):
    drift_sales_draws = daily_totals(drift_forecast.sales, drift_forecast.dates)
    return (drift_sales_draws,)


@app.cell(hide_code=True)
def _(
    ACCENT,
    AMBER,
    INK,
    MUTED,
    SLATE,
    alt,
    data,
    drift_forecast,
    drift_sales_draws,
    fan,
    figure,
    forecast,
    held_out,
    mo,
    np,
    sales_actual,
    sales_draws,
    sales_posterior,
    swatch,
):
    def _panel(current, drifting, actual, color, title, show_axis):
        _now = fan(forecast.dates, current, "current")
        _new = fan(forecast.dates, drifting, "drifting level").assign(actual=actual)
        _x = alt.X(
            "date:T",
            title=None,
            axis=alt.Axis(format="%a %b %-d", labels=show_axis, ticks=show_axis, labelAngle=0),
        )
        return alt.layer(
            alt.Chart(_now)
            .mark_area(color=color, opacity=0.14)
            .encode(x=_x, y="lo90:Q", y2="hi90:Q"),
            alt.Chart(_now)
            .mark_line(color=color, strokeWidth=2)
            .encode(x=_x, y=alt.Y("mean:Q", title=title)),
            alt.Chart(_new)
            .mark_area(color=SLATE, opacity=0.16)
            .encode(x=_x, y="lo90:Q", y2="hi90:Q"),
            alt.Chart(_new)
            .mark_line(color=SLATE, strokeWidth=2.2, strokeDash=[5, 3])
            .encode(x=_x, y="mean:Q"),
            alt.Chart(_new).mark_circle(color=INK, size=30, opacity=0.9).encode(x=_x, y="actual:Q"),
        ).properties(height=130, width="container")

    _order = list(data.groups["product"].to_numpy(int))
    _scale = np.asarray(sales_posterior["drift_scale"]).mean(axis=0)
    figure(
        mo.vstack(
            [
                _panel(sales_draws, drift_sales_draws, sales_actual, MUTED, "Units sold", False),
                _panel(
                    forecast.initiations,
                    drift_forecast.initiations,
                    held_out.initiations,
                    AMBER,
                    "Returns started",
                    False,
                ),
                _panel(
                    forecast.receipts,
                    drift_forecast.receipts,
                    held_out.receipts,
                    ACCENT,
                    "Returns received",
                    True,
                ),
            ],
            gap=0,
        ),
        "Two sales forecasters, one set of fitted return stages",
        f"Solid lines and light bands: the Part 4 forecast. {swatch(SLATE)}Dashed slate line "
        "and band: the drifting level, fitted only to the training days and pushed through "
        f"the same return stages. Bands are 90% intervals; {swatch(INK)}dots are held-out "
        f"actuals. The drifting model's daily log-rate step is about "
        f"{_scale[_order.index(0)]:.3f} (product A) and {_scale[_order.index(1)]:.3f} "
        "(product B): day-to-day volatility, not a trend.",
    )
    return


@app.cell(hide_code=True)
def _(
    cards,
    crps,
    drift_forecast,
    drift_sales_draws,
    forecast,
    held_out,
    np,
    sales_actual,
    sales_draws,
):
    def _interval(draws):
        return (
            f"{draws.mean():,.0f}"
            f"<small>[{np.quantile(draws, 0.05):,.0f}–{np.quantile(draws, 0.95):,.0f}]</small>"
        )

    _actual = held_out.receipts.to_numpy()
    cards(
        [
            (
                "Units sold, next 28 days",
                _interval(drift_sales_draws.sum(axis=1)),
                f"Drifting level. Part 4 model: <b>{sales_draws.sum(axis=1).mean():,.0f}</b>. "
                f"Actual: <b>{sales_actual.sum():,}</b>.",
                None,
            ),
            (
                "Returns started",
                _interval(drift_forecast.initiations.sum(axis=1)),
                f"Part 4 model: <b>{forecast.initiations.sum(axis=1).mean():,.0f}</b>. "
                f"Actual: <b>{held_out.initiations.sum():,}</b>.",
                "amber",
            ),
            (
                "Returns received",
                _interval(drift_forecast.receipts.sum(axis=1)),
                f"Part 4 model: <b>{forecast.receipts.sum(axis=1).mean():,.0f}</b>. "
                f"Actual: <b>{held_out.receipts.sum():,}</b>.",
                "accent",
            ),
            (
                "Daily receipt CRPS",
                f"{crps(drift_forecast.receipts, _actual):.2f}",
                f"Part 4 model: {crps(forecast.receipts, _actual):.2f}. Lower is better.",
                None,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(data, drift_forecast, drift_sales, mo, np):
    _open_now = int((data.units.initiation_date.notna() & data.units.receipt_date.isna()).sum())
    _closed = (np.asarray(drift_forecast.dates, "datetime64[D]").astype(int) + 3) % 7 >= 5
    _checks = [
        (
            "Every return is counted exactly once",
            np.array_equal(
                drift_forecast.open_returns + drift_forecast.receipts.cumsum(axis=1),
                _open_now + drift_forecast.initiations.cumsum(axis=1),
            ),
            "open + cumulative receipts = open today + cumulative initiations, per draw",
        ),
        (
            "The stages saw exactly the forecaster's sales",
            np.array_equal(drift_forecast.sales.counts, drift_sales.counts),
            "the sales that were propagated equal the drifting model's draws",
        ),
        (
            "No receipts on a closed warehouse day",
            bool((drift_forecast.receipts[:, _closed] == 0).all()),
            "Saturday and Sunday receipts are exactly zero in every draw",
        ),
    ]
    mo.accordion(
        {
            "Consistency checks on the propagated forecast": mo.Html(
                '<ul class="ttn-checks">'
                + "".join(
                    f'<li class="{"" if ok else "ttn-check--fail"}">{"✓" if ok else "✗"} {label} '
                    f"<code>{detail}</code></li>"
                    for label, ok, detail in _checks
                )
                + "</ul>"
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Against a simple baseline

    A common shortcut for warehouse planning: forecast each day's receipts as the average
    for that weekday over the last eight weeks.
    """)
    return


@app.cell
def _(daily_history, held_out, pd):
    recent = daily_history.tail(56).assign(weekday=lambda d: d.date.dt.dayofweek)
    weekday_mean = recent.groupby("weekday")["Returns received"].mean()
    weekday_avg = pd.Series(
        weekday_mean.reindex(held_out.date.dt.dayofweek).to_numpy(),
        index=held_out.date,
        name="weekday_avg",
    )
    return (weekday_avg,)


@app.cell(hide_code=True)
def _(
    ACCENT,
    AMBER,
    INK,
    RED,
    alt,
    fan,
    figure,
    forecast,
    held_out,
    weekday_avg,
    pd,
    storm_spans,
    swatch,
):
    _frame = fan(forecast.dates, forecast.receipts, "ttenet").assign(
        weekday_avg=weekday_avg.to_numpy(), actual=held_out.receipts.to_numpy()
    )
    _x = alt.X("date:T", title=None, axis=alt.Axis(format="%a %b %-d", labelAngle=0))
    _lines = pd.concat(
        [
            _frame[["date", "mean"]].rename(columns={"mean": "value"}).assign(series="ttenet mean"),
            _frame[["date", "weekday_avg"]]
            .rename(columns={"weekday_avg": "value"})
            .assign(series="Weekday average"),
        ]
    )
    _chart = (
        alt.Chart(storm_spans[storm_spans.end > _frame.date.min()])
        .mark_rect(color=AMBER, opacity=0.14)
        .encode(x="start:T", x2="end:T")
        + alt.Chart(_frame)
        .mark_area(color=ACCENT, opacity=0.14)
        .encode(x=_x, y="lo90:Q", y2="hi90:Q")
        + alt.Chart(_lines)
        .mark_line(strokeWidth=2.2)
        .encode(
            x=_x,
            y=alt.Y("value:Q", title="Returns received per day"),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=["ttenet mean", "Weekday average"], range=[ACCENT, RED]),
            ),
        )
        + alt.Chart(_frame).mark_circle(color=INK, size=34).encode(x=_x, y="actual:Q")
    ).properties(height=240, width="container")
    figure(
        _chart,
        "The baseline doesn't know about the storm or the backlog. The model forecasts "
        "a post-storm catch-up, but does not guarantee a one-day spike.",
        f"{swatch(RED)}The weekday average repeats the recent past. {swatch(ACCENT)}ttenet "
        "knows which units are in transit, how old they are, and that the storm will hold "
        f"them up. {swatch(INK)}Dots: held-out actuals.",
    )
    return


@app.cell(hide_code=True)
def _(TABLE_THEME, crps, ct, forecast, held_out, weekday_avg, np, pd):
    _actual = held_out.receipts.to_numpy()
    _model_mean = forecast.receipts.mean(axis=0)
    _model_total = forecast.receipts.sum(axis=1)  # one 28-day total per posterior draw
    _storm_week = slice(5, 14)
    _scores = pd.DataFrame(
        [
            {
                "Method": "Weekday average (last 8 weeks)",
                "mae": float(np.abs(weekday_avg.to_numpy() - _actual).mean()),
                "crps": float(np.abs(weekday_avg.to_numpy() - _actual).mean()),
                "storm": float(
                    np.abs(weekday_avg.to_numpy()[_storm_week] - _actual[_storm_week]).mean()
                ),
                "total": float(weekday_avg.sum()),
                "total_lo": np.nan,  # a point forecast has no interval
                "total_hi": np.nan,
            },
            {
                "Method": "ttenet",
                "mae": float(np.abs(_model_mean - _actual).mean()),
                "crps": crps(forecast.receipts, _actual),
                "storm": float(np.abs(_model_mean[_storm_week] - _actual[_storm_week]).mean()),
                "total": float(_model_total.mean()),
                "total_lo": float(np.quantile(_model_total, 0.05)),
                "total_hi": float(np.quantile(_model_total, 0.95)),
            },
        ]
    )
    (
        ct.CoefTable(_scores, rows="Method")
        .estimate("Daily MAE", "mae", fmt=ct.Number(decimals=2))
        .estimate("Daily CRPS", "crps", fmt=ct.Number(decimals=2))
        .estimate("MAE, storm fortnight", "storm", fmt=ct.Number(decimals=2))
        .estimate(
            f"28-day total (actual {_actual.sum():,})",
            "total",
            ci=("total_lo", "total_hi"),
            fmt=ct.Number(decimals=0),
        )
        .with_theme(TABLE_THEME)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For a point forecast, CRPS reduces to absolute error, so the baseline's two columns
    match.

    On the monthly total the two land in the same place, and that is expected here. This
    simulated world is flat: sales rates and return behaviour never change, and the last
    eight weeks already contain a storm with its dip and catch-up. Repeating the recent
    average therefore gets a month's total about right; its daily misses largely cancel
    when summed. One 28-day window is also a single noisy draw, so it cannot show whether
    either method is biased; the actual total falls inside the model's 90% interval.

    The shortcut's weakness is *when* receipts arrive, which is what a warehouse staffs
    for: around the storm its daily error is several times the model's. It also has no
    answer for *returns still owed*, no interval, and no way to take in a sales forecast
    or a storm scenario. In a world that changes (sales growth, a promo, a new policy) the
    recent average would miss the total too.
    """)
    return


@app.cell(hide_code=True)
def _(section):
    section("wrap-up", "Conclusion", "Chain the forecasts")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Returns forecasting is an old problem. Splitting it into a chain of small forecasts,
    sales then initiation then receipt, is what makes it convenient. ttenet fits the chain
    in one call and propagates through it in another. Because each stage consumes the
    previous stage's forecast draw by draw, the same machinery answers daily receipts,
    returns still owed, "what if the storm lingers?" and "what if a different model
    forecasts sales?" without refitting the return stages, with uncertainty carried end
    to end.

    ### Limits

    - Finite follow-up can't fully separate "never" from "very late". Quiet days lower a
      unit's chance of being susceptible but never settle it; the susceptible share leans
      on mature cohorts and the priors. Check calibration on held-out periods.
    - One event of each type per unit, daily resolution, no repeated return attempts, no
      inventory feedback, and weather is an input you supply. "Returns still owed" assumes
      the warehouse keeps reopening.
    - The posterior is a variational (AutoNormal) approximation. For publication-grade
      intervals, hand `network.numpyro_model(...)` to NUTS.

    The command-line version of this example lives in `examples/retail_returns.py`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html(
        '<p class="ttn-footer">Simulated data throughout: every "truth" in this post is the '
        "simulator's, and none of it reaches the estimator.</p>"
    )
    return


if __name__ == "__main__":
    app.run()
