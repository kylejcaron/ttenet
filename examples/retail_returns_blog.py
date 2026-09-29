# /// script
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
    import dataclasses
    import inspect
    from html import escape

    import altair as alt
    import coeftable as ct
    import marimo as mo
    import numpy as np
    import pandas as pd
    from retail_returns import (
        initiation_covariates,
        receipt_covariates,
        sales_covariates,
        sales_model,
        simulate_retail,
        storms,
    )

    from ttenet import CountProcess, CureProcess, RetailData, RetailReturnModel, date_grid

    return (
        CountProcess,
        CureProcess,
        RetailData,
        RetailReturnModel,
        alt,
        ct,
        dataclasses,
        date_grid,
        escape,
        initiation_covariates,
        inspect,
        mo,
        np,
        pd,
        receipt_covariates,
        sales_covariates,
        sales_model,
        simulate_retail,
        storms,
    )


@app.cell(hide_code=True)
def _(alt, ct, escape, mo):
    INK, MUTED, ACCENT, AMBER, RED = "#292d26", "#62695d", "#42644d", "#956017", "#a23c3c"
    RULE, GRID, PAPER, SLATE = "#dcded1", "#ecefe4", "#fffef9", "#4c5a6b"

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

    def section(anchor, kicker, title, lede=""):
        """A numbered section opener that the sticky navigation can jump to."""
        lede_html = f'<p class="ttn-section-lede">{lede}</p>' if lede else ""
        return mo.Html(
            f'<section id="{anchor}" class="ttn-section">'
            f'<p class="ttn-section-kicker">{kicker}</p>'
            f'<h2 class="ttn-section-title">{title}</h2>{lede_html}</section>'
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

    def swatch(color, dashed=False, block=False):
        """Inline legend key: a line, a dashed line, or a shaded block for bands."""
        if block:
            style = f"background: {color}; opacity: 0.35; height: 11px; width: 13px"
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
            <a href="#scenarios">Scenarios</a>
            <a href="#simpler">Simpler?</a>
            <a href="#wrap-up">Wrap-up</a>
          </nav>
        </header>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html("""
        <div id="top" class="ttn-hero">
          <p class="ttn-kicker">Survival analysis · Time series · Supply chain · Bayesian modeling</p>
          <h1 class="ttn-title">Returns you haven't seen yet</h1>
          <p class="ttn-dek">
            Forecasting retail returns from first principles. Every sale starts a clock.
            Most of those clocks never ring, and the ones that do ring at the warehouse on
            a weekday, unless it's storming.
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
        "Observed returns are not eventual returns",
        "If you staff a returns warehouse, book refund liabilities, or plan to restock "
        "returned units, you care about the returns that <em>will</em> arrive. The data only "
        "shows the ones that already have.",
    )
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
def _(cards, cohort_rates, forecast, held_out, np):
    _recent = cohort_rates.tail(4)  # the last month of sales
    _gap = (_recent.eventual - _recent.observed).mean()
    _owed = forecast.expected_existing_receipts
    _next = forecast.receipts.sum(axis=1)
    cards(
        [
            (
                "Naive return rate, last month's sales",
                f"−{_gap * 100:.0f} pts",
                "Average shortfall vs. the rate those same units eventually reach.",
                "red",
            ),
            (
                "Returns still owed by past sales",
                f"{_owed.mean():,.0f}",
                "Expected eventual receipts from units already sold, including open returns.",
                "accent",
            ),
            (
                "Receipts, next 28 days",
                f"{_next.mean():,.0f}"
                f"<small>[{np.quantile(_next, 0.05):,.0f}–{np.quantile(_next, 0.95):,.0f}]</small>",
                f"Posterior mean and 90% interval. Held-out actual: <b>{held_out.receipts.sum():,}</b>.",
                None,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    That gap is **right-censoring**. A unit sold three weeks ago with no return yet hasn't
    told us "I won't be returned"; it has only told us "not yet". Averaging over those units
    treats "not yet" as "never", and every forecast built on that average inherits the bias.

    The rest of this post:

    - frames returns as a network of **censored clocks** (sale → return initiated → return
      received) where some clocks never ring at all,
    - fits sales and both return stages **jointly** with
      [ttenet](https://github.com/kylejcaron/ttenet),
    - forecasts **daily warehouse receipts** and the **returns still owed** by past sales,
    - asks "what if?" about demand and weather, and
    - checks whether a simpler approach would have been good enough.
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
def _(AS_OF, RetailData, TRAINING_START, date_grid, ledger):
    data = RetailData.from_units(
        ledger,
        as_of=AS_OF,  # end-of-day snapshot; anything later is hidden
        calendar=date_grid(TRAINING_START, AS_OF),  # the sales observation window
        group_by=["product"],
    )
    return (data,)


@app.cell(hide_code=True)
def _(AS_OF, data, np):
    # RetailData imposes no deadline; the 90-day policy belongs to the initiation process.
    _within_policy = (AS_OF - data.units.sale_date.to_numpy().astype("datetime64[D]")).astype(
        int
    ) <= 90
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
def _(data, escape, mo, pd, unit_status):
    _pill = {
        "Received": "received",
        "Return in transit": "transit",
        "Could still return": "eligible",
        "Window closed": "expired",
    }
    _examples = (
        data.units.assign(status=unit_status)
        .sort_values("sale_date")
        .groupby("status", sort=False)
        .nth([40, 41])
        .sort_values("sale_date")
    )

    def _day(value, missing="not yet"):
        if pd.isna(value):
            return f'<span class="ttn-missing">{missing}</span>'
        return pd.Timestamp(value).strftime("%b %-d")

    _rows = "".join(
        "<tr>"
        f'<td class="ttn-mono">{escape(str(row.item_id))}</td>'
        f"<td>{'A' if row.product == 0 else 'B'}</td>"
        f"<td>{_day(row.sale_date)}</td>"
        f"<td>{_day(row.initiation_date, 'never' if row.status == 'Window closed' else 'not yet')}</td>"
        f"<td>{_day(row.receipt_date, '—' if row.status == 'Window closed' else 'not yet')}</td>"
        f'<td><span class="ttn-pill ttn-pill--{_pill[row.status]}">{row.status}</span></td>'
        "</tr>"
        for row in _examples.itertuples()
    )
    mo.vstack(
        [
            mo.Html(
                '<div class="ttn-table-wrap"><table class="ttn-table"><thead><tr>'
                "<th>item_id</th><th>Product</th><th>Sold</th><th>Return initiated</th>"
                "<th>Return received</th><th>Status at snapshot</th>"
                f"</tr></thead><tbody>{_rows}</tbody></table></div>"
            ),
            mo.accordion(
                {
                    f"Browse all {len(data.units):,} units": mo.ui.table(
                        data.units[
                            ["item_id", "product", "sale_date", "initiation_date", "receipt_date"]
                        ],
                        page_size=8,
                        selection=None,
                    )
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(cards, data, pd, truth, unit_status):
    _counts = pd.Series(unit_status).value_counts()
    _open_ids = data.units.item_id[unit_status == "Return in transit"]
    _abandoned = truth.set_index("item_id").abandoned_truth.loc[_open_ids].mean()

    cards(
        [
            (
                "Received",
                f"{_counts.get('Received', 0):,}",
                "Back on the shelf. The only fully observed outcome.",
                "accent",
            ),
            (
                "Return in transit",
                f"{_counts.get('Return in transit', 0):,}",
                f"Initiated, not received. Secretly, <b>{_abandoned:.0%}</b> of these are "
                "abandoned; the ledger can't say which.",
                "amber",
            ),
            (
                "Could still return",
                f"{_counts.get('Could still return', 0):,}",
                "No return yet, still inside the 90-day window.",
                None,
            ),
            (
                "Window closed",
                f"{_counts.get('Window closed', 0):,}",
                "Kept for good: past the deadline with no return.",
                None,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Only the first and last cards are settled. Everything in the middle is **censored**:
    we know the clock is still running, not where it will stop. The in-transit pile is the
    sneakiest: in this world about one in five initiated returns is never shipped back (the
    customer keeps the prepaid label in a drawer). Those never leave "in transit", so after
    six months they are nearly all of it. The ledger records no "abandoned" event; the model
    has to learn it from how long open returns stay open.

    Here are the same events as daily time series.
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
def _(section):
    section(
        "model",
        "Part 2",
        "Modeling the data generating process",
        "Rather than regress daily receipts on lagged sales and hope, write down how each "
        "unit actually moves through the system, then let the aggregate fall out.",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For each unit, four things happen (or don't):

    1. **Sales.** Product $p$ sells $\text{sales}_{t,p} \sim \text{Poisson}\big(\exp(\alpha_p + \gamma_p \sin(2\pi\,\text{weekday}_t / 7))\big)$ units on day $t$.
    2. **Will this unit ever be returned?** A sale is *susceptible* with probability $\pi^{\text{init}}_p = \sigma(a + b\,x_p)$. The rest are **cured**: kept forever, however long you wait.
    3. **When is the return initiated?** A susceptible unit of age $a$ starts a return on day $t$ with discrete hazard $h(a, t) = \sigma\big(\ell_{\min(a,\,15)} + \beta^\top x_t\big)$, but only through day 90 of the policy.
    4. **Does it arrive, and when?** An initiated return arrives at all with probability $\pi^{\text{rec}}$, on a second clock whose hazard drops during storms and is *exactly zero* on weekends.

    The $\ell$'s are a flexible, regularized random-walk baseline over age, so nothing forces
    a lognormal or Weibull shape on the delay. The whole thing is a small directed graph:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid(
        """
    flowchart LR
        W([weekday]) --> S
        S((Sales)) -- "each unit starts a clock" --> I((Return<br/>initiated))
        S -. "cured: never returned" .-> K[kept]
        P([product]) --> I
        D([90-day policy]) -. deadline .-> I
        I -- "second clock" --> R((Return<br/>received))
        I -. "cured: abandoned" .-> A[never arrives]
        T([storms]) --> R
        C([weekends closed]) -. hard zero .-> R
        classDef stage fill:#fffef9,stroke:#42644d,stroke-width:2px,color:#292d26;
        classDef cov fill:#f5f5ed,stroke:#dcded1,color:#62695d;
        classDef sink fill:#f2f0e9,stroke:#dcded1,color:#62695d,stroke-dasharray:4 3;
        class S,I,R stage;
        class W,P,D,T,C cov;
        class K,A sink;
    """,
        theme="base",
        theme_variables={
            "fontFamily": "system-ui, -apple-system, 'Segoe UI', sans-serif",
            "fontSize": "13px",
            "lineColor": "#62695d",
            "edgeLabelBackground": "#fffef9",
            "primaryTextColor": "#292d26",
        },
    )
    return


@app.cell(hide_code=True)
def _(callout):
    callout(
        "Key point",
        r"""A unit that has been quiet for $a$ days is not evidence against a return. Its
        likelihood contribution is $(1-\pi) + \pi\,S(a)$: either it belongs to the cured
        fraction, or it will return and simply hasn't yet. Given that silence, the chance it
        is still headed back is $\pi S(a) / \big((1-\pi) + \pi S(a)\big)$, which shrinks the
        longer it stays quiet. That single idea is what fixes the collapsing line in the
        first chart.""",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Writing it down

    The sales process is an ordinary NumPyro model following the
    [`numpyro_forecast`](https://github.com/juanitorduz/numpyro_forecast) protocol: it
    receives covariates with time on axis `-2`, and `predict` handles the observed prefix
    versus the future suffix.
    """)
    return


@app.cell(hide_code=True)
def _(initiation_covariates, inspect, mo, receipt_covariates, sales_model):
    mo.md(
        "\n\n".join(
            f"```python\n{inspect.getsource(fn).rstrip()}\n```"
            for fn in (sales_model, initiation_covariates, receipt_covariates)
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each return stage is a `CureProcess`: a mixture-cure, discrete-time hazard with its own
    age clock. The deadline and the warehouse's opening days are declared, not learned.
    """)
    return


@app.cell
def _(CountProcess, CureProcess, RetailReturnModel, mo, sales_model):
    model = RetailReturnModel(
        sales=CountProcess(model=sales_model),
        initiation=CureProcess(age_bins=16, deadline_days=90),
        receipt=CureProcess(age_bins=16, allowed_weekdays=range(5)),  # Mon-Fri only
    )
    mo.show_code(position="above")
    return (model,)


@app.cell(hide_code=True)
def _(section):
    section(
        "fit",
        "Part 3",
        "Fitting everything at once",
        "One call fits the sales process and both return stages as a single NumPyro model. "
        "Calendar covariates are passed as providers, so the same function describes history "
        "and the future.",
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
    Because this world is simulated, we know every true parameter. A correctly specified
    model should put its posterior mass around them. Here is the recovery check, including
    the parts the ledger never shows directly: how many units are *ever* returned, and how
    many initiated returns never arrive.
    """)
    return


@app.cell(hide_code=True)
def _(SLATE, TABLE_THEME, ct, data, dataclasses, fitted, np, pd):
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    _ip, _rp = fitted.initiation_fit.parameters, fitted.receipt_fit.parameters
    _posterior = fitted.network.posterior
    _order = list(data.groups["product"].to_numpy(int))  # series order is explicit
    _a, _b = _order.index(0), _order.index(1)
    _ci, _cb = np.asarray(_ip.cure_intercept), np.asarray(_ip.cure_beta)[:, 0]
    _beta_i, _beta_r = np.asarray(_ip.beta), np.asarray(_rp.beta)
    _rate = np.exp(np.asarray(_posterior["sales/log_rate"]))
    _weekday = np.asarray(_posterior["sales/weekday_effect"])

    _rows = [
        ("Sales", "Daily sales rate, product A", _rate[:, _a], 12.0),
        ("Sales", "Daily sales rate, product B", _rate[:, _b], 8.0),
        ("Sales", "Weekday effect, product A", _weekday[:, _a], 0.3),
        ("Sales", "Weekday effect, product B", _weekday[:, _b], -0.2),
        ("Return initiation", "Ever returned, product A", _sigmoid(_ci), _sigmoid(-0.8)),
        ("Return initiation", "Ever returned, product B", _sigmoid(_ci + _cb), _sigmoid(-0.35)),
        ("Return initiation", "Hazard shift, product B", _beta_i[:, 0], -0.25),
        ("Return initiation", "Hazard shift, weekday", _beta_i[:, 1], 0.3),
        (
            "Return receipt",
            "Initiated return ever arrives",
            _sigmoid(np.asarray(_rp.cure_intercept)),
            0.8,
        ),
        ("Return receipt", "Hazard shift, storm day", _beta_r[:, 0], -2.0),
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
    (
        ct.CoefTable(recovery, rows="Parameter", groups="stage")
        .estimate("Posterior", "estimate", ci=("lower", "upper"), fmt=ct.Number(decimals=2))
        .estimate("Truth", "truth", fmt=ct.Number(decimals=2))
        .forest(
            "Posterior vs. truth",
            of="Posterior",
            ref=float("nan"),  # no zero line: each row's reference is its true value
            scale="row",
            width=260,
            annotations=[ct.Rule(at="truth", axis="x", color="#292D26")],
        )
        .with_theme(
            # Sign is not good or bad here: one neutral colour for every interval.
            dataclasses.replace(TABLE_THEME, favorable=SLATE, unfavorable=SLATE, inconclusive=SLATE)
        )
    )
    return (recovery,)


@app.cell(hide_code=True)
def _(ACCENT, AMBER, INK, alt, figure, fitted, mo, np, pd, swatch):
    def _curve(parameters, truth_logit, stage):
        _ages = np.arange(0, 31)
        _logits = np.asarray(parameters.age_logits)
        _hazard = 1 / (1 + np.exp(-_logits[:, np.minimum(_ages, _logits.shape[1] - 1)]))
        _truth = 1 / (1 + np.exp(-truth_logit(_ages)))
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
        fitted.initiation_fit.parameters,
        lambda a: -2.5 + 0.35 * np.cos(np.minimum(a, 15) / 5),
        "Initiation: days since sale",
    )
    _rec = _curve(
        fitted.receipt_fit.parameters,
        lambda a: -1.0 + 0.2 * np.cos(np.minimum(a, 15) / 5),
        "Receipt: days since initiation",
    )

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
        "The two clocks, learned without assuming a shape",
        f"Baseline daily hazard for a susceptible unit ({swatch(AMBER)}initiation for product A "
        f"on a neutral weekday, {swatch(ACCENT)}receipt on a clear weekday) with 90% posterior "
        f"bands; {swatch(INK, dashed=True)}dashed lines are the true hazards. Ages past 15 days "
        "share the last baseline bin.",
    )
    return


@app.cell(hide_code=True)
def _(AS_OF, data, fitted, np, pd, truth):
    # The model's belief about each still-eligible unit: the probability it starts a return
    # before its deadline, given it has been quiet through the snapshot. Same formula as the
    # "Key point" above, evaluated per unit and per posterior draw.
    _params = fitted.initiation_fit.parameters
    _age_logits = np.asarray(_params.age_logits)
    _beta = np.asarray(_params.beta)
    _cure_int = np.asarray(_params.cure_intercept)
    _cure_beta = np.asarray(_params.cure_beta)[:, 0]

    _eligible = data.units[
        data.units.initiation_date.isna()
        & ((AS_OF - data.units.sale_date.to_numpy().astype("datetime64[D]")).astype(int) <= 90)
    ]
    _sale = _eligible.sale_date.to_numpy().astype("datetime64[D]")
    _product = _eligible["product"].to_numpy(float)
    _ages = np.arange(91)
    _days = _sale[:, None] + _ages[None, :].astype("timedelta64[D]")
    _weekday = ((_days.astype("datetime64[D]").astype(int) + 3) % 7).astype(float)  # Mon=0
    _season = np.sin(2 * np.pi * _weekday / 7)
    _quiet = _ages[None, :] <= (AS_OF - _sale).astype(int)[:, None]
    _age_index = np.minimum(_ages, _age_logits.shape[1] - 1)

    _p_future = np.empty((len(_cure_int), len(_eligible)))
    for _d in range(len(_cure_int)):
        _logit = (
            _age_logits[_d, _age_index][None, :]
            + _beta[_d, 0] * _product[:, None]
            + _beta[_d, 1] * _season
        )
        _log_survive = -np.logaddexp(0.0, _logit)  # log(1 - hazard)
        _log_s_quiet = np.where(_quiet, _log_survive, 0.0).sum(1)
        _log_s_ahead = np.where(_quiet, 0.0, _log_survive).sum(1)
        _still_susceptible = 1 / (
            1 + np.exp(-(_cure_int[_d] + _cure_beta[_d] * _product + _log_s_quiet))
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
            "window_opens": pd.Timestamp(str(AS_OF - np.timedelta64(90, "D"))),
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
    )
    return


@app.cell
def _(AS_OF, HORIZON, SEED, date_grid, fitted, np, sales_covariates):
    forecast_dates = date_grid(AS_OF + np.timedelta64(1, "D"), AS_OF + np.timedelta64(HORIZON, "D"))
    forecast = fitted.forecast(
        horizon=HORIZON,
        covariates={"sales": sales_covariates(forecast_dates)},  # the weather is a known input
        seed=SEED + 5,
    )
    return forecast, forecast_dates


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

    return crps, fan


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
        "the model never saw. The storm in early July is a supplied weather input, not a "
        "forecast: receipts dip during it, then the backlog lands as soon as it clears. "
        "Weekend receipts are exactly zero in every draw.",
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

    Planning often needs a number that no finite horizon gives you: how many units, **already
    sold**, will eventually come back through the door? ttenet computes it analytically per
    posterior draw and splits it by where each unit is today.
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
        "arrived in the simulated future. Most owed returns sit with customers who haven't "
        f'clicked "return" yet {swatch(AMBER)}, which is exactly the population the naive '
        "approach treats as zero. Open returns are shrunk too: an in-transit return is an "
        "observed status, not a promise it will arrive.",
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
        "Because every step is explicit, you can swap an input and propagate it. Future "
        "sales and weather are both inputs here; the posterior stays fixed.",
    )
    return


@app.cell(hide_code=True)
def _(AS_OF, mo, np, storm_spans):
    _next = storm_spans[storm_spans.start > str(AS_OF)].iloc[0]
    storm_start = np.datetime64(_next.start.date())
    _label = f"{_next.start:%b %-d}–{(_next.end - np.timedelta64(1, 'D')):%b %-d}"
    demand_scale = mo.ui.slider(
        start=0.5,
        stop=2.0,
        step=0.25,
        value=1.5,
        label="Future sales vs. forecast",
        show_value=True,
    )
    weather = mo.ui.dropdown(
        options={
            f"Storm as forecast ({_label})": "as_forecast",
            "Clear skies": "clear",
            "Storm lingers ten days": "long",
        },
        value="Storm lingers ten days",
        label="Weather",
    )
    mo.Html(
        '<div class="ttn-controls">'
        + mo.hstack([demand_scale, weather], justify="start", gap=2.5, wrap=True).text
        + "</div>"
    )
    return demand_scale, storm_start, weather


@app.cell
def _(np, storm_start, storms, weather):
    def scenario_receipts(frame, calendar):
        """Receipt covariates for the chosen weather; history keeps its cached values."""
        days = np.asarray(calendar, dtype="datetime64[D]")
        storm = storms(days)
        if weather.value == "clear":
            storm = np.zeros_like(storm)
        elif weather.value == "long":
            lingering = (days >= storm_start) & (days < storm_start + np.timedelta64(10, "D"))
            storm = np.maximum(storm, lingering.astype(float))
        return {"features": np.broadcast_to(storm[None, :, None], (len(frame), len(days), 1))}

    return (scenario_receipts,)


@app.cell
def _(
    HORIZON,
    SEED,
    dataclasses,
    demand_scale,
    fitted,
    forecast,
    forecast_dates,
    np,
    sales_covariates,
    scenario_receipts,
):
    # Scale the model's own sales draws; keeping draw identity pairs each path with its posterior draw.
    scenario_sales = dataclasses.replace(
        forecast.sales,
        counts=np.rint(np.asarray(forecast.sales.counts) * demand_scale.value).astype(np.int64),
    )
    scenario = fitted.forecast(
        horizon=HORIZON,
        covariates={"sales": sales_covariates(forecast_dates), "receipts": scenario_receipts},
        future_sales=scenario_sales,
        seed=SEED + 5,
    )
    return (scenario,)


@app.cell(hide_code=True)
def _(
    AMBER,
    MUTED,
    SLATE,
    alt,
    fan,
    figure,
    forecast,
    forecast_dates,
    pd,
    scenario,
    scenario_receipts,
    swatch,
):
    _frame = fan(forecast.dates, forecast.receipts, "Baseline forecast")
    _alt_frame = fan(scenario.dates, scenario.receipts, "Scenario")
    _stormy = scenario_receipts(pd.DataFrame(index=[0]), forecast_dates)["features"][0, :, 0] > 0
    _storm_days = pd.DataFrame({"start": pd.DatetimeIndex(forecast_dates)[_stormy]})
    _storm_days["end"] = _storm_days.start + pd.Timedelta(days=1)
    _x = alt.X("date:T", title=None, axis=alt.Axis(format="%a %b %-d", labelAngle=0))
    _chart = (
        alt.Chart(_storm_days).mark_rect(color=AMBER, opacity=0.14).encode(x="start:T", x2="end:T")
        + alt.Chart(_alt_frame)
        .mark_area(color=SLATE, opacity=0.16)
        .encode(x=_x, y="lo90:Q", y2="hi90:Q")
        + alt.Chart(_frame)
        .mark_line(color=MUTED, strokeDash=[5, 4], strokeWidth=1.8)
        .encode(x=_x, y=alt.Y("mean:Q", title="Returns received per day"))
        + alt.Chart(_alt_frame).mark_line(color=SLATE, strokeWidth=2.4).encode(x=_x, y="mean:Q")
    ).properties(height=240, width="container")
    figure(
        _chart,
        "Scenario receipts vs. the baseline forecast",
        f"{swatch(SLATE)}Scenario mean with 90% interval; {swatch(MUTED, dashed=True)}baseline "
        "mean. Extra sales show up in initiations first and reach the warehouse with a lag. "
        "Weather moves receipts immediately, and a longer storm mostly <em>delays</em> returns "
        "rather than losing them.",
    )
    return


@app.cell(hide_code=True)
def _(cards, forecast, scenario):
    _base = forecast.receipts.sum(axis=1).mean()
    _scen = scenario.receipts.sum(axis=1).mean()
    _base_owed = forecast.initiations.sum(axis=1).mean()
    _scen_owed = scenario.initiations.sum(axis=1).mean()
    cards(
        [
            (
                "Receipts, 28 days",
                f"{_scen:,.0f}<small>{_scen - _base:+,.0f}</small>",
                f"Baseline {_base:,.0f}. The difference is what the warehouse feels this month.",
                "accent",
            ),
            (
                "Returns initiated, 28 days",
                f"{_scen_owed:,.0f}<small>{_scen_owed - _base_owed:+,.0f}</small>",
                f"Baseline {_base_owed:,.0f}. Extra sales show up here first.",
                "amber",
            ),
            (
                "Open at horizon end",
                f"{scenario.open_returns[:, -1].mean():,.0f}"
                f"<small>{scenario.open_returns[:, -1].mean() - forecast.open_returns[:, -1].mean():+,.0f}</small>",
                "Initiated but not yet received on the last forecast day.",
                None,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(callout):
    callout(
        "Worth knowing",
        """A supplied sales scenario *replaces* generated sales; it is not extra data and it
        does not update the posterior. Scaling the model's own sales draws keeps each path
        paired with the posterior draw that produced it, so parameter uncertainty and demand
        uncertainty stay aligned.""",
        tone="warn",
    )
    return


@app.cell(hide_code=True)
def _(section):
    section(
        "simpler",
        "Part 6",
        "Can we do something simpler?",
        "Was all of this necessary? A reasonable tabular baseline: forecast each day's "
        "receipts as the average for that weekday over the last eight weeks.",
    )
    return


@app.cell
def _(daily_history, held_out, pd):
    recent = daily_history.tail(56).assign(weekday=lambda d: d.date.dt.dayofweek)
    weekday_mean = recent.groupby("weekday")["Returns received"].mean()
    naive = pd.Series(
        weekday_mean.reindex(held_out.date.dt.dayofweek).to_numpy(),
        index=held_out.date,
        name="naive",
    )
    return (naive,)


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
    naive,
    pd,
    storm_spans,
    swatch,
):
    _frame = fan(forecast.dates, forecast.receipts, "ttenet").assign(
        naive=naive.to_numpy(), actual=held_out.receipts.to_numpy()
    )
    _x = alt.X("date:T", title=None, axis=alt.Axis(format="%a %b %-d", labelAngle=0))
    _lines = pd.concat(
        [
            _frame[["date", "mean"]].rename(columns={"mean": "value"}).assign(series="ttenet mean"),
            _frame[["date", "naive"]]
            .rename(columns={"naive": "value"})
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
        "The baseline doesn't know about the storm or the backlog",
        f"{swatch(RED)}The weekday average repeats the recent past. {swatch(ACCENT)}ttenet "
        "knows which units are in transit, how old they are, and that the storm will hold "
        f"them up. {swatch(INK)}Dots: held-out actuals.",
    )
    return


@app.cell(hide_code=True)
def _(TABLE_THEME, crps, ct, forecast, held_out, naive, np, pd):
    _actual = held_out.receipts.to_numpy()
    _model_mean = forecast.receipts.mean(axis=0)
    _storm_week = slice(5, 14)
    _scores = pd.DataFrame(
        [
            {
                "Method": "Weekday average (last 8 weeks)",
                "mae": float(np.abs(naive.to_numpy() - _actual).mean()),
                "crps": float(np.abs(naive.to_numpy() - _actual).mean()),
                "storm": float(np.abs(naive.to_numpy()[_storm_week] - _actual[_storm_week]).mean()),
                "total": float(naive.sum() - _actual.sum()),
            },
            {
                "Method": "ttenet",
                "mae": float(np.abs(_model_mean - _actual).mean()),
                "crps": crps(forecast.receipts, _actual),
                "storm": float(np.abs(_model_mean[_storm_week] - _actual[_storm_week]).mean()),
                "total": float(forecast.receipts.sum(axis=1).mean() - _actual.sum()),
            },
        ]
    )
    (
        ct.CoefTable(_scores, rows="Method")
        .estimate("Daily MAE", "mae", fmt=ct.Number(decimals=2))
        .estimate("Daily CRPS", "crps", fmt=ct.Number(decimals=2))
        .estimate("MAE, storm fortnight", "storm", fmt=ct.Number(decimals=2))
        .estimate("28-day total error", "total", fmt=ct.Number(decimals=0, signed=True))
        .with_theme(TABLE_THEME)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For a point forecast, CRPS reduces to absolute error, so the baseline's two columns
    match. The baseline also has no answer at all for *returns still owed*, and its return
    rate, if you tried to build one from recent cohorts, is the collapsing red line from
    the very first chart.
    """)
    return


@app.cell(hide_code=True)
def _(section):
    section("wrap-up", "Conclusion", "First principles, again")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This wasn't a tabular prediction problem, and forcing it into one throws away the
    structure that makes it tractable. Writing down the data generating process turned it
    into three small, explainable pieces:

    1. sales arrive as $\text{sales}_{t,p} \sim \text{Poisson}(\lambda_{t,p})$,
    2. each unit may start a return, on a mixture-cure clock bounded by a 90-day policy,
    3. each initiated return may arrive, on a second mixture-cure clock that respects
       storms and a closed warehouse.

    Fit jointly, those pieces recover parameters the ledger never shows directly, forecast
    daily receipts with uncertainty that held up on a held-out month, put a number on
    returns still owed, and answer "what if?" questions without refitting.

    ### Where this is honest about its limits

    - Finite follow-up can't fully separate "never" from "very late"; the cure fraction
      leans on mature cohorts and the priors. Check calibration on held-out periods.
    - Ages beyond the learned baseline reuse the last bin: flexible, not assumption-free.
    - One event of each type per unit, daily resolution, no repeated return attempts, and
      no inventory feedback. Weather is an input you supply, never forecast.
    - The default posterior is a variational (AutoNormal) approximation. For
      publication-grade intervals, hand `network.numpyro_model(...)` to NUTS.

    ### Run it yourself

    ```bash
    uv sync --all-extras
    uv run marimo edit examples/retail_returns_blog.py
    ```

    The command-line version of the same example, with the same simulator, lives in
    `examples/retail_returns.py`.
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
