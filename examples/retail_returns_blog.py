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
#     "statsmodels>=0.14",
# ]
# [tool.marimo.display]
# theme = "light"
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(
    width="full",
    app_title="Returns you haven't seen yet",
    css_file="retail_returns_blog.css",
)


@app.cell
def _():
    # Kept alone so the page styling below still renders if a heavier import fails.
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import jax
    import jax.numpy as jnp
    import numpy as np
    import numpyro
    import numpyro.distributions as dist
    import numpyro_forecast
    import pandas as pd
    from numpyro_forecast import Horizon, draw_posterior, innovations, predict

    import ttenet
    from ttenet import RetailData, SalesForecast, date_grid

    return (
        Horizon,
        RetailData,
        SalesForecast,
        date_grid,
        dist,
        draw_posterior,
        innovations,
        jax,
        jnp,
        np,
        numpyro,
        numpyro_forecast,
        pd,
        predict,
        ttenet,
    )


@app.cell(hide_code=True)
def _():
    # The interactive figures live beside this notebook: Python computes, they only draw.
    from retail_returns_widgets import ScenarioForecast

    return (ScenarioForecast,)


@app.cell(hide_code=True)
def _():
    # The synthetic world lives in retail_returns_world.py, shared with the command-line
    # example. Receipts there follow the same discretized Weibull law the receipt stage
    # fits, so every fitted receipt parameter has a true value to check.
    from retail_returns_world import (
        initiation_covariates,
        promotions,
        receipt_covariates,
        sales_covariates,
        sales_model,
        simulate_retail,
        storms,
    )

    return (
        initiation_covariates,
        promotions,
        receipt_covariates,
        sales_covariates,
        sales_model,
        simulate_retail,
        storms,
    )


@app.cell(hide_code=True)
def _():
    # How the essay looks (palette, themes, one chart per figure) lives in
    # retail_returns_presentation.py; the numbers behind it in retail_returns_analysis.py.
    from retail_returns_presentation import (
        ACCENT,
        AMBER,
        AMBER_LIGHT,
        INK,
        MODEL_NAME,
        MUTED,
        RATE,
        RATE_LIGHT,
        RED,
        SLATE,
        STORM,
        VIOLET,
        baseline_chart,
        callout,
        cards,
        checks_list,
        cohort_chart,
        convolution_chart,
        convolution_icon,
        drift_chart,
        figure,
        forecast_chart,
        hazard_charts,
        interval_value,
        nav_header,
        owed_chart,
        purchase_timeline,
        recovery_view,
        return_fates_chart,
        sale_echo,
        scores_view,
        section,
        source_md,
        stage_cards,
        swatch,
        tldr_chart,
        wait_charts,
    )

    return (
        ACCENT,
        AMBER,
        AMBER_LIGHT,
        INK,
        MODEL_NAME,
        MUTED,
        RATE,
        RATE_LIGHT,
        RED,
        SLATE,
        STORM,
        VIOLET,
        baseline_chart,
        callout,
        cards,
        checks_list,
        cohort_chart,
        convolution_chart,
        convolution_icon,
        drift_chart,
        figure,
        forecast_chart,
        hazard_charts,
        interval_value,
        nav_header,
        owed_chart,
        purchase_timeline,
        recovery_view,
        return_fates_chart,
        sale_echo,
        scores_view,
        section,
        source_md,
        stage_cards,
        swatch,
        tldr_chart,
        wait_charts,
    )


@app.cell(hide_code=True)
def _():
    from retail_returns_analysis import (
        actual_sales,
        baseline_scores,
        check_closed_days,
        check_counted_once,
        check_sales_propagated,
        crps,
        daily_counts,
        daily_totals,
        eventual_counts,
        hazard_curves,
        held_out_counts,
        lagged_rate,
        post_storm_check,
        recovery_table,
        return_fates,
        sarimax_baseline,
        storm_runs,
        time_between_events,
        typical_return,
        weekday_average,
        weekly_cohort_rates,
    )
    from retail_returns_scenario import scenario_data, storm_labels

    return (
        actual_sales,
        baseline_scores,
        check_closed_days,
        check_counted_once,
        check_sales_propagated,
        crps,
        daily_counts,
        daily_totals,
        eventual_counts,
        hazard_curves,
        held_out_counts,
        lagged_rate,
        post_storm_check,
        recovery_table,
        return_fates,
        sarimax_baseline,
        scenario_data,
        storm_labels,
        storm_runs,
        time_between_events,
        typical_return,
        weekday_average,
        weekly_cohort_rates,
    )


@app.cell(hide_code=True)
def _(nav_header):
    nav_header("Returns forecasting")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html("""
        <div id="top" class="ttn-hero">
          <p class="ttn-kicker">Forecasting · Time-to-event · Supply chain · Bayesian modeling</p>
          <h1 class="ttn-title">Forecasting Returns</h1>
          <p class="ttn-dek">
            Forecasting retail returns highlights interesting first-principles solutions: <b>forecast chaining</b> and <b>time-to-event convolutions</b>. We create a model where we fit the forecast chain once, then push any
            sales forecast through it to get a returns forecast, uncertainty and all.
          </p>
          <ul class="ttn-byline">
            <li><b>Kyle Caron</b></li>
            <li>September 2026</li>
            <li>Built with <b>numpyro_forecast</b>, ttenet (a demo package) &amp; marimo</li>
          </ul>
        </div>
    """)
    return


@app.cell(hide_code=True)
def _(section):
    section(
        "tldr",
        "TL;DR",
        "Returns are a time-dependent chain of forecasts",
        "Next week's warehouse receipts depend on the returns customers start this week, "
        "which depend on what sold last month and on what sells this month. We show how we can "
        "improve inference and forecast performance with a lesser-known solution: forecasting "
        "with a chain of time-to-event convolutions.",
    )
    return


@app.cell(hide_code=True)
def _(HORIZON, daily_history, mo):
    # Days relative to the snapshot: the whole ledger to the end of the forecast. The
    # default shows nine weeks of history.
    tldr_window = mo.ui.range_slider(
        start=1 - len(daily_history),
        stop=HORIZON,
        step=1,
        value=[-62, HORIZON],
        label="History shown",
        show_value=False,
        full_width=True,
    )
    return (tldr_window,)


@app.cell(hide_code=True)
def _(
    AS_OF,
    MUTED,
    SLATE,
    STORM,
    daily_history,
    figure,
    forecast,
    held_out,
    mo,
    pd,
    promo_spans,
    sales_actual,
    sales_draws,
    storms,
    swatch,
    tldr_chart,
    tldr_window,
):
    _past = promo_spans[promo_spans.end <= pd.Timestamp(AS_OF)].iloc[-1]
    _planned = promo_spans[promo_spans.start > pd.Timestamp(AS_OF)].iloc[0]
    _last = _planned.end - pd.Timedelta(days=1)
    _first, _end = (pd.Timestamp(AS_OF) + pd.Timedelta(days=d) for d in tldr_window.value)
    figure(
        mo.vstack(
            [
                tldr_chart(
                    forecast,
                    sales_draws,
                    sales_actual,
                    held_out,
                    storms,
                    promo_spans,
                    daily_history,
                    (_first, _end),
                ),
                mo.hstack(
                    [
                        tldr_window,
                        mo.Html(
                            f'<span class="ttn-range-dates">{_first:%b %-d} – {_end:%b %-d}</span>'
                        ),
                    ],
                    widths=[1, 0],
                    align="center",
                    gap=1,
                ),
            ]
        ),
        "One sales forecast, propagated twice",
        [
            (swatch(MUTED), "Historical observations"),
            (
                swatch(SLATE, block=True),
                f"Sales: the {_past.start:%b} {_past.start.day} sale's returns are in by today; "
                f"the {_planned.start:%b} {_planned.start.day}–{_last.day} sale's land late.",
            ),
            (swatch(STORM, block=True, opacity=0.55), "Storm: receipts stall, then catch up."),
            ("", "Bands are 50% and 90% intervals."),
        ],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The usual shortcut multiplies a sales forecast by a return rate and shifts it by a typical lag. In regular weeks it does fine: steady sales make steady returns. It breaks when something changes: a promo, a storm, a new policy, or a quarter whose recent sales haven't had time to come back yet.

    This post treats returns as a **chain of forecasts** instead (the *chained convolution
    model*, or just *the chain*), built on [numpyro_forecast](https://github.com/juanitorduz/numpyro_forecast): sales is an
    ordinary `numpyro_forecast` model, and each return stage is a time-to-event model. A small demo package,
    [ttenet](https://github.com/kylejcaron/ttenet), wires them together: one call fits the
    chain, one call pushes any sales forecast through it with uncertainty intact, and covariates can impact any part of the chain:
    * Promos, seasonality, stockouts, etc. can impact the demand forecast.
    * Policy changes, fulfillment speed, quality control, etc. can impact the return initiation.
    * Storms, warehouse availability, etc. can impact when returns are actually received.
    """)
    return


@app.cell(hide_code=True)
def _(callout, convolution_icon):
    callout(
        "The trick: a survival convolution",
        """We break returns down into a conversion rate and a time-to-event (survival analysis) process, both downstream of sales. But survival models are famously hard to turn into forecasts, so we use a __survival convolution__.

    The survival convolution spreads each day's sales over the days that follow using the time-to-event curve, then adds them up.

    That gives us properties that match how the
    data is generated: delays like storms or
    warehouse closures are covariates on a
    calendar day, so they hit every order in
    transit that day, whichever day it was sold.
    """,
        icon=convolution_icon(size=48),
    )
    return


@app.cell(hide_code=True)
def _(fitted, forecast, sales_draws):
    from retail_returns_chain import convolution_kernels, follow_sales

    # Every forecast sale day, followed forward nine weeks through both fitted timing curves.
    conv_days = 63
    conv_followed = follow_sales(
        convolution_kernels(fitted), sales_draws.mean(axis=0), forecast.dates[0], conv_days
    )
    return (conv_followed,)


@app.cell(hide_code=True)
def _(AS_OF, HORIZON, forecast, mo, pd, promo_spans):
    _planned = promo_spans[promo_spans.start > pd.Timestamp(AS_OF)].iloc[0].start
    conv_day = mo.ui.slider(
        start=0,
        stop=HORIZON - 1,
        step=1,
        value=int((_planned - pd.Timestamp(forecast.dates[0])).days),
        label="Sale day",
        show_value=False,
        full_width=True,
    )
    return (conv_day,)


@app.cell(hide_code=True)
def _(
    ACCENT,
    AMBER,
    INK,
    conv_day,
    conv_followed,
    convolution_chart,
    figure,
    forecast,
    mo,
    pd,
    sales_draws,
    swatch,
):
    _dates = pd.DatetimeIndex(forecast.dates)
    _day = _dates[conv_day.value]
    _units = sales_draws.mean(axis=0)[conv_day.value]
    figure(
        mo.vstack(
            [
                mo.md(
                    "The plot below shows the convolution in full effect. One day's sales "
                    "are propagated forward into future returns initiated and future returns "
                    "received."
                ),
                convolution_chart(
                    conv_followed, sales_draws.mean(axis=0), _dates[0], conv_day.value
                ),
                mo.hstack(
                    [conv_day, mo.Html(f'<span class="ttn-range-dates">{_day:%a %b %-d}</span>')],
                    widths=[1, 0],
                    align="center",
                    gap=1,
                ),
            ]
        ),
        "Follow one day's sales forward",
        [
            (
                swatch(INK, block=True, opacity=0.9),
                f"{_day:%b %-d}: {_units:.0f} units sold → "
                f"{conv_followed['started'][conv_day.value].sum():.0f} returns started → "
                f"{conv_followed['received'][conv_day.value].sum():.0f} received.",
            ),
            (swatch(AMBER, block=True, opacity=0.9), "Returns started each day."),
            (swatch(ACCENT, block=True, opacity=0.9), "Returns received each day."),
            ("", "Each bar stacks one block per sale day; darker blocks sold later."),
        ],
    )
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
            "Close-ups: how long each step takes, and a sale's returns",
            "What the data can and cannot see yet",
            "Why the return-rate shortcut breaks",
        ),
    )
    return


@app.cell
def _(np, simulate_retail):
    SEED = 21
    TRAINING_DAYS, HORIZON = 180, 28
    VOLUME = 30.0  # scales both products' daily sales
    START = np.datetime64("2026-01-01")
    TRAINING_START = np.datetime64("2026-01-31")
    AS_OF = START + np.timedelta64(TRAINING_DAYS - 1, "D")

    # A simulated world: we get to see every unit's full future, the model never does.
    truth, _ = simulate_retail(SEED, TRAINING_DAYS, HORIZON, volume=VOLUME)
    ledger = truth.drop(columns="abandoned_truth")
    return AS_OF, HORIZON, SEED, START, TRAINING_START, VOLUME, ledger, truth


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
def _(AS_OF, data, mo, purchase_timeline, typical_return):
    mo.vstack(
        [
            mo.md(
                """
    ### Follow one purchase

    Every unit runs on two clocks. The first starts at the sale and stops if the customer
    starts a return, which the policy allows through day 90. The second starts then and
    stops when the parcel is scanned in at the warehouse. Here is one typical return from
    the data:
    """
            ),
            purchase_timeline(typical_return(data.units, AS_OF)),
            # A published page has no Python to page through data, so show a fixed sample.
            mo.accordion(
                {
                    f"A sample of the data: the first 25 of {len(data.units):,} units": mo.ui.table(
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
def _(
    ACCENT,
    AMBER,
    AS_OF,
    data,
    figure,
    mo,
    source_md,
    swatch,
    time_between_events,
    wait_charts,
):
    _to_start, _in_transit = time_between_events(data.units, AS_OF)
    mo.vstack(
        [
            figure(
                wait_charts(_to_start, _in_transit),
                "How long each step takes",
                [
                    (
                        swatch(AMBER, block=True, opacity=0.85),
                        "Time from sale to starting a return.",
                    ),
                    (
                        swatch(ACCENT, block=True, opacity=0.85),
                        "In transit: usually under two weeks.",
                    ),
                ],
            ),
            mo.accordion({"How these waits are computed": source_md(time_between_events)}),
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    ACCENT,
    AMBER,
    AS_OF,
    SLATE,
    daily_history,
    figure,
    pd,
    promo_spans,
    sale_echo,
    swatch,
):
    _sale = promo_spans[promo_spans.end <= pd.Timestamp(AS_OF)].iloc[-1]
    figure(
        sale_echo(daily_history, _sale, before=21),
        "A spike in sales leads to an increase in returns",
        [
            (
                swatch(SLATE, block=True),
                f"The {_sale.start:%b %-d} sale: about four times the usual orders, for one week.",
            ),
            (swatch(AMBER), "Returns started peak about a week later."),
            (swatch(ACCENT), "Returns received peak about two weeks after the sale."),
        ],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Today many units are still mid-story: a return could start tomorrow, or a
    parcel could still be on its way. Those records are **censored**: the clock is still
    running and the ledger cannot say where it will stop. Open returns are the sneakiest.
    In this world about one in five initiated returns is never shipped back, and no
    "abandoned" event is ever recorded, so the model has to learn that share from how long open
    returns stay open.
    """)
    return


@app.cell(hide_code=True)
def _(
    AS_OF,
    HORIZON,
    START,
    daily_counts,
    data,
    np,
    promotions,
    storm_runs,
    storms,
):
    daily_history = daily_counts(data.units, START, AS_OF)
    # The last six observed weeks lead into every forecast figure: long enough to show the
    # May sale and its returns before the snapshot.
    recent_history = daily_history.tail(42)
    # Contiguous storm spans over the full simulated period, for shading.
    storm_spans = storm_runs(storms, START, AS_OF + np.timedelta64(HORIZON, "D"))
    promo_spans = storm_runs(promotions, START, AS_OF + np.timedelta64(HORIZON, "D"))
    return daily_history, promo_spans, recent_history, storm_spans


@app.cell(hide_code=True)
def _(cohort_rates, mo):
    _recent = cohort_rates.tail(4)
    _seen, _eventual = _recent.observed.mean(), _recent.eventual.mean()
    mo.md(rf"""
    ### Why the return-rate shortcut breaks

    The shortcut is one number, *"we expect {_eventual:.0%} of orders to start a return"*, applied
    to sales after a typical lag.

    | What it misses | What goes wrong |
    | :--- | :--- |
    | Recent orders haven't finished returning | Last four weeks: **{_seen:.0%}** returned so far, **{_eventual:.0%}** eventually |
    | Return rates change | A change shows up only once orders finish returning, about 90 days later |
    | Returns spread over weeks | One lag lands a whole sale's returns at once |
    | Storms | Warehouse receipts stall, then catch up |
    | The warehouse shuts at weekends | Receipts pile up on Mondays |
    | Some started returns never ship | Returns started overstate returns received |

    The chain below asks *will* an order be returned and *when* separately, so "not yet"
    never counts as "never".
    """)
    return


@app.cell(hide_code=True)
def _(INK, RED, SLATE, cohort_chart, cohort_rates, figure, swatch):
    figure(
        cohort_chart(cohort_rates),
        "Recent sales look like they never get returned",
        [
            (swatch(RED), "Returned so far: collapses for recent weeks, which haven't had time."),
            (swatch(INK, dashed=True), "Eventually returned (the truth): doesn't fall at all."),
            (
                swatch(SLATE),
                "The model: adds each open unit's chance of a later return. Band: 90% interval.",
            ),
        ],
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
            "Modeling returns: will it, and when?",
            "What each stage learns and what it is told",
            "Why will-it and when are fitted together",
        ),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Modeling returns

    Each return stage is two models in one:
    1. a **conversion rate** (will this order ever be returned?)
    2. a **time to event** (if so, when?).

    An order with no return yet is either one that will never come back, or one that hasn't
    come back *yet*.
    """)
    return


@app.cell(hide_code=True)
def _(
    AMBER,
    AMBER_LIGHT,
    AS_OF,
    figure,
    return_fates,
    return_fates_chart,
    swatch,
    truth,
):
    _fates = return_fates(truth, AS_OF)
    figure(
        return_fates_chart(_fates),
        "Where a batch of orders stands, day by day",
        [
            (
                swatch(AMBER, block=True, opacity=0.9),
                "Returned so far: climbs at the pace of the time to event.",
            ),
            (
                swatch(AMBER_LIGHT, block=True, opacity=0.9),
                "Will return, not yet: silent today, but coming.",
            ),
            (swatch("#e3e5da", block=True, opacity=1), "Never returns: the rest."),
            ("", "Orders sold more than 90 days ago, so every fate is known."),
        ],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    They have to be fitted together, because you can't learn one without the other while the
    data is still censored. Looking back three months, we can see both the return rate and
    how long the returns took. But what about two weeks ago, when many returns haven't had
    time to happen yet? We shouldn't wait 90 days to learn when we can do better.



    ### The full model

    The full model chains three forecasts:
    1. Sales
    2. Returns started
    3. Returns received

    Steps 2 and 3 each get their own conversion rate and time to event estimate.

    Chaining them is the **survival convolution** from the TL;DR: each day's sales spread
    forward through the time to event, then summed, which turns the return stages into a
    daily forecast.
    """)
    return


@app.cell
def _(dist, mo, sales_model):
    from ttenet import CountProcess, EventProcess, RetailReturnModel, WeibullFamily

    model = RetailReturnModel(
        sales=CountProcess(model=sales_model),
        # age_bins=16 specifies discrete hazard functions bins
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
def _(model, stage_cards):
    stage_cards(model.receipt.family)
    return


@app.cell(hide_code=True)
def _(callout):
    callout(
        "One likelihood, both parts",
        r"""An order silent for $a$ days is either never returning or still on its way:
        $(1-\pi) + \pi\,S(a)$, with conversion rate $\pi$ and time-to-event survival $S(a)$.
        That is what fixes the collapsing line in Part 1.""",
    )
    return


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
    `product`, the weekday sine and the promotion flag are defined by formula rather than by
    position.
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

    The same model in more detail. Each tab stands on its own; skip them on a first read and come back when you want the math or the code.
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
    value, including what the data never shows directly: the share of units susceptible
    to returning and the share of initiated returns that ever arrive. The bars put every
    row on one scale, the posterior divided by the truth: a bar that covers 1 contains the
    truth, and a narrow bar is a precise estimate.

    One row misses: the weekday effect on starting a return comes out low. In this world
    nothing can be returned before the parcel arrives on day 3. The model's curve over age
    can drop to zero for those days, but it is built to change smoothly, so it blurs the
    cliff and the weekday effect soaks up the difference. It's a small limitation with an
    easy next step: tell the model returns can't start before delivery.
    """)
    return


@app.cell(hide_code=True)
def _(VOLUME, data, fitted, recovery_table, recovery_view):
    recovery = recovery_table(fitted, data, VOLUME)
    recovery_view(recovery)
    return


@app.cell(hide_code=True)
def _(
    ACCENT,
    AMBER,
    INK,
    figure,
    fitted,
    hazard_charts,
    hazard_curves,
    swatch,
):
    figure(
        hazard_charts(*hazard_curves(fitted)),
        "The two clocks: one flexible, one Weibull",
        [
            (
                swatch(AMBER),
                "Starting a return: a flexible curve. It learns that nothing starts before the "
                "parcel arrives on day 3.",
            ),
            (swatch(ACCENT), "Arriving at the warehouse: a Weibull curve, clear weather."),
            (swatch(INK, dashed=True), "The simulator's true daily chance."),
            ("", "Bands are 90% intervals. Product A, neutral weekday, warehouse open."),
        ],
    )
    return


@app.cell(hide_code=True)
def _(AS_OF, data, fitted, truth, weekly_cohort_rates):
    cohort_rates = weekly_cohort_rates(data.units, truth, fitted, AS_OF)
    return (cohort_rates,)


@app.cell(hide_code=True)
def _(section):
    section(
        "forecast",
        "Part 4",
        "Does it work?",
        "The fitted chain forecasts the next four weeks: it simulates new sales and pushes every "
        "cohort, old and new, through both clocks. Then we score it on held-out days, against "
        "three planning shortcuts and a standard time-series model.",
        points=(
            "Receipts against held-out days",
            "Four baselines, scored on held-out days",
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
        covariates={"sales": sales_covariates(forecast_dates)},  # weekdays and the sale calendar
        seed=SEED + 5,
    )
    mo.show_code(position="above")
    return (forecast,)


@app.cell(hide_code=True)
def _(AS_OF, eventual_counts, forecast, held_out_counts, truth):
    held_out = held_out_counts(truth, forecast.dates)
    eventual_truth = eventual_counts(truth, AS_OF)
    return eventual_truth, held_out


@app.cell(hide_code=True)
def _(actual_sales, daily_totals, forecast, truth):
    # Daily totals of the sales forecast that feeds the chain, and what actually sold.
    sales_draws = daily_totals(forecast.sales, forecast.dates)
    sales_actual = actual_sales(truth, forecast.dates)
    return sales_actual, sales_draws


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The TL;DR's "Follow one day's sales forward" figure shows how that call builds the forecast: each day's sales spread
    forward through both clocks, then added up.
    """)
    return


@app.cell(hide_code=True)
def _(forecast, held_out, post_storm_check, storm_spans):
    receipt_check = post_storm_check(forecast, held_out, storm_spans)
    return (receipt_check,)


@app.cell(hide_code=True)
def _(
    AMBER,
    MUTED,
    SLATE,
    figure,
    forecast,
    forecast_chart,
    held_out,
    promo_spans,
    receipt_check,
    recent_history,
    storm_spans,
    swatch,
):
    figure(
        forecast_chart(forecast, held_out, storm_spans, promo_spans, recent_history),
        "Daily forecasts against what actually happened",
        [
            (swatch(MUTED), "Historical observations"),
            (swatch(SLATE, block=True), "The May and July sales."),
            (
                swatch(AMBER, block=True),
                "Storm: receipts stall, then the backlog drains. "
                f"{receipt_check['date']:%a %b} {receipt_check['date'].day}: forecast "
                f"{receipt_check['mean']:.0f} ({receipt_check['lo']:.0f}–{receipt_check['hi']:.0f}), "
                f"actual {receipt_check['actual']}.",
            ),
            ("", "Bands are 50% and 90% intervals."),
        ],
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
                f"Actual: <b>{held_out.receipts.sum():,}</b>",
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
                "Lower is better.",
                None,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comparing against four baselines

    Three common shortcuts for warehouse planning, and one standard forecaster:

    - **Weekday average:** each weekday's receipts over the last eight weeks.
    - **Settled return rate × sales, lagged:** the shortcut from the introduction, at its best.
      The rate and lag come from orders 60–115 days old, whose returns have nearly all
      arrived; the lag is rounded to whole weeks.
    - **Return rate so far × sales, lagged:** the same, with the rate taken naively from
      recent orders.
    - **SARIMAX with lagged sales + return initiations:** a seasonal ARIMA model of daily
      receipts. Its regressors are average sales over windows one to ten weeks back, average
      initiations over windows one to 30 days back, and weekend, Monday and storm indicators.
      It learns its own delay curves, one weight per window, none below zero (a sale can't
      remove returns). Initiations after today come from a second SARIMAX on lagged sales.

    The rate shortcuts and SARIMAX use the chain's own sales forecast and know the warehouse
    is shut at weekends; SARIMAX also gets the storm dates.
    """)
    return


@app.cell
def _(
    AS_OF,
    daily_history,
    data,
    forecast,
    held_out,
    lagged_rate,
    sales_draws,
    sarimax_baseline,
    storm_spans,
    weekday_average,
):
    weekday_avg = weekday_average(daily_history, held_out)
    _sales = sales_draws.mean(axis=0)
    # The careful shortcut: a rate from orders old enough to have finished returning.
    lagged = lagged_rate(data.units, AS_OF, _sales, forecast.dates, lag_step=7)
    # The naive one: returns so far over units sold in the last eight weeks.
    naive_lagged = lagged_rate(
        data.units, AS_OF, _sales, forecast.dates, mature_days=0, cohort_days=56, lag_step=7
    )
    baselines = {
        "Weekday average (last 8 weeks)": weekday_avg,
        f"Settled return rate × sales, lagged {lagged.attrs['lag']} days": lagged,
        f"Return rate so far × sales, lagged {naive_lagged.attrs['lag']} days": naive_lagged,
    }

    # A standard forecaster: SARIMAX on daily receipts with lagged sales and lagged return
    # initiations as regressors, given the same sales forecast and storm dates as the chain.
    sarimax = sarimax_baseline(daily_history, _sales, forecast.dates, storm_spans)
    baselines["SARIMAX with lagged sales + return initiations"] = sarimax

    return baselines, lagged, naive_lagged, sarimax


@app.cell(hide_code=True)
def _(daily_history, forecast, sales_draws, sarimax_baseline, storm_spans):
    # How much SARIMAX's forecast depends on the sales lag windows it is given: other reasonable
    # choices, each scored on nothing but its 28-day total. Fits that fail to converge are left out.
    _weekly = [(7 + 7 * i, 13 + 7 * i) for i in range(12)]
    _fits = {
        _name: sarimax_baseline(
            daily_history, sales_draws.mean(axis=0), forecast.dates, storm_spans, sales_lags=_lags
        )
        for _name, _lags in {
            "weekly windows to 90 days": _weekly,
            "weekly to 48 days, then 49-90": _weekly[:6] + [(49, 90)],
            "two-week windows to 90 days": [
                (7, 20),
                (21, 34),
                (35, 48),
                (49, 62),
                (63, 76),
                (77, 90),
            ],
            "four windows to 90 days": [(7, 13), (14, 27), (28, 55), (56, 90)],
            "daily to 30 days, then 31-60 and 61-90": [(k, k) for k in range(7, 31)]
            + [(31, 60), (61, 90)],
        }.items()
    }
    sarimax_lag_totals = {_n: float(_f.sum()) for _n, _f in _fits.items() if _f.attrs["converged"]}
    return (sarimax_lag_totals,)


@app.cell(hide_code=True)
def _(baselines, mo):
    _short = ["Weekday average", "Settled return rate", "Return rate so far", "SARIMAX"]
    baseline_pick = mo.ui.radio(
        options=dict(zip(_short, baselines)),
        value="SARIMAX",
        label="Compare the chain with",
        inline=True,
    )
    return (baseline_pick,)


@app.cell(hide_code=True)
def _(
    ACCENT,
    AMBER,
    MUTED,
    RATE,
    RATE_LIGHT,
    RED,
    SLATE,
    VIOLET,
    baseline_chart,
    baseline_pick,
    baselines,
    figure,
    forecast,
    held_out,
    lagged,
    mo,
    naive_lagged,
    np,
    promo_spans,
    recent_history,
    storm_spans,
    swatch,
):
    _names = list(baselines)
    _style = {
        _names[0]: (
            RED,
            "Repeating recent receipts misses the planned sale",
            "Weekday average of the last eight weeks: can't know about July's sale.",
        ),
        _names[1]: (
            RATE,
            "A single lag piles each sale's returns into one week",
            f"{lagged.attrs['rate']:.0%} of sales reach the warehouse (the rate settled orders show), "
            f"{lagged.attrs['lag']} days later: each sale's returns land at once.",
        ),
        _names[2]: (
            RATE_LIGHT,
            "Counting returns so far undercounts every sale",
            f"The same, with the rate so far from recent orders "
            f"({naive_lagged.attrs['rate']:.0%}): too low, they haven't finished returning.",
        ),
        _names[3]: (
            VIOLET,
            "A standard time-series model, given every upstream series",
            "SARIMAX on daily receipts with lagged sales and return initiations.",
        ),
    }
    _color, _title, _row = _style[baseline_pick.value]
    # One y-axis for every choice: the tallest of all shortcuts, the chain's band, the actuals
    # and the ledger lead-in, so switching doesn't rescale the chart.
    _y_max = 1.05 * max(
        max(float(np.max(v)) for v in baselines.values()),
        float(np.quantile(forecast.receipts, 0.95, axis=0).max()),
        float(held_out.receipts.max()),
        float(recent_history["Returns received"].max()),
    )
    figure(
        mo.vstack(
            [
                baseline_pick,
                baseline_chart(
                    forecast,
                    held_out,
                    {baseline_pick.value: baselines[baseline_pick.value]},
                    storm_spans,
                    promo_spans,
                    recent_history,
                    colors=[_color],
                    y_max=_y_max,
                ),
            ]
        ),
        _title,
        [
            (swatch(_color, dashed=True), _row),
            (
                swatch(ACCENT),
                "The chain: tracks every unit sold, simulates the July sale, expects the storm.",
            ),
            (swatch(MUTED), "Historical observations"),
            (swatch(SLATE, block=True), "The May and July sales."),
            (swatch(AMBER, block=True), "The storm."),
        ],
    )
    return


@app.cell(hide_code=True)
def _(MODEL_NAME, baseline_scores, baselines, forecast, held_out, scores_view):
    receipt_scores = baseline_scores(forecast, held_out, baselines, MODEL_NAME)
    scores_view(receipt_scores, held_out.receipts.sum())
    return (receipt_scores,)


@app.cell(hide_code=True)
def _(
    AS_OF,
    held_out,
    lagged,
    mo,
    np,
    receipt_scores,
    sarimax,
    sarimax_lag_totals,
    truth,
):
    _weekday, _lagged, _naive, _sarimax, _model = (receipt_scores.iloc[i] for i in range(5))
    _actual = held_out.receipts.sum()
    _peak = int(np.argmax(lagged.to_numpy()))
    _day = held_out.date.iloc[_peak]
    _from_new_sales = (
        (truth.sale_date > AS_OF) & truth.receipt_date.isin(held_out.date)
    ).sum() / _actual
    _pile = lagged.iloc[_peak] / held_out.receipts.iloc[_peak]
    _best = min(_weekday.mae, _lagged.mae, _naive.mae, _sarimax.mae)
    _band = np.quantile(sarimax.attrs["draws"], [0.05, 0.95], axis=0)
    _sarimax_cover = np.mean((held_out.receipts >= _band[0]) & (held_out.receipts <= _band[1]))
    _lag_totals = list(sarimax_lag_totals.values())
    _covers = (
        ", and its interval covers the actual"
        if _model.total_lo <= _actual <= _model.total_hi
        else ""
    )
    mo.md(rf"""
    *(For a point forecast CRPS equals absolute error, so the shortcuts' two columns match.)*

    - **Weekday average** misses {1 - _weekday.total / _actual:.0%} of the month. Much of this
      window's returns come from the July sale, which hadn't happened yet ({_from_new_sales:.0%}
      of receipts), and no history of past receipts can contain them.
    - **Settled return rate** gets the month about right but the days wrong. One lag drops the
      whole sale's returns on a single day, {_day:%A %b} {_day.day}: {_pile:.1f} times what
      actually arrived.
    - **Return rate so far** misses {1 - _naive.total / _actual:.0%}. Recent orders that haven't
      returned *yet* count as *never*, so its rate comes out too low.
    - **SARIMAX** is the strongest baseline, but misses {abs(_sarimax.total / _actual - 1):.0%} of
      the month. It learns its delay curves from daily totals, where only one past sale stands
      out, so its 90% intervals hold {_sarimax_cover:.0%} of days, and other reasonable sales lag
      windows put the month anywhere from {min(_lag_totals):,.0f} to {max(_lag_totals):,.0f}.
    - **The chain** lands within {abs(_model.total / _actual - 1):.0%} of the month{_covers}.
      Its daily error is {_best / _model.mae:.1f} times smaller than the best baseline's, and
      it is the only one that can say how many returns are still owed or what a storm would do.
    """)
    return


@app.cell(hide_code=True)
def _(
    ACCENT,
    AMBER,
    INK,
    eventual_truth,
    figure,
    forecast,
    mo,
    owed_chart,
    swatch,
):
    figure(
        mo.vstack(
            [
                mo.md(
                    "How many units **already sold** will eventually come back? No finite "
                    "horizon answers that, but the chain can, and it can say *why*."
                ),
                owed_chart(forecast, eventual_truth),
            ]
        ),
        "An interpretable model: where the owed returns are",
        [
            (swatch(ACCENT, block=True, opacity=0.9), "In transit."),
            (
                swatch(AMBER, block=True, opacity=0.9),
                "Not started yet: uncertain, spread over weeks.",
            ),
            (swatch(INK, block=True, opacity=0.9), "Total: the refund reserve."),
            ("", "Each line is a 90% interval; the tick marks the actual observation."),
        ],
    )
    return


@app.cell(hide_code=True)
def _(section):
    section(
        "scenarios",
        "Part 5",
        "Asking “what if?”",
        "This is where the chain pays off. Its inputs can change without a refit: the storm "
        "forecast feeding the receipt stage, or the sales forecast feeding the whole chain. The "
        "fitted return stages stay exactly as they are.",
        points=(
            "Choose a storm scenario",
            "Receipts and returns in transit against the forecast storm",
            "A drifting sales model through the same stages",
        ),
    )
    return


@app.cell(hide_code=True)
def _(AS_OF, storm_labels, storm_spans):
    storm_start, weather_labels = storm_labels(storm_spans, AS_OF)
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
def _(
    HORIZON,
    SEED,
    fitted,
    forecast,
    forecast_dates,
    sales_covariates,
    storm_scenario,
):
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
    scenario_data,
    scenario_runs,
    storm_scenario,
    storms,
    weather_labels,
):
    mo.ui.anywidget(
        ScenarioForecast(
            data=scenario_data(
                AS_OF, forecast, scenario_runs, weather_labels, storms, storm_scenario
            )
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
def _(mo):
    mo.md(r"""
    ### A second sales model

    The alternative is a **local level**: each product's log sales rate takes a small
    random-walk step every day, so the forecast can follow a drifted level instead of
    returning to a fixed mean. It is ordinary numpyro_forecast code (`Horizon`, `predict`,
    `innovations`), fitted on the training sales alone, and its 28-day draws go through the
    return stages fitted in Part 3 by row: sales draw
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
        promo_lift = numpyro.sample("promo_lift", dist.Normal(0, 1))
        with numpyro.plate("product", 2, dim=-1):  # one random walk per product
            steps = innovations(h, "steps", dist.Normal(0.0, 1.0))
        level = level0 + jnp.cumsum(steps * drift_scale, axis=-2)
        calendar = covariates[..., :1] * weekday_effect + covariates[..., 1:] * promo_lift
        predict(h, lambda value: dist.Poisson(jnp.exp(value)), level + calendar)

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
    from numpyro.infer import SVI, Trace_ELBO, init_to_median
    from numpyro.infer.autoguide import AutoNormal

    # Fit on the training window only: data.sales stops at the as_of snapshot.
    training_sales = jnp.asarray(data.sales)  # [day, product] counts
    # Start every random-walk step at its prior median (zero): a flat level. The default
    # random start draws each step anywhere in (-2, 2), and SVI can settle with the walk
    # soaking up the weekday swing.
    guide = AutoNormal(drifting_sales_model, init_loc_fn=init_to_median)
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
    return drift_forecast, drift_sales, sales_posterior


@app.cell(hide_code=True)
def _(daily_totals, drift_forecast):
    drift_sales_draws = daily_totals(drift_forecast.sales, drift_forecast.dates)
    return (drift_sales_draws,)


@app.cell(hide_code=True)
def _(
    INK,
    SLATE,
    data,
    drift_chart,
    drift_forecast,
    drift_sales_draws,
    figure,
    forecast,
    held_out,
    np,
    sales_actual,
    sales_draws,
    sales_posterior,
    swatch,
):
    _order = list(data.groups["product"].to_numpy(int))
    _scale = np.asarray(sales_posterior["drift_scale"]).mean(axis=0)
    figure(
        drift_chart(
            forecast, drift_forecast, sales_draws, drift_sales_draws, sales_actual, held_out
        ),
        "Two sales forecasters, one set of fitted return stages",
        [
            ("", "Solid lines and light bands: the Part 4 forecast."),
            (
                swatch(SLATE, dashed=True),
                "A drifting-level sales model, pushed through the same return stages.",
            ),
            (swatch(INK), "Dots: what actually happened."),
            (
                "",
                f"Bands are 90% intervals. The drift is day-to-day noise (daily step "
                f"{_scale[_order.index(0)]:.3f} for A, {_scale[_order.index(1)]:.3f} for B), "
                "not a trend.",
            ),
        ],
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
    interval_value,
    sales_actual,
    sales_draws,
):
    _actual = held_out.receipts.to_numpy()
    cards(
        [
            (
                "Units sold, next 28 days",
                interval_value(drift_sales_draws.sum(axis=1)),
                f"Drifting level. Part 4 model: <b>{sales_draws.sum(axis=1).mean():,.0f}</b>. "
                f"Actual: <b>{sales_actual.sum():,}</b>.",
                None,
            ),
            (
                "Returns started",
                interval_value(drift_forecast.initiations.sum(axis=1)),
                f"Part 4 model: <b>{forecast.initiations.sum(axis=1).mean():,.0f}</b>. "
                f"Actual: <b>{held_out.initiations.sum():,}</b>.",
                "amber",
            ),
            (
                "Returns received",
                interval_value(drift_forecast.receipts.sum(axis=1)),
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
def _(
    check_closed_days,
    check_counted_once,
    check_sales_propagated,
    checks_list,
    data,
    drift_forecast,
    drift_sales,
    mo,
):
    mo.accordion(
        {
            "Consistency checks on the propagated forecast": checks_list(
                [
                    check_counted_once(data.units, drift_forecast),
                    check_sales_propagated(drift_forecast, drift_sales),
                    check_closed_days(drift_forecast),
                ]
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(section):
    section("wrap-up", "Conclusion", "Chain the forecasts")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Returns forecasting is an old problem. Splitting it into a chain of smaller forecasts, with time-to-event convolutions to link them makes it interpretable and outperforms classical baselines.


    While the convolutions were difficult to implement in `ttenet` (a demo package for time-to-event bayesian networks), `numpyro_forecast` makes using them and chaining forecasts easy.

    This allows us to match the data generating process in our model, and build a better and more interpretable forecast than we could have otherwise.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
