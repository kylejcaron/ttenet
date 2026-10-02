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
    from retail_returns_widgets import (
        ChainStepper,
        PurchaseStory,
        ScenarioForecast,
    )

    return ChainStepper, PurchaseStory, ScenarioForecast


@app.cell(hide_code=True)
def _():
    # The synthetic world lives in retail_returns_world.py, shared with the command-line
    # example. Receipts there follow the same discretized Weibull law the receipt stage
    # fits, so every fitted receipt parameter has a true value to check.
    from retail_returns_world import (
        initiation_covariates,
        receipt_covariates,
        sales_covariates,
        sales_model,
        simulate_retail,
        storms,
    )

    return (
        initiation_covariates,
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
        INK,
        MODEL_NAME,
        MUTED,
        RED,
        SLATE,
        STORM,
        baseline_chart,
        callout,
        cards,
        checks_list,
        cohort_chart,
        daily_history_chart,
        drift_chart,
        figure,
        forecast_chart,
        hazard_charts,
        interval_value,
        nav_header,
        owed_chart,
        recovery_view,
        scores_view,
        section,
        source_md,
        stage_cards,
        status_figure,
        swatch,
        tldr_chart,
    )

    return (
        ACCENT,
        AMBER,
        INK,
        MODEL_NAME,
        MUTED,
        RED,
        SLATE,
        STORM,
        baseline_chart,
        callout,
        cards,
        checks_list,
        cohort_chart,
        daily_history_chart,
        drift_chart,
        figure,
        forecast_chart,
        hazard_charts,
        interval_value,
        nav_header,
        owed_chart,
        recovery_view,
        scores_view,
        section,
        source_md,
        stage_cards,
        status_figure,
        swatch,
        tldr_chart,
    )


@app.cell(hide_code=True)
def _():
    from retail_returns_analysis import (
        abandoned_share,
        actual_sales,
        baseline_scores,
        check_closed_days,
        check_counted_once,
        check_owed_decomposes,
        check_sales_propagated,
        classify_units,
        crps,
        daily_counts,
        daily_totals,
        eventual_counts,
        hazard_curves,
        held_out_counts,
        post_storm_check,
        recovery_table,
        storm_runs,
        weekday_average,
        weekly_cohort_rates,
    )
    from retail_returns_scenario import scenario_data, storm_labels

    return (
        abandoned_share,
        actual_sales,
        baseline_scores,
        check_closed_days,
        check_counted_once,
        check_owed_decomposes,
        check_sales_propagated,
        classify_units,
        crps,
        daily_counts,
        daily_totals,
        eventual_counts,
        hazard_curves,
        held_out_counts,
        post_storm_check,
        recovery_table,
        scenario_data,
        storm_labels,
        storm_runs,
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
          <h1 class="ttn-title">Returns you haven't seen yet</h1>
          <p class="ttn-dek">
            Forecasting retail returns as a chain of forecasts. Sales feed return
            initiations, which feed warehouse receipts. Fit the chain once, then push any
            sales forecast through it, uncertainty and all.
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
    STORM,
    figure,
    forecast,
    held_out,
    sales_actual,
    sales_draws,
    storms,
    swatch,
    tldr_chart,
):
    figure(
        tldr_chart(forecast, sales_draws, sales_actual, held_out, storms),
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
def _(cards, forecast, held_out, interval_value, sales_actual, sales_draws):
    cards(
        [
            (
                "Units sold, next 28 days",
                interval_value(sales_draws.sum(axis=1)),
                f"The sales forecast going in. Actual: <b>{sales_actual.sum():,}</b>.",
                None,
            ),
            (
                "Returns started",
                interval_value(forecast.initiations.sum(axis=1)),
                f"Propagated once. Actual: <b>{held_out.initiations.sum():,}</b>.",
                "amber",
            ),
            (
                "Returns received",
                interval_value(forecast.receipts.sum(axis=1)),
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
    as a **chain of forecasts** instead, built on
    [numpyro_forecast](https://github.com/juanitorduz/numpyro_forecast): sales is an
    ordinary numpyro_forecast model, and each return stage is a time-to-event model that
    speaks the same protocol. A small demo package,
    [ttenet](https://github.com/kylejcaron/ttenet), wires them together: one call fits the
    chain, one call pushes any sales forecast through it with uncertainty intact, and a
    storm scenario or a different sales model is one argument.
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
    VOLUME = 10.0  # scales both products' daily sales
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
def _(AS_OF, classify_units, data):
    # RetailData imposes no deadline; the 90-day policy belongs to the initiation process.
    unit_status = classify_units(data.units, AS_OF)
    return (unit_status,)


@app.cell(hide_code=True)
def _(AS_OF, PurchaseStory, data, mo, pd, storms, truth):
    from retail_returns_story import purchase_story

    _story = purchase_story(truth, AS_OF, storms)
    _sale = pd.Timestamp(_story["sale"])
    _sold = f"{_sale:%A, %B} {_sale.day}"
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
def _(AS_OF, abandoned_share, data, status_figure, truth, unit_status):
    status_figure(unit_status, abandoned_share(data.units, unit_status, truth), AS_OF)
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
def _(AS_OF, HORIZON, START, daily_counts, data, np, storm_runs, storms):
    daily_history = daily_counts(data.units, START, AS_OF)
    # Contiguous storm spans over the full simulated period, for shading.
    storm_spans = storm_runs(storms, START, AS_OF + np.timedelta64(HORIZON, "D"))
    return daily_history, storm_spans


@app.cell(hide_code=True)
def _(
    AMBER,
    TRAINING_START,
    daily_history,
    daily_history_chart,
    figure,
    storm_spans,
    swatch,
):
    figure(
        daily_history_chart(daily_history, storm_spans, TRAINING_START),
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
def _(RED, SLATE, cohort_chart, cohort_rates, figure, swatch):
    figure(
        cohort_chart(cohort_rates),
        "Recent sales look like they never get returned",
        f"{swatch(RED)}<b>Returns so far</b> collapse for recent cohorts: those units simply "
        "haven't had time. The truth (dashed) doesn't fall at all. The model's estimate "
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

    Sales are an ordinary [numpyro_forecast](https://github.com/juanitorduz/numpyro_forecast)
    model (`sales_model`, in the tabs below): `Horizon` and `predict` split the observed
    days from the forecast days. Each return stage is an `EventProcess`: initiation takes
    the defaults, a flexible random-walk hazard with a 90-day deadline; receipt uses the
    packaged Weibull family on a warehouse that opens Monday to Friday. `dist` is
    `numpyro.distributions`.
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
def _(model, stage_cards):
    stage_cards(model.receipt.family)
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
        "Built on numpyro_forecast",
        """Every stage speaks the numpyro_forecast protocol. Sales is any model written with
        `Horizon` and `predict`, so you can bring your own. Each return stage is observed
        through `predict` too, with a time-to-event law in place of a count. Forecasting
        replays every posterior draw through those same programs, and a different
        numpyro_forecast sales model can replace the first stage without touching the other
        two (Part 6).""",
    )
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
def _(VOLUME, data, fitted, recovery_table, recovery_view):
    recovery = recovery_table(fitted, data, VOLUME)
    recovery_view(recovery)
    return (recovery,)


@app.cell(hide_code=True)
def _(ACCENT, AMBER, INK, figure, fitted, hazard_charts, hazard_curves, swatch):
    figure(
        hazard_charts(*hazard_curves(fitted)),
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
def _(AS_OF, data, fitted, truth, weekly_cohort_rates):
    cohort_rates = weekly_cohort_rates(data.units, truth, fitted, AS_OF)
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
def _(forecast, held_out, post_storm_check, storm_spans):
    receipt_check = post_storm_check(forecast, held_out, storm_spans)
    return (receipt_check,)


@app.cell(hide_code=True)
def _(
    INK,
    figure,
    forecast,
    forecast_chart,
    held_out,
    receipt_check,
    storm_spans,
    swatch,
):
    figure(
        forecast_chart(forecast, held_out, storm_spans),
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
    sold** will eventually come back? The fitted chain computes it per posterior draw and
    splits it by where each unit is today.
    """)
    return


@app.cell(hide_code=True)
def _(AMBER, eventual_truth, figure, forecast, owed_chart, swatch):
    figure(
        owed_chart(forecast, eventual_truth),
        "Where the owed returns are hiding",
        "Bars are posterior means with 90% intervals; the black tick is how many actually "
        f"arrived in the simulated future. Most owed returns sit with customers who haven't "
        f'clicked "return" yet {swatch(AMBER)}, the population a return-rate shortcut '
        "undercounts. Eventual counts assume the warehouse keeps opening on weekdays.",
    )
    return


@app.cell(hide_code=True)
def _(
    check_closed_days,
    check_counted_once,
    check_owed_decomposes,
    checks_list,
    data,
    forecast,
):
    checks_list(
        [
            check_counted_once(data.units, forecast),
            check_closed_days(forecast),
            check_owed_decomposes(forecast),
        ]
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
def _(mo):
    mo.md(r"""
    ### Against a simple baseline

    A common shortcut for warehouse planning: forecast each day's receipts as the average
    for that weekday over the last eight weeks.
    """)
    return


@app.cell
def _(daily_history, held_out, weekday_average):
    weekday_avg = weekday_average(daily_history, held_out)
    return (weekday_avg,)


@app.cell(hide_code=True)
def _(
    ACCENT, INK, RED, baseline_chart, figure, forecast, held_out, storm_spans, swatch, weekday_avg
):
    figure(
        baseline_chart(forecast, held_out, weekday_avg, storm_spans),
        "The baseline doesn't know about the storm or the backlog. The model forecasts "
        "a post-storm catch-up, but does not guarantee a one-day spike.",
        f"{swatch(RED)}The weekday average repeats the recent past. {swatch(ACCENT)}The chain "
        "knows which units are in transit, how old they are, and that the storm will hold "
        f"them up. {swatch(INK)}Dots: held-out actuals.",
    )
    return


@app.cell(hide_code=True)
def _(MODEL_NAME, baseline_scores, forecast, held_out, scores_view, weekday_avg):
    scores_view(
        baseline_scores(forecast, held_out, weekday_avg, MODEL_NAME),
        held_out.receipts.sum(),
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
    sales then initiation then receipt, is what makes it convenient. Each stage is a
    numpyro_forecast program, so the chain inherits its protocol: write the sales model with
    `Horizon` and `predict`, fit, and forecast. ttenet, a small demo package, fits the chain
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
