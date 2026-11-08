"""Data-science helpers for the returns essay: tables and numbers, no drawing.

Everything here takes plain arrays, frames and fitted objects and returns plain arrays and
frames, so each piece can be read, tested and reused without marimo or a plotting library.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from retail_returns_world import (
    HAZARD_PRODUCT,
    HAZARD_WEEKDAY,
    PROMO_LIFT,
    RECEIPT_RECEIVABLE,
    RECEIPT_SCALE,
    RECEIPT_SHAPE,
    RECEIPT_STORM,
    SALES_RATES,
    SUSCEPTIBLE,
    SUSCEPTIBLE_PRODUCT,
    WEEKDAY_SWING,
    sigmoid,
    true_initiation_hazard,
    true_receipt_hazard,
)

from ttenet import TimingInputs

POLICY_DAYS = 90


# --- Forecast summaries ---------------------------------------------------------------


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


def bands(draws):
    """Mean and 50%/90% predictive intervals of a [draw, day] array, as JSON-ready lists."""
    draws = np.asarray(draws, dtype=float)
    lo90, lo50, hi50, hi90 = np.quantile(draws, [0.05, 0.25, 0.75, 0.95], axis=0)
    return {
        "mean": draws.mean(axis=0).tolist(),
        "lo90": lo90.tolist(),
        "hi90": hi90.tolist(),
        "lo50": lo50.tolist(),
        "hi50": hi50.tolist(),
    }


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


# --- What the ledger shows ------------------------------------------------------------


def return_fates(truth, as_of, policy_days=90):
    """In hindsight, where a batch of orders stands at each age: returned so far, will
    return but hasn't yet, or never returns.

    The batch is every order sold at least ``policy_days`` days before ``as_of`` that
    could still be followed to the end of its return window, so its fate is known.
    Shares are of the batch, by age in days since the sale.
    """
    cutoff = pd.Timestamp(str(as_of)) - pd.Timedelta(days=policy_days)
    batch = truth[truth.sale_date <= cutoff]
    started = (batch.initiation_date - batch.sale_date).dt.days  # NaN: never starts
    ages = np.arange(policy_days + 1)
    so_far = np.array([(started <= age).mean() for age in ages])
    eventually = started.notna().mean()
    return pd.DataFrame(
        {
            "age": ages,
            "Returned so far": so_far,
            "Will return, not yet": eventually - so_far,
            "Never returns": 1 - eventually,
        }
    )


def time_between_events(units, as_of):
    """Days from sale to return started, and from return started to received.

    Uses only what the ledger shows by ``as_of``; units still waiting are left out.
    """
    today = pd.Timestamp(str(as_of))
    started = units[units.initiation_date <= today]  # missing dates compare False
    received = started[started.receipt_date <= today]
    to_start = (started.initiation_date - started.sale_date).dt.days
    in_transit = (received.receipt_date - received.initiation_date).dt.days
    return to_start, in_transit


def typical_return(units, as_of):
    """One received return whose two waits are both typical: the unit closest to the median
    days from sale to return start and the median days in transit, among units the ledger
    shows received by ``as_of``. Ties go to the earliest row."""
    sold = units.sale_date.to_numpy().astype("datetime64[D]")
    started = units.initiation_date.to_numpy().astype("datetime64[D]")
    received = units.receipt_date.to_numpy().astype("datetime64[D]")
    done = ~np.isnat(received) & (received <= np.datetime64(as_of, "D"))
    if not done.any():
        raise ValueError(f"no unit is received by {as_of}")
    wait = (started - sold).astype(int)[done]
    transit = (received - started).astype(int)[done]
    distance = np.abs(wait - np.median(wait)) + np.abs(transit - np.median(transit))
    row = np.flatnonzero(done)[np.argmin(distance)]
    return {
        "sold": pd.Timestamp(sold[row]),
        "started": pd.Timestamp(started[row]),
        "received": pd.Timestamp(received[row]),
    }


def daily_counts(units, start, as_of):
    """Daily sales, returns initiated and returns received over ``[start, as_of]``."""
    days = pd.date_range(str(start), str(as_of), freq="D")

    def per_day(column):
        return units.groupby(column).size().reindex(days, fill_value=0).values

    return pd.DataFrame(
        {
            "date": days,
            "Sales": per_day("sale_date"),
            "Returns initiated": per_day("initiation_date"),
            "Returns received": per_day("receipt_date"),
        }
    )


def storm_runs(storms, start, end):
    """One row per contiguous run of storm days in ``[start, end]``, with an exclusive end."""
    days = pd.date_range(str(start), str(end), freq="D")
    flag = storms(days.values.astype("datetime64[D]")) > 0
    edges = np.flatnonzero(np.diff(np.r_[0, flag.astype(int), 0]))
    return pd.DataFrame(
        {
            "start": days[edges[::2]],
            "end": days[edges[1::2] - 1] + pd.Timedelta(days=1),
        }
    )


def weekly_cohort_rates(units, truth, fitted, as_of, min_units=60):
    """Weekly sale cohorts: share returned so far, share that eventually returns, and the model.

    The model's belief about each still-eligible unit is the probability it starts a return
    before its deadline, given it has been quiet through the snapshot (the censoring formula),
    evaluated per unit and per posterior draw.
    """
    params = fitted.initiation_fit.parameters
    age_logits = np.asarray(params.age_logits)
    beta = np.asarray(params.beta)
    susceptibility_intercept = np.asarray(params.susceptibility_intercept)
    susceptibility_beta = np.asarray(params.susceptibility_beta)[:, 0]

    eligible = units[
        units.initiation_date.isna()
        & ((as_of - units.sale_date.to_numpy().astype("datetime64[D]")).astype(int) < POLICY_DAYS)
    ]
    sale = eligible.sale_date.to_numpy().astype("datetime64[D]")
    product = eligible["product"].to_numpy(float)
    ages = np.arange(POLICY_DAYS + 1)
    days = sale[:, None] + ages[None, :].astype("timedelta64[D]")
    weekday = ((days.astype("datetime64[D]").astype(int) + 3) % 7).astype(float)  # Mon=0
    season = np.sin(2 * np.pi * weekday / 7)
    quiet = ages[None, :] <= (as_of - sale).astype(int)[:, None]
    age_index = np.minimum(ages, age_logits.shape[1] - 1)

    p_future = np.empty((len(susceptibility_intercept), len(eligible)))
    for d in range(len(susceptibility_intercept)):
        logit = (
            age_logits[d, age_index][None, :] + beta[d, 0] * product[:, None] + beta[d, 1] * season
        )
        log_survive = -np.logaddexp(0.0, logit)  # log(1 - hazard)
        log_s_quiet = np.where(quiet, log_survive, 0.0).sum(1)
        log_s_ahead = np.where(quiet, 0.0, log_survive).sum(1)
        still_susceptible = 1 / (
            1
            + np.exp(
                -(susceptibility_intercept[d] + susceptibility_beta[d] * product + log_s_quiet)
            )
        )
        p_future[d] = still_susceptible * (1 - np.exp(log_s_ahead))

    sold = truth[truth.sale_date <= pd.Timestamp(str(as_of))].copy()
    sold["week"] = pd.to_datetime(sold.sale_date).dt.to_period("W-SUN").dt.start_time
    sold["observed"] = sold.initiation_date <= pd.Timestamp(str(as_of))
    sold["eventual"] = sold.initiation_date.notna()
    grouped = sold.groupby("week")
    weeks = grouped.size().index

    unit_week = sold.set_index("item_id").week
    future_by_week = (
        pd.DataFrame(p_future.T, index=unit_week.loc[eligible.item_id].values)
        .groupby(level=0)
        .sum()
        .reindex(weeks, fill_value=0.0)
    )
    model_rate = (
        grouped["observed"].sum().to_numpy()[:, None] + future_by_week.to_numpy()
    ) / grouped.size().to_numpy()[:, None]

    rates = pd.DataFrame(
        {
            "week": weeks,
            "units": grouped.size().to_numpy(),
            "observed": grouped["observed"].mean().to_numpy(),
            "eventual": grouped["eventual"].mean().to_numpy(),
            "model": model_rate.mean(1),
            "model_lo": np.quantile(model_rate, 0.05, axis=1),
            "model_hi": np.quantile(model_rate, 0.95, axis=1),
            "window_opens": pd.Timestamp(str(as_of - np.timedelta64(POLICY_DAYS - 1, "D"))),
        }
    )
    # The partial first and last calendar weeks hold only a few days of sales.
    return rates[rates.units >= min_units].reset_index(drop=True)


# --- Held-out outcomes ----------------------------------------------------------------


def held_out_counts(truth, dates):
    """Actual returns initiated and received on each forecast date."""
    days = pd.DatetimeIndex(dates)

    def per_day(column):
        return truth.groupby(column).size().reindex(days, fill_value=0).values

    return pd.DataFrame(
        {
            "date": days,
            "initiations": per_day("initiation_date"),
            "receipts": per_day("receipt_date"),
        }
    )


def actual_sales(truth, dates):
    """Units actually sold on each forecast date."""
    return (
        truth.groupby("sale_date").size().reindex(pd.DatetimeIndex(dates), fill_value=0).to_numpy()
    )


def eventual_counts(truth, as_of):
    """How many already-sold units eventually arrive, split by where each stands at ``as_of``."""
    cutoff = pd.Timestamp(str(as_of))
    past = truth[truth.sale_date <= cutoff]
    later = past.receipt_date > cutoff
    return {
        "open": int((later & (past.initiation_date <= cutoff)).sum()),
        "uninitiated": int((later & (past.initiation_date > cutoff)).sum()),
    }


def post_storm_check(forecast, held_out, spans):
    """The first open day after the storm, where the drained backlog is expected to land."""
    dates = pd.DatetimeIndex(forecast.dates)
    after = spans[spans.end > dates.min()].iloc[0].end
    while after.dayofweek >= 5:  # the warehouse is closed on weekends
        after += pd.Timedelta(days=1)
    i = int(dates.get_loc(after))
    lo, hi = np.quantile(forecast.receipts[:, i], [0.05, 0.95])
    return {
        "date": after,
        "actual": int(held_out.receipts.to_numpy()[i]),
        "mean": float(forecast.receipts[:, i].mean()),
        "lo": float(lo),
        "hi": float(hi),
    }


# --- The fitted model against the simulator's truth ----------------------------------


def hazard_curve(fit, true_hazard, stage, max_age=30):
    """Daily hazard of a fitted stage at every posterior draw: a neutral day, product A.

    The stage replays its own timing law, the random walk or the Weibull, so the curve is
    whatever that stage actually implies.
    """
    named = fit.parameters._asdict() if fit.family is None else dict(fit.parameters)
    ages = np.arange(0, max_age + 1)
    inputs = TimingInputs(
        ages=jnp.asarray(ages)[:, None],
        features=jnp.zeros((len(ages), 1, named["beta"].shape[-1])),
        susceptibility_features=jnp.zeros(
            (
                1,
                named["susceptibility_beta"].shape[-1] if "susceptibility_beta" in named else 0,
            )
        ),
    )
    log_hazard = jax.vmap(lambda draw: fit.timing(inputs, draw=draw).timing.log_hazard)(
        jnp.arange(fit.draws)
    )
    hazard = np.exp(np.asarray(log_hazard))[:, :, 0]  # [draw, age]
    return pd.DataFrame(
        {
            "age": ages,
            "mean": hazard.mean(0),
            "lo": np.quantile(hazard, 0.05, axis=0),
            "hi": np.quantile(hazard, 0.95, axis=0),
            "truth": true_hazard(ages),
            "stage": stage,
        }
    )


def hazard_curves(fitted):
    """Fitted initiation and receipt hazards beside the simulator's."""
    return (
        hazard_curve(fitted.initiation_fit, true_initiation_hazard, "Initiation: days since sale"),
        hazard_curve(fitted.receipt_fit, true_receipt_hazard, "Receipt: days since initiation"),
    )


def recovery_table(fitted, data, volume):
    """Every fitted parameter beside its true value, with 90% posterior intervals.

    ``ratio`` columns divide by the truth so every row lands on one axis. Every truth here is
    nonzero; dividing by a negative truth flips the interval's ends, hence the min/max.
    """
    initiation, receipt = fitted.initiation_fit.parameters, fitted.receipt_fit.parameters
    posterior = fitted.network.posterior
    order = list(data.groups["product"].to_numpy(int))  # series order is explicit
    a, b = order.index(0), order.index(1)
    susceptibility = np.asarray(initiation.susceptibility_intercept)
    susceptibility_b = np.asarray(initiation.susceptibility_beta)[:, 0]
    beta = np.asarray(initiation.beta)
    rate = np.exp(np.asarray(posterior["sales/log_rate"]))
    weekday = np.asarray(posterior["sales/weekday_effect"])

    rows = [
        ("Sales", "Daily sales rate, product A", rate[:, a], volume * SALES_RATES[0]),
        ("Sales", "Daily sales rate, product B", rate[:, b], volume * SALES_RATES[1]),
        ("Sales", "Weekday effect, product A", weekday[:, a], WEEKDAY_SWING[0]),
        ("Sales", "Weekday effect, product B", weekday[:, b], WEEKDAY_SWING[1]),
        (
            "Sales",
            "Promotion lift (log-multiplier of the daily rate)",
            np.asarray(posterior["sales/promo_lift"]),
            PROMO_LIFT,
        ),
        (
            "Return initiation",
            "Susceptible, product A",
            sigmoid(susceptibility),
            sigmoid(SUSCEPTIBLE),
        ),
        (
            "Return initiation",
            "Susceptible, product B",
            sigmoid(susceptibility + susceptibility_b),
            sigmoid(SUSCEPTIBLE + SUSCEPTIBLE_PRODUCT),
        ),
        ("Return initiation", "Hazard shift, product B", beta[:, 0], HAZARD_PRODUCT),
        ("Return initiation", "Hazard shift, weekday", beta[:, 1], HAZARD_WEEKDAY),
        (
            "Return receipt",
            "Initiated return ever arrives",
            sigmoid(np.asarray(receipt["susceptibility_intercept"])),
            RECEIPT_RECEIVABLE,
        ),
        ("Return receipt", "Weibull scale (days)", np.asarray(receipt["scale"]), RECEIPT_SCALE),
        ("Return receipt", "Weibull shape", np.asarray(receipt["shape"]), RECEIPT_SHAPE),
        (
            "Return receipt",
            "Storm effect (log-multiplier of each day's hazard increment)",
            np.asarray(receipt["beta"])[:, 0],
            RECEIPT_STORM,
        ),
    ]
    table = pd.DataFrame(
        [
            {
                "stage": stage,
                "Parameter": name,
                "estimate": float(np.mean(draws)),
                "lower": float(np.quantile(draws, 0.05)),
                "upper": float(np.quantile(draws, 0.95)),
                "truth": truth_value,
            }
            for stage, name, draws, truth_value in rows
        ]
    )
    lo, hi = table.lower / table.truth, table.upper / table.truth
    return table.assign(
        ratio=table.estimate / table.truth,
        ratio_lower=np.minimum(lo, hi),
        ratio_upper=np.maximum(lo, hi),
    )


# --- A standard time-series baseline --------------------------------------------------

# Regressors are mean daily counts over windows of days before the forecast day: a coarse
# delay curve, one weight per window, each bounded at zero (an upstream event can add
# downstream events, never remove them). Receipts get sales from a week back (no return
# starts, ships and arrives faster) and initiations from a day back; the initiation model,
# which supplies future initiations, gets sales from a day back.
SARIMAX_SALES_LAGS = ((7, 13), (14, 20), (21, 27), (28, 41), (42, 69))
SARIMAX_INITIATION_LAGS = ((1, 3), (4, 6), (7, 9), (10, 13), (14, 20), (21, 30))
INITIATION_SALES_LAGS = ((1, 3), (4, 7), (8, 14), (15, 28), (29, 56))


def _lag_windows(series, index, lags, prefix):
    """Mean of ``series`` over each ``(lo, hi)`` window of days before each day in ``index``."""
    return pd.DataFrame(
        {
            f"{prefix}_{lo}_{hi}": (
                sum(series.shift(k, freq="D") for k in range(lo, hi + 1)) / (hi - lo + 1)
            ).reindex(index)
            for lo, hi in lags
        },
        index=index,
    )


def _bounded_sarimax(y, train, future):
    """Fit SARIMAX(1,0,0)(1,0,0)7 with lag weights bounded at zero; forecast ``future``.

    Trains on every day with all regressors defined. Returns the fit and the forecast mean
    and standard deviation.
    """
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    usable = train.notna().all(axis=1).to_numpy()
    model = SARIMAX(
        y[usable].asfreq("D"),
        exog=train[usable].asfreq("D"),
        order=(1, 0, 0),
        seasonal_order=(1, 0, 0, 7),
        trend="c",
    )
    # Regression coefficients are not transformed, so these bounds are on the weights.
    lagged = ("sales_", "initiations_")
    bounds = [(0, None) if n.startswith(lagged) else (None, None) for n in model.param_names]
    with warnings.catch_warnings():  # non-convergence is reported to the caller instead
        warnings.simplefilter("ignore", ConvergenceWarning)
        fit = model.fit(disp=False, method="lbfgs", maxiter=1000, bounds=bounds)
    predicted = fit.get_forecast(len(future), exog=future)
    return fit, predicted.predicted_mean.to_numpy(), np.sqrt(predicted.var_pred_mean.to_numpy())


def sarimax_baseline(
    daily,
    future_sales,
    dates,
    storm_spans,
    sales_lags=SARIMAX_SALES_LAGS,
    initiation_lags=SARIMAX_INITIATION_LAGS,
    draws=1000,
    seed=0,
):
    """A standard forecaster: SARIMAX(1,0,0)(1,0,0)7 on daily receipts, with lagged sales,
    lagged return initiations, weekend and Monday dummies and the storm days as regressors.

    Initiations after today are not yet known, so a second SARIMAX forecasts them from
    lagged sales and its mean fills those days in. Lag weights are bounded at zero:
    unbounded, an old sales window and the intercept trade off to fit the slow trend, and a
    negative weight wipes out the forecast once a past sale enters that window.

    ``daily`` is the ledger's daily counts (``date``, ``Sales``, ``Returns initiated``,
    ``Returns received``); ``future_sales`` one expected sales value per forecast date in
    ``dates``. Returns the daily mean as a series; ``attrs["draws"]`` holds ``[draws, days]``
    samples of its Gaussian forecast, clipped at zero with closed weekends set to zero, so it
    can be scored as a distribution, and ``attrs["converged"]`` whether both fits converged.
    """
    history = daily.set_index(pd.DatetimeIndex(daily.date))
    days = pd.DatetimeIndex(dates)
    sales = pd.concat(
        [
            history["Sales"].astype(float),
            pd.Series(np.asarray(future_sales, dtype=float), index=days),
        ]
    )
    initiated = history["Returns initiated"].astype(float)
    initiation_fit, future_initiations, _ = _bounded_sarimax(
        initiated,
        _lag_windows(sales, initiated.index, INITIATION_SALES_LAGS, "sales"),
        _lag_windows(sales, days, INITIATION_SALES_LAGS, "sales"),
    )
    initiations = pd.concat(
        [initiated, pd.Series(np.clip(future_initiations, 0, None), index=days)]
    )

    def regressors(index):
        frame = pd.concat(
            [
                _lag_windows(sales, index, sales_lags, "sales"),
                _lag_windows(initiations, index, initiation_lags, "initiations"),
            ],
            axis=1,
        )
        frame["closed"] = (index.dayofweek >= 5).astype(float)
        frame["monday"] = (index.dayofweek == 0).astype(float)
        storm = np.zeros(len(index), dtype=bool)
        for span in storm_spans.itertuples():
            storm |= np.asarray((index >= span.start) & (index < span.end))
        frame["storm"] = storm.astype(float)
        return frame

    receipts = history["Returns received"].astype(float)
    receipt_fit, mean, sd = _bounded_sarimax(receipts, regressors(receipts.index), regressors(days))
    samples = np.random.default_rng(seed).normal(mean, sd, size=(draws, len(days)))
    samples = np.clip(samples, 0, None)
    samples[:, days.dayofweek >= 5] = 0.0
    series = pd.Series(samples.mean(axis=0), index=days, name="sarimax")
    series.attrs.update(
        draws=samples,
        converged=bool(
            initiation_fit.mle_retvals["converged"] and receipt_fit.mle_retvals["converged"]
        ),
    )
    return series


# --- The weekday-average baseline -----------------------------------------------------


def weekday_average(daily, held_out, weeks=8):
    """The usual shortcut: each forecast day's receipts are that weekday's recent average."""
    recent = daily.tail(7 * weeks).assign(weekday=lambda d: d.date.dt.dayofweek)
    by_weekday = recent.groupby("weekday")["Returns received"].mean()
    return pd.Series(
        by_weekday.reindex(held_out.date.dt.dayofweek).to_numpy(),
        index=held_out.date,
        name="weekday_avg",
    )


def lagged_rate(units, as_of, future_sales, dates, mature_days=60, cohort_days=56, lag_step=1):
    """The other usual shortcut: receipts are a return rate times sales, shifted by one lag.

    Both numbers come from cohorts old enough to have nearly finished returning (sold
    ``mature_days`` to ``mature_days + cohort_days`` days before ``as_of``), so the rate is
    not dragged down by recent units. The lag is their median sale-to-receipt time, rounded
    to the nearest multiple of ``lag_step`` days (7 for whole weeks, as a planner would).
    Sales before ``as_of`` are the ledger's; later sales are ``future_sales``, one value per
    forecast date. Receipts that would land on a closed weekend move to the next Monday.
    ``attrs`` keep the rate and lag the shortcut used.
    """
    as_of = np.datetime64(as_of, "D")
    sold = units.sale_date.to_numpy().astype("datetime64[D]")
    received = units.receipt_date.to_numpy().astype("datetime64[D]")
    age = (as_of - sold).astype(int)
    cohort = (age >= mature_days) & (age < mature_days + cohort_days)
    seen = cohort & (received <= as_of)  # NaT compares False: unreceived units stay out
    rate = seen.sum() / cohort.sum()
    median = np.median((received[seen] - sold[seen]).astype(int))
    lag = int(lag_step * round(median / lag_step))

    days = pd.DatetimeIndex(dates)
    past = units.groupby("sale_date").size()
    sales = pd.concat([past, pd.Series(np.asarray(future_sales, dtype=float), index=days)])
    point = rate * sales.reindex(days - pd.Timedelta(days=lag), fill_value=0).to_numpy()
    weekend = days.dayofweek >= 5
    for i in np.flatnonzero(weekend):
        monday = i + (7 - days[i].dayofweek)
        if monday < len(days):
            point[monday] += point[i]
    point[weekend] = 0.0
    series = pd.Series(point, index=days, name="lagged_rate")
    series.attrs.update(rate=float(rate), lag=lag)
    return series


def baseline_scores(forecast, held_out, baselines, model_label):
    """Daily error, daily CRPS and 28-day total for each method.

    ``baselines`` maps a label to a forecast. A plain point forecast has no interval, so its
    CRPS is its absolute error and its total interval is missing; a forecast series carrying
    ``attrs["draws"]`` (``[draws, days]``) is scored as a distribution.
    """
    actual = held_out.receipts.to_numpy()
    rows = []
    for label, baseline in baselines.items():
        point = np.asarray(baseline, dtype=float)
        draws = getattr(baseline, "attrs", {}).get("draws")
        totals = None if draws is None else np.asarray(draws).sum(axis=1)
        rows.append(
            {
                "Method": label,
                "mae": float(np.abs(point - actual).mean()),
                "crps": float(np.abs(point - actual).mean())
                if draws is None
                else crps(draws, actual),
                "total": float(point.sum()),
                "total_lo": np.nan if totals is None else float(np.quantile(totals, 0.05)),
                "total_hi": np.nan if totals is None else float(np.quantile(totals, 0.95)),
            }
        )
    model_mean = forecast.receipts.mean(axis=0)
    model_total = forecast.receipts.sum(axis=1)  # one 28-day total per posterior draw
    rows.append(
        {
            "Method": model_label,
            "mae": float(np.abs(model_mean - actual).mean()),
            "crps": crps(forecast.receipts, actual),
            "total": float(model_total.mean()),
            "total_lo": float(np.quantile(model_total, 0.05)),
            "total_hi": float(np.quantile(model_total, 0.95)),
        }
    )
    return pd.DataFrame(rows)


# --- Consistency checks on a propagated forecast --------------------------------------
# Each returns (label, passed, detail); the essay shows them as a checklist.


def check_counted_once(units, forecast):
    open_now = int((units.initiation_date.notna() & units.receipt_date.isna()).sum())
    return (
        "Every return is counted exactly once",
        np.array_equal(
            forecast.open_returns + forecast.receipts.cumsum(axis=1),
            open_now + forecast.initiations.cumsum(axis=1),
        ),
        "open + cumulative receipts = open today + cumulative initiations, per draw",
    )


def check_closed_days(forecast):
    closed = (np.asarray(forecast.dates, "datetime64[D]").astype(int) + 3) % 7 >= 5  # Sat, Sun
    return (
        "No receipts on a closed warehouse day",
        bool((forecast.receipts[:, closed] == 0).all()),
        "Saturday and Sunday receipts are exactly zero in every draw",
    )


def check_sales_propagated(forecast, sales):
    return (
        "The stages saw exactly the forecaster's sales",
        np.array_equal(forecast.sales.counts, sales.counts),
        "the sales that were propagated equal the drifting model's draws",
    )
