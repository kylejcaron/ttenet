"""JSON for the article's "Sales -> Return Initiation -> Receipt -> Forecast" chain explainer.

Everything here is read off the fitted model, never hand-built:

* The two timing curves replay each fitted stage's own timing law -- the random-walk
  baseline for initiation, the discretized Weibull for receipt -- at every posterior draw,
  through the same ``ttenet.event_times.survival_kernel`` the forecast uses.
  ``kernel.log_mass`` is the unconditional per-unit probability that the event lands on each
  day; ``kernel.log_tail`` is the probability of no event on any plotted day. Bars are
  therefore shares of ALL units entering the stage (all purchases, all initiated returns),
  not hazards and not conditioned on susceptibility.
* The curves use one hypothetical covariate setting: product A, a neutral weekday term, clear
  weather, and every calendar day open. It teaches the shape of each clock; it is not a
  calendar forecast and not conditioned on a unit being susceptible.
* The daily sales and receipt panels summarize the forecast's own predictive draws.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ttenet.event_times import TimingInputs, survival_kernel

# Pulled into the plot: initiation shows 15 days past the policy so the cutoff is visible.
_INITIATION_MARGIN_DAYS = 15
_MASS_TOLERANCE = 1e-6


def _stage_curve(fit, ages, exposure):
    """Per-posterior-draw native timing masses of ``fit`` at one hypothetical covariate setting.

    ``ages`` is ``[A]``; ``exposure`` is a boolean ``[A]`` of ages that can still fire. Returns
    arrays with a leading draw axis: ``mass [D, A]``, ``rest [D]`` (units that never fire, the
    non-susceptible share), ``survivors [D]`` (susceptible units still waiting after the last
    age) and ``tail [D]`` (no event on any age: ``rest + survivors``).
    """
    n_features, n_susceptibility = fit.feature_widths
    ages = jnp.asarray(ages)[:, None]
    exposure = jnp.asarray(exposure)[:, None]
    inputs = TimingInputs(
        ages=ages,
        features=jnp.zeros((ages.shape[0], 1, n_features)),
        susceptibility_features=jnp.zeros((1, n_susceptibility)),
    )

    def one_draw(draw):
        law = fit.timing(inputs, draw=draw)
        kernel = survival_kernel(
            law.timing, law.susceptibility_logits, allowed=True, exposure=exposure
        )
        survivors = kernel.log_susceptible + jnp.sum(kernel.log_survival_step, axis=-2)
        return (
            jnp.exp(kernel.log_mass[:, 0]),
            jnp.exp(kernel.log_rest[0]),
            jnp.exp(survivors[0]),
            jnp.exp(kernel.log_tail[0]),
        )

    with jax.enable_x64(True):
        mass, rest, survivors, tail = jax.vmap(one_draw)(jnp.arange(fit.draws))
        mass, rest, survivors, tail = (
            np.asarray(x, dtype=float) for x in (mass, rest, survivors, tail)
        )
    if not (np.all(np.isfinite(mass)) and np.all(mass >= 0)):
        raise ValueError("a fitted timing law produced non-finite or negative daily masses")
    drift = np.abs(mass.sum(axis=1) + tail - 1)
    if np.any(drift > _MASS_TOLERANCE):
        raise ValueError(f"timing masses and tail do not sum to one (max drift {drift.max():.2e})")
    return mass, rest, survivors, tail


def _band(draws, axis=0):
    """Posterior mean and 90% interval along ``axis``, as JSON lists (scalars if reduced fully)."""
    draws = np.asarray(draws, dtype=float)
    mean = draws.mean(axis=axis)
    lo, hi = np.quantile(draws, [0.05, 0.95], axis=axis)
    return {"mean": mean.tolist(), "lo90": lo.tolist(), "hi90": hi.tolist()}


def _summary(draws):
    """Mean and 50%/90% predictive intervals of a ``[draw, day]`` array."""
    draws = np.asarray(draws, dtype=float)
    lo90, lo50, hi50, hi90 = np.quantile(draws, [0.05, 0.25, 0.75, 0.95], axis=0)
    return {
        "mean": draws.mean(axis=0).tolist(),
        "lo90": lo90.tolist(),
        "hi90": hi90.tolist(),
        "lo50": lo50.tolist(),
        "hi50": hi50.tolist(),
    }


def _total(draws):
    """Mean and 90% interval of the per-draw horizon sum (not a sum of daily quantiles)."""
    return {k: float(v) for k, v in _band(np.asarray(draws, dtype=float).sum(axis=1)).items()}


def chain_data(fitted, forecast, sales_draws, storm=None, receipt_days=30):
    """Plain-JSON payload for the chain explainer, from a fitted model and its forecast.

    ``fitted`` is the ``FittedRetailReturnModel``; ``forecast`` its ``ReturnForecast``;
    ``sales_draws`` the ``[draw, day]`` daily sales totals feeding it. ``storm`` is an optional
    boolean mask over ``forecast.dates`` of the supplied weather input (``None`` when there is
    none to shade). ``receipt_days`` is the last elapsed day plotted for receipts; receipt has
    no deadline, so what lands later is reported as ``beyond``.

    Schema (probabilities are fractions in [0, 1]; the page formats percent)::

        as_of, datesISO[H], weekend[H], storm[H] | None
        sales, receipt_forecast: {mean, lo90, hi90, lo50, hi50}[H]   # predictive, counts per day
        totals: {sales, receipt_forecast: {mean, lo90, hi90}}        # per-draw horizon sums
        draws: {forecast, initiation, receipt}
        initiation | receipt: {
          setting: str,                       # the hypothetical covariate setting
          denominator: str,                   # what the bars are a share of
          family: str,                        # "age_baseline" (default family) or the family class
          ages[A], prob: {mean, lo90, hi90}[A],
          total: {mean, lo90, hi90},          # shares that land on a plotted day
          never: {mean, lo90, hi90},          # non-susceptible: never start / never arrive
          beyond: {mean, lo90, hi90},         # susceptible, still waiting after the last age
          (initiation) policy_days, (receipt) plot_days
        }
    """
    receipts = np.asarray(forecast.receipts, dtype=float)
    sales = np.asarray(sales_draws, dtype=float)
    dates = np.asarray(forecast.dates).astype("datetime64[D]")
    horizon = len(dates)
    if receipts.ndim != 2 or receipts.shape[1] != horizon:
        raise ValueError("forecast.receipts must be [draw, day] over forecast.dates")
    if sales.ndim != 2 or sales.shape[1] != horizon:
        raise ValueError("sales_draws must be [draw, day] over forecast.dates")
    if storm is not None:
        storm = np.asarray(storm, dtype=bool)
        if storm.shape != (horizon,):
            raise ValueError("storm must be a boolean mask over forecast.dates")
    if not isinstance(receipt_days, (int, np.integer)) or receipt_days < 1:
        raise ValueError("receipt_days must be a positive integer")

    policy_days = fitted.history.policy_days
    if policy_days is None:
        init_ages = np.arange(0, 106)
        init_exposure = np.ones(len(init_ages), dtype=bool)
    else:
        init_ages = np.arange(0, policy_days + _INITIATION_MARGIN_DAYS + 1)
        init_exposure = init_ages <= policy_days
    init_mass, init_rest, init_survivors, _ = _stage_curve(
        fitted.initiation_fit, init_ages, init_exposure
    )

    recv_ages = np.arange(0, int(receipt_days) + 1)
    recv_mass, recv_rest, recv_survivors, _ = _stage_curve(
        fitted.receipt_fit, recv_ages, np.ones(len(recv_ages), dtype=bool)
    )

    return {
        "as_of": str(dates[0] - np.timedelta64(1, "D")),
        "datesISO": [str(d) for d in dates],
        "weekend": (_weekday(dates) >= 5).tolist(),
        "storm": None if storm is None else storm.tolist(),
        "sales": _summary(sales),
        "receipt_forecast": _summary(receipts),
        "totals": {"sales": _total(sales), "receipt_forecast": _total(receipts)},
        "draws": {
            "forecast": int(receipts.shape[0]),
            "initiation": int(fitted.initiation_fit.draws),
            "receipt": int(fitted.receipt_fit.draws),
        },
        "initiation": {
            "setting": (
                "Hypothetical setting: product A, neutral weekday term, every day open. "
                "Not conditioned on a unit being susceptible and not a calendar forecast."
            ),
            "denominator": "all purchases",
            "family": _family(fitted.initiation_fit),
            "ages": init_ages.tolist(),
            "policy_days": None if policy_days is None else int(policy_days),
            "prob": _band(init_mass),
            "total": _band(init_mass.sum(axis=1)),
            "never": _band(init_rest),
            "beyond": _band(init_survivors),
        },
        "receipt": {
            "setting": (
                "Hypothetical setting: clear weather, every day open (no weekend closure). "
                "Not conditioned on a return being receivable and not a calendar forecast."
            ),
            "denominator": "all initiated returns",
            "family": _family(fitted.receipt_fit),
            "ages": recv_ages.tolist(),
            "plot_days": int(receipt_days),
            "prob": _band(recv_mass),
            "total": _band(recv_mass.sum(axis=1)),
            "never": _band(recv_rest),
            "beyond": _band(recv_survivors),
        },
    }


def _family(fit):
    return "age_baseline" if fit.family is None else type(fit.family).__name__


def _weekday(dates):
    """Monday=0 .. Sunday=6 of ``datetime64[D]`` dates (1970-01-01 was a Thursday)."""
    return (dates.astype("datetime64[D]").astype(np.int64) + 3) % 7
