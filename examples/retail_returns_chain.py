"""The survival convolution behind the TL;DR's "Follow one day's sales forward" figures.

Everything here is read off the fitted model, never hand-built: each stage's timing curve
replays that stage's own timing law (the random-walk baseline for initiation, the discretized
Weibull for receipt) at every posterior draw, through the same
``ttenet.event_times.survival_kernel`` the forecast uses. ``kernel.log_mass`` is the
unconditional per-unit probability that the event lands on each day, so the curves are
shares of all units entering the stage, not hazards. They use one hypothetical covariate
setting: product A, a neutral weekday term and clear weather.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ttenet.event_times import TimingInputs, survival_kernel

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


def _weekday(dates):
    """Monday=0 .. Sunday=6 of ``datetime64[D]`` dates (1970-01-01 was a Thursday)."""
    return (dates.astype("datetime64[D]").astype(np.int64) + 3) % 7


# --- "Follow one day's sales forward": the survival convolution, one sale day at a time ---


def convolution_kernels(fitted, receipt_days=60):
    """Posterior-mean daily timing masses of both fitted stages, for the TL;DR figure.

    Same hypothetical setting as ``chain_data`` (product A, neutral weekday term, clear
    weather), except that receipt respects the warehouse's weekday opening: one receipt
    curve per weekday a return can start on, since closed days carry no chance of arrival.
    Returns ``{"initiation": [A0], "receipt": [7, A1]}`` with Monday=0 rows.
    """
    policy_days = fitted.history.policy_days
    init_ages = np.arange(0, (policy_days if policy_days is not None else 105) + 1)
    init_mass, *_ = _stage_curve(
        fitted.initiation_fit, init_ages, np.ones(len(init_ages), dtype=bool)
    )
    recv_ages = np.arange(0, int(receipt_days) + 1)
    receipt = np.stack(
        [
            _stage_curve(fitted.receipt_fit, recv_ages, (start + recv_ages) % 7 < 5)[0].mean(0)
            for start in range(7)
        ]
    )
    return {"initiation": init_mass.mean(0), "receipt": receipt}


def follow_sales(kernels, sales, first_date, days):
    """Expected returns started and received from each day's sales, per calendar day.

    ``sales`` is ``[S]`` expected units sold on ``S`` consecutive days from ``first_date``;
    the result covers ``days`` calendar days from ``first_date``. Returns ``started`` and
    ``received`` arrays of shape ``[S, days]``: row ``s`` is what sale day ``s`` contributes.
    Started returns spread through the initiation curve; each day's started returns spread
    again through the receipt curve for that weekday. That double spread, summed over sale
    days, is the survival convolution.
    """
    sales = np.asarray(sales, dtype=float)
    first = np.datetime64(first_date, "D")
    init, recv = kernels["initiation"], kernels["receipt"]
    started = np.zeros((len(sales), days))
    received = np.zeros((len(sales), days))
    for s, units in enumerate(sales):
        span = min(len(init), days - s)
        started[s, s : s + span] = units * init[:span]
        for u in range(s, days):
            if started[s, u] == 0:
                continue
            weekday = int(_weekday(np.array([first + np.timedelta64(u, "D")]))[0])
            reach = min(recv.shape[1], days - u)
            received[s, u : u + reach] += started[s, u] * recv[weekday, :reach]
    return {"started": started, "received": received}
