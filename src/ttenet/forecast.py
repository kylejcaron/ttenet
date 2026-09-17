"""Calendar-day, count-conserving event forecasts built on one generic kernel.

``forecast_events`` propagates a single parent -> child event (e.g. sale ->
initiation, or initiation -> receipt) forward from ``as_of`` through a
calendar-day horizon, given a stage's fitted mixture-cure parameters, known
historical origin/own-event dates, and an exact realized-arrivals array for
new parent events during the horizon. ``forecast_returns`` composes two
calls to this kernel (initiation, then receipt, the second consuming the
first's ``events`` output as its ``arrivals``) to reproduce the full retail
return process, and adds the exact analytic eventual-receipt expectations
for the existing historical population via ``expected_return_receipts``.

Every cohort (one historical row, or one future dated sales cohort) is
tracked as a conserved integer count, never expanded per unit: a historical
row simply starts with count 1. Population mass is exact within every
posterior/count draw and no unit is ever materialized or destroyed.

Host float64 hazard/cure arithmetic
------------------------------------
This module is host-side NumPy simulation, not JAX-traced code: parameter
draws are ordinary float64 NumPy arrays, and every day's hazard/susceptibility
value is computed directly from the *documented* linear equations behind
``survival.stage_hazard``/``survival.susceptibility``
(``age_logits[min(age, K-1)] + x(t) @ beta`` and
``cure_intercept + z @ cure_beta``) rather than by calling those JAX
functions and re-deriving a logit from their sigmoid output. Two reasons:

* Day-by-day cumulative log-survival tracking over long horizons needs
  ``log(1 - hazard)``. Forming that from an already-sigmoid-transformed
  hazard loses precision as the hazard approaches 0 or 1 (float32 rounds
  ``sigmoid(logit)`` to exactly 1 once ``logit`` exceeds ~16.7, turning
  ``log(1 - hazard)`` into ``-inf``). Computed instead as
  ``-log(1 + exp(hazard_logit)) = -logaddexp(0, hazard_logit)`` directly from
  the pre-sigmoid logit, this stays finite and accurate for any realistic
  finite logit in float64.
* ``conditional_susceptibility``'s ``pi * S / ((1 - pi) + pi * S)`` is
  exactly ``sigmoid(cure_logit + log_survival)`` in logit space (dividing
  numerator/denominator by ``1 - pi`` turns the ratio into ``sigmoid`` of the
  sum of logits). Evaluating the combined logit directly avoids ever forming
  ``pi`` and ``1 - pi`` separately, so it cannot hit the indeterminate
  ``0 / 0`` that a rounded ``pi == 1`` meets a ``log_survival == -inf``
  history with.

Both are the *same* mixture-cure model implemented by ``survival.py`` and
``models.py``; this module never redefines the model, it
only evaluates its equations without a lossy sigmoid/logit round trip. If any
row's as-of conditioning still produces a nonfinite susceptibility/log-
survival combination (e.g. a genuinely impossible supplied history), forecast
construction fails loudly for that row rather than silently treating it as
certain cure.

Generic event kernel mechanics
--------------------------------
``forecast_events`` packs cohorts into parent-date pools. A unit, or a count
cohort whose parents arrive on only one day per draw, needs one pool with a
draw-specific clock. Cohorts with arrivals on several days retain separate
pools for the actual arrival dates. Empty pools are not allocated.
Historical pools start with susceptibility conditioned on all observed
survival through ``as_of``; future pools start at their original susceptibility.
Future pools cannot produce events or contribute to pending stocks before
their parent day. An age-zero trial runs on that day, allowing same-day
transitions. Every pool retains its original root-cohort identity.

A unit's *marginal* (cure-integrated) hazard on a given day is the raw stage
hazard multiplied by the *current posterior probability of being
susceptible given survival so far*, recomputed every day from a running
cumulative log-survival state per pool. This is mathematically equivalent
to drawing each unit's latent cure status once and simulating only the
susceptible sub-population forward, without ever having to materialize or
track that latent draw. Once a pool's age exceeds a configured
``deadline_days``, its members stop accruing hazard (frozen survival, zero
event probability) but remain counted in ``pending`` forever -- ``eligible``
is the subset still within the deadline.

Receipt tail assumption
------------------------
Receipt age uses the final fitted baseline bin beyond the learned ages.
Eventual receipt estimates assume positive tail exposure continues and
receiving eventually reopens. For a susceptible unit, cumulative tail hazard
then diverges and eventual receipt probability is one.

The ``expected_*_receipts`` fields concern the historical population at the
forecast origin, not future sales. Their initiation component still depends
on the supplied calendar through the policy deadline (when one is
configured; an unbounded ``policy_days=None`` initiation reduces the
eventual-initiation probability to the posterior conditional susceptibility
itself, exactly like the already-unbounded receipt stage). Simulated
``receipts`` include future sales and process noise, so a realized path
need not be below the historical-population eventual expectation.
Finite-horizon expected receipts from the same historical population cannot
exceed that eventual expectation. No receipt deadline is invented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .dates import date_grid, elapsed_days, to_day
from .integration import SalesForecast, _validate_counts
from .survival import StageParameters


@dataclass(frozen=True)
class ReturnForecast:
    """Calendar-day forecast of return initiations, receipts, and outstanding counts.

    All ``[draw, ...]`` fields carry a leading posterior/count draw axis
    (size 1 when no posterior draws or future-count draws were supplied).
    ``initiations``, ``receipts``, ``eligible``, and ``open_returns`` are
    stochastic per-draw simulations over the supplied ``calendar``/horizon.
    ``expected_existing_receipts``, ``expected_uninitiated_receipts``, and
    ``expected_open_receipts`` are exact analytic expectations under the
    continuing-tail assumption documented on this module and are independent
    of the simulated draws. See the module docstring for the exact
    definition of the tail assumption. ``sales`` is the bundled
    :class:`~ttenet.integration.SalesForecast` source that produced the
    future cohorts, if any was supplied.
    """

    dates: Any  # [horizon] datetime64[D], as_of+1 .. as_of+horizon inclusive
    initiations: Any  # [draw, horizon] integer counts
    receipts: Any  # [draw, horizon] integer counts
    eligible: Any  # [draw, horizon] uninitiated and not past deadline, end-of-day
    open_returns: Any  # [draw, horizon] initiated, not received, end-of-day
    expected_existing_receipts: Any  # [draw] eventual receipts from past sales only
    expected_uninitiated_receipts: Any  # [draw] eventual-receipt component from pending initiations
    expected_open_receipts: Any  # [draw] eventual-receipt component from existing open returns
    sales: Any = None  # SalesForecast | None -- the future-cohort source, if supplied


@dataclass(frozen=True)
class EventForecast:
    """Calendar-day forecast of one generic parent -> child event propagation.

    ``events[D, horizon, C]`` retains the root-cohort axis through every
    descendant. ``pending[D, horizon]`` is the aggregate end-of-day count
    whose parent has occurred but whose own event has not, including expired
    or cured units. ``eligible[D, horizon]`` excludes units past the stage's
    deadline and equals ``pending`` for an unbounded stage.
    """

    events: Any
    pending: Any
    eligible: Any


# --------------------------------------------------------------------------
# Parameter normalization
# --------------------------------------------------------------------------


def _stack_draws(value, base_ndim, name):
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == base_ndim:
        return arr[None, ...]
    if arr.ndim == base_ndim + 1:
        return arr
    raise ValueError(
        f"{name} must have ndim {base_ndim} (single draw) or {base_ndim + 1} "
        f"(posterior draws); got ndim {arr.ndim}"
    )


def _normalize_stage_parameters(parameters, name):
    age_logits = _stack_draws(parameters.age_logits, 1, f"{name}.age_logits")
    beta = _stack_draws(parameters.beta, 1, f"{name}.beta")
    cure_intercept = _stack_draws(parameters.cure_intercept, 0, f"{name}.cure_intercept")
    cure_beta = _stack_draws(parameters.cure_beta, 1, f"{name}.cure_beta")
    draws = {age_logits.shape[0], beta.shape[0], cure_intercept.shape[0], cure_beta.shape[0]}
    draws.discard(1)
    if len(draws) > 1:
        raise ValueError(
            f"{name} parameter fields disagree on posterior draw count: {sorted(draws)}"
        )
    d = next(iter(draws)) if draws else 1
    return StageParameters(age_logits, beta, cure_intercept, cure_beta), d


def _broadcast_stage(params, d):
    return StageParameters(
        np.broadcast_to(params.age_logits, (d,) + params.age_logits.shape[1:]),
        np.broadcast_to(params.beta, (d,) + params.beta.shape[1:]),
        np.broadcast_to(params.cure_intercept, (d,) + params.cure_intercept.shape[1:]),
        np.broadcast_to(params.cure_beta, (d,) + params.cure_beta.shape[1:]),
    )


def _align_draws(*counts):
    unique = sorted({c for c in counts if c != 1})
    if len(unique) > 1:
        raise ValueError(
            f"Cannot align posterior parameter draws with arrival/count draws: sizes {unique} "
            "must each be 1 or share a single common value"
        )
    return unique[0] if unique else 1


# --------------------------------------------------------------------------
# Feature / allowed / count validation
# --------------------------------------------------------------------------


def _prepare_features(features, n_rows, calendar_len, width, name):
    if features is None:
        if width != 0:
            raise ValueError(
                f"{name} is required because the fitted model has feature width {width}"
            )
        return np.zeros((n_rows, calendar_len, 0), dtype=np.float64)
    arr = np.asarray(features, dtype=np.float64)
    expected_shape = (n_rows, calendar_len, width)
    if tuple(arr.shape) != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}; got {tuple(arr.shape)}")
    return arr


def _prepare_cure_features(features, n_rows, width, name):
    if features is None:
        if width != 0:
            raise ValueError(
                f"{name} is required because the fitted model has cure feature width {width}"
            )
        return np.zeros((n_rows, 0), dtype=np.float64)
    arr = np.asarray(features, dtype=np.float64)
    expected_shape = (n_rows, width)
    if tuple(arr.shape) != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}; got {tuple(arr.shape)}")
    return arr


def _prepare_allowed(allowed, n_rows, calendar_len, name):
    if allowed is None:
        return np.ones((n_rows, calendar_len), dtype=bool)
    arr = np.asarray(allowed, dtype=bool)
    if tuple(arr.shape) == (calendar_len,):
        return np.broadcast_to(arr[None, :], (n_rows, calendar_len))
    if tuple(arr.shape) == (n_rows, calendar_len):
        return arr
    raise ValueError(
        f"{name} must have shape ({calendar_len},) or ({n_rows}, {calendar_len}); got {tuple(arr.shape)}"
    )


def _prepare_sales_forecast(future_sales):
    """Accept a :class:`SalesForecast` directly, or parse a plain DataFrame."""
    if future_sales is None:
        return None
    if isinstance(future_sales, SalesForecast):
        return future_sales
    if isinstance(future_sales, pd.DataFrame):
        return SalesForecast.from_frame(future_sales)
    raise ValueError(
        "future_sales must be a SalesForecast or a DataFrame accepted by SalesForecast.from_frame"
    )


def _future_cohorts_from_sales(sales_forecast, history_item_ids, as_of, horizon):
    if sales_forecast is None:
        empty = pd.DataFrame(columns=["item_id", "sale_date"])
        return empty, np.zeros(0, dtype="datetime64[D]"), np.zeros((1, 0), dtype=np.int64)
    cohorts = sales_forecast.cohorts
    ids = cohorts["item_id"].to_numpy()
    overlap = set(ids) & set(history_item_ids)
    if overlap:
        raise ValueError(
            f"future_sales item_id overlaps historical item_id values: {sorted(map(str, overlap))}"
        )
    sale_dates = np.asarray(cohorts["sale_date"].to_numpy(), dtype="datetime64[D]")
    output_start = as_of + np.timedelta64(1, "D")
    output_end = as_of + np.timedelta64(horizon, "D")
    if np.any((sale_dates < output_start) | (sale_dates > output_end)):
        raise ValueError("future_sales sale_date must fall within the output horizon")
    counts_array = np.asarray(sales_forecast.counts, dtype=np.int64)
    return cohorts, sale_dates, counts_array


def _validate_calendar(calendar, as_of, horizon, earliest_origin, latest_deadline):
    cal = np.asarray(to_day(calendar)).reshape(-1)
    if cal.size == 0:
        raise ValueError("calendar must not be empty")
    if np.any(np.isnat(cal)):
        raise ValueError("calendar must not contain missing dates")
    order = np.argsort(cal, kind="stable")
    if not np.array_equal(order, np.arange(cal.size)):
        raise ValueError("calendar must be sorted ascending")
    expected = np.asarray(date_grid(cal[0], cal[-1]))
    if cal.size != expected.size or not np.array_equal(cal, expected):
        raise ValueError(
            "calendar must be an inclusive, contiguous daily grid (see dates.date_grid)"
        )
    output_end = as_of + np.timedelta64(horizon, "D")
    if cal[0] > earliest_origin:
        raise ValueError(
            "calendar must start on or before the earliest historical sale/initiation date "
            "that still needs coverage"
        )
    if cal[-1] < output_end:
        raise ValueError("calendar must extend through the output horizon (as_of + horizon)")
    if cal[-1] < latest_deadline:
        raise ValueError(
            "calendar must extend through the latest uninitiated historical initiation deadline"
        )
    return cal


def _elapsed(origin, target):
    return np.asarray(elapsed_days(origin, target)).astype(np.int64)


# --------------------------------------------------------------------------
# Host float64 hazard/cure logit primitives (see module docstring)
# --------------------------------------------------------------------------


def _sigmoid(x):
    """Numerically stable sigmoid for float64 NumPy arrays."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def _softplus(x):
    """``log(1 + exp(x))``, stable via ``logaddexp(0, x)``."""
    return np.logaddexp(0.0, x)


def _linear_logit(features, weights):
    """``features @ weights`` contracted over the trailing feature axis.

    ``features`` is ``[..., P]`` (no draw axis); ``weights`` is ``[D, P]``.
    Returns ``[D, ...]`` -- the same contraction as
    ``survival._linear_effect`` with the draw axis made explicit and leading.
    """
    return np.tensordot(
        np.asarray(weights, dtype=np.float64),
        np.asarray(features, dtype=np.float64),
        axes=([-1], [-1]),
    )


def _hazard_logit(params, ages, features, allowed):
    """Host float64 hazard logit, mirroring ``survival.stage_hazard`` exactly.

    ``ages`` is ``[D, ...]`` (callers always broadcast a leading draw axis
    onto ages, even where the age trajectory itself does not vary by draw).
    ``features``/``allowed`` are ``[..., P]``/``[...]`` and shared across
    draws. Returns ``-inf`` (so the corresponding hazard probability is
    exactly zero, never a rounded near-zero) wherever ``ages < 0`` or
    ``allowed`` is ``False``, exactly matching ``stage_hazard``'s masking.
    """
    ages = np.asarray(ages)
    num_bins = params.age_logits.shape[-1]
    age_index = np.clip(ages, 0, num_bins - 1).astype(np.int64)
    d = params.age_logits.shape[0]
    flat_index = age_index.reshape(d, -1)
    baseline = np.take_along_axis(params.age_logits, flat_index, axis=1).reshape(age_index.shape)
    logit = baseline + _linear_logit(features, params.beta)
    valid = ages >= 0
    if allowed is not None:
        valid = valid & np.asarray(allowed, dtype=bool)[None, ...]
    return np.where(valid, logit, -np.inf)


def _cure_logit(params, cure_features):
    """Host float64 cure logit, mirroring ``survival.susceptibility`` exactly.

    ``cure_features`` is ``[N, Q]`` (draw-independent). Returns ``[D, N]``.
    """
    return params.cure_intercept[:, None] + _linear_logit(cure_features, params.cure_beta)


def _require_finite(values, name):
    values = np.asarray(values)
    if not np.all(np.isfinite(values)):
        bad = np.argwhere(~np.isfinite(values))[:5].tolist()
        raise ValueError(
            f"{name} produced a nonfinite value at draw/row indices {bad} (possibly more) -- this is "
            "an impossible or degenerate conditional history (an infinite hazard/cure logit combined "
            "with a fully decayed survival), not a valid cure outcome, and is rejected rather than "
            "silently treated as certain cure"
        )
    return values


def _conditional_probability(cure_logit, log_survival, name):
    """``sigmoid(cure_logit + log_survival)``, i.e. ``conditional_susceptibility`` in logit space."""
    return _require_finite(_sigmoid(cure_logit + log_survival), name)


def _gather_window(full_arr, row_positions, cal_idx):
    """Gather ``full_arr[row_positions[i], cal_idx[i, k], ...]`` for every (i, k)."""
    selected = np.asarray(full_arr)[np.asarray(row_positions)]  # [N, T, ...]
    n = selected.shape[0]
    return selected[np.arange(n)[:, None], cal_idx]  # [N, W, ...]


def _window_log_survival(
    params,
    full_features,
    full_allowed,
    row_positions,
    origin_idx,
    start_age,
    end_age,
    max_width,
    calendar_len,
):
    """Cumulative ``sum(log(1 - hazard))`` over ages [start_age, end_age] inclusive, per row/draw.

    Computed as ``-softplus(hazard_logit)`` directly from the pre-sigmoid
    logit (see module docstring) rather than ``log1p(-sigmoid(logit))``.
    Each row ``i`` has its own calendar origin (``origin_idx[i]``); age ``a``
    for row ``i`` corresponds to calendar position ``origin_idx[i] + a``.
    Rows whose window is empty (``start_age[i] > end_age[i]``) contribute
    exactly 0. Returns ``[D, N]``.
    """
    n = row_positions.shape[0]
    d = params.age_logits.shape[0]
    if n == 0:
        return np.zeros((d, 0))
    ages_rel = start_age[:, None] + np.arange(max_width)[None, :]  # [N, W]
    ages = np.broadcast_to(ages_rel[None, :, :], (d, n, max_width))
    cal_idx = np.clip(origin_idx[:, None] + ages_rel, 0, calendar_len - 1)
    features = _gather_window(full_features, row_positions, cal_idx)  # [N, W, P]
    allowed = _gather_window(full_allowed, row_positions, cal_idx)  # [N, W]
    within = (
        (ages_rel <= end_age[:, None]) & (start_age[:, None] <= end_age[:, None]) & (ages_rel >= 0)
    )
    logit = _hazard_logit(params, ages, features, allowed)  # [D, N, W]
    log_survive = np.where(within[None, :, :], -_softplus(logit), 0.0)
    return np.sum(log_survive, axis=-1)


# --------------------------------------------------------------------------
# Generic event kernel
# --------------------------------------------------------------------------


def _arrival_pools(arrivals, initial_pending, age0):
    """Pack only realized parent-date pools; single-entry cohorts need one slot.

    A historical unit may enter a descendant on different days in different
    draws, but still needs only one clock per draw. Count cohorts with several
    parent dates retain separate pools, grouped for exact integer reductions.
    """
    draws, horizon, cohorts = arrivals.shape
    entries = np.broadcast_to(initial_pending, (draws, cohorts)).astype(np.int64)
    quantities = entries.copy()
    parent_days = np.broadcast_to(-age0, (draws, cohorts)).copy()
    possible = np.empty((horizon, cohorts), dtype=bool)
    for day in range(horizon):
        incoming = arrivals[:, day, :]
        present = incoming > 0
        entries += present
        quantities += incoming
        np.copyto(parent_days, day + 1, where=present)
        possible[day] = present.any(axis=0)

    maximum_entries = entries.max(axis=0, initial=0)
    single = np.flatnonzero(maximum_entries == 1)
    multiple = np.flatnonzero(maximum_entries > 1)
    occupied = np.column_stack((initial_pending[multiple], possible[:, multiple].T))
    groups, days = np.nonzero(occupied)
    pool_cohorts = np.concatenate((single, multiple[groups]))
    active_cohorts = np.concatenate((single, multiple))
    sizes = occupied.sum(axis=1)
    starts = np.concatenate((np.arange(single.size), single.size + np.cumsum(sizes) - sizes))
    initial_pool = np.full(cohorts, -1, dtype=np.int64)
    initial_pool[active_cohorts] = starts

    counts = np.empty((draws, pool_cohorts.size), dtype=np.int64)
    origins = np.empty_like(counts)
    counts[:, : single.size] = quantities[:, single]
    origins[:, : single.size] = parent_days[:, single]
    multi_counts = counts[:, single.size :]
    multi_counts[:] = arrivals[:, days - 1, multiple[groups]]
    multi_counts[:, days == 0] = 1
    origins[:, single.size :] = np.where(days == 0, -age0[multiple[groups]], days)
    return counts, origins, pool_cohorts, active_cohorts, starts, initial_pool


def forecast_events(
    parameters,
    *,
    origins,
    observed,
    arrivals,
    calendar,
    as_of,
    horizon,
    features=None,
    cure_features=None,
    allowed=None,
    deadline_days=None,
    seed=0,
) -> EventForecast:
    """Simulate one calendar-day parent -> child event propagation.

    ``origins``/``observed`` are ``[C]`` dates: each cohort's known
    historical immediate-parent event and its own event, ``NaT`` for
    unknown/future. A cohort with a known origin ``<= as_of`` and no known
    own event starts with exactly one pending unit (its actual age at
    ``as_of``); a cohort whose own event is already observed never produces
    it again. A cohort with unknown origin (``NaT``) starts with zero
    pending units -- it can only enter later through ``arrivals`` (e.g. a
    future sales cohort, whose "origin" is simply the day its arrival lands
    at age zero).

    ``arrivals`` is ``[D or 1, horizon, C]``: the exact realized count of
    *new* parent events for each cohort on each forecast day (e.g. a root
    sale cohort's future dated sale quantities, or -- for a child stage --
    the parent stage's own ``EventForecast.events`` output, used unchanged).
    Every arriving unit begins its own age-0 clock that day, so a parent
    event and its child's own trial may both happen the same calendar day.

    See the module docstring for the age-bucket propagation mechanics, the
    host float64 hazard/cure arithmetic rationale, and the exact
    ``pending``/``eligible`` semantics on :class:`EventForecast`.
    """
    if horizon < 1:
        raise ValueError("horizon must be a positive number of days")
    if deadline_days is not None:
        if not isinstance(deadline_days, (int, np.integer)) or isinstance(deadline_days, bool):
            raise ValueError("deadline_days must be an integer or None")
        if deadline_days < 0:
            raise ValueError("deadline_days must be nonnegative")

    params, d_params = _normalize_stage_parameters(parameters, "parameters")

    origins = np.asarray(to_day(origins)).reshape(-1)
    observed = np.asarray(to_day(observed)).reshape(-1)
    if origins.shape != observed.shape:
        raise ValueError("origins and observed must have the same shape")
    n_cohorts = int(origins.shape[0])

    origin_known = ~np.isnat(origins)
    event_known = ~np.isnat(observed)
    if np.any(event_known & ~origin_known):
        raise ValueError("an observed own event cannot occur without a known parent/origin date")
    if np.any(event_known & origin_known & (observed < origins)):
        raise ValueError("observed own event precedes its parent/origin date")
    if deadline_days is not None:
        deadline_date = origins + np.timedelta64(int(deadline_days), "D")
        if np.any(event_known & origin_known & (observed > deadline_date)):
            raise ValueError(f"observed own event exceeds the {deadline_days}-day deadline")

    as_of = np.datetime64(to_day(as_of), "D")

    arrivals_arr = np.asarray(arrivals)
    if arrivals_arr.ndim == 2:
        arrivals_arr = arrivals_arr[None, ...]
    if arrivals_arr.ndim != 3:
        raise ValueError("arrivals must have ndim 2 (single draw) or 3 (draws, horizon, cohorts)")
    if arrivals_arr.shape[1:] != (horizon, n_cohorts):
        raise ValueError(
            f"arrivals must have shape (draws, {horizon}, {n_cohorts}); got {arrivals_arr.shape}"
        )
    arrivals_arr = _validate_counts(arrivals_arr, "arrivals", copy=False)
    d_arrivals = arrivals_arr.shape[0]
    if d_arrivals == 0:
        raise ValueError("arrivals must contain at least one draw")

    d = _align_draws(d_params, d_arrivals)
    params = _broadcast_stage(params, d)
    arrivals_arr = np.broadcast_to(arrivals_arr, (d, horizon, n_cohorts))

    p = int(params.beta.shape[-1])
    q = int(params.cure_beta.shape[-1])

    earliest_origin = origins[origin_known].min() if np.any(origin_known) else as_of
    cal = _validate_calendar(calendar, as_of, horizon, earliest_origin, as_of)
    calendar_len = int(cal.size)
    asof_idx = int(_elapsed(cal[0], as_of))

    features_arr = _prepare_features(features, n_cohorts, calendar_len, p, "features")
    cure_arr = _prepare_cure_features(cure_features, n_cohorts, q, "cure_features")
    allowed_arr = _prepare_allowed(allowed, n_cohorts, calendar_len, "allowed")

    already_done = event_known & (observed <= as_of)
    initial_pending_mask = origin_known & (origins <= as_of) & ~already_done
    safe_origins = np.where(origin_known, origins, as_of)
    age0 = np.where(initial_pending_mask, _elapsed(safe_origins, as_of), 0).astype(np.int64)
    # Different posterior draws are alternative worlds, never one population.
    available = np.iinfo(np.int64).max - int(initial_pending_mask.sum())
    incoming = arrivals_arr.reshape(d, horizon * n_cohorts)
    threshold = available // max(horizon * n_cohorts, 1)
    for row in incoming[incoming.max(axis=1, initial=0) > threshold]:
        if sum(map(int, row)) > available:
            raise ValueError("total available population in a draw exceeds int64 count support")

    n, parent_days, pool_cohorts, active_cohorts, starts, initial_pool = _arrival_pools(
        arrivals_arr,
        initial_pending_mask,
        age0,
    )
    cure_logit = _cure_logit(params, cure_arr[pool_cohorts])
    log_ssus = np.zeros_like(n, dtype=np.float64)
    positions = np.flatnonzero(initial_pending_mask)
    if positions.size:
        last_age = age0[positions]
        if deadline_days is not None:
            last_age = np.minimum(last_age, deadline_days)
        log_ssus[:, initial_pool[positions]] = _window_log_survival(
            params,
            features_arr,
            allowed_arr,
            positions,
            _elapsed(cal[0], origins[positions]),
            np.zeros(positions.size, dtype=np.int64),
            last_age,
            int(last_age.max()) + 1,
            calendar_len,
        )

    rng = np.random.default_rng(seed)

    events_out = np.zeros((d, horizon, n_cohorts), dtype=np.int64)
    pending_out = np.zeros((d, horizon), dtype=np.int64)
    eligible_out = np.zeros((d, horizon), dtype=np.int64)

    for t in range(1, horizon + 1):
        day_idx = t - 1
        ci = asof_idx + t

        ages = t - parent_days
        started = ages >= 0
        elig_mask = started if deadline_days is None else started & (ages <= deadline_days)
        hazard_logit = _hazard_logit(
            params,
            ages,
            features_arr[pool_cohorts, ci],
            allowed_arr[pool_cohorts, ci],
        )
        hazard = _sigmoid(hazard_logit)

        cond_pi = _conditional_probability(
            cure_logit,
            log_ssus,
            "forecast_events simulation",
        )

        p_event = np.where(elig_mask, cond_pi * hazard, 0.0)
        events = rng.binomial(n, p_event)
        n -= events
        log_ssus -= np.where(elig_mask, _softplus(hazard_logit), 0.0)

        events_out[:, day_idx, active_cohorts] = np.add.reduceat(events, starts, axis=1)
        pending_out[:, day_idx] = np.where(started, n, 0).sum(axis=1)
        eligible_out[:, day_idx] = np.where(elig_mask, n, 0).sum(axis=1)

    return EventForecast(events=events_out, pending=pending_out, eligible=eligible_out)


# --------------------------------------------------------------------------
# Analytic eventual-receipt expectations
# --------------------------------------------------------------------------


def expected_return_receipts(
    history,
    initiation,
    receipt,
    *,
    calendar,
    initiation_features=None,
    receipt_features=None,
    initiation_cure_features=None,
    receipt_cure_features=None,
    initiation_allowed=None,
    receipt_allowed=None,
):
    """Exact analytic eventual-receipt expectations for the existing historical population.

    Returns ``(expected_uninitiated_receipts, expected_open_receipts)``,
    each ``[draw]``, under the continuing-tail assumption documented on this
    module. Reuses the same as-of conditioning math the generic kernel's
    hazard/cure primitives are built from, so callers needing these extras
    (e.g. a retail wrapper) never need to run a second simulation.

    ``history.policy_days`` may be ``None`` (unbounded initiation deadline):
    the eventual initiation probability then reduces to the posterior
    conditional susceptibility itself, exactly like the already-unbounded
    receipt stage's own eventual-receipt expectation.
    """
    init_params, d_init = _normalize_stage_parameters(initiation, "initiation")
    recv_params, d_recv = _normalize_stage_parameters(receipt, "receipt")
    d = _align_draws(d_init, d_recv)
    init_params = _broadcast_stage(init_params, d)
    recv_params = _broadcast_stage(recv_params, d)

    frame = history.frame.reset_index(drop=True)
    n_hist = len(frame)
    as_of = np.datetime64(history.as_of, "D")
    policy_days = history.policy_days

    p_init = int(init_params.beta.shape[-1])
    p_recv = int(recv_params.beta.shape[-1])
    q_init = int(init_params.cure_beta.shape[-1])
    q_recv = int(recv_params.cure_beta.shape[-1])

    if n_hist:
        sale_date_hist = np.asarray(to_day(frame["sale_date"].to_numpy()))
        initiation_date_hist = np.asarray(to_day(frame["initiation_date"].to_numpy()))
        receipt_date_hist = np.asarray(to_day(frame["receipt_date"].to_numpy()))
        eligible_mask = frame["eligible"].to_numpy(dtype=bool)
        open_mask = ~np.isnat(initiation_date_hist) & np.isnat(receipt_date_hist)
    else:
        sale_date_hist = np.zeros(0, dtype="datetime64[D]")
        initiation_date_hist = np.zeros(0, dtype="datetime64[D]")
        eligible_mask = np.zeros(0, dtype=bool)
        open_mask = np.zeros(0, dtype=bool)

    elig_positions = np.where(eligible_mask)[0]
    open_positions = np.where(open_mask)[0]
    n_elig = int(elig_positions.size)
    n_open = int(open_positions.size)

    elig_sale_date = sale_date_hist[elig_positions]
    open_init_date = initiation_date_hist[open_positions]

    deadline_candidates = []
    if n_elig and policy_days is not None:
        deadline_candidates.append((elig_sale_date + np.timedelta64(int(policy_days), "D")).max())
    latest_deadline = max(deadline_candidates) if deadline_candidates else as_of
    earliest_origin = sale_date_hist.min() if n_hist else as_of

    cal = _validate_calendar(calendar, as_of, 0, earliest_origin, latest_deadline)
    calendar_len = int(cal.size)
    cal0 = cal[0]

    init_features = _prepare_features(
        initiation_features, n_hist, calendar_len, p_init, "initiation_features"
    )
    recv_features = _prepare_features(
        receipt_features, n_hist, calendar_len, p_recv, "receipt_features"
    )
    init_cure = _prepare_cure_features(
        initiation_cure_features, n_hist, q_init, "initiation_cure_features"
    )
    recv_cure = _prepare_cure_features(
        receipt_cure_features, n_hist, q_recv, "receipt_cure_features"
    )
    init_allowed = _prepare_allowed(initiation_allowed, n_hist, calendar_len, "initiation_allowed")
    recv_allowed = _prepare_allowed(receipt_allowed, n_hist, calendar_len, "receipt_allowed")

    # -- as-of conditioning: eligible (uninitiated) historical rows -----------------------
    elig_age0 = _elapsed(elig_sale_date, as_of)  # [n_elig]
    elig_origin_idx = _elapsed(cal0, elig_sale_date)  # [n_elig]
    elig_feature_row = elig_positions

    elig_init_cure_logit = _cure_logit(init_params, init_cure[elig_feature_row])  # [D, n_elig]
    max_w_elig_asof = int(elig_age0.max()) + 1 if n_elig else 0
    elig_log_ssus_asof = _window_log_survival(
        init_params,
        init_features,
        init_allowed,
        elig_feature_row,
        elig_origin_idx,
        np.zeros(n_elig, dtype=np.int64),
        elig_age0,
        max_w_elig_asof,
        calendar_len,
    )  # [D, n_elig]
    elig_recv_cure_logit = _cure_logit(recv_params, recv_cure[elig_feature_row])  # [D, n_elig]

    # -- as-of conditioning: pre-existing open (initiated, not received) historical rows --
    open_age0 = _elapsed(open_init_date, as_of)  # [n_open]
    open_origin_idx = _elapsed(cal0, open_init_date)  # [n_open]
    open_feature_row = open_positions

    open_recv_cure_logit = _cure_logit(recv_params, recv_cure[open_feature_row])  # [D, n_open]
    max_w_open_asof = int(open_age0.max()) + 1 if n_open else 0
    open_log_ssus_asof = _window_log_survival(
        recv_params,
        recv_features,
        recv_allowed,
        open_feature_row,
        open_origin_idx,
        np.zeros(n_open, dtype=np.int64),
        open_age0,
        max_w_open_asof,
        calendar_len,
    )  # [D, n_open]

    # -- analytic exact expected remaining receipts (continuing-tail assumption) ----------
    # Open returns: eventual receipt given receipt-susceptible is certain (see module
    # docstring), so the expectation is exactly the posterior conditional susceptibility.
    if n_open:
        expected_open_receipts = np.sum(
            _conditional_probability(
                open_recv_cure_logit, open_log_ssus_asof, "receipt as-of conditioning"
            ),
            axis=1,
        )
    else:
        expected_open_receipts = np.zeros(d)

    # Remaining receipts require initiating (by the deadline, when one is configured) and
    # being receipt-susceptible. Use posterior susceptibility times susceptible event
    # probability; subtracting two large log-marginal survivals can lose the entire
    # remaining-event probability to cancellation.
    if n_elig:
        conditional_init = _conditional_probability(
            elig_init_cure_logit, elig_log_ssus_asof, "initiation as-of conditioning"
        )
        if policy_days is None:
            p_initiate = conditional_init
        else:
            remaining_width = (
                int(policy_days) - elig_age0
            )  # >= 0 because eligible implies age0 <= policy_days
            max_w_deadline = int(remaining_width.max())
            extra_log_ssus = _window_log_survival(
                init_params,
                init_features,
                init_allowed,
                elig_feature_row,
                elig_origin_idx,
                elig_age0 + 1,
                np.full(n_elig, int(policy_days), dtype=np.int64),
                max_w_deadline,
                calendar_len,
            )  # [D, n_elig]
            p_initiate = conditional_init * -np.expm1(extra_log_ssus)
        pi_receipt_raw = _sigmoid(elig_recv_cure_logit)
        expected_uninitiated_receipts = np.sum(p_initiate * pi_receipt_raw, axis=1)
    else:
        expected_uninitiated_receipts = np.zeros(d)

    return expected_uninitiated_receipts, expected_open_receipts


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def forecast_returns(
    history,
    initiation,
    receipt,
    *,
    calendar,
    horizon,
    future_sales=None,
    initiation_features=None,
    receipt_features=None,
    initiation_cure_features=None,
    receipt_cure_features=None,
    initiation_allowed=None,
    receipt_allowed=None,
    seed=0,
):
    """Simulate calendar-day return initiations/receipts from ``history.as_of``.

    A thin two-event composition of :func:`forecast_events`: the initiation
    kernel call's ``events`` output feeds unchanged as the receipt kernel
    call's ``arrivals``. ``future_sales`` is a
    :class:`~ttenet.integration.SalesForecast`, or a plain DataFrame parsed
    via ``SalesForecast.from_frame`` (requiring ``sale_date``/``quantity``
    columns). See the module docstring for the sequential-simulation
    mechanics and the exact receipt tail assumption behind
    ``expected_*_receipts``.
    """
    if horizon < 1:
        raise ValueError("horizon must be a positive number of days")

    init_params, d_init = _normalize_stage_parameters(initiation, "initiation")
    recv_params, d_recv = _normalize_stage_parameters(receipt, "receipt")

    frame = history.frame.reset_index(drop=True)
    n_hist = len(frame)
    as_of = np.datetime64(history.as_of, "D")
    policy_days = history.policy_days

    history_item_ids = frame["item_id"].to_numpy() if n_hist else np.array([])
    sales_forecast = _prepare_sales_forecast(future_sales)
    future_frame, sale_dates_future, counts_array = _future_cohorts_from_sales(
        sales_forecast, history_item_ids, as_of, horizon
    )
    n_future = len(future_frame)
    d_counts = counts_array.shape[0]

    d = _align_draws(d_init, d_recv, d_counts)
    init_params = _broadcast_stage(init_params, d)
    recv_params = _broadcast_stage(recv_params, d)
    counts_array = np.broadcast_to(counts_array, (d, n_future)).astype(np.int64)

    n_total = n_hist + n_future
    p_init = int(init_params.beta.shape[-1])
    p_recv = int(recv_params.beta.shape[-1])
    q_init = int(init_params.cure_beta.shape[-1])
    q_recv = int(recv_params.cure_beta.shape[-1])

    # -- historical row classification ---------------------------------------------------
    if n_hist:
        sale_date_hist = np.asarray(to_day(frame["sale_date"].to_numpy()))
        initiation_date_hist = np.asarray(to_day(frame["initiation_date"].to_numpy()))
        receipt_date_hist = np.asarray(to_day(frame["receipt_date"].to_numpy()))
        eligible_mask = frame["eligible"].to_numpy(dtype=bool)
    else:
        sale_date_hist = np.zeros(0, dtype="datetime64[D]")
        initiation_date_hist = np.zeros(0, dtype="datetime64[D]")
        receipt_date_hist = np.zeros(0, dtype="datetime64[D]")
        eligible_mask = np.zeros(0, dtype=bool)

    elig_positions = np.where(eligible_mask)[0]
    n_elig = int(elig_positions.size)
    elig_sale_date = sale_date_hist[elig_positions]

    # -- calendar coverage (historical eligible deadlines only, per the retail contract:
    # future cohorts' own deadlines are not required, since forecast_events needs no
    # calendar beyond as_of + horizon) --------------------------------------------------
    deadline_candidates = []
    if n_elig and policy_days is not None:
        deadline_candidates.append((elig_sale_date + np.timedelta64(int(policy_days), "D")).max())
    latest_deadline = max(deadline_candidates) if deadline_candidates else as_of
    earliest_origin = sale_date_hist.min() if n_hist else as_of

    cal = _validate_calendar(calendar, as_of, horizon, earliest_origin, latest_deadline)
    calendar_len = int(cal.size)

    # -- features / cure features / allowed --------------------------------------------------
    init_features = _prepare_features(
        initiation_features, n_total, calendar_len, p_init, "initiation_features"
    )
    recv_features = _prepare_features(
        receipt_features, n_total, calendar_len, p_recv, "receipt_features"
    )
    init_cure = _prepare_cure_features(
        initiation_cure_features, n_total, q_init, "initiation_cure_features"
    )
    recv_cure = _prepare_cure_features(
        receipt_cure_features, n_total, q_recv, "receipt_cure_features"
    )
    init_allowed = _prepare_allowed(initiation_allowed, n_total, calendar_len, "initiation_allowed")
    recv_allowed = _prepare_allowed(receipt_allowed, n_total, calendar_len, "receipt_allowed")

    output_dates = np.asarray(
        date_grid(as_of + np.timedelta64(1, "D"), as_of + np.timedelta64(horizon, "D"))
    )

    nat = np.datetime64("NaT", "D")
    future_nat = np.full(n_future, nat, dtype="datetime64[D]")

    # -- initiation kernel call: parent = sale, own event = initiation --------------------
    init_origins = np.concatenate([sale_date_hist, future_nat])
    init_observed = np.concatenate([initiation_date_hist, future_nat])
    init_arrivals = np.zeros((d, horizon, n_total), dtype=np.int64)
    if n_future:
        future_day = _elapsed(as_of, sale_dates_future)  # 1..horizon, validated above
        future_cols = n_hist + np.arange(n_future)
        init_arrivals[:, future_day - 1, future_cols] = counts_array

    seed_seq = np.random.SeedSequence(seed)
    init_seed, recv_seed = seed_seq.spawn(2)

    init_result = forecast_events(
        init_params,
        origins=init_origins,
        observed=init_observed,
        arrivals=init_arrivals,
        calendar=cal,
        as_of=as_of,
        horizon=horizon,
        features=init_features,
        cure_features=init_cure,
        allowed=init_allowed,
        deadline_days=policy_days,
        seed=init_seed,
    )

    # -- receipt kernel call: parent = initiation, own event = receipt --------------------
    # A child uses the upstream EventForecast.events unchanged as its arrivals, so every
    # initiated unit (historical or future, on whatever day it actually initiates)
    # immediately keeps its own independently correct receipt-age clock.
    recv_origins = np.concatenate([initiation_date_hist, future_nat])
    recv_observed = np.concatenate([receipt_date_hist, future_nat])

    recv_result = forecast_events(
        recv_params,
        origins=recv_origins,
        observed=recv_observed,
        arrivals=init_result.events,
        calendar=cal,
        as_of=as_of,
        horizon=horizon,
        features=recv_features,
        cure_features=recv_cure,
        allowed=recv_allowed,
        deadline_days=None,  # retail receipts have no imposed deadline
        seed=recv_seed,
    )

    initiations = init_result.events.sum(axis=-1)
    eligible = init_result.eligible
    receipts = recv_result.events.sum(axis=-1)
    open_returns = recv_result.pending

    expected_uninitiated_receipts, expected_open_receipts = expected_return_receipts(
        history,
        init_params,
        recv_params,
        calendar=cal,
        initiation_features=init_features[:n_hist],
        receipt_features=recv_features[:n_hist],
        initiation_cure_features=init_cure[:n_hist],
        receipt_cure_features=recv_cure[:n_hist],
        initiation_allowed=init_allowed[:n_hist],
        receipt_allowed=recv_allowed[:n_hist],
    )
    expected_existing_receipts = expected_uninitiated_receipts + expected_open_receipts

    return ReturnForecast(
        dates=output_dates,
        initiations=initiations,
        receipts=receipts,
        eligible=eligible,
        open_returns=open_returns,
        expected_existing_receipts=np.asarray(expected_existing_receipts),
        expected_uninitiated_receipts=np.asarray(expected_uninitiated_receipts),
        expected_open_receipts=np.asarray(expected_open_receipts),
        sales=sales_forecast,
    )
