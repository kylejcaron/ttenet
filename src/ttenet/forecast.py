"""Calendar-day, count-conserving return forecasts for two linked mixture-cure stages.

``forecast_returns`` propagates a retail return process (sale -> initiation ->
receipt) forward from ``history.as_of`` through a calendar-day horizon. It
conditions every still-outstanding historical unit on its own observed
survival, then sequentially simulates initiation and receipt day by day for
both the conditioned historical population and any supplied future-sales
cohorts. Historical units are simulated one at a time (each row is exactly
one sold unit); future cohorts are simulated as conserved integer counts via
day-by-day binomial thinning, so population mass is exact within every
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

Sequential simulation mechanics
--------------------------------
Every day, initiation is simulated before receipt, so a unit that initiates
today is immediately eligible for receipt the same day. A unit's *marginal*
(cure-integrated) hazard on a given day is the raw stage hazard multiplied by
the *current posterior probability of being susceptible given survival so
far*, recomputed every day from a running cumulative log-survival state.
This is mathematically equivalent to drawing each unit's latent cure status
once and simulating only the susceptible sub-population forward, without
ever having to materialize or track that latent draw.

Receipt cohorts are always tracked by their own initiation date (age since
initiation), never by sale date, because their age clocks differ from the
initiation clock.

Receipt tail assumption
------------------------
Receipt age uses the final fitted baseline bin beyond the learned ages.
Eventual receipt estimates assume positive tail exposure continues and
receiving eventually reopens. For a susceptible unit, cumulative tail hazard
then diverges and eventual receipt probability is one.

The ``expected_*_receipts`` fields concern the historical population at the
forecast origin, not future sales. Their initiation component still depends
on the supplied calendar through the policy deadline. Simulated ``receipts``
include future sales and process noise, so a realized path need not be below
the historical-population eventual expectation. Finite-horizon expected
receipts from the same historical population cannot exceed that eventual
expectation. No receipt deadline is invented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .dates import date_grid, elapsed_days, to_day
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
    definition of the tail assumption.
    """

    dates: Any  # [horizon] datetime64[D], as_of+1 .. as_of+horizon inclusive
    initiations: Any  # [draw, horizon] integer counts
    receipts: Any  # [draw, horizon] integer counts
    eligible: Any  # [draw, horizon] uninitiated and not past deadline, end-of-day
    open_returns: Any  # [draw, horizon] initiated, not received, end-of-day
    expected_existing_receipts: Any  # [draw] eventual receipts from past sales only
    expected_uninitiated_receipts: Any  # [draw] eventual-receipt component from pending initiations
    expected_open_receipts: Any  # [draw] eventual-receipt component from existing open returns


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
            f"Cannot align posterior parameter draws with future-count draws: sizes {unique} "
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


def _validate_count_array(values, name):
    arr = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite counts")
    if np.any(arr < 0):
        raise ValueError(f"{name} must be nonnegative")
    if not np.all(np.equal(np.mod(arr, 1.0), 0.0)):
        raise ValueError(f"{name} must be integer-valued")


def _prepare_counts(future_counts, future_frame, n_future):
    if future_counts is not None:
        counts = np.asarray(future_counts)
        if counts.ndim != 2 or counts.shape[1] != n_future:
            raise ValueError(
                f"future_counts must have shape (draws, {n_future}); got {counts.shape}"
            )
        _validate_count_array(counts, "future_counts")
        return counts.astype(np.int64), counts.shape[0]
    if n_future == 0:
        return np.zeros((1, 0), dtype=np.int64), 1
    if "quantity" not in future_frame.columns:
        raise ValueError(
            "future_sales requires a quantity column when future_counts is not supplied"
        )
    quantity = future_frame["quantity"].to_numpy()
    _validate_count_array(quantity, "future_sales.quantity")
    return quantity.astype(np.int64)[None, :], 1


def _prepare_future_cohorts(future_sales, history_item_ids, as_of, horizon):
    if future_sales is None or len(future_sales) == 0:
        empty = pd.DataFrame(columns=["item_id", "sale_date"])
        return empty, np.zeros(0, dtype="datetime64[D]")
    if "item_id" not in future_sales.columns or "sale_date" not in future_sales.columns:
        raise ValueError("future_sales requires item_id and sale_date columns")
    ids = future_sales["item_id"].to_numpy()
    if pd.Series(ids).duplicated().any():
        raise ValueError("future_sales item_id values must be unique")
    overlap = set(ids) & set(history_item_ids)
    if overlap:
        raise ValueError(
            f"future_sales item_id overlaps historical item_id values: {sorted(map(str, overlap))}"
        )
    sale_dates = np.asarray(to_day(future_sales["sale_date"].to_numpy()))
    if np.isnat(sale_dates).any():
        raise ValueError("future_sales sale_date must not be missing")
    output_start = as_of + np.timedelta64(1, "D")
    output_end = as_of + np.timedelta64(horizon, "D")
    if np.any((sale_dates < output_start) | (sale_dates > output_end)):
        raise ValueError("future_sales sale_date must fall within the output horizon")
    out = future_sales.reset_index(drop=True).assign(sale_date=sale_dates)
    return out, sale_dates


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
            "calendar must extend through the latest uninitiated/future-cohort initiation deadline"
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
    future_counts=None,
    initiation_features=None,
    receipt_features=None,
    initiation_cure_features=None,
    receipt_cure_features=None,
    initiation_allowed=None,
    receipt_allowed=None,
    seed=0,
):
    """Simulate calendar-day return initiations/receipts from ``history.as_of``.

    See the module docstring for the sequential-simulation mechanics and the
    exact receipt tail assumption behind ``expected_*_receipts``.
    """
    if horizon < 1:
        raise ValueError("horizon must be a positive number of days")

    init_params, d_init = _normalize_stage_parameters(initiation, "initiation")
    recv_params, d_recv = _normalize_stage_parameters(receipt, "receipt")

    frame = history.frame.reset_index(drop=True)
    n_hist = len(frame)
    as_of = np.datetime64(history.as_of, "D")
    policy_days = int(history.policy_days)

    history_item_ids = frame["item_id"].to_numpy() if n_hist else np.array([])
    future_frame, sale_dates_future = _prepare_future_cohorts(
        future_sales, history_item_ids, as_of, horizon
    )
    n_future = len(future_frame)
    counts_array, d_counts = _prepare_counts(future_counts, future_frame, n_future)

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
        open_mask = ~np.isnat(initiation_date_hist) & np.isnat(receipt_date_hist)
    else:
        sale_date_hist = np.zeros(0, dtype="datetime64[D]")
        initiation_date_hist = np.zeros(0, dtype="datetime64[D]")
        receipt_date_hist = np.zeros(0, dtype="datetime64[D]")
        eligible_mask = np.zeros(0, dtype=bool)
        open_mask = np.zeros(0, dtype=bool)

    elig_positions = np.where(eligible_mask)[0]
    open_positions = np.where(open_mask)[0]
    n_elig = int(elig_positions.size)
    n_open = int(open_positions.size)

    elig_sale_date = sale_date_hist[elig_positions]
    open_init_date = initiation_date_hist[open_positions]

    # -- calendar coverage ------------------------------------------------------------------
    deadline_candidates = []
    if n_elig:
        deadline_candidates.append((elig_sale_date + np.timedelta64(policy_days, "D")).max())
    if n_future:
        deadline_candidates.append((sale_dates_future + np.timedelta64(policy_days, "D")).max())
    latest_deadline = max(deadline_candidates) if deadline_candidates else as_of
    earliest_origin = sale_date_hist.min() if n_hist else as_of

    cal = _validate_calendar(calendar, as_of, horizon, earliest_origin, latest_deadline)
    calendar_len = int(cal.size)
    cal0 = cal[0]
    asof_idx = int(_elapsed(cal0, as_of))

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

    # Remaining receipts require initiating by the deadline and being
    # receipt-susceptible. Use posterior susceptibility times susceptible
    # event probability; subtracting two large log-marginal survivals can
    # lose the entire remaining-event probability to cancellation.
    if n_elig:
        remaining_width = (
            policy_days - elig_age0
        )  # >= 0 because eligible implies age0 <= policy_days
        max_w_deadline = int(remaining_width.max())
        extra_log_ssus = _window_log_survival(
            init_params,
            init_features,
            init_allowed,
            elig_feature_row,
            elig_origin_idx,
            elig_age0 + 1,
            np.full(n_elig, policy_days, dtype=np.int64),
            max_w_deadline,
            calendar_len,
        )  # [D, n_elig]
        conditional_init = _conditional_probability(
            elig_init_cure_logit, elig_log_ssus_asof, "initiation as-of conditioning"
        )
        p_initiate = conditional_init * -np.expm1(extra_log_ssus)
        pi_receipt_raw = _sigmoid(elig_recv_cure_logit)
        expected_uninitiated_receipts = np.sum(p_initiate * pi_receipt_raw, axis=1)
    else:
        expected_uninitiated_receipts = np.zeros(d)

    expected_existing_receipts = expected_open_receipts + expected_uninitiated_receipts

    # -- day-by-day simulation state -------------------------------------------------------
    rng = np.random.default_rng(seed)

    def _binomial(n_arr, p_arr):
        n_clip = np.clip(np.round(n_arr), 0, None).astype(np.int64)
        p_clip = np.clip(p_arr, 0.0, 1.0)
        return rng.binomial(n_clip, p_clip)

    # historical eligible units: age going into simulation day 1 is elig_age0 + 1 (as-of
    # conditioning already covers ages 0..elig_age0 inclusive).
    elig_init_active = np.ones((d, n_elig), dtype=bool)
    elig_init_age = np.broadcast_to(elig_age0 + 1, (d, n_elig)).astype(np.int64).copy()
    elig_init_log_ssus = np.broadcast_to(elig_log_ssus_asof, (d, n_elig)).astype(np.float64).copy()

    elig_open_active = np.zeros((d, n_elig), dtype=bool)
    elig_open_age = np.zeros((d, n_elig), dtype=np.int64)
    elig_open_log_ssus = np.zeros((d, n_elig), dtype=np.float64)

    hist_open_active = np.ones((d, n_open), dtype=bool)
    hist_open_age = np.broadcast_to(open_age0 + 1, (d, n_open)).astype(np.int64).copy()
    hist_open_log_ssus = np.broadcast_to(open_log_ssus_asof, (d, n_open)).astype(np.float64).copy()

    fut_feature_row = n_hist + np.arange(n_future)
    fut_init_remaining = np.zeros((d, n_future), dtype=np.int64)
    fut_init_age = np.zeros((d, n_future), dtype=np.int64)
    fut_init_log_ssus = np.zeros((d, n_future), dtype=np.float64)
    fut_activation_day = (
        _elapsed(as_of, sale_dates_future) if n_future else np.zeros(0, dtype=np.int64)
    )
    fut_init_cure_logit = (
        _cure_logit(init_params, init_cure[fut_feature_row]) if n_future else np.zeros((d, 0))
    )
    fut_recv_cure_logit = (
        _cure_logit(recv_params, recv_cure[fut_feature_row]) if n_future else np.zeros((d, 0))
    )

    bucket_width = horizon
    fut_open_n = np.zeros((d, n_future, bucket_width), dtype=np.int64)
    fut_open_log_ssus = np.zeros((d, n_future, bucket_width), dtype=np.float64)

    initiations = np.zeros((d, horizon), dtype=np.int64)
    receipts = np.zeros((d, horizon), dtype=np.int64)
    eligible = np.zeros((d, horizon), dtype=np.int64)
    open_returns = np.zeros((d, horizon), dtype=np.int64)

    for t in range(1, horizon + 1):
        ci = asof_idx + t
        out_idx = t - 1

        # activate future cohorts sold today (age zero, same-day initiation permitted)
        if n_future:
            activate = fut_activation_day == t
            if np.any(activate):
                fut_init_remaining[:, activate] = counts_array[:, activate]
                fut_init_age[:, activate] = 0
                fut_init_log_ssus[:, activate] = 0.0

        # -- initiation: historical eligible units -----------------------------------------
        if n_elig:
            elig_init_active = elig_init_active & (elig_init_age <= policy_days)
            feats = init_features[elig_feature_row, ci, :]
            allow = init_allowed[elig_feature_row, ci]
            hazard_logit = _hazard_logit(init_params, elig_init_age, feats, allow)
            hazard = _sigmoid(hazard_logit)
            cond_pi = _conditional_probability(
                elig_init_cure_logit, elig_init_log_ssus, "initiation simulation"
            )
            p_event = np.where(elig_init_active, cond_pi * hazard, 0.0)
            events = np.where(
                elig_init_active, _binomial(elig_init_active.astype(np.int64), p_event), 0
            )
            triggered = events.astype(bool)
            survived = elig_init_active & ~triggered
            elig_init_log_ssus = np.where(
                survived, elig_init_log_ssus - _softplus(hazard_logit), elig_init_log_ssus
            )
            elig_init_active = elig_init_active & ~triggered
            elig_init_age = elig_init_age + 1
            elig_open_active = elig_open_active | triggered
            elig_open_age = np.where(triggered, 0, elig_open_age)
            elig_open_log_ssus = np.where(triggered, 0.0, elig_open_log_ssus)
            today_elig_init = events.sum(axis=1)
        else:
            today_elig_init = np.zeros(d, dtype=np.int64)

        # -- initiation: future cohorts ------------------------------------------------------
        if n_future:
            fut_init_remaining = np.where(fut_init_age <= policy_days, fut_init_remaining, 0)
            active = fut_init_remaining > 0
            feats = init_features[fut_feature_row, ci, :]
            allow = init_allowed[fut_feature_row, ci]
            hazard_logit = _hazard_logit(init_params, fut_init_age, feats, allow)
            hazard = _sigmoid(hazard_logit)
            cond_pi = _conditional_probability(
                fut_init_cure_logit, fut_init_log_ssus, "future initiation simulation"
            )
            p_event = np.where(active, cond_pi * hazard, 0.0)
            events = _binomial(np.where(active, fut_init_remaining, 0), p_event)
            fut_init_remaining = fut_init_remaining - events
            fut_init_log_ssus = np.where(
                active, fut_init_log_ssus - _softplus(hazard_logit), fut_init_log_ssus
            )
            fut_init_age = fut_init_age + 1
            fut_open_n[:, :, 0] = fut_open_n[:, :, 0] + events
            fut_open_log_ssus[:, :, 0] = 0.0
            today_fut_init = events.sum(axis=1)
        else:
            today_fut_init = np.zeros(d, dtype=np.int64)

        initiations[:, out_idx] = today_elig_init + today_fut_init

        # -- receipt: historical eligible-triggered opens (same-day receipt permitted) -----
        if n_elig:
            active = elig_open_active
            feats = recv_features[elig_feature_row, ci, :]
            allow = recv_allowed[elig_feature_row, ci]
            hazard_logit = _hazard_logit(recv_params, elig_open_age, feats, allow)
            hazard = _sigmoid(hazard_logit)
            cond_pi = _conditional_probability(
                elig_recv_cure_logit, elig_open_log_ssus, "receipt simulation"
            )
            p_event = np.where(active, cond_pi * hazard, 0.0)
            events = np.where(active, _binomial(active.astype(np.int64), p_event), 0)
            received = events.astype(bool)
            survived = active & ~received
            elig_open_log_ssus = np.where(
                survived, elig_open_log_ssus - _softplus(hazard_logit), elig_open_log_ssus
            )
            elig_open_active = elig_open_active & ~received
            elig_open_age = elig_open_age + 1
            today_elig_receipt = events.sum(axis=1)
        else:
            today_elig_receipt = np.zeros(d, dtype=np.int64)

        # -- receipt: pre-existing historical open units -------------------------------------
        if n_open:
            active = hist_open_active
            feats = recv_features[open_feature_row, ci, :]
            allow = recv_allowed[open_feature_row, ci]
            hazard_logit = _hazard_logit(recv_params, hist_open_age, feats, allow)
            hazard = _sigmoid(hazard_logit)
            cond_pi = _conditional_probability(
                open_recv_cure_logit, hist_open_log_ssus, "open receipt simulation"
            )
            p_event = np.where(active, cond_pi * hazard, 0.0)
            events = np.where(active, _binomial(active.astype(np.int64), p_event), 0)
            received = events.astype(bool)
            survived = active & ~received
            hist_open_log_ssus = np.where(
                survived, hist_open_log_ssus - _softplus(hazard_logit), hist_open_log_ssus
            )
            hist_open_active = hist_open_active & ~received
            hist_open_age = hist_open_age + 1
            today_hist_open_receipt = events.sum(axis=1)
        else:
            today_hist_open_receipt = np.zeros(d, dtype=np.int64)

        # -- receipt: future cohorts, tracked as an age-bucket cascade ------------------------
        if n_future:
            ages_grid = np.broadcast_to(np.arange(bucket_width)[None, :], (n_future, bucket_width))
            ages_grid_d = np.broadcast_to(ages_grid[None, :, :], (d, n_future, bucket_width))
            feats_today = recv_features[fut_feature_row, ci, :]  # [F, P]
            allow_today = recv_allowed[fut_feature_row, ci]  # [F]
            feats_grid = np.broadcast_to(
                feats_today[:, None, :], (n_future, bucket_width, feats_today.shape[-1])
            )
            allow_grid = np.broadcast_to(allow_today[:, None], (n_future, bucket_width))
            hazard_logit = _hazard_logit(
                recv_params, ages_grid_d, feats_grid, allow_grid
            )  # [D, F, B]
            cure_grid = np.broadcast_to(
                fut_recv_cure_logit[:, :, None], (d, n_future, bucket_width)
            )
            cond_pi = _conditional_probability(
                cure_grid, fut_open_log_ssus, "future receipt simulation"
            )
            hazard = _sigmoid(hazard_logit)
            p_event = cond_pi * hazard
            events = _binomial(fut_open_n, p_event)
            fut_open_n = fut_open_n - events
            fut_open_log_ssus = fut_open_log_ssus - _softplus(hazard_logit)
            today_fut_receipt = events.sum(axis=(1, 2))
            # Only age buckets when another step will consume them. Shifting
            # on the last day would discard the oldest still-open cohort.
            if t < horizon:
                fut_open_n = np.concatenate(
                    [np.zeros((d, n_future, 1), dtype=np.int64), fut_open_n[:, :, :-1]], axis=-1
                )
                fut_open_log_ssus = np.concatenate(
                    [np.zeros((d, n_future, 1)), fut_open_log_ssus[:, :, :-1]], axis=-1
                )
        else:
            today_fut_receipt = np.zeros(d, dtype=np.int64)

        receipts[:, out_idx] = today_elig_receipt + today_hist_open_receipt + today_fut_receipt

        eligible[:, out_idx] = (
            elig_init_active.sum(axis=1) if n_elig else np.zeros(d, dtype=np.int64)
        ) + (fut_init_remaining.sum(axis=1) if n_future else np.zeros(d, dtype=np.int64))
        open_returns[:, out_idx] = (
            (elig_open_active.sum(axis=1) if n_elig else np.zeros(d, dtype=np.int64))
            + (hist_open_active.sum(axis=1) if n_open else np.zeros(d, dtype=np.int64))
            + (fut_open_n.sum(axis=(1, 2)) if n_future else np.zeros(d, dtype=np.int64))
        )

    return ReturnForecast(
        dates=output_dates,
        initiations=initiations,
        receipts=receipts,
        eligible=eligible,
        open_returns=open_returns,
        expected_existing_receipts=np.asarray(expected_existing_receipts),
        expected_uninitiated_receipts=np.asarray(expected_uninitiated_receipts),
        expected_open_receipts=np.asarray(expected_open_receipts),
    )
