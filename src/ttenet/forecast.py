"""Calendar-day, count-conserving event forecasts from the shared survival law.

``forecast_events`` propagates a single parent -> child event (e.g. sale ->
initiation, or initiation -> receipt) forward from ``as_of`` through a
calendar-day horizon, given a stage's fitted timing family, known historical
origin/own-event dates, and an exact realized-arrivals array for new parent
events during the horizon. ``forecast_returns`` composes two calls (initiation,
then receipt, the second consuming the first's ``events`` output as its
``arrivals``) to reproduce the full retail return process, and adds the exact
eventual-receipt expectations for the existing historical population via
``expected_return_receipts``.

Every cohort (one historical row, or one future dated sales cohort) is
tracked as a conserved integer count, never expanded per unit: a historical
row simply starts with count 1. Population mass is exact within every
posterior/count draw and no unit is ever materialized or destroyed.

Stage parameters
----------------
Each stage is a ``StageParameters`` (one draw, or a leading draw axis) or a
``StageFit``. Both replay through ``StageFit.timing``: the default family
evaluates ``event_times.default_timing`` on the draw's parameters; a custom
family reruns its ``family.model`` with the draw's named sites
and resolved shared values substituted, so new ages, features and horizons
re-evaluate the family rather than extend a frozen training grid. The host
only prepares ``TimingInputs`` (ages, regressors) and administrative masks
(closures, exposure, known pre-entry survival); cure marginalization,
conditioning and date masses come from ``event_times.survival_kernel``.

Native allocation driver
------------------------
``forecast_events`` packs cohorts into immediate-parent-date pools. A unit,
or a count cohort whose parents arrive on only one day per draw, needs one
pool with a draw-specific clock; cohorts with arrivals on several days keep
separate pools for the actual arrival dates; empty pools are not allocated.
Historical pools enter the horizon as selected survivors: their
susceptibility logits are conditioned on the known event-free run through
``as_of`` (the kernel's pre-entry rule evaluated on the history window), so
no historical event path is materialized. Future pools cannot fire before
their parent day; an age-zero trial runs on that day, allowing same-day
transitions. Every pool retains its original root-cohort identity.

For every posterior draw the pools form a NumPyro Forecast program: an
upstream ``Horizon`` over the forecast days with an empty observed prefix,
and ``predict`` with the native ``EventTime`` law for pools statically known
to hold one unit or ``CohortEventTime`` for counted pools. The generated
suffix is read from the program's ``forecast`` site through
``numpyro.infer.Predictive`` with a singleton batch, one program per draw,
so draw-specific parent dates and counts never cross other draws' posterior
rows; ``forecast`` in ``numpyro_forecast`` runs the same ``Predictive`` call
but refuses the empty posterior of a fixed family, which this driver must
replay ``StageFit.num_samples`` times. Draws are mapped in fixed-shape blocks
under a local ``jax.enable_x64`` scope so counts stay exact int64; nothing
here samples on the host or changes global precision.

Once a pool's age exceeds a configured ``deadline_days``, its members stop
accruing hazard but remain counted in ``pending`` forever; ``eligible`` is
the subset still within the deadline.

Tail assumptions
----------------
Eventual expectations need each family's typed ``tail`` declaration:
``ProperTail()`` means a susceptible unit eventually fires under the
documented continuing-exposure assumption (positive hazard keeps running and
closed days eventually reopen); the default family's final fitted baseline
bin continues beyond the learned ages and is proper. ``FiniteTail(last_age=a)``
integrates the calendar-masked kernel through age ``a``,
closures included -- a terminal atom falling on a closure does not fire.
``UnknownTail()`` cannot support an eventual expectation and is
refused unless a deadline bounds the stage. Beyond the supplied calendar,
finite windows continue all-open with each row's last calendar features
held constant; proper initiation mass that remains after the calendar ends
completes with the stationary receipt probability computed under that same
continuation from the same family.

The ``expected_*_receipts`` fields concern the historical population at the
forecast origin, not future sales. Their initiation component depends on the
supplied calendar through the policy deadline (or the family's finite
support); an unbounded proper initiation reduces to the conditional
susceptibility itself. Simulated ``receipts`` include future sales and
process noise, so a realized path need not be below the historical-
population eventual expectation. No receipt deadline is invented.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import lax, random
from jax.nn import sigmoid
from numpyro.infer import Predictive
from numpyro_forecast import Horizon, predict

from .dates import date_grid, elapsed_days, to_day
from .distributions import CohortEventTime, EventTime
from .event_times import TimingInputs, survival_kernel
from .families import FiniteTail, ProperTail, TailBehavior, validate_family
from .integration import SalesForecast, _validate_counts
from .models import StageFit
from .survival import StageParameters

_BLOCK_CELLS = 1 << 22
"""Per-block budget of ``[time, pool]`` cells: draws are mapped in blocks this size."""


@dataclass(frozen=True)
class ReturnForecast:
    """Calendar-day forecast of return initiations, receipts, and outstanding counts.

    All ``[draw, ...]`` fields carry a leading posterior/count draw axis
    (size 1 when no posterior draws or future-count draws were supplied).
    ``initiations``, ``receipts``, ``eligible``, and ``open_returns`` are
    stochastic per-draw simulations over the supplied ``calendar``/horizon.
    ``expected_existing_receipts``, ``expected_uninitiated_receipts``, and
    ``expected_open_receipts`` are exact analytic expectations under the
    tail assumptions documented on this module and are independent of the
    simulated draws. ``sales`` is the bundled
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
# Stage parameter resolution
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


def _stage_tail(fit) -> TailBehavior:
    """The family's typed tail declaration; the default family is proper."""
    if fit.family is None:
        return ProperTail()
    validate_family(fit.family)
    return fit.family.tail


class _Stage(NamedTuple):
    """A stage ready to replay: its fit, draw count, regressor widths and tail."""

    fit: StageFit
    draws: int
    widths: tuple[int, int] | None  # (P, Q) for the default family, None for a custom one
    tail: TailBehavior


def _resolve_stage(parameters, name):
    """Accept ``StageParameters`` (one draw or stacked) or a ``StageFit``."""
    if isinstance(parameters, StageFit):
        fit = parameters
    else:
        normalized, draws = _normalize_stage_parameters(parameters, name)
        fit = StageFit(_broadcast_stage(normalized, draws), np.zeros(0))
    widths = None
    if fit.family is None:
        widths = (int(fit.parameters.beta.shape[-1]), int(fit.parameters.cure_beta.shape[-1]))
    return _Stage(fit, fit.draws, widths, _stage_tail(fit))


def _x64(tree):
    """Device copies of a pytree with float64/int64 leaves (inside ``jax.enable_x64``)."""

    def promote(leaf):
        value = jnp.asarray(leaf)
        if jnp.issubdtype(value.dtype, jnp.floating):
            return value.astype(jnp.float64)
        if jnp.issubdtype(value.dtype, jnp.integer):
            return value.astype(jnp.int64)
        return value

    return jax.tree.map(promote, tree)


def _root_key(seed):
    sequence = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    return random.key(int(sequence.generate_state(1, dtype=np.uint32)[0]))


def _map_draws(block, draws, cells, key, *args):
    """Run a jitted block function over every draw index in fixed-shape blocks.

    Block ``i`` receives draw indices ``arange(i * size, (i + 1) * size) % draws``
    and gathers its own rows of the per-draw arguments, so every block shares
    one shape and compiles once; wrapped rows of the last block are
    discarded. Outputs are concatenated along the draw axis.
    """
    size = max(1, min(draws, _BLOCK_CELLS // max(int(cells), 1)))
    pieces = []
    for index, start in enumerate(range(0, draws, size)):
        indices = jnp.asarray(np.arange(start, start + size) % draws)
        result = block(random.fold_in(key, index), indices, *args)
        valid = min(size, draws - start)
        pieces.append(jax.tree.map(lambda value: np.asarray(value)[:valid], result))
    return jax.tree.map(lambda *values: np.concatenate(values), *pieces)


# --------------------------------------------------------------------------
# Feature / allowed / count validation
# --------------------------------------------------------------------------


def _prepare_features(features, n_rows, calendar_len, width, name):
    """``[N, T, P]`` regressors; ``width`` None accepts any P (a custom family's own)."""
    if features is None:
        if width:
            raise ValueError(
                f"{name} is required because the fitted model has feature width {width}"
            )
        return np.zeros((n_rows, calendar_len, 0), dtype=np.float64)
    arr = np.asarray(features, dtype=np.float64)
    expected = (n_rows, calendar_len) + (() if width is None else (width,))
    if arr.ndim != 3 or tuple(arr.shape[: len(expected)]) != expected:
        shown = expected if width is not None else expected + ("P",)
        raise ValueError(f"{name} must have shape {shown}; got {tuple(arr.shape)}")
    return arr


def _prepare_cure_features(features, n_rows, width, name):
    if features is None:
        if width:
            raise ValueError(
                f"{name} is required because the fitted model has cure feature width {width}"
            )
        return np.zeros((n_rows, 0), dtype=np.float64)
    arr = np.asarray(features, dtype=np.float64)
    expected = (n_rows,) + (() if width is None else (width,))
    if arr.ndim != 2 or tuple(arr.shape[: len(expected)]) != expected:
        shown = expected if width is not None else expected + ("Q",)
        raise ValueError(f"{name} must have shape {shown}; got {tuple(arr.shape)}")
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


class _StageInputs(NamedTuple):
    """Validated calendar-indexed regressors and closures of one stage's rows."""

    features: Any  # [N, T, P]
    cure_features: Any  # [N, Q]
    allowed: Any  # [N, T]


def _stage_inputs(stage, features, cure_features, allowed, n_rows, calendar_len, *, prefix=""):
    """Validate a stage's host regressors; a custom family's widths are its own."""
    p, q = stage.widths if stage.widths is not None else (None, None)
    return _StageInputs(
        _prepare_features(features, n_rows, calendar_len, p, f"{prefix}features"),
        _prepare_cure_features(cure_features, n_rows, q, f"{prefix}cure_features"),
        _prepare_allowed(allowed, n_rows, calendar_len, f"{prefix}allowed"),
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
            "calendar must extend through the latest uninitiated historical initiation "
            "deadline or finite initiation support window"
        )
    return cal


def _elapsed(origin, target):
    return np.asarray(elapsed_days(origin, target)).astype(np.int64)


# --------------------------------------------------------------------------
# Age-relative windows of the shared kernel
# --------------------------------------------------------------------------


class _AgeWindow(NamedTuple):
    """``[A, N]`` age-relative grid of selected rows with their calendar regressors.

    Cell ``(a, n)`` is row ``n`` at age ``a`` on calendar day
    ``origin[n] + a``. Days past the supplied calendar continue all-open with
    the last calendar day's regressors held constant.
    """

    ages: Any  # [A, N]
    features: Any  # [A, N, P]
    cure_features: Any  # [N, Q]
    allowed: Any  # [A, N]
    pre_entry: Any  # [A, N] known event-free ages
    exposure: Any  # [A, N] ages still able to fire


def _age_window(inputs, rows, origin_idx, span, *, known_through, exposed_through):
    """Build an ``_AgeWindow`` of ``span`` ages for ``rows`` of a stage's inputs.

    ``known_through[n]`` is the last age with a known event-free outcome
    (``-1`` for none); ``exposed_through[n]`` is the last age that may still
    fire (``-1`` for no exposed window); exposure starts the age after the
    known run.
    """
    rows = np.asarray(rows, dtype=np.int64)
    origin_idx = np.asarray(origin_idx, dtype=np.int64)
    n = rows.shape[0]
    calendar_len = inputs.allowed.shape[1]
    ages = np.broadcast_to(np.arange(span, dtype=np.int64)[:, None], (span, n))
    calendar_day = origin_idx[None, :] + ages
    within = calendar_day < calendar_len
    index = np.clip(calendar_day, 0, calendar_len - 1)
    row_index = np.broadcast_to(rows[None, :], (span, n))
    known_through = np.asarray(known_through, dtype=np.int64)[None, :]
    exposed_through = np.asarray(exposed_through, dtype=np.int64)[None, :]
    return _AgeWindow(
        ages=ages,
        features=inputs.features[row_index, index],
        cure_features=inputs.cure_features[rows],
        allowed=np.where(within, inputs.allowed[row_index, index], True),
        pre_entry=ages <= known_through,
        exposure=(ages > known_through) & (ages <= exposed_through),
    )


def _window_kernel(fit, draw, window):
    """The shared kernel of one posterior draw on an age window."""
    timing, logits = fit.timing(
        TimingInputs(window.ages, window.features, window.cure_features), draw=draw
    )
    return survival_kernel(
        timing,
        logits,
        allowed=window.allowed,
        exposure=window.exposure,
        pre_entry=window.pre_entry,
    )


def _window_probability(kernel, *, eventual):
    """Per-row probability of firing: within the exposed window, or eventually (proper tail)."""
    susceptible = jnp.exp(kernel.log_susceptible)
    if eventual:
        return susceptible
    return susceptible * -jnp.expm1(jnp.sum(kernel.log_survival_step, axis=-2))


# --------------------------------------------------------------------------
# Native pool allocation
# --------------------------------------------------------------------------


class _PoolLayout(NamedTuple):
    """Draw-independent description of one pool partition over the forecast days."""

    features: Any  # [horizon, pools, P]
    cure_features: Any  # [pools, Q]
    allowed: Any  # [horizon, pools]
    cohort: Any  # [pools] root-cohort index of each pool
    historical: Any  # [h] partition positions of pools pending at as_of
    entry_rows: Any  # [h] rows of the history window those pools condition on


class _Partition(NamedTuple):
    layout: _PoolLayout
    parent_day: Any  # [pools] forecast day of the immediate parent event (<= 0: pending at as_of)
    counts: Any  # [pools] integer population


class _PoolInputs(NamedTuple):
    """Covariates of the pool program over the forecast days.

    ``allowed`` is the time-major ``[horizon, pools]`` calendar grid the
    upstream ``Horizon`` is derived from; the pools' metadata and other
    regressors ride along as pytree leaves.
    """

    parent_day: Any
    counts: Any
    entry_logits: Any  # [h] susceptibility logits conditioned on the event-free run to as_of
    historical: Any  # [h]
    features: Any  # [horizon, pools, P]
    cure_features: Any  # [pools, Q]
    allowed: Any  # [horizon, pools]


def _exposed(ages, deadline):
    started = ages >= 0
    return started if deadline is None else started & (ages <= deadline)


def _pool_model(inputs, data=None, *, fit, draw, deadline, counted):
    """NumPyro Forecast program of one posterior draw's pools over the horizon."""
    h = Horizon.from_data(inputs.allowed, data)
    ages = jnp.arange(1, h.duration + 1)[:, None] - inputs.parent_day[None, :]
    timing, logits = fit.timing(
        TimingInputs(ages, inputs.features, inputs.cure_features), draw=draw
    )
    logits = logits.at[inputs.historical].set(inputs.entry_logits)
    kernel = survival_kernel(
        timing,
        logits,
        allowed=inputs.allowed,
        exposure=_exposed(ages, deadline) & (inputs.counts > 0)[None, :],
    )
    if counted:
        predict(
            h,
            lambda log_mass: CohortEventTime(log_mass, kernel.log_tail, inputs.counts),
            kernel.log_mass,
        )
    else:
        predict(
            h,
            lambda log_hazard: EventTime(kernel=kernel._replace(log_hazard=log_hazard)),
            kernel.log_hazard,
        )


def _draw_forecast(key, draw, fit, history, unit, bulk, *, deadline, num_cohorts):
    """One posterior draw: native allocations of every pool, reduced to cohorts and stocks."""
    entry_logits = _window_kernel(fit, draw, history).susceptibility_logits
    events = pending = eligible = None
    undefined = jnp.asarray(False)
    for partition, counted, subkey in zip(
        (unit, bulk), (False, True), random.split(key, 2), strict=True
    ):
        if partition is None:
            continue
        layout, parent_day, counts = partition
        horizon, pools = layout.allowed.shape
        inputs = _PoolInputs(
            parent_day,
            counts,
            entry_logits[layout.entry_rows],
            layout.historical,
            layout.features,
            layout.cure_features,
            layout.allowed,
        )
        program = functools.partial(
            _pool_model, fit=fit, draw=draw, deadline=deadline, counted=counted
        )
        predictive = Predictive(program, num_samples=1, return_sites=["forecast"], parallel=True)
        allocation = predictive(subkey, inputs, jnp.zeros((0, pools), counts.dtype))["forecast"][0]
        undefined = undefined | jnp.any(allocation < 0)
        ages = jnp.arange(1, horizon + 1)[:, None] - parent_day[None, :]
        remaining = counts[None, :] - jnp.cumsum(allocation, axis=0)
        by_cohort = jax.ops.segment_sum(allocation.T, layout.cohort, num_segments=num_cohorts).T
        started = jnp.sum(jnp.where(ages >= 0, remaining, 0), axis=1)
        within = jnp.sum(jnp.where(_exposed(ages, deadline), remaining, 0), axis=1)
        events = by_cohort if events is None else events + by_cohort
        pending = started if pending is None else pending + started
        eligible = within if eligible is None else eligible + within
    return events, pending, eligible, undefined


@functools.lru_cache(maxsize=None)
def _forecast_block(deadline, num_cohorts, has_unit, has_bulk):
    """Jitted, draw-mapped forecast of one static pool configuration.

    A single-draw fit pairs with every arrival/count draw, and a single
    arrival draw with every posterior draw: the fit row of draw ``i`` is
    ``i % fit.draws`` and the pool row ``i`` is gathered from the stacked
    per-draw metadata (already broadcast to the aligned draw count).
    """
    per_draw = functools.partial(_draw_forecast, deadline=deadline, num_cohorts=num_cohorts)
    mapped = _Partition(None, 0, 0)
    axes = (0, 0, None, None, mapped if has_unit else None, mapped if has_bulk else None)
    vmapped = jax.vmap(per_draw, in_axes=axes)

    def rows(partition, indices):
        if partition is None:
            return None
        return partition._replace(
            parent_day=partition.parent_day[indices], counts=partition.counts[indices]
        )

    def block(key, indices, fit, history, unit, bulk):
        keys = random.split(key, indices.shape[0])
        draws = indices % fit.draws
        return vmapped(keys, draws, fit, history, rows(unit, indices), rows(bulk, indices))

    return jax.jit(block)


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
    sizes = occupied.sum(axis=1)
    starts = np.concatenate((np.arange(single.size), single.size + np.cumsum(sizes) - sizes))
    initial_pool = np.full(cohorts, -1, dtype=np.int64)
    initial_pool[np.concatenate((single, multiple))] = starts

    counts = np.empty((draws, pool_cohorts.size), dtype=np.int64)
    origins = np.empty_like(counts)
    counts[:, : single.size] = quantities[:, single]
    origins[:, : single.size] = parent_days[:, single]
    multi_counts = counts[:, single.size :]
    multi_counts[:] = arrivals[:, days - 1, multiple[groups]]
    multi_counts[:, days == 0] = 1
    origins[:, single.size :] = np.where(days == 0, -age0[multiple[groups]], days)
    return counts, origins, pool_cohorts, initial_pool


def _partition(index, counts, parent_days, pool_cohorts, inputs, future_idx):
    """Host-packed pool partition over the forecast days, or ``None`` when empty."""
    if index.size == 0:
        return None
    cohorts = pool_cohorts[index]
    layout = _PoolLayout(
        features=inputs.features[cohorts[:, None], future_idx[None, :]].transpose(1, 0, 2),
        cure_features=inputs.cure_features[cohorts],
        allowed=inputs.allowed[cohorts[:, None], future_idx[None, :]].T,
        cohort=cohorts,
        historical=np.zeros(0, dtype=np.int64),
        entry_rows=np.zeros(0, dtype=np.int64),
    )
    return _Partition(layout, parent_days[:, index], counts[:, index])


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
    """Forecast one calendar-day parent -> child event propagation.

    ``parameters`` is a ``StageParameters`` (one draw or a leading draw axis)
    or a ``StageFit`` of either family. ``origins``/``observed`` are ``[C]``
    dates: each cohort's known historical immediate-parent event and its own
    event, ``NaT`` for unknown/future. A cohort with a known origin
    ``<= as_of`` and no known own event starts with exactly one pending unit
    (its actual age at ``as_of``); a cohort whose own event is already
    observed never produces it again. A cohort with unknown origin (``NaT``)
    starts with zero pending units -- it can only enter later through
    ``arrivals`` (e.g. a future sales cohort, whose "origin" is simply the
    day its arrival lands at age zero).

    ``arrivals`` is ``[D or 1, horizon, C]``: the exact realized count of
    *new* parent events for each cohort on each forecast day (e.g. a root
    sale cohort's future dated sale quantities, or -- for a child stage --
    the parent stage's own ``EventForecast.events`` output, used unchanged).
    Every arriving unit begins its own age-0 clock that day, so a parent
    event and its child's own trial may both happen the same calendar day.

    See the module docstring for the native allocation driver and the exact
    ``pending``/``eligible`` semantics on :class:`EventForecast`.
    """
    if horizon < 1:
        raise ValueError("horizon must be a positive number of days")
    if deadline_days is not None:
        if not isinstance(deadline_days, (int, np.integer)) or isinstance(deadline_days, bool):
            raise ValueError("deadline_days must be an integer or None")
        if deadline_days < 0:
            raise ValueError("deadline_days must be nonnegative")
        deadline_days = int(deadline_days)

    stage = _resolve_stage(parameters, "parameters")

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
        deadline_date = origins + np.timedelta64(deadline_days, "D")
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

    d = _align_draws(stage.draws, d_arrivals)
    arrivals_arr = np.broadcast_to(arrivals_arr, (d, horizon, n_cohorts))

    earliest_origin = origins[origin_known].min() if np.any(origin_known) else as_of
    cal = _validate_calendar(calendar, as_of, horizon, earliest_origin, as_of)
    calendar_len = int(cal.size)
    asof_idx = int(_elapsed(cal[0], as_of))
    inputs = _stage_inputs(stage, features, cure_features, allowed, n_cohorts, calendar_len)

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

    counts, parent_days, pool_cohorts, initial_pool = _arrival_pools(
        arrivals_arr, initial_pending_mask, age0
    )
    if pool_cohorts.size == 0:
        zeros = np.zeros((d, horizon), dtype=np.int64)
        return EventForecast(
            events=np.zeros((d, horizon, n_cohorts), dtype=np.int64),
            pending=zeros,
            eligible=zeros.copy(),
        )

    # Historical pools are one unit in every draw; their history window conditions entry.
    positions = np.flatnonzero(initial_pending_mask)
    known_through = age0[positions]
    if deadline_days is not None:
        known_through = np.minimum(known_through, deadline_days)
    history = _age_window(
        inputs,
        positions,
        _elapsed(cal[0], origins[positions]),
        int(known_through.max()) + 1 if positions.size else 0,
        known_through=known_through,
        exposed_through=np.full(positions.size, -1, dtype=np.int64),
    )

    one_unit = np.all(counts <= 1, axis=0)
    unit_index = np.flatnonzero(one_unit)
    bulk_index = np.flatnonzero(~one_unit)
    future_idx = asof_idx + 1 + np.arange(horizon)
    packed = (counts, parent_days, pool_cohorts, inputs, future_idx)
    unit = _partition(unit_index, *packed)
    bulk = _partition(bulk_index, *packed)
    if positions.size:
        unit_position = np.full(pool_cohorts.size, -1, dtype=np.int64)
        unit_position[unit_index] = np.arange(unit_index.size)
        historical = unit_position[initial_pool[positions]]
        assert np.all(historical >= 0), "historical pools are unit pools"
        unit = unit._replace(
            layout=unit.layout._replace(
                historical=historical, entry_rows=np.arange(positions.size, dtype=np.int64)
            )
        )

    cells = horizon * pool_cohorts.size + history.allowed.size
    with jax.enable_x64(True):
        block = _forecast_block(deadline_days, n_cohorts, unit is not None, bulk is not None)
        events, pending, eligible, undefined = _map_draws(
            block, d, cells, _root_key(seed), _x64(stage.fit), _x64(history), _x64(unit), _x64(bulk)
        )
    if undefined.any():
        raise ValueError(
            "forecast_events reached an impossible or degenerate conditional history in draws "
            f"{np.flatnonzero(undefined)[:5].tolist()} (possibly more) -- an infinite hazard or "
            "susceptibility logit combined with a fully decayed survival is rejected rather than "
            "silently treated as certain cure"
        )
    return EventForecast(
        events=events.astype(np.int64),
        pending=pending.astype(np.int64),
        eligible=eligible.astype(np.int64),
    )


# --------------------------------------------------------------------------
# Eventual-receipt expectations
# --------------------------------------------------------------------------


def _window_end(tail, deadline, stage):
    """Last age at which ``stage`` can still fire, or ``None`` for an unbounded proper tail."""
    bounds = [deadline] if deadline is not None else []
    if isinstance(tail, FiniteTail):
        bounds.append(int(tail.last_age))
    if bounds:
        return int(min(bounds))
    if isinstance(tail, ProperTail):
        return None
    raise ValueError(
        f"the {stage} family declares an unknown tail, so its eventual {stage} probability "
        "is undefined: declare a proper continuation or finite support, or bound the stage "
        "with a deadline, before requesting eventual expectations"
    )


class _Stream(NamedTuple):
    """Per-row receipt regressors for the initiation-day contraction of a finite receipt law."""

    origin_idx: Any  # [N] calendar index of each row's sale
    features: Any  # [T, N, P] calendar-indexed receipt regressors of the rows
    allowed: Any  # [T, N]
    cure_features: Any  # [N, Q]
    first_future: Any  # calendar index of the first day after as_of


def _completion(fit, draw, stream, start, span):
    """Receipt probability of fresh units initiating on calendar index ``start``.

    The finite support ``0..span-1`` follows the calendar's closures and
    regressors, continuing all-open with the last regressors held past the
    calendar end.
    """
    n = stream.origin_idx.shape[0]
    calendar_len = stream.allowed.shape[0]
    ages = jnp.broadcast_to(jnp.arange(span)[:, None], (span, n))
    calendar_day = start + ages
    index = jnp.clip(calendar_day, 0, calendar_len - 1)
    columns = jnp.arange(n)[None, :]
    allowed = jnp.where(calendar_day < calendar_len, stream.allowed[index, columns], True)
    timing, logits = fit.timing(
        TimingInputs(ages, stream.features[index, columns], stream.cure_features), draw=draw
    )
    kernel = survival_kernel(timing, logits, allowed=allowed, exposure=jnp.ones_like(allowed))
    return _window_probability(kernel, eventual=False)


def _draw_expectations(
    init_draw,
    recv_draw,
    init_fit,
    recv_fit,
    open_window,
    elig_window,
    recv_probe,
    stream,
    stationary,
    *,
    init_bounded,
    recv_span,
    future_days,
):
    """One posterior draw's eventual receipts from open and uninitiated rows."""
    open_receipts = jnp.float64(0.0)
    if open_window is not None:
        kernel = _window_kernel(recv_fit, recv_draw, open_window)
        open_receipts = _window_probability(kernel, eventual=recv_span is None).sum()
    uninitiated = jnp.float64(0.0)
    if elig_window is not None:
        kernel = _window_kernel(init_fit, init_draw, elig_window)
        if recv_span is None:
            # Proper receipt: every initiated susceptible unit eventually arrives.
            initiates = _window_probability(kernel, eventual=not init_bounded)
            _, recv_logits = recv_fit.timing(
                TimingInputs(recv_probe.ages, recv_probe.features, recv_probe.cure_features),
                draw=recv_draw,
            )
            uninitiated = jnp.sum(initiates * sigmoid(recv_logits))
        else:
            log_mass = kernel.log_mass  # [A, N] initiation mass by age
            span = log_mass.shape[0]

            def day(total, offset):
                start = stream.first_future + offset
                age = start - stream.origin_idx
                row_mass = jnp.take_along_axis(
                    log_mass, jnp.clip(age, 0, span - 1)[None, :], axis=0
                )[0]
                mass = jnp.where(age < span, jnp.exp(row_mass), 0.0)
                completion = _completion(recv_fit, recv_draw, stream, start, recv_span)
                return total + jnp.sum(mass * completion), None

            uninitiated, _ = lax.scan(day, jnp.float64(0.0), jnp.arange(future_days))
            if not init_bounded:
                # Proper initiation mass left after the calendar completes under the
                # stationary continuation of the same receipt family.
                remaining = jnp.exp(
                    kernel.log_susceptible + jnp.sum(kernel.log_survival_step, axis=-2)
                )
                completion = _window_probability(
                    _window_kernel(recv_fit, recv_draw, stationary), eventual=False
                )
                uninitiated = uninitiated + jnp.sum(remaining * completion)
    return uninitiated, open_receipts


@functools.lru_cache(maxsize=None)
def _expectation_block(init_bounded, recv_span, future_days):
    """Jitted, draw-mapped expectations; single-draw fits pair with every draw."""
    per_draw = functools.partial(
        _draw_expectations,
        init_bounded=init_bounded,
        recv_span=recv_span,
        future_days=future_days,
    )
    vmapped = jax.vmap(per_draw, in_axes=(0, 0) + (None,) * 7)

    def block(key, indices, init_fit, recv_fit, *windows):
        del key
        return vmapped(
            indices % init_fit.draws, indices % recv_fit.draws, init_fit, recv_fit, *windows
        )

    return jax.jit(block)


class _HistoryRows(NamedTuple):
    """Date arrays and row classes of a ``RetailHistory`` frame."""

    sale: Any  # [N] datetime64[D]
    initiation: Any  # [N] datetime64[D], NaT when uninitiated
    receipt: Any  # [N] datetime64[D], NaT when not received
    eligible: Any  # [n_elig] positions of uninitiated rows still within the policy window
    open: Any  # [n_open] positions of initiated rows awaiting receipt
    item_ids: Any


def _history_rows(history):
    frame = history.frame.reset_index(drop=True)
    if len(frame):
        sale = np.asarray(to_day(frame["sale_date"].to_numpy()))
        initiation = np.asarray(to_day(frame["initiation_date"].to_numpy()))
        receipt = np.asarray(to_day(frame["receipt_date"].to_numpy()))
        eligible = frame["eligible"].to_numpy(dtype=bool)
        item_ids = frame["item_id"].to_numpy()
    else:
        sale = initiation = receipt = np.zeros(0, dtype="datetime64[D]")
        eligible = np.zeros(0, dtype=bool)
        item_ids = np.array([])
    return _HistoryRows(
        sale,
        initiation,
        receipt,
        np.flatnonzero(eligible),
        np.flatnonzero(~np.isnat(initiation) & np.isnat(receipt)),
        item_ids,
    )


class _ExpectationWindows(NamedTuple):
    """Host-packed inputs of ``_draw_expectations`` (absent groups are ``None``)."""

    open: _AgeWindow | None
    eligible: _AgeWindow | None
    probe: _AgeWindow | None  # fresh-unit receipt logits of eligible rows (proper receipt)
    stream: _Stream | None  # finite receipt contraction over initiation days
    stationary: _AgeWindow | None  # finite receipt completion past the calendar


def _open_window(recv, rows, origin_idx, age0, recv_end):
    """Open returns: known event-free through ``age0``, exposed through the receipt support."""
    through = np.full(age0.shape, -1 if recv_end is None else recv_end, dtype=np.int64)
    span = int(max(age0.max(), through.max())) + 1
    return _age_window(recv, rows, origin_idx, span, known_through=age0, exposed_through=through)


def _eligible_windows(init, recv, rows, origin_idx, age0, init_end, recv_end, asof_idx):
    """Uninitiated rows: the initiation window plus the receipt completion inputs."""
    calendar_len = init.allowed.shape[1]
    n = rows.shape[0]
    none = np.full(n, -1, dtype=np.int64)
    if init_end is not None:
        through = np.full(n, init_end, dtype=np.int64)
    elif recv_end is None:
        through = none  # proper initiation and receipt: eventual = conditional susceptibility
    else:
        through = calendar_len - 1 - origin_idx  # proper initiation streamed through the calendar
    span = int(max(age0.max(), through.max())) + 1
    eligible = _age_window(
        init, rows, origin_idx, span, known_through=age0, exposed_through=through
    )
    if recv_end is None:
        probe = _age_window(
            recv,
            rows,
            np.full(n, asof_idx, dtype=np.int64),
            1,
            known_through=none,
            exposed_through=none,
        )
        return eligible, probe, None, None, 0
    stream = _Stream(
        origin_idx=origin_idx,
        features=recv.features[rows].transpose(1, 0, 2),
        allowed=recv.allowed[rows].T,
        cure_features=recv.cure_features[rows],
        first_future=np.int64(asof_idx + 1),
    )
    stationary = _age_window(
        recv,
        rows,
        np.full(n, calendar_len, dtype=np.int64),
        recv_end + 1,
        known_through=none,
        exposed_through=np.full(n, recv_end, dtype=np.int64),
    )
    latest = calendar_len - 1 if init_end is None else int((origin_idx + init_end).max())
    return eligible, None, stream, stationary, max(0, min(latest, calendar_len - 1) - asof_idx)


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
    """Exact eventual-receipt expectations for the existing historical population.

    Returns ``(expected_uninitiated_receipts, expected_open_receipts)``,
    each ``[draw]``, under each family's declared tail behavior (see the
    module docstring). Open returns contribute their conditional receipt
    probability; uninitiated eligible rows contribute the probability of
    initiating within their window (the policy deadline and/or the
    initiation family's finite support) times the receipt probability of a
    fresh unit, which for a finite receipt law is contracted day by day over
    the actual initiation dates.

    ``history.policy_days`` may be ``None`` (unbounded initiation deadline):
    a proper initiation family then initiates eventually with its conditional
    susceptibility, and a finite one within its support; an unknown tail
    without a deadline is refused.
    """
    init_stage = _resolve_stage(initiation, "initiation")
    recv_stage = _resolve_stage(receipt, "receipt")
    d = _align_draws(init_stage.draws, recv_stage.draws)
    as_of = np.datetime64(history.as_of, "D")
    init_end = _window_end(init_stage.tail, history.policy_days, "initiation")
    recv_end = _window_end(recv_stage.tail, None, "receipt")

    rows = _history_rows(history)
    n_hist = rows.sale.shape[0]
    elig_sale_date = rows.sale[rows.eligible]
    open_init_date = rows.initiation[rows.open]
    latest_required = _required_calendar_end(elig_sale_date, init_end, as_of)
    earliest_origin = rows.sale.min() if n_hist else as_of
    cal = _validate_calendar(calendar, as_of, 0, earliest_origin, latest_required)
    calendar_len = int(cal.size)
    asof_idx = int(_elapsed(cal[0], as_of))
    init = _stage_inputs(
        init_stage,
        initiation_features,
        initiation_cure_features,
        initiation_allowed,
        n_hist,
        calendar_len,
        prefix="initiation_",
    )
    recv = _stage_inputs(
        recv_stage,
        receipt_features,
        receipt_cure_features,
        receipt_allowed,
        n_hist,
        calendar_len,
        prefix="receipt_",
    )

    windows = _ExpectationWindows(None, None, None, None, None)
    cells = future_days = 0
    if rows.open.size:
        open_window = _open_window(
            recv,
            rows.open,
            _elapsed(cal[0], open_init_date),
            _elapsed(open_init_date, as_of),
            recv_end,
        )
        windows = windows._replace(open=open_window)
        cells += open_window.allowed.size
    if rows.eligible.size:
        eligible, probe, stream, stationary, future_days = _eligible_windows(
            init,
            recv,
            rows.eligible,
            _elapsed(cal[0], elig_sale_date),
            _elapsed(elig_sale_date, as_of),
            init_end,
            recv_end,
            asof_idx,
        )
        windows = windows._replace(
            eligible=eligible, probe=probe, stream=stream, stationary=stationary
        )
        cells += eligible.allowed.size
        cells += 0 if recv_end is None else (recv_end + 1) * rows.eligible.size
    if cells == 0:
        return np.zeros(d), np.zeros(d)

    with jax.enable_x64(True):
        recv_span = None if recv_end is None else recv_end + 1
        block = _expectation_block(init_end is not None, recv_span, future_days)
        uninitiated, open_receipts = _map_draws(
            block,
            d,
            cells,
            random.key(0),
            _x64(init_stage.fit),
            _x64(recv_stage.fit),
            *_x64(windows),
        )
    for name, values in (("initiation", uninitiated), ("receipt", open_receipts)):
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"{name} as-of conditioning produced a nonfinite expectation -- an impossible or "
                "degenerate conditional history (an infinite hazard/cure logit combined with a "
                "fully decayed survival) is rejected rather than silently treated as certain cure"
            )
    return np.asarray(uninitiated, dtype=np.float64), np.asarray(open_receipts, dtype=np.float64)


def _required_calendar_end(elig_sale_date, init_end, as_of):
    """Latest calendar day the uninitiated rows' bounded initiation windows reach."""
    if init_end is None or elig_sale_date.size == 0:
        return as_of
    return (elig_sale_date + np.timedelta64(init_end, "D")).max()


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
    """Forecast calendar-day return initiations/receipts from ``history.as_of``.

    A thin two-event composition of :func:`forecast_events`: the initiation
    call's ``events`` output feeds unchanged as the receipt call's
    ``arrivals``. ``initiation``/``receipt`` are ``StageParameters`` or
    ``StageFit`` values. ``future_sales`` is a
    :class:`~ttenet.integration.SalesForecast`, or a plain DataFrame parsed
    via ``SalesForecast.from_frame`` (requiring ``sale_date``/``quantity``
    columns). See the module docstring for the sequential-simulation
    mechanics and the tail assumptions behind ``expected_*_receipts``.
    """
    if horizon < 1:
        raise ValueError("horizon must be a positive number of days")

    init_stage = _resolve_stage(initiation, "initiation")
    recv_stage = _resolve_stage(receipt, "receipt")
    as_of = np.datetime64(history.as_of, "D")
    policy_days = history.policy_days
    init_end = _window_end(init_stage.tail, policy_days, "initiation")
    _window_end(recv_stage.tail, None, "receipt")

    rows = _history_rows(history)
    n_hist = rows.sale.shape[0]
    sales_forecast = _prepare_sales_forecast(future_sales)
    future_frame, sale_dates_future, counts_array = _future_cohorts_from_sales(
        sales_forecast, rows.item_ids, as_of, horizon
    )
    n_future = len(future_frame)
    d = _align_draws(init_stage.draws, recv_stage.draws, counts_array.shape[0])
    counts_array = np.broadcast_to(counts_array, (d, n_future)).astype(np.int64)
    n_total = n_hist + n_future

    # -- calendar coverage (historical eligible windows only, per the retail contract:
    # future cohorts' own deadlines are not required, since forecast_events needs no
    # calendar beyond as_of + horizon) --------------------------------------------------
    latest_required = _required_calendar_end(rows.sale[rows.eligible], init_end, as_of)
    earliest_origin = rows.sale.min() if n_hist else as_of
    cal = _validate_calendar(calendar, as_of, horizon, earliest_origin, latest_required)
    calendar_len = int(cal.size)
    init = _stage_inputs(
        init_stage,
        initiation_features,
        initiation_cure_features,
        initiation_allowed,
        n_total,
        calendar_len,
        prefix="initiation_",
    )
    recv = _stage_inputs(
        recv_stage,
        receipt_features,
        receipt_cure_features,
        receipt_allowed,
        n_total,
        calendar_len,
        prefix="receipt_",
    )

    output_dates = np.asarray(
        date_grid(as_of + np.timedelta64(1, "D"), as_of + np.timedelta64(horizon, "D"))
    )

    future_nat = np.full(n_future, np.datetime64("NaT", "D"), dtype="datetime64[D]")

    # -- initiation call: parent = sale, own event = initiation --------------------------
    init_arrivals = np.zeros((d, horizon, n_total), dtype=np.int64)
    if n_future:
        future_day = _elapsed(as_of, sale_dates_future)  # 1..horizon, validated above
        future_cols = n_hist + np.arange(n_future)
        init_arrivals[:, future_day - 1, future_cols] = counts_array

    seed_seq = np.random.SeedSequence(seed)
    init_seed, recv_seed = seed_seq.spawn(2)

    init_result = forecast_events(
        init_stage.fit,
        origins=np.concatenate([rows.sale, future_nat]),
        observed=np.concatenate([rows.initiation, future_nat]),
        arrivals=init_arrivals,
        calendar=cal,
        as_of=as_of,
        horizon=horizon,
        features=init.features,
        cure_features=init.cure_features,
        allowed=init.allowed,
        deadline_days=policy_days,
        seed=init_seed,
    )

    # -- receipt call: parent = initiation, own event = receipt --------------------------
    # A child uses the upstream EventForecast.events unchanged as its arrivals, so every
    # initiated unit (historical or future, on whatever day it actually initiates)
    # immediately keeps its own independently correct receipt-age clock.
    recv_result = forecast_events(
        recv_stage.fit,
        origins=np.concatenate([rows.initiation, future_nat]),
        observed=np.concatenate([rows.receipt, future_nat]),
        arrivals=init_result.events,
        calendar=cal,
        as_of=as_of,
        horizon=horizon,
        features=recv.features,
        cure_features=recv.cure_features,
        allowed=recv.allowed,
        deadline_days=None,  # retail receipts have no imposed deadline
        seed=recv_seed,
    )

    initiations = init_result.events.sum(axis=-1)
    eligible = init_result.eligible
    receipts = recv_result.events.sum(axis=-1)
    open_returns = recv_result.pending

    expected_uninitiated_receipts, expected_open_receipts = expected_return_receipts(
        history,
        init_stage.fit,
        recv_stage.fit,
        calendar=cal,
        initiation_features=init.features[:n_hist],
        receipt_features=recv.features[:n_hist],
        initiation_cure_features=init.cure_features[:n_hist],
        receipt_cure_features=recv.cure_features[:n_hist],
        initiation_allowed=init.allowed[:n_hist],
        receipt_allowed=recv.allowed[:n_hist],
    )
    expected_uninitiated_receipts = np.broadcast_to(expected_uninitiated_receipts, (d,))
    expected_open_receipts = np.broadcast_to(expected_open_receipts, (d,))
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
