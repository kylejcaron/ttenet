"""Discrete-time mixture-cure stage observations, native observation, and fitting.

Host-side preparation (`make_observations`, `make_event_observations`) is
plain NumPy/pandas and runs before inference: it converts a history's dates
into calendar-indexed arrays and validates coverage, so it never touches
JAX tracers. Everything downstream is ordinary differentiable JAX/NumPyro:
a timing family (the default `StageParameters` adapter or a custom
`event_time_model`) produces a `TimingLaw`, the shared `event_times` kernel
masks it by the stage's administrative exposure and closures and conditions
entry, and the native `EventTime` distribution registers every unit's
observed trajectory through NumPyro Forecast's `Horizon`/`predict`. The
public `stage_log_likelihood` scores the same kernel per unit.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random, tree_util
from jax.tree_util import register_dataclass
from numpyro import handlers
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro_forecast import Horizon, predict

from .dates import to_day
from .distributions import EventTime
from .event_times import (
    SurvivalKernel,
    TimingInputs,
    TimingLaw,
    default_timing,
    kernel_unit_log_prob,
    survival_kernel,
)
from .processes import tail_behavior
from .survival import StageParameters

_STAGES = ("initiation", "receipt")


@partial(
    register_dataclass,
    data_fields=[
        "ages",
        "features",
        "cure_features",
        "at_risk",
        "allowed",
        "event_index",
        "pre_entry",
        "exposure",
    ],
    meta_fields=[],
)
@dataclasses.dataclass(frozen=True)
class StageObservations:
    """Calendar-indexed, likelihood-ready observations for one stage.

    ``ages: [N, T]`` -- elapsed days since the unit's clock origin at each
    calendar day (sale date for ``"initiation"``, initiation date for
    ``"receipt"``); negative before the clock starts.
    ``features: [N, T, P]`` -- time-varying hazard regressors, calendar-day
    indexed.
    ``cure_features: [N, Q]`` -- static susceptibility regressors.
    ``at_risk: [N, T]`` -- observed exposure window: ``True`` for every
    calendar day the unit was under observation *from its entry date
    inclusive* through and including its event or censoring day, clipped at
    ``as_of`` (and at the policy deadline for uninitiated ``"initiation"``
    rows). Distinct from ``allowed``: a closed day can still be "at risk"
    (observed), it simply carries zero hazard.
    ``allowed: [N, T]`` -- hard calendar closures (e.g. weekends), zero
    hazard regardless of the learned baseline/regressors.
    ``event_index: [N]`` -- calendar index of the observed event day, or
    ``-1`` if the unit was censored (no event observed by ``as_of``).
    ``pre_entry: [N, T] | None`` -- known event-free exposure *before* a
    unit's snapshot entry date, clipped at the policy deadline; ``None``
    (the default) means every unit's history is fully observed from its
    clock origin, which is exactly the ordinary (non-conditional-entry)
    likelihood. When supplied, the cumulative survival over these days
    shifts the susceptibility logit at entry, so a unit's already-known
    event-free run before it entered the observed snapshot still informs
    its posterior cure probability without re-litigating a likelihood
    contribution for those days directly.
    ``exposure: [N, T] | None`` -- administrative exposure of the generative
    law: ``True`` from the unit's entry (``max(origin, entry)``) through
    ``min(deadline, as_of)`` whether or not its event was observed earlier.
    It never encodes the observed stopping time, so a posterior predictive
    draw may fire a unit on a later date than its recorded event. Builders
    always populate it; ``None`` (hand-built observations) infers it from
    ``at_risk``: a censored unit's at-risk days, and an event unit's at-risk
    days plus every later date, which leaves the observed likelihood exactly
    as ``at_risk`` describes it (an event outside ``at_risk`` is impossible).
    Negative ages are never exposed, whichever route supplies the mask.

    Registered as a JAX pytree (all fields are array data, no static
    metadata) so instances can be passed directly as NumPyro model/SVI
    arguments without breaking JIT.
    """

    ages: Any
    features: Any
    cure_features: Any
    at_risk: Any
    allowed: Any
    event_index: Any
    pre_entry: Any = None
    exposure: Any = None


_DRAW_AXES = {"age_logits": 2, "beta": 2, "cure_intercept": 1, "cure_beta": 2}


def _fit_parameter_leaves(parameters, event_time_model):
    """Validate the family-specific posterior container without copying its arrays."""
    if event_time_model is None:
        if not isinstance(parameters, StageParameters):
            raise TypeError(
                "a default-family StageFit carries StageParameters; a custom family "
                "needs its event_time_model alongside its named posterior mapping"
            )
        for name, value in zip(StageParameters._fields, parameters, strict=True):
            if jnp.ndim(value) != _DRAW_AXES[name]:
                raise ValueError(f"StageFit.parameters.{name} must carry a leading draw axis")
        return list(parameters)
    if not callable(event_time_model):
        raise TypeError("event_time_model must be callable")
    if not isinstance(parameters, Mapping) or any(not isinstance(name, str) for name in parameters):
        raise TypeError(
            "a custom-family StageFit carries a mapping from the factory's site "
            "names to posterior arrays"
        )
    return tree_util.tree_leaves(dict(parameters))


@partial(
    register_dataclass,
    data_fields=["parameters", "losses", "shared"],
    meta_fields=["event_time_model", "num_samples"],
)
@dataclasses.dataclass(frozen=True)
class StageFit:
    """Fit result: posterior draws of one stage's timing family and the loss trace.

    For the default family ``parameters`` is a ``StageParameters`` whose
    every field carries a leading posterior-draw axis and
    ``event_time_model`` is ``None``. For a custom family ``event_time_model``
    is the fitted factory and ``parameters`` maps the factory's own sample
    sites and optimized ``numpyro.param`` values (node scope stripped) to
    arrays with the same leading draw axis; ``shared`` is the shared model's
    returned mapping resolved at those same draws (``None`` without one), so
    replaying the factory never re-samples shared values. ``num_samples``
    records the fit's actual draw count; it is what lets a fully fixed
    factory with no arrays at all replay the requested number of draws. The
    two-argument form ``StageFit(parameters, losses)`` remains valid for the
    default family and takes its draw count from the arrays. ``losses`` is
    the per-step ELBO trace from ``SVI.run`` (empty when nothing was fitted).

    ``timing`` replays one draw's law at new ages and regressors; the draw
    index may be traced, so ``jax.vmap`` over ``jnp.arange(fit.draws)``
    evaluates every draw.
    """

    parameters: Any
    losses: Any
    event_time_model: Callable | None = None
    shared: Any = None
    num_samples: int | None = None

    def __post_init__(self):
        leaves = _fit_parameter_leaves(self.parameters, self.event_time_model)
        leaves += tree_util.tree_leaves(self.shared)
        if any(isinstance(leaf, jax.core.Tracer) for leaf in leaves):
            return
        if self.num_samples is not None and (
            isinstance(self.num_samples, bool)
            or not isinstance(self.num_samples, (int, np.integer))
            or self.num_samples < 1
        ):
            raise ValueError("num_samples must be a positive integer or None")
        sizes = {int(np.shape(leaf)[0]) for leaf in leaves}
        if len(sizes) > 1:
            raise ValueError(f"posterior arrays disagree on their draw axis: {sorted(sizes)}")
        if self.num_samples is None:
            if not sizes:
                raise ValueError(
                    "a fit without posterior arrays must record num_samples, its actual draw count"
                )
        elif sizes and sizes != {int(self.num_samples)}:
            raise ValueError(
                f"num_samples={self.num_samples} disagrees with the posterior draw axis "
                f"{sizes.pop()}"
            )

    @property
    def draws(self) -> int:
        """Number of posterior draws this fit replays."""
        if self.num_samples is not None:
            return int(self.num_samples)
        if self.event_time_model is None:
            return int(np.shape(self.parameters.cure_intercept)[0])
        return int(np.shape(tree_util.tree_leaves((dict(self.parameters), self.shared))[0])[0])

    def timing(self, inputs: TimingInputs, *, draw, key=None) -> tuple[TimingLaw, Any]:
        """Timing law and susceptibility logits of posterior draw ``draw`` at ``inputs``.

        The default family evaluates ``default_timing`` on that draw's
        ``StageParameters``. A custom family reruns its factory with the
        draw's named values substituted at its sample and ``numpyro.param``
        sites and the draw's resolved shared values passed through; a factory
        that reaches a site absent from the fitted mapping is refused rather
        than silently sampled from its prior. ``key`` seeds that replay (every
        site is substituted, so it only matters for factories with other
        NumPyro randomness).
        """

        def select(leaf):
            return jnp.asarray(leaf)[draw]

        if self.event_time_model is None:
            return default_timing(tree_util.tree_map(select, self.parameters), inputs)
        parameters = tree_util.tree_map(select, dict(self.parameters))
        shared = tree_util.tree_map(select, self.shared)
        return _replay_family(self.event_time_model, parameters, shared, inputs, key)


def _select_rows(
    array: Any | None, row_mask: np.ndarray, total_rows: int, *, name: str
) -> np.ndarray | None:
    if array is None:
        return None
    array = np.asarray(array)
    required_ndim = {"features": 3, "cure_features": 2, "allowed": 2}[name]
    if array.ndim != required_ndim:
        raise ValueError(f"{name} must have {required_ndim} dimensions")
    if name != "allowed" and not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite numeric values")
    if array.shape[0] != total_rows:
        raise ValueError(
            f"{name} must have {total_rows} leading rows aligned to the full history frame, "
            f"got {array.shape[0]}"
        )
    return array[row_mask]


def _prepare_event_rows(
    frame: Any,
    *,
    origin_column: str,
    event_column: str,
    as_of: np.datetime64,
    deadline_days: int | None,
    entry_dates: Any | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """Validate dates and select rows contributing to a stage likelihood."""
    total_rows = len(frame)
    origin_full = np.asarray(to_day(frame[origin_column]))
    event_full = np.asarray(to_day(frame[event_column]))
    if origin_full.shape[0] != total_rows or event_full.shape[0] != total_rows:
        raise ValueError("origin_column and event_column must align with the full frame")
    if entry_dates is None:
        entry_full = origin_full.copy()
    else:
        entry_full = np.asarray(to_day(entry_dates))
        if entry_full.shape[0] != total_rows:
            raise ValueError("entry_dates must align with the full history frame")

    origin_known = ~np.isnat(origin_full)
    event_known = ~np.isnat(event_full)
    if np.any(event_known & ~origin_known):
        raise ValueError("an observed event cannot occur without a known parent/origin date")
    if np.any(event_known & origin_known & (event_full < origin_full)):
        raise ValueError("event date precedes its origin date")

    deadline_full = None
    if deadline_days is not None:
        deadline_full = origin_full + np.timedelta64(int(deadline_days), "D")
        if np.any(event_known & origin_known & (event_full > deadline_full)):
            raise ValueError(f"observed event exceeds the {deadline_days}-day deadline")
    if np.any(origin_known & np.isnat(entry_full)):
        raise ValueError("entry_dates must be known wherever origin_column is known")

    has_event_full = event_known & (event_full <= as_of)
    completed_before_entry = has_event_full & (event_full < entry_full)
    row_mask = origin_known & ~completed_before_entry
    return (
        origin_full[row_mask],
        event_full[row_mask],
        entry_full[row_mask],
        None if deadline_full is None else deadline_full[row_mask],
        has_event_full[row_mask],
        row_mask,
    )


def _validate_calendar(calendar: Any, origin: np.ndarray, as_of: np.datetime64) -> np.ndarray:
    """Normalize and validate the inclusive daily calendar."""
    calendar = np.asarray(calendar, dtype="datetime64[D]")
    if calendar.ndim != 1 or calendar.shape[0] == 0:
        raise ValueError("calendar must be a nonempty one-dimensional date array")
    if np.isnat(calendar).any():
        raise ValueError("calendar must not contain missing dates")
    if calendar.shape[0] > 1 and not np.all(
        np.diff(calendar).astype("timedelta64[D]").astype(np.int64) == 1
    ):
        raise ValueError("calendar must be an inclusive, contiguous daily grid")
    if origin.size > 0 and calendar[0] > origin.min():
        raise ValueError("calendar must start at or before the earliest original origin date")
    if calendar[-1] < as_of:
        raise ValueError("calendar must extend through as_of")
    return calendar


def _construct_exposure(
    origin: np.ndarray,
    event: np.ndarray,
    entry: np.ndarray,
    deadline: np.ndarray | None,
    has_event: np.ndarray,
    calendar: np.ndarray,
    as_of: np.datetime64,
    entry_dates_given: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Construct age, observed at-risk, administrative exposure, event-index and pre-entry masks.

    ``at_risk`` stops at the observed event; ``exposure`` is the generative
    law's window from entry through ``min(deadline, as_of)`` regardless of the
    event, so a calendar extending past ``as_of`` never adds exposure.
    """
    num_days = calendar.shape[0]
    calendar_start = calendar[0]
    stop_date = (
        np.minimum(deadline, as_of)
        if deadline is not None
        else np.full(origin.size, as_of, dtype="datetime64[D]")
    )
    end_date = np.where(has_event, event, stop_date)
    end_index = (end_date - calendar_start).astype("timedelta64[D]").astype(np.int64)
    stop_index = (stop_date - calendar_start).astype("timedelta64[D]").astype(np.int64)
    event_raw_index = (event - calendar_start).astype("timedelta64[D]").astype(np.int64)
    if np.any(has_event & ((event_raw_index < 0) | (event_raw_index >= num_days))):
        raise ValueError("calendar does not cover an observed event date")
    event_index = np.where(has_event, event_raw_index, -1).astype(np.int64)

    ages = (calendar[None, :] - origin[:, None]).astype("timedelta64[D]").astype(np.int64)
    day_index = np.arange(num_days)
    risk_origin = np.maximum(origin, entry)
    risk_origin_idx = (risk_origin - calendar_start).astype("timedelta64[D]").astype(np.int64)
    entered = day_index[None, :] >= risk_origin_idx[:, None]
    at_risk = entered & (day_index[None, :] <= end_index[:, None])
    exposure = entered & (day_index[None, :] <= stop_index[:, None])
    if not entry_dates_given:
        return ages, at_risk, exposure, event_index, None

    origin_idx = (origin - calendar_start).astype("timedelta64[D]").astype(np.int64)
    pre_entry_stop = entry - np.timedelta64(1, "D")
    if deadline is not None:
        pre_entry_stop = np.minimum(pre_entry_stop, deadline)
    pre_entry_stop_idx = (pre_entry_stop - calendar_start).astype("timedelta64[D]").astype(np.int64)
    pre_entry = (day_index[None, :] >= origin_idx[:, None]) & (
        day_index[None, :] <= pre_entry_stop_idx[:, None]
    )
    return ages, at_risk, exposure, event_index, pre_entry


def _observation_allowed(allowed, row_mask, total_rows, num_rows, num_days):
    """Normalize global or per-row closure masks onto the selected exposure grid."""
    if allowed is None:
        return np.ones((num_rows, num_days), dtype=bool)
    values = np.asarray(allowed)
    if not np.isin(values, [False, True]).all():
        raise ValueError("allowed must contain boolean or zero/one values")
    values = values.astype(bool)
    if values.ndim == 1:
        if values.shape[0] != num_days:
            raise ValueError("a 1-D allowed mask must match the calendar length")
        return np.broadcast_to(values, (num_rows, num_days))
    selected = _select_rows(values, row_mask, total_rows, name="allowed")
    if selected.shape[-1] != num_days:
        raise ValueError("allowed must be indexed against the exact calendar length")
    return selected


def _normalize_observation_arrays(
    features: Any | None,
    cure_features: Any | None,
    allowed: Any | None,
    row_mask: np.ndarray,
    total_rows: int,
    num_days: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align covariates and closure masks after stage rows are selected."""
    num_rows = int(row_mask.sum())
    features_full = _select_rows(features, row_mask, total_rows, name="features")
    if features_full is None:
        features_arr = np.zeros((num_rows, num_days, 0))
    else:
        if features_full.shape[1] != num_days:
            raise ValueError("features must be indexed against the exact calendar length")
        features_arr = features_full

    cure_full = _select_rows(cure_features, row_mask, total_rows, name="cure_features")
    cure_arr = np.zeros((num_rows, 0)) if cure_full is None else cure_full
    allowed_arr = _observation_allowed(allowed, row_mask, total_rows, num_rows, num_days)
    return features_arr, cure_arr, allowed_arr


def make_event_observations(
    frame: Any,
    *,
    origin_column: str,
    event_column: str,
    as_of: Any,
    calendar: Any,
    deadline_days: int | None = None,
    entry_dates: Any | None = None,
    features: Any | None = None,
    cure_features: Any | None = None,
    allowed: Any | None = None,
) -> StageObservations:
    """Build calendar-indexed :class:`StageObservations` for one generic event stage.

    Generalizes the sale->initiation->receipt stage clock to an arbitrary
    parent/child event pair: ``origin_column`` is the immediate-parent event
    (e.g. sale date), ``event_column`` is this stage's own event (e.g.
    initiation date). ``deadline_days`` is an optional integer offset from
    origin beyond which an unobserved own-event stops accruing exposure
    (``None`` means unbounded, e.g. receipt).

    ``entry_dates`` (aligned to the full ``frame``, one date per row)
    supports conditional-entry snapshots: a row's clock origin stays at
    ``origin_column`` (so its age and pre-entry survival are correct), but
    its *observed* exposure (``at_risk``) only starts at
    ``max(origin, entry)``, while known event-free exposure from origin
    through the day before entry (clipped at the deadline) is recorded in
    ``pre_entry`` for :func:`stage_log_likelihood` to condition on. ``None``
    (the default) means every row's entry equals its origin -- an ordinary
    complete-origin history, with ``pre_entry`` left ``None`` on the result.
    A row whose own event is already observed *before* its entry date
    contributes no new information given that entry and is excluded
    entirely (not zeroed out).

    ``features`` (``[frame rows, len(calendar), P]``), ``cure_features``
    (``[frame rows, Q]``), and ``allowed`` (``[len(calendar)]`` or
    ``[frame rows, len(calendar)]``) are indexed against the *full* frame
    row order, matching every row -- including rows this stage will
    ultimately drop (unknown origin, or an event completed before entry) --
    before the stage's row subset is selected, so alignment never shifts
    under a caller's feature array. ``calendar`` must be an inclusive,
    contiguous daily grid that starts at or before the earliest original
    origin date (pre-entry susceptibility needs that historical exposure)
    and extends through ``as_of``.
    """
    as_of_day = to_day(as_of)
    total_rows = len(frame)
    origin, event, entry, deadline, has_event, row_mask = _prepare_event_rows(
        frame,
        origin_column=origin_column,
        event_column=event_column,
        as_of=as_of_day,
        deadline_days=deadline_days,
        entry_dates=entry_dates,
    )
    calendar_arr = _validate_calendar(calendar, origin, as_of_day)
    ages, at_risk, exposure, event_index, pre_entry = _construct_exposure(
        origin,
        event,
        entry,
        deadline,
        has_event,
        calendar_arr,
        as_of_day,
        entry_dates is not None,
    )
    features_arr, cure_arr, allowed_arr = _normalize_observation_arrays(
        features, cure_features, allowed, row_mask, total_rows, calendar_arr.shape[0]
    )
    return StageObservations(
        ages=jnp.asarray(ages),
        features=jnp.asarray(features_arr, dtype=jnp.float32),
        cure_features=jnp.asarray(cure_arr, dtype=jnp.float32),
        at_risk=jnp.asarray(at_risk),
        allowed=jnp.asarray(allowed_arr),
        event_index=jnp.asarray(event_index),
        pre_entry=None if pre_entry is None else jnp.asarray(pre_entry),
        exposure=jnp.asarray(exposure),
    )


def make_observations(
    history: Any,
    stage: str,
    calendar: Any,
    *,
    features: Any | None = None,
    cure_features: Any | None = None,
    allowed: Any | None = None,
) -> StageObservations:
    """Build calendar-indexed :class:`StageObservations` for one stage.

    Thin wrapper over :func:`make_event_observations` using the standard
    sale/initiation/receipt column names and ``history.policy_days`` as the
    initiation deadline (``None`` means unbounded). ``history`` is a
    ``RetailHistory``-shaped object (``frame``, ``as_of``, ``policy_days``).

    ``"initiation"`` uses every row of ``history.frame`` (all historical
    sales); ``"receipt"`` uses only rows with an observed initiation date,
    and resets the clock origin to that initiation date. ``features``
    (``[frame rows, len(calendar), P]``), ``cure_features``
    (``[frame rows, Q]``), and ``allowed`` (``[len(calendar)]`` or
    ``[frame rows, len(calendar)]``) are indexed against the *full*
    ``history.frame`` row order; for ``"receipt"`` the matching row subset
    is selected automatically so feature/cure/mask rows stay aligned with
    the units actually observed in that stage. Omitted arrays default to
    zero-width features/cure-features or an all-allowed mask. No
    conditional-entry snapshot semantics here (``entry_dates`` is always
    ``None``); existing public behavior is maintained except that
    ``policy_days=None`` now means an unbounded initiation deadline.
    """
    if stage == "initiation":
        origin_column = "sale_date"
        event_column = "initiation_date"
        deadline_days = history.policy_days
    elif stage == "receipt":
        origin_column = "initiation_date"
        event_column = "receipt_date"
        deadline_days = None
    else:
        raise ValueError(f"stage must be one of {_STAGES}, got {stage!r}")
    return make_event_observations(
        history.frame,
        origin_column=origin_column,
        event_column=event_column,
        as_of=history.as_of,
        calendar=calendar,
        deadline_days=deadline_days,
        features=features,
        cure_features=cure_features,
        allowed=allowed,
    )


def timing_inputs(observations: StageObservations) -> TimingInputs:
    """Time-major ``TimingInputs`` (``[T, N]`` ages, ``[T, N, P]`` features) of a stage.

    This is everything a timing family may see: elapsed ages and regressors,
    never the observed outcomes or administrative masks.
    """
    features = jnp.asarray(observations.features)
    return TimingInputs(
        ages=jnp.asarray(observations.ages).T,
        features=jnp.moveaxis(features, 0, 1),
        cure_features=jnp.asarray(observations.cure_features),
    )


def _observation_masks(observations: StageObservations):
    """Time-major closure, administrative exposure and pre-entry masks of a stage.

    Exposure is inferred for hand-built observations without one (see
    :class:`StageObservations`); negative ages are excluded from exposure and
    pre-entry because the shared kernel cannot see ages.
    """
    ages = jnp.asarray(observations.ages)
    present = ages >= 0
    allowed = jnp.asarray(observations.allowed, dtype=bool)
    if observations.exposure is None:
        event_index = jnp.asarray(observations.event_index)
        after_event = (event_index >= 0)[:, None] & (
            jnp.arange(ages.shape[-1])[None, :] > event_index[:, None]
        )
        exposure = jnp.asarray(observations.at_risk, dtype=bool) | after_event
    else:
        exposure = jnp.asarray(observations.exposure, dtype=bool)
    exposure = exposure & present
    if observations.pre_entry is None:
        pre_entry = None
    else:
        pre_entry = (jnp.asarray(observations.pre_entry, dtype=bool) & present).T
    return allowed.T, exposure.T, pre_entry


def observation_kernel(
    timing: TimingLaw, susceptibility_logits: Any, observations: StageObservations
) -> SurvivalKernel:
    """Mask a family's timing law by the stage's exposure, closures and pre-entry survival.

    The result is the one law used for the observed likelihood, the native
    observation site and in-sample prediction: ``kernel_unit_log_prob`` of it
    at ``observations.event_index`` is each unit's observed log probability.
    """
    allowed, exposure, pre_entry = _observation_masks(observations)
    return survival_kernel(
        timing, susceptibility_logits, allowed=allowed, exposure=exposure, pre_entry=pre_entry
    )


def _trajectories(observations: StageObservations) -> tuple[Any, Any]:
    """Integer ``[T, N]`` trajectories and impossible-unit mask, never invalid censoring."""
    event_index = jnp.asarray(observations.event_index)
    units, days = jnp.asarray(observations.ages).shape
    if event_index.shape != (units,) or not jnp.issubdtype(event_index.dtype, jnp.integer):
        raise ValueError("event_index must be an integer vector with one index per unit")
    valid = (event_index >= -1) & (event_index < days)
    if not isinstance(valid, jax.core.Tracer) and not np.asarray(valid).all():
        raise ValueError("event_index must be -1 or an index within the observed calendar")
    data = (jnp.arange(days)[:, None] == event_index[None, :]).astype(jnp.int32)
    return data, ~valid


def _observe(
    kernel: SurvivalKernel, inputs: TimingInputs, data: Any | None, *, massless=None
) -> None:
    """Register the stage's trajectories at the native observation site.

    ``data`` is the observed ``[T, N]`` trajectory grid, or ``None`` to draw
    the in-sample posterior predictive at ``"obs"``. The compact kernel route
    scores the observation from log hazards without forming date masses; the
    predictor passed to ``predict`` is the kernel's own log hazard.
    """
    horizon = Horizon.from_data(inputs.ages, data)
    predict(
        horizon,
        lambda log_hazard: EventTime(
            kernel=kernel._replace(log_hazard=log_hazard), massless=massless
        ),
        kernel.log_hazard,
    )


def _validated_law(law: Any, inputs: TimingInputs) -> tuple[TimingLaw, Any]:
    """Check a factory's ``(TimingLaw, susceptibility_logits)`` and broadcast it to the inputs."""
    if not isinstance(law, tuple) or len(law) != 2 or not isinstance(law[0], TimingLaw):
        raise TypeError("event_time_model must return (TimingLaw, susceptibility_logits)")
    timing, logits = law
    cells = tuple(np.shape(inputs.ages))
    units = cells[:-2] + cells[-1:]
    values = (
        ("log_hazard", jnp.asarray(timing.log_hazard), cells),
        ("log_survival_step", jnp.asarray(timing.log_survival_step), cells),
        ("susceptibility_logits", jnp.asarray(logits), units),
    )
    resolved = []
    for name, value, shape in values:
        try:
            resolved.append(jnp.broadcast_to(value, shape))
        except ValueError:
            raise ValueError(
                f"event_time_model {name} has shape {value.shape}, which does not "
                f"broadcast to the {shape} inputs"
            ) from None
    return TimingLaw(resolved[0], resolved[1]), resolved[2]


def _replay_family(event_time_model, parameters, shared, inputs, key) -> tuple[TimingLaw, Any]:
    """Run a factory with one draw's named values substituted at its sites."""
    key = random.PRNGKey(0) if key is None else key
    model = handlers.substitute(handlers.seed(event_time_model, key), data=dict(parameters))
    with handlers.trace() as trace:
        law = model(inputs, shared)
    missing = sorted(
        name
        for name, site in trace.items()
        if site["type"] in ("sample", "param")
        and not site.get("is_observed", False)
        and name not in parameters
    )
    if missing:
        raise ValueError(
            f"event_time_model reaches sites absent from the fitted posterior: {missing}"
        )
    return _validated_law(law, inputs)


def stage_log_likelihood(parameters: StageParameters, observations: StageObservations) -> Any:
    """Per-unit mixture-cure log likelihood of the default family, ``[N]``.

    One posterior draw of ``StageParameters`` is turned into its timing law
    and scored through the shared kernel: an event on date ``d`` has mass
    ``pi * S_{<d} * h_d`` and censoring has the residual
    ``(1 - pi) + pi * S_{<T}``, with hazard exactly zero on closed, unexposed
    or negative-age dates and susceptibility conditioned on any known
    pre-entry survival (``logit(pi) + log S_pre``). Large finite logits keep
    finite values and gradients; an impossible outcome (an event on a closed
    or unexposed date) scores ``-inf``. The native observation site of
    ``stage_model`` sums exactly these per-unit scores.
    """
    inputs = timing_inputs(observations)
    timing, logits = default_timing(parameters, inputs)
    kernel = observation_kernel(timing, logits, observations)
    return kernel_unit_log_prob(kernel, jnp.asarray(observations.event_index))


def sample_stage_parameters(
    observations: StageObservations, *, age_bins: int = 30
) -> StageParameters:
    """Sample one stage's prior and return the resulting :class:`StageParameters`.

    Sample sites and priors:

    - ``age_scale ~ HalfNormal(1)``: random-walk step-size controlling how
      quickly the age baseline can vary between adjacent bins. This is the
      regularizer that keeps the baseline flexible without being a
      prescribed parametric duration family: it shrinks the fit toward a
      flat hazard unless the data support a shape, and how much curvature
      is allowed is itself learned rather than fixed.
    - ``age_init ~ Normal(0, 2)``: logit-hazard at age 0.
    - ``age_steps ~ Normal(0, 1)^(age_bins - 1)``: standardized random-walk
      innovations; the cumulative, ``age_scale``-scaled sum on top of
      ``age_init`` gives the ``age_bins`` baseline logits, deterministically
      recorded as ``age_logits``.
    - ``beta ~ Normal(0, 1)^P``: linear effects of the time-varying hazard
      regressors.
    - ``cure_intercept ~ Normal(0, 2)``: logit susceptibility intercept.
    - ``cure_beta ~ Normal(0, 1)^Q``: linear effects of the static
      susceptibility regressors.

    Extracted from ``stage_model`` so a custom NumPyro prior callback (e.g.
    ``CureProcess.parameter_model``) can sample or otherwise construct its
    own :class:`StageParameters` under a caller-controlled scope while
    reusing these exact sample sites and priors when no override is given.
    """
    num_features = observations.features.shape[-1]
    num_cure_features = observations.cure_features.shape[-1]

    age_scale = numpyro.sample("age_scale", dist.HalfNormal(1.0))
    age_init = numpyro.sample("age_init", dist.Normal(0.0, 2.0))
    age_steps = numpyro.sample(
        "age_steps", dist.Normal(0.0, 1.0).expand([max(age_bins - 1, 0)]).to_event(1)
    )
    age_logits = jnp.concatenate([age_init[None], age_init + jnp.cumsum(age_steps) * age_scale])
    age_logits = numpyro.deterministic("age_logits", age_logits)

    beta = numpyro.sample("beta", dist.Normal(0.0, 1.0).expand([num_features]).to_event(1))
    cure_intercept = numpyro.sample("cure_intercept", dist.Normal(0.0, 2.0))
    cure_beta = numpyro.sample(
        "cure_beta", dist.Normal(0.0, 1.0).expand([num_cure_features]).to_event(1)
    )

    return StageParameters(
        age_logits=age_logits, beta=beta, cure_intercept=cure_intercept, cure_beta=cure_beta
    )


def stage_model(
    observations: StageObservations, *, age_bins: int = 30, event_time_model=None
) -> None:
    """NumPyro model for one mixture-cure stage.

    Without ``event_time_model`` the prior of :func:`sample_stage_parameters`
    feeds the default timing adapter; with one, the factory is called as
    ``event_time_model(timing_inputs(observations), None)`` and samples its
    own named sites. Either law is masked by the stage's exposure, closures
    and pre-entry survival and observed at the native ``"obs"`` site as an
    integer ``[T, N]`` trajectory grid through ``Horizon``/``predict``.
    """
    inputs = timing_inputs(observations)
    if event_time_model is None:
        parameters = sample_stage_parameters(observations, age_bins=age_bins)
        timing, logits = default_timing(parameters, inputs)
    else:
        timing, logits = _validated_law(event_time_model(inputs, None), inputs)
    data, impossible = _trajectories(observations)
    _observe(observation_kernel(timing, logits, observations), inputs, data, massless=impossible)


def _run_svi(
    program, trace, latent, parameter_names, *, keys, num_steps, num_samples, learning_rate
):
    """Optimize real model parameters and draw the nonempty guide's named sites."""

    def empty_guide():
        for name, site in trace.items():
            if site["type"] == "sample" and not site["is_observed"]:
                numpyro.sample(name, site["fn"])

    guide = AutoNormal(program) if latent else empty_guide
    svi = SVI(program, guide, numpyro.optim.Adam(learning_rate), Trace_ELBO())
    result = svi.run(keys[0], num_steps, progress_bar=False)
    losses = result.losses
    if not np.isfinite(losses).all():
        raise FloatingPointError("nonfinite SVI losses; inspect covariate scaling and priors")
    posterior = (
        Predictive(
            guide,
            params=result.params,
            num_samples=num_samples,
            return_sites=[
                name
                for name, site in trace.items()
                if site["type"] == "sample" and not site["is_observed"]
            ],
            parallel=True,
        )(keys[1])
        if latent
        else {}
    )
    params = {name: result.params[name] for name in parameter_names}
    return posterior, params, losses


def _validated_fit_trace(program, key):
    """Reject impossible model data before fitting and identify real inference sites."""
    trace = handlers.trace(handlers.seed(program, key)).get_trace()
    latent = False
    for site in trace.values():
        if site["type"] != "sample":
            continue
        if not np.isfinite(site["fn"].log_prob(site["value"])).all():
            raise ValueError(
                "model gives nonfinite density to the observations; check closures and priors"
            )
        latent |= not site["is_observed"] and np.size(site["value"]) > 0
    parameter_names = [name for name, site in trace.items() if site["type"] == "param"]
    return trace, latent, parameter_names


def _fit_program(program, resolver, return_sites, *, num_steps, num_samples, seed, learning_rate):
    """Fit a zero-argument NumPyro program with SVI and resolve named draws.

    Returns ``(posterior, params, losses, resolved)``: guide draws of every
    latent sample site, optimized ``numpyro.param`` values of the program
    itself (never the guide's), the loss trace, and ``return_sites`` of
    ``resolver`` evaluated under those draws. A program with no latent site
    and no parameter has nothing to fit and yields an empty trace.
    """
    keys = random.split(random.PRNGKey(seed), 4)
    trace, latent, parameter_names = _validated_fit_trace(program, keys[0])
    params = {}
    if latent or parameter_names:
        posterior, params, losses = _run_svi(
            program,
            trace,
            latent,
            parameter_names,
            keys=keys[1:3],
            num_steps=num_steps,
            num_samples=num_samples,
            learning_rate=learning_rate,
        )
    else:
        # A fully specified model has no parameters to optimize; still simulate its noise.
        posterior, losses = {}, np.empty(0)
    # Empty latent vectors are real named sites, even when there is no guide.
    # Preserve their draw axis so family replay cannot mistake them for missing priors.
    for name, site in trace.items():
        if site["type"] == "sample" and not site["is_observed"] and np.size(site["value"]) == 0:
            value = jnp.asarray(site["value"])
            posterior.setdefault(name, jnp.broadcast_to(value, (num_samples,) + value.shape))
    resolved = {}
    if return_sites:
        resolved = Predictive(
            resolver,
            posterior_samples=posterior or None,
            params=params,
            num_samples=num_samples,
            return_sites=return_sites,
            condition_deterministic=True,
            parallel=True,
        )(keys[3])
    if any(
        not np.isfinite(value).all()
        for value in tree_util.tree_leaves((posterior, params, resolved))
    ):
        raise FloatingPointError("nonfinite posterior parameter draws")
    return dict(posterior), params, losses, resolved


def _named_posterior(posterior, params, *, prefix, num_samples):
    """A factory's local sample-site draws and optimized values, scope stripped, per draw."""
    local = {
        name[len(prefix) :]: jnp.asarray(value)
        for name, value in posterior.items()
        if name.startswith(prefix)
    }
    for name, value in params.items():
        if name.startswith(prefix):
            value = jnp.asarray(value)
            local[name[len(prefix) :]] = jnp.broadcast_to(value, (num_samples,) + value.shape)
    return local


def fit_stage(
    observations: StageObservations,
    *,
    age_bins: int = 30,
    event_time_model=None,
    num_steps: int = 500,
    num_samples: int = 100,
    seed: int = 0,
    learning_rate: float = 0.02,
) -> StageFit:
    """Fit ``stage_model`` with SVI (AutoNormal guide) and draw the posterior.

    Ordinary NumPyro inference, not a custom engine: ``SVI`` with an
    ``AutoNormal`` mean-field guide and the ``Adam`` optimizer minimizes the
    ``Trace_ELBO``; ``num_samples`` guide draws are then resolved through
    the model. For the default family the result's ``parameters`` is a
    ``StageParameters`` (including the deterministic ``age_logits``) with a
    leading draw axis. With ``event_time_model`` the factory's own sample
    sites and optimized ``numpyro.param`` values form the named posterior
    mapping instead, and the factory itself is kept for replay. A family
    with nothing to fit still records ``num_samples`` draws. Advanced users
    can call ``stage_model`` directly with any NumPyro inference algorithm
    (e.g. MCMC) instead of this convenience.
    """
    for name, value in (
        ("age_bins", age_bins),
        ("num_steps", num_steps),
        ("num_samples", num_samples),
    ):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if observations.ages.shape[0] == 0:
        raise ValueError("fitting requires at least one observed or censored item")
    if (
        not np.isfinite(observations.features).all()
        or not np.isfinite(observations.cure_features).all()
    ):
        raise ValueError("fitting requires finite feature values")
    if event_time_model is not None:
        tail_behavior(event_time_model)
    program = partial(
        stage_model, observations, age_bins=age_bins, event_time_model=event_time_model
    )
    resolver = partial(sample_stage_parameters, observations, age_bins=age_bins)
    sites = list(StageParameters._fields) if event_time_model is None else []
    posterior, params, losses, resolved = _fit_program(
        program,
        resolver,
        sites,
        num_steps=num_steps,
        num_samples=num_samples,
        seed=seed,
        learning_rate=learning_rate,
    )
    if event_time_model is None:
        parameters = StageParameters(*[resolved[name] for name in StageParameters._fields])
        return StageFit(parameters, losses, num_samples=num_samples)
    return StageFit(
        _named_posterior(posterior, params, prefix="", num_samples=num_samples),
        losses,
        event_time_model=event_time_model,
        num_samples=num_samples,
    )


def predict_stage(fit: StageFit, observations: StageObservations, *, seed: int = 0) -> Any:
    """In-sample posterior predictive trajectories ``[draws, T, N]`` of a fitted stage.

    Each posterior draw replays its timing law on the observations' ages and
    regressors, applies the same administrative exposure, closures and
    pre-entry conditioning as fitting, and samples the native observation
    site with no data attached. The recorded outcomes play no part: a unit
    whose event was observed can fire on any exposed date, or not at all.
    Draws are integer time-major grids with at most one event per unit.
    """
    inputs = timing_inputs(observations)

    def model(draw):
        timing, logits = fit.timing(inputs, draw=draw)
        _observe(observation_kernel(timing, logits, observations), inputs, None)

    def sample(draw, key):
        return handlers.trace(handlers.seed(model, key)).get_trace(draw)["obs"]["value"]

    keys = random.split(random.PRNGKey(seed), fit.draws)
    return jax.vmap(sample)(jnp.arange(fit.draws), keys)
