"""Discrete-time mixture-cure stage observations, likelihood, and NumPyro fitting.

Host-side preparation (`make_observations`) is plain NumPy/pandas and runs
before inference: it converts a history's dates into calendar-indexed
arrays and validates coverage, so it never touches JAX tracers. Everything
downstream (`stage_log_likelihood`, `stage_model`, `fit_stage`) is ordinary
differentiable JAX/NumPyro built from the single-draw primitives in
`survival.py`.
"""

from __future__ import annotations

import dataclasses
from functools import partial
from typing import Any

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from jax.nn import log_sigmoid
from jax.tree_util import register_dataclass
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

from .dates import to_day
from .survival import StageParameters, _cure_logits, _hazard_logits

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
    likelihood. When supplied, ``stage_log_likelihood`` uses the cumulative
    survival over these days to compute a *conditional* susceptibility at
    entry, so a unit's already-known event-free run before it entered the
    observed snapshot still informs its posterior cure probability without
    re-litigating a likelihood contribution for those days directly.

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


@partial(register_dataclass, data_fields=["parameters", "losses"], meta_fields=[])
@dataclasses.dataclass(frozen=True)
class StageFit:
    """SVI fit result: posterior stage-parameter draws and the loss trace.

    ``parameters`` carries a leading posterior-draw axis on every field
    (see ``StageParameters``). ``losses`` is the per-step ELBO loss trace
    from ``SVI.run``.
    """

    parameters: StageParameters
    losses: Any


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Construct age, exposure, event-index, and conditional-entry masks."""
    num_days = calendar.shape[0]
    calendar_start = calendar[0]
    stop_date = (
        np.minimum(deadline, as_of)
        if deadline is not None
        else np.full(origin.size, as_of, dtype="datetime64[D]")
    )
    end_date = np.where(has_event, event, stop_date)
    end_index = (end_date - calendar_start).astype("timedelta64[D]").astype(np.int64)
    event_raw_index = (event - calendar_start).astype("timedelta64[D]").astype(np.int64)
    if np.any(has_event & ((event_raw_index < 0) | (event_raw_index >= num_days))):
        raise ValueError("calendar does not cover an observed event date")
    event_index = np.where(has_event, event_raw_index, -1).astype(np.int64)

    ages = (calendar[None, :] - origin[:, None]).astype("timedelta64[D]").astype(np.int64)
    day_index = np.arange(num_days)
    risk_origin = np.maximum(origin, entry)
    risk_origin_idx = (risk_origin - calendar_start).astype("timedelta64[D]").astype(np.int64)
    at_risk = (day_index[None, :] >= risk_origin_idx[:, None]) & (
        day_index[None, :] <= end_index[:, None]
    )
    if not entry_dates_given:
        return ages, at_risk, event_index, None

    origin_idx = (origin - calendar_start).astype("timedelta64[D]").astype(np.int64)
    pre_entry_stop = entry - np.timedelta64(1, "D")
    if deadline is not None:
        pre_entry_stop = np.minimum(pre_entry_stop, deadline)
    pre_entry_stop_idx = (pre_entry_stop - calendar_start).astype("timedelta64[D]").astype(np.int64)
    pre_entry = (day_index[None, :] >= origin_idx[:, None]) & (
        day_index[None, :] <= pre_entry_stop_idx[:, None]
    )
    return ages, at_risk, event_index, pre_entry


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
    ages, at_risk, event_index, pre_entry = _construct_exposure(
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


def stage_log_likelihood(parameters: StageParameters, observations: StageObservations) -> Any:
    """Per-unit mixture-cure log likelihood, including exact hard-zero days.

    Work from logits, never log rounded sigmoid probabilities: large finite
    logits must retain finite likelihoods and gradients under float32 too.
    Censoring includes the last exposed day; event survival excludes its day.

    When ``observations.pre_entry`` is not ``None``, this is a
    conditional-entry likelihood: the cure logit used for the post-entry
    event/censor term is first shifted by the cumulative log-survival over
    the known event-free days *before* entry, i.e.
    ``sigmoid(cure_intercept + z @ beta + logS_pre)`` -- exactly
    ``conditional_susceptibility`` in logit space (see ``survival.py``).
    This avoids ever forming and subtracting two large log-marginal
    survivals. ``pre_entry is None`` (the default) makes ``logS_pre`` zero
    everywhere, reducing exactly to the ordinary full-origin likelihood.
    """
    ages = observations.ages
    at_risk = observations.at_risk
    event_index = observations.event_index
    logits = _hazard_logits(parameters, ages, observations.features)
    open_day = (ages >= 0) & observations.allowed
    log_hazard = jnp.where(open_day, log_sigmoid(logits), -jnp.inf)
    log_survival_day = jnp.where(open_day, log_sigmoid(-logits), 0.0)
    cure_logits = _cure_logits(parameters, observations.cure_features)
    if observations.pre_entry is None:
        log_survival_pre_entry = 0.0
    else:
        log_survival_pre_entry = jnp.sum(
            jnp.where(observations.pre_entry, log_survival_day, 0.0), axis=-1
        )
    conditional_cure_logits = cure_logits + log_survival_pre_entry
    log_pi, log_not_pi = log_sigmoid(conditional_cure_logits), log_sigmoid(-conditional_cure_logits)

    num_days = ages.shape[-1]
    is_event = event_index >= 0
    event_stop = jnp.where(is_event, event_index, num_days)
    survival_mask = at_risk & (jnp.arange(num_days)[None, :] < event_stop[:, None])
    log_survival = jnp.sum(jnp.where(survival_mask, log_survival_day, 0.0), axis=-1)
    safe_index = jnp.where(is_event, event_index, 0)
    event_log_hazard = jnp.take_along_axis(log_hazard, safe_index[:, None], axis=-1)[:, 0]
    event_is_exposed = jnp.take_along_axis(at_risk, safe_index[:, None], axis=-1)[:, 0]
    event_log_likelihood = jnp.where(
        event_is_exposed, log_pi + event_log_hazard + log_survival, -jnp.inf
    )
    censored_log_likelihood = jnp.logaddexp(log_not_pi, log_pi + log_survival)
    return jnp.where(is_event, event_log_likelihood, censored_log_likelihood)


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


def stage_model(observations: StageObservations, *, age_bins: int = 30) -> None:
    """NumPyro model for one mixture-cure stage.

    Delegates prior sampling to :func:`sample_stage_parameters` and adds the
    per-unit ``stage_log_likelihood`` to the log joint density via
    ``numpyro.factor``.
    """
    parameters = sample_stage_parameters(observations, age_bins=age_bins)
    log_likelihood = stage_log_likelihood(parameters, observations)
    numpyro.factor("stage_log_likelihood", log_likelihood)


def fit_stage(
    observations: StageObservations,
    *,
    age_bins: int = 30,
    num_steps: int = 500,
    num_samples: int = 100,
    seed: int = 0,
    learning_rate: float = 0.02,
) -> StageFit:
    """Fit ``stage_model`` with SVI (AutoNormal guide) and draw the posterior.

    Ordinary NumPyro inference, not a custom engine: ``SVI`` with an
    ``AutoNormal`` mean-field guide and the ``Adam`` optimizer minimizes the
    ``Trace_ELBO``. Posterior draws (including the deterministic
    ``age_logits`` site) are then produced with ``Predictive`` conditioned
    on the fitted guide, giving every ``StageParameters`` field a leading
    posterior-draw axis of size ``num_samples``. Advanced users can call
    ``stage_model`` directly with any NumPyro inference algorithm (e.g.
    MCMC) instead of this convenience.
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
    guide = AutoNormal(stage_model)
    optimizer = numpyro.optim.Adam(step_size=learning_rate)
    svi = SVI(stage_model, guide, optimizer, loss=Trace_ELBO())
    svi_key, predictive_key = random.split(random.PRNGKey(seed))
    result = svi.run(svi_key, num_steps, observations, age_bins=age_bins, progress_bar=False)
    if not np.isfinite(result.losses).all():
        raise FloatingPointError("nonfinite SVI losses; inspect covariate scaling and model priors")

    predictive = Predictive(
        stage_model,
        guide=guide,
        params=result.params,
        num_samples=num_samples,
        return_sites=["age_logits", "beta", "cure_intercept", "cure_beta"],
        condition_deterministic=True,
    )
    draws = predictive(predictive_key, observations, age_bins=age_bins)
    if any(not np.isfinite(value).all() for value in draws.values()):
        raise FloatingPointError("nonfinite posterior draws")
    parameters = StageParameters(
        age_logits=draws["age_logits"],
        beta=draws["beta"],
        cure_intercept=draws["cure_intercept"],
        cure_beta=draws["cure_beta"],
    )
    return StageFit(parameters=parameters, losses=result.losses)
