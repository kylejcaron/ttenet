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
from typing import Any, Optional

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
    data_fields=["ages", "features", "cure_features", "at_risk", "allowed", "event_index"],
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
    ``at_risk: [N, T]`` -- observation/censoring window: ``True`` for every
    calendar day the unit was under observation, through and including its
    event or censoring day, clipped at ``as_of`` (and at the policy deadline
    for uninitiated ``"initiation"`` rows). Distinct from ``allowed``: a
    closed day can still be "at risk" (observed), it simply carries zero
    hazard.
    ``allowed: [N, T]`` -- hard calendar closures (e.g. weekends), zero
    hazard regardless of the learned baseline/regressors.
    ``event_index: [N]`` -- calendar index of the observed event day, or
    ``-1`` if the unit was censored (no event observed by ``as_of``).

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


def _stage_clock(
    history: Any, stage: str
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Per-stage clock origin, event date, deadline, and full-frame row mask.

    Returns ``(origin, event_date, deadline_or_none, row_mask)`` as host
    NumPy arrays. ``row_mask`` selects which rows of ``history.frame``
    participate in this stage (all sale rows for ``"initiation"``, only
    initiated rows for ``"receipt"``); ``origin``/``event_date`` are already
    filtered by ``row_mask``.
    """
    frame = history.frame
    if stage == "initiation":
        row_mask = np.ones(len(frame), dtype=bool)
        origin = to_day(frame["sale_date"])
        event_date = to_day(frame["initiation_date"])
        deadline = origin + np.timedelta64(int(history.policy_days), "D")
        return origin, event_date, deadline, row_mask
    if stage == "receipt":
        initiation_date = to_day(frame["initiation_date"])
        row_mask = ~np.isnat(initiation_date)
        origin = initiation_date[row_mask]
        event_date = to_day(frame["receipt_date"])[row_mask]
        return origin, event_date, None, row_mask
    raise ValueError(f"stage must be one of {_STAGES}, got {stage!r}")


def _select_rows(
    array: Optional[Any], row_mask: np.ndarray, total_rows: int, *, name: str
) -> Optional[np.ndarray]:
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


def make_observations(
    history: Any,
    stage: str,
    calendar: Any,
    *,
    features: Optional[Any] = None,
    cure_features: Optional[Any] = None,
    allowed: Optional[Any] = None,
) -> StageObservations:
    """Build calendar-indexed :class:`StageObservations` for one stage.

    ``history`` is a ``RetailHistory``-shaped object (``frame``, ``as_of``,
    ``policy_days``); ``calendar`` is an inclusive, contiguous daily
    ``datetime64[D]`` grid (e.g. from ``dates.date_grid``) that must start
    at or before the stage's earliest clock origin and extend through
    ``as_of`` -- day resolution beyond ``as_of`` is irrelevant here since
    future days never contribute to the likelihood.

    ``"initiation"`` uses every row of ``history.frame`` (all historical
    sales); ``"receipt"`` uses only rows with an observed initiation date,
    and resets the clock origin to that initiation date. ``features``
    (``[frame rows, len(calendar), P]``), ``cure_features``
    (``[frame rows, Q]``), and ``allowed`` (``[len(calendar)]`` or
    ``[frame rows, len(calendar)]``) are indexed against the *full*
    ``history.frame`` row order; for ``"receipt"`` the matching row subset
    is selected automatically so feature/cure/mask rows stay aligned with
    the units actually observed in that stage. Omitted arrays default to
    zero-width features/cure-features or an all-allowed mask.
    """
    total_rows = len(history.frame)
    origin, event_date, deadline, row_mask = _stage_clock(history, stage)
    n_rows = origin.shape[0]

    calendar = np.asarray(calendar, dtype="datetime64[D]")
    if calendar.ndim != 1 or calendar.shape[0] == 0:
        raise ValueError("calendar must be a nonempty one-dimensional date array")
    if np.isnat(calendar).any():
        raise ValueError("calendar must not contain missing dates")
    if calendar.shape[0] > 1 and not np.all(
        np.diff(calendar).astype("timedelta64[D]").astype(np.int64) == 1
    ):
        raise ValueError("calendar must be an inclusive, contiguous daily grid")
    num_days = calendar.shape[0]
    calendar_start = calendar[0]
    as_of = to_day(history.as_of)

    if n_rows > 0 and calendar_start > origin.min():
        raise ValueError("calendar must start at or before the stage's earliest clock origin")
    if calendar[-1] < as_of:
        raise ValueError("calendar must extend through as_of")

    has_event = ~np.isnat(event_date) & (event_date <= as_of)
    if stage == "initiation":
        stop_date = np.minimum(deadline, as_of)
    else:
        stop_date = np.full(n_rows, as_of, dtype="datetime64[D]")

    end_date = np.where(has_event, event_date, stop_date)
    end_index = (end_date - calendar_start).astype("timedelta64[D]").astype(np.int64)
    event_raw_index = (event_date - calendar_start).astype("timedelta64[D]").astype(np.int64)
    if np.any(has_event & ((event_raw_index < 0) | (event_raw_index >= num_days))):
        raise ValueError("calendar does not cover an observed event date")
    event_index = np.where(has_event, event_raw_index, -1).astype(np.int64)

    ages = (calendar[None, :] - origin[:, None]).astype("timedelta64[D]").astype(np.int64)
    day_index = np.arange(num_days)
    at_risk = (ages >= 0) & (day_index[None, :] <= end_index[:, None])

    features_full = _select_rows(features, row_mask, total_rows, name="features")
    if features_full is None:
        features_arr = np.zeros((n_rows, num_days, 0))
    else:
        if features_full.shape[1] != num_days:
            raise ValueError("features must be indexed against the exact calendar length")
        features_arr = features_full

    cure_full = _select_rows(cure_features, row_mask, total_rows, name="cure_features")
    cure_arr = np.zeros((n_rows, 0)) if cure_full is None else cure_full

    if allowed is None:
        allowed_arr = np.ones((n_rows, num_days), dtype=bool)
    else:
        allowed_input = np.asarray(allowed)
        if not np.isin(allowed_input, [False, True]).all():
            raise ValueError("allowed must contain boolean or zero/one values")
        allowed_input = allowed_input.astype(bool)
        if allowed_input.ndim == 1:
            if allowed_input.shape[0] != num_days:
                raise ValueError("a 1-D allowed mask must match the calendar length")
            allowed_arr = np.broadcast_to(allowed_input, (n_rows, num_days))
        else:
            allowed_full = _select_rows(allowed_input, row_mask, total_rows, name="allowed")
            if allowed_full.shape[-1] != num_days:
                raise ValueError("allowed must be indexed against the exact calendar length")
            allowed_arr = allowed_full

    return StageObservations(
        ages=jnp.asarray(ages),
        features=jnp.asarray(features_arr, dtype=jnp.float32),
        cure_features=jnp.asarray(cure_arr, dtype=jnp.float32),
        at_risk=jnp.asarray(at_risk),
        allowed=jnp.asarray(allowed_arr),
        event_index=jnp.asarray(event_index),
    )


def stage_log_likelihood(parameters: StageParameters, observations: StageObservations) -> Any:
    """Per-unit mixture-cure log likelihood, including exact hard-zero days.

    Work from logits, never log rounded sigmoid probabilities: large finite
    logits must retain finite likelihoods and gradients under float32 too.
    Censoring includes the last exposed day; event survival excludes its day.
    """
    ages = observations.ages
    at_risk = observations.at_risk
    event_index = observations.event_index
    logits = _hazard_logits(parameters, ages, observations.features)
    open_day = (ages >= 0) & observations.allowed
    log_hazard = jnp.where(open_day, log_sigmoid(logits), -jnp.inf)
    log_survival_day = jnp.where(open_day, log_sigmoid(-logits), 0.0)
    cure_logits = _cure_logits(parameters, observations.cure_features)
    log_pi, log_not_pi = log_sigmoid(cure_logits), log_sigmoid(-cure_logits)

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


def stage_model(observations: StageObservations, *, age_bins: int = 30) -> None:
    """NumPyro model for one mixture-cure stage.

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

    The per-unit ``stage_log_likelihood`` is added to the log joint density
    via ``numpyro.factor``.
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

    parameters = StageParameters(
        age_logits=age_logits, beta=beta, cure_intercept=cure_intercept, cure_beta=cure_beta
    )
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
        exclude_deterministic=False,
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
