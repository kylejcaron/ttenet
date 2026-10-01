"""Shared discrete-time mixture-cure event-time kernel.

One unit either fires on exactly one calendar date or never fires. The law
is a mixture: with susceptibility ``pi`` the unit follows a timing law
whose per-date event probability is a hazard ``h_t`` on exposed dates, and
with probability ``1 - pi`` it is cured and never fires. In log space, with
``log S_{<t} = sum_{s<t} log(1 - h_s)`` (the exclusive log survival),

* date mass:  ``log pi + log S_{<t} + log h_t``
* residual:   ``logaddexp(log(1 - pi), log pi + log S_{<T})``

Every quantity here is a log probability. Hazards may be exactly zero
(``-inf``) on closed or unexposed dates and stays may be exactly zero
(``-inf``) on structurally certain dates; those infinities are boundary
values, never intermediate cancellations, so nothing below subtracts two
infinities or forms ``1 - p`` from a rounded probability.

Layout is time-major: cells are ``[..., time, cohort]`` and per-unit values
are ``[..., cohort]``. Leading axes broadcast (posterior draws, samples).
Timing families receive only ages and regressors -- never realized
outcomes or administrative masks -- so the same law serves fitting,
conditioning and forecasting.

All functions are pure JAX and safe to trace, ``jit``, ``vmap`` and
differentiate. The only host-side check (pre-entry/exposure disjointness)
runs when the masks are concrete and is documented on ``survival_kernel``.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.nn import log_sigmoid

from .survival import StageParameters, _hazard_logits, _susceptibility_logits

_LOG_HALF = math.log(0.5)


class TimingInputs(NamedTuple):
    """What a timing family may see: elapsed ages and regressors only.

    ages: ``[..., time, cohort]`` days elapsed since the immediate parent
        date; negative before the parent date. Calendar closures do not
        pause age.
    features: ``[..., time, cohort, P]`` time-varying hazard regressors.
    susceptibility_features: ``[..., cohort, Q]`` static susceptibility regressors.
    """

    ages: Any
    features: Any
    susceptibility_features: Any


class TimingLaw(NamedTuple):
    """Per-cell log hazard and log stay of the susceptible timing law.

    On exposed cells ``exp(log_hazard) + exp(log_survival_step) == 1``.
    Either may be ``-inf`` for an exact boundary probability. Values on
    cells that will be masked must still be finite or exact boundaries
    (never NaN) so masked gradients stay clean.
    """

    log_hazard: Any
    log_survival_step: Any


class EventLaw(NamedTuple):
    """A family's complete law for one stage: susceptible timing plus susceptibility.

    ``timing`` is the per-cell :class:`TimingLaw` and ``susceptibility_logits``
    is ``[..., cohort]`` (or anything that broadcasts to it) -- the logit of
    the probability that a unit is susceptible at all. Every family model,
    packaged or custom, returns one of these.
    """

    timing: TimingLaw
    susceptibility_logits: Any


class SurvivalKernel(NamedTuple):
    """Exposure-masked timing law plus conditioned susceptibility logits.

    ``log_hazard`` and ``log_survival_step`` are ``[..., time, cohort]`` with
    hazard ``-inf`` and stay ``0`` on unexposed cells. ``susceptibility_logits``
    is ``[..., cohort]``, already conditioned on any known pre-entry survival.
    """

    log_hazard: Any
    log_survival_step: Any
    susceptibility_logits: Any

    @property
    def log_susceptible(self) -> Any:
        return log_sigmoid(self.susceptibility_logits)

    @property
    def log_rest(self) -> Any:
        return log_sigmoid(-self.susceptibility_logits)

    @property
    def log_mass(self) -> Any:
        """``[..., time, cohort]`` log probability of the event on each date."""
        return (
            self.log_susceptible[..., None, :]
            + exclusive_log_survival(self.log_survival_step)
            + self.log_hazard
        )

    @property
    def log_tail(self) -> Any:
        """``[..., cohort]`` log probability of no event on any date."""
        return _logaddexp(
            self.log_rest, self.log_susceptible + jnp.sum(self.log_survival_step, axis=-2)
        )


@jax.custom_jvp
def _logaddexp(x: Any, y: Any) -> Any:
    """``logaddexp`` whose derivative is zero, not NaN, where both operands are ``-inf``.

    Two exhausted log masses add to an exhausted log mass; the library
    derivative forms ``exp(-inf - -inf)`` there. Every finite case keeps the
    ordinary softmax weights.
    """
    return jnp.logaddexp(x, y)


@_logaddexp.defjvp
def _logaddexp_jvp(primals, tangents):
    x, y = primals
    tangent_x, tangent_y = tangents
    out = jnp.logaddexp(x, y)
    finite = jnp.isfinite(out)
    anchor = jnp.where(finite, out, 0.0)
    weight_x = jnp.where(finite, jnp.exp(x - anchor), 0.0)
    weight_y = jnp.where(finite, jnp.exp(y - anchor), 0.0)
    return out, tangent_x * weight_x + tangent_y * weight_y


def log1mexp(x: Any) -> Any:
    """``log(1 - exp(x))`` for a log probability ``x <= 0``.

    Exactly ``-inf`` at ``x == 0`` (a certain stay leaves no hazard) and
    ``0`` at ``x == -inf``. Uses ``log(-expm1(x))`` above ``log(1/2)`` and
    ``log1p(-exp(x))`` below it, each branch evaluated only on an operand
    where it is well conditioned, so gradients are finite everywhere,
    including at both boundaries. Operands above zero are rounding noise on
    a log probability and are clamped to the boundary.
    """
    x = jnp.minimum(jnp.asarray(x), 0.0)
    boundary = x == 0.0
    near = x > _LOG_HALF
    near_value = jnp.log(-jnp.expm1(jnp.where(near & ~boundary, x, -1.0)))
    far_value = jnp.log1p(-jnp.exp(jnp.where(near, -1.0, x)))
    return jnp.where(boundary, -jnp.inf, jnp.where(near, near_value, far_value))


def exclusive_log_survival(log_survival_step: Any) -> Any:
    """``log S_{<t}``: cumulative log stay over the dates strictly before ``t``.

    Time is axis ``-2``. The first date gets ``0``; an empty time axis gives
    an empty result. A structural ``-inf`` stay propagates to every later
    date by accumulation alone; the inclusive sum is shifted, never
    subtracted, so no infinities cancel.
    """
    log_survival_step = jnp.asarray(log_survival_step)
    inclusive = jnp.cumsum(log_survival_step, axis=-2)
    return jnp.concatenate(
        [jnp.zeros_like(log_survival_step[..., :1, :]), inclusive[..., :-1, :]], axis=-2
    )


def _log_suffix_sums(log_weights: Any) -> Any:
    """``log sum_{a' >= a} exp(w[a'])`` along the last axis; finite gradients when exhausted."""
    return lax.associative_scan(_logaddexp, log_weights, reverse=True, axis=log_weights.ndim - 1)


def timing_from_log_masses(log_mass: Any, log_tail: Any, ages: Any) -> TimingLaw:
    """Timing law of a discrete susceptible mass on ages ``0..A-1`` plus a beyond-grid atom.

    ``log_mass`` is ``[..., A]`` log mass per age and ``log_tail`` is ``[...]``
    the log mass of susceptible units that have no event on any grid age;
    the leading axes broadcast against the ``[..., cohort]`` layout of
    ``ages``: ``[C, A]`` grids are per cohort, while ``[D, 1, A]`` grids
    are per posterior draw for an age array ``[T, C]``. Only ratios matter,
    so the masses need not be normalized.

    Hazard at age ``a`` is ``mass[a] / remaining[a]`` and stay is
    ``remaining[a+1] / remaining[a]`` where ``remaining[a]`` is the log
    suffix mass including the atom; ratios are differences of log suffix
    sums, never ``1 - p``. Once the remaining mass is zero (a proper law
    with no atom past its last age) no susceptible unit can still be
    present, and past the grid only the atom remains: both give hazard
    ``-inf`` and stay ``0``.

    That off-grid padding is neutral bookkeeping for calendar cells the
    grid does not describe. It does not say what the atom's units do
    later -- whether they are known never to fire, fire at some
    unmodelled later age, or are simply unresolved -- and therefore
    cannot by itself justify an eventual expectation or a proper/finite
    continuation. The family's typed tail declaration specifies that.

    Negative ages look up age ``0``; exposure masking is the caller's
    concern, as for every timing family.
    """
    log_mass = jnp.asarray(log_mass)
    log_tail = jnp.asarray(log_tail)
    ages = jnp.asarray(ages)
    support = log_mass.shape[-1]
    batch = jnp.broadcast_shapes(log_mass.shape[:-1], log_tail.shape)
    grid = jnp.concatenate(
        [
            jnp.broadcast_to(log_mass, batch + (support,)),
            jnp.broadcast_to(log_tail, batch)[..., None],
        ],
        axis=-1,
    )
    remaining = _log_suffix_sums(grid)
    here, after = remaining[..., :-1], remaining[..., 1:]
    reachable = ~jnp.isneginf(here)
    anchor = jnp.where(reachable, here, 0.0)
    grid_log_hazard = jnp.where(reachable, jnp.minimum(log_mass - anchor, 0.0), -jnp.inf)
    grid_log_stay = jnp.where(reachable, jnp.minimum(after - anchor, 0.0), 0.0)
    beyond = jnp.zeros_like(grid[..., :1])
    grid_log_hazard = jnp.concatenate([grid_log_hazard, beyond - jnp.inf], axis=-1)
    grid_log_stay = jnp.concatenate([grid_log_stay, beyond], axis=-1)

    lead = jnp.broadcast_shapes(grid.shape[:-1], ages.shape[:-2] + ages.shape[-1:])
    index = jnp.broadcast_to(
        jnp.clip(ages, 0, support)[..., None], lead[:-1] + ages.shape[-2:] + (1,)
    )

    def lookup(values):
        values = jnp.broadcast_to(values, lead + values.shape[-1:])[..., None, :, :]
        return jnp.take_along_axis(values, index, axis=-1)[..., 0]

    return TimingLaw(lookup(grid_log_hazard), lookup(grid_log_stay))


def timing_from_log_survival(log_survival_start: Any, log_survival_end: Any) -> TimingLaw:
    """Timing law of a susceptible continuous law from its log survival at bin edges.

    ``log_survival_start`` is ``log S(a)`` and ``log_survival_end`` is
    ``log S(a + 1)`` for the bin ``[a, a + 1)`` of every cell; the two arrays
    broadcast against each other to the ``[..., time, cohort]`` cell axes.
    The stay is the conditional probability of outliving the bin,
    ``log S(a + 1) - log S(a)``, formed directly in log space, and the hazard
    is its complement through :func:`log1mexp`; the two never go through
    ``1 - p``.

    Boundary values are exact. A finite ``log S(a)`` with ``log S(a + 1) ==
    -inf`` is a certain terminal bin: stay ``-inf``, hazard ``0``. An exhausted
    cell, ``-inf`` at both edges, has no susceptible unit left to fire: hazard
    ``-inf`` and stay ``0`` by substitution, never ``-inf - -inf``. Gradients
    are finite at every boundary. ``NaN`` propagates, and a pair that is not a
    non-increasing log probability (``log S(a + 1) > log S(a)``) yields a stay
    above one rather than a laundered law.

    Callers evaluate the continuous law at elapsed ages only; exposure,
    closures, entry conditioning and susceptibility stay in the shared core.
    """
    start, end = jnp.broadcast_arrays(
        jnp.asarray(log_survival_start), jnp.asarray(log_survival_end)
    )
    exhausted = jnp.isneginf(start) & jnp.isneginf(end)
    log_stay = jnp.where(exhausted, 0.0, end) - jnp.where(exhausted, 0.0, start)
    return TimingLaw(log_hazard=log1mexp(log_stay), log_survival_step=log_stay)


def default_timing(parameters: StageParameters, inputs: TimingInputs) -> EventLaw:
    """Timing law and susceptibility logits of the default logistic stage family.

    Hazard logits are ``age_logits[min(age, K-1)] + x(t) @ beta`` and the
    susceptibility logit is ``susceptibility_intercept + z @ susceptibility_beta``, both from
    the ``survival`` primitives. Log hazard and log stay are
    ``log_sigmoid(+-logit)`` so large finite logits keep finite values and
    gradients.

    Negative ages look up age zero; administrative adapters must exclude
    them from both exposure and pre-entry masks, as for other families.
    """
    logits = _hazard_logits(parameters, inputs.ages, inputs.features)
    timing = TimingLaw(log_hazard=log_sigmoid(logits), log_survival_step=log_sigmoid(-logits))
    return EventLaw(timing, _susceptibility_logits(parameters, inputs.susceptibility_features))


def _require_disjoint(pre_entry: Any, exposure: Any) -> None:
    if isinstance(pre_entry, jax.core.Tracer) or isinstance(exposure, jax.core.Tracer):
        return
    overlap = np.logical_and(np.asarray(pre_entry, dtype=bool), np.asarray(exposure, dtype=bool))
    if np.any(overlap):
        raise ValueError(
            "pre_entry and exposure must be disjoint: a date is either a known "
            "event-free day before entry or an exposed day, never both"
        )


def survival_kernel(
    timing: TimingLaw,
    susceptibility_logits: Any,
    *,
    allowed: Any,
    exposure: Any,
    pre_entry: Any | None = None,
) -> SurvivalKernel:
    """Mask a timing law by exposure and condition susceptibility on pre-entry survival.

    ``allowed`` (calendar closures), ``exposure`` (administrative exposure)
    and ``pre_entry`` (known event-free dates before entry) are boolean
    masks broadcast to the timing cells. Unexposed or closed cells get
    hazard ``-inf`` and stay ``0``. Pre-entry dates shift the susceptibility
    logit by their accumulated log stay -- ``logit(pi) + log S_pre`` -- which
    is exactly ``conditional_susceptibility`` in logit space and stays finite
    for histories whose survival underflows.

    ``pre_entry`` and ``exposure`` must be disjoint; overlapping dates would
    count survival twice. Concrete masks are checked and rejected here;
    traced masks are the calling adapter's responsibility.
    """
    allowed = jnp.asarray(allowed, dtype=bool)
    exposure = jnp.asarray(exposure, dtype=bool)
    logits = jnp.asarray(susceptibility_logits)
    if pre_entry is not None:
        _require_disjoint(pre_entry, exposure)
        known = allowed & jnp.asarray(pre_entry, dtype=bool)
        logits = logits + jnp.sum(jnp.where(known, timing.log_survival_step, 0.0), axis=-2)
    exposed = allowed & exposure
    return SurvivalKernel(
        log_hazard=jnp.where(exposed, timing.log_hazard, -jnp.inf),
        log_survival_step=jnp.where(exposed, timing.log_survival_step, 0.0),
        susceptibility_logits=logits,
    )


def kernel_unit_log_prob(kernel: SurvivalKernel, event_index: Any) -> Any:
    """``[..., cohort]`` log probability of each unit's observed outcome.

    ``event_index`` is the date index of the event, or ``-1`` for no event.
    An event date uses ``log pi + log S_{<d} + log h_d``; no event uses the
    residual ``logaddexp(log(1 - pi), log pi + log S_{<T})``. Only a masked
    sum over the dates before the outcome and one gather are needed, never
    the full date-mass array. Indices outside ``[-1, T-1]`` are impossible
    outcomes with ``-inf`` log probability.
    """
    event_index = jnp.asarray(event_index)
    logits = kernel.susceptibility_logits
    shape = jnp.broadcast_shapes(
        event_index.shape[:-1] + (1,) + event_index.shape[-1:],
        kernel.log_hazard.shape,
        logits.shape[:-1] + (1,) + logits.shape[-1:],
    )
    days = shape[-2]
    log_susceptible, log_rest = kernel.log_susceptible, kernel.log_rest
    is_event = event_index >= 0
    if days == 0:
        none = jnp.broadcast_to(_logaddexp(log_rest, log_susceptible), shape[:-2] + shape[-1:])
        return jnp.where(event_index == -1, none, -jnp.inf)
    stop = jnp.where(is_event, event_index, days)
    before = jnp.arange(days)[:, None] < stop[..., None, :]
    log_survival_before = jnp.sum(jnp.where(before, kernel.log_survival_step, 0.0), axis=-2)
    safe_index = jnp.broadcast_to(
        jnp.clip(event_index, 0, days - 1)[..., None, :], shape[:-2] + (1,) + shape[-1:]
    )
    log_hazard = jnp.broadcast_to(kernel.log_hazard, shape)
    event_log_hazard = jnp.take_along_axis(log_hazard, safe_index, axis=-2)[..., 0, :]
    event = log_susceptible + log_survival_before + event_log_hazard
    none = _logaddexp(log_rest, log_susceptible + log_survival_before)
    valid = (event_index >= -1) & (event_index < days)
    return jnp.where(valid, jnp.where(is_event, event, none), -jnp.inf)
