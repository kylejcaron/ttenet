"""Native event-time laws: one event date per unit, or counts per homogeneous pool.

Both laws share one representation. For a unit or pool ``c`` and window of
``T`` dates, ``log_mass[..., t, c]`` is the unnormalized log weight of a first
event on date ``t`` and ``log_tail[..., c]`` the weight of no event by the
window end (cure and surviving susceptibles, marginalized together). The
normalized categorical over ``T + 1`` categories is the law of one unit; a
pool of ``n`` units allocates ``Multinomial(n, ...)`` counts over the same
categories, and ``n = 1`` is the unit law exactly.

``EventTime`` also accepts the shared ``SurvivalKernel`` (log hazard, log
survival step and susceptibility logits) and then scores observations without
materializing ``log_mass``; its window and prefix surgeries stay in kernel
form by shifting the susceptibility logits, so fitting, conditioning and
forecasting evaluate one law from one set of components.

Values are time-major ``[..., time, cohort]`` arrays: binary trajectories with
at most one event per unit for ``EventTime``, integer counts whose column
totals never exceed the pool for ``CohortEventTime``. Leading axes of the
parameters are batch axes; values broadcast against them, so an unbatched
observation scores every batch member.

Precision follows the JAX configuration where a law is built: int32/float32
by default, int64/float64 inside ``jax.enable_x64(True)``. Nothing here
changes global configuration.
"""

from __future__ import annotations

import math
from decimal import Decimal, localcontext

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from jax import lax, random
from numpyro.distributions import constraints
from numpyro.distributions.util import lazy_property
from numpyro.util import not_jax_tracer
from numpyro_forecast.surgery import prefix_condition, slice_time

from ttenet._count_sampling import allocate_counts
from ttenet.event_times import SurvivalKernel, _logaddexp, kernel_unit_log_prob, log1mexp


class _LogMass(constraints.ParameterFreeConstraint):
    """A log weight: finite, or exactly ``-inf`` for a structural zero."""

    def __call__(self, value):
        return jnp.isfinite(value) | jnp.isneginf(value)

    def feasible_like(self, prototype):
        return jnp.zeros_like(prototype)


def _integer_dtype():
    return jnp.dtype(jax.dtypes.canonicalize_dtype(np.int64))


def _trajectory_values(value):
    """Refuse host integer observations that JAX's current dtype would wrap."""
    if isinstance(value, (jax.Array, jax.core.Tracer)):
        return jnp.asarray(value)
    host = np.asarray(value)
    if host.dtype.kind in "iu" and host.size:
        bounds = np.iinfo(_integer_dtype())
        if int(host.min()) < bounds.min or int(host.max()) > bounds.max:
            raise ValueError(
                f"observations exceed the {_integer_dtype()} range of current JAX "
                "precision; use jax.enable_x64(True) for int64 values"
            )
    return jnp.asarray(host)


def _log_sum_mass(values, axis):
    """Log-sum along ``axis`` that is exactly ``-inf``, with finite gradients, for no mass.

    Only structural ``-inf`` entries are absent; a NaN operand stays NaN.
    """
    present = ~jnp.all(jnp.isneginf(values), axis=axis, keepdims=True)
    safe = jnp.where(present, values, 0.0)
    result = jax.scipy.special.logsumexp(safe, axis=axis, keepdims=True)
    return jnp.squeeze(jnp.where(present, result, -jnp.inf), axis=axis)


def _log_softmax(weights):
    """Normalized log weights over the category axis ``-2``, exact for the dominant one.

    With ``top`` the largest weight and ``rest`` the sum of ``exp(w - top)``
    over the others, ``log p_k = (w_k - top) - log1p(rest)``. The dominant
    category's share is a single ``log1p``; every other difference
    ``w_k - top`` is exact whenever ``w_k`` lies within a factor of two of
    ``top`` (Sterbenz), so no log probability is rounded at the scale of the
    raw weights. ``w - logsumexp(w)`` rounds at the ulp of the weights, and a
    count of ``2**53`` multiplies that: with weights near ``-700`` it is a
    full nat. Columns without mass stay ``-inf`` throughout (no NaN); a NaN
    weight keeps its column NaN rather than turning it into an impossibility.
    """
    index = jnp.argmax(weights, axis=-2, keepdims=True)
    top = jnp.take_along_axis(weights, index, axis=-2)
    present = ~jnp.isneginf(top)
    anchor = jnp.where(present, top, 0.0)
    shifted = jnp.where(present, weights - anchor, -jnp.inf)
    position = jnp.arange(weights.shape[-2])[:, None]
    rest = jnp.where(position == index, 0.0, jnp.exp(shifted)).sum(axis=-2, keepdims=True)
    return shifted - jnp.log1p(rest)


def _unit_event_index(value, days):
    """Per-unit validity and event date (``-1`` for no event) of binary trajectories."""
    if value.ndim < 2:
        raise ValueError("event-time values must be [..., time, cohort] arrays")
    binary = (value == 0) | (value == 1)
    events = value == 1
    count = events.sum(axis=-2)
    valid = jnp.all(binary, axis=-2) & (count <= 1)
    if days == 0:
        return valid, jnp.full(count.shape, -1, jnp.int32)
    date = jnp.where(count == 1, jnp.argmax(events, axis=-2), -1).astype(jnp.int32)
    return valid, date


def _prefix_length(data, days):
    """Length of a time-major prefix, refusing shapes that cannot be a prefix of ``days``."""
    if data.ndim < 2:
        raise ValueError("prefix data must be [..., time, cohort] arrays")
    start = data.shape[-2]
    if start > days:
        raise ValueError(f"prefix of {start} dates is longer than the {days}-date horizon")
    return start


def _refuse_infeasible_prefix(feasible, reason):
    """Refuse a concretely infeasible prefix; traced prefixes carry their sentinel instead."""
    if not_jax_tracer(feasible) and not bool(np.all(np.asarray(feasible))):
        raise ValueError(f"prefix is not a feasible observation of this law: {reason}")


def _contiguous_window(index, days):
    start, stop, step = index.indices(days)
    if step != 1:
        raise NotImplementedError("event-time windows must be contiguous")
    return start, max(start, stop)


class EventTime(dist.Distribution):
    """One event date, or none, per unit as a binary ``[time, cohort]`` trajectory.

    Build it from explicit ``log_mass`` ``[..., time, cohort]`` and
    ``log_tail`` ``[..., cohort]`` log weights, or from a ``SurvivalKernel``
    (``kernel=``), whose event masses are only formed when the categorical
    view (``log_mass``, ``mean``, ``log_probabilities``) is requested. A
    kernel normalizes every unit to total mass one, so the kernel route also
    carries ``massless`` ``[..., cohort]``: units whose conditional law has no
    mass at all (a traced prefix that is not a trajectory of this law). Every
    outcome of such a unit is impossible, including the empty suffix. Units
    are independent given the parameters; dates are not.
    """

    arg_constraints = {"log_mass": _LogMass(), "log_tail": _LogMass()}
    support = constraints.independent(constraints.boolean, 2)
    pytree_data_fields = ("log_mass", "log_tail", "kernel", "massless")

    def __init__(
        self, log_mass=None, log_tail=None, *, kernel=None, massless=None, validate_args=None
    ):
        if kernel is not None:
            if log_mass is not None or log_tail is not None:
                raise ValueError(
                    "EventTime takes either log_mass and log_tail or a kernel, not both"
                )
            if not isinstance(kernel, SurvivalKernel):
                raise TypeError("kernel must be a SurvivalKernel")
            hazard = jnp.asarray(kernel.log_hazard)
            stay = jnp.asarray(kernel.log_survival_step)
            logits = jnp.asarray(kernel.susceptibility_logits)
            if hazard.ndim < 2 or stay.ndim < 2:
                raise ValueError(
                    "kernel log_hazard and log_survival_step must be [..., time, cohort]"
                )
            if hazard.shape[-2] != stay.shape[-2]:
                raise ValueError("kernel log_hazard and log_survival_step must share the time axis")
            massless = jnp.asarray(False if massless is None else massless, dtype=bool)
            batch_shape = jnp.broadcast_shapes(
                hazard.shape[:-2], stay.shape[:-2], logits.shape[:-1], massless.shape[:-1]
            )
            cohorts = jnp.broadcast_shapes(
                hazard.shape[-1:], stay.shape[-1:], logits.shape[-1:], massless.shape[-1:]
            )
            event_shape = (hazard.shape[-2],) + cohorts
            self.kernel = SurvivalKernel(
                jnp.broadcast_to(hazard, batch_shape + event_shape),
                jnp.broadcast_to(stay, batch_shape + event_shape),
                jnp.broadcast_to(logits, batch_shape + cohorts),
            )
            self.massless = jnp.broadcast_to(massless, batch_shape + cohorts)
        else:
            if massless is not None:
                raise ValueError("massless units are a kernel-route property; pass a kernel")
            if log_mass is None:
                raise ValueError("EventTime requires log_mass and log_tail, or a kernel")
            if log_tail is None:
                raise ValueError("EventTime requires log_tail alongside log_mass")
            log_mass = jnp.asarray(log_mass)
            log_tail = jnp.asarray(log_tail)
            if log_mass.ndim < 2:
                raise ValueError("log_mass must be [..., time, cohort]")
            batch_shape = jnp.broadcast_shapes(log_mass.shape[:-2], log_tail.shape[:-1])
            cohorts = jnp.broadcast_shapes(log_mass.shape[-1:], log_tail.shape[-1:])
            event_shape = (log_mass.shape[-2],) + cohorts
            self.kernel = None
            self.massless = None
            self.log_mass = jnp.broadcast_to(log_mass, batch_shape + event_shape)
            self.log_tail = jnp.broadcast_to(log_tail, batch_shape + cohorts)
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @classmethod
    def from_kernel(cls, kernel, *, validate_args=None):
        return cls(kernel=kernel, validate_args=validate_args)

    @lazy_property
    def log_mass(self):
        return jnp.where(self.massless[..., None, :], -jnp.inf, self.kernel.log_mass)

    @lazy_property
    def log_tail(self):
        return jnp.where(self.massless, -jnp.inf, self.kernel.log_tail)

    @property
    def log_weights(self):
        return jnp.concatenate([self.log_mass, self.log_tail[..., None, :]], axis=-2)

    @property
    def log_probabilities(self):
        return _log_softmax(self.log_weights)

    @property
    def mean(self):
        return jnp.exp(self.log_probabilities[..., :-1, :])

    @property
    def variance(self):
        log_p = self.log_probabilities[..., :-1, :]
        return jnp.exp(log_p) * -jnp.expm1(log_p)

    @property
    def _float_dtype(self):
        source = self.log_mass if self.kernel is None else self.kernel.log_hazard
        return source.dtype

    def sample(self, key, sample_shape=()):
        """One inverse-CDF uniform per unit: a log-uniform threshold against the cumulative law.

        A unit whose law has no mass, or NaN mass, cannot draw a trajectory;
        its rows are the visible off-support sentinel ``-1`` rather than a
        plausible run of no events.
        """
        days, cohorts = self.event_shape
        shape = tuple(sample_shape) + self.batch_shape + (cohorts,)
        log_u = -random.exponential(key, shape, self._float_dtype)
        if self.kernel is None:
            weights = self.log_weights
            cumulative = lax.cumlogsumexp(weights, axis=weights.ndim - 2)
            total = cumulative[..., -1, :]
            threshold = (total + log_u)[..., None, :]
            date = jnp.sum(cumulative[..., :-1, :] < threshold, axis=-2)
            undefined = ~jnp.isfinite(total)
        else:
            kernel = self.kernel
            log_susceptible = kernel.log_susceptible
            susceptible = log_u <= log_susceptible
            log_v = log_u - jnp.where(susceptible, log_susceptible, 0.0)
            survival = jnp.cumsum(kernel.log_survival_step, axis=-2)
            date = jnp.sum(survival > log_v[..., None, :], axis=-2)
            undefined = self.massless | jnp.isnan(kernel.log_tail)
            if days:
                # A date without event mass is never an event, whatever the survival
                # bookkeeping says: this keeps sampling consistent with log_prob.
                hazard = jnp.broadcast_to(kernel.log_hazard, shape[:-1] + (days, cohorts))
                index = jnp.clip(date, 0, days - 1)[..., None, :]
                selected = jnp.take_along_axis(hazard, index, axis=-2)[..., 0, :]
                susceptible = susceptible & ~((date < days) & jnp.isneginf(selected))
                undefined = undefined | jnp.any(jnp.isnan(kernel.log_hazard), axis=-2)
            date = jnp.where(susceptible, date, days)
        trajectory = (jnp.arange(days)[:, None] == date[..., None, :]).astype(_integer_dtype())
        return jnp.where(undefined[..., None, :], -1, trajectory)

    def unit_log_prob(self, value):
        """Log probability of each unit's trajectory; ``-inf`` off the support."""
        value = _trajectory_values(value)
        if value.ndim < 2:
            raise ValueError("event-time values must be [..., time, cohort] arrays")
        days, cohorts = self.event_shape
        shape = jnp.broadcast_shapes(value.shape[:-2], self.batch_shape)
        value = jnp.broadcast_to(value, shape + (days, cohorts))
        valid, date = _unit_event_index(value, days)
        if days == 0:
            log_tail = jnp.broadcast_to(self.log_tail, shape + (cohorts,))
            certain = jnp.where(jnp.isneginf(log_tail), -jnp.inf, log_tail - log_tail)
            return jnp.where(valid, certain, -jnp.inf).astype(self._float_dtype)
        if self.kernel is None:
            probabilities = jnp.broadcast_to(self.log_probabilities, shape + (days + 1, cohorts))
            index = jnp.where(date < 0, days, date)[..., None, :]
            selected = jnp.take_along_axis(probabilities, index, axis=-2)[..., 0, :]
        else:
            selected = kernel_unit_log_prob(self.kernel, date)
            valid = valid & ~self.massless
        return jnp.where(valid, selected, -jnp.inf)

    def log_prob(self, value):
        return self.unit_log_prob(value).sum(axis=-1)


def _window_logits(kernel, log_before):
    """Susceptibility logits of the law restricted to a window after ``log_before`` survival.

    Events before the window join its no-event category: with ``a`` the
    susceptible and ``b`` the cured share, the window law has ``a' = a S``
    and ``b' = b + a (1 - S)`` where ``S = exp(log_before)``.
    """
    log_a = kernel.log_susceptible
    log_b = kernel.log_rest
    log_rest = _logaddexp(log_b, log_a + log1mexp(log_before))
    return log_a + log_before - log_rest


@slice_time.register
def _(law: EventTime, index: slice) -> EventTime:
    """Exact window marginal: events outside the window join its no-event mass."""
    start, stop = _contiguous_window(index, law.event_shape[0])
    if law.kernel is None:
        weights = law.log_weights
        outside = jnp.concatenate([weights[..., :start, :], weights[..., stop:, :]], axis=-2)
        return EventTime(weights[..., start:stop, :], _log_sum_mass(outside, -2))
    kernel = law.kernel
    logits = kernel.susceptibility_logits
    if start:
        logits = _window_logits(kernel, kernel.log_survival_step[..., :start, :].sum(axis=-2))
    return EventTime(
        kernel=SurvivalKernel(
            kernel.log_hazard[..., start:stop, :],
            kernel.log_survival_step[..., start:stop, :],
            logits,
        ),
        massless=law.massless,
    )


def _prefix_log_survival(log_survival_step, stop):
    """Per-unit log stay summed over the prefix dates strictly before ``stop``."""
    before = jnp.arange(log_survival_step.shape[-2])[:, None] < stop[..., None, :]
    return jnp.sum(jnp.where(before, log_survival_step, 0.0), axis=-2)


@prefix_condition.register
def _(law: EventTime, data) -> EventTime:
    """Exact suffix law given the observed prefix.

    Units that fired are absorbed (no further event, certainly); units still
    pending condition on their prefix survival. A prefix outside the law's
    support (entries other than 0/1, more than one event, an event on a date
    without mass, or no event where surviving the prefix has no mass) is
    refused when it is concrete; under tracing such a unit keeps no mass at
    all (explicit weights ``-inf``, or ``massless`` on the kernel route), so
    every suffix is impossible rather than a certain cure. NaN components
    are neither refused nor masked: they stay NaN.
    """
    data = _trajectory_values(data)
    days = law.event_shape[0]
    start = _prefix_length(data, days)
    valid, date = _unit_event_index(data, start)
    _refuse_infeasible_prefix(valid, "entries must be 0/1 with at most one event per unit")
    pending = valid & (date < 0)
    absorbed = valid & (date >= 0)
    shape = jnp.broadcast_shapes(valid.shape[:-1], law.batch_shape)
    cohorts = law.event_shape[1:]
    safe_index = jnp.broadcast_to(
        jnp.clip(date, 0, max(days - 1, 0))[..., None, :], shape + (1,) + cohorts
    )

    def observed_rows(values):
        """Per unit, the row of ``values`` on its event date (meaningless when pending)."""
        if days == 0:
            return jnp.zeros(shape + cohorts, values.dtype)
        rows = jnp.broadcast_to(values, shape + (days,) + cohorts)
        return jnp.take_along_axis(rows, safe_index, axis=-2)[..., 0, :]

    if law.kernel is None:
        remaining = jnp.all(jnp.isneginf(law.log_mass[..., start:, :]), axis=-2)
        remaining = remaining & jnp.isneginf(law.log_tail)
        exhausted = (absorbed & jnp.isneginf(observed_rows(law.log_mass))) | (pending & remaining)
        _refuse_infeasible_prefix(~exhausted, "the observed prefix has no mass under this law")
        pending = pending & ~exhausted
        absorbed = absorbed & ~exhausted
        log_mass = jnp.where(pending[..., None, :], law.log_mass[..., start:, :], -jnp.inf)
        log_tail = jnp.where(pending, law.log_tail, jnp.where(absorbed, 0.0, -jnp.inf))
        return EventTime(log_mass, log_tail)
    kernel = law.kernel
    logits = kernel.susceptibility_logits
    stop = jnp.where(absorbed, date, start)
    log_survival = _prefix_log_survival(kernel.log_survival_step, stop)
    observed_hazard = observed_rows(kernel.log_hazard)
    # The observed event needs susceptible mass, prefix survival and hazard on its
    # date; a pending unit needs some way to have had no event: cure or survival.
    event_exhausted = absorbed & (
        jnp.isneginf(logits) | jnp.isneginf(log_survival) | jnp.isneginf(observed_hazard)
    )
    stay_exhausted = pending & jnp.isposinf(logits) & jnp.isneginf(log_survival)
    exhausted = event_exhausted | stay_exhausted
    _refuse_infeasible_prefix(~exhausted, "the observed prefix has no mass under this law")
    massless = law.massless | ~valid | exhausted
    survived = logits + log_survival
    logits = jnp.where(pending, survived, jnp.where(absorbed, -jnp.inf, jnp.inf))
    logits = jnp.where(massless, 0.0, logits)
    return EventTime(
        kernel=SurvivalKernel(
            kernel.log_hazard[..., start:, :], kernel.log_survival_step[..., start:, :], logits
        ),
        massless=massless,
    )


# ----------------------------------------------------------------- counted pools


def _integer_counts(total_count):
    """Counts in the traced integer dtype, refusing host values it cannot hold.

    Host arrays are cast deliberately: JAX would otherwise truncate an int64
    ``2**53 + 1`` to int32 ``1`` with only a warning when x64 is disabled, and
    wrap a negative value below the int32 range into a plausible pool.
    """
    if isinstance(total_count, jax.Array):
        counts = total_count
    else:
        host = np.asarray(total_count)
        if host.dtype.kind not in "iu":
            raise TypeError(f"total_count must be integer-valued, got dtype {host.dtype}")
        if host.size and int(host.min()) < 0:
            raise ValueError(f"total_count must be nonnegative, got {int(host.min())}")
        dtype = _integer_dtype()
        limit = np.iinfo(dtype).max
        if host.size and int(host.max()) > limit:
            raise ValueError(
                f"total_count {int(host.max())} exceeds the {np.dtype(dtype)} range of the "
                "current JAX precision; build the law inside jax.enable_x64(True)"
            )
        counts = jnp.asarray(host.astype(dtype))
    if not jnp.issubdtype(counts.dtype, jnp.integer):
        raise TypeError(f"total_count must be integer-typed, got dtype {counts.dtype}")
    return counts


_LOG_2PI = math.log(2.0 * math.pi)
_STIRLING_SERIES = (1.0 / 12.0, 1.0 / 360.0, 1.0 / 1260.0, 1.0 / 1680.0, 1.0 / 1188.0)


def _stirling_table(last):
    """delta(m) for m = 1..last from exact factorials, correctly rounded to float64."""
    with localcontext() as context:
        context.prec = 40
        half_log_2pi = (2 * Decimal("3.141592653589793238462643383279502884197")).ln() / 2
        return np.array(
            [
                float(
                    Decimal(math.factorial(m)).ln()
                    - ((Decimal(m) + Decimal("0.5")) * Decimal(m).ln() - m + half_log_2pi)
                )
                for m in range(1, last + 1)
            ]
        )


_STIRLING_TABLE = _stirling_table(15)


def _stirling_error(count):
    """delta(m) = log(m!) - [(m + 1/2) log m - m + log(2 pi) / 2] for m >= 1.

    Tabulated through 15 and the asymptotic series beyond, so no factorial or
    lgamma of a large argument is ever formed: delta(2**53) is ~1e-17 while
    the lgamma values it replaces are ~3e17 with an ulp of 64.
    """
    tabulated = count <= 15
    index = jnp.clip(count, 1, 15).astype(jnp.int32) - 1
    safe = jnp.where(tabulated, 16.0, count)
    inverse_square = 1.0 / (safe * safe)
    s0, s1, s2, s3, s4 = _STIRLING_SERIES
    nested = s3 - s4 * inverse_square
    nested = s2 - nested * inverse_square
    nested = s1 - nested * inverse_square
    series = (s0 - nested * inverse_square) / safe
    table = jnp.asarray(_STIRLING_TABLE, count.dtype)
    return jnp.where(tabulated, table[index], series)


_DEVIANCE_TERMS = 10
"""Series terms for the near-mode deviance; the ratio v**2 is below 0.01 there."""


def _deviance(x, mu, log_mu):
    """Loader's ``x log(x / mu) + mu - x``, evaluated without cancellation.

    Near the mode (``|x - mu| < (x + mu) / 10``) the series in
    ``v = (x - mu) / (x + mu)``,

        (x - mu) v + 2 x sum_{j >= 1} v**(2j + 1) / (2j + 1),

    keeps relative accuracy where ``x log(x / mu)`` and ``mu - x`` would
    cancel. Elsewhere the direct form takes ``log(x / mu)`` from the ratio
    (an absolute error of a few eps, so ``x`` multiplies eps rather than
    the eps-scaled ``log n`` of ``log x - log mu``) and falls back to
    ``log x - log_mu`` where ``mu`` is below ``x / 2**100``: there the
    ratio form is at its conditioning floor anyway, and the quotient can
    overflow when ``mu`` is a subnormal that the backend did not flush,
    which would turn a possible count into an infinite deviance. ``x = 0``
    contributes ``mu``; ``x > 0`` against ``log_mu = -inf`` is the caller's
    impossibility flag and only needs to stay finite here. Both branches
    are computed from guarded operands so the unselected branch is finite
    and its gradient is exactly zero.
    """
    positive = x > 0
    near = positive & (jnp.abs(x - mu) < 0.1 * (x + mu))
    v = jnp.where(near, (x - mu) / jnp.where(near, x + mu, 1.0), 0.0)
    v2 = v * v
    term = 2.0 * x * v
    series = (x - mu) * v
    for j in range(1, _DEVIANCE_TERMS + 1):
        term = term * v2
        series = series + term / (2 * j + 1)
    safe_x = jnp.where(positive, x, 1.0)
    tiny = mu < safe_x * 2.0**-100
    ratio = jnp.log(safe_x / jnp.where(tiny, 1.0, mu))
    underflow = jnp.log(safe_x) - jnp.where(jnp.isneginf(log_mu), 0.0, log_mu)
    log_ratio = jnp.where(tiny, underflow, ratio)
    direct = jnp.where(positive, x * log_ratio + mu - x, mu)
    return jnp.where(near, series, direct)


def _multinomial_log_pmf(occupancy, total, log_p):
    """Saddle-point form of the multinomial log mass.

    With ``mu_k = n p_k`` and the deviance ``D = sum_k [x_k log(x_k / mu_k) + mu_k - x_k]``
    over every category (``x_k = 0`` contributes ``mu_k``; the linear terms
    sum to zero because ``sum_k mu_k = n``):

        log P = -D + [log(2 pi n) - sum_{x_k > 0} log(2 pi x_k)] / 2
                + delta(n) - sum_{x_k > 0} delta(x_k)

    with ``delta`` the Stirling error. Each deviance term is stationary at
    ``x_k = mu_k``, so the rounding of ``mu_k`` costs ``|x_k - mu_k| eps``
    rather than ``x_k eps``: the multinomial's own gradient conditioning.
    Nothing forms a factorial, a large lgamma difference, or ``x_k log p_k``.
    Rows with ``n = 0`` have log mass 0; a positive count on a ``-inf``
    category gives ``-inf``.
    """
    dtype = log_p.dtype
    x = occupancy.astype(dtype)
    positive = occupancy > 0
    possible = ~jnp.isneginf(log_p)
    pool = jnp.where(total > 0, total, 1).astype(dtype)
    log_pool = jnp.log(pool)
    mu = pool[..., None, :] * jnp.exp(log_p)
    deviance = _deviance(x, mu, log_pool[..., None, :] + log_p).sum(axis=-2)
    log_count = jnp.where(positive, jnp.log(jnp.where(positive, x, 1.0)), 0.0).sum(axis=-2)
    stirling = jnp.where(positive, _stirling_error(jnp.where(positive, x, 1.0)), 0.0).sum(axis=-2)
    occupied = positive.sum(axis=-2)
    value = (
        0.5 * (log_pool - log_count)
        + 0.5 * _LOG_2PI * (1 - occupied)
        + _stirling_error(pool)
        - stirling
        - deviance
    )
    impossible = jnp.any(positive & ~possible, axis=-2)
    return jnp.where(total > 0, jnp.where(impossible, -jnp.inf, value), 0.0)


def _feasible_counts(value, total):
    """Counts in the pool dtype and a per-cohort feasibility flag, without overflow.

    Feasible means every count is a whole number in ``[0, pool]`` and the
    column total is at most the pool. The total is validated through the
    running sum: with nonnegative addends a wrapped partial sum is the only
    way the sequence can decrease, so an overflowing column is rejected
    instead of wrapping into a plausible residual. Float values must lie in
    their dtype's exact-integer range.
    """
    limit = total[..., None, :]
    if jnp.issubdtype(value.dtype, jnp.floating):
        # Range-check in the value's own dtype: a float beyond the pool dtype's
        # range must never reach the integer cast.
        exact = 2.0 ** (jnp.finfo(value.dtype).nmant + 1)
        whole = jnp.isfinite(value) & (value == jnp.floor(value)) & (jnp.abs(value) <= exact)
        in_range = whole & (value >= 0) & (value <= limit.astype(value.dtype))
    else:
        in_range = (value >= 0) & (value <= limit)
    counts = jnp.where(in_range, value, 0).astype(total.dtype)
    running = jnp.cumsum(counts, axis=-2)
    ordered = jnp.all(running[..., 1:, :] >= running[..., :-1, :], axis=-2)
    feasible = jnp.all(in_range, axis=-2) & ordered & (counts.sum(axis=-2) <= total)
    return counts, feasible


class CohortEventTime(dist.Distribution):
    """Integer allocation of homogeneous pools across event dates.

    ``log_mass[..., time, cohort]`` and ``log_tail[..., cohort]`` are the
    unnormalized log weights of ``EventTime``; ``total_count[..., cohort]`` is
    the integer pool. A value is the count array ``[..., time, cohort]``; the
    difference between the pool and a column's total is the no-event
    residual, which is never stored. Pools are independent given the
    parameters; dates are not. Work and memory scale with dates and pools,
    never with the pool sizes.
    """

    arg_constraints = {
        "log_mass": _LogMass(),
        "log_tail": _LogMass(),
        "total_count": constraints.nonnegative_integer,
    }
    support = constraints.independent(constraints.nonnegative_integer, 2)

    def __init__(self, log_mass, log_tail, total_count, *, validate_args=None):
        log_mass = jnp.asarray(log_mass)
        log_tail = jnp.asarray(log_tail)
        total_count = _integer_counts(total_count)
        if log_mass.ndim < 2:
            raise ValueError("log_mass must be [..., time, cohort]")
        batch_shape = jnp.broadcast_shapes(
            log_mass.shape[:-2], log_tail.shape[:-1], total_count.shape[:-1]
        )
        cohorts = jnp.broadcast_shapes(
            log_mass.shape[-1:], log_tail.shape[-1:], total_count.shape[-1:]
        )
        event_shape = (log_mass.shape[-2],) + cohorts
        self.log_mass = jnp.broadcast_to(log_mass, batch_shape + event_shape)
        self.log_tail = jnp.broadcast_to(log_tail, batch_shape + cohorts)
        self.total_count = jnp.broadcast_to(total_count, batch_shape + cohorts)
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @property
    def log_weights(self):
        return jnp.concatenate([self.log_mass, self.log_tail[..., None, :]], axis=-2)

    @property
    def log_probabilities(self):
        return _log_softmax(self.log_weights)

    @property
    def mean(self):
        return self.total_count[..., None, :] * jnp.exp(self.log_probabilities[..., :-1, :])

    @property
    def variance(self):
        log_p = self.log_probabilities[..., :-1, :]
        return self.total_count[..., None, :] * jnp.exp(log_p) * -jnp.expm1(log_p)

    def sample(self, key, sample_shape=()):
        days, cohorts = self.event_shape
        shape = tuple(sample_shape) + self.batch_shape
        weights = jnp.broadcast_to(self.log_weights, shape + (days + 1, cohorts))
        total = jnp.broadcast_to(self.total_count, shape + (cohorts,))
        return allocate_counts(key, weights, total)

    def cohort_log_prob(self, value):
        """Multinomial log mass of each pool's counts; ``-inf`` off the support."""
        value = _trajectory_values(value)
        if value.ndim < 2:
            raise ValueError("count values must be [..., time, cohort] arrays")
        days, cohorts = self.event_shape
        shape = jnp.broadcast_shapes(value.shape[:-2], self.batch_shape)
        value = jnp.broadcast_to(value, shape + (days, cohorts))
        total = jnp.broadcast_to(self.total_count, shape + (cohorts,))
        counts, feasible = _feasible_counts(value, total)
        residual = total - counts.sum(axis=-2)
        occupancy = jnp.concatenate([counts, residual[..., None, :]], axis=-2)
        log_p = jnp.broadcast_to(self.log_probabilities, shape + (days + 1, cohorts))
        return jnp.where(feasible, _multinomial_log_pmf(occupancy, total, log_p), -jnp.inf)

    def log_prob(self, value):
        return self.cohort_log_prob(value).sum(axis=-1)


@slice_time.register
def _(law: CohortEventTime, index: slice) -> CohortEventTime:
    """Exact window marginal: dates outside the window join its no-event category."""
    start, stop = _contiguous_window(index, law.event_shape[0])
    weights = law.log_weights
    outside = jnp.concatenate([weights[..., :start, :], weights[..., stop:, :]], axis=-2)
    return CohortEventTime(weights[..., start:stop, :], _log_sum_mass(outside, -2), law.total_count)


@prefix_condition.register
def _(law: CohortEventTime, data) -> CohortEventTime:
    """Exact suffix law: observed counts leave the pool; remaining dates renormalize.

    A prefix outside the law's support (negative or fractional counts, more
    events than the pool including column sums that would overflow, events
    on a date without mass, or units left over with no remaining mass to
    hold them) is refused when it is concrete. Under tracing it yields a
    pool of ``-1``, which scores every suffix ``-inf``, has a negative mean
    and allocates negative counts, so it is never mistaken for a pool with
    no events left. NaN weights are neither refused nor masked.
    """
    data = _trajectory_values(data)
    start = _prefix_length(data, law.event_shape[0])
    counts, feasible = _feasible_counts(data, law.total_count)
    _refuse_infeasible_prefix(feasible, "counts must be whole, nonnegative and within the pool")
    remaining = law.total_count - counts.sum(axis=-2)
    on_structural_zero = jnp.any((counts > 0) & jnp.isneginf(law.log_mass[..., :start, :]), axis=-2)
    nothing_left = jnp.isneginf(law.log_tail) & jnp.all(
        jnp.isneginf(law.log_mass[..., start:, :]), axis=-2
    )
    exhausted = on_structural_zero | ((remaining > 0) & nothing_left)
    _refuse_infeasible_prefix(
        ~(feasible & exhausted), "the observed prefix has no mass under this law"
    )
    remaining = jnp.where(feasible & ~exhausted, remaining, -1)
    return CohortEventTime(law.log_mass[..., start:, :], law.log_tail, remaining)
