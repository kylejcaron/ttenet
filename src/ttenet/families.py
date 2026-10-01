"""Event-time families: typed tail declarations and the packaged discretized Weibull.

A family is any object with a NumPyro ``model(inputs, shared) -> EventLaw``
and a static ``tail`` declaration. The model sees :class:`TimingInputs` only
-- elapsed ages and regressors, never realized outcomes, calendars or
exposure -- samples its own named sites and returns the susceptible timing
law with the susceptibility logits. Exposure, closures, cure
marginalization, entry conditioning, native observation and count
propagation stay in the shared core, so swapping the family never touches
fitting, the native distributions or the network.

The tail says what the law does past any age the data describe:
:class:`ProperTail` (every susceptible unit eventually fires under
continuing exposure), :class:`FiniteTail` (fully specified through
``last_age``) or :class:`UnknownTail` (an unresolved residual that cannot
support eventual expectations). Nothing here infers a tail from a law.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Protocol

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from .event_times import EventLaw, TimingInputs, TimingLaw, log1mexp


def _finite_support(last_age: Any) -> int:
    if isinstance(last_age, bool) or not isinstance(last_age, (int, np.integer)) or last_age < 0:
        raise ValueError("FiniteTail.last_age must be a nonnegative integer")
    return int(last_age)


@dataclass(frozen=True)
class ProperTail:
    """Under continuing exposure the susceptible event probability tends to one."""


@dataclass(frozen=True)
class FiniteTail:
    """The susceptible law is fully specified through ``last_age``; nothing fires later."""

    last_age: int

    def __post_init__(self):
        object.__setattr__(self, "last_age", _finite_support(self.last_age))


@dataclass(frozen=True)
class UnknownTail:
    """An unresolved residual category that cannot support eventual expectations."""


type TailBehavior = ProperTail | FiniteTail | UnknownTail

_TAILS = (ProperTail, FiniteTail, UnknownTail)


class Family(Protocol):
    """Structural interface of an event-time family.

    ``model`` is a NumPyro model of the stage's law and ``tail`` its static
    tail declaration. Any object with these two attributes is a family;
    :class:`EventFamily` wraps a plain function and :class:`WeibullFamily`
    is the packaged law.
    """

    @property
    def tail(self) -> TailBehavior: ...

    def model(self, inputs: TimingInputs, shared: Any) -> EventLaw: ...


def validate_family(family: Any) -> Any:
    """Refuse anything that is not a family with an exact tail declaration.

    Checks that ``family.model`` is callable and that ``family.tail`` is one
    of :class:`ProperTail`, :class:`FiniteTail` (with a nonnegative integer
    ``last_age``) or :class:`UnknownTail` -- the exact classes, not
    look-alikes or subclasses -- without running the model. Returns the
    family unchanged so callers can validate inline.
    """
    if not callable(getattr(family, "model", None)):
        raise TypeError(
            "family.model must be a callable NumPyro model (inputs, shared) -> EventLaw"
        )
    tail = getattr(family, "tail", None)
    if type(tail) not in _TAILS:
        raise ValueError("family.tail must be ProperTail(), FiniteTail(last_age) or UnknownTail()")
    return family


@dataclass(frozen=True)
class EventFamily:
    """A plain NumPyro function ``(inputs, shared) -> EventLaw`` with its tail declaration."""

    model: Callable[[TimingInputs, Any], EventLaw]
    tail: TailBehavior

    def __post_init__(self):
        validate_family(self)


def _fixed_or_distribution(name: str, prior: Any, *, positive: bool) -> Any:
    """A scalar NumPyro distribution, or a fixed real scalar normalized to ``float``."""
    if isinstance(prior, dist.Distribution):
        if prior.shape() != ():
            raise ValueError(f"{name} must be a scalar NumPyro distribution")
        return prior
    array = np.asarray(prior)
    real = np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.floating)
    if array.ndim != 0 or not real:
        raise TypeError(f"{name} must be a NumPyro distribution or a fixed real number")
    value = float(array)
    if not np.isfinite(value) or (positive and value <= 0.0):
        bound = "a finite positive" if positive else "a finite"
        raise ValueError(f"a fixed {name} must be {bound} number")
    return value


def _site(name: str, prior: Any) -> Any:
    """Sample ``name`` from a distribution, or use the fixed value without a site."""
    if isinstance(prior, dist.Distribution):
        return numpyro.sample(name, prior)
    return jnp.asarray(prior)


@jax.custom_jvp
def _negative_exp(value: Any) -> Any:
    """Negative cumulative hazard; overflow is a certain-event boundary."""
    return -jnp.exp(value)


@_negative_exp.defjvp
def _negative_exp_jvp(primals, tangents):
    (value,), (tangent,) = primals, tangents
    result = _negative_exp(value)
    boundary = jnp.isneginf(result)
    slope = jnp.where(boundary, 0.0, result)
    tangent = jnp.where(boundary, 0.0, tangent)
    return result, slope * tangent


def _timing_from_log_increment(log_increment: Any) -> TimingLaw:
    """Convert a log cumulative-hazard increment without losing tiny hazards."""
    log_stay = _negative_exp(log_increment)
    # Below machine epsilon, log(1 - exp(-delta)) rounds to log(delta).
    tiny = log_increment < jnp.log(jnp.finfo(log_increment.dtype).eps)
    log_hazard = jnp.where(tiny, log_increment, log1mexp(log_stay))
    return TimingLaw(log_hazard, log_stay)


def _weibull_log_increment(age: Any, log_scale: Any, shape: Any) -> Any:
    """Log of ``((a + 1) / scale)**shape - (a / scale)**shape``.

    Use the power ratio rather than subtracting nearly equal cumulative
    hazards. The age-zero bin is explicit; neither path exponentiates a
    potentially overflowing cumulative hazard.
    """
    positive = age > 0
    safe_age = jnp.where(positive, age, 1)
    log_ratio = jnp.log(shape) + jnp.log(jnp.log1p(1.0 / safe_age))
    ratio = _timing_from_log_increment(log_ratio)
    # log(expm1(x)) = x + log(1 - exp(-x)); reuse the stable small-x logs.
    log_expm1_ratio = ratio.log_hazard - ratio.log_survival_step
    later = shape * (jnp.log(safe_age) - log_scale) + log_expm1_ratio
    first = -shape * log_scale
    increment = jnp.where(positive, later, first)
    valid = (shape > 0) & jnp.isfinite(shape) & jnp.isfinite(log_scale)
    return jnp.where(valid, increment, jnp.nan)


def _regressor_effect(features: Any, weights: Any) -> Any:
    return jnp.tensordot(features, weights, axes=([-1], [-1]))


@dataclass(frozen=True)
class WeibullFamily:
    """A discretized Weibull timing law with a logistic susceptibility.

    With scale ``lambda`` and shape ``k`` the continuous susceptible survival
    is ``log S(a) = -(a / lambda) ** k`` for ``a >= 0``; each day's law is the
    bin ``[a, a + 1)`` of that curve. Its analytic cumulative-hazard increment
    avoids subtracting nearly equal survival values and remains a certain
    event if the cumulative hazard overflows. Negative ages evaluate age
    zero; the core masks them. Shape ``2`` makes the hazard rise with age.

    ``scale_prior`` and ``shape_prior`` are scalar NumPyro distributions sampled at
    the sites ``"scale"`` and ``"shape"``, or fixed positive numbers with no
    site; ``susceptibility_prior`` is the susceptibility logit intercept,
    likewise sampled at ``"cure_intercept"`` or fixed. Time-varying
    regressors scale each day's cumulative-hazard increment by
    ``exp(x_t @ beta)`` (``beta ~ Normal(0, 1)`` at ``"beta"``) and static
    regressors shift the logit by ``z @ cure_beta`` (``cure_beta ~ Normal(0,
    1)`` at ``"cure_beta"``); both sites exist only when regressors are
    supplied. The tail is proper: under continuing exposure every
    susceptible unit eventually fires.

    Fixed values are concrete real scalars (Python, NumPy or JAX) and are
    stored as plain ``float``; distributions are stored as given. The
    family is therefore immutable, hashable static data -- a fitted stage
    keeps it as pytree metadata -- and never holds an array.
    """

    scale_prior: dist.Distribution | float
    shape_prior: dist.Distribution | float = 2.0
    susceptibility_prior: dist.Distribution | float = dist.Normal(0.0, 2.0)

    tail: ClassVar[TailBehavior] = ProperTail()

    def __post_init__(self):
        for name, positive in (
            ("scale_prior", True),
            ("shape_prior", True),
            ("susceptibility_prior", False),
        ):
            value = _fixed_or_distribution(name, getattr(self, name), positive=positive)
            object.__setattr__(self, name, value)

    def model(self, inputs: TimingInputs, shared: Any) -> EventLaw:
        scale = _site("scale", self.scale_prior)
        shape = _site("shape", self.shape_prior)
        ages = jnp.asarray(inputs.ages)
        age = jnp.maximum(ages, 0)
        log_scale = jnp.log(scale)
        log_increment = _weibull_log_increment(age, log_scale, shape)
        width = inputs.features.shape[-1]
        if width:
            beta = numpyro.sample("beta", dist.Normal(0.0, 1.0).expand([width]).to_event(1))
            log_increment = log_increment + _regressor_effect(inputs.features, beta)
        logits = _site("cure_intercept", self.susceptibility_prior)
        cure_width = inputs.cure_features.shape[-1]
        if cure_width:
            cure_beta = numpyro.sample(
                "cure_beta", dist.Normal(0.0, 1.0).expand([cure_width]).to_event(1)
            )
            logits = logits + _regressor_effect(inputs.cure_features, cure_beta)
        units = ages.shape[:-2] + ages.shape[-1:]
        return EventLaw(_timing_from_log_increment(log_increment), jnp.broadcast_to(logits, units))
