"""Experimental survival convolutions using NumPyro Forecast's native protocol.

This example module is not a replacement for TTENet's public API. EventTime
represents one event date (or no event) per unit, not independent daily counts.
A cure kernel can be used directly by predict(), or composed with another
kernel to obtain exact expected downstream events.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from jax import random
from jax.nn import log_sigmoid
from jax.scipy.special import logsumexp
from numpyro.distributions import constraints
from numpyro_forecast.surgery import prefix_condition, slice_time


def logsumexp_or_zero_mass(values, axis, keepdims=False):
    """Log-sum with an exact -inf result and finite derivatives for all-zero mass."""
    present = jnp.any(jnp.isfinite(values), axis=axis, keepdims=True)
    safe = jnp.where(present, values, 0.0)
    result = jnp.where(present, logsumexp(safe, axis=axis, keepdims=True), -jnp.inf)
    return result if keepdims else jnp.squeeze(result, axis=axis)


class _LogMass(constraints.ParameterFreeConstraint):
    def __call__(self, value):
        return jnp.isfinite(value) | jnp.isneginf(value)

    def feasible_like(self, prototype):
        return jnp.zeros_like(prototype)


class EventTime(dist.Distribution):
    """A categorical event date exposed as a binary [day, unit] trajectory.

    log_mass[..., day, unit] contains first-event masses; log_tail[..., unit]
    contains the probability of no event by the window end, INCLUDING cure.
    Both are log weights, normalized together. Leading axes are distribution
    batch axes. Units are independent given parameters, dates are not.
    """

    arg_constraints = {"log_mass": _LogMass(), "log_tail": _LogMass()}
    support = constraints.independent(constraints.boolean, 2)

    def __init__(self, log_mass, log_tail, *, validate_args=None):
        self.log_mass = jnp.asarray(log_mass)
        self.log_tail = jnp.broadcast_to(
            log_tail, self.log_mass.shape[:-2] + self.log_mass.shape[-1:]
        )
        super().__init__(
            self.log_mass.shape[:-2], self.log_mass.shape[-2:], validate_args=validate_args
        )

    @property
    def log_weights(self):
        return jnp.concatenate([self.log_mass, self.log_tail[..., None, :]], axis=-2)

    @property
    def log_probabilities(self):
        weights = self.log_weights
        return weights - logsumexp(weights, axis=-2, keepdims=True)

    @property
    def mean(self):
        return jnp.exp(self.log_probabilities[..., :-1, :])

    def sample(self, key, sample_shape=()):
        days, units = self.event_shape
        date = random.categorical(
            key,
            jnp.swapaxes(self.log_weights, -1, -2),
            shape=sample_shape + self.batch_shape + (units,),
        )
        return (jnp.arange(days)[:, None] == date[..., None, :]).astype(jnp.int32)

    def unit_log_prob(self, value):
        value = jnp.asarray(value)
        days, units = self.event_shape
        shape = jnp.broadcast_shapes(value.shape[:-2], self.batch_shape)
        if days == 0:
            return jnp.zeros(shape + (units,))
        event = jnp.any(value == 1, axis=-2)
        date = jnp.where(event, jnp.argmax(value, axis=-2), days)
        probabilities = jnp.broadcast_to(self.log_probabilities, shape + (days + 1, units))
        index = jnp.broadcast_to(date[..., None, :], shape + (1, units))
        selected = jnp.take_along_axis(probabilities, index, axis=-2)[..., 0, :]
        valid = jnp.all((value == 0) | (value == 1), axis=-2) & (value.sum(axis=-2) <= 1)
        return jnp.where(valid, selected, -jnp.inf)

    def log_prob(self, value):
        return self.unit_log_prob(value).sum(axis=-1)


@slice_time.register
def _(law: EventTime, index: slice) -> EventTime:
    """Exact window marginal: events outside the window join its no-event mass."""
    start, stop, step = index.indices(law.event_shape[0])
    if step != 1:
        raise NotImplementedError("event-time windows must be contiguous")
    weights = law.log_weights
    outside = jnp.concatenate([weights[..., :start, :], weights[..., stop:, :]], axis=-2)
    return EventTime(weights[..., start:stop, :], logsumexp_or_zero_mass(outside, -2))


@prefix_condition.register
def _(law: EventTime, data) -> EventTime:
    """Exact suffix law conditional on the observed event-free/absorbed state."""
    start = data.shape[-2]
    pending = ~jnp.any(data != 0, axis=-2)
    return EventTime(
        jnp.where(pending[..., None, :], law.log_mass[..., start:, :], -jnp.inf),
        jnp.where(pending, law.log_tail, 0.0),
    )


def survival_kernel(
    parameters, origin, features, cure_features, allowed, *, entry=None, deadline=None
):
    """Build exact first-event masses from susceptible hazards and cure logits.

    features [day, unit, feature] and allowed [day, unit] are calendar inputs.
    origin [*origin_batch, unit] retains each source date. Passing all possible
    origins builds K[source_day, event_day, unit] for a survival convolution;
    passing only actual origins avoids materializing that quadratic kernel.
    Selected survivors can have entry > origin; pre-entry survival conditions
    their probabilities without contributing a second observation likelihood.
    """
    origin = jnp.asarray(origin)
    entry = origin if entry is None else jnp.asarray(entry)
    day = jnp.arange(features.shape[0])[:, None]
    age = day - origin[..., None, :]
    hazard = parameters.age_logits[jnp.clip(age, 0, len(parameters.age_logits) - 1)]
    hazard = hazard + jnp.einsum("tnp,p->tn", features, parameters.beta)
    open_day = (age >= 0) & allowed
    if deadline is not None:
        open_day &= age <= deadline
    log_failure = jnp.where(open_day, log_sigmoid(-hazard), 0.0)
    log_before = jnp.cumsum(log_failure, axis=-2) - log_failure
    cure = parameters.cure_intercept + cure_features @ parameters.cure_beta
    log_pi, log_cured = log_sigmoid(cure), log_sigmoid(-cure)
    before_entry = day < entry[..., None, :]
    log_pre = jnp.where(before_entry, log_failure, 0.0).sum(axis=-2)
    log_entry_survival = jnp.logaddexp(log_cured, log_pi + log_pre)
    mass = jnp.where(
        open_day & ~before_entry,
        log_pi[..., None, :] + log_before + log_sigmoid(hazard) - log_entry_survival[..., None, :],
        -jnp.inf,
    )
    tail = jnp.logaddexp(log_cured, log_pi + log_failure.sum(axis=-2)) - log_entry_survival
    return EventTime(mass, tail)


def convolve_event_times(parent_mass, child_kernel):
    """Exact mean: sum over parent dates, retaining each unit, not a Poisson model.

    parent_mass [source_day, unit], child_kernel.mean [source_day, day, unit].
    A stationary lag-only kernel reduces to ordinary discrete convolution.
    """
    return jnp.einsum("sn,stn->tn", parent_mass, child_kernel.mean)
