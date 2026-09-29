"""JAX numerical primitives for discrete-time mixture-cure survival stages.

Every function here operates on a *single* posterior draw: the fields of
``StageParameters`` are un-batched (``age_logits`` is ``[K]``, ``beta`` is
``[P]``, ``cure_intercept`` is a scalar, ``cure_beta`` is ``[Q]``). Callers
that hold a leading posterior-draw axis (e.g. the output of
``models.fit_stage``) are responsible for ``jax.vmap``-ing these primitives
over that axis, or for indexing a single draw before calling them.

All functions are pure JAX and safe to trace, ``jit``, ``vmap``, and
differentiate. They perform no host-side validation and never attempt to
convert traced values to concrete NumPy arrays; shape/shape-compatibility
checks that require concrete inspection belong to host-side callers
(``models.make_observations``, ``models.fit_stage``).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp
from jax.nn import sigmoid


class StageParameters(NamedTuple):
    """One posterior draw of a stage's parameters.

    age_logits: ``[K]`` flexible age-baseline logits, one per age bin.
    beta: ``[P]`` linear effects for time-varying hazard regressors.
    cure_intercept: scalar susceptibility intercept.
    cure_beta: ``[Q]`` linear effects for static susceptibility regressors.

    A leading posterior-draw axis (``[draw, K]``, ``[draw, P]``, ``[draw]``,
    ``[draw, Q]``) is valid wherever a *batch* of draws is being carried
    around (e.g. ``models.StageFit.parameters``); the numerical primitives
    in this module consume one draw at a time.
    """

    age_logits: Any
    beta: Any
    cure_intercept: Any
    cure_beta: Any


def _linear_effect(features: Any, weights: Any) -> Any:
    """``features @ weights`` contracted over the trailing feature axis.

    Correct for zero-width feature/weight arrays: contracting over an
    empty axis yields exact zeros with the right broadcast shape, so no
    special-casing of "no regressors" is required.
    """
    return jnp.tensordot(features, weights, axes=([-1], [-1]))


def _hazard_logits(parameters: StageParameters, ages: Any, features: Any) -> Any:
    age_index = jnp.clip(jnp.asarray(ages), 0, parameters.age_logits.shape[-1] - 1)
    return parameters.age_logits[age_index] + _linear_effect(features, parameters.beta)


def _cure_logits(parameters: StageParameters, cure_features: Any) -> Any:
    return parameters.cure_intercept + _linear_effect(cure_features, parameters.cure_beta)


def stage_hazard(
    parameters: StageParameters, ages: Any, features: Any, allowed: Any | None = None
) -> Any:
    """Daily hazard ``h(a, t) = sigmoid(age_logits[min(a, K-1)] + x(t) @ beta)``.

    ``ages`` and ``features`` share every leading/trailing axis except the
    feature axis (e.g. ``ages: [N, T]``, ``features: [N, T, P]``). The age
    index is clipped to the final baseline bin for tail ages (the last bin
    hazard is reused for arbitrarily old ages/receipts). Hazard is forced to
    exactly zero (not merely small) wherever ``ages < 0`` -- a unit is not
    yet at risk before its clock starts, so pre-origin padding never
    accumulates spurious survival -- and wherever ``allowed`` is ``False``
    (a hard closure, e.g. a weekend or holiday, contributes zero hazard that
    day regardless of the learned baseline/regressors).

    ``allowed`` defaults to all-True (no hard closures) when omitted.
    """
    ages = jnp.asarray(ages)
    logits = _hazard_logits(parameters, ages, features)
    valid = ages >= 0
    if allowed is not None:
        valid = valid & jnp.asarray(allowed)
    return jnp.where(valid, sigmoid(logits), 0.0)


def susceptibility(parameters: StageParameters, cure_features: Any) -> Any:
    """Susceptibility probability ``pi = sigmoid(cure_intercept + z @ cure_beta)``.

    ``cure_features`` is static at stage entry (``[..., Q]``); the result
    broadcasts over its leading axes. This is latent susceptibility, not a
    realized event probability: a susceptible unit can still miss a finite
    observation/policy window.
    """
    logits = _cure_logits(parameters, cure_features)
    return sigmoid(logits)


def conditional_susceptibility(probability: Any, log_survival: Any) -> Any:
    """Posterior susceptibility after surviving event-free with no closure.

    Given prior susceptibility ``pi`` and the log susceptible-survival
    function ``log_survival = sum(log(1 - h_k))`` accumulated over every day
    a unit was at risk and event-free through the conditioning point,
    returns ``pi * S / ((1 - pi) + pi * S)`` where ``S = exp(log_survival)``:
    the probability the unit remains susceptible (has not been cured) given
    it has not yet had the event. Computed in log-space via
    ``logaddexp`` for numerical stability rather than forming ``S``
    directly, so it stays accurate for long, deeply-survived histories
    where ``S`` itself would underflow to zero.

    Conditioning on a zero-probability history (``pi=1`` and ``S=0``) is
    undefined and returns NaN. Host-side forecasting must reject that
    contradiction rather than arbitrarily relabel a certain return as cured.
    """
    probability = jnp.asarray(probability)
    log_pi = jnp.log(probability)
    log_not_pi = jnp.log1p(-probability)
    log_numerator = log_pi + log_survival
    log_denominator = jnp.logaddexp(log_not_pi, log_numerator)
    return jnp.exp(log_numerator - log_denominator)
