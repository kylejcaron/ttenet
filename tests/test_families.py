"""Behavioral tests for the typed family declarations and the packaged Weibull.

Oracles are float64 NumPy computations in probability space on the
continuous Weibull survival ``S(a) = exp(-(a / scale) ** shape)``; the
family under test works in log space through the shared kernel.
"""

from __future__ import annotations

import types

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro import handlers

from ttenet.event_times import EventLaw, TimingInputs, exclusive_log_survival, survival_kernel
from ttenet.families import (
    EventFamily,
    FiniteTail,
    ProperTail,
    WeibullFamily,
    validate_family,
)


def _inputs(ages, features=None, susceptibility_features=None) -> TimingInputs:
    ages = jnp.asarray(ages)
    days, cohorts = ages.shape
    if features is None:
        features = jnp.zeros((days, cohorts, 0))
    if susceptibility_features is None:
        susceptibility_features = jnp.zeros((cohorts, 0))
    return TimingInputs(
        ages=ages,
        features=jnp.asarray(features),
        susceptibility_features=jnp.asarray(susceptibility_features),
    )


def _weibull_log_survival(ages, scale, shape):
    return -((np.maximum(np.asarray(ages, dtype=np.float64), 0) / scale) ** shape)


def _bin_log_stay(ages, scale, shape):
    ages = np.maximum(np.asarray(ages, dtype=np.float64), 0)
    return _weibull_log_survival(ages + 1, scale, shape) - _weibull_log_survival(ages, scale, shape)


def _law(family, inputs, data=None, seed=0) -> EventLaw:
    model = handlers.seed(family.model, seed)
    if data is not None:
        model = handlers.substitute(model, data=data)
    return model(inputs, None)


# --- Packaged Weibull ----------------------------------------------------------


def test_weibull_fixed_parameters_have_no_sites_and_match_the_analytic_discrete_law():
    # Scale 6, shape 2, fixed logit .3: no NumPyro site is created, each cell
    # is the bin [a, a+1) of the continuous survival, and a negative age
    # evaluates age zero (the core masks it).
    family = WeibullFamily(scale_prior=6.0, shape_prior=2.0, susceptibility_logit_prior=0.3)
    ages = np.array([[0, -1], [1, 2], [2, 3], [3, 4], [4, 5]])
    inputs = _inputs(ages)
    trace = handlers.trace(family.model).get_trace(inputs, None)
    assert trace == {}
    law = family.model(inputs, None)
    assert isinstance(law, EventLaw)
    log_stay = _bin_log_stay(ages, 6.0, 2.0)
    np.testing.assert_allclose(np.array(law.timing.log_survival_step), log_stay, rtol=1e-6)
    np.testing.assert_allclose(
        np.exp(np.array(law.timing.log_hazard)), 1 - np.exp(log_stay), rtol=1e-6
    )
    np.testing.assert_allclose(np.array(law.susceptibility_logits), [0.3, 0.3], rtol=1e-6)
    assert law.susceptibility_logits.shape == (2,)


def test_weibull_prior_sites_and_regressor_semantics():
    # Distribution priors create exactly the named sites; regressors scale the
    # bin's cumulative-hazard increment by exp(x_t @ beta) and shift the
    # susceptibility logit by z @ susceptibility_beta.
    family = WeibullFamily(
        scale_prior=dist.LogNormal(np.log(5.0), 0.5), shape_prior=dist.LogNormal(np.log(2.0), 0.3)
    )
    rng = np.random.default_rng(5)
    ages = np.arange(4)[:, None] + np.array([[0, 3]])
    features = rng.normal(size=(4, 2, 2))
    susceptibility_features = rng.normal(size=(2, 1))
    inputs = _inputs(ages, features, susceptibility_features)
    trace = handlers.trace(handlers.seed(family.model, 1)).get_trace(inputs, None)
    assert set(trace) == {
        "scale",
        "shape",
        "susceptibility_intercept",
        "beta",
        "susceptibility_beta",
    }
    assert all(site["type"] == "sample" for site in trace.values())
    assert trace["beta"]["value"].shape == (2,) and trace["susceptibility_beta"]["value"].shape == (
        1,
    )
    assert trace["scale"]["value"].shape == () and trace["shape"]["value"].shape == ()

    beta, susceptibility_beta = np.array([0.2, -0.4]), np.array([0.5])
    law = _law(
        family,
        inputs,
        data={
            "scale": 6.0,
            "shape": 2.0,
            "susceptibility_intercept": 0.3,
            "beta": beta,
            "susceptibility_beta": susceptibility_beta,
        },
    )
    log_stay = _bin_log_stay(ages, 6.0, 2.0) * np.exp(features @ beta)
    np.testing.assert_allclose(np.array(law.timing.log_survival_step), log_stay, rtol=1e-5)
    np.testing.assert_allclose(
        np.exp(np.array(law.timing.log_hazard)), 1 - np.exp(log_stay), rtol=1e-5
    )
    np.testing.assert_allclose(
        np.array(law.susceptibility_logits),
        0.3 + susceptibility_features @ susceptibility_beta,
        rtol=1e-6,
    )

    fixed_sites = handlers.trace(handlers.seed(WeibullFamily(6.0).model, 2)).get_trace(inputs, None)
    assert set(fixed_sites) == {"susceptibility_intercept", "beta", "susceptibility_beta"}


def test_weibull_masses_match_simulated_floor_of_continuous_weibull():
    # The discrete law is the day on which a continuous Weibull(6, 2) delay
    # lands: its masses over ages 0..11 and the remaining mass past age 11
    # agree with a large simulated sample of floor(6 * (-log U) ** .5).
    rng = np.random.default_rng(11)
    delays = np.floor(6.0 * (-np.log(rng.random(400_000))) ** 0.5).astype(int)
    empirical = np.bincount(np.minimum(delays, 12), minlength=13) / delays.size
    law = WeibullFamily(6.0, 2.0, 0.0).model(_inputs(np.arange(12)[:, None]), None)
    log_mass = exclusive_log_survival(law.timing.log_survival_step) + law.timing.log_hazard
    masses = np.exp(np.array(log_mass[:, 0], dtype=np.float64))
    np.testing.assert_allclose(masses, empirical[:12], atol=3e-3)
    np.testing.assert_allclose(1 - masses.sum(), empirical[12], atol=3e-3)
    np.testing.assert_allclose(1 - masses.sum(), np.exp(-4.0), rtol=1e-5)


def test_weibull_replay_is_differentiable_and_jittable_at_the_first_bin_and_far_ages():
    # Substituting one draw's values at the sites is how the core replays a
    # fitted family. Gradients in scale and shape are finite on the age-zero
    # bin (where log age is undefined) and far out where stays underflow,
    # and the compiled law equals the eager one.
    family = WeibullFamily(dist.LogNormal(0.0, 1.0), dist.LogNormal(0.0, 1.0), 0.0)
    inputs = _inputs(np.array([[0], [1], [40], [400]]))

    def replay(scale, shape):
        law = handlers.substitute(family.model, data={"scale": scale, "shape": shape})(inputs, None)
        return law.timing

    def total(scale, shape):
        timing = replay(scale, shape)
        return jnp.sum(timing.log_hazard) + jnp.sum(timing.log_survival_step)

    gradients = jax.grad(total, argnums=(0, 1))(jnp.array(0.5), jnp.array(2.0))
    assert all(bool(jnp.all(jnp.isfinite(value))) for value in gradients)
    eager = replay(jnp.array(0.5), jnp.array(2.0))
    compiled = jax.jit(replay)(jnp.array(0.5), jnp.array(2.0))
    np.testing.assert_array_equal(np.array(compiled.log_hazard), np.array(eager.log_hazard))
    np.testing.assert_array_equal(
        np.array(compiled.log_survival_step), np.array(eager.log_survival_step)
    )
    assert bool(jnp.all(jnp.isfinite(eager.log_hazard))) and float(eager.log_hazard[-1, 0]) <= 0.0
    # Scale derivative of the first bin's log hazard: hazard 1 - exp(-(1/s)^k).
    first = jax.grad(lambda s: replay(s, jnp.array(2.0)).log_hazard[0, 0])(jnp.array(0.5))
    s, k = 0.5, 2.0
    expected = -(k / s) * (1 / s) ** k * np.exp(-((1 / s) ** k)) / (1 - np.exp(-((1 / s) ** k)))
    np.testing.assert_allclose(float(first), expected, rtol=1e-5)


def test_weibull_family_is_static_jit_data_for_fixed_jax_values_and_prior_replay():
    # A fitted stage keeps its family as pytree metadata, so the family must
    # hash and compare as a static argument: fixed JAX scalars normalize to
    # plain floats at construction and distribution priors hash by identity.
    inputs = _inputs(np.arange(3)[:, None])
    fixed = WeibullFamily(jnp.array(6.0), jnp.float32(2), jnp.array(0.5))
    law = jax.jit(lambda family, inputs: family.model(inputs, None), static_argnums=0)(
        fixed, inputs
    )
    expected = _bin_log_stay(np.arange(3)[:, None], 6.0, 2.0)
    np.testing.assert_allclose(np.array(law.timing.log_survival_step), expected, rtol=1e-6)
    np.testing.assert_allclose(np.array(law.susceptibility_logits), [0.5], rtol=1e-6)

    prior = WeibullFamily(dist.LogNormal(0.0, 1.0), dist.LogNormal(0.0, 1.0), dist.Normal(0.0, 2.0))

    def replay(family, inputs, scale, shape, intercept):
        data = {"scale": scale, "shape": shape, "susceptibility_intercept": intercept}
        return handlers.substitute(family.model, data=data)(inputs, None)

    compiled = jax.jit(replay, static_argnums=0)
    law = compiled(prior, inputs, jnp.array(6.0), jnp.array(2.0), jnp.array(0.3))
    np.testing.assert_allclose(np.array(law.timing.log_survival_step), expected, rtol=1e-6)
    np.testing.assert_allclose(np.array(law.susceptibility_logits), [0.3], rtol=1e-6)
    with pytest.raises(TypeError):
        jax.jit(lambda scale: WeibullFamily(scale).model(inputs, None))(jnp.array(6.0))


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"scale_prior": 0.0}, ValueError),
        ({"scale_prior": -6.0}, ValueError),
        ({"scale_prior": np.inf}, ValueError),
        ({"scale_prior": 6.0, "shape_prior": 0.0}, ValueError),
        ({"scale_prior": 6.0, "susceptibility_logit_prior": np.nan}, ValueError),
        ({"scale_prior": True}, TypeError),
        ({"scale_prior": "6"}, TypeError),
        ({"scale_prior": [6.0]}, TypeError),
        ({"scale_prior": 6.0, "shape_prior": np.array([2.0, 2.0])}, TypeError),
        ({"scale_prior": 6.0, "susceptibility_logit_prior": None}, TypeError),
    ],
)
def test_weibull_rejects_fixed_values_outside_its_domain(kwargs, error):
    with pytest.raises(error):
        WeibullFamily(**kwargs)


# --- Declarations ----------------------------------------------------------------


def _plain_model(inputs, shared):
    raise AssertionError("validation must not run the model")


class _Lookalike(ProperTail):
    pass


@pytest.mark.parametrize(
    ("family", "error"),
    [
        (types.SimpleNamespace(model="not callable", tail=ProperTail()), TypeError),
        (types.SimpleNamespace(tail=ProperTail()), TypeError),
        (_plain_model, TypeError),
        (types.SimpleNamespace(model=_plain_model, tail={"kind": "proper"}), ValueError),
        (types.SimpleNamespace(model=_plain_model, tail=ProperTail), ValueError),
        (types.SimpleNamespace(model=_plain_model, tail=_Lookalike()), ValueError),
        (types.SimpleNamespace(model=_plain_model, tail=None), ValueError),
    ],
)
def test_validate_family_refuses_missing_models_and_lookalike_tails(family, error):
    with pytest.raises(error):
        validate_family(family)


def test_event_family_validates_at_construction():
    with pytest.raises(TypeError):
        EventFamily("not callable", ProperTail())
    with pytest.raises(ValueError):
        EventFamily(_plain_model, {"kind": "finite", "last_age": 3})


@pytest.mark.parametrize("last_age", [True, False, -1, 2.0, np.float64(2), "3", None])
def test_finite_tail_needs_a_nonnegative_integer_support(last_age):
    with pytest.raises(ValueError):
        FiniteTail(last_age)


def test_weibull_overflow_does_not_lose_arrivals_on_reopening():
    family = WeibullFamily(dist.LogNormal(-5.0, 0.1), dist.LogNormal(3.0, 0.1), 0.0)
    inputs = _inputs(jnp.arange(3, dtype=jnp.float32)[:, None])
    allowed = jnp.array([[False], [True], [True]])

    def masses(parameters):
        law = _law(family, inputs, {"scale": parameters[0], "shape": parameters[1]})
        kernel = survival_kernel(
            law.timing, law.susceptibility_logits, allowed=allowed, exposure=jnp.ones_like(allowed)
        )
        return jnp.exp(kernel.log_mass), jnp.exp(kernel.log_tail)

    parameters = jnp.array([0.01, 20.0])
    events, tail = jax.jit(masses)(parameters)
    np.testing.assert_allclose(events[:, 0], [0.0, 0.5, 0.0], atol=1e-6)
    np.testing.assert_allclose(tail, [0.5], atol=1e-6)
    gradient = jax.grad(lambda values: masses(values)[0].sum())(parameters)
    np.testing.assert_allclose(gradient, [0.0, 0.0], atol=1e-6)


def test_weibull_small_bin_hazard_survives_large_age_cancellation():
    age = 10_000_000
    scale = 10_000_000.0
    family = WeibullFamily(scale, 2.0, 0.0)
    law = _law(family, _inputs(jnp.array([[age]], dtype=jnp.int32)))
    increment = (2 * age + 1) / scale**2
    np.testing.assert_allclose(law.timing.log_hazard, np.log(-np.expm1(-increment)), rtol=2e-6)


def test_weibull_regressor_overflow_keeps_first_bin_and_gradient_valid():
    family = WeibullFamily(6.0, 2.0, 0.0)
    inputs = _inputs(jnp.array([[0.0], [1.0]]), features=jnp.ones((2, 1, 1)))

    def hazards(beta):
        return _law(family, inputs, {"beta": beta[None]}).timing.log_hazard

    np.testing.assert_allclose(jax.jit(hazards)(jnp.array(100.0)), [[0.0], [0.0]], atol=1e-6)
    np.testing.assert_allclose(jax.grad(lambda beta: hazards(beta).sum())(jnp.array(100.0)), 0.0)


def test_weibull_global_scale_prior_refuses_per_cohort_batches():
    with pytest.raises(ValueError, match="scalar"):
        WeibullFamily(scale_prior=dist.LogNormal(jnp.zeros(2), 0.3))
