"""Pluggable event-time families through the standalone fitting API.

A family samples its own named NumPyro sites in ``model`` and returns an
``EventLaw``; the fitting core applies exposure, closures, susceptibility and
conditioning. These tests pin the consumer-visible contract: real named
posteriors, exact likelihoods against closed-form survival, and refusals of
ambiguous or undeclared setups.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro import handlers
from numpyro.infer.util import log_density

from ttenet.event_times import EventLaw, TimingInputs, TimingLaw, log1mexp
from ttenet.families import EventFamily, FiniteTail, ProperTail, UnknownTail, WeibullFamily
from ttenet.models import StageFit, StageObservations, fit_stage, predict_stage, stage_model
from ttenet.processes import EventProcess
from ttenet.survival import StageParameters


def _weibull_log_stay(ages, scale, shape):
    """``log S(a+1) - log S(a)`` of a Weibull, with clean gradients at age zero."""
    age = jnp.maximum(ages, 0)
    positive = age > 0
    safe_age = jnp.where(positive, age, 1)
    cumulative = jnp.where(positive, jnp.exp(shape * (jnp.log(safe_age) - jnp.log(scale))), 0.0)
    following = jnp.exp(shape * (jnp.log(age + 1) - jnp.log(scale)))
    return -(following - cumulative)


def _weibull_family(*, scale=None, shape=2.0, susceptibility_logit=None, tail=ProperTail()):
    """Discrete shape-``shape`` Weibull: sampled ``scale`` unless fixed, optional ``susceptibility``."""

    def model(inputs, shared):
        lam = scale if scale is not None else numpyro.sample("scale", dist.LogNormal(1.0, 0.5))
        pi = (
            susceptibility_logit
            if susceptibility_logit is not None
            else numpyro.sample("susceptibility", dist.Normal(0.0, 2.0))
        )
        log_stay = _weibull_log_stay(inputs.ages, lam, shape)
        return EventLaw(
            TimingLaw(log1mexp(log_stay), log_stay), jnp.broadcast_to(pi, inputs.ages.shape[-1:])
        )

    return EventFamily(model, tail)


def _observations(ages, event_index, **overrides):
    ages = jnp.asarray(ages)
    n, t = ages.shape
    fields = dict(
        ages=ages,
        features=jnp.zeros((n, t, 0)),
        susceptibility_features=jnp.zeros((n, 0)),
        at_risk=jnp.ones((n, t), dtype=bool),
        allowed=jnp.ones((n, t), dtype=bool),
        event_index=jnp.asarray(event_index),
        exposure=jnp.ones((n, t), dtype=bool),
    )
    fields.update(overrides)
    return StageObservations(**fields)


def _survival(age, scale, shape=2.0):
    return np.exp(-((age / scale) ** shape))


def test_weibull_family_likelihood_and_gradient_match_closed_form_survival():
    # Shape 2, scale 3, pi=.7, all dates open: an event at age 2 has mass
    # pi*(S(2)-S(3)); censoring after four exposed dates has mass
    # (1-pi)+pi*S(4). The gradient in the scale is checked against central
    # differences of that float64 identity, through the native observation.
    family = _weibull_family(susceptibility_logit=float(np.log(0.7 / 0.3)))
    observations = _observations(ages=[[0, 1, 2, 3]] * 2, event_index=[2, -1])

    def observed_density(scale):
        model = handlers.block(
            handlers.substitute(stage_model, data={"scale": scale}), hide=["scale"]
        )
        return log_density(model, (observations,), {"family": family}, {})[0]

    def oracle(scale):
        event = 0.7 * (_survival(2, scale) - _survival(3, scale))
        censored = 0.3 + 0.7 * _survival(4, scale)
        return np.log(event) + np.log(censored)

    value, gradient = jax.value_and_grad(observed_density)(jnp.array(3.0))
    np.testing.assert_allclose(float(value), oracle(3.0), rtol=1e-6)
    step = 1e-6
    expected = (oracle(3.0 + step) - oracle(3.0 - step)) / (2 * step)
    np.testing.assert_allclose(float(gradient), expected, rtol=1e-4)


def test_weibull_family_respects_closures_and_delayed_entry_from_the_core():
    # Closed date 1 contributes no hazard and no survival decay; two known
    # event-free pre-entry dates condition the susceptibility exactly.
    family = _weibull_family(scale=3.0, susceptibility_logit=float(np.log(0.7 / 0.3)))
    observations = _observations(
        ages=[[0, 1, 2, 3]],
        event_index=[-1],
        allowed=jnp.array([[True, False, True, True]]),
        at_risk=jnp.array([[False, False, True, True]]),
        exposure=jnp.array([[False, False, True, True]]),
        pre_entry=jnp.array([[True, True, False, False]]),
    )
    value = log_density(stage_model, (observations,), {"family": family}, {})[0]
    stay = _survival(1, 3.0) / _survival(0, 3.0)  # only age 0 is open before entry
    pre = 0.7 * stay / (0.3 + 0.7 * stay)
    post = _survival(4, 3.0) / _survival(2, 3.0)
    np.testing.assert_allclose(float(value), np.log((1 - pre) + pre * post), rtol=1e-6)


def test_fit_stage_custom_family_exposes_named_posterior_and_optimized_values():
    def model(inputs, shared):
        scale = numpyro.sample("scale", dist.LogNormal(1.0, 0.3))
        shape = numpyro.param("shape", jnp.array(1.5), constraint=dist.constraints.positive)
        log_stay = _weibull_log_stay(inputs.ages, scale, shape)
        return EventLaw(
            TimingLaw(log1mexp(log_stay), log_stay), jnp.full(inputs.ages.shape[-1:], 1.0)
        )

    family = EventFamily(model, ProperTail())
    rng = np.random.default_rng(0)
    n, t = 30, 12
    ages = np.broadcast_to(np.arange(t), (n, t))
    delays = np.floor(3.0 * np.sqrt(-np.log(rng.random(n)))).astype(int)
    event_index = np.where(delays < t, delays, -1)
    observations = _observations(ages=ages, event_index=event_index)
    fit = fit_stage(observations, family=family, num_steps=40, num_samples=9, seed=2)
    assert fit.family is family and fit.shared is None
    assert fit.num_samples == 9 and fit.draws == 9
    assert set(fit.parameters) == {"scale", "shape"}
    assert fit.parameters["scale"].shape == (9,) and fit.parameters["shape"].shape == (9,)
    assert len(np.unique(np.asarray(fit.parameters["scale"]))) > 1
    np.testing.assert_array_equal(fit.parameters["shape"], np.full(9, fit.parameters["shape"][0]))
    assert float(fit.parameters["shape"][0]) > 0 and float(fit.parameters["shape"][0]) != 1.5
    assert np.isfinite(fit.losses).all() and fit.losses.shape == (40,)
    inputs = TimingInputs(
        ages=jnp.array([[0, 20], [1, 21]]),
        features=jnp.zeros((2, 2, 0)),
        susceptibility_features=jnp.zeros((2, 0)),
    )
    law = fit.timing(inputs, draw=4)
    assert isinstance(law, EventLaw)
    scale, shape = fit.parameters["scale"][4], fit.parameters["shape"][4]
    expected = _weibull_log_stay(inputs.ages, scale, shape)
    np.testing.assert_allclose(law.timing.log_survival_step, expected, rtol=1e-5)
    np.testing.assert_allclose(law.susceptibility_logits, [1.0, 1.0])


def test_fit_stage_fixed_family_keeps_the_requested_draw_count():
    family = _weibull_family(scale=3.0, susceptibility_logit=0.5)
    observations = _observations(ages=[[0, 1, 2]] * 3, event_index=[1, -1, 2])
    fit = fit_stage(observations, family=family, num_steps=5, num_samples=4)
    assert dict(fit.parameters) == {}
    assert fit.num_samples == 4 and fit.draws == 4
    laws = jax.vmap(
        lambda draw: fit.timing(
            TimingInputs(jnp.array([[0], [1]]), jnp.zeros((2, 1, 0)), jnp.zeros((1, 0))),
            draw=draw,
        )
    )(jnp.arange(fit.draws))
    assert laws.timing.log_hazard.shape == (4, 2, 1)
    assert laws.susceptibility_logits.shape == (4, 1)
    np.testing.assert_allclose(laws.timing.log_survival_step[:, 0, 0], -1 / 9, rtol=1e-6)


def test_custom_fit_timing_refuses_sites_missing_from_its_posterior():
    family = _weibull_family()
    fit = StageFit({"susceptibility": jnp.zeros(3)}, jnp.zeros(0), family=family, num_samples=3)
    inputs = TimingInputs(jnp.array([[0], [1]]), jnp.zeros((2, 1, 0)), jnp.zeros((1, 0)))
    with pytest.raises(ValueError, match="scale"):
        fit.timing(inputs, draw=0)


def test_stage_fit_validates_draw_metadata_against_its_arrays():
    parameters = StageParameters(
        jnp.zeros((3, 2)), jnp.zeros((3, 0)), jnp.zeros(3), jnp.zeros((3, 0))
    )
    with pytest.raises(ValueError, match="num_samples"):
        StageFit(parameters, jnp.zeros(0), num_samples=5)
    with pytest.raises(ValueError, match="num_samples"):
        StageFit({}, jnp.zeros(0), family=_weibull_family(scale=1.0, susceptibility_logit=0.0))
    with pytest.raises(TypeError, match="StageParameters"):
        StageFit({"scale": jnp.ones(3)}, jnp.zeros(0))


def test_ambiguous_family_configuration_is_rejected():
    def prior(observations, shared):
        return StageParameters(jnp.zeros(1), jnp.empty(0), 0.0, jnp.empty(0))

    with pytest.raises(ValueError, match="ambiguous"):
        EventProcess(parameter_model=prior, family=_weibull_family())
    with pytest.raises(TypeError):
        EventProcess(family="not a family")


def test_family_without_valid_tail_declaration_is_refused():
    observations = _observations(ages=[[0, 1]], event_index=[-1])
    model = _weibull_family(scale=2.0, susceptibility_logit=0.0).model

    def bare_function(inputs, shared):
        return model(inputs, shared)

    class DictTail:
        tail = {"kind": "proper"}

        @staticmethod
        def model(inputs, shared):
            return model(inputs, shared)

    for undeclared in (bare_function, DictTail()):
        with pytest.raises((TypeError, ValueError)):
            EventProcess(family=undeclared)
        with pytest.raises((TypeError, ValueError)):
            fit_stage(observations, family=undeclared, num_steps=1, num_samples=1)
    for last_age in (-1, True, 2.5):
        with pytest.raises(ValueError, match="last_age"):
            EventProcess(family=EventFamily(model, FiniteTail(last_age=last_age)))
        with pytest.raises(ValueError, match="last_age"):
            fit_stage(
                observations,
                family=EventFamily(model, FiniteTail(last_age=last_age)),
                num_steps=1,
                num_samples=1,
            )
    finite = _weibull_family(scale=2.0, susceptibility_logit=0.0, tail=FiniteTail(last_age=4))
    assert EventProcess(family=finite).family is finite
    unknown = _weibull_family(scale=2.0, susceptibility_logit=0.0, tail=UnknownTail())
    assert EventProcess(family=unknown).family is unknown


def test_family_model_must_return_a_named_event_law():
    law_model = _weibull_family(scale=2.0, susceptibility_logit=0.0).model

    def unnamed(inputs, shared):
        return tuple(law_model(inputs, shared))

    observations = _observations(ages=[[0, 1]], event_index=[-1])
    with pytest.raises(TypeError, match="EventLaw"):
        fit_stage(
            observations, family=EventFamily(unnamed, ProperTail()), num_steps=1, num_samples=1
        )


def test_packaged_weibull_family_fits_and_replays_through_the_public_api():
    family = WeibullFamily(scale_prior=dist.LogNormal(1.0, 0.3), susceptibility_logit_prior=1.0)
    assert isinstance(family.tail, ProperTail)
    rng = np.random.default_rng(1)
    n, t = 40, 10
    ages = np.broadcast_to(np.arange(t), (n, t))
    delays = np.floor(3.0 * np.sqrt(-np.log(rng.random(n)))).astype(int)
    observations = _observations(ages, np.where(delays < t, delays, -1))
    fit = fit_stage(observations, family=family, num_steps=30, num_samples=6, seed=3)
    assert fit.family is family and set(fit.parameters) == {"scale"}
    assert len(np.unique(np.asarray(fit.parameters["scale"]))) > 1
    inputs = TimingInputs(jnp.array([[0, 3], [1, 4]]), jnp.zeros((2, 2, 0)), jnp.zeros((2, 0)))
    law = fit.timing(inputs, draw=2)
    expected = _weibull_log_stay(inputs.ages, fit.parameters["scale"][2], 2.0)
    np.testing.assert_allclose(law.timing.log_survival_step, expected, rtol=1e-5)
    np.testing.assert_allclose(law.susceptibility_logits, [1.0, 1.0], rtol=1e-6)


@pytest.mark.parametrize("index,days", [(2, 2), (-2, 2), (0.5, 2), (0, 0)])
def test_invalid_event_indices_are_not_fitted_as_censoring(index, days):
    observations = _observations([list(range(days))], [index])
    family = _weibull_family(scale=3.0, susceptibility_logit=1.0)
    with pytest.raises(ValueError, match="event_index"):
        log_density(stage_model, (observations,), {"family": family}, {})


def test_traced_invalid_event_indices_keep_impossible_likelihood():
    family = _weibull_family(scale=3.0, susceptibility_logit=1.0)

    def score(index):
        observations = _observations([[0, 1]], jnp.reshape(index, (1,)))
        return log_density(stage_model, (observations,), {"family": family}, {})[0]

    scores = np.asarray(jax.jit(jax.vmap(score))(jnp.array([-1, 0, 1, 2, -2])))
    assert np.isfinite(scores[:3]).all()
    assert np.isneginf(scores[3:]).all()


@pytest.mark.parametrize("optimized", [False, True])
def test_zero_size_latent_sites_remain_replayable_with_or_without_parameters(optimized):
    def model(inputs, shared):
        beta = numpyro.sample(
            "beta", dist.Normal(0, 1).expand([inputs.features.shape[-1]]).to_event(1)
        )
        rate = numpyro.param("rate", jnp.array(0.5)) if optimized else jnp.array(0.5)
        stay = -rate * jnp.exp(inputs.features @ beta)
        logits = jnp.full(inputs.ages.shape[-1], np.log(4.0))
        return EventLaw(TimingLaw(log1mexp(stay), stay), logits)

    family = EventFamily(model, ProperTail())
    observations = _observations(
        [[0, 1, 2, 3]], [-1], allowed=jnp.array([[True, False, True, True]])
    )
    fitted = fit_stage(observations, family=family, num_steps=2, num_samples=2048)
    paths = np.asarray(predict_stage(fitted, observations, seed=19))
    assert not paths[:, 1].any()
    assert (paths.sum(axis=1) <= 1).all()
    rate = float(fitted.parameters["rate"][0]) if optimized else 0.5
    expected = 0.8 * (1 - np.exp(-3 * rate))
    error = abs(paths.sum(axis=(1, 2)).mean() - expected)
    assert error < 6 * np.sqrt(expected * (1 - expected) / len(paths))
