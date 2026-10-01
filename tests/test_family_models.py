"""Pluggable event-time families through the standalone fitting API.

A family factory samples its own named NumPyro sites and returns a timing law;
the fitting core applies exposure, closures, cure and conditioning. These
tests pin the consumer-visible contract: real named posteriors, exact
likelihoods against closed-form survival, and refusals of ambiguous setups.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro import handlers
from numpyro.infer.util import log_density

from ttenet.event_times import TimingInputs, TimingLaw, log1mexp
from ttenet.models import StageFit, StageObservations, fit_stage, predict_stage, stage_model
from ttenet.processes import CureProcess
from ttenet.survival import StageParameters

_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "event_time_families.py"


def _weibull_log_stay(ages, scale, shape):
    """``log S(a+1) - log S(a)`` of a Weibull, with clean gradients at age zero."""
    age = jnp.maximum(ages, 0)
    positive = age > 0
    safe_age = jnp.where(positive, age, 1)
    cumulative = jnp.where(positive, jnp.exp(shape * (jnp.log(safe_age) - jnp.log(scale))), 0.0)
    following = jnp.exp(shape * (jnp.log(age + 1) - jnp.log(scale)))
    return -(following - cumulative)


def _weibull_family(*, scale=None, shape=2.0, cure_logit=None):
    """Discrete shape-``shape`` Weibull: sampled ``scale`` unless fixed, optional ``cure``."""

    def event_time_model(inputs, shared):
        lam = scale if scale is not None else numpyro.sample("scale", dist.LogNormal(1.0, 0.5))
        pi = cure_logit if cure_logit is not None else numpyro.sample("cure", dist.Normal(0.0, 2.0))
        log_stay = _weibull_log_stay(inputs.ages, lam, shape)
        return TimingLaw(log1mexp(log_stay), log_stay), jnp.broadcast_to(pi, inputs.ages.shape[-1:])

    event_time_model.tail_behavior = {"kind": "proper"}
    return event_time_model


def _observations(ages, event_index, **overrides):
    ages = jnp.asarray(ages)
    n, t = ages.shape
    fields = dict(
        ages=ages,
        features=jnp.zeros((n, t, 0)),
        cure_features=jnp.zeros((n, 0)),
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
    family = _weibull_family(cure_logit=float(np.log(0.7 / 0.3)))
    observations = _observations(ages=[[0, 1, 2, 3]] * 2, event_index=[2, -1])

    def observed_density(scale):
        model = handlers.block(
            handlers.substitute(stage_model, data={"scale": scale}), hide=["scale"]
        )
        return log_density(model, (observations,), {"event_time_model": family}, {})[0]

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
    family = _weibull_family(scale=3.0, cure_logit=float(np.log(0.7 / 0.3)))
    observations = _observations(
        ages=[[0, 1, 2, 3]],
        event_index=[-1],
        allowed=jnp.array([[True, False, True, True]]),
        at_risk=jnp.array([[False, False, True, True]]),
        exposure=jnp.array([[False, False, True, True]]),
        pre_entry=jnp.array([[True, True, False, False]]),
    )
    value = log_density(stage_model, (observations,), {"event_time_model": family}, {})[0]
    stay = _survival(1, 3.0) / _survival(0, 3.0)  # only age 0 is open before entry
    pre = 0.7 * stay / (0.3 + 0.7 * stay)
    post = _survival(4, 3.0) / _survival(2, 3.0)
    np.testing.assert_allclose(float(value), np.log((1 - pre) + pre * post), rtol=1e-6)


def test_fit_stage_custom_family_exposes_named_posterior_and_optimized_values():
    def family(inputs, shared):
        scale = numpyro.sample("scale", dist.LogNormal(1.0, 0.3))
        shape = numpyro.param("shape", jnp.array(1.5), constraint=dist.constraints.positive)
        log_stay = _weibull_log_stay(inputs.ages, scale, shape)
        return TimingLaw(log1mexp(log_stay), log_stay), jnp.full(inputs.ages.shape[-1:], 1.0)

    family.tail_behavior = {"kind": "proper"}
    rng = np.random.default_rng(0)
    n, t = 30, 12
    ages = np.broadcast_to(np.arange(t), (n, t))
    delays = np.floor(3.0 * np.sqrt(-np.log(rng.random(n)))).astype(int)
    event_index = np.where(delays < t, delays, -1)
    observations = _observations(ages=ages, event_index=event_index)
    fit = fit_stage(observations, event_time_model=family, num_steps=40, num_samples=9, seed=2)
    assert fit.event_time_model is family and fit.shared is None
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
        cure_features=jnp.zeros((2, 0)),
    )
    timing, logits = fit.timing(inputs, draw=4)
    scale, shape = fit.parameters["scale"][4], fit.parameters["shape"][4]
    expected = _weibull_log_stay(inputs.ages, scale, shape)
    np.testing.assert_allclose(timing.log_survival_step, expected, rtol=1e-5)
    np.testing.assert_allclose(logits, [1.0, 1.0])


def test_fit_stage_fixed_family_keeps_the_requested_draw_count():
    family = _weibull_family(scale=3.0, cure_logit=0.5)
    observations = _observations(ages=[[0, 1, 2]] * 3, event_index=[1, -1, 2])
    fit = fit_stage(observations, event_time_model=family, num_steps=5, num_samples=4)
    assert dict(fit.parameters) == {}
    assert fit.num_samples == 4 and fit.draws == 4
    laws = jax.vmap(
        lambda draw: fit.timing(
            TimingInputs(jnp.array([[0], [1]]), jnp.zeros((2, 1, 0)), jnp.zeros((1, 0))),
            draw=draw,
        )
    )(jnp.arange(fit.draws))
    assert laws[0].log_hazard.shape == (4, 2, 1) and laws[1].shape == (4, 1)
    np.testing.assert_allclose(laws[0].log_survival_step[:, 0, 0], -1 / 9, rtol=1e-6)


def test_custom_fit_timing_refuses_sites_missing_from_its_posterior():
    family = _weibull_family()
    fit = StageFit({"cure": jnp.zeros(3)}, jnp.zeros(0), event_time_model=family, num_samples=3)
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
        StageFit({}, jnp.zeros(0), event_time_model=_weibull_family(scale=1.0, cure_logit=0.0))
    with pytest.raises(TypeError, match="StageParameters"):
        StageFit({"scale": jnp.ones(3)}, jnp.zeros(0))


def test_ambiguous_family_configuration_is_rejected():
    def prior(observations, shared):
        return StageParameters(jnp.zeros(1), jnp.empty(0), 0.0, jnp.empty(0))

    with pytest.raises(ValueError, match="ambiguous"):
        CureProcess(parameter_model=prior, event_time_model=_weibull_family())
    with pytest.raises(TypeError):
        CureProcess(event_time_model="not callable")


def test_family_without_declared_tail_behavior_is_refused():
    def family(inputs, shared):
        return _weibull_family(scale=2.0, cure_logit=0.0)(inputs, shared)

    with pytest.raises(ValueError, match="tail_behavior"):
        CureProcess(event_time_model=family)
    observations = _observations(ages=[[0, 1]], event_index=[-1])
    with pytest.raises(ValueError, match="tail_behavior"):
        fit_stage(observations, event_time_model=family, num_steps=1, num_samples=1)
    family.tail_behavior = {"kind": "finite"}
    with pytest.raises(ValueError, match="last_age"):
        CureProcess(event_time_model=family)
    family.tail_behavior = {"kind": "finite", "last_age": 4}
    assert CureProcess(event_time_model=family).event_time_model is family


def test_example_weibull_plugin_is_a_proper_shape_two_family():
    spec = importlib.util.spec_from_file_location("event_time_families", _EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    family = module.weibull_family(scale=4.0, shape=2.0, cure=float(np.log(0.8 / 0.2)))
    assert family.tail_behavior == {"kind": "proper"}
    inputs = TimingInputs(
        ages=jnp.array([[-1, 0], [0, 1], [1, 2]]),
        features=jnp.zeros((3, 2, 0)),
        cure_features=jnp.zeros((2, 0)),
    )
    with handlers.seed(rng_seed=0):
        timing, logits = family(inputs, None)
    stay = np.exp(np.asarray(timing.log_survival_step, dtype=np.float64))
    expected = _survival(np.maximum(np.asarray(inputs.ages), 0) + 1, 4.0) / _survival(
        np.maximum(np.asarray(inputs.ages), 0), 4.0
    )
    np.testing.assert_allclose(stay, expected, rtol=1e-6)
    np.testing.assert_allclose(np.exp(timing.log_hazard) + stay, 1.0, rtol=1e-6)
    np.testing.assert_allclose(logits, np.log(0.8 / 0.2), rtol=1e-6)
    sampled = module.weibull_family(scale=dist.LogNormal(1.0, 0.3), shape=dist.LogNormal(0.5, 0.2))
    trace = handlers.trace(handlers.seed(sampled, rng_seed=1)).get_trace(inputs, None)
    assert {name for name, site in trace.items() if site["type"] == "sample"} == {
        "scale",
        "shape",
        "cure_intercept",
    }


@pytest.mark.parametrize("index,days", [(2, 2), (-2, 2), (0.5, 2), (0, 0)])
def test_invalid_event_indices_are_not_fitted_as_censoring(index, days):
    observations = _observations([list(range(days))], [index])
    family = _weibull_family(scale=3.0, cure_logit=1.0)
    with pytest.raises(ValueError, match="event_index"):
        log_density(stage_model, (observations,), {"event_time_model": family}, {})


def test_traced_invalid_event_indices_keep_impossible_likelihood():
    family = _weibull_family(scale=3.0, cure_logit=1.0)

    def score(index):
        observations = _observations([[0, 1]], jnp.reshape(index, (1,)))
        return log_density(stage_model, (observations,), {"event_time_model": family}, {})[0]

    scores = np.asarray(jax.jit(jax.vmap(score))(jnp.array([-1, 0, 1, 2, -2])))
    assert np.isfinite(scores[:3]).all()
    assert np.isneginf(scores[3:]).all()


@pytest.mark.parametrize("optimized", [False, True])
def test_zero_size_latent_sites_remain_replayable_with_or_without_parameters(optimized):
    def family(inputs, shared):
        beta = numpyro.sample(
            "beta", dist.Normal(0, 1).expand([inputs.features.shape[-1]]).to_event(1)
        )
        rate = numpyro.param("rate", jnp.array(0.5)) if optimized else jnp.array(0.5)
        stay = -rate * jnp.exp(inputs.features @ beta)
        return TimingLaw(log1mexp(stay), stay), jnp.full(inputs.ages.shape[-1], np.log(4.0))

    family.tail_behavior = {"kind": "proper"}
    observations = _observations(
        [[0, 1, 2, 3]], [-1], allowed=jnp.array([[True, False, True, True]])
    )
    fitted = fit_stage(observations, event_time_model=family, num_steps=2, num_samples=2048)
    paths = np.asarray(predict_stage(fitted, observations, seed=19))
    assert not paths[:, 1].any()
    assert (paths.sum(axis=1) <= 1).all()
    rate = float(fitted.parameters["rate"][0]) if optimized else 0.5
    expected = 0.8 * (1 - np.exp(-3 * rate))
    error = abs(paths.sum(axis=(1, 2)).mean() - expected)
    assert error < 6 * np.sqrt(expected * (1 - expected) / len(paths))
