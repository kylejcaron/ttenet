"""Throwaway survival convolutions on the calendar axis, retaining unit identity.

K[start, date, unit] is an unconditional first-event mass (a subprobability
kernel). Compose K_init and K_receipt to obtain exact expected receipt dates.
Nonstationary weather and closures make this a triangular kernel product,
not necessarily a stationary np.convolve. Categorical path draws preserve
population and lineage; independent Poisson draws from the same means do not.
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.util import log_density
from numpyro_forecast import Horizon, draw_posterior, forecast
from prototype_native_common import (
    TRUTH,
    chain_prefix_supported,
    check_paths,
    fit_and_forecast,
    make_case,
    parameters,
    prefix_supported,
    sample_theta,
)

from ttenet.models import StageObservations, stage_log_likelihood


def calendar_kernel(case, theta, duration, deadline=None):
    """Exact first-event probability for every origin, date and unit."""
    p = parameters(theta)
    ages = jnp.arange(duration)[None, :, None] - jnp.arange(duration)[:, None, None]
    logits = p.age_logits[jnp.clip(ages, 0, len(p.age_logits) - 1)]
    logits = (
        logits
        + jnp.einsum("ntp,p->tn", jnp.asarray(case.features[:, :duration]), p.beta)[None, :, :]
    )
    eligible = (ages >= 0) & jnp.asarray(case.allowed[:, :duration].T)[None, :, :]
    if deadline is not None:
        eligible = eligible & (ages <= deadline)
    log_failure = jnp.where(eligible, jax.nn.log_sigmoid(-logits), 0.0)
    log_before = jnp.cumsum(log_failure, axis=1) - log_failure
    log_pi = jax.nn.log_sigmoid(p.cure_intercept + jnp.asarray(case.cure_features) @ p.cure_beta)
    return jnp.where(eligible, jnp.exp(log_pi + log_before + jax.nn.log_sigmoid(logits)), 0.0)


def initiation_masses(case, theta, duration):
    # Full original clocks needed even for left-truncated selected survivors.
    kernel = calendar_kernel(case, theta, len(case.calendar), case.deadline)
    mass = kernel[jnp.asarray(case.origin), :, jnp.arange(len(case.origin))].T
    before_entry = jnp.arange(len(case.calendar))[:, None] < jnp.asarray(case.entry)[None, :]
    survival_entry = 1 - jnp.where(before_entry, mass, 0).sum(axis=0)
    return jnp.where(before_entry, 0.0, mass / survival_entry)[:duration]


def receipt_masses(kernel, initiation):
    known = initiation.any(axis=0)
    origin = jnp.argmax(initiation, axis=0)
    mass = kernel[origin, :, jnp.arange(initiation.shape[-1])].T
    return jnp.where(known[None, :], mass, 0.0)


def categorical_probabilities(mass):
    none = jnp.maximum(0.0, 1 - mass.sum(axis=0, keepdims=True))
    return jnp.concatenate([mass, none], axis=0).T


def event_distribution(mass):
    probabilities = categorical_probabilities(mass)
    positive = probabilities > 0
    # Structural zeros must stay impossible. Avoid differentiating log(0)
    # even on categories not selected by the observed event date.
    logits = jnp.where(positive, jnp.log(jnp.where(positive, probabilities, 1.0)), -jnp.inf)
    return dist.Categorical(logits=logits).to_event(1)


def observe_times(name, mass, data):
    stop = data.shape[0]
    event_time = jnp.where(data.any(axis=0), jnp.argmax(data, axis=0), stop)
    numpyro.sample(name, event_distribution(mass[:stop]), obs=event_time)


def conditional_future(mass, data):
    stop = data.shape[0]
    survived = 1 - mass[:stop].sum(axis=0)
    return jnp.where(data.any(axis=0)[None, :], 0.0, mass[stop:] / survived)


def draw_times(name, mass):
    date = numpyro.sample(name, event_distribution(mass))
    return (jnp.arange(mass.shape[0])[:, None] == date[None, :]).astype(jnp.int32)


def make_model(case, chained=False):
    n = len(case.origin)

    def model(covariates, data=None):
        h = Horizon.from_data(covariates, data)
        theta = sample_theta()
        init_data = data[:, :n]
        supported = chain_prefix_supported(case, data) if chained else prefix_supported(case, data)
        numpyro.factor("support", jnp.where(supported, 0.0, -jnp.inf))
        mass = initiation_masses(case, theta, h.duration)
        observe_times("initiation_time", mass, init_data)
        if h.future:
            future = draw_times("initiation_time_future", conditional_future(mass, init_data))
            initiation = jnp.concatenate([init_data, future], axis=0)
        else:
            initiation = init_data
        if chained:
            # A shared theta is deliberate in this small experiment, not a claim
            # that the two production stages have identical parameters.
            kernel = calendar_kernel(case, theta, h.duration)
            rec_mass = receipt_masses(kernel, initiation)
            rec_data = data[:, n:]
            observe_times("receipt_time", rec_mass, rec_data)
            if h.future:
                rec_future = draw_times(
                    "receipt_time_future", conditional_future(rec_mass, rec_data)
                )
                future = jnp.concatenate([future, rec_future], axis=-1)
        if h.future:
            numpyro.deterministic("forecast", future)

    return model


def chain_experiment(case, draws=8192):
    theta = jnp.asarray(TRUTH)
    n, stop, total = len(case.origin), case.t_obs, len(case.calendar)
    init_data = jnp.asarray(case.data[:stop])
    rec_data = np.zeros_like(init_data)
    historical = np.flatnonzero(np.asarray(init_data).any(axis=0))
    assert len(historical) >= 2
    completed = historical[0]
    rec_data[int(jnp.argmax(init_data[:, completed])), completed] = 1
    data = jnp.concatenate([init_data, jnp.asarray(rec_data)], axis=-1)
    model = make_model(case, chained=True)
    # Compare the actual two-stage native trace against both existing unit likelihoods.
    origin = jnp.where(init_data.any(axis=0), jnp.argmax(init_data, axis=0), total + 1)
    event = jnp.where(jnp.asarray(rec_data).any(axis=0), jnp.argmax(rec_data, axis=0), -1)
    day = jnp.arange(stop)[None, :]
    receipt_obs = StageObservations(
        ages=day - origin[:, None],
        features=jnp.asarray(case.features[:, :stop]),
        cure_features=jnp.asarray(case.cure_features),
        allowed=jnp.asarray(case.allowed[:, :stop]),
        at_risk=(day >= origin[:, None]) & (day <= jnp.where(event >= 0, event, stop - 1)[:, None]),
        event_index=event,
    )

    def chain_density(t):
        return log_density(model, (case.covariates[:stop], data), {}, {"theta": t})[0]

    def reference_density(t):
        return (
            case.reference(t).sum()
            + stage_log_likelihood(parameters(t), receipt_obs).sum()
            + dist.Normal(jnp.zeros(5), 1).log_prob(t).sum()
        )

    np.testing.assert_allclose(chain_density(theta), reference_density(theta), atol=2e-5)
    np.testing.assert_allclose(
        jax.grad(chain_density)(theta), jax.grad(reference_density)(theta), atol=2e-5
    )
    guide = AutoNormal(model)
    svi = SVI(model, guide, numpyro.optim.Adam(0.025), Trace_ELBO())
    fitted = svi.run(random.PRNGKey(205), 100, case.covariates[:stop], data, progress_bar=False)
    assert np.isfinite(fitted.losses).all()
    learned = draw_posterior(random.PRNGKey(206), guide, fitted.params, 128)
    learned_paths = np.asarray(forecast(random.PRNGKey(207), model, learned, data, case.covariates))
    check_paths(case, learned_paths[:, :, :n])
    learned_init = np.concatenate(
        [np.broadcast_to(init_data, (128, stop, n)), learned_paths[:, :, :n]], axis=1
    )
    learned_rec = np.concatenate(
        [np.broadcast_to(rec_data, (128, stop, n)), learned_paths[:, :, n:]], axis=1
    )
    assert (learned_rec.cumsum(axis=1) <= learned_init.cumsum(axis=1)).all()
    assert (learned_rec.sum(axis=1) <= 1).all()
    assert not learned_paths[:, :, n:][:, ~case.allowed[:, stop:].T].any()
    posterior = {"theta": jnp.broadcast_to(theta, (draws, 5))}
    samples = np.asarray(forecast(random.PRNGKey(201), model, posterior, data, case.covariates))
    initiation, receipts = samples[:, :, :n], samples[:, :, n:]
    check_paths(case, initiation)
    full_init = np.concatenate([np.broadcast_to(init_data, (draws, stop, n)), initiation], axis=1)
    full_rec = np.concatenate([np.broadcast_to(rec_data, (draws, stop, n)), receipts], axis=1)
    assert (full_rec.sum(axis=1) <= 1).all()
    assert (full_rec.cumsum(axis=1) <= full_init.cumsum(axis=1)).all()
    assert not receipts[:, ~case.allowed[:, stop:].T].any()
    assert receipts.any()
    # Exact survival convolution for uninitiated units plus residual-life
    # forecasts for already-open returns, with already-received items removed.
    init_mass = initiation_masses(case, theta, total)
    init_future = conditional_future(init_mass, init_data)
    parent_mass = jnp.concatenate([jnp.zeros((stop, n)), init_future], axis=0)
    receipt_kernel = calendar_kernel(case, theta, total)
    composed = jnp.einsum("sn,stn->tn", parent_mass, receipt_kernel)[stop:]
    open_mass = receipt_masses(receipt_kernel, init_data)
    expected = np.asarray(composed + conditional_future(open_mass, jnp.asarray(rec_data)))
    error = float(np.max(abs(receipts.mean(axis=0) - expected)))
    assert error < 0.025
    assert np.all(expected[:, completed] == 0)
    # The same expectation does not imply the same stochastic process.
    independent = np.random.default_rng(204).poisson(expected, size=(draws,) + expected.shape)
    poisson_violations = int(np.any(independent.sum(axis=1) > 1, axis=1).sum())
    assert poisson_violations > 0
    return {
        "shape": list(samples.shape),
        "kernel_composition_pmf_error": error,
        "joint_chain_loglik_error": float(abs(chain_density(theta) - reference_density(theta))),
        "joint_chain_gradient_error": float(
            jnp.max(abs(jax.grad(chain_density)(theta) - jax.grad(reference_density)(theta)))
        ),
        "joint_fit_shape": list(learned_paths.shape),
        "joint_fit_final_elbo": float(fitted.losses[-1]),
        "lineage_violations": 0,
        "known_received_items": 1,
        "known_open_items": len(historical) - 1,
        "independent_poisson_draws_exceeding_one_receipt_per_unit": poisson_violations,
        "comparison_draws": draws,
        "kernel_shape": list(receipt_kernel.shape),
    }


def run(steps=100, draws=128):
    case = make_case()
    model = make_model(case)
    x, y = case.covariates[: case.t_obs], jnp.asarray(case.data[: case.t_obs])
    ll_errors, gradient_errors = [], []
    for value in (TRUTH, TRUTH + 0.3, TRUTH - 0.5):
        theta = jnp.asarray(value)

        def native(t):
            return log_density(model, (x, y), {}, {"theta": t})[0]

        def baseline(t):
            return case.reference(t).sum() + dist.Normal(jnp.zeros(5), 1).log_prob(t).sum()

        ll_errors.append(float(abs(native(theta) - baseline(theta))))
        gradient_errors.append(
            float(jnp.max(abs(jax.grad(native)(theta) - jax.grad(baseline)(theta))))
        )
        np.testing.assert_allclose(native(theta), baseline(theta), atol=2e-5)
        np.testing.assert_allclose(jax.grad(native)(theta), jax.grad(baseline)(theta), atol=2e-5)
    fitted = fit_and_forecast(model, case, steps, draws)
    theta = jnp.asarray(TRUTH)
    mass = conditional_future(initiation_masses(case, theta, len(case.calendar)), y)
    np.testing.assert_allclose(mass, case.analytic_future(theta), atol=2e-6)
    return {
        "approach": "survival convolution kernels + exact categorical event times",
        "max_loglik_error": max(ll_errors),
        "max_gradient_error": max(gradient_errors),
        "fit": fitted,
        "chain": chain_experiment(case),
        "limitation": "Dense [origin,date,unit] kernels scale quadratically in calendar length; not a production replacement.",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
