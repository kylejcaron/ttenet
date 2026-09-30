"""Focused experimental survival-convolution model, native NumPyro Forecast end to end.

Run: uv run --extra forecast python examples/survival_convolution.py

The fixture is 12 identified units on 42 calendar days with an observed prefix
of 24 days. It includes selected survivors, future births, heterogeneous
covariates, a deadline and closures. No Markov/Poisson/adapter alternatives are
included. This is not a production API cutover or a calibration benchmark.
"""

import argparse
import json
from importlib.metadata import version

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from _survival_convolution_fixture import TRUTH, check_paths, make_case, parameters, sample_theta
from jax import random
from numpyro import handlers
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.util import log_density
from numpyro_forecast import Horizon, draw_posterior, forecast, predict, predict_in_sample
from numpyro_forecast.surgery import prefix_condition, slice_time
from survival_convolution_model import EventTime, convolve_event_times, survival_kernel

from ttenet.models import StageObservations, stage_log_likelihood


def covariates(case):
    """Pack real calendar features and closures into the upstream [time, feature] layout."""
    values = np.concatenate([case.features.transpose(1, 0, 2), case.allowed.T[..., None]], axis=-1)
    return jnp.asarray(values.reshape(len(case.calendar), -1))


def kernel(case, theta, x, origin, *, entry=None, deadline=None):
    inputs = x.reshape(x.shape[0], len(case.origin), -1)
    return survival_kernel(
        parameters(theta),
        origin,
        inputs[..., :-1],
        jnp.asarray(case.cure_features),
        inputs[..., -1] > 0.5,
        entry=entry,
        deadline=deadline,
    )


def observe_and_predict(h, law):
    predict(h, lambda mass: EventTime(mass, law.log_tail), law.log_mass)


def make_model(case, chained=False):
    n = len(case.origin)

    def model(x, data=None):
        h = Horizon.from_data(x, data)
        if not chained:
            law = kernel(
                case, sample_theta(), x, case.origin, entry=case.entry, deadline=case.deadline
            )
            observe_and_predict(h, law)
            return
        init_data = None if data is None else data[:, :n]
        rec_data = None if data is None else data[:, n:]
        with handlers.trace() as parent, handlers.scope(prefix="initiation"):
            law = kernel(
                case, sample_theta(), x, case.origin, entry=case.entry, deadline=case.deadline
            )
            observe_and_predict(Horizon.from_data(x, init_data), law)
        if data is None:
            initiation = parent["initiation/obs"]["value"]
        elif h.future:
            initiation = jnp.concatenate([init_data, parent["initiation/obs_future"]["value"]])
        else:
            initiation = init_data
        origin = jnp.where(initiation.any(axis=0), jnp.argmax(initiation, axis=0), h.duration)
        with handlers.trace() as child, handlers.scope(prefix="receipt"):
            law = kernel(case, sample_theta(), x, origin)
            observe_and_predict(Horizon.from_data(x, rec_data), law)
        if h.future:
            numpyro.deterministic(
                "forecast",
                jnp.concatenate(
                    [
                        parent["initiation/obs_future"]["value"],
                        child["receipt/obs_future"]["value"],
                    ],
                    axis=-1,
                ),
            )
        elif data is None:
            # Expose a joint prior predictive without copying the training ledger
            # into the guide's deterministic posterior sites.
            numpyro.deterministic(
                "obs",
                jnp.concatenate(
                    [
                        parent["initiation/obs"]["value"],
                        child["receipt/obs"]["value"],
                    ],
                    axis=-1,
                ),
            )

    return model


def fit(model, x, data, steps, draws):
    guide = AutoNormal(model)
    svi = SVI(model, guide, numpyro.optim.Adam(0.025), Trace_ELBO())
    fitted = svi.run(random.PRNGKey(101), steps, x[: len(data)], data, progress_bar=False)
    assert np.isfinite(fitted.losses).all()
    posterior = draw_posterior(random.PRNGKey(102), guide, fitted.params, draws)
    paths = np.asarray(forecast(random.PRNGKey(103), model, posterior, data, x))
    return posterior, paths, float(fitted.losses[-1])


def distribution_checks():
    mass = np.array([[0.2, 0.0], [0.0, 0.5], [0.3, 0.0]])
    log_mass = np.full_like(mass, -np.inf)
    np.log(mass, out=log_mass, where=mass > 0)
    law = EventTime(jnp.asarray(log_mass), jnp.log(jnp.array([0.5, 0.5])))
    paths = jnp.concatenate([jnp.zeros((1, 3)), jnp.eye(3)]).astype(jnp.int32)
    paths = jnp.broadcast_to(paths[:, :, None], (4, 3, 2))
    np.testing.assert_allclose(jnp.exp(law.unit_log_prob(paths)).sum(axis=0), 1, atol=2e-7)
    window = slice_time(law, slice(1, 3))
    np.testing.assert_allclose(window.mean, mass[1:], atol=1e-7)
    prefix = jnp.array([[1, 0]], dtype=jnp.int32)
    suffix = prefix_condition(law, prefix)
    np.testing.assert_allclose(suffix.mean, np.array([[0.0, 0.5], [0.0, 0.0]]), atol=1e-7)
    full = jnp.array([[1, 0], [0, 1], [0, 0]])
    split_logp = slice_time(law, slice(0, 1)).log_prob(prefix) + suffix.log_prob(full[1:])
    np.testing.assert_allclose(split_logp, law.log_prob(full), atol=1e-7)
    for bad in (full.at[2, 0].set(1), full.at[0, 1].set(1), full.at[0, 0].set(2)):
        assert np.isneginf(law.log_prob(bad))
    samples = np.asarray(law.sample(random.PRNGKey(81), (8192,)))
    assert (samples.sum(axis=1) <= 1).all() and not samples[:, mass == 0].any()
    # Includes zero-event windows, batched laws, and differentiable structural zeros.
    empty = slice_time(law, slice(0, 0))
    assert float(empty.log_prob(jnp.zeros((0, 2), dtype=jnp.int32))) == 0.0
    batched = EventTime(jnp.stack([law.log_mass, law.log_mass]), law.log_tail)
    assert batched.sample(random.PRNGKey(82), (4,)).shape == (4, 2, 3, 2)
    gradient = jax.grad(lambda log_mass: EventTime(log_mass, law.log_tail).log_prob(full))(
        law.log_mass
    )
    assert np.isfinite(gradient).all()
    return {"normalization_and_conditioning": "passed", "impossible_paths_rejected": 3}


def likelihood_checks(case, model, x, data):
    worst_ll, worst_grad = 0.0, 0.0
    for stop in (10, case.t_obs, len(case.calendar)):

        def native(theta):
            return log_density(model, (x[:stop], data[:stop]), {}, {"theta": theta})[0]

        def reference(theta):
            return case.reference(theta, stop).sum() + dist.Normal(0, 1).log_prob(theta).sum()

        for values in (TRUTH, TRUTH + 0.3, TRUTH - 0.5):
            theta = jnp.asarray(values)
            ll_error = float(abs(native(theta) - reference(theta)))
            grad_error = float(jnp.max(abs(jax.grad(native)(theta) - jax.grad(reference)(theta))))
            assert ll_error < 3e-5 and grad_error < 3e-5
            worst_ll, worst_grad = max(worst_ll, ll_error), max(worst_grad, grad_error)
    return {"max_loglik_error": worst_ll, "max_gradient_error": worst_grad}


def check_chain(case, history, paths):
    n, stop = len(case.origin), len(history)
    check_paths(case, paths[:, :, :n])
    full = np.concatenate([np.broadcast_to(history, (len(paths), stop, 2 * n)), paths], axis=1)
    initiation, receipt = full[:, :, :n], full[:, :, n:]
    assert (receipt.sum(axis=1) <= 1).all()
    assert (receipt.cumsum(axis=1) <= initiation.cumsum(axis=1)).all()
    assert not paths[:, :, n:][:, ~case.allowed[:, stop:].T].any()


def chain_experiment(case, x, steps, draws):
    n, stop = len(case.origin), case.t_obs
    initiation = jnp.asarray(case.data[:stop])
    receipt = np.zeros_like(initiation)
    historical = np.flatnonzero(np.asarray(initiation).any(axis=0))
    completed = historical[0]
    receipt[int(jnp.argmax(initiation[:, completed])), completed] = 1
    history = jnp.concatenate([initiation, jnp.asarray(receipt)], axis=-1)
    model = make_model(case, chained=True)
    theta = jnp.asarray(TRUTH)
    theta_receipt = theta + jnp.array([0.5, 0, 0, 0.5, 0])
    origin = jnp.where(initiation.any(axis=0), initiation.argmax(axis=0), len(x))
    event = jnp.where(jnp.asarray(receipt).any(axis=0), jnp.argmax(receipt, axis=0), -1)
    day = jnp.arange(stop)[None, :]
    receipt_obs = StageObservations(
        ages=day - origin[:, None],
        features=jnp.asarray(case.features[:, :stop]),
        cure_features=jnp.asarray(case.cure_features),
        allowed=jnp.asarray(case.allowed[:, :stop]),
        at_risk=(day >= origin[:, None]) & (day <= jnp.where(event >= 0, event, stop - 1)[:, None]),
        event_index=event,
    )

    def native(values):
        return log_density(
            model,
            (x[:stop], history),
            {},
            {
                "initiation/theta": values[:5],
                "receipt/theta": values[5:],
            },
        )[0]

    def reference(values):
        return (
            case.reference(values[:5]).sum()
            + stage_log_likelihood(parameters(values[5:]), receipt_obs).sum()
            + dist.Normal(0, 1).log_prob(values).sum()
        )

    values = jnp.concatenate([theta, theta_receipt])
    np.testing.assert_allclose(native(values), reference(values), atol=3e-5)
    np.testing.assert_allclose(jax.grad(native)(values), jax.grad(reference)(values), atol=3e-5)
    posterior, fitted_paths, loss = fit(model, x, history, steps, draws)
    check_chain(case, history, fitted_paths)
    in_sample = np.asarray(predict_in_sample(random.PRNGKey(92), model, posterior, x[:stop]))
    assert in_sample.shape == (draws, stop, 2 * n)
    assert (in_sample.sum(axis=1) <= 1).all()
    assert (in_sample[:, :, n:].cumsum(axis=1) <= in_sample[:, :, :n].cumsum(axis=1)).all()
    predicted_mass = np.asarray(
        jax.vmap(
            lambda theta: (
                kernel(
                    case, theta, x[:stop], case.origin, entry=case.entry, deadline=case.deadline
                ).mean
            )
        )(posterior["initiation/theta"])
    )
    standard_error = np.sqrt((predicted_mass * (1 - predicted_mass)).sum(axis=0)) / draws
    difference = in_sample[:, :, :n].mean(axis=0) - predicted_mass.mean(axis=0)
    assert np.all(difference[standard_error == 0] == 0)
    predictive_z = float(
        np.max(abs(difference[standard_error > 0] / standard_error[standard_error > 0]))
    )
    assert predictive_z < 6, "in-sample draws disagree with their posterior predictive law"
    fixed = {
        "initiation/theta": jnp.broadcast_to(theta, (8192, 5)),
        "receipt/theta": jnp.broadcast_to(theta_receipt, (8192, 5)),
    }
    paths = np.asarray(forecast(random.PRNGKey(93), model, fixed, history, x))
    check_chain(case, history, paths)
    # Survival convolution over all possible future initiation dates, plus
    # conditional residual life of already-open returns. Known receipts stay absorbed.
    law = kernel(case, theta, x, case.origin, entry=case.entry, deadline=case.deadline)
    future_mass = prefix_condition(law, initiation).mean
    parent_mass = jnp.concatenate([jnp.zeros((stop, n)), future_mass])
    origins = jnp.broadcast_to(jnp.arange(len(x))[:, None], (len(x), n))
    receipt_kernel = kernel(case, theta_receipt, x, origins)
    expected = convolve_event_times(parent_mass, receipt_kernel)[stop:]
    open_law = kernel(case, theta_receipt, x, origin)
    expected += prefix_condition(open_law, jnp.asarray(receipt)).mean
    empirical = paths[:, :, n:].mean(axis=0)
    error = float(np.max(abs(empirical - np.asarray(expected))))
    assert error < 0.025
    return {
        "joint_loglik_error": float(abs(native(values) - reference(values))),
        "joint_gradient_error": float(
            jnp.max(abs(jax.grad(native)(values) - jax.grad(reference)(values)))
        ),
        "fitted_shape": list(fitted_paths.shape),
        "posterior_sites": sorted(posterior),
        "predict_in_sample_shape": list(in_sample.shape),
        "predictive_max_z": predictive_z,
        "final_elbo": loss,
        "convolution_draws": 8192,
        "convolution_max_probability_error": error,
        "lineage_violations": 0,
        "known_received": 1,
        "known_open": len(historical) - 1,
    }


def run(steps=100, draws=128):
    case = make_case()
    x, data = covariates(case), jnp.asarray(case.data)
    model = make_model(case)
    distribution = distribution_checks()
    parity = likelihood_checks(case, model, x, data)
    history = data[: case.t_obs]
    posterior, paths, loss = fit(model, x, history, steps, draws)
    check_paths(case, paths)
    prior = predict_in_sample(random.PRNGKey(104), model, posterior, x[: case.t_obs])
    assert prior.shape == (draws, case.t_obs, len(case.origin))
    fixed = {"theta": jnp.broadcast_to(jnp.asarray(TRUTH), (8192, 5))}
    samples = np.asarray(forecast(random.PRNGKey(105), model, fixed, history, x))
    check_paths(case, samples)
    law = kernel(case, TRUTH, x, case.origin, entry=case.entry, deadline=case.deadline)
    np.testing.assert_allclose(
        prefix_condition(law, history).mean, case.analytic_future(TRUTH), atol=2e-6
    )
    short = forecast(random.PRNGKey(106), model, posterior, history, x[: case.t_obs + 3])
    assert short.shape == (draws, 3, len(case.origin))
    # Future outcomes are not model inputs; verify that changing hidden fixture
    # labels cannot affect forecasts and that actual future covariates do affect them.
    case.data[case.t_obs :] = 1 - case.data[case.t_obs :]
    np.testing.assert_array_equal(samples, forecast(random.PRNGKey(105), model, fixed, history, x))
    case.data[case.t_obs :] = 1 - case.data[case.t_obs :]
    closed = x.reshape(len(x), len(case.origin), -1).at[case.t_obs :, :, -1].set(0).reshape(x.shape)
    no_events = forecast(random.PRNGKey(107), model, posterior, history, closed)
    assert not np.asarray(no_events).any() and samples.any()
    return {
        "numpyro_forecast_version": version("numpyro-forecast"),
        "distribution": distribution,
        "likelihood": parity,
        "single_stage": {
            "forecast_shape": list(paths.shape),
            "final_elbo": loss,
            "predict_in_sample_shape": list(prior.shape),
            "closure_scenario": "zero events",
            "hidden_outcome_leakage": False,
        },
        "chain": chain_experiment(case, x, steps, draws),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--draws", type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(run(args.steps, args.draws), indent=2))
