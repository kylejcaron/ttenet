"""Throwaway exact factor + categorical first-event sampler, native forecast driver.

Unlike predict(), this adapter supplies the existing cure likelihood directly.
It retains unit identities and calendar clocks but does not provide an obs
sample site for upstream predict_in_sample/to_datatree diagnostics.
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer.util import log_density
from numpyro_forecast import Horizon, forecast
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


def observations(case, data):
    """Build the existing likelihood input from the supplied prefix, never truth suffix."""
    stop = data.shape[-2]
    day = jnp.arange(stop)[None, :]
    origin, entry = jnp.asarray(case.origin)[:, None], jnp.asarray(case.entry)[:, None]
    event_index = jnp.where(jnp.any(data, axis=0), jnp.argmax(data, axis=0), -1)
    end = jnp.where(event_index >= 0, event_index, stop - 1)[:, None]
    return StageObservations(
        ages=day - origin,
        features=jnp.asarray(case.features[:, :stop]),
        cure_features=jnp.asarray(case.cure_features),
        at_risk=(day >= entry) & (day <= end) & (day <= origin + case.deadline),
        allowed=jnp.asarray(case.allowed[:, :stop]),
        event_index=event_index,
        pre_entry=(day >= origin) & (day < entry) & (day <= origin + case.deadline),
    )


def sample_first_event(name, q, pending):
    """One categorical draw over future dates plus the no-event outcome, per unit."""
    q = jnp.where(pending[None, :], q, 0.0)
    survival = jnp.cumprod(1 - q, axis=0)
    before = jnp.concatenate([jnp.ones_like(q[:1]), survival[:-1]], axis=0)
    masses = jnp.concatenate([q * before, survival[-1:]], axis=0)
    date = numpyro.sample(name, dist.Categorical(probs=masses.T).to_event(1))
    return (jnp.arange(q.shape[0])[:, None] == date[None, :]).astype(jnp.int32)


def make_model(case):
    def model(covariates, data=None):
        h = Horizon.from_data(covariates, data)
        theta = sample_theta()
        if data is not None:
            numpyro.factor("support", jnp.where(prefix_supported(case, data), 0.0, -jnp.inf))
            numpyro.factor(
                "cure_likelihood", stage_log_likelihood(parameters(theta), observations(case, data))
            )
        if h.future:
            q, _ = case.arrays(theta, h.duration)
            paths = sample_first_event("first_event_future", q[h.t_obs :], ~jnp.any(data, axis=0))
            numpyro.deterministic("forecast", paths)

    return model


def receipt_q(case, theta, origin, duration):
    """Receipt clock starts on the sampled initiation date, including same-day events."""
    p = parameters(theta)
    ages = jnp.arange(duration)[:, None] - origin[None, :]
    logits = p.age_logits[jnp.clip(ages, 0, len(p.age_logits) - 1)]
    logits += jnp.einsum("ntp,p->tn", jnp.asarray(case.features[:, :duration]), p.beta)
    opened = (ages >= 0) & jnp.asarray(case.allowed[:, :duration].T)
    log_day = jnp.where(opened, jax.nn.log_sigmoid(-logits), 0.0)
    log_before = jnp.cumsum(log_day, axis=0) - log_day
    cure = p.cure_intercept + jnp.asarray(case.cure_features) @ p.cure_beta
    return jnp.where(opened, jax.nn.sigmoid(cure + log_before) * jax.nn.sigmoid(logits), 0.0)


def make_chain_model(case):
    n = len(case.origin)

    def model(covariates, data=None):
        h = Horizon.from_data(covariates, data)
        theta = sample_theta()
        receipt_theta = numpyro.sample("receipt_theta", dist.Normal(jnp.zeros(5), 1.0).to_event(1))
        init_data, rec_data = data[:, :n], data[:, n:]
        numpyro.factor("support", jnp.where(chain_prefix_supported(case, data), 0.0, -jnp.inf))
        numpyro.factor(
            "initiation_likelihood",
            stage_log_likelihood(parameters(theta), observations(case, init_data)),
        )
        # Unknown parent origins are after the entire horizon, never silently age zero.
        past_origin = jnp.where(
            init_data.any(axis=0), jnp.argmax(init_data, axis=0), h.duration + 1
        )
        q_past = receipt_q(case, receipt_theta, past_origin, h.t_obs)
        rec_alive = jnp.cumsum(rec_data, axis=0) - rec_data == 0
        probs = jnp.where(rec_alive, q_past, 0.0)
        numpyro.factor("receipt_likelihood", dist.Bernoulli(probs=probs).log_prob(rec_data).sum())
        if h.future:
            q, _ = case.arrays(theta, h.duration)
            initiation = sample_first_event(
                "initiation_future", q[h.t_obs :], ~init_data.any(axis=0)
            )
            combined = jnp.concatenate([init_data, initiation], axis=0)
            origin = jnp.where(combined.any(axis=0), jnp.argmax(combined, axis=0), h.duration + 1)
            receipts = sample_first_event(
                "receipt_future",
                receipt_q(case, receipt_theta, origin, h.duration)[h.t_obs :],
                ~rec_data.any(axis=0),
            )
            numpyro.deterministic("forecast", jnp.concatenate([initiation, receipts], axis=-1))

    return model


def chain_smoke(case, draws=2048):
    """Real native driver over two stages, with historical open and received items."""
    n = len(case.origin)
    # Synthetic full receipt ledger generated from the actual full initiation ledger.
    origin = np.where(case.data.any(axis=0), case.data.argmax(axis=0), len(case.calendar) + 1)
    q = np.asarray(receipt_q(case, jnp.asarray(TRUTH), jnp.asarray(origin), len(case.calendar)))
    rng = np.random.default_rng(73)
    received = np.zeros_like(case.data)
    pending = np.ones(n, dtype=bool)
    for day in range(len(case.calendar)):
        received[day] = pending & (rng.random(n) < q[day])
        pending &= ~received[day].astype(bool)
    historical = np.flatnonzero(case.data[: case.t_obs].any(axis=0))
    assert len(historical) >= 2
    # Force both known-received and still-open states to be exercised, on valid dates.
    received[:, historical[0]] = 0
    received[origin[historical[0]], historical[0]] = 1
    received[:, historical[1]] = 0
    history = np.concatenate([case.data[: case.t_obs], received[: case.t_obs]], axis=-1)
    model = make_chain_model(case)
    posterior = {
        "theta": jnp.broadcast_to(TRUTH, (draws, 5)),
        "receipt_theta": jnp.broadcast_to(TRUTH, (draws, 5)),
    }
    output = np.asarray(
        forecast(random.PRNGKey(74), model, posterior, jnp.asarray(history), case.covariates)
    )
    init = output[:, :, :n]
    rec = output[:, :, n:]
    check_paths(case, init)
    full_init = np.concatenate(
        [np.broadcast_to(history[:, :n], (draws, case.t_obs, n)), init], axis=1
    )
    full_rec = np.concatenate(
        [np.broadcast_to(history[:, n:], (draws, case.t_obs, n)), rec], axis=1
    )
    assert np.all(full_rec.sum(axis=1) <= 1)
    assert np.all(full_rec.cumsum(axis=1) <= full_init.cumsum(axis=1)), (
        "receipt before its own parent"
    )
    assert not np.any(rec[:, ~case.allowed[:, case.t_obs :].T])
    assert np.any(rec) and np.any(init), "fixture failed to exercise transitions"
    return {
        "shape": list(output.shape),
        "lineage_violations": 0,
        "historical_receipts": int(history[:, n:].sum()),
        "future_receipts": int(rec.sum()),
    }


def run(steps=100, draws=128):
    case = make_case()
    model = make_model(case)
    x, y = case.covariates[: case.t_obs], jnp.asarray(case.data[: case.t_obs])
    errors, gradients = [], []
    for value in (TRUTH, TRUTH + 0.3, TRUTH - 0.5):
        theta = jnp.asarray(value)

        def native(t):
            return log_density(model, (x, y), {}, {"theta": t})[0]

        def baseline(t):
            prior = dist.Normal(jnp.zeros(5), 1.0).log_prob(t).sum()
            return case.reference(t).sum() + prior

        errors.append(float(abs(native(theta) - baseline(theta))))
        gradients.append(
            float(jnp.max(jnp.abs(jax.grad(native)(theta) - jax.grad(baseline)(theta))))
        )
        np.testing.assert_allclose(native(theta), baseline(theta), atol=2e-5)
        np.testing.assert_allclose(jax.grad(native)(theta), jax.grad(baseline)(theta), atol=2e-5)
    fitted = fit_and_forecast(model, case, steps, draws)
    posterior = {"theta": jnp.broadcast_to(jnp.asarray(TRUTH), (8192, 5))}
    fixed = np.asarray(forecast(random.PRNGKey(104), model, posterior, y, case.covariates))
    check_paths(case, fixed)
    expected = case.analytic_future(jnp.asarray(TRUTH))
    pmf_error = float(np.max(abs(fixed.mean(axis=0) - expected)))
    assert pmf_error < 0.025
    # A native forecast driver can run the same model at a shorter horizon.
    short = forecast(
        random.PRNGKey(105), model, {"theta": posterior["theta"][:32]}, y, case.covariates[:28]
    )
    assert short.shape == (32, 4, len(case.origin))
    # Mutating hidden outcomes does not change the conditioned forecast.
    case.data[case.t_obs :] = 1 - case.data[case.t_obs :]
    replay = forecast(random.PRNGKey(104), model, posterior, y, case.covariates)
    np.testing.assert_array_equal(fixed, replay)
    case.data[case.t_obs :] = 1 - case.data[case.t_obs :]
    return {
        "approach": "existing likelihood + first-event sampler + native forecast",
        "max_loglik_error": max(errors),
        "max_gradient_error": max(gradients),
        "fixed_parameter_pmf_error": pmf_error,
        "fit": fitted,
        "short_horizon_shape": list(short.shape),
        "hidden_outcome_leakage": False,
        "chain": chain_smoke(case),
        "limitation": "No obs sample site: forecast/draw_posterior work, predict_in_sample is not supported.",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
