"""Throwaway exact single-event trajectory distribution on the native forecast driver.

Question: can one genuine ``numpyro.distributions.Distribution`` over per-unit
``[day, unit]`` event paths, registered with the upstream time-axis surgeries,
carry the existing cure / delayed-entry / closure likelihood through an
unmodified ``numpyro_forecast.predict`` + ``forecast``, including chained stages?
Run with: uv run --extra forecast python examples/prototype_native_distribution.py
"""

import json
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from jax.nn import log_sigmoid
from jax.scipy.special import logit, logsumexp
from numpyro import handlers
from numpyro.distributions import constraints
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.util import log_density
from numpyro_forecast import Horizon, draw_posterior, forecast, predict, predict_in_sample
from numpyro_forecast.surgery import prefix_condition, slice_time
from prototype_native_common import (
    TRUTH,
    check_paths,
    fit_and_forecast,
    make_case,
    parameters,
    sample_theta,
)

RECEIPT_TRUTH = np.array([1.0, -0.3, 0.1, 1.5, -0.6])
LOG_HALF = float(np.log(0.5))
# Float32 error model: per-term rounding 2^-24 on terms of magnitude <= ~10, summed
# over a few hundred day/unit cells with mixed signs; 5e-5 is ~10x the expected drift.
TOL = 5e-5


def log1mexp(x):
    """``log(1 - exp(x))`` for ``x <= 0``: exactly ``0`` at ``-inf`` with a zero gradient."""
    small = x < LOG_HALF
    return jnp.where(
        small,
        jnp.log1p(-jnp.exp(jnp.where(small, x, LOG_HALF))),
        jnp.log(-jnp.expm1(jnp.where(small, LOG_HALF, x))),
    )


def product_logit(a, b):
    """``logit(sigmoid(a) * sigmoid(b))`` without ever forming a probability."""
    return (
        log_sigmoid(a)
        + log_sigmoid(b)
        - jnp.logaddexp(log_sigmoid(-a), log_sigmoid(a) + log_sigmoid(-b))
    )


def exclusive_cumsum(x, axis):
    return jnp.cumsum(x, axis=axis) - x


class _ExtendedReal(constraints.ParameterFreeConstraint):
    """Finite or ``-inf`` (a structural zero); never ``+inf`` or NaN."""

    def __call__(self, x):
        return (x == x) & (x < jnp.inf)

    def feasible_like(self, prototype):
        return jnp.zeros_like(prototype)


class SingleEventTrajectory(dist.Distribution):
    """At-most-one-event paths over ``[time, unit]`` driven by conditional event log-odds.

    ``logits[..., t, n]`` is ``logit P(event at t | no event before t)`` for unit ``n``
    (``-inf`` on days the unit cannot fire: closed, unborn, past its deadline, before a
    delayed entry, or already absorbed). Cure and any survival before the window are
    already marginalized into that conditional hazard ``q``, so a path's probability is
    the product of conditional day terms: ``P(event at e) = q_e prod_{t<e} (1 - q_t)`` and
    ``P(no event) = prod_t (1 - q_t)``. The family is closed under time slicing and
    prefix conditioning, which is what makes the upstream surgeries exact.
    """

    arg_constraints = {"logits": _ExtendedReal()}
    support = constraints.independent(constraints.boolean, 2)

    def __init__(self, logits, *, validate_args=None):
        self.logits = jnp.asarray(logits)
        super().__init__(
            self.logits.shape[:-2], self.logits.shape[-2:], validate_args=validate_args
        )

    @property
    def log_hazard(self):
        return log_sigmoid(self.logits)

    @property
    def log_survival_day(self):
        return log_sigmoid(-self.logits)

    @property
    def mean(self):
        """Exact per-cell event probability: hazard times survival through the day before."""
        return jnp.exp(self.log_hazard + exclusive_cumsum(self.log_survival_day, -2))

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        hit = jnp.log(random.uniform(key, shape)) < self.log_hazard
        first = hit & (jnp.cumsum(hit, axis=-2) == 1)
        return first.astype(jnp.int32)

    def unit_log_prob(self, value):
        """Per-unit path log-probability ``[..., unit]``; ``-inf`` for impossible paths."""
        value = jnp.asarray(value)
        is_event = value == 1
        alive = exclusive_cumsum(value, -2) == 0
        day = jnp.where(is_event, self.log_hazard, jnp.where(alive, self.log_survival_day, 0.0))
        valid = jnp.all((value == 0) | is_event, axis=-2) & (jnp.sum(value, axis=-2) <= 1)
        return jnp.where(valid, jnp.sum(day, axis=-2), -jnp.inf)

    def log_prob(self, value):
        return jnp.sum(self.unit_log_prob(value), axis=-1)


@slice_time.register
def _(noise_dist: SingleEventTrajectory, index: slice) -> SingleEventTrajectory:
    """Exact marginal law of a time window; a prefix window is a plain parameter slice."""
    start, stop, step = index.indices(noise_dist.event_shape[0])
    if step != 1:
        raise NotImplementedError("single-event paths only slice contiguous windows")
    window = noise_dist.logits[..., start:stop, :]
    if start == 0:
        return SingleEventTrajectory(window)
    # A unit may already have fired before the window. Folding that absorbed mass into the
    # conditional hazard keeps the family closed: q'_e = q_e * P(alive at e | none in window).
    log_alive = jnp.sum(noise_dist.log_survival_day[..., :start, :], axis=-2, keepdims=True)
    certain = log_alive == 0
    safe = jnp.where(certain, -1.0, log_alive)
    logit_alive = jnp.where(certain, jnp.inf, safe - log1mexp(safe))
    cumulative = exclusive_cumsum(log_sigmoid(-window), -2)
    return SingleEventTrajectory(product_logit(window, logit_alive + cumulative))


@prefix_condition.register
def _(noise_dist: SingleEventTrajectory, data) -> SingleEventTrajectory:
    """Exact conditional of the suffix given the observed prefix.

    A unit that fired in the prefix is absorbed (all future logits ``-inf``); a unit that did
    not is event-free at ``t``, and its remaining conditional hazards are unchanged, still
    carrying its original age clock, pre-entry survival and marginalized cure.
    """
    t = data.shape[-2]
    pending = ~jnp.any(data != 0, axis=-2, keepdims=True)
    return SingleEventTrajectory(jnp.where(pending, noise_dist.logits[..., t:, :], -jnp.inf))


def stage_logits(case, theta, origin, entry, deadline, duration):
    """Conditional event log-odds ``[day, unit]`` for one cure stage clocked from ``origin``.

    ``q_t = sigmoid(cure + log S_t) * sigmoid(h_t)`` with ``S_t`` the susceptible survival
    from the original origin (pre-entry days included), i.e. exactly
    ``P(event at t | no event before t)`` with cure marginalized. Days before ``entry``
    accrue survival but cannot fire: the selected-survivor conditioning. ``-inf`` marks
    every day the unit cannot fire; ``deadline=None`` means unbounded exposure.
    """
    p = parameters(theta)
    day = jnp.arange(duration)[:, None]
    ages = day - origin[None, :]
    logits = p.age_logits[jnp.clip(ages, 0, len(p.age_logits) - 1)]
    logits = logits + jnp.einsum("ntp,p->tn", jnp.asarray(case.features[:, :duration]), p.beta)
    open_day = (ages >= 0) & jnp.asarray(case.allowed[:, :duration].T)
    if deadline is not None:
        open_day &= ages <= deadline
    log_survival_before = exclusive_cumsum(jnp.where(open_day, log_sigmoid(-logits), 0.0), 0)
    cure = p.cure_intercept + jnp.asarray(case.cure_features) @ p.cure_beta
    fire = open_day & (day >= entry[None, :])
    return jnp.where(fire, product_logit(cure[None, :] + log_survival_before, logits), -jnp.inf)


def initiation_logits(case, theta, duration):
    return stage_logits(
        case, theta, jnp.asarray(case.origin), jnp.asarray(case.entry), case.deadline, duration
    )


def make_model(case):
    def model(covariates, data=None):
        h = Horizon.from_data(covariates, data)
        theta = sample_theta()
        predict(h, SingleEventTrajectory, initiation_logits(case, theta, h.duration))

    return model


def receipt_logits(case, theta, origin, duration):
    """Receipt clock starts on each unit's own initiation day; unbounded, same-day allowed."""
    return stage_logits(case, theta, origin, origin, None, duration)


def make_chain_model(case):
    n = len(case.origin)

    def model(covariates, data=None):
        h = Horizon.from_data(covariates, data)
        init_data = None if data is None else data[..., :n]
        rec_data = None if data is None else data[..., n:]
        with handlers.trace() as parent, handlers.scope(prefix="initiation"):
            theta = sample_theta()
            predict(
                Horizon.from_data(covariates, init_data),
                SingleEventTrajectory,
                initiation_logits(case, theta, h.duration),
            )
        # Each unit's receipt clock starts on its own initiation day: observed in the prefix,
        # or this draw's sampled future initiation. Never initiated means never eligible.
        if data is None:
            initiation = parent["initiation/obs"]["value"]
        elif h.future:
            initiation = jnp.concatenate(
                [init_data, parent["initiation/obs_future"]["value"]], axis=-2
            )
        else:
            initiation = init_data
        origin = jnp.where(initiation.any(axis=-2), jnp.argmax(initiation, axis=-2), h.duration)
        with handlers.trace() as child, handlers.scope(prefix="receipt"):
            theta_receipt = sample_theta()
            predict(
                Horizon.from_data(covariates, rec_data),
                SingleEventTrajectory,
                receipt_logits(case, theta_receipt, origin, h.duration),
            )
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

    return model


def unit_paths(length):
    """Every at-most-one-event path of one unit: row 0 none, row k an event on day k-1."""
    return jnp.concatenate([jnp.zeros((1, length)), jnp.eye(length)], axis=0).astype(jnp.int32)


def path_table(d):
    """Per-unit log-probability of every path, ``[path, unit]``."""
    paths = unit_paths(d.event_shape[0])
    return d.unit_log_prob(jnp.broadcast_to(paths[:, :, None], paths.shape + d.event_shape[1:]))


def max_z(frequency, probability, draws):
    """Largest Monte Carlo z-score over non-degenerate cells (``0 < p < 1``)."""
    live = (probability > 0) & (probability < 1)
    se = np.sqrt(probability[live] * (1 - probability[live]) / draws)
    return float(np.max(np.abs(frequency[live] - probability[live]) / se))


def log_error(a, b):
    """Largest absolute log-probability difference; a shared ``-inf`` is agreement, a lone one is ``inf``."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    both = np.isneginf(a) & np.isneginf(b)
    error = float(np.max(np.abs(np.where(both, 0.0, a) - np.where(both, 0.0, b))))
    assert not np.isnan(error)
    return error


def distribution_probes(draws=200_000):
    """Exactness of the family itself on synthetic hazards with structural zeros."""
    boundary = np.array([-1e-8, -1e-4, -1.0, -20.0])
    np.testing.assert_allclose(
        log1mexp(jnp.asarray(boundary)),
        np.log(-np.expm1(boundary)),
        rtol=2e-6,
    )
    key_q, key_s = random.split(random.PRNGKey(7))
    logits = logit(random.uniform(key_q, (6, 3), minval=0.05, maxval=0.6))
    logits = logits.at[1, 0].set(-jnp.inf).at[:2, 2].set(-jnp.inf)
    d = SingleEventTrajectory(logits)
    table = np.asarray(path_table(d))
    normalization = float(np.max(np.abs(logsumexp(table, axis=0))))
    assert normalization < 1e-5, normalization

    samples = np.asarray(d.sample(key_s, (draws,)))
    assert samples.sum(axis=1).max() <= 1 and np.all(samples[:, np.isinf(np.asarray(logits))] == 0)
    index = np.where(samples.any(axis=1), samples.argmax(axis=1) + 1, 0)
    frequency = np.stack([(index == k).mean(axis=0) for k in range(7)])
    sampler_z = max_z(frequency, np.exp(table), draws)
    assert sampler_z < 5, sampler_z

    # Window marginals by brute force: sum the full-path masses consistent with each window path.
    full_index = np.arange(7) - 1  # -1 = no event, else event day
    slice_error = 0.0
    for start, stop in ((0, 3), (2, 5), (3, 6)):
        sliced = np.asarray(path_table(slice_time(d, slice(start, stop))))
        window_index = np.arange(stop - start + 1) - 1
        consistent = np.where(
            window_index[:, None] < 0,
            (full_index[None, :] < start) | (full_index[None, :] >= stop),
            full_index[None, :] == start + window_index[:, None],
        )
        brute = logsumexp(np.where(consistent[:, :, None], table[None], -np.inf), axis=1)
        slice_error = max(slice_error, log_error(sliced, brute))
    assert slice_error < 1e-5, slice_error

    # Prefix conditional by brute force: unit 0 fired on day 0, unit 1 pending, unit 2 fired day 2.
    t = 3
    data = jnp.zeros((t, 3), jnp.int32).at[0, 0].set(1).at[2, 2].set(1)
    conditional = np.asarray(path_table(prefix_condition(d, data)))
    prefix_mass = logsumexp(table[[0, 4, 5, 6]], axis=0)  # no event before day 3
    brute = np.full_like(conditional, -np.inf)
    brute[:, [0, 2]] = np.where(np.arange(4)[:, None] == 0, 0.0, -np.inf)
    brute[:, 1] = table[[0, 4, 5, 6], 1] - prefix_mass[1]
    prefix_error = log_error(conditional, brute)
    assert prefix_error < 1e-5, prefix_error
    # Chain rule the driver relies on: obs-prefix mass plus conditional suffix mass is the joint.
    prefix_only = np.asarray(path_table(slice_time(d, slice(None, t))))[0, 1]
    chain_error = log_error(prefix_only + conditional[:, 1], table[[0, 4, 5, 6], 1])
    assert chain_error < 1e-5, chain_error

    rejected = {
        "repeat_event": d.unit_log_prob(
            jnp.zeros((6, 3), jnp.int32).at[2, 0].set(1).at[4, 0].set(1)
        )[0],
        "closed_day_event": d.unit_log_prob(jnp.zeros((6, 3), jnp.int32).at[1, 0].set(1))[0],
        "unborn_day_event": d.unit_log_prob(jnp.zeros((6, 3), jnp.int32).at[0, 2].set(1))[2],
        "non_binary": d.unit_log_prob(jnp.zeros((6, 3), jnp.int32).at[2, 1].set(2))[1],
        "after_depletion": prefix_condition(d, data).unit_log_prob(
            jnp.zeros((3, 3), jnp.int32).at[1, 0].set(1)
        )[0],
        "joint_with_one_bad_unit": d.log_prob(
            jnp.zeros((6, 3), jnp.int32).at[1, 0].set(1).at[3, 1].set(1)
        ),
    }
    assert all(float(v) == -np.inf for v in rejected.values()), rejected
    accepted = float(d.log_prob(jnp.zeros((6, 3), jnp.int32).at[2, 0].set(1).at[3, 2].set(1)))
    assert np.isfinite(accepted)
    gradient = jax.grad(lambda x: SingleEventTrajectory(x).log_prob(data))(logits[:t])
    assert np.isfinite(np.asarray(gradient)).all()
    # Leading batch axes: [*batch, time, unit] parameters give [*batch] log_prob and [*, *batch, time, unit] draws.
    batched = SingleEventTrajectory(jnp.stack([logits, logits.at[3, 1].set(0.0)]))
    assert batched.batch_shape == (2,) and batched.event_shape == (6, 3)
    assert batched.sample(key_s, (5,)).shape == (5, 2, 6, 3)
    accepted_pair = batched.log_prob(jnp.zeros((6, 3), jnp.int32).at[2, 0].set(1).at[3, 2].set(1))
    assert (
        accepted_pair.shape == (2,)
        and float(accepted_pair[0]) == accepted
        and float(accepted_pair[1]) != accepted
    )
    assert prefix_condition(batched, data).logits.shape == (2, 3, 3)
    assert slice_time(batched, slice(2, 5)).logits.shape == (2, 3, 3)
    return {
        "normalization_error": normalization,
        "sampler_max_z": sampler_z,
        "sampler_draws": draws,
        "slice_marginal_error": slice_error,
        "prefix_conditional_error": prefix_error,
        "chain_rule_error": chain_error,
        "rejected": {k: float(v) for k, v in rejected.items()},
        "accepted_log_prob": accepted,
        "gradient_finite_with_structural_zeros": True,
        "batched_shapes": {"batch": [2], "event": [6, 3], "sample": [5, 2, 6, 3]},
    }


def parity(case, model):
    """Per-unit likelihood and gradient against the existing exact likelihood, several stops."""
    thetas = [TRUTH, TRUTH + 0.3, TRUTH - 0.5]
    thetas += [np.asarray(random.normal(k, (5,))) for k in random.split(random.PRNGKey(11), 3)]
    worst = {
        "unit_loglik": 0.0,
        "model_log_density": 0.0,
        "unit_gradient": 0.0,
        "model_gradient": 0.0,
    }
    for stop in (10, case.t_obs, len(case.calendar)):
        x, y = case.covariates[:stop], jnp.asarray(case.data[:stop])

        def native_units(t):
            return SingleEventTrajectory(initiation_logits(case, t, stop)).unit_log_prob(y)

        def native_model(t):
            return log_density(model, (x, y), {}, {"theta": t})[0]

        def baseline_units(t):
            return case.reference(t, stop)

        def baseline_model(t):
            return baseline_units(t).sum() + dist.Normal(0.0, 1.0).log_prob(t).sum()

        for value in thetas:
            theta = jnp.asarray(value, dtype=jnp.float32)
            errors = {
                "unit_loglik": np.max(np.abs(native_units(theta) - baseline_units(theta))),
                "model_log_density": np.abs(native_model(theta) - baseline_model(theta)),
                "unit_gradient": np.max(
                    np.abs(
                        jax.grad(lambda t: native_units(t).sum())(theta)
                        - jax.grad(lambda t: baseline_units(t).sum())(theta)
                    )
                ),
                "model_gradient": np.max(
                    np.abs(jax.grad(native_model)(theta) - jax.grad(baseline_model)(theta))
                ),
            }
            assert np.isfinite(jax.grad(native_model)(theta)).all()
            for name, error in errors.items():
                assert error < TOL, (name, stop, value, error)
                worst[name] = max(worst[name], float(error))
    worst.update(
        stops=[10, case.t_obs, len(case.calendar)], theta_values=len(thetas), tolerance=TOL
    )
    return worst


def forecast_law(case, model, y, law_draws):
    """Many fixed-theta native forecast draws against the closed-form future law."""
    theta = jnp.asarray(TRUTH)
    posterior = {"theta": jnp.broadcast_to(theta, (law_draws, 5))}
    fixed = np.asarray(forecast(random.PRNGKey(104), model, posterior, y, case.covariates))
    summary = check_paths(case, fixed)
    expected = case.analytic_future(theta)
    conditioned = prefix_condition(
        SingleEventTrajectory(initiation_logits(case, theta, len(case.calendar))), y
    )
    law_error = float(np.max(np.abs(np.asarray(conditioned.mean) - expected)))
    assert law_error < 1e-6, law_error
    frequency = fixed.mean(axis=0)
    assert np.all(frequency[expected == 0] == 0), "mass on a structurally impossible cell"
    cell_z = max_z(frequency, expected, law_draws)
    no_event_z = max_z(1 - frequency.sum(axis=0), 1 - expected.sum(axis=0), law_draws)
    assert cell_z < 5 and no_event_z < 5, (cell_z, no_event_z)
    summary.update(
        draws=law_draws,
        closed_form_law_error=law_error,
        max_cell_z=cell_z,
        max_unit_no_event_z=no_event_z,
        max_abs_pmf_error=float(np.max(np.abs(frequency - expected))),
    )
    return fixed, summary


def simulate_receipts(case, theta, seed=73):
    """Full-calendar receipt ledger driven by the actual full initiation ledger.

    The first two units initiated inside the prefix are pinned to the two historical
    states inference must handle: received on its own (open) initiation day, and still open.
    """
    full = case.data
    origin = jnp.asarray(np.where(full.any(axis=0), full.argmax(axis=0), full.shape[0]))
    q = np.asarray(jax.nn.sigmoid(receipt_logits(case, jnp.asarray(theta), origin, full.shape[0])))
    rng = np.random.default_rng(seed)
    received = np.zeros_like(full)
    pending = np.ones(full.shape[1], dtype=bool)
    for day in range(full.shape[0]):
        received[day] = pending & (rng.random(full.shape[1]) < q[day])
        pending &= ~received[day].astype(bool)
    done, still_open = np.flatnonzero(full[: case.t_obs].any(axis=0))[:2]
    received[:, [done, still_open]] = 0
    received[full[:, done].argmax(), done] = 1
    return received


def chain_expected_receipts(case, history, theta, theta_receipt):
    """Exact future receipt law by composing the parent event-time law with the child kernel.

    ``E[receipt on r] = sum_d P(initiation on d | history) * P(receipt on r | initiation on d, history)``,
    with ``d`` known for units initiated in the prefix. This is an expectation over exact
    per-unit paths, not an independent-Poisson approximation.
    """
    n = len(case.origin)
    duration, t = len(case.calendar), history.shape[0]
    init_hist, rec_hist = history[:, :n], history[:, n:]
    init_future = prefix_condition(
        SingleEventTrajectory(initiation_logits(case, theta, duration)), init_hist
    ).mean
    known = init_hist.any(axis=0)
    weight = jnp.zeros((duration, n)).at[t:].set(init_future)
    weight = jnp.where(
        known[None, :],
        jnp.arange(duration)[:, None] == jnp.argmax(init_hist, axis=0)[None, :],
        weight,
    )

    def kernel(day):
        origin = jnp.full((n,), day)
        return prefix_condition(
            SingleEventTrajectory(receipt_logits(case, theta_receipt, origin, duration)), rec_hist
        ).mean

    kernels = jax.vmap(kernel)(jnp.arange(duration))  # [origin day, future day, unit]
    return np.asarray(jnp.einsum("dn,dfn->fn", weight, kernels))


def chain_smoke(case, steps, draws, law_draws):
    """Two scoped stages through the native driver; receipts keep their own parent's day."""
    n, t = len(case.origin), case.t_obs
    received = simulate_receipts(case, RECEIPT_TRUTH)
    history = jnp.asarray(np.concatenate([case.data[:t], received[:t]], axis=-1))
    assert int(history[:, n:].sum()) > 0 and int(history[:, :n].sum()) > int(history[:, n:].sum())
    model = make_chain_model(case)
    guide = AutoNormal(model)
    svi = SVI(model, guide, numpyro.optim.Adam(0.025), Trace_ELBO())
    fitted = svi.run(random.PRNGKey(201), steps, case.covariates[:t], history, progress_bar=False)
    assert np.isfinite(fitted.losses).all()
    posterior = draw_posterior(random.PRNGKey(202), guide, fitted.params, draws)
    fitted_paths = np.asarray(
        forecast(random.PRNGKey(203), model, posterior, history, case.covariates)
    )
    assert fitted_paths.shape == (draws, len(case.calendar) - t, 2 * n)

    theta, theta_receipt = jnp.asarray(TRUTH), jnp.asarray(RECEIPT_TRUTH)
    fixed = {
        "initiation/theta": jnp.broadcast_to(theta, (law_draws, 5)),
        "receipt/theta": jnp.broadcast_to(theta_receipt, (law_draws, 5)),
    }
    sites = Predictive(
        model,
        posterior_samples=fixed,
        return_sites=["initiation/obs_future", "receipt/obs_future", "forecast"],
    )(random.PRNGKey(204), case.covariates, history)
    paths = np.asarray(sites["forecast"])
    np.testing.assert_array_equal(
        paths,
        np.concatenate(
            [np.asarray(sites["initiation/obs_future"]), np.asarray(sites["receipt/obs_future"])],
            axis=-1,
        ),
    )
    init, rec = paths[..., :n], paths[..., n:]
    check_paths(case, init)
    full_init = np.concatenate(
        [np.broadcast_to(np.asarray(history[:, :n]), (law_draws, t, n)), init], axis=1
    )
    full_rec = np.concatenate(
        [np.broadcast_to(np.asarray(history[:, n:]), (law_draws, t, n)), rec], axis=1
    )
    never = len(case.calendar)
    init_day = np.where(full_init.any(axis=1), full_init.argmax(axis=1), never)
    rec_day = np.where(full_rec.any(axis=1), full_rec.argmax(axis=1), never)
    assert np.all(full_rec.sum(axis=1) <= 1), "repeat receipt"
    assert np.all((rec_day == never) | (rec_day >= init_day)), "receipt before its own initiation"
    assert np.all((rec_day == never) | (init_day < never)), "receipt without initiation"
    assert not np.any(rec[:, ~case.allowed[:, t:].T]), "receipt on a closed day"
    expected = chain_expected_receipts(case, history, theta, theta_receipt)
    frequency = rec.mean(axis=0)
    assert np.all(frequency[expected == 0] == 0)
    receipt_z = max_z(frequency, expected, law_draws)
    assert receipt_z < 5, receipt_z
    future_then_receipt = int(np.sum((init_day >= t) & (init_day < never) & (rec_day < never)))
    open_history_receipt = int(np.sum((init_day < t) & (rec_day >= t) & (rec_day < never)))
    assert future_then_receipt > 0 and open_history_receipt > 0, (
        "fixture failed to exercise lineage"
    )
    return {
        "posterior_sites": sorted(posterior),
        "elbo_final": float(fitted.losses[-1]),
        "fitted_forecast_shape": list(fitted_paths.shape),
        "scoped_sites": sorted(sites),
        "historical_initiations": int(history[:, :n].sum()),
        "historical_receipts": int(history[:, n:].sum()),
        "law_draws": law_draws,
        "future_initiation_then_receipt_paths": future_then_receipt,
        "open_history_receipt_paths": open_history_receipt,
        "lineage_violations": 0,
        "receipt_law_max_z": receipt_z,
        "receipt_law_max_abs_error": float(np.max(np.abs(frequency - expected))),
    }


def run(steps=100, draws=128, law_draws=16384):
    case = make_case()
    model = make_model(case)
    y = jnp.asarray(case.data[: case.t_obs])
    probes = distribution_probes()
    parity_summary = parity(case, model)
    fitted = fit_and_forecast(model, case, steps, draws)
    fixed, law = forecast_law(case, model, y, law_draws)

    posterior = {"theta": jnp.broadcast_to(jnp.asarray(TRUTH), (32, 5))}
    in_sample = np.asarray(
        predict_in_sample(random.PRNGKey(106), model, posterior, case.covariates[: case.t_obs])
    )
    _, eligible = case.arrays(jnp.asarray(TRUTH))
    assert (
        in_sample.shape == (32, case.t_obs, len(case.origin)) and np.isin(in_sample, [0, 1]).all()
    )
    assert (
        in_sample.sum(axis=1).max() <= 1
        and not in_sample[:, ~np.asarray(eligible)[: case.t_obs]].any()
    )
    short = np.asarray(
        forecast(random.PRNGKey(105), model, posterior, y, case.covariates[: case.t_obs + 3])
    )
    assert short.shape == (32, 3, len(case.origin)) and short.sum(axis=1).max() <= 1
    assert not short[:, ~np.asarray(eligible)[case.t_obs : case.t_obs + 3]].any()
    # Flipping every hidden future outcome leaves the conditioned forecast bit-identical.
    hidden = replace(
        case, data=np.concatenate([case.data[: case.t_obs], 1 - case.data[case.t_obs :]])
    )
    replay = np.asarray(
        forecast(
            random.PRNGKey(104),
            make_model(hidden),
            {"theta": jnp.broadcast_to(jnp.asarray(TRUTH), (law_draws, 5))},
            y,
            case.covariates,
        )
    )
    np.testing.assert_array_equal(fixed, replay)

    return {
        "approach": "SingleEventTrajectory distribution + registered slice_time/prefix_condition, upstream predict/forecast unchanged",
        "distribution_probes": probes,
        "parity": parity_summary,
        "fit": fitted,
        "forecast_law": law,
        "predict_in_sample_shape": list(in_sample.shape),
        "short_horizon_shape": list(short.shape),
        "long_horizon_shape": list(fixed.shape),
        "hidden_outcome_leakage": False,
        "chain": chain_smoke(case, steps, draws, law_draws // 2),
        "extension_cost": {
            "upstream_edits": 0,
            "distribution": "one Distribution subclass: logits [*batch, time, unit], sample/log_prob/unit_log_prob/mean",
            "registrations": [
                "slice_time (prefix: parameter slice; interior window: absorbed mass folded into conditional hazard)",
                "prefix_condition (absorb fired units with -inf logits, keep the rest unchanged)",
            ],
            "chaining": "predict() returns None and only a top-level 'forecast' site is read by the driver, so a "
            "parent's sampled path is recovered with handlers.trace around the scoped predict() and the "
            "two scoped forecasts are re-registered as one combined 'forecast' deterministic",
            "dtype": "support declared boolean, so predict() requires integer-dtyped data",
        },
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
