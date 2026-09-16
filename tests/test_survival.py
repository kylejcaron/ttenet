"""Analytical, boundary, and invariant tests for survival.py and models.py."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from ttenet.dates import date_grid
from ttenet.models import StageObservations, make_observations, stage_log_likelihood
from ttenet.survival import (
    StageParameters,
    conditional_susceptibility,
    stage_hazard,
    susceptibility,
)


@dataclasses.dataclass(frozen=True)
class _History:
    """Minimal RetailHistory-shaped stand-in (frame/as_of/policy_days)."""

    frame: pd.DataFrame
    as_of: np.datetime64
    policy_days: int


def _constant_hazard_params(
    hazard: float, cure_probability: float, num_bins: int = 3
) -> StageParameters:
    """StageParameters producing a flat hazard and a flat cure probability."""
    logit = float(np.log(hazard / (1 - hazard)))
    cure_logit = float(np.log(cure_probability / (1 - cure_probability)))
    return StageParameters(
        age_logits=jnp.full((num_bins,), logit),
        beta=jnp.zeros(0),
        cure_intercept=jnp.array(cure_logit),
        cure_beta=jnp.zeros(0),
    )


def _observations(
    ages, event_index, at_risk=None, allowed=None, num_features=0
) -> StageObservations:
    ages = jnp.asarray(ages)
    n, t = ages.shape
    if at_risk is None:
        at_risk = jnp.ones((n, t), dtype=bool)
    if allowed is None:
        allowed = jnp.ones((n, t), dtype=bool)
    return StageObservations(
        ages=ages,
        features=jnp.zeros((n, t, num_features)),
        cure_features=jnp.zeros((n, 0)),
        at_risk=jnp.asarray(at_risk),
        allowed=jnp.asarray(allowed),
        event_index=jnp.asarray(event_index),
    )


# --- Analytical likelihood -------------------------------------------------


def test_stage_log_likelihood_matches_analytical_probabilities():
    # h=.5 constant, pi=.4. Event on second exposed day (index 1): .4*.5*.5=.1.
    # Censor after two exposed days: .6+.4*.25=.7.
    params = _constant_hazard_params(hazard=0.5, cure_probability=0.4)
    observations = _observations(ages=[[0, 1], [0, 1]], event_index=[1, -1])
    log_likelihood = stage_log_likelihood(params, observations)
    np.testing.assert_allclose(np.exp(np.array(log_likelihood)), [0.1, 0.7], rtol=1e-6)


def test_conditional_susceptibility_matches_analytical_value():
    # Same two-exposed-day censoring case: posterior susceptibility is .1/.7 = 1/7.
    hazard = 0.5
    pi = 0.4
    log_survival = 2 * np.log(1 - hazard)
    result = conditional_susceptibility(jnp.array(pi), jnp.array(log_survival))
    np.testing.assert_allclose(float(result), 1 / 7, rtol=1e-6)


def test_conditional_susceptibility_boundary_invariants():
    pi = jnp.array([0.1, 0.4, 0.9])
    # No elapsed at-risk time (S=1, log_survival=0): posterior susceptibility is unchanged.
    np.testing.assert_allclose(np.array(conditional_susceptibility(pi, jnp.zeros(3))), np.array(pi))
    # A unit survived essentially forever without an event: overwhelming evidence of cure.
    long_survival = conditional_susceptibility(pi, jnp.full(3, -1e4))
    np.testing.assert_allclose(np.array(long_survival), np.zeros(3), atol=1e-6)


# --- Hard closures and hazard primitives -----------------------------------


def test_observed_event_on_closed_day_is_negative_infinite_not_clipped():
    params = _constant_hazard_params(hazard=0.5, cure_probability=0.4)
    observations = _observations(
        ages=[[0, 1]],
        event_index=[1],
        allowed=[[True, False]],  # the event day itself is a hard closure
    )
    log_likelihood = stage_log_likelihood(params, observations)
    assert log_likelihood[0] == -jnp.inf


def test_closed_day_does_not_count_as_a_cure_or_decay_survival():
    # A censored unit with one open day (h=.5) and one closed day in between
    # should get the same likelihood as if the closed day were simply absent:
    # survival only decays on the open day.
    params = _constant_hazard_params(hazard=0.5, cure_probability=0.4)
    with_closure = _observations(ages=[[0, 1, 2]], event_index=[-1], allowed=[[True, False, True]])
    without_closure = _observations(ages=[[0, 2]], event_index=[-1], allowed=[[True, True]])
    ll_with = stage_log_likelihood(params, with_closure)
    ll_without = stage_log_likelihood(params, without_closure)
    np.testing.assert_allclose(np.array(ll_with), np.array(ll_without), rtol=1e-6)


def test_same_day_event_uses_age_zero_hazard():
    # An event on the origin day itself (age 0) has probability pi*h0.
    params = _constant_hazard_params(hazard=0.3, cure_probability=0.7)
    observations = _observations(ages=[[0, 1, 2]], event_index=[0])
    log_likelihood = stage_log_likelihood(params, observations)
    np.testing.assert_allclose(np.exp(np.array(log_likelihood)), [0.7 * 0.3], rtol=1e-6)


def test_negative_ages_have_zero_hazard_and_do_not_accumulate_survival():
    params = _constant_hazard_params(hazard=0.9, cure_probability=0.5)  # aggressive hazard
    ages = jnp.array([[-3, -2, -1, 0, 1]])
    hazard = stage_hazard(params, ages, jnp.zeros((1, 5, 0)))
    np.testing.assert_array_equal(np.array(hazard[0, :3]), np.zeros(3))
    assert float(hazard[0, 3]) > 0 and float(hazard[0, 4]) > 0

    # A censored unit observed (at_risk) across pre-origin padding plus two
    # real at-risk days must match a unit with only the two real days: the
    # padding must not decay survival even if it is (incorrectly) marked
    # at_risk by a careless caller.
    padded = _observations(
        ages=[[-2, -1, 0, 1]], event_index=[-1], at_risk=[[True, True, True, True]]
    )
    unpadded = _observations(ages=[[0, 1]], event_index=[-1])
    ll_padded = stage_log_likelihood(params, padded)
    ll_unpadded = stage_log_likelihood(params, unpadded)
    np.testing.assert_allclose(np.array(ll_padded), np.array(ll_unpadded), rtol=1e-6)


def test_stage_hazard_clips_tail_age_to_final_baseline_bin():
    age_logits = jnp.array([-1.0, 0.0, 2.0])
    params = StageParameters(
        age_logits=age_logits,
        beta=jnp.zeros(0),
        cure_intercept=jnp.array(0.0),
        cure_beta=jnp.zeros(0),
    )
    ages = jnp.array([[0, 1, 2, 5, 1000]])
    hazard = stage_hazard(params, ages, jnp.zeros((1, 5, 0)))
    expected_tail = jax.nn.sigmoid(age_logits[-1])
    np.testing.assert_allclose(float(hazard[0, 2]), float(expected_tail), rtol=1e-6)
    np.testing.assert_allclose(float(hazard[0, 3]), float(expected_tail), rtol=1e-6)
    np.testing.assert_allclose(float(hazard[0, 4]), float(expected_tail), rtol=1e-6)


def test_susceptibility_uses_static_cure_regressors():
    params = StageParameters(
        age_logits=jnp.zeros(2),
        beta=jnp.zeros(0),
        cure_intercept=jnp.array(0.0),
        cure_beta=jnp.array([1.5, -0.5]),
    )
    cure_features = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    result = susceptibility(params, cure_features)
    expected = jax.nn.sigmoid(jnp.array([1.5, -0.5, 0.0]))
    np.testing.assert_allclose(np.array(result), np.array(expected), rtol=1e-6)


# --- Gradients ---------------------------------------------------------------


def test_gradients_are_finite_for_ordinary_valid_inputs():
    rng = np.random.default_rng(0)
    n, t, p, q = 6, 8, 2, 1
    ages = jnp.array(np.tile(np.arange(t), (n, 1)) - rng.integers(0, 3, size=(n, 1)))
    features = jnp.array(rng.normal(size=(n, t, p)).astype(np.float32))
    cure_features = jnp.array(rng.normal(size=(n, q)).astype(np.float32))
    event_index = jnp.array([3, -1, 5, -1, 0, 7])
    observations = StageObservations(
        ages=ages,
        features=features,
        cure_features=cure_features,
        at_risk=jnp.ones((n, t), dtype=bool),
        allowed=jnp.ones((n, t), dtype=bool),
        event_index=event_index,
    )

    def total_log_likelihood(age_logits, beta, cure_intercept, cure_beta):
        params = StageParameters(age_logits, beta, cure_intercept, cure_beta)
        return jnp.sum(stage_log_likelihood(params, observations))

    grads = jax.grad(total_log_likelihood, argnums=(0, 1, 2, 3))(
        jnp.zeros(4), jnp.zeros(p), jnp.array(0.1), jnp.zeros(q)
    )
    for grad in grads:
        assert bool(jnp.all(jnp.isfinite(grad)))


@pytest.mark.parametrize("logit", [40.0, -100.0])
def test_extreme_finite_logits_preserve_log_likelihood_and_gradients(logit):
    observations = _observations(ages=[[0, 1], [0, 1]], event_index=[1, -1])
    params = StageParameters(jnp.array([logit]), jnp.empty(0), jnp.array(logit), jnp.empty(0))
    log_h = -np.logaddexp(0.0, -logit)
    log_s = -np.logaddexp(0.0, logit)
    expected = [2 * log_h + log_s, np.logaddexp(log_s, log_h + 2 * log_s)]
    actual = stage_log_likelihood(params, observations)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    gradients = jax.grad(lambda value: stage_log_likelihood(value, observations).sum())(params)
    assert all(np.isfinite(value).all() for value in gradients)


def test_closed_day_event_gradient_has_no_nan_leakage():
    # The -inf branch must not poison gradients with NaN even though it is
    # reached: a hard-zero hazard day is a real, differentiable constant
    # zero as a function of parameters, not an undefined 0/0 or log(-x).
    observations = _observations(ages=[[0, 1]], event_index=[1], allowed=[[True, False]])

    def loss(age_logits):
        params = StageParameters(age_logits, jnp.zeros(0), jnp.array(0.0), jnp.zeros(0))
        return jnp.sum(stage_log_likelihood(params, observations))

    grad = jax.grad(loss)(jnp.zeros(3))
    assert not bool(jnp.any(jnp.isnan(grad)))


# --- make_observations --------------------------------------------------


def _history(as_of="2026-02-01", policy_days=90, **columns) -> _History:
    frame = pd.DataFrame(columns)
    return _History(frame=frame, as_of=np.datetime64(as_of, "D"), policy_days=policy_days)


def test_make_observations_initiation_ages_and_event_index():
    history = _history(
        item_id=["a", "b"],
        sale_date=pd.to_datetime(["2026-01-01", "2026-01-05"]),
        initiation_date=pd.to_datetime(["2026-01-03", pd.NaT]),
        receipt_date=pd.to_datetime([pd.NaT, pd.NaT]),
    )
    calendar = date_grid("2026-01-01", "2026-02-01")
    observations = make_observations(history, "initiation", calendar)

    assert observations.ages.shape == (2, len(calendar))
    np.testing.assert_array_equal(np.array(observations.ages[0, :5]), [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(np.array(observations.ages[1, :5]), [-4, -3, -2, -1, 0])
    np.testing.assert_array_equal(np.array(observations.event_index), [2, -1])
    # Item b never initiates; at-risk continues through as_of (policy deadline is far later).
    assert int(observations.at_risk[1].sum()) == 28


def test_make_observations_policy_day_90_inclusive_91_exclusive():
    # An uninitiated sale on day 0: at risk through age 90 inclusive, then
    # censored (out of the initiation window) starting age 91.
    as_of = "2026-06-01"
    history = _history(
        as_of=as_of,
        policy_days=90,
        item_id=["a"],
        sale_date=pd.to_datetime(["2026-01-01"]),
        initiation_date=pd.to_datetime([pd.NaT]),
        receipt_date=pd.to_datetime([pd.NaT]),
    )
    calendar = date_grid("2026-01-01", as_of)
    observations = make_observations(history, "initiation", calendar)
    at_risk = np.array(observations.at_risk[0])
    assert at_risk[90]
    assert not at_risk[91]
    assert int(at_risk.sum()) == 91  # ages 0..90 inclusive


def test_make_observations_receipt_filters_to_initiated_rows_and_resets_age():
    history = _history(
        item_id=["a", "b", "c"],
        sale_date=pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-01"]),
        initiation_date=pd.to_datetime(["2026-01-05", pd.NaT, "2026-01-10"]),
        receipt_date=pd.to_datetime(["2026-01-08", pd.NaT, pd.NaT]),
    )
    calendar = date_grid("2026-01-01", "2026-02-01")
    observations = make_observations(history, "receipt", calendar)

    # Only rows a and c (initiated); b is excluded from the receipt stage entirely.
    assert observations.ages.shape[0] == 2
    # Age resets at initiation, not at sale: item a's age is 0 on 2026-01-05.
    np.testing.assert_array_equal(np.array(observations.ages[0, :6]), [-4, -3, -2, -1, 0, 1])
    np.testing.assert_array_equal(np.array(observations.event_index), [7, -1])


def test_receipt_likelihood_keeps_each_items_regressors_and_closures():
    history = _history(
        as_of="2026-01-03",
        item_id=["a", "b", "c"],
        sale_date=pd.to_datetime(["2026-01-01"] * 3),
        initiation_date=pd.to_datetime(["2026-01-02", pd.NaT, "2026-01-02"]),
        receipt_date=pd.to_datetime(["2026-01-02", pd.NaT, pd.NaT]),
    )
    calendar = date_grid("2026-01-01", "2026-01-03")
    features = np.broadcast_to(np.array([0.0, 99.0, np.log(3)])[:, None, None], (3, 3, 1))
    cure_features = np.array([[np.log(2 / 3)], [99.0], [np.log(4)]])
    allowed = np.array([[True, True, True], [False, False, False], [False, False, True]])
    observations = make_observations(
        history,
        "receipt",
        calendar,
        features=features,
        cure_features=cure_features,
        allowed=allowed,
    )
    parameters = StageParameters(jnp.zeros(1), jnp.ones(1), 0.0, jnp.ones(1))
    # a: same-day receipt, .4*.5; c: one exposed day with h=.75 and pi=.8.
    np.testing.assert_allclose(
        np.exp(stage_log_likelihood(parameters, observations)), [0.2, 0.4], rtol=1e-6
    )


def test_cure_regression_rejects_an_ambiguous_one_dimensional_vector():
    history = _history(
        item_id=["a", "b"],
        sale_date=pd.to_datetime(["2026-01-01"] * 2),
        initiation_date=pd.to_datetime([pd.NaT, pd.NaT]),
        receipt_date=pd.to_datetime([pd.NaT, pd.NaT]),
    )
    with pytest.raises(ValueError):
        make_observations(
            history,
            "initiation",
            date_grid("2026-01-01", "2026-02-01"),
            cure_features=np.array([1.0, 2.0]),
        )


def test_make_observations_calendar_must_cover_earliest_origin():
    history = _history(
        item_id=["a"],
        sale_date=pd.to_datetime(["2026-01-01"]),
        initiation_date=pd.to_datetime([pd.NaT]),
        receipt_date=pd.to_datetime([pd.NaT]),
    )
    late_calendar = date_grid("2026-01-02", "2026-02-01")
    with pytest.raises(ValueError):
        make_observations(history, "initiation", late_calendar)


def test_make_observations_calendar_must_extend_through_as_of():
    history = _history(
        item_id=["a"],
        sale_date=pd.to_datetime(["2026-01-01"]),
        initiation_date=pd.to_datetime([pd.NaT]),
        receipt_date=pd.to_datetime([pd.NaT]),
    )
    short_calendar = date_grid("2026-01-01", "2026-01-15")
    with pytest.raises(ValueError):
        make_observations(history, "initiation", short_calendar)


def test_make_observations_future_calendar_days_are_not_at_risk():
    # A calendar extending well past as_of (e.g. shared with a forecast
    # horizon) must not leak future days into the observation window.
    history = _history(
        as_of="2026-01-10",
        item_id=["a"],
        sale_date=pd.to_datetime(["2026-01-01"]),
        initiation_date=pd.to_datetime([pd.NaT]),
        receipt_date=pd.to_datetime([pd.NaT]),
    )
    calendar = date_grid("2026-01-01", "2026-03-01")  # far beyond as_of
    observations = make_observations(history, "initiation", calendar)
    at_risk = np.array(observations.at_risk[0])
    as_of_index = 9  # 2026-01-10 is the 10th day (index 9)
    assert at_risk[as_of_index]
    assert not np.any(at_risk[as_of_index + 1 :])
