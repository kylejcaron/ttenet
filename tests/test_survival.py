"""Analytical, boundary, and invariant tests for survival.py and models.py."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from numpyro import handlers
from numpyro.infer.util import log_density

from ttenet.dates import date_grid
from ttenet.event_times import TimingInputs, TimingLaw
from ttenet.models import (
    StageFit,
    StageObservations,
    fit_stage,
    make_event_observations,
    make_observations,
    predict_stage,
    stage_log_likelihood,
    stage_model,
)
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
    hazard: float, susceptibility_probability: float, num_bins: int = 3
) -> StageParameters:
    """StageParameters producing a flat hazard and a flat susceptibility probability."""
    logit = float(np.log(hazard / (1 - hazard)))
    susceptibility_logit = float(
        np.log(susceptibility_probability / (1 - susceptibility_probability))
    )
    return StageParameters(
        age_logits=jnp.full((num_bins,), logit),
        beta=jnp.zeros(0),
        susceptibility_intercept=jnp.array(susceptibility_logit),
        susceptibility_beta=jnp.zeros(0),
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
        susceptibility_features=jnp.zeros((n, 0)),
        at_risk=jnp.asarray(at_risk),
        allowed=jnp.asarray(allowed),
        event_index=jnp.asarray(event_index),
    )


# --- Analytical likelihood -------------------------------------------------


def test_stage_log_likelihood_matches_analytical_probabilities():
    # h=.5 constant, pi=.4. Event on second exposed day (index 1): .4*.5*.5=.1.
    # Censor after two exposed days: .6+.4*.25=.7.
    params = _constant_hazard_params(hazard=0.5, susceptibility_probability=0.4)
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
    params = _constant_hazard_params(hazard=0.5, susceptibility_probability=0.4)
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
    params = _constant_hazard_params(hazard=0.5, susceptibility_probability=0.4)
    with_closure = _observations(ages=[[0, 1, 2]], event_index=[-1], allowed=[[True, False, True]])
    without_closure = _observations(ages=[[0, 2]], event_index=[-1], allowed=[[True, True]])
    ll_with = stage_log_likelihood(params, with_closure)
    ll_without = stage_log_likelihood(params, without_closure)
    np.testing.assert_allclose(np.array(ll_with), np.array(ll_without), rtol=1e-6)


def test_same_day_event_uses_age_zero_hazard():
    # An event on the origin day itself (age 0) has probability pi*h0.
    params = _constant_hazard_params(hazard=0.3, susceptibility_probability=0.7)
    observations = _observations(ages=[[0, 1, 2]], event_index=[0])
    log_likelihood = stage_log_likelihood(params, observations)
    np.testing.assert_allclose(np.exp(np.array(log_likelihood)), [0.7 * 0.3], rtol=1e-6)


def test_negative_ages_have_zero_hazard_and_do_not_accumulate_survival():
    params = _constant_hazard_params(
        hazard=0.9, susceptibility_probability=0.5
    )  # aggressive hazard
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
        susceptibility_intercept=jnp.array(0.0),
        susceptibility_beta=jnp.zeros(0),
    )
    ages = jnp.array([[0, 1, 2, 5, 1000]])
    hazard = stage_hazard(params, ages, jnp.zeros((1, 5, 0)))
    expected_tail = jax.nn.sigmoid(age_logits[-1])
    np.testing.assert_allclose(float(hazard[0, 2]), float(expected_tail), rtol=1e-6)
    np.testing.assert_allclose(float(hazard[0, 3]), float(expected_tail), rtol=1e-6)
    np.testing.assert_allclose(float(hazard[0, 4]), float(expected_tail), rtol=1e-6)


def test_susceptibility_uses_static_regressors():
    params = StageParameters(
        age_logits=jnp.zeros(2),
        beta=jnp.zeros(0),
        susceptibility_intercept=jnp.array(0.0),
        susceptibility_beta=jnp.array([1.5, -0.5]),
    )
    susceptibility_features = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    result = susceptibility(params, susceptibility_features)
    expected = jax.nn.sigmoid(jnp.array([1.5, -0.5, 0.0]))
    np.testing.assert_allclose(np.array(result), np.array(expected), rtol=1e-6)


# --- Gradients ---------------------------------------------------------------


def test_gradients_are_finite_for_ordinary_valid_inputs():
    rng = np.random.default_rng(0)
    n, t, p, q = 6, 8, 2, 1
    ages = jnp.array(np.tile(np.arange(t), (n, 1)) - rng.integers(0, 3, size=(n, 1)))
    features = jnp.array(rng.normal(size=(n, t, p)).astype(np.float32))
    susceptibility_features = jnp.array(rng.normal(size=(n, q)).astype(np.float32))
    event_index = jnp.array([3, -1, 5, -1, 0, 7])
    observations = StageObservations(
        ages=ages,
        features=features,
        susceptibility_features=susceptibility_features,
        at_risk=jnp.ones((n, t), dtype=bool),
        allowed=jnp.ones((n, t), dtype=bool),
        event_index=event_index,
    )

    def total_log_likelihood(age_logits, beta, susceptibility_intercept, susceptibility_beta):
        params = StageParameters(age_logits, beta, susceptibility_intercept, susceptibility_beta)
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
    susceptibility_features = np.array([[np.log(2 / 3)], [99.0], [np.log(4)]])
    allowed = np.array([[True, True, True], [False, False, False], [False, False, True]])
    observations = make_observations(
        history,
        "receipt",
        calendar,
        features=features,
        susceptibility_features=susceptibility_features,
        allowed=allowed,
    )
    parameters = StageParameters(jnp.zeros(1), jnp.ones(1), 0.0, jnp.ones(1))
    # a: same-day receipt, .4*.5; c: one exposed day with h=.75 and pi=.8.
    np.testing.assert_allclose(
        np.exp(stage_log_likelihood(parameters, observations)), [0.2, 0.4], rtol=1e-6
    )


def test_susceptibility_regression_rejects_an_ambiguous_one_dimensional_vector():
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
            susceptibility_features=np.array([1.0, 2.0]),
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


# --- Conditional-entry likelihood (pre_entry) -------------------------------


def test_pre_entry_conditions_susceptibility_logit_on_known_prior_survival():
    # h=.5 constant, pi=.4. Two known event-free pre-entry days (S_pre=.25),
    # then censored through two more post-entry at-risk days (S_post=.25).
    # The conditional-entry likelihood must equal the exact Bayes identity
    # for "survive all 4 days given already known to survive the first 2":
    # ((1-pi) + pi*S_pre*S_post) / ((1-pi) + pi*S_pre) = 0.625 / 0.7.
    params = _constant_hazard_params(hazard=0.5, susceptibility_probability=0.4)
    observations = StageObservations(
        ages=jnp.array([[0, 1, 2, 3]]),
        features=jnp.zeros((1, 4, 0)),
        susceptibility_features=jnp.zeros((1, 0)),
        at_risk=jnp.array([[False, False, True, True]]),
        allowed=jnp.ones((1, 4), dtype=bool),
        event_index=jnp.array([-1]),
        pre_entry=jnp.array([[True, True, False, False]]),
    )
    log_likelihood = stage_log_likelihood(params, observations)
    np.testing.assert_allclose(np.exp(np.array(log_likelihood)), [0.625 / 0.7], rtol=1e-6)


# --- make_event_observations (generic conditional-entry stage clock) --------


def test_make_observations_none_policy_days_is_an_unbounded_deadline():
    # policy_days=None must behave as an unbounded initiation deadline: an
    # uninitiated row stays at risk straight through as_of regardless of age.
    history = _history(
        as_of="2026-06-01",
        policy_days=None,
        item_id=["a"],
        sale_date=pd.to_datetime(["2026-01-01"]),
        initiation_date=pd.to_datetime([pd.NaT]),
        receipt_date=pd.to_datetime([pd.NaT]),
    )
    calendar = date_grid("2026-01-01", "2026-06-01")
    observations = make_observations(history, "initiation", calendar)
    at_risk = np.array(observations.at_risk[0])
    as_of_index = int(
        (np.datetime64("2026-06-01") - np.datetime64("2026-01-01")) / np.timedelta64(1, "D")
    )
    assert at_risk[as_of_index]
    assert int(at_risk.sum()) == as_of_index + 1


def test_make_event_observations_entry_splits_pre_entry_and_at_risk():
    # A row entering a snapshot on day 3 keeps its origin-based age clock,
    # but at_risk only starts at entry; pre_entry covers the known
    # event-free run from origin through the day before entry.
    frame = pd.DataFrame(
        {"origin": pd.to_datetime(["2026-01-01"]), "own_event": pd.to_datetime([pd.NaT])}
    )
    calendar = date_grid("2026-01-01", "2026-01-10")
    observations = make_event_observations(
        frame,
        origin_column="origin",
        event_column="own_event",
        as_of=np.datetime64("2026-01-10", "D"),
        calendar=calendar,
        entry_dates=pd.to_datetime(["2026-01-04"]),
    )
    pre_entry = np.array(observations.pre_entry[0])
    at_risk = np.array(observations.at_risk[0])
    # origin=day0, entry=day3: pre-entry covers days 0,1,2; at_risk starts day3.
    np.testing.assert_array_equal(pre_entry[:3], [True, True, True])
    assert not np.any(pre_entry[3:])
    assert not np.any(at_risk[:3])
    assert at_risk[3]


def test_make_event_observations_event_before_entry_excludes_row():
    # A row whose own event already happened before its snapshot entry
    # contributes no new likelihood: it must be dropped entirely, not
    # zeroed out or forced into a degenerate probability.
    frame = pd.DataFrame(
        {
            "origin": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "own_event": pd.to_datetime(["2026-01-02", pd.NaT]),
        }
    )
    calendar = date_grid("2026-01-01", "2026-01-10")
    observations = make_event_observations(
        frame,
        origin_column="origin",
        event_column="own_event",
        as_of=np.datetime64("2026-01-10", "D"),
        calendar=calendar,
        entry_dates=pd.to_datetime(["2026-01-05", "2026-01-05"]),
    )
    # Row 0's event (day 1) precedes its entry (day 4): excluded. Only row 1 remains.
    assert observations.ages.shape[0] == 1


def test_make_event_observations_event_without_parent_raises():
    frame = pd.DataFrame(
        {"origin": pd.to_datetime([pd.NaT]), "own_event": pd.to_datetime(["2026-01-02"])}
    )
    with pytest.raises(ValueError):
        make_event_observations(
            frame,
            origin_column="origin",
            event_column="own_event",
            as_of=np.datetime64("2026-01-10", "D"),
            calendar=date_grid("2026-01-01", "2026-01-10"),
        )


def test_make_event_observations_event_exceeds_deadline_raises():
    frame = pd.DataFrame(
        {
            "origin": pd.to_datetime(["2026-01-01"]),
            "own_event": pd.to_datetime(["2026-01-15"]),  # 14 days later
        }
    )
    with pytest.raises(ValueError, match="deadline"):
        make_event_observations(
            frame,
            origin_column="origin",
            event_column="own_event",
            as_of=np.datetime64("2026-01-20", "D"),
            calendar=date_grid("2026-01-01", "2026-01-20"),
            deadline_days=10,
        )


def test_entry_before_parent_starts_exposure_at_parent_event():
    frame = pd.DataFrame(
        {
            "origin": pd.to_datetime(["2026-01-05"]),
            "own_event": pd.to_datetime(["2026-01-06"]),
        }
    )
    observations = make_event_observations(
        frame,
        origin_column="origin",
        event_column="own_event",
        as_of="2026-01-10",
        calendar=date_grid("2026-01-01", "2026-01-10"),
        entry_dates=["2026-01-01"],
    )
    parameters = _constant_hazard_params(hazard=0.5, susceptibility_probability=0.4)
    # Parent day zero survives, then the event happens at age one: .4*.5*.5.
    np.testing.assert_allclose(
        np.exp(stage_log_likelihood(parameters, observations)), [0.1], rtol=1e-6
    )


# --- Native observation: log joint, exposure and posterior predictive ---------


_PRIOR_SITES = (
    "age_scale",
    "age_init",
    "age_steps",
    "beta",
    "susceptibility_intercept",
    "susceptibility_beta",
)


def _constant_sites(hazard, susceptibility_probability):
    """Default-model sample-site values giving one flat hazard bin and a flat susceptibility."""
    return {
        "age_scale": jnp.array(1.0),
        "age_init": jnp.array(float(np.log(hazard / (1 - hazard)))),
        "age_steps": jnp.zeros(0),
        "beta": jnp.zeros(0),
        "susceptibility_intercept": jnp.array(
            float(np.log(susceptibility_probability / (1 - susceptibility_probability)))
        ),
        "susceptibility_beta": jnp.zeros(0),
    }


def _observation_log_density(sites, observations):
    """Log density of the native observation site alone, as a function of the prior sites."""
    model = handlers.block(
        handlers.substitute(stage_model, data=sites), hide=list(_PRIOR_SITES) + ["age_logits"]
    )
    return log_density(model, (observations,), {"age_bins": 1}, {})[0]


def test_native_log_joint_matches_analytic_probabilities_and_gradients():
    # h=.5, pi=.4: event on the second exposed day is .4*.5*.5=.1; censoring
    # after two exposed days is .6+.4*.25=.7. The observation site of the
    # NumPyro model carries exactly that likelihood, and its gradients are the
    # closed forms d/d(age_init) = (1-2h) + .4*2(1-h)(-h(1-h))/.7 = -1/7 and
    # d/d(susceptibility_intercept) = (1-pi) + (-pi(1-pi) + pi(1-pi)*.25)/.7 = 12/35.
    observations = _observations(ages=[[0, 1], [0, 1]], event_index=[1, -1])
    sites = _constant_sites(0.5, 0.4)
    value, gradients = jax.value_and_grad(_observation_log_density)(sites, observations)
    np.testing.assert_allclose(float(value), np.log(0.1) + np.log(0.7), rtol=1e-6)
    np.testing.assert_allclose(float(gradients["age_init"]), -1 / 7, rtol=1e-5)
    np.testing.assert_allclose(float(gradients["susceptibility_intercept"]), 12 / 35, rtol=1e-5)
    assert all(np.isfinite(np.asarray(value)).all() for value in gradients.values())


def test_native_log_joint_conditions_delayed_entry_like_the_public_helper():
    # Two known event-free pre-entry days, then two exposed censored days:
    # ((1-pi) + pi*S_pre*S_post) / ((1-pi) + pi*S_pre) = .625/.7. The gradient
    # is checked against central differences of that float64 identity.
    observations = StageObservations(
        ages=jnp.array([[0, 1, 2, 3]]),
        features=jnp.zeros((1, 4, 0)),
        susceptibility_features=jnp.zeros((1, 0)),
        at_risk=jnp.array([[False, False, True, True]]),
        allowed=jnp.ones((1, 4), dtype=bool),
        event_index=jnp.array([-1]),
        pre_entry=jnp.array([[True, True, False, False]]),
    )

    def oracle(age_init, susceptibility_intercept):
        hazard = 1 / (1 + np.exp(-age_init))
        pi = 1 / (1 + np.exp(-susceptibility_intercept))
        survive = (1 - hazard) ** 2
        return np.log(((1 - pi) + pi * survive * survive) / ((1 - pi) + pi * survive))

    sites = _constant_sites(0.5, 0.4)
    value, gradients = jax.value_and_grad(_observation_log_density)(sites, observations)
    np.testing.assert_allclose(float(value), np.log(0.625 / 0.7), rtol=1e-6)
    step = 1e-6
    theta, susceptibility_logit = float(sites["age_init"]), float(sites["susceptibility_intercept"])
    expected_theta = (
        oracle(theta + step, susceptibility_logit) - oracle(theta - step, susceptibility_logit)
    ) / (2 * step)
    expected_susceptibility = (
        oracle(theta, susceptibility_logit + step) - oracle(theta, susceptibility_logit - step)
    ) / (2 * step)
    np.testing.assert_allclose(float(gradients["age_init"]), expected_theta, rtol=1e-4)
    np.testing.assert_allclose(
        float(gradients["susceptibility_intercept"]), expected_susceptibility, rtol=1e-4
    )


def test_builder_exposure_is_administrative_and_stops_at_as_of():
    # Exposure runs from the clock origin through min(deadline, as_of) even for
    # a unit whose event is observed earlier, while at_risk still stops at the
    # event; a calendar extending past as_of never adds exposure.
    history = _history(
        as_of="2026-01-10",
        policy_days=5,
        item_id=["event", "expired", "late"],
        sale_date=pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-08"]),
        initiation_date=pd.to_datetime(["2026-01-03", pd.NaT, pd.NaT]),
        receipt_date=pd.to_datetime([pd.NaT, pd.NaT, pd.NaT]),
    )
    calendar = date_grid("2026-01-01", "2026-02-01")
    observations = make_observations(history, "initiation", calendar)
    exposure = np.array(observations.exposure)
    at_risk = np.array(observations.at_risk)
    ages = np.array(observations.ages)
    assert exposure.shape == at_risk.shape
    np.testing.assert_array_equal(np.flatnonzero(at_risk[0]), [0, 1, 2])
    np.testing.assert_array_equal(np.flatnonzero(exposure[0]), [0, 1, 2, 3, 4, 5])
    np.testing.assert_array_equal(np.flatnonzero(exposure[1]), [0, 1, 2, 3, 4, 5])
    np.testing.assert_array_equal(np.flatnonzero(exposure[2]), [7, 8, 9])
    assert not exposure[:, 10:].any()
    assert not exposure[ages < 0].any()


def test_pre_entry_and_exposure_never_overlap_or_precede_the_origin():
    frame = pd.DataFrame(
        {"origin": pd.to_datetime(["2026-01-03"]), "own_event": pd.to_datetime([pd.NaT])}
    )
    observations = make_event_observations(
        frame,
        origin_column="origin",
        event_column="own_event",
        as_of="2026-01-10",
        calendar=date_grid("2026-01-01", "2026-01-12"),
        deadline_days=4,
        entry_dates=["2026-01-05"],
    )
    pre_entry = np.array(observations.pre_entry[0])
    exposure = np.array(observations.exposure[0])
    np.testing.assert_array_equal(np.flatnonzero(pre_entry), [2, 3])
    np.testing.assert_array_equal(np.flatnonzero(exposure), [4, 5, 6])
    assert not (pre_entry & exposure).any()


def test_manual_observations_without_exposure_keep_their_event_window():
    # Legacy hand-built observations carry only at_risk. An event on a day
    # outside that window stays impossible, and censoring still sums survival
    # over exactly the at_risk days.
    params = _constant_hazard_params(hazard=0.5, susceptibility_probability=0.4)
    outside = _observations(ages=[[0, 1, 2]], event_index=[2], at_risk=[[True, True, False]])
    assert float(stage_log_likelihood(params, outside)[0]) == -np.inf
    censored = _observations(
        ages=[[0, 1, 2, 3]], event_index=[-1], at_risk=[[True, True, False, False]]
    )
    np.testing.assert_allclose(np.exp(stage_log_likelihood(params, censored)), [0.7], rtol=1e-6)


def _constant_fit(hazard, susceptibility_probability, draws):
    single = _constant_hazard_params(hazard, susceptibility_probability, num_bins=1)
    parameters = jax.tree_util.tree_map(
        lambda leaf: jnp.broadcast_to(leaf, (draws,) + leaf.shape), single
    )
    return StageFit(parameters, jnp.zeros(0), num_samples=draws)


def test_predict_stage_draws_the_fitted_law_not_the_stored_observations():
    # Two units exposed on four open days (h=.5, pi=.4); one closed day.
    # Posterior predictive trajectories must follow the law: the unit with an
    # observed day-1 event can fire on any exposed day, including after its
    # observed event, and the no-event share is .6+.4*.5**4=.625.
    draws = 4000
    observations = StageObservations(
        ages=jnp.array([[0, 1, 2, 3, 4]] * 2),
        features=jnp.zeros((2, 5, 0)),
        susceptibility_features=jnp.zeros((2, 0)),
        at_risk=jnp.array([[True, True, False, False, False], [True] * 5]),
        allowed=jnp.array([[True, True, True, False, True]] * 2),
        event_index=jnp.array([1, -1]),
        exposure=jnp.ones((2, 5), dtype=bool),
    )
    paths = np.asarray(predict_stage(_constant_fit(0.5, 0.4, draws), observations, seed=3))
    assert paths.shape == (draws, 5, 2)
    assert np.issubdtype(paths.dtype, np.integer)
    assert set(np.unique(paths)) <= {0, 1}
    assert (paths.sum(axis=1) <= 1).all()
    assert not paths[:, 3, :].any()
    observed = np.zeros((5, 2), dtype=int)
    observed[1, 0] = 1
    assert not np.array_equal(paths, np.broadcast_to(observed, paths.shape))
    assert paths[:, 2, 0].sum() > 0 and paths[:, 4, 0].sum() > 0
    no_event = 1 - paths.sum(axis=1)
    np.testing.assert_allclose(no_event.mean(axis=0), 0.625, atol=0.03)
    np.testing.assert_allclose(paths[:, 0].mean(axis=0), 0.2, atol=0.03)
    replay = np.asarray(predict_stage(_constant_fit(0.5, 0.4, draws), observations, seed=3))
    np.testing.assert_array_equal(paths, replay)


def test_fit_stage_default_family_keeps_stage_parameters_with_draw_axes():
    rng = np.random.default_rng(3)
    sales = pd.to_datetime("2026-01-01") + pd.to_timedelta(rng.integers(0, 10, 40), unit="D")
    delay = rng.geometric(0.3, 40) - 1
    initiated = rng.random(40) < 0.6
    events = pd.Series(sales + pd.to_timedelta(delay, unit="D")).where(initiated)
    history = _history(
        as_of="2026-01-25",
        policy_days=30,
        item_id=np.arange(40),
        sale_date=sales,
        initiation_date=events,
        receipt_date=pd.to_datetime([pd.NaT] * 40),
    )
    observations = make_observations(history, "initiation", date_grid("2026-01-01", "2026-01-25"))
    fit = fit_stage(observations, age_bins=4, num_steps=30, num_samples=7, seed=1)
    assert fit.family is None and fit.shared is None
    assert fit.num_samples == 7 and fit.draws == 7
    assert fit.parameters.age_logits.shape == (7, 4)
    assert fit.parameters.susceptibility_intercept.shape == (7,)
    assert fit.parameters.beta.shape == (7, 0) and fit.parameters.susceptibility_beta.shape == (
        7,
        0,
    )
    assert np.isfinite(fit.losses).all() and fit.losses.shape == (30,)
    inputs = TimingInputs(
        ages=jnp.array([[0, 5], [1, 6], [2, 7]]),
        features=jnp.zeros((3, 2, 0)),
        susceptibility_features=jnp.zeros((2, 0)),
    )
    timing, logits = fit.timing(inputs, draw=2)
    assert isinstance(timing, TimingLaw)
    assert timing.log_hazard.shape == (3, 2) and logits.shape == (2,)
    expected = jax.nn.log_sigmoid(fit.parameters.age_logits[2][jnp.minimum(inputs.ages, 3)])
    np.testing.assert_allclose(timing.log_hazard, expected, rtol=1e-6)
    # The legacy two-argument result still describes its draws from the arrays.
    legacy = StageFit(fit.parameters, fit.losses)
    assert legacy.draws == 7 and legacy.num_samples is None
