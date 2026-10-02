"""Analytical, boundary, and invariant tests for event_times.py.

Layout everywhere is time-major: ``[time, cohort]`` cells, ``[cohort]``
per-unit quantities. Oracles are independent float64 NumPy computations in
probability space; the module under test works in log space.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ttenet.event_times import (
    EventLaw,
    SurvivalKernel,
    TimingInputs,
    TimingLaw,
    default_timing,
    exclusive_log_survival,
    kernel_unit_log_prob,
    log1mexp,
    survival_kernel,
    timing_from_log_masses,
    timing_from_log_survival,
)
from ttenet.survival import StageParameters, stage_hazard, susceptibility


def _logit(probability: float) -> float:
    return float(np.log(probability / (1 - probability)))


def _probabilities(log_values) -> np.ndarray:
    return np.exp(np.array(log_values, dtype=np.float64))


def _constant_hazard_params(
    hazard: float, susceptibility_probability: float, num_bins: int = 3
) -> StageParameters:
    """StageParameters producing a flat hazard and a flat susceptibility probability."""
    return StageParameters(
        age_logits=jnp.full((num_bins,), _logit(hazard)),
        beta=jnp.zeros(0),
        susceptibility_intercept=jnp.array(_logit(susceptibility_probability)),
        susceptibility_beta=jnp.zeros(0),
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


def _full_masks(days: int, cohorts: int):
    return jnp.ones((days, cohorts), dtype=bool)


def _oracle_masses(pi, hazard, *, allowed, exposure, pre_entry=None):
    """Probability-space mixture-cure law: ``[T, C]`` date masses and ``[C]`` residual.

    ``hazard`` is the family's per-cell event probability; closed or unexposed
    cells contribute zero hazard. A known event-free pre-entry run conditions
    the susceptibility by the exact Bayes identity before the exposed window.
    """
    hazard = np.asarray(hazard, dtype=np.float64)
    allowed = np.broadcast_to(np.asarray(allowed, dtype=bool), hazard.shape)
    exposure = np.broadcast_to(np.asarray(exposure, dtype=bool), hazard.shape)
    pi = np.broadcast_to(np.asarray(pi, dtype=np.float64), hazard.shape[-1:])
    if pre_entry is not None:
        known = allowed & np.broadcast_to(np.asarray(pre_entry, dtype=bool), hazard.shape)
        survived = np.prod(np.where(known, 1 - hazard, 1.0), axis=0)
        pi = pi * survived / ((1 - pi) + pi * survived)
    active = np.where(allowed & exposure, hazard, 0.0)
    stay = 1 - active
    before = np.concatenate([np.ones((1,) + stay.shape[1:]), np.cumprod(stay, axis=0)[:-1]])
    return pi * before * active, (1 - pi) + pi * np.prod(stay, axis=0)


def _unit_log_probs(kernel: SurvivalKernel, days: int, cohorts: int) -> np.ndarray:
    """Compact unit log probabilities for every date and for no event, ``[T + 1, C]``."""
    rows = [kernel_unit_log_prob(kernel, jnp.full((cohorts,), day)) for day in range(days)]
    rows.append(kernel_unit_log_prob(kernel, jnp.full((cohorts,), -1)))
    return np.array(jnp.stack(rows), dtype=np.float64)


# --- Analytical likelihood -------------------------------------------------


def test_unit_log_prob_matches_analytical_event_and_censor_probabilities():
    # h=.5 constant, pi=.4. Event on second exposed day: .4*.5*.5=.1.
    # Censor after two exposed days: .6+.4*.25=.7.
    params = _constant_hazard_params(hazard=0.5, susceptibility_probability=0.4)
    timing, logits = default_timing(params, _inputs([[0, 0], [1, 1]]))
    masks = _full_masks(2, 2)
    kernel = survival_kernel(timing, logits, allowed=masks, exposure=masks)
    log_prob = kernel_unit_log_prob(kernel, jnp.array([1, -1]))
    np.testing.assert_allclose(_probabilities(log_prob), [0.1, 0.7], rtol=1e-6)
    np.testing.assert_allclose(_probabilities(kernel.log_mass[:, 0]), [0.2, 0.1], rtol=1e-6)
    np.testing.assert_allclose(_probabilities(kernel.log_tail), [0.7, 0.7], rtol=1e-6)


def test_pre_entry_conditions_susceptibility_on_known_survival():
    # h=.5, pi=.4. Two known event-free pre-entry days (S_pre=.25), then two
    # exposed days. Censored: ((1-pi)+pi*S_pre*S_post)/((1-pi)+pi*S_pre)=.625/.7.
    # Event on the first exposed day: conditioned pi is .1/.7, times h: 1/14.
    params = _constant_hazard_params(hazard=0.5, susceptibility_probability=0.4)
    timing, logits = default_timing(params, _inputs([[0, 0], [1, 1], [2, 2], [3, 3]]))
    pre_entry = jnp.array([[True], [True], [False], [False]])
    kernel = survival_kernel(
        timing, logits, allowed=_full_masks(4, 2), exposure=~pre_entry, pre_entry=pre_entry
    )
    log_prob = kernel_unit_log_prob(kernel, jnp.array([-1, 2]))
    np.testing.assert_allclose(_probabilities(log_prob), [0.625 / 0.7, 1 / 14], rtol=1e-6)
    np.testing.assert_allclose(_probabilities(kernel.log_susceptible), [1 / 7, 1 / 7], rtol=1e-6)
    assert bool(jnp.all(jnp.isneginf(kernel.log_mass[:2])))


def test_calendar_hazard_variation_and_closures_match_probability_oracle():
    # Age-varying baseline, a storm regressor on day 2 for cohort 0, cohort 1
    # starting one day later, and a calendar closure on day 1 for everyone.
    age_logits = np.array([-1.0, 0.0, 0.5, 1.0])
    beta = np.array([1.2])
    susceptibility_intercept, susceptibility_beta = 0.3, np.array([0.7])
    params = StageParameters(
        jnp.array(age_logits),
        jnp.array(beta),
        jnp.array(susceptibility_intercept),
        jnp.array(susceptibility_beta),
    )
    ages = np.array([[0, -1], [1, 0], [2, 1], [3, 2], [4, 3]])
    storm = np.zeros((5, 2, 1))
    storm[2, 0, 0] = 1.0
    susceptibility_features = np.array([[1.0], [-0.5]])
    allowed = np.array([[True, True], [False, False], [True, True], [True, True], [True, True]])

    def kernel_for(ages, features, allowed):
        timing, logits = default_timing(params, _inputs(ages, features, susceptibility_features))
        return survival_kernel(
            timing, logits, allowed=jnp.asarray(allowed), exposure=jnp.asarray(ages >= 0)
        )

    def oracle_for(ages, features, allowed):
        hazard = 1 / (1 + np.exp(-(age_logits[np.clip(ages, 0, 3)] + features @ beta)))
        pi = 1 / (
            1 + np.exp(-(susceptibility_intercept + susceptibility_features @ susceptibility_beta))
        )
        return _oracle_masses(pi, hazard, allowed=allowed, exposure=ages >= 0)

    kernel = kernel_for(ages, storm, allowed)
    mass, tail = oracle_for(ages, storm, allowed)
    np.testing.assert_allclose(_probabilities(kernel.log_mass), mass, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(_probabilities(kernel.log_tail), tail, rtol=1e-5)
    np.testing.assert_allclose(mass.sum(axis=0) + tail, 1.0, rtol=1e-12)
    with np.errstate(divide="ignore"):
        expected = np.log(np.concatenate([mass, tail[None]]))
    np.testing.assert_allclose(_unit_log_probs(kernel, 5, 2), expected, rtol=1e-5)
    # Closed and unexposed dates carry exactly zero event mass.
    assert np.all(np.isneginf(np.array(kernel.log_mass[1])))
    assert np.isneginf(float(kernel.log_mass[0, 1]))
    # The storm moves mass onto its date and out of the residual for the
    # affected cohort only.
    calm = kernel_for(ages, np.zeros_like(storm), allowed)
    assert float(kernel.log_mass[2, 0]) > float(calm.log_mass[2, 0])
    assert float(kernel.log_tail[0]) < float(calm.log_tail[0])
    np.testing.assert_allclose(np.array(kernel.log_mass[:, 1]), np.array(calm.log_mass[:, 1]))
    # A closure is the same as the date being absent from the calendar: ages
    # keep advancing across it, so the following open date uses its own age.
    keep = [0, 2, 3, 4]
    reduced = kernel_for(ages[keep], storm[keep], allowed[keep])
    np.testing.assert_allclose(
        np.array(reduced.log_mass), np.array(kernel.log_mass)[keep], rtol=1e-6
    )
    np.testing.assert_allclose(np.array(reduced.log_tail), np.array(kernel.log_tail), rtol=1e-6)


# --- Exclusive accumulation and structural boundaries -------------------------


def test_exclusive_log_survival_handles_empty_time_and_structural_zero():
    half = float(np.log(0.5))
    step = jnp.array([[half, 0.0], [-jnp.inf, half], [half, half]])
    expected = np.array([[0.0, 0.0], [half, 0.0], [-np.inf, half]])
    np.testing.assert_allclose(np.array(exclusive_log_survival(step)), expected, rtol=1e-6)
    assert exclusive_log_survival(jnp.zeros((0, 2))).shape == (0, 2)
    batched = exclusive_log_survival(jnp.stack([step, 2 * step]))
    np.testing.assert_allclose(np.array(batched[1]), 2 * expected, rtol=1e-6)

    def finite_total(value):
        result = exclusive_log_survival(value)
        return jnp.sum(jnp.where(jnp.isfinite(result), result, 0.0))

    assert bool(jnp.all(jnp.isfinite(jax.grad(finite_total)(step))))


def test_certain_hazard_day_absorbs_all_susceptible_mass():
    # A family with a structurally certain event on day 1 (log stay -inf):
    # mass is pi*h0, then pi*(1-h0), nothing afterwards, residual exactly 1-pi.
    h0, h2, pi = 0.3, 0.6, 0.4
    hazard = [[h0], [1.0], [h2]]
    timing = TimingLaw(
        log_hazard=jnp.log(jnp.array(hazard)),
        log_survival_step=jnp.array([[np.log1p(-h0)], [-jnp.inf], [np.log1p(-h2)]]),
    )
    masks = _full_masks(3, 1)
    kernel = survival_kernel(timing, jnp.array([_logit(pi)]), allowed=masks, exposure=masks)
    expected_mass, expected_tail = _oracle_masses(
        pi, hazard, allowed=np.ones((3, 1), bool), exposure=np.ones((3, 1), bool)
    )
    np.testing.assert_allclose(_probabilities(kernel.log_mass), expected_mass, rtol=1e-6)
    np.testing.assert_allclose(_probabilities(kernel.log_tail), expected_tail, rtol=1e-6)
    assert np.isneginf(float(kernel.log_mass[2, 0]))
    log_probs = _unit_log_probs(kernel, 3, 1)[:, 0]
    np.testing.assert_allclose(log_probs[:2], np.log(expected_mass[:2, 0]), rtol=1e-6)
    assert np.isneginf(log_probs[2])
    np.testing.assert_allclose(log_probs[3], np.log(1 - pi), rtol=1e-6)


def test_empty_time_kernel_puts_all_mass_on_no_event():
    params = _constant_hazard_params(hazard=0.5, susceptibility_probability=0.4)
    timing, logits = default_timing(params, _inputs(jnp.zeros((0, 2), dtype=jnp.int32)))
    masks = _full_masks(0, 2)
    kernel = survival_kernel(timing, logits, allowed=masks, exposure=masks)
    assert kernel.log_mass.shape == (0, 2)
    np.testing.assert_allclose(np.array(kernel.log_tail), [0.0, 0.0], atol=1e-6)
    log_prob = kernel_unit_log_prob(kernel, jnp.array([-1, 0]))
    np.testing.assert_allclose(float(log_prob[0]), 0.0, atol=1e-6)
    assert np.isneginf(float(log_prob[1]))


def test_kernel_unit_log_prob_treats_out_of_range_dates_as_impossible():
    params = _constant_hazard_params(hazard=0.5, susceptibility_probability=0.4)
    timing, logits = default_timing(params, _inputs([[0, 0], [1, 1]]))
    masks = _full_masks(2, 2)
    kernel = survival_kernel(timing, logits, allowed=masks, exposure=masks)
    log_prob = kernel_unit_log_prob(kernel, jnp.array([2, -2]))
    assert bool(jnp.all(jnp.isneginf(log_prob)))


# --- Finite-mass timing adapter ---------------------------------------------


def test_timing_from_log_masses_terminal_atom_and_exhausted_support():
    # Susceptible law on ages 0..2 with masses .2/.3/.1 and a beyond-grid atom
    # .4 with no event on any grid age. Hazards are ratios of suffix masses;
    # past the grid only the atom remains, so hazard is zero and stay is one.
    grid = np.array([0.2, 0.3, 0.1])
    atom = 0.4
    ages = np.array([[0, -1], [1, 0], [2, 1], [3, 2], [4, 3], [5, 4]])
    timing = timing_from_log_masses(jnp.log(grid), jnp.log(atom), jnp.asarray(ages))
    suffix = np.cumsum(np.append(grid, atom)[::-1])[::-1]
    on_grid = np.clip(ages, 0, 2)
    expected_hazard = np.where(ages >= 3, 0.0, grid[on_grid] / suffix[on_grid])
    hazard, stay = _probabilities(timing.log_hazard), _probabilities(timing.log_survival_step)
    np.testing.assert_allclose(hazard, expected_hazard, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(hazard + stay, 1.0, rtol=1e-6)
    # Ratios ignore the overall scale of the susceptible masses.
    shifted = timing_from_log_masses(jnp.log(grid) + 3.0, jnp.log(atom) + 3.0, jnp.asarray(ages))
    for scaled, original in zip(shifted, timing, strict=True):
        np.testing.assert_allclose(np.array(scaled), np.array(original), rtol=1e-6, atol=1e-7)
    pi = 0.4
    kernel = survival_kernel(
        timing,
        jnp.full((2,), _logit(pi)),
        allowed=_full_masks(6, 2),
        exposure=jnp.asarray(ages >= 0),
    )
    expected_mass = np.where((ages >= 0) & (ages < 3), pi * grid[on_grid], 0.0)
    np.testing.assert_allclose(_probabilities(kernel.log_mass), expected_mass, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(_probabilities(kernel.log_tail), (1 - pi) + pi * atom, rtol=1e-6)


def test_timing_from_log_masses_proper_grid_exhausts_support_with_finite_gradients():
    # No terminal atom and a trailing structural zero: the last reachable age
    # fires with certainty (log stay -inf), later ages are unreachable, the
    # residual is exactly the cured mass, and gradients stay finite.
    log_grid = jnp.log(jnp.array([0.5, 0.5, 0.0]))
    log_atom = jnp.array(-jnp.inf)
    ages = jnp.arange(5)[:, None]
    half = np.log(0.5)
    timing = timing_from_log_masses(log_grid, log_atom, ages)
    np.testing.assert_allclose(
        np.array(timing.log_hazard[:, 0]), [half, 0.0, -np.inf, -np.inf, -np.inf]
    )
    np.testing.assert_allclose(
        np.array(timing.log_survival_step[:, 0]), [half, -np.inf, 0.0, 0.0, 0.0]
    )
    pi = 0.4
    masks = _full_masks(5, 1)
    kernel = survival_kernel(timing, jnp.array([_logit(pi)]), allowed=masks, exposure=masks)
    np.testing.assert_allclose(
        _probabilities(kernel.log_mass[:, 0]), [0.2, 0.2, 0.0, 0.0, 0.0], rtol=1e-6
    )
    np.testing.assert_allclose(_probabilities(kernel.log_tail), [1 - pi], rtol=1e-6)
    log_probs = _unit_log_probs(kernel, 5, 1)[:, 0]
    assert np.all(np.isneginf(log_probs[2:5]))
    np.testing.assert_allclose(log_probs[5], np.log(1 - pi), rtol=1e-6)

    def finite_total(log_grid, log_atom):
        law = timing_from_log_masses(log_grid, log_atom, ages)
        kernel = survival_kernel(law, jnp.array([_logit(pi)]), allowed=masks, exposure=masks)
        values = jnp.concatenate([kernel.log_mass[:, 0], kernel.log_tail])
        return jnp.sum(jnp.where(jnp.isfinite(values), values, 0.0))

    gradients = jax.grad(finite_total, argnums=(0, 1))(log_grid, log_atom)
    assert all(bool(jnp.all(jnp.isfinite(value))) for value in gradients)


# --- Boundary log-survival adapter -------------------------------------------


def test_timing_from_log_survival_matches_discretized_continuous_law():
    # Weibull(scale 6, shape 2) on ages 0..4 for one cohort and an exponential
    # (rate 1/2) for another: each bin's stay is S(a+1)/S(a), the hazard its
    # complement, and the two add to one on every cell.
    ages = np.arange(5)[:, None]
    log_survival = np.where(np.array([[True, False]]), -((ages / 6.0) ** 2), -ages / 2.0)
    log_survival_next = np.where(
        np.array([[True, False]]), -(((ages + 1) / 6.0) ** 2), -(ages + 1) / 2.0
    )
    timing = timing_from_log_survival(jnp.array(log_survival), jnp.array(log_survival_next))
    stay = np.exp(log_survival_next - log_survival)
    np.testing.assert_allclose(_probabilities(timing.log_survival_step), stay, rtol=1e-6)
    np.testing.assert_allclose(_probabilities(timing.log_hazard), 1 - stay, rtol=1e-6)
    np.testing.assert_allclose(
        _probabilities(timing.log_hazard) + _probabilities(timing.log_survival_step), 1.0, rtol=1e-6
    )
    np.testing.assert_allclose(_probabilities(timing.log_hazard[:, 1]), 1 - np.exp(-0.5), rtol=1e-6)


def test_timing_from_log_survival_structural_endpoints_keep_exact_values_and_finite_gradients():
    # Finite -> finite is an ordinary bin, finite -> -inf a certain terminal
    # bin (stay exactly zero, hazard one) and -inf -> -inf an exhausted cell
    # with nothing left to fire (hazard zero, stay one, never -inf - -inf).
    start = jnp.array([0.0, -1.0, -jnp.inf])
    end = jnp.array([-0.5, -jnp.inf, -jnp.inf])
    timing = timing_from_log_survival(start, end)
    np.testing.assert_allclose(
        _probabilities(timing.log_hazard), [1 - np.exp(-0.5), 1.0, 0.0], rtol=1e-6
    )
    np.testing.assert_allclose(
        _probabilities(timing.log_survival_step), [np.exp(-0.5), 0.0, 1.0], rtol=1e-6
    )
    assert np.isneginf(float(timing.log_survival_step[1]))
    assert np.isneginf(float(timing.log_hazard[2]))
    assert float(timing.log_hazard[1]) == 0.0 and float(timing.log_survival_step[2]) == 0.0

    def total(field):
        return lambda s, e: jnp.sum(getattr(timing_from_log_survival(s, e), field))

    for field in ("log_hazard", "log_survival_step"):
        gradients = jax.grad(total(field), argnums=(0, 1))(start, end)
        assert all(bool(jnp.all(jnp.isfinite(value))) for value in gradients)
    # Interior gradient of the hazard in the end-point: d/dx log(1 - e^-x) = 1 / expm1(x).
    derivative = jax.grad(lambda x: timing_from_log_survival(jnp.array(0.0), -x).log_hazard)(
        jnp.array(0.5)
    )
    np.testing.assert_allclose(float(derivative), 1 / np.expm1(0.5), rtol=1e-6)
    compiled = jax.jit(timing_from_log_survival)(start, end)
    np.testing.assert_array_equal(np.array(compiled.log_hazard), np.array(timing.log_hazard))
    np.testing.assert_array_equal(
        np.array(compiled.log_survival_step), np.array(timing.log_survival_step)
    )


def test_timing_from_log_survival_keeps_invalid_inputs_invalid():
    # NaN propagates through both fields; a survival that rises across a bin
    # is not a probability and must not be laundered into one, whether the
    # rise is finite (stay above one) or from an exhausted start (stay +inf).
    timing = timing_from_log_survival(
        jnp.array([jnp.nan, -0.5, -1.0, -jnp.inf]), jnp.array([-0.5, jnp.nan, -0.5, -1.0])
    )
    assert np.isnan(np.array(timing.log_hazard[:2])).all()
    assert np.isnan(np.array(timing.log_survival_step[:2])).all()
    assert float(timing.log_survival_step[2]) > 0.0
    assert np.isposinf(float(timing.log_survival_step[3]))


def test_timing_from_log_survival_broadcasts_cells_and_feeds_the_kernel():
    # One continuous curve per cohort evaluated at the cells' own bin edges:
    # ``[T, 1]`` starts against ``[T, C]`` ends broadcast to ``[T, C]``, and
    # the kernel turns the law into the mixture masses pi * (S(a) - S(a+1))
    # with residual (1 - pi) + pi * S(T); exposure is the kernel's job alone.
    days, pi = 4, 0.7
    ages = np.arange(days)[:, None]
    scale = np.array([[3.0, 5.0]])
    log_survival = -((ages / scale) ** 1.5)
    log_survival_next = -(((ages + 1) / scale) ** 1.5)
    timing = timing_from_log_survival(jnp.array(log_survival[:, :1]), jnp.array(log_survival_next))
    assert timing.log_hazard.shape == (days, 2) and timing.log_survival_step.shape == (days, 2)
    timing = timing_from_log_survival(jnp.array(log_survival), jnp.array(log_survival_next))
    kernel = survival_kernel(
        timing,
        jnp.full(2, _logit(pi)),
        allowed=_full_masks(days, 2),
        exposure=_full_masks(days, 2),
    )
    masses = pi * (np.exp(log_survival) - np.exp(log_survival_next))
    np.testing.assert_allclose(_probabilities(kernel.log_mass), masses, rtol=1e-5)
    np.testing.assert_allclose(
        _probabilities(kernel.log_tail), (1 - pi) + pi * np.exp(log_survival_next[-1]), rtol=1e-5
    )


# --- Default StageParameters adapter -----------------------------------------


def test_default_timing_matches_stage_parameter_primitives():
    rng = np.random.default_rng(3)
    days, cohorts, p, q = 7, 4, 2, 3
    params = StageParameters(
        age_logits=jnp.array(rng.normal(size=4)),
        beta=jnp.array(rng.normal(size=p)),
        susceptibility_intercept=jnp.array(rng.normal()),
        susceptibility_beta=jnp.array(rng.normal(size=q)),
    )
    ages = jnp.array(np.arange(days)[:, None] + rng.integers(0, 6, size=(1, cohorts)))
    features = jnp.array(rng.normal(size=(days, cohorts, p)))
    susceptibility_features = jnp.array(rng.normal(size=(cohorts, q)))
    law = default_timing(params, TimingInputs(ages, features, susceptibility_features))
    assert isinstance(law, EventLaw)
    timing, logits = law.timing, law.susceptibility_logits
    hazard = np.array(stage_hazard(params, ages, features))
    np.testing.assert_allclose(_probabilities(timing.log_hazard), hazard, rtol=1e-5)
    np.testing.assert_allclose(
        _probabilities(timing.log_survival_step), 1 - hazard, rtol=1e-4, atol=1e-6
    )
    np.testing.assert_allclose(
        np.array(jax.nn.sigmoid(logits)),
        np.array(susceptibility(params, susceptibility_features)),
        rtol=1e-6,
    )


# --- Extreme magnitudes ------------------------------------------------------


def test_long_history_underflow_stays_finite_in_log_space():
    # 5000 event-free days at h=.5: the susceptible survival 2^-5000 is far
    # below float range, yet log masses stay finite and the censored
    # likelihood is exactly the cured mass.
    days, pi = 5000, 0.4
    params = _constant_hazard_params(hazard=0.5, susceptibility_probability=pi)
    ages = jnp.broadcast_to(jnp.arange(days)[:, None], (days, 2))
    timing, logits = default_timing(params, _inputs(ages))
    masks = _full_masks(days, 2)
    kernel = survival_kernel(timing, logits, allowed=masks, exposure=masks)
    expected_last = np.log(pi) + days * np.log(0.5)
    np.testing.assert_allclose(float(kernel.log_mass[-1, 0]), expected_last, rtol=1e-5)
    assert float(jnp.exp(kernel.log_mass[-1, 0])) == 0.0
    log_prob = kernel_unit_log_prob(kernel, jnp.array([days - 1, -1]))
    np.testing.assert_allclose(float(log_prob[0]), expected_last, rtol=1e-5)
    np.testing.assert_allclose(float(log_prob[1]), np.log(1 - pi), atol=1e-6)
    np.testing.assert_allclose(float(kernel.log_tail[0]), np.log(1 - pi), atol=1e-6)
    # The same history known before entry conditions the susceptibility to a
    # finite, deeply negative logit instead of a fake certain cure.
    pre_entry = jnp.concatenate([jnp.ones((days, 2), bool), jnp.zeros((2, 2), bool)])
    timing, logits = default_timing(params, _inputs(jnp.concatenate([ages, ages[:2] + days])))
    conditioned = survival_kernel(
        timing, logits, allowed=_full_masks(days + 2, 2), exposure=~pre_entry, pre_entry=pre_entry
    )
    np.testing.assert_allclose(
        float(conditioned.log_susceptible[0]), _logit(pi) + days * np.log(0.5), rtol=1e-5
    )
    censored = kernel_unit_log_prob(conditioned, jnp.array([-1, -1]))
    np.testing.assert_allclose(np.array(censored), 0.0, atol=1e-6)


@pytest.mark.parametrize("logit", [-100.0, -25.0, 0.0, 40.0, 100.0])
def test_extreme_logits_keep_finite_values_and_gradients(logit):
    params = StageParameters(jnp.array([logit]), jnp.empty(0), jnp.array(logit), jnp.empty(0))
    inputs = _inputs([[0, 0], [1, 1]])
    masks = _full_masks(2, 2)
    event_index = jnp.array([1, -1])
    log_h = -np.logaddexp(0.0, -logit)
    log_s = -np.logaddexp(0.0, logit)
    expected = [2 * log_h + log_s, np.logaddexp(log_s, log_h + 2 * log_s)]

    def kernel_for(params):
        timing, logits = default_timing(params, inputs)
        return survival_kernel(timing, logits, allowed=masks, exposure=masks)

    def unit_log_prob(params):
        return kernel_unit_log_prob(kernel_for(params), event_index)

    def law_total(params):
        kernel = kernel_for(params)
        return jnp.sum(kernel.log_mass) + jnp.sum(kernel.log_tail)

    np.testing.assert_allclose(np.array(unit_log_prob(params)), expected, rtol=1e-6, atol=1e-6)
    for total in (lambda value: unit_log_prob(value).sum(), law_total):
        gradients = jax.grad(total)(params)
        assert all(bool(jnp.all(jnp.isfinite(value))) for value in gradients)


def test_log1mexp_boundaries_and_gradients():
    # Interior accuracy where the naive log1p(-exp(x)) loses digits in float32.
    near = jnp.float32(-1e-6)
    np.testing.assert_allclose(float(log1mexp(near)), np.log(-np.expm1(-1e-6)), rtol=1e-5)
    far = jnp.float32(-25.0)
    np.testing.assert_allclose(float(log1mexp(far)), -np.exp(-25.0), rtol=1e-5)
    np.testing.assert_allclose(float(jax.grad(log1mexp)(far)), -np.exp(-25.0), rtol=1e-5)
    np.testing.assert_allclose(float(log1mexp(jnp.float32(np.log(0.5)))), np.log(0.5), rtol=1e-6)
    # Boundary operands: zero log stay is a certain event, -inf is no event.
    assert np.isneginf(float(log1mexp(jnp.float32(0.0))))
    assert float(log1mexp(jnp.float32(-jnp.inf))) == 0.0
    for value in (0.0, -jnp.inf, -100.0):
        assert np.isfinite(float(jax.grad(log1mexp)(jnp.float32(value))))
    # Elementwise over arrays, including mixed boundaries.
    result = log1mexp(jnp.array([0.0, -0.5, -3.0, -jnp.inf]))
    expected = [-np.inf, np.log(1 - np.exp(-0.5)), np.log(1 - np.exp(-3.0)), 0.0]
    np.testing.assert_allclose(np.array(result), expected, rtol=1e-6)


# --- Masks and batching ------------------------------------------------------


def test_survival_kernel_rejects_overlapping_pre_entry_and_exposure():
    params = _constant_hazard_params(hazard=0.5, susceptibility_probability=0.4)
    timing, logits = default_timing(params, _inputs([[0], [1], [2]]))
    allowed = _full_masks(3, 1)
    pre_entry = jnp.array([[True], [True], [False]])
    with pytest.raises(ValueError, match="disjoint"):
        survival_kernel(timing, logits, allowed=allowed, exposure=allowed, pre_entry=pre_entry)
    disjoint = survival_kernel(
        timing, logits, allowed=allowed, exposure=~pre_entry, pre_entry=pre_entry
    )
    traced = jax.jit(
        lambda allowed, exposure, pre_entry: survival_kernel(
            timing, logits, allowed=allowed, exposure=exposure, pre_entry=pre_entry
        )
    )(allowed, ~pre_entry, pre_entry)
    for eager, compiled in zip(disjoint, traced, strict=True):
        np.testing.assert_allclose(np.array(eager), np.array(compiled))


def test_kernel_unit_log_prob_broadcasts_batched_kernels_and_values():
    inputs = _inputs([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    masks = _full_masks(3, 3)
    single = [
        survival_kernel(*default_timing(params, inputs), allowed=masks, exposure=masks)
        for params in (_constant_hazard_params(0.3, 0.4), _constant_hazard_params(0.6, 0.8))
    ]
    batched = jax.tree.map(lambda *xs: jnp.stack(xs), *single)
    assert batched.log_mass.shape == (2, 3, 3)
    assert batched.log_tail.shape == (2, 3)
    values = jnp.array([[0, 2, -1], [1, -1, 1], [-1, 0, 2]])
    shared = kernel_unit_log_prob(batched, values[0])
    assert shared.shape == (2, 3)
    for draw in range(2):
        expected = kernel_unit_log_prob(single[draw], values[0])
        np.testing.assert_allclose(np.array(shared[draw]), np.array(expected))
        np.testing.assert_allclose(
            np.array(batched.log_mass[draw]), np.array(single[draw].log_mass)
        )
    stacked = kernel_unit_log_prob(batched, values[:, None, :])
    assert stacked.shape == (3, 2, 3)
    for sample in range(3):
        for draw in range(2):
            expected = kernel_unit_log_prob(single[draw], values[sample])
            np.testing.assert_allclose(np.array(stacked[sample, draw]), np.array(expected))
    sampled = kernel_unit_log_prob(single[0], values)
    assert sampled.shape == (3, 3)
    for sample in range(3):
        expected = kernel_unit_log_prob(single[0], values[sample])
        np.testing.assert_allclose(np.array(sampled[sample]), np.array(expected))


def test_empty_kernel_rejects_out_of_domain_event_indices():
    timing = TimingLaw(jnp.empty((0, 3)), jnp.empty((0, 3)))
    mask = jnp.empty((0, 3), dtype=bool)
    kernel = survival_kernel(timing, jnp.zeros(3), allowed=mask, exposure=mask)
    indices = jnp.array([-2, -1, 0])
    np.testing.assert_array_equal(
        kernel_unit_log_prob(kernel, indices), jnp.array([-jnp.inf, 0.0, -jnp.inf])
    )


def test_mass_adapter_does_not_turn_nan_weights_into_a_valid_law():
    timing = timing_from_log_masses(
        jnp.array([jnp.nan, 0.0]), jnp.array(-jnp.inf), jnp.arange(3)[:, None]
    )
    assert np.isnan(float(timing.log_hazard[0, 0]))
    assert np.isnan(float(timing.log_survival_step[0, 0]))


def test_mass_adapter_distinguishes_per_draw_from_per_cohort_grids():
    masses = jnp.array([[0.2, 0.3], [0.6, 0.1]])
    tails = jnp.array([0.5, 0.3])
    ages = jnp.array([[0, 0], [1, 1]])
    per_cohort = timing_from_log_masses(jnp.log(masses), jnp.log(tails), ages)
    np.testing.assert_allclose(
        jnp.exp(per_cohort.log_hazard), [[0.2, 0.6], [0.375, 0.25]], rtol=1e-6
    )
    per_draw = timing_from_log_masses(jnp.log(masses[:, None, :]), jnp.log(tails[:, None]), ages)
    np.testing.assert_allclose(
        jnp.exp(per_draw.log_hazard),
        [[[0.2, 0.2], [0.375, 0.375]], [[0.6, 0.6], [0.25, 0.25]]],
        rtol=1e-6,
    )


def test_mass_adapter_broadcasts_per_cohort_grids_against_a_shared_atom():
    # Per-cohort grids [C, A] with one scalar beyond-grid atom; cohort 1 has a
    # structural zero at age 1. On the grid, hazard is m[a] / R[a] and stay is
    # R[a + 1] / R[a] with R the suffix mass including the atom; past the grid
    # hazard is zero and stay one. In the log weights w (the atom last), every
    # finite cell has d log h[a] / d w[b] = delta[a, b] - m[b] / R[a] and
    # d log s[a] / d w[b] = m[b] / R[a + 1] - m[b] / R[a], each share present
    # only where b lies in that suffix.
    masses = np.array([[0.2, 0.3, 0.1], [0.5, 0.0, 0.2]])
    atom = 0.4
    ages = np.array([[-1, 0], [0, 1], [1, 2], [2, 3], [3, 4]])
    log_masses = jnp.log(jnp.asarray(masses))
    log_atom = jnp.log(jnp.asarray(atom))

    def law(log_masses, log_atom):
        timing = timing_from_log_masses(log_masses, log_atom, jnp.asarray(ages))
        return timing.log_hazard, timing.log_survival_step

    log_hazard, log_stay = law(log_masses, log_atom)
    weights = np.concatenate([masses, np.full((2, 1), atom)], axis=1)
    remaining = np.cumsum(weights[:, ::-1], axis=1)[:, ::-1]
    cohort = np.arange(2)[None, :]
    on_grid = np.clip(ages, 0, 2)
    beyond = ages >= 3
    np.testing.assert_allclose(
        _probabilities(log_hazard),
        np.where(beyond, 0.0, masses[cohort, on_grid] / remaining[cohort, on_grid]),
        rtol=1e-6,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        _probabilities(log_stay),
        np.where(beyond, 1.0, remaining[cohort, on_grid + 1] / remaining[cohort, on_grid]),
        rtol=1e-6,
        atol=1e-8,
    )
    (hazard_mass, hazard_atom), (stay_mass, stay_atom) = jax.jacobian(law, argnums=(0, 1))(
        log_masses, log_atom
    )
    index = np.arange(4)
    for (t, c), age in np.ndenumerate(ages):
        if age < 3:
            a = max(int(age), 0)
            share = np.where(index >= a, weights[c] / remaining[c, a], 0.0)
            after = np.where(index > a, weights[c] / remaining[c, a + 1], 0.0)
            rows = {"hazard": np.eye(4)[a] - share, "stay": after - share}
        else:
            rows = {"hazard": np.zeros(4), "stay": np.zeros(4)}
        cells = (
            ("hazard", log_hazard, hazard_mass, hazard_atom),
            ("stay", log_stay, stay_mass, stay_atom),
        )
        for name, output, mass_jacobian, atom_jacobian in cells:
            if not np.isfinite(float(output[t, c])):
                continue
            expected = np.zeros((2, 3))
            expected[c] = rows[name][:3]
            np.testing.assert_allclose(np.array(mass_jacobian[t, c]), expected, atol=1e-6)
            np.testing.assert_allclose(float(atom_jacobian[t, c]), rows[name][3], atol=1e-6)


def test_mass_adapter_broadcasts_per_draw_grids_and_atoms_over_cohorts():
    # Per-draw grids [D, 1, A] with one scalar atom give every cohort of draw d
    # the law of that draw's grid alone.
    draws = jnp.array([[0.2, 0.3], [0.6, 0.1], [0.4, 0.4]])
    ages = jnp.array([[0, 0], [1, 1]])
    per_draw = timing_from_log_masses(jnp.log(draws[:, None, :]), jnp.log(0.5), ages)
    assert per_draw.log_hazard.shape == (3, 2, 2)
    for draw in range(3):
        single = timing_from_log_masses(jnp.log(draws[draw]), jnp.log(0.5), ages)
        for broadcast, alone in zip(per_draw, single, strict=True):
            np.testing.assert_allclose(np.array(broadcast[draw]), np.array(alone), rtol=1e-6)
    # Per-cohort grids [C, A] against per-draw atoms [D, 1]: draw 0 pairs every
    # cohort with atom .5, draw 1 with atom .3.
    masses = jnp.array([[0.2, 0.3], [0.6, 0.1]])
    atoms = jnp.array([[0.5], [0.3]])
    mixed = timing_from_log_masses(jnp.log(masses), jnp.log(atoms), ages)
    assert mixed.log_hazard.shape == (2, 2, 2)
    np.testing.assert_allclose(
        _probabilities(mixed.log_hazard),
        [[[0.2, 0.6 / 1.2], [0.3 / 0.8, 0.1 / 0.6]], [[0.2 / 0.8, 0.6], [0.3 / 0.6, 0.1 / 0.4]]],
        rtol=1e-6,
    )
