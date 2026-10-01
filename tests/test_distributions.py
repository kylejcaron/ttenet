"""Consumer-visible contracts of the native event-time laws.

Every expected number here comes from an independent derivation (the
mixture-cure identities, multinomial moments, ``math.lgamma`` at small counts,
or a recorded 60-digit Decimal Stirling evaluation for the large-count cases),
never from the implementation under test.
"""

from __future__ import annotations

import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from numpyro_forecast.surgery import prefix_condition, slice_time

from ttenet.distributions import CohortEventTime, EventTime
from ttenet.event_times import SurvivalKernel

LOG_HALF = math.log(0.5)
SUSCEPTIBLE_LOGIT = math.log(0.4 / 0.6)


def _explicit(masses, tail):
    """EventTime from event probabilities [day, unit] and no-event probabilities [unit]."""
    return EventTime(jnp.log(jnp.asarray(masses)), jnp.log(jnp.asarray(tail)))


def _cure_kernel(hazard, days, units=1, logit=SUSCEPTIBLE_LOGIT, closed=()):
    """Constant-hazard mixture-cure kernel with optional closed calendar days."""
    log_hazard = jnp.full((days, units), math.log(hazard))
    log_stay = jnp.full((days, units), math.log1p(-hazard))
    for day in closed:
        log_hazard = log_hazard.at[day].set(-jnp.inf)
        log_stay = log_stay.at[day].set(0.0)
    return SurvivalKernel(log_hazard, log_stay, jnp.full((units,), logit))


def _paths(days, units):
    """Every binary trajectory with at most one event per unit."""
    for dates in itertools.product(range(days + 1), repeat=units):
        yield (jnp.arange(days)[:, None] == jnp.asarray(dates)[None, :]).astype(jnp.int32)


def _multinomial_log_pmf(counts, total, probabilities):
    occupancy = list(counts) + [total - sum(counts)]
    value = math.lgamma(total + 1)
    for x, p in zip(occupancy, probabilities, strict=True):
        value += x * math.log(p) - math.lgamma(x + 1)
    return value


# --------------------------------------------------------------------- EventTime


def test_unit_masses_follow_the_mixture_cure_law_on_both_routes():
    kernel = _cure_kernel(0.5, days=2)
    laws = [EventTime(kernel=kernel), EventTime.from_kernel(kernel)]
    laws.append(EventTime(laws[0].log_mass, laws[0].log_tail))
    laws.append(_explicit([[0.2], [0.1]], [0.7]))
    day0 = jnp.array([[1], [0]], dtype=jnp.int32)
    day1 = jnp.array([[0], [1]], dtype=jnp.int32)
    censored = jnp.zeros((2, 1), dtype=jnp.int32)
    for law in laws:
        assert law.batch_shape == ()
        assert law.event_shape == (2, 1)
        np.testing.assert_allclose(law.unit_log_prob(day0), [math.log(0.2)], rtol=1e-5)
        np.testing.assert_allclose(law.unit_log_prob(day1), [math.log(0.1)], rtol=1e-5)
        np.testing.assert_allclose(law.log_prob(censored), math.log(0.7), rtol=1e-5)
        np.testing.assert_allclose(law.mean, [[0.2], [0.1]], atol=1e-6)
        np.testing.assert_allclose(law.variance, [[0.16], [0.09]], atol=1e-6)
        np.testing.assert_allclose(jnp.exp(law.log_probabilities).sum(axis=-2), 1.0, atol=1e-6)


def test_ambiguous_or_incomplete_construction_is_refused():
    kernel = _cure_kernel(0.5, days=2)
    log_mass = jnp.log(jnp.array([[0.2], [0.1]]))
    with pytest.raises(ValueError, match="kernel"):
        EventTime(log_mass, jnp.log(jnp.array([0.7])), kernel=kernel)
    with pytest.raises(ValueError, match="kernel"):
        EventTime()
    with pytest.raises(ValueError, match="log_tail"):
        EventTime(log_mass)
    with pytest.raises(ValueError, match="log_mass"):
        EventTime(log_tail=jnp.log(jnp.array([0.7])))


def test_repeated_and_closed_date_events_are_impossible():
    closed_kernel = _cure_kernel(0.5, days=3, closed=(1,))
    laws = [EventTime(kernel=closed_kernel), _explicit([[0.2], [0.0], [0.1]], [0.7])]
    repeated = jnp.array([[1], [0], [1]], dtype=jnp.int32)
    doubled = jnp.array([[2], [0], [0]], dtype=jnp.int32)
    closed = jnp.array([[0], [1], [0]], dtype=jnp.int32)
    for law in laws:
        assert law.log_prob(repeated) == -jnp.inf
        assert law.log_prob(doubled) == -jnp.inf
        assert law.log_prob(closed) == -jnp.inf
        assert law.log_prob(jnp.array([[0], [0], [-1]], dtype=jnp.int32)) == -jnp.inf
        np.testing.assert_allclose(law.mean[1], 0.0)
        total = jnp.exp(law.log_probabilities).sum(axis=-2)
        np.testing.assert_allclose(total, 1.0, atol=1e-6)


def test_event_time_prefix_keeps_observed_units_absorbed():
    law = EventTime(
        jnp.log(jnp.array([[0.2, 0.0], [0.0, 0.5], [0.3, 0.0]])),
        jnp.log(jnp.array([0.5, 0.5])),
    )
    prefix = jnp.array([[1, 0]], dtype=jnp.int32)
    future = prefix_condition(law, prefix)
    assert future.event_shape == (2, 2)
    np.testing.assert_allclose(future.mean, [[0.0, 0.5], [0.0, 0.0]], atol=1e-6)
    full = jnp.array([[1, 0], [0, 1], [0, 0]], dtype=jnp.int32)
    np.testing.assert_allclose(
        slice_time(law, slice(0, 1)).log_prob(prefix) + future.log_prob(full[1:]),
        law.log_prob(full),
        rtol=1e-5,
    )
    # An absorbed unit cannot fire again; a pending unit renormalizes.
    assert future.unit_log_prob(full[1:])[0] == 0.0
    np.testing.assert_allclose(future.unit_log_prob(full[1:])[1], math.log(0.5), atol=1e-6)
    assert future.log_prob(jnp.array([[0, 0], [1, 0]], dtype=jnp.int32)) == -jnp.inf


def test_kernel_route_prefix_and_window_reproduce_conditional_entry_numbers():
    kernel = _cure_kernel(0.5, days=2)
    for law in (EventTime(kernel=kernel), _explicit([[0.2], [0.1]], [0.7])):
        survived = jnp.zeros((1, 1), dtype=jnp.int32)
        future = prefix_condition(law, survived)
        np.testing.assert_allclose(future.mean, [[0.125]], atol=1e-6)
        np.testing.assert_allclose(
            future.log_prob(jnp.zeros((1, 1), dtype=jnp.int32)), math.log(0.875), rtol=1e-5
        )
        window = slice_time(law, slice(1, 2))
        np.testing.assert_allclose(window.mean, [[0.1]], atol=1e-6)
        np.testing.assert_allclose(
            window.log_prob(jnp.zeros((1, 1), dtype=jnp.int32)), math.log(0.9), rtol=1e-5
        )
        head = slice_time(law, slice(0, 1))
        np.testing.assert_allclose(
            head.log_prob(jnp.ones((1, 1), jnp.int32)), math.log(0.2), rtol=1e-5
        )
        for path in _paths(2, 1):
            conditioned = prefix_condition(law, path[:1])
            np.testing.assert_allclose(
                head.log_prob(path[:1]) + conditioned.log_prob(path[1:]),
                law.log_prob(path),
                rtol=1e-5,
            )


def test_kernel_route_surgeries_match_the_explicit_law():
    key = random.PRNGKey(3)
    days, units = 6, 4
    k_h, k_l = random.split(key)
    hazard_logit = random.normal(k_h, (days, units))
    log_hazard = jax.nn.log_sigmoid(hazard_logit)
    log_stay = jax.nn.log_sigmoid(-hazard_logit)
    closed = jnp.zeros((days, units), bool).at[2, :].set(True).at[4, 1].set(True)
    log_hazard = jnp.where(closed, -jnp.inf, log_hazard)
    log_stay = jnp.where(closed, 0.0, log_stay)
    logits = random.normal(k_l, (units,)).at[3].set(-jnp.inf)
    kernel = SurvivalKernel(log_hazard, log_stay, logits)
    compact = EventTime(kernel=kernel)
    explicit = EventTime(compact.log_mass, compact.log_tail)
    np.testing.assert_allclose(compact.log_probabilities, explicit.log_probabilities, atol=1e-5)
    prefix = jnp.zeros((3, units), jnp.int32).at[1, 0].set(1)
    pairs = [
        (slice_time(compact, slice(2, 5)), slice_time(explicit, slice(2, 5))),
        (slice_time(compact, slice(0, 3)), slice_time(explicit, slice(0, 3))),
        (prefix_condition(compact, prefix), prefix_condition(explicit, prefix)),
    ]
    for kernel_law, explicit_law in pairs:
        assert kernel_law.event_shape == explicit_law.event_shape
        np.testing.assert_allclose(kernel_law.mean, explicit_law.mean, atol=1e-5)
        for path in _paths(3, 1):
            value = jnp.tile(path, (1, units))
            kernel_lp = kernel_law.unit_log_prob(value)
            explicit_lp = explicit_law.unit_log_prob(value)
            finite = jnp.isfinite(explicit_lp)
            np.testing.assert_array_equal(jnp.isfinite(kernel_lp), finite)
            np.testing.assert_allclose(kernel_lp[finite], explicit_lp[finite], atol=1e-5)


def test_direct_infeasible_unit_prefix_is_refused_and_traced_paths_are_impossible():
    kernel = _cure_kernel(0.5, days=3)
    laws = [EventTime(kernel=kernel), _explicit([[0.2], [0.1], [0.05]], [0.65])]
    repeated = jnp.array([[1], [1]], dtype=jnp.int32)
    doubled = jnp.array([[2], [0]], dtype=jnp.int32)
    for law in laws:
        with pytest.raises(ValueError, match="prefix"):
            prefix_condition(law, repeated)
        with pytest.raises(ValueError, match="prefix"):
            prefix_condition(law, doubled)

        @jax.jit
        def traced(data, law=law):
            future = prefix_condition(law, data)
            suffix = jnp.zeros((1, 1), dtype=jnp.int32)
            return (
                slice_time(law, slice(0, 2)).log_prob(data),
                future.log_prob(suffix),
                future.log_prob(jnp.ones((1, 1), dtype=jnp.int32)),
            )

        observed, silent, fired = traced(repeated)
        assert observed == -jnp.inf
        assert silent == -jnp.inf
        assert fired == -jnp.inf
        observed, silent, fired = traced(jnp.zeros((2, 1), dtype=jnp.int32))
        np.testing.assert_allclose(observed, math.log(0.7), rtol=1e-5)
        np.testing.assert_allclose(silent, math.log(0.65 / 0.7), rtol=1e-5)
        np.testing.assert_allclose(fired, math.log(0.05 / 0.7), rtol=1e-5)


def test_impossible_traced_units_stay_impossible_in_empty_suffixes_and_windows():
    kernel = _cure_kernel(0.5, days=3, units=2)
    laws = [EventTime(kernel=kernel), _explicit([[0.2] * 2, [0.1] * 2, [0.05] * 2], [0.65] * 2)]
    nothing = jnp.zeros((0, 2), dtype=jnp.int32)
    invalid_full = jnp.array([[1, 0], [1, 0], [0, 0]], dtype=jnp.int32)
    invalid_head = jnp.array([[1, 0], [1, 0]], dtype=jnp.int32)
    absorbed_full = jnp.array([[0, 0], [1, 0], [0, 1]], dtype=jnp.int32)
    pending_full = jnp.zeros((3, 2), dtype=jnp.int32)
    for law in laws:

        @jax.jit
        def traced(full, head, law=law):
            exhausted = prefix_condition(law, full)
            window = slice_time(prefix_condition(law, head), slice(0, 0))
            return (
                exhausted.unit_log_prob(nothing),
                window.unit_log_prob(nothing),
                exhausted.log_tail,
                prefix_condition(law, head).mean,
                prefix_condition(law, head).sample(random.PRNGKey(0), (3,)),
            )

        exhausted, window, tail, mean, draws = traced(invalid_full, invalid_head)
        assert exhausted[0] == -jnp.inf and exhausted[1] == 0.0
        assert window[0] == -jnp.inf and window[1] == 0.0
        assert tail[0] == -jnp.inf and bool(jnp.isfinite(tail[1]))
        np.testing.assert_array_equal(mean[:, 0], 0.0)
        np.testing.assert_allclose(mean[:, 1], [0.05 / 0.7], rtol=1e-5)
        # An impossible unit never draws a plausible cure: its trajectory is a
        # visible off-support sentinel, while its sibling keeps drawing the law.
        assert draws.shape == (3, 1, 2)
        assert bool(jnp.all(draws[:, :, 0] < 0))
        assert bool(jnp.all((draws[:, :, 1] == 0) | (draws[:, :, 1] == 1)))
        for full in (absorbed_full, pending_full):
            exhausted, window, tail, _, _ = traced(full, full[:2])
            np.testing.assert_array_equal(exhausted, 0.0)
            np.testing.assert_array_equal(window, 0.0)
            assert bool(jnp.all(jnp.isfinite(tail)))


def test_observed_prefixes_without_mass_are_refused_or_impossible():
    closed_kernel = SurvivalKernel(
        jnp.array([[-jnp.inf], [LOG_HALF]]), jnp.array([[0.0], [LOG_HALF]]), jnp.array([0.3])
    )
    cured_kernel = _cure_kernel(0.5, days=2, logit=-jnp.inf)
    fired = jnp.array([[1]], dtype=jnp.int32)
    survived = jnp.array([[0]], dtype=jnp.int32)
    cases = [
        (EventTime(kernel=closed_kernel), fired),
        (EventTime(kernel=cured_kernel), fired),
        (_explicit([[0.0], [0.5]], [0.5]), fired),
        (
            EventTime(
                kernel=SurvivalKernel(
                    jnp.zeros((1, 1)), jnp.array([[-jnp.inf]]), jnp.array([jnp.inf])
                )
            ),
            survived,
        ),
        (_explicit([[1.0]], [0.0]), survived),
    ]
    for law, prefix in cases:
        with pytest.raises(ValueError, match="prefix"):
            prefix_condition(law, prefix)
        days = law.event_shape[0]
        full = jnp.concatenate([prefix, jnp.zeros((days - 1, 1), jnp.int32)])

        @jax.jit
        def traced(head, full, law=law):
            future = prefix_condition(law, head)
            return (
                future.log_prob(jnp.zeros((days - 1, 1), jnp.int32)),
                future.log_tail,
                future.sample(random.PRNGKey(0), (2,)),
                prefix_condition(law, full).log_prob(jnp.zeros((0, 1), jnp.int32)),
            )

        silent, tail, draws, exhausted = traced(prefix, full)
        assert silent == -jnp.inf
        assert tail[0] == -jnp.inf
        assert exhausted == -jnp.inf
        assert draws.shape == (2, days - 1, 1)
        assert bool(jnp.all(draws < 0))


def test_pool_prefixes_without_mass_are_refused_or_carry_the_sentinel():
    weights = jnp.log(jnp.array([0.0, 0.5, 0.5]))
    law = CohortEventTime(weights[:-1, None], weights[-1:], jnp.asarray([4]))
    with pytest.raises(ValueError, match="prefix"):
        prefix_condition(law, jnp.array([[1]], dtype=jnp.int32))
    exhausted_tail = CohortEventTime(
        jnp.log(jnp.array([[0.5], [0.5]])), jnp.array([-jnp.inf]), jnp.asarray([4])
    )
    with pytest.raises(ValueError, match="prefix"):
        prefix_condition(exhausted_tail, jnp.array([[1], [2]], dtype=jnp.int32))
    remaining = prefix_condition(exhausted_tail, jnp.array([[1], [3]], dtype=jnp.int32))
    assert int(remaining.total_count[0]) == 0
    traced = jax.jit(lambda data: prefix_condition(law, data).total_count)
    assert int(traced(jnp.array([[1]], dtype=jnp.int32))[0]) < 0
    assert int(traced(jnp.array([[0]], dtype=jnp.int32))[0]) == 4


def test_undefined_laws_never_draw_a_plausible_cure():
    no_mass = _explicit([[0.0], [0.0]], [0.0])
    assert bool(jnp.all(no_mass.sample(random.PRNGKey(0), (3,)) < 0))
    assert no_mass.log_prob(jnp.zeros((2, 1), jnp.int32)) == -jnp.inf
    pool = CohortEventTime(no_mass.log_mass, no_mass.log_tail, jnp.asarray([3]))
    assert bool(jnp.all(pool.sample(random.PRNGKey(0), (3,)) < 0))
    assert pool.log_prob(jnp.zeros((2, 1), jnp.int32)) == -jnp.inf
    empty = CohortEventTime(no_mass.log_mass, no_mass.log_tail, jnp.asarray([0]))
    np.testing.assert_array_equal(empty.sample(random.PRNGKey(0), (3,)), 0)
    assert empty.log_prob(jnp.zeros((2, 1), jnp.int32)) == 0.0


def test_nan_inputs_propagate_instead_of_becoming_certain_cure():
    zeros = jnp.zeros((2, 1), dtype=jnp.int32)
    kernel = SurvivalKernel(
        jnp.full((2, 1), LOG_HALF), jnp.full((2, 1), LOG_HALF), jnp.array([jnp.nan])
    )
    law = EventTime(kernel=kernel)
    assert bool(jnp.isnan(law.log_prob(zeros)))
    assert bool(jnp.isnan(prefix_condition(law, zeros[:1]).log_prob(zeros[1:])))
    assert bool(jnp.isnan(slice_time(law, slice(1, 2)).log_prob(zeros[1:])))
    assert bool(jnp.all(law.sample(random.PRNGKey(0), (2,)) < 0))
    # Concrete explicit weights refuse NaN at the boundary; traced ones (as under
    # inference) must carry it through rather than hide it as an impossibility.
    weights = jnp.array([[jnp.nan], [LOG_HALF]])
    with pytest.raises(ValueError, match="log_mass"):
        EventTime(weights, jnp.array([LOG_HALF]))
    explicit = EventTime(weights, jnp.array([LOG_HALF]), validate_args=False)
    assert bool(jnp.isnan(explicit.log_prob(zeros)))
    assert bool(jnp.isnan(explicit.log_prob(zeros.at[1].set(1))))
    assert bool(jnp.all(explicit.sample(random.PRNGKey(0), (2,)) < 0))
    pool = CohortEventTime(
        explicit.log_mass, explicit.log_tail, jnp.asarray([3]), validate_args=False
    )
    assert bool(jnp.isnan(pool.log_prob(zeros)))
    assert bool(jnp.isnan(pool.log_prob(zeros.at[1].set(2))))
    assert bool(jnp.all(pool.sample(random.PRNGKey(0), (2,)) < 0))


def test_window_gradients_survive_certain_susceptibility_after_closed_prefix():
    hazard = jnp.array([[-jnp.inf], [LOG_HALF]])
    stay = jnp.array([[0.0], [LOG_HALF]])
    fired = jnp.ones((1, 1), dtype=jnp.int32)

    def windowed(stay):
        kernel = SurvivalKernel(hazard, stay, jnp.array([jnp.inf]))
        return slice_time(EventTime(kernel=kernel), slice(1, 2)).log_prob(fired)

    np.testing.assert_allclose(windowed(stay), LOG_HALF, rtol=1e-6)
    assert bool(jnp.all(jnp.isfinite(jax.grad(windowed)(stay))))


def test_malformed_values_and_prefixes_are_refused():
    unit = EventTime(kernel=_cure_kernel(0.5, days=2))
    pool = _pool_law([0.2, 0.3, 0.5], 7)
    with pytest.raises(ValueError, match="time"):
        unit.log_prob(jnp.array([1], dtype=jnp.int32))
    with pytest.raises(ValueError, match="time"):
        pool.log_prob(jnp.array([2], dtype=jnp.int32))
    with pytest.raises(ValueError, match="time"):
        prefix_condition(unit, jnp.array([0], dtype=jnp.int32))
    with pytest.raises(ValueError, match="prefix"):
        prefix_condition(unit, jnp.zeros((3, 1), dtype=jnp.int32))
    with pytest.raises(ValueError, match="prefix"):
        prefix_condition(pool, jnp.zeros((3, 1), dtype=jnp.int32))


def test_float_counts_are_range_checked_before_any_integer_cast():
    with jax.enable_x64(True):
        law = CohortEventTime(
            jnp.zeros((1, 1)), jnp.array([-jnp.inf]), jnp.asarray([5], dtype=jnp.int32)
        )
        assert law.total_count.dtype == jnp.int32
        assert law.log_prob(jnp.array([[2.0**40]])) == -jnp.inf
        assert law.log_prob(jnp.array([[-(2.0**40)]])) == -jnp.inf
        assert law.log_prob(jnp.array([[5.0]])) == 0.0
        with pytest.raises(ValueError, match="prefix"):
            prefix_condition(law, jnp.array([[2.0**40]]))
    assert not jax.config.jax_enable_x64


@pytest.mark.parametrize("dtype", [jnp.uint8, jnp.uint16, jnp.uint32])
def test_unsigned_jax_pools_cannot_wrap_impossible_draws_into_positive_counts(dtype):
    def sample(total):
        law = CohortEventTime(jnp.full((1, 1), -jnp.inf), jnp.array([-jnp.inf]), total)
        return law.sample(random.PRNGKey(0))

    total = jnp.array([3], dtype)
    for execute in (sample, jax.jit(sample)):
        with pytest.raises(TypeError, match="signed"):
            execute(total)
    # Host integers are range-checked and converted to the signed native dtype.
    np.testing.assert_array_equal(sample(np.array([3], np.uint32)), [[-1]])


def test_tiny_expected_counts_keep_finite_log_mass():
    law = CohortEventTime(jnp.array([[-95.0]]), jnp.array([0.0]), jnp.asarray([1]))
    np.testing.assert_allclose(law.log_prob(jnp.array([[1]], dtype=jnp.int32)), -95.0, atol=1e-4)
    with jax.enable_x64(True):
        law = CohortEventTime(jnp.array([[-740.0]]), jnp.array([0.0]), np.array([1]))
        np.testing.assert_allclose(
            law.log_prob(jnp.array([[1]], dtype=jnp.int64)), -740.0, atol=1e-9
        )
        gradient = jax.grad(
            lambda w: CohortEventTime(w[:1, None], w[1:], np.array([1])).log_prob(
                jnp.array([[1]], dtype=jnp.int64)
            )
        )(jnp.array([-740.0, 0.0]))
        np.testing.assert_allclose(gradient, [1.0, -1.0], atol=1e-9)
    assert not jax.config.jax_enable_x64


def test_batched_laws_score_unbatched_and_sample_shaped_values():
    logits = jnp.array([SUSCEPTIBLE_LOGIT, 0.0, 2.0])
    hazard = jnp.full((3, 2, 2), LOG_HALF)
    kernel = SurvivalKernel(hazard, hazard, jnp.broadcast_to(logits[:, None], (3, 2)))
    compact = EventTime(kernel=kernel)
    explicit = EventTime(compact.log_mass, compact.log_tail)
    value = jnp.array([[1, 0], [0, 0]], dtype=jnp.int32)
    for law in (compact, explicit):
        assert law.batch_shape == (3,)
        assert law.event_shape == (2, 2)
        scores = law.unit_log_prob(value)
        assert scores.shape == (3, 2)
        pi = jax.nn.sigmoid(logits)
        np.testing.assert_allclose(scores[:, 0], jnp.log(pi * 0.5), rtol=1e-5)
        np.testing.assert_allclose(scores[:, 1], jnp.log(1 - pi + pi * 0.25), rtol=1e-5)
        assert law.log_prob(value).shape == (3,)
        stacked = jnp.broadcast_to(value, (5, 3, 2, 2))
        assert law.unit_log_prob(stacked).shape == (5, 3, 2)
        assert law.log_prob(stacked).shape == (5, 3)
        draws = law.sample(random.PRNGKey(0), (4,))
        assert draws.shape == (4, 3, 2, 2)
        assert jnp.issubdtype(draws.dtype, jnp.integer)
        assert bool(jnp.all(draws.sum(axis=-2) <= 1))
        assert bool(jnp.all((draws == 0) | (draws == 1)))
        assert law.mean.shape == (3, 2, 2)
        assert law.sample(random.PRNGKey(1)).shape == (3, 2, 2)


def test_unit_samples_follow_the_law_on_both_routes():
    kernel = _cure_kernel(0.4, days=3, units=2, closed=(1,))
    compact = EventTime(kernel=kernel)
    explicit = EventTime(compact.log_mass, compact.log_tail)
    draws = 20_000
    for law, seed in ((compact, 11), (explicit, 12)):
        samples = law.sample(random.PRNGKey(seed), (draws,))
        assert samples.shape == (draws, 3, 2)
        assert bool(jnp.all(samples.sum(axis=-2) <= 1))
        assert int(samples[:, 1, :].sum()) == 0
        frequency = samples.mean(axis=0)
        error = jnp.sqrt(law.mean * (1 - law.mean) / draws)
        assert bool(jnp.all(jnp.abs(frequency - law.mean) <= 5 * error + 1e-9))
        no_event = 1 - samples.sum(axis=-2).mean(axis=0)
        expected = jnp.exp(law.log_probabilities[-1])
        assert bool(
            jnp.all(jnp.abs(no_event - expected) <= 5 * jnp.sqrt(expected * (1 - expected) / draws))
        )


def test_empty_windows_and_full_prefixes_have_unit_mass():
    kernel = _cure_kernel(0.5, days=2, units=3)
    for law in (EventTime(kernel=kernel), _explicit([[0.2] * 3, [0.1] * 3], [0.7] * 3)):
        empty = slice_time(law, slice(0, 0))
        assert empty.event_shape == (0, 3)
        nothing = jnp.zeros((0, 3), dtype=jnp.int32)
        assert empty.log_prob(nothing) == 0.0
        np.testing.assert_array_equal(empty.unit_log_prob(nothing), jnp.zeros(3))
        assert empty.mean.shape == (0, 3)
        assert empty.sample(random.PRNGKey(0), (2,)).shape == (2, 0, 3)
        full = prefix_condition(law, jnp.zeros((2, 3), jnp.int32).at[0, 1].set(1))
        assert full.event_shape == (0, 3)
        assert full.log_prob(nothing) == 0.0
        assert full.sample(random.PRNGKey(0)).shape == (0, 3)
        assert prefix_condition(law, nothing).event_shape == (2, 3)
        np.testing.assert_allclose(prefix_condition(law, nothing).mean, law.mean, atol=1e-6)


def test_structural_zero_gradients_stay_finite():
    hazard = jnp.array([[math.log(0.3)], [-jnp.inf], [math.log(0.2)]])
    stay = jnp.array([[math.log(0.7)], [0.0], [math.log(0.8)]])
    logits = jnp.array([0.3])
    value = jnp.array([[0], [0], [1]], dtype=jnp.int32)

    def compact(h, s, susceptibility_logits):
        return EventTime(kernel=SurvivalKernel(h, s, susceptibility_logits)).log_prob(value)

    grads = jax.grad(compact, argnums=(0, 1, 2))(hazard, stay, logits)
    for gradient in grads:
        assert bool(jnp.all(jnp.isfinite(gradient)))
    assert grads[0][1, 0] == 0.0

    law = EventTime(kernel=SurvivalKernel(hazard, stay, logits))

    def explicit(log_mass, log_tail):
        return EventTime(log_mass, log_tail).log_prob(value)

    mass_gradient, tail_gradient = jax.grad(explicit, argnums=(0, 1))(law.log_mass, law.log_tail)
    assert bool(jnp.all(jnp.isfinite(mass_gradient))) and bool(jnp.all(jnp.isfinite(tail_gradient)))
    assert mass_gradient[1, 0] == 0.0
    censored = jnp.zeros((3, 1), dtype=jnp.int32)
    grads = jax.grad(
        lambda h, s, susceptibility_logits: EventTime(
            kernel=SurvivalKernel(h, s, susceptibility_logits)
        ).log_prob(censored),
        argnums=(0, 1, 2),
    )(hazard, stay, logits)
    for gradient in grads:
        assert bool(jnp.all(jnp.isfinite(gradient)))


# --------------------------------------------------------------- CohortEventTime


def _pool_law(probabilities, total):
    """Counted law with explicit category probabilities (last one is no-event)."""
    weights = jnp.log(jnp.asarray(probabilities))
    return CohortEventTime(weights[:-1, None], weights[-1:], jnp.asarray([total]))


def test_pool_moments_pmf_and_conditioning_at_seven_units():
    law = _pool_law([0.2, 0.3, 0.5], 7)
    assert law.event_shape == (2, 1)
    np.testing.assert_allclose(law.mean[:, 0], [1.4, 2.1], rtol=1e-6)
    np.testing.assert_allclose(law.variance[:, 0], [1.12, 1.47], rtol=1e-6)
    total = 0.0
    for x1 in range(8):
        for x2 in range(8 - x1):
            value = jnp.array([[x1], [x2]], dtype=jnp.int32)
            expected = _multinomial_log_pmf([x1, x2], 7, [0.2, 0.3, 0.5])
            np.testing.assert_allclose(law.log_prob(value), expected, rtol=2e-5)
            total += math.exp(float(law.log_prob(value)))
    assert abs(total - 1.0) < 1e-4
    assert law.log_prob(jnp.array([[4], [4]], dtype=jnp.int32)) == -jnp.inf
    assert law.log_prob(jnp.array([[-1], [2]], dtype=jnp.int32)) == -jnp.inf
    assert law.log_prob(jnp.array([[0.5], [2.0]])) == -jnp.inf

    future = prefix_condition(law, jnp.array([[2]], dtype=jnp.int32))
    assert int(future.total_count[0]) == 5
    np.testing.assert_allclose(jnp.exp(future.log_probabilities)[:, 0], [0.375, 0.625], rtol=1e-6)
    np.testing.assert_allclose(future.mean[:, 0], [1.875], rtol=1e-6)
    full = jnp.array([[2], [3]], dtype=jnp.int32)
    np.testing.assert_allclose(
        slice_time(law, slice(0, 1)).log_prob(full[:1]) + future.log_prob(full[1:]),
        law.log_prob(full),
        rtol=1e-5,
    )
    np.testing.assert_allclose(law.log_prob(full), math.log(0.0567), rtol=1e-5)


def test_pool_samples_reproduce_multinomial_covariance():
    law = _pool_law([0.2, 0.3, 0.5], 7)
    draws = 20_000
    samples = law.sample(random.PRNGKey(7), (draws,))
    assert samples.shape == (draws, 2, 1)
    assert jnp.issubdtype(samples.dtype, jnp.integer)
    counts = np.asarray(samples[:, :, 0], dtype=np.int64)
    assert counts.min() >= 0 and counts.sum(axis=1).max() <= 7
    mean = counts.mean(axis=0)
    z = (mean - np.array([1.4, 2.1])) / np.sqrt(np.array([1.12, 1.47]) / draws)
    assert np.abs(z).max() <= 5, f"sample means {mean} deviate by z = {z}"
    variance = counts.var(axis=0)
    assert np.all(np.abs(variance - [1.12, 1.47]) < 0.06)
    covariance = np.cov(counts.T)[0, 1]
    assert abs(covariance + 0.42) < 0.05


def test_single_unit_pool_is_the_unit_law():
    log_mass = jnp.log(jnp.array([[0.2, 0.0, 0.1], [0.0, 0.5, 0.2], [0.3, 0.0, 0.0]]))
    log_tail = jnp.log(jnp.array([0.5, 0.5, 0.7]))
    unit = EventTime(log_mass, log_tail)
    pool = CohortEventTime(log_mass, log_tail, jnp.ones(3, dtype=jnp.int32))
    np.testing.assert_allclose(pool.mean, unit.mean, atol=1e-6)
    np.testing.assert_allclose(pool.variance, unit.variance, atol=1e-6)
    for path in _paths(3, 3):
        unit_scores = unit.unit_log_prob(path)
        pool_scores = pool.cohort_log_prob(path)
        np.testing.assert_array_equal(jnp.isfinite(unit_scores), jnp.isfinite(pool_scores))
        finite = jnp.isfinite(unit_scores)
        np.testing.assert_allclose(pool_scores[finite], unit_scores[finite], atol=1e-5)
    prefix = jnp.array([[1, 0, 0]], dtype=jnp.int32)
    np.testing.assert_allclose(
        prefix_condition(pool, prefix).mean, prefix_condition(unit, prefix).mean, atol=1e-6
    )


def test_pool_samples_are_conserved_integers_with_closures_and_batches():
    log_mass = jnp.log(jnp.array([[0.3, 0.1], [0.0, 0.0], [0.4, 0.5]]))
    log_tail = jnp.log(jnp.array([0.3, 0.4]))
    totals = jnp.array([[5, 0], [7, 1000]], dtype=jnp.int32)
    law = CohortEventTime(log_mass, log_tail, totals)
    assert law.batch_shape == (2,)
    assert law.event_shape == (3, 2)
    samples = jax.jit(lambda key: law.sample(key, (50,)))(random.PRNGKey(2))
    assert samples.shape == (50, 2, 3, 2)
    assert jnp.issubdtype(samples.dtype, jnp.integer)
    assert bool(jnp.all(samples >= 0))
    assert bool(jnp.all(samples.sum(axis=-2) <= totals))
    assert int(samples[:, 0, :, 1].sum()) == 0
    assert int(samples[:, :, 1, :].sum()) == 0
    assert int(samples[:, 1, :, 1].sum()) > 0
    value = jnp.array([[1, 0], [0, 0], [2, 0]], dtype=jnp.int32)
    assert law.log_prob(value).shape == (2,)
    assert law.cohort_log_prob(value).shape == (2, 2)
    assert law.log_prob(jnp.broadcast_to(value, (4, 2, 3, 2))).shape == (4, 2)
    assert law.log_prob(jnp.array([[0, 0], [1, 0], [0, 0]], dtype=jnp.int32))[0] == -jnp.inf


def test_zero_population_absorbs_and_infeasible_prefixes_are_refused_or_impossible():
    law = _pool_law([0.2, 0.3, 0.5], 7)
    empty = CohortEventTime(law.log_mass, law.log_tail, jnp.zeros(1, dtype=jnp.int32))
    zeros = jnp.zeros((2, 1), dtype=jnp.int32)
    assert empty.log_prob(zeros) == 0.0
    assert empty.log_prob(zeros.at[0].set(1)) == -jnp.inf
    np.testing.assert_array_equal(empty.mean, 0.0)
    np.testing.assert_array_equal(empty.variance, 0.0)
    assert int(empty.sample(random.PRNGKey(0), (3,)).sum()) == 0
    exhausted = prefix_condition(law, jnp.array([[7]], dtype=jnp.int32))
    assert int(exhausted.total_count[0]) == 0
    assert exhausted.log_prob(jnp.zeros((1, 1), jnp.int32)) == 0.0

    with pytest.raises(ValueError, match="prefix"):
        prefix_condition(law, jnp.array([[8]], dtype=jnp.int32))
    with pytest.raises(ValueError, match="prefix"):
        prefix_condition(law, jnp.array([[1.5]]))
    with pytest.raises(ValueError, match="total_count"):
        CohortEventTime(law.log_mass, law.log_tail, np.array([-1]))
    with pytest.raises(TypeError, match="total_count"):
        CohortEventTime(law.log_mass, law.log_tail, jnp.asarray([2.0]))

    paired = CohortEventTime(
        jnp.concatenate([law.log_mass, law.log_mass], axis=-1),
        jnp.concatenate([law.log_tail, law.log_tail]),
        jnp.array([7, 7], dtype=jnp.int32),
    )

    @jax.jit
    def traced(prefix):
        future = prefix_condition(paired, prefix)
        suffix = jnp.zeros((1, 2), dtype=jnp.int32)
        return future.cohort_log_prob(suffix), future.mean, future.sample(random.PRNGKey(0))

    scores, mean, draw = traced(jnp.array([[8, 2]], dtype=jnp.int32))
    assert scores[0] == -jnp.inf
    np.testing.assert_allclose(scores[1], 5 * math.log(0.625), rtol=1e-5)
    assert bool(jnp.all(mean[:, 0] < 0))
    np.testing.assert_allclose(mean[:, 1], [1.875], rtol=1e-5)
    assert bool(jnp.all(draw[:, 0] < 0))
    assert 0 <= int(draw[0, 1]) <= 5


def test_window_marginal_moves_outside_events_into_no_event_mass():
    weights = jnp.log(jnp.array([0.1, 0.2, 0.3, 0.15, 0.25]))
    law = CohortEventTime(weights[:-1, None], weights[-1:], jnp.asarray([9]))
    window = slice_time(law, slice(1, 3))
    assert window.event_shape == (2, 1)
    np.testing.assert_allclose(jnp.exp(window.log_probabilities)[:, 0], [0.2, 0.3, 0.5], rtol=1e-6)
    assert int(window.total_count[0]) == 9
    np.testing.assert_allclose(
        window.log_prob(jnp.array([[2], [3]], dtype=jnp.int32)),
        _multinomial_log_pmf([2, 3], 9, [0.2, 0.3, 0.5]),
        rtol=2e-5,
    )
    with pytest.raises(NotImplementedError):
        slice_time(law, slice(0, 4, 2))


def test_int64_pools_keep_odd_counts_under_local_x64():
    quantity = 2**53 + 1
    with pytest.raises(ValueError, match="x64"):
        CohortEventTime(jnp.zeros((1, 1)), jnp.array([-50.0]), np.array([quantity]))
    with jax.enable_x64(True):
        near_certain = CohortEventTime(jnp.zeros((1, 1)), jnp.array([-50.0]), np.array([quantity]))
        assert near_certain.total_count.dtype == jnp.int64
        draws = near_certain.sample(random.PRNGKey(0), (4,))
        assert draws.dtype == jnp.int64
        assert [int(x) for x in draws[:, 0, 0]] == [quantity] * 4
        assert near_certain.log_prob(jnp.array([[quantity]], dtype=jnp.int64)) > -1e-5

        certain = CohortEventTime(jnp.zeros((1, 1)), jnp.array([-jnp.inf]), np.array([quantity]))
        assert certain.log_prob(jnp.array([[quantity]], dtype=jnp.int64)) == 0.0
        assert certain.log_prob(jnp.array([[quantity - 1]], dtype=jnp.int64)) == -jnp.inf

        two_days = CohortEventTime(jnp.zeros((2, 1)), jnp.array([-50.0]), np.array([quantity]))
        remaining = prefix_condition(two_days, jnp.array([[quantity - 2]], dtype=jnp.int64))
        assert int(remaining.total_count[0]) == 2
        assert [int(x) for x in remaining.sample(random.PRNGKey(5), (3,))[:, 0, 0]] == [2] * 3

        rare = CohortEventTime(
            jnp.array([[math.log(1.0 / quantity)]]),
            jnp.array([math.log1p(-1.0 / quantity)]),
            np.array([quantity]),
        )
        np.testing.assert_allclose(
            rare.log_prob(jnp.array([[1]], dtype=jnp.int64)), -0.9999999999999992, atol=1e-9
        )
        np.testing.assert_allclose(
            rare.log_prob(jnp.array([[0]], dtype=jnp.int64)), -1.0000000000000002, atol=1e-9
        )

        maximum = np.iinfo(np.int64).max
        wide = CohortEventTime(jnp.zeros((3, 1)), jnp.array([-jnp.inf]), np.array([maximum]))
        assert wide.log_prob(jnp.array([[maximum], [maximum], [2]], dtype=jnp.int64)) == -jnp.inf
    assert not jax.config.jax_enable_x64


def test_log_pmf_and_gradient_stay_exact_at_huge_counts():
    with jax.enable_x64(True):
        total = np.array([2**62])
        offset = 10**9
        value = jnp.array([[2**61 + offset]], dtype=jnp.int64)

        def log_prob(weights):
            return CohortEventTime(weights[:1, None], weights[1:], total).log_prob(value)

        weights = jnp.zeros(2)
        np.testing.assert_allclose(log_prob(weights), -22.147034818997234, atol=1e-7)
        gradient = jax.grad(log_prob)(weights)
        np.testing.assert_allclose(gradient, [offset, -offset], rtol=1e-9)

        small = jnp.log(jnp.array([0.25, 0.35, 0.4]))
        counts = jnp.array([[3], [5]], dtype=jnp.int64)

        def small_log_prob(w):
            return CohortEventTime(w[:-1, None], w[-1:], np.array([11])).log_prob(counts)

        analytic = jax.grad(small_log_prob)(small)
        step = 1e-6
        for k in range(3):
            unit = jnp.zeros(3).at[k].set(step)
            numeric = (small_log_prob(small + unit) - small_log_prob(small - unit)) / (2 * step)
            np.testing.assert_allclose(analytic[k], numeric, atol=1e-6)
        # The gradient with respect to the raw weights is the residual x - n p
        # under the softmax normalization.
        p = jnp.exp(small - jax.scipy.special.logsumexp(small))
        np.testing.assert_allclose(analytic, jnp.array([3, 5, 3]) - 11 * p, atol=1e-9)
    assert not jax.config.jax_enable_x64


def test_ten_million_unit_pool_samples_and_scores_at_default_precision():
    law = _pool_law([0.2, 0.3, 0.5], 10_000_000)
    samples = law.sample(random.PRNGKey(9), (2000,))
    assert jnp.issubdtype(samples.dtype, jnp.integer)
    counts = np.asarray(samples[:, :, 0], dtype=np.int64)
    assert counts.min() >= 0 and counts.sum(axis=1).max() <= 10_000_000
    expected = np.array([2_000_000.0, 3_000_000.0])
    standard_error = np.sqrt(np.array([1.6e6, 2.1e6]) / 2000)
    assert np.all(np.abs(counts.mean(axis=0) - expected) < 5 * standard_error)
    gradient = jax.grad(
        lambda w: CohortEventTime(w[:-1, None], w[-1:], np.array([10_000_000])).log_prob(
            jnp.array([[1_999_000], [3_002_000]], dtype=jnp.int32)
        )
    )(jnp.log(jnp.array([0.2, 0.3, 0.5])))
    assert bool(jnp.all(jnp.isfinite(gradient)))
    np.testing.assert_allclose(gradient, [-1000.0, 2000.0, -1000.0], rtol=2e-3)


def _binomial_pmf(trials, probability):
    """Exact Binomial(trials, probability) mass over 0..trials from integer combinations."""
    return np.array(
        [
            math.comb(trials, k) * probability**k * (1 - probability) ** (trials - k)
            for k in range(trials + 1)
        ]
    )


def test_pool_marginals_follow_exact_binomial_frequencies():
    # Each date's marginal of Multinomial(40, (0.3, 0.2, 0.5)) is binomial, as is
    # the residual; a date at expected count 8 and one near 12 of the remaining
    # pool exercise both small- and large-mean sampling regimes.
    trials, draws = 40, 40_000
    law = _pool_law([0.3, 0.2, 0.5], trials)
    samples = np.asarray(law.sample(random.PRNGKey(17), (draws,))[:, :, 0], dtype=np.int64)
    assert samples.min() >= 0 and samples.sum(axis=1).max() <= trials
    residual = trials - samples.sum(axis=1)
    for counts, probability in ((samples[:, 0], 0.3), (samples[:, 1], 0.2), (residual, 0.5)):
        pmf = _binomial_pmf(trials, probability)
        frequency = np.bincount(counts, minlength=trials + 1) / draws
        resolved = pmf * draws >= 25
        error = np.sqrt(pmf * (1 - pmf) / draws)
        z = ((frequency - pmf) / error)[resolved]
        assert np.abs(z).max() <= 5, f"p = {probability}: cell deviations {z}"
        rest, rest_expected = frequency[~resolved].sum(), pmf[~resolved].sum()
        rest_error = math.sqrt(rest_expected * (1 - rest_expected) / draws)
        assert abs(rest - rest_expected) <= 5 * rest_error


THIRTEEN = (0.005, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.225)


def test_thirteen_category_pool_reproduces_every_multinomial_moment():
    trials, draws = 1000, 4000
    probabilities = np.array(THIRTEEN)
    law = _pool_law(THIRTEEN, trials)
    assert law.event_shape == (12, 1)
    samples = np.asarray(law.sample(random.PRNGKey(13), (draws,))[:, :, 0], dtype=np.int64)
    assert samples.min() >= 0 and samples.sum(axis=1).max() <= trials
    occupancy = np.concatenate([samples, trials - samples.sum(axis=1, keepdims=True)], axis=1)
    mean = trials * probabilities
    variance = mean * (1 - probabilities)
    covariance = -trials * np.outer(probabilities, probabilities)
    z_mean = (occupancy.mean(axis=0) - mean) / np.sqrt(variance / draws)
    assert np.abs(z_mean).max() <= 6, z_mean
    excess_kurtosis = (1 - 6 * probabilities * (1 - probabilities)) / variance
    z_variance = (occupancy.var(axis=0) - variance) / (
        variance * np.sqrt((2 + excess_kurtosis) / draws)
    )
    assert np.abs(z_variance).max() <= 6, z_variance
    off_diagonal = ~np.eye(len(THIRTEEN), dtype=bool)
    sample_covariance = np.cov(occupancy.T, bias=True)
    error = np.sqrt((np.outer(variance, variance) + covariance**2) / draws)
    z_covariance = ((sample_covariance - covariance) / error)[off_diagonal]
    assert np.abs(z_covariance).max() <= 6, z_covariance


def test_year_long_pool_allocations_respect_closures_and_date_masses():
    # 365 dates with weekend closures and a sawtooth date weight: every date has
    # a distinct neighbour, so any misrouting between dates or into the no-event
    # category moves a per-date mean by many standard errors.
    days, draws = 365, 600
    day = np.arange(days)
    open_day = (day % 7) < 5
    weight = np.where(open_day, 1.0 + day % 11, 0.0)
    probabilities = 0.7 * weight / weight.sum()
    pools = np.array([0, 1, 2, 3, 5, 8, 13, 21, 50, 300, 1000], dtype=np.int32)
    log_mass = np.where(open_day, np.log(np.where(open_day, probabilities, 1.0)), -np.inf)
    law = CohortEventTime(
        jnp.broadcast_to(jnp.asarray(log_mass)[:, None], (days, len(pools))),
        jnp.full((len(pools),), math.log(0.3)),
        pools,
    )
    samples = law.sample(random.PRNGKey(23), (draws,))
    assert samples.shape == (draws, days, len(pools))
    assert jnp.issubdtype(samples.dtype, jnp.integer)
    counts = np.asarray(samples, dtype=np.int64)
    assert counts.min() >= 0
    assert np.all(counts.sum(axis=1) <= pools)
    assert counts[:, :, 0].sum() == 0
    assert counts[:, ~open_day, :].sum() == 0
    population = pools.sum()
    per_date = counts.sum(axis=2).mean(axis=0)[open_day]
    expected = population * probabilities[open_day]
    z_date = (per_date - expected) / np.sqrt(expected * (1 - probabilities[open_day]) / draws)
    assert np.abs(z_date).max() <= 6, z_date
    total = counts.sum(axis=(1, 2)).mean()
    z_total = (total - 0.7 * population) / math.sqrt(0.7 * 0.3 * population / draws)
    assert abs(z_total) <= 6, z_total


def test_vmapped_pool_draws_equal_individual_draws():
    law = CohortEventTime(
        jnp.log(jnp.array([[0.2, 0.45], [0.3, 0.45]])),
        jnp.log(jnp.array([0.5, 0.1])),
        jnp.array([7, 1000], dtype=jnp.int32),
    )
    keys = random.split(random.PRNGKey(29), 6)
    batched = jax.vmap(law.sample)(keys)
    single = jnp.stack([law.sample(key) for key in keys])
    assert batched.shape == (6, 2, 2)
    np.testing.assert_array_equal(batched, single)
    assert int(single[:, 0, 1].min()) < int(single[:, 0, 1].max())


@pytest.mark.parametrize("quantity", [2**32 + 1, 2**32 + 2, -(2**32) + 1])
def test_host_observations_never_wrap_into_feasible_trajectories(quantity):
    laws = [_explicit([[0.2], [0.3]], [0.5]), _pool_law([0.2, 0.3, 0.5], 7)]
    observed = np.array([[quantity], [0]], dtype=np.int64)
    for law in laws:
        with pytest.raises(ValueError, match="range|precision"):
            law.log_prob(observed)
        with pytest.raises(ValueError, match="range|precision"):
            prefix_condition(law, observed[:1])
        with jax.enable_x64():
            assert bool(jnp.isneginf(law.log_prob(observed)))
