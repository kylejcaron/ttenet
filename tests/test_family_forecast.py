"""Forecasts and eventual expectations replayed through fitted event-time families."""

import functools

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import pytest
from jax.nn import log_sigmoid

from ttenet.data import prepare_history
from ttenet.dates import date_grid
from ttenet.event_times import EventLaw, TimingLaw, log1mexp, timing_from_log_masses
from ttenet.families import EventFamily, FiniteTail, ProperTail, UnknownTail, WeibullFamily
from ttenet.forecast import expected_return_receipts, forecast_events, forecast_returns
from ttenet.integration import SalesForecast
from ttenet.models import StageFit, StageObservations, fit_stage
from ttenet.survival import StageParameters

AS_OF = np.datetime64("2026-03-01", "D")
NAT = np.datetime64("NaT", "D")


def _logit(p):
    return float(np.log(p / (1.0 - p)))


def _params(k=5, hazard=0.5, susceptibility=0.5, draws=None):
    fields = (
        np.full(k, _logit(hazard)),
        np.zeros(0),
        np.array(_logit(susceptibility)),
        np.zeros(0),
    )
    if draws is not None:
        fields = tuple(np.broadcast_to(f, (draws,) + f.shape) for f in fields)
    return StageParameters(*fields)


def _history(rows, policy_days=90):
    return prepare_history(pd.DataFrame(rows), as_of=AS_OF, policy_days=policy_days)


def _calendar(start="2025-01-01", horizon=120):
    return date_grid(np.datetime64(start, "D"), AS_OF + np.timedelta64(horizon, "D"))


def _fit(family, parameters, draws, feature_widths=(0, 0)):
    return StageFit(
        parameters=parameters,
        losses=np.zeros(0),
        family=family,
        num_samples=draws,
        feature_widths=feature_widths,
    )


def _constant_model(inputs, shared):
    """Constant-hazard family whose named sites are replayed from the fit."""
    hazard_logit = numpyro.sample("hazard_logit", dist.Normal(0.0, 1.0))
    susceptibility_logit = numpyro.sample("susceptibility_logit", dist.Normal(0.0, 1.0))
    logits = jnp.broadcast_to(hazard_logit, inputs.ages.shape)
    timing = TimingLaw(log_sigmoid(logits), log_sigmoid(-logits))
    return EventLaw(timing, jnp.broadcast_to(susceptibility_logit, inputs.ages.shape[-1:]))


_constant_family = EventFamily(_constant_model, ProperTail())


def _fixed_constant_family(hazard, susceptibility):
    """Zero-latent constant-hazard family: no sample or param sites at all."""

    def model(inputs, shared):
        logits = jnp.full(inputs.ages.shape, _logit(hazard))
        timing = TimingLaw(log_sigmoid(logits), log_sigmoid(-logits))
        return EventLaw(timing, jnp.full(inputs.ages.shape[-1:], _logit(susceptibility)))

    return EventFamily(model, ProperTail())


def _weibull_family(scale, shape, susceptibility):
    """Discretized Weibull: log S(a) = -(a / scale) ** shape on the [a, a + 1) bins."""

    def model(inputs, shared):
        ages = jnp.maximum(inputs.ages, 0).astype(jnp.float64)
        stay = -(((ages + 1.0) / scale) ** shape) + (ages / scale) ** shape
        return EventLaw(
            TimingLaw(log1mexp(stay), stay), jnp.full(ages.shape[-1:], _logit(susceptibility))
        )

    return EventFamily(model, ProperTail())


def _grid_family(masses, atom, susceptibility, tail):
    """Discrete susceptible masses on ages 0..A-1 plus a beyond-grid atom."""

    def model(inputs, shared):
        timing = timing_from_log_masses(
            jnp.log(jnp.asarray(masses, dtype=jnp.float64)), jnp.log(jnp.float64(atom)), inputs.ages
        )
        return EventLaw(timing, jnp.full(inputs.ages.shape[-1:], _logit(susceptibility)))

    return EventFamily(model, tail)


def _future_arrivals(draws, horizon, cohorts, day, cohort, count):
    arrivals = np.zeros((draws, horizon, cohorts), dtype=np.int64)
    arrivals[:, day - 1, cohort] = count
    return arrivals


# --------------------------------------------------------------------------
# forecast_events through StageFit replay
# --------------------------------------------------------------------------


def test_default_stage_fit_replays_identically_to_stage_parameters():
    draws = 3
    parameters = _params(hazard=0.3, susceptibility=0.7, draws=draws)
    fit = StageFit(parameters=parameters, losses=np.zeros(5))
    kwargs = dict(
        origins=[AS_OF - np.timedelta64(2, "D"), NAT],
        observed=[NAT, NAT],
        arrivals=_future_arrivals(draws, 6, 2, day=2, cohort=1, count=400),
        calendar=_calendar(),
        as_of=AS_OF,
        horizon=6,
        seed=11,
    )
    from_parameters = forecast_events(parameters, **kwargs)
    from_fit = forecast_events(fit, **kwargs)
    np.testing.assert_array_equal(from_fit.events, from_parameters.events)
    np.testing.assert_array_equal(from_fit.pending, from_parameters.pending)
    np.testing.assert_array_equal(from_fit.eligible, from_parameters.eligible)
    assert 0 < from_fit.events[:, 1:, 1].sum() < 3 * 400


def test_custom_family_replays_each_draws_named_sites():
    # Draw 0: certain hazard and susceptibility; draw 1: zero hazard; draw 2:
    # certain cure. The same ten arriving units must fire on their arrival
    # day in draw 0 only, so each draw sees exactly its own posterior row.
    posterior = {
        "hazard_logit": jnp.array([50.0, -50.0, 50.0]),
        "susceptibility_logit": jnp.array([50.0, 50.0, -50.0]),
    }
    fit = _fit(_constant_family, posterior, 3)
    result = forecast_events(
        fit,
        origins=[NAT],
        observed=[NAT],
        arrivals=_future_arrivals(3, 4, 1, day=2, cohort=0, count=10),
        calendar=_calendar(),
        as_of=AS_OF,
        horizon=4,
        seed=0,
    )
    np.testing.assert_array_equal(
        result.events[:, :, 0], [[0, 10, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    )
    np.testing.assert_array_equal(result.pending, [[0, 0, 0, 0], [0, 10, 10, 10], [0, 10, 10, 10]])
    np.testing.assert_array_equal(result.eligible, result.pending)


def test_zero_latent_family_replays_the_requested_draw_count_and_seed():
    fit = _fit(_weibull_family(scale=3.0, shape=2.0, susceptibility=0.8), {}, 4)
    kwargs = dict(
        origins=[NAT],
        observed=[NAT],
        arrivals=_future_arrivals(1, 5, 1, day=1, cohort=0, count=1000),
        calendar=_calendar(),
        as_of=AS_OF,
        horizon=5,
    )
    first = forecast_events(fit, seed=3, **kwargs)
    replay = forecast_events(fit, seed=3, **kwargs)
    other = forecast_events(fit, seed=4, **kwargs)
    assert first.events.shape == (4, 5, 1)
    np.testing.assert_array_equal(first.events, replay.events)
    assert not np.array_equal(first.events, other.events)
    assert np.all(first.events.sum(axis=1) <= 1000)
    # Weibull shape 2 at scale 3: the age-0 bin fires with 1 - exp(-1/9).
    expected = 1000 * 0.8 * -np.expm1(-1.0 / 9.0)
    assert abs(first.events[:, 0, 0].mean() - expected) < 25
    # Draw alignment follows the fit's draw count, not the parameter arrays.
    with pytest.raises(ValueError, match="align"):
        forecast_events(fit, seed=0, **{**kwargs, "arrivals": np.ones((5, 5, 1), dtype=np.int64)})


def test_mass_grid_family_conditions_history_and_times_future_events():
    # Susceptible units fire at age one with certainty and never at age zero.
    # A unit already at age one with no event is therefore known to be cured,
    # while a fresh 1000-unit cohort fires at age one with its prior
    # susceptibility 1/2 and nowhere else.
    fit = _fit(_grid_family([0.0, 1.0], 0.0, 0.5, FiniteTail(last_age=1)), {}, 1)
    result = forecast_events(
        fit,
        origins=[AS_OF - np.timedelta64(1, "D"), NAT],
        observed=[NAT, NAT],
        arrivals=_future_arrivals(1, 4, 2, day=1, cohort=1, count=1000),
        calendar=_calendar(),
        as_of=AS_OF,
        horizon=4,
        seed=5,
    )
    np.testing.assert_array_equal(result.events[0, :, 0], 0)
    assert result.events[0, 0, 1] == 0
    assert 420 < result.events[0, 1, 1] < 580
    np.testing.assert_array_equal(result.events[0, 2:, 1], 0)
    np.testing.assert_array_equal(result.pending[0], 1 + 1000 - np.cumsum(result.events[0, :, 1]))


def test_weibull_family_respects_closures_and_deadlines_in_the_shared_kernel():
    fit = _fit(_weibull_family(scale=2.0, shape=2.0, susceptibility=1 - 1e-9), {}, 2)
    calendar = _calendar()
    allowed = np.ones(calendar.size, dtype=bool)
    allowed[int(np.flatnonzero(calendar == AS_OF + np.timedelta64(2, "D"))[0])] = False
    result = forecast_events(
        fit,
        origins=[NAT],
        observed=[NAT],
        arrivals=_future_arrivals(2, 6, 1, day=1, cohort=0, count=2000),
        calendar=calendar,
        as_of=AS_OF,
        horizon=6,
        allowed=allowed,
        deadline_days=2,
        seed=0,
    )
    assert np.all(result.events[:, 0, 0] > 0)
    np.testing.assert_array_equal(result.events[:, 1, 0], 0)  # closed calendar day
    assert np.all(result.events[:, 2, 0] > 0)
    np.testing.assert_array_equal(result.events[:, 3:, 0], 0)  # past the inclusive deadline
    np.testing.assert_array_equal(result.eligible[:, 3:], 0)
    np.testing.assert_array_equal(result.pending[:, 3:], result.pending[:, 2:3].repeat(3, axis=1))


def test_forecast_events_replays_a_hand_built_fit_only_with_its_declared_widths():
    # Declared with one timing and one susceptibility regressor, the packaged
    # Weibull replays only forecasts that supply regressors of exactly those
    # widths. Omitting either would silently drop a fitted effect, so it is
    # refused; matching widths replay the fit.
    family = WeibullFamily(scale_prior=dist.LogNormal(1.0, 0.3), susceptibility_logit_prior=0.5)
    posterior = {
        "scale": jnp.array([3.0, 4.0]),
        "beta": jnp.array([[0.5], [-0.5]]),
        "susceptibility_beta": jnp.array([[1.0], [1.0]]),
    }
    fit = _fit(family, posterior, 2, feature_widths=(1, 1))
    calendar = _calendar()
    kwargs = dict(
        origins=[NAT],
        observed=[NAT],
        arrivals=_future_arrivals(2, 4, 1, day=1, cohort=0, count=1000),
        calendar=calendar,
        as_of=AS_OF,
        horizon=4,
    )
    features = np.zeros((1, calendar.size, 1))
    susceptibility_features = np.zeros((1, 1))
    with pytest.raises(ValueError):
        forecast_events(fit, **kwargs)
    with pytest.raises(ValueError):
        forecast_events(fit, features=features, **kwargs)
    with pytest.raises(ValueError):
        forecast_events(fit, susceptibility_features=susceptibility_features, **kwargs)
    result = forecast_events(
        fit, features=features, susceptibility_features=susceptibility_features, **kwargs
    )
    # Zero regressors leave each draw's own law: the age-0 bin fires with
    # sigmoid(.5) * (1 - exp(-1 / scale**2)) per arriving unit.
    expected = 1000 * (1 / (1 + np.exp(-0.5))) * -np.expm1(-1.0 / np.array([9.0, 16.0]))
    assert result.events.shape == (2, 4, 1)
    np.testing.assert_allclose(result.events[:, 0, 0], expected, atol=4 * np.sqrt(expected.max()))


def test_forecast_returns_runs_mixed_families_with_count_draws():
    history = _history(
        {
            "item_id": ["open", "elig"],
            "sale_date": ["2026-02-01", "2026-02-20"],
            "initiation_date": ["2026-02-27", None],
            "receipt_date": [None, None],
        }
    )
    receipt = _fit(_weibull_family(scale=4.0, shape=2.0, susceptibility=0.6), {}, 3)
    future = SalesForecast(
        cohorts=pd.DataFrame(
            {"item_id": ["f"], "sale_date": [str(AS_OF + np.timedelta64(1, "D"))]}
        ),
        counts=np.array([[30], [60], [0]]),
    )
    result = forecast_returns(
        history,
        _params(hazard=0.4, susceptibility=0.9),
        receipt,
        calendar=_calendar(),
        horizon=12,
        future_sales=future,
        seed=2,
    )
    assert result.initiations.shape == result.receipts.shape == (3, 12)
    assert np.all(result.initiations.sum(axis=1) <= 1 + future.counts[:, 0])
    assert np.all(result.receipts.sum(axis=1) <= 2 + future.counts[:, 0])
    np.testing.assert_array_equal(
        result.open_returns + result.receipts.cumsum(axis=1),
        1 + result.initiations.cumsum(axis=1),
    )
    assert result.expected_existing_receipts.shape == (3,)
    assert np.all(result.expected_open_receipts > 0)


# --------------------------------------------------------------------------
# Eventual expectations under declared tail behavior
# --------------------------------------------------------------------------


def test_custom_proper_receipt_open_return_uses_conditional_susceptibility():
    # h=.5, pi=.4, two exposed no-event dates: .4*.25/(.6+.4*.25) = 1/7 per draw.
    history = _history(
        {
            "item_id": ["open"],
            "sale_date": ["2026-01-01"],
            "initiation_date": [str(AS_OF - np.timedelta64(1, "D"))],
            "receipt_date": [None],
        }
    )
    receipt = _fit(_fixed_constant_family(hazard=0.5, susceptibility=0.4), {}, 3)
    uninitiated, open_receipts = expected_return_receipts(
        history, _params(), receipt, calendar=_calendar()
    )
    np.testing.assert_allclose(open_receipts, np.full(3, 1.0 / 7.0), rtol=1e-9)
    np.testing.assert_allclose(uninitiated, np.zeros(3))


def test_finite_receipt_family_integrates_through_last_age_with_closures():
    # Masses .5/.5 on ages 0/1, pi=.8. An open return at age 0 that did not
    # fire has conditional susceptibility .8*.5/(.2+.4) = 2/3; age 1 is the
    # last supported age and fires with certainty, unless that date is closed.
    history = _history(
        {
            "item_id": ["open"],
            "sale_date": ["2026-01-01"],
            "initiation_date": [str(AS_OF)],
            "receipt_date": [None],
        }
    )
    receipt = _fit(_grid_family([0.5, 0.5], 0.0, 0.8, FiniteTail(last_age=1)), {}, 2)
    calendar = _calendar()
    _, open_receipts = expected_return_receipts(history, _params(), receipt, calendar=calendar)
    np.testing.assert_allclose(open_receipts, [2.0 / 3.0] * 2, rtol=1e-9)
    closed = np.ones((1, calendar.size), dtype=bool)
    closed[0, int(np.flatnonzero(calendar == AS_OF + np.timedelta64(1, "D"))[0])] = False
    _, closed_receipts = expected_return_receipts(
        history, _params(), receipt, calendar=calendar, receipt_allowed=closed
    )
    np.testing.assert_allclose(closed_receipts, [0.0, 0.0], atol=1e-12)


def test_finite_receipt_contraction_streams_over_actual_initiation_days():
    # Certainly susceptible initiation at h=.5 with a two-day policy from a
    # sale on as_of: initiation on day 1 w.p. .5, day 2 w.p. .25. Receipt
    # fires on the initiation day itself (single age-0 mass) with pi=.5, so a
    # receiving closure on day 1 removes the day-1 initiations' receipts.
    history = _history({"item_id": ["sold"], "sale_date": [str(AS_OF)]}, policy_days=2)
    initiation = StageParameters(np.zeros(1), np.zeros(0), np.array(50.0), np.zeros(0))
    receipt = _fit(_grid_family([1.0], 0.0, 0.5, FiniteTail(last_age=0)), {}, 1)
    calendar = _calendar()
    uninitiated, _ = expected_return_receipts(history, initiation, receipt, calendar=calendar)
    np.testing.assert_allclose(uninitiated, [0.75 * 0.5], rtol=1e-9)
    closed = np.ones((1, calendar.size), dtype=bool)
    closed[0, int(np.flatnonzero(calendar == AS_OF + np.timedelta64(1, "D"))[0])] = False
    uninitiated, _ = expected_return_receipts(
        history, initiation, receipt, calendar=calendar, receipt_allowed=closed
    )
    np.testing.assert_allclose(uninitiated, [0.25 * 0.5], rtol=1e-9)


def test_finite_initiation_family_bounds_an_unbounded_policy_window():
    # No policy deadline, but the initiation law has no mass past age 1 and
    # fires there with certainty for susceptible units, so the eventual
    # initiation probability is the conditional susceptibility times one.
    history = _history({"item_id": ["sold"], "sale_date": [str(AS_OF)]}, policy_days=None)
    initiation = _fit(_grid_family([0.5, 0.5], 0.0, 1 - 1e-12, FiniteTail(last_age=1)), {}, 1)
    calendar = _calendar()
    uninitiated, _ = expected_return_receipts(
        history, initiation, _params(susceptibility=0.5), calendar=calendar
    )
    np.testing.assert_allclose(uninitiated, [0.5], rtol=1e-9)
    # Known finite initiation support still needs the calendar's actual
    # future closures and features through its last age.
    with pytest.raises(ValueError, match="calendar"):
        expected_return_receipts(
            history, initiation, _params(susceptibility=0.5), calendar=_calendar(horizon=0)
        )


def test_finite_receipt_beyond_the_calendar_uses_the_stated_continuation():
    # Proper initiation (h=.5, certainly susceptible, no deadline) from a sale
    # on as_of with a calendar ending two days later: initiation on day 1
    # w.p. .5, day 2 w.p. .25, after the calendar w.p. .25. Receipt masses
    # .5/.5 on ages 0/1 with pi=.5 and a receiving closure on day 2 only.
    # Day-1 initiations: age 0 open (h=.5), age 1 closed -> .5*.5 = .25.
    # Day-2 initiations: age 0 closed (its mass is lost, the clock runs on),
    # age 1 falls past the calendar, which continues all-open with the last
    # features held, and fires with certainty -> .5*1 = .5. Post-calendar
    # initiations complete under that stationary continuation with the full
    # .5*(.5+.5) = .5. Total .5*.25 + .25*.5 + .25*.5 = .375.
    history = _history({"item_id": ["sold"], "sale_date": [str(AS_OF)]}, policy_days=None)
    initiation = StageParameters(np.zeros(1), np.zeros(0), np.array(50.0), np.zeros(0))
    receipt = _fit(_grid_family([0.5, 0.5], 0.0, 0.5, FiniteTail(last_age=1)), {}, 2)
    calendar = _calendar(horizon=2)
    closed = np.ones((1, calendar.size), dtype=bool)
    closed[0, int(np.flatnonzero(calendar == AS_OF + np.timedelta64(2, "D"))[0])] = False
    uninitiated, _ = expected_return_receipts(
        history, initiation, receipt, calendar=calendar, receipt_allowed=closed
    )
    np.testing.assert_allclose(uninitiated, [0.375] * 2, rtol=1e-9)
    uninitiated, _ = expected_return_receipts(history, initiation, receipt, calendar=calendar)
    np.testing.assert_allclose(uninitiated, [0.5] * 2, rtol=1e-9)


def test_finite_open_receipt_window_past_the_calendar_continues_all_open():
    # Open return initiated on as_of under receipt masses .5/.5 (pi=.8): the
    # conditional susceptibility after the age-0 miss is 2/3 and age 1 falls
    # one day past a calendar that ends on as_of, where exposure continues.
    history = _history(
        {
            "item_id": ["open"],
            "sale_date": ["2026-01-01"],
            "initiation_date": [str(AS_OF)],
            "receipt_date": [None],
        }
    )
    receipt = _fit(_grid_family([0.5, 0.5], 0.0, 0.8, FiniteTail(last_age=1)), {}, 1)
    _, open_receipts = expected_return_receipts(
        history, _params(), receipt, calendar=_calendar(horizon=0)
    )
    np.testing.assert_allclose(open_receipts, [2.0 / 3.0], rtol=1e-9)


def test_unknown_tail_refuses_eventual_expectations_but_not_finite_horizon_forecasts():
    history = _history(
        {
            "item_id": ["open"],
            "sale_date": ["2026-01-01"],
            "initiation_date": [str(AS_OF - np.timedelta64(1, "D"))],
            "receipt_date": [None],
        }
    )
    receipt = _fit(_grid_family([0.3, 0.3], 0.4, 0.5, UnknownTail()), {}, 2)
    with pytest.raises(ValueError, match="eventual"):
        expected_return_receipts(history, _params(), receipt, calendar=_calendar())
    with pytest.raises(ValueError, match="eventual"):
        forecast_returns(history, _params(), receipt, calendar=_calendar(), horizon=3)
    result = forecast_events(
        receipt,
        origins=[AS_OF - np.timedelta64(1, "D"), NAT],
        observed=[NAT, NAT],
        arrivals=_future_arrivals(2, 3, 2, day=1, cohort=1, count=50),
        calendar=_calendar(),
        as_of=AS_OF,
        horizon=3,
    )
    assert result.events.shape == (2, 3, 2)
    assert np.all(result.events.sum(axis=1) <= [1, 50])


def test_post_calendar_receipt_atom_does_not_reuse_the_final_closure():
    history = _history({"item_id": ["sold"], "sale_date": [str(AS_OF)]}, policy_days=None)
    initiation = StageParameters(np.zeros(1), np.zeros(0), np.array(50.0), np.zeros(0))
    receipt = _fit(_grid_family([1.0], 0.0, 0.5, FiniteTail(last_age=0)), {}, 1)
    calendar = _calendar(horizon=2)
    closed = np.ones((1, calendar.size), dtype=bool)
    closed[:, -1] = False
    uninitiated, _ = expected_return_receipts(
        history, initiation, receipt, calendar=calendar, receipt_allowed=closed
    )
    # Day 1 contributes .5*.5, day 2 is closed, and the .25 initiation
    # probability after the calendar has all-open receipt probability .5.
    np.testing.assert_allclose(uninitiated, [0.375], rtol=1e-9)


@pytest.mark.parametrize("has_history", [False, True])
def test_singleton_fits_expand_expectations_to_sales_scenario_draws(has_history):
    history = _history(
        {
            "item_id": ["open"] if has_history else [],
            "sale_date": [str(AS_OF - np.timedelta64(2, "D"))] if has_history else [],
            "initiation_date": [str(AS_OF - np.timedelta64(1, "D"))] if has_history else [],
        }
    )
    future = SalesForecast(
        pd.DataFrame({"item_id": ["new"], "sale_date": [str(AS_OF + np.timedelta64(1, "D"))]}),
        np.array([[1], [2], [3]]),
    )
    result = forecast_returns(
        history,
        _params(susceptibility=0.4),
        _params(susceptibility=0.4),
        future_sales=future,
        calendar=_calendar(),
        horizon=2,
    )
    expected = np.full(3, 1 / 7 if has_history else 0)
    np.testing.assert_array_equal(result.expected_uninitiated_receipts, np.zeros(3))
    np.testing.assert_allclose(result.expected_open_receipts, expected, rtol=1e-9)
    np.testing.assert_allclose(result.expected_existing_receipts, expected, rtol=1e-9)


def test_one_unit_pools_absorb_absent_draws_and_keep_multiple_parent_dates():
    fit = _fit(
        _constant_family,
        {"hazard_logit": jnp.full(3, 50.0), "susceptibility_logit": jnp.full(3, 50.0)},
        3,
    )
    arrivals = np.array([[[1], [1], [0]], [[0], [0], [0]], [[0], [0], [1]]])
    result = forecast_events(
        fit,
        origins=[NAT],
        observed=[NAT],
        arrivals=arrivals,
        calendar=_calendar(),
        as_of=AS_OF,
        horizon=3,
    )
    np.testing.assert_array_equal(result.events, arrivals)
    np.testing.assert_array_equal(result.pending, np.zeros((3, 3), dtype=int))


# --------------------------------------------------------------------------
# Regressor widths of a fitted custom law bind every standalone forecast
# --------------------------------------------------------------------------
#
# The law below reads the supplied regressors directly and owns no coefficient
# site, so nothing in the posterior reveals how many columns it was trained on:
# only the width recorded by ``fit_stage`` can refuse a forecast whose
# regressors were dropped, padded or reshaped.

_FEATURE_VALUE = 1.0
_SUSCEPTIBILITY_VALUE = 0.7
_TRAINED_WIDTHS = [(0, 0), (1, 1), (2, 0), (0, 2)]


def _feature_law(inputs, shared):
    stay = -jnp.exp(-2.0 + inputs.features.sum(axis=-1))
    logits = 0.5 + inputs.susceptibility_features.sum(axis=-1)
    return EventLaw(
        TimingLaw(log1mexp(stay), stay), jnp.broadcast_to(logits, inputs.ages.shape[-1:])
    )


@functools.cache
def _feature_fit(p, q):
    """A deterministic custom fit trained on ``p`` timing and ``q`` susceptibility columns."""
    observations = StageObservations(
        ages=jnp.broadcast_to(jnp.arange(3), (2, 3)),
        features=jnp.ones((2, 3, p)),
        susceptibility_features=jnp.ones((2, q)),
        at_risk=jnp.ones((2, 3), dtype=bool),
        allowed=jnp.ones((2, 3), dtype=bool),
        event_index=jnp.array([0, -1]),
    )
    fit = fit_stage(
        observations, family=EventFamily(_feature_law, ProperTail()), num_steps=2, num_samples=3
    )
    assert dict(fit.parameters) == {}
    return fit


def _feature_law_numbers(p, q):
    """Per-age survival and susceptibility of ``_feature_law`` on the forecast regressors."""
    survive = np.exp(-np.exp(-2.0 + p * _FEATURE_VALUE))
    susceptibility = 1.0 / (1.0 + np.exp(-(0.5 + q * _SUSCEPTIBILITY_VALUE)))
    return survive, susceptibility


def _width_cases(p, q):
    """Supplied ``(features, susceptibility_features)`` widths; ``None`` omits the argument."""
    cases = [(None, None), (None, q), (p, None), (p, q), (p + 1, q), (p, q + 1), (p + 1, q + 1)]
    if p:
        cases.append((p - 1, q))
    if q:
        cases.append((p, q - 1))
    return list(dict.fromkeys(cases))


def _regressors(prefix, widths, rows, calendar_size):
    features, susceptibility = widths
    supplied = {}
    if features is not None:
        supplied[f"{prefix}features"] = np.full((rows, calendar_size, features), _FEATURE_VALUE)
    if susceptibility is not None:
        supplied[f"{prefix}susceptibility_features"] = np.full(
            (rows, susceptibility), _SUSCEPTIBILITY_VALUE
        )
    return supplied


def _assert_trained_widths_bind(call, prefix, trained, rows, calendar_size, check):
    """Every supplied width pair other than the trained one is refused; the trained pair replays."""
    for widths in _width_cases(*trained):
        regressors = _regressors(prefix, widths, rows, calendar_size)
        if (widths[0] or 0, widths[1] or 0) == trained:
            check(call(**regressors))
            continue
        wrong = (
            f"{prefix}features"
            if (widths[0] or 0) != trained[0]
            else (f"{prefix}susceptibility_features")
        )
        with pytest.raises(ValueError, match=wrong):
            call(**regressors)


@pytest.mark.parametrize("trained", _TRAINED_WIDTHS)
def test_forecast_events_binds_a_site_free_custom_law_to_its_trained_widths(trained):
    fit = _feature_fit(*trained)
    calendar = _calendar()
    units = 20000
    survive, susceptibility = _feature_law_numbers(*trained)
    expected = units * susceptibility * (1.0 - survive)

    def call(**regressors):
        return forecast_events(
            fit,
            origins=[NAT],
            observed=[NAT],
            arrivals=_future_arrivals(3, 4, 1, day=1, cohort=0, count=units),
            calendar=calendar,
            as_of=AS_OF,
            horizon=4,
            **regressors,
        )

    def check(result):
        np.testing.assert_allclose(result.events[:, 0, 0], expected, atol=5 * np.sqrt(expected))

    _assert_trained_widths_bind(call, "", trained, 1, calendar.size, check)


def _return_scenario(stage, trained):
    """History and exact ``(uninitiated, open)`` receipt expectations with the custom law at ``stage``."""
    survive, susceptibility = _feature_law_numbers(*trained)
    if stage == "receipt":
        # One return initiated the day before as_of: two exposed event-free days.
        history = _history(
            {
                "item_id": ["open"],
                "sale_date": ["2026-01-01"],
                "initiation_date": [str(AS_OF - np.timedelta64(1, "D"))],
                "receipt_date": [None],
            }
        )
        cured = 1.0 - susceptibility + susceptibility * survive**2
        return history, (0.0, susceptibility * survive**2 / cured)
    # One sale on as_of with a two-day policy: one event-free age-0 exposure,
    # then initiation within ages 1-2 and the default fresh-unit receipt of .5.
    history = _history({"item_id": ["sold"], "sale_date": [str(AS_OF)]}, policy_days=2)
    conditional = susceptibility * survive / (1.0 - susceptibility + susceptibility * survive)
    return history, (conditional * (1.0 - survive**2) * 0.5, 0.0)


def _stage_pair(stage, fit):
    return (_params(), fit) if stage == "receipt" else (fit, _params())


@pytest.mark.parametrize("stage", ["initiation", "receipt"])
@pytest.mark.parametrize("trained", _TRAINED_WIDTHS)
def test_expected_return_receipts_binds_a_site_free_custom_law_to_its_trained_widths(
    stage, trained
):
    history, (uninitiated, open_receipts) = _return_scenario(stage, trained)
    initiation, receipt = _stage_pair(stage, _feature_fit(*trained))
    calendar = _calendar()

    def call(**regressors):
        return expected_return_receipts(
            history, initiation, receipt, calendar=calendar, **regressors
        )

    def check(result):
        np.testing.assert_allclose(result[0], np.full(3, uninitiated), rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(result[1], np.full(3, open_receipts), rtol=1e-9, atol=1e-12)

    _assert_trained_widths_bind(call, f"{stage}_", trained, 1, calendar.size, check)


@pytest.mark.parametrize("stage", ["initiation", "receipt"])
@pytest.mark.parametrize("trained", _TRAINED_WIDTHS)
def test_forecast_returns_binds_a_site_free_custom_law_to_its_trained_widths(stage, trained):
    history, (uninitiated, open_receipts) = _return_scenario(stage, trained)
    initiation, receipt = _stage_pair(stage, _feature_fit(*trained))
    future = SalesForecast(
        cohorts=pd.DataFrame(
            {"item_id": ["f"], "sale_date": [str(AS_OF + np.timedelta64(1, "D"))]}
        ),
        counts=np.array([[3]]),
    )
    calendar = _calendar()

    def call(**regressors):
        return forecast_returns(
            history,
            initiation,
            receipt,
            calendar=calendar,
            horizon=4,
            future_sales=future,
            **regressors,
        )

    def check(result):
        np.testing.assert_allclose(
            result.expected_uninitiated_receipts, np.full(3, uninitiated), rtol=1e-9, atol=1e-12
        )
        np.testing.assert_allclose(
            result.expected_open_receipts, np.full(3, open_receipts), rtol=1e-9, atol=1e-12
        )

    _assert_trained_widths_bind(call, f"{stage}_", trained, 2, calendar.size, check)
