"""Analytically grounded tests for ttenet.forecast.forecast_returns."""

import numpy as np
import pandas as pd
import pytest

from ttenet.data import prepare_history
from ttenet.dates import date_grid
from ttenet.forecast import forecast_returns
from ttenet.survival import StageParameters

AS_OF = np.datetime64("2026-03-01", "D")


def _logit(p):
    return float(np.log(p / (1.0 - p)))


def _params(k=5, hazard=0.5, cure=0.5, p=0, q=0):
    return StageParameters(
        age_logits=np.full(k, _logit(hazard)),
        beta=np.zeros((p,)),
        cure_intercept=np.array(_logit(cure)),
        cure_beta=np.zeros((q,)),
    )


def _history(rows, as_of=AS_OF, policy_days=90):
    return prepare_history(pd.DataFrame(rows), as_of=as_of, policy_days=policy_days)


def _calendar(start="2024-01-01", end=None, horizon=120):
    end = end or (AS_OF + np.timedelta64(horizon, "D"))
    return date_grid(
        np.datetime64(start, "D"), np.datetime64(end, "D") if isinstance(end, str) else end
    )


# --------------------------------------------------------------------------
# Boundary / validation
# --------------------------------------------------------------------------


def test_calendar_must_be_contiguous_daily_grid():
    history = _history(
        {
            "item_id": ["a"],
            "sale_date": ["2026-02-01"],
            "initiation_date": [None],
            "receipt_date": [None],
        }
    )
    params = _params()
    gappy = np.array(["2024-01-01", "2024-01-03", "2026-06-01"], dtype="datetime64[D]")
    with pytest.raises(ValueError, match="contiguous"):
        forecast_returns(history, params, params, calendar=gappy, horizon=10)


def test_calendar_must_cover_output_horizon():
    history = _history(
        {
            "item_id": ["a"],
            "sale_date": ["2026-02-01"],
            "initiation_date": [None],
            "receipt_date": [None],
        }
    )
    params = _params()
    short_calendar = _calendar(end=AS_OF + np.timedelta64(3, "D"))
    with pytest.raises(ValueError, match="horizon"):
        forecast_returns(history, params, params, calendar=short_calendar, horizon=10)


def test_calendar_must_cover_initiation_deadline():
    # sold yesterday, 90-day policy deadline is ~89 days past the short output horizon
    history = _history(
        {
            "item_id": ["a"],
            "sale_date": [str(AS_OF - np.timedelta64(1, "D"))],
            "initiation_date": [None],
            "receipt_date": [None],
        }
    )
    params = _params()
    short_calendar = _calendar(end=AS_OF + np.timedelta64(5, "D"))
    with pytest.raises(ValueError, match="deadline"):
        forecast_returns(history, params, params, calendar=short_calendar, horizon=5)


def test_missing_required_features_rejected():
    history = _history(
        {
            "item_id": ["a"],
            "sale_date": ["2026-02-01"],
            "initiation_date": [None],
            "receipt_date": [None],
        }
    )
    params_with_features = _params(p=2)
    with pytest.raises(ValueError, match="initiation_features"):
        forecast_returns(history, params_with_features, _params(), calendar=_calendar(), horizon=10)


def test_feature_width_mismatch_rejected():
    history = _history(
        {
            "item_id": ["a"],
            "sale_date": ["2026-02-01"],
            "initiation_date": [None],
            "receipt_date": [None],
        }
    )
    params = _params(p=2)
    calendar = _calendar()
    wrong_width = np.zeros((1, len(calendar), 3))
    with pytest.raises(ValueError, match="initiation_features"):
        forecast_returns(
            history,
            params,
            _params(),
            calendar=calendar,
            horizon=10,
            initiation_features=wrong_width,
        )


@pytest.mark.parametrize("value", [-1.0, 1.25, float("nan"), float("inf")])
def test_invalid_future_counts_rejected(value):
    history = _history({"item_id": [], "sale_date": [], "initiation_date": [], "receipt_date": []})
    params = _params()
    calendar = _calendar()
    future_sales = pd.DataFrame(
        {"item_id": ["f1"], "sale_date": [str(AS_OF + np.timedelta64(1, "D"))]}
    )
    with pytest.raises(ValueError):
        forecast_returns(
            history,
            params,
            params,
            calendar=calendar,
            horizon=10,
            future_sales=future_sales,
            future_counts=np.array([[value]]),
        )


def test_future_sales_duplicate_item_id_rejected():
    history = _history({"item_id": [], "sale_date": [], "initiation_date": [], "receipt_date": []})
    params = _params()
    future_sales = pd.DataFrame(
        {
            "item_id": ["f1", "f1"],
            "sale_date": [str(AS_OF + np.timedelta64(1, "D"))] * 2,
            "quantity": [1, 1],
        }
    )
    with pytest.raises(ValueError, match="unique"):
        forecast_returns(
            history, params, params, calendar=_calendar(), horizon=10, future_sales=future_sales
        )


def test_future_sales_item_id_overlap_with_history_rejected():
    history = _history(
        {
            "item_id": ["shared"],
            "sale_date": ["2026-02-01"],
            "initiation_date": [None],
            "receipt_date": [None],
        }
    )
    params = _params()
    future_sales = pd.DataFrame(
        {"item_id": ["shared"], "sale_date": [str(AS_OF + np.timedelta64(1, "D"))], "quantity": [1]}
    )
    with pytest.raises(ValueError, match="overlap"):
        forecast_returns(
            history, params, params, calendar=_calendar(), horizon=10, future_sales=future_sales
        )


def test_future_sales_outside_horizon_rejected():
    history = _history({"item_id": [], "sale_date": [], "initiation_date": [], "receipt_date": []})
    params = _params()
    future_sales = pd.DataFrame(
        {"item_id": ["f1"], "sale_date": [str(AS_OF + np.timedelta64(20, "D"))], "quantity": [1]}
    )
    with pytest.raises(ValueError, match="horizon"):
        forecast_returns(
            history, params, params, calendar=_calendar(), horizon=10, future_sales=future_sales
        )


def test_mismatched_posterior_draws_rejected():
    history = _history({"item_id": [], "sale_date": [], "initiation_date": [], "receipt_date": []})
    init3 = StageParameters(
        age_logits=np.zeros((3, 5)),
        beta=np.zeros((3, 0)),
        cure_intercept=np.zeros((3,)),
        cure_beta=np.zeros((3, 0)),
    )
    future_sales = pd.DataFrame(
        {"item_id": ["f1"], "sale_date": [str(AS_OF + np.timedelta64(1, "D"))]}
    )
    future_counts = np.zeros(
        (5, 1), dtype=int
    )  # 5 draws, incompatible with the 3 posterior draws above
    with pytest.raises(ValueError, match="align"):
        forecast_returns(
            history,
            init3,
            init3,
            calendar=_calendar(),
            horizon=10,
            future_sales=future_sales,
            future_counts=future_counts,
        )


def test_single_draw_broadcasts_against_future_count_draws():
    history = _history({"item_id": [], "sale_date": [], "initiation_date": [], "receipt_date": []})
    params = _params(hazard=0.9, cure=1 - 1e-9)  # single draw
    future_sales = pd.DataFrame(
        {"item_id": ["f1"], "sale_date": [str(AS_OF + np.timedelta64(1, "D"))]}
    )
    future_counts = np.array([[0], [100]])  # 2 draws
    result = forecast_returns(
        history,
        params,
        params,
        calendar=_calendar(),
        horizon=5,
        seed=0,
        future_sales=future_sales,
        future_counts=future_counts,
    )
    assert result.initiations.shape == (2, 5)
    assert result.initiations[0].sum() == 0
    assert result.initiations[1].sum() > 0


# --------------------------------------------------------------------------
# Core dynamics and closed-form / analytical checks
# --------------------------------------------------------------------------


def test_analytical_conditional_susceptibility_matches_closed_form():
    # From survival.py's documented example: h=.5, pi=.4, censored after two exposed days
    # (ages 0 and 1) has probability .6 + .4*.25 = .7 and conditional susceptibility 1/7.
    history = _history(
        {
            "item_id": ["z"],
            "sale_date": ["2026-01-01"],
            "initiation_date": [str(AS_OF - np.timedelta64(1, "D"))],
            "receipt_date": [None],
        }
    )
    receipt = _params(hazard=0.5, cure=0.4)
    result = forecast_returns(history, receipt, receipt, calendar=_calendar(), horizon=5, seed=0)
    np.testing.assert_allclose(result.expected_open_receipts, [1.0 / 7.0], rtol=1e-9)
    np.testing.assert_allclose(result.expected_existing_receipts, result.expected_open_receipts)
    np.testing.assert_allclose(result.expected_uninitiated_receipts, [0.0])


def test_day90_inclusive_vs_day91_policy_boundary_closed_form():
    policy_days = 10
    # p1: one remaining allowed day (age0 = policy_days - 1); p2: deadline already passed today.
    raw = {
        "item_id": ["p1", "p2"],
        "sale_date": [
            str(AS_OF - np.timedelta64(policy_days - 1, "D")),
            str(AS_OF - np.timedelta64(policy_days, "D")),
        ],
        "initiation_date": [None, None],
        "receipt_date": [None, None],
    }
    history = _history(raw, policy_days=policy_days)
    assert history.frame["eligible"].all()  # both still within [0, policy_days]

    h, pi_init = 0.2, 0.6
    init = _params(k=15, hazard=h, cure=pi_init)
    receipt = _params(
        k=15, hazard=h, cure=1 - 1e-9
    )  # near-certain receipt susceptibility isolates p_initiate
    result = forecast_returns(history, init, receipt, calendar=_calendar(), horizon=5, seed=0)

    def smarg(pi, s):
        return (1 - pi) + pi * s

    s_asof_p1 = (1 - h) ** policy_days  # ages 0..(policy_days-1) inclusive
    s_deadline_p1 = (1 - h) ** (policy_days + 1)  # one more day, exactly at the deadline
    p_initiate_p1 = 1 - smarg(pi_init, s_deadline_p1) / smarg(pi_init, s_asof_p1)
    pi_receipt_raw = 1.0 / (1.0 + np.exp(-(_logit(1 - 1e-9))))
    expected = p_initiate_p1 * pi_receipt_raw  # p2 contributes exactly 0: its window is empty
    np.testing.assert_allclose(result.expected_uninitiated_receipts, [expected], rtol=1e-6)


def test_no_weekend_receipts():
    history = _history(
        {
            "item_id": ["w"],
            "sale_date": ["2026-01-01"],
            "initiation_date": ["2026-02-20"],
            "receipt_date": [None],
        }
    )
    params = _params(hazard=0.9, cure=1 - 1e-9)
    calendar = _calendar()
    allowed = np.array([pd.Timestamp(d).weekday() < 5 for d in calendar])
    result = forecast_returns(
        history, params, params, calendar=calendar, horizon=14, seed=0, receipt_allowed=allowed
    )
    weekend_mask = np.array([pd.Timestamp(d).weekday() >= 5 for d in result.dates])
    assert weekend_mask.any()
    assert np.all(result.receipts[:, weekend_mask] == 0)


def test_same_day_initiation_and_receipt():
    # Item sold exactly on as_of: age-zero risk on day 1 for both stages, so a near-certain
    # hazard should initiate and receive on the very first simulated day (event ordering:
    # initiation before receipt, same day).
    history = _history(
        {
            "item_id": ["x"],
            "sale_date": [str(AS_OF)],
            "initiation_date": [None],
            "receipt_date": [None],
        }
    )
    params = _params(hazard=1 - 1e-9, cure=1 - 1e-9)
    result = forecast_returns(history, params, params, calendar=_calendar(), horizon=3, seed=0)
    assert result.initiations[0, 0] == 1
    assert result.receipts[0, 0] == 1
    assert result.open_returns[0, 0] == 0
    assert result.eligible[0, 0] == 0


def test_heterogeneous_origins_share_calendar_day_features():
    # Two open items with different initiation dates (different ages on any given calendar
    # day). A feature that spikes hazard on exactly one calendar day must fire for both on
    # that same calendar day, proving features are indexed by calendar day, not by age.
    raw = {
        "item_id": ["A", "B"],
        "sale_date": ["2026-01-01", "2026-01-01"],
        "initiation_date": [
            str(AS_OF - np.timedelta64(1, "D")),
            str(AS_OF - np.timedelta64(5, "D")),
        ],
        "receipt_date": [None, None],
    }
    history = _history(raw)
    calendar = _calendar()
    k = 20
    age_logits = np.full(k, _logit(1e-4))
    beta = np.array([15.0])
    receipt = StageParameters(
        age_logits=age_logits,
        beta=beta,
        cure_intercept=np.array(_logit(1 - 1e-9)),
        cure_beta=np.zeros((0,)),
    )
    init = _params(k=k, p=1)

    storm_date = AS_OF + np.timedelta64(5, "D")
    storm_idx = int(np.where(calendar == storm_date)[0][0])
    n_total = len(history.frame)
    receipt_features = np.zeros((n_total, len(calendar), 1))
    receipt_features[:, storm_idx, 0] = 1.0
    initiation_features = np.zeros((n_total, len(calendar), 1))

    result = forecast_returns(
        history,
        init,
        receipt,
        calendar=calendar,
        horizon=10,
        seed=0,
        receipt_features=receipt_features,
        initiation_features=initiation_features,
    )
    storm_out_idx = int(np.where(result.dates == storm_date)[0][0])
    assert result.receipts[0, storm_out_idx] == 2
    assert result.receipts.sum() == 2  # both received exactly on the storm day, nowhere else


def test_tail_bin_reused_and_no_artificial_receipt_deadline():
    # Open item far beyond any plausible deadline (age_since_initiation = 199) and beyond
    # the fitted age bins (K=30): receipt has no policy deadline, and ages >= K-1 reuse the
    # final fitted bin's hazard indefinitely (the documented tail assumption).
    history = _history(
        {
            "item_id": ["old"],
            "sale_date": ["2025-01-01"],
            "initiation_date": [str(AS_OF - np.timedelta64(199, "D"))],
            "receipt_date": [None],
        },
        policy_days=365,
    )
    calendar = _calendar(start="2024-01-01")
    k = 30
    age_logits = np.full(k, -50.0)
    age_logits[-1] = _logit(0.01)
    params = StageParameters(
        age_logits=age_logits,
        beta=np.zeros((0,)),
        cure_intercept=np.array(_logit(0.9)),
        cure_beta=np.zeros((0,)),
    )
    result = forecast_returns(history, params, params, calendar=calendar, horizon=5, seed=0)

    h_early = 1.0 / (1.0 + np.exp(50.0))
    h_tail = 0.01
    ages = np.arange(200)
    idx = np.clip(ages, 0, k - 1)
    hazards = np.where(idx == k - 1, h_tail, h_early)
    survival = np.prod(1 - hazards)
    pi = 0.9
    expected = pi * survival / ((1 - pi) + pi * survival)
    np.testing.assert_allclose(result.expected_open_receipts, [expected], rtol=1e-6)
    assert result.expected_open_receipts[0] > 0  # not artificially zeroed by an invented deadline


def test_zero_cure_probability_suppresses_events():
    history = _history(
        {
            "item_id": ["a", "b"],
            "sale_date": ["2026-02-01", "2026-01-01"],
            "initiation_date": [None, "2026-02-20"],
            "receipt_date": [None, None],
        }
    )
    params = _params(hazard=0.9, cure=1e-22)  # essentially nobody is susceptible
    result = forecast_returns(history, params, params, calendar=_calendar(), horizon=20, seed=0)
    np.testing.assert_allclose(result.expected_open_receipts, [0.0], atol=1e-15)
    np.testing.assert_allclose(result.expected_uninitiated_receipts, [0.0], atol=1e-15)
    assert result.initiations.sum() == 0
    assert result.receipts.sum() == 0


def test_abandonment_lowers_expected_receipts_below_raw_susceptibility():
    # An open item that has survived many days of a substantial daily hazard without being
    # received is, by Bayesian updating, far more likely to be cured than its prior
    # susceptibility pi suggested (regression toward the cured subpopulation).
    raw_pi = 0.5
    history = _history(
        {
            "item_id": ["w"],
            "sale_date": ["2025-11-01"],
            "initiation_date": ["2026-01-01"],
            "receipt_date": [None],
        }
    )
    receipt = _params(hazard=0.3, cure=raw_pi)
    result = forecast_returns(history, receipt, receipt, calendar=_calendar(), horizon=5, seed=0)
    assert float(result.expected_open_receipts[0]) < raw_pi
    assert float(result.expected_open_receipts[0]) > 0.0


def test_expired_uninitiated_excluded_from_eligible():
    history = _history(
        {
            "item_id": ["expired"],
            "sale_date": ["2025-01-01"],
            "initiation_date": [None],
            "receipt_date": [None],
        },
        policy_days=90,
    )
    assert not history.frame["eligible"].iloc[0]  # far past the 90-day policy window
    params = _params(hazard=0.9, cure=1 - 1e-9)
    result = forecast_returns(history, params, params, calendar=_calendar(), horizon=10, seed=0)
    assert np.all(result.eligible == 0)
    assert np.all(result.initiations == 0)


def test_received_rows_excluded_from_simulation():
    history = _history(
        {
            "item_id": ["done"],
            "sale_date": ["2026-01-01"],
            "initiation_date": ["2026-01-05"],
            "receipt_date": ["2026-01-10"],
        }
    )
    params = _params(hazard=0.9, cure=1 - 1e-9)
    result = forecast_returns(history, params, params, calendar=_calendar(), horizon=10, seed=0)
    assert np.all(result.eligible == 0)
    assert np.all(result.open_returns == 0)
    assert np.all(result.receipts == 0)
    np.testing.assert_allclose(result.expected_existing_receipts, [0.0])


# --------------------------------------------------------------------------
# Conservation and draw-preservation
# --------------------------------------------------------------------------


def test_mass_conservation_and_expected_receipts_identity():
    raw = {
        "item_id": ["elig1", "elig2", "open1", "recv1"],
        "sale_date": ["2026-02-20", "2025-12-01", "2026-01-01", "2026-01-01"],
        "initiation_date": [None, None, "2026-02-15", "2026-01-05"],
        "receipt_date": [None, None, None, "2026-01-10"],
    }
    history = _history(raw)
    calendar = _calendar()
    init = _params(hazard=0.3, cure=0.7)
    receipt = _params(hazard=0.25, cure=0.6)
    future_sales = pd.DataFrame(
        {
            "item_id": ["f1", "f2"],
            "sale_date": [str(AS_OF + np.timedelta64(1, "D")), str(AS_OF + np.timedelta64(3, "D"))],
        }
    )
    future_counts = np.array([[10, 25], [3, 50], [0, 0]])  # 3 draws
    result = forecast_returns(
        history,
        init,
        receipt,
        calendar=calendar,
        horizon=30,
        seed=0,
        future_sales=future_sales,
        future_counts=future_counts,
    )

    historical_eligible = 2  # elig1, elig2
    historical_open = 1  # open1
    source = historical_eligible + historical_open + future_counts.sum(axis=1)
    assert np.all(result.receipts.sum(axis=1) <= source)
    assert np.all(result.initiations.sum(axis=1) <= historical_eligible + future_counts.sum(axis=1))
    assert np.all(result.eligible >= 0)
    assert np.all(result.open_returns >= 0)
    np.testing.assert_allclose(
        result.expected_existing_receipts,
        result.expected_open_receipts + result.expected_uninitiated_receipts,
    )


def test_uncertain_future_counts_preserve_draws_without_averaging():
    history = _history({"item_id": [], "sale_date": [], "initiation_date": [], "receipt_date": []})
    params = _params(hazard=0.9, cure=1 - 1e-9)
    future_sales = pd.DataFrame(
        {"item_id": ["f1"], "sale_date": [str(AS_OF + np.timedelta64(1, "D"))]}
    )
    future_counts = np.array([[0], [7], [500]])
    result = forecast_returns(
        history,
        params,
        params,
        calendar=_calendar(),
        horizon=5,
        seed=0,
        future_sales=future_sales,
        future_counts=future_counts,
    )
    # draws must reflect their own source count exactly, not some averaged blend
    assert result.initiations[0].sum() == 0
    assert 0 < result.initiations[1].sum() <= 7
    assert result.initiations[2].sum() > 50  # near-certain hazard on a 500-unit cohort


def test_counts_never_negative_or_exceed_remaining_population():
    history = _history(
        {
            "item_id": ["a", "b"],
            "sale_date": ["2026-02-20", "2026-01-01"],
            "initiation_date": [None, "2026-02-25"],
            "receipt_date": [None, None],
        }
    )
    params = _params(hazard=0.6, cure=0.8)
    for seed in range(8):
        result = forecast_returns(
            history, params, params, calendar=_calendar(), horizon=25, seed=seed
        )
        assert np.all(result.initiations >= 0)
        assert np.all(result.receipts >= 0)
        assert np.all(result.eligible >= 0)
        assert np.all(result.open_returns >= 0)
        assert result.initiations.sum() <= 1  # only one eligible historical unit
        assert result.receipts.sum() <= 2  # at most the open unit plus any newly initiated unit


def test_future_abandoned_returns_remain_open_through_last_forecast_day():
    history = _history({"item_id": [], "sale_date": []})
    initiate = StageParameters(np.array([50.0]), np.empty(0), 50.0, np.empty(0))
    abandon = StageParameters(np.array([0.0]), np.empty(0), -1000.0, np.empty(0))
    future = pd.DataFrame(
        {
            "item_id": ["future"],
            "sale_date": [AS_OF + np.timedelta64(1, "D")],
            "quantity": [12],
        }
    )
    result = forecast_returns(
        history,
        initiate,
        abandon,
        calendar=_calendar(),
        horizon=3,
        future_sales=future,
    )
    np.testing.assert_array_equal(result.open_returns, [[12, 12, 12]])
    np.testing.assert_array_equal(
        result.open_returns + result.receipts.cumsum(axis=1),
        result.initiations.cumsum(axis=1),
    )


def test_future_sales_cannot_silently_disappear_without_a_sale_date():
    history = _history({"item_id": [], "sale_date": []})
    future = pd.DataFrame({"item_id": ["future"], "sale_date": [None], "quantity": [12]})
    with pytest.raises(ValueError):
        forecast_returns(
            history,
            _params(),
            _params(),
            calendar=_calendar(),
            horizon=3,
            future_sales=future,
        )


def test_strong_susceptibility_does_not_cancel_remaining_event_probability():
    history = _history(
        {"item_id": ["sale"], "sale_date": [AS_OF]},
        policy_days=1,
    )
    certain_susceptible = StageParameters(np.zeros(1), np.empty(0), 1e16, np.empty(0))
    result = forecast_returns(
        history,
        certain_susceptible,
        _params(hazard=0.5, cure=0.5),
        calendar=_calendar(),
        horizon=1,
    )
    # Still susceptible after age zero; one remaining day at hazard .5,
    # followed by receipt susceptibility .5, gives .25 eventual receipts.
    np.testing.assert_allclose(result.expected_uninitiated_receipts, [0.25], rtol=1e-12)
