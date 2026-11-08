"""Contracts for the returns essay's data code: the synthetic world, the numbers computed from
it, and the JSON the interactive figures draw. None of this touches marimo or a plotting
library, so a wrong number or a broken payload fails here rather than as a wrong figure."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import retail_returns_analysis as analysis
import retail_returns_chain as chain
import retail_returns_scenario as scenario
import retail_returns_world as world

# --- The synthetic world ----------------------------------------------------------------


def test_simulated_ledger_obeys_the_business_rules():
    truth, _ = world.simulate_retail(seed=3, training_days=60, horizon=14, volume=3.0)
    again, _ = world.simulate_retail(seed=3, training_days=60, horizon=14, volume=3.0)
    pd.testing.assert_frame_equal(truth, again)

    started = truth.initiation_date.notna()
    received = truth.receipt_date.notna()
    assert not (received & ~started).any(), "a receipt needs an initiation"
    assert not (received & truth.abandoned_truth).any(), "an abandoned return never arrives"
    assert (truth.receipt_date[received].dt.dayofweek < 5).all(), "the warehouse is closed weekends"
    assert (truth.receipt_date[received] >= truth.initiation_date[received]).all()
    age = (truth.initiation_date - truth.sale_date).dt.days[started]
    assert age.min() == world.DELIVERY_DAYS, "no return starts before the parcel arrives"
    assert age.max() <= 90, "the policy allows starting a return through day 90"
    assert truth.abandoned_truth[started].any() and received.any(), "the sample exercises both"


def test_true_receipt_hazard_is_the_discretized_weibull():
    # Surviving every daily bin up to age A must equal the continuous Weibull survival.
    ages = np.arange(0, 20)
    survived = np.cumprod(1 - world.true_receipt_hazard(ages))
    weibull = np.exp(-(((ages + 1) / world.RECEIPT_SCALE) ** world.RECEIPT_SHAPE))
    np.testing.assert_allclose(survived, weibull)


# --- Numbers computed from a ledger -----------------------------------------------------


def _ledger(**columns):
    return pd.DataFrame({"item_id": [f"u{i}" for i in range(len(columns["sale_date"]))]} | columns)


def test_typical_return_ignores_receipts_after_the_snapshot():
    as_of = pd.Timestamp("2026-04-30")
    # (sold, days to return start, days in transit). The first row has the median timing but
    # arrives on May 1, after the snapshot, so the ledger cannot show it as received.
    rows = [
        ("2026-04-12", 12, 7),
        ("2026-03-01", 10, 5),
        ("2026-03-02", 12, 7),
        ("2026-03-01", 30, 20),
    ]
    sold = pd.to_datetime([s for s, _, _ in rows])
    started = sold + pd.to_timedelta([w for _, w, _ in rows], unit="D")
    units = _ledger(
        sale_date=sold,
        initiation_date=started,
        receipt_date=started + pd.to_timedelta([t for _, _, t in rows], unit="D"),
    )
    assert analysis.typical_return(units, as_of)["sold"] == pd.Timestamp("2026-03-02")


def test_storm_runs_have_exclusive_ends():
    def storms(days):
        age = (np.asarray(days) - np.datetime64("2026-01-01")).astype(int)
        return ((age >= 2) & (age <= 3)) | (age == 8)

    spans = analysis.storm_runs(storms, "2026-01-01", "2026-01-10")
    assert spans.start.dt.day.tolist() == [3, 9]
    assert spans.end.dt.day.tolist() == [5, 10]


def test_daily_totals_sums_cohorts_onto_their_sale_day():
    sales = SimpleNamespace(
        cohorts=pd.DataFrame({"sale_date": ["2026-02-01", "2026-02-01", "2026-02-03"]}),
        counts=np.array([[1, 2, 5], [0, 4, 6]]),
    )
    totals = analysis.daily_totals(sales, pd.date_range("2026-02-01", periods=3))
    np.testing.assert_array_equal(totals, [[3, 0, 5], [4, 0, 6]])


def test_crps_of_a_point_forecast_is_its_absolute_error():
    actual = np.array([3.0, 0.0, 7.0])
    point = np.tile([2.0, 1.0, 7.0], (5, 1))
    assert analysis.crps(point, actual) == pytest.approx(np.abs(point[0] - actual).mean())
    spread = np.vstack([point - 1, point + 1])
    assert analysis.crps(spread, actual) > 0


def test_eventual_counts_split_owed_returns_by_where_they_stand():
    truth = _ledger(
        sale_date=pd.to_datetime(["2026-03-01"] * 4),
        initiation_date=pd.to_datetime(["2026-03-02", "2026-03-02", "2026-03-20", "2026-03-02"]),
        receipt_date=pd.to_datetime(["2026-03-05", "2026-04-02", "2026-04-03", None]),
    )
    counts = analysis.eventual_counts(truth, np.datetime64("2026-03-10"))
    # Unit 1 is in transit at the snapshot, unit 2 had not started one; 0 is received, 3 is lost.
    assert counts == {"open": 1, "uninitiated": 1}


def test_post_storm_check_lands_on_the_next_open_day():
    dates = pd.date_range("2026-07-01", periods=14)  # a Wednesday start
    spans = pd.DataFrame(
        {"start": [pd.Timestamp("2026-07-02")], "end": [pd.Timestamp("2026-07-04")]}
    )
    receipts = np.tile(np.arange(14), (4, 1))
    forecast = SimpleNamespace(dates=dates.values, receipts=receipts)
    held_out = SimpleNamespace(receipts=pd.Series(np.arange(14) * 10))
    check = analysis.post_storm_check(forecast, held_out, spans)
    assert check["date"] == pd.Timestamp("2026-07-06")  # the storm ends Saturday; Monday opens
    assert (check["mean"], check["actual"]) == (5.0, 50)


def test_weekday_average_looks_back_exactly_eight_weeks():
    days = pd.date_range("2026-01-05", periods=84)  # starts on a Monday
    receipts = np.where(np.arange(84) < 28, 1000, days.dayofweek.to_numpy() * 10)
    daily = pd.DataFrame({"date": days, "Returns received": receipts})
    held_out = pd.DataFrame({"date": pd.date_range(days[-1] + pd.Timedelta(days=1), periods=7)})
    average = analysis.weekday_average(daily, held_out)
    # The older, huge weeks must not leak in: Monday averages 0, Tuesday 10, and so on.
    assert average.to_numpy().tolist() == [d * 10.0 for d in held_out.date.dt.dayofweek]


def test_lagged_rate_uses_mature_receipts_and_moves_weekends_to_monday():
    as_of = np.datetime64("2026-04-30")
    day = pd.Timestamp
    units = pd.DataFrame(
        {
            # Four mature units (age 70), one recent and one ancient; only the mature
            # receipts seen by the snapshot set the rate (2 of 4) and the lag (median 11).
            "sale_date": [day("2026-02-19")] * 4 + [day("2026-04-22"), day("2025-10-12")],
            "receipt_date": [
                day("2026-03-01"),
                day("2026-03-03"),
                day("2026-05-05"),  # after the snapshot: not yet seen
                pd.NaT,
                day("2026-04-25"),
                day("2025-10-15"),
            ],
        }
    )
    dates = pd.date_range("2026-05-01", periods=14)  # a Friday start
    lagged = analysis.lagged_rate(units, as_of, np.full(14, 10.0), dates)
    assert (lagged.attrs["rate"], lagged.attrs["lag"]) == (0.5, 11)
    expected = pd.Series(0.0, index=dates)
    expected["2026-05-04"] = 0.5  # the Apr 22 sale lands on Sunday May 3, so Monday
    expected["2026-05-12":"2026-05-14"] = 5.0  # forecast sales from May 1-3
    pd.testing.assert_series_equal(lagged, expected, check_names=False)


def test_baseline_scores_treat_each_baseline_as_a_point_forecast():
    actual = np.tile([4.0, 2.0], 7)
    held_out = SimpleNamespace(receipts=pd.Series(actual))
    forecast = SimpleNamespace(receipts=np.tile(actual, (6, 1)) + np.array([[1], [-1]] * 3))
    baselines = {"Flat": pd.Series(np.full(14, 3.0)), "Exact": pd.Series(actual)}
    scores = analysis.baseline_scores(forecast, held_out, baselines, "Model")
    scores = scores.set_index("Method")
    point = scores.loc["Flat"]
    assert point.crps == point.mae == pytest.approx(1.0)
    assert scores.loc["Exact"].mae == 0.0
    assert np.isnan(point.total_lo) and np.isnan(point.total_hi)
    model = scores.loc["Model"]
    assert model.total == pytest.approx(42.0)
    assert model.total_lo <= model.total <= model.total_hi


def test_baseline_scores_score_a_baseline_with_draws_as_a_distribution():
    actual = np.tile([4.0, 2.0], 7)
    held_out = SimpleNamespace(receipts=pd.Series(actual))
    forecast = SimpleNamespace(receipts=np.tile(actual, (2, 1)))
    # Mean exactly right, draws one unit either side: CRPS = E|X-y| - E|X-X'|/2 = 1 - 1/2.
    baseline = pd.Series(actual)
    baseline.attrs["draws"] = np.stack([actual + 1, actual - 1])
    scores = analysis.baseline_scores(forecast, held_out, {"Spread": baseline}, "Model")
    spread = scores.set_index("Method").loc["Spread"]
    assert spread.mae == 0.0
    assert spread.crps == pytest.approx(0.5)
    assert spread.total_lo < spread.total < spread.total_hi


def _initiation_fit(logit, draws=3):
    return SimpleNamespace(
        parameters=SimpleNamespace(
            age_logits=np.full((draws, 16), logit),
            beta=np.zeros((draws, 2)),
            susceptibility_intercept=np.zeros(draws),
            susceptibility_beta=np.zeros((draws, 1)),
        )
    )


def test_cohort_rates_never_fall_below_what_the_ledger_already_shows():
    as_of = np.datetime64("2026-03-31")
    sale = pd.date_range("2026-03-03", "2026-03-30").repeat(2)
    units = _ledger(
        sale_date=sale,
        initiation_date=pd.to_datetime([None] * len(sale)),
        receipt_date=pd.to_datetime([None] * len(sale)),
        product=0.0,
    )
    started = np.arange(len(sale)) % 4 == 0
    units.loc[started, "initiation_date"] = units.sale_date[started] + pd.Timedelta(days=1)
    truth = units.copy()
    truth.loc[~started & (np.arange(len(sale)) % 3 == 0), "initiation_date"] = (
        as_of + np.timedelta64(5, "D")
    )

    # With no hazard left, a quiet unit adds nothing: the model is exactly the ledger's rate.
    quiet = analysis.weekly_cohort_rates(
        units, truth, SimpleNamespace(initiation_fit=_initiation_fit(-50.0)), as_of, min_units=1
    )
    np.testing.assert_allclose(quiet.model, quiet.observed, atol=1e-9)

    # With some hazard it can only add, never subtract, and a share stays a share.
    live = analysis.weekly_cohort_rates(
        units, truth, SimpleNamespace(initiation_fit=_initiation_fit(-3.0)), as_of, min_units=1
    )
    assert (live.model >= live.observed - 1e-12).all() and (live.model <= 1).all()
    assert (live.model_lo <= live.model).all() and (live.model <= live.model_hi).all()
    assert (live.model > live.observed).any()


# --- The storm-scenario payload --------------------------------------------------------


def _run(receipts, open_returns=None, sales=None):
    receipts = np.asarray(receipts, dtype=float)
    return SimpleNamespace(
        receipts=receipts,
        open_returns=receipts * 0 if open_returns is None else np.asarray(open_returns),
        sales=SimpleNamespace(counts=np.arange(receipts.shape[0]) if sales is None else sales),
    )


def test_scenario_gap_is_a_paired_difference_not_a_difference_of_bands():
    draws = np.arange(5)[:, None]  # each draw has its own level, so unpaired bands would differ
    baseline = _run(5 + draws + np.zeros((5, 10)))
    change = np.array([0, 0, -3, -3, 6, 0, 2, 0, 0, 0])
    shifted = _run(baseline.receipts + change)
    storm_baseline = np.arange(10) == 2
    storm_days = (np.arange(10) >= 2) & (np.arange(10) <= 5)

    payload = scenario.scenario_payload(shifted, baseline, storm_baseline, storm_days)

    gap = payload["gap"]
    np.testing.assert_allclose(gap["mean"], np.cumsum(change))
    # Same shift in every draw: a paired gap has no width even though the draws differ a lot.
    np.testing.assert_allclose(gap["lo90"], gap["hi90"])
    first, week = payload["windows"]
    assert (first["start"], first["end"]) == (2, 5)  # the union of both runs' stormy days
    assert (week["start"], week["end"]) == (6, 9)  # the days after, up to the horizon
    assert first["receipt"]["diff"]["mean"] == pytest.approx(0.0)
    assert week["receipt"]["diff"]["mean"] == pytest.approx(2.0)
    assert payload["totals"]["receipt"]["scenario"]["mean"] == pytest.approx(
        payload["totals"]["receipt"]["baseline"]["mean"] + change.sum()
    )
    assert payload["storm"]["scenario"] == storm_days.tolist()


def test_scenario_week_after_is_dropped_when_the_storm_runs_to_the_horizon():
    baseline = _run(np.ones((3, 6)))
    payload = scenario.scenario_payload(baseline, baseline, np.arange(6) >= 4, np.arange(6) >= 4)
    assert [(w["start"], w["end"]) for w in payload["windows"]] == [(4, 5)]


def test_scenario_payload_refuses_runs_that_do_not_share_sales_draws():
    baseline = _run(np.ones((3, 6)), sales=np.array([1, 2, 3]))
    other = _run(np.ones((3, 6)), sales=np.array([1, 2, 4]))
    flags = np.arange(6) == 2
    with pytest.raises(AssertionError, match="do not pair"):
        scenario.scenario_payload(other, baseline, flags, flags)


def test_scenario_data_keys_every_run_and_reads_each_storm_provider():
    dates = pd.date_range("2026-07-01", periods=8).values
    baseline = SimpleNamespace(**vars(_run(np.ones((3, 8)))), dates=dates)
    runs = {"as_forecast": baseline, "clear": baseline}

    def storms(days):
        return (np.arange(len(days)) == 3).astype(float)

    def storm_scenario(weather):
        def receipts(frame, calendar):
            flag = np.zeros(len(calendar)) if weather == "clear" else storms(calendar)
            return {"features": flag[None, :, None]}

        return receipts

    labels = {"clear": "No storm", "as_forecast": "As forecast", "long": "Lingers"}
    data = scenario.scenario_data("2026-06-30", baseline, runs, labels, storms, storm_scenario)
    assert data["order"] == ["clear", "as_forecast", "long"] and data["default"] == "as_forecast"
    assert data["dates"][0] == "2026-07-01" and len(data["dates"]) == 8
    assert set(data["scenarios"]) == {"as_forecast", "clear"}
    assert data["scenarios"]["as_forecast"]["storm"]["scenario"][3] is True
    assert not any(data["scenarios"]["clear"]["storm"]["scenario"])
    assert (
        data["scenarios"]["clear"]["storm"]["baseline"][3] is True
    )  # the baseline keeps its storm


def test_storm_labels_name_the_next_storm_after_the_snapshot():
    spans = pd.DataFrame(
        {
            "start": pd.to_datetime(["2026-05-10", "2026-07-05", "2026-09-01"]),
            "end": pd.to_datetime(["2026-05-15", "2026-07-10", "2026-09-06"]),
        }
    )
    start, labels = scenario.storm_labels(spans, np.datetime64("2026-06-30"))
    assert start == np.datetime64("2026-07-05")
    assert labels["as_forecast"] == "Storm as forecast (Jul 5–9)"
    assert labels["long"] == "Storm lingers ten days (Jul 5–14)"


# --- The survival convolution ---------------------------------------------------------


def test_follow_sales_spreads_each_start_day_through_its_own_weekday_curve():
    # Half of a day's units start a return that day, half the next. Each start then arrives
    # the same day with a share that identifies the weekday curve used: (weekday + 1) / 10.
    kernels = {
        "initiation": np.array([0.5, 0.5]),
        "receipt": np.array([[(w + 1) / 10, 0.0] for w in range(7)]),
    }
    monday = "2026-06-29"
    followed = chain.follow_sales(kernels, [10.0, 20.0], monday, 4)
    np.testing.assert_allclose(followed["started"], [[5, 5, 0, 0], [0, 10, 10, 0]])
    # Mon 5 x 0.1, Tue 5 x 0.2; the second sale day's starts land Tue and Wed.
    np.testing.assert_allclose(followed["received"], [[0.5, 1.0, 0, 0], [0, 2.0, 3.0, 0]])
