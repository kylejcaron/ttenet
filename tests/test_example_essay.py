"""Contracts for the returns essay's data code: the synthetic world, the numbers computed from
it, and the JSON the interactive figures draw. None of this touches marimo or a plotting
library, so a wrong number or a broken payload fails here rather than as a wrong figure."""

from types import SimpleNamespace

import numpy as np
import numpyro.distributions as dist
import pandas as pd
import pytest
import retail_returns_analysis as analysis
import retail_returns_chain as chain
import retail_returns_scenario as scenario
import retail_returns_story as story
import retail_returns_world as world

import ttenet

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
    assert age.between(0, 90).all(), "the policy allows starting a return through day 90"
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


def test_status_boundary_follows_the_inclusive_day_90_deadline():
    as_of = np.datetime64("2026-04-30")
    sold = [as_of - np.timedelta64(n, "D") for n in (89, 90, 89, 89)]
    units = _ledger(
        sale_date=pd.to_datetime(sold),
        initiation_date=pd.to_datetime([None, None, as_of - np.timedelta64(5, "D"), as_of]),
        receipt_date=pd.to_datetime([None, None, None, as_of]),
    )
    status = analysis.classify_units(units, as_of)
    # Age 89 can still start a return tomorrow; age 90 has had its last chance by end of day.
    assert list(status) == ["Could still return", "Window closed", "Return in transit", "Received"]


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


def test_baseline_scores_treat_the_baseline_as_a_point_forecast():
    actual = np.tile([4.0, 2.0], 7)  # long enough to include the storm fortnight (days 5-13)
    held_out = SimpleNamespace(receipts=pd.Series(actual))
    forecast = SimpleNamespace(receipts=np.tile(actual, (6, 1)) + np.array([[1], [-1]] * 3))
    baseline = pd.Series(np.full(14, 3.0))
    scores = analysis.baseline_scores(forecast, held_out, baseline, "Model").set_index("Method")
    point = scores.loc["Weekday average (last 8 weeks)"]
    assert point.crps == point.mae == pytest.approx(1.0)
    assert np.isnan(point.total_lo) and np.isnan(point.total_hi)
    model = scores.loc["Model"]
    assert model.total == pytest.approx(42.0)
    assert model.total_lo <= model.total <= model.total_hi


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


# --- The purchase story ----------------------------------------------------------------


def _storm_on(*ages):
    def storms(days):
        age = (np.asarray(days, dtype="datetime64[D]") - np.datetime64("2026-01-01")).astype(int)
        return np.isin(age, ages).astype(float)

    return storms


def _story_ledger(extra=()):
    sale = pd.Timestamp("2026-01-01")
    rows = [
        # (initiated age, received age): A candidates, B candidates, C candidates
        (3, 5),  # in transit two days, across the storm
        (10, 20),  # in transit longer, no storm: loses to the one with storm days
        (None, None),  # B: never started
        (80, None),  # C candidate
        (60, None),
        (None, None),
        *extra,
    ]

    def day(age):
        return None if age is None else sale + pd.Timedelta(days=age)

    return pd.DataFrame(
        {
            "item_id": [f"sale_{i}" for i in range(len(rows))],
            "sale_date": sale,
            "initiation_date": pd.to_datetime([day(i) for i, _ in rows]),
            "receipt_date": pd.to_datetime([day(r) for _, r in rows]),
            "product": [float(i % 2) for i in range(len(rows))],
        }
    )


def test_purchase_story_picks_the_three_cases_by_rule():
    as_of = np.datetime64("2026-01-01") + np.timedelta64(100, "D")
    payload = story.purchase_story(_story_ledger(), as_of, _storm_on(4, 5))
    a, b, c = payload["records"]
    assert (a["init"], a["recv"]) == (3, 5)  # the wait that overlaps the storm beats the longer one
    assert a["storm"] == [4, 5] and a["title"] == "Received during a storm"
    assert (b["init"], b["recv"], b["order"]) == (None, None, "SYN-2")  # the first quiet unit
    assert (c["init"], c["recv"]) == (80, None)  # the latest start without a receipt
    assert payload["cohort_units"] == 6 and payload["initial_day"] == 3
    assert payload["sale"] == "2026-01-01" and payload["snapshot"] == str(as_of)


def test_purchase_story_without_a_storm_prefers_the_longest_wait():
    as_of = np.datetime64("2026-01-01") + np.timedelta64(100, "D")
    payload = story.purchase_story(_story_ledger(), as_of, _storm_on())
    assert payload["records"][0]["init"] == 10 and payload["records"][0]["storm"] is None


def test_purchase_story_needs_all_three_cases_and_a_valid_ledger():
    as_of = np.datetime64("2026-01-01") + np.timedelta64(100, "D")
    no_quiet = _story_ledger().query("initiation_date.notna()")
    with pytest.raises(ValueError, match="no initiation through day 90"):
        story.purchase_story(no_quiet, as_of, _storm_on())
    with pytest.raises(ValueError, match="no sales on"):
        story.purchase_story(_story_ledger(), as_of + np.timedelta64(1, "D"), _storm_on())
    late = _story_ledger(extra=[(95, None)])
    with pytest.raises(ValueError, match="outside days 0-90"):
        story.purchase_story(late, as_of, _storm_on())
    with pytest.raises(ValueError, match="missing columns"):
        story.purchase_story(_story_ledger().drop(columns="product"), as_of, _storm_on())


# --- The chain explainer, from a real (tiny) fit ---------------------------------------


@pytest.fixture(scope="module")
def tiny_chain():
    truth, _ = world.simulate_retail(seed=4, training_days=45, horizon=7, volume=2.0)
    as_of = np.datetime64("2026-01-01") + np.timedelta64(44, "D")
    data = ttenet.RetailData.from_units(
        truth.drop(columns="abandoned_truth"),
        as_of=as_of,
        calendar=ttenet.date_grid("2026-01-11", as_of),
        group_by=["product"],
    )
    model = ttenet.RetailReturnModel(
        sales=ttenet.CountProcess(model=world.sales_model),
        initiation=ttenet.EventProcess(age_bins=4, deadline_days=90),
        receipt=ttenet.EventProcess(
            family=ttenet.WeibullFamily(
                scale_prior=dist.LogNormal(1.8, 0.3),
                shape_prior=dist.LogNormal(0.7, 0.2),
                susceptibility_logit_prior=dist.Normal(1.0, 1.0),
            ),
            allowed_weekdays=range(5),
        ),
    )
    fitted = model.fit(
        data,
        covariates={
            "sales": world.sales_covariates(data.calendar),
            "initiations": world.initiation_covariates,
            "receipts": world.receipt_covariates,
        },
        mode="joint",
        num_steps=20,
        num_samples=12,
        seed=1,
    )
    dates = ttenet.date_grid(as_of + np.timedelta64(1, "D"), as_of + np.timedelta64(7, "D"))
    forecast = fitted.forecast(
        horizon=7, covariates={"sales": world.sales_covariates(dates)}, seed=2
    )
    return fitted, forecast


def test_chain_timing_bars_and_remainders_account_for_every_unit(tiny_chain):
    fitted, forecast = tiny_chain
    sales = analysis.daily_totals(forecast.sales, forecast.dates)
    payload = chain.chain_data(fitted, forecast, sales, storm=np.zeros(7, dtype=bool))
    for stage in ("initiation", "receipt"):
        part = payload[stage]
        # Shares on a plotted day + never + still waiting after the last day = every unit.
        covered = part["total"]["mean"] + part["never"]["mean"] + part["beyond"]["mean"]
        assert covered == pytest.approx(1.0, abs=1e-5)
        assert len(part["prob"]["mean"]) == len(part["ages"])
    assert payload["initiation"]["policy_days"] == 90
    assert payload["draws"]["forecast"] == forecast.receipts.shape[0]
    assert len(payload["datesISO"]) == 7 and payload["weekend"].count(True) in (1, 2)
    assert payload["totals"]["receipt_forecast"]["mean"] == pytest.approx(
        forecast.receipts.sum(axis=1).mean()
    )


def test_chain_data_validates_what_the_figure_will_draw(tiny_chain):
    fitted, forecast = tiny_chain
    sales = analysis.daily_totals(forecast.sales, forecast.dates)
    with pytest.raises(ValueError, match="storm must be"):
        chain.chain_data(fitted, forecast, sales, storm=np.zeros(3, dtype=bool))
    with pytest.raises(ValueError, match="sales_draws"):
        chain.chain_data(fitted, forecast, sales[:, :3])
    with pytest.raises(ValueError, match="receipt_days"):
        chain.chain_data(fitted, forecast, sales, receipt_days=0)
