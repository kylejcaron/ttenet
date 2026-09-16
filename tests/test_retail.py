"""Behavioral contracts for reusable model configurations and fitted forecasts."""

import numpy as np
import pandas as pd
import pytest

import ttenet


def _sales():
    return pd.DataFrame(
        {
            "item_id": ["a", "b", "c", "d"],
            "sale_date": ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"],
            "initiation_date": ["2026-01-02", "2026-01-03", None, None],
            "receipt_date": ["2026-01-03", None, None, None],
        }
    )


def test_refitting_configuration_does_not_change_an_existing_forecast():
    model = ttenet.RetailReturnModel(policy_days=3, age_bins=2)
    calendar = ttenet.date_grid("2026-01-01", "2026-01-10")
    history = ttenet.prepare_history(_sales(), as_of="2026-01-04", policy_days=3)
    first = model.fit(history, calendar=calendar, num_steps=3, num_samples=8, seed=2)
    before = first.forecast(calendar=calendar, horizon=3, seed=9)

    # Caller edits and a subsequent fit must not replace an earlier fit's population.
    history.frame["initiation_date"] = pd.Timestamp("2026-01-03")
    history.frame["receipt_date"] = pd.Timestamp("2026-01-04")
    complete = _sales().assign(initiation_date="2026-01-03", receipt_date="2026-01-04")
    second = model.fit(
        complete,
        as_of="2026-01-04",
        calendar=calendar,
        num_steps=3,
        num_samples=8,
        seed=4,
    )
    after = first.forecast(calendar=calendar, horizon=3, seed=9)
    completed = second.forecast(calendar=calendar, horizon=3, seed=9)

    np.testing.assert_array_equal(after.receipts, before.receipts)
    np.testing.assert_array_equal(after.open_returns, before.open_returns)
    np.testing.assert_array_equal(
        after.expected_existing_receipts, before.expected_existing_receipts
    )
    np.testing.assert_array_equal(completed.expected_existing_receipts, np.zeros(8))
    np.testing.assert_array_equal(completed.receipts, np.zeros((8, 3), dtype=int))


def test_feature_builder_aligns_future_cohort_closures_and_uncertain_counts():
    def features(frame, calendar):
        return {
            "receipt_allowed": np.broadcast_to(
                frame["receiving_open"].to_numpy(bool)[:, None], (len(frame), len(calendar))
            )
            & (calendar >= np.datetime64("2026-01-06"))[None, :]
        }

    model = ttenet.RetailReturnModel(policy_days=3, feature_builder=features)
    history = ttenet.prepare_history(
        pd.DataFrame(
            {
                "item_id": ["old"],
                "sale_date": ["2026-01-01"],
                "initiation_date": ["2026-01-01"],
                "receipt_date": [None],
                "receiving_open": [False],
            }
        ),
        as_of="2026-01-04",
        policy_days=3,
    )
    parameters = ttenet.StageParameters(
        age_logits=np.array([[1000.0]]),
        beta=np.empty((1, 0)),
        cure_intercept=np.array([1000.0]),
        cure_beta=np.empty((1, 0)),
    )
    # Real fixed posterior parameters isolate the exact cohort transition contract.
    stage_fit = ttenet.StageFit(parameters, np.empty(0))
    fitted = ttenet.FittedRetailReturnModel(model, history, stage_fit, stage_fit)
    future = pd.DataFrame(
        {
            "item_id": ["open", "closed"],
            "sale_date": ["2026-01-05", "2026-01-05"],
            "quantity": [3, 7],
            "receiving_open": [True, False],
        }
    )
    result = fitted.forecast(
        calendar=ttenet.date_grid("2026-01-01", "2026-01-08").astype(str).tolist(),
        horizon=2,
        future_sales=future,
        future_counts=np.array([[3, 7], [5, 11]]),
        seed=1,
    )
    np.testing.assert_array_equal(result.initiations, [[10, 0], [16, 0]])
    np.testing.assert_array_equal(result.receipts, [[0, 3], [0, 5]])
    np.testing.assert_array_equal(result.open_returns, [[11, 8], [17, 12]])


def test_misspelled_feature_keys_cannot_silently_disable_closures():
    def misspelled_features(frame, calendar):
        return {"receipt_allowd": np.zeros(len(calendar), dtype=bool)}

    model = ttenet.RetailReturnModel(policy_days=3, feature_builder=misspelled_features)
    with pytest.raises(ValueError):
        model.fit(
            _sales(),
            as_of="2026-01-04",
            calendar=ttenet.date_grid("2026-01-01", "2026-01-04"),
            num_steps=1,
            num_samples=1,
        )


def test_prepared_history_cannot_silently_truncate_a_different_model_policy():
    history = ttenet.prepare_history(_sales(), as_of="2026-01-04", policy_days=3)
    model = ttenet.RetailReturnModel(policy_days=3.5)
    with pytest.raises(ValueError):
        model.fit(
            history,
            calendar=ttenet.date_grid("2026-01-01", "2026-01-04"),
            num_steps=1,
            num_samples=1,
        )
