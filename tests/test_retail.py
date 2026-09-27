"""Behavioral contracts for reusable configurations and fitted retail forecasts."""

import jax.numpy as jnp
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


def _data(units):
    return ttenet.RetailData.from_units(
        units,
        as_of="2026-01-04",
        calendar=ttenet.date_grid("2026-01-01", "2026-01-04"),
    )


def _no_sales():
    return ttenet.SalesForecast.from_frame(pd.DataFrame(columns=["sale_date", "quantity"]))


def test_refitting_configuration_does_not_change_an_existing_forecast():
    model = ttenet.RetailReturnModel(
        initiation=ttenet.CureProcess(age_bins=2, deadline_days=3),
        receipt=ttenet.CureProcess(age_bins=2),
    )
    data = _data(_sales())
    first = model.fit(data, num_steps=3, num_samples=8, seed=2)
    before = first.forecast(horizon=3, future_sales=_no_sales(), seed=9)
    data.units["initiation_date"] = pd.Timestamp("2026-01-03")
    data.units["receipt_date"] = pd.Timestamp("2026-01-04")
    complete = _data(_sales().assign(initiation_date="2026-01-03", receipt_date="2026-01-04"))
    second = model.fit(complete, num_steps=3, num_samples=8, seed=4, mode="modular")
    after = first.forecast(horizon=3, future_sales=_no_sales(), seed=9)
    completed = second.forecast(horizon=3, future_sales=_no_sales(), seed=9)
    np.testing.assert_array_equal(after.receipts, before.receipts)
    np.testing.assert_array_equal(after.open_returns, before.open_returns)
    np.testing.assert_array_equal(
        after.expected_existing_receipts, before.expected_existing_receipts
    )
    np.testing.assert_array_equal(completed.expected_existing_receipts, np.zeros(8))
    np.testing.assert_array_equal(completed.receipts, np.zeros((8, 3), dtype=int))


def _certain(observations, shared):
    return ttenet.StageParameters(jnp.array([1000.0]), jnp.empty(0), 1000.0, jnp.empty(0))


def test_future_cohort_closures_and_uncertain_counts_stay_aligned():
    def receipt_covariates(frame, calendar):
        return {
            "allowed": np.broadcast_to(
                frame["receiving_open"].to_numpy(bool)[:, None],
                (len(frame), len(calendar)),
            )
            & (calendar >= np.datetime64("2026-01-06"))[None, :]
        }

    model = ttenet.RetailReturnModel(
        initiation=ttenet.CureProcess(age_bins=1, deadline_days=3, parameter_model=_certain),
        receipt=ttenet.CureProcess(age_bins=1, parameter_model=_certain),
    )
    data = _data(
        pd.DataFrame(
            {
                "item_id": ["old"],
                "sale_date": ["2026-01-01"],
                "initiation_date": ["2026-01-01"],
                "receipt_date": [None],
                "receiving_open": [False],
            }
        )
    )
    fitted = model.fit(
        data,
        covariates={"receipts": receipt_covariates},
        num_steps=1,
        num_samples=2,
    )
    future = ttenet.SalesForecast(
        pd.DataFrame(
            {
                "item_id": ["open", "closed"],
                "sale_date": ["2026-01-05"] * 2,
                "receiving_open": [True, False],
            }
        ),
        np.array([[3, 7], [5, 11]]),
    )
    result = fitted.forecast(horizon=2, future_sales=future, seed=1)
    np.testing.assert_array_equal(result.initiations, [[10, 0], [16, 0]])
    np.testing.assert_array_equal(result.receipts, [[0, 3], [0, 5]])
    np.testing.assert_array_equal(result.open_returns, [[11, 8], [17, 12]])


def test_invalid_covariate_provider_cannot_silently_disable_closures():
    model = ttenet.RetailReturnModel()
    with pytest.raises(ValueError, match="unknown"):
        model.fit(_data(_sales()), covariates={"receipts": lambda frame, days: {"allowd": False}})
    with pytest.raises(TypeError, match="mapping"):
        model.fit(_data(_sales()), covariates={"receipts": lambda frame, days: None})


def test_policy_days_cannot_be_silently_truncated():
    with pytest.raises(ValueError):
        ttenet.CureProcess(deadline_days=3.5)
