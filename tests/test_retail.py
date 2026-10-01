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


def test_custom_receipt_family_preserves_eventual_outstanding_expectation():
    def fixed_initiation(observations, shared):
        return ttenet.StageParameters(
            jnp.array([0.0]), jnp.empty(0), np.log(0.4 / 0.6), jnp.empty(0)
        )

    def receipt_family(inputs, shared):
        return (
            ttenet.TimingLaw(
                jnp.full_like(inputs.ages, np.log(0.5), dtype=float),
                jnp.full_like(inputs.ages, np.log(0.5), dtype=float),
            ),
            jnp.full(inputs.ages.shape[-1], np.log(0.4 / 0.6)),
        )

    receipt_family.tail_behavior = {"kind": "proper"}
    model = ttenet.RetailReturnModel(
        initiation=ttenet.CureProcess(
            age_bins=1, deadline_days=3, parameter_model=fixed_initiation
        ),
        receipt=ttenet.CureProcess(event_time_model=receipt_family),
    )
    units = pd.DataFrame(
        {
            "item_id": ["pending"],
            "sale_date": ["2026-01-01"],
            "initiation_date": ["2026-01-03"],
            "receipt_date": [None],
        }
    )
    fitted = model.fit(_data(units), num_steps=1, num_samples=3)
    result = fitted.forecast(horizon=4, future_sales=_no_sales(), seed=3)
    # Receipt was event-free on ages zero and one. Its remaining susceptible
    # share is .4 * .5**2 / (.6 + .4 * .5**2) = 1/7; no sale remains uninitiated.
    np.testing.assert_allclose(result.expected_open_receipts, np.full(3, 1 / 7), rtol=1e-6)
    np.testing.assert_array_equal(result.expected_uninitiated_receipts, np.zeros(3))
    np.testing.assert_allclose(result.expected_existing_receipts, np.full(3, 1 / 7), rtol=1e-6)
    assert np.issubdtype(result.receipts.dtype, np.integer)
    assert np.all(result.receipts.sum(axis=1) <= 1)


def test_finite_initiation_support_extends_eventual_retail_covariates():
    def initiation_family(inputs, shared):
        log_mass = jnp.concatenate([jnp.full(10, -jnp.inf), jnp.zeros(1)])
        law = ttenet.timing_from_log_masses(log_mass, -jnp.inf, inputs.ages)
        return law, jnp.full(inputs.ages.shape[-1], 50.0)

    initiation_family.tail_behavior = {"kind": "finite", "last_age": 10}

    def receipt_parameters(observations, shared):
        return ttenet.StageParameters(jnp.zeros(1), jnp.empty(0), np.log(0.4 / 0.6), jnp.empty(0))

    model = ttenet.RetailReturnModel(
        initiation=ttenet.CureProcess(event_time_model=initiation_family),
        receipt=ttenet.CureProcess(age_bins=1, parameter_model=receipt_parameters),
    )
    units = pd.DataFrame({"item_id": ["sold"], "sale_date": ["2026-01-04"]})
    fitted = model.fit(_data(units), num_steps=1, num_samples=3)
    result = fitted.forecast(horizon=1, future_sales=_no_sales())
    np.testing.assert_array_equal(result.initiations, np.zeros((3, 1), dtype=int))
    np.testing.assert_allclose(result.expected_existing_receipts, [0.4] * 3, rtol=1e-6)
    closed = fitted.forecast(
        horizon=1,
        future_sales=_no_sales(),
        covariates={
            "initiations": lambda frame, dates: {"allowed": dates != np.datetime64("2026-01-14")}
        },
    )
    np.testing.assert_array_equal(closed.expected_existing_receipts, np.zeros(3))
