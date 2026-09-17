"""Joint posterior learning and dated lineage, not container plumbing."""

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import pytest

import ttenet


def _certain_parameters(observations, shared):
    return ttenet.StageParameters(jnp.array([50.0]), jnp.empty(0), 50.0, jnp.empty(0))


def _certain_process(**kwargs):
    return ttenet.CureProcess(age_bins=1, parameter_model=_certain_parameters, **kwargs)


def _empty_future():
    return ttenet.SalesForecast.from_frame(pd.DataFrame(columns=["sale_date", "quantity"]))


def test_arbitrary_descendant_uses_parent_date_and_keeps_root_cohort():
    root = ttenet.CountNode("sales")
    initiation = ttenet.EventNode("initiations", root, _certain_process(deadline_days=1))
    receipt = ttenet.EventNode("receipts", initiation, _certain_process())
    inspection = ttenet.EventNode(
        "inspections", receipt, _certain_process(deadline_days=1), event_column="inspection_date"
    )
    model = ttenet.ForecastNetwork(nodes=[inspection, root, receipt, initiation])
    units = pd.DataFrame(columns=["item_id", "sale_date", "product", "inspection_date"])
    data = ttenet.RetailData.from_units(
        units,
        as_of="2026-01-01",
        calendar=["2026-01-01"],
        event_columns={"inspections": "inspection_date"},
    )
    covariates = {
        "receipts": lambda frame, days: {
            "allowed": (frame["product"].to_numpy()[:, None] == 0)
            & (days[None, :] >= np.datetime64("2026-01-03"))
        },
        "inspections": lambda frame, days: {"allowed": days >= np.datetime64("2026-01-04")},
    }
    fitted = model.fit(data, covariates=covariates, num_steps=1, num_samples=3)
    future = ttenet.SalesForecast(
        pd.DataFrame(
            {
                "item_id": ["p0", "p1"],
                "sale_date": ["2026-01-02"] * 2,
                "product": [0, 1],
            }
        ),
        np.array([[3, 7], [5, 11], [1, 0]]),
    )
    result = fitted.forecast(horizon=4, future_sales=future, seed=9)
    np.testing.assert_array_equal(
        result.counts["sales"], [[10, 0, 0, 0], [16, 0, 0, 0], [1, 0, 0, 0]]
    )
    np.testing.assert_array_equal(
        result.counts["receipts"], [[0, 3, 0, 0], [0, 5, 0, 0], [0, 1, 0, 0]]
    )
    np.testing.assert_array_equal(
        result.counts["inspections"], [[0, 0, 3, 0], [0, 0, 5, 0], [0, 0, 1, 0]]
    )
    np.testing.assert_array_equal(
        result.nodes["receipts"].pending + result.counts["receipts"].cumsum(axis=1),
        result.counts["initiations"].cumsum(axis=1),
    )
    assert not result.nodes["inspections"].events[:, :, 1].any()


def _shared_model():
    return {"demand": numpyro.sample("demand", dist.Normal(0, 2))}


def _shared_sales(covariates, data=None, *, shared):
    nf = pytest.importorskip("numpyro_forecast")
    horizon = nf.Horizon.from_data(covariates, data)
    rate = jnp.broadcast_to(jnp.exp(shared["demand"]), (covariates.shape[-2], 1))
    nf.predict(horizon, lambda value: dist.Poisson(value), rate)


def _shared_event(observations, shared):
    return ttenet.StageParameters(
        jnp.reshape(shared["demand"], (1,)), jnp.empty(0), 50.0, jnp.empty(0)
    )


def test_return_observations_update_shared_sales_posterior():
    """Identical sales, different return evidence: sales predictions must change."""
    pytest.importorskip("numpyro_forecast")
    root = ttenet.CountNode("sales", ttenet.CountProcess(_shared_sales))
    child = ttenet.EventNode(
        "initiations",
        root,
        ttenet.CureProcess(age_bins=1, deadline_days=0, parameter_model=_shared_event),
    )
    model = ttenet.ForecastNetwork([root, child], shared_model=_shared_model)
    base = pd.DataFrame({"item_id": range(20), "sale_date": ["2026-01-01"] * 20})
    predictions = []
    for initiated in (False, True):
        units = base.assign(initiation_date="2026-01-01" if initiated else None)
        data = ttenet.RetailData.from_units(units, as_of="2026-01-01", calendar=["2026-01-01"])
        fitted = model.fit(data, num_steps=150, num_samples=100, seed=12)
        result = fitted.forecast(horizon=1, seed=2)
        predictions.append(result.counts["sales"].mean())
        # Reordered same-posterior paths remain paired, and replace rather than add sales.
        reordered = ttenet.SalesForecast(
            result.sales.cohorts,
            result.sales.counts[::-1],
            draw_ids=result.sales.draw_ids[::-1],
            posterior_id=result.sales.posterior_id,
        )
        scenario = fitted.forecast(horizon=1, future_sales=reordered, seed=2)
        np.testing.assert_array_equal(scenario.counts["sales"], result.counts["sales"])
        np.testing.assert_array_equal(scenario.counts["initiations"], result.counts["initiations"])
    assert predictions[1] > 2 * predictions[0]
    with pytest.raises(ValueError, match="shared"):
        model.fit(data, mode="modular", num_steps=1, num_samples=1)


def test_starting_population_receives_without_in_window_sales_or_clock_reset():
    units = pd.DataFrame(
        {
            "item_id": ["old_open", "expired", "recent", "completed"],
            "sale_date": ["2025-10-01", "2025-10-01", "2025-12-31", "2025-12-20"],
            "initiation_date": ["2025-12-28", None, None, "2025-12-21"],
            "receipt_date": [None, None, None, "2025-12-22"],
        }
    )
    data = ttenet.RetailData.from_units(units, as_of="2026-01-01", calendar=["2026-01-01"])
    model = ttenet.RetailReturnModel(
        initiation=_certain_process(deadline_days=90),
        receipt=_certain_process(),
    )
    covariates = {
        "initiations": lambda frame, days: {
            "allowed": (frame["item_id"].to_numpy()[:, None] != "recent")
            | (days[None, :] >= np.datetime64("2026-01-02"))
        },
        "receipts": lambda frame, days: {
            "allowed": (frame["item_id"].to_numpy()[:, None] != "old_open")
            | (days[None, :] >= np.datetime64("2026-01-02"))
        },
    }
    fitted = model.fit(data, covariates=covariates, num_steps=1, num_samples=3)
    result = fitted.forecast(horizon=2, future_sales=_empty_future())
    assert data.sales.sum() == 0
    np.testing.assert_array_equal(result.initiations, [[1, 0]] * 3)
    np.testing.assert_array_equal(result.receipts, [[2, 0]] * 3)
    np.testing.assert_array_equal(result.open_returns, 0)
    np.testing.assert_allclose(result.expected_existing_receipts, 2)
    with pytest.raises(ValueError, match="sales"):
        fitted.forecast(horizon=2)


def test_internal_sales_identifiers_cannot_collide_with_historical_units():
    nf = pytest.importorskip("numpyro_forecast")

    def count_model(covariates, data=None):
        horizon = nf.Horizon.from_data(covariates, data)
        nf.predict(
            horizon, lambda rate: dist.Poisson(rate), jnp.full((covariates.shape[-2], 1), 4.0)
        )

    data = ttenet.RetailData.from_units(
        pd.DataFrame(
            {
                "item_id": ["future_0", "future1_0", "ordinary"],
                "sale_date": ["2026-01-01"] * 3,
                "initiation_date": ["2026-01-01"] * 3,
            }
        ),
        as_of="2026-01-01",
        calendar=["2026-01-01"],
    )
    root = ttenet.CountNode("sales", ttenet.CountProcess(count_model))
    child = ttenet.EventNode("initiations", root, _certain_process(deadline_days=0))
    fitted = ttenet.ForecastNetwork([root, child]).fit(data, num_steps=1, num_samples=4)
    result = fitted.forecast(horizon=1)
    assert set(result.sales.cohorts.item_id).isdisjoint(data.units.item_id)
    np.testing.assert_array_equal(result.counts["initiations"], result.counts["sales"])


def test_numpyro_parameter_only_count_model_forecasts_at_the_fitted_rate():
    nf = pytest.importorskip("numpyro_forecast")

    def count_model(covariates, data=None):
        rate = numpyro.param("rate", 1.0, constraint=dist.constraints.positive)
        horizon = nf.Horizon.from_data(covariates, data)
        rates = jnp.broadcast_to(rate, (covariates.shape[-2], 1))
        nf.predict(horizon, lambda value: dist.Poisson(value), rates)

    data = ttenet.RetailData.from_units(
        pd.DataFrame({"item_id": range(50), "sale_date": ["2026-01-01"] * 50}),
        as_of="2026-01-01",
        calendar=["2026-01-01"],
    )
    model = ttenet.ForecastNetwork([ttenet.CountNode("sales", ttenet.CountProcess(count_model))])
    fitted = model.fit(data, num_steps=180, num_samples=100, learning_rate=0.05, seed=9)
    result = fitted.forecast(horizon=1, seed=8)
    assert result.counts["sales"].mean() > 25


def test_optimized_event_parameters_are_resolved_before_simulating():
    def event_prior(observations, shared):
        logits = numpyro.param("age", jnp.array([-4.0]))
        return ttenet.StageParameters(logits, jnp.empty(0), 50.0, jnp.empty(0))

    data = ttenet.RetailData.from_units(
        pd.DataFrame(
            {
                "item_id": range(20),
                "sale_date": ["2026-01-01"] * 20,
                "initiation_date": ["2026-01-01"] * 20,
            }
        ),
        as_of="2026-01-01",
        calendar=["2026-01-01"],
    )
    root = ttenet.CountNode("sales")
    child = ttenet.EventNode(
        "initiations",
        root,
        ttenet.CureProcess(age_bins=1, deadline_days=0, parameter_model=event_prior),
    )
    fitted = ttenet.ForecastNetwork([root, child]).fit(
        data,
        num_steps=150,
        num_samples=10,
        learning_rate=0.05,
    )
    future = ttenet.SalesForecast.from_frame(
        pd.DataFrame({"sale_date": ["2026-01-02"], "quantity": [1000]}),
    )
    result = fitted.forecast(horizon=1, future_sales=future)
    assert result.counts["initiations"].mean() > 500
