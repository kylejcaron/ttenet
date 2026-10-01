"""Joint posterior learning and dated lineage, not container plumbing."""

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import pytest

import ttenet
from ttenet.event_times import log1mexp


def _certain_parameters(observations, shared):
    return ttenet.StageParameters(jnp.array([50.0]), jnp.empty(0), 50.0, jnp.empty(0))


def _certain_process(**kwargs):
    return ttenet.CureProcess(age_bins=1, parameter_model=_certain_parameters, **kwargs)


def _empty_future():
    return ttenet.SalesForecast.from_frame(pd.DataFrame(columns=["sale_date", "quantity"]))


@pytest.mark.parametrize("mode", ["joint", "modular"])
def test_arbitrary_descendants_and_siblings_use_parent_dates_and_keep_root_cohorts(mode):
    root = ttenet.CountNode("sales")
    initiation = ttenet.EventNode("initiations", root, _certain_process(deadline_days=1))
    receipt = ttenet.EventNode("receipts", initiation, _certain_process())
    inspection = ttenet.EventNode(
        "inspections", receipt, _certain_process(deadline_days=1), event_column="inspection_date"
    )
    restock = ttenet.EventNode(
        "restocks", receipt, _certain_process(deadline_days=0), event_column="restock_date"
    )
    model = ttenet.ForecastNetwork(nodes=[inspection, restock, root, receipt, initiation])
    units = pd.DataFrame(
        columns=["item_id", "sale_date", "product", "inspection_date", "restock_date"]
    )
    data = ttenet.RetailData.from_units(
        units,
        as_of="2026-01-01",
        calendar=["2026-01-01"],
        event_columns={"inspections": "inspection_date", "restocks": "restock_date"},
    )
    covariates = {
        "receipts": lambda frame, days: {
            "allowed": (frame["product"].to_numpy()[:, None] == 0)
            & (days[None, :] >= np.datetime64("2026-01-03"))
        },
        "inspections": lambda frame, days: {"allowed": days >= np.datetime64("2026-01-04")},
    }
    fitted = model.fit(data, covariates=covariates, num_steps=1, num_samples=3, mode=mode)
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
    # Both distinct sibling events may happen for the same physical units.
    np.testing.assert_array_equal(result.nodes["restocks"].events, result.nodes["receipts"].events)


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


def _weibull_receipts(inputs, shared):
    """Shape-2 Weibull receipt timing with a sampled scale and certain susceptibility."""
    scale = numpyro.sample("scale", dist.LogNormal(np.log(3.0), 0.3))
    age = jnp.maximum(inputs.ages, 0)
    log_stay = -(((age + 1) / scale) ** 2 - (age / scale) ** 2)
    return ttenet.TimingLaw(log1mexp(log_stay), log_stay), jnp.full(inputs.ages.shape[-1], 50.0)


_weibull_receipts.tail_behavior = {"kind": "proper"}


def test_mixed_family_network_propagates_through_the_custom_stage():
    """Default -> Weibull -> default: the custom stage keeps its own named posterior."""
    root = ttenet.CountNode("sales")
    initiation = ttenet.EventNode("initiations", root, _certain_process(deadline_days=1))
    receipt = ttenet.EventNode(
        "receipts", initiation, ttenet.CureProcess(event_time_model=_weibull_receipts)
    )
    inspection = ttenet.EventNode(
        "inspections", receipt, _certain_process(deadline_days=1), event_column="inspection_date"
    )
    model = ttenet.ForecastNetwork(nodes=[root, initiation, receipt, inspection])
    units = pd.DataFrame(
        {
            "item_id": ["a", "b", "c", "d"],
            "sale_date": ["2026-01-01", "2026-01-01", "2026-01-03", "2026-01-05"],
            "initiation_date": ["2026-01-01", "2026-01-01", "2026-01-03", "2026-01-05"],
            "receipt_date": ["2026-01-03", "2026-01-06", None, None],
            "inspection_date": ["2026-01-03", "2026-01-06", None, None],
        }
    )
    data = ttenet.RetailData.from_units(
        units,
        as_of="2026-01-10",
        calendar=ttenet.date_grid("2026-01-01", "2026-01-10"),
        event_columns={"inspections": "inspection_date"},
    )
    fitted = model.fit(data, num_steps=20, num_samples=5, seed=4)
    fit = fitted.stage_fits["receipts"]
    assert fit.event_time_model is _weibull_receipts
    assert set(fit.parameters) == {"scale"} and fit.parameters["scale"].shape == (5,)
    assert fit.num_samples == 5 and fit.draws == 5 and fit.shared is None
    assert isinstance(fitted.stage_fits["initiations"].parameters, ttenet.StageParameters)
    assert isinstance(fitted.stage_fits["inspections"].parameters, ttenet.StageParameters)
    inputs = ttenet.TimingInputs(
        ages=jnp.array([[0, 3], [1, 4]]),
        features=jnp.zeros((2, 2, 0)),
        cure_features=jnp.zeros((2, 0)),
    )
    timing, logits = fit.timing(inputs, draw=1)
    scale = fit.parameters["scale"][1]
    expected = -(((inputs.ages + 1) / scale) ** 2 - (inputs.ages / scale) ** 2)
    np.testing.assert_allclose(timing.log_survival_step, expected, rtol=1e-5)
    np.testing.assert_allclose(logits, 50.0)
    future = ttenet.SalesForecast(
        pd.DataFrame({"item_id": ["p0", "p1"], "sale_date": ["2026-01-11", "2026-01-12"]}),
        np.array([[30, 20]] * 5),
    )
    result = fitted.forecast(horizon=14, future_sales=future, seed=7)
    initiations = result.counts["initiations"]
    receipts = result.counts["receipts"]
    inspections = result.counts["inspections"]
    np.testing.assert_array_equal(initiations[:, :2], [[30, 20]] * 5)
    assert np.issubdtype(receipts.dtype, np.integer) and (receipts >= 0).all()
    assert (receipts.cumsum(axis=1) <= initiations.cumsum(axis=1) + 2).all()
    np.testing.assert_array_equal(
        result.nodes["receipts"].pending + receipts.cumsum(axis=1),
        initiations.cumsum(axis=1) + 2,
    )
    np.testing.assert_array_equal(inspections, receipts)
    assert receipts.sum() > 0
    # Lineage: the historical open units (c, d) are the only cohorts before new sales.
    assert not result.nodes["receipts"].events[:, :, :2].any()


def _derived_shared_model():
    demand = numpyro.sample("demand", dist.Normal(0.0, 0.5))
    return {"demand": demand, "rate": jnp.exp(demand)}


def _shared_rate_family(inputs, shared):
    log_stay = jnp.broadcast_to(-shared["rate"], inputs.ages.shape)
    return ttenet.TimingLaw(log1mexp(log_stay), log_stay), jnp.full(inputs.ages.shape[-1], 0.0)


_shared_rate_family.tail_behavior = {"kind": "proper"}


def test_custom_family_fit_records_the_shared_models_derived_values_per_draw():
    root = ttenet.CountNode("sales")
    child = ttenet.EventNode(
        "initiations",
        root,
        ttenet.CureProcess(deadline_days=3, event_time_model=_shared_rate_family),
    )
    model = ttenet.ForecastNetwork([root, child], shared_model=_derived_shared_model)
    units = pd.DataFrame(
        {
            "item_id": range(12),
            "sale_date": ["2026-01-01"] * 12,
            "initiation_date": ["2026-01-01"] * 4 + ["2026-01-02"] * 2 + [None] * 6,
        }
    )
    data = ttenet.RetailData.from_units(units, as_of="2026-01-04", calendar=["2026-01-04"])
    fitted = model.fit(data, num_steps=40, num_samples=6, seed=5)
    fit = fitted.stage_fits["initiations"]
    assert fit.event_time_model is _shared_rate_family
    assert dict(fit.parameters) == {}
    assert fit.num_samples == 6 and fit.draws == 6
    assert set(fit.shared) == {"demand", "rate"}
    np.testing.assert_array_equal(fit.shared["demand"], fitted.posterior["shared/demand"])
    np.testing.assert_allclose(fit.shared["rate"], np.exp(fit.shared["demand"]), rtol=1e-6)
    assert len(np.unique(np.asarray(fit.shared["demand"]))) > 1
    inputs = ttenet.TimingInputs(
        ages=jnp.array([[0, 2], [1, 3]]),
        features=jnp.zeros((2, 2, 0)),
        cure_features=jnp.zeros((2, 0)),
    )
    for draw in range(6):
        timing, _ = fit.timing(inputs, draw=draw)
        np.testing.assert_allclose(timing.log_survival_step, -np.exp(fit.shared["demand"][draw]))
