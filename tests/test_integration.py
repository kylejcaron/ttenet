import numpy as np
import pandas as pd
import pytest

from ttenet.integration import SalesForecast


def test_real_numpyro_sales_uncertainty_reaches_return_counts():
    nf = pytest.importorskip("numpyro_forecast")
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    import pandas as pd
    from jax import random

    from ttenet import StageParameters, date_grid, forecast_returns, prepare_history

    def sales_model(covariates, data=None):
        h = nf.Horizon.from_data(covariates, data)
        rate = numpyro.sample("rate", dist.LogNormal(0, 1))
        nf.predict(h, lambda eta: dist.Poisson(eta), jnp.ones_like(covariates) * rate)

    sales = nf.forecast(
        random.PRNGKey(17),
        sales_model,
        {"rate": jnp.array([1.0, 100.0])},
        jnp.zeros((2, 1), dtype=jnp.int32),
        jnp.zeros((4, 1)),
    )
    dates = ["2026-01-02", "2026-01-03"]
    future_sales = SalesForecast.from_numpyro_forecast(sales, dates)
    history = prepare_history(pd.DataFrame(columns=["item_id", "sale_date"]), as_of="2026-01-01")
    certain = StageParameters(np.array([50.0]), np.empty(0), 50.0, np.empty(0))
    result = forecast_returns(
        history,
        certain,
        certain,
        calendar=date_grid("2026-01-01", "2026-04-03"),
        horizon=2,
        future_sales=future_sales,
        seed=4,
    )
    # Every sale initiates and arrives on the sale day in this limiting model.
    # Averaging the upstream sales draws would violate this per-path identity.
    np.testing.assert_array_equal(result.receipts, np.asarray(sales)[:, :, 0])
    assert result.receipts[0].sum() < 30
    assert result.receipts[1].sum() > 100


@pytest.mark.parametrize("value", [-1.0, 1.25, float("nan"), float("inf")])
def test_invalid_sales_counts_are_not_silently_rounded(value):
    with pytest.raises(ValueError):
        SalesForecast.from_numpyro_forecast(np.array([[[value]]]), ["2026-10-01"])


def test_sales_calendar_must_match_forecast_time_axis():
    with pytest.raises(ValueError):
        SalesForecast.from_numpyro_forecast(np.ones((2, 3, 1)), ["2026-10-01"])
    with pytest.raises(ValueError):
        SalesForecast.from_numpyro_forecast(np.ones((2, 2, 1)), ["2026-10-01", "2026-10-01"])


def test_long_sales_paths_align_by_draw_and_cohort_not_row_position():
    frame = pd.DataFrame(
        {
            "draw": ["z", "a", "a", "z"],
            "sale_date": ["2026-01-02", "2026-01-03", "2026-01-02", "2026-01-03"],
            "product": ["red", "blue", "red", "blue"],
            "quantity": [1, 70, 7, 10],
        }
    )
    sales = SalesForecast.from_frame(frame)
    round_trip = sales.to_frame()
    keys = ["draw", "sale_date", "product"]
    expected = (
        frame.assign(sale_date=pd.to_datetime(frame.sale_date))
        .set_index(keys)
        .quantity.sort_index()
    )
    actual = round_trip.set_index(keys).quantity.sort_index()
    pd.testing.assert_series_equal(actual, expected, check_index_type=False)
    with pytest.raises(ValueError, match="missing"):
        SalesForecast.from_frame(frame.iloc[:3])
    with pytest.raises(ValueError, match="duplicate"):
        SalesForecast.from_frame(pd.concat([frame, frame.iloc[:1]]))


def test_all_numpyro_groups_retain_their_date_and_receiving_rules():
    from ttenet import StageParameters, date_grid, forecast_returns, prepare_history

    samples = np.array([[[1, 2], [3, 4]], [[10, 20], [30, 40]]])
    sales = SalesForecast.from_numpyro_forecast(
        samples,
        ["2026-01-02", "2026-01-03"],
        groups=pd.DataFrame({"product": ["red", "blue"]}),
    )
    history = prepare_history(pd.DataFrame(columns=["item_id", "sale_date"]), as_of="2026-01-01")
    params = StageParameters(np.array([50.0]), np.empty(0), 50.0, np.empty(0))
    calendar = date_grid("2026-01-01", "2026-01-03")
    allowed = np.broadcast_to((sales.cohorts["product"] == "red").to_numpy()[:, None], (4, 3))
    result = forecast_returns(
        history,
        params,
        params,
        calendar=calendar,
        horizon=2,
        future_sales=sales,
        receipt_allowed=allowed,
    )
    np.testing.assert_array_equal(result.receipts, [[1, 3], [10, 30]])
    np.testing.assert_array_equal(result.open_returns, [[2, 6], [20, 60]])


def test_int64_cohort_counts_do_not_round_through_float64():
    from ttenet import StageParameters, date_grid, forecast_returns, prepare_history

    quantity = 2**53 + 1
    sales = SalesForecast.from_frame(
        pd.DataFrame({"sale_date": ["2026-01-02"], "quantity": [quantity]})
    )
    history = prepare_history(pd.DataFrame(columns=["item_id", "sale_date"]), as_of="2026-01-01")
    params = StageParameters(np.array([50.0]), np.empty(0), 50.0, np.empty(0))
    result = forecast_returns(
        history,
        params,
        params,
        calendar=date_grid("2026-01-01", "2026-01-02"),
        horizon=1,
        future_sales=sales,
    )
    assert int(result.receipts[0, 0]) == quantity
    maximum = np.iinfo(np.int64).max
    with pytest.raises(ValueError, match="total"):
        SalesForecast(
            pd.DataFrame({"item_id": ["a", "b"], "sale_date": ["2026-01-02"] * 2}),
            np.array([[maximum, maximum]], dtype=np.int64),
        )


def test_group_metadata_cannot_replace_forecast_dates():
    with pytest.raises(ValueError):
        SalesForecast.from_numpyro_forecast(
            np.ones((1, 2, 1)),
            ["2026-01-02", "2026-01-03"],
            groups=pd.DataFrame({"sale_date": ["2025-01-01"]}),
        )


def test_heterogeneous_draw_labels_keep_their_quantity_identity():
    source = SalesForecast(
        pd.DataFrame({"item_id": ["a"], "sale_date": ["2026-01-02"]}),
        np.array([[2], [7]]),
        draw_ids=[1, "1"],
    )
    restored = SalesForecast.from_frame(source.to_frame()).to_frame().set_index("draw")
    assert restored.loc[1, "quantity"] == 2
    assert restored.loc["1", "quantity"] == 7


def test_object_backed_integer_quantities_preserve_large_counts():
    source = SalesForecast.from_frame(
        pd.DataFrame(
            {
                "sale_date": ["2026-01-02"],
                "quantity": pd.Series([2**53 + 1], dtype=object),
            }
        )
    )
    assert int(source.to_frame().quantity.iloc[0]) == 2**53 + 1
