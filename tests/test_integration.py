import numpy as np
import pytest

from ttenet.integration import sales_cohorts


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
    cohorts, counts = sales_cohorts(sales, dates)
    history = prepare_history(pd.DataFrame(columns=["item_id", "sale_date"]), as_of="2026-01-01")
    certain = StageParameters(np.array([50.0]), np.empty(0), 50.0, np.empty(0))
    result = forecast_returns(
        history,
        certain,
        certain,
        calendar=date_grid("2026-01-01", "2026-04-03"),
        horizon=2,
        future_sales=cohorts,
        future_counts=counts,
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
        sales_cohorts(np.array([[[value]]]), ["2026-10-01"])


def test_sales_calendar_must_match_forecast_time_axis():
    with pytest.raises(ValueError):
        sales_cohorts(np.ones((2, 3, 1)), ["2026-10-01"])
    with pytest.raises(ValueError):
        sales_cohorts(np.ones((2, 2, 1)), ["2026-10-01", "2026-10-01"])
