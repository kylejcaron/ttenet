"""Jointly fit sales and return processes, including a pre-window population.

Run: uv run python examples/retail_returns.py --steps 150 --draws 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import numpyro.distributions as dist
import pandas as pd
from retail_returns_world import (
    initiation_covariates,
    receipt_covariates,
    sales_covariates,
    sales_model,
    simulate_retail,
)

from ttenet import (
    CountProcess,
    EventProcess,
    RetailData,
    RetailReturnModel,
    WeibullFamily,
    date_grid,
    prepare_history,
)


def run(steps=150, draws=40, seed=21, plot=None):
    if steps <= 0 or draws <= 0:
        raise ValueError("steps and draws must be positive")
    training_days, horizon = 180, 28
    truth, sales = simulate_retail(seed, training_days, horizon)
    as_of = np.datetime64("2026-01-01") + np.timedelta64(training_days - 1, "D")
    # Only business observations reach the estimator; latent abandonment labels do not.
    observed = truth.drop(columns="abandoned_truth")
    training_start = np.datetime64("2026-01-31")
    data = RetailData.from_units(
        observed,
        as_of=as_of,
        calendar=date_grid(training_start, as_of),
        group_by=["product"],
    )
    history = prepare_history(data.units, as_of=as_of, policy_days=90)
    starting = prepare_history(
        observed,
        as_of=training_start - np.timedelta64(1, "D"),
        policy_days=90,
    )
    output_dates = date_grid(as_of + np.timedelta64(1, "D"), as_of + np.timedelta64(horizon, "D"))
    initiated = int(history.frame["initiation_date"].notna().sum())
    pre_window_sales = int((data.units.sale_date < pd.Timestamp(training_start)).sum())
    assert data.sales.sum() + pre_window_sales == len(history.frame)
    # Series order is carried explicitly by the adapter, never assumed from product labels.
    product_order = data.groups["product"].to_numpy(int)
    np.testing.assert_array_equal(data.sales, sales[30:training_days, product_order])
    print(
        f"Jointly fitting {int(data.sales.sum())} in-window sales and {initiated} initiations; "
        f"retaining {pre_window_sales} earlier unit histories",
        flush=True,
    )
    model = RetailReturnModel(
        sales=CountProcess(model=sales_model),
        initiation=EventProcess(age_bins=16, deadline_days=90),
        receipt=EventProcess(
            family=WeibullFamily(
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
            "sales": sales_covariates(data.calendar),
            "initiations": initiation_covariates,
            "receipts": receipt_covariates,
        },
        mode="joint",
        num_steps=steps,
        num_samples=draws,
        seed=seed,
    )
    if any(not np.isfinite(loss).all() for loss in fitted.losses.values()):
        raise RuntimeError("joint fitting produced nonfinite losses")
    # The network generates one SalesForecast and propagates that same draw through both stages.
    result = fitted.forecast(
        horizon=horizon,
        covariates={"sales": sales_covariates(output_dates)},
        seed=seed + 5,
    )
    output_weekend = pd.DatetimeIndex(result.dates).dayofweek >= 5
    assert np.all(result.receipts[:, output_weekend] == 0)
    assert np.all(result.initiations >= 0) and np.all(result.receipts >= 0)
    historical_open = int(
        (history.frame.initiation_date.notna() & history.frame.receipt_date.isna()).sum()
    )
    historical_eligible = int(history.frame.eligible.sum())
    assert np.all(
        result.initiations.sum(axis=1) <= historical_eligible + result.sales.counts.sum(axis=1)
    )
    assert np.all(result.receipts.sum(axis=1) <= historical_open + result.initiations.sum(axis=1))
    np.testing.assert_array_equal(
        result.open_returns + result.receipts.cumsum(axis=1),
        historical_open + result.initiations.cumsum(axis=1),
    )
    np.testing.assert_allclose(
        result.expected_existing_receipts,
        result.expected_open_receipts + result.expected_uninitiated_receipts,
    )
    actual_receipts = np.array(
        [(truth.receipt_date == pd.Timestamp(date)).sum() for date in result.dates]
    )
    predicted = result.receipts.mean(axis=0)
    # Empirical CRPS for counts, computed per forecast date from predictive draws.
    pairwise = np.abs(result.receipts[:, None, :] - result.receipts[None, :, :]).mean(axis=(0, 1))
    crps = (np.abs(result.receipts - actual_receipts).mean(axis=0) - 0.5 * pairwise).mean()
    summary = {
        "fit_mode": fitted.network.mode,
        "sales_window_start": str(training_start),
        "in_window_sales": int(data.sales.sum()),
        "pre_window_sales": pre_window_sales,
        "starting_eligible_uninitiated": int(starting.frame.eligible.sum()),
        "starting_open_returns": int(
            (starting.frame.initiation_date.notna() & starting.frame.receipt_date.isna()).sum()
        ),
        "historical_sales": len(history.frame),
        "eligible_uninitiated_at_origin": historical_eligible,
        "initiated_not_received_at_origin": historical_open,
        "expected_remaining_receipts_from_past_sales": float(
            result.expected_existing_receipts.mean()
        ),
        "expected_from_uninitiated": float(result.expected_uninitiated_receipts.mean()),
        "expected_from_open_returns": float(result.expected_open_receipts.mean()),
        "forecast_receipts_mean": float(result.receipts.sum(axis=1).mean()),
        "forecast_receipts_90pct_interval": np.quantile(
            result.receipts.sum(axis=1), [0.05, 0.95]
        ).tolist(),
        "held_out_receipts": int(actual_receipts.sum()),
        "daily_receipt_mae": float(np.abs(predicted - actual_receipts).mean()),
        "daily_receipt_crps": float(crps),
        "weekend_receipts": int(result.receipts[:, output_weekend].sum()),
        "abandoned_initiations_in_synthetic_truth": int(truth.abandoned_truth.sum()),
        "posterior_draws": draws,
        "finite_stage_and_sales_losses": True,
        "mass_conservation": True,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if plot:
        import matplotlib.pyplot as plt

        lo, hi = np.quantile(result.receipts, [0.05, 0.95], axis=0)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.fill_between(result.dates, lo, hi, alpha=0.25, label="90% predictive interval")
        ax.plot(result.dates, predicted, label="Mean predicted receipts")
        ax.plot(result.dates, actual_receipts, "o", label="Held-out receipts")
        ax.set(
            ylabel="Received units per day", title="Retail returns: joint sales and event network"
        )
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        destination = Path(plot)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination)
        plt.close(fig)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--plot", type=str)
    args = parser.parse_args()
    run(args.steps, args.draws, args.seed, args.plot)
