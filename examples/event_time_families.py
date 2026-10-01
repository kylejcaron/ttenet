"""Pluggable event-time families: the packaged discretized Weibull through the public API.

A timing family is an object with a NumPyro ``model(inputs, shared) -> EventLaw``
that samples its own named sites from ``TimingInputs`` (ages and regressors only)
and a static ``tail`` declaration. The package core applies administrative
exposure, calendar closures, cure marginalization, entry conditioning, native
observation and count propagation, so swapping the family never touches fitting,
the native distributions or the network.

``WeibullFamily`` is the packaged law: with scale ``lambda`` and shape ``k`` the
continuous survival is ``log S(a) = -(a / lambda) ** k`` for ``a >= 0``, so each
day's law is the ``[a, a + 1)`` bin of that curve -- a nonzero same-day trial, and
shape 2 makes the hazard rise with age -- a genuinely different law from the
default piecewise-constant logistic baseline, not a renamed constant hazard.

Run::

    uv run python examples/event_time_families.py
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pandas as pd

from ttenet import (
    CountNode,
    CureProcess,
    EventNode,
    ForecastNetwork,
    RetailData,
    SalesForecast,
    StageParameters,
    TimingInputs,
    WeibullFamily,
    date_grid,
    fit_stage,
    make_event_observations,
)
from ttenet.models import predict_stage

# --- Synthetic sale -> initiation -> receipt -> inspection history ----------------


def _simulate_units(rng, *, days, sales_per_day, as_of):
    start = np.datetime64("2026-01-01", "D")
    missing = np.datetime64("NaT", "D")
    sale = np.repeat(start + np.arange(days), sales_per_day)
    n = sale.size
    # Initiation: susceptible with probability .55, then a geometric delay (daily .12).
    initiates = rng.random(n) < 0.55
    initiation_delay = rng.geometric(0.12, n) - 1
    initiation = np.where(initiates & (initiation_delay <= 30), sale + initiation_delay, missing)
    # Receipt: Weibull(scale 6, shape 2) days after initiation, every initiation susceptible.
    receipt_delay = np.floor(6.0 * (-np.log(rng.random(n))) ** 0.5).astype(int)
    receipt = np.where(~np.isnat(initiation), initiation + receipt_delay, missing)
    # Inspection: usually the day after receipt (daily hazard .7).
    inspection_delay = rng.geometric(0.7, n) - 1
    inspection = np.where(~np.isnat(receipt), receipt + inspection_delay, missing)
    frame = pd.DataFrame(
        {
            "item_id": [f"unit_{i}" for i in range(n)],
            "sale_date": sale,
            "initiation_date": initiation,
            "receipt_date": receipt,
            "inspection_date": inspection,
        }
    )
    for column in ("initiation_date", "receipt_date", "inspection_date"):
        values = frame[column].to_numpy(dtype="datetime64[D]")
        frame[column] = np.where(values <= as_of, values, missing)
    return frame


def _summary(name, values):
    values = np.asarray(values, dtype=float)
    return f"{name}: mean {values.mean():.3f}, sd {values.std():.3f}"


def main():
    rng = np.random.default_rng(7)
    as_of = np.datetime64("2026-02-15", "D")
    units = _simulate_units(rng, days=40, sales_per_day=8, as_of=as_of)
    calendar = date_grid("2026-01-01", as_of)
    receipts = make_event_observations(
        units,
        origin_column="initiation_date",
        event_column="receipt_date",
        as_of=as_of,
        calendar=calendar,
    )
    print(
        f"receipt stage: {receipts.ages.shape[0]} initiated units, "
        f"{int((receipts.event_index >= 0).sum())} observed receipts"
    )

    # 1. Native fitting: the default family and the packaged Weibull fit the same
    #    observations; only the family argument differs.
    default = fit_stage(receipts, age_bins=12, num_steps=300, num_samples=60, seed=1)
    family = WeibullFamily(
        scale_prior=dist.LogNormal(np.log(5.0), 0.5),
        shape_prior=dist.LogNormal(np.log(2.0), 0.3),
    )
    weibull = fit_stage(receipts, family=family, num_steps=300, num_samples=60, seed=1)
    assert isinstance(default.parameters, StageParameters)
    print("default family posterior:", _summary("cure", default.parameters.cure_intercept))
    print("weibull family posterior sites:", sorted(weibull.parameters))
    for name in ("scale", "shape", "cure_intercept"):
        print("  " + _summary(name, weibull.parameters[name]))
    print("  simulated truth: scale 6, shape 2, every initiated unit susceptible")

    # 2. In-sample prediction: posterior predictive trajectories follow the fitted
    #    law on the same exposure, not the recorded outcomes.
    observed = np.asarray(receipts.event_index)
    paths = np.asarray(predict_stage(weibull, receipts, seed=3))
    predicted_index = np.where(paths.sum(axis=1) == 1, paths.argmax(axis=1), -1)
    fired = observed >= 0
    agreement = (predicted_index[:, fired] == observed[fired]).mean()
    print(
        f"in-sample predictive: {paths.shape[0]} draws x {paths.shape[1]} days x "
        f"{paths.shape[2]} units; observed receipt dates reproduced in "
        f"{agreement:.1%} of draws (an echo would give 100%)"
    )
    print(
        "  mean predicted receipts per unit "
        f"{paths.sum(axis=(1, 2)).mean() / paths.shape[2]:.3f} vs observed "
        f"{fired.mean():.3f}"
    )

    # 3. A mixed-family network: default initiation, Weibull receipt, default
    #    inspection. Nothing in the network knows which family a node uses.
    data = RetailData.from_units(
        units,
        as_of=as_of,
        calendar=date_grid("2026-02-01", as_of),
        event_columns={"inspections": "inspection_date"},
    )
    sales = CountNode("sales")
    initiations = EventNode("initiations", sales, CureProcess(age_bins=12, deadline_days=30))
    receipt_node = EventNode("receipts", initiations, CureProcess(family=family))
    inspections = EventNode(
        "inspections",
        receipt_node,
        CureProcess(age_bins=4, deadline_days=7),
        event_column="inspection_date",
    )
    network = ForecastNetwork([sales, initiations, receipt_node, inspections])
    fitted = network.fit(data, num_steps=300, num_samples=40, seed=2)
    receipt_fit = fitted.stage_fits["receipts"]
    print(
        "network receipt fit:",
        f"{receipt_fit.draws} draws of {sorted(receipt_fit.parameters)}; "
        f"tail {receipt_fit.family.tail}",
    )

    # 4. Longer horizon than training: the family is re-evaluated at ages the
    #    training window never contained, under the same posterior draws.
    horizon = 90
    first_day = as_of + np.timedelta64(1, "D")
    future = SalesForecast.from_frame(
        pd.DataFrame(
            {
                "sale_date": date_grid(first_day, first_day + np.timedelta64(9, "D")),
                "quantity": np.full(10, 12),
            }
        )
    )
    result = fitted.forecast(horizon=horizon, future_sales=future, seed=5)
    for name in ("initiations", "receipts", "inspections"):
        counts = result.counts[name]
        print(
            f"  {name}: {counts.sum(axis=1).mean():.1f} expected over {horizon} days "
            f"(80% interval {np.percentile(counts.sum(axis=1), 10):.0f}-"
            f"{np.percentile(counts.sum(axis=1), 90):.0f})"
        )
    late_ages = TimingInputs(
        ages=jnp.array([[80], [120]]),
        features=jnp.zeros((2, 1, 0)),
        cure_features=jnp.zeros((1, 0)),
    )
    law = receipt_fit.timing(late_ages, draw=0)
    print(
        "  receipt hazard replayed at ages 80 and 120 (never observed in training):",
        np.round(np.exp(np.asarray(law.timing.log_hazard[:, 0])), 4),
    )
    open_returns = int((units["initiation_date"].notna() & units["receipt_date"].isna()).sum())
    conserved = (
        result.counts["receipts"].cumsum(axis=1)
        <= result.counts["initiations"].cumsum(axis=1) + open_returns
    ).all()
    print(f"  receipts never exceed initiations plus the {open_returns} open returns: {conserved}")


if __name__ == "__main__":
    main()
