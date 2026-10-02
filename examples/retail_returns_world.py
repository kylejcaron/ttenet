"""The synthetic retail world shared by the returns essay and the command-line example.

Two products, a 90-day return policy, abandonment, storms, weekend receipt closures, and
the model inputs that describe them. Receipts arrive by the same discretized Weibull law
the receipt stage fits, so every fitted receipt parameter has a true value to check.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro_forecast import Horizon, predict

from ttenet import date_grid

# The true receipt law. An initiated return is receivable with probability
# RECEIPT_RECEIVABLE; a receivable one arrives at age a (days since initiation) with
# cumulative-hazard increment ((a+1)/scale)^shape - (a/scale)^shape, scaled by
# exp(RECEIPT_STORM) on storm days. Weekends accrue no hazard: the warehouse is closed.
RECEIPT_SCALE, RECEIPT_SHAPE, RECEIPT_STORM, RECEIPT_RECEIVABLE = 6.0, 1.8, -1.5, 0.8

# Daily sales rates before ``volume``, and the weekday swing of each product.
SALES_RATES = (1.2, 0.8)
WEEKDAY_SWING = (0.3, -0.2)

# Return initiation. A sale is susceptible with probability
# sigmoid(SUSCEPTIBLE + SUSCEPTIBLE_PRODUCT * product). A susceptible unit of age a starts a
# return on a day with hazard sigmoid(HAZARD_BASE + HAZARD_AGE * cos(min(a, 15) / 5)
# + HAZARD_PRODUCT * product + HAZARD_WEEKDAY * sin(2 * pi * weekday / 7)), through day 90.
SUSCEPTIBLE, SUSCEPTIBLE_PRODUCT = -0.8, 0.45
HAZARD_BASE, HAZARD_AGE, HAZARD_PRODUCT, HAZARD_WEEKDAY = -2.5, 0.35, -0.25, 0.3


def sigmoid(value):
    return 1 / (1 + np.exp(-value))


def true_initiation_hazard(age):
    """The simulator's daily initiation hazard: product A, a neutral weekday."""
    return sigmoid(HAZARD_BASE + HAZARD_AGE * np.cos(np.minimum(age, 15) / 5))


def true_receipt_hazard(age):
    """The simulator's receipt hazard for the Weibull bin [a, a + 1) on a clear, open day."""
    scaled = np.asarray(age, float) / RECEIPT_SCALE
    increment = (scaled + 1 / RECEIPT_SCALE) ** RECEIPT_SHAPE - scaled**RECEIPT_SHAPE
    return 1 - np.exp(-increment)


def storms(days):
    """An explicitly supplied historical/future weather scenario, not a forecast."""
    age = (np.asarray(days, dtype="datetime64[D]") - np.datetime64("2026-01-01")).astype(int)
    return (
        ((age >= 45) & (age <= 49)) | ((age >= 125) & (age <= 129)) | ((age >= 185) & (age <= 189))
    ).astype(float)


def simulate_retail(seed=21, training_days=180, horizon=28, volume=1.0):
    """Simulate unit histories; ``volume`` scales both products' daily sales rates."""
    rng = np.random.default_rng(seed)
    start = np.datetime64("2026-01-01")
    sales_days = date_grid(start, start + np.timedelta64(training_days + horizon - 1, "D"))
    weekday = pd.DatetimeIndex(sales_days).dayofweek.to_numpy()
    season = np.sin(2 * np.pi * weekday / 7)
    rates = volume * np.exp(np.log(SALES_RATES) + season[:, None] * np.array(WEEKDAY_SWING))
    daily_sales = rng.poisson(rates)
    rows = []
    for day_index, date in enumerate(sales_days):
        for product in range(2):
            for _ in range(daily_sales[day_index, product]):
                initiated = np.datetime64("NaT", "D")
                received = np.datetime64("NaT", "D")
                susceptible = rng.random() < sigmoid(SUSCEPTIBLE + SUSCEPTIBLE_PRODUCT * product)
                if susceptible:
                    for age in range(91):
                        day = date + np.timedelta64(age, "D")
                        week = pd.Timestamp(day).dayofweek
                        hazard = sigmoid(
                            HAZARD_BASE
                            + HAZARD_AGE * np.cos(min(age, 15) / 5)
                            + HAZARD_PRODUCT * product
                            + HAZARD_WEEKDAY * np.sin(2 * np.pi * week / 7)
                        )
                        if rng.random() < hazard:
                            initiated = day
                            break
                abandoned = False
                if not np.isnat(initiated):
                    abandoned = rng.random() >= RECEIPT_RECEIVABLE
                    if not abandoned:
                        # Continue until receipt, not until the sale's policy expires.
                        age = 0
                        while np.isnat(received):
                            day = initiated + np.timedelta64(age, "D")
                            if pd.Timestamp(day).dayofweek < 5:
                                increment = (
                                    ((age + 1) / RECEIPT_SCALE) ** RECEIPT_SHAPE
                                    - (age / RECEIPT_SCALE) ** RECEIPT_SHAPE
                                ) * np.exp(RECEIPT_STORM * float(storms([day])[0]))
                                if rng.random() < 1 - np.exp(-increment):
                                    received = day
                            age += 1
                rows.append(
                    {
                        "item_id": f"sale_{len(rows)}",
                        "sale_date": date,
                        "initiation_date": initiated,
                        "receipt_date": received,
                        "product": float(product),
                        "abandoned_truth": abandoned,
                    }
                )
    return pd.DataFrame(rows), daily_sales


def initiation_covariates(frame, calendar):
    """Product attributes and calendar features with fixed meanings across windows."""
    product = frame["product"].to_numpy(float)
    weekday = pd.DatetimeIndex(calendar).dayofweek.to_numpy()
    features = np.empty((len(frame), len(calendar), 2))
    features[:, :, 0] = product[:, None]
    features[:, :, 1] = np.sin(2 * np.pi * weekday / 7)[None, :]
    return {"features": features, "susceptibility_features": product[:, None]}


def receipt_covariates(frame, calendar):
    return {
        "features": np.broadcast_to(
            storms(calendar)[None, :, None],
            (len(frame), len(calendar), 1),
        ),
    }


def sales_covariates(calendar):
    weekday = pd.DatetimeIndex(calendar).dayofweek.to_numpy()
    return jnp.asarray(np.sin(2 * np.pi * weekday / 7)[:, None])


def sales_model(covariates, data=None):
    h = Horizon.from_data(covariates, data)
    log_rate = numpyro.sample("log_rate", dist.Normal(0, 0.6).expand([2]).to_event(1))
    weekday_effect = numpyro.sample("weekday_effect", dist.Normal(0, 0.4).expand([2]).to_event(1))
    eta = log_rate + covariates * weekday_effect
    predict(h, lambda value: dist.Poisson(jnp.exp(value)), eta)
