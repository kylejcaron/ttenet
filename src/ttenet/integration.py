"""Composition with count forecasts, including NumPyro Forecast's draw layout."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .dates import to_day


def sales_cohorts(samples, dates, *, series: int = 0, prefix: str = "future"):
    """Convert ``[draw, time, observation]`` sales draws into dated count cohorts.

    Returns ``(future_sales, future_counts)`` for :func:`forecast_returns`.
    Pass BOTH outputs: the frame's ``quantity`` is the first draw, while
    ``future_counts`` preserves the complete predictive distribution. Each row
    represents a homogeneous sales cohort, not an individual sold unit.

    Forecasts must already represent integer sales counts. Continuous demand
    predictions cannot be rounded into a calibrated count distribution here.
    """
    draws = np.asarray(samples)
    if draws.ndim != 3 or any(size == 0 for size in draws.shape):
        raise ValueError("sales samples must have nonempty shape [draw, time, observation]")
    if not isinstance(series, (int, np.integer)) or not 0 <= series < draws.shape[2]:
        raise ValueError("series must index the observation axis")
    if not np.issubdtype(draws.dtype, np.number) or np.iscomplexobj(draws):
        raise ValueError("sales samples must be real integer counts")
    selected = draws[:, :, series]
    if (
        not np.isfinite(selected).all()
        or (selected < 0).any()
        or (selected != np.floor(selected)).any()
        or (selected >= float(np.iinfo(np.int64).max)).any()
    ):
        raise ValueError("sales samples must be finite nonnegative int64 counts")
    days = np.asarray(to_day(dates))
    if days.ndim != 1 or len(days) != draws.shape[1] or np.isnat(days).any():
        raise ValueError("dates must match the forecast time axis and contain no missing days")
    if len(days) > 1 and not np.all(np.diff(days) == np.timedelta64(1, "D")):
        raise ValueError("forecast dates must be consecutive, unique, increasing calendar days")
    counts = selected.astype(np.int64)
    cohorts = pd.DataFrame(
        {
            "item_id": [f"{prefix}_{i}" for i in range(len(days))],
            "sale_date": days,
            "quantity": counts[0],
        }
    )
    return cohorts, counts
