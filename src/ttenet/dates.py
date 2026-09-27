"""Date conversion, calendar grids, and calendar-derived regressors.

All dates in this library are UTC calendar days. Timestamps carrying a
timezone or a time-of-day component are converted to UTC first and then
truncated to the containing calendar day; naive timestamps are treated as
already-UTC wall-clock days. Missing dates are represented as ``NaT``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_ARRAY_LIKE = (list, tuple, np.ndarray, pd.Series, pd.Index)


def _is_array_like(values: Any) -> bool:
    return isinstance(values, _ARRAY_LIKE)


def to_day(values: Any) -> Any:
    """Convert a scalar or array-like of dates to UTC calendar days.

    Accepts strings, ``datetime``/``date`` objects, ``pandas.Timestamp``,
    ``numpy.datetime64``, or arrays/Series thereof. Timezone-aware inputs are
    converted to UTC before the time-of-day is discarded. Missing values
    (``None``, ``NaN``, ``NaT``, empty string) become ``NaT``.

    Returns a scalar ``numpy.datetime64[D]`` for scalar input, or a
    ``numpy.ndarray`` of dtype ``datetime64[D]`` for array-like input.
    """
    is_array = _is_array_like(values)
    parsed = pd.to_datetime(values, utc=True, errors="raise", format="mixed")

    if is_array:
        index = pd.DatetimeIndex(parsed)
        if index.tz is not None:
            index = index.tz_localize(None)
        return index.to_numpy().astype("datetime64[D]")

    if parsed is pd.NaT or pd.isna(parsed):
        return np.datetime64("NaT", "D")
    if parsed.tzinfo is not None:
        parsed = parsed.tz_localize(None)
    return np.datetime64(parsed.to_datetime64(), "D")


def date_grid(start: Any, end: Any) -> np.ndarray:
    """Build an inclusive daily ``datetime64[D]`` grid from ``start`` to ``end``."""
    start_day = to_day(start)
    end_day = to_day(end)
    if pd.isna(start_day) or pd.isna(end_day):
        raise ValueError("date_grid requires non-missing start and end dates")
    if start_day > end_day:
        raise ValueError(f"date_grid start {start_day} must not be after end {end_day}")
    return np.arange(start_day, end_day + np.timedelta64(1, "D"), dtype="datetime64[D]")


def elapsed_days(origin: Any, dates: Any) -> Any:
    """Integer day differences ``dates - origin``, broadcasting like NumPy.

    Either argument may be missing (``NaT``); the corresponding result is
    ``NaN``. Both arguments are normalized with :func:`to_day` first, so
    strings/timestamps/arrays are all accepted.
    """
    origin_day = to_day(origin)
    dates_day = to_day(dates)
    diff = (dates_day - origin_day) / np.timedelta64(1, "D")
    return diff


def calendar_features(dates: Any) -> pd.DataFrame:
    """Build calendar regressors: ``date``, ``weekday``, and its sin/cos encoding.

    ``weekday`` is 0 (Monday) through 6 (Sunday). ``weekday_sin``/``weekday_cos``
    encode weekday on the unit circle with a 7-day period.
    """
    days = to_day(dates)
    days = np.atleast_1d(days)
    weekday = pd.DatetimeIndex(days).weekday.to_numpy()
    angle = 2.0 * np.pi * weekday / 7.0
    return pd.DataFrame(
        {
            "date": days,
            "weekday": weekday,
            "weekday_sin": np.sin(angle),
            "weekday_cos": np.cos(angle),
        }
    )


def allowed_days(dates: Any, *, weekdays: Any = range(7), closed_dates: Any = ()) -> np.ndarray:
    """Boolean mask: ``True`` where ``dates`` fall on an allowed weekday and are
    not individually closed.

    ``weekdays`` is the set of permitted weekday integers (0=Monday..6=Sunday).
    ``closed_dates`` is an explicit collection of hard-closed calendar days,
    overriding an otherwise-allowed weekday.
    """
    days = np.atleast_1d(to_day(dates))
    weekday_values = pd.DatetimeIndex(days).weekday.to_numpy()
    allowed_weekdays = np.array(sorted(set(weekdays)), dtype=int)
    mask = np.isin(weekday_values, allowed_weekdays)

    closed_list = list(closed_dates)
    if closed_list:
        closed_days = np.atleast_1d(to_day(closed_list))
        mask = mask & ~np.isin(days, closed_days)

    return mask
