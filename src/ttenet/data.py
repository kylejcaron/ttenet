"""Canonical retail return histories and covariate expansion.

Accepts unit histories as tabular rows, longitudinal snapshots, or dated
event/change records and reduces them to a single canonical
:class:`RetailHistory`. Every conversion is strictly isolated to information
knowable as of the supplied cutoff: anything dated after ``as_of`` is treated
as unknown (never as a silently-repaired or imputed value), and invalid
observed sequences are rejected rather than coerced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from .dates import to_day

_LAYOUTS = {"tabular", "longitudinal", "changes"}
_VALID_EVENTS = {"sale", "initiation", "receipt"}


@dataclass(frozen=True)
class RetailHistory:
    """A canonical, as-of-isolated retail unit history.

    ``frame`` has canonical columns ``item_id``, ``sale_date``,
    ``initiation_date``, ``receipt_date``, ``age_since_sale``,
    ``age_since_initiation``, ``eligible``, plus any extra static feature
    columns from the input. Row order matches the original item order (rows
    are only ever dropped, never reordered), and the index is a plain
    ``RangeIndex``.
    """

    frame: pd.DataFrame
    as_of: np.datetime64
    policy_days: Optional[int]


def _empty_day_array(n: int) -> np.ndarray:
    return np.full(n, np.datetime64("NaT", "D"), dtype="datetime64[D]")


def _prepare_tabular(data: pd.DataFrame) -> pd.DataFrame:
    """Reduce a one-row-per-item table to the shared raw item frame."""
    if "item_id" not in data.columns or "sale_date" not in data.columns:
        raise ValueError("tabular layout requires 'item_id' and 'sale_date' columns")
    if data["item_id"].duplicated().any():
        dupes = sorted(str(v) for v in data.loc[data["item_id"].duplicated(), "item_id"].unique())
        raise ValueError(f"duplicate item_id values in tabular input: {dupes}")

    n = len(data)
    out = pd.DataFrame({"item_id": data["item_id"].to_numpy()})
    out["sale_date"] = to_day(data["sale_date"])
    out["initiation_date"] = (
        to_day(data["initiation_date"])
        if "initiation_date" in data.columns
        else _empty_day_array(n)
    )
    out["receipt_date"] = (
        to_day(data["receipt_date"]) if "receipt_date" in data.columns else _empty_day_array(n)
    )

    reserved = {"item_id", "sale_date", "initiation_date", "receipt_date"}
    for col in data.columns:
        if col not in reserved:
            out[col] = data[col].to_numpy()
    return out


def _prepare_longitudinal(data: pd.DataFrame, as_of: np.datetime64) -> pd.DataFrame:
    """Select each item's latest snapshot known by ``as_of``."""
    required = {"item_id", "recorded_date", "sale_date"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"longitudinal layout requires columns: {sorted(missing)}")

    recorded_day = to_day(data["recorded_date"])
    order_map: dict = {}
    for pos, item in enumerate(data["item_id"].to_numpy()):
        order_map.setdefault(item, pos)

    reserved = {
        "item_id",
        "recorded_date",
        "sale_date",
        "initiation_date",
        "receipt_date",
        "_recorded_day",
    }
    feature_cols = [c for c in data.columns if c not in reserved]
    columns = ["item_id", "sale_date", "initiation_date", "receipt_date", *feature_cols]

    visible_mask = ~np.isnat(recorded_day) & (recorded_day <= as_of)
    visible = data.loc[visible_mask].reset_index(drop=True)
    if visible.empty:
        return pd.DataFrame(columns=columns)
    visible["_recorded_day"] = recorded_day[visible_mask]

    dup_key = visible[["item_id"]].copy()
    dup_key["_recorded_day"] = visible["_recorded_day"]
    if dup_key.duplicated().any():
        item = dup_key.loc[dup_key.duplicated(), "item_id"].iloc[0]
        raise ValueError(f"item '{item}' has more than one longitudinal snapshot on the same date")

    idx = visible.groupby("item_id", sort=False)["_recorded_day"].idxmax()
    latest = visible.loc[idx].copy()
    latest["_order"] = latest["item_id"].map(order_map)
    latest = latest.sort_values("_order", kind="stable").reset_index(drop=True)

    n = len(latest)
    out = pd.DataFrame({"item_id": latest["item_id"].to_numpy()})
    out["sale_date"] = to_day(latest["sale_date"])
    out["initiation_date"] = (
        to_day(latest["initiation_date"])
        if "initiation_date" in latest.columns
        else _empty_day_array(n)
    )
    out["receipt_date"] = (
        to_day(latest["receipt_date"]) if "receipt_date" in latest.columns else _empty_day_array(n)
    )
    for col in feature_cols:
        out[col] = latest[col].to_numpy()
    return out


def _prepare_changes(data: pd.DataFrame, as_of: np.datetime64) -> pd.DataFrame:
    """Pivot a sale/initiation/receipt event log, visible-as-of-cutoff only."""
    required = {"item_id", "date", "event"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"changes layout requires columns: {sorted(missing)}")

    date_day = to_day(data["date"])
    order_map: dict = {}
    for pos, item in enumerate(data["item_id"].to_numpy()):
        order_map.setdefault(item, pos)

    reserved = {"item_id", "date", "event"}
    feature_cols = [c for c in data.columns if c not in reserved]
    columns = ["item_id", "sale_date", "initiation_date", "receipt_date", *feature_cols]

    working = data.copy()
    working["_date_day"] = date_day
    visible = working.loc[~np.isnat(date_day) & (date_day <= as_of)]
    bad_events = sorted(set(visible["event"].unique()) - _VALID_EVENTS)
    if bad_events:
        raise ValueError(f"unknown event types in changes input: {bad_events}")

    pivot_rows = []
    for item_id, group in visible.groupby("item_id", sort=False):
        counts = group["event"].value_counts()
        for event in _VALID_EVENTS:
            if counts.get(event, 0) > 1:
                raise ValueError(f"item '{item_id}' has duplicate '{event}' change events")

        sale_rows = group.loc[group["event"] == "sale"]
        if sale_rows.empty:
            raise ValueError(
                f"item '{item_id}' has an observed initiation/receipt event without an "
                "observed sale event"
            )
        sale_row = sale_rows.iloc[0]
        init_rows = group.loc[group["event"] == "initiation"]
        receipt_rows = group.loc[group["event"] == "receipt"]

        row = {
            "item_id": item_id,
            "sale_date": sale_row["_date_day"],
            "initiation_date": (
                init_rows["_date_day"].iloc[0] if not init_rows.empty else np.datetime64("NaT", "D")
            ),
            "receipt_date": (
                receipt_rows["_date_day"].iloc[0]
                if not receipt_rows.empty
                else np.datetime64("NaT", "D")
            ),
        }
        for col in feature_cols:
            row[col] = sale_row[col]
        pivot_rows.append((order_map[item_id], row))

    if not pivot_rows:
        return pd.DataFrame(columns=columns)

    pivot_rows.sort(key=lambda pair: pair[0])
    return pd.DataFrame([row for _, row in pivot_rows], columns=columns)


def _hide_future(days: np.ndarray, as_of: np.datetime64) -> np.ndarray:
    days = days.copy()
    future = ~np.isnat(days) & (days > as_of)
    days[future] = np.datetime64("NaT", "D")
    return days


def _finalize(raw: pd.DataFrame, as_of: np.datetime64, policy_days: Optional[int]) -> pd.DataFrame:
    """Apply as-of isolation, validate event ordering/policy, add derived columns."""
    frame = raw.reset_index(drop=True)
    n = len(frame)

    sale = (
        np.asarray(frame["sale_date"].to_numpy(), dtype="datetime64[D]")
        if n
        else _empty_day_array(0)
    )
    initiation = (
        np.asarray(frame["initiation_date"].to_numpy(), dtype="datetime64[D]")
        if n
        else _empty_day_array(0)
    )
    receipt = (
        np.asarray(frame["receipt_date"].to_numpy(), dtype="datetime64[D]")
        if n
        else _empty_day_array(0)
    )

    if np.isnat(sale).any():
        raise ValueError("sale_date is required for every item")

    # Sales after the cutoff have not happened yet: exclude those rows entirely.
    keep = sale <= as_of
    frame = frame.loc[keep].reset_index(drop=True)
    sale, initiation, receipt = sale[keep], initiation[keep], receipt[keep]

    # Hide any remaining future information rather than repairing it.
    initiation = _hide_future(initiation, as_of)
    receipt = _hide_future(receipt, as_of)

    init_observed = ~np.isnat(initiation)
    receipt_observed = ~np.isnat(receipt)

    if (initiation[init_observed] < sale[init_observed]).any():
        raise ValueError("observed initiation_date cannot precede sale_date")

    elapsed_to_initiation = (initiation - sale) / np.timedelta64(1, "D")
    if (
        policy_days is not None
        and init_observed.any()
        and (elapsed_to_initiation[init_observed] > policy_days).any()
    ):
        raise ValueError(
            f"observed initiation_date exceeds the {policy_days}-day policy deadline for at "
            "least one item"
        )

    if (receipt_observed & ~init_observed).any():
        raise ValueError("observed receipt_date without an observed initiation_date")

    both_observed = receipt_observed & init_observed
    if (receipt[both_observed] < initiation[both_observed]).any():
        raise ValueError("observed receipt_date cannot precede initiation_date")

    age_since_sale = (as_of - sale) / np.timedelta64(1, "D")
    age_since_initiation = np.where(
        init_observed, (as_of - initiation) / np.timedelta64(1, "D"), np.nan
    )
    within_deadline = True if policy_days is None else (age_since_sale <= policy_days)
    eligible = (~init_observed) & (age_since_sale >= 0) & within_deadline

    out = pd.DataFrame({"item_id": frame["item_id"].to_numpy()})
    out["sale_date"] = sale
    out["initiation_date"] = initiation
    out["receipt_date"] = receipt
    out["age_since_sale"] = pd.array(age_since_sale, dtype="Int64")
    out["age_since_initiation"] = pd.array(age_since_initiation, dtype="Int64")
    out["eligible"] = eligible.astype(bool)

    reserved = {
        "item_id",
        "sale_date",
        "initiation_date",
        "receipt_date",
        "age_since_sale",
        "age_since_initiation",
        "eligible",
    }
    for col in frame.columns:
        if col not in reserved:
            out[col] = frame[col].to_numpy()

    return out


def prepare_history(
    data: pd.DataFrame,
    *,
    as_of: Any,
    layout: str = "tabular",
    policy_days: Optional[int] = 90,
) -> RetailHistory:
    """Build a canonical, as-of-isolated :class:`RetailHistory`.

    ``layout`` selects the input shape:

    - ``"tabular"``: one row per item with ``item_id``, ``sale_date``, and
      optional ``initiation_date``/``receipt_date`` columns. Any other columns
      are static per-item features that survive conversion unchanged.
    - ``"longitudinal"``: repeated full-state snapshots per item keyed by
      ``recorded_date``; the latest snapshot with ``recorded_date <= as_of`` is
      used. Items with no snapshot known by ``as_of`` are excluded.
    - ``"changes"``: a long event log with ``item_id``, ``date``, and ``event``
      in ``{"sale", "initiation", "receipt"}``. Extra columns present on an
      item's ``sale`` event row are its static features.

    Every date is truncated to a UTC calendar day (see :func:`ttenet.dates.to_day`).
    Any date after ``as_of`` is unknown as of that cutoff: a sale after the
    cutoff excludes the row entirely (the item is not yet a known sale), while
    an initiation or receipt after the cutoff is hidden (set to ``NaT``) rather
    than repaired. Validation (ordering, the ``policy_days`` initiation
    deadline, and "no receipt without an observed initiation") only applies to
    dates that remain visible after this cutoff is applied -- so a future,
    not-yet-visible initiation can never trigger a policy-deadline error.

    ``policy_days`` is the initiation deadline in days after sale; ``None``
    means there is no deadline at all (every uninitiated, non-future-sale
    item is eligible, with no upper age bound and no policy-deadline
    validation). The default (``90``) is unchanged.

    Raises ``ValueError`` on missing required columns, duplicate/ambiguous
    item identifiers or events, a missing mandatory sale, or any of the
    invalid observed sequences above. Never silently repairs or imputes.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if layout not in _LAYOUTS:
        raise ValueError(f"unknown layout '{layout}'; expected one of {sorted(_LAYOUTS)}")
    invalid_policy = policy_days is not None and (
        not isinstance(policy_days, (int, np.integer))
        or isinstance(policy_days, bool)
        or policy_days < 0
    )
    if invalid_policy:
        raise ValueError("policy_days must be None or a non-negative integer")

    as_of_day = to_day(as_of)
    if pd.isna(as_of_day):
        raise ValueError("as_of must be a valid, non-missing date")

    if layout == "tabular":
        raw = _prepare_tabular(data)
    elif layout == "longitudinal":
        raw = _prepare_longitudinal(data, as_of_day)
    else:
        raw = _prepare_changes(data, as_of_day)

    normalized_policy = None if policy_days is None else int(policy_days)
    frame = _finalize(raw, as_of_day, normalized_policy)
    return RetailHistory(frame=frame, as_of=as_of_day, policy_days=normalized_policy)


def _to_numeric_strict(series: pd.Series, column_name: str) -> np.ndarray:
    raw_null = series.isna()
    numeric = pd.to_numeric(series, errors="coerce")
    bad = numeric.isna() & ~raw_null
    if bad.any():
        raise ValueError(f"non-numeric value(s) in covariate column '{column_name}'")
    return numeric.to_numpy(dtype=np.float64)


def expand_covariates(
    records: pd.DataFrame,
    items: Any,
    dates: Any,
    columns: Any,
    *,
    layout: str = "changes",
) -> np.ndarray:
    """Expand sparse or dense covariate records onto an explicit
    ``[item, calendar_day, feature]`` array, in the exact order of ``items``,
    ``dates``, and ``columns``.

    ``layout="changes"``: ``records`` has ``item_id``, ``date``, and one column
    per requested feature. A non-missing cell is a value change effective on
    that date; a missing cell means "no change on this date" and is skipped,
    never overwriting the item's last known value. Every requested date is
    filled with the most recent known value at or before it, independently
    per item and per feature (a step function) -- never from a later record
    (no backward fill). A requested date strictly before an item's first known
    value for a feature is an error, never silently imputed.

    ``layout="longitudinal"``: ``records`` must contain an explicit row for
    every ``(item, date)`` pair spanned by ``items``/``dates``, with every
    requested feature column populated. Any missing row or missing value is an
    error; there is no forward propagation in this layout.
    """
    items = list(items)
    columns = list(columns)
    dates_arr = np.atleast_1d(to_day(dates))
    n_items, n_days, n_features = len(items), len(dates_arr), len(columns)

    if n_features == 0:
        return np.zeros((n_items, n_days, 0), dtype=np.float64)

    if not isinstance(records, pd.DataFrame):
        raise TypeError("records must be a pandas DataFrame")
    required = {"item_id", "date", *columns}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"records missing required columns: {sorted(missing)}")

    work = records[["item_id", "date", *columns]].copy()
    work["date"] = to_day(work["date"])

    if layout == "changes":
        result = np.empty((n_items, n_days, n_features), dtype=np.float64)
        grouped = {key: sub for key, sub in work.groupby("item_id", sort=False)}
        for f_idx, col in enumerate(columns):
            for i, item in enumerate(items):
                sub = grouped.get(item)
                if sub is not None:
                    values = _to_numeric_strict(sub[col], col)
                    known = ~np.isnan(values)
                    known_dates = sub["date"].to_numpy()[known]
                    known_values = values[known]
                else:
                    known_dates = _empty_day_array(0)
                    known_values = np.array([], dtype=np.float64)

                order = np.argsort(known_dates)
                known_dates = known_dates[order]
                known_values = known_values[order]
                if pd.Index(known_dates).has_duplicates:
                    raise ValueError(
                        f"duplicate same-day changes for item '{item}', covariate '{col}'"
                    )

                position = np.searchsorted(known_dates, dates_arr, side="right") - 1
                if (position < 0).any():
                    bad_date = dates_arr[position < 0][0]
                    raise ValueError(
                        f"item '{item}' has no known value for covariate '{col}' on or "
                        f"before {bad_date}; sparse changes cannot be imputed before the "
                        "first known value"
                    )
                result[i, :, f_idx] = known_values[position]
        return result

    if layout == "longitudinal":
        for col in columns:
            work[col] = _to_numeric_strict(work[col], col)
        if work.duplicated(subset=["item_id", "date"]).any():
            item = work.loc[work.duplicated(subset=["item_id", "date"]), "item_id"].iloc[0]
            raise ValueError(f"duplicate longitudinal covariate rows for item '{item}'")

        indexed = work.set_index(["item_id", "date"])[columns]
        full_index = pd.MultiIndex.from_product([items, dates_arr], names=["item_id", "date"])
        aligned = indexed.reindex(full_index)
        missing_mask = aligned.isna().any(axis=1).to_numpy()
        if missing_mask.any():
            first = aligned.index[missing_mask][0]
            raise ValueError(
                f"missing longitudinal covariate coverage for item '{first[0]}' on {first[1]}"
            )
        return aligned.to_numpy(dtype=np.float64).reshape(n_items, n_days, n_features)

    raise ValueError(f"unknown covariate layout '{layout}'; expected 'changes' or 'longitudinal'")
