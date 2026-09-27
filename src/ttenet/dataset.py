"""Canonical joint, cohort-aware retail unit ledger.

``RetailData.from_units`` reduces a full unit ledger -- plus an optional
survivor-selected starting snapshot -- to the single canonical container the
cohort-aware return network is fit and forecast against: a canonical per-unit
``units`` table, daily in-window ``sales`` counts, per-unit ``entry_dates``,
and grouping. It never imposes a process deadline (that belongs to the
process/node configuration downstream); it only isolates what is knowable as
of ``as_of`` and separates "old, informative history" from "counted,
in-window sales".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from .data import (
    _LAYOUTS,
    _hide_future,
    _prepare_changes,
    _prepare_longitudinal,
    _prepare_tabular,
    prepare_history,
)
from .dates import date_grid, to_day

_DEFAULT_EVENT_COLUMNS = {
    "sales": "sale_date",
    "initiations": "initiation_date",
    "receipts": "receipt_date",
}


def _resolve_event_columns(event_columns: Optional[Mapping[str, str]]) -> dict:
    event_cols = dict(_DEFAULT_EVENT_COLUMNS)
    if event_columns is not None:
        if not isinstance(event_columns, Mapping):
            raise TypeError("event_columns must be a mapping of node name to column name")
        if any(
            not isinstance(k, str) or not isinstance(v, str) or not k or not v
            for k, v in event_columns.items()
        ):
            raise ValueError("event_columns must contain nonempty string names")
        if any(k in event_cols and event_cols[k] != v for k, v in event_columns.items()):
            raise ValueError("canonical retail event columns cannot be remapped")
        event_cols.update(event_columns)
    if len(set(event_cols.values())) != len(event_cols):
        raise ValueError("event_columns must map distinct node names to distinct column names")
    return event_cols


def _isolate(raw, event_cols, as_of_day, *, allow_missing_sales=False):
    """As-of isolate every mapped event column: drop future sales rows, hide
    (rather than repair) every other future event date."""
    frame = raw.reset_index(drop=True)
    sales_col = event_cols["sales"]

    sale = np.asarray(frame[sales_col].to_numpy(), dtype="datetime64[D]")
    if not allow_missing_sales and np.isnat(sale).any():
        raise ValueError(f"'{sales_col}' is required for every item")

    keep = np.isnat(sale) | (sale <= as_of_day)
    frame = frame.loc[keep].reset_index(drop=True)
    frame[sales_col] = sale[keep]

    for name, col in event_cols.items():
        if name == "sales":
            continue
        vals = np.asarray(to_day(frame[col]))
        frame[col] = _hide_future(vals, as_of_day)

    return frame


def _validate_observation_window(calendar: np.ndarray, as_of_day: np.datetime64) -> np.ndarray:
    if calendar.size == 0:
        raise ValueError("calendar must not be empty")
    if np.isnat(calendar).any():
        raise ValueError("calendar must not contain missing dates")
    order = np.argsort(calendar, kind="stable")
    if not np.array_equal(order, np.arange(calendar.size)):
        raise ValueError("calendar must be sorted ascending")
    expected = date_grid(calendar[0], calendar[-1])
    if calendar.size != expected.size or not np.array_equal(calendar, expected):
        raise ValueError("calendar must be a contiguous daily grid (see dates.date_grid)")
    if calendar[-1] != as_of_day:
        raise ValueError("calendar must be an observation window ending exactly at as_of")
    return calendar


def _validate_starting_items(
    starting_items: pd.DataFrame, event_cols: dict, calendar0: np.datetime64
) -> pd.DataFrame:
    """Normalize and validate a survivor-selected snapshot alone, before any merge."""
    if "item_id" not in starting_items.columns:
        raise ValueError("starting_items requires an 'item_id' column")
    if starting_items["item_id"].duplicated().any():
        dupes = sorted(
            str(v)
            for v in starting_items.loc[starting_items["item_id"].duplicated(), "item_id"].unique()
        )
        raise ValueError(f"duplicate item_id values in starting_items: {dupes}")

    sales_col = event_cols["sales"]
    if sales_col not in starting_items.columns:
        raise ValueError(f"starting_items requires the '{sales_col}' column")

    if starting_items["item_id"].isna().any():
        raise ValueError("starting_items item_id must not be missing")
    out = (
        starting_items.drop(
            columns=["age_since_sale", "age_since_initiation", "eligible"],
            errors="ignore",
        )
        .reset_index(drop=True)
        .copy()
    )
    sale = np.asarray(to_day(out[sales_col]))
    if np.isnat(sale).any():
        raise ValueError("every starting_items row must have a known sale date")
    if (sale >= calendar0).any():
        raise ValueError(
            "starting_items must not include new sales on or after the first calendar day"
        )
    out[sales_col] = sale

    receipts_col = event_cols.get("receipts")
    if receipts_col is not None and receipts_col in out.columns:
        receipt = np.asarray(to_day(out[receipts_col]))
        if (~np.isnat(receipt)).any():
            raise ValueError(
                "starting_items must not include completed receipts: a snapshot describes "
                "only still-outstanding units"
            )

    for name, col in event_cols.items():
        if name in ("sales",) or col not in out.columns:
            continue
        vals = np.asarray(to_day(out[col]))
        if name != "receipts" and (~np.isnat(vals) & (vals >= calendar0)).any():
            raise ValueError(
                f"starting_items must not include post-boundary '{col}' events (on or after "
                "the first calendar day)"
            )
        out[col] = vals

    return out


def _merge_ledger_and_starting(
    isolated: pd.DataFrame,
    starting: Optional[pd.DataFrame],
    event_cols: dict,
    calendar0: np.datetime64,
) -> tuple:
    """Merge the as-of-isolated full ledger with an optional starting-items
    snapshot, returning ``(units, entry_dates)``.

    Ledger-only units keep their own sale date as ``entry_dates`` (full,
    informative pre-window history); units driven by ``starting`` (matched
    to a ledger row or standalone) get ``entry_dates = calendar0``.
    """
    sales_col = event_cols["sales"]
    date_cols = list(dict.fromkeys(event_cols.values()))

    if starting is None:
        entry_dates = np.asarray(isolated[sales_col].to_numpy(), dtype="datetime64[D]")
        return isolated.reset_index(drop=True), entry_dates

    ledger_by_id = {v: i for i, v in enumerate(isolated["item_id"].to_numpy())}
    starting_ids = list(starting["item_id"].to_numpy())
    starting_by_id = {v: i for i, v in enumerate(starting_ids)}

    extra_cols = sorted(
        (set(isolated.columns) | set(starting.columns)) - set(date_cols) - {"item_id"}
    )

    order = list(isolated["item_id"].to_numpy())
    order.extend(item_id for item_id in starting_ids if item_id not in ledger_by_id)

    date_values: dict = {col: [] for col in date_cols}
    extra_values: dict = {col: [] for col in extra_cols}
    entry_dates = []

    for item_id in order:
        in_ledger = item_id in ledger_by_id
        in_starting = item_id in starting_by_id
        lrow = isolated.iloc[ledger_by_id[item_id]] if in_ledger else None
        srow = starting.iloc[starting_by_id[item_id]] if in_starting else None

        entry_dates.append(calendar0 if in_starting else np.datetime64(lrow[sales_col], "D"))

        for col in date_cols:
            l_val = (
                np.datetime64(lrow[col], "D")
                if in_ledger and col in isolated.columns and not pd.isna(lrow[col])
                else np.datetime64("NaT", "D")
            )
            s_val = (
                np.datetime64(srow[col], "D")
                if in_starting and col in starting.columns and not pd.isna(srow[col])
                else np.datetime64("NaT", "D")
            )
            l_known, s_known = not np.isnat(l_val), not np.isnat(s_val)

            if col == sales_col:
                if in_ledger and in_starting:
                    if l_known and l_val != s_val:
                        raise ValueError(
                            f"conflicting sale date between starting_items and unit_history "
                            f"for item {item_id!r}"
                        )
                    date_values[col].append(s_val)
                else:
                    date_values[col].append(l_val if in_ledger else s_val)
                continue

            if l_known and s_known:
                if l_val != s_val:
                    raise ValueError(
                        f"conflicting '{col}' between starting_items and unit_history for "
                        f"item {item_id!r}"
                    )
                date_values[col].append(s_val)
            elif s_known:
                date_values[col].append(s_val)
            elif l_known:
                if in_starting and l_val < calendar0:
                    raise ValueError(
                        f"unit_history has a prior '{col}' event before the starting_items "
                        f"boundary that is not reflected in the snapshot for item {item_id!r}"
                    )
                date_values[col].append(l_val)
            else:
                date_values[col].append(np.datetime64("NaT", "D"))

        for col in extra_cols:
            l_val = lrow[col] if in_ledger and col in isolated.columns else None
            s_val = srow[col] if in_starting and col in starting.columns else None
            l_known, s_known = not pd.isna(l_val), not pd.isna(s_val)
            if l_known and s_known:
                if l_val != s_val:
                    raise ValueError(
                        f"conflicting '{col}' attribute between starting_items and "
                        f"unit_history for item {item_id!r}"
                    )
                extra_values[col].append(s_val)
            elif s_known:
                extra_values[col].append(s_val)
            elif l_known:
                extra_values[col].append(l_val)
            else:
                extra_values[col].append(np.nan)

    units = pd.DataFrame({"item_id": order})
    for col in date_cols:
        units[col] = np.array(date_values[col], dtype="datetime64[D]")
    for col in extra_cols:
        units[col] = extra_values[col]

    return units, np.array(entry_dates, dtype="datetime64[D]")


def _build_groups(units: pd.DataFrame, group_by: Tuple[str, ...]) -> tuple:
    if not group_by:
        group_index = np.zeros(len(units), dtype=np.int64)
        return group_index, pd.DataFrame(index=range(1))

    unknown = [c for c in group_by if c not in units.columns]
    if unknown:
        raise ValueError(f"unknown group_by column(s): {unknown}")

    sub = units[list(group_by)]
    if sub.isna().to_numpy().any():
        raise ValueError(f"group_by column(s) {list(group_by)} must not contain missing values")

    keys = list(sub.itertuples(index=False, name=None))
    index_map: dict = {}
    order: list = []
    group_index = np.empty(len(units), dtype=np.int64)
    for i, key in enumerate(keys):
        if key not in index_map:
            index_map[key] = len(order)
            order.append(key)
        group_index[i] = index_map[key]

    groups = pd.DataFrame(order, columns=list(group_by))
    return group_index, groups


def _compute_sales(
    sale_dates: np.ndarray, calendar: np.ndarray, group_index: np.ndarray, n_groups: int
) -> np.ndarray:
    t = len(calendar)
    sales = np.zeros((t, n_groups), dtype=np.int64)
    if len(sale_dates) == 0 or n_groups == 0:
        return sales
    offsets = (sale_dates - calendar[0]) / np.timedelta64(1, "D")
    in_window = (offsets >= 0) & (offsets < t)
    idx = offsets[in_window].astype(np.int64)
    grp = group_index[in_window]
    np.add.at(sales, (idx, grp), 1)
    return sales


@dataclass(frozen=True)
class RetailData:
    """A canonical, as-of-isolated, cohort-aware retail unit ledger.

    ``units`` is the canonical per-unit table: ``item_id`` plus every mapped
    ``event_columns`` date column (as-of isolated) plus any extra static
    feature/grouping columns. ``sales`` is the daily in-window unit-sale
    count grid, ``[len(calendar), len(groups)]``: only sales whose date
    falls within ``calendar`` are counted -- pre-window sales remain in
    ``units`` (informative full history) but never enter ``sales``.
    ``entry_dates`` is aligned to ``units``: each unit's own sale date, or
    ``calendar[0]`` for units driven by an optional starting-items snapshot.

    No process deadline belongs here; that is the process/node
    configuration's responsibility downstream.
    """

    units: pd.DataFrame
    as_of: np.datetime64
    calendar: np.ndarray
    group_by: Tuple[str, ...]
    groups: pd.DataFrame
    sales: np.ndarray
    entry_dates: np.ndarray
    event_columns: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.units, pd.DataFrame):
            raise TypeError("units must be a pandas DataFrame")
        units = self.units.reset_index(drop=True).copy()

        as_of_day = to_day(self.as_of)
        if pd.isna(as_of_day):
            raise ValueError("as_of must be a valid, non-missing date")

        calendar = np.array(to_day(self.calendar)).reshape(-1)
        calendar.setflags(write=False)

        group_by = tuple(self.group_by)

        if not isinstance(self.groups, pd.DataFrame):
            raise TypeError("groups must be a pandas DataFrame")
        groups = self.groups.reset_index(drop=True).copy()

        sales = np.array(self.sales, dtype=np.int64)
        if sales.shape != (len(calendar), len(groups)):
            raise ValueError(
                f"sales must have shape ({len(calendar)}, {len(groups)}); got {sales.shape}"
            )
        sales.setflags(write=False)

        entry_dates = np.array(to_day(self.entry_dates)).reshape(-1)
        if len(entry_dates) != len(units):
            raise ValueError("entry_dates must align with units (one entry date per unit row)")
        entry_dates.setflags(write=False)

        if not isinstance(self.event_columns, Mapping):
            raise TypeError("event_columns must be a mapping")
        event_columns = dict(self.event_columns)
        if "sales" not in event_columns:
            raise ValueError("event_columns must define a 'sales' node")

        object.__setattr__(self, "units", units)
        object.__setattr__(self, "as_of", as_of_day)
        object.__setattr__(self, "calendar", calendar)
        object.__setattr__(self, "group_by", group_by)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "sales", sales)
        object.__setattr__(self, "entry_dates", entry_dates)
        object.__setattr__(self, "event_columns", event_columns)

    def copy(self) -> "RetailData":
        """Return an independent copy with independent data/array copies."""
        return RetailData(
            units=self.units,
            as_of=self.as_of,
            calendar=self.calendar,
            group_by=self.group_by,
            groups=self.groups,
            sales=self.sales,
            entry_dates=self.entry_dates,
            event_columns=self.event_columns,
        )

    @property
    def context_calendar(self) -> np.ndarray:
        """Daily grid from ``min(calendar[0], earliest unit sale)`` through ``as_of``."""
        sales_col = self.event_columns["sales"]
        earliest = self.calendar[0]
        if len(self.units):
            observed = to_day(pd.Series(self.units[sales_col]).min())
            if not pd.isna(observed):
                earliest = min(earliest, observed)
        return date_grid(earliest, self.as_of)

    @classmethod
    def from_units(
        cls,
        unit_history: pd.DataFrame,
        *,
        as_of: Any,
        calendar: Any,
        group_by: Tuple[str, ...] = (),
        starting_items: Optional[pd.DataFrame] = None,
        layout: str = "tabular",
        event_columns: Optional[Mapping[str, str]] = None,
    ) -> "RetailData":
        """Build a canonical :class:`RetailData` from a complete unit ledger.

        ``unit_history`` is a complete ledger: every known unit, sold at any
        time (including before ``calendar`` begins). ``as_of`` is the cutoff
        -- anything dated after it is unknown and hidden (or, for a sale,
        excludes the row entirely, since an unsold-as-of-cutoff item is not
        yet a known unit). ``calendar`` is a contiguous daily observation
        window ending exactly at ``as_of``; sales dated before
        ``calendar[0]`` remain in ``units`` as informative history but are
        never counted in ``sales``.

        ``layout`` selects the input shape, matching
        :func:`ttenet.data.prepare_history` (``"tabular"``, ``"longitudinal"``,
        ``"changes"``). Canonical retail date columns retain their names.

        ``event_columns`` maps EventNode names to ``unit_history`` date
        columns, merged over the default
        ``{'sales': 'sale_date', 'initiations': 'initiation_date',
        'receipts': 'receipt_date'}``; extra entries add optional
        additional node mappings, normalized/as-of-isolated the same way.

        ``starting_items`` is an optional survivor-selected snapshot as of
        the end of the day before ``calendar[0]``: every date field in it
        must be strictly before ``calendar[0]`` (a new sale, a completed
        receipt, or any other post-boundary event in the snapshot alone is
        rejected as an inconsistent snapshot). Items in both
        ``starting_items`` and ``unit_history`` are merged into one row
        (never double-counted); a value known in both sources must agree
        (conflicting sale/prior-event dates or grouping attrs raise), a
        value known only in the snapshot is preserved, and a later
        (post-boundary) ledger event fills in what the snapshot could not
        yet know. Items present only in ``starting_items`` (absent from the
        ledger) are retained as censored.
        """
        if not isinstance(unit_history, pd.DataFrame):
            raise TypeError("unit_history must be a pandas DataFrame")
        if layout not in _LAYOUTS:
            raise ValueError(f"unknown layout '{layout}'; expected one of {sorted(_LAYOUTS)}")

        as_of_day = to_day(as_of)
        if pd.isna(as_of_day):
            raise ValueError("as_of must be a valid, non-missing date")

        event_cols = _resolve_event_columns(event_columns)

        group_by = (group_by,) if isinstance(group_by, str) else tuple(group_by)
        reserved = {"item_id", "quantity", "draw", *event_cols.values()}
        if len(set(group_by)) != len(group_by) or set(group_by) & reserved:
            raise ValueError(
                "group_by requires distinct static attribute columns, not identifiers or event dates"
            )
        if layout == "tabular":
            if starting_items is not None and "sale_date" not in unit_history:
                unit_history = unit_history.assign(sale_date=pd.NaT)
            raw = _prepare_tabular(unit_history)
        elif layout == "longitudinal":
            raw = _prepare_longitudinal(unit_history, as_of_day)
        else:
            raw = _prepare_changes(unit_history, as_of_day)
        raw = raw.drop(
            columns=["age_since_sale", "age_since_initiation", "eligible"],
            errors="ignore",
        )
        if raw["item_id"].isna().any():
            raise ValueError("unit_history item_id must not be missing")

        for name, col in event_cols.items():
            if col not in raw.columns:
                raise ValueError(f"unit_history is missing event column '{col}' for node '{name}'")

        isolated = _isolate(
            raw,
            event_cols,
            as_of_day,
            allow_missing_sales=starting_items is not None,
        )

        calendar_arr = np.asarray(to_day(calendar)).reshape(-1)
        calendar_arr = _validate_observation_window(calendar_arr, as_of_day)

        prepared_starting = None
        if starting_items is not None:
            if not isinstance(starting_items, pd.DataFrame):
                raise TypeError("starting_items must be a pandas DataFrame")
            prepared_starting = _validate_starting_items(
                starting_items, event_cols, calendar_arr[0]
            )

        units, entry_dates = _merge_ledger_and_starting(
            isolated, prepared_starting, event_cols, calendar_arr[0]
        )
        units = prepare_history(units, as_of=as_of_day, policy_days=None).frame

        group_index, groups = _build_groups(units, tuple(group_by))

        sales_col = event_cols["sales"]
        sale_dates = np.asarray(units[sales_col].to_numpy(), dtype="datetime64[D]")
        sales = _compute_sales(sale_dates, calendar_arr, group_index, len(groups))

        return cls(
            units=units,
            as_of=as_of_day,
            calendar=calendar_arr,
            group_by=tuple(group_by),
            groups=groups,
            sales=sales,
            entry_dates=entry_dates,
            event_columns=event_cols,
        )
