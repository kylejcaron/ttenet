"""Real simulated-ledger records for the article's "follow one purchase" figure.

The figure is a censoring lesson, not a forecast. It follows three units from one sale
day as the ledger looked at the snapshot: every date below comes from ledger columns that
exist at the snapshot (``sale_date``, ``initiation_date``, ``receipt_date``) and the
supplied storm calendar. The simulator's ``abandoned_truth`` label is never read.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW = 90  # the policy limits *starting* a return: day 90 is still allowed, day 91 is not
SNAPSHOT_AGE = 100  # the snapshot is end-of-day ``as_of``, 100 days after the story's sale day

_COLUMNS = ("item_id", "sale_date", "initiation_date", "receipt_date", "product")


def _days(values) -> np.ndarray:
    return pd.to_datetime(pd.Series(values)).to_numpy().astype("datetime64[D]")


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive ``(first, last)`` index pairs of consecutive true values."""
    edges = np.flatnonzero(np.diff(np.r_[0, flags.astype(int), 0]))
    return [(int(a), int(b) - 1) for a, b in zip(edges[::2], edges[1::2])]


def _storm_days(sale, storms, first_age: int, last_age: int) -> np.ndarray:
    ages = np.arange(first_age, last_age + 1)
    return np.asarray(storms(sale + ages.astype("timedelta64[D]")), dtype=float) > 0


def _storm_interval(sale, storms, first_age: int, last_age: int):
    """Longest known storm run inside ``[first_age, last_age]`` as ``[start, end]`` ages."""
    runs = _runs(_storm_days(sale, storms, first_age, last_age))
    if not runs:
        return None
    start, end = max(runs, key=lambda run: run[1] - run[0])
    return [first_age + start, first_age + end]


def purchase_story(truth: pd.DataFrame, as_of, storms) -> dict:
    """Pick three real units from one sale day and describe them for the figure.

    ``truth`` is the simulated unit ledger (``item_id``, ``sale_date``,
    ``initiation_date``, ``receipt_date``, ``product``); ``as_of`` is the end-of-day
    snapshot date; ``storms`` maps an array of ``datetime64[D]`` to storm flags.
    The story's sale day is ``as_of - 100 days`` so the snapshot is day 100. On that
    cohort, selected in ledger order:

    * A: receipt observed by the snapshot after at least one day in transit (among those,
      the one whose wait overlaps the most storm days; ties go to the earliest row);
    * B: no initiation through day 90 (the first such unit);
    * C: initiated but no receipt observed by the snapshot (the latest initiation).

    Each record's ``init``/``recv`` are ages in days since the sale, ``recv`` is ``None``
    unless the receipt is on or before the snapshot, and ``storm`` is the longest known
    storm run between the return start and the receipt (or snapshot), or ``None``.
    Raises ``ValueError`` when the ledger lacks any of the three cases.
    """
    missing = [name for name in _COLUMNS if name not in truth.columns]
    if missing:
        raise ValueError(f"ledger is missing columns: {missing}")
    snapshot = np.datetime64(as_of, "D")
    sale = snapshot - np.timedelta64(SNAPSHOT_AGE, "D")

    ledger = truth.reset_index(drop=True)
    sold = _days(ledger.sale_date)
    cohort = np.flatnonzero(sold == sale)
    if len(cohort) == 0:
        raise ValueError(f"the ledger has no sales on {sale}")

    initiated = _days(ledger.initiation_date)[cohort]
    received = _days(ledger.receipt_date)[cohort]
    has_init = ~np.isnat(initiated)
    has_recv = ~np.isnat(received)
    init_age = np.where(has_init, (initiated - sale).astype(int), -1)
    if (init_age[has_init] > WINDOW).any() or (init_age[has_init] < 0).any():
        raise ValueError(f"an initiation on {sale}'s cohort falls outside days 0-{WINDOW}")
    if (has_recv & ~has_init).any() or (received[has_recv] < initiated[has_recv]).any():
        raise ValueError("a receipt precedes or lacks its initiation in the ledger")
    recv_age = np.where(has_recv, (received - sale).astype(int), -1)

    seen_init = has_init & (initiated <= snapshot)
    seen_recv = has_recv & (received <= snapshot)
    transit = recv_age - init_age
    received_a = np.flatnonzero(seen_init & seen_recv & (transit >= 1))
    quiet_b = np.flatnonzero(~has_init)
    open_c = np.flatnonzero(seen_init & ~seen_recv)
    for name, found in (
        ("received after at least a day in transit", received_a),
        (f"no initiation through day {WINDOW}", quiet_b),
        ("initiated with no receipt by the snapshot", open_c),
    ):
        if len(found) == 0:
            raise ValueError(f"no unit on {sale} ({len(cohort)} sold) is {name}")

    overlap = [
        int(_storm_days(sale, storms, int(init_age[i]), int(recv_age[i])).sum()) for i in received_a
    ]
    # Most storm days first, then the longest wait; the earliest ledger row breaks remaining ties.
    best = max(range(len(received_a)), key=lambda k: (overlap[k], transit[received_a[k]], -k))
    a = int(received_a[best])
    b = int(quiet_b[0])
    c = int(open_c[int(np.argmax(init_age[open_c]))])  # argmax keeps the earliest tie

    def unit(slot: int) -> tuple[str, str, int]:
        row = ledger.iloc[int(cohort[slot])]
        order = "SYN-" + str(row["item_id"]).removeprefix("sale_")
        return order, f"Product {int(row['product'])}", int(row["product"])

    a_init, a_recv, c_init = int(init_age[a]), int(recv_age[a]), int(init_age[c])
    a_storm = _storm_interval(sale, storms, a_init, a_recv)
    c_storm = _storm_interval(sale, storms, c_init, SNAPSHOT_AGE)
    if a_storm is None:
        a_title = "Started, then received"
    elif a_storm[1] < a_recv:
        a_title = "Received after a storm"
    else:
        a_title = "Received during a storm"

    records = []
    for rid, letter, slot, title, blurb, init, recv, storm in (
        (
            "a",
            "A",
            a,
            a_title,
            f"Started day {a_init}, scanned in day {a_recv}.",
            a_init,
            a_recv,
            a_storm,
        ),
        (
            "b",
            "B",
            b,
            "No return started",
            f"The {WINDOW}-day window closes with nothing initiated.",
            None,
            None,
            None,
        ),
        (
            "c",
            "C",
            c,
            "Started, not received",
            f"Started day {c_init}, no receipt by the snapshot.",
            c_init,
            None,
            c_storm,
        ),
    ):
        order, item, product = unit(slot)
        records.append(
            {
                "id": rid,
                "letter": letter,
                "order": order,
                "item": item,
                "product": product,
                "title": title,
                "blurb": blurb,
                "init": init,
                "recv": recv,
                "storm": storm,
            }
        )

    every_receipt = _days(ledger.receipt_date)
    observed = every_receipt[~np.isnat(every_receipt) & (every_receipt <= snapshot)]
    weekend_receipts = int((pd.DatetimeIndex(observed).dayofweek >= 5).sum())
    return {
        "sale": str(sale),
        "snapshot": str(snapshot),
        "window90": WINDOW,
        "snapshot100": SNAPSHOT_AGE,
        "initial_day": a_init,
        "cohort_units": int(len(cohort)),
        "weekend_receipts": weekend_receipts,
        "records": records,
    }
