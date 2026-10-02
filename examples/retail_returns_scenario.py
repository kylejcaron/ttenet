"""Everything the storm-scenario figure draws, computed in Python.

The figure is plain JavaScript over this payload, so it keeps working in a static export:
every scenario is summarized up front and the buttons only choose between them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from retail_returns_analysis import bands

#: Display order of the scenario buttons, and the one shown first.
ORDER = ("clear", "as_forecast", "long")
DEFAULT = "as_forecast"
#: How many days the lingering storm lasts.
LINGER_DAYS = 10


def storm_labels(spans, as_of):
    """First storm day after ``as_of`` and the button label of each scenario."""
    upcoming = spans[spans.start > str(as_of)].iloc[0]
    first = upcoming.start
    last = upcoming.end - pd.Timedelta(days=1)
    lingers_to = first + pd.Timedelta(days=LINGER_DAYS - 1)
    labels = {
        "clear": "No storm",
        "as_forecast": f"Storm as forecast ({first:%b} {first.day}–{last.day})",
        "long": f"Storm lingers ten days ({first:%b} {first.day}–{lingers_to.day})",
    }
    return np.datetime64(first.date()), labels


def total(draws):
    """An interval for a per-draw quantity (a 28-day sum, a final-day count)."""
    values = np.asarray(draws)
    lo, hi = np.quantile(values, [0.05, 0.95])
    return {"mean": float(values.mean()), "lo90": float(lo), "hi90": float(hi)}


def scenario_payload(scenario, baseline, storm_baseline, storm_days):
    """One scenario against the baseline run.

    The runs share a seed and the same sales draws, so draw ``i`` of a scenario pairs with
    draw ``i`` of the baseline and their difference is a real per-draw quantity.
    ``storm_baseline`` and ``storm_days`` are the boolean storm days each run was given.
    """
    assert np.array_equal(scenario.sales.counts, baseline.sales.counts), "draws do not pair"
    arms = {"baseline": baseline, "scenario": scenario}
    difference = scenario.receipts - baseline.receipts

    # Planning windows: the days either run calls stormy, then the week that follows.
    stormy = np.flatnonzero(storm_baseline | storm_days)
    first, last = int(stormy[0]), int(stormy[-1])
    spans = [(first, last), (last + 1, min(last + 7, len(storm_baseline) - 1))]
    return {
        "storm": {
            "baseline": storm_baseline.tolist(),
            "scenario": storm_days.tolist(),
        },
        "panels": {
            "receipt": {arm: bands(run.receipts) for arm, run in arms.items()},
            "open": {arm: bands(run.open_returns) for arm, run in arms.items()},
        },
        # Receipts so far, scenario minus baseline, summarized over paired draws.
        "gap": bands(np.cumsum(difference, axis=1)),
        "windows": [
            {
                "start": start,
                "end": end,
                "receipt": {
                    **{
                        arm: total(run.receipts[:, start : end + 1].sum(axis=1))
                        for arm, run in arms.items()
                    },
                    "diff": total(difference[:, start : end + 1].sum(axis=1)),
                },
            }
            for start, end in spans
            if start <= end
        ],
        "totals": {
            "receipt": {arm: total(run.receipts.sum(axis=1)) for arm, run in arms.items()},
            "open_end": {arm: total(run.open_returns[:, -1]) for arm, run in arms.items()},
        },
    }


def scenario_data(as_of, baseline, runs, labels, storms, storm_scenario):
    """The full figure payload: one entry per scenario run, keyed by weather.

    ``runs`` maps weather to a forecast, ``storms`` is the supplied storm calendar and
    ``storm_scenario(weather)`` returns the receipt-covariate provider each run was given.
    """
    storm_baseline = storms(baseline.dates) > 0
    placeholder = pd.DataFrame(index=[0])

    def storm_days(weather):
        features = storm_scenario(weather)(placeholder, baseline.dates)["features"]
        return features[0, :, 0] > 0

    return {
        "as_of": str(as_of),
        "dates": np.asarray(baseline.dates, dtype="datetime64[D]").astype(str).tolist(),
        "weather_labels": labels,
        "order": list(ORDER),
        "default": DEFAULT,
        "scenarios": {
            weather: scenario_payload(run, baseline, storm_baseline, storm_days(weather))
            for weather, run in runs.items()
        },
    }
