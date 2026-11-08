"""Vega-Lite templates and complete data for the essay's browser-owned controls."""

import numpy as np
import pandas as pd
from retail_returns_presentation import baseline_chart, convolution_panels, tldr_panels


def _panel_data(charts):
    """Store shared Altair datasets once, keeping each panel's references explicit."""
    datasets = {}
    panels = []
    for chart in charts:
        spec = chart.to_dict()
        own_data = spec.pop("datasets", {})
        datasets.update(own_data)
        panels.append({"spec": spec, "datasets": list(own_data)})
    return {"panels": panels, "datasets": datasets}


def history_data(
    forecast,
    sales_draws,
    sales_actual,
    held_out,
    storms,
    promos,
    history,
    as_of,
    horizon,
    title,
    caption,
):
    """One history/forecast template; moving either handle changes only the date axes."""
    snapshot = pd.Timestamp(as_of)
    initial = [-62, int(horizon)]
    window = [snapshot + pd.Timedelta(days=day) for day in initial]
    return {
        "kind": "history",
        **_panel_data(
            tldr_panels(
                forecast, sales_draws, sales_actual, held_out, storms, promos, history, window
            )
        ),
        "as_of": snapshot.strftime("%Y-%m-%d"),
        "min": 1 - len(history),
        "max": int(horizon),
        "default": initial,
        "title": title,
        "caption": caption,
    }


def sale_day_data(followed, sales, first_date, default, title, caption, description):
    """Every sale-day contribution, plus the selected day's caption totals."""
    sales = np.asarray(sales, dtype=float)
    data = _panel_data(convolution_panels(followed, sales, first_date, default))
    return {
        "kind": "sale",
        **data,
        "highlight_datasets": [
            name for name, rows in data["datasets"].items() if rows and "picked" in rows[0]
        ],
        "dates": pd.date_range(first_date, periods=len(sales)).strftime("%Y-%m-%d").tolist(),
        "sales": sales.tolist(),
        "started": np.asarray(followed["started"]).sum(axis=1).tolist(),
        "received": np.asarray(followed["received"]).sum(axis=1).tolist(),
        "default": int(default),
        "title": title,
        "caption": caption,
        "description": description,
    }


def baseline_data(forecast, held_out, baselines, spans, promos, recent, choices, caption):
    """All four shortcuts on one fixed axis, sharing the same chain and observations."""
    y_max = 1.05 * max(
        max(float(np.max(values)) for values in baselines.values()),
        float(np.quantile(forecast.receipts, 0.95, axis=0).max()),
        float(held_out.receipts.max()),
        float(recent["Returns received"].max()),
    )
    data = _panel_data(
        [
            baseline_chart(
                forecast,
                held_out,
                baselines,
                spans,
                promos,
                recent,
                colors=[choice["color"] for choice in choices],
                y_max=y_max,
            )
        ]
    )
    lines = next(
        name
        for name, rows in data["datasets"].items()
        if rows and "series" in rows[0] and "value" in rows[0]
    )
    model = data["datasets"][lines][0]["series"]
    return {
        "kind": "baseline",
        **data,
        "line_dataset": lines,
        "model_series": model,
        "choices": choices,
        "default": next(i for i, choice in enumerate(choices) if choice["label"] == "SARIMAX"),
        "caption": caption,
        "y_max": y_max,
    }
