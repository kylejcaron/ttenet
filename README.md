# TTENet

Calendar-aware survival forecasting for retail returns, built with JAX and
NumPyro. Model two linked events:

**Sale → return initiated → return received**

Some purchases never initiate a return; some initiated returns are abandoned.
TTENet fits both stages from censored item histories and forecasts daily event
counts from outstanding items and uncertain future sales.

## Install and run

Python 3.12 or newer:

```bash
uv sync --all-extras
uv run python examples/retail_returns.py --steps 150 --draws 40
```

For a longer fit and an optional chart:

```bash
uv run python examples/retail_returns.py --steps 600 --draws 100 \
  --plot /tmp/retail-returns.png
```

The example simulates two product groups, a 90-day return policy, customer
abandonment, storms, and weekend warehouse closures. It fits both survival
stages, fits a real NumPyro sales model, uses
[`numpyro_forecast.forecast`](https://github.com/juanitorduz/numpyro_forecast)
for future-sales draws, and reports held-out return forecasts and outstanding
estimates. Synthetic abandonment labels are never passed into fitting.

This is a modeling toolkit, not an automatic selection or calibration system.
The short command is an execution smoke test; inspect convergence and
out-of-time calibration before using a fit operationally.

## Histories and clocks

One row per sold unit:

```python
import pandas as pd
from ttenet import prepare_history

sales = pd.DataFrame(
    {
        "item_id": ["a", "b", "c"],
        "sale_date": ["2026-01-01", "2026-01-03", "2026-01-04"],
        "initiation_date": ["2026-01-10", None, "2026-01-15"],
        "receipt_date": ["2026-01-14", None, None],
    }
)
history = prepare_history(sales, as_of="2026-01-20", policy_days=90)
```

`prepare_history` derives ages and eligibility automatically. Missing event
dates mean “not observed yet,” not a known cure or abandonment. Events after
`as_of` are hidden, and future sales are excluded from the historical snapshot.

Three input layouts use the same output:

| `layout` | Input |
| --- | --- |
| `"tabular"` | `item_id`, `sale_date`, optional `initiation_date`, `receipt_date`, static features |
| `"longitudinal"` | Complete snapshots with the same columns plus `recorded_date`; latest snapshot known by the cutoff |
| `"changes"` | Event records with `item_id`, `date`, `event` (`sale`, `initiation`, `receipt`); static features on the sale record |

Use `expand_covariates(records, items, dates, columns, layout="changes")` to
align sparse feature changes to a daily grid. Sparse cells preserve prior
values independently per column. It never fills backward from a future record.
`layout="longitudinal"` requires exact daily coverage. Supply explicit initial
values and future scenarios where needed.
Multiple updates to different features on one date are allowed; multiple
non-missing values for the same item, feature, and date are rejected as ambiguous.

Date helpers include `to_day`, `date_grid`, `elapsed_days`,
`calendar_features`, and `allowed_days`. Dates use **UTC calendar days**:
timezone-aware timestamps are converted to UTC before selecting the day.
Naive dates are interpreted in UTC. Convert business-local timestamps to the
desired business dates explicitly if UTC day boundaries are unsuitable.

### Policy and observation conventions

- `as_of` is end-of-day; forecasts begin the following day.
- Origin day is age zero. Same-day initiation and receipt are supported.
- A 90-day policy permits initiation through `sale_date + 90 days`, inclusive.
- Weekends do not silently extend the deadline.
- Receipt may occur after the sale's initiation deadline.
- Initiation and receipt have separate allowed-day masks. An online return can
  be initiated on Sunday even when the warehouse cannot receive it.
- A hard closure has zero event hazard. If the actual event can occur but is
  merely reported later, model the relevant receipt/reporting event instead of
  masking the wrong process.

## Fit flexible cure-capable stages

```python
from ttenet import date_grid, fit_stage, make_observations

calendar = date_grid("2026-01-01", "2026-06-01")
initiation_data = make_observations(history, "initiation", calendar)
receipt_data = make_observations(history, "receipt", calendar)

initiation_fit = fit_stage(initiation_data, age_bins=30, num_steps=500, num_samples=100, seed=1)
receipt_fit = fit_stage(receipt_data, age_bins=30, num_steps=500, num_samples=100, seed=2)
```

For regressors, supply:

- `features`: `[historical_item, calendar_day, feature]` for timing hazards;
  these can combine static attributes, date features, and changing records.
- `cure_features`: `[historical_item, feature]` for susceptibility.
- `allowed`: `[calendar_day]` or `[historical_item, calendar_day]`.

Both stage preparations take arrays in the **full history row order**; receipt
preparation selects initiated items itself. Negative pre-origin ages and
post-cutoff days never contribute likelihood.

The susceptible daily hazard has a regularized, flexible age baseline plus
linear regressors. Susceptibility has a separate logistic regression. This
avoids prescribing a lognormal or Weibull duration curve; it is not
assumption-free extrapolation. Ages beyond the learned baseline use the final
age-bin hazard.

`fit_stage` uses NumPyro SVI with `AutoNormal` and returns parameter draws plus
loss history. For another inference method, pass `stage_model` and
`StageObservations` directly to NumPyro. `stage_log_likelihood` and the
`survival` primitives are independently usable JAX functions.

## Forecast existing and future sales

```python
from ttenet import forecast_returns

result = forecast_returns(
    history,
    initiation_fit.parameters,
    receipt_fit.parameters,
    calendar=calendar,
    horizon=28,
    seed=3,
)
```

Optional `future_sales` is a frame of homogeneous count cohorts:
`item_id`, `sale_date`, `quantity`. Optional `future_counts[draw, cohort]`
replaces deterministic quantities with uncertain sales draws. Large cohorts
stay counts; they are not expanded into individual unit records.

For models with regressors, pass `initiation_features`, `receipt_features`,
`initiation_cure_features`, and `receipt_cure_features` in the order
**historical rows, then future cohort rows**. Pass stage-specific
`initiation_allowed` and `receipt_allowed` masks. Reuse the same regressor
definitions, scaling, ordering, and historical values used during fitting.
Feature omission for a fitted nonzero-width coefficient vector is an error.

The supplied calendar must cover the historical origin dates, output horizon,
and **every uninitiated/future cohort's initiation deadline**. That extra
coverage makes total remaining-return expectations possible even for a shorter
output horizon. Future weather and other covariates are explicit scenarios;
TTENet does not silently invent them.

### Results

Daily arrays have shape `[draw, forecast_day]`:

| Field | Meaning |
| --- | --- |
| `initiations` | Newly initiated returns |
| `receipts` | Physically received units |
| `eligible` | Uninitiated units still within policy, end-of-day |
| `open_returns` | Initiated but not received, end-of-day |

Per-draw estimates at the forecast origin:

| Field | Meaning |
| --- | --- |
| `expected_uninitiated_receipts` | Eventual receipts from historical sales not yet initiated |
| `expected_open_receipts` | Eventual receipts from already initiated, unreceived returns |
| `expected_existing_receipts` | Their total; excludes future sales |

“Open” is an observed status, not proof that an item will arrive. Abandoned
returns remain unreceived unless the business records an explicit closure;
this package does not invent observed cancellation dates.

Each forecast conditions old items on their observed survival. A 60-day-old
unreturned sale is not restarted as a fresh purchase. Binomial transitions
preserve population counts in every path, including same-day transitions.

## Compose with NumPyro Forecast

```python
from ttenet import sales_cohorts

# sales_samples comes from numpyro_forecast.forecast:
# [posterior_draw, future_day, sales_series]
future_sales, future_counts = sales_cohorts(sales_samples, future_dates, series=0)
```

Pass **both** outputs to `forecast_returns`. The frame's `quantity` is the
first path; `future_counts` preserves all paths. The adapter rejects fractional,
negative, missing, or infinite count predictions instead of rounding them.
The example shows the complete fitting and composition code.

Equal-length draw axes are paired by index; a single parameter draw can
broadcast. Independently fitted models imply a modular independence
assumption, not a jointly inferred sales/return posterior. Preserve aligned
draws yourself when supplying a joint posterior.

## Statistical limits

For susceptibility `pi` and susceptible survival `S`, observed non-events have
probability `(1 - pi) + pi*S`. Conditional susceptibility is
`pi*S / ((1 - pi) + pi*S)`. Right-censored observations therefore contribute
survival information without becoming known never-returners.

The initiation policy limits actual return probability: some susceptible
items may miss the deadline. **`pi` is not the realized return rate.** Expected
receipts integrate the policy-limited initiation probability.

Eventual receipt estimates assume the final positive receipt hazard continues
and receiving eventually reopens. Finite-horizon predictions honor the exact
supplied calendar. Finite follow-up cannot identify cure separately from a
very long tail without assumptions; use mature cohorts, prior sensitivity,
and held-out calibration. SVI uncertainty is approximate.

Current scope is daily, single-unit histories and homogeneous future cohorts
with at most one initiation and one receipt each. No repeated attempts,
partial-unit returns, or rental/inventory feedback.

## Development

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```
