# TTENet

Calendar-aware survival forecasting for retail returns, built with JAX and
NumPyro. Model two linked events:

**Sale → return initiated → return received**

Some purchases never initiate a return; some initiated returns are abandoned.
TTENet fits both stages from censored item histories and forecasts daily event
counts from outstanding items and uncertain future sales. The public entry
point is `RetailReturnModel`; the functions it calls underneath remain
directly usable for advanced configurations.

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
abandonment, storms, and weekend warehouse closures. It builds a
`RetailReturnModel`, fits it with `.fit(...)`, fits a real NumPyro sales
model, uses
[`numpyro_forecast.forecast`](https://github.com/juanitorduz/numpyro_forecast)
for future-sales draws, and calls the fitted model's `.forecast(...)` to
report held-out return forecasts and outstanding estimates. Synthetic
abandonment labels are never passed into fitting.

This is a modeling toolkit, not an automatic selection or calibration system.
The short command is an execution smoke test; inspect convergence and
out-of-time calibration before using a fit operationally.

## Fit and forecast with `RetailReturnModel`

`RetailReturnModel` is frozen, reusable configuration; `.fit(...)` returns a
separate `FittedRetailReturnModel` snapshot without mutating the model or any
earlier fit:

```python
import pandas as pd
from ttenet import RetailReturnModel, allowed_days, date_grid

sales = pd.DataFrame(
    {
        "item_id": ["a", "b", "c"],
        "sale_date": ["2026-01-01", "2026-01-03", "2026-01-04"],
        "initiation_date": ["2026-01-10", None, "2026-01-15"],
        "receipt_date": ["2026-01-14", None, None],
        "product": [0.0, 1.0, 0.0],
    }
)
calendar = date_grid("2026-01-01", "2026-06-01")


def retail_features(frame, calendar):
    """Pure: same rows and calendar always produce the same arrays."""
    product = frame["product"].to_numpy(float)
    return {
        "initiation_cure_features": product[:, None],
        "receipt_allowed": allowed_days(calendar, weekdays=range(5)),
    }


model = RetailReturnModel(policy_days=90, age_bins=16, feature_builder=retail_features)
fitted = model.fit(
    sales, as_of="2026-01-20", calendar=calendar, num_steps=150, num_samples=40, seed=21
)

result = fitted.forecast(calendar=calendar, horizon=28, seed=26)
```

Raw data (as above) requires `as_of` and uses the same `layout` values as
`prepare_history` (`"tabular"`, `"longitudinal"`, `"changes"`; default
`"tabular"`). A `RetailHistory` you already built can be passed directly,
without `as_of` or `layout`, as long as its `policy_days` matches the model:

```python
from ttenet import prepare_history

history = prepare_history(sales, as_of="2026-01-20", policy_days=90)
fitted = model.fit(history, calendar=calendar, num_steps=150, num_samples=40, seed=21)
```

`fit` copies that history's frame, so later edits to your `history.frame`
never change `fitted.history`. Fitting the same `model` again — even against
a completed population — returns an independent `FittedRetailReturnModel`;
it never changes an earlier fit's `.forecast()` output.

`fitted` exposes `model`, `history`, `initiation_fit`, and `receipt_fit`.
The two stage fits expose posterior `.parameters` and the full `.losses` trace
for diagnostics. Inspect the loss trajectory and held-out calibration;
comparing the first and last losses alone is not a convergence test.

`.forecast(*, calendar, horizon, future_sales=None, future_counts=None, seed=0)`
returns the same `ReturnForecast` described in [Results](#results) below.
Fit-time rows are the canonical historical units; at forecast time the model
builds features for historical rows followed by `future_sales` cohort rows,
in that order — you do not manually concatenate the two populations' feature arrays.

## The feature builder contract

`feature_builder(frame, calendar)` is the single hook for regressors and
allowed-day closures. Its calendar is a normalized UTC `datetime64[D]` array.
It receives historical rows while fitting and historical rows followed by
future cohort rows while forecasting; it
must be a **pure function** returning the same arrays for the same rows and
calendar, since fitted coefficients assume fixed feature definitions and
scaling. It returns a mapping using any subset of these keys:

| Key | Shape | Effect |
| --- | --- | --- |
| `initiation_features` | `[rows, len(calendar), P]` | initiation timing regressors |
| `receipt_features` | `[rows, len(calendar), P]` | receipt timing regressors |
| `initiation_cure_features` | `[rows, Q]` | initiation susceptibility regressors |
| `receipt_cure_features` | `[rows, Q]` | receipt susceptibility regressors |
| `initiation_allowed` | `[len(calendar)]` or `[rows, len(calendar)]` | initiation closures |
| `receipt_allowed` | `[len(calendar)]` or `[rows, len(calendar)]` | receipt closures |

Any other key raises `ValueError` immediately — a misspelled key can never
silently disable a closure mask or drop a regressor. Omitted feature keys mean
zero regressors for that component; omitted allowed-day masks mean all days
open. With `feature_builder=None`, both stages have no regressors and all days
are open, matching the low-level defaults.

The builder owns feature *definitions*, not the wrapper: column order,
encoding, and any training-derived scaling must stay exactly the same between
the fit-time and forecast-time calls. `RetailReturnModel` does not learn,
invert, or cache preprocessing, and it never invents future covariate
scenarios (weather, promotions, etc.) — the calendar and any per-day
regressors your builder produces for future dates are yours to supply
explicitly, the same as with the low-level API below.

Future cohort rows must include any static columns used by your builder,
such as `product` in the example above. `sales_cohorts` supplies identifiers,
sale dates, and quantities; attach the selected series' product attributes
before forecasting, as the full retail example does.

## Model configuration vs. fitted results

- `RetailReturnModel(policy_days=90, age_bins=30, feature_builder=None)` is
  immutable configuration. `age_bins` applies to both stages. It holds no fit
  state and can be reused for any number of independent `.fit()` calls.
- `FittedRetailReturnModel(model, history, initiation_fit, receipt_fit)` is a
  point-in-time snapshot. Both dataclasses are `frozen=True`, which blocks
  reassigning their fields, but does **not** deep-freeze nested state:
  the fitted history frame, feature arrays, and state captured by the builder
  remain mutable. Treat fitted contents and builder inputs as read-only, and
  keep captured feature definitions/scaling fixed. Caller-owned input
  histories are copied when fitting.
- The class changes no statistics: it sequences the same `prepare_history`,
  `make_observations`, `fit_stage`, and `forecast_returns` calls described
  below, using seeds `seed` (initiation) and `seed + 1` (receipt) for the two
  stages. No new inference backend, scan rewrite, or feature-expression
  language is introduced; NumPyro Forecast integration is unchanged.

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
The resulting `RetailHistory` can be passed straight to `RetailReturnModel.fit`.

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

## Results

Both `FittedRetailReturnModel.forecast` and the low-level `forecast_returns` return
a `ReturnForecast`. Daily arrays have shape `[draw, forecast_day]`:

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

## Advanced: low-level fitting and forecasting

`RetailReturnModel` is a thin sequencing wrapper. For direct control over one
stage, custom inference (e.g. MCMC), or a fit lifecycle that doesn't fit the
two-dataclass shape, call the underlying functions yourself:

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
`survival` primitives are independently usable JAX functions. `RetailReturnModel.fit`
calls exactly `make_observations` and `fit_stage` for you, using your
`feature_builder`'s mapping in place of the explicit `features`/`cure_features`/
`allowed` arguments above.

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
`FittedRetailReturnModel.forecast` builds exactly these arrays for you by calling
your `feature_builder` once with `history.frame` and `future_sales` concatenated
in that order.

The supplied calendar must cover the historical origin dates, output horizon,
and **every uninitiated/future cohort's initiation deadline**. That extra
coverage makes total remaining-return expectations possible even for a shorter
output horizon. Future weather and other covariates are explicit scenarios;
TTENet does not silently invent them.

## Compose with NumPyro Forecast

```python
from ttenet import sales_cohorts

# sales_samples comes from numpyro_forecast.forecast:
# [posterior_draw, future_day, sales_series]
future_sales, future_counts = sales_cohorts(sales_samples, future_dates, series=0)
```

Pass **both** outputs to `fitted.forecast(...)` (or the low-level
`forecast_returns`). The frame's `quantity` is the first path; `future_counts`
preserves all paths. The adapter rejects fractional, negative, missing, or
infinite count predictions instead of rounding them. The example shows the
complete fitting and composition code, including a real NumPyro sales model
and `numpyro_forecast.forecast` call — this integration is unchanged by the
`RetailReturnModel` wrapper.

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
partial-unit returns, or rental/inventory feedback. `RetailReturnModel` changes
none of this: it configures and sequences the same fitting and forecasting
calls documented above.

## Development

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```
