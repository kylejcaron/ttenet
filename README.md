# TTENet

Calendar-aware count and time-to-event forecasting with JAX and NumPyro.

**Sale → return initiated → return received**

Some purchases never initiate a return; some initiated returns never arrive.
TTENet fits these event processes from censored unit histories and
propagates dated cohorts through the network. Sales can be fitted jointly or
supplied as one external forecast value. Existing items retain their original
clocks, including items sold before the sales observation window.

## Install and run

Python 3.12 or newer:

```bash
uv sync --all-extras
uv run python examples/retail_returns.py --steps 150 --draws 40
```

For a longer fit and a chart:

```bash
uv run python examples/retail_returns.py --steps 600 --draws 100 \
  --plot /tmp/retail-returns.png
```

The example jointly fits a real NumPyro sales model and two return processes.
It includes two products, a 90-day initiation policy, abandonment, storms,
weekend receipt closures, and histories preceding the sales training window.
It prints held-out scores and checks count conservation. Latent abandonment
labels never enter fitting. Short runs are execution checks, not convergence
or calibration guarantees.

The same example as an illustrated, interactive essay (a marimo notebook with
parameter recovery, held-out checks, and demand/weather scenario controls):

```bash
uv run marimo run examples/retail_returns_blog.py   # or `marimo edit` to see the code
```

### Native survival convolutions

This focused example expresses susceptibility-capable return stages through
[`numpyro_forecast`](https://github.com/juanitorduz/numpyro_forecast)'s native
`Horizon`, `predict`, `draw_posterior`, `forecast`, and `predict_in_sample` APIs:

```bash
uv run python examples/survival_convolution.py
```

The production [shared kernel](src/ttenet/event_times.py) builds first-event
probabilities `K[source_day, event_day, unit]` from susceptibility and survival.
Composing initiation and receipt kernels gives exact expected receipt counts.
Weather and closures make these calendar-dependent kernels, rather than a
single stationary lag curve. The `EventTime` distribution samples one event
date or no event per unit; its registered `slice_time` and `prefix_condition`
operations let upstream `predict` handle censoring and the forecast boundary.
Chained samples retain each unit's actual parent date and conserve counts.

The [runnable example](examples/survival_convolution.py) checks likelihood and
gradient agreement with TTENet, conditional and in-sample predictions, delayed
entry, closures, and analytic convolution against sampled receipts. It fits
initiation and receipt stages jointly on a synthetic 12-unit, 42-day ledger.

The example supplies future units and source dates rather than generating
sales. Network and retail APIs combine the same native event law with their
sales forecasts. Sampling uses actual origins; this small demonstration's
explicit all-origin mean-convolution oracle uses quadratic calendar storage,
unlike production source-aware pool propagation. Short variational fits
establish execution, not calibration or production-scale performance.

## Fit and forecast

A count process follows the functional
[`numpyro_forecast`](https://github.com/juanitorduz/numpyro_forecast) model
protocol: `model(covariates, data=None)`, with `obs` and `forecast` sites.
`Horizon` and `predict` handle the observed prefix and future suffix:

```python
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro_forecast import Horizon, predict
from ttenet import CountProcess, EventProcess, RetailData, RetailReturnModel, date_grid


def sales_model(covariates, data=None):
    horizon = Horizon.from_data(covariates, data)
    rate = numpyro.sample("rate", dist.LogNormal(0, 1).expand([2]).to_event(1))
    rates = jnp.broadcast_to(rate, (covariates.shape[-2], 2))
    predict(horizon, lambda value: dist.Poisson(value), rates)


def initiation_covariates(frame, days):
    return {"susceptibility_features": frame["product"].to_numpy(float)[:, None]}


unit_history = pd.DataFrame(
    {
        "item_id": ["a", "b", "c"],
        "sale_date": ["2026-01-01", "2026-01-03", "2026-01-04"],
        "initiation_date": ["2026-01-10", None, "2026-01-15"],
        "receipt_date": ["2026-01-14", None, None],
        "product": [0.0, 1.0, 0.0],
    }
)
observed = RetailData.from_units(
    unit_history,
    as_of="2026-01-20",
    calendar=date_grid("2026-01-03", "2026-01-20"),
    group_by=["product"],
)
model = RetailReturnModel(
    sales=CountProcess(model=sales_model),
    initiation=EventProcess(age_bins=16, deadline_days=90),
    receipt=EventProcess(age_bins=16, allowed_weekdays=range(5)),
)
fitted = model.fit(
    observed,
    covariates={"initiations": initiation_covariates},
    mode="joint",
    num_steps=150,
    num_samples=40,
    seed=21,
)
result = fitted.forecast(horizon=28, seed=26)
```

The January 1 purchase is retained, but is **not** counted as a January 3 sale.
The count matrix `observed.sales[day, group]` covers only `observed.calendar`.
`observed.groups` defines the series order, in first-appearance order; the
count model's observation axis must use that order. Without `group_by`, there
is one aggregate sales series.

Configuration is reusable; each `.fit()` returns a separate, copied data
snapshot. `fitted.losses` contains one `"joint"` ELBO trace, or separate
node traces in modular mode. `initiation_fit` and `receipt_fit` expose resolved
posterior `StageParameters`; in joint mode their losses refer to the same
joint objective, not separately optimized stage objectives. Nested frames,
mappings, and callback state are read-only by convention, not deeply frozen.

## One future-sales value

Use `SalesForecast` for both fixed scenarios and uncertain trajectories:

```python
from ttenet import SalesForecast

future_sales = SalesForecast.from_frame(
    pd.DataFrame(
        {
            "sale_date": ["2026-02-01", "2026-02-01"],
            "product": [0.0, 1.0],
            "quantity": [100, 60],
        }
    )
)
scenario = fitted.forecast(horizon=28, future_sales=future_sales, seed=26)
```

- No `draw` column: fixed quantities, broadcast over posterior draws.
- A `draw` column: complete trajectories. Every draw/cohort cell is required
  exactly once; missing cells are not implicit zero sales.
- Optional `item_id` identifies cohorts. Without it, date and static attributes
  identify cohorts and identifiers are generated.
- `.cohorts` holds dates and static attributes; `.counts[draw, cohort]` holds
  quantities; `.draw_ids` labels paths. `.to_frame()` returns the complete
  long table. Counts are never rounded or replaced by their mean.
- Large quantities remain counts, not expanded unit records.

To adapt an existing NumPyro Forecast result, preserve every product series:

```python
future_sales = SalesForecast.from_numpyro_forecast(
    sales_samples,  # [draw, future_day, group]
    future_dates,
    groups=observed.groups,
)
result = fitted.forecast(horizon=28, future_sales=future_sales)
```

One source draw broadcasts; otherwise source and parameter draw counts must
match. External trajectories pair by row and do not become a jointly inferred
posterior merely by composition. Internally generated sources carry posterior
identity, so reordering their draw labels preserves the original pairing.

| Modeled sales process | External `future_sales` | Behavior |
| --- | --- | --- |
| Present | Absent | Generate sales from the fitted posterior |
| Absent | Present | Use the supplied sales scenario |
| Present | Present | Replace internal generation with the scenario |
| Absent | Absent | Error: the future source is unspecified |

A supplied scenario is **not additional sales** and does not condition the
posterior on a hypothetical future observation. For no new sales, pass an
explicit empty `SalesForecast.from_frame` with `sale_date` and `quantity`
columns. There is no separate public count-array argument.

## Covariates and closures

`covariates` maps node names to their inputs:

- `"sales"`: the count model's array, time at axis `-2`. Fitting supplies the
  observation window; forecasting supplies the next `horizon` days. The
  fitted prefix is retained automatically. With no regressors, a zero-width
  array still carries the required time axis.
- `"initiations"`, `"receipts"`, or another event node: a mapping below, or a
  pure `provider(frame, days)` callable returning that mapping.

| Event key | Shape | Meaning |
| --- | --- | --- |
| `features` | `[rows, days, P]` | Time-varying timing regressors |
| `susceptibility_features` | `[rows, Q]` | Static susceptibility regressors |
| `allowed` | `[days]` or `[rows, days]` | Boolean hard-open mask |

During fitting, event providers receive historical unit rows and the calendar
from the earliest original sale through `as_of`. During forecasting, they
receive **historical rows followed by new sale cohorts**, and requested future
dates. Cached historical feature values and masks remain unchanged. A fitted
callable can be reused for future dates; an explicit array mapping needs a
future mapping. Product attributes are carried into generated sale cohorts.

Feature order, encoding, and learned scaling must stay fixed. Historical
`susceptibility_features` cannot change in a scenario: they describe susceptibility at
stage entry. Missing data during risk exposure is an error, not zero-filled
weather. Only inactive pre-birth padding is filled. Unknown node/feature keys
and non-mapping provider results fail rather than silently removing closures.

`EventProcess.allowed_weekdays` and the supplied `allowed` mask are ANDed.
Each stage has its own mask. A closed warehouse does not stop online return
initiations, and a closure does not make a unit non-susceptible.

For retail's **eventual historical receipt expectation**, future event
covariates must extend through the latest still-eligible historical sale's
initiation deadline, even if that exceeds `horizon`. Providers receive those
extra dates. The network does not invent future weather. Generic network
forecasts of daily events only require the requested horizon.

## Starting populations

When full `unit_history` is available, include units sold or initiated before
the first training day. No separate starting table is required:

- Earlier sales do not enter the in-window sales likelihood.
- An uninitiated old unit retains age since its original sale and its deadline.
- An already initiated unit retains age since initiation; its receipt can
  occur after the sale's 90-day deadline.
- Complete pre-window event histories remain informative once, not twice.
- Already received units do not produce another receipt.

If only surviving items were recorded at the beginning, pass an optional
`starting_items` snapshot to `RetailData.from_units`. The snapshot is taken at
**end-of-day before the first observed day**. Include known sale/initiation
dates and static group attributes. Later `unit_history` rows with the same
identifier supply subsequent outcomes; missing original dates can be supplied
by the snapshot. Items absent from the later ledger remain censored. Conflicting
origins, prior events, or static attributes are errors, as are post-boundary
snapshot events, new-window sales, and already completed snapshot receipts.

Selected survivors need a conditional-entry likelihood. If susceptible
survival before entry is `S_pre`, the post-entry susceptibility logit is:

```text
logit(pi_at_entry) = logit(pi) + log(S_pre)
```

Pre-entry hazards therefore still need their original historical covariates.
Those days are used for selection conditioning, not counted again as newly
observed follow-up. Stages already completed in the snapshot contribute no
second event likelihood. Do not tag a complete birth-cohort history as a
selected snapshot: doing so discards its pre-entry event information.

Unknown starting ages/counts are not silently guessed. Initial populations
belong to the data, not to reusable model configuration.

## Explicit graphs and joint learning

The retail class is a convenience over the same graph implementation:

```python
from ttenet import CountNode, EventNode, ForecastNetwork

sales = CountNode("sales", model.sales)
initiations = EventNode("initiations", sales, model.initiation)
receipts = EventNode("receipts", initiations, model.receipt)
network = ForecastNetwork(nodes=[sales, initiations, receipts])
network_fit = network.fit(
    observed,
    covariates={"initiations": initiation_covariates},
    mode="joint",
    num_samples=40,
)
network_result = network_fit.forecast(horizon=28)
```

A graph has one count root and acyclic, single-source event nodes. Event nodes
may have further descendants with explicit `event_column` mappings supplied
to `RetailData.from_units`. Each child receives actual dated parent events
and retains root-cohort identity, not just aggregate daily totals. Same-day
transitions are allowed. Branches represent distinct events, **not competing
risks**. `network_result.counts` contains daily arrays by node name;
`network_result.nodes[name].events[draw, day, cohort]` retains lineage, with
`pending[draw, day]` and deadline-limited `eligible[draw, day]` population counts.

`mode="joint"` fits one scoped NumPyro model and posterior. `mode="modular"`
explicitly fits independent blocks and composes them. When all parents are
observed and priors are independent, the joint posterior still factorizes;
putting components in one class does not create cross-stage learning.

For genuine coupling, supply `shared_model()` to the network or retail model.
It samples and returns shared latent values under the `shared` scope:

- The count callable receives an additional `shared=values` keyword.
- A custom `EventProcess(parameter_model=...)` callback receives
  `(observations, shared)` and returns one `StageParameters` value. It may
  sample conditional NumPyro priors or use `sample_stage_parameters` and
  transform those parameters with shared values.
- A custom `EventProcess(family=...)` calls `family.model(inputs, shared)` with
  `TimingInputs` and receives a named `EventLaw(timing, susceptibility_logits)`;
  its real named sample and optimized parameter sites are replayed per draw.
- The same shared posterior draw reaches every process during prediction.
- Shared effects require joint mode; modular mode rejects them.

The default inference helper uses an `AutoNormal` variational guide; its
uncertainty is approximate. `network.numpyro_model(observed, covariates=...)`
returns an ordinary zero-argument joint model for direct NumPyro NUTS/MCMC use.
There is no additional independent likelihood for aggregate returns on top
of the observed unit events.

Model `numpyro.param` sites are optimized point estimates, retained in
`network_fit.params` and reused for prediction; they do not carry posterior
uncertainty. Use `numpyro.sample` priors for uncertain parameters.

## Histories and clocks

`RetailData.from_units` accepts the same layouts as the low-level
`prepare_history` adapter:

| `layout` | Input |
| --- | --- |
| `"tabular"` | `item_id`, `sale_date`, optional `initiation_date`, `receipt_date`, static attributes |
| `"longitudinal"` | Complete snapshots plus `recorded_date`; latest known snapshot |
| `"changes"` | `item_id`, `date`, `event` in sale/initiation/receipt; attributes on sale records |

Missing event dates mean “not observed yet,” not known abandonment. Dates
beyond `as_of` are hidden; future sales are excluded from history. Invalid
observed ordering is rejected, not repaired. `prepare_history(...,
policy_days=90)` remains available for functional workflows; `policy_days=None`
means no initiation deadline. `RetailData` itself imposes no process deadline.

`expand_covariates(records, items, dates, columns, layout="changes")` expands
sparse per-item changes forward, never backward from future records. Dense
`layout="longitudinal"` requires exact daily coverage. Conflicting same-day
updates to one feature are errors.

Date helpers: `to_day`, `date_grid`, `elapsed_days`, `calendar_features`, and
`allowed_days`. Dates are **UTC calendar days**; convert business-local dates
explicitly if UTC boundaries are unsuitable.

- `as_of` is end-of-day; forecasting starts the next day.
- Origin day has age zero. Initiation and receipt can occur on the same day.
- A 90-day policy permits initiation through sale date + 90 days, inclusive.
- Weekends do not extend the deadline. Receipt has no retail policy deadline.
- A reporting delay is not a hard closure of an event that actually occurred.

## Retail results

`FittedRetailReturnModel.forecast` returns `ReturnForecast`; daily arrays have
shape `[draw, forecast_day]`:

| Field | Meaning |
| --- | --- |
| `sales` | The bundled source `SalesForecast` |
| `initiations` | Newly initiated returns |
| `receipts` | Physically received units |
| `eligible` | Uninitiated, within-policy units at end-of-day |
| `open_returns` | Initiated, unreceived units at end-of-day |

Per-draw analytic estimates at the origin:

| Field | Meaning |
| --- | --- |
| `expected_uninitiated_receipts` | Eventual receipts from uninitiated historical units |
| `expected_open_receipts` | Eventual receipts from existing unreceived returns |
| `expected_existing_receipts` | Their sum; excludes future sales |

“Open” is an observed status, not proof that an item will arrive. Abandoned
returns remain unreceived unless the business records an explicit closure;
TTENet does not invent cancellation dates. Binomial transitions conserve
integer population counts in every forecast draw.

## Functional numerical APIs

For direct control, use `prepare_history`, `make_observations`, `fit_stage`,
and `forecast_returns`. Event arrays use full historical row order; receipt
preparation selects initiated units itself:

```python
from ttenet import prepare_history, make_observations, fit_stage, forecast_returns

history = prepare_history(unit_history, as_of="2026-01-20", policy_days=90)
calendar = date_grid("2026-01-01", "2026-06-01")
a = fit_stage(make_observations(history, "initiation", calendar), age_bins=16)
b = fit_stage(make_observations(history, "receipt", calendar), age_bins=16)
result = forecast_returns(history, a, b, calendar=calendar, horizon=28)
```

Those two `fit_stage` calls are independent, not joint fitting. Optional
`future_sales` is one `SalesForecast` or a deterministic quantity frame.
Explicit `initiation_features`, `receipt_features`, static susceptibility features,
and allowed masks use **historical rows followed by future cohorts**. Missing
features for fitted nonzero-width coefficients are errors. The calendar
covers original clocks, the output horizon, and historical initiation deadlines
needed for eventual expectations; new sales' post-horizon deadlines need no
extra coverage.

`make_event_observations` supports arbitrary origin/event columns and explicit
`entry_dates`. `StageObservations`, `stage_log_likelihood`, `stage_model`, and
`sample_stage_parameters` are ordinary JAX/NumPyro building blocks.
`StageObservations.exposure` is administrative follow-up through `as_of` and
the deadline, independent of the observed event's stopping time.
`predict_stage(fit, observations, seed=0)` generates native in-sample trajectories
with shape `[draw, day, unit]`; it does not echo recorded outcomes.
`forecast_events` propagates one source-aware event stage. Fitting, graph and
functional retail forecasts use the same `survival_kernel` and native
`EventTime` / `CohortEventTime` laws through NumPyro Forecast `predict`.
Forecasts use bounded draw blocks under local JAX float64/int64 precision.
No global JAX precision setting is changed; no NumPy event sampler is used.
Direct `CohortEventTime` callers must supply signed JAX integer pools so
impossible-path sentinels remain negative. Host integer arrays are range-checked
and converted to the native signed dtype; local x64 is required for wide pools.
Whole host floating-point observations are converted to native integers when
that conversion is exact. Other host floats must be exactly representable at
the active JAX precision; lossy conversion is refused rather than rounding an
invalid observation onto the event/count support.
Integer totals are exact, but floating sampling proposals do not guarantee
arbitrary-int64 distributional accuracy above `2**53`.

## Interchangeable event-time families

Select a packaged family without changing inference, calendar bookkeeping, or
network propagation:

```python
import math

import numpyro.distributions as dist

from ttenet import EventProcess, WeibullFamily

receipt = EventProcess(
    family=WeibullFamily(
        scale_prior=dist.LogNormal(math.log(6.0), 0.3),
        shape_prior=dist.LogNormal(math.log(2.0), 0.2),
        susceptibility_logit_prior=dist.Normal(1.0, 1.0),
    ),
    allowed_weekdays=(0, 1, 2, 3, 4),
)
```

`WeibullFamily` discretizes susceptible survival at daily bin boundaries. Its
priors can also be fixed real scalars; scale and shape must be positive. Named
sites are `scale`, `shape`, `susceptibility_intercept`, and, when regressors are present,
`beta` and `susceptibility_beta`. Fixed parameters create no sample sites. Distribution
priors must be scalar and have suitable support. `family=None` retains the default regularized
random-walk hazard; its `parameter_model` callback still customizes those priors.
Combining `family` with `parameter_model` is ambiguous and rejected.

**Migration.** `CureProcess` is now `EventProcess`, and every name that
denotes the susceptibility logit or its regressors moved from `cure_*` to
`susceptibility_*`: the `StageParameters` fields `susceptibility_intercept` and
`susceptibility_beta`, the `susceptibility_features` inputs (including the
prefixed `initiation_`/`receipt_` keyword arguments and provider-dictionary
key), the `susceptibility_intercept`/`susceptibility_beta` sample sites, and
`WeibullFamily(susceptibility_prior=...)`, now `susceptibility_logit_prior`
because it is a prior on the logit intercept. There are no aliases: update
callers, saved posterior-site dictionaries, and custom `parameter_model`
callbacks. Mixture-cure mathematics, and never-event wording, is unchanged.

Custom plain NumPyro functions use the frozen `EventFamily` wrapper:

```python
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from ttenet import (
    EventProcess,
    EventFamily,
    EventLaw,
    ProperTail,
    timing_from_log_survival,
)


def exponential_delay(inputs, shared):
    rate = numpyro.sample("rate", dist.LogNormal(-1.0, 0.3))
    susceptibility = numpyro.sample("susceptibility", dist.Normal(0.0, 1.0))
    ages = jnp.maximum(inputs.ages, 0)
    return EventLaw(
        timing=timing_from_log_survival(-rate * ages, -rate * (ages + 1)),
        susceptibility_logits=jnp.broadcast_to(susceptibility, ages.shape[:-2] + ages.shape[-1:]),
    )


family = EventFamily(model=exponential_delay, tail=ProperTail())
custom_receipt = EventProcess(family=family, allowed_weekdays=(0, 1, 2, 3, 4))
```

`family.model(inputs, shared)` receives only ages and regressors:

| Input | Shape and meaning |
| --- | --- |
| `inputs.ages` | `[day, cohort]`, elapsed days from the immediate parent |
| `inputs.features` | `[day, cohort, feature]`, time-varying regressors |
| `inputs.susceptibility_features` | `[cohort, feature]`, static susceptibility regressors |

It returns `EventLaw(timing=TimingLaw(log_hazard, log_survival_step),
susceptibility_logits=...)`, with susceptibility logits `[cohort]`. The common
kernel applies susceptibility marginalization, administrative exposure, elapsed-age closures and
selected-survivor conditioning. Families receive no observed outcomes or
administrative masks. Re-evaluating a family at a new horizon replays its fitted
named NumPyro sites; missing posterior sites are errors, not new prior draws.
`fit_stage(observations, family=family)` uses the same contract.
Replay must also reach every fitted site: omitting regressors cannot silently
drop their learned coefficients. Latent fits use NumPyro's feasible initialization,
not the density of one prior draw, to establish a valid starting point. Fixed
and parameter-only laws still have their observation density checked before fitting.

Default `StageFit.parameters` remains `StageParameters`. For a custom family
it is a mapping from local sample/optimized parameter names to arrays with a
leading draw axis. `StageFit.shared` contains the actual resolved shared values
from those same draws; `.draws` also preserves requested draws for fixed,
zero-latent families without inventing posterior sites.
`.timing(inputs, draw=i)` returns a replayed `EventLaw`.
Pass the whole `StageFit` to forecasting so the family and shared metadata survive.

Every family carries an explicit, immutable tail declaration:

- `ProperTail()`: susceptible units eventually fire under continuing exposure
  and reopening.
- `FiniteTail(last_age=a)`: integrate the masked law through inclusive age `a`;
  a terminal atom on a closure does not imply eventual completion.
- `UnknownTail()`: finite-horizon forecasts are supported, but eventual receipt
  estimates are refused.

Eventual finite-support receipt estimates honor supplied calendar covariates and
closures; beyond that calendar they explicitly assume all-open continuation
with the last regressors held constant. They contract each possible initiation
date with the receipt law, rather than substituting a finite output horizon
for an eventual probability.

`timing_from_log_survival(log_S_a, log_S_next)` forms daily conditional survival
and hazard from susceptible log survival at bin boundaries, including terminal
and exhausted-support cells. It does not discretize arbitrary NumPyro
distributions automatically or accept non-monotone survival as a valid law.
`timing_from_log_masses` adapts discrete masses and a residual tail to the same
interface. These are nonparametric-compatible seams, not a shipped nonparametric
prior or estimator.
Mass grids and residual tails broadcast over their leading axes: `[cohort, age]`
can share a scalar tail, and `[draw, 1, age]` can pair with `[draw, 1]` tails
before being evaluated on a `[day, cohort]` age grid.

Run `uv run python examples/event_time_families.py` for sampled Weibull parameters,
native in-sample predictions and a mixed default → Weibull → default network
forecasting beyond its training window.

## Statistical limits

The default susceptible hazard has a regularized random-walk age baseline and
linear regressors. Susceptibility has a separate logistic regression. Ages past
the learned baseline use the last bin; this is flexible, not assumption-free.
For susceptibility `pi` and susceptible survival `S`, a censored observation
has probability `(1-pi) + pi*S`, and current susceptibility is
`pi*S / ((1-pi) + pi*S)`.

The initiation deadline limits actual returns: **`pi` is not the realized
return rate**. Eventual receipt estimates assume a positive receipt tail and
that receiving reopens. Finite-horizon paths honor the supplied masks. A
permanently closed weekday configuration produces no eventual events.
Finite follow-up cannot distinguish cure from arbitrarily long delay without
assumptions. Use mature cohorts, prior sensitivity, and held-out calibration.

Daily resolution, single-unit histories, homogeneous future cohorts, and at
most one event of each type per unit. No repeated return attempts, partial
units, automatic weather forecasting, or rental/inventory feedback.

Event trajectories retain a dense `[draw, day, cohort]` array per node.
Simulation packs historical units and single-date count arrivals into one
pool per cohort; only genuinely different parent dates need separate pools.
Memory still scales with posterior draws, horizon, cohort count, and graph size.

## Development

```bash
uvx nox                    # lint, complexity, and tests on Python 3.12-3.14
uvx nox -s tests-3.14      # one interpreter
uvx nox -s lint            # ruff check + format check
uvx nox -s complexity      # complexipy vs. complexipy-snapshot.json
```

`complexity` fails when a function is added over complexipy's limit of 15 or an
existing over-limit function gets more complex. After simplifying a function,
regenerate the baseline with `uv run complexipy src --plain --snapshot-create`.
