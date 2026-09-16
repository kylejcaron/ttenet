# Retail Survival Forecasting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fit two censored cure-capable return stages and forecast calendar-day retail return initiations, receipts, and outstanding populations from past and uncertain future sales.

**Architecture:** Host-side dates/history adapters feed JAX discrete-time mixture-cure likelihoods. A count-conserving cohort simulator conditions old histories and propagates new sales through initiation and receipt. Optional NumPyro Forecast integration supplies real posterior predictive future-sales counts.

**Tech Stack:** Python >=3.12, NumPy, pandas, JAX, NumPyro, pytest, uv; optional numpyro-forecast>=0.3,<0.4 and matplotlib for the worked example.

## Global Constraints

- Follow `docs/superpowers/specs/2026-09-16-retail-survival-design.md`.
- UTC calendar days; as-of is end-of-day; forecast starts as-of + 1 day.
- Age zero and same-day transitions are supported. Initiation may occur on sale + policy_days inclusive, default 90; receipt can occur later.
- Keep user's `.gitignore`, `.kata.toml`, and `AGENTS.md` changes untouched and out of commits.
- Worktree path `/Users/kylejcaron/ttenet`, branch `feat/retail-survival`. Do not merge or push.
- Implementers own disjoint files. Skip builds, linters, formatters, and tests during the concurrent edit wave. Coordinator validates once the wave is stable. Do not commit from concurrent agents.
- Every test must defend an observable boundary, invariant, or mathematical result. No source-text or mock-forwarding tests.
- All numerical stage parameters are posterior-draw aware; do not replace draws with means.

## Shared contract and file map

`src/ttenet/dates.py`, `data.py`: date conversion, canonical history, covariate expansion. `survival.py`, `models.py`: stage parameters, conditional likelihoods, NumPyro fitting. `forecast.py`: cohort propagation and outstanding estimates. `integration.py`: NumPyro Forecast sales draws to future cohorts. `__init__.py`: public exports. `examples/retail_returns.py`: runnable end-to-end example. Packaging is `pyproject.toml`, tests are under `tests/`.

```python
# data.py
@dataclass(frozen=True)
class RetailHistory:
    frame: pd.DataFrame  # canonical columns, RangeIndex, original item order
    as_of: np.datetime64  # datetime64[D]
    policy_days: int


def prepare_history(data, *, as_of, layout="tabular", policy_days=90) -> RetailHistory: ...
def expand_covariates(records, items, dates, columns, *, layout="changes") -> np.ndarray: ...


# changes: item_id/date + feature columns; sparse cells are unchanged.
# longitudinal: exact daily rows; missing required coverage is an error.
# Returns [item, calendar_day, feature]. No backward fill.


# dates.py
def to_day(values): ...  # scalar or array, datetime64[D], missing allowed as NaT


def date_grid(start, end): ...  # inclusive daily numpy array


def elapsed_days(origin, dates): ...  # integer day differences


def calendar_features(dates) -> pd.DataFrame: ...  # date, weekday, weekday_sin/cos


def allowed_days(dates, *, weekdays=range(7), closed_dates=()) -> np.ndarray: ...


# survival.py
class StageParameters(NamedTuple):
    age_logits: Any  # [K], or posterior [draw, K]
    beta: Any  # [P], or [draw, P]
    cure_intercept: Any  # scalar or [draw]
    cure_beta: Any  # [Q], or [draw, Q]


def stage_hazard(parameters, ages, features, allowed=None): ...
def susceptibility(parameters, cure_features): ...
def conditional_susceptibility(probability, log_survival): ...


# Numerical functions operate on one parameter draw; caller vmaps/selects draws.


# models.py
@dataclass(frozen=True)
class StageObservations:
    ages: Any  # [N,T], negative before origin
    features: Any  # [N,T,P]
    cure_features: Any  # [N,Q]
    at_risk: Any  # [N,T], includes event day, clipped at as_of/deadline
    allowed: Any  # [N,T], hard closures, distinct from observation/censor mask
    event_index: Any  # [N], calendar index, -1 when censored


@dataclass(frozen=True)
class StageFit:
    parameters: StageParameters  # leading posterior draw dimension
    losses: Any


def make_observations(
    history, stage, calendar, *, features=None, cure_features=None, allowed=None
) -> StageObservations: ...
def stage_log_likelihood(parameters, observations): ...  # [N], differentiable


def stage_model(observations, *, age_bins=30): ...
def fit_stage(
    observations, *, age_bins=30, num_steps=500, num_samples=100, seed=0, learning_rate=0.02
) -> StageFit: ...


# forecast.py
@dataclass(frozen=True)
class ReturnForecast:
    dates: Any  # [horizon]
    initiations: Any  # [draw,horizon], integer counts
    receipts: Any
    eligible: Any  # uninitiated and not past deadline, end-of-day
    open_returns: Any  # initiated, not received; not a count of known abandonments
    expected_existing_receipts: Any  # [draw], total eventual receipts from past sales
    expected_uninitiated_receipts: Any  # [draw]
    expected_open_receipts: Any  # [draw]


def forecast_returns(
    history,
    initiation,
    receipt,
    *,
    calendar,
    horizon,
    future_sales=None,
    future_counts=None,
    initiation_features=None,
    receipt_features=None,
    initiation_cure_features=None,
    receipt_cure_features=None,
    initiation_allowed=None,
    receipt_allowed=None,
    seed=0,
) -> ReturnForecast: ...


# future_sales: DataFrame item_id,sale_date,quantity (cohort rows, extra features allowed).
# future_counts: optional nonnegative integer [draw, future_cohort] overriding quantity.
# Both parameter arguments accept one draw or a leading posterior draw axis.
# Feature arrays [history rows + future cohorts, len(calendar), P].
# Cure features [all rows,Q]. Allowed [T] or [all rows,T].
# None -> width-zero features or all-allowed masks. If model width >0, omission errors.
# Calendar must start by earliest historical origin and extend through BOTH output
# horizon and all uninitiated/future cohort initiation deadlines for eventual estimates.
# Future cohorts must occur within the output horizon; reject duplicate ids and overlap.
```

## Task 1: Canonical histories, clocks, and regressors

**Owner:** data implementer. **Files:** create `src/ttenet/dates.py`, `src/ttenet/data.py`, `tests/test_data.py`.

- [x] Implement shared signatures with descriptive docstrings and validated canonical frames. Tabular uses canonical columns; absent event-date columns mean unobserved. Longitudinal uses `recorded_date` and full snapshots; changes use `date,event` with sale/initiation/receipt. Filter records to as-of before deriving known outcomes. Preserve static features on sale rows. Canonical output adds `age_since_sale`, nullable `age_since_initiation`, `eligible`.
- [x] Derive eligibility as no observed initiation and `0 <= age_since_sale <= policy_days`. Dates after cutoff become NaT; sales after cutoff are excluded. Reject impossible observed sequences and initiation after deadline. Receipts without observed initiation cannot enter receipt risk sets.
- [x] Implement per-column forward propagation for sparse change records. Do not interpret an unchanged sparse cell as missing and overwrite its prior value. Missing values before the first known value are errors on requested risk dates.
- [x] Add meaningful boundary tests, including this behavior:

```python
history = prepare_history(
    pd.DataFrame(
        {
            "item_id": ["a", "b"],
            "sale_date": ["2026-01-01"] * 2,
            "initiation_date": ["2026-01-10", "2026-04-02"],
        }
    ),
    as_of="2026-01-05",
)
assert history.frame.initiation_date.isna().all()
assert history.frame.eligible.all()
```

Also cover equivalent three layouts, day-90/day-91, timezone normalization, future-only sparse feature rejection, and same-day events. Return precise file/behavior report; no concurrent validation commands.

## Task 2: Stable cure likelihood and flexible NumPyro stages

**Owner:** numerical implementer. **Files:** create `src/ttenet/survival.py`, `src/ttenet/models.py`, `tests/test_survival.py`.

- [x] Implement single-draw numerical primitives with JAX arrays and masked hazards. Tail age index is clipped to the final baseline bin. Negative ages must not accidentally accumulate survival; caller risk masks exclude them.
- [x] Implement likelihood in log space; risk mask includes only observed days through event/censor/deadline. Avoid undefined gradients from computing unused impossible branches. Observed event on a closed day has negative-infinite likelihood, never clipped to a small nonzero probability.

```python
# At h=.5, pi=.4, event on second exposed day has probability .4*.5*.5=.1.
# Censor after two exposed days has probability .6+.4*.25=.7.
np.testing.assert_allclose(np.exp(log_likelihood), [0.1, 0.7], rtol=1e-6)
# Conditional susceptibility after those two days is .1/.7 = 1/7.
```

- [x] `make_observations` uses the exact calendar and source-row feature indexing. Initiation uses all historical sales; receipt uses initiated sales only. Filter corresponding feature/cure/mask rows for receipt, resetting age at initiation. Require calendar history coverage; future days are not likelihood contributions.
- [x] Implement NumPyro stage model with regularized flexible age baseline, linear time-varying regressors, and static susceptibility regression. Document site names and priors; provide real SVI with AutoNormal returning draws and losses. No custom inference engine.
- [x] Tests defend analytical probabilities, hard-zero days, same-day events, masking/censoring, finite gradients for ordinary valid inputs, and tail continuation. Skip validation during concurrent wave.

## Task 3: Conditional, count-conserving return forecasts

**Owner:** forecast implementer. **Files:** create `src/ttenet/forecast.py`, `tests/test_forecast.py`.

- [x] Implement the shared API using host-side NumPy count simulation and stable logit arithmetic consistent with the JAX stage likelihood. Validate feature widths/calendar coverage/count support at the boundary. Normalize single vs posterior parameter draws without averaging; align or broadcast a single parameter draw with future count draws.
- [x] For historical uninitiated items, compute conditional susceptibility from all observed initiation risk days. For historical initiated/not-received items, condition receipt susceptibility from initiation through as-of. Exclude received units and expired uninitiated units. Future cohorts start with count quantities, not materialized unit rows.
- [x] Sequentially simulate initiation then receipt each calendar day, permitting same-day receipt of new initiations. Maintain separate receipt cohorts by initiation date, since their age clocks differ. Binomial conditional hazards include the current posterior susceptible fraction; update survival for survivors. Counts may never be negative or exceed remaining source population.
- [x] Compute exact expected remaining receipts at origin using the policy-limited initiation event probability and receipt susceptibility, plus conditioned existing open returns. Do not count future sales in `expected_existing_receipts`. Report the receipt tail assumption in docstrings.
- [x] Include a deterministic mass-conservation case with all-susceptible near-certain hazards, a zero-cure probability case, a conditional analytical example, abandonment, day-90 policy, no-weekend receipts, late receipt after initiation expiry, heterogeneous origin clocks, and uncertain future counts.

```python
assert np.all(
    result.receipts.sum(axis=1)
    <= (historical_open + historical_eligible + future_counts.sum(axis=1))
)
assert np.all(result.receipts[:, weekend_output_mask] == 0)
np.testing.assert_allclose(
    result.expected_existing_receipts,
    result.expected_open_receipts + result.expected_uninitiated_receipts,
)
```

No test should merely assert these field copies; use analytically known expected probabilities and independently known source counts. Skip all validation during concurrent wave.

## Task 4: Packaging, real forecasting integration, and worked example

**Owner:** coordinator/integration. **Files:** create `pyproject.toml`, `src/ttenet/__init__.py`, `src/ttenet/integration.py`, `examples/retail_returns.py`, `tests/test_integration.py`; update `README.md`.

- [x] Package with hatchling and uv; base dependencies NumPy/pandas/JAX/NumPyro. Optional forecast and example extras. Dev dependencies pytest/ruff. Expose the agreed APIs without duplicate abstractions.
- [x] Implement `sales_cohorts(samples, dates, *, series=0, prefix="future") -> (DataFrame, np.ndarray)` for actual NumPyro Forecast samples `[draw,time,obs]`. Preserve each integer draw exactly, one homogeneous future cohort per date for the selected series. Reject invalid count draws rather than rounding or silently replacing with expectations.
- [x] Example simulates real item-level truth using cure draws and age/calendar-dependent hazards, then creates a censored training snapshot. Set policy 90, weekend receipt closures, and a storm regressor. Fit both stages with real SVI, keeping future records hidden. Fit a NumPyro sales model and obtain actual `numpyro_forecast.forecast` draws for future quantities. Run return forecasts, print outstanding estimates and held-out count forecast scores, and optionally save a plot.
- [x] README documents install, runnable command, low-level likelihood and fitting APIs, three input representations, exact date/policy semantics, scenario coverage, finite-data cure limitations, and continuing receipt tail. No claims of generic event networks or perfect cure identification.

## Verification and review gates

- [x] Install: `uv sync --all-extras`.
- [x] Focused combined tests: `uv run pytest -q` (the repository begins with no existing suite).
- [x] Execute actual example: `uv run python examples/retail_returns.py --steps 150 --draws 40`.
- [x] Validate generated counts, zero weekend receipts, policy boundaries, finite fitted losses/draws, outstanding decomposition, and held-out scores from the example output.
- [x] Run `uv run ruff check .` and `uv run ruff format --check .`, applying the formatter at integration.
- [x] Obtain independent numerical and data/API reviews; fix actionable findings and rerun their covering cases. Reviewers skip build/lint/test commands.
- [x] Commit only implementation/doc/example files and lockfile; preserve unrelated existing working-tree changes. Close tracked work with exact command/output evidence. Leave branch unmerged and report run instructions and modeling limitations.

## Completion evidence

- `uv run pytest -q`: **97 passed**.
- `uv run ruff check .` and `uv run ruff format --check .`: passed.
- `uv build`: source distribution and wheel built successfully.
- `uv run python examples/retail_returns.py --steps 150 --draws 40 --plot /tmp/ttenet-retail-returns.png`: real SVI fits and upstream sales forecasting completed with finite losses, exact daily outstanding-count conservation, and zero weekend receipts. Plot inspected.
- Example: 366 historical sales, 112 initiations, 142 eligible uninitiated units, and 22 open returns. Expected remaining receipts from past sales: 9.9532. Forecast receipts including future sales: mean 15.8, total 90% interval [8.95, 25.15], versus 10 held-out receipts. These are synthetic smoke-run results, not a convergence or calibration guarantee.
- Direct public `stage_model` NUTS smoke: five warmup steps, three finite posterior draws using `StageObservations` as a pytree.
- Independent data, numerical, forecast, and integration reviews completed; final integration verdict: spec compliance PASS, code quality PASS, no shipping blockers.
- `roborev fix --list --branch feat/retail-survival`: no open jobs.

### Implementation decisions and regression coverage

- Forecast propagation stays in host float64 NumPy rather than passing sigmoid probabilities through JAX float32. It uses the same logit-defined hazards, avoids saturation/cancellation, and does not change global JAX configuration.
- Likelihood uses `log_sigmoid` directly on finite logits; closure days remain exactly impossible. Regression cases cover finite gradients at strong logits and cancellation-free remaining-initiation probability.
- Reproduced and fixed stale derived clocks, duplicated longitudinal indices, conflicting same-feature/day updates, missing future sale dates, and the final-day loss of abandoned future-return cohorts.
- Actual fitted SVI-to-forecast execution covers posterior-axis compatibility, including zero-width receipt cure features. Kept behavioral regressions rather than adding shape-only plumbing tests.
- Source distribution explicitly selects package, examples, tests, and public metadata; local agent/kata configuration is excluded. User-owned repository configuration remains unchanged and outside implementation commits.

## Follow-up: thin public model interface

**Goal:** Expose reusable configuration and separate fitted forecasts without replacing the functional numerical core. The user selected this direction after seeing the functional API.

**Architecture:** New `src/ttenet/retail.py` holds frozen `RetailReturnModel` and `FittedRetailReturnModel`; `src/ttenet/__init__.py` exports them. A single feature-builder callable uses the six existing forecast keyword names. Explicit calendars/scenarios, numerical priors, count simulation, and upstream sales forecasting remain unchanged.

### Public contract

```python
model = RetailReturnModel(policy_days=90, age_bins=16, feature_builder=retail_features)
fitted = model.fit(
    sales,
    as_of=cutoff,
    calendar=calendar,
    num_steps=150,
    num_samples=40,
    seed=21,
)
result = fitted.forecast(
    calendar=calendar,
    horizon=28,
    future_sales=future_sales,
    future_counts=future_counts,
    seed=26,
)
```

`retail_features(frame, calendar)` returns a mapping with keys drawn from `initiation_features`, `receipt_features`, `initiation_cure_features`, `receipt_cure_features`, `initiation_allowed`, `receipt_allowed`. Historical rows are canonical and future records are hidden at fitting; forecasting builds features once for historical rows followed by future cohorts. Missing keys retain existing zero-regressor/all-open defaults; unknown keys fail explicitly.

`fit` also accepts a copied `RetailHistory` with matching policy, without cutoff/layout overrides. It calls `make_observations` for each stage and `fit_stage` with seeds `seed` and `seed + 1`. Returned attributes are `model`, `history`, `initiation_fit`, `receipt_fit`; each call creates a distinct fitted snapshot. All existing inference controls pass through unchanged.

### Implementation

- [ ] Main: add behavior-first tests in `tests/test_retail.py`. Exercise actual SVI fits, then fit the same configuration to a completed population and verify the earlier fit's forecasts are unchanged. Use real fixed posterior parameters to verify two future cohorts with different receipt closures preserve draw-specific quantities and remain correctly ordered. Reject misspelled feature keys rather than silently allowing closed-day receipts. Observe missing-API failures before implementation.
- [ ] Model owner: implement `src/ttenet/retail.py` and exports only. Reuse `prepare_history`, `make_observations`, `fit_stage`, and `forecast_returns`; no duplicate validation of numerical shapes, date rules, or likelihoods. Copy caller-owned canonical histories; keep fit state separate from configuration.
- [ ] Example/docs owner: migrate `examples/retail_returns.py` and the README's primary call flow to the class. Adapt the existing `feature_arrays` helper to return the shared mapping and include the existing weekend mask. Keep SVI seeds, upstream `numpyro_forecast.forecast`, sales draws, forecast seed, scores, and exact conservation checks unchanged. Keep low-level documentation for direct numerical use.
- [ ] Main: run `uv run pytest -q`, `uv run python examples/retail_returns.py --steps 150 --draws 40 --plot /tmp/ttenet-retail-class.png`, Ruff checks, and `uv build`. Compare deterministic example metrics with the preceding functional run. Obtain scoped API/behavior review and resolve findings.

Parallel ownership: model and example/docs edits are disjoint; both implementers skip builds, tests, formatters, and linters. Main owns tests, design/plan updates, and final verification. Preserve `.gitignore`, `.kata.toml`, and `AGENTS.md`; stay on `feat/retail-survival`, without merge/push.
