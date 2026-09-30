# Native Survival-Convolution Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace every package survival-model execution path with one cure-capable event-time kernel and native NumPyro Forecast distributions, but only after demonstrating correctness, cohort scalability, and a worthwhile integration benefit.

**Architecture:** Keep `CureProcess`, `StageParameters`, the model priors, and the public fit/forecast contracts. One log-space survival kernel supplies event likelihoods, conditional future laws, integer cohort sampling, and convolution expectations. Use upstream `Horizon`, `predict`, `draw_posterior`, `forecast`, and `predict_in_sample`; do not introduce another forecasting driver or leave two production backends.

**Tech Stack:** Python >=3.12, NumPy >=2.0, pandas >=2.2, JAX >=0.10.0,<0.11.3, NumPyro >=0.22.0, numpyro-forecast >=0.4.0,<0.5, pytest, marimo, Ruff, nox.

## Global constraints

- This change currently delivers a plan and a blog explanation, not the production migration.
- Worktree: `/Users/kylejcaron/ttenet/.worktrees/native-survival-convolution`; branch: `prototype/native-survival-convolution`; draft PR: https://github.com/kylejcaron/ttenet/pull/6.
- Durable parent: `5jvf`. Task references and real prerequisites appear below. Leave implementation children open until their deliverables are verified.
- Task 1 is a go/no-go gate. A failed gate stops the cutover; do not relax it, narrow integer support, substitute independent Poisson counts, or call a partially migrated package complete.
- Preserve public signatures, result shapes, validations, custom priors, optimized `numpyro.param` values, and the default regularized random-walk age prior. The prototype's five-parameter prior is a fixture, not the replacement package model.
- Preserve Python 3.12/3.13/3.14 coverage and existing dependency version ranges. Promote numpyro-forecast to a required dependency only when introducing the accepted native backend.
- No global JAX precision mutation at import, hidden host callbacks masquerading as native forecasting, unit expansion of count cohorts, or mandatory dense all-origin kernels.
- Existing random seeds must remain reproducible within the new implementation. Different sampling algorithms need distributional parity, not bit-identical draws against the old generator.
- Keep tests of public behavior. Remove tests tied only to obsolete implementation details rather than re-pinning those details.
- Existing unrelated work and PRs remain untouched. Do not merge PR #6 as part of this plan.

---

## 1. What is established, and what is not

Current production paths are split:

| Path | Current implementation | Required destination |
|---|---|---|
| Stage likelihood and SVI | `models.stage_log_likelihood`, `stage_model`, `fit_stage` | Shared event-time law; native observation sites |
| Network likelihood | `network._event_model` adds a separate `numpyro.factor` | The same native law under each node's scope |
| Survival mathematics | `survival.py` JAX primitives; repeated host formulas in `forecast.py` | One log-space mathematical implementation |
| Cohort simulation | `forecast.forecast_events`, NumPy binomial depletion of actual parent-date pools | Exact cohort allocation from the shared kernel, driven through native predictions |
| Historical eventual expectations | `forecast.expected_return_receipts` | Shared probabilities and explicit tail assumptions |
| Network / retail forecast | `FittedNetwork._forecast`, `FittedRetailReturnModel.forecast` | Native cure stages preserving existing graph and result semantics |
| Demonstration | `examples/survival_convolution_model.py` and its fixture | Examples importing the package implementation |

On 2026-09-30, this command passed on the current draft branch:

```bash
/usr/bin/time -l uv run --extra forecast python examples/survival_convolution.py
```

Observed: numpyro-forecast 0.4.0; maximum single-stage likelihood error `3.8147e-6`, gradient error `9.5367e-7`; joint likelihood error `0`, gradient error `2.3842e-7`; 8,192 chained draws with no lineage violations and maximum receipt-probability discrepancy `0.0057565`. The full process took 20.22 seconds and reported 2,339,504,128 bytes maximum RSS on this Darwin arm64 workstation. These totals include compilation, fitting, and several checks: **they are not a forecast benchmark or evidence of a speedup**.

Still unproved: native count-cohort sampling at existing integer limits, realistic memory/runtime, all package entry/mask contracts, stochastic future sales integration, shared priors across a full graph, and stable extreme/long-history behavior. Native API reuse is an architectural benefit; improved statistical accuracy is not expected merely from changing representation.

## 2. Model and representation

For a source at calendar day `s`, unit/cohort `c`, and fixed parameter draw, define

\[
K_{s,t,c}=\pi_c\,S_c(s,t^-)\,h_c(t-s,t),\qquad t\ge s.
\]

`S` is susceptible survival before the event day. A closure contributes zero hazard, not an additional cure observation. Deadlines are inclusive; a child starts at its immediate parent's actual event day, allowing a same-day transition.

The finite-window no-event category has mass

\[
q_{s,c}=1-\sum_t K_{s,t,c}=(1-\pi_c)+\pi_c S_c(s,T).
\]

It includes cure **and** susceptibility surviving the finite horizon (including policy-expired survivors). Do not equate it with permanent cure or compute it by unstable `1 - sum(probabilities)` in production. Use log survival and `logaddexp`.

For selected survivors entering after their origin, shift the susceptibility logit by pre-entry log survival, then accumulate only post-entry survival. This preserves the existing stable conditional-entry likelihood; do not blindly promote the prototype's subtraction of large log-survival terms.

For fixed parameters and a homogeneous cohort, composition gives

\[
(K_A\circ K_B)_{s,t,c}=\sum_u K_{A,s,u,c}K_{B,u,t,c}.
\]

This is a calendar-dependent kernel composition. It becomes ordinary lag convolution only with stationary hazards/exposure. Historical open stages require their separately conditioned residual-life contribution. Convolve within each posterior draw and cohort, then aggregate: multiplying posterior-averaged kernels discards parameter dependence.

**Expectations are not trajectories.** For paths, allocate a pool of `n` units jointly across event dates plus the no-event category, then pass the actual parent-date allocations to its child. An independent Poisson draw per day is not an acceptable substitute.

### Candidate internal contract

Task 1 must validate this contract before its production implementation. These are proposed new internal names, not existing public APIs:

```python
class SurvivalKernel(NamedTuple):
    log_mass: Array  # [..., time, cohort]
    log_tail: Array  # [..., cohort], no event by the window end


def survival_kernel(
    parameters: StageParameters,
    *,
    ages: Array,
    features: Array,
    cure_features: Array,
    allowed: Array,
    exposure: Array,
    pre_entry: Array | None = None,
) -> SurvivalKernel:
    """One draw; retain calendar time at -2 and cohort identity at -1."""
```

`exposure` represents administrative entry/exit/deadline bounds, not the realized event's stopping time. Observation-specific event masks must never leak into a generative forecast law. `pre_entry` conditions on known prior survival without restarting the age clock or counting that evidence twice.

- `EventTime(log_mass, log_tail)`: unit trajectories; `unit_log_prob(data)` retains the cohort axis; `log_prob(data)` sums cohorts.
- `CohortEventTime(log_mass, log_tail, total_count)`: the corresponding integer allocation law for homogeneous, actual-origin pools. Unit count one is the same law, not different cure mathematics.
- Both distributions support `sample`, `mean`, `log_prob`, and native `slice_time` / `prefix_condition`. Slicing marginalizes outside-window event mass into the window's no-event category. Prefix conditioning removes observed events from remaining population and renormalizes future categories, with zero population absorbing.
- The native call is `predict(h, lambda mass: EventTime(mass, kernel.log_tail), kernel.log_mass)`, or its count-cohort equivalent. It returns `None`; obtain generated paths from NumPyro sites, not a fabricated return value.
- Validate infeasible observations at host boundaries; impossible paths have `-inf` likelihood. Never turn an impossible history into certain cure as a fallback.
- Keep the public `stage_log_likelihood` entry point as a real evaluator of the shared law, not as a second formula. Existing public convenience functions are not deprecated aliases to remove.

## 3. Adoption gates

All four gates are mandatory. Record measured results, environment, reference revision, seeds, dimensions, and pass/fail in `docs/superpowers/plans/2026-09-30-native-survival-convolution-evidence.md` during Task 1; append final results after Task 6. Do not create a blank evidence file now.

### A. Statistical and numerical equivalence

- Ordinary float32 likelihood/gradient cases: `atol=3e-5, rtol=1e-5`; analytic extreme-logit cases retain their existing tighter tolerances and finite-gradient assertions.
- Retain closed-day impossibility, closure-neutral survival, inclusive deadlines, same-day events, empty windows/cohorts, heterogeneous origins, tail-bin reuse, and selected-survivor conditioning.
- Check both synthetic and package observations: entry before origin, entry after the fitting window, events before entry excluded, and old origins preceding the sales training window.
- Long-history cases at 365 and 3,650 days and logits `-100`, `0`, `40`, `100`: compare against independent float64/log-space identities, not the new kernel against itself.
- For Monte Carlo checks use exact moments and deterministic seeds. Require integer conservation and chronology exactly; use analytically justified sampling-error bounds for means/covariances. Exact path equality with the old PRNG is not required.

### B. Population and public-contract preservation

- No event before birth/parent date; no duplicate unit event; no child counts exceeding realized parent arrivals; retain root-cohort identity on arbitrary-depth graphs.
- Preserve `pending` versus `eligible`: policy-expired units remain pending but cannot initiate after their deadline.
- Preserve int64 quantities, including the existing `2**53 + 1` near-certain-event contract and overflow rejection. Test default JAX precision as well as explicit x64. Simply turning on x64 does not prove a sampler preserves these integers.
- Cohort storage and work must depend on cohort/date pools, not total unit quantity. Increasing a fixed pool from 10 to 10,000,000 units must not materialize more trajectories.
- Preserve latent/shared and optimized parameters, conditional-entry semantics, finite versus eventual expectations, and posterior draw labels/paired sales scenarios.

### C. Genuine native integration

- Real SVI and upstream `draw_posterior`, `forecast`, and `predict_in_sample` execute the custom distributions. The registered surgeries must handle multiple observed-prefix and future-horizon lengths without monkeypatching upstream.
- Observed likelihood and generated suffix come from upstream `predict`; no second day-by-day simulation hidden behind a deterministic `forecast` site.
- Demonstrate fit-time and forecast-time shape changes, integer observed data, zero future arrivals, sparse versus dense parent dates, arbitrary descendant nodes, and no hidden-outcome leakage.
- Avoid a dynamically growing unit axis inside a traced model. Host cohort preparation may remain, but native sampling must consume fixed-shape pool data; prove how pools and lineage map back to public results.

### D. Measured engineering benefit

- Benchmark the old committed implementation and candidate in separate processes on the same machine, using identical parameters and inputs. Do not compare the prototype's whole 20-second script against one forecast call.
- Measure cold compile+first call, median of five warmed calls after synchronization, and peak RSS. Include the native driver, conditioning, packing, and output aggregation in end-to-end forecast timing.
- Baseline matrix: `(cohorts, history_days, horizon, draws)` = `(100,60,28,32)`, `(1000,180,90,128)`, `(5000,365,365,32)`. Include sparse and multi-day parent arrivals, homogeneous quantities 1 and 10,000,000, heterogeneous closures, and both unit-history and future-cohort populations.
- Proposed acceptance budget: warmed forecast and peak RSS each <=1.25x the old implementation for every matrix case; cold wall time <=2x. Measure fitting/gradient timing separately and reject a >1.25x warmed regression there too. These are prospective budgets, not measured claims; changing them requires a recorded user-approved tradeoff.
- In addition, the cutover must actually remove duplicated likelihood/hazard/conditioning implementations and use native survival observation sites. Passing a timing budget while retaining the old implementation behind a wrapper does not satisfy the goal.
- If dense kernels exceed the budget, stream or batch actual-origin pools and mean contractions. Do not assume an FFT is valid for weather-, calendar-, or unit-dependent hazards.

## 4. File ownership and scope

| File | Responsibility after migration |
|---|---|
| `src/ttenet/survival.py` | `StageParameters`, one stable log-space kernel and public probability primitives |
| `src/ttenet/distributions.py` (new) | Unit/cohort event-time laws and native surgery registrations; no data-frame work |
| `src/ttenet/models.py` | Observation adapters, existing priors, standalone native stage fitting |
| `src/ttenet/forecast.py` | Host validation/pool preparation, public result assembly, shared-kernel expectations; no second cure formula |
| `src/ttenet/network.py` | Scoped graph composition, shared priors, posterior/draw alignment and native cure forecasting |
| `src/ttenet/processes.py`, `src/ttenet/retail.py`, `src/ttenet/__init__.py` | Preserve public facade, configuration, contracts and exports |
| `src/ttenet/integration.py` | Preserve source counts and draw identity; only change if needed to connect the native path |
| `pyproject.toml`, `uv.lock`, `noxfile.py` | Required native dependency, clean optional-extra cutover and verification environments |
| `tests/test_survival.py`, `tests/test_forecast.py` | Mathematical and population boundary contracts |
| `tests/test_network.py`, `tests/test_integration.py`, `tests/test_retail.py` | Consumer-visible full-network behavior |
| `tests/test_distributions.py` (new) | Uncertain distribution/surgery edges and count conservation, not source-text/wiring assertions |
| `examples/survival_convolution.py` | Runnable native example importing the package |
| `examples/retail_returns.py`, `examples/retail_returns_blog.py`, `README.md` | Actual public usage and concise explanation |

`data.py`, `dataset.py` and `dates.py` are not alternative survival engines; leave their data semantics alone. Do not convert the count-root model to a survival process. Sibling events remain distinct events, not competing risks.

## Task 0: Deliver the plan and a truthful blog explanation

**Issue:** `ersy`. **Depends on:** nothing. **Scope of the current change.**

**Files:** this plan; `examples/retail_returns_blog.py`, immediately after the censoring callout in Part 2.

- [ ] Commit this plan and its adoption gates; attach one durable issue per task with the dependencies below.
- [ ] Add one short hidden-code markdown cell titled “From clocks to a survival convolution.” Explain probability mass spreading through calendar time, then show the kernel-composition identity and a native `Horizon`/`predict` sketch.
- [ ] Mark the sketch as the native prototype, not the currently executed `CureProcess` backend. State that the kernel builder and event-time distribution are TTENet components, not built-in NumPyro Forecast models.
- [ ] Keep the code block short, mathematical expectations distinct from sampled paths, and the same typography as adjacent prose. No new widget, styling system, dependency, or chart is needed.
- [ ] Run the notebook with its real defaults, inspect the new section in a browser at desktop and narrow width, and run `marimo check` plus Ruff. Record only observed results.

## Task 1: Prove native cohort sampling and establish adoption evidence

**Issue:** `jtr1`. **Depends on:** Task 0. **No production cutover in this task.**

**Files:** `examples/survival_convolution.py`, `examples/survival_convolution_model.py`; create the evidence document specified above after measurements. Benchmark/reproduction scripts may remain throwaway unless they provide a reproducible consumer-relevant check worth retaining.

**Consumes:** current production `StageParameters`, `forecast_events`, `stage_log_likelihood`; native 0.4.0 API. **Produces:** a pass/fail report and a demonstrated count-cohort allocation/shape/precision strategy satisfying all adoption gates.

- [ ] Re-run the current example and existing focused contracts before choosing a sampler:

```bash
uv run --extra forecast python examples/survival_convolution.py
uv run --extra forecast pytest -q tests/test_survival.py tests/test_forecast.py tests/test_integration.py tests/test_network.py tests/test_retail.py
```

- [ ] Exercise the candidate native count distribution, not just unit trajectories. For one homogeneous pool with `n=7` and category probabilities `[0.2,0.3,0.5]`, compare event means `[1.4,2.1]`, variances `[1.12,1.47]`, and covariance `-0.42`; require integer nonnegative samples and total observed events <=7. After observing two events in category one, the remaining two categories have probabilities `[0.375,0.625]` for the five remaining units.
- [ ] Run the existing `test_int64_cohort_counts_do_not_round_through_float64` against the candidate end-to-end path. Also check exact total preservation in non-degenerate allocations; a near-certain special case alone does not establish safe integer accounting.
- [ ] Prove native tracing with real count draws, delayed entry, changing horizon lengths and actual-origin pools. Record whether precision and static-shape requirements are compatible with the package, not merely with a specially configured demo.
- [ ] Run gates A–D and write actual timings, memory, correctness results, and the chosen allocation strategy. Preserve the old baseline revision in the report so later tasks can rerun it after obsolete code is removed.
- [ ] Record **GO** only if every gate passes. If not, finish with **NO-GO**, exact failing scenarios, and a concrete alternative for review; downstream tasks remain blocked. Do not treat a failed gate as successful completion of a prerequisite.

## Task 2: Implement the shared survival law and native distribution

**Issue:** `jggf`. **Depends on:** successful Task 1.

**Files:** `src/ttenet/survival.py`; new `src/ttenet/distributions.py`; `tests/test_survival.py`; new `tests/test_distributions.py`; dependency/configuration files and existing commands referring to the removed extra.

**Consumes:** the validated kernel/count contract and precision strategy. **Produces:** `SurvivalKernel`, `survival_kernel`, `EventTime`, `CohortEventTime`, native surgery registrations, and a default installation capable of executing them.

- [ ] Establish regression cases before implementation. This tiny law has an independent exact oracle:

```python
def test_event_time_prefix_keeps_observed_units_absorbed():
    law = EventTime(
        jnp.log(jnp.array([[0.2, 0.0], [0.0, 0.5], [0.3, 0.0]])),
        jnp.log(jnp.array([0.5, 0.5])),
    )
    prefix = jnp.array([[1, 0]], dtype=jnp.int32)
    future = prefix_condition(law, prefix)
    np.testing.assert_allclose(future.mean, [[0.0, 0.5], [0.0, 0.0]])
    full = jnp.array([[1, 0], [0, 1], [0, 0]], dtype=jnp.int32)
    np.testing.assert_allclose(
        slice_time(law, slice(0, 1)).log_prob(prefix)
        + future.log_prob(full[1:]),
        law.log_prob(full),
    )
```

- [ ] Add the count-pool moment/conditioning cases from Task 1, impossible repeated events, empty-prefix/full-prefix/empty-window cases, batched laws, sample axes, structural-zero gradients, and underflow-resistant delayed entry. Retain failing-before evidence for defects found while promoting the prototype.
- [ ] Implement log masses from log hazards and log survival; condition entry by shifting susceptibility logits. Share the numerical calculation between both distributions. Implement the native surgeries according to the contract above, including absorbing zero remaining counts.
- [ ] Move `numpyro-forecast>=0.4.0,<0.5` into required dependencies. Remove the redundant `forecast` extra rather than leave an empty compatibility shim; update README, example invocations/docstrings, CI/nox and installation guidance in the same change. Keep the `examples` extra for notebook/plot dependencies. Update the lockfile normally.
- [ ] Run `uv run pytest -q tests/test_survival.py tests/test_distributions.py`, followed by an actual fit/native forecast and in-sample draw using the new production distributions. Confirm both default precision and the approved large-count strategy.
- [ ] Commit only this independently usable probability/distribution layer and the installation migration.

## Task 3: Route all survival fitting through native observation sites

**Issue:** `rb47`. **Depends on:** Task 2.

**Files:** `src/ttenet/models.py`, `src/ttenet/network.py`; `tests/test_survival.py`, `tests/test_network.py`.

**Consumes:** shared kernel and distributions. **Produces:** existing standalone and network fit APIs backed by native observations, with unchanged parameter/result semantics.

- [ ] Preserve the existing analytic cases, notably event probability `0.1` and censor probability `0.7` for `h=0.5, pi=0.4`, and conditional-entry censor probability `0.625/0.7`. Exercise their native log joint and parameter gradients, not just the public likelihood helper.
- [ ] Adapt `StageObservations` into integer time-major trajectories and administrative exposure inputs. Preserve masks and row selection; do not condition generated data on the observed event's stopping mask.
- [ ] Keep `sample_stage_parameters` and custom `parameter_model(observations, shared)` unchanged in meaning. Use the following native registration inside the existing node scope, with full-horizon kernel inputs:

```python
h = Horizon.from_data(covariates, data)
kernel = survival_kernel(parameters, **exposure_inputs)
predict(h, lambda log_mass: EventTime(log_mass, kernel.log_tail), kernel.log_mass)
```

`covariates`, `data`, and `exposure_inputs` are the adapter's aligned calendar arrays, not a time-length-only dummy standing in for real feature data. Do not also add the old `numpyro.factor`, which would double-count observations.

- [ ] Implement exported `stage_log_likelihood` with the same shared law's per-unit probabilities. Preserve `StageFit.parameters`, losses, posterior sample names needed by consumers, and deterministic resolved parameter extraction.
- [ ] Run `test_return_observations_update_shared_sales_posterior` and `test_optimized_event_parameters_are_resolved_before_simulating`; fit standalone, joint, and modular examples. Check that observed returns still affect a genuinely shared sales latent.
- [ ] Verify upstream posterior predictive draws have the expected distribution rather than echoing stored observations, then commit.

## Task 4: Replace cohort forecasts and expectations with the shared kernel

**Issue:** `k3hb`. **Depends on:** Task 2; can run independently of Task 3 if file ownership is kept separate.

**Files:** `src/ttenet/forecast.py`; `tests/test_forecast.py`, `tests/test_integration.py`.

**Consumes:** native cohort law and exact integer strategy accepted in Task 1. **Produces:** the existing `EventForecast` / `ReturnForecast` contracts through shared-kernel sampling and expectations.

- [ ] Retain host validation and actual parent-date pool preparation. Each pool retains root-cohort identity, immediate parent date, and integer population; split heterogeneous covariates into distinct pools instead of applying an averaged hazard.
- [ ] For native forecast input, represent surviving historical pools as selected populations conditioned at the forecast boundary; represent future pools with no exposure before their source day. Do not materialize unobserved full historical event paths simply to satisfy a fixed array shape.
- [ ] Sample event-date allocations jointly from `CohortEventTime` through upstream `forecast`. Derive public pending/eligible totals from exactly the same allocations and deadlines. Pool counts and residual counts remain integers throughout.
- [ ] Compute finite expected downstream counts by streaming the contraction over actual parent-day blocks. For uninitiated historical units add conditional initiation-to-receipt composition; for known open returns use conditional receipt residual life. Do not condition twice or count already-received units again.
- [ ] Preserve eventual receipt assumptions explicitly: the tail baseline is reused, and unbounded susceptible stages eventually fire only under the documented continuing-exposure/reopening assumption. Do not replace eventual expectations with a finite-horizon sum.
- [ ] Remove host `_hazard_logit`, `_cure_logit`, and independent conditional-survival mathematics when their final callers migrate. Keep validation/pool utilities that still serve the public API.
- [ ] Run `uv run pytest -q tests/test_forecast.py tests/test_integration.py`, including exact int64 counts, overflow rejection, tail behavior and old-origin histories. Smoke the real retail CLI and rerun forecast performance gates before committing.

## Task 5: Integrate native cure stages across full forecast networks

**Issue:** `kz91`. **Depends on:** Tasks 3 and 4.

**Files:** `src/ttenet/network.py`, `src/ttenet/retail.py`, `src/ttenet/integration.py` only where integration requires it; `tests/test_network.py`, `tests/test_retail.py`, `tests/test_integration.py`.

**Consumes:** migrated fitting and cohort propagation. **Produces:** one complete production path for every EventNode, not only the retail two-stage special case.

- [ ] Retain host graph validation, calendar/covariate resolution and observed populations. Wire each node to native distribution execution; use its actual immediate-parent allocations, not an aggregate mean or the original sales date.
- [ ] Preserve model-generated and externally supplied `SalesForecast` paths, draw labels and posterior pairing. Reordered same-fit scenario draws must remain paired; unrelated externally supplied draws retain the documented row-pairing semantics.
- [ ] Exercise sale → initiation → receipt → inspection and sibling events; retain root-cohort labels and immediate-parent deadlines at every depth. Siblings are not competing risks.
- [ ] Run real joint and modular fitting plus default-sales, fixed-sales and uncertain-sales scenarios. Change only future weather/closures and verify historical evidence is unchanged and future predictions change accordingly.
- [ ] Run `uv run pytest -q tests/test_network.py tests/test_retail.py tests/test_integration.py`, the full native example, and `uv run --extra examples python examples/retail_returns.py --steps 150 --draws 40`. Require all conservation assertions and gate budgets to pass before committing.

## Task 6: Retire duplicate implementations and verify the public cutover

**Issue:** `rsb4`. **Depends on:** Task 5.

**Files:** obsolete internals in migrated source files; `examples/survival_convolution.py`, `examples/survival_convolution_model.py`, `examples/_survival_convolution_fixture.py`; README, blog, installation commands and evidence document.

**Consumes:** the working complete native path. **Produces:** one maintainable backend, accurate examples/documentation and measured end-to-end acceptance evidence.

- [ ] Make the focused example import production kernel/distributions. Delete `examples/survival_convolution_model.py` once no callers remain. Remove duplicate legacy likelihood formulas from the synthetic fixture; retain independent analytic oracles in tests rather than a second production implementation.
- [ ] Check every public survival entry point from `src/ttenet/__init__.py`, including low-level `fit_stage`, `stage_model`, `stage_log_likelihood`, `forecast_events`, `forecast_returns`, and all network/retail methods. No unmigrated execution path, backend selector, dead private helper, or obsolete alias remains.
- [ ] Update the blog's short convolution section from “prototype” to the real package implementation, showing actual public imports where helpful. Remove the temporary prototype caveat only after the notebook executes the migrated backend. Keep the explanation brief: event mass, composition, native time-axis conditioning, and exact per-draw population conservation.
- [ ] Run required verification, recording actual results rather than writing expected counts into the report:

```bash
uv sync --all-extras
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run complexipy src --plain
uv run python examples/survival_convolution.py
uv run python examples/retail_returns.py --steps 150 --draws 40
uv run marimo check examples/retail_returns_blog.py
uv run python examples/retail_returns_blog.py
uvx nox -s tests-3.12 tests-3.13 tests-3.14 lint complexity
```

- [ ] Render the actual notebook in a browser, inspect the new prose/math/code at desktop and narrow widths, and exercise demand/weather controls. Confirm all displayed forecasts come from the migrated package and no errors are hidden by script-mode shortcuts.
- [ ] Repeat the Task 1 benchmark matrix against the recorded old revision. Append measured before/after results and confirm all four adoption gates. Do not reset the complexity baseline just to conceal a regression.
- [ ] Update draft PR #6 with the real production scope, results, and any explicitly approved tradeoffs; commit and push. Close verified children with evidence, then the parent. Do not merge without authorization.

## Dependency order and review checkpoints

```text
Task 0: plan + honest blog explanation
  -> Task 1: adoption gate
    -> Task 2: common kernel + native distributions
      -> Task 3: fitting -----------+
      -> Task 4: cohort forecasts --+-> Task 5: complete network
                                       -> Task 6: clean cutover + final evidence
```

Only Tasks 3 and 4 are independent implementation slices after the shared law exists. A reviewer may reject either without invalidating the other's file ownership. Before each task, rehydrate its issue, record the intended approach, and record consequential divergences before editing. Keep scope and status truthful: completing this planning change does not close the migration parent.
