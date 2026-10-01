# Native Survival-Convolution Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace duplicated survival-model mathematics with one cure-capable, family-agnostic probability kernel; use native NumPyro Forecast observations and scalable count forecasting, explicitly documenting the native overhead subsequently approved by the user.

**Architecture:** Keep `CureProcess` and existing default-model contracts. The current `StageParameters` hazard model, alternative parametric families, and eventually learned nonparametric laws all feed the same cure/conditioning/cohort machinery. Use one native NumPyro Forecast model for observation, conditioning, posterior replay and count forecasting. NumPy remains an independent oracle and the existing-backend benchmark; it is not an allowed production allocation path.

**Tech Stack:** Python >=3.12, NumPy >=2.0, pandas >=2.2, JAX >=0.10.0,<0.11.3, NumPyro >=0.22.0, numpyro-forecast >=0.4.0,<0.5, pytest, marimo, Ruff, nox.

## Global constraints

- The user has authorized full implementation of this plan, followed by a draft PR containing the verified package cutover.
- Worktree: `/Users/kylejcaron/ttenet/.worktrees/native-survival-convolution`; branch: `prototype/native-survival-convolution`. Prototype/planning PR #6 is closed; create its replacement only after implementation and acceptance.
- Durable parent: `5jvf`. Task references and real prerequisites appear below. Leave implementation children open until their deliverables are verified.
- Task 1 gates the execution design. After reviewing the native performance evidence and the multinomial/calendar semantics, the user explicitly said “ok then proceed!” The accepted design is pure-JAX unified NumPyro execution, with no NumPy hybrid or compiled extension. Record the measured performance tradeoffs without claiming the original prospective target passed. Correctness and integer-support gates remain binding.
- Preserve existing public call signatures, result shapes, validations, custom priors, optimized `numpyro.param` values, and the default regularized random-walk age prior. Additive family-selection configuration is in scope. The prototype's five-parameter prior is a fixture, not the replacement package model.
- Preserve Python 3.12/3.13/3.14 coverage and existing dependency version ranges. Promote numpyro-forecast to a required dependency only when introducing the accepted native backend.
- No global JAX precision mutation at import, hidden host callbacks masquerading as native forecasting, unit expansion of count cohorts, or mandatory dense all-origin kernels.
- Existing random seeds must remain reproducible within the new implementation. Different sampling algorithms need distributional parity, not bit-identical draws against the old generator.
- Keep tests of public behavior. Remove tests tied only to obsolete implementation details rather than re-pinning those details.
- Existing unrelated work and PRs remain untouched. Do not merge the replacement implementation PR without authorization.
- Distribution-family interchangeability is a cutover requirement, not a documentation promise. Ship the current default plus one demonstrably different family through the same backend; reserve a compatible seam for nonparametric models without implementing a full nonparametric inference framework in this migration.

### Approved execution-design revision

The user's latest clarification requires one unified NumPyro model. Fitting, conditioning, posterior replay and counted forecasting must execute the shared native distributions. Earlier approval to investigate a shared-kernel/NumPy fallback remains useful benchmark and oracle evidence, but does not authorize that split execution as production architecture. There must be one implementation of hazard, cure, calendar masking, entry conditioning, tail semantics and count-allocation law, with no legacy/new backend selector or native-looking host callback.

Task 1 measured fitting, native count sampling and the rejected fallback. The user subsequently authorized implementation after discussing the measured overhead and the 365-day/storm use cases. This is **GO for implementation with recorded tradeoffs**, not a claim of performance parity or completed release acceptance. Remeasure the fully native retail workflow and completed package; do not promote the unvalidated experimental RBG sampler.

---

## 1. What is established, and what is not

At plan creation, production paths were split (the table is the migration baseline, not the final implementation):

| Path | Current implementation | Required destination |
|---|---|---|
| Stage likelihood and SVI | `models.stage_log_likelihood`, `stage_model`, `fit_stage` | Shared event-time law; native observation sites |
| Network likelihood | `network._event_model` adds a separate `numpyro.factor` | The same native law under each node's scope |
| Survival mathematics | `survival.py` JAX primitives; repeated host formulas in `forecast.py` | One log-space mathematical implementation |
| Cohort simulation | `forecast.forecast_events`, NumPy binomial depletion of actual parent-date pools | Native `CohortEventTime` allocation from the shared probability kernel |
| Historical eventual expectations | `forecast.expected_return_receipts` | Shared probabilities and explicit tail assumptions |
| Network / retail forecast | `FittedNetwork._forecast`, `FittedRetailReturnModel.forecast` | Shared-law cure stages preserving existing graph and result semantics |
| Demonstration | `examples/survival_convolution_model.py` and its fixture | Examples importing the package implementation |

On 2026-09-30, this command passed on the current draft branch:

```bash
/usr/bin/time -l uv run --extra forecast python examples/survival_convolution.py
```

Observed: numpyro-forecast 0.4.0; maximum single-stage likelihood error `3.8147e-6`, gradient error `9.5367e-7`; joint likelihood error `0`, gradient error `2.3842e-7`; 8,192 chained draws with no lineage violations and maximum receipt-probability discrepancy `0.0057565`. The full process took 20.22 seconds and reported 2,339,504,128 bytes maximum RSS on this Darwin arm64 workstation. These totals include compilation, fitting, and several checks: **they are not a forecast benchmark or evidence of a speedup**.

At plan creation, still unproved: native count-cohort sampling at existing integer limits, realistic memory/runtime, all package entry/mask contracts, stochastic future sales integration, shared priors across a full graph, and stable extreme/long-history behavior. Final production evidence is appended in the companion evidence document. Native API reuse is an architectural benefit; improved statistical accuracy is not expected merely from changing representation.

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

Index stages as `K_0` (initiation), `K_1` (receipt), and so on. For fixed parameters and a homogeneous cohort, composition gives

\[
(K_0 K_1)_{s,t,c}=\sum_u K_{0,s,u,c}K_{1,u,t,c}.
\]

This is a calendar-dependent kernel composition. It becomes ordinary lag convolution only with stationary hazards/exposure. Historical open stages require their separately conditioned residual-life contribution. Convolve within each posterior draw and cohort, then aggregate: multiplying posterior-averaged kernels discards parameter dependence.

**Expectations are not trajectories.** For paths, allocate a pool of `n` units jointly across event dates plus the no-event category, then pass the actual parent-date allocations to its child. An independent Poisson draw per day is not an acceptable substitute.

### Implemented internal contract

The accepted compact representation keeps exposure-masked timing and conditioned susceptibility. Its derived properties expose full event masses and the no-event tail without requiring those arrays in every fitting evaluation:

```python
class SurvivalKernel(NamedTuple):
    log_hazard: Array  # [..., time, cohort], exposure-masked
    log_survival_step: Array  # [..., time, cohort], exposure-masked
    susceptibility_logits: Array  # [..., cohort], conditioned at entry
    # Derived properties: log_mass [..., time, cohort], log_tail [..., cohort].


class TimingLaw(NamedTuple):
    log_hazard: Array  # susceptible event probability, [..., time, cohort]
    log_survival_step: Array  # susceptible no-event probability on that day


def survival_kernel(
    timing: TimingLaw,
    susceptibility_logits: Array,
    *,
    allowed: Array,
    exposure: Array,
    pre_entry: Array | None = None,
) -> SurvivalKernel:
    """Marginalize cure and condition entry, independently of the timing family."""
```

`exposure` represents administrative entry/exit/deadline bounds, not the realized event's stopping time. Observation-specific event masks must never leak into a generative forecast law. `pre_entry` conditions on known prior survival without restarting the age clock or counting that evidence twice.

- `EventTime(log_mass, log_tail)`: unit trajectories; `unit_log_prob(data)` retains the cohort axis; `log_prob(data)` sums cohorts.
- `CohortEventTime(log_mass, log_tail, total_count)`: the corresponding integer allocation law for homogeneous, actual-origin pools. Unit count one is the same law, not different cure mathematics.
- Both distributions support `sample`, `mean`, `log_prob`, and native `slice_time` / `prefix_condition`. Slicing marginalizes outside-window event mass into the window's no-event category. Prefix conditioning removes observed events from remaining population and renormalizes future categories, with zero population absorbing.
- The native compact call is `predict(h, lambda log_hazard: EventTime(kernel=kernel._replace(log_hazard=log_hazard)), kernel.log_hazard)`, or its count-cohort equivalent. It returns `None`; obtain generated paths from NumPyro sites, not a fabricated return value.
- Validate infeasible observations at host boundaries; impossible paths have `-inf` likelihood. Never turn an impossible history into certain cure as a fallback.
- Keep the public `stage_log_likelihood` entry point as a real evaluator of the shared law, not as a second formula. Existing public convenience functions are not deprecated aliases to remove.

### Plug-and-play time-to-event families

The backend consumes a `TimingLaw`, not `StageParameters.age_logits`. Any discrete event-time law can be expressed as conditional event/no-event probabilities; using this representation does **not** require a logistic hazard model or a finite-dimensional parametric family.

Public extension: `CureProcess(event_time_model=callable)`. Like the existing prior callback, the factory receives timing inputs and shared latents, samples named NumPyro parameters, and returns `(TimingLaw, susceptibility_logits)`. Its inputs include only `ages`, `features`, and `cure_features`; administrative `allowed`, `exposure`, and `pre_entry` masks and realized outcome labels remain exclusively in the shared core. Calling the model at a longer horizon re-evaluates the factory under the same posterior parameter sites, rather than extending a frozen training-grid array. Reject simultaneous `event_time_model` and legacy `parameter_model` configuration as ambiguous; the legacy callback remains fully supported for the default family.

The callable also declares static `tail_behavior`: `{"kind": "proper"}` for a family whose event probability tends to one under the documented continuing-exposure assumption; `{"kind": "finite", "last_age": a}` for fully specified finite support; or `{"kind": "unknown"}` for an unresolved residual category. Keep this metadata outside traced array leaves. A declared continuation must be evaluable by replaying the same factory at future ages; the default constant final-hazard continuation and Weibull both use the proper case. For finite support, integrate the calendar-masked kernel through `last_age`: a terminal atom falling on a closure does not justify assuming all remaining units eventually fire.

The default adapter builds daily log hazards from existing `StageParameters` and preserves all prior sample-site names. Other factories carry their own named parameters: do not hide Weibull scale/shape or learned probability masses inside `age_logits`. Generic fitted execution stores/replays the native posterior and factory rather than insisting on `StageParameters` for every family. Keep the existing default `StageFit.parameters` result unchanged; expose custom-family named posterior parameters as a documented mapping instead of fabricating default-model parameters. Exported low-level `StageParameters` convenience functions remain default-family entry points into the common backend.

Family-specific code supplies susceptible timing and susceptibility logits. The shared backend alone applies cure marginalization, administrative entry/deadlines, structural-zero closures, censoring, native time-axis surgery, cohort allocation, and convolution. This boundary prevents every new family from reimplementing survival bookkeeping.

**Second-family proof:** implement a discretized Weibull timing factory as an example plugin, not a second backend or a mandatory replacement prior. With positive scale `lambda` and shape `k`, use continuous `log S(a)=-(a/lambda)^k` for `a>=0`, so daily `log_survival_step(a)=log S(a+1)-log S(a)`. Derive `log_hazard` with a stable log-complement. The `[a,a+1)` bin gives a nonzero same-day trial. Use shape 2 as well as shape 1 so the demonstration is not merely a renamed constant hazard.

**Calendar semantics:** the default elapsed-age clock continues through closed dates, while closures replace that day's probabilities with `log_hazard=-inf`, `log_survival_step=0`. Apply that rule once in the common kernel. Do not quietly switch a parametric distribution to an exposure-time clock; that would be a separate, explicit modeling choice.

**Nonparametric-compatible contract:** prove a discrete probability-mass adapter in tests. Given learned masses `p[a]` and a residual tail `q`, the at-risk mass at age `a` is `R[a]=sum(p[a:])+q`, with hazard `p[a]/R[a]` and survival step `R[a+1]/R[a]`. Compute these ratios in log space. This admits histogram/Dirichlet masses, spline-based survival, and other learned monotone survival laws without changing native distributions, the graph, or count allocation. A full nonparametric prior/estimator is a later modeling feature, not required now.

**Tail contract:** each family declares proper finite support, a specified continuation, or unknown residual tail. The default last-bin continuation is specific to the default model; Weibull uses its own survival continuation. Residual mass beyond a learned grid is not automatically cure or zero hazard forever. Finite-horizon predictions may use a known residual category, but eventual-receipt expectations require an identified continuation/proper-tail assumption; reject that request clearly when the selected family cannot supply it. Do not silently return zero or reuse the default model's tail. Zero residual risk after a certain event also needs explicit support handling: use exclusive survival accumulation, not `cumsum(log_stay) - log_stay`, which produces `-inf - -inf` at a terminal atom.

## 3. Adoption gates

Task 1 requires gates A–D on the candidate/prototype, using adapters to exercise existing public-contract cases without switching production callsites. Task 6 repeats A–D on the migrated package and additionally requires final release gate E. Record measured results, environment, reference revision, seeds, dimensions, and pass/fail in `docs/superpowers/plans/2026-09-30-native-survival-convolution-evidence.md` during Task 1; append final results after Task 6. Do not create a blank evidence file now.

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
- Observed likelihood comes from upstream `predict`. Native forecasts must obtain generated suffixes from upstream rather than hiding host sampling behind a native-looking site. NumPy allocation is permitted only as an independent oracle and benchmark.
- Demonstrate fit-time and forecast-time shape changes, integer observed data, zero future arrivals, sparse versus dense parent dates, arbitrary descendant nodes, and no hidden-outcome leakage.
- Avoid a dynamically growing unit axis inside a traced model. Host cohort preparation may remain, but native sampling must consume fixed-shape pool data; prove how pools and lineage map back to public results.
- Run the default and discretized Weibull families through native fitting and posterior replay at new horizons, then conditioning and joint native cohort forecasts. Add a finite-mass adapter contract test with an explicit terminal atom and an unknown-tail case; prove the backend has no required age-logit representation.

### D. Measured engineering benefit

- Benchmark the old committed implementation and candidate in separate processes on the same machine, using identical parameters and inputs. Do not compare the prototype's whole 20-second script against one forecast call.
- Measure cold compile+first call, median of five warmed calls after synchronization, and peak RSS. Include the native driver, conditioning, packing, and output aggregation in end-to-end forecast timing.
- Baseline matrix: `(cohorts, history_days, horizon, draws)` = `(100,60,28,32)`, `(1000,180,90,128)`, `(5000,365,365,32)`. Include sparse and multi-day parent arrivals, homogeneous quantities 1 and 10,000,000, heterogeneous closures, and both unit-history and future-cohort populations.
- Keep warmed forecast and fitting/gradient regression ratios <=1.25x as prospective optimization targets, not falsely passed gates. The user authorized proceeding with the unified native design after the overhead was explained. Measure peak RSS and mitigate scalable regressions; report absolute cold costs, fixed runtime footprint and scalable memory independently. Shared-kernel/NumPy measurements remain diagnostics only. The draft PR must report actual package/native retail timings and any remaining regressions.
- The candidate must demonstrate one reusable mathematical implementation and native observation sites, plus identify the exact legacy callers/formulas it can replace. Production deletion is not a prerequisite for Task 1; it is required by final release gate E.
- If dense kernels exceed the budget, stream or batch actual-origin pools and mean contractions. Do not assume an FFT is valid for weather-, calendar-, or unit-dependent hazards.

### E. Final release gate (Task 6 only)

- Every public survival execution path uses the accepted shared native probability distributions; correctness/native gates A–C pass again against the actual package, and gate D supplies measured final performance evidence under the approved tradeoff.
- Duplicated likelihood/hazard/conditioning implementations, the host event simulator and the example-only backend are removed after their final callers migrate. No legacy/new backend selector or host simulator hidden behind a native-looking wrapper remains.
- Full public-contract checks, all supported Python versions, real examples, the rendered blog, and the measured final benchmark report pass. Task 1's GO authorizes implementation, not a claim that this release gate already passed.

## 4. File ownership and scope

| File | Responsibility after migration |
|---|---|
| `src/ttenet/survival.py` | `StageParameters`, one stable log-space kernel and public probability primitives |
| `src/ttenet/distributions.py` (new) | Unit/cohort event-time laws and native surgery registrations; no data-frame work |
| `src/ttenet/event_times.py` (new) | Family input/factory contract and default hazard adapter; no duplicate cure/conditioning engine |
| `examples/event_time_families.py` (new) | Runnable discretized Weibull plugin and comparison through the public configuration |
| `tests/test_event_times.py` (new) | Family adapters, terminal atoms, continuation semantics and plug-in behavior |
| `src/ttenet/models.py` | Observation adapters, existing priors, standalone native stage fitting |
| `src/ttenet/forecast.py` | Host validation/pool preparation, public result assembly, shared-kernel expectations; no second cure formula |
| `src/ttenet/network.py` | Scoped graph composition, shared priors, posterior/draw alignment and shared-law cure forecasting |
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

**Issue:** `ersy`. **Depends on:** nothing. **Completed planning phase.**

**Files:** this plan; `examples/retail_returns_blog.py`, immediately after the censoring callout in Part 2.

- [x] Commit this plan and its adoption gates; attach one durable issue per task with the dependencies below.
- [x] Add one short hidden-code markdown cell titled “From clocks to a survival convolution.” Explain probability mass spreading through calendar time, then show the kernel-composition identity and a native `Horizon`/`predict` sketch.
- [x] Mark the sketch as the native prototype, not the currently executed `CureProcess` backend. State that the kernel builder and event-time distribution are TTENet components, not built-in NumPyro Forecast models.
- [x] Keep the code block short, mathematical expectations distinct from sampled paths, and the same typography as adjacent prose. No new widget, styling system, dependency, or chart is needed.
- [x] Run the notebook with its real defaults, inspect the new section in a browser at desktop and narrow width, and run `marimo check` plus Ruff. Record only observed results.

## Task 1: Prove native cohort sampling and establish adoption evidence

**Issue:** `jtr1`. **Depends on:** Task 0. **No production cutover in this task.**

**Files:** `examples/survival_convolution.py`, `examples/survival_convolution_model.py`; create the evidence document specified above after measurements. Benchmark/reproduction scripts may remain throwaway unless they provide a reproducible consumer-relevant check worth retaining.

**Consumes:** current production `StageParameters`, `forecast_events`, `stage_log_likelihood`; native 0.4.0 API. **Produces:** a pass/fail report and demonstrated count-cohort allocation/shape/precision strategy satisfying candidate gates A–D under the approved execution-design revision.

- [x] Re-run the current example and existing focused contracts before choosing a sampler:

```bash
uv run --extra forecast python examples/survival_convolution.py
uv run --extra forecast pytest -q tests/test_survival.py tests/test_forecast.py tests/test_integration.py tests/test_network.py tests/test_retail.py
```

- [x] Exercise the candidate native count distribution, not just unit trajectories. For one homogeneous pool with `n=7` and category probabilities `[0.2,0.3,0.5]`, compare event means `[1.4,2.1]`, variances `[1.12,1.47]`, and covariance `-0.42`; require integer nonnegative samples and total observed events <=7. After observing two events in category one, the remaining two categories have probabilities `[0.375,0.625]` for the five remaining units.
- [x] Run the existing `test_int64_cohort_counts_do_not_round_through_float64` contract against the candidate end-to-end path. Exact near-certain completion and conservation pass; float-proposal support limitations above `2**53` are recorded rather than overstated.
- [x] Prove native tracing with real count draws, delayed entry, changing horizon lengths and actual-origin pools. Record precision/static-shape compatibility and the unresolved native performance regression.
- [x] Validate the family boundary using the existing hazard model, a discretized Weibull law with shape 2, and a discrete-mass adapter with a terminal atom. Require native posterior replay at a longer horizon without assuming `age_logits`; retain evidence that cure, closures and conditioning behave identically across adapters.
- [x] Run gates A–D and write actual forecast, fitting, gradient, retail-workflow timings, memory and correctness results. Compare the optimized native sampler with the shared-kernel/NumPy diagnostic to isolate the sampler regression. Preserve the old baseline revision so later tasks can rerun it after obsolete code is removed.
- [x] Record **GO for implementation with documented tradeoffs** after the user's explicit “ok then proceed!” following native-performance and model-semantics explanations. Do not claim the original prospective warm target passed. NumPy hybrid and compiled extensions are excluded; retain pure JAX and the validated precision strategy. Production retirement/release gate E remains deferred to Task 6.

## Task 2: Implement the shared survival law and native distribution

**Issue:** `jggf`. **Depends on:** successful Task 1.

**Files:** `src/ttenet/survival.py`; new `src/ttenet/distributions.py`, `src/ttenet/event_times.py`, `tests/test_distributions.py`, `tests/test_event_times.py`; `tests/test_survival.py`; dependency/configuration files and existing commands referring to the removed extra.

**Consumes:** the validated kernel/count/family contracts and precision strategy. **Produces:** `TimingLaw`, `SurvivalKernel`, `survival_kernel`, the default timing adapter, `EventTime`, `CohortEventTime`, native surgery registrations, and a default installation capable of executing them. Task 1's exploratory Weibull probe is promoted to a runnable public plugin in Task 3.

- [x] Establish regression cases before implementation. This tiny law has an independent exact oracle:

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
        slice_time(law, slice(0, 1)).log_prob(prefix) + future.log_prob(full[1:]),
        law.log_prob(full),
    )
```

- [x] Add the count-pool moment/conditioning cases from Task 1, impossible repeated events, empty-prefix/full-prefix/empty-window cases, batched laws, sample axes, structural-zero gradients, and underflow-resistant delayed entry. Retain failing-before evidence for defects found while promoting the prototype.
- [x] Implement log masses from log hazards and log survival; condition entry by shifting susceptibility logits. Share the numerical calculation between both distributions. Implement the native surgeries according to the contract above, including absorbing zero remaining counts.
- [x] Move `numpyro-forecast>=0.4.0,<0.5` into required dependencies. Remove the redundant `forecast` extra rather than leave an empty compatibility shim; update README, example invocations/docstrings, CI/nox and installation guidance in the same change. Keep the `examples` extra for notebook/plot dependencies. Update the lockfile normally.
- [x] Run `uv run pytest -q tests/test_survival.py tests/test_distributions.py tests/test_event_times.py`, followed by an actual fit/native forecast and in-sample draw using the new production distributions. Confirm both default precision and the approved large-count strategy.
- [x] Commit only this independently usable probability/distribution layer and the installation migration.

## Task 3: Route all survival fitting through native observation sites

**Issue:** `rb47`. **Depends on:** Task 2.

**Files:** `src/ttenet/models.py`, `src/ttenet/network.py`, `src/ttenet/processes.py`; new `examples/event_time_families.py`; `tests/test_survival.py`, `tests/test_network.py`, `tests/test_event_times.py`.

**Consumes:** shared family/kernel contracts and distributions. **Produces:** existing standalone and network fit APIs backed by native observations, unchanged default parameter/result semantics, and the additive event-time factory configuration.

- [x] Preserve the existing analytic cases, notably event probability `0.1` and censor probability `0.7` for `h=0.5, pi=0.4`, and conditional-entry censor probability `0.625/0.7`. Exercise their native log joint and parameter gradients, not just the public likelihood helper.
- [x] Adapt `StageObservations` into integer time-major trajectories and administrative exposure inputs. Preserve masks and row selection; do not condition generated data on the observed event's stopping mask.
- [x] Keep `sample_stage_parameters` and custom `parameter_model(observations, shared)` unchanged in meaning for the default family. Convert their resolved parameters with the default timing adapter; for another family replay its named posterior parameters through its factory. Both routes produce `timing` and `susceptibility_logits` for the same native registration:

```python
h = Horizon.from_data(covariates, data)
kernel = survival_kernel(timing, susceptibility_logits, **exposure_inputs)
predict(
    h,
    lambda log_hazard: EventTime(kernel=kernel._replace(log_hazard=log_hazard)),
    kernel.log_hazard,
)
```

`covariates` and `data` are aligned calendar arrays, not a time-length-only dummy standing in for real feature data. `exposure_inputs` contains only the common kernel's `allowed`, `exposure` and optional `pre_entry` masks; the family adapter has already consumed ages and regressors. Do not also add the old `numpyro.factor`, which would double-count observations.

- [x] Implement exported `stage_log_likelihood` with the same shared law's per-unit probabilities. Preserve `StageFit.parameters`, losses, posterior sample names needed by consumers, and deterministic resolved parameter extraction.
- [x] Add `CureProcess(event_time_model=...)` and prove changing only that factory selects Weibull without modifying fitting, native distributions or network propagation. Preserve the default `StageParameters` result; document custom-family named parameter mappings. Check custom shared priors and reject the ambiguous combination of both factory and legacy prior callback.
- [x] Run `test_return_observations_update_shared_sales_posterior` and `test_optimized_event_parameters_are_resolved_before_simulating`; fit standalone, joint, and modular examples. Check that observed returns still affect a genuinely shared sales latent.
- [x] Verify upstream posterior predictive draws have the expected distribution rather than echoing stored observations, then commit.

## Task 4: Replace cohort forecasts and expectations with the shared kernel

**Issue:** `k3hb`. **Depends on:** Task 2; can run independently of Task 3 if file ownership is kept separate.

**Files:** `src/ttenet/forecast.py`; `tests/test_forecast.py`, `tests/test_integration.py`.

**Consumes:** the common probability law and native integer strategy accepted in Task 1. **Produces:** existing `EventForecast` / `ReturnForecast` contracts through native shared-kernel sampling and expectations.

- [x] Retain host validation and actual parent-date pool preparation. Each pool retains root-cohort identity, immediate parent date, and integer population; split heterogeneous covariates into distinct pools instead of applying an averaged hazard.
- [x] Represent surviving historical pools as selected populations conditioned at the forecast boundary; future pools have no exposure before their source day. Do not materialize unobserved full historical event paths simply to satisfy a fixed array shape.
- [x] Allocate event dates jointly through native `CohortEventTime` and upstream NumPyro Forecast execution. Host code may validate and pack actual-origin pools, then map native integer allocations back to public results; it must not sample event dates. Derive pending/eligible totals from the same allocations and deadlines, preserving integer pools and residuals.
- [x] Compute finite expected downstream counts by streaming the contraction over actual parent-day blocks. For uninitiated historical units add conditional initiation-to-receipt composition; for known open returns use conditional receipt residual life. Do not condition twice or count already-received units again.
- [x] Preserve eventual receipt assumptions explicitly: the tail baseline is reused, and unbounded susceptible stages eventually fire only under the documented continuing-exposure/reopening assumption. Do not replace eventual expectations with a finite-horizon sum.
- [x] Remove host `_hazard_logit`, `_cure_logit`, and independent conditional-survival mathematics when their final callers migrate. Keep validation/pool utilities that still serve the public API.
- [x] Run `uv run pytest -q tests/test_forecast.py tests/test_integration.py`, including exact int64 counts, overflow rejection, tail behavior and old-origin histories. Smoke the real retail CLI and rerun forecast performance gates before committing.

## Task 5: Integrate shared-law cure stages across full forecast networks

**Issue:** `kz91`. **Depends on:** Tasks 3 and 4.

**Files:** `src/ttenet/network.py`, `src/ttenet/retail.py`, `src/ttenet/integration.py` only where integration requires it; `tests/test_network.py`, `tests/test_retail.py`, `tests/test_integration.py`.

**Consumes:** migrated fitting and cohort propagation. **Produces:** one complete production path for every EventNode, not only the retail two-stage special case.

- [x] Retain host graph validation, calendar/covariate resolution and observed populations. Wire each node to the common native probability distributions; use its actual immediate-parent allocations, not an aggregate mean or original sales date.
- [x] Preserve model-generated and externally supplied `SalesForecast` paths, draw labels and posterior pairing. Reordered same-fit scenario draws must remain paired; unrelated externally supplied draws retain the documented row-pairing semantics.
- [x] Exercise sale → initiation → receipt → inspection and sibling events; retain root-cohort labels and immediate-parent deadlines at every depth. Siblings are not competing risks.
- [x] Run real joint and modular fitting plus default-sales, fixed-sales and uncertain-sales scenarios. Change only future weather/closures and verify historical evidence is unchanged and future predictions change accordingly.
- [x] Run a mixed-family graph: default initiation, Weibull receipt, and default inspection. Confirm shape-2 Weibull likelihood/gradients against its analytic survival, then forecast the same cohort inputs and closures without a family-specific graph branch. Include a native in-sample prediction and a horizon longer than training.
- [x] Run `uv run pytest -q tests/test_network.py tests/test_retail.py tests/test_integration.py`, the full native example, and `uv run --extra examples python examples/retail_returns.py --steps 150 --draws 40`. Require conservation/native correctness assertions and record real workflow timings before committing.

## Task 6: Retire duplicate implementations and verify the public cutover

**Issue:** `rsb4`. **Depends on:** Task 5.

**Files:** obsolete internals in migrated source files; `examples/survival_convolution.py`, `examples/survival_convolution_model.py`, `examples/_survival_convolution_fixture.py`; README, blog, installation commands and evidence document.

**Consumes:** the complete accepted shared-law execution path. **Produces:** one maintainable probability implementation, accurate examples/documentation and measured end-to-end acceptance evidence.

- [x] Make the focused example import production kernel/distributions. Delete `examples/survival_convolution_model.py` once no callers remain. Remove duplicate legacy likelihood formulas from the synthetic fixture; retain independent analytic oracles in tests rather than a second production implementation.
- [x] Check every public survival entry point from `src/ttenet/__init__.py`, including low-level `fit_stage`, `stage_model`, `stage_log_likelihood`, `forecast_events`, `forecast_returns`, and all network/retail methods. No unmigrated execution path, backend selector, dead private helper, or obsolete alias remains.
- [x] Update the blog's short convolution section from “prototype” to the real package implementation, showing actual public imports where helpful. Remove the temporary prototype caveat only after the notebook executes the migrated backend. Keep the explanation brief: event mass, composition, native time-axis conditioning, and exact per-draw population conservation.
- [x] Document the factory contract and runnable Weibull plugin. Explain that nonparametric-compatible means an interchangeable probability-law interface, not an already-shipped nonparametric estimator. Include `tests/test_event_times.py` in the full suite and run `uv run python examples/event_time_families.py`; retain the explicit unknown-tail failure for eventual expectations.
- [x] Run required verification, recording actual results rather than writing expected counts into the report:

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

- [x] Render the actual notebook in a browser, inspect the new prose/math/code at desktop and narrow widths, and exercise demand/weather controls. Confirm all displayed forecasts come from the migrated package and no errors are hidden by script-mode shortcuts.
- [x] Repeat the Task 1 benchmark matrix against the recorded old revision. Append measured before/after results and confirm gates A–E on the completed package. Do not reset the complexity baseline just to conceal a regression.
- [x] Create a new draft PR with the real production scope, results, and any explicitly approved tradeoffs; commit and push. Close verified children with evidence, then the parent. Do not merge without authorization.

## Dependency order and review checkpoints

```text
Task 0: plan + honest blog explanation
  -> Task 1: adoption gate
    -> Task 2: common kernel + native distributions
      -> Task 3: fitting -----------+
      -> Task 4: cohort forecasts --+-> Task 5: complete network
                                       -> Task 6: clean cutover + final evidence
```

Only Tasks 3 and 4 are independent implementation slices after the shared law exists. A reviewer may reject either without invalidating the other's file ownership. Before each task, rehydrate its issue, record the intended approach, and record consequential divergences before editing. Close the migration parent only after the complete implementation is verified and its new draft PR exists.
