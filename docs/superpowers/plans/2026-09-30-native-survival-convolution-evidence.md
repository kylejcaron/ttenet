# Native Survival-Convolution Adoption Evidence

## Current decision

**GO for unified native implementation with recorded tradeoffs.** After the measured overhead and multinomial/365-day/storm semantics were explained, the user explicitly said “ok then proceed!” Proceed with pure JAX and native NumPyro distributions; no NumPy hybrid, compiled extension or experimental RBG promotion. The validated native tree/unit-count partition is about 2.5× slower on the large-32 fixture, so the original prospective performance target is not claimed passed. Final package/native retail performance and full correctness remain release evidence requirements.

The original relative cold-start budget is superseded: a sub-second startup cost is acceptable for substantial repeated workloads. Warm execution, likelihood/gradient correctness, integer accounting, scalable memory and genuine native execution still matter. Production code has not been switched.

## Environment and reference

Measured 2026-09-30 on the same arm64 host, macOS 15.7.4, Python 3.13.5. Installed packages: NumPy 2.5.3, JAX/jaxlib 0.11.0, NumPyro 0.22.0, NumPyro Forecast 0.4.0.

Reference revision: `8a85dfd661740e9df6c1a61b4e28a3262ab3afea`. Its production implementation remains unchanged during this gate.

## Existing behavioral and native-integration baselines

- `uv run --all-extras pytest -q tests/test_survival.py tests/test_forecast.py tests/test_integration.py tests/test_network.py tests/test_retail.py`: **87 passed**, 15.16 s.
- `.venv/bin/python examples/survival_convolution.py --steps 100 --draws 120`: passed in 20.90 s. Native `draw_posterior`, `forecast`, and `predict_in_sample` execute the existing unit-event distribution.
  - Maximum single-stage log-likelihood error: `3.814697265625e-06`; gradient error: `9.5367431640625e-07`.
  - Joint log-likelihood error: `0`; gradient error: `2.384185791015625e-07`.
  - 8,192 chained draws: zero lineage violations; maximum convolution probability error `0.005756467580795288`.
  - Impossible paths rejected; closed-day events zero; no hidden-outcome leakage.

These are execution and numerical checks, not posterior-calibration claims or proof of count-cohort support.

## Matched end-to-end forecast measurements

Each backend ran in a separate fresh process. Cold timing includes the first call and JAX initialization/compilation, but excludes module imports and shared input-fixture creation. Warm timing is the median of five subsequent calls, including host packing, native input conversion, prefix conditioning, driver execution, output materialization/synchronization, and public-result aggregation. Peak RSS is total process high-water memory, reported in bytes by macOS `resource.getrusage`; it includes imports and runtime overhead.

Fixture seed `7341`, forecast seed `519`; calendar starts `2024-01-01`. Half the root cohorts are historical units with uniformly varied origins; one fifth of those have already completed the event. The remaining cohorts receive one future arrival on varied dates. Weekday closures plus every-thirteenth-day closures for every seventh cohort give heterogeneous exposure. The model has 16 age bins, two hazard features, and one susceptibility feature. Parameter values and all inputs match across backends; PRNG algorithms intentionally differ.

The unit candidate uses the existing `EventTime` distribution and asserts every packed count is one. It computes the full log-space kernel, delegates observed-prefix conditioning and suffix draws to upstream `predict`/`forecast`, and aggregates actual-origin pools into the same `EventForecast` outputs. It is a unit-pool specialization, not evidence for arbitrary counts. Native numerical work is inside a local `jax.enable_x64()` context; no import-time precision mutation is required.

| Case `(cohorts, history, horizon, draws)` | Quantity/layout | Backend | Cold seconds | Warm median seconds | Peak RSS bytes |
| --- | --- | --- | ---: | ---: | ---: |
| `(100,60,28,32)` | 1 / sparse | Existing NumPy backend | 0.007670458 | 0.006551417 | 183189504 |
| `(100,60,28,32)` | 1 / sparse | Native unit candidate | 0.441980916 | 0.003509500 | 366788608 |

For this case: warmed time is **0.536×** reference (passes ≤1.25×); cold time **57.6×** (fails ≤2×); peak RSS **2.002×** (fails ≤1.25×). A sub-second absolute cold cost can still violate the approved relative budget. The fixed native-runtime footprint and scalable workload memory must not be conflated.

This is an early failing gate, not the complete benchmark matrix. Long-history, family-plugin, large-count, and all-workload acceptance remain unproven; downstream migration tasks remain blocked until adoption criteria can be met or an explicit tradeoff is approved.

## Additional measured workloads and memory mitigation

The same fixture generator and five-warm-call protocol produced the following results. Quantities describe future pools; historical observations still represent individual units.

| Case | Quantity/layout | Backend and batching | Cold seconds | Warm median seconds | Peak RSS bytes |
| --- | --- | --- | ---: | ---: | ---: |
| `(1000,180,90,128)` | 1 / sparse | Existing | 0.642667959 | 0.640673500 | 694075392 |
| `(1000,180,90,128)` | 1 / sparse | Native unit, unbatched | 1.135702833 | 0.686080333 | 1963081728 |
| `(1000,180,90,128)` | 1 / sparse | Native unit, upstream batch 4 | 0.791797417 | 0.387011792 | 826261504 |
| `(5000,365,365,32)` | 1 / sparse | Existing | 3.120588250 | 2.538880458 | 1705050112 |
| `(5000,365,365,32)` | 1 / sparse | Native unit, upstream batch 1 | 1.777052500 | 1.433260917 | 2881519616 |
| `(5000,365,365,32)` | 1 / sparse | Native unit, streamed batch 1 | 1.523908000 | 1.310747959 | 1320583168 |
| `(100,60,28,32)` | 10000000 / sparse | Existing | 0.007609667 | 0.006747375 | 182976512 |
| `(100,60,28,32)` | 10000000 / sparse | Native cohort, JAX binomial | 0.505503792 | 0.029781500 | 414711808 |
| `(1000,180,90,128)` | 10000000 / sparse | Existing | 0.642297208 | 0.628899209 | 677724160 |
| `(1000,180,90,128)` | 10000000 / sparse | Native cohort, JAX binomial, streamed batch 16 | 3.265171917 | 2.731392333 | 743981056 |

Streaming means each posterior chunk is passed to genuine upstream `forecast`, then its actual-origin pool counts are immediately aggregated into the public output. It does not introduce a separate event simulator. Upstream batching alone still retains the complete pool-output array before public aggregation; streaming avoids that additional full-size array.

The medium batched unit case and large streamed unit case meet the original cold/warm/RSS budgets. The counted medium case meets the memory budget but takes **4.343×** as long warmed. This is a steady-state sampler regression, not just compilation.

An alternate NumPyro binomial sampler was also measured and rejected: small counted case warm `0.025969875` s, but medium streamed batch 4 warm `7.538782625` s. The candidate retains JAX's sampler.

These are stage-level benchmarks, not a measured slowdown of the complete retail notebook. Multi-day-arrival cases, the largest counted case, and fitting/gradient performance have not been measured here; the existing failures are sufficient for NO-GO under the original budgets.

## Count-cohort feasibility checks

`.venv/bin/python examples/_cohort_candidate_probe.py` executed the counted native law, including:

- A 200,000-draw check of the `n=7` multinomial law and its negative cross-date covariance; its 36 feasible probability masses sum to `1.0` at default precision after the final numerical repair.
- Conditioning on two observed events leaves five units with probabilities approximately `[0.375, 0.625]`; the checked chain-rule log-probability difference is `2.384185791015625e-07`.
- Under locally scoped x64, the near-certain event count is exactly `9007199254740993` (`2**53 + 1`), including through upstream forecasting. Non-degenerate allocations preserve integer residuals and show both odd and even category counts.
- Native prefix absorption, zero populations, structural closures, heterogeneous source dates, longer forecast horizons, in-sample prediction, batching, and structural-zero gradients.

Counts above int32 require the locally scoped x64 boundary. The direct candidate constructor refuses unsupported host counts instead of permitting JAX's silent truncation. A public package cutover must supply that boundary; callers must not lose the existing int64 forecast contract.

Sampling checks alone did not establish extreme-count likelihood accuracy. Independent review exposed and then verified repairs for:

1. Certain allocation at `n=int64.max` previously yielded NaN because integer `count + 1` overflowed before `gammaln`; the verified log probability is now exactly `0`.
2. At `n=2**53+1`, one event with probability `1/n` previously yielded `-38.7368005696771`. The final stable calculation produces exactly `-1.0`, matching the independent 60-digit oracle.

The complete numerical probe was rerun after the final repair. Terminal-atom prefix absorption has zero remaining population, mean, and variance; negative host counts are rejected before casting; overflowing observed columns are rejected. The tested central-difference gradient discrepancy is `1.03e-10`.

Further independent probes exposed near-mode cancellation in `count * (log_share - log_probability)`. A Loader-style deviance series now avoids subtracting large, nearly equal category contributions; this is a general numerical repair, not input-specific exceptions.

- `n=2**62`, first-category count `n//2 + 1000000000`: the repaired log probability and oracle both equal `-22.147034818997234` (previous error: `-0.43368` nats).
- `n=2**53+1`, first-category count `n//2 + 30000000`: both equal `-18.794031775254467` (previous error: `+0.80016` nats).
- Both opposite-tail offsets also match their Decimal oracles. At `n=2**62` with unequal probabilities `[0.3,0.7]`, the checked near-mode discrepancy is approximately `4.9e-7` nats.
- Three-way large-count and shifted-log-weight cases differ by less than `6e-10` nats. Far-tail cases have errors at output floating-point precision; analytic gradient comparisons also pass the probe's tolerances.

The sampler was not changed by this numerical repair. The timing tables above precede the final probability-scoring repair; the final package must be remeasured before any performance acceptance claim.

Two integration constraints remain explicit: the tested float32 large-count log PMF differs from its high-precision oracle by `0.10877` nats, so the public count path must use x64; impossible observed prefixes must be rejected at the host boundary before native sampling. Under tracing, the candidate marks an infeasible remaining pool as `-1`, but the sampler does not raise a Python exception. These probes do not establish all other adoption criteria.

## Independent event-time family probe

A throwaway native-model probe ran a discretized Weibull family with shape 2 through real SVI, upstream posterior sampling, forecasting, and in-sample prediction:

- Posterior sites are `log_scale` and `cure_logit`, with no age-logit representation.
- 40 SVI steps have finite final loss `13.106949806213379`.
- Forecast shape changes from `[64,10,8]` to `[64,17,8]` when replayed at a longer horizon; in-sample shape is `[64,8,8]`.
- Observed units remain absorbed; closures and source dates are respected; closing the entire future produces zero events.
- A finite discrete law reproduces masses `[0.2,0.3,0.5,0]`. Closing its terminal-atom day leaves residual probability `0.5`, rather than forcing an event.
- An unresolved-tail discrete law retains residual probability `0.5`.

This proves a candidate family-independent numerical/native interface, not the new public `CureProcess` API, a fitted network with arbitrary families, or eventual-expectation rejection for an unknown tail. Those production acceptance criteria remain pending.

## User-approved continuation

The user's later clarification supersedes the earlier fallback interpretation: “unified model” means one native NumPyro execution path, not merely one mathematical kernel feeding separate samplers. NumPy measurements below remain useful comparison/oracle evidence only. No hidden host callback, split production sampler or legacy/new backend selector is allowed.

### Native count-sampler optimization

The optimized candidate changes JAX binomial inversion/BTRS loop termination so inactive lanes do not keep a vector loop running. It retains the same finite-precision proposal/acceptance machinery; exact int64 residual accounting does not prove extreme-count distributional accuracy.

| Counted case, quantity 10,000,000 / sparse | Existing warm seconds | Original native warm seconds | Masked native warm seconds | Masked / existing |
| --- | ---: | ---: | ---: | ---: |
| `(1000,180,90,128)` | 0.628899209 | 2.731392333 | 2.107693250 | 3.351× |
| `(5000,365,365,32)` | 2.751694000 | 11.932200000 | 9.695855584 | 3.524× |
| `(5000,365,365,128)` | 13.025100000 | 47.937800000 | 40.025300208 | 3.073× |

The optimization helps but does not meet the warm target. Native count performance remains the production adoption blocker.

The parent executed the full 100,000-draw lane/multinomial/closure checks for stock, unmasked and masked variants. Moment/covariance checks and exact structural count bookkeeping passed. These checks also deliberately demonstrate a limitation: float64 proposals lose full integer support at sufficiently large counts, and the BTRS log-acceptance bound suffers cancellation. At `n=2**53+1`, measured bound error reaches approximately one nat; at balanced `n=2**62`, proposals occupy a coarse integer lattice. Near-certain odd-count completion remains exact.

An independent run through `numpy.random.Generator.binomial` confirms the existing NumPy sampler also loses full support at extreme balanced counts: `n=2**55, p=0.5` produced only four residues modulo eight, and `n=2**62, p=0.5` only one. At `n=10,000,000` and `2**40`, all eight residues appeared and the deterministic residue checks passed. This baseline qualifies the existing public int64 conservation contract; it is not a new claim of arbitrary-int64 distributional accuracy or a production strategy.

### Independent fitting measurements

Fitting uses the actual regularized random-walk prior, real `StageObservations`, selected-survivor entry conditioning, inclusive deadline 45, two hazard regressors, two cure regressors and heterogeneous closures. Fixture seed is 7341. Native inputs use the whole administrative exposure calendar, not the observed event's stopping mask. Each timed backend runs in a separate process with explicit compilation and synchronization.

The ordinary event-mass distribution agrees numerically but exceeds the warm fitting target. A compact native event-time distribution preserves the complete law while evaluating the observed event/censor likelihood without materializing all event masses; mean and sampling remain available from the same law.

| Fitting case `(rows,days)` | Existing warm 100-step SVI chunk | Full event-mass native | Compact native before layout fix | Compact / existing |
| --- | ---: | ---: | ---: | ---: |
| `(100,60)` | 0.009842834 | 0.014747792 | 0.010453167 | 1.062× |
| `(1000,180)` | 0.101813125 | 0.137645000 | 0.118302958 | 1.162× |
| `(5000,365)` | 0.973540250 | 1.251193000 | 1.304145125 | 1.340× |

The final compact reduction keeps each unit's calendar contiguous. In separately matched large-case runs, its 100-step SVI median is `1.129293875` seconds versus `0.980129708` existing (**1.152×**, within the prospective target), with identical displayed loss `7576.4375`. Raw likelihood-gradient medians are `0.013411083` versus `0.009951333` (**1.348×**); joint-model gradients are `0.013543125` versus `0.010089875` (**1.342×**). Native fitting therefore remains numerically accepted but does not pass every prospective performance measure.

The medium compact likelihood/gradient checks pass against the existing path and independent float64 reference, including prior/extreme-logit cases. Full-law checks also pass for a 3,650-day history. A sliced-window probe exposed a NaN gradient at hazard logit -25; safe operands in unused `log1mexp` branches repair it. Fresh checks at logits -100, -25, 0, 40 and 100 produce finite gradients. Production must additionally cap administrative exposure at `as_of` or require the fitting calendar to end there; the throwaway native adapter was tested only with `calendar[-1] == as_of`.

### Rejected shared-kernel / NumPy diagnostic

A deterministic JAX kernel derives per-day conditional event probabilities from the cure-conditioned logit and exclusive susceptible survival; the diagnostic adapter then uses NumPy for allocation. Against an independent calculation, the checked maximum probability error is `2.08e-17`, and near-certain accounting retains exactly `9007199254740993` events.

This isolates native allocation as the dominant regression. With adaptive draw batches, all 12 matrix rows are below the prospective 1.25× warm target; medium sparse rows take `0.526–0.564` seconds versus `0.598–0.627`, while large sparse rows take `2.406–2.610` versus `2.633–2.785`. Large multi-arrival rows take `4.822–5.410` versus `5.533–6.145`. Large peak RSS is 25–31% below the existing processes. Small fixed-runtime RSS remains 1.65–1.78× because JAX is loaded.

The generalized adapter passed structural checks for draw-specific source dates, multi-day count arrivals, inclusive deadline-zero same-day events, expired-but-pending historical units, all-closed calendars and int64 conservation. Its grouping preserves packed pool order rather than sorting cohort labels. These results are diagnostic only: the user explicitly rejected this split execution as the production architecture.

### Real retail workflow

The actual retail example was run in separate processes with 150 joint SVI steps and 40 posterior draws; each process made one cold and five warm forecasts. The native fitting comparison changes only the observation registration; both rows still use the existing NumPy forecast implementation.

| Backend | Fit cold seconds | First forecast seconds | Warm forecast median seconds | Total measured seconds | Peak RSS bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing | 8.477390834 | 0.618403042 | 0.079459959 | 9.564026625 | 1564196864 |
| Compact native observations | 10.287874416 | 0.555297500 | 0.077791209 | 11.301035084 | 1700380672 |

Both runs have finite final joint loss `1492.4757080078125`, exact mass conservation, zero weekend receipts, mean forecast receipts 15.05 and the same reported interval `[9.9,21.0]`. Expected historical receipts differ by less than `5e-8`. Compact native fitting takes 1.214× as long in this measured cold workflow.

An earlier existing-path run took 92.7 seconds and its unchanged warm forecast took 0.588 seconds; the clean paired rerun above does not reproduce that slowdown. The earlier result is excluded from performance conclusions.

## Final adoption interpretation

The complete matrix and independent reviews support three separate facts:

1. The shared family-independent probability core is mathematically sound for the exercised moderate-count fixtures, native Weibull fitting/replay and count moments. Production must retain the existing host validation prologue and explicitly bound fitting exposure at `as_of`.
2. Compact native unit observations preserve likelihoods, gradients and SVI results; final large SVI is 1.152× existing, while raw gradients remain about 1.34×.
3. The date-sequential native allocator is 3.07–3.52× slower on measured large workloads. A balanced binomial tree and separate native unit/count population sites reduce this, but the best large-32 result remains about 1.92× existing. The NumPy diagnostic cannot satisfy the clarified unified-model requirement.

Result after the user's subsequent “ok then proceed!”: **GO for implementation with documented native overhead.** The unified-model requirement and correctness gates remain binding. The draft PR must expose final measured performance; this approval does not imply benchmark parity.

### Native-only sampler follow-through

The parent executed the balanced-tree sampler's deterministic replay, sample/batch shape, moderate-count moments and all covariances, closure/structural-zero, prefix/window surgery and int64 completion probes. Three-category `n=7` means are `[1.39816, 2.10501, 3.49683]` over 100,000 draws; maximum covariance error is 1.37 standard errors. The `n=10,000,000` case is below one covariance standard error. Both 13-category fixtures pass. Eager infeasible prefixes are refused; the traced sentinel path and `2**53+1` near-certain completion pass. The float-proposal limitations remain explicit.

The tree reduces sequential binomial calls from 90 to 7 in the medium fixture, but widens vectors and increases lane-iterations. At large scale, an all-population tree is slower than the masked sequential candidate. Separating known one-unit pools into native `EventTime` sites and bulk pools into native `CohortEventTime` sites avoids unnecessary binomial work on unit pools.

| Native-only candidate | Medium warm seconds | Large-32 warm seconds | Large-32 peak RSS |
| --- | ---: | ---: | ---: |
| Tree, all pools, draw batch 16 | 1.867669125 | 12.694316958 | 5,572,771,840 |
| Tree, separate unit/count sites; best measured draw batching | 1.317742333 | 6.867807708 | 1,931,149,312 |
| Same sites; distribution-local XLA random-bit generation and scalar predictive map | 1.345373958 | 5.274946958 | 1,934,639,104 |

Existing matching sparse quantity-10,000,000 reference times are `0.628899209` and `2.751694000` seconds. Native table medians use five warm repetitions; the best large cases use draw batch one and the medium cases use batch 16. The final large native cold call is `7.341539500` seconds, versus its warm median `5.274946958`; five warm samples range from `5.118519625` to `6.190069125`.

Scan unrolling at factors 4, 8, 16 and 32 does not materially improve the medium native chain and increases compilation costs. Distribution-local XLA random-bit generation cannot be used under the current vmapped rejection loops: the attempted vmapped run timed out, consistent with the first-key batching semantics of JAX's experimental RBG. The successful comparison uses scalar predictive mapping; it is not approved for production, and its native distribution-local RNG variant still needs separate distributional validation. Upstream NumPyro also strips typed-key implementation information in `Predictive`, so passing an RBG key directly is refused. These constraints are part of the tradeoff, not hidden workarounds.

## Production probability-layer verification

Default `uv sync --locked` installs `numpyro-forecast 0.4.0`; importing `Horizon`, `predict`, and `forecast` succeeds without an optional extra.

The parent exercised the production `EventTime` and `CohortEventTime` through 50-step AutoNormal SVI, 32 posterior draws, native `forecast` replay from a five-date training window to nine dates, and `predict_in_sample`. Both runs have finite losses, integer trajectories, exact pool bounds, absorbing unit events, and zero events on a closed date. Future arrays have shape `[32,4,2]`; in-sample arrays have shape `[32,5,2]`. Conditional native means change with future weather without changing historical inputs. Cohort future events total 6 in calm weather and 4 under the storm scenario; the unit sample totals happen to be zero in both, so weather sensitivity is established by its conditional means rather than claiming sampled totals must differ.

Initial focused checks: 47 shared-kernel/survival tests pass. The combined run has 66 passes and one statistical-test assertion defect (vector `atol` rejected by NumPy's error formatting); the corrected standardized-error assertion passes. New regressions expose two distinct empty-window defects: a traced impossible full prefix scored zero instead of `-inf`, and a zero-date compact kernel treated event index `-2` as no event. After those fixes, the parent observes **69 passing tests** and repeats the actual native API smoke successfully.

Independent production-layer numerical review finds no high-severity shared-kernel defects. The parent reproduces a NaN mass-grid suffix incorrectly becoming a valid zero-hazard cell; replacing the exhaustion predicate with an exact negative-infinity check preserves NaN rather than silently repairing the law. Per-draw `[D,1,A]` versus per-cohort `[C,A]` mass grids have an explicit behavioral test and documented layout. The default family documents administrative negative-age masking. The updated shared-kernel suite has **22 passing tests**, and Ruff passes for the kernel, survival helpers, exports and kernel tests. The distribution review identifies further impossible-prefix sampling/support, NaN, shape and count-range boundaries to correct before layer closure.

After the numerical-review corrections, the parent observes **79 passing shared-kernel/distribution/survival tests in 43.64 seconds**, then repeats the actual unit/cohort native API smoke successfully. Impossible traced observations now draw visible negative sentinels rather than plausible zero events; eager zero-probability prefixes are refused. NaN numerics remain distinct from structural zero mass. Rank, prefix-length and float-to-integer domain checks have consumer regressions. Ruff passes after replacing two ambiguous test argument names and formatting the new modules/tests.

Fresh pinned-reference measurements from `8a85dfd661740e9df6c1a61b4e28a3262ab3afea`, in a detached benchmark worktree with the same installed dependencies: real retail (150 SVI steps, 40 draws) fits in 7.303102625 seconds, first forecast 0.575479917 seconds, warm median 0.073085167 seconds, total 8.312623750 seconds, peak RSS 1,511,702,528 bytes. Its final loss is `1492.4757080078125`, expected historical receipts `3.962345190929851`, mean forecast receipts 15.05, and conservation/finite-loss/closed-weekend checks pass. The large-32 sparse quantity-10,000,000 fixture has cold 2.738233625 seconds, five-call warm median 2.616348875 seconds and peak RSS 1,650,868,224 bytes. These are refreshed reference results, not measurements of the still-unmigrated production forecast path.

Final desk review accepts the distribution repairs. The parent also reproduces host int64 observations wrapping into feasible int32 events/counts, records three failing-before regressions, and adds a host-only pre-conversion range guard; JAX arrays/tracers bypass host inspection. The resulting distribution suite has **32 passing tests**, the actual native API smoke passes again, and Ruff lint/format checks pass. The full public compatibility suite before this final host guard has **185 passing tests in 57.79 seconds**. No unresolved probability-layer review finding remains. Default fitting and production cohort propagation migrate in the next tasks; this layer alone is not the completed cutover.
