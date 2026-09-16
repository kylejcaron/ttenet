# Retail survival forecasting

## Authorization and scope

The user approved retail sales with two observed transitions (initiation and receipt), abandonment, both outstanding measures, date conveniences, calendar regressors, and a 90-day initiation policy. The user then explicitly requested a plan followed by immediate implementation, superseding further interactive design/spec approval pauses.

Build a Python library and runnable end-to-end example, not rental inventory optimization. Preserve existing uncommitted repository configuration. Work on `feat/retail-survival`; do not merge or push.

## Alternatives and decision

1. **Discrete-time mixture-cure stages with calendar-aware propagation (chosen).** Learn flexible age hazards and regression effects from censored item histories; forecast coherent calendar-day counts. Handles policy boundaries and closed days directly. Daily resolution and explicit tail assumptions are the tradeoffs.
2. Continuous-time parametric duration models. Compact, but hard closures and changing regressors complicate integration and restrict duration shapes.
3. Aggregate stationary convolution. Fast for homogeneous, stationary cohorts, but insufficient as the primary abstraction for conditional outstanding populations and calendar regressors.

The primitive is a cure-capable event stage; retail links two such stages. No generic graph executor is required for these two linked stages.

## Process and clocks

Sale -> initiation -> receipt. Each sold unit initiates at most once; each initiation is received at most once. An initiation may never occur, and an initiated return may be abandoned. Receipt is not an independent prediction from initiation.

Dates represent UTC calendar days. As-of means end of that day; forecasts begin the following day. Sale and initiation days have age zero, and same-day transitions are supported. Initiation is allowed through sale date + 90 calendar days, inclusive. The policy is configurable and is never silently extended by weekends. Receipt is not subject to the sale's initiation deadline.

Initiation clock: days since sale. Receipt clock: days since initiation. Calendar date is a separate clock. Storms and other supplied regressors can affect either stage. Each stage also has its own exact allowed-day mask. A closed day has zero hazard and does not itself mark a unit as cured.

## Data conveniences

Accept tabular unit histories, longitudinal snapshots, and event change records, producing the same canonical history. Canonical columns: `item_id`, `sale_date`, `initiation_date`, `receipt_date`; extra static numeric columns survive conversion. Future events must be hidden at the as-of cutoff. Invalid ordering, duplicate identifiers/ambiguous events, initiation beyond policy, and receipt without initiation are errors, not silently repaired.

Longitudinal input uses `recorded_date` with complete snapshots; use the latest snapshot known by as-of. Change records use `date` and `event` in `sale`, `initiation`, `receipt`. Covariate changes use a separate dated table: expand sparse updates forward only, never backward from future records. Dense longitudinal covariates require actual coverage. Helpers convert dates, generate inclusive daily grids, compute clocks, construct calendar features and allowed-day masks.

Data preparation is pandas/NumPy; likelihoods and numerical primitives are JAX-compatible. Model input arrays are explicit and inspectable.

## Statistical model

For each stage, susceptibility probability is `pi = sigmoid(cure_intercept + z @ cure_beta)`. Susceptible daily hazard is `h(a,t) = sigmoid(age_logits[min(a,K-1)] + x(t) @ beta)` on allowed days, zero otherwise. The age baseline is flexible with regularizing priors rather than a prescribed parametric duration family. Covariates may combine static attributes, longitudinal changes, and date features. Susceptibility regressors are static at stage entry; changing regressors affect timing hazards.

For observed event on day d, likelihood is `pi * h_d * product_{k<d}(1-h_k)`. For censoring after day c, likelihood is `(1-pi) + pi * product_{k<=c}(1-h_k)`. Compute these in log space, correctly including age-zero risk and ignoring padded/out-of-risk dates. Do not label censored rows as known cures.

Initiation risk ends at the policy deadline. Thus `pi` is latent susceptibility, NOT automatically the realized return rate: some susceptible items also miss the policy window. Report derived event probabilities, not `pi`, as expected actual returns. This distinction is explicit because finite follow-up does not identify cure and a long tail without assumptions.

Receipt uses the final age-bin hazard beyond the fitted bins. Eventual-receipt expectations assume this positive tail continues and receiving eventually reopens; finite-horizon draws use only supplied calendar scenarios. Do not infer an extra shipping deadline or label all open returns abandoned.

Inference is ordinary NumPyro. Provide a small SVI convenience function returning posterior stage-parameter draws and loss history; advanced users can use the model directly with MCMC. Posterior draws retain a draw axis. No claim that variational uncertainty is exact.

## Forecasting and outstanding quantities

Condition old cohorts on their observed history. For an unobserved stage, posterior susceptibility after elapsed survival is `pi*S_susceptible / ((1-pi)+pi*S_susceptible)`. Forecast remaining events from that state, never restart clocks.

Historical rows are sold units. Future-sales rows are homogeneous cohorts with sale date and nonnegative integer quantity; per-draw quantities may carry uncertain sales forecasts. Do not expand large sales counts into unit rows. Retain counts and sample binomial transitions, so counts conserve mass within every draw.

Return daily initiation and receipt draws, daily eligible-not-initiated counts, and daily initiated-not-received counts. Return expected remaining receipts at the forecast origin for the historical population, split between pending initiations and existing open returns. Closed/expired sales contribute no further initiations. Never-returning/abandoned latent units are not observable labels in the input.

To compute all remaining initiation probability, the supplied scenario calendar must extend through the latest uninitiated sale's policy deadline, even if the requested output horizon is shorter. Require coverage rather than invent future weather. Receipt's eventual probability uses the documented continuing-tail assumption. Future covariates can be supplied as explicit scenarios; parameter and sales uncertainty are propagated draw by draw.

## NumPyro Forecast integration

Use the real optional `numpyro_forecast>=0.3,<0.4` functional API, not the old class API. Compose its posterior predictive sales counts (sample, time, observation) with TTENet future cohorts without replacing uncertain draws by their mean. Reject fractional, negative, or nonfinite counts. A worked example must fit a real NumPyro sales model, call `numpyro_forecast.forecast`, and pass its draws into the return forecast. The return-stage likelihood remains event-history based; do not pretend aggregate daily counts are independent likelihoods for the two observed stages.

## Validation and error handling

Defend the real risks: as-of leakage; sparse changes using future values; policy day 90 vs 91; zero-hazard days; same-day and after-day-90 receipts; cure-conditioned old items; abandonment; mass conservation; correct expected receipts on an analytically solvable case; finite log likelihoods and gradients under ordinary censoring; posterior parameter shapes; real NumPyro Forecast draw preservation.

Run an end-to-end synthetic retail example with a 90-day policy, non-returners, abandoned initiations, weekend receipt closures, a storm regressor, two fitted stages, future-sales uncertainty, and reported outstanding quantities. Include a held-out calendar forecast and score it. The example should save a small plot or print machine-checkable summaries; correctness comes from executed invariants and analytical cases, not a picture alone.

## Limits made explicit

Daily resolution, conditional independence between units given parameters/covariates, no partial-unit returns or repeated return attempts, no inventory feedback, no automatic forecasting of unknown weather, and no identification of infinite-lifetime cure from finite data. Forecasts depend on the supplied future-sales mix, date scenarios, and documented tail convention.

## Thin public model interface

The user selected a thin public class after reviewing the functional API. Use two frozen dataclasses, not a stateful fit-in-place estimator or a model inheritance hierarchy:

- `RetailReturnModel(policy_days=90, age_bins=30, feature_builder=None)` owns reusable configuration. Both stages use `age_bins`; existing stage functions remain available for advanced configurations.
- `.fit(data, *, calendar, as_of=None, layout="tabular", num_steps=500, num_samples=100, seed=0, learning_rate=0.02)` returns a separate `FittedRetailReturnModel`. Raw data require `as_of` and use the existing layout adapters. A `RetailHistory` is accepted without `as_of`/layout overrides, must match the model policy, and is copied so subsequent caller edits do not change the fit's history.
- `FittedRetailReturnModel` retains `model`, `history`, `initiation_fit`, and `receipt_fit`. `.forecast(*, calendar, horizon, future_sales=None, future_counts=None, seed=0)` returns the existing `ReturnForecast`. Fitting the same configuration again must not change an earlier fit's forecasts.

`feature_builder(frame, calendar)` is a pure callable returning a mapping with any of the existing forecast keyword names: `initiation_features`, `receipt_features`, `initiation_cure_features`, `receipt_cure_features`, `initiation_allowed`, `receipt_allowed`. Unknown keys are errors rather than silently ignored regressors or closure masks. With no builder, both stages have no regressors and all days are allowed. Fit-time rows are canonical historical units; forecast-time rows are historical units followed by future cohorts. The wrapper handles that ordering. Array shape validation stays in the existing numerical boundary functions.

Keep calendars and future covariate scenarios explicit. The builder must preserve feature definitions, ordering, and fixed training-derived scaling; the wrapper does not learn preprocessing or forecast weather. Frozen containers do not make nested arrays/dataframes/callback state deeply immutable; treat fitted contents and builder inputs as read-only.

The numerical model, priors, simulator, and optional sales-forecast integration are unchanged. No scan rewrite, serialization framework, feature-expression language, or additional inference backend is included. The README and full worked example lead with the class interface; low-level functions remain supported numerical building blocks.
