# Retail survival forecasting

## Authorization and scope

The user approved retail sales with two observed transitions (initiation and receipt), abandonment, both outstanding measures, date conveniences, calendar regressors, and a 90-day initiation policy. The user then explicitly requested a plan followed by immediate implementation, superseding further interactive design/spec approval pauses.

Build a Python library and runnable end-to-end example, not rental inventory optimization. Preserve existing uncommitted repository configuration. Work on `feat/retail-survival`; do not merge or push.

## Alternatives and decision

1. **Discrete-time mixture-cure stages with calendar-aware propagation (chosen).** Learn flexible age hazards and regression effects from censored item histories; forecast coherent calendar-day counts. Handles policy boundaries and closed days directly. Daily resolution and explicit tail assumptions are the tradeoffs.
2. Continuous-time parametric duration models. Compact, but hard closures and changing regressors complicate integration and restrict duration shapes.
3. Aggregate stationary convolution. Fast for homogeneous, stationary cohorts, but insufficient as the primary abstraction for conditional outstanding populations and calendar regressors.

The primitive is a cure-capable event stage. A typed count-root/event-node graph retains dated root-cohort lineage; the retail interface is its sale → initiation → receipt specialization, not a second implementation.

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

Use the optional `numpyro_forecast>=0.3,<0.4` functional model protocol (`Horizon`, `predict`, root `obs` and `forecast` sites). Preserve posterior predictive count samples without averaging. `SalesForecast` binds dated cohort metadata, integer quantities, and draw identity in one value. Reject fractional, negative, nonfinite, or overflowing counts. Fit the root count likelihood and the event-history likelihoods in one scoped NumPyro program for joint mode. Do not add aggregate initiation/receipt likelihoods on top of the unit-history evidence.

## Validation and error handling

Defend the real risks: as-of leakage; sparse changes using future values; policy day 90 vs 91; zero-hazard days; same-day and after-day-90 receipts; cure-conditioned old items; abandonment; mass conservation; correct expected receipts on an analytically solvable case; finite log likelihoods and gradients under ordinary censoring; posterior parameter shapes; real NumPyro Forecast draw preservation.

Run an end-to-end synthetic retail example with a 90-day policy, non-returners, abandoned initiations, weekend receipt closures, a storm regressor, two fitted stages, future-sales uncertainty, and reported outstanding quantities. Include a held-out calendar forecast and score it. The example should save a small plot or print machine-checkable summaries; correctness comes from executed invariants and analytical cases, not a picture alone.

## Limits made explicit

Daily resolution, conditional independence between units given parameters/covariates, no partial-unit returns or repeated return attempts, no inventory feedback, no automatic forecasting of unknown weather, and no identification of infinite-lifetime cure from finite data. Forecasts depend on the supplied future-sales mix, date scenarios, and documented tail convention.

## Public network and retail interface

The user approved this clean cutover after reviewing the single-sales/joint-network mockups and starting-population semantics:

```python
model = RetailReturnModel(
    sales=CountProcess(model=sales_model),
    initiation=CureProcess(age_bins=30, deadline_days=90),
    receipt=CureProcess(age_bins=30, allowed_weekdays=range(5)),
)
observed = RetailData.from_units(
    unit_history,
    as_of=cutoff,
    calendar=training_calendar,
    group_by=["product"],
)
fitted = model.fit(observed, covariates=historical_covariates, mode="joint", num_samples=500)
forecast = fitted.forecast(horizon=28, covariates=future_covariates)
```

`CountNode`, `EventNode`, and `ForecastNetwork` expose the same underlying execution path. One count root supplies dated unit cohorts; each event node has one immediate source and its own origin/age, cure process, optional inclusive deadline, and hard weekday mask. Arbitrary event descendants are supported; branching children are separate events, not competing risks. Data uses a canonical sale-date root; custom event columns are explicit. Reject cycles, missing sources, duplicate names, and reuse of one observed event column as multiple likelihoods.

`joint` uses one NumPyro program, one SVI guide, and one posterior draw across the entire graph. `modular` explicitly fits independent blocks. Fully observed parents and independent priors still factorize in joint mode. Optional `shared_model()` samples shared latent values: count callbacks receive `shared=values`; custom event `parameter_model(observations, values)` callbacks return `StageParameters` sampled from conditional priors. This provides real cross-stage learning without a feature-expression language. Shared models require joint mode. The default guide is mean-field variational inference, not exact posterior uncertainty; the joint model is directly usable with NumPyro MCMC.

`SalesForecast.from_frame` accepts sale dates, quantities, static attributes, and optional draw labels. `from_numpyro_forecast` accepts all `[draw, time, group]` trajectories and group metadata. Missing draw/cohort cells are errors, not implicit zeros. The object owns cohort metadata and count paths; the public `future_counts` argument and tuple-returning sales adapter are removed. One source draw broadcasts; otherwise source and parameter draw counts must match. Same-posterior source labels preserve pairing.

At forecasting, supplied `future_sales` replaces internal generation; it is an external scenario, not extra sales or posterior conditioning. A modeled source can generate internally. With no modeled source, an explicit sales scenario is required; an empty `SalesForecast` states that no new sales will occur.

Covariates map node names to count-model arrays or event mappings/callables returning `features`, `cure_features`, and `allowed`. Event providers receive the cohort frame and requested dates. Fitting uses the original-origin context calendar. Forecasting retains cached historical exposures and requests future values for historical units followed by new cohorts. Only pre-birth padding may be zero-filled; missing values during risk exposure are errors. Historical cure features are static. Learned feature definitions/scaling remain fixed; no future weather is invented.

Configuration and fitted snapshots are separate frozen containers; fitted data is copied. Nested frames, mappings, and callback state remain read-only by convention. There is no in-place refit, inheritance framework, serialization system, or automatic preprocessing.

## Starting populations and observation entry

`RetailData.calendar` is the root sales observation window and ends at `as_of`. Full `unit_history` may precede its first day. Old purchases never enter the in-window sales count likelihood, but their original sale and initiation clocks remain. Complete historical event evidence contributes once, including completed pre-window events.

Optional `starting_items` is a survivor-selected snapshot at end-of-day before the first observed day. Merge its known histories with later outcomes by item identifier; do not duplicate units or overwrite known origins/events. Reject conflicting static groups, future snapshot events, completed snapshot receipts, and purchases on/after the window start. Already completed stages contribute no second event likelihood. A parent event arising after entry starts its child's clock normally.

For selected survivors, condition on survival to entry. With susceptible pre-entry survival `S_pre`, the post-entry susceptibility logit is `logit(pi) + log(S_pre)`. Evaluate the post-entry event/censor likelihood with that conditional susceptibility, rather than treating selected survivors as a complete birth cohort or subtracting large marginal likelihoods. Original-origin covariates are required to compute this selection probability. Unknown starting ages/counts are not silently imputed.

## Implementation and verification boundaries

The event-history likelihood stays vectorized JAX. The shared cohort propagation kernel uses host float64 logits/log-survival and binomial counts, retaining exact dated lineage without expanding quantities into units. A scan rewrite is not required for joint fitting and must not compromise numerical stability. All public forecast paths use this same kernel.

Verify joint shared-effect learning with real inference, native count forecast uncertainty, explicit source replacement, arbitrary descendant clock propagation, starting-population receipts without in-window sales, conditional-entry likelihood/gradients, existing strong-logit regressions, same-day events, inclusive deadlines, hard closures, fit isolation, and draw-wise stock conservation. Run the complete retail example with a sales window starting after the earliest unit history, plus the existing suite, formatter/linter, and package build. Leave the branch unmerged.
