"""The essay's model overview: configured stages and inspectable, real source."""

import ast
import inspect
import textwrap
from pathlib import Path

from ttenet import CountProcess, EventProcess
from ttenet.event_times import SurvivalKernel, default_timing, exclusive_log_survival
from ttenet.families import _timing_from_log_increment, _weibull_log_increment
from ttenet.models import _observe, observation_kernel, sample_stage_parameters
from ttenet.network import FittedNetwork, ForecastNetwork, _event_model


def _source(obj):
    """Keep original line numbers; only docstrings and validation start folded."""
    lines, start = inspect.getsourcelines(obj)
    code = textwrap.dedent("".join(lines))
    block = ast.parse(code).body[0]
    folds = []
    if (
        block.body
        and isinstance(block.body[0], ast.Expr)
        and isinstance(block.body[0].value, ast.Constant)
        and isinstance(block.body[0].value.value, str)
    ):
        doc = block.body[0]
        if doc.end_lineno > doc.lineno:
            folds.append({"start": doc.lineno - 1, "end": doc.end_lineno - 1, "label": "Docstring"})
    if isinstance(block, ast.ClassDef):
        for method in block.body:
            if isinstance(method, ast.FunctionDef) and method.name == "__post_init__":
                folds.append(
                    {
                        "start": method.lineno - 1,
                        "end": method.end_lineno - 1,
                        "label": "Configuration validation",
                    }
                )
    return {
        "name": obj.__qualname__,
        "file": Path(inspect.getsourcefile(obj)).name,
        "start": start,
        "lines": code.rstrip().splitlines(),
        "folds": folds,
    }


def _note(text, *, math=False):
    return {"text": text, "math": math}


def model_overview_data(model, initiation_covariates, receipt_covariates):
    """Describe this essay's configured model; no fitting or forecast work happens here."""
    sales_model = model.sales.model
    family = model.receipt.family
    rho = family.susceptibility_logit_prior
    scale, shape = family.scale_prior, family.shape_prior
    return {
        "stages": [
            {
                "id": "sales",
                "name": "Sales",
                "code": f"sales=CountProcess(\n    model={sales_model.__name__}),",
                "question": "How many units?",
                "takeaway": "Weekdays and planned promotions shape the daily sales forecast.",
                "rule": "Given: promo calendar",
            },
            {
                "id": "initiation",
                "name": "Return starts",
                "code": (
                    "initiation=EventProcess(\n"
                    f"    age_bins={model.initiation.age_bins}, "
                    f"deadline_days={model.initiation.deadline_days}),"
                ),
                "question": "Will it happen, and when?",
                "takeaway": "Learn the chance of a return and the time from sale to starting it.",
                "rule": f"Given: {model.initiation.deadline_days}-day window",
            },
            {
                "id": "receipt",
                "name": "Return receipts",
                "code": (
                    "receipt=EventProcess(\n"
                    f"    family={type(family).__name__}(...),\n"
                    "    allowed_weekdays=range(5)),"
                ),
                "question": "A new clock, not the sale clock.",
                "takeaway": (
                    "Each initiation starts a receipt clock, with its own arrival "
                    "probability and warehouse calendar."
                ),
                "rule": "Given: weekdays only",
            },
        ],
        "pieces": {
            "sales": {
                "kicker": "CountProcess · how many",
                "title": "Your sales model supplies the counts.",
                "main": (
                    "CountProcess holds the callable you provide. Here, that is a "
                    "Poisson model with weekday and promo effects."
                ),
                "body": (
                    "During fitting it observes recorded sales. During forecasting, the "
                    "same callable generates future counts under fitted parameter draws."
                ),
                "equation": "calendar + fitted parameters → daily sales",
                "source": _source(sales_model),
                "more_title": "This notebook's count model and covariates",
                "notes": [
                    _note(
                        "sales[t, p] ∼ Poisson(exp(η[t, p]))\n"
                        "η = product baseline + weekday effect + promo lift",
                        math=True,
                    ),
                    _note(
                        "Baseline: Normal(0, 0.6). Weekday effects: Normal(0, 0.4). "
                        "Promo lift: Normal(0, 1). These are the example's priors, not "
                        "requirements of CountProcess."
                    ),
                    _note(
                        "The callable follows numpyro_forecast's protocol: time is on "
                        "axis −2; predict separates the observed prefix from the future suffix."
                    ),
                    _note(
                        "An event covariate provider is called as provider(frame, calendar). "
                        "The frame contains historical units followed by new sales cohorts."
                    ),
                    _note(
                        "features[unit, day, P] contains time-varying regressors; "
                        "susceptibility_features[unit, Q] contains static attributes known "
                        "when the source event happens. Static features cannot change for "
                        "existing units in a scenario."
                    ),
                    _note(
                        "An optional allowed mask closes additional days. Features must "
                        "keep their meaning across fitting and forecasting: the default "
                        "family shifts hazard logits; WeibullFamily multiplies cumulative-"
                        "hazard increments by exp(x·β). Those coefficients are not interchangeable."
                    ),
                ],
                "extra_sources": [
                    _source(CountProcess),
                    _source(ForecastNetwork._count_model),
                    _source(initiation_covariates),
                    _source(receipt_covariates),
                ],
            },
            "initiation": {
                "kicker": "EventProcess · first clock",
                "title": "A return probability and a flexible timing law.",
                "main": (
                    "One part learns which units are susceptible to a return. The other "
                    "learns when those returns start, measured from the sale date."
                ),
                "body": (
                    "The default family uses an age-binned hazard with product and weekday "
                    "effects. A unit still waiting is not automatically counted as a never-return."
                ),
                "equation": "sale date → initiation clock",
                "source": _source(default_timing),
                "more_title": "Learned timing; declared policy",
                "notes": [
                    _note(
                        "π = σ(θ₀ + θ₁ · product)\n"
                        f"h(age, t) = σ(ℓ[min(age, {model.initiation.age_bins - 1})] + β · x[t])",
                        math=True,
                    ),
                    _note(
                        f"Age effects follow a random walk over {model.initiation.age_bins} "
                        "bins: age_scale ∼ HalfNormal(1), age_init ∼ Normal(0, 2), "
                        "standardized age_steps ∼ Normal(0, 1). Hazard and susceptibility "
                        "coefficients have Normal(0, 1) priors; the susceptibility intercept "
                        "has a Normal(0, 2) prior."
                    ),
                    _note(
                        f"The {model.initiation.deadline_days}-day policy is supplied, not "
                        "learned, and is inclusive from the sale date. Initiations are "
                        "allowed every day."
                    ),
                    _note("P(no initiation yet) = (1 − π) + πS(age)", math=True),
                    _note(
                        "That combines never returning with still waiting. Both parts must "
                        "be fitted together while outcomes are censored."
                    ),
                ],
                "extra_sources": [_source(EventProcess), _source(sample_stage_parameters)],
            },
            "receipt": {
                "kicker": "EventProcess · second clock",
                "title": "Restart the clock when the return starts.",
                "main": (
                    "The receipt stage learns whether a started return reaches the warehouse "
                    "and how long that takes. Its age begins at initiation, not at sale."
                ),
                "body": (
                    "Here, WeibullFamily supplies the timing law. Warehouse closures mask "
                    "the daily hazard; they do not pause the age clock."
                ),
                "equation": "initiation date → receipt clock",
                "source": _source(type(family).model),
                "more_title": "A different family; the same event machinery",
                "notes": [
                    _note("π = σ(ρ)\nS(age) = exp(−(age / λ)ᵏ)", math=True),
                    _note(
                        f"ρ ∼ Normal({float(rho.loc):g}, {float(rho.scale):g}). "
                        f"Scale λ ∼ LogNormal({float(scale.loc):g}, {float(scale.scale):g}); "
                        f"shape k ∼ LogNormal({float(shape.loc):g}, {float(shape.scale):g}). "
                        "These priors come from the model configured above."
                    ),
                    _note(
                        "Storm regressors multiply each day's cumulative-hazard increment. "
                        "Scalar scale and shape priors create named sample sites; fixed "
                        "positive values create no site. Susceptibility is parameterized "
                        "on log-odds; its fixed value may be any finite number."
                    ),
                    _note(
                        "Regressors introduce beta or susceptibility_beta only when their "
                        "features are present. Receipts are allowed Monday–Friday, with no "
                        "receipt deadline. The propensity and timing are learned; calendar "
                        "rules are declared."
                    ),
                    _note(
                        "The same propagation function runs for initiation and receipt, "
                        "using different fitted stage models and upstream flows."
                    ),
                ],
                "extra_sources": [
                    _source(_weibull_log_increment),
                    _source(_timing_from_log_increment),
                ],
            },
            "convolution": {
                "kicker": "Shared mechanism · survival convolution",
                "title": "Spread each cohort forward, then add.",
                "main": (
                    "A cohort contributes its size × its event probability on each day. "
                    "Add those contributions to obtain the daily expectation."
                ),
                "body": (
                    "The forecaster samples paths rather than just passing those averages "
                    "onward. Initiation counts from one draw become that same draw's "
                    "receipt-stage arrivals."
                ),
                "equation": "K(s, t) = π × S(s, t⁻) × h(t − s, t)",
                "source": _source(SurvivalKernel.log_mass.fget),
                "more_title": "Waiting, calendar rules and propagation",
                "notes": [
                    _note("expected events(t) = Σₛ N(s) · K(s, t)", math=True),
                    _note(
                        "With parameters fixed, new homogeneous cohorts compose as "
                        "(K₀K₁)(s, t) = Σᵤ K₀(s, u)K₁(u, t). Storms, weekends and "
                        "cohort features make these calendar-aware kernels; one fixed "
                        "shifted curve is only a special case."
                    ),
                    _note(
                        "Orders already sold enter at their current age, conditioned on "
                        "their observed history, not restarted at age zero. Mass outside "
                        "the forecast window means not by then, not necessarily never."
                    ),
                    _note("P(no event yet) = (1 − π) + πS(age)", math=True),
                    _note(
                        "A family states the delay law and susceptibility. The shared core "
                        "handles exposure, closures, cure marginalization, dated cohorts "
                        "and integer counts. A custom NumPyro law joins through EventFamily."
                    ),
                    _note(
                        "The sources below show cumulative survival, the event-time "
                        "likelihood through numpyro_forecast.predict, and propagation of "
                        "actual parent dates. Private helpers may change."
                    ),
                ],
                "extra_sources": [
                    _source(exclusive_log_survival),
                    _source(_event_model),
                    _source(observation_kernel),
                    _source(_observe),
                    _source(FittedNetwork._forecast),
                ],
            },
        },
    }
