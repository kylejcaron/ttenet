"""Calendar-aware time-to-event and retail return forecasting."""

from .data import RetailHistory, expand_covariates, prepare_history
from .dataset import RetailData
from .dates import allowed_days, calendar_features, date_grid, elapsed_days, to_day
from .distributions import CohortEventTime, EventTime
from .event_times import (
    EventLaw,
    SurvivalKernel,
    TimingInputs,
    TimingLaw,
    survival_kernel,
    timing_from_log_masses,
    timing_from_log_survival,
)
from .families import (
    EventFamily,
    Family,
    FiniteTail,
    ProperTail,
    TailBehavior,
    UnknownTail,
    WeibullFamily,
    validate_family,
)
from .forecast import EventForecast, ReturnForecast, forecast_events, forecast_returns
from .integration import SalesForecast
from .models import (
    StageFit,
    StageObservations,
    fit_stage,
    make_event_observations,
    make_observations,
    predict_stage,
    sample_stage_parameters,
    stage_log_likelihood,
    stage_model,
)
from .network import FittedNetwork, ForecastNetwork, NetworkForecast
from .processes import CountNode, CountProcess, CureProcess, EventNode
from .retail import FittedRetailReturnModel, RetailReturnModel
from .survival import (
    StageParameters,
    conditional_susceptibility,
    stage_hazard,
    susceptibility,
)

__all__ = [
    "CohortEventTime",
    "CountNode",
    "CountProcess",
    "CureProcess",
    "EventFamily",
    "EventForecast",
    "EventLaw",
    "EventNode",
    "EventTime",
    "Family",
    "FiniteTail",
    "FittedNetwork",
    "ForecastNetwork",
    "NetworkForecast",
    "ProperTail",
    "RetailData",
    "SalesForecast",
    "FittedRetailReturnModel",
    "RetailHistory",
    "RetailReturnModel",
    "ReturnForecast",
    "StageFit",
    "StageObservations",
    "StageParameters",
    "SurvivalKernel",
    "TailBehavior",
    "TimingInputs",
    "TimingLaw",
    "UnknownTail",
    "WeibullFamily",
    "allowed_days",
    "calendar_features",
    "conditional_susceptibility",
    "date_grid",
    "elapsed_days",
    "expand_covariates",
    "fit_stage",
    "forecast_events",
    "forecast_returns",
    "make_event_observations",
    "make_observations",
    "predict_stage",
    "prepare_history",
    "sample_stage_parameters",
    "stage_hazard",
    "stage_log_likelihood",
    "stage_model",
    "susceptibility",
    "survival_kernel",
    "timing_from_log_masses",
    "timing_from_log_survival",
    "to_day",
    "validate_family",
]
