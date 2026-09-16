"""Calendar-aware time-to-event and retail return forecasting."""

from .data import RetailHistory, expand_covariates, prepare_history
from .dates import allowed_days, calendar_features, date_grid, elapsed_days, to_day
from .forecast import ReturnForecast, forecast_returns
from .integration import sales_cohorts
from .models import (
    StageFit,
    StageObservations,
    fit_stage,
    make_observations,
    stage_log_likelihood,
    stage_model,
)
from .retail import FittedRetailReturnModel, RetailReturnModel
from .survival import (
    StageParameters,
    conditional_susceptibility,
    stage_hazard,
    susceptibility,
)

__all__ = [
    "FittedRetailReturnModel",
    "RetailHistory",
    "RetailReturnModel",
    "ReturnForecast",
    "StageFit",
    "StageObservations",
    "StageParameters",
    "allowed_days",
    "calendar_features",
    "conditional_susceptibility",
    "date_grid",
    "elapsed_days",
    "expand_covariates",
    "fit_stage",
    "forecast_returns",
    "make_observations",
    "prepare_history",
    "sales_cohorts",
    "stage_hazard",
    "stage_log_likelihood",
    "stage_model",
    "susceptibility",
    "to_day",
]
