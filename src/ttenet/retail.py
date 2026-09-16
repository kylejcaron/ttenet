"""Reusable configuration and independent fitted retail return forecasts.

These classes compose the existing history adapters, stage fits, and forecast
kernel without changing their numerical behavior. A pure feature builder
receives a frame and normalized UTC calendar, returning the documented
``initiation_*``/``receipt_*`` feature and closure keywords. Definitions and
training-derived scaling must remain fixed across fitting and forecasting.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from .data import RetailHistory, prepare_history
from .dates import to_day
from .forecast import ReturnForecast, forecast_returns
from .models import StageFit, fit_stage, make_observations

FeatureBuilder = Callable[[pd.DataFrame, np.ndarray], Mapping[str, Any]]

_FEATURE_KEYS = frozenset(
    {
        "initiation_features",
        "receipt_features",
        "initiation_cure_features",
        "receipt_cure_features",
        "initiation_allowed",
        "receipt_allowed",
    }
)


def _build_feature_mapping(
    feature_builder: Optional[FeatureBuilder], frame: pd.DataFrame, calendar: Any
) -> dict:
    """Call ``feature_builder`` once (if supplied) and validate its keys.

    With no builder, both stages get zero-width regressors and an
    all-allowed calendar mask (the ``make_observations``/``forecast_returns``
    defaults). A non-mapping result or any key outside the six documented
    ``initiation_*``/``receipt_*`` names is an error -- a misspelled key
    must never silently fall back to "no closures".
    """
    if feature_builder is None:
        return {}
    result = feature_builder(frame, calendar)
    if not isinstance(result, Mapping):
        raise TypeError("feature_builder must return a mapping")
    unknown = sorted(set(result) - _FEATURE_KEYS)
    if unknown:
        raise ValueError(f"feature_builder returned unknown keys: {unknown}")
    return dict(result)


def _stage_observation_kwargs(features: dict, stage: str) -> dict:
    return {
        "features": features.get(f"{stage}_features"),
        "cure_features": features.get(f"{stage}_cure_features"),
        "allowed": features.get(f"{stage}_allowed"),
    }


@dataclass(frozen=True)
class RetailReturnModel:
    """Reusable retail return model configuration.

    Owns only configuration -- ``policy_days`` (the initiation deadline),
    ``age_bins`` (shared by both stages' age baselines), and an optional
    ``feature_builder``. It never holds fit state: calling :meth:`fit`
    repeatedly on the same instance returns independent
    :class:`FittedRetailReturnModel` results and never mutates a previous
    one. The lower-level stage functions (``make_observations``,
    ``fit_stage``) remain directly usable for configurations this wrapper
    does not expose.
    """

    policy_days: int = 90
    age_bins: int = 30
    feature_builder: Optional[FeatureBuilder] = None

    def _resolve_history(self, data: Any, *, as_of: Any, layout: str) -> RetailHistory:
        if isinstance(data, RetailHistory):
            if as_of is not None:
                raise ValueError("as_of must not be supplied when data is already a RetailHistory")
            if layout != "tabular":
                raise ValueError(
                    "layout must not be overridden when data is already a RetailHistory"
                )
            if data.policy_days != self.policy_days:
                raise ValueError(
                    f"RetailHistory.policy_days ({data.policy_days}) does not match "
                    f"RetailReturnModel.policy_days ({self.policy_days})"
                )
            # Copy so later caller edits to the original history's frame can
            # never change what this fit was actually trained on.
            return RetailHistory(
                frame=data.frame.copy(), as_of=data.as_of, policy_days=data.policy_days
            )
        if as_of is None:
            raise ValueError("as_of is required when fitting from raw data")
        return prepare_history(data, as_of=as_of, layout=layout, policy_days=self.policy_days)

    def fit(
        self,
        data: Any,
        *,
        calendar: Any,
        as_of: Any = None,
        layout: str = "tabular",
        num_steps: int = 500,
        num_samples: int = 100,
        seed: int = 0,
        learning_rate: float = 0.02,
    ) -> "FittedRetailReturnModel":
        """Fit both stages and return an independent :class:`FittedRetailReturnModel`.

        ``data`` is either raw input for ``prepare_history`` (in which case
        ``as_of`` is required and ``layout`` selects the adapter), or an
        existing :class:`RetailHistory` (in which case ``as_of`` must be
        omitted, ``layout`` must stay ``"tabular"``, and its
        ``policy_days`` must match this model) that is copied before use.
        The optional ``feature_builder`` is called once with the resulting
        canonical history frame and ``calendar``; its output is split into
        the matching ``initiation``/``receipt`` ``features``/
        ``cure_features``/``allowed`` arguments of ``make_observations``.
        The receipt stage uses ``seed + 1`` for a separate random seed.
        """
        history = self._resolve_history(data, as_of=as_of, layout=layout)
        calendar = np.asarray(to_day(calendar)).reshape(-1)
        features = _build_feature_mapping(self.feature_builder, history.frame, calendar)

        initiation_observations = make_observations(
            history, "initiation", calendar, **_stage_observation_kwargs(features, "initiation")
        )
        receipt_observations = make_observations(
            history, "receipt", calendar, **_stage_observation_kwargs(features, "receipt")
        )

        initiation_fit = fit_stage(
            initiation_observations,
            age_bins=self.age_bins,
            num_steps=num_steps,
            num_samples=num_samples,
            seed=seed,
            learning_rate=learning_rate,
        )
        receipt_fit = fit_stage(
            receipt_observations,
            age_bins=self.age_bins,
            num_steps=num_steps,
            num_samples=num_samples,
            seed=seed + 1,
            learning_rate=learning_rate,
        )
        return FittedRetailReturnModel(
            model=self, history=history, initiation_fit=initiation_fit, receipt_fit=receipt_fit
        )


@dataclass(frozen=True)
class FittedRetailReturnModel:
    """An independent fit: configuration, history, and both stages' posteriors.

    ``model`` and ``history`` are the configuration and (copied) canonical
    history this fit was produced from; ``initiation_fit``/``receipt_fit``
    are the ``StageFit`` posterior draws for each stage. Nested arrays and
    the ``history`` frame are not deeply frozen -- callers must treat them
    as read-only, since :meth:`forecast` may be called repeatedly against
    the same fit.
    """

    model: RetailReturnModel
    history: RetailHistory
    initiation_fit: StageFit
    receipt_fit: StageFit

    def forecast(
        self,
        *,
        calendar: Any,
        horizon: int,
        future_sales: Optional[pd.DataFrame] = None,
        future_counts: Any = None,
        seed: int = 0,
    ) -> ReturnForecast:
        """Simulate forward from this fit and return the existing ``ReturnForecast``.

        The optional ``feature_builder`` is called once with the historical
        units followed by ``future_sales`` rows (combined via
        ``pandas.concat(..., ignore_index=True)`` when future sales are
        supplied, preserving row order) and ``calendar``; its output is
        passed straight through to ``forecast_returns`` under the matching
        ``initiation_*``/``receipt_*`` keyword. Calendar dates are normalized
        to UTC days; future sales counts and covariate scenarios remain
        caller-supplied. No future weather or sales scenario is invented.
        """
        calendar = np.asarray(to_day(calendar)).reshape(-1)
        frame = self.history.frame
        if future_sales is not None and len(future_sales):
            frame = pd.concat([frame, future_sales], ignore_index=True)
        features = _build_feature_mapping(self.model.feature_builder, frame, calendar)

        return forecast_returns(
            self.history,
            self.initiation_fit.parameters,
            self.receipt_fit.parameters,
            calendar=calendar,
            horizon=horizon,
            future_sales=future_sales,
            future_counts=future_counts,
            initiation_features=features.get("initiation_features"),
            receipt_features=features.get("receipt_features"),
            initiation_cure_features=features.get("initiation_cure_features"),
            receipt_cure_features=features.get("receipt_cure_features"),
            initiation_allowed=features.get("initiation_allowed"),
            receipt_allowed=features.get("receipt_allowed"),
            seed=seed,
        )
