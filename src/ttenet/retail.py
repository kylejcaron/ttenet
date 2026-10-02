"""The sale → initiation → receipt specialization of the cohort network."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .data import RetailHistory, prepare_history
from .families import FiniteTail, validate_family
from .forecast import ReturnForecast, expected_return_receipts
from .network import FittedNetwork, ForecastNetwork
from .processes import (
    CountNode,
    CountProcess,
    EventNode,
    EventProcess,
    _positive_integer,
)


@dataclass(frozen=True)
class RetailReturnModel:
    """Reusable retail configuration, backed by the same explicit network API.

    ``sales=None`` fits the return processes but requires an external
    ``SalesForecast`` at prediction time, including an explicit empty scenario
    when no new sales are expected. Receipt has no extra policy deadline.
    """

    sales: CountProcess | None = None
    initiation: EventProcess = field(default_factory=lambda: EventProcess(deadline_days=90))
    receipt: EventProcess = field(default_factory=EventProcess)
    shared_model: Callable | None = None

    def __post_init__(self):
        if not isinstance(self.initiation, EventProcess) or not isinstance(
            self.receipt, EventProcess
        ):
            raise TypeError("initiation and receipt must be EventProcess values")
        if self.receipt.deadline_days is not None:
            raise ValueError(
                "retail receipts have no deadline; use ForecastNetwork for other event policies"
            )
        # Constructing the graph validates the count/shared process configuration too.
        self.network

    @property
    def network(self):
        sales = CountNode("sales", self.sales)
        initiation = EventNode("initiations", sales, self.initiation, "initiation_date")
        receipt = EventNode("receipts", initiation, self.receipt, "receipt_date")
        return ForecastNetwork((sales, initiation, receipt), shared_model=self.shared_model)

    def fit(
        self,
        data,
        *,
        covariates=None,
        mode="joint",
        num_steps=500,
        num_samples=100,
        seed=0,
        learning_rate=0.02,
    ):
        """Fit a ``RetailData`` snapshot jointly, or with explicit modular inference."""
        fitted = self.network.fit(
            data,
            covariates=covariates,
            mode=mode,
            num_steps=num_steps,
            num_samples=num_samples,
            seed=seed,
            learning_rate=learning_rate,
        )
        history = prepare_history(
            fitted.data.units,
            as_of=fitted.data.as_of,
            policy_days=self.initiation.deadline_days,
        )
        return FittedRetailReturnModel(self, fitted, history)


@dataclass(frozen=True)
class FittedRetailReturnModel:
    """A fitted network plus retail-specific outstanding-population summaries."""

    model: RetailReturnModel
    network: FittedNetwork
    history: RetailHistory

    @property
    def initiation_fit(self):
        return self.network.stage_fits["initiations"]

    @property
    def receipt_fit(self):
        return self.network.stage_fits["receipts"]

    @property
    def losses(self):
        """One joint ELBO trace, or separate traces keyed by node in modular mode."""
        return self.network.losses

    def forecast(self, *, horizon, covariates=None, future_sales=None, seed=0):
        """Forecast daily returns and eventual receipts from the existing population.

        Event covariates cover future days through the latest still-eligible
        historical sale's initiation deadline, even when that exceeds ``horizon``.
        This extra coverage is needed for the eventual expectation, not for new
        sales' post-horizon returns. The unbounded receipt tail assumes continued
        positive exposure and reopening, unless all weekdays are explicitly closed.
        """
        _positive_integer("horizon", horizon)
        through = self.history.as_of + np.timedelta64(int(horizon), "D")
        policy = self.model.initiation.deadline_days
        family = self.model.initiation.family
        if family is not None:
            validate_family(family)
            if isinstance(family.tail, FiniteTail):
                last_age = family.tail.last_age
                policy = last_age if policy is None else min(policy, last_age)
        eligible = self.history.frame.loc[self.history.frame.eligible, "sale_date"]
        if policy is not None and len(eligible) and self.model.receipt.allowed_weekdays:
            deadline = np.asarray(eligible, dtype="datetime64[D]").max() + np.timedelta64(
                int(policy), "D"
            )
            through = max(through, deadline)
        result, context = self.network._forecast(
            horizon=horizon,
            covariates=covariates,
            future_sales=future_sales,
            seed=seed,
            through=through,
        )
        n = len(self.history.frame)
        initiation = context.features["initiations"]
        receipt = context.features["receipts"]
        if self.model.receipt.allowed_weekdays:
            uninitiated, open_returns = expected_return_receipts(
                self.history,
                self.initiation_fit,
                self.receipt_fit,
                calendar=context.calendar,
                initiation_features=initiation.features[:n],
                receipt_features=receipt.features[:n],
                initiation_susceptibility_features=initiation.susceptibility_features[:n],
                receipt_susceptibility_features=receipt.susceptibility_features[:n],
                initiation_allowed=initiation.allowed[:n],
                receipt_allowed=receipt.allowed[:n],
            )
            if not self.model.initiation.allowed_weekdays:
                uninitiated = np.zeros_like(uninitiated)
        else:
            uninitiated = open_returns = np.zeros(self.network.num_samples)
        draws = result.counts["receipts"].shape[0]
        uninitiated = np.broadcast_to(uninitiated, (draws,))
        open_returns = np.broadcast_to(open_returns, (draws,))
        return ReturnForecast(
            dates=result.dates,
            initiations=result.counts["initiations"],
            receipts=result.counts["receipts"],
            eligible=result.nodes["initiations"].eligible,
            open_returns=result.nodes["receipts"].pending,
            expected_existing_receipts=uninitiated + open_returns,
            expected_uninitiated_receipts=uninitiated,
            expected_open_receipts=open_returns,
            sales=result.sales,
        )
