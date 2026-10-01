"""Joint NumPyro inference and posterior-paired propagation over dated cohort edges."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable
from uuid import uuid4

import jax.numpy as jnp
import numpy as np
import numpyro
import pandas as pd
from jax import random, tree_util
from numpyro import handlers
from numpyro.infer import Predictive

from .dataset import RetailData
from .dates import date_grid, to_day
from .event_times import default_timing
from .forecast import EventForecast, forecast_events
from .integration import SalesForecast
from .models import (
    StageFit,
    _fit_program,
    _named_posterior,
    _observe,
    _trajectories,
    _validated_law,
    make_event_observations,
    observation_kernel,
    sample_stage_parameters,
    timing_inputs,
)
from .processes import CountNode, EventNode, _positive_integer
from .survival import StageParameters

_FEATURE_KEYS = frozenset({"features", "cure_features", "allowed"})


@dataclass(frozen=True)
class _Features:
    features: np.ndarray
    cure_features: np.ndarray
    allowed: np.ndarray

    def kwargs(self):
        return {key: getattr(self, key) for key in _FEATURE_KEYS}


def _covariate_mapping(value, names):
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("covariates must map node names to their covariates")
    unknown = set(value) - set(names)
    if unknown:
        raise ValueError(f"unknown covariate nodes: {sorted(unknown)}")
    return dict(value)


def _resolve_features(process, provider, frame, calendar, *, widths=None, default_cure=None):
    value = (
        {} if provider is None else provider(frame, calendar) if callable(provider) else provider
    )
    if not isinstance(value, Mapping):
        raise TypeError("event covariates must be a mapping or a callable returning one")
    if set(value) - _FEATURE_KEYS:
        raise ValueError(f"unknown event covariate keys: {sorted(set(value) - _FEATURE_KEYS)}")
    n, t = len(frame), len(calendar)
    features = np.asarray(value.get("features", np.empty((n, t, 0))), dtype=np.float64)
    cure = np.asarray(
        value.get("cure_features", np.empty((n, 0)) if default_cure is None else default_cure),
        dtype=np.float64,
    )
    if features.ndim != 3 or features.shape[:2] != (n, t):
        raise ValueError(f"features must have shape ({n}, {t}, P)")
    if cure.ndim != 2 or cure.shape[0] != n:
        raise ValueError(f"cure_features must have shape ({n}, Q)")
    if not np.isfinite(features).all() or not np.isfinite(cure).all():
        raise ValueError("event covariates must be finite over the requested calendar")
    if widths is not None and (features.shape[-1], cure.shape[-1]) != widths:
        raise ValueError(
            "future covariates must preserve fitted feature widths; missing values are not imputed"
        )
    mask = np.asarray(value.get("allowed", True))
    if mask.dtype != bool:
        raise ValueError("allowed must contain boolean values")
    if mask.shape not in ((), (t,), (n, t)):
        raise ValueError(f"allowed must have shape ({t},) or ({n}, {t})")
    weekdays = pd.DatetimeIndex(calendar).dayofweek.to_numpy()
    allowed = np.broadcast_to(mask, (n, t)) & np.isin(weekdays, process.allowed_weekdays)[None, :]
    return _Features(features.copy(), cure.copy(), allowed)


def _count_covariates(value, duration):
    result = np.empty((duration, 0)) if value is None else np.asarray(value)
    if result.ndim < 2 or result.shape[-2] != duration:
        raise ValueError(f"sales covariates must have {duration} days on time axis -2")
    if not np.issubdtype(result.dtype, np.number) or not np.isfinite(result).all():
        raise ValueError("sales covariates must be finite numeric arrays")
    return jnp.asarray(result)


def _parameter_sites(nodes):
    return [
        f"{node.name}/__resolved_{name}"
        for node in nodes
        if node.process.family is None
        for name in StageParameters._fields
    ]


_SHARED_SITE = "shared/__resolved_"


def _shared_sites(count):
    return [f"{_SHARED_SITE}{index}" for index in range(count)]


def _record_shared(shared):
    """Record every leaf of the shared model's returned mapping as a resolved site."""
    for index, leaf in enumerate(tree_util.tree_leaves(shared)):
        numpyro.deterministic(f"{_SHARED_SITE}{index}", jnp.asarray(leaf))


def _default_parameters(node, observations, shared):
    """Sample or construct the default family's single-draw parameters and record them."""
    if node.process.parameter_model is None:
        parameters = sample_stage_parameters(observations, age_bins=node.process.age_bins)
    else:
        parameters = node.process.parameter_model(observations, shared)
    if not isinstance(parameters, StageParameters):
        raise TypeError("parameter_model must return StageParameters")
    shapes = (
        (node.process.age_bins,),
        (observations.features.shape[-1],),
        (),
        (observations.cure_features.shape[-1],),
    )
    for name, value, shape in zip(StageParameters._fields, parameters, shapes, strict=True):
        if np.shape(value) != shape:
            raise ValueError(f"{node.name}.{name} must have single-draw shape {shape}")
        numpyro.deterministic(f"__resolved_{name}", jnp.asarray(value))
    return parameters


def _event_model(node, observations, shared, *, likelihood=True):
    """One node's timing law under its scope, observed at the native site when fitting."""
    inputs = timing_inputs(observations)
    with handlers.scope(prefix=node.name):
        if node.process.family is None:
            parameters = _default_parameters(node, observations, shared)
            timing, logits = default_timing(parameters, inputs)
        else:
            timing, logits = _validated_law(node.process.family.model(inputs, shared), inputs)
        if likelihood:
            kernel = observation_kernel(timing, logits, observations)
            data, impossible = _trajectories(observations)
            _observe(kernel, inputs, data, massless=impossible)


@dataclass(frozen=True)
class _Prepared:
    data: RetailData
    columns: dict[str, str]
    covariates: dict[str, Any]
    sales_covariates: Any
    features: dict[str, _Features]
    observations: dict[str, Any]


def _resolved_stage_fit(node, posterior, params, losses, resolved, shared, num_samples):
    """Keep default parameters or the custom family's own named posterior and object."""
    if node.process.family is None:
        parameters = StageParameters(
            *[resolved[f"{node.name}/__resolved_{name}"] for name in StageParameters._fields]
        )
        return StageFit(parameters, losses, num_samples=num_samples)
    return StageFit(
        _named_posterior(posterior, params, prefix=f"{node.name}/", num_samples=num_samples),
        losses,
        family=node.process.family,
        shared=shared,
        num_samples=num_samples,
    )


@dataclass(frozen=True)
class ForecastNetwork:
    """One count root and an acyclic graph of single-source cure-capable events.

    Joint mode is one NumPyro program/posterior. Independent priors and fully
    observed parents still factorize. For cross-stage learning, ``shared_model()``
    samples shared latent values; a count model receives ``shared=values`` and
    each custom event ``parameter_model(observations, values)`` or timing family
    ``family.model(inputs, values)`` receives them too. Every event node
    observes its units at one native event-time site under its own scope; a
    custom family's fit keeps its named posterior and the resolved shared
    values per draw. Shared effects require joint mode. Branches are distinct,
    not competing events.
    """

    nodes: tuple[CountNode | EventNode, ...]
    shared_model: Callable | None = None
    _ordered: tuple[CountNode | EventNode, ...] = field(init=False, repr=False)

    def __post_init__(self):
        nodes = tuple(self.nodes)
        if any(not isinstance(node, (CountNode, EventNode)) for node in nodes):
            raise TypeError("nodes must be CountNode or EventNode values")
        names = [node.name for node in nodes]
        if any(
            not isinstance(name, str) or not name or "/" in name or name == "shared"
            for name in names
        ):
            raise ValueError(
                "node names must be nonempty, unique, contain no '/', and not be 'shared'"
            )
        if len(set(names)) != len(names):
            raise ValueError("node names must be unique")
        roots = [node for node in nodes if isinstance(node, CountNode)]
        if len(roots) != 1:
            raise ValueError("a forecast network requires exactly one count root")
        by_name = {node.name: node for node in nodes}
        remaining = [node for node in nodes if isinstance(node, EventNode)]
        for node in remaining:
            if node.source_name not in by_name:
                raise ValueError(f"unknown source {node.source_name!r} for {node.name!r}")
            if not isinstance(node.source, str) and by_name[node.source_name] is not node.source:
                raise ValueError("source node must be the same node object included in nodes")
        ordered = roots[:]
        seen = {roots[0].name}
        while remaining:
            ready = [node for node in remaining if node.source_name in seen]
            if not ready:
                raise ValueError("event network contains a cycle")
            ordered.extend(ready)
            seen.update(node.name for node in ready)
            remaining = [node for node in remaining if node not in ready]
        if self.shared_model is not None and not callable(self.shared_model):
            raise TypeError("shared_model must be callable")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "_ordered", tuple(ordered))

    @property
    def root(self):
        return self._ordered[0]

    @property
    def event_nodes(self):
        return self._ordered[1:]

    def _shared(self):
        if self.shared_model is None:
            return None
        with handlers.scope(prefix="shared"):
            return self.shared_model()

    def _count_model(self, covariates, data, shared):
        if self.root.process is None:
            return
        with handlers.trace() as trace:
            with handlers.scope(prefix=self.root.name):
                kwargs = {} if self.shared_model is None else {"shared": shared}
                self.root.process.model(covariates, data, **kwargs)
        observed = trace.get(f"{self.root.name}/obs")
        if data is not None and (
            observed is None
            or observed["type"] != "sample"
            or not observed["is_observed"]
            or np.shape(observed["value"]) != np.shape(data)
        ):
            raise ValueError("CountProcess must observe data at 'obs' with shape [time, group]")
        future = trace.get(f"{self.root.name}/forecast")
        if future is not None:
            numpyro.deterministic("forecast", future["value"])
        elif covariates.shape[-2] > (0 if data is None else data.shape[-2]):
            raise ValueError("CountProcess must record a 'forecast' site for future days")

    def _prepare(self, data, covariates):
        if not isinstance(data, RetailData):
            raise TypeError("fit requires RetailData.from_units(...)")
        data = data.copy()
        if self.root.event_column != "sale_date":
            raise ValueError(
                "RetailData count roots use sale_date; normalize root dates in the adapter"
            )
        values = _covariate_mapping(covariates, [node.name for node in self.nodes])
        columns = {self.root.name: self.root.event_column}
        for node in self.event_nodes:
            column = node.event_column or data.event_columns.get(node.name)
            if column is None or column not in data.units:
                raise ValueError(f"no observed date column for event node {node.name!r}")
            columns[node.name] = column
        if len(set(columns.values())) != len(columns):
            raise ValueError("each node must correspond to a distinct observed event column")
        features, observations = {}, {}
        for node in self.event_nodes:
            resolved = _resolve_features(
                node.process, values.get(node.name), data.units, data.context_calendar
            )
            features[node.name] = resolved
            observations[node.name] = make_event_observations(
                data.units,
                origin_column=columns[node.source_name],
                event_column=columns[node.name],
                as_of=data.as_of,
                calendar=data.context_calendar,
                deadline_days=node.process.deadline_days,
                entry_dates=data.entry_dates,
                **resolved.kwargs(),
            )
        sales_covariates = _count_covariates(values.get(self.root.name), len(data.calendar))
        return _Prepared(data, columns, values, sales_covariates, features, observations)

    def _shared_structure(self):
        """Pytree structure of the shared model's returned mapping, from one prior run."""
        if self.shared_model is None:
            return None
        return tree_util.tree_structure(handlers.seed(self._shared, random.PRNGKey(0))())

    def _programs(self, prepared, selected):
        events = [node for node in self.event_nodes if node.name in selected]
        record_shared = any(node.process.family is not None for node in events)

        def program():
            shared = self._shared()
            if self.root.name in selected:
                self._count_model(
                    prepared.sales_covariates, jnp.asarray(prepared.data.sales), shared
                )
            for node in events:
                _event_model(node, prepared.observations[node.name], shared)

        def resolver():
            shared = self._shared()
            if record_shared:
                _record_shared(shared)
            for node in events:
                _event_model(node, prepared.observations[node.name], shared, likelihood=False)

        return program, resolver, events

    def numpyro_model(self, data, *, covariates=None):
        """Return the ordinary zero-argument joint model, usable directly with NUTS/MCMC."""
        prepared = self._prepare(data, covariates)
        return self._programs(prepared, {node.name for node in self.nodes})[0]

    def _fit_selection(self, prepared, selected, structure, **options):
        """Fit one joint/modular block and resolve its real local/shared draw values."""
        program, resolver, events = self._programs(prepared, selected)
        custom = any(node.process.family is not None for node in events)
        record = custom and structure is not None
        shared_sites = _shared_sites(structure.num_leaves) if record else []
        post, params, loss, resolved = _fit_program(
            program, resolver, _parameter_sites(events) + shared_sites, **options
        )
        shared = structure.unflatten([resolved[name] for name in shared_sites]) if record else None
        stages = {
            node.name: _resolved_stage_fit(
                node, post, params, loss, resolved, shared, options["num_samples"]
            )
            for node in events
        }
        return post, params, loss, stages

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
        """Fit one joint AutoNormal guide, or explicitly independent modular guides.

        Covariates map node names to arrays (count root) or mappings/callables
        returning features, cure_features and allowed masks (event nodes).
        Event fit providers receive the original-origin context calendar, not
        just the root sales observation window. No historical prefix is imputed.
        """
        if mode not in ("joint", "modular"):
            raise ValueError("mode must be 'joint' or 'modular'")
        if mode == "modular" and self.shared_model is not None:
            raise ValueError("shared effects require joint inference")
        for name, value in (("num_steps", num_steps), ("num_samples", num_samples)):
            _positive_integer(name, value)
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        prepared = self._prepare(data, covariates)
        selections = (
            [("joint", {node.name for node in self.nodes})]
            if mode == "joint"
            else [
                (node.name, {node.name})
                for node in self._ordered
                if isinstance(node, EventNode) or node.process is not None
            ]
        )
        structure = self._shared_structure()
        posterior, params, stage_fits, losses = {}, {}, {}, {}
        for index, (label, selected) in enumerate(selections):
            post, model_params, loss, stages = self._fit_selection(
                prepared,
                selected,
                structure,
                num_steps=num_steps,
                num_samples=num_samples,
                seed=seed + index,
                learning_rate=learning_rate,
            )
            posterior.update(post)
            params.update(model_params)
            losses[label] = loss
            stage_fits.update(stages)
        return FittedNetwork(
            self,
            prepared,
            mode,
            posterior,
            params,
            stage_fits,
            losses,
            num_samples,
            uuid4().hex,
        )


@dataclass(frozen=True)
class NetworkForecast:
    """Paired integer trajectories; event arrays also retain root-cohort lineage."""

    dates: np.ndarray
    sales: SalesForecast
    counts: Mapping[str, np.ndarray]
    nodes: Mapping[str, EventForecast]
    cohorts: pd.DataFrame


@dataclass(frozen=True)
class _ForecastContext:
    calendar: np.ndarray
    features: Mapping[str, _Features]


@dataclass(frozen=True)
class FittedNetwork:
    """An independent data snapshot and posterior; nested mappings are read-only by convention."""

    model: ForecastNetwork
    _prepared: _Prepared
    mode: str
    posterior: Mapping[str, Any]
    params: Mapping[str, Any]
    stage_fits: Mapping[str, StageFit]
    losses: Mapping[str, Any]
    num_samples: int
    posterior_id: str

    @property
    def data(self):
        return self._prepared.data

    def _sales(self, horizon, values, future_sales, key):
        if future_sales is not None:
            result = (
                future_sales
                if isinstance(future_sales, SalesForecast)
                else SalesForecast.from_frame(future_sales)
            )
            if result.posterior_id == self.posterior_id:
                if result.counts.shape[0] != self.num_samples:
                    raise ValueError("joint sales must contain every fitted posterior draw")
                lookup = {label: i for i, label in enumerate(result.draw_ids)}
                if set(lookup) != set(range(self.num_samples)):
                    raise ValueError(
                        "joint sales draw_ids must identify the fitted posterior draws"
                    )
                order = [lookup[i] for i in range(self.num_samples)]
                if order != list(range(self.num_samples)):
                    result = SalesForecast(
                        result.cohorts,
                        result.counts[order],
                        posterior_id=self.posterior_id,
                    )
            return result
        if self.model.root.process is None:
            raise ValueError(
                "future sales require a CountProcess or an explicit SalesForecast scenario"
            )
        root = self.model.root.name
        if root not in values and self._prepared.sales_covariates.shape[-1] != 0:
            raise ValueError("future sales covariates are required; unknown values are not imputed")
        future = _count_covariates(values.get(root), horizon)
        past = self._prepared.sales_covariates
        if future.shape[:-2] != past.shape[:-2] or future.shape[-1] != past.shape[-1]:
            raise ValueError("future sales covariates must preserve fitted batch and feature axes")
        covariates = jnp.concatenate([past, future], axis=-2)

        def count_program(covariates, data):
            self.model._count_model(covariates, data, self.model._shared())

        samples = Predictive(
            count_program,
            posterior_samples=dict(self.posterior) or None,
            params=dict(self.params),
            num_samples=self.num_samples,
            return_sites=["forecast"],
            parallel=True,
        )(key, covariates, jnp.asarray(self.data.sales))["forecast"]
        dates = date_grid(
            self.data.as_of + np.timedelta64(1, "D"), self.data.as_of + np.timedelta64(horizon, "D")
        )
        used_prefixes = {
            item_id.rpartition("_")[0]
            for item_id in self.data.units.item_id
            if isinstance(item_id, str)
        }
        prefix, suffix = "future", 0
        while prefix in used_prefixes:
            suffix += 1
            prefix = f"future{suffix}"
        return SalesForecast.from_numpyro_forecast(
            samples,
            dates,
            groups=self.data.groups,
            prefix=prefix,
            posterior_id=self.posterior_id,
        )

    def _future_features(self, node, values, frame, future_days):
        old = self._prepared.features[node.name]
        provider = values.get(node.name)
        if node.name not in values:
            historical = self._prepared.covariates.get(node.name)
            if callable(historical):
                provider = historical
            elif historical:
                raise ValueError(f"future covariates for {node.name!r} are required")
        n = len(self.data.units)
        future = _resolve_features(
            node.process,
            provider,
            frame,
            future_days,
            widths=(old.features.shape[-1], old.cure_features.shape[-1]),
            default_cure=old.cure_features if len(frame) == n else None,
        )
        if not np.array_equal(future.cure_features[:n], old.cure_features):
            raise ValueError(
                "historical cure_features are static at stage entry and cannot change in a scenario"
            )
        prefix = old.features.shape[1]
        features = np.empty((len(frame), prefix + len(future_days), old.features.shape[-1]))
        features[:n, :prefix] = old.features
        features[n:, :prefix] = 0  # These cohorts do not exist yet; padding is never exposure.
        features[:, prefix:] = future.features
        allowed = np.empty((len(frame), prefix + len(future_days)), dtype=bool)
        allowed[:n, :prefix] = old.allowed
        allowed[n:, :prefix] = False
        allowed[:, prefix:] = future.allowed
        return _Features(features, future.cure_features, allowed)

    def _forecast(self, *, horizon, covariates=None, future_sales=None, seed=0, through=None):
        _positive_integer("horizon", horizon)
        values = _covariate_mapping(covariates, [node.name for node in self.model.nodes])
        keys = random.split(random.PRNGKey(seed), len(self.model.event_nodes) + 1)
        sales = self._sales(horizon, values, future_sales, keys[0])
        d_sales = sales.counts.shape[0]
        if d_sales != self.num_samples and 1 not in (d_sales, self.num_samples):
            raise ValueError(
                "sales and posterior draws must align: equal counts, or one deterministic draw"
            )
        draws = max(d_sales, self.num_samples)
        dates = date_grid(
            self.data.as_of + np.timedelta64(1, "D"), self.data.as_of + np.timedelta64(horizon, "D")
        )
        sale_dates = np.asarray(to_day(sales.cohorts.sale_date))
        if ((sale_dates < dates[0]) | (sale_dates > dates[-1])).any():
            raise ValueError("future sales dates must fall within the requested horizon")
        if set(sales.cohorts.item_id) & set(self.data.units.item_id):
            raise ValueError("future sales cohort identifiers overlap historical item identifiers")
        frame = pd.concat([self.data.units, sales.cohorts], ignore_index=True)
        n = len(self.data.units)
        arrivals = np.zeros((draws, horizon, len(frame)), dtype=np.int64)
        slots = (sale_dates - dates[0]).astype(int)
        arrivals[:, slots, np.arange(n, len(frame))] = sales.counts
        counts = {self.model.root.name: arrivals.sum(axis=-1)}
        flows = {self.model.root.name: arrivals}
        results, features = {}, {}
        end = dates[-1] if through is None else max(dates[-1], np.datetime64(through, "D"))
        future_days = date_grid(dates[0], end)
        calendar = np.concatenate([self.data.context_calendar, future_days])
        for index, node in enumerate(self.model.event_nodes):
            resolved = self._future_features(node, values, frame, future_days)
            features[node.name] = resolved
            origins = np.full(len(frame), np.datetime64("NaT", "D"))
            observed = origins.copy()
            origins[:n] = to_day(self.data.units[self._prepared.columns[node.source_name]])
            observed[:n] = to_day(self.data.units[self._prepared.columns[node.name]])
            result = forecast_events(
                self.stage_fits[node.name],
                origins=origins,
                observed=observed,
                arrivals=flows[node.source_name],
                calendar=calendar,
                as_of=self.data.as_of,
                horizon=horizon,
                deadline_days=node.process.deadline_days,
                seed=int(random.bits(keys[index + 1], dtype=jnp.uint32)),
                **resolved.kwargs(),
            )
            results[node.name] = result
            flows[node.name] = result.events
            counts[node.name] = result.events.sum(axis=-1)
        return NetworkForecast(dates, sales, counts, results, frame), _ForecastContext(
            calendar, features
        )

    def forecast(self, *, horizon, covariates=None, future_sales=None, seed=0):
        """Propagate one joint draw through every edge; an external source replaces generation.

        Event providers receive future dates and historical units followed by new
        root cohorts. Cached historical exposures are retained. External stochastic
        sources need the same draw count (one draw broadcasts); pairing is by row,
        not a claim of joint conditioning. Sources generated by this fit carry draw
        identity, so a reordered same-posterior scenario is paired by draw label.
        """
        return self._forecast(
            horizon=horizon, covariates=covariates, future_sales=future_sales, seed=seed
        )[0]
