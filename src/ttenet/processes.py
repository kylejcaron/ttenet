"""Declarative processes and dated cohort edges for a NumPyro network."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable

import numpy as np

_TAIL_KINDS = ("proper", "finite", "unknown")


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def tail_behavior(event_time_model) -> dict:
    """Validated static tail metadata declared by a timing-family factory.

    A factory declares ``tail_behavior`` as ``{"kind": "proper"}`` (event
    probability tends to one under continuing exposure), ``{"kind":
    "finite", "last_age": a}`` (fully specified support through age ``a``) or
    ``{"kind": "unknown"}`` (an unresolved residual category that cannot
    support eventual expectations). Returns a plain copy; refuses anything
    else, so a family's tail is never silently assumed.
    """
    if not callable(event_time_model):
        raise TypeError("event_time_model must be callable")
    declared = getattr(event_time_model, "tail_behavior", None)
    if not isinstance(declared, Mapping) or declared.get("kind") not in _TAIL_KINDS:
        raise ValueError(
            "event_time_model must declare tail_behavior as {'kind': 'proper'}, "
            "{'kind': 'finite', 'last_age': a} or {'kind': 'unknown'}"
        )
    kind = declared["kind"]
    expected = {"kind", "last_age"} if kind == "finite" else {"kind"}
    if set(declared) != expected:
        raise ValueError(f"tail_behavior {kind!r} takes exactly the keys {sorted(expected)}")
    if kind == "finite":
        last_age = declared["last_age"]
        if (
            isinstance(last_age, bool)
            or not isinstance(last_age, (int, np.integer))
            or last_age < 0
        ):
            raise ValueError("tail_behavior 'finite' needs a nonnegative integer last_age")
        return {"kind": "finite", "last_age": int(last_age)}
    return {"kind": kind}


@dataclass(frozen=True)
class CountProcess:
    """A count model following ``numpyro_forecast``'s ``(covariates, data)`` protocol.

    The model observes ``data`` at ``obs`` and records a future ``forecast``
    site, normally using ``Horizon`` and ``predict``. In a network with a
    ``shared_model``, it additionally receives the keyword ``shared``.
    """

    model: Callable

    def __post_init__(self):
        if not callable(self.model):
            raise TypeError("CountProcess.model must be callable")


@dataclass(frozen=True)
class CureProcess:
    """One event per source unit, with its own age clock and possible non-occurrence.

    A deadline is inclusive and relative to the immediate source event, not
    necessarily the root sale. The default family is the regularized
    random-walk age baseline of ``stage_model``; ``parameter_model(observations,
    shared)`` may instead sample custom NumPyro priors and must return
    ``StageParameters`` for that same family. ``event_time_model(inputs,
    shared)`` selects a different timing family altogether: it receives
    ``TimingInputs`` (ages and regressors only), samples its own named
    NumPyro sites, returns ``(TimingLaw, susceptibility_logits)`` and
    declares static ``tail_behavior`` (see :func:`tail_behavior`). Exposure,
    closures, cure marginalization and entry conditioning stay in the shared
    core. The two callbacks are alternatives, never combined.
    """

    age_bins: int = 30
    deadline_days: int | None = None
    allowed_weekdays: tuple[int, ...] = tuple(range(7))
    parameter_model: Callable | None = None
    event_time_model: Callable | None = None

    def __post_init__(self):
        _positive_integer("age_bins", self.age_bins)
        if self.deadline_days is not None and (
            isinstance(self.deadline_days, bool)
            or not isinstance(self.deadline_days, (int, np.integer))
            or self.deadline_days < 0
        ):
            raise ValueError("deadline_days must be a nonnegative integer or None")
        weekdays = tuple(self.allowed_weekdays)
        if any(
            isinstance(day, bool) or not isinstance(day, (int, np.integer)) or not 0 <= day <= 6
            for day in weekdays
        ):
            raise ValueError("allowed_weekdays must contain integers from 0 to 6")
        object.__setattr__(self, "allowed_weekdays", tuple(dict.fromkeys(weekdays)))
        if self.parameter_model is not None and not callable(self.parameter_model):
            raise TypeError("parameter_model must be callable")
        if self.event_time_model is not None:
            if self.parameter_model is not None:
                raise ValueError(
                    "parameter_model and event_time_model are ambiguous together: a stage "
                    "uses either the default family's prior or a custom timing family"
                )
            tail_behavior(self.event_time_model)


@dataclass(frozen=True)
class CountNode:
    """The network's dated root population; no process means external sales only."""

    name: str
    process: CountProcess | None = None
    event_column: str = "sale_date"

    def __post_init__(self):
        if self.process is not None and not isinstance(self.process, CountProcess):
            raise TypeError("a CountNode process must be a CountProcess or None")


@dataclass(frozen=True)
class EventNode:
    """A cure-capable child event, preserving its source's root cohort identity.

    Multiple children of one source are distinct events, not competing risks.
    ``event_column`` defaults to the node's entry in ``RetailData.event_columns``.
    """

    name: str
    source: CountNode | EventNode | str
    process: CureProcess
    event_column: str | None = None

    def __post_init__(self):
        if not isinstance(self.process, CureProcess):
            raise TypeError("an EventNode process must be a CureProcess")

    @property
    def source_name(self):
        return self.source if isinstance(self.source, str) else self.source.name
