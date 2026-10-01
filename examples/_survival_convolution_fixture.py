"""Synthetic ledger and structural path checks for the production convolution demo."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from jax import random

from ttenet import EventTime, TimingInputs, survival_kernel
from ttenet.event_times import default_timing
from ttenet.models import make_event_observations, stage_log_likelihood
from ttenet.survival import StageParameters

TRUTH = np.array([-0.2, -0.8, 0.25, -0.35, 0.55])


def parameters(theta):
    return StageParameters(
        jnp.array([-2.2, -1.8, -1.4, -1.1, -0.9]) + theta[0],
        theta[1:3],
        theta[3],
        theta[4:5],
    )


def sample_theta():
    return numpyro.sample("theta", dist.Normal(jnp.zeros(5), 1.0).to_event(1))


@dataclass
class CalendarCase:
    calendar: np.ndarray
    origin: np.ndarray
    entry: np.ndarray
    deadline: int
    features: np.ndarray  # [unit, day, feature]
    cure_features: np.ndarray
    allowed: np.ndarray  # [unit, day]
    data: np.ndarray  # [day, unit], full truth; only prefix reaches fitting
    t_obs: int

    def reference(self, theta, stop=None):
        stop = self.t_obs if stop is None else stop
        event = np.where(self.data[:stop].any(axis=0), self.data[:stop].argmax(axis=0), -1)
        frame = pd.DataFrame(
            {
                "origin": self.calendar[self.origin],
                "event": [self.calendar[i] if i >= 0 else np.datetime64("NaT") for i in event],
            }
        )
        obs = make_event_observations(
            frame,
            origin_column="origin",
            event_column="event",
            as_of=self.calendar[stop - 1],
            calendar=self.calendar[:stop],
            deadline_days=self.deadline,
            entry_dates=self.calendar[self.entry],
            features=self.features[:, :stop],
            cure_features=self.cure_features,
            allowed=self.allowed[:, :stop],
        )
        return stage_log_likelihood(parameters(theta), obs)

    def exposure(self):
        ages = np.arange(len(self.calendar))[None, :] - self.origin[:, None]
        return (
            (ages >= 0)
            & (ages <= self.deadline)
            & (np.arange(len(self.calendar))[None, :] >= self.entry[:, None])
        ).T


def make_case():
    calendar = np.arange(np.datetime64("2026-01-01"), np.datetime64("2026-02-12"))
    origin = np.array([0, 0, 0, 4, 4, 4, 12, 12, 20, 20, 28, 28])
    entry = origin.copy()
    entry[:2] = 5  # selected survivors, original age clocks remain unchanged
    product = np.array([0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1])
    day = np.arange(len(calendar))
    weekday = pd.DatetimeIndex(calendar).dayofweek.to_numpy()
    weather = ((day >= 14) & (day <= 17)) | ((day >= 29) & (day <= 31))
    features = np.stack(
        [
            np.broadcast_to(weather, (len(origin), len(day))),
            np.broadcast_to(product[:, None], (len(origin), len(day))),
        ],
        axis=-1,
    ).astype(float)
    allowed = np.broadcast_to(weekday < 5, (len(origin), len(day))).copy()
    allowed[7, 18:21] = False  # genuinely unit-specific calendar closure
    case = CalendarCase(
        calendar,
        origin,
        entry,
        18,
        features,
        product[:, None].astype(float),
        allowed,
        np.zeros((len(day), len(origin)), dtype=np.int32),
        24,
    )
    ages = np.arange(len(day))[:, None] - origin[None, :]
    valid = (ages >= 0) & (ages <= case.deadline)
    pre_entry = valid & (day[:, None] < entry[None, :])
    timing, logits = default_timing(
        parameters(jnp.asarray(TRUTH)),
        TimingInputs(ages, features.transpose(1, 0, 2), case.cure_features),
    )
    law = EventTime.from_kernel(
        survival_kernel(
            timing,
            logits,
            allowed=allowed.T,
            exposure=valid & ~pre_entry,
            pre_entry=pre_entry,
        )
    )
    case.data = np.array(law.sample(random.key(31)), copy=True)
    return case


def check_paths(case, draws):
    values = np.asarray(draws)
    assert values.ndim == 3 and values.shape[1:] == case.data[case.t_obs :].shape
    assert np.isin(values, [0, 1]).all(), "non-binary unit events"
    prefix = case.data[: case.t_obs].sum(axis=0)
    assert np.all(values.sum(axis=1) + prefix[None, :] <= 1), "repeat event"
    eligible = case.exposure() & case.allowed.T
    assert np.all(values[:, ~eligible[case.t_obs :]] == 0), "event on closed/ineligible day"
    return {"shape": list(values.shape), "repeat_events": 0, "closed_day_events": 0}
