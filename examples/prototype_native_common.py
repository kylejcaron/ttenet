"""Throwaway fixture for exact calendar-axis NumPyro Forecast cure experiments.

Question: can native forecast drivers preserve the existing per-unit cure
likelihood, delayed entry and at-most-one-event paths without aggregating units?
Run the comparison with: uv run python examples/prototype_numpyro_native.py
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from jax import random
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro_forecast import draw_posterior, forecast

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


def prefix_supported(case, data):
    day = jnp.arange(data.shape[0])[:, None]
    age = day - jnp.asarray(case.origin)[None, :]
    eligible = (
        (age >= 0)
        & (age <= case.deadline)
        & (day >= jnp.asarray(case.entry)[None, :])
        & jnp.asarray(case.allowed[:, : data.shape[0]].T)
    )
    cells = (data == 0) | ((data == 1) & eligible)
    return jnp.all(cells) & jnp.all(data.sum(axis=0) <= 1)


def chain_prefix_supported(case, data):
    n = len(case.origin)
    initiation, receipt = data[:, :n], data[:, n:]
    receipt_cells = (receipt == 0) | (
        (receipt == 1) & jnp.asarray(case.allowed[:, : data.shape[0]].T)
    )
    return (
        prefix_supported(case, initiation)
        & jnp.all(receipt_cells)
        & jnp.all(receipt.sum(axis=0) <= 1)
        & jnp.all(receipt.cumsum(axis=0) <= initiation.cumsum(axis=0))
    )


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

    @property
    def covariates(self):
        return jnp.arange(len(self.calendar), dtype=jnp.float32)[:, None]

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

    def arrays(self, theta, duration=None):
        duration = len(self.calendar) if duration is None else duration
        p = parameters(theta)
        ages = jnp.arange(duration)[:, None] - jnp.asarray(self.origin)[None, :]
        logits = p.age_logits[jnp.clip(ages, 0, len(p.age_logits) - 1)]
        logits = logits + jnp.einsum("ntp,p->tn", jnp.asarray(self.features[:, :duration]), p.beta)
        eligible = (ages >= 0) & (ages <= self.deadline) & jnp.asarray(self.allowed[:, :duration].T)
        log_survival_day = jnp.where(eligible, jax.nn.log_sigmoid(-logits), 0.0)
        log_s_before = jnp.cumsum(log_survival_day, axis=0) - log_survival_day
        cure = p.cure_intercept + jnp.asarray(self.cure_features) @ p.cure_beta
        # Effective hazard conditional on remaining event-free; cure is marginalized.
        q = jax.nn.sigmoid(cure[None, :] + log_s_before) * jax.nn.sigmoid(logits)
        entered = jnp.arange(duration)[:, None] >= jnp.asarray(self.entry)[None, :]
        return jnp.where(eligible & entered, q, 0.0), eligible & entered

    def analytic_future(self, theta):
        q, _ = self.arrays(theta)
        q = np.asarray(q)[self.t_obs :]
        pending = ~self.data[: self.t_obs].any(axis=0)
        survival = np.concatenate([np.ones((1, q.shape[1])), np.cumprod(1 - q, axis=0)[:-1]])
        return q * survival * pending[None, :]


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
    q = np.asarray(case.arrays(jnp.asarray(TRUTH))[0])
    rng = np.random.default_rng(31)
    alive = np.ones(len(origin), dtype=bool)
    for t in day:
        event = alive & (rng.random(len(origin)) < q[t])
        case.data[t] = event
        alive &= ~event
    return case


def check_paths(case, draws):
    values = np.asarray(draws)
    assert values.ndim == 3 and values.shape[1:] == case.data[case.t_obs :].shape
    assert np.isin(values, [0, 1]).all(), "non-binary unit events"
    prefix = case.data[: case.t_obs].sum(axis=0)
    assert np.all(values.sum(axis=1) + prefix[None, :] <= 1), "repeat event"
    _, eligible = case.arrays(jnp.asarray(TRUTH))
    assert np.all(values[:, ~np.asarray(eligible)[case.t_obs :]] == 0), (
        "event on closed/ineligible day"
    )
    return {"shape": list(values.shape), "repeat_events": 0, "closed_day_events": 0}


def fit_and_forecast(model, case, steps=100, draws=128):
    x, y = case.covariates[: case.t_obs], jnp.asarray(case.data[: case.t_obs])
    guide = AutoNormal(model)
    svi = SVI(model, guide, numpyro.optim.Adam(0.025), Trace_ELBO())
    fitted = svi.run(random.PRNGKey(101), steps, x, y, progress_bar=False)
    assert np.isfinite(fitted.losses).all(), "non-finite ELBO"
    posterior = draw_posterior(random.PRNGKey(102), guide, fitted.params, draws)
    samples = forecast(random.PRNGKey(103), model, posterior, y, case.covariates)
    summary = check_paths(case, samples)
    summary.update(elbo_final=float(fitted.losses[-1]), posterior_sites=sorted(posterior))
    return summary
