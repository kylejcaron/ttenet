"""Custom-family stage fits produced by a forecast network replay standalone.

A network fits each custom event node at the regressor widths of its
observations. The extracted ``StageFit`` must carry those widths with it:
replayed on its own it applies the fitted law at the same widths and refuses
any other, whether the posterior came from one joint program with shared
values or from an independent modular block.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import pytest

import ttenet
from ttenet.event_times import log1mexp


def _regressor_law_model(inputs, shared):
    """Stay ``-exp(log_rate + sum(features))``; logit ``0.5 + sum(susceptibility_features)``.

    ``log_rate`` is the shared model's value under joint fitting and ``-2``
    without one; the law samples no local site of its own.
    """
    log_rate = -2.0 if shared is None else shared["log_rate"]
    stay = -jnp.exp(log_rate + inputs.features.sum(axis=-1))
    logits = 0.5 + inputs.susceptibility_features.sum(axis=-1)
    return ttenet.EventLaw(ttenet.TimingLaw(log1mexp(stay), stay), logits)


_regressor_law = ttenet.EventFamily(_regressor_law_model, ttenet.ProperTail())


def _shared_log_rate():
    return {"log_rate": numpyro.sample("log_rate", dist.Normal(-2.0, 0.5))}


def _network_fit(mode):
    """The ``initiations`` fit of a network fitted with two timing and one static regressor."""
    root = ttenet.CountNode("sales")
    child = ttenet.EventNode(
        "initiations", root, ttenet.EventProcess(deadline_days=3, family=_regressor_law)
    )
    model = ttenet.ForecastNetwork(
        [root, child], shared_model=_shared_log_rate if mode == "joint" else None
    )
    units = pd.DataFrame(
        {
            "item_id": range(6),
            "sale_date": ["2026-01-01"] * 6,
            "initiation_date": ["2026-01-01"] * 2 + ["2026-01-02"] * 2 + [None] * 2,
        }
    )
    data = ttenet.RetailData.from_units(units, as_of="2026-01-04", calendar=["2026-01-04"])
    rng = np.random.default_rng(mode == "joint")
    covariates = {
        "features": rng.normal(scale=0.5, size=(6, len(data.context_calendar), 2)),
        "susceptibility_features": rng.normal(size=(6, 1)),
    }
    fitted = model.fit(
        data, covariates={"initiations": covariates}, mode=mode, num_steps=20, num_samples=4, seed=3
    )
    return fitted.stage_fits["initiations"]


def _inputs(rng, p, q):
    return ttenet.TimingInputs(
        ages=jnp.array([[0, 2], [1, 3]]),
        features=jnp.asarray(rng.normal(size=(2, 2, p))),
        susceptibility_features=jnp.asarray(rng.normal(size=(2, q))),
    )


@pytest.mark.parametrize("mode", ["joint", "modular"])
@pytest.mark.parametrize("widths", [(0, 1), (3, 1), (2, 0), (2, 2)])
def test_network_custom_fit_refuses_standalone_replay_at_other_regressor_widths(mode, widths):
    # Eager, closed over by a traced replay, and carried through tracing as
    # a pytree argument (its shared draws become tracers): refused alike.
    fit = _network_fit(mode)
    inputs = _inputs(np.random.default_rng(1), *widths)
    with pytest.raises(ValueError):
        fit.timing(inputs, draw=0)
    with pytest.raises(ValueError):
        jax.jit(lambda inputs, draw: fit.timing(inputs, draw=draw))(inputs, 0)
    with pytest.raises(ValueError):
        jax.jit(lambda fit, inputs: fit.timing(inputs, draw=0))(fit, inputs)


@pytest.mark.parametrize("mode", ["joint", "modular"])
def test_network_custom_fit_replays_the_closed_form_law_at_its_fitted_widths(mode):
    fit = _network_fit(mode)
    inputs = _inputs(np.random.default_rng(2), 2, 1)
    laws = jax.jit(
        lambda fit, inputs: jax.vmap(lambda draw: fit.timing(inputs, draw=draw))(
            jnp.arange(fit.draws)
        )
    )(fit, inputs)
    if mode == "joint":
        log_rate = np.asarray(fit.shared["log_rate"])
        assert log_rate.shape == (4,) and len(np.unique(log_rate)) > 1
    else:
        assert fit.shared is None
        log_rate = np.full(4, -2.0)
    expected_stay = -np.exp(log_rate[:, None, None] + np.asarray(inputs.features).sum(axis=-1))
    expected_logits = 0.5 + np.asarray(inputs.susceptibility_features).sum(axis=-1)
    np.testing.assert_allclose(laws.timing.log_survival_step, expected_stay, rtol=1e-6)
    np.testing.assert_allclose(laws.susceptibility_logits, [expected_logits] * 4, rtol=1e-6)
