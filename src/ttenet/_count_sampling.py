"""Integer allocation of homogeneous pools across event dates.

A pool of ``n`` units with normalized date probabilities ``p[t]`` and no-event
probability ``q`` is allocated by the chain of conditional binomials

    X[t] | X[<t] ~ Binomial(n - sum(X[<t]), p[t] / (p[t] + ... + p[T-1] + q)),

so work and memory scale with dates times pools, never with ``n``. Both
conditional sides come from log-space suffix sums, so neither is formed by
``1 - p`` cancellation. The float binomial variate is drawn on the smaller
side and the other side is completed by integer subtraction, which keeps
column totals exact for every pool the integer dtype holds.

Precision follows the JAX configuration where the draw is traced: int32
pools with float32 proposals by default, int64 pools with float64 proposals
inside ``jax.enable_x64(True)``. Nothing here changes global configuration.
Two limits of the float proposal are inherent: a pool above the float
dtype's exact-integer range (``2**24`` for float32, ``2**53`` for float64)
is rounded before the *drawn* side is sampled, and the drawn side itself
lives on the float lattice above that range. The completed side and the
column total remain exact integers.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, random


def _binomial(key, total, log_event, log_stay):
    """Binomial(total, exp(log_event)) drawn on the smaller side, completed in integers.

    ``log_event`` and ``log_stay`` are the two conditional log probabilities
    (both at most zero). Whichever is smaller is passed to the float sampler;
    the complement side is ``total`` minus that draw in the integer dtype.
    """
    small = log_event <= log_stay
    side = jnp.exp(jnp.where(small, log_event, log_stay))
    drawn = random.binomial(key, total.astype(side.dtype), side, dtype=side.dtype)
    drawn = drawn.astype(total.dtype)
    return jnp.where(small, drawn, total - drawn)


def allocate_counts(key, log_weights, total):
    """Counts ``[..., T, C]`` allocating ``total`` ``[..., C]`` under ``log_weights`` ``[..., T+1, C]``.

    The last weight row is the no-event category; its residual count is
    ``total - counts.sum(-2)`` and is not returned. Pools and weights must
    already share their leading axes. A negative pool, or a positive pool
    whose weights have no mass at all (or NaN mass), has no allocation: every
    date receives the visible sentinel ``-1``, so an impossible or undefined
    population is never reported as a plausible run of zero events. An
    empty pool draws zeros whatever its weights.
    """
    days = log_weights.shape[-2] - 1
    float_dtype = jnp.dtype(jax.dtypes.canonicalize_dtype(np.float64))
    weights = log_weights.astype(float_dtype)
    pool = jnp.maximum(total, 0)
    counts = jnp.zeros(pool.shape[:-1] + (days,) + pool.shape[-1:], pool.dtype)
    if days == 0:
        return counts
    suffix = lax.cumlogsumexp(weights, axis=weights.ndim - 2, reverse=True)

    def step(carry, day):
        remaining, key, counts = carry
        key, subkey = random.split(key)
        mass = lax.dynamic_index_in_dim(weights, day, axis=-2, keepdims=False)
        here = lax.dynamic_index_in_dim(suffix, day, axis=-2, keepdims=False)
        after = lax.dynamic_index_in_dim(suffix, day + 1, axis=-2, keepdims=False)
        open_mass = jnp.isfinite(here)
        log_event = jnp.where(open_mass, jnp.minimum(mass - here, 0.0), -jnp.inf)
        log_stay = jnp.where(open_mass, jnp.minimum(after - here, 0.0), 0.0)
        count = _binomial(subkey, remaining, log_event, log_stay)
        counts = lax.dynamic_update_index_in_dim(counts, count, day, axis=-2)
        return (remaining - count, key, counts), None

    (_, _, counts), _ = lax.scan(step, (pool, key, counts), jnp.arange(days))
    undefined = (total < 0) | ((total > 0) & ~jnp.isfinite(suffix[..., 0, :]))
    return jnp.where(undefined[..., None, :], -1, counts)
