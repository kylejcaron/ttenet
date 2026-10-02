"""Integer allocation of homogeneous pools across event dates.

A pool of ``n`` units with normalized date probabilities ``p[t]`` and no-event
probability ``q`` follows ``Multinomial(n, (p[0..T-1], q))``. The allocation
factorizes it over a balanced binary tree whose ``K = T + 1`` leaves are the
categories ``[no-event, date 0, ..., date T-1]``: every internal node holds
the log mass of its subtree, the root holds the pool, and each node splits
its count into its left subtree with ``Binomial(count, m_left / (m_left +
m_right))``, the right subtree receiving the integer difference. A
multinomial restricted to a subset of categories, given the subset total,
is the multinomial with renormalized probabilities, so the tree is the exact
law whatever its shape. All nodes of one level split in one vectorized
call, so a window of ``T`` dates costs ``ceil(log2(T + 1))`` sequential
binomial rounds instead of ``T``; work and memory scale with dates times
pools, never with ``n``.

Split probabilities stay in log space: with both children finite,
``log p_left = log_sigmoid(a - b)`` and ``log p_right = log_sigmoid(b - a)``,
so the dominant child's share is one ``log1p`` and nothing is formed by
``1 - p`` cancellation. A child without mass makes its split certain and
draws nothing; a level whose nodes are all certain, empty or absent runs no
sampling iterations at all.

The binomial variate is drawn on the smaller conditional side by an owned
sampler (geometric inversion when ``n q <= 10``, Hoermann's transformed
rejection otherwise), and the other side is completed by integer
subtraction, which keeps column totals exact for every pool the integer
dtype holds. Each lane keeps its first accepted proposal, so a lane's draw
never depends on how long other lanes keep a loop alive, and only lanes
still pending extend a loop. The rejection test's acceptance bound is
evaluated in a regrouped form whose rounding error grows with the distance
from the mode rather than with the pool.

Precision follows the JAX configuration where the draw is traced: int32
pools with float32 proposals by default, int64 pools with float64 proposals
inside ``jax.enable_x64(True)``. Nothing here changes global configuration.
Two limits of the float proposal are inherent: a pool above the float
dtype's exact-integer range (``2**24`` for float32, ``2**53`` for float64)
is rounded before the *drawn* side is sampled, and the drawn side itself
lives on the float lattice above that range. The completed side and the
column total remain exact integers. No accuracy is claimed for the drawn
side of arbitrary int64 pools beyond this.
"""

from __future__ import annotations

import math
from decimal import Decimal, localcontext

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, random

_INVERSION_LIMIT = 10.0
"""Largest ``n q`` handled by geometric inversion; larger lanes use rejection."""

_STIRLING_SERIES = (1.0 / 12.0, 1.0 / 360.0, 1.0 / 1260.0, 1.0 / 1680.0, 1.0 / 1188.0)


def _stirling_table(last):
    """delta(m) for m = 1..last from exact factorials, correctly rounded to float64."""
    with localcontext() as context:
        context.prec = 40
        half_log_2pi = (2 * Decimal("3.141592653589793238462643383279502884197")).ln() / 2
        return np.array(
            [
                float(
                    Decimal(math.factorial(m)).ln()
                    - ((Decimal(m) + Decimal("0.5")) * Decimal(m).ln() - m + half_log_2pi)
                )
                for m in range(1, last + 1)
            ]
        )


_STIRLING_TABLE = _stirling_table(15)


def _stirling_error(count):
    """delta(m) = log(m!) - [(m + 1/2) log m - m + log(2 pi) / 2] for m >= 1.

    Tabulated through 15 and the asymptotic series beyond, so no factorial or
    lgamma of a large argument is ever formed: delta(2**53) is ~1e-17 while
    the lgamma values it replaces are ~3e17 with an ulp of 64.
    """
    tabulated = count <= 15
    index = jnp.clip(count, 1, 15).astype(jnp.int32) - 1
    safe = jnp.where(tabulated, 16.0, count)
    inverse_square = 1.0 / (safe * safe)
    s0, s1, s2, s3, s4 = _STIRLING_SERIES
    nested = s3 - s4 * inverse_square
    nested = s2 - nested * inverse_square
    nested = s1 - nested * inverse_square
    series = (s0 - nested * inverse_square) / safe
    table = jnp.asarray(_STIRLING_TABLE, count.dtype)
    return jnp.where(tabulated, table[index], series)


def _stirling_tail(k):
    """log(k!) - [(k + 1/2) log(k + 1) - (k + 1) + log(2 pi) / 2], which is delta(k + 1)."""
    return _stirling_error(k + 1.0)


# ------------------------------------------------------------------ binomial


def _inversion(key, n, q, lanes):
    """Successes among ``n`` trials with probability ``q`` by geometric inversion.

    A lane is pending while its running sum of geometric waiting times is at
    most ``n``. The loop ends when no lane in ``lanes`` is pending, so lanes
    outside the regime add no iterations, and a lane's value is fixed once
    it stops pending whatever the other lanes do. Lanes outside ``lanes``
    return ``-1``; their probability is replaced by a finite constant so no
    lane divides by a zero log stay.
    """
    log_stay = jnp.log1p(-jnp.where(lanes, q, 0.5))

    def pending(geom_sum):
        return lanes & (geom_sum <= n)

    def body(carry):
        successes, geom_sum, key = carry
        key, subkey = random.split(key)
        successes = successes + pending(geom_sum).astype(successes.dtype)
        u = random.uniform(subkey, n.shape, n.dtype)
        geom_sum = geom_sum + jnp.ceil(jnp.log(u) / log_stay)
        return successes, geom_sum, key

    zeros = jnp.zeros_like(n)
    successes, _, _ = lax.while_loop(lambda c: pending(c[1]).any(), body, (zeros, zeros, key))
    return successes - 1.0


def _btrs_constants(n, q):
    """Hoermann's transformed-rejection constants and the loop-invariant bound terms."""
    stddev = jnp.sqrt(n * q * (1.0 - q))
    b = 1.15 + 2.53 * stddev
    a = -0.0873 + 0.0248 * b + 0.01 * q
    c = n * q + 0.5
    v_r = 0.92 - 4.2 / b
    r = q / (1.0 - q)
    alpha = (2.83 + 5.1 / b) * stddev
    m = jnp.floor((n + 1.0) * q)
    mode_log_ratio = jnp.log(r * (n - m + 1.0) / (m + 1.0))
    mode_tails = _stirling_tail(m) + _stirling_tail(n - m)
    return dict(
        n=n,
        a=a,
        b=b,
        c=c,
        v_r=v_r,
        alpha=alpha,
        m=m,
        mode_log_ratio=mode_log_ratio,
        mode_tails=mode_tails,
    )


def _btrs_log_ratio(constants, k):
    """log f(k) / f(m) of Binomial(n, q) at the float mode ``m``, without cancellation.

    With ``delta = k - m``, ``A = m + 1`` and ``B = n - m + 1`` (all exact
    floats below the dtype's exact-integer range) the usual bound

        (m + 1/2) log(A / (r B)) + (n + 1) log(B / (n - k + 1))
        + (k + 1/2) log(r (n - k + 1) / (k + 1)) + tails

    regroups identically to

        delta log(r B / A) - (n - k + 1/2) log1p(-delta / B) - (k + 1/2) log1p(delta / A)
        + tail(m) + tail(n - m) - tail(k) - tail(n - k).

    ``delta / B`` and ``delta / A`` are quotients of exact integers, so each
    ``log1p`` carries a relative error of a few ulp and the products an
    absolute error of order ``|delta| eps``; ``log(r B / A)`` is a ratio
    near one, exact to a few ``eps`` absolutely. The usual form multiplies
    ``n + 1`` by the log of a rounded ratio, an ``n eps`` error that reaches
    about one nat at ``2**53``.
    """
    n, m = constants["n"], constants["m"]
    delta = k - m
    return (
        delta * constants["mode_log_ratio"]
        - (n - k + 0.5) * jnp.log1p(-delta / (n - m + 1.0))
        - (k + 0.5) * jnp.log1p(delta / (m + 1.0))
        + constants["mode_tails"]
        - _stirling_tail(k)
        - _stirling_tail(n - k)
    )


def _btrs_proposal(key, constants):
    """One transformed-rejection proposal per lane: ``(accepted, k, next key)``.

    The bound is evaluated at the proposal clipped into ``[0, n]`` so the
    ``log1p`` operands stay inside their domain; an out-of-range proposal
    is rejected by ``in_range`` regardless of the bound.
    """
    n, a, b, c, v_r, alpha = (constants[name] for name in ("n", "a", "b", "c", "v_r", "alpha"))
    key, key_u, key_v = random.split(key, 3)
    u = random.uniform(key_u, n.shape, n.dtype) - 0.5
    v = random.uniform(key_v, n.shape, n.dtype)
    us = 0.5 - jnp.abs(u)
    quick = (us >= 0.07) & (v <= v_r)
    k = jnp.floor((2.0 * a / us + b) * u + c)
    in_range = (k >= 0.0) & (k <= n)
    v = jnp.log(v * alpha / (a / (us * us) + b))
    bound = _btrs_log_ratio(constants, jnp.clip(k, 0.0, n))
    return quick | (in_range & (v <= bound)), k, key


def _btrs(key, n, q, lanes):
    """Transformed rejection over ``lanes``; each lane keeps its first accepted proposal.

    Lanes outside ``lanes`` start accepted with value zero and never extend
    the loop; their operands are replaced by a pool and probability inside
    the rejection regime so every lane's constants and proposals stay finite.
    """
    constants = _btrs_constants(jnp.where(lanes, n, 32.0), jnp.where(lanes, q, 0.5))

    def body(carry):
        k_out, accepted, key = carry
        accept, k, key = _btrs_proposal(key, constants)
        k_out = jnp.where(accept & ~accepted, k, k_out)
        return k_out, accepted | accept, key

    k_out, _, _ = lax.while_loop(lambda c: (~c[1]).any(), body, (jnp.zeros_like(n), ~lanes, key))
    return k_out


def _split_binomial(key, count, log_left, log_right):
    """Binomial(count, exp(log_left)) drawn on the smaller side, completed in the integer dtype.

    ``log_left`` and ``log_right`` are the two conditional log probabilities
    (each at most zero, ``-inf`` allowed) of nonnegative integer ``count``.
    Returns the left count; the right count is ``count`` minus it.
    """
    shape = jnp.broadcast_shapes(count.shape, log_left.shape, log_right.shape)
    count, log_left, log_right = (jnp.broadcast_to(x, shape) for x in (count, log_left, log_right))
    small = log_left <= log_right
    side = jnp.exp(jnp.where(small, log_left, log_right))
    # exp can round the smaller side a few ulp above one half; 1 - side is exact there.
    flip = side > 0.5
    q = jnp.where(flip, 1.0 - side, side)
    n = count.astype(q.dtype)
    active = (count > 0) & (q > 0.0)
    inversion = active & (n * q <= _INVERSION_LIMIT)
    rejection = active & ~inversion
    key_inversion, key_rejection = random.split(key)
    inverse = _inversion(key_inversion, n, q, inversion)
    rejected = _btrs(key_rejection, n, q, rejection)
    drawn = jnp.where(inversion, inverse, jnp.where(rejection, rejected, 0.0)).astype(count.dtype)
    drawn = jnp.where(flip, count - drawn, drawn)
    return jnp.where(small, drawn, count - drawn)


# ---------------------------------------------------------------------- tree


def _split_log_probabilities(left_mass, right_mass):
    """Conditional log probabilities of two children from their subtree log masses.

    Both live: ``log_sigmoid(+-(a - b))``, so the dominant child's share is
    one ``log1p`` and the other side is the exact difference minus it.
    Otherwise the split is certain: a right child without mass (including
    a parent without any mass, or with undefined mass) sends everything
    left, a left child without mass sends everything right.
    """
    live_left, live_right = jnp.isfinite(left_mass), jnp.isfinite(right_mass)
    both = live_left & live_right
    gap = jnp.where(both, left_mass, 0.0) - jnp.where(both, right_mass, 0.0)
    log_left = jnp.where(both, jax.nn.log_sigmoid(gap), jnp.where(live_right, -jnp.inf, 0.0))
    log_right = jnp.where(both, jax.nn.log_sigmoid(-gap), jnp.where(live_right, 0.0, -jnp.inf))
    return log_left, log_right


def _pad_even(masses):
    """One ``-inf`` node on the right when a level has an odd number of nodes."""
    if masses.shape[-2] % 2 == 0:
        return masses
    padding = [(0, 0)] * (masses.ndim - 2) + [(0, 1), (0, 0)]
    return jnp.pad(masses, padding, constant_values=-jnp.inf)


def _tree_levels(log_weights):
    """Subtree log masses per level, leaves first: ``K, ceil(K / 2), ..., 1`` nodes.

    Leaf ``0`` is the no-event category so a pool without any mass travels
    the leftmost path to the residual, and the padding node, always a right
    child, never receives a unit.
    """
    leaves = jnp.concatenate([log_weights[..., -1:, :], log_weights[..., :-1, :]], axis=-2)
    levels = [leaves]
    while levels[-1].shape[-2] > 1:
        even = _pad_even(levels[-1])
        levels.append(jnp.logaddexp(even[..., ::2, :], even[..., 1::2, :]))
    return levels


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
    pool = jnp.maximum(total, 0)[..., None, :]
    if days == 0:
        return jnp.zeros(pool.shape[:-2] + (0,) + pool.shape[-1:], pool.dtype)
    levels = _tree_levels(log_weights.astype(float_dtype))
    root = levels[-1]
    counts = jnp.broadcast_to(pool, jnp.broadcast_shapes(pool.shape, root.shape))
    for children in reversed(levels[:-1]):
        key, subkey = random.split(key)
        even = _pad_even(children)
        log_left, log_right = _split_log_probabilities(even[..., ::2, :], even[..., 1::2, :])
        left = _split_binomial(subkey, counts, log_left, log_right)
        pairs = jnp.stack([left, counts - left], axis=-2)
        counts = pairs.reshape(pairs.shape[:-3] + (2 * pairs.shape[-3],) + pairs.shape[-1:])
        counts = counts[..., : children.shape[-2], :]
    undefined = (total < 0) | ((total > 0) & ~jnp.isfinite(root[..., 0, :]))
    return jnp.where(undefined[..., None, :], -1, counts[..., 1:, :])
