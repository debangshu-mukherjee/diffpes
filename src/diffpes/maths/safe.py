r"""Provide named gradient-safe elementary operations.

Extended Summary
----------------
This module centralizes guarded elementary operations used by differentiable
physics paths. Each helper sanitizes the input to an unsafe branch before an
outer ``jnp.where`` selects the documented guard value. This double-``where``
pattern prevents a NaN or infinity produced by an inactive branch from
polluting reverse-mode gradients, as described in the JAX FAQ.

Routine Listings
----------------
:func:`safe_arccos`
    Evaluate arccos with saturated values and zero boundary gradients.
:func:`safe_arctan2`
    Evaluate arctan2 with a zero value and gradient at the origin.
:func:`safe_divide`
    Divide with a fallback and zero quotient gradients at zero denominators.
:func:`safe_log`
    Evaluate log with a finite floor and zero gradients below it.
:func:`safe_norm`
    Calculate a Euclidean norm with a zero gradient at zero vectors.
:func:`safe_power`
    Raise positive inputs to a power and return zero otherwise.
:func:`safe_sqrt`
    Evaluate sqrt on positive inputs and return zero otherwise.

Notes
-----
These helpers choose explicit subgradients on guarded sets. They are intended
for numerical guards whose boundary convention is part of the caller's
contract, not for replacing a known nonzero analytic limiting derivative.
"""

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Bool, Float, jaxtyped

from diffpes.types import ScalarFloat


@jaxtyped(typechecker=beartype)
def safe_divide(
    numerator: Float[Array, " ..."],
    denominator: Float[Array, " ..."],
    fallback: ScalarFloat = 0.0,
) -> Float[Array, " ..."]:
    """Divide with a fallback and zero quotient gradients at zero denominators.

    :see: :class:`~.test_safe.TestSafeDivide`

    Parameters
    ----------
    numerator : Float[Array, " ..."]
        Dividend array.
    denominator : Float[Array, " ..."]
        Divisor array broadcast-compatible with ``numerator``.
    fallback : ScalarFloat
        Value returned wherever ``denominator`` is zero.

    Returns
    -------
    quotient : Float[Array, " ..."]
        Broadcast quotient with ``fallback`` at zero denominators.

    Notes
    -----
    The denominator is replaced by one before evaluating the inactive
    division branch. At a zero denominator the selected subgradient with
    respect to both quotient operands is zero; a traced ``fallback`` retains
    its ordinary selected-value gradient.
    """
    nonzero: Bool[Array, " ..."] = denominator != 0.0
    sanitized_denominator: Float[Array, " ..."] = jnp.where(
        nonzero, denominator, 1.0
    )
    divided: Float[Array, " ..."] = numerator / sanitized_denominator
    quotient: Float[Array, " ..."] = jnp.where(nonzero, divided, fallback)
    return quotient


@jaxtyped(typechecker=beartype)
def safe_sqrt(x: Float[Array, " ..."]) -> Float[Array, " ..."]:
    """Evaluate sqrt on positive inputs and return zero otherwise.

    :see: :class:`~.test_safe.TestSafeSqrt`

    Parameters
    ----------
    x : Float[Array, " ..."]
        Real input array.

    Returns
    -------
    roots : Float[Array, " ..."]
        Principal square roots, with zero for ``x <= 0``.

    Notes
    -----
    Non-positive inputs are replaced by one before the square root is
    evaluated. The value and selected subgradient are both zero for
    ``x <= 0``.
    """
    positive: Bool[Array, " ..."] = x > 0.0
    sanitized_x: Float[Array, " ..."] = jnp.where(positive, x, 1.0)
    positive_roots: Float[Array, " ..."] = jnp.sqrt(sanitized_x)
    roots: Float[Array, " ..."] = jnp.where(positive, positive_roots, 0.0)
    return roots


@jaxtyped(typechecker=beartype)
def safe_norm(
    x: Float[Array, " ... n"],
    axis: int = -1,
    keepdims: bool = False,
) -> Float[Array, " ..."]:
    """Calculate a Euclidean norm with a zero gradient at zero vectors.

    :see: :class:`~.test_safe.TestSafeNorm`

    Parameters
    ----------
    x : Float[Array, " ... n"]
        Real vectors.
    axis : int
        (**static** — a compile-time constant; changing it triggers
        retracing) Axis containing vector components.
    keepdims : bool
        (**static** — a compile-time constant; changing it triggers
        retracing) Whether the reduced axis remains with length one.

    Returns
    -------
    norms : Float[Array, " ..."]
        Euclidean norms reduced along ``axis``.

    Notes
    -----
    The squared norm is routed through :func:`safe_sqrt`. A zero vector has
    value zero and the selected gradient is the zero vector.
    """
    squared_norms: Float[Array, " ..."] = jnp.sum(
        x * x, axis=axis, keepdims=keepdims
    )
    norms: Float[Array, " ..."] = safe_sqrt(squared_norms)
    return norms


@jaxtyped(typechecker=beartype)
def safe_arccos(x: Float[Array, " ..."]) -> Float[Array, " ..."]:
    """Evaluate arccos with saturated values and zero boundary gradients.

    :see: :class:`~.test_safe.TestSafeArccos`

    Parameters
    ----------
    x : Float[Array, " ..."]
        Real cosine values.

    Returns
    -------
    angles : Float[Array, " ..."]
        Angles in radians, saturated to ``pi`` below -1 and zero above 1.

    Notes
    -----
    Inputs strictly inside ``(-1, 1)`` use ordinary ``arccos``. Values at
    or beyond either endpoint are selected from constants, giving zero
    subgradients while avoiding the infinite endpoint derivative.
    """
    interior: Bool[Array, " ..."] = jnp.abs(x) < 1.0
    sanitized_x: Float[Array, " ..."] = jnp.where(interior, x, 0.0)
    interior_angles: Float[Array, " ..."] = jnp.arccos(sanitized_x)
    saturated_angles: Float[Array, " ..."] = jnp.where(x <= -1.0, jnp.pi, 0.0)
    angles: Float[Array, " ..."] = jnp.where(
        interior, interior_angles, saturated_angles
    )
    return angles


@jaxtyped(typechecker=beartype)
def safe_arctan2(
    y: Float[Array, " ..."], x: Float[Array, " ..."]
) -> Float[Array, " ..."]:
    """Evaluate arctan2 with a zero value and gradient at the origin.

    :see: :class:`~.test_safe.TestSafeArctan2`

    Parameters
    ----------
    y : Float[Array, " ..."]
        Vertical coordinates.
    x : Float[Array, " ..."]
        Horizontal coordinates broadcast-compatible with ``y``.

    Returns
    -------
    angles : Float[Array, " ..."]
        Four-quadrant angles in radians, with zero at ``(0, 0)``.

    Notes
    -----
    At the indeterminate origin, sanitized coordinates ``(0, 1)`` keep the
    inactive branch finite. The selected value and both coordinate
    subgradients at the origin are zero.
    """
    away_from_origin: Bool[Array, " ..."] = (x != 0.0) | (y != 0.0)
    sanitized_x: Float[Array, " ..."] = jnp.where(away_from_origin, x, 1.0)
    sanitized_y: Float[Array, " ..."] = jnp.where(away_from_origin, y, 0.0)
    ordinary_angles: Float[Array, " ..."] = jnp.arctan2(
        sanitized_y, sanitized_x
    )
    angles: Float[Array, " ..."] = jnp.where(
        away_from_origin, ordinary_angles, 0.0
    )
    return angles


@jaxtyped(typechecker=beartype)
def safe_log(
    x: Float[Array, " ..."], floor: ScalarFloat = 1e-300
) -> Float[Array, " ..."]:
    """Evaluate log with a finite floor and zero gradients below it.

    :see: :class:`~.test_safe.TestSafeLog`

    Parameters
    ----------
    x : Float[Array, " ..."]
        Real input array.
    floor : ScalarFloat
        Positive lower bound used before taking the logarithm.

    Returns
    -------
    logarithms : Float[Array, " ..."]
        Natural logarithms of ``maximum(x, floor)``.

    Notes
    -----
    Inputs at or below the positive floor are replaced before logarithm
    evaluation. Their selected subgradient with respect to ``x`` is zero.
    """
    above_floor: Bool[Array, " ..."] = x > floor
    sanitized_x: Float[Array, " ..."] = jnp.where(above_floor, x, floor)
    logarithms: Float[Array, " ..."] = jnp.log(sanitized_x)
    return logarithms


@jaxtyped(typechecker=beartype)
def safe_power(
    x: Float[Array, " ..."], exponent: ScalarFloat
) -> Float[Array, " ..."]:
    """Raise positive inputs to a power and return zero otherwise.

    :see: :class:`~.test_safe.TestSafePower`

    Parameters
    ----------
    x : Float[Array, " ..."]
        Real bases.
    exponent : ScalarFloat
        Real exponent, including non-integer values.

    Returns
    -------
    powers : Float[Array, " ..."]
        ``x**exponent`` for positive ``x`` and zero otherwise.

    Notes
    -----
    Non-positive bases are replaced by one before exponentiation, which
    prevents fractional powers from entering the complex plane or producing
    NaNs. The selected subgradient with respect to both inputs is zero on
    the guarded set.
    """
    positive: Bool[Array, " ..."] = x > 0.0
    sanitized_x: Float[Array, " ..."] = jnp.where(positive, x, 1.0)
    positive_powers: Float[Array, " ..."] = jnp.power(sanitized_x, exponent)
    powers: Float[Array, " ..."] = jnp.where(positive, positive_powers, 0.0)
    return powers


__all__: list[str] = [
    "safe_arccos",
    "safe_arctan2",
    "safe_divide",
    "safe_log",
    "safe_norm",
    "safe_power",
    "safe_sqrt",
]
