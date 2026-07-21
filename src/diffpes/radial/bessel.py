"""Compute spherical Bessel functions in JAX.

Extended Summary
----------------
The module computes :math:`j_l(x)` with stable low-order seeds and an
upward recurrence. A small-argument limit avoids division by zero at the
origin.

Routine Listings
----------------
:func:`spherical_bessel_jl`
    Evaluate spherical Bessel function :math:`j_l(x)`.
"""

import math

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, Integer, jaxtyped

from diffpes.types import SMALL_ARGUMENT


def _odd_double_factorial(order: int) -> float:
    r"""Compute odd double factorial ``(order)!!`` for odd ``order``.

    Extended Summary
    ----------------
    The function computes the double factorial :math:`n!!` for positive odd
    integers:

    .. math::

        n!! = 1 \cdot 3 \cdot 5 \cdots n = \prod_{k=0}^{(n-1)/2} (2k + 1)

    The spherical Bessel function uses this value in its small-argument
    limit:

    .. math::

        j_l(x) \approx \frac{x^l}{(2l+1)!!} \quad \text{for } |x| \ll 1

    The function applies ``math.prod`` to ``range(1, order+1, 2)``. It then
    converts the exact integer result to a float.

    Parameters
    ----------
    order : int
        A positive odd integer.

    Returns
    -------
    value : float
        The double factorial ``order!!``.

    Raises
    ------
    ValueError
        If ``order`` is not a positive odd integer.
    """
    if order < 1 or order % 2 == 0:
        msg: str = "order must be a positive odd integer"
        raise ValueError(msg)
    product: int = math.prod(range(1, order + 1, 2))
    value: float = float(product)
    return value


@jaxtyped(typechecker=beartype)
def spherical_bessel_jl(
    order: int,
    x: Float[Array, " ..."],
) -> Float[Array, " ..."]:
    r"""Evaluate spherical Bessel function :math:`j_l(x)`.

    The function computes the spherical Bessel function of the first kind.
    It uses closed-form seeds for l=0 and l=1. It uses an upward recurrence
    for l >= 2.

    **Seed values:**

    .. math::

        j_0(x) = \frac{\sin x}{x}

        j_1(x) = \frac{\sin x}{x^2} - \frac{\cos x}{x}

    **Upward recurrence** (for l >= 2, starting from l=1):

    .. math::

        j_{l+1}(x) = \frac{2l + 1}{x} \, j_l(x) - j_{l-1}(x)

    The function implements the recurrence with ``jax.lax.fori_loop``. The
    loop runs from index 1 to ``order - 1``. It carries the pair
    :math:`(j_{l-1}, j_l)` and produces :math:`j_{l+1}` at each step.

    **Small-argument limit:**

    For :math:`|x| < 10^{-8}` (the module constant ``SMALL_ARGUMENT``),
    the function uses the leading-order Taylor expansion:

    .. math::

        j_l(x) \approx \frac{x^l}{(2l+1)!!}

    This expansion avoids division by zero in the seed formulas. The
    ``small_mask`` value selects the recurrence or the small-argument limit.
    The ``x_safe`` input replaces the masked entries with 1.0. This replacement
    prevents invalid recurrence values from affecting the gradients.

    **Numerical stability notes:**

    - The upward recurrence remains stable when :math:`j_l(x)` dominates the
      recurrence at moderate x.
    - A downward Miller recurrence offers more stability when l is very large
      relative to x.
    - ARPES simulations do not use this regime. They use l <= 5 and usually
      use kr values from O(1) to O(10).
    - The ``jnp.where`` mask defines the gradients everywhere, including at
      x = 0.

    :see: :class:`~.test_bessel.TestSphericalBesselJl`

    Parameters
    ----------
    order : int
        Non-negative angular momentum order.
    x : Float[Array, " ..."]
        Real argument array.

    Returns
    -------
    values : Float[Array, " ..."]
        ``j_l(x)`` evaluated element-wise with the same shape as ``x``.

    Raises
    ------
    ValueError
        If ``order`` is negative.

    Notes
    -----
    The ``order`` parameter is a Python integer, not a JAX tracer. A change to
    this parameter causes JAX to trace the function again. The loop bounds
    and the double-factorial constant depend on ``order`` statically. The
    series branch preserves the nonzero analytic limiting gradient near zero.
    """
    if order < 0:
        msg: str = "order must be non-negative"
        raise ValueError(msg)

    x_arr: Float[Array, " ..."] = jnp.asarray(x, dtype=jnp.float64)
    small_mask: Float[Array, " ..."] = jnp.abs(x_arr) < SMALL_ARGUMENT
    x_safe: Float[Array, " ..."] = jnp.where(small_mask, 1.0, x_arr)

    j0_nonzero: Float[Array, " ..."] = jnp.sin(x_safe) / x_safe
    values: Float[Array, " ..."]
    if order == 0:
        values = jnp.where(small_mask, 1.0, j0_nonzero)
        return values

    j1_nonzero: Float[Array, " ..."] = (
        jnp.sin(x_safe) / (x_safe * x_safe) - jnp.cos(x_safe) / x_safe
    )
    if order == 1:
        j1_limit: Float[Array, " ..."] = x_arr / 3.0
        values = jnp.where(small_mask, j1_limit, j1_nonzero)
        return values

    def _recurrence_step(
        index: Integer[Array, ""],
        state: tuple[Float[Array, " ..."], Float[Array, " ..."]],
    ) -> tuple[Float[Array, " ..."], Float[Array, " ..."]]:
        previous: Float[Array, " ..."]
        current: Float[Array, " ..."]
        previous, current = state
        index_arr: Float[Array, " "] = jnp.asarray(index, dtype=jnp.float64)
        next_value: Float[Array, " ..."] = (
            (2.0 * index_arr + 1.0) / x_safe
        ) * current - previous
        recurrence_state: tuple[Float[Array, " ..."], Float[Array, " ..."]] = (
            current,
            next_value,
        )
        return recurrence_state

    jl_nonzero: Float[Array, " ..."]
    loop_state: tuple[Float[Array, " ..."], Float[Array, " ..."]] = (
        jax.lax.fori_loop(
            1,
            order,
            _recurrence_step,
            (j0_nonzero, j1_nonzero),
        )
    )
    jl_nonzero = loop_state[1]
    double_factorial: float = _odd_double_factorial(2 * order + 1)
    small_limit: Float[Array, " ..."] = (x_arr**order) / double_factorial
    values = jnp.where(small_mask, small_limit, jl_nonzero)
    return values


__all__: list[str] = ["spherical_bessel_jl"]
