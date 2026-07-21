"""Compute mathematical utilities for ARPES simulations.

Extended Summary
----------------
The module provides JAX-compatible implementations of the Faddeeva function
and data normalization routines. The ARPES simulation pipeline uses these
functions. Complex parameter packing provides the required optimizer boundary
for complex physics.

Routine Listings
----------------
:func:`faddeeva`
    Evaluate the Faddeeva function w(z) = exp(-z^2) erfc(-iz).
:func:`pack_complex`
    Pack complex parameters as stacked real values.
:func:`unpack_complex`
    Unpack stacked real parameters into complex values.
:func:`zscore_normalize`
    Apply z-score normalization (zero-mean, unit-variance).

Notes
-----
The Faddeeva implementation uses a 64-term Taylor series. The ODE
w'(z) = -2z w(z) + 2i/sqrt(pi) defines the coefficients. The series provides
double-precision accuracy for |z| < 6.
"""

import math

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Complex, Float, Int, jaxtyped

from diffpes.maths import safe_divide
from diffpes.types import N_TAYLOR


def _faddeeva_taylor_coeffs() -> Complex[Array, " N"]:
    r"""Compute Taylor coefficients of w(z) through a JAX scan.

    Extended Summary
    ----------------
    The function computes the first ``N_TAYLOR`` coefficients in the Taylor
    expansion of the Faddeeva function about the origin.

    **Mathematical derivation:**

    The Faddeeva function satisfies the ODE:

    .. math::

        w'(z) = -2z \, w(z) + \frac{2i}{\sqrt{\pi}}

    Substituting the power series ansatz :math:`w(z) = \sum_{n=0}^\infty
    a_n z^n` and matching coefficients of :math:`z^n` on both sides
    yields the two-term recurrence:

    .. math::

        a_0 = 1, \quad a_1 = \frac{2i}{\sqrt{\pi}}, \quad
        a_{n+1} = \frac{-2 \, a_{n-1}}{n+1} \quad (n \ge 1)

    The recurrence connects the even-indexed coefficients. It separately
    connects the odd-indexed coefficients. Even coefficients are real, and
    odd coefficients are purely imaginary. This structure reflects the
    symmetry :math:`w(-z) = 2 e^{-z^2} - w(z)`.

    **Implementation:**

    The function implements the recurrence with ``jax.lax.scan``. The scan
    carries the pair :math:`(a_{n-1}, a_n)`. At each zero-based step n, the
    scan computes :math:`a_{n+2} = -2 a_n / (n + 2)`. It produces coefficients
    :math:`a_2` through :math:`a_{N-1}`. The function joins these coefficients
    with the seed values :math:`a_0, a_1`.

    The module reverses the resulting array into descending power order. It
    stores the array in ``_W_POLY`` for Horner's method in ``jnp.polyval``.

    Returns
    -------
    coeffs : Complex[Array, " N"]
        Taylor coefficients :math:`[a_0, a_1, \ldots, a_{N-1}]` in
        ascending power order, where ``N = N_TAYLOR``.
    """
    c0: Complex[Array, " "] = jnp.array(1.0 + 0j, dtype=jnp.complex128)
    c1: Complex[Array, " "] = jnp.array(
        2.0j / math.sqrt(math.pi), dtype=jnp.complex128
    )

    def body(
        carry: tuple[Complex[Array, " "], Complex[Array, " "]],
        n: Int[Array, ""],
    ) -> tuple[
        tuple[Complex[Array, " "], Complex[Array, " "]],
        Complex[Array, " "],
    ]:
        c_prev: Complex[Array, " "]
        c_curr: Complex[Array, " "]
        c_prev, c_curr = carry
        next_c: Complex[Array, " "] = (-2.0 * c_prev) / (
            jnp.asarray(n, dtype=jnp.float64) + 2.0
        )
        scan_output: tuple[
            tuple[Complex[Array, " "], Complex[Array, " "]],
            Complex[Array, " "],
        ] = ((c_curr, next_c), next_c)
        return scan_output

    scan_result: tuple[
        tuple[Complex[Array, " "], Complex[Array, " "]],
        Complex[Array, " N2"],
    ] = jax.lax.scan(
        body,
        (c0, c1),
        jnp.arange(N_TAYLOR - 2, dtype=jnp.int32),
    )
    rest: Complex[Array, " N2"] = scan_result[1]
    full: Complex[Array, " N"] = jnp.concatenate([c0[None], c1[None], rest])
    return full


_W_POLY: Complex[Array, " N"] = _faddeeva_taylor_coeffs()[::-1]


@jaxtyped(typechecker=beartype)
def faddeeva(
    z: Complex[Array, " ..."],
) -> Complex[Array, " ..."]:
    r"""Evaluate the Faddeeva function w(z) = exp(-z^2) erfc(-iz).

    The function computes the Faddeeva function for arbitrary complex arrays.
    It applies Horner's method to a precomputed Taylor polynomial.

    The following equation defines the Faddeeva function:

    .. math::

        w(z) = e^{-z^2} \operatorname{erfc}(-iz)
             = e^{-z^2} \left(1 + \frac{2i}{\sqrt{\pi}}
               \int_0^z e^{t^2} dt \right)

    The Voigt profile uses the real part of the Faddeeva function along the
    imaginary axis. ARPES simulations use it to convolve Lorentzian lifetime
    broadening with Gaussian instrument resolution.

    **Implementation:**

    1. **Cast to complex128**: Convert the input to ``jnp.complex128`` with
       ``jnp.asarray``. This type provides sufficient precision for the
       64-term polynomial.

    2. **Apply Horner's method**: Call ``jnp.polyval`` with ``_W_POLY`` and the
       converted input. ``_W_POLY`` stores the coefficients in descending
       power order. The ``unroll=8`` hint lets XLA pipeline the inner loop.

    The ODE :math:`w'(z) = -2z w(z) + 2i/\sqrt{\pi}` defines the Taylor
    coefficients. The `_faddeeva_taylor_coeffs` function computes them during
    module import. Its docstring gives the full recurrence derivation.

    :see: :class:`~.test_math.TestFaddeeva`

    Parameters
    ----------
    z : Complex[Array, " ..."]
        Complex argument(s), arbitrary shape.

    Returns
    -------
    w : Complex[Array, " ..."]
        Faddeeva function values, same shape as ``z``.

    Notes
    -----
    The result has approximately 15 significant digits for ``|z| < 6``. The
    Taylor series converges slowly for ``|z| >= 6``. Use an asymptotic
    expansion or a continued fraction outside the convergence domain. The
    function does not select these alternatives automatically.

    The function is fully differentiable via JAX autodiff, which
    is important for gradient-based optimization of broadening
    parameters in inverse-fitting workflows.
    """
    z_c: Complex[Array, " ..."] = jnp.asarray(z, dtype=jnp.complex128)
    w: Complex[Array, " ..."] = jnp.polyval(_W_POLY, z_c, unroll=8)
    return w


@jaxtyped(typechecker=beartype)
def pack_complex(
    z: Complex[Array, " ..."],
) -> Float[Array, " ... 2"]:
    """Pack complex parameters as stacked real values.

    Complex parameters cross the optimizer and Fisher-information boundary as
    stacked reals, while values remain complex inside the physics pipeline.
    This function is the sanctioned complex-to-real crossing point.

    :see: :class:`~.test_math.TestPackComplex`

    Parameters
    ----------
    z : Complex[Array, " ..."]
        Complex-valued physics parameters of arbitrary shape.

    Returns
    -------
    packed : Float[Array, " ... 2"]
        Real-valued parameters with real and imaginary components in the final
        axis, in that order.

    Notes
    -----
    The function forms the final axis with
    ``jnp.stack([z.real, z.imag], axis=-1)``. This operation preserves the
    component dtype and exposes independent real optimizer coordinates.

    See Also
    --------
    unpack_complex : Restore complex values inside the physics pipeline.
    """
    packed: Float[Array, " ... 2"] = jnp.stack([z.real, z.imag], axis=-1)
    return packed


@jaxtyped(typechecker=beartype)
def unpack_complex(
    p: Float[Array, " ... 2"],
) -> Complex[Array, " ..."]:
    """Unpack stacked real parameters into complex values.

    Complex parameters cross the optimizer and Fisher-information boundary as
    stacked reals, while values remain complex inside the physics pipeline.
    This function is the sanctioned real-to-complex crossing point.

    :see: :class:`~.test_math.TestUnpackComplex`

    Parameters
    ----------
    p : Float[Array, " ... 2"]
        Real-valued optimizer parameters whose final axis stores real and
        imaginary components, in that order.

    Returns
    -------
    unpacked : Complex[Array, " ..."]
        Complex-valued physics parameters with the packing axis removed.

    Notes
    -----
    ``jax.lax.complex`` combines the final-axis components without changing
    their precision. For a real loss, JAX differentiates the two packed
    components as ordinary real optimizer coordinates.

    See Also
    --------
    pack_complex : Expose complex parameters at the real optimizer boundary.
    """
    unpacked: Complex[Array, " ..."] = jax.lax.complex(p[..., 0], p[..., 1])
    return unpacked


@jaxtyped(typechecker=beartype)
def zscore_normalize(
    data: Float[Array, " ..."],
) -> Float[Array, " ..."]:
    r"""Apply z-score normalization (zero-mean, unit-variance).

    The function transforms a float array to zero mean and unit standard
    deviation. This transformation prepares simulated and experimental ARPES
    spectra for comparison. The z-score transformation is:

    .. math::

        \hat{x}_i = \frac{x_i - \bar{x}}{\sigma}

    where :math:`\bar{x}` is the global mean and :math:`\sigma` is
    the population standard deviation (:math:`\text{ddof}=0`).

    **Implementation details:**

    1. **Compute the statistics**: Compute the global mean with ``jnp.mean``.
       Compute the population standard deviation with ``jnp.std`` and ddof=0.

    2. **Guard against zero deviation**: Pass the centered values and standard
       deviation to :func:`~diffpes.maths.safe_divide`. The function selects
       zero for a constant input and defines a zero subgradient.

    3. **Normalize the values**: Compute ``(data - mean) / safe_std`` for each
       element and return the result.

    :see: :class:`~.test_math.TestZscoreNormalize`

    Parameters
    ----------
    data : Float[Array, " ..."]
        Input data array of any shape.

    Returns
    -------
    normalized : Float[Array, " ..."]
        Normalized data with mean 0 and standard deviation 1
        (or all zeros if the input is constant).

    Notes
    -----
    The function computes one global mean and standard deviation over all
    elements. For each-axis normalization, reshape the data and call the
    function on each slice.

    This function is differentiable via JAX autodiff with respect to
    the input ``data``. The gradient propagates through both the
    mean-subtraction and the division by standard deviation.
    """
    mean_val: Float[Array, " "] = jnp.mean(data)
    std_val: Float[Array, " "] = jnp.std(data)
    centered: Float[Array, " ..."] = data - mean_val
    normalized: Float[Array, " ..."] = safe_divide(centered, std_val)
    return normalized


__all__: list[str] = [
    "faddeeva",
    "pack_complex",
    "unpack_complex",
    "zscore_normalize",
]
