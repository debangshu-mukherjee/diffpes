"""Mathematical utility functions for ARPES simulations.

Extended Summary
----------------
Provides JAX-compatible implementations of the Faddeeva function
(complex error function) and data normalization routines used
throughout the ARPES simulation pipeline. Complex parameter packing
provides the sanctioned optimizer boundary for complex-valued physics.

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
The Faddeeva implementation uses a 64-term Taylor series
derived from the ODE w'(z) = -2z w(z) + 2i/sqrt(pi),
giving double-precision accuracy for |z| < 6.
"""

import math

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Complex, Float, Int, jaxtyped

from diffpes.maths import safe_divide
from diffpes.types import N_TAYLOR


def _faddeeva_taylor_coeffs() -> Complex[Array, " N"]:
    r"""Taylor coefficients of w(z) = exp(-z^2) erfc(-iz) via JAX scan.

    Extended Summary
    ----------------
    Computes the first ``N_TAYLOR`` coefficients of the Taylor expansion
    of the Faddeeva function about the origin.

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

    Note that the recurrence couples even-indexed coefficients among
    themselves and odd-indexed coefficients among themselves (the
    series separates into even and odd parts). Even coefficients are
    real and odd coefficients are purely imaginary, reflecting the
    symmetry :math:`w(-z) = 2 e^{-z^2} - w(z)`.

    **Implementation:**

    The recurrence is implemented with ``jax.lax.scan`` (no Python
    for-loop) for efficiency and XLA compatibility. The scan carry
    is the pair :math:`(a_{n-1}, a_n)`, and at each step n (0-indexed),
    the next coefficient is :math:`a_{n+2} = -2 a_n / (n + 2)`. The
    scan produces coefficients :math:`a_2` through :math:`a_{N-1}`,
    which are concatenated with the seed values :math:`a_0, a_1` to
    form the complete coefficient vector.

    The resulting array is reversed (descending power order) at module
    level and stored in ``_W_POLY`` for use with ``jnp.polyval``
    (Horner's method).

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

    Computes the Faddeeva (scaled complex complementary error) function
    for arbitrary complex arrays using a precomputed Taylor polynomial
    evaluated via Horner's method.

    The Faddeeva function is defined as:

    .. math::

        w(z) = e^{-z^2} \operatorname{erfc}(-iz)
             = e^{-z^2} \left(1 + \frac{2i}{\sqrt{\pi}}
               \int_0^z e^{t^2} dt \right)

    It is closely related to the Voigt profile used in spectroscopy:
    the Voigt function is the real part of the Faddeeva function
    evaluated along the imaginary axis. In ARPES simulations, it
    appears when convolving Lorentzian lifetime broadening with
    Gaussian instrumental resolution.

    **Implementation:**

    1. **Cast to complex128** -- convert the input to ``jnp.complex128``
       via ``jnp.asarray`` to ensure sufficient precision for the 64-term
       polynomial evaluation.

    2. **Evaluate via Horner's method** -- call ``jnp.polyval`` with
       the module-level constant ``_W_POLY`` (coefficients stored in
       descending-power order, i.e. reversed from the natural
       :math:`a_0, \ldots, a_{N-1}` ordering) and the cast input. The
       ``unroll=8`` hint allows XLA to pipeline the inner loop for
       better throughput on accelerators.

    The Taylor coefficients are derived from the ODE
    :math:`w'(z) = -2z w(z) + 2i/\sqrt{\pi}` and precomputed at
    module import time by `_faddeeva_taylor_coeffs`. See that
    function's docstring for the full recurrence derivation.

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
    Accuracy is approximately double-precision (~15 significant digits)
    for ``|z| < 6``. For ``|z| >= 6`` the Taylor series converges slowly
    and
    an asymptotic expansion or continued-fraction representation should
    be used instead. The current implementation does **not** fall back
    to an asymptotic form, so callers must ensure inputs stay within the
    convergence domain.

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
    The final axis is formed exactly as
    ``jnp.stack([z.real, z.imag], axis=-1)``. This preserves the component
    dtype and exposes independent real optimizer coordinates.

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

    Transforms an arbitrary float array so that the output has zero mean
    and unit standard deviation, which is a common preprocessing step
    before comparing simulated and experimental ARPES spectra. The
    z-score transformation is:

    .. math::

        \hat{x}_i = \frac{x_i - \bar{x}}{\sigma}

    where :math:`\bar{x}` is the global mean and :math:`\sigma` is
    the population standard deviation (:math:`\text{ddof}=0`).

    **Implementation details:**

    1. **Compute statistics** -- calculate the global mean and standard
       deviation of the input array using ``jnp.mean`` and ``jnp.std``
       (population std, i.e. ddof=0).

    2. **Guard against zero std** -- route the centered values and standard
       deviation through :func:`~diffpes.maths.safe_divide`, selecting its
       zero fallback for constant inputs. This produces an all-zeros output,
       the mathematically sensible degenerate value, with a zero selected
       subgradient rather than an inactive division-by-zero contaminating
       reverse-mode autodiff.

    3. **Normalize** -- compute ``(data - mean) / safe_std`` element-wise
       and return.

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
    The normalization is computed over **all** elements (global mean
    and std), not per-axis. For per-axis normalization, reshape the
    data and call this function on each slice separately.

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
