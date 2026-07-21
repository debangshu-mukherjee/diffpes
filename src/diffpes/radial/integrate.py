r"""Evaluate radial integrals on fixed grids.

Extended Summary
----------------
The module provides fixed-grid quadrature for dipole radial integrals

.. math::

    B^{l'}(k) = (i)^{l'} \int_0^\infty R(r) r^3 j_{l'}(k r) dr

The module uses JAX-traceable composite trapezoidal integration.

Routine Listings
----------------
:func:`radial_integral`
    Evaluate dipole radial integral on a fixed radial grid.
"""

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Complex, Float, jaxtyped

from .bessel import spherical_bessel_jl


@jaxtyped(typechecker=beartype)
def radial_integral(
    k: Float[Array, " ..."],
    r: Float[Array, " R"],
    radial_values: Float[Array, " R"],
    l_prime: int,
) -> Complex[Array, " ..."]:
    r"""Evaluate dipole radial integral on a fixed radial grid.

    The function computes the dipole radial integral in the partial-wave
    expansion of the photoemission matrix element:

    .. math::

        B^{l'}(k) = i^{l'} \int_0^\infty R(r) \, r^3 \, j_{l'}(k r) \, dr

    The computation proceeds in five steps:

    1. **Form the outer product**: Form the 2-D array :math:`kr` by
       broadcasting ``k`` (shape ``...``) against ``r`` (shape ``R``) via
       ``jnp.expand_dims``. This produces shape ``(... , R)``.

    2. **Evaluate the Bessel function**: Evaluate :math:`j_{l'}(kr)`
       element-wise using `spherical_bessel_jl`. This function handles the
       small-argument limit internally.

    3. **Assemble the integrand**: Multiply the Bessel values by
       :math:`R(r) \cdot r^3`. The :math:`r^3` factor combines the volume
       element and the dipole factor.

    4. **Apply trapezoidal quadrature**: Integrate over the radial grid
       using ``jnp.trapezoid`` along the last axis. The grid need not
       be uniform; ``jnp.trapezoid`` handles non-uniform spacing via
       the ``x`` argument.

    5. **Apply the phase factor**: Multiply the real integral by the complex
       phase :math:`i^{l'}` to obtain the final complex-valued result.

    The function supports batched evaluation. If ``k`` has shape ``(...)``, the
    output retains its leading shape. The integration contracts the last axis.

    :see: :class:`~.test_integrate.TestRadialIntegral`

    Parameters
    ----------
    k : Float[Array, " ..."]
        Momentum magnitudes at which the function computes the integral.
    r : Float[Array, " R"]
        Monotonic radial grid.
    radial_values : Float[Array, " R"]
        Radial wavefunction values sampled on ``r``.
    l_prime : int
        Final-state angular momentum order.

    Returns
    -------
    values : Complex[Array, " ..."]
        Complex radial integral values with the same leading shape as ``k``.

    Raises
    ------
    ValueError
        If ``l_prime`` is negative.

    Notes
    -----
    The result accuracy depends on the density and extent of the radial grid.
    Slater-type orbitals decay as :math:`e^{-\zeta r}`. A grid with 200--500
    points usually suffices when it extends to :math:`r \sim 10/\zeta`.

    The function uses the trapezoidal rule for its simplicity and JAX
    compatibility. Higher-order quadrature can improve accuracy for smooth
    integrands on coarse grids. Such quadrature requires a custom JAX-traceable
    implementation.
    """
    if l_prime < 0:
        msg: str = "l_prime must be non-negative"
        raise ValueError(msg)

    k_arr: Float[Array, " ..."] = jnp.asarray(k, dtype=jnp.float64)
    r_arr: Float[Array, " R"] = jnp.asarray(r, dtype=jnp.float64)
    radial_arr: Float[Array, " R"] = jnp.asarray(
        radial_values, dtype=jnp.float64
    )

    kr: Float[Array, " ... R"] = jnp.expand_dims(k_arr, axis=-1) * r_arr
    bessel_vals: Float[Array, " ... R"] = spherical_bessel_jl(l_prime, kr)
    radial_factor: Float[Array, " R"] = radial_arr * (r_arr**3)
    integrand: Float[Array, " ... R"] = bessel_vals * radial_factor
    real_integral: Float[Array, " ..."] = jnp.trapezoid(
        integrand,
        x=r_arr,
        axis=-1,
    )

    phase: Complex[Array, " "] = jnp.asarray(
        (1j) ** l_prime, dtype=jnp.complex128
    )
    values: Complex[Array, " ..."] = phase * real_integral.astype(
        jnp.complex128
    )
    return values


__all__: list[str] = ["radial_integral"]
