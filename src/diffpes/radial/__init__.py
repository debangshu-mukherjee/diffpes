r"""Provide differentiable radial primitives for ARPES matrix elements.

Extended Summary
----------------
The subpackage provides JAX-compatible spherical Bessel functions and atomic
radial wavefunctions. It also computes radial integrals on a fixed grid. The
dipole matrix element pipeline uses these functions. The central quantity is
the radial integral

.. math::

    B^{l'}(k) = (i)^{l'} \int_0^\infty R(r)\, r^3\, j_{l'}(kr)\, dr

The functions compute this integral with composite trapezoidal quadrature.

The following list describes the submodules:

- :mod:`bessel`
    Compute spherical Bessel functions in JAX.
- :mod:`integrate`
    Evaluate radial integrals on fixed grids.
- :mod:`wavefunctions`
    Evaluate atomic radial wavefunction models in JAX.

Routine Listings
----------------
:func:`hydrogenic_radial`
    Evaluate normalized hydrogenic radial function.
:func:`radial_integral`
    Evaluate dipole radial integral on a fixed radial grid.
:func:`slater_radial`
    Evaluate normalized Slater-type radial function.
:func:`spherical_bessel_jl`
    Evaluate spherical Bessel function :math:`j_l(x)`.

Notes
-----
All functions support JAX transformations and automatic differentiation.
The Bessel and Laguerre recurrences use ``jax.lax.fori_loop``.
"""

from .bessel import spherical_bessel_jl
from .integrate import radial_integral
from .wavefunctions import hydrogenic_radial, slater_radial

__all__: list[str] = [
    "hydrogenic_radial",
    "radial_integral",
    "slater_radial",
    "spherical_bessel_jl",
]
