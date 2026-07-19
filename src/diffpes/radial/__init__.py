r"""Differentiable radial primitives for ARPES matrix elements.

Extended Summary
----------------
Provides JAX-compatible spherical Bessel functions, atomic radial
wavefunctions, and fixed-grid radial-integral evaluation used by the
dipole-matrix-element pipeline.  The central quantity is the radial
integral

.. math::

    B^{l'}(k) = (i)^{l'} \int_0^\infty R(r)\, r^3\, j_{l'}(kr)\, dr

evaluated on a fixed radial grid via composite trapezoidal quadrature.

The submodules are organized as follows:

- :mod:`bessel`
    Spherical Bessel functions in JAX.
- :mod:`integrate`
    Radial-integral evaluation utilities.
- :mod:`wavefunctions`
    Atomic radial wavefunction models in JAX.

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
All functions are JAX-compatible and support JIT compilation and
automatic differentiation.  Recurrences (Bessel, Laguerre) use
``jax.lax.fori_loop`` for traceability.
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
