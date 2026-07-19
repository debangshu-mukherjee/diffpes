r"""Angular matrix elements for dipole photoemission.

Extended Summary
----------------
Provides Gaunt coefficients, real spherical harmonics, and full
dipole matrix element assembly for the differentiable ARPES
forward model.  The dipole matrix element for orbital
:math:`(n, l, m)` combines radial integrals, Gaunt coefficients,
and spherical harmonics of the photoelectron direction:

.. math::

    M(\mathbf{k}, n, l, m) = \sum_{l', m'} B_{n,l}^{l'}(|\mathbf{k}|)
        \cdot G(l, m, l', m') \cdot Y_{l'}^{m'}(\hat{k})
        \cdot \hat{e}_{q(m'-m)}

The submodules are organized as follows:

- :mod:`dipole`
    Full dipole matrix element assembly.
- :mod:`gaunt`
    Gaunt coefficient table for dipole transitions.
- :mod:`spherical_harmonics`
    Real spherical harmonics in JAX.

Routine Listings
----------------
:obj:`GAUNT_TABLE`
    Module-level precomputed Gaunt coefficient table for l_max=4.
:obj:`L_MAX`
    Maximum angular momentum supported by the precomputed table.
:func:`build_gaunt_table`
    Build the dipole Gaunt coefficient lookup table.
:func:`dipole_intensities_all_orbitals`
    Compute |M|^2 for all orbitals in the basis.
:func:`dipole_intensity_orbital`
    Compute |M|^2 for one orbital.
:func:`dipole_matrix_element_single`
    Compute dipole matrix element for a single orbital (n, l, m).
:func:`gaunt_lookup`
    Look up a single Gaunt coefficient from the precomputed table.
:func:`real_spherical_harmonic`
    Evaluate a single real spherical harmonic.
:func:`real_spherical_harmonics_all`
    Evaluate all real spherical harmonics up to l_max.

Notes
-----
All functions are JAX-compatible and support JIT compilation and
automatic differentiation.  The Gaunt table is computed once at
import time using pure Python and stored as a JAX array for O(1)
lookup during traced computation.
"""

from .dipole import (
    dipole_intensities_all_orbitals,
    dipole_intensity_orbital,
    dipole_matrix_element_single,
)
from .gaunt import GAUNT_TABLE, L_MAX, build_gaunt_table, gaunt_lookup
from .spherical_harmonics import (
    real_spherical_harmonic,
    real_spherical_harmonics_all,
)

__all__: list[str] = [
    "GAUNT_TABLE",
    "L_MAX",
    "build_gaunt_table",
    "dipole_intensities_all_orbitals",
    "dipole_intensity_orbital",
    "dipole_matrix_element_single",
    "gaunt_lookup",
    "real_spherical_harmonic",
    "real_spherical_harmonics_all",
]
