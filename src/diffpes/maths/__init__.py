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
- :mod:`safe`
    Provide named gradient-safe elementary operations.
- :mod:`spherical_harmonics`
    Real spherical harmonics in JAX.

Routine Listings
----------------
:obj:`GAUNT_TABLE`
    Module-level precomputed Gaunt coefficient table for l_max=4.
:func:`build_gaunt_table`
    Build the dipole Gaunt coefficient lookup table.
:func:`dipole_intensities_all_orbitals`
    Compute ``|M|^2`` for all orbitals in the basis.
:func:`dipole_intensity_orbital`
    Compute ``|M|^2`` for one orbital.
:func:`dipole_matrix_element_single`
    Compute dipole matrix element for a single orbital (n, l, m).
:func:`gaunt_lookup`
    Look up a single Gaunt coefficient from the precomputed table.
:func:`real_spherical_harmonic`
    Evaluate a single real spherical harmonic.
:func:`real_spherical_harmonics_all`
    Evaluate all real spherical harmonics up to l_max.
:func:`safe_arccos`
    Evaluate arccos with saturated values and zero boundary gradients.
:func:`safe_arctan2`
    Evaluate arctan2 with a zero value and gradient at the origin.
:func:`safe_divide`
    Divide with a fallback and zero quotient gradients at zero denominators.
:func:`safe_log`
    Evaluate log with a finite floor and zero gradients below it.
:func:`safe_norm`
    Compute a Euclidean norm with a zero gradient at zero vectors.
:func:`safe_power`
    Raise positive inputs to a power and return zero otherwise.
:func:`safe_sqrt`
    Evaluate sqrt on positive inputs and return zero otherwise.

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
from .gaunt import GAUNT_TABLE, build_gaunt_table, gaunt_lookup
from .safe import (
    safe_arccos,
    safe_arctan2,
    safe_divide,
    safe_log,
    safe_norm,
    safe_power,
    safe_sqrt,
)
from .spherical_harmonics import (
    real_spherical_harmonic,
    real_spherical_harmonics_all,
)

__all__: list[str] = [
    "GAUNT_TABLE",
    "build_gaunt_table",
    "dipole_intensities_all_orbitals",
    "dipole_intensity_orbital",
    "dipole_matrix_element_single",
    "gaunt_lookup",
    "real_spherical_harmonic",
    "real_spherical_harmonics_all",
    "safe_arccos",
    "safe_arctan2",
    "safe_divide",
    "safe_log",
    "safe_norm",
    "safe_power",
    "safe_sqrt",
]
