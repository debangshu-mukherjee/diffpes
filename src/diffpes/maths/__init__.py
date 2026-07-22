r"""Compute angular matrix elements for dipole photoemission.

Extended Summary
----------------
The subpackage provides Gaunt coefficients and real spherical harmonics. It
also assembles the full dipole matrix element for the differentiable ARPES
forward model. The dipole matrix element for orbital
:math:`(n, l, m)` combines radial integrals, Gaunt coefficients,
and spherical harmonics of the photoelectron direction:

.. math::

    M(\mathbf{k}, n, l, m) = \sum_{l', m'} B_{n,l}^{l'}(|\mathbf{k}|)
        \cdot G(l, m, l', m') \cdot Y_{l'}^{m'}(\hat{k})
        \cdot \hat{e}_{q(m'-m)}

The following list describes the submodules:

- :mod:`dipole`
    Assemble full dipole matrix elements.
- :mod:`gaunt`
    Build the Gaunt coefficient table for dipole transitions.
- :mod:`rotations`
    Construct differentiable three-dimensional rotations.
- :mod:`safe`
    Provide named gradient-safe elementary operations.
- :mod:`spherical_harmonics`
    Compute real spherical harmonics in JAX.

Routine Listings
----------------
:obj:`GAUNT_TABLE`
    Module-level precomputed Gaunt coefficient table for l_max=4.
:func:`build_gaunt_table`
    Build the dipole Gaunt coefficient lookup table.
:func:`bond_angles`
    Convert a Cartesian bond to safe polar and azimuthal angles.
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
:func:`real_harmonic_unitary`
    Construct the complex-to-real harmonic basis-function unitary.
:func:`rodrigues_rotation`
    Construct a rotation matrix with Rodrigues' formula.
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
:func:`wigner_d`
    Construct a Wigner D matrix for an active z--y--z rotation.
:func:`wigner_small_d`
    Construct a Wigner small-d matrix from its finite factorial sum.

Notes
-----
All functions support JAX transformations and automatic differentiation. Pure
Python computes the Gaunt table once during import. The module stores the
table as a JAX array for constant-time lookup during traced computation.
"""

from .dipole import (
    dipole_intensities_all_orbitals,
    dipole_intensity_orbital,
    dipole_matrix_element_single,
)
from .gaunt import GAUNT_TABLE, build_gaunt_table, gaunt_lookup
from .rotations import (
    bond_angles,
    real_harmonic_unitary,
    rodrigues_rotation,
    wigner_d,
    wigner_small_d,
)
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
    "bond_angles",
    "build_gaunt_table",
    "dipole_intensities_all_orbitals",
    "dipole_intensity_orbital",
    "dipole_matrix_element_single",
    "gaunt_lookup",
    "real_spherical_harmonic",
    "real_spherical_harmonics_all",
    "real_harmonic_unitary",
    "rodrigues_rotation",
    "safe_arccos",
    "safe_arctan2",
    "safe_divide",
    "safe_log",
    "safe_norm",
    "safe_power",
    "safe_sqrt",
    "wigner_d",
    "wigner_small_d",
]
