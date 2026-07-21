"""Provide native tight-binding tools and ARPES-side adapters.

Extended Summary
----------------
The native tight-binding layer provides model construction and Slater-Koster
coupling. It adds spin-orbit coupling, slabs, and degeneracy-safe
diagonalization as the plan series progresses. It also consumes
``DiagonalizedBands`` from other electronic-structure sources.

This module retains:

- **ARPES-side adapters** that stay here permanently:
  ``vasp_to_diagonalized``, ``eigenvector_orbital_weights``,
  ``orbital_coefficients``.
- **Current fixtures, superseded by plan 04**:
  ``build_hamiltonian_k``, ``diagonalize_single_k``,
  ``diagonalize_tb``.

The following list describes the submodules:

- :mod:`diagonalize`
    Diagonalize bands and adapt VASP outputs.
- :mod:`hamiltonian`
    Build tight-binding Hamiltonians in JAX.
- :mod:`kspace`
    Build differentiable paths and fixed-shape rasters in k-space.
- :mod:`projections`
    Convert eigenvectors to orbital weights.

Routine Listings
----------------
:func:`build_hamiltonian_k`
    Build the Bloch Hamiltonian H(k) at a single k-point.
:func:`build_arpes_kmesh`
    Build a fixed-kz ARPES raster in fractional coordinates.
:func:`build_bz_mesh`
    Build a fixed-shape reciprocal mesh and its first-zone mask.
:func:`build_kmesh_hv`
    Build a photon-energy raster in fractional coordinates.
:func:`build_kpath`
    Build a labeled path between k-space anchors.
:func:`diagonalize_single_k`
    Diagonalize H(k) at a single k-point.
:func:`diagonalize_tb`
    Diagonalize a TB model at all k-points.
:func:`eigenvector_orbital_weights`
    Compute orbital weights from eigenvectors.
:func:`first_bz_mask`
    Mark Cartesian points inside the first Brillouin zone.
:func:`kpath_arc_length`
    Compute cumulative Cartesian distance along a k-path.
:func:`kpoints_cart_to_frac`
    Convert Cartesian momenta to fractional k-points.
:func:`kpoints_frac_to_cart`
    Convert fractional k-points to Cartesian momenta.
:func:`orbital_coefficients`
    Return the raw complex orbital coefficients.
:func:`vasp_to_diagonalized`
    Convert VASP BandStructure + OrbitalProjection to DiagonalizedBands.
"""

from .diagonalize import (
    diagonalize_single_k,
    diagonalize_tb,
    vasp_to_diagonalized,
)
from .hamiltonian import (
    build_hamiltonian_k,
)
from .kspace import (
    build_arpes_kmesh,
    build_bz_mesh,
    build_kmesh_hv,
    build_kpath,
    first_bz_mask,
    kpath_arc_length,
    kpoints_cart_to_frac,
    kpoints_frac_to_cart,
)
from .projections import eigenvector_orbital_weights, orbital_coefficients

__all__: list[str] = [
    "build_arpes_kmesh",
    "build_bz_mesh",
    "build_hamiltonian_k",
    "build_kmesh_hv",
    "build_kpath",
    "diagonalize_single_k",
    "diagonalize_tb",
    "eigenvector_orbital_weights",
    "first_bz_mask",
    "kpath_arc_length",
    "kpoints_cart_to_frac",
    "kpoints_frac_to_cart",
    "orbital_coefficients",
    "vasp_to_diagonalized",
]
