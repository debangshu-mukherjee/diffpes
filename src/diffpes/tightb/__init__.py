"""Provide native tight-binding tools and ARPES-side adapters.

Extended Summary
----------------
The native tight-binding layer provides model construction and Slater-Koster
coupling. It adds spin-orbit coupling, slabs, and degeneracy-safe
diagonalization as the plan series progresses. It also consumes
``DiagonalizedBands`` from other electronic-structure sources.

The package exposes native basis-position-gauge Hamiltonian assembly,
degeneracy-regularized diagonalization, an eigenvalues-only fast path, and
ARPES-side adapters. Analytic chain and graphene models live only in the test
fixture layer.

The following list describes the submodules:

- :mod:`diagonalize`
    Diagonalize native bands and adapt atom-resolved VASP projections.
- :mod:`hamiltonian`
    Assemble native tight-binding Bloch Hamiltonians.
- :mod:`kspace`
    Build differentiable paths and fixed-shape rasters in k-space.
- :mod:`projections`
    Convert eigenvectors to orbital weights.

Routine Listings
----------------
:func:`bloch_hamiltonian`
    Assemble one basis-position-gauge Bloch Hamiltonian.
:func:`bloch_hamiltonian_batch`
    Assemble Bloch Hamiltonians for a batch of fractional k-points.
:func:`build_arpes_kmesh`
    Build a fixed-kz ARPES raster in fractional coordinates.
:func:`build_bz_mesh`
    Build a fixed-shape reciprocal mesh and its first-zone mask.
:func:`build_kmesh_hv`
    Build a photon-energy raster in fractional coordinates.
:func:`build_kpath`
    Build a labeled path between k-space anchors.
:func:`diagonalize_tb`
    Diagonalize a native tight-binding model over k-points.
:func:`eigh_safe`
    Diagonalize a Hermitian matrix with a regularized eigenvector JVP.
:func:`eigvalsh_bands`
    Compute only native tight-binding eigenvalues over k-points.
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
    Convert atom-resolved VASP projections to approximate band vectors.
"""

from .diagonalize import (
    diagonalize_tb,
    eigh_safe,
    eigvalsh_bands,
    vasp_to_diagonalized,
)
from .hamiltonian import (
    bloch_hamiltonian,
    bloch_hamiltonian_batch,
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
    "bloch_hamiltonian",
    "bloch_hamiltonian_batch",
    "build_arpes_kmesh",
    "build_bz_mesh",
    "build_kmesh_hv",
    "build_kpath",
    "diagonalize_tb",
    "eigh_safe",
    "eigvalsh_bands",
    "eigenvector_orbital_weights",
    "first_bz_mask",
    "kpath_arc_length",
    "kpoints_cart_to_frac",
    "kpoints_frac_to_cart",
    "orbital_coefficients",
    "vasp_to_diagonalized",
]
