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
- :mod:`projections`
    Convert eigenvectors to orbital weights.

Routine Listings
----------------
:func:`build_hamiltonian_k`
    Build the Bloch Hamiltonian H(k) at a single k-point.
:func:`diagonalize_single_k`
    Diagonalize H(k) at a single k-point.
:func:`diagonalize_tb`
    Diagonalize a TB model at all k-points.
:func:`eigenvector_orbital_weights`
    Compute orbital weights from eigenvectors.
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
from .projections import eigenvector_orbital_weights, orbital_coefficients

__all__: list[str] = [
    "build_hamiltonian_k",
    "diagonalize_single_k",
    "diagonalize_tb",
    "eigenvector_orbital_weights",
    "orbital_coefficients",
    "vasp_to_diagonalized",
]
