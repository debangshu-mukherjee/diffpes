"""Provide native tight-binding tools and ARPES-side adapters.

Extended Summary
----------------
The native tight-binding layer of diffpes provides model construction,
Slater-Koster coupling, spin-orbit coupling, slabs, and degeneracy-safe
diagonalization as the plan series is implemented. It also consumes
``DiagonalizedBands`` from other electronic-structure sources.

This module retains:

- **ARPES-side adapters** that stay here permanently:
  ``vasp_to_diagonalized``, ``eigenvector_orbital_weights``,
  ``orbital_coefficients``.
- **Current fixtures, superseded by plan 04**:
  ``build_hamiltonian_k``, ``diagonalize_single_k``,
  ``diagonalize_tb``, ``make_1d_chain_model``,
  ``make_graphene_model``.

The submodules are organized as follows:

- :mod:`diagonalize`
    Differentiable band diagonalization and VASP adapter.
- :mod:`hamiltonian`
    Tight-binding Hamiltonian builder in JAX.
- :mod:`projections`
    Eigenvector to orbital weight conversions.

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
:func:`make_1d_chain_model`
    Create a 1D chain tight-binding model.
:func:`make_graphene_model`
    Create a graphene pz tight-binding model.
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
    make_1d_chain_model,
    make_graphene_model,
)
from .projections import eigenvector_orbital_weights, orbital_coefficients

__all__: list[str] = [
    "build_hamiltonian_k",
    "diagonalize_single_k",
    "diagonalize_tb",
    "eigenvector_orbital_weights",
    "make_1d_chain_model",
    "make_graphene_model",
    "orbital_coefficients",
    "vasp_to_diagonalized",
]
