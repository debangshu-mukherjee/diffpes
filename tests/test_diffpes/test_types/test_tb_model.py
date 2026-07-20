"""Test tight-binding model and diagonalized-band carriers.

Extended Summary
----------------
Covers construction, PyTree and gradient behavior, connectivity bounds, and eigensystem validation for the carriers defined in ``diffpes.types.tb_model``.
"""

import jax
import jax.numpy as jnp

from diffpes.types import (
    make_diagonalized_bands,
    make_orbital_basis,
    make_tb_model,
)
from tests._assertions import assert_rejects


class TestDiagonalizedBands:
    """Tests for :func:`diffpes.types.make_diagonalized_bands`.

    Validates construction of ``DiagonalizedBands`` PyTrees that store
    eigenvalues and eigenvectors from tight-binding Hamiltonian
    diagonalization. Covers basic creation with shape verification and
    JAX PyTree flatten/unflatten round-trip fidelity for both real
    eigenvalue and complex eigenvector arrays.
    """

    def test_creation(self) -> None:
        """Verify DiagonalizedBands stores eigenvalues and eigenvectors with correct shapes.

        Constructs a DiagonalizedBands with 5 k-points, 3 bands, and
        4 orbitals. Asserts ``eigenvalues`` shape is (5, 3) and
        ``eigenvectors`` shape is (5, 3, 4), confirming the factory
        correctly stores the real eigenvalue matrix and the complex
        eigenvector tensor with the expected dimensions
        (K, B) and (K, B, O) respectively.
        """
        K, B, O = 5, 3, 4
        diag = make_diagonalized_bands(
            eigenvalues=jnp.zeros((K, B)),
            eigenvectors=jnp.ones((K, B, O), dtype=jnp.complex128),
            kpoints=jnp.zeros((K, 3)),
            fermi_energy=0.0,
        )
        assert diag.eigenvalues.shape == (K, B)
        assert diag.eigenvectors.shape == (K, B, O)

    def test_pytree_round_trip(self) -> None:
        """Verify DiagonalizedBands survives a JAX PyTree flatten/unflatten round-trip.

        Constructs a 2-k-point, 2-band, 2-orbital DiagonalizedBands
        with specific eigenvalues [[1, 2], [3, 4]] and identity
        eigenvectors. Flattens and reconstructs via JAX tree utilities.
        Asserts that the restored ``eigenvalues`` match the originals
        via ``jnp.allclose``, confirming both the real eigenvalue and
        complex eigenvector leaf arrays, plus the scalar Fermi energy
        auxiliary data, survive the round-trip.
        """
        K, B, O = 2, 2, 2
        diag = make_diagonalized_bands(
            eigenvalues=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            eigenvectors=jnp.eye(2, dtype=jnp.complex128)[None].repeat(
                2, axis=0
            ),
            kpoints=jnp.zeros((2, 3)),
        )
        leaves, treedef = jax.tree_util.tree_flatten(diag)
        diag2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert jnp.allclose(diag2.eigenvalues, diag.eigenvalues)


class TestTBModel:
    """Tests for :func:`diffpes.types.make_tb_model`.

    Validates construction of ``TBModel`` PyTrees that hold
    tight-binding Hamiltonian parameters: hopping amplitudes, lattice
    vectors, hopping connectivity indices, and an orbital basis. Covers
    basic creation with shape verification and gradient flow through
    the differentiable ``hopping_params`` field (essential for
    inverse fitting of tight-binding models).
    """

    def test_creation(self) -> None:
        """Verify TBModel stores hopping parameters with the correct shape.

        Constructs a minimal TBModel with 2 hopping parameters
        (forward and backward nearest-neighbor along x), a cubic
        lattice, and a single-orbital basis. Asserts
        ``hopping_params`` shape is (2,), confirming the factory
        correctly stores the differentiable hopping array alongside
        the static auxiliary data (hopping_indices, n_orbitals,
        orbital_basis).
        """
        basis = make_orbital_basis(
            n_values=(1,),
            l_values=(0,),
            m_values=(0,),
        )
        model = make_tb_model(
            hopping_params=jnp.array([1.0, 1.0]),
            lattice_vectors=jnp.eye(3),
            hopping_indices=((0, 0, (1, 0, 0)), (0, 0, (-1, 0, 0))),
            n_orbitals=1,
            orbital_basis=basis,
        )
        assert model.hopping_params.shape == (2,)

    def test_pytree_gradient(self) -> None:
        """Verify JAX gradients flow through the hopping_params field of TBModel.

        Defines a loss function ``loss(m) = sum(hopping_params^2)`` and
        computes ``jax.grad(loss)`` with respect to a TBModel having a
        single hopping parameter of value 1.0. Asserts the gradient is
        2.0 (``d/d(t) t^2 = 2*t = 2.0``), confirming that TBModel is
        a valid differentiable JAX PyTree and that gradients correctly
        propagate through its leaf arrays while treating hopping_indices
        and other structure as static auxiliary data.
        """
        basis = make_orbital_basis(
            n_values=(1,),
            l_values=(0,),
            m_values=(0,),
        )
        model = make_tb_model(
            hopping_params=jnp.array([1.0]),
            lattice_vectors=jnp.eye(3),
            hopping_indices=((0, 0, (1, 0, 0)),),
            n_orbitals=1,
            orbital_basis=basis,
        )

        def loss(m):
            return jnp.sum(m.hopping_params**2)

        grad = jax.grad(loss)(model)
        assert jnp.allclose(grad.hopping_params, 2.0)


def test_tb_model_rejects_out_of_range_orbital_index() -> None:
    """Reject connectivity referencing an absent orbital."""
    basis = make_orbital_basis((1,), (0,), (0,))
    assert_rejects(
        make_tb_model,
        hopping_params=jnp.ones(1),
        lattice_vectors=jnp.eye(3),
        hopping_indices=((0, 1, (0, 0, 0)),),
        n_orbitals=1,
        orbital_basis=basis,
        match="hopping orbital indices must be in",
    )


def test_diagonalized_bands_reject_nonfinite_eigenvectors() -> None:
    """Reject non-finite eigenvectors and incompatible K/B dimensions."""
    assert_rejects(
        make_diagonalized_bands,
        eigenvalues=jnp.zeros((1, 1)),
        eigenvectors=jnp.array([[[jnp.nan + 0.0j]]]),
        kpoints=jnp.zeros((1, 3)),
        match="eigenvectors finite",
    )
    assert_rejects(
        make_diagonalized_bands,
        eigenvalues=jnp.zeros((1, 1)),
        eigenvectors=jnp.zeros((2, 1, 1), dtype=jnp.complex128),
        kpoints=jnp.zeros((1, 3)),
        match="eigenvalues and eigenvectors must agree",
    )
