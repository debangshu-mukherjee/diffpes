"""Validate tight-binding and diagonalized-band carriers and factories.

The cases cover PyTree reconstruction, differentiable hopping parameters,
eigensystem validation, connectivity bounds, and two analytic model presets.
"""

import chex
import jax
import jax.numpy as jnp

from diffpes.types import (
    DiagonalizedBands,
    OrbitalBasis,
    TBModel,
    make_1d_chain_model,
    make_diagonalized_bands,
    make_graphene_model,
    make_orbital_basis,
    make_tb_model,
)
from tests._assertions import assert_rejects


def _one_orbital_basis() -> OrbitalBasis:
    """Create the single-s-orbital basis shared by model tests.

    Returns
    -------
    basis : OrbitalBasis
        Validated static basis for one ``s`` orbital.

    Notes
    -----
    Delegates construction to the public factory so tests do not duplicate the
    quantum-number validation contract.
    """
    basis: OrbitalBasis = make_orbital_basis((1,), (0,), (0,), labels=("s",))
    return basis


class TestDiagonalizedBands:
    """Validate :class:`~diffpes.types.DiagonalizedBands` as a JAX PyTree.

    Real eigenvalues and complex eigenvectors must survive JAX reconstruction
    without changing their k-point and orbital axes.

    :see: :class:`~diffpes.types.DiagonalizedBands`
    """

    def test_pytree_round_trip(self) -> None:
        """Preserve a two-point eigensystem through JAX reconstruction.

        The check compares a ``2 x 2`` energy matrix and its complex identity
        eigenvectors after a PyTree round trip.

        Notes
        -----
        The test constructs the carrier through its factory, flattens and unflattens it
        with JAX, and compares both numerical leaves with Chex.
        """
        bands: DiagonalizedBands = make_diagonalized_bands(
            eigenvalues=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            eigenvectors=jnp.eye(2, dtype=jnp.complex128)[None].repeat(
                2, axis=0
            ),
            kpoints=jnp.zeros((2, 3)),
        )
        leaves: list[object]
        tree: jax.tree_util.PyTreeDef
        leaves, tree = jax.tree_util.tree_flatten(bands)
        restored: DiagonalizedBands = jax.tree_util.tree_unflatten(
            tree, leaves
        )

        chex.assert_trees_all_close(restored.eigenvalues, bands.eigenvalues)
        chex.assert_trees_all_close(restored.eigenvectors, bands.eigenvectors)


class TestTBModel:
    """Validate :class:`~diffpes.types.TBModel` gradient behavior.

    Hopping amplitudes must remain differentiable leaves while connectivity and
    the orbital basis remain static model structure.

    :see: :class:`~diffpes.types.TBModel`
    """

    def test_hopping_gradient(self) -> None:
        """Differentiate a quadratic loss through hopping amplitudes.

        At hopping value 1 eV, the derivative of :math:`t^2` must equal 2
        within the default Chex numerical tolerance.

        Notes
        -----
        The test constructs a one-hop model, differentiates an independently defined
        quadratic loss with JAX, and compares the hopping leaf with 2.
        """
        model: TBModel = make_tb_model(
            hopping_params=jnp.array([1.0]),
            lattice_vectors=jnp.eye(3),
            hopping_indices=((0, 0, (1, 0, 0)),),
            n_orbitals=1,
            orbital_basis=_one_orbital_basis(),
        )

        def loss(candidate: TBModel) -> jax.Array:
            """Return the quadratic hopping-amplitude loss."""
            result: jax.Array = jnp.sum(candidate.hopping_params**2)
            return result

        gradient: TBModel = jax.grad(loss)(model)

        chex.assert_trees_all_close(gradient.hopping_params, 2.0)


class TestMakeDiagonalizedBands:
    """Validate :func:`~diffpes.types.make_diagonalized_bands`.

    The factory must preserve compatible eigensystem dimensions and reject
    non-finite eigenvectors or mismatched k-point axes.

    :see: :func:`~diffpes.types.make_diagonalized_bands`
    """

    def test_constructs_expected_shapes(self) -> None:
        """Construct a five-point, three-band, four-orbital eigensystem.

        The check gates the exact eigenvalue and eigenvector dimensions after
        dtype normalization by the factory.

        Notes
        -----
        Supplies finite zero energies and unit complex vectors, then checks both
        carrier shapes with Chex.
        """
        bands: DiagonalizedBands = make_diagonalized_bands(
            eigenvalues=jnp.zeros((5, 3)),
            eigenvectors=jnp.ones((5, 3, 4), dtype=jnp.complex128),
            kpoints=jnp.zeros((5, 3)),
        )

        chex.assert_shape(bands.eigenvalues, (5, 3))
        chex.assert_shape(bands.eigenvectors, (5, 3, 4))

    def test_rejects_invalid_eigensystems(self) -> None:
        """Reject non-finite vectors and inconsistent k-point dimensions.

        The check covers the numerical finiteness guard and the static leading
        dimension contract between eigenvalues and eigenvectors.

        Notes
        -----
        The test uses the eager-and-JIT rejection helper first with a NaN vector and
        then with unequal one- and two-point eigensystem arrays.
        """
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


class TestMakeTBModel:
    """Validate :func:`~diffpes.types.make_tb_model`.

    The factory must preserve hopping arrays and reject connectivity that
    references an orbital outside the declared basis.

    :see: :func:`~diffpes.types.make_tb_model`
    """

    def test_constructs_hopping_array(self) -> None:
        """Store two nearest-neighbor hopping amplitudes.

        The check verifies the one-dimensional hopping shape for forward and
        backward translations of a single orbital.

        Notes
        -----
        The test constructs a validated model with two static connectivity entries and
        checks the differentiable hopping leaf shape with Chex.
        """
        model: TBModel = make_tb_model(
            hopping_params=jnp.array([1.0, 1.0]),
            lattice_vectors=jnp.eye(3),
            hopping_indices=((0, 0, (1, 0, 0)), (0, 0, (-1, 0, 0))),
            n_orbitals=1,
            orbital_basis=_one_orbital_basis(),
        )

        chex.assert_shape(model.hopping_params, (2,))

    def test_rejects_out_of_range_orbital_index(self) -> None:
        """Reject connectivity referencing an absent second orbital.

        The check isolates the static orbital-index bound for a model declaring
        exactly one orbital.

        Notes
        -----
        Supplies connectivity ending at index one and matches the factory's
        allowed-index diagnostic.
        """
        basis: OrbitalBasis = _one_orbital_basis()

        assert_rejects(
            make_tb_model,
            hopping_params=jnp.ones(1),
            lattice_vectors=jnp.eye(3),
            hopping_indices=((0, 1, (0, 0, 0)),),
            n_orbitals=1,
            orbital_basis=basis,
            match="hopping orbital indices must be in",
        )


class TestMake1dChainModel:
    """Validate :func:`~diffpes.types.make_1d_chain_model`.

    The preset must construct one orbital with two translated nearest-neighbor
    hopping entries and preserve the supplied amplitude.

    :see: :func:`~diffpes.types.make_1d_chain_model`
    """

    def test_constructs_chain_connectivity(self) -> None:
        """Construct the two-hop one-orbital chain preset.

        The check verifies two amplitudes of ``-1.5`` eV and the declared
        single-orbital model size.

        Notes
        -----
        The test calls the convenience factory with an explicit hopping value and
        compares the carrier's static size and numerical hopping leaf.
        """
        model: TBModel = make_1d_chain_model(t=-1.5)

        assert model.n_orbitals == 1
        chex.assert_trees_all_close(
            model.hopping_params, jnp.array([-1.5, -1.5])
        )


class TestMakeGrapheneModel:
    """Validate :func:`~diffpes.types.make_graphene_model`.

    The preset must construct the two-sublattice honeycomb model with the
    expected number of nearest-neighbor hopping entries.

    :see: :func:`~diffpes.types.make_graphene_model`
    """

    def test_constructs_two_orbital_model(self) -> None:
        """Construct the two-orbital graphene preset.

        The check verifies the A/B orbital count and that every generated
        hopping amplitude equals the supplied ``-2.7`` eV value.

        Notes
        -----
        The test calls the convenience factory, checks its static orbital count, and
        compares all numerical hopping leaves with an independent constant.
        """
        model: TBModel = make_graphene_model(t=-2.7)
        expected: jax.Array = jnp.full(model.hopping_params.shape, -2.7)

        assert model.n_orbitals == 2
        chex.assert_trees_all_close(model.hopping_params, expected)
