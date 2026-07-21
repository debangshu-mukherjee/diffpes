"""Validate orbital-basis and Slater-parameter carriers and factories.

The cases cover static PyTree metadata, differentiable radial parameters,
default labels and coefficients, and quantum-number and shape validation.
"""

import chex
import jax
import jax.numpy as jnp
import pytest

from diffpes.types import (
    OrbitalBasis,
    SlaterParams,
    make_orbital_basis,
    make_slater_params,
)
from tests._assertions import assert_rejects


class TestOrbitalBasis:
    """Validate :class:`~diffpes.types.OrbitalBasis` as static PyTree data.

    Quantum numbers and labels must survive JAX reconstruction without
    becoming differentiable leaves.

    :see: :class:`~diffpes.types.OrbitalBasis`
    """

    def test_pytree_round_trip(self) -> None:
        """Preserve three orbital labels and quantum-number tuples.

        The check compares ``s``, ``p``, and ``d`` metadata before and after a
        JAX PyTree round trip.

        Notes
        -----
        The test constructs the basis with explicit tuples, flattens and unflattens it,
        and compares every static field exactly.
        """
        basis: OrbitalBasis = make_orbital_basis(
            n_values=(1, 2, 3),
            l_values=(0, 1, 2),
            m_values=(0, 0, 0),
            labels=("s", "p", "d"),
        )
        leaves: list[object]
        tree: jax.tree_util.PyTreeDef
        leaves, tree = jax.tree_util.tree_flatten(basis)
        restored: OrbitalBasis = jax.tree_util.tree_unflatten(tree, leaves)

        assert restored.n_values == basis.n_values
        assert restored.l_values == basis.l_values
        assert restored.m_values == basis.m_values
        assert restored.labels == basis.labels


class TestSlaterParams:
    """Validate :class:`~diffpes.types.SlaterParams` gradient behavior.

    Slater exponents and coefficients must remain differentiable JAX leaves
    while their orbital basis remains static structure.

    :see: :class:`~diffpes.types.SlaterParams`
    """

    def test_zeta_gradient(self) -> None:
        r"""Differentiate a quadratic loss through the Slater exponent.

        At exponent 2 inverse Angstrom, the derivative of :math:`\zeta^2`
        must equal 4 within the default Chex numerical tolerance.

        Notes
        -----
        The test constructs one radial parameter carrier, differentiates an independent
        quadratic loss with JAX, and compares its ``zeta`` gradient with 4.
        """
        basis: OrbitalBasis = make_orbital_basis((1,), (0,), (0,))
        params: SlaterParams = make_slater_params(
            zeta=jnp.array([2.0]), orbital_basis=basis
        )

        def loss(candidate: SlaterParams) -> jax.Array:
            """Return the quadratic Slater-exponent loss."""
            result: jax.Array = jnp.sum(candidate.zeta**2)
            return result

        gradient: SlaterParams = jax.grad(loss)(params)

        chex.assert_trees_all_close(gradient.zeta, 4.0)


class TestMakeOrbitalBasis:
    """Validate :func:`~diffpes.types.make_orbital_basis`.

    The factory must generate stable labels and reject inconsistent tuple
    lengths or quantum numbers outside their physical ranges.

    :see: :func:`~diffpes.types.make_orbital_basis`
    """

    def test_generates_default_labels(self) -> None:
        """Generate zero-indexed labels for unlabeled orbitals.

        The check verifies the independent naming convention ``orb_0`` and
        ``orb_1`` for a two-orbital basis.

        Notes
        -----
        The test constructs a valid two-orbital basis without labels and compares the
        immutable generated tuple exactly.
        """
        basis: OrbitalBasis = make_orbital_basis(
            n_values=(1, 2),
            l_values=(0, 1),
            m_values=(0, 0),
        )

        assert basis.labels == ("orb_0", "orb_1")

    def test_rejects_mismatched_lengths(self) -> None:
        """Reject unequal quantum-number and label tuple lengths.

        The check covers both the three required quantum-number axes and the
        optional label axis.

        Notes
        -----
        The test calls the factory once with mismatched quantum numbers and once with an
        extra label, matching the shared static diagnostic.
        """
        with pytest.raises(ValueError, match="same length"):
            make_orbital_basis((1, 2), (0,), (0, 0))
        with pytest.raises(ValueError, match="same length"):
            make_orbital_basis((1,), (0,), (0,), labels=("s", "extra"))

    def test_rejects_invalid_quantum_numbers(self) -> None:
        """Reject principal quantum numbers below one.

        The check isolates the lower bound on ``n`` from the angular and
        magnetic quantum-number constraints.

        Notes
        -----
        Supplies a length-consistent one-orbital tuple with ``n=0`` and matches
        the factory's principal-quantum-number diagnostic.
        """
        assert_rejects(
            make_orbital_basis,
            (0,),
            (0,),
            (0,),
            match="n_values must all be at least 1",
        )


class TestMakeSlaterParams:
    """Validate :func:`~diffpes.types.make_slater_params`.

    The factory must provide unit default coefficients, normalize dtype, and
    reject incompatible or nonpositive radial parameters.

    :see: :func:`~diffpes.types.make_slater_params`
    """

    def test_supplies_unit_coefficients(self) -> None:
        """Create one unit coefficient per orbital when omitted.

        The check expects a ``2 x 1`` float64 matrix of ones for two Slater
        exponents.

        Notes
        -----
        The test constructs a two-orbital basis and radial carrier without coefficients,
        then compares shape, dtype, and values with Chex.
        """
        basis: OrbitalBasis = make_orbital_basis((1, 2), (0, 1), (0, 0))
        params: SlaterParams = make_slater_params(
            zeta=jnp.array([1.0, 1.5]), orbital_basis=basis
        )

        chex.assert_shape(params.coefficients, (2, 1))
        assert params.coefficients.dtype == jnp.float64
        chex.assert_trees_all_close(params.coefficients, jnp.ones((2, 1)))

    def test_casts_explicit_coefficients_to_float64(self) -> None:
        """Promote explicit float32 coefficients to float64.

        The check verifies the project-wide precision policy for a two-orbital,
        two-term Slater expansion.

        Notes
        -----
        Supplies a float32 coefficient matrix and checks its stored shape and
        dtype after factory normalization.
        """
        basis: OrbitalBasis = make_orbital_basis((1, 2), (0, 1), (0, 0))
        coefficients: jax.Array = jnp.array(
            [[0.8, 0.2], [0.6, 0.4]], dtype=jnp.float32
        )
        params: SlaterParams = make_slater_params(
            zeta=jnp.array([1.0, 1.5]),
            orbital_basis=basis,
            coefficients=coefficients,
        )

        chex.assert_shape(params.coefficients, (2, 2))
        assert params.coefficients.dtype == jnp.float64

    def test_rejects_invalid_radial_arrays(self) -> None:
        """Reject nonpositive exponents and coefficient-axis mismatches.

        The check covers the physical lower bound on ``zeta`` and the static
        first-axis agreement with the orbital basis.

        Notes
        -----
        The test uses the eager-and-JIT rejection helper with zero exponent and then a
        two-row coefficient matrix for a one-orbital basis.
        """
        basis: OrbitalBasis = make_orbital_basis((1,), (0,), (0,))
        assert_rejects(
            make_slater_params,
            zeta=jnp.array([0.0]),
            orbital_basis=basis,
            match="zeta positive",
        )
        assert_rejects(
            make_slater_params,
            zeta=jnp.array([1.0]),
            orbital_basis=basis,
            coefficients=jnp.ones((2, 1)),
            match="coefficients first dimension must match",
        )
