"""Validate crystal-geometry carriers and reciprocal-lattice construction.

The cases cover PyTree reconstruction, an analytic cubic reciprocal lattice,
and rejection of a left-handed real-space basis.
"""

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from diffpes.types import CrystalGeometry, make_crystal_geometry
from tests._assertions import assert_rejects


class TestCrystalGeometry:
    """Validate :class:`~diffpes.types.CrystalGeometry` as a JAX PyTree.

    Numeric leaves and static chemical symbols must survive JAX flattening and
    reconstruction without changing the crystal description.

    :see: :class:`~diffpes.types.CrystalGeometry`
    """

    def test_pytree_round_trip(self) -> None:
        """Preserve array leaves and static symbols through a PyTree round trip.

        The check compares a 3 Angstrom cubic lattice and one silicon symbol
        after JAX flattening and reconstruction.

        Notes
        -----
        Constructs a one-atom geometry, flattens and unflattens it with JAX,
        then uses Chex for the numerical and static comparisons.
        """
        lattice: Float[Array, "3 3"] = jnp.eye(3) * 3.0
        coords: Float[Array, "1 3"] = jnp.zeros((1, 3))
        geometry: CrystalGeometry = make_crystal_geometry(
            lattice=lattice,
            coords=coords,
            symbols=("Si",),
            atom_counts=[1],
        )
        leaves: list[object]
        tree: jax.tree_util.PyTreeDef
        leaves, tree = jax.tree.flatten(geometry)
        restored: CrystalGeometry = jax.tree.unflatten(tree, leaves)

        chex.assert_trees_all_close(restored.lattice, geometry.lattice)
        chex.assert_equal(restored.symbols, geometry.symbols)


class TestMakeCrystalGeometry:
    """Validate :func:`~diffpes.types.make_crystal_geometry`.

    The factory must compute the analytic reciprocal basis of a cubic lattice
    and reject non-right-handed input cells.

    :see: :func:`~diffpes.types.make_crystal_geometry`
    """

    def test_computes_cubic_reciprocal_lattice(self) -> None:
        r"""Match the analytic reciprocal lattice for a cubic cell.

        For lattice constant 5 Angstrom, every reciprocal diagonal must equal
        :math:`2\pi/5` inverse Angstrom within ``atol=1e-10``.

        Notes
        -----
        Constructs the cubic geometry, calculates the closed-form reciprocal
        matrix independently, and compares both matrices with Chex.
        """
        lattice_constant: float = 5.0
        lattice: Float[Array, "3 3"] = jnp.eye(3) * lattice_constant
        geometry: CrystalGeometry = make_crystal_geometry(
            lattice=lattice,
            coords=jnp.zeros((1, 3)),
            symbols=("X",),
            atom_counts=[1],
        )
        expected: Float[Array, "3 3"] = (
            jnp.eye(3) * 2.0 * jnp.pi / lattice_constant
        )

        chex.assert_trees_all_close(
            geometry.reciprocal_lattice, expected, atol=1e-10
        )

    def test_rejects_left_handed_lattice(self) -> None:
        """Reject a finite real-space lattice with negative handedness.

        The check verifies the determinant-sign convention independently of
        coordinate and species validation.

        Notes
        -----
        Supplies a diagonal lattice with determinant ``-1`` and matches the
        factory's right-handedness diagnostic.
        """
        assert_rejects(
            make_crystal_geometry,
            lattice=jnp.diag(jnp.array([1.0, 1.0, -1.0])),
            coords=jnp.zeros((1, 3)),
            symbols=("X",),
            atom_counts=[1],
            match="lattice must be right-handed",
        )
