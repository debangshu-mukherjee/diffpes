"""Test crystal-geometry construction and validation.

Extended Summary
----------------
Covers reciprocal-lattice construction, PyTree round trips, and handedness rejection for the carrier defined in ``diffpes.types.geometry``.
"""

import chex
import jax
import jax.numpy as jnp

from diffpes.types import make_crystal_geometry
from tests._assertions import assert_rejects


class TestMakeCrystalGeometry(chex.TestCase):
    """Tests for :func:`diffpes.types.geometry.make_crystal_geometry`.

    Verifies correct construction of ``CrystalGeometry`` PyTrees including
    output shape validation, automatic reciprocal lattice computation for
    orthogonal lattices, and JAX PyTree round-trip (flatten/unflatten)
    fidelity.
    """

    def test_basic_creation(self) -> None:
        """Verify that a CrystalGeometry is created with correct field shapes.

        Test Logic
        ----------
        1. **Construct geometry**:
           Build a simple cubic lattice (3 Angstrom) with two atoms
           using ``make_crystal_geometry``.

        2. **Assert shapes**:
           Check that ``lattice`` is (3, 3), ``reciprocal_lattice`` is
           (3, 3), and ``coords`` is (2, 3).

        3. **Assert static field**:
           Confirm that ``symbols`` is preserved as ``("Si",)``.

        Asserts
        -------
        Output array shapes match expected dimensions and the ``symbols``
        tuple is stored unchanged as auxiliary data.
        """
        lattice = jnp.eye(3) * 3.0
        coords = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        geom = make_crystal_geometry(
            lattice=lattice,
            coords=coords,
            symbols=("Si",),
            atom_counts=[2],
        )
        chex.assert_shape(geom.lattice, (3, 3))
        chex.assert_shape(geom.reciprocal_lattice, (3, 3))
        chex.assert_shape(geom.coords, (2, 3))
        chex.assert_equal(geom.symbols, ("Si",))

    def test_reciprocal_lattice_orthogonal(self) -> None:
        """Verify that the reciprocal lattice is correct for a cubic cell.

        Test Logic
        ----------
        1. **Construct orthogonal lattice**:
           Create a simple cubic lattice with parameter ``a = 5.0``
           Angstroms (diagonal identity matrix scaled by ``a``).

        2. **Compute expected reciprocal lattice**:
           For a cubic cell, ``b_i = 2 pi / a`` along each axis, so
           the expected reciprocal lattice is ``eye(3) * 2 pi / a``.

        3. **Compare numerically**:
           Assert element-wise closeness between the factory-computed
           ``reciprocal_lattice`` and the analytical expectation.

        Asserts
        -------
        ``geom.reciprocal_lattice`` matches the analytical ``2 pi / a``
        diagonal matrix to within ``atol=1e-10``.
        """
        a = 5.0
        lattice = jnp.eye(3) * a
        geom = make_crystal_geometry(
            lattice=lattice,
            coords=jnp.zeros((1, 3)),
            symbols=("X",),
            atom_counts=[1],
        )
        expected = jnp.eye(3) * 2.0 * jnp.pi / a
        chex.assert_trees_all_close(
            geom.reciprocal_lattice, expected, atol=1e-10
        )

    def test_pytree_flatten_unflatten(self) -> None:
        """Verify that CrystalGeometry survives a JAX PyTree round-trip.

        Test Logic
        ----------
        1. **Create geometry**:
           Build a minimal CrystalGeometry with one atom.

        2. **Flatten and unflatten**:
           Use ``jax.tree.flatten`` and ``jax.tree.unflatten`` to
           simulate the round-trip JAX performs during ``jit``/``grad``.

        3. **Compare restored fields**:
           Assert that the numeric ``lattice`` array and the auxiliary
           ``symbols`` tuple are identical after reconstruction.

        Asserts
        -------
        ``restored.lattice`` is close to the original and
        ``restored.symbols`` equals the original, confirming both
        children and auxiliary data survive the round-trip.
        """
        lattice = jnp.eye(3) * 3.0
        coords = jnp.array([[0.0, 0.0, 0.0]])
        geom = make_crystal_geometry(
            lattice=lattice,
            coords=coords,
            symbols=("Si",),
            atom_counts=[1],
        )
        leaves, treedef = jax.tree.flatten(geom)
        restored = jax.tree.unflatten(treedef, leaves)
        chex.assert_trees_all_close(restored.lattice, geom.lattice)
        chex.assert_equal(restored.symbols, geom.symbols)


def test_crystal_geometry_rejects_left_handed_lattice() -> None:
    """Reject a finite but left-handed real-space lattice."""
    assert_rejects(
        make_crystal_geometry,
        lattice=jnp.diag(jnp.array([1.0, 1.0, -1.0])),
        coords=jnp.zeros((1, 3)),
        symbols=("X",),
        atom_counts=[1],
        match="lattice must be right-handed",
    )
