"""Test scalar and spin-orbit volumetric carriers.

Extended Summary
----------------
Covers defaults, PyTree round trips, and static grid-shape validation for the carriers defined in ``diffpes.types.volumetric``.
"""

import jax
import jax.numpy as jnp

from diffpes.types import make_soc_volumetric_data, make_volumetric_data
from tests._assertions import assert_rejects


class TestVolumetricPyTree:
    """PyTree round-trip tests for VolumetricData and SOCVolumetricData.

    Exercises the ``tree_flatten`` / ``tree_unflatten`` methods and
    the ``atom_counts=None`` default path in both factories.
    """

    def test_volumetric_data_pytree_round_trip(self) -> None:
        """Verify VolumetricData survives a JAX PyTree flatten/unflatten round-trip.

        Constructs a minimal VolumetricData, flattens and unflattens it,
        then asserts the ``charge`` array is preserved.
        """
        lattice = jnp.eye(3, dtype=jnp.float64)
        coords = jnp.zeros((2, 3), dtype=jnp.float64)
        charge = jnp.ones((4, 4, 4), dtype=jnp.float64)
        vol = make_volumetric_data(
            lattice=lattice,
            coords=coords,
            charge=charge,
            grid_shape=(4, 4, 4),
            symbols=("Fe", "Co"),
        )
        leaves, treedef = jax.tree_util.tree_flatten(vol)
        vol2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert jnp.allclose(vol2.charge, vol.charge)
        assert vol2.grid_shape == vol.grid_shape
        assert vol2.symbols == vol.symbols

    def test_volumetric_data_atom_counts_none_default(self) -> None:
        """Verify that atom_counts=None creates a zero-length int32 array.

        When ``atom_counts`` is omitted, the factory should default to
        ``jnp.zeros(0, dtype=jnp.int32)`` so the field is always an array.
        """
        vol = make_volumetric_data(
            lattice=jnp.eye(3, dtype=jnp.float64),
            coords=jnp.zeros((1, 3), dtype=jnp.float64),
            charge=jnp.ones((2, 2, 2), dtype=jnp.float64),
            grid_shape=(2, 2, 2),
            symbols=("Fe",),
        )
        assert vol.atom_counts.dtype == jnp.int32
        assert vol.atom_counts.shape == (0,)

    def test_soc_volumetric_data_pytree_round_trip(self) -> None:
        """Verify SOCVolumetricData survives a JAX PyTree flatten/unflatten round-trip."""
        lattice = jnp.eye(3, dtype=jnp.float64)
        coords = jnp.zeros((1, 3), dtype=jnp.float64)
        charge = jnp.ones((2, 2, 2), dtype=jnp.float64)
        mag = jnp.zeros((2, 2, 2), dtype=jnp.float64)
        mag_vec = jnp.zeros((2, 2, 2, 3), dtype=jnp.float64)
        vol = make_soc_volumetric_data(
            lattice=lattice,
            coords=coords,
            charge=charge,
            magnetization=mag,
            magnetization_vector=mag_vec,
            grid_shape=(2, 2, 2),
            symbols=("Fe",),
        )
        leaves, treedef = jax.tree_util.tree_flatten(vol)
        vol2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert jnp.allclose(vol2.charge, vol.charge)
        assert vol2.grid_shape == vol.grid_shape

    def test_soc_volumetric_data_atom_counts_none_default(self) -> None:
        """Verify that atom_counts=None creates a zero-length int32 array in SOCVolumetricData."""
        vol = make_soc_volumetric_data(
            lattice=jnp.eye(3, dtype=jnp.float64),
            coords=jnp.zeros((1, 3), dtype=jnp.float64),
            charge=jnp.ones((2, 2, 2), dtype=jnp.float64),
            magnetization=jnp.zeros((2, 2, 2), dtype=jnp.float64),
            magnetization_vector=jnp.zeros((2, 2, 2, 3), dtype=jnp.float64),
            grid_shape=(2, 2, 2),
            symbols=("Fe",),
        )
        assert vol.atom_counts.dtype == jnp.int32
        assert vol.atom_counts.shape == (0,)


def test_volumetric_factories_reject_grid_mismatches() -> None:
    """Reject scalar and SOC grids inconsistent with static metadata."""
    assert_rejects(
        make_volumetric_data,
        lattice=jnp.eye(3),
        coords=jnp.zeros((1, 3)),
        charge=jnp.zeros((2, 2, 2)),
        grid_shape=(1, 1, 1),
        match="grid_shape must match charge shape",
    )
    assert_rejects(
        make_soc_volumetric_data,
        lattice=jnp.eye(3),
        coords=jnp.zeros((1, 3)),
        charge=jnp.zeros((1, 1, 1)),
        magnetization=jnp.zeros((1, 1, 1)),
        magnetization_vector=jnp.zeros((2, 1, 1, 3)),
        grid_shape=(1, 1, 1),
        match="grid_shape must match magnetization_vector spatial shape",
    )
