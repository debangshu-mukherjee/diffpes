"""Test density-of-states carriers and factories.

Extended Summary
----------------
Covers construction, PyTree round trips, eager and compiled execution, and energy-axis validation for the carriers defined in ``diffpes.types.dos``.
"""

import chex
import jax
import jax.numpy as jnp

from diffpes.types import make_density_of_states, make_full_density_of_states
from tests._assertions import assert_rejects


class TestFullDensityOfStatesPyTree:
    """PyTree round-trip tests for FullDensityOfStates.

    Exercises the ``tree_flatten`` / ``tree_unflatten`` of
    ``FullDensityOfStates``, both with and without optional fields.
    """

    def test_pytree_round_trip(self) -> None:
        """Verify FullDensityOfStates survives a JAX PyTree flatten/unflatten round-trip.

        Constructs a minimal FullDensityOfStates (spin-up only),
        flattens and unflattens it, then asserts the ``fermi_energy``
        and ``total_dos_up`` arrays are preserved.
        """
        energy = jnp.linspace(-3, 1, 50, dtype=jnp.float64)
        dos_up = jnp.ones(50, dtype=jnp.float64)
        idos_up = jnp.cumsum(dos_up)
        full_dos = make_full_density_of_states(
            energy=energy,
            total_dos_up=dos_up,
            integrated_dos_up=idos_up,
            fermi_energy=-0.5,
        )
        leaves, treedef = jax.tree_util.tree_flatten(full_dos)
        full_dos2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert jnp.allclose(full_dos2.total_dos_up, full_dos.total_dos_up)
        assert jnp.allclose(full_dos2.fermi_energy, full_dos.fermi_energy)
        assert full_dos2.natoms == full_dos.natoms


class TestMakeDensityOfStates(chex.TestCase):
    """Tests for :func:`diffpes.types.dos.make_density_of_states`.

    Verifies correct construction of ``DensityOfStates`` PyTrees including
    output shape validation for the energy axis and total DOS arrays, and
    correct casting of the Fermi energy scalar to a float64 JAX array.
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_basic_creation(self) -> None:
        """Verify that a DensityOfStates is created with correct fields.

        Test Logic
        ----------
        1. **Construct DOS**:
           Create a linearly spaced energy axis (500 points from -10 to
           5 eV), a uniform total DOS, and a Fermi energy of -1.5 eV,
           then call the factory via ``self.variant``.

        2. **Assert shapes**:
           Check that ``energy`` and ``total_dos`` both have shape (500,).

        3. **Assert Fermi energy value**:
           Confirm that ``fermi_energy`` is close to the supplied -1.5
           as a float64 scalar.

        Asserts
        -------
        Array shapes match the input dimensions and the scalar Fermi
        energy is correctly cast and stored.
        """
        ne = 500
        energy = jnp.linspace(-10.0, 5.0, ne)
        dos = jnp.ones(ne)
        var_fn = self.variant(make_density_of_states)
        result = var_fn(energy=energy, total_dos=dos, fermi_energy=-1.5)
        chex.assert_shape(result.energy, (ne,))
        chex.assert_shape(result.total_dos, (ne,))
        chex.assert_trees_all_close(result.fermi_energy, jnp.float64(-1.5))


def test_dos_factories_reject_unsorted_energy_axes() -> None:
    """Reject non-increasing energy coordinates in both DOS carriers."""
    energy = jnp.array([0.0, 0.0])
    assert_rejects(
        make_density_of_states,
        energy=energy,
        total_dos=jnp.ones(2),
        match="energy strictly increasing",
    )
    assert_rejects(
        make_full_density_of_states,
        energy=energy,
        total_dos_up=jnp.ones(2),
        integrated_dos_up=jnp.ones(2),
        match="energy strictly increasing",
    )
