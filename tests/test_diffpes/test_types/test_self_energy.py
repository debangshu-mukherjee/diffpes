"""Validate self-energy configuration carriers and factory modes.

The cases cover differentiable leaf reconstruction, default and polynomial
configuration, and the static and traced validation rules for tabulated data.
"""

import chex
import jax
import jax.numpy as jnp
import pytest

from diffpes.types import SelfEnergyConfig, make_self_energy_config
from tests._assertions import assert_rejects


class TestSelfEnergyConfig:
    """Validate :class:`~diffpes.types.SelfEnergyConfig` as a JAX PyTree.

    The static mode and differentiable coefficient leaves must survive JAX
    flattening and reconstruction together.

    :see: :class:`~diffpes.types.SelfEnergyConfig`
    """

    def test_pytree_round_trip(self) -> None:
        """Preserve self-energy mode and coefficients through reconstruction.

        The check compares a constant 0.2 eV broadening before and after a JAX
        PyTree round trip.

        Notes
        -----
        Constructs the carrier through its factory, flattens and unflattens it
        with JAX, then compares the static and array fields independently.
        """
        config: SelfEnergyConfig = make_self_energy_config(gamma=0.2)
        leaves: list[object]
        tree: jax.tree_util.PyTreeDef
        leaves, tree = jax.tree_util.tree_flatten(config)
        restored: SelfEnergyConfig = jax.tree_util.tree_unflatten(tree, leaves)

        assert restored.mode == config.mode
        chex.assert_trees_all_close(restored.coefficients, config.coefficients)


class TestMakeSelfEnergyConfig:
    """Validate :func:`~diffpes.types.make_self_energy_config`.

    The factory must construct supported constant and polynomial modes while
    rejecting invalid selectors and malformed tabulated coordinates.

    :see: :func:`~diffpes.types.make_self_energy_config`
    """

    def test_constructs_constant_default(self) -> None:
        """Construct the documented constant 0.1 eV default broadening.

        The check verifies the static mode, one-element coefficient shape, and
        scalar value to ``pytest.approx`` precision.

        Notes
        -----
        Calls the factory without arguments and compares every field defining
        the default constant configuration.
        """
        config: SelfEnergyConfig = make_self_energy_config()

        assert config.mode == "constant"
        chex.assert_shape(config.coefficients, (1,))
        assert float(config.coefficients[0]) == pytest.approx(0.1)

    def test_constructs_polynomial_mode(self) -> None:
        """Preserve two explicit polynomial coefficients.

        The check verifies the static polynomial selector and coefficients
        ``[0.01, 0.1]`` used for energy-dependent broadening.

        Notes
        -----
        Constructs the polynomial configuration with a JAX array and compares
        the stored mode and coefficients directly.
        """
        coefficients: jax.Array = jnp.array([0.01, 0.1])
        config: SelfEnergyConfig = make_self_energy_config(
            mode="polynomial", coefficients=coefficients
        )

        assert config.mode == "polynomial"
        chex.assert_trees_all_close(config.coefficients, coefficients)

    def test_rejects_missing_tabulated_nodes(self) -> None:
        """Reject tabulated broadening without interpolation nodes.

        The check enforces the required coordinate set for a tabulated
        self-energy configuration.

        Notes
        -----
        Selects tabulated mode without ``energy_nodes`` and matches the static
        factory diagnostic.
        """
        with pytest.raises(ValueError, match="energy_nodes required"):
            make_self_energy_config(mode="tabulated")

    def test_rejects_invalid_mode(self) -> None:
        """Reject a self-energy mode outside the supported static set.

        The check isolates selector validation from all numerical parameter
        checks.

        Notes
        -----
        Supplies ``mode="invalid"`` and matches the factory's allowed-mode
        diagnostic.
        """
        with pytest.raises(ValueError, match="mode must be"):
            make_self_energy_config(mode="invalid")

    def test_rejects_unsorted_or_mismatched_nodes(self) -> None:
        """Reject repeated nodes and unequal node and coefficient lengths.

        The check covers both tabulated-axis invariants with otherwise finite
        numerical inputs.

        Notes
        -----
        Uses the shared eager-and-JIT rejection helper for repeated nodes and
        then for a three-node, two-coefficient length mismatch.
        """
        assert_rejects(
            make_self_energy_config,
            mode="tabulated",
            coefficients=jnp.ones(2),
            energy_nodes=jnp.array([0.0, 0.0]),
            match="energy nodes strictly increasing",
        )
        assert_rejects(
            make_self_energy_config,
            mode="tabulated",
            coefficients=jnp.ones(2),
            energy_nodes=jnp.arange(3.0),
            match="energy_nodes and coefficients must have the same length",
        )
