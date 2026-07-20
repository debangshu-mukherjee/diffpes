"""Test self-energy configuration construction and validation.

Extended Summary
----------------
Covers configuration modes, defaults, PyTree round trips, and tabulated-node rejection for the carrier defined in ``diffpes.types.self_energy``.
"""

import jax
import jax.numpy as jnp
import pytest

from diffpes.types import make_self_energy_config
from tests._assertions import assert_rejects


class TestSelfEnergyConfig:
    """Tests for :func:`diffpes.types.make_self_energy_config`.

    Validates construction of ``SelfEnergyConfig`` PyTrees that
    parameterize the imaginary part of the electron self-energy for
    ARPES spectral function broadening. Covers the default constant
    mode, polynomial mode, validation errors for tabulated mode
    without energy nodes and for invalid mode strings, and JAX PyTree
    flatten/unflatten round-trip fidelity.
    """

    def test_constant_default(self) -> None:
        """Verify default SelfEnergyConfig uses constant mode with gamma=0.1.

        Calls ``make_self_energy_config()`` with no arguments. Asserts
        ``mode`` is ``"constant"``, ``coefficients`` shape is (1,),
        and the single coefficient value is approximately 0.1 (the
        default gamma broadening), verified via ``pytest.approx``.
        This confirms the factory's default initialization path.
        """
        config = make_self_energy_config()
        assert config.mode == "constant"
        assert config.coefficients.shape == (1,)
        assert float(config.coefficients[0]) == pytest.approx(0.1)

    def test_polynomial(self) -> None:
        """Verify polynomial mode accepts explicit coefficients and stores mode string.

        Constructs a SelfEnergyConfig with ``mode="polynomial"`` and
        two coefficients [0.01, 0.1] (representing a linear polynomial
        in energy). Asserts ``mode`` is ``"polynomial"``, confirming
        the factory accepts this mode and stores the mode string as
        auxiliary data without error.
        """
        config = make_self_energy_config(
            mode="polynomial",
            coefficients=jnp.array([0.01, 0.1]),
        )
        assert config.mode == "polynomial"

    def test_tabulated_requires_nodes(self) -> None:
        """Verify tabulated mode raises ValueError when energy_nodes are not provided.

        Calls ``make_self_energy_config(mode="tabulated")`` without
        supplying ``energy_nodes``. Asserts ``ValueError`` is raised
        with a message matching ``"energy_nodes required"``, confirming
        the factory enforces that tabulated interpolation mode requires
        an explicit set of energy grid nodes.
        """
        with pytest.raises(ValueError, match="energy_nodes required"):
            make_self_energy_config(mode="tabulated")

    def test_invalid_mode_raises(self) -> None:
        """Verify an unrecognized mode string raises ValueError.

        Calls ``make_self_energy_config(mode="invalid")``. Asserts
        ``ValueError`` is raised with a message matching
        ``"mode must be"``, confirming the factory validates the mode
        string against the allowed set (``"constant"``,
        ``"polynomial"``, ``"tabulated"``) and rejects unknown values.
        """
        with pytest.raises(ValueError, match="mode must be"):
            make_self_energy_config(mode="invalid")

    def test_pytree_round_trip(self) -> None:
        """Verify SelfEnergyConfig survives a JAX PyTree flatten/unflatten round-trip.

        Constructs a constant-mode SelfEnergyConfig with ``gamma=0.2``.
        Flattens via ``jax.tree_util.tree_flatten`` and reconstructs
        via ``jax.tree_util.tree_unflatten``. Asserts that the restored
        ``mode`` string matches the original and the ``coefficients``
        array matches via ``jnp.allclose``, confirming that both the
        auxiliary string data and the differentiable leaf arrays survive
        the round-trip.
        """
        config = make_self_energy_config(gamma=0.2)
        leaves, treedef = jax.tree_util.tree_flatten(config)
        config2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert config2.mode == config.mode
        assert jnp.allclose(config2.coefficients, config.coefficients)


def test_self_energy_rejects_unsorted_nodes() -> None:
    """Reject non-increasing and length-mismatched tabulated nodes."""
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
