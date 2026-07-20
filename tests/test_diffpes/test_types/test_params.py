"""Test simulation and polarization parameter carriers.

Extended Summary
----------------
Covers defaults, custom values, PyTree round trips, and factory rejection contracts for the carriers defined in ``diffpes.types.params``.
"""

import chex
import jax
import jax.numpy as jnp

from diffpes.types import make_polarization_config, make_simulation_params
from tests._assertions import assert_rejects


class TestMakeSimulationParams(chex.TestCase):
    """Tests for :func:`diffpes.types.params.make_simulation_params`.

    Verifies correct construction of ``SimulationParams`` PyTrees including
    default parameter values, custom value passthrough, and JAX PyTree
    round-trip (flatten/unflatten) fidelity.
    """

    def test_defaults(self) -> None:
        """Verify that default parameter values match expected constants.

        Test Logic
        ----------
        1. **Construct with no arguments**:
           Call ``make_simulation_params()`` with all defaults.

        2. **Assert each default value**:
           Check ``energy_min`` (-3.0), ``energy_max`` (1.0),
           ``fidelity`` (25000), ``sigma`` (0.04), and ``gamma`` (0.1)
           against their documented default values.

        Asserts
        -------
        Each default parameter matches the expected constant, confirming
        the factory's default-value specification is correct.
        """
        params = make_simulation_params()
        chex.assert_trees_all_close(params.energy_min, jnp.float64(-3.0))
        chex.assert_trees_all_close(params.energy_max, jnp.float64(1.0))
        chex.assert_equal(params.fidelity, 25000)
        chex.assert_trees_all_close(params.sigma, jnp.float64(0.04))
        chex.assert_trees_all_close(params.gamma, jnp.float64(0.1))

    def test_custom_values(self) -> None:
        """Verify that custom parameter values are stored correctly.

        Test Logic
        ----------
        1. **Construct with custom arguments**:
           Call the factory with non-default values for all parameters,
           including ``temperature=300.0`` and ``photon_energy=21.2``.

        2. **Spot-check one custom field**:
           Assert that ``params.temperature`` equals the supplied
           ``300.0`` as a float64 JAX scalar.

        Asserts
        -------
        ``params.temperature`` matches the custom input value,
        confirming that user-supplied arguments override defaults and
        are correctly cast to float64.
        """
        params = make_simulation_params(
            energy_min=-5.0,
            energy_max=2.0,
            fidelity=1000,
            sigma=0.08,
            gamma=0.2,
            temperature=300.0,
            photon_energy=21.2,
        )
        chex.assert_trees_all_close(params.temperature, jnp.float64(300.0))

    def test_pytree_compatible(self) -> None:
        """Verify that SimulationParams survives a JAX PyTree round-trip.

        Test Logic
        ----------
        1. **Create params**:
           Build a default ``SimulationParams`` instance.

        2. **Flatten and unflatten**:
           Use ``jax.tree.flatten`` and ``jax.tree.unflatten`` to
           simulate the round-trip JAX performs during ``jit``/``grad``.

        3. **Compare restored field**:
           Assert that ``restored.sigma`` is close to the original
           ``params.sigma``.

        Asserts
        -------
        ``restored.sigma`` matches the original value, confirming that
        both JAX-traced children and the auxiliary ``fidelity`` int
        survive the flatten/unflatten round-trip.
        """
        params = make_simulation_params()
        leaves, treedef = jax.tree.flatten(params)
        restored = jax.tree.unflatten(treedef, leaves)
        chex.assert_trees_all_close(restored.sigma, params.sigma)


class TestMakePolarizationConfig(chex.TestCase):
    """Tests for :func:`diffpes.types.params.make_polarization_config`.

    Verifies correct construction of ``PolarizationConfig`` PyTrees including
    default polarization type and angular values, as well as explicit LVP
    (linear vertical polarization) configuration.
    """

    def test_defaults(self) -> None:
        """Verify that default polarization config is unpolarized with scalar angles.

        Test Logic
        ----------
        1. **Construct with no arguments**:
           Call ``make_polarization_config()`` with all defaults.

        2. **Assert polarization type**:
           Check that ``polarization_type`` defaults to ``"unpolarized"``.

        3. **Assert angle shapes**:
           Confirm that ``theta`` and ``phi`` are 0-D scalar arrays.

        Asserts
        -------
        Default polarization type is ``"unpolarized"`` and angular fields
        are scalar JAX arrays, confirming the factory's default behavior.
        """
        config = make_polarization_config()
        chex.assert_equal(config.polarization_type, "unpolarized")
        chex.assert_shape(config.theta, ())
        chex.assert_shape(config.phi, ())

    def test_lvp(self) -> None:
        """Verify that an LVP polarization config stores the correct type string.

        Test Logic
        ----------
        1. **Construct with LVP settings**:
           Call the factory with ``theta=0.7854``, ``phi=0.0``, and
           ``polarization_type="LVP"`` (linear vertical polarization,
           i.e., s-polarization).

        2. **Assert polarization type**:
           Check that ``config.polarization_type`` is ``"LVP"``.

        Asserts
        -------
        The auxiliary ``polarization_type`` string is stored as ``"LVP"``,
        confirming that user-supplied string arguments are passed through
        unchanged.
        """
        config = make_polarization_config(
            theta=0.7854,
            phi=0.0,
            polarization_type="LVP",
        )
        chex.assert_equal(config.polarization_type, "LVP")


def test_simulation_params_reject_audit_probes() -> None:
    """Reject both invalid simulation-parameter probes from the audit."""
    assert_rejects(
        make_simulation_params, sigma=-1.0, match="sigma must be positive"
    )
    assert_rejects(
        make_simulation_params,
        energy_min=5.0,
        energy_max=-5.0,
        match="energy_min must be less than energy_max",
    )


def test_polarization_config_rejects_unknown_type() -> None:
    """Reject polarization selectors outside the static allowed set."""
    assert_rejects(
        make_polarization_config,
        polarization_type="unknown",
        match="polarization_type must be one of",
    )
