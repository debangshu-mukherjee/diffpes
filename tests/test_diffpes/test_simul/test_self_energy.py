"""Validate energy-dependent self-energy evaluation.

Extended Summary
----------------
Validates :func:`diffpes.simul.self_energy.evaluate_self_energy`, which
computes lifetime broadening as a function of binding energy. Exercise
constant, polynomial, and tabulated modes. Verify their values and JAX
gradients.

"""

import jax
import jax.numpy as jnp
import pytest
from beartype.typing import Any, Callable
from jaxtyping import Array

import diffpes
from diffpes.simul import evaluate_self_energy
from diffpes.types import make_self_energy_config


class TestEvaluateSelfEnergy:
    """Validate :func:`diffpes.simul.self_energy.evaluate_self_energy`.

    Validates the three self-energy evaluation modes -- constant, polynomial,
    and tabulated -- by checking exact functional values at known energy
    points. Also verifies JAX differentiability of the constant and
    polynomial modes by asserting that ``jax.grad`` produces finite
    gradients with respect to the self-energy coefficients.

    :see: :func:`~diffpes.simul.evaluate_self_energy`
    """

    def test_constant_mode(self) -> None:
        """Verify that constant mode returns the same gamma at every energy.

        The test establishes the constant mode contract for evaluate self energy with
        the concrete values and array shapes described below.

        Notes
        -----
        1. **Setup**: Create a self-energy config with mode="constant"
           and gamma=0.15 eV. Evaluate at 100 energy points spanning
           [-3, 1] eV.
        2. **Check**: Assert all 100 returned gamma values equal 0.15.

        **Expected assertions**

        Every output element equals 0.15. Thus, constant mode returns a
        uniform broadening width.
        """
        config: diffpes.types.SelfEnergyConfig
        energy: Array
        gamma: Array

        config = make_self_energy_config(gamma=0.15, mode="constant")
        energy = jnp.linspace(-3, 1, 100)
        gamma = evaluate_self_energy(energy, config)
        assert jnp.allclose(gamma, 0.15)

    def test_polynomial_mode(self) -> None:
        """Verify that polynomial mode evaluates gamma(E) = c0 + c1*E correctly.

        The test establishes the polynomial mode contract for evaluate self energy with
        the concrete values and array shapes described below.

        Notes
        -----
        1. **Setup**: Create a polynomial configuration with coefficients
           [0.05, 0.1]. Evaluate gamma(E) = 0.05*E + 0.1 at three points.
        2. **Check**: Assert the returned values match the analytically
           expected [0.1, 0.15, 0.05] to within atol=1e-10.

        **Expected assertions**

        The results match the analytic values at three energies. The
        coefficient order follows ``jnp.polyval`` semantics.
        """
        config: diffpes.types.SelfEnergyConfig
        energy: Array
        gamma: Array
        expected: Array

        config = make_self_energy_config(
            mode="polynomial",
            coefficients=jnp.array([0.05, 0.1]),
        )
        energy = jnp.array([0.0, 1.0, -1.0])
        gamma = evaluate_self_energy(energy, config)
        expected = jnp.array([0.1, 0.15, 0.05])
        assert jnp.allclose(gamma, expected, atol=1e-10)

    def test_tabulated_mode(self) -> None:
        """Verify that tabulated mode interpolates gamma between energy nodes.

        The test establishes the tabulated mode contract for evaluate self energy with
        the concrete values and array shapes described below.

        Notes
        -----
        1. **Setup**: Create a self-energy config with mode="tabulated",
           energy_nodes=[-3.0, 0.0, 1.0] and coefficients=[0.05, 0.1, 0.2].
           Evaluate at E = [0.0, 0.5]. The first point is a node, and
           the second point is midway between two nodes.
        2. **Check exact node**: Assert gamma(0.0) = 0.1 (direct lookup).
        3. **Check interpolated point**: Assert gamma(0.5) = 0.15, the
           linear interpolation midpoint between 0.1 and 0.2.

        **Expected assertions**

        The function returns exact node values and correct interpolated
        values between nodes.
        """
        nodes: Array
        coeffs: Array
        config: diffpes.types.SelfEnergyConfig
        energy: Array
        gamma: Array

        nodes = jnp.array([-3.0, 0.0, 1.0])
        coeffs = jnp.array([0.05, 0.1, 0.2])
        config = make_self_energy_config(
            mode="tabulated",
            coefficients=coeffs,
            energy_nodes=nodes,
        )
        energy = jnp.array([0.0, 0.5])
        gamma = evaluate_self_energy(energy, config)
        assert float(gamma[0]) == pytest.approx(0.1, abs=1e-10)
        assert float(gamma[1]) == pytest.approx(0.15, abs=1e-10)

    def test_constant_gradient(self) -> None:
        """Verify that the gradient w.r.t. constant self-energy coefficient is finite.

        The test establishes the constant gradient contract for evaluate self energy
        with the concrete values and array shapes described below.

        Notes
        -----
        1. **Setup**: Define a loss from one constant coefficient.
           Evaluate two energies and sum their gamma values.
        2. **Differentiate**: Call ``jax.grad(loss)(0.1)`` to compute
           the gradient w.r.t. the constant coefficient.
        3. **Check finiteness**: Assert the gradient is finite.

        **Expected assertions**

        The gradient w.r.t. the constant self-energy coefficient is
        finite, confirming that the constant evaluation path is
        differentiable through JAX and suitable for inverse fitting.
        """
        grad: Array

        def loss(coeff):
            config: diffpes.types.SelfEnergyConfig
            energy: Array

            config = make_self_energy_config(
                mode="constant",
                coefficients=jnp.array([coeff]),
            )
            energy = jnp.array([0.0, 1.0])
            return jnp.sum(evaluate_self_energy(energy, config))

        grad = jax.grad(loss)(jnp.array(0.1))
        assert jnp.isfinite(grad)

    def test_polynomial_gradient(self) -> None:
        """Verify that gradients w.r.t. polynomial coefficients are finite.

        The test establishes the polynomial gradient contract for evaluate self energy
        with the concrete values and array shapes described below.

        Notes
        -----
        1. **Setup**: Define a polynomial loss from coefficients [c1,
           c0]. Evaluate 50 energies and sum their gamma values.
        2. **Differentiate**: Call ``jax.grad(loss)`` with initial
           coefficients [0.01, 0.1] to compute the gradient vector.
        3. **Check finiteness**: Assert all gradient components are
           finite.

        **Expected assertions**

        All gradient components w.r.t. the polynomial coefficients are
        finite, confirming that the polynomial evaluation via
        ``jnp.polyval`` is differentiable through JAX. This enables
        fitting energy-dependent self-energy models to experimental
        linewidths.
        """
        grad: Array

        def loss(coeffs):
            config: diffpes.types.SelfEnergyConfig
            energy: Array

            config = make_self_energy_config(
                mode="polynomial",
                coefficients=coeffs,
            )
            energy = jnp.linspace(-1, 1, 50)
            return jnp.sum(evaluate_self_energy(energy, config))

        grad = jax.grad(loss)(jnp.array([0.01, 0.1]))
        assert jnp.all(jnp.isfinite(grad))


class TestSelfEnergyErrors:
    """Validate invalid mode handling in evaluate_self_energy.

    Validates that ``evaluate_self_energy`` raises ``ValueError`` when
    given a ``SelfEnergyConfig`` with an unsupported mode string.

    :see: :func:`~diffpes.simul.evaluate_self_energy`
    """

    def test_unknown_mode_raises(self) -> None:
        """Verify that an unknown self-energy mode raises ValueError.

        Construct a ``SelfEnergyConfig`` with mode="bad_mode". Call
        ``evaluate_self_energy`` and require ``ValueError`` with the
        expected message.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        config: diffpes.types.SelfEnergyConfig
        energy: Array

        from diffpes.types import SelfEnergyConfig

        config = SelfEnergyConfig(
            coefficients=jnp.array([0.1]),
            energy_nodes=None,
            mode="bad_mode",
        )
        energy = jnp.array([0.0, 1.0])
        with pytest.raises(ValueError, match="Unknown self-energy mode"):
            evaluate_self_energy(energy, config)
