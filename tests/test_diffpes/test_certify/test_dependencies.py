"""Tests for JAX-native certification dependency analysis."""

import jax
import jax.numpy as jnp

from diffpes.certify.dependencies import (
    dependency_map,
    information_spectrum,
    linearized_forward,
    sensitivity_map,
)


class TestDependencies:
    """Verify structural, local, and spectral information flow."""

    def test_dependency_map_distinguishes_disconnected_leaf(self):
        """Trace only leaves consumed by the output JAXPR."""

        def forward(inputs):
            x, unused = inputs
            del unused
            return x**2

        inputs = (jnp.array([2.0]), jnp.array([7.0]))
        result = dependency_map("org.diffpes.model.test", forward, inputs)
        assert result.structural.shape == (1, 2)
        assert result.structural.tolist() == [[True, False]]
        assert result.traced.tolist() == [[True, False]]

    def test_linearization_reuses_exact_jvp(self):
        """Retain the linear map of a cubic function."""
        value, pushforward = linearized_forward(
            lambda x: x**3, jnp.array([2.0])
        )
        assert jnp.allclose(value, 8.0)
        assert jnp.allclose(pushforward(jnp.ones(1)), 12.0)

    def test_information_spectrum_and_gradient(self):
        """Recover and differentiate the singular value of x squared."""

        def singular(x):
            spectrum = information_spectrum(
                lambda candidate: candidate**2,
                x,
                rank=1,
                iterations=3,
            )
            return spectrum.singular_values[0]

        x = jnp.array([2.0])
        assert jnp.allclose(singular(x), 4.0, rtol=1e-12)
        assert jnp.allclose(jax.grad(singular)(x), 2.0, rtol=1e-9)

    def test_sensitivity_map_has_output_by_input_orientation(self):
        """Store output projections along rows and input probes in columns."""
        inputs = jnp.array([2.0])
        directions = jnp.ones((1, 1))
        result = sensitivity_map(
            ("$[0]",),
            ("square",),
            lambda x: x**2,
            inputs,
            directions,
            jnp.ones(1),
        )
        assert result.sensitivities.shape == (1, 1)
        assert jnp.allclose(result.sensitivities, 4.0)
