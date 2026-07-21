"""Validate momentum resolution broadening.

Extended Summary
----------------
Apply :func:`diffpes.simul.resolution.apply_momentum_broadening` to a 2D
ARPES intensity map. Test the identity limit, smoothing, conservation,
shape, and gradients.

"""

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from diffpes.simul import apply_momentum_broadening


class TestApplyMomentumBroadening:
    """Validate :func:`diffpes.simul.resolution.apply_momentum_broadening`.

    Validate Gaussian momentum broadening along the k-axis. Verify the
    identity limit, smoothing, conservation, shape, and JAX gradients.

    :see: :func:`~diffpes.simul.apply_momentum_broadening`
    """

    def test_identity_with_zero_dk(self) -> None:
        """Verify that vanishing dk returns approximately the original intensity.

        The test establishes the identity with zero dk contract for apply momentum
        broadening with the concrete values and array shapes described below.

        Notes
        -----
        1. **Setup**: Create a uniform intensity map of shape (20, 50)
           and a linearly spaced k_distances array from 0 to 1. Set
           dk = 1e-15 so that the kernel approaches a delta function.
        2. **Apply broadening**: Call ``apply_momentum_broadening`` with
           the near-zero dk.
        3. **Compare**: Assert the result is element-wise close to the
           original intensity within atol=1e-3.

        **Expected assertions**

        The broadened intensity equals the input to within 1e-3,
        confirming the correct identity limit when the broadening
        width vanishes.
        """
        K: int
        E: int
        intensity: Array
        k_distances: Array
        result: Array

        K, E = 20, 50
        intensity = jnp.ones((K, E))
        k_distances = jnp.linspace(0, 1, K)

        result = apply_momentum_broadening(intensity, k_distances, 1e-15)
        assert jnp.allclose(result, intensity, atol=1e-3)

    def test_smoothing_effect(self) -> None:
        """Verify that finite dk smooths a delta-like peak along the k-axis.

        The test establishes the smoothing effect contract for apply momentum
        broadening with the concrete values and array shapes described below.

        Notes
        -----
        1. **Setup**: Create a zero intensity map of shape (50, 10).
           Set k-index 25 to 1.0 across all energy channels.
        2. **Apply broadening**: Call ``apply_momentum_broadening`` with
           dk=0.1, which should spread the peak to neighbouring
           k-points via the Gaussian kernel.
        3. **Check peak reduction**: Assert that the peak value at index
           25 falls below 1.0.
        4. **Check neighbour activation**: Assert that indices 24 and 26
           have positive intensity.

        **Expected assertions**

        The peak decreases and its neighbours gain intensity. Thus, the
        Gaussian convolution smooths sharp k-space features.
        """
        K: int
        E: int
        intensity: Array
        k_distances: Array
        result: Array

        K, E = 50, 10

        intensity = jnp.zeros((K, E))
        intensity = intensity.at[25, :].set(1.0)
        k_distances = jnp.linspace(0, 1, K)

        result = apply_momentum_broadening(intensity, k_distances, 0.1)

        assert float(result[25, 0]) < 1.0

        assert float(result[24, 0]) > 0.0
        assert float(result[26, 0]) > 0.0

    def test_conservation(self) -> None:
        """Verify approximate conservation of intensity under broadening.

        The test establishes the conservation contract for apply momentum broadening
        with the concrete values and array shapes described below.

        Notes
        -----
        1. **Setup**: Create a 2D intensity map (30 x 20) with a
           spatially varying (sinusoidal) k-profile broadcast across
           all energy channels. Use dk=0.05 for moderate broadening.
        2. **Apply broadening**: Call ``apply_momentum_broadening``.
        3. **Compare totals**: Assert the summed intensity before and
           after broadening agree within 10% relative tolerance.

        **Expected assertions**

        The total intensity changes by less than 10%. Grid boundaries
        truncate the Gaussian kernel and prevent exact conservation.
        """
        K: int
        E: int
        intensity: Array
        k_distances: Array
        result: Array

        K, E = 30, 20
        intensity = jnp.abs(
            jnp.sin(jnp.linspace(0, 3, K))[:, None]
        ) * jnp.ones((1, E))
        k_distances = jnp.linspace(0, 1, K)
        result = apply_momentum_broadening(intensity, k_distances, 0.05)
        assert float(jnp.sum(result)) == pytest.approx(
            float(jnp.sum(intensity)), rel=0.1
        )

    def test_output_shape(self) -> None:
        """Verify that the output shape matches the input shape.

        The test establishes the output shape contract for apply momentum broadening
        with the concrete values and array shapes described below.

        Notes
        -----
        1. **Setup**: Create a uniform intensity map of shape (15, 25)
           and apply momentum broadening with dk=0.1.
        2. **Check shape**: Assert the output shape is ``(15, 25)``,
           identical to the input.

        **Expected assertions**

        The output shape equals the input shape ``(K, E)``, confirming
        that the convolution does not alter the grid dimensions.
        """
        K: int
        E: int
        intensity: Array
        k_distances: Array
        result: Array

        K, E = 15, 25
        intensity = jnp.ones((K, E))
        k_distances = jnp.linspace(0, 1, K)
        result = apply_momentum_broadening(intensity, k_distances, 0.1)
        assert result.shape == (K, E)

    def test_gradient_wrt_dk(self) -> None:
        """Verify that the gradient of total intensity w.r.t. dk is finite.

        The test establishes the gradient wrt dk contract for apply momentum broadening
        with the concrete values and array shapes described below.

        Notes
        -----
        1. **Setup**: Create a uniform intensity map of shape (10, 5).
           Define a loss that sums the broadened intensity.
        2. **Differentiate**: Call ``jax.grad(loss)(0.1)`` to compute
           the gradient of the loss with respect to dk.
        3. **Check finiteness**: Assert the gradient is finite.

        **Expected assertions**

        The gradient with respect to dk is finite. This result supports
        inverse fitting with dk as a learnable instrument parameter.
        """
        K: int
        E: int
        intensity: Array
        k_distances: Array
        grad: Array

        K, E = 10, 5
        intensity = jnp.ones((K, E))
        k_distances = jnp.linspace(0, 1, K)

        def loss(dk):
            return jnp.sum(
                apply_momentum_broadening(intensity, k_distances, dk)
            )

        grad = jax.grad(loss)(jnp.array(0.1))
        assert jnp.isfinite(grad)
