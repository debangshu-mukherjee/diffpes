"""Tests for ARPES math utility functions.

Extended Summary
----------------
Exercises the Faddeeva function, z-score normalization, and the sanctioned
complex-to-stacked-real optimizer boundary. Packing tests cover generic
asymmetric complex data, exact round trips, precision preservation, JIT,
vectorization, real-coordinate gradients, and JAX's pinned Wirtinger
convention.
"""

import chex
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any, Callable
from jaxtyping import Array, Complex, Float, jaxtyped

from diffpes.utils import (
    faddeeva,
    pack_complex,
    unpack_complex,
    zscore_normalize,
)


@jaxtyped(typechecker=beartype)
def _packed_norm_squared(
    packed: Float[Array, " ... 2"],
) -> Float[Array, ""]:
    """Evaluate squared complex magnitude from packed real coordinates."""
    unpacked: Complex[Array, " ..."] = unpack_complex(packed)
    loss: Float[Array, ""] = jnp.sum(jnp.abs(unpacked) ** 2)
    return loss


@jaxtyped(typechecker=beartype)
def _complex_abs_squared(z: Complex[Array, ""]) -> Float[Array, ""]:
    """Evaluate squared magnitude for the Wirtinger convention pin."""
    loss: Float[Array, ""] = jnp.abs(z) ** 2
    return loss


class TestFaddeeva(chex.TestCase):
    """Tests for :func:`diffpes.utils.math.faddeeva`.

    Verifies correctness of the Faddeeva function (w(z) = exp(-z^2) erfc(-iz))
    implementation including evaluation on the real axis, the known value at the
    origin, and evaluation on the imaginary axis. Each test is run both with and
    without JIT compilation to ensure consistent behavior.

    :see: :func:`~diffpes.utils.faddeeva`
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_real_axis(self) -> None:
        """Verify that faddeeva returns finite values with correct shape on the real axis.

        This case establishes the real axis contract for faddeeva with the concrete
        values and array shapes described below.

        Notes
        -----
        1. **Build real-valued input**:
           Create 100 points linearly spaced in [-3, 3] and cast to complex
           by adding 0j.

        2. **Evaluate**:
           Call the faddeeva function (via the chex variant wrapper) on the
           complex array.

        3. **Check output**:
           Assert the result shape is (100,) and that all real parts are
           finite (no NaN or Inf).

        **Expected assertions**

        Output shape matches (100,) and all real components are finite,
        confirming the Taylor-series evaluation is numerically stable on
        the real line within ``|x| <= 3``.
        """
        x: Array
        z: Array
        var_fn: Callable[..., Any]
        w: Array

        x = jnp.linspace(-3.0, 3.0, 100)
        z = x + 0j
        var_fn = self.variant(faddeeva)
        w = var_fn(z)
        chex.assert_shape(w, (100,))
        chex.assert_tree_all_finite(jnp.real(w))

    @chex.variants(with_jit=True, without_jit=True)
    def test_zero(self) -> None:
        """Verify that faddeeva(0) returns approximately 1.0.

        This case establishes the zero contract for faddeeva with the concrete values
        and array shapes described below.

        Notes
        -----
        1. **Build scalar input**:
           Create the complex scalar z = 0 + 0j.

        2. **Evaluate**:
           Call the faddeeva function (via the chex variant wrapper) on the
           scalar input.

        3. **Check known value**:
           The Faddeeva function at the origin satisfies w(0) = erfc(0) = 1.
           Assert that the real part of the result is close to 1.0 within
           an absolute tolerance of 0.05.

        **Expected assertions**

        Real part of w(0) is within 0.05 of 1.0, validating the seed
        coefficient a_0 = 1 of the Taylor expansion.
        """
        z: Array
        var_fn: Callable[..., Any]
        w: Array

        z = jnp.array(0.0 + 0j)
        var_fn = self.variant(faddeeva)
        w = var_fn(z)
        chex.assert_trees_all_close(jnp.real(w), jnp.float64(1.0), atol=0.05)

    @chex.variants(with_jit=True, without_jit=True)
    def test_imaginary_axis(self) -> None:
        """Verify that faddeeva returns finite values with correct shape on the imaginary axis.

        This case establishes the imaginary axis contract for faddeeva with the concrete
        values and array shapes described below.

        Notes
        -----
        1. **Build purely imaginary input**:
           Create an array of three imaginary values z = i * [0.5, 1.0, 2.0].

        2. **Evaluate**:
           Call the faddeeva function (via the chex variant wrapper) on the
           imaginary array.

        3. **Check output**:
           Assert the result shape is (3,) and that all real parts are
           finite (no NaN or Inf).

        **Expected assertions**

        Output shape matches (3,) and all real components are finite,
        confirming numerical stability along the imaginary axis where
        w(iy) grows exponentially as exp(y^2) erfc(y).
        """
        y: Array
        z: Array
        var_fn: Callable[..., Any]
        w: Array

        y = jnp.array([0.5, 1.0, 2.0])
        z = 1j * y
        var_fn = self.variant(faddeeva)
        w = var_fn(z)
        chex.assert_shape(w, (3,))
        chex.assert_tree_all_finite(jnp.real(w))


class TestZscoreNormalize(chex.TestCase):
    """Tests for :func:`diffpes.utils.math.zscore_normalize`.

    Verifies correctness of z-score normalization including the standard case
    (mean becomes 0, std becomes 1), the degenerate constant-input case
    (zero-variance guard), and multi-dimensional input support.

    :see: :func:`~diffpes.utils.zscore_normalize`
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_normalized_stats(self) -> None:
        """Verify that zscore_normalize produces zero mean and unit standard deviation.

        This case establishes the normalized stats contract for zscore normalize with
        the concrete values and array shapes described below.

        Notes
        -----
        1. **Build a simple 1-D array**:
           Create the float64 array [1, 2, 3, 4, 5] with non-zero variance.

        2. **Normalize**:
           Call zscore_normalize (via the chex variant wrapper) on the data.

        3. **Check statistics**:
           Assert the mean of the result is 0.0 and the standard deviation
           is 1.0, both within an absolute tolerance of 1e-10.

        **Expected assertions**

        Output mean is approximately 0 and output std is approximately 1,
        confirming the core z-score transformation (x - mu) / sigma.
        """
        data: Array
        var_fn: Callable[..., Any]
        result: Array

        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64)
        var_fn = self.variant(zscore_normalize)
        result = var_fn(data)
        chex.assert_trees_all_close(
            jnp.mean(result), jnp.float64(0.0), atol=1e-10
        )
        chex.assert_trees_all_close(
            jnp.std(result), jnp.float64(1.0), atol=1e-10
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_constant_input(self) -> None:
        """Verify that zscore_normalize returns all zeros for a constant array.

        This case establishes the constant input contract for zscore normalize with the
        concrete values and array shapes described below.

        Notes
        -----
        1. **Build a constant array**:
           Create a length-10 array of all ones (std = 0).

        2. **Normalize**:
           Call zscore_normalize (via the chex variant wrapper). The function
           should detect the zero standard deviation and use the safe divisor
           of 1.0 to avoid division by zero.

        3. **Check output**:
           Assert the result equals a length-10 zero array within an absolute
           tolerance of 1e-10.

        **Expected assertions**

        Output is all zeros, confirming the zero-variance guard path
        produces the correct degenerate result.
        """
        data: Array
        var_fn: Callable[..., Any]
        result: Array

        data = jnp.ones(10, dtype=jnp.float64)
        var_fn = self.variant(zscore_normalize)
        result = var_fn(data)
        chex.assert_trees_all_close(
            result, jnp.zeros(10, dtype=jnp.float64), atol=1e-10
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_2d_input(self) -> None:
        """Verify that zscore_normalize handles 2-D arrays with global statistics.

        This case establishes the 2d input contract for zscore normalize with the
        concrete values and array shapes described below.

        Notes
        -----
        1. **Build a 2-D array**:
           Create a (3, 4) float array from arange(12) so that the data has
           non-trivial variance across both axes.

        2. **Normalize**:
           Call zscore_normalize (via the chex variant wrapper) on the 2-D
           input. The function computes global (not per-axis) mean and std.

        3. **Check shape and mean**:
           Assert the output shape is preserved as (3, 4) and the global mean
           of the result is approximately 0.0 within an absolute tolerance
           of 1e-10.

        **Expected assertions**

        Output shape is (3, 4) and global mean is approximately 0, confirming
        that zscore_normalize operates element-wise over arbitrary-rank inputs
        using global statistics.
        """
        data: Array
        var_fn: Callable[..., Any]
        result: Array

        data = jnp.arange(12.0).reshape(3, 4)
        var_fn = self.variant(zscore_normalize)
        result = var_fn(data)
        chex.assert_shape(result, (3, 4))
        chex.assert_trees_all_close(
            jnp.mean(result), jnp.float64(0.0), atol=1e-10
        )


class TestPackComplex(chex.TestCase):
    """Validate :func:`~diffpes.utils.math.pack_complex`.

    Covers exact complex round trips, complex128-to-float64 preservation, JIT
    compilation, and vectorization on generic data whose real and imaginary
    magnitudes are deliberately asymmetric.

    :see: :func:`~diffpes.utils.pack_complex`
    """

    def test_round_trip_and_dtype(self) -> None:
        """Preserve generic complex128 values exactly through a JIT round trip.

        The input uses unequal real and imaginary components so an accidental
        component swap or real-symmetric implementation cannot pass.

        Notes
        -----
        Packs a ``(2, 2)`` complex128 array under ``jax.jit``, asserts the
        ``(2, 2, 2)`` float64 representation, and unpacks it under ``jax.jit``
        before checking bitwise equality.
        """
        complex_values: Complex[Array, "2 2"] = jnp.array(
            [[1.0 + 2.0j, -3.0 + 0.5j], [7.0 - 4.0j, 0.25 + 9.0j]],
            dtype=jnp.complex128,
        )
        packed: Float[Array, "2 2 2"] = jax.jit(pack_complex)(complex_values)
        round_tripped: Complex[Array, "2 2"] = jax.jit(unpack_complex)(packed)

        chex.assert_shape(packed, (2, 2, 2))
        chex.assert_equal(packed.dtype, jnp.dtype("float64"))
        chex.assert_trees_all_equal(round_tripped, complex_values)

    def test_vmap(self) -> None:
        """Vectorize packing independently over a leading parameter batch.

        The result must retain batch and parameter axes while appending only
        the two-component packing axis.

        Notes
        -----
        Applies ``jax.vmap`` to three generic two-element complex parameter
        vectors and compares with one direct packing operation exactly.
        """
        complex_values: Complex[Array, "3 2"] = jnp.array(
            [
                [1.0 + 4.0j, 2.0 - 3.0j],
                [-5.0 + 0.25j, 7.0 + 8.0j],
                [9.0 - 2.0j, -1.5 + 6.0j],
            ],
            dtype=jnp.complex128,
        )
        vmapped: Float[Array, "3 2 2"] = jax.vmap(pack_complex)(complex_values)
        direct: Float[Array, "3 2 2"] = pack_complex(complex_values)
        unpacked: Complex[Array, "3 2"] = jax.vmap(unpack_complex)(vmapped)

        chex.assert_shape(vmapped, (3, 2, 2))
        chex.assert_trees_all_equal(vmapped, direct)
        chex.assert_trees_all_equal(unpacked, complex_values)


class TestUnpackComplex(chex.TestCase):
    """Validate :func:`~diffpes.utils.math.unpack_complex`.

    Covers exact stacked-real round trips, float64-to-complex128 preservation,
    and equivalence between complex magnitude gradients and ordinary real
    optimizer-coordinate gradients.

    :see: :func:`~diffpes.utils.unpack_complex`
    """

    def test_round_trip_and_dtype(self) -> None:
        """Preserve generic stacked float64 values exactly through a JIT round trip.

        The final input axis contains deliberately asymmetric real and
        imaginary coordinates to detect ordering errors.

        Notes
        -----
        Unpacks a ``(2, 3, 2)`` float64 array under ``jax.jit``, asserts a
        ``(2, 3)`` complex128 result, and packs it again before checking exact
        coordinate equality.
        """
        packed_values: Float[Array, "2 3 2"] = jnp.array(
            [
                [[1.0, 2.0], [-3.0, 0.5], [7.0, -4.0]],
                [[0.25, 9.0], [6.0, -8.0], [-2.5, 11.0]],
            ],
            dtype=jnp.float64,
        )
        unpacked: Complex[Array, "2 3"] = jax.jit(unpack_complex)(
            packed_values
        )
        round_tripped: Float[Array, "2 3 2"] = jax.jit(pack_complex)(unpacked)

        chex.assert_shape(unpacked, (2, 3))
        chex.assert_equal(unpacked.dtype, jnp.dtype("complex128"))
        chex.assert_trees_all_equal(round_tripped, packed_values)

    def test_gradient_equivalence(self) -> None:
        """Match complex-magnitude gradients to packed real coordinates exactly.

        For ``p = stack([x, y])``, the real optimizer gradient of
        ``sum(abs(unpack_complex(p))**2)`` must equal ``stack([2x, 2y])``.

        Notes
        -----
        Differentiates a JIT-compiled loss on generic float64 coordinates and
        checks exact equality with twice the input, pinning an optax-safe real
        gradient with no Wirtinger bookkeeping at the optimizer boundary.
        """
        packed_values: Float[Array, "3 2"] = jnp.array(
            [[1.0, 2.0], [-3.0, 0.5], [7.0, -4.0]], dtype=jnp.float64
        )
        gradient: Float[Array, "3 2"] = jax.jit(
            jax.grad(_packed_norm_squared)
        )(packed_values)
        expected: Float[Array, "3 2"] = 2.0 * packed_values

        chex.assert_trees_all_equal(gradient, expected)


class TestComplexAutodiffConvention(chex.TestCase):
    """Pin JAX's complex-gradient convention at the packing boundary.

    Guards the exact conjugated Wirtinger convention assumed when complex
    physics values are related to stacked real optimizer coordinates.

    :see: :func:`~diffpes.utils.pack_complex`
    :see: :func:`~diffpes.utils.unpack_complex`
    """

    def test_wirtinger_convention(self) -> None:
        """Pin ``grad(abs(z)**2)`` at ``1+1j`` to exactly ``2-2j``.

        This convention determines how real-loss complex gradients relate to
        gradients of the two stacked real optimizer coordinates.

        Notes
        -----
        Applies reverse-mode autodiff to the scalar complex128 point
        ``1+1j`` and checks bitwise equality with the JAX-cookbook convention
        ``2-2j`` so any upstream convention change fails loudly.
        """
        z: Complex[Array, ""] = jnp.asarray(1.0 + 1.0j, dtype=jnp.complex128)
        gradient: Complex[Array, ""] = jax.grad(_complex_abs_squared)(z)
        expected: Complex[Array, ""] = jnp.asarray(
            2.0 - 2.0j, dtype=jnp.complex128
        )

        chex.assert_trees_all_equal(gradient, expected)
