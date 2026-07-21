"""Validate the ARPES mathematical utilities.

Extended Summary
----------------
The tests exercise the Faddeeva function, z-score normalization, and the
complex packing boundary. They cover JIT, vectorization, precision, round
trips, and gradients.
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
    """Compute squared complex magnitude from packed real coordinates.

    Parameters
    ----------
    packed : Float[Array, " ... 2"]
        Packed real and imaginary coordinates.

    Returns
    -------
    loss : Float[Array, ""]
        Sum of the squared complex magnitudes.

    Notes
    -----
    The helper unpacks the coordinates and sums ``abs(z)**2`` for the
    gradient test.
    """
    unpacked: Complex[Array, " ..."] = unpack_complex(packed)
    loss: Float[Array, ""] = jnp.sum(jnp.abs(unpacked) ** 2)
    return loss


@jaxtyped(typechecker=beartype)
def _complex_abs_squared(z: Complex[Array, ""]) -> Float[Array, ""]:
    """Compute squared magnitude for the Wirtinger convention test.

    Parameters
    ----------
    z : Complex[Array, ""]
        Complex scalar under test.

    Returns
    -------
    loss : Float[Array, ""]
        Squared magnitude of ``z``.

    Notes
    -----
    The helper supplies a real scalar loss to ``jax.grad``.
    """
    loss: Float[Array, ""] = jnp.abs(z) ** 2
    return loss


class TestFaddeeva(chex.TestCase):
    """Validate :func:`diffpes.utils.math.faddeeva`.

    The tests cover both coordinate axes and the analytic value at the origin.
    Each test uses compiled and uncompiled execution.

    :see: :func:`~diffpes.utils.faddeeva`
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_real_axis(self) -> None:
        """Verify ``faddeeva`` returns finite values on the real axis.

        The function preserves the input shape and produces finite real
        components for the specified interval.

        Notes
        -----
        The test uses 100 complex points across ``[-3, 3]``. It checks shape
        ``(100,)`` and the finite real components under both JAX variants.
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
        """Verify ``faddeeva(0)`` returns approximately 1.0.

        The Faddeeva function satisfies ``w(0) = erfc(0) = 1``.

        Notes
        -----
        The test supplies the complex scalar zero. It compares the real
        component with 1.0 at an absolute tolerance of 0.05 under both variants.
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
        """Verify ``faddeeva`` returns finite values on the imaginary axis.

        The function preserves the input shape and produces finite real
        components for three imaginary inputs.

        Notes
        -----
        The test uses ``i*[0.5, 1.0, 2.0]``. It checks shape ``(3,)`` and
        the finite real components under both JAX variants.
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
    """Validate :func:`diffpes.utils.math.zscore_normalize`.

    The tests cover standard data, constant data, and two-dimensional data.
    They verify the global normalization and the zero-variance guard.

    :see: :func:`~diffpes.utils.zscore_normalize`
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_normalized_stats(self) -> None:
        """Verify ``zscore_normalize`` produces zero mean and unit deviation.

        Standard z-score normalization produces a mean of zero and a standard
        deviation of one.

        Notes
        -----
        The test normalizes ``[1, 2, 3, 4, 5]``. It compares both statistics
        with their analytic values at an absolute tolerance of 1e-10.
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
        """Verify ``zscore_normalize`` returns zeros for a constant array.

        The zero-variance guard produces a finite zero array.

        Notes
        -----
        The test normalizes an array of ten ones under both JAX variants.
        It compares the result with ten zeros at an absolute tolerance of 1e-10.
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
        """Verify ``zscore_normalize`` handles arrays with two dimensions.

        The function preserves the two-dimensional shape and uses global
        statistics across both axes.

        Notes
        -----
        The test normalizes ``arange(12)`` with shape ``(3, 4)``. It checks the
        shape and compares the global mean with zero at a tolerance of 1e-10.
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

    The tests cover exact round trips, dtype preservation, JIT, and
    vectorization. Asymmetric complex values reveal an incorrect coordinate order.

    :see: :func:`~diffpes.utils.pack_complex`
    """

    def test_round_trip_and_dtype(self) -> None:
        """Preserve generic complex128 values exactly through a JIT round trip.

        The input uses unequal real and imaginary components. These values
        reveal a component swap or a real-symmetric implementation.

        Notes
        -----
        The test packs and unpacks a ``(2, 2)`` array with ``jax.jit``. It
        checks the packed shape, both dtypes, and exact value equality.
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

        The result retains the batch and parameter axes. It appends only the
        two-component packing axis.

        Notes
        -----
        The test applies ``jax.vmap`` to three complex parameter vectors. It
        compares the result with direct packing and checks the exact round trip.
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

    The tests cover exact round trips and dtype preservation. They also compare
    complex magnitude gradients with gradients in real coordinates.

    :see: :func:`~diffpes.utils.unpack_complex`
    """

    def test_round_trip_and_dtype(self) -> None:
        """Preserve generic stacked float64 values exactly through a JIT round trip.

        The final input axis contains asymmetric real and imaginary coordinates.
        These values reveal an incorrect coordinate order.

        Notes
        -----
        The test unpacks and packs a ``(2, 3, 2)`` array with ``jax.jit``.
        It checks the unpacked shape, both dtypes, and exact coordinate equality.
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

        For ``p = stack([x, y])``, the real gradient equals
        ``stack([2x, 2y])``.

        Notes
        -----
        The test differentiates a compiled loss on generic float64 coordinates.
        It compares the gradient with twice the input by exact equality.
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

    The tests fix the conjugated Wirtinger convention at the boundary between
    complex physics values and real optimizer coordinates.

    :see: :func:`~diffpes.utils.pack_complex`
    :see: :func:`~diffpes.utils.unpack_complex`
    """

    def test_wirtinger_convention(self) -> None:
        """Pin ``grad(abs(z)**2)`` at ``1+1j`` to exactly ``2-2j``.

        The convention determines the relation between complex gradients and
        gradients of stacked real coordinates.

        Notes
        -----
        The test applies reverse-mode autodiff at ``1+1j``. It checks exact
        equality with ``2-2j`` to detect a change in the JAX convention.
        """
        z: Complex[Array, ""] = jnp.asarray(1.0 + 1.0j, dtype=jnp.complex128)
        gradient: Complex[Array, ""] = jax.grad(_complex_abs_squared)(z)
        expected: Complex[Array, ""] = jnp.asarray(
            2.0 - 2.0j, dtype=jnp.complex128
        )

        chex.assert_trees_all_equal(gradient, expected)
