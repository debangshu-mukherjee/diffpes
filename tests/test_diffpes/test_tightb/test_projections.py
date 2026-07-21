"""Validate the eigenvector projection utilities.

Extended Summary
----------------
The tests validate ``eigenvector_orbital_weights`` and
``orbital_coefficients`` from :mod:`diffpes.tightb.projections`.
They cover complex orbital weights and the identity operation for orbital
coefficients.

"""

import jax.numpy as jnp
from beartype.typing import Any, Callable
from jaxtyping import Array

from diffpes.tightb import (
    eigenvector_orbital_weights,
    orbital_coefficients,
)


class TestEigenvectorOrbitalWeights:
    """Validate :func:`diffpes.tightb.projections.eigenvector_orbital_weights`.

    The tests verify the squared magnitude of each complex coefficient.
    They also verify the shape and the real dtype of the result.

    :see: :func:`~diffpes.tightb.eigenvector_orbital_weights`
    """

    def test_real_eigenvectors_give_squared_values(self) -> None:
        """Verify real eigenvectors produce their squared magnitudes.

        For a real coefficient, ``|c|^2 = c^2``. The expected weights have
        values 0.36 and 0.64.

        Notes
        -----
        The test uses one k-point, two bands, and two orbitals. It compares
        the weights with the squared input at an absolute tolerance of 1e-12.
        """
        evecs: Array
        weights: Array
        expected: Array

        evecs = jnp.array([[[0.6, 0.8], [0.8, -0.6]]], dtype=jnp.complex128)
        weights = eigenvector_orbital_weights(evecs)
        expected = jnp.array([[[0.36, 0.64], [0.64, 0.36]]], dtype=jnp.float64)
        assert jnp.allclose(weights, expected, atol=1e-12)

    def test_complex_eigenvectors_give_modulus_squared(self) -> None:
        """Verify complex eigenvectors produce ``|c|^2`` per coefficient.

        The coefficient ``(3+4j)/5`` has a squared magnitude of 1.0.
        The expected result distinguishes the squared magnitude from the
        squared real part.

        Notes
        -----
        The test fills a two-band array with the complex coefficient. It
        compares every weight with 1.0 at an absolute tolerance of 1e-12.
        """
        c: complex
        evecs: Array
        weights: Array

        c = (3.0 + 4.0j) / 5.0
        evecs = jnp.array([[[c, c], [c, c]]], dtype=jnp.complex128)
        weights = eigenvector_orbital_weights(evecs)

        assert jnp.allclose(weights, 1.0, atol=1e-12)

    def test_output_shape_matches_input(self) -> None:
        """Verify the output shape equals the input shape (K, B, O).

        The weight function preserves every axis of the eigenvector array.

        Notes
        -----
        The test supplies an array with shape ``(3, 4, 5)``. It compares the
        output shape with the input shape.
        """
        K: int
        B: int
        O: int
        evecs: Array
        weights: Array

        K, B, O = 3, 4, 5
        evecs = jnp.ones((K, B, O), dtype=jnp.complex128)
        weights = eigenvector_orbital_weights(evecs)
        assert weights.shape == (K, B, O)

    def test_output_is_real(self) -> None:
        """Verify squared magnitudes have a float64 dtype.

        Squared magnitudes of complex coefficients have a real dtype.

        Notes
        -----
        The test supplies a complex128 array. It checks that the result has a
        floating-point dtype.
        """
        evecs: Array
        weights: Array

        evecs = jnp.ones((2, 2, 2), dtype=jnp.complex128)
        weights = eigenvector_orbital_weights(evecs)
        assert jnp.issubdtype(weights.dtype, jnp.floating)


class TestOrbitalCoefficients:
    """Validate :func:`diffpes.tightb.projections.orbital_coefficients`.

    The tests verify that the identity function preserves the complex values,
    shape, and dtype of its input.

    :see: :func:`~diffpes.tightb.orbital_coefficients`
    """

    def test_identity_preserves_values(self) -> None:
        """Verify ``orbital_coefficients`` returns its input unchanged.

        The function preserves each complex orbital coefficient.

        Notes
        -----
        The test supplies a complex128 eigenvector array. It compares the
        output and input at an absolute tolerance of 1e-15.
        """
        c: complex
        evecs: Array
        out: Array

        c = 0.5 + 0.3j
        evecs = jnp.array([[[c, 0.0], [0.0, c]]], dtype=jnp.complex128)
        out = orbital_coefficients(evecs)
        assert jnp.allclose(out, evecs, atol=1e-15)

    def test_identity_preserves_shape_and_dtype(self) -> None:
        """Verify ``orbital_coefficients`` preserves the shape and complex128 dtype.

        The function retains all eigenvector axes and the complex dtype.

        Notes
        -----
        The test supplies a complex128 array with shape ``(2, 3, 4)``. It
        compares the output shape and dtype with the input properties.
        """
        K: int
        B: int
        O: int
        evecs: Array
        out: Array

        K, B, O = 2, 3, 4
        evecs = jnp.zeros((K, B, O), dtype=jnp.complex128)
        out = orbital_coefficients(evecs)
        assert out.shape == (K, B, O)
        assert out.dtype == jnp.complex128
