"""Tests for eigenvector projection utilities.

Extended Summary
----------------
Validates ``eigenvector_orbital_weights`` and ``orbital_coefficients``
from :mod:`diffpes.tightb.projections`. Covers the modulus-squared
computation for complex eigenvectors and verifies that
``orbital_coefficients`` is an identity function preserving shape,
dtype, and complex values.

"""

import jax.numpy as jnp
from beartype.typing import Any, Callable
from jaxtyping import Array

from diffpes.tightb import (
    eigenvector_orbital_weights,
    orbital_coefficients,
)


class TestEigenvectorOrbitalWeights:
    """Tests for :func:`diffpes.tightb.projections.eigenvector_orbital_weights`.

    Verifies that the function correctly computes ``|c|^2`` for complex
    eigenvectors and that the result shape and dtype match expectations.

    :see: :func:`~diffpes.tightb.eigenvector_orbital_weights`
    """

    def test_real_eigenvectors_give_squared_values(self) -> None:
        """Verify that real eigenvectors produce their squared magnitudes.

        For purely real input, ``|c|^2 = c^2``. Uses a single k-point,
        two bands, and two orbitals. Asserts the output matches the
        squared input to within 1e-12.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        evecs: Array
        weights: Array
        expected: Array

        evecs = jnp.array([[[0.6, 0.8], [0.8, -0.6]]], dtype=jnp.complex128)
        weights = eigenvector_orbital_weights(evecs)
        expected = jnp.array([[[0.36, 0.64], [0.64, 0.36]]], dtype=jnp.float64)
        assert jnp.allclose(weights, expected, atol=1e-12)

    def test_complex_eigenvectors_give_modulus_squared(self) -> None:
        """Verify complex eigenvectors produce ``|c|^2`` per coefficient.

        Uses (3+4j)/5 which has ``|c|^2 = (9+16)/25 = 1.0``. Checks that
        the
        function computes the squared modulus, not the real part squared.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        c: complex
        evecs: Array
        weights: Array

        c = (3.0 + 4.0j) / 5.0
        evecs = jnp.array([[[c, c], [c, c]]], dtype=jnp.complex128)
        weights = eigenvector_orbital_weights(evecs)

        assert jnp.allclose(weights, 1.0, atol=1e-12)

    def test_output_shape_matches_input(self) -> None:
        """Verify the output shape equals the input shape (K, B, O).

        This case establishes the output shape matches input contract for eigenvector
        orbital weights with the concrete values and array shapes described below.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
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
        """Verify the output dtype is float64 (real), not complex.

        This case establishes the output is real contract for eigenvector orbital
        weights with the concrete values and array shapes described below.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        evecs: Array
        weights: Array

        evecs = jnp.ones((2, 2, 2), dtype=jnp.complex128)
        weights = eigenvector_orbital_weights(evecs)
        assert jnp.issubdtype(weights.dtype, jnp.floating)


class TestOrbitalCoefficients:
    """Tests for :func:`diffpes.tightb.projections.orbital_coefficients`.

    Verifies that the identity function returns its input unchanged,
    preserving complex values, shape, and dtype.

    :see: :func:`~diffpes.tightb.orbital_coefficients`
    """

    def test_identity_preserves_values(self) -> None:
        """Verify that orbital_coefficients returns its input unchanged.

        Constructs a complex eigenvector array and asserts that the
        output is identical to the input (element-wise equality).

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        c: complex
        evecs: Array
        out: Array

        c = 0.5 + 0.3j
        evecs = jnp.array([[[c, 0.0], [0.0, c]]], dtype=jnp.complex128)
        out = orbital_coefficients(evecs)
        assert jnp.allclose(out, evecs, atol=1e-15)

    def test_identity_preserves_shape_and_dtype(self) -> None:
        """Verify that orbital_coefficients preserves shape and complex128 dtype.

        This case establishes the identity preserves shape and dtype contract for
        orbital coefficients with the concrete values and array shapes described below.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
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
