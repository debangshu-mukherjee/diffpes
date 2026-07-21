"""Validate radial integrals.

Extended Summary
----------------
The tests validate ``radial_integral`` for a Slater radial wavefunction
and a spherical Bessel function. They compare the ``l'=0`` numerical
integral with an analytical result for Slater 1s orbitals. They also compare
the ``zeta`` autodiff gradient with a central finite difference.

"""

import chex
import jax
import jax.numpy as jnp
from beartype.typing import Any, Callable
from jaxtyping import Array

from diffpes.radial import radial_integral, slater_radial


class TestRadialIntegral(chex.TestCase):
    """Validate radial-integral values and derivatives.

    The tests compare the numerical radial overlap integral with analytical
    results for Slater-type orbitals. They also compare
    JAX autodiff gradients for ``zeta`` with finite differences.

    :see: :func:`~diffpes.radial.radial_integral`
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_l0_slater_matches_analytic_integral(self) -> None:
        """Verify the l'=0 radial integral matches the analytical Fourier transform.

        The test uses a Slater 1s orbital with ``zeta=1.2``. Its
        ``j_0(kr)`` overlap integral has a closed-form result.
        The test uses a dense 25000-point grid up to r=50 Bohr and test k-values
        [0.2, 0.8, 1.4].  Asserts the real part of the numerical integral
        agrees with the analytical expression to within 5e-3 in both
        absolute and relative tolerance.  Run under both JIT and eager
        modes via ``chex.variants``.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        zeta: float
        r: Array
        radial: Array
        k: Array
        fn: Callable[..., Any]
        numeric: Array
        norm: Array
        expected: Array

        zeta = 1.2
        r = jnp.linspace(0.0, 50.0, 25000, dtype=jnp.float64)
        radial = slater_radial(r, n=1, zeta=zeta)
        k = jnp.array([0.2, 0.8, 1.4], dtype=jnp.float64)
        fn = self.variant(
            lambda kvals: radial_integral(kvals, r, radial, l_prime=0)
        )

        numeric = jnp.real(fn(k))
        norm = ((2.0 * zeta) ** 1.5) / jnp.sqrt(2.0)
        expected = (
            2.0 * norm * (3.0 * zeta**2 - k**2) / ((zeta**2 + k**2) ** 3)
        )
        chex.assert_trees_all_close(
            numeric, expected, atol=5.0e-3, rtol=5.0e-3
        )

    def test_gradient_wrt_zeta_matches_finite_difference(self) -> None:
        """Verify autodiff gradient w.r.t. zeta matches finite differences.

        The test defines the real part of the ``l'=0`` radial integral as
        a scalar objective. It uses a Slater 1s orbital at ``k=0.9`` and
        ``zeta=1.1``. The test differentiates with ``jax.grad`` and compares with
        a central finite-difference estimate with step eps=5e-4.  Uses
        an 18000-point grid up to r=45 Bohr.  Asserts agreement to
        within ``5e-3`` for both tolerances. This result confirms smooth
        derivatives through the integration, radial construction, and Bessel
        evaluation.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        r: Array
        k: Array
        zeta0: Array
        eps: Array
        grad_auto: Array
        fd: Array

        r = jnp.linspace(0.0, 45.0, 18000, dtype=jnp.float64)
        k = jnp.asarray(0.9, dtype=jnp.float64)
        zeta0 = jnp.asarray(1.1, dtype=jnp.float64)
        eps = jnp.asarray(5.0e-4, dtype=jnp.float64)

        def objective(zeta: chex.Numeric) -> chex.Array:
            radial: Array
            value: Array

            radial = slater_radial(r, n=1, zeta=jnp.asarray(zeta))
            value = radial_integral(k, r, radial, l_prime=0)
            return jnp.real(value)

        grad_auto = jax.grad(objective)(zeta0)
        fd = (objective(zeta0 + eps) - objective(zeta0 - eps)) / (2.0 * eps)
        chex.assert_trees_all_close(grad_auto, fd, atol=5.0e-3, rtol=5.0e-3)


class TestRadialIntegrateErrors:
    """Validate invalid input handling in radial_integral.

    Validates that ``radial_integral`` raises ``ValueError`` for
    negative ``l_prime`` values.

    :see: :func:`~diffpes.radial.radial_integral`
    """

    def test_negative_l_prime_raises(self) -> None:
        """Verify that l_prime < 0 raises ValueError.

        The test calls ``radial_integral`` with ``l_prime=-1`` and expects a
        ``ValueError`` that matches "non-negative". This input covers the
        guard at the top of the function.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        r: Array
        radial: Array
        k: Array

        import jax.numpy as jnp
        import pytest

        from diffpes.radial import radial_integral, slater_radial

        r = jnp.linspace(0.0, 10.0, 100, dtype=jnp.float64)
        radial = slater_radial(r, n=1, zeta=1.0)
        k = jnp.array([1.0], dtype=jnp.float64)
        with pytest.raises(ValueError, match="non-negative"):
            radial_integral(k, r, radial, l_prime=-1)
