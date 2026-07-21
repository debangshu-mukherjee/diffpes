"""Validate radial wavefunction models.

Extended Summary
----------------
The tests validate the ``slater_radial`` and ``hydrogenic_radial``
constructors. The Slater tests verify normalization and compare autodiff
gradients with finite differences. The hydrogenic tests compare the 1s and
2p radial functions with analytical expressions. They also verify the
boundary condition ``R_{2p}(0) = 0``.

"""

import chex
import jax
import jax.numpy as jnp
import pytest
from beartype.typing import Any, Callable
from jaxtyping import Array

from diffpes.radial import hydrogenic_radial, slater_radial


class TestSlaterRadial(chex.TestCase):
    """Validate Slater radial normalization and autodiff gradients.

    The tests verify the normalization of the Slater-type orbital.
    They also compare the ``jax.grad`` result for ``zeta`` with central
    finite differences.

    :see: :func:`~diffpes.radial.slater_radial`
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_normalization(self) -> None:
        """Verify normalization of the Slater 2s orbital to unity.

        The test constructs ``R(r)`` for ``n=2`` and ``zeta=1.3`` on a
        20000-point grid through 30 Bohr. It integrates ``|R(r)|^2 * r^2`` with
        trapezoidal rule.  Asserts the integral is within 2e-3 of 1.0.
        The dense grid and large cutoff ensure the exponential tail
        contributes negligibly.  Run under both JIT and eager modes.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        r: Array
        fn: Callable[..., Any]
        radial: Array
        norm: Array

        r = jnp.linspace(0.0, 30.0, 20000, dtype=jnp.float64)
        fn = self.variant(lambda radius: slater_radial(radius, n=2, zeta=1.3))
        radial = fn(r)
        norm = jnp.trapezoid((radial**2) * (r**2), x=r)
        chex.assert_trees_all_close(norm, jnp.asarray(1.0), atol=2.0e-3)

    def test_gradient_wrt_zeta_matches_finite_difference(self) -> None:
        """Verify autodiff gradient of Slater sum w.r.t. zeta matches FD.

        The test defines a scalar objective = sum(R(r; zeta)) for n=2, zeta=1.15
        on a 500-point grid up to r=8 Bohr.  Differentiates with
        ``jax.grad`` and compares against a central finite-difference
        estimate with step eps=1e-4.  Asserts agreement to within 2e-4
        (atol and rtol), confirming the normalization constant, power-law
        prefactor, and exponential are all smoothly differentiable.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        r: Array
        zeta0: Array
        eps: Array
        grad_auto: Array
        fd: Array

        r = jnp.linspace(0.0, 8.0, 500, dtype=jnp.float64)
        zeta0 = jnp.asarray(1.15, dtype=jnp.float64)
        eps = jnp.asarray(1.0e-4, dtype=jnp.float64)

        def objective(zeta: chex.Numeric) -> chex.Array:
            return jnp.sum(slater_radial(r, n=2, zeta=jnp.asarray(zeta)))

        grad_auto = jax.grad(objective)(zeta0)
        fd = (objective(zeta0 + eps) - objective(zeta0 - eps)) / (2.0 * eps)
        chex.assert_trees_all_close(grad_auto, fd, atol=2.0e-4, rtol=2.0e-4)


class TestHydrogenicRadial(chex.TestCase):
    """Validate hydrogenic radial wavefunctions against analytical expressions.

    The tests compare ``hydrogenic_radial`` for hydrogen with closed-form
    expressions. They verify the 1s ground state and the 2p boundary condition.
    All radial functions with ``l > 0`` vanish at the origin.

    :see: :func:`~diffpes.radial.hydrogenic_radial`
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_1s_matches_analytic_expression(self) -> None:
        """Verify R_{10}(r) = 2*exp(-r) for the hydrogen 1s orbital.

        The test evaluates the hydrogenic radial function for ``n=1``,
        ``l=0``, and ``Z_eff=1``. It uses four radii from 0.0 to 2.5 Bohr.
        It compares the result with the analytical 1s expression.
        The test asserts element-wise agreement to within 1e-10.  The r=0 point
        tests the boundary condition R_{10}(0) = 2, and the larger r
        values test the exponential decay.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        r: Array
        fn: Callable[..., Any]
        expected: Array

        r = jnp.array([0.0, 0.3, 1.0, 2.5], dtype=jnp.float64)
        fn = self.variant(
            lambda radius: hydrogenic_radial(
                radius,
                n=1,
                angular_momentum=0,
                z_eff=1.0,
            )
        )
        expected = 2.0 * jnp.exp(-r)
        chex.assert_trees_all_close(fn(r), expected, atol=1.0e-10)

    @chex.variants(with_jit=True, without_jit=True)
    def test_2p_is_zero_at_origin(self) -> None:
        """Verify the 2p (n=2, l=1) radial function vanishes at r=0.

        The test evaluates R_{21}(r=0) for Z_eff=1.  All hydrogenic radial
        functions with l > 0 contain a factor r^l and therefore must
        vanish at the origin.  Asserts the output is zero to within
        1e-12, testing this critical boundary condition / edge case.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        fn: Callable[..., Any]
        value_at_origin: Array

        fn = self.variant(
            lambda radius: hydrogenic_radial(
                radius,
                n=2,
                angular_momentum=1,
                z_eff=1.0,
            )
        )
        value_at_origin = fn(jnp.asarray([0.0], dtype=jnp.float64))
        chex.assert_trees_all_close(
            value_at_origin,
            jnp.asarray([0.0], dtype=jnp.float64),
            atol=1.0e-12,
        )


class TestSlaterRadialErrors:
    """Validate invalid input handling in slater_radial.

    Validates that ``slater_radial`` raises ``ValueError`` when the
    principal quantum number ``n`` is less than 1.

    :see: :func:`~diffpes.radial.slater_radial`
    """

    def test_n_zero_raises(self) -> None:
        """Verify that n=0 raises ValueError.

        The test calls ``slater_radial`` with ``n=0`` and expects a
        ``ValueError`` that matches "n must be >= 1".

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        r: Array

        r = jnp.array([1.0], dtype=jnp.float64)
        with pytest.raises(ValueError, match="n must be >= 1"):
            slater_radial(r, n=0, zeta=1.0)


class TestHydrogenicRadialErrors:
    """Validate invalid input handling in hydrogenic_radial.

    Validates that ``hydrogenic_radial`` raises ``ValueError`` for
    invalid principal quantum numbers (n < 1) and for angular momentum
    that violates 0 <= l < n.

    :see: :func:`~diffpes.radial.hydrogenic_radial`
    """

    def test_n_zero_raises(self) -> None:
        """Verify that n=0 raises ValueError for hydrogenic_radial.

        The test calls ``hydrogenic_radial`` with ``n=0`` and expects a
        ``ValueError`` that matches "n must be >= 1".

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        r: Array

        r = jnp.array([1.0], dtype=jnp.float64)
        with pytest.raises(ValueError, match="n must be >= 1"):
            hydrogenic_radial(r, n=0, angular_momentum=0, z_eff=1.0)

    def test_angular_momentum_equals_n_raises(self) -> None:
        """Verify that angular_momentum >= n raises ValueError.

        The test calls ``hydrogenic_radial`` with ``n=2`` and
        ``angular_momentum=2``. This input violates the angular-momentum
        constraint. The test expects a matching ``ValueError``.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        r: Array

        r = jnp.array([1.0], dtype=jnp.float64)
        with pytest.raises(ValueError, match="angular_momentum"):
            hydrogenic_radial(r, n=2, angular_momentum=2, z_eff=1.0)


class TestLaguerreRecurrence:
    """Validate the Laguerre polynomial recurrence path.

    Exercises the ``order >= 2`` branch of ``_associated_laguerre``
    that uses ``jax.lax.fori_loop`` for the recurrence, and validates
    the error paths for negative order or alpha.

    :see: :func:`~diffpes.radial.hydrogenic_radial`
    """

    def test_negative_order_raises(self) -> None:
        """Verify that order < 0 raises ValueError in _associated_laguerre.

        The test establishes the negative order raises contract for laguerre recurrence
        with the concrete values and array shapes described below.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        r: Array

        from diffpes.radial.wavefunctions import _associated_laguerre

        r = jnp.array([1.0], dtype=jnp.float64)
        with pytest.raises(ValueError, match="non-negative"):
            _associated_laguerre(-1, 0.5, r)

    def test_negative_alpha_raises(self) -> None:
        """Verify that alpha < 0 raises ValueError in _associated_laguerre.

        The test establishes the negative alpha raises contract for laguerre recurrence
        with the concrete values and array shapes described below.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        r: Array

        from diffpes.radial.wavefunctions import _associated_laguerre

        r = jnp.array([1.0], dtype=jnp.float64)
        with pytest.raises(ValueError, match="non-negative"):
            _associated_laguerre(2, -0.1, r)

    def test_order_one_early_return(self) -> None:
        """Verify that order=1 takes the early-return branch.

        L_1^0(x) = 1 - x, so at x=0 the value is 1.0 and at x=1 it is 0.0.
        This exercises the ``if order == 1: return laguerre_one`` path.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        x: Array
        result: Array
        expected: Array

        from diffpes.radial.wavefunctions import _associated_laguerre

        x = jnp.array([0.0, 1.0], dtype=jnp.float64)
        result = _associated_laguerre(1, 0.0, x)
        expected = jnp.array([1.0, 0.0], dtype=jnp.float64)
        chex.assert_trees_all_close(result, expected, atol=1e-10)

    def test_order_two_uses_recurrence(self) -> None:
        """Verify that order=2 executes the fori_loop recurrence branch.

        The order=0 and order=1 branches return early; order >= 2
        uses the upward recurrence. Checks the known value
        L_2^0(0) = 1 - 0 + 0 = 1.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        x: Array
        result: Array

        from diffpes.radial.wavefunctions import _associated_laguerre

        x = jnp.array([0.0], dtype=jnp.float64)

        result = _associated_laguerre(2, 0.0, x)
        chex.assert_trees_all_close(result, jnp.array([1.0]), atol=1e-10)
