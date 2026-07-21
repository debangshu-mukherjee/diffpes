"""Tests for spherical Bessel functions.

Extended Summary
----------------
Validates the ``spherical_bessel_jl`` implementation for orders l=0, 1, 2
against closed-form analytical expressions, tests the singular k=0 limit
(boundary condition), and verifies the autodiff gradient of j_0 against
its known derivative.  All closed-form tests are run both with and without
JIT compilation via ``chex.variants``.

"""

import chex
import jax
import jax.numpy as jnp
from beartype.typing import Any, Callable
from jaxtyping import Array

from diffpes.radial import spherical_bessel_jl


class TestSphericalBesselJl(chex.TestCase):
    """Validate low-order spherical Bessel j_l(x) behavior and derivatives.

    Tests cover the three lowest-order spherical Bessel functions j_0, j_1,
    and j_2 against their closed-form expressions, the k=0 boundary
    condition (j_0(0)=1, j_l(0)=0 for l>0), and the autodiff gradient of
    j_0 against its analytical derivative d/dx[sin(x)/x].  Each variant
    test runs both with and without JAX JIT to catch tracing issues.

    :see: :func:`~diffpes.radial.spherical_bessel_jl`
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_j0_and_j1_match_closed_form(self) -> None:
        """Verify j_0 and j_1 match their closed-form expressions.

        Uses test points x = [0.2, 0.7, 1.5] (avoiding x=0 singularity).
        j_0(x) = sin(x)/x and j_1(x) = sin(x)/x^2 - cos(x)/x are the
        standard analytical forms.  Asserts element-wise agreement to
        within 1e-10, run under both JIT and eager modes via
        ``chex.variants``.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        x: Array
        j0_fn: Callable[..., Any]
        j1_fn: Callable[..., Any]
        expected_j0: Array
        expected_j1: Array

        x = jnp.array([0.2, 0.7, 1.5], dtype=jnp.float64)
        j0_fn = self.variant(lambda values: spherical_bessel_jl(0, values))
        j1_fn = self.variant(lambda values: spherical_bessel_jl(1, values))

        expected_j0 = jnp.sin(x) / x
        expected_j1 = jnp.sin(x) / (x * x) - jnp.cos(x) / x
        chex.assert_trees_all_close(j0_fn(x), expected_j0, atol=1.0e-10)
        chex.assert_trees_all_close(j1_fn(x), expected_j1, atol=1.0e-10)

    @chex.variants(with_jit=True, without_jit=True)
    def test_j2_matches_closed_form(self) -> None:
        """Verify j_2 matches its closed-form expression.

        Uses test points x = [0.4, 1.1, 2.4].  The analytical form is
        j_2(x) = (3/x^3 - 1/x)*sin(x) - (3/x^2)*cos(x).  Asserts
        element-wise agreement to within 1e-10, confirming the recursion
        or series implementation is accurate for the l=2 case.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        x: Array
        fn: Callable[..., Any]
        expected: Array

        x = jnp.array([0.4, 1.1, 2.4], dtype=jnp.float64)
        fn = self.variant(lambda values: spherical_bessel_jl(2, values))
        expected = ((3.0 / (x**3)) - (1.0 / x)) * jnp.sin(x) - (
            3.0 / (x * x)
        ) * jnp.cos(x)
        chex.assert_trees_all_close(fn(x), expected, atol=1.0e-10)

    @chex.variants(with_jit=True, without_jit=True)
    def test_zero_argument_limits(self) -> None:
        """Verify the x=0 boundary conditions: j_0(0)=1, j_l(0)=0 for l>0.

        Evaluates j_0, j_1, and j_3 at x=0.0.  The mathematical limits
        are j_0(0) = 1 and j_l(0) = 0 for all l >= 1.  This is a critical
        edge case because the naive sin(x)/x formula is 0/0 at x=0, so
        the implementation must handle the removable singularity.  Asserts
        agreement to within 1e-12.  The l=3 case also confirms higher-
        order terms beyond the three tested in the closed-form tests.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        zero: Array
        j0_fn: Callable[..., Any]
        j1_fn: Callable[..., Any]
        j3_fn: Callable[..., Any]

        zero = jnp.array([0.0], dtype=jnp.float64)
        j0_fn = self.variant(lambda values: spherical_bessel_jl(0, values))
        j1_fn = self.variant(lambda values: spherical_bessel_jl(1, values))
        j3_fn = self.variant(lambda values: spherical_bessel_jl(3, values))
        chex.assert_trees_all_close(
            j0_fn(zero), jnp.array([1.0]), atol=1.0e-12
        )
        chex.assert_trees_all_close(
            j1_fn(zero), jnp.array([0.0]), atol=1.0e-12
        )
        chex.assert_trees_all_close(
            j3_fn(zero), jnp.array([0.0]), atol=1.0e-12
        )

    def test_j0_gradient_matches_analytic_derivative(self) -> None:
        """Verify autodiff gradient of j_0 matches the analytical derivative.

        Differentiates j_0(x) at x=1.3 using ``jax.grad`` and compares
        against the closed-form derivative d/dx[sin(x)/x] =
        (x*cos(x) - sin(x))/x^2.  Asserts agreement to within 1e-10,
        confirming the Bessel implementation supports reverse-mode AD
        for downstream radial-integral differentiation.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        x0: Array
        grad_fn: Callable[..., Any]
        grad_val: Array
        expected_grad: Array

        x0 = jnp.asarray(1.3, dtype=jnp.float64)

        def objective(x: chex.Numeric) -> chex.Array:
            return spherical_bessel_jl(0, jnp.asarray(x))

        grad_fn = jax.grad(objective)
        grad_val = grad_fn(x0)
        expected_grad = (x0 * jnp.cos(x0) - jnp.sin(x0)) / (x0 * x0)
        chex.assert_trees_all_close(grad_val, expected_grad, atol=1.0e-10)


class TestBesselErrors:
    """Tests for invalid input handling in the Bessel module.

    Validates that ``spherical_bessel_jl`` and the private helper
    ``_odd_double_factorial`` raise ``ValueError`` for out-of-range inputs.

    :see: :func:`~diffpes.radial.spherical_bessel_jl`
    """

    def test_negative_order_raises(self) -> None:
        """Verify that a negative order raises ValueError.

        Calls ``spherical_bessel_jl`` with ``order=-1`` and asserts a
        ``ValueError`` is raised, covering the guard at the top of the
        function.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        x: Array

        import jax.numpy as jnp
        import pytest

        from diffpes.radial import spherical_bessel_jl

        x = jnp.array([1.0], dtype=jnp.float64)
        with pytest.raises(ValueError, match="non-negative"):
            spherical_bessel_jl(-1, x)

    def test_odd_double_factorial_even_input_raises(self) -> None:
        """Verify that an even input to _odd_double_factorial raises ValueError.

        The helper ``_odd_double_factorial`` is used internally to compute
        the small-argument Taylor coefficient. It requires a positive odd
        integer; even inputs are invalid.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        import pytest

        from diffpes.radial.bessel import _odd_double_factorial

        with pytest.raises(ValueError, match="positive odd integer"):
            _odd_double_factorial(0)

    def test_odd_double_factorial_even_positive_raises(self) -> None:
        """Verify that a positive even input to _odd_double_factorial raises ValueError.

        This case establishes the odd double factorial even positive raises contract for
        bessel errors with the concrete values and array shapes described below.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        import pytest

        from diffpes.radial.bessel import _odd_double_factorial

        with pytest.raises(ValueError, match="positive odd integer"):
            _odd_double_factorial(4)
