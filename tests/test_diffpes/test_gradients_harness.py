"""Validate the shared gradient harness against external analytic truths.

Extended Summary
----------------
Exercises scale-aware finite differences, JAX's complex-to-real convention,
complex-step differentiation, and planted wrong and zero gradients. These
self-tests establish gates 01.G3 and 01.G4 before physics code relies on the
shared harness.
"""

import chex
import jax
import jax.numpy as jnp
import pytest
from jax import test_util
from jaxtyping import Array, Complex, Float

from diffpes.simul.crosssections import heuristic_weights
from tests._gradients import (
    RTOL_LADDER,
    assert_grad_matches_fd,
    assert_nonzero_grad,
    central_fd_grad,
    complex_step_derivative,
    fd_step,
    gradient_gate,
    random_generic_complex,
)


@jax.custom_jvp
def _wrong_sine(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Return sine with a deliberately incorrect ten-percent tangent."""
    result: Float[Array, "..."] = jnp.sin(x)
    return result


@_wrong_sine.defjvp
def _wrong_sine_jvp(
    primals: tuple[Float[Array, "..."], ...],
    tangents: tuple[Float[Array, "..."], ...],
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """Plant a tangent scaled by 1.1 for harness-defect detection."""
    (x,) = primals
    (x_tangent,) = tangents
    primal: Float[Array, "..."] = _wrong_sine(x)
    tangent: Float[Array, "..."] = 1.1 * jnp.cos(x) * x_tangent
    result: tuple[Float[Array, "..."], Float[Array, "..."]] = (
        primal,
        tangent,
    )
    return result


@jax.custom_jvp
def _near_wrong_sine(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Return sine with a deliberately incorrect five-digit tangent."""
    result: Float[Array, "..."] = jnp.sin(x)
    return result


@_near_wrong_sine.defjvp
def _near_wrong_sine_jvp(
    primals: tuple[Float[Array, "..."], ...],
    tangents: tuple[Float[Array, "..."], ...],
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """Plant a tangent scaled by 1.00001 to pin the detection floor."""
    (x,) = primals
    (x_tangent,) = tangents
    primal: Float[Array, "..."] = _near_wrong_sine(x)
    tangent: Float[Array, "..."] = 1.00001 * jnp.cos(x) * x_tangent
    result: tuple[Float[Array, "..."], Float[Array, "..."]] = (
        primal,
        tangent,
    )
    return result


class TestGradientHarness(chex.TestCase):
    """Validate the shared gradient harness and gates 01.G3 and 01.G4."""

    def test_closed_form_truths(self) -> None:
        """Verify analytic smooth gradients pass at relative tolerance 1e-6.

        Notes
        -----
        Checks sine, a Gaussian sum, the two-dimensional Rosenbrock function,
        and a mixed-unit rational monomial through both forward- and
        reverse-mode checks plus elementwise finite differences (01.G3).
        """
        sine_input: Float[Array, "3"] = jnp.array([-0.7, 0.2, 1.1])
        assert_grad_matches_fd(lambda x: jnp.sum(jnp.sin(x)), sine_input)
        gaussian_input: Float[Array, "3"] = jnp.array([-1.2, 0.3, 0.9])
        assert_grad_matches_fd(
            lambda x: jnp.sum(jnp.exp(-(x**2))), gaussian_input
        )
        rosenbrock_input: Float[Array, "2"] = jnp.array([-0.8, 1.4])
        assert_grad_matches_fd(
            lambda x: (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2,
            rosenbrock_input,
            regime="stiff",
        )
        mixed_scale: Float[Array, "3"] = jnp.array([1e-3, 5.0, 3.0])
        gradient_gate(
            lambda x: x[0] ** 2 * x[1] / x[2],
            mixed_scale,
            regime="smooth",
        )

    def test_fd_step_scales_per_element(self) -> None:
        """Verify finite-difference steps scale with each parameter magnitude.

        Notes
        -----
        Compares a mixed-unit vector against the exact
        ``eps**(1/3) * max(abs(theta), 1e-3)`` prescription (01.G3).
        """
        theta: Float[Array, "3"] = jnp.array([1e-4, 5.0, -3.0])
        actual: Float[Array, "3"] = fd_step(theta)
        ratio: Float[Array, "3"] = actual / actual[0]
        expected_ratio: Float[Array, "3"] = jnp.array([1.0, 5000.0, 3000.0])
        chex.assert_trees_all_close(ratio, expected_ratio)

    def test_wirtinger_convention(self) -> None:
        """Pin JAX's C-to-R gradient as d/dRe minus i times d/dIm.

        Notes
        -----
        Checks the exact gradient ``2-2j`` of ``abs(z)**2`` at ``1+1j`` and
        compares the harness on generic asymmetric complex data at 1e-8
        relative tolerance (01.G3).
        """
        exact: Complex[Array, ""] = jax.grad(lambda z: jnp.abs(z) ** 2)(
            jnp.asarray(1.0 + 1.0j)
        )
        chex.assert_trees_all_equal(exact, jnp.asarray(2.0 - 2.0j))
        values: Complex[Array, "4"] = random_generic_complex(
            jax.random.key(20260713), (4,)
        )
        automatic: Complex[Array, "4"] = jax.grad(
            lambda z: jnp.sum(jnp.abs(z) ** 2)
        )(values)
        finite_difference: Complex[Array, "4"] = central_fd_grad(
            lambda z: jnp.sum(jnp.abs(z) ** 2), values
        )
        chex.assert_trees_all_close(
            automatic, finite_difference, rtol=1e-8, atol=1e-10
        )

    def test_planted_wrong_gradient(self) -> None:
        """Verify a ten-percent tangent defect fails every tolerance rung.

        Notes
        -----
        A ``custom_jvp`` retains the correct sine primal but scales its
        derivative by 1.1; the shared gate must raise for gate 01.G4.
        """
        theta: Float[Array, "3"] = jnp.array([-0.4, 0.2, 0.8])
        for regime in RTOL_LADDER:
            with self.subTest(regime=regime), pytest.raises(AssertionError):
                assert_grad_matches_fd(
                    lambda x: jnp.sum(_wrong_sine(x)), theta, regime=regime
                )

    def test_detection_floor(self) -> None:
        """Verify a one-part-in-100000 defect fails the smooth tolerance.

        Notes
        -----
        Uses a planted ``1.00001*cos(x)`` tangent and demands detection at the
        strictest 1e-6 relative rung, documenting gate 01.G4's floor.
        """
        theta: Float[Array, "3"] = jnp.array([-0.4, 0.2, 0.8])
        with pytest.raises(AssertionError):
            assert_grad_matches_fd(
                lambda x: jnp.sum(_near_wrong_sine(x)), theta
            )

    def test_planted_zero_gradient(self) -> None:
        """Verify finite-but-zero stopped gradients fail both tripwires.

        Notes
        -----
        The primal remains ``sum(x**2)`` while ``stop_gradient`` removes all
        autodiff sensitivity. Finite differences and the independent norm
        check must each raise for gate 01.G4.
        """
        theta: Float[Array, "2"] = jnp.array([1.0, -2.0])

        def stopped_loss(x: Float[Array, "2"]) -> Float[Array, ""]:
            result: Float[Array, ""] = jnp.sum(jax.lax.stop_gradient(x) ** 2)
            return result

        with pytest.raises(AssertionError):
            assert_grad_matches_fd(stopped_loss, theta)
        with pytest.raises(AssertionError):
            assert_nonzero_grad(stopped_loss, theta)

    def test_in_tree_zero_gradient(self) -> None:
        """Verify the known heuristic photon-energy dead gradient is caught.

        Notes
        -----
        Differentiates the sum of heuristic orbital weights at 30 eV, where
        the piecewise lookup has exactly zero sensitivity, and requires the
        zero-gradient tripwire to raise for gate 01.G4.
        """
        photon_energy: Float[Array, ""] = jnp.asarray(30.0)
        with pytest.raises(AssertionError):
            assert_nonzero_grad(
                lambda energy: jnp.sum(heuristic_weights(energy)),
                photon_energy,
            )

    def test_complex_step_derivative(self) -> None:
        """Verify complex-step sine accuracy and reject modulus-squared.

        Notes
        -----
        Compares the holomorphic sine derivative with cosine at relative
        tolerance 1e-15, then confirms conjugation in ``abs(x)**2`` triggers
        the zero-imaginary guard (01.G3).
        """
        x: Float[Array, "3"] = jnp.array([-0.4, 0.2, 0.8])
        derivative: Float[Array, "3"] = complex_step_derivative(jnp.sin, x)
        chex.assert_trees_all_close(
            derivative, jnp.cos(x), rtol=1e-15, atol=0.0
        )
        with pytest.raises(ValueError, match="non-holomorphic"):
            complex_step_derivative(lambda value: jnp.abs(value) ** 2, x)

    def test_check_grads_semantics_anchor(self) -> None:
        """Pin JAX check_grads behavior on truth and a planted tangent defect.

        Notes
        -----
        Calls JAX's external directional-gradient instrument directly on sine
        and the 1.1-scaled ``custom_jvp`` plant, establishing gate 01.G3's
        semantic anchor independently of the wrapper.
        """
        theta: Float[Array, "3"] = jnp.array([-0.4, 0.2, 0.8])
        test_util.check_grads(
            lambda x: jnp.sum(jnp.sin(x)),
            (theta,),
            order=1,
            modes=("fwd", "rev"),
            eps=1e-5,
        )
        with pytest.raises(AssertionError):
            test_util.check_grads(
                lambda x: jnp.sum(_wrong_sine(x)),
                (theta,),
                order=1,
                modes=("fwd", "rev"),
                eps=1e-5,
            )
