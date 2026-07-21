"""Tests for ARPES broadening functions.

Extended Summary
----------------
Exercises the broadening module: gaussian, voigt, and fermi_dirac.
Each test is run with and without JIT (via chex.variants) to ensure
correctness under transformation. Tests cover normalization (unit
integral), peak position, symmetry, reduction of Voigt to Gaussian
when Lorentzian width is negligible, and Fermi-Dirac behaviour at
and away from the Fermi level. All test logic and assertions are
documented in the docstrings of each test class and method.

"""

import chex
import jax
import jax.numpy as jnp
import mpmath as mp
from beartype.typing import Any, Callable
from jaxtyping import Array, Float
from scipy import stats

from diffpes.simul import (
    fermi_dirac,
    gaussian,
    voigt,
)
from diffpes.types import KB_EV_PER_K
from tests._assertions import assert_rejects
from tests._gradients import assert_grad_matches_fd


def _fermi_value_and_gradients(
    theta: Float[Array, "3"],
) -> Float[Array, "4"]:
    """Evaluate Fermi occupation and its three parameter derivatives."""

    def occupation(parameters: Float[Array, "3"]) -> Float[Array, ""]:
        value: Float[Array, ""] = fermi_dirac(
            parameters[0], parameters[1], parameters[2]
        )
        return value

    value: Float[Array, ""] = occupation(theta)
    derivatives: Float[Array, "3"] = jax.grad(occupation)(theta)
    result: Float[Array, "4"] = jnp.concatenate(
        [jnp.reshape(value, (1,)), derivatives]
    )
    return result


def _voigt_width_loss(
    widths: Float[Array, "2"],
) -> Float[Array, ""]:
    """Reduce a pseudo-Voigt profile without symmetry cancellation."""
    energy_axis: Float[Array, "17"] = jnp.linspace(-1.3, 1.7, 17)
    weights: Float[Array, "17"] = jnp.linspace(0.7, 1.4, 17)
    profile: Float[Array, "17"] = voigt(
        energy_axis, 0.17, widths[0], widths[1]
    )
    loss: Float[Array, ""] = jnp.sum(weights * profile)
    return loss


class TestGaussian(chex.TestCase):
    """Tests for :func:`diffpes.simul.broadening.gaussian`.

    Verifies the normalized Gaussian broadening profile, including
    normalization (unit integral), peak position accuracy, and
    symmetry about the center energy.

    :see: :func:`~diffpes.simul.gaussian`
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_normalization(self) -> None:
        """Verify that the Gaussian profile integrates to unity.

        This case establishes the normalization contract for gaussian with the concrete
        values and array shapes described below.

        Notes
        -----
        1. **Create dense energy grid**:
           A wide energy range [-10, 10] eV with 100,000 points ensures
           accurate numerical integration via the rectangle rule.

        2. **Evaluate Gaussian**:
           Computes the profile centered at 0.0 eV with sigma = 0.5 eV.

        3. **Numerical integration**:
           Sums profile values multiplied by the energy step size to
           approximate the integral.

        **Expected assertions**

        The numerical integral is within 1e-3 of 1.0, confirming
        proper normalization of the Gaussian density.
        """
        e_range: Array
        sigma: float
        var_fn: Callable[..., Any]
        profile: Array
        de: Array
        integral: Array

        e_range = jnp.linspace(-10.0, 10.0, 100000)
        sigma = 0.5
        var_fn = self.variant(gaussian)
        profile = var_fn(e_range, 0.0, sigma)
        de = e_range[1] - e_range[0]
        integral = jnp.sum(profile) * de
        chex.assert_trees_all_close(integral, jnp.float64(1.0), atol=1e-3)

    @chex.variants(with_jit=True, without_jit=True)
    def test_peak_position(self) -> None:
        """Verify that the Gaussian peak occurs at the specified center energy.

        This case establishes the peak position contract for gaussian with the concrete
        values and array shapes described below.

        Notes
        -----
        1. **Create energy grid**:
           A symmetric range [-5, 5] eV with 10,001 points gives a step
           size of 0.001 eV for precise peak location.

        2. **Evaluate Gaussian**:
           Computes the profile centered at 1.5 eV with sigma = 0.3 eV.

        3. **Locate peak**:
           Finds the energy corresponding to the maximum profile value
           using ``jnp.argmax``.

        **Expected assertions**

        The peak energy is within 0.01 eV of the requested center (1.5 eV),
        confirming the centering parameter works correctly.
        """
        e_range: Array
        center: float
        var_fn: Callable[..., Any]
        profile: Array
        peak_idx: Array
        peak_energy: Array

        e_range = jnp.linspace(-5.0, 5.0, 10001)
        center = 1.5
        var_fn = self.variant(gaussian)
        profile = var_fn(e_range, center, 0.3)
        peak_idx = jnp.argmax(profile)
        peak_energy = e_range[peak_idx]
        chex.assert_trees_all_close(
            peak_energy, jnp.float64(center), atol=0.01
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_symmetry(self) -> None:
        """Verify that the Gaussian profile is symmetric about its center.

        This case establishes the symmetry contract for gaussian with the concrete
        values and array shapes described below.

        Notes
        -----
        1. **Create symmetric energy grid**:
           A range [-5, 5] eV centered at zero with an odd number of
           points (1001) ensures the center point lies exactly at 0.0 eV.

        2. **Evaluate Gaussian**:
           Computes the profile centered at 0.0 eV with sigma = 0.5 eV.

        3. **Compare with reversed profile**:
           Checks that the profile array equals its reverse, which holds
           for a symmetric function on a symmetric grid.

        **Expected assertions**

        Each element matches its mirror element to within 1e-10,
        confirming the even-function symmetry G(-E) = G(E).
        """
        e_range: Array
        var_fn: Callable[..., Any]
        profile: Array

        e_range = jnp.linspace(-5.0, 5.0, 1001)
        var_fn = self.variant(gaussian)
        profile = var_fn(e_range, 0.0, 0.5)
        chex.assert_trees_all_close(profile, profile[::-1], atol=1e-10)


class TestVoigt(chex.TestCase):
    """Tests for :func:`diffpes.simul.broadening.voigt`.

    Verifies the pseudo-Voigt broadening profile, including its
    limiting behavior (reduction to Gaussian when gamma approaches zero),
    peak position accuracy, and output finiteness.

    :see: :func:`~diffpes.simul.voigt`
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_reduces_to_gaussian(self) -> None:
        """Verify that the Voigt profile reduces to a Gaussian when gamma is negligible.

        This case establishes the reduces to gaussian contract for voigt with the
        concrete values and array shapes described below.

        Notes
        -----
        1. **Set near-zero Lorentzian width**:
           Uses gamma = 1e-10 eV so the Lorentzian contribution vanishes,
           leaving only the Gaussian component.

        2. **Evaluate both profiles**:
           Computes the Voigt profile and a pure Gaussian with the same
           sigma on the same energy grid.

        3. **Compare element-wise**:
           Checks that the Voigt output matches the pure Gaussian.

        **Expected assertions**

        All values agree to within 1e-3, confirming the correct
        Gaussian limiting behavior of the pseudo-Voigt approximation.
        """
        e_range: Array
        sigma: float
        gamma: float
        var_fn: Callable[..., Any]
        v_profile: Array
        g_profile: Array

        e_range = jnp.linspace(-5.0, 5.0, 10001)
        sigma = 0.5
        gamma = 1e-10
        var_fn = self.variant(voigt)
        v_profile = var_fn(e_range, 0.0, sigma, gamma)
        g_profile = gaussian(e_range, 0.0, sigma)
        chex.assert_trees_all_close(v_profile, g_profile, atol=1e-3)

    @chex.variants(with_jit=True, without_jit=True)
    def test_peak_position(self) -> None:
        """Verify that the Voigt profile peaks at the specified center energy.

        This case establishes the peak position contract for voigt with the concrete
        values and array shapes described below.

        Notes
        -----
        1. **Create energy grid**:
           A range [-5, 5] eV with 10,001 points for sub-meV resolution.

        2. **Evaluate Voigt profile**:
           Computes the profile centered at -1.0 eV with sigma = 0.3 eV
           and gamma = 0.1 eV (mixed Gaussian-Lorentzian regime).

        3. **Locate peak**:
           Finds the energy corresponding to the maximum profile value.

        **Expected assertions**

        The peak energy is within 0.01 eV of the requested center (-1.0 eV),
        confirming that both Gaussian and Lorentzian components share the
        same center.
        """
        e_range: Array
        center: Array
        var_fn: Callable[..., Any]
        profile: Array
        peak_idx: Array
        peak_energy: Array

        e_range = jnp.linspace(-5.0, 5.0, 10001)
        center = -1.0
        var_fn = self.variant(voigt)
        profile = var_fn(e_range, center, 0.3, 0.1)
        peak_idx = jnp.argmax(profile)
        peak_energy = e_range[peak_idx]
        chex.assert_trees_all_close(
            peak_energy, jnp.float64(center), atol=0.01
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_positive_values(self) -> None:
        """Verify that the Voigt profile produces finite values everywhere.

        This case establishes the positive values contract for voigt with the concrete
        values and array shapes described below.

        Notes
        -----
        1. **Evaluate Voigt profile**:
           Computes the profile on [-5, 5] eV with sigma = 0.5 eV and
           gamma = 0.2 eV, a typical mixed-broadening regime.

        2. **Check finiteness**:
           Ensures no NaN or infinity values appear in the output, which
           could arise from division-by-zero in the Lorentzian component
           or overflow in intermediate calculations.

        **Expected assertions**

        All profile values are finite (no NaN or Inf), confirming
        numerical stability of the pseudo-Voigt implementation.
        """
        e_range: Array
        var_fn: Callable[..., Any]
        profile: Array

        e_range = jnp.linspace(-5.0, 5.0, 1001)
        var_fn = self.variant(voigt)
        profile = var_fn(e_range, 0.0, 0.5, 0.2)
        chex.assert_tree_all_finite(profile)

    @chex.variants(with_jit=True, without_jit=True)
    def test_exact_boundary_profiles_match_scipy(self) -> None:
        """Match the Gaussian and Lorentzian endpoint rays to SciPy.

        Extended Summary
        ----------------
        Verifies the Thompson-Cox-Hastings mixing law reduces to the exact
        normalized Gaussian at ``gamma = 0`` and the exact normalized Cauchy
        density at ``sigma = 0``, both to ``rtol=1e-12``.

        Notes
        -----
        Evaluates both rays eagerly and under JIT on an asymmetric energy
        grid, then compares against ``scipy.stats.norm.pdf`` and
        ``scipy.stats.cauchy.pdf`` as independent external truths.
        """
        var_fn: Callable[..., Any]

        energy_axis: Float[Array, "31"] = jnp.linspace(-1.7, 2.1, 31)
        center: float = 0.13
        sigma: float = 0.27
        gamma: float = 0.19
        var_fn = self.variant(voigt)
        gaussian_profile: Float[Array, "31"] = var_fn(
            energy_axis, center, sigma, 0.0
        )
        lorentzian_profile: Float[Array, "31"] = var_fn(
            energy_axis, center, 0.0, gamma
        )
        expected_gaussian: Float[Array, "31"] = jnp.asarray(
            stats.norm.pdf(energy_axis, loc=center, scale=sigma)
        )
        expected_lorentzian: Float[Array, "31"] = jnp.asarray(
            stats.cauchy.pdf(energy_axis, loc=center, scale=gamma)
        )
        chex.assert_trees_all_close(
            gaussian_profile, expected_gaussian, rtol=1e-12, atol=0.0
        )
        chex.assert_trees_all_close(
            lorentzian_profile, expected_lorentzian, rtol=1e-12, atol=0.0
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_boundary_gradients_match_fd_plateau(self) -> None:
        """Match both boundary-ray gradients to a multistep FD plateau.

        Extended Summary
        ----------------
        Verifies sensitivities on the pure-Gaussian and pure-Lorentzian rays
        remain finite, nonzero, and central-FD-correct to the stiff
        ``rtol=1e-5`` gate.

        Notes
        -----
        Applies the shared program-wide gradient harness at three step scales
        to both width vectors, for eager and JIT-transformed scalar losses.
        """
        widths: Array
        scale_floor: float

        loss: Array

        loss = self.variant(_voigt_width_loss)
        boundary_widths: tuple[Float[Array, "2"], ...] = (
            jnp.array([0.23, 0.0], dtype=jnp.float64),
            jnp.array([0.0, 0.19], dtype=jnp.float64),
        )
        for widths in boundary_widths:
            derivatives: Float[Array, "2"] = jax.grad(loss)(widths)
            chex.assert_tree_all_finite(derivatives)
            chex.assert_trees_all_equal(derivatives != 0.0, jnp.ones(2, bool))
            for scale_floor in (0.5, 1.0, 2.0):
                assert_grad_matches_fd(
                    loss,
                    widths,
                    regime="stiff",
                    scale_floor=scale_floor,
                )

    def test_simultaneous_zero_width_is_rejected(self) -> None:
        """Reject the singular zero-width point eagerly and under JIT.

        Extended Summary
        ----------------
        Verifies ``sigma = gamma = 0`` is reported as undefined instead of
        returning a pointwise value with a fabricated derivative.

        Notes
        -----
        Uses the shared rejection helper on a finite energy grid, exercising
        both direct execution and ``equinox.filter_jit``.
        """
        energy_axis: Float[Array, "5"] = jnp.linspace(-1.0, 1.0, 5)
        assert_rejects(
            voigt,
            energy_axis,
            0.0,
            0.0,
            0.0,
            match="sigma and gamma must not both be zero",
        )


class TestFermiDirac(chex.TestCase):
    """Tests for :func:`diffpes.simul.broadening.fermi_dirac`.

    Verifies the Fermi-Dirac thermal distribution function, including
    the exact value at the Fermi level, asymptotic limits deep below
    and high above the Fermi energy, and the bounded output range [0, 1].

    :see: :func:`~diffpes.simul.fermi_dirac`
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_at_fermi_level(self) -> None:
        """Verify that the Fermi-Dirac function equals 0.5 at the Fermi energy.

        This case establishes the at fermi level contract for fermi dirac with the
        concrete values and array shapes described below.

        Notes
        -----
        1. **Evaluate at E = Ef**:
           Calls ``fermi_dirac(0.0, 0.0, 300.0)`` where energy and Fermi
           level are both 0.0 eV at room temperature (300 K).

        2. **Check analytic result**:
           At E = Ef the exponent is zero, so f(Ef) = 1/(1+1) = 0.5
           regardless of temperature.

        **Expected assertions**

        The result is within 1e-5 of 0.5, confirming the fundamental
        property f(Ef) = 0.5 at finite temperature.
        """
        var_fn: Callable[..., Any]
        result: Array

        var_fn = self.variant(fermi_dirac)
        result = var_fn(0.0, 0.0, 300.0)
        chex.assert_trees_all_close(result, jnp.float64(0.5), atol=1e-5)

    @chex.variants(with_jit=True, without_jit=True)
    def test_deep_below_fermi(self) -> None:
        """Verify that states deep below the Fermi level are fully occupied.

        This case establishes the deep below fermi contract for fermi dirac with the
        concrete values and array shapes described below.

        Notes
        -----
        1. **Evaluate far below Ef**:
           Calls ``fermi_dirac(-5.0, 0.0, 15.0)`` where E - Ef = -5.0 eV
           at T = 15 K (kT ~ 1.3 meV), making the exponent ~ -3900.

        2. **Check saturation**:
           For large negative exponents, exp(x) approaches 0 and the
           occupation approaches 1/(1+0) = 1.

        **Expected assertions**

        The result is within 1e-5 of 1.0, confirming full occupation
        of deeply bound states.
        """
        var_fn: Callable[..., Any]
        result: Array

        var_fn = self.variant(fermi_dirac)
        result = var_fn(-5.0, 0.0, 15.0)
        chex.assert_trees_all_close(result, jnp.float64(1.0), atol=1e-5)

    @chex.variants(with_jit=True, without_jit=True)
    def test_high_above_fermi(self) -> None:
        """Verify that states far above the Fermi level are unoccupied.

        This case establishes the high above fermi contract for fermi dirac with the
        concrete values and array shapes described below.

        Notes
        -----
        1. **Evaluate far above Ef**:
           Calls ``fermi_dirac(5.0, 0.0, 15.0)`` where E - Ef = +5.0 eV
           at T = 15 K, making the exponent ~ +3900.

        2. **Check vanishing occupation**:
           For large positive exponents, exp(x) dominates and the
           occupation approaches 1/(1+inf) = 0.

        **Expected assertions**

        The result is within 1e-5 of 0.0, confirming that states
        well above the Fermi energy are effectively empty.
        """
        var_fn: Callable[..., Any]
        result: Array

        var_fn = self.variant(fermi_dirac)
        result = var_fn(5.0, 0.0, 15.0)
        chex.assert_trees_all_close(result, jnp.float64(0.0), atol=1e-5)

    @chex.variants(with_jit=True, without_jit=True)
    def test_range_0_to_1(self) -> None:
        """Verify that the Fermi-Dirac output lies within [0, 1].

        This case establishes the range 0 to 1 contract for fermi dirac with the
        concrete values and array shapes described below.

        Notes
        -----
        1. **Evaluate at a typical energy**:
           Calls ``fermi_dirac(-0.5, 0.0, 300.0)`` at E = -0.5 eV
           relative to Ef = 0.0 eV at room temperature (300 K), where
           the result should be between 0 and 1 (closer to 1).

        2. **Bound checks**:
           Asserts the scalar result is non-negative and does not
           exceed unity using plain Python comparisons.

        **Expected assertions**

        The occupation value satisfies 0 <= f(E) <= 1, which must
        hold for any valid probability/occupation function.
        """
        var_fn: Callable[..., Any]
        result: Array

        var_fn = self.variant(fermi_dirac)
        result = var_fn(-0.5, 0.0, 300.0)
        assert float(result) >= 0.0
        assert float(result) <= 1.0

    @chex.variants(with_jit=True, without_jit=True)
    def test_value_and_gradients_match_closed_forms(self) -> None:
        """Match occupations and all derivatives across the f64 ladder.

        Extended Summary
        ----------------
        Verifies values against a 50-decimal-digit ``mpmath`` logistic and
        derivatives with respect to energy, Fermi energy, and temperature
        against the analytic Fermi-Dirac formulas to ``rtol=1e-12``.

        Notes
        -----
        Evaluates ``x`` in ``{0, ±1, ±10, ±100, ±700}`` at 5, 15, and
        300 K, both eagerly and under JIT. Closed forms are evaluated from
        the independently computed occupation after f64 rounding, including
        the representable saturation convention.
        """
        temperature: float
        x_value: float

        occupation_high_precision: Array
        var_fn: Callable[..., Any]

        x_values: tuple[float, ...] = (
            0.0,
            1.0,
            -1.0,
            10.0,
            -10.0,
            100.0,
            -100.0,
            700.0,
            -700.0,
        )
        temperatures: tuple[float, ...] = (5.0, 15.0, 300.0)
        parameters: list[list[float]] = []
        expected_rows: list[list[float]] = []
        with mp.workdps(50):
            for temperature in temperatures:
                thermal_energy: float = KB_EV_PER_K * temperature
                for x_value in x_values:
                    energy: float = x_value * thermal_energy
                    occupation_high_precision = 1 / (
                        1 + mp.exp(mp.mpf(str(x_value)))
                    )
                    occupation: float = float(occupation_high_precision)
                    occupation_factor: float = occupation * (1.0 - occupation)
                    energy_derivative: float = (
                        -occupation_factor / thermal_energy
                    )
                    fermi_derivative: float = -energy_derivative
                    temperature_derivative: float = (
                        occupation_factor * x_value / temperature
                    )
                    parameters.append([energy, 0.0, temperature])
                    expected_rows.append(
                        [
                            occupation,
                            energy_derivative,
                            fermi_derivative,
                            temperature_derivative,
                        ]
                    )
        parameter_array: Float[Array, "27 3"] = jnp.asarray(parameters)
        expected: Float[Array, "27 4"] = jnp.asarray(expected_rows)
        var_fn = self.variant(_fermi_value_and_gradients)
        actual: Float[Array, "27 4"] = jax.vmap(var_fn)(parameter_array)
        chex.assert_tree_all_finite(actual)
        chex.assert_trees_all_close(actual, expected, rtol=1e-12, atol=0.0)

    @chex.variants(with_jit=True, without_jit=True)
    def test_extreme_arguments_have_finite_zero_gradients(self) -> None:
        """Keep the audit probe and extreme tails free of NaN gradients.

        Extended Summary
        ----------------
        Verifies the original ``+1 eV`` at 15 K failure and tails through
        ``|x| = 5000`` have finite values and finite derivatives, with exact
        zero derivatives after f64 underflow or saturation.

        Notes
        -----
        Evaluates the occupation and all three gradients eagerly and under
        JIT, then checks the positive audit/tail values and every saturated
        derivative exactly.
        """
        var_fn: Callable[..., Any]

        temperature: float = 15.0
        thermal_energy: float = KB_EV_PER_K * temperature
        parameters: Float[Array, "3 3"] = jnp.array(
            [
                [1.0, 0.0, temperature],
                [5000.0 * thermal_energy, 0.0, temperature],
                [-5000.0 * thermal_energy, 0.0, temperature],
            ],
            dtype=jnp.float64,
        )
        var_fn = self.variant(_fermi_value_and_gradients)
        results: Float[Array, "3 4"] = jax.vmap(var_fn)(parameters)
        chex.assert_tree_all_finite(results)
        chex.assert_trees_all_equal(results[:, 1:], jnp.zeros((3, 3)))
        chex.assert_trees_all_equal(results[:2, 0], jnp.zeros(2))
        chex.assert_trees_all_equal(results[2, 0], jnp.float64(1.0))

    @chex.variants(with_jit=True, without_jit=True)
    def test_gradients_match_central_finite_differences(self) -> None:
        """Match all three smooth derivatives to central differences.

        Extended Summary
        ----------------
        Verifies energy, Fermi-energy, and temperature sensitivities are
        finite, nonzero, and central-FD-correct at ``x = 1`` and 15 K to the
        smooth ``rtol=1e-6`` gate.

        Notes
        -----
        Runs the shared gradient harness against eager and JIT-transformed
        scalar functions on the three-parameter vector.
        """
        var_fn: Callable[..., Any]

        theta: Float[Array, "3"] = jnp.array(
            [KB_EV_PER_K * 15.0, 0.0, 15.0], dtype=jnp.float64
        )

        def occupation(parameters: Float[Array, "3"]) -> Float[Array, ""]:
            value: Float[Array, ""] = fermi_dirac(
                parameters[0], parameters[1], parameters[2]
            )
            return value

        var_fn = self.variant(occupation)
        derivatives: Float[Array, "3"] = jax.grad(var_fn)(theta)
        chex.assert_tree_all_finite(derivatives)
        chex.assert_trees_all_equal(derivatives != 0.0, jnp.ones(3, bool))
        assert_grad_matches_fd(var_fn, theta, regime="smooth")
