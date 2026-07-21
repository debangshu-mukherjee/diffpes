"""Validate ARPES cross-section weight functions.

Extended Summary
----------------
Exercises heuristic_weights and yeh_lindau_weights. Heuristic tests
verify the two-regime model (p-enhanced below 50 eV, d-enhanced above),
output shape (9,), and JIT compatibility. Yeh-Lindau tests verify
interpolation at a fixed photon energy, output shape, and non-negative
weights. Each class and method docstring documents its test logic.

"""

import chex
import jax.numpy as jnp
from beartype.typing import Any, Callable
from jaxtyping import Array

from diffpes.simul import (
    heuristic_weights,
    yeh_lindau_weights,
)


class TestHeuristicWeights(chex.TestCase):
    """Validate :func:`diffpes.simul.crosssections.heuristic_weights`.

    The tests verify p-orbital enhancement below 50 eV and d-orbital
    enhancement above 50 eV. They check the output shape and JIT compatibility.

    :see: :func:`~diffpes.simul.heuristic_weights`
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_low_energy_p_enhanced(self) -> None:
        """Verify enhanced p-orbital weights in the low-energy regime.

        The test establishes the low energy p enhanced contract for heuristic weights
        with the concrete values and array shapes described below.

        Notes
        -----
        1. **Compute weights at 30 eV**:
           Call ``heuristic_weights(30.0)`` below the 50 eV threshold.

        2. **Check output shape**:
           Confirms the result has shape ``(9,)`` matching the 9-orbital basis.

        3. **Check p-orbital enhancement**:
           Verify weights 2.0 for p orbitals and 1.0 for the first d orbital.

        **Expected assertions**

        Output shape is ``(9,)``, p-orbitals (indices 1-3) equal 2.0, and
        d-orbital (index 4) equals 1.0.
        """
        var_fn: Callable[..., Any]
        w: Array

        var_fn = self.variant(heuristic_weights)
        w = var_fn(30.0)
        chex.assert_shape(w, (9,))
        assert float(w[1]) == 2.0
        assert float(w[2]) == 2.0
        assert float(w[3]) == 2.0
        assert float(w[4]) == 1.0

    @chex.variants(with_jit=True, without_jit=True)
    def test_high_energy_d_enhanced(self) -> None:
        """Verify enhanced d-orbital weights in the high-energy regime.

        The test establishes the high energy d enhanced contract for heuristic weights
        with the concrete values and array shapes described below.

        Notes
        -----
        1. **Compute weights at 60 eV**:
           Call ``heuristic_weights(60.0)`` above the 50 eV threshold.

        2. **Check p- and d-orbital values**:
           Verify weight 1.0 for a p orbital and 2.0 for two d orbitals.

        **Expected assertions**

        p-orbital (index 1) equals 1.0, and d-orbitals (indices 4, 8)
        equal 2.0.
        """
        var_fn: Callable[..., Any]
        w: Array

        var_fn = self.variant(heuristic_weights)
        w = var_fn(60.0)
        assert float(w[1]) == 1.0
        assert float(w[4]) == 2.0
        assert float(w[8]) == 2.0


class TestYehLindauWeights(chex.TestCase):
    """Validate :func:`diffpes.simul.crosssections.yeh_lindau_weights`.

    Verifies the interpolated Yeh-Lindau cross-section weights including
    exact values at tabulation points, correct interpolation at intermediate
    energies, positivity of all weights, and JIT compatibility.

    :see: :func:`~diffpes.simul.yeh_lindau_weights`
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_at_20_eV(self) -> None:
        """Verify exact cross-section values at the 20 eV tabulation point.

        The test establishes the at 20 eV contract for yeh lindau weights with the
        concrete values and array shapes described below.

        Notes
        -----
        1. **Compute weights at 20 eV**:
           Call ``yeh_lindau_weights(20.0)`` at the lowest table energy.

        2. **Check output shape**:
           Confirms the result has shape ``(9,)``.

        3. **Check known tabulated values**:
           Compare selected s, p, and d weights with their table values.

        **Expected assertions**

        Output shape is ``(9,)`` and weights at indices 0, 1, 4 match
        the tabulated values within tolerance 1e-5.
        """
        var_fn: Callable[..., Any]
        w: Array

        var_fn = self.variant(yeh_lindau_weights)
        w = var_fn(20.0)
        chex.assert_shape(w, (9,))
        chex.assert_trees_all_close(w[0], jnp.float64(0.1), atol=1e-5)
        chex.assert_trees_all_close(w[1], jnp.float64(0.6), atol=1e-5)
        chex.assert_trees_all_close(w[4], jnp.float64(2.0), atol=1e-5)

    @chex.variants(with_jit=True, without_jit=True)
    def test_interpolated(self) -> None:
        """Verify that interpolation at 30 eV produces positive s and p weights.

        The test establishes the interpolated contract for yeh lindau weights with the
        concrete values and array shapes described below.

        Notes
        -----
        1. **Compute weights at 30 eV**:
           Calls ``yeh_lindau_weights(30.0)``, an energy between the
           20 eV and 40 eV tabulation points, requiring interpolation.

        2. **Check output shape**:
           Confirms the result has shape ``(9,)``.

        3. **Check s- and p-orbital weights are positive**:
           Verify positive interpolated s-orbital and p-orbital weights.

        **Expected assertions**

        Output shape is ``(9,)`` and weights at indices 0 and 1 are
        strictly positive.
        """
        var_fn: Callable[..., Any]
        w: Array

        var_fn = self.variant(yeh_lindau_weights)
        w = var_fn(30.0)
        chex.assert_shape(w, (9,))
        assert float(w[0]) > 0.0
        assert float(w[1]) > 0.0

    @chex.variants(with_jit=True, without_jit=True)
    def test_all_positive(self) -> None:
        """Verify that all 9 orbital weights are strictly positive at 40 eV.

        The test establishes the all positive contract for yeh lindau weights with the
        concrete values and array shapes described below.

        Notes
        -----
        1. **Compute weights at 40 eV**:
           Calls ``yeh_lindau_weights(40.0)`` at a tabulation point.

        2. **Check positivity of every element**:
           Iterates over all 9 orbital weight entries and asserts each
           is strictly positive using ``chex.assert_scalar_positive``.

        **Expected assertions**

        Every element of the 9-element weight vector is strictly positive,
        ensuring no orbital receives a zero or negative cross-section.
        """
        i: int

        var_fn: Callable[..., Any]
        w: Array

        var_fn = self.variant(yeh_lindau_weights)
        w = var_fn(40.0)
        for i in range(9):
            chex.assert_scalar_positive(float(w[i]))
