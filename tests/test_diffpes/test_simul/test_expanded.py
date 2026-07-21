"""Validate expanded-input simulation wrappers.

Extended Summary
----------------
The tests validate expanded wrappers that accept arrays and scalars.
They compare each wrapper with its corresponding core simulation.
They verify energy-window construction and dispatch for every level.
They also verify that the SOC level requires spin data.

"""

import chex
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any, Callable
from jaxtyping import Array, Float, jaxtyped

import diffpes
from diffpes.simul import (
    simulate_advanced,
    simulate_advanced_expanded,
    simulate_basic,
    simulate_basic_expanded,
    simulate_basicplus_expanded,
    simulate_expanded,
    simulate_expert_expanded,
    simulate_novice_expanded,
    simulate_soc,
    simulate_soc_expanded,
)
from diffpes.types import (
    ArpesSpectrum,
    make_band_structure,
    make_expanded_simulation_params,
    make_orbital_projection,
    make_polarization_config,
    make_spin_orbital_projection,
)


@jaxtyped(typechecker=beartype)
def _make_synthetic_data(
    nk: int = 12,
    nb: int = 4,
    na: int = 3,
) -> tuple[Float[Array, "nk nb"], Float[Array, "nk nb na 9"]]:
    """Generate synthetic eigenband and orbital projection arrays for testing.

    The helper creates raw arrays for the expanded-input API.
    It spaces eigenvalues from -2.5 to 0.75 eV with float64 precision.
    It sets all orbital projections to 0.1.

    Parameters
    ----------
    nk : int, optional
        Number of k-points. Default is 12.
    nb : int, optional
        Number of bands. Default is 4.
    na : int, optional
        Number of atoms. Default is 3.

    Returns
    -------
    eigenbands : jnp.ndarray
        Band eigenvalues of shape ``(nk, nb)`` in float64, linearly
        spaced from -2.5 to 0.75 eV.
    surface_orb : jnp.ndarray
        Uniform orbital projections of shape ``(nk, nb, na, 9)`` in
        float64 with all entries set to 0.1.
    """
    eigenbands: Float[Array, "nk nb"] = jnp.linspace(
        -2.5, 0.75, nk * nb, dtype=jnp.float64
    ).reshape(nk, nb)
    surface_orb: Float[Array, "nk nb na 9"] = jnp.ones(
        (nk, nb, na, 9), dtype=jnp.float64
    )
    surface_orb = surface_orb * 0.1
    synthetic_data: tuple[
        Float[Array, "nk nb"], Float[Array, "nk nb na 9"]
    ] = (eigenbands, surface_orb)
    return synthetic_data


class TestExpandedParams(chex.TestCase):
    """Validate :func:`diffpes.simul.expanded.make_expanded_simulation_params`.

    The tests verify the energy window from eigenband extrema and default
    padding. They also verify all forwarded scalar parameters.

    :see: :func:`~diffpes.types.make_expanded_simulation_params`
    """

    def test_energy_window_matches_expanded_default(self) -> None:
        """Verify derivation of the energy bounds from eigenbands.

        The test establishes the energy window matches expanded default contract for
        expanded params with the concrete values and array shapes described below.

        Notes
        -----
        1. **Setup**:
           Create a small eigenband array with known extrema:
           min = -2.0, max = 1.0. Use the default energy padding of 1.0.

        2. **Build params**:
           Call ``make_expanded_simulation_params`` with fidelity=100.

        3. **Check energy bounds**:
           Assert that ``energy_min`` equals ``min(eigenbands) - 1.0 = -3.0``
           and ``energy_max`` equals ``max(eigenbands) + 1.0 = 2.0``,
           each within a tolerance of 1e-12.

        4. **Check fidelity**:
           Check that the fidelity parameter equals 100.

        **Expected assertions**

        The derived energy window matches ``[min - padding, max + padding]``.
        The fidelity value remains exactly 100.
        """
        eigenbands: Array
        params: diffpes.types.SimulationParams

        eigenbands = jnp.array([[-2.0, 0.25], [1.0, -1.0]], dtype=jnp.float64)
        params = make_expanded_simulation_params(
            eigenbands=eigenbands,
            fidelity=100,
        )
        chex.assert_trees_all_close(
            params.energy_min, jnp.float64(-3.0), atol=1e-12
        )
        chex.assert_trees_all_close(
            params.energy_max, jnp.float64(2.0), atol=1e-12
        )
        chex.assert_equal(params.fidelity, 100)


class TestSimulateNoviceExpanded(chex.TestCase):
    """Validate :func:`~diffpes.simul.simulate_novice_expanded`.

    Covers the plain-array novice wrapper at a reduced energy-grid fidelity
    while retaining all required Voigt parameters.

    :see: :func:`~diffpes.simul.simulate_novice_expanded`
    """

    def test_returns_requested_energy_grid(self) -> None:
        """Return one intensity row per k-point on the requested energy grid.

        The twelve-point synthetic band path and fidelity of 48 must produce a
        finite ``(12, 48)`` intensity array.

        Notes
        -----
        The test passes the raw synthetic arrays and explicit novice parameters to the
        wrapper, then checks its shape and finite values with Chex.
        """
        eigenbands: Array
        surface_orb: Array
        spectrum: diffpes.types.ArpesSpectrum

        eigenbands, surface_orb = _make_synthetic_data()
        spectrum = simulate_novice_expanded(
            eigenbands=eigenbands,
            surface_orb=surface_orb,
            ef=0.0,
            sigma=0.04,
            gamma=0.1,
            fidelity=48,
            temperature=15.0,
            photon_energy=21.2,
        )
        chex.assert_shape(spectrum.intensity, (12, 48))
        chex.assert_tree_all_finite(spectrum.intensity)


class TestSimulateBasicExpanded(chex.TestCase):
    """Validate :func:`diffpes.simul.expanded.simulate_basic_expanded`.

    Verifies that the expanded basic wrapper produces results identical to
    manually constructing PyTree inputs and calling the core
    :func:`~diffpes.simul.spectrum.simulate_basic` function directly.

    :see: :func:`~diffpes.simul.simulate_basic_expanded`
    """

    def test_matches_core_basic_simulation(self) -> None:
        """Verify that the expanded wrapper matches the core basic simulation.

        The test establishes the matches core basic simulation contract for expanded
        basic wrapper with the concrete values and array shapes described below.

        Notes
        -----
        1. **Build reference manually**:
           Construct equivalent PyTrees from the raw inputs.

        2. **Run core simulation**:
           Call ``simulate_basic`` directly with the PyTree inputs to
           obtain the expected spectrum.

        3. **Run expanded wrapper**:
           Call the expanded wrapper with equivalent inputs.

        4. **Compare**:
           Assert that both intensity arrays and energy axes match to
           within 1e-12 absolute tolerance.

        **Expected assertions**

        The expanded and core results have identical intensity and energy axes.
        This agreement verifies construction of all intermediate PyTrees.
        """
        eigenbands: Array
        surface_orb: Array
        params: diffpes.types.SimulationParams
        kpoints: Array
        bands: diffpes.types.BandStructure
        orb_proj: diffpes.types.OrbitalProjection
        expected: diffpes.types.ArpesSpectrum
        wrapped: diffpes.types.ArpesSpectrum

        eigenbands, surface_orb = _make_synthetic_data()
        params = make_expanded_simulation_params(
            eigenbands=eigenbands,
            fidelity=240,
            sigma=0.06,
            temperature=20.0,
            photon_energy=35.0,
        )
        kpoints = jnp.zeros((eigenbands.shape[0], 3), dtype=jnp.float64)
        bands = make_band_structure(
            eigenvalues=eigenbands,
            kpoints=kpoints,
            fermi_energy=0.0,
        )
        orb_proj = make_orbital_projection(projections=surface_orb)
        expected = simulate_basic(bands, orb_proj, params)
        wrapped = simulate_basic_expanded(
            eigenbands=eigenbands,
            surface_orb=surface_orb,
            ef=0.0,
            sigma=0.06,
            fidelity=240,
            temperature=20.0,
            photon_energy=35.0,
        )
        chex.assert_trees_all_close(
            wrapped.intensity, expected.intensity, atol=1e-12
        )
        chex.assert_trees_all_close(
            wrapped.energy_axis, expected.energy_axis, atol=1e-12
        )


class TestSimulateBasicplusExpanded(chex.TestCase):
    """Validate :func:`~diffpes.simul.simulate_basicplus_expanded`.

    Covers the Yeh-Lindau weighted wrapper for raw arrays at a representative
    ultraviolet photon energy.

    :see: :func:`~diffpes.simul.simulate_basicplus_expanded`
    """

    def test_returns_finite_cross_section_weighted_spectrum(self) -> None:
        """Return finite intensity at the requested basicplus fidelity.

        The wrapper must retain twelve k-points and construct forty-eight energy
        samples without non-finite cross-section weights.

        Notes
        -----
        The test evaluates the raw synthetic arrays at 21.2 eV and checks the output
        intensity shape and finiteness with Chex.
        """
        eigenbands: Array
        surface_orb: Array
        spectrum: diffpes.types.ArpesSpectrum

        eigenbands, surface_orb = _make_synthetic_data()
        spectrum = simulate_basicplus_expanded(
            eigenbands=eigenbands,
            surface_orb=surface_orb,
            ef=0.0,
            sigma=0.04,
            fidelity=48,
            temperature=15.0,
            photon_energy=21.2,
        )
        chex.assert_shape(spectrum.intensity, (12, 48))
        chex.assert_tree_all_finite(spectrum.intensity)


class TestSimulateAdvancedExpanded(chex.TestCase):
    """Validate :func:`diffpes.simul.expanded.simulate_advanced_expanded`.

    The tests verify conversion of incident angles from degrees to radians.
    They compare the wrapper result with a core call that uses radians.

    :see: :func:`~diffpes.simul.simulate_advanced_expanded`
    """

    def test_degree_conversion_matches_core_advanced(self) -> None:
        """Verify that degree-input angles produce the same result as radian-input.

        The test establishes the degree conversion matches core advanced contract for
        expanded advanced wrapper with the concrete values and array shapes described
        below.

        Notes
        -----
        1. **Build reference manually**:
           Construct equivalent PyTrees from the raw inputs.

        2. **Run core simulation**:
           Call ``simulate_advanced`` directly with the manually built
           PyTrees and polarization config to obtain the expected spectrum.

        3. **Run expanded wrapper**:
           Call the expanded wrapper with equivalent inputs.

        4. **Compare**:
           Assert that both intensity arrays match to within 1e-12
           absolute tolerance.

        **Expected assertions**

        The expanded and core intensities are identical. This agreement
        verifies conversion of both incident angles.
        """
        eigenbands: Array
        surface_orb: Array
        params: diffpes.types.SimulationParams
        kpoints: Array
        bands: diffpes.types.BandStructure
        orb_proj: diffpes.types.OrbitalProjection
        pol: diffpes.types.PolarizationConfig
        expected: diffpes.types.ArpesSpectrum
        wrapped: diffpes.types.ArpesSpectrum

        eigenbands, surface_orb = _make_synthetic_data()
        params = make_expanded_simulation_params(
            eigenbands=eigenbands,
            fidelity=220,
            sigma=0.05,
            temperature=25.0,
            photon_energy=21.2,
        )
        kpoints = jnp.zeros((eigenbands.shape[0], 3), dtype=jnp.float64)
        bands = make_band_structure(
            eigenvalues=eigenbands,
            kpoints=kpoints,
            fermi_energy=0.0,
        )
        orb_proj = make_orbital_projection(projections=surface_orb)
        pol = make_polarization_config(
            theta=jnp.deg2rad(jnp.float64(45.0)),
            phi=jnp.deg2rad(jnp.float64(30.0)),
            polarization_angle=jnp.float64(0.25),
            polarization_type="LHP",
        )
        expected = simulate_advanced(bands, orb_proj, params, pol)
        wrapped = simulate_advanced_expanded(
            eigenbands=eigenbands,
            surface_orb=surface_orb,
            ef=0.0,
            sigma=0.05,
            fidelity=220,
            temperature=25.0,
            photon_energy=21.2,
            polarization="LHP",
            incident_theta=45.0,
            incident_phi=30.0,
            polarization_angle=0.25,
        )
        chex.assert_trees_all_close(
            wrapped.intensity, expected.intensity, atol=1e-12
        )


class TestSimulateExpertExpanded(chex.TestCase):
    """Validate :func:`~diffpes.simul.simulate_expert_expanded`.

    Covers the full dipole and Voigt wrapper for raw arrays under unpolarized
    illumination at a reduced test fidelity.

    :see: :func:`~diffpes.simul.simulate_expert_expanded`
    """

    def test_returns_finite_unpolarized_spectrum(self) -> None:
        """Return finite expert intensity for unpolarized illumination.

        The expert wrapper must preserve twelve k-points and form forty-eight
        energy samples after its polarization and matrix-element stages.

        Notes
        -----
        The test evaluates the synthetic raw arrays with explicit Voigt and incidence
        parameters, then checks the output shape and finite values with Chex.
        """
        eigenbands: Array
        surface_orb: Array
        spectrum: diffpes.types.ArpesSpectrum

        eigenbands, surface_orb = _make_synthetic_data()
        spectrum = simulate_expert_expanded(
            eigenbands=eigenbands,
            surface_orb=surface_orb,
            ef=0.0,
            sigma=0.04,
            gamma=0.1,
            fidelity=48,
            temperature=15.0,
            photon_energy=21.2,
            polarization="unpolarized",
            incident_theta=45.0,
            incident_phi=0.0,
            polarization_angle=0.0,
        )
        chex.assert_shape(spectrum.intensity, (12, 48))
        chex.assert_tree_all_finite(spectrum.intensity)


class TestSimulateExpanded(chex.TestCase):
    """Validate :func:`diffpes.simul.expanded.simulate_expanded`.

    Verifies that the level-based dispatch function correctly routes to
    the appropriate expanded wrapper and produces identical results.

    :see: :func:`~diffpes.simul.simulate_expanded`
    """

    def test_dispatch_expert_matches_direct_wrapper(self) -> None:
        """Verify that dispatching with level="Expert" matches the direct wrapper.

        The test establishes the dispatch expert matches direct wrapper contract for
        expanded dispatch with the concrete values and array shapes described below.

        Notes
        -----
        1. **Run direct wrapper**:
           Call the direct wrapper with explicit parameters.

        2. **Run dispatcher**:
           Call ``simulate_expanded`` with ``level="Expert"`` and the
           same parameters. Note the capitalized level string, which
           tests case-insensitive dispatch.

        3. **Compare**:
           Assert that both intensity arrays match to within 1e-12
           absolute tolerance.

        **Expected assertions**

        The dispatched intensity is numerically identical to the direct
        expert wrapper, confirming that ``simulate_expanded`` correctly
        routes to ``simulate_expert_expanded`` and that level matching
        is case-insensitive.
        """
        eigenbands: Array
        surface_orb: Array
        expected: diffpes.types.ArpesSpectrum
        dispatched: diffpes.types.ArpesSpectrum

        eigenbands, surface_orb = _make_synthetic_data()
        expected = simulate_expert_expanded(
            eigenbands=eigenbands,
            surface_orb=surface_orb,
            ef=0.0,
            sigma=0.04,
            gamma=0.1,
            fidelity=200,
            temperature=15.0,
            photon_energy=11.0,
            polarization="unpolarized",
            incident_theta=45.0,
            incident_phi=0.0,
            polarization_angle=0.0,
        )
        dispatched = simulate_expanded(
            level="Expert",
            eigenbands=eigenbands,
            surface_orb=surface_orb,
            ef=0.0,
            sigma=0.04,
            gamma=0.1,
            fidelity=200,
            temperature=15.0,
            photon_energy=11.0,
            polarization="unpolarized",
            incident_theta=45.0,
            incident_phi=0.0,
            polarization_angle=0.0,
        )
        chex.assert_trees_all_close(
            dispatched.intensity, expected.intensity, atol=1e-12
        )

    def test_dispatch_novice_matches_direct_wrapper(self) -> None:
        """Verify that dispatching with level='novice' matches the direct wrapper.

        The test establishes the dispatch novice matches direct wrapper contract for
        expanded dispatch with the concrete values and array shapes described below.

        Notes
        -----
        1. **Run direct wrapper**:
           Call the direct wrapper with explicit parameters.

        2. **Run dispatcher**:
           Call ``simulate_expanded`` with ``level="novice"`` and the
           same parameters.

        3. **Compare**:
           Assert that both intensity arrays match to within 1e-12
           absolute tolerance.

        **Expected assertions**

        The dispatched intensity is numerically identical to the direct
        novice wrapper, confirming that ``simulate_expanded`` correctly
        routes to ``simulate_novice_expanded``.
        """
        eigenbands: Array
        surface_orb: Array
        expected: diffpes.types.ArpesSpectrum
        dispatched: diffpes.types.ArpesSpectrum

        eigenbands, surface_orb = _make_synthetic_data()
        expected = simulate_novice_expanded(
            eigenbands=eigenbands,
            surface_orb=surface_orb,
            ef=0.0,
            sigma=0.04,
            gamma=0.1,
            fidelity=200,
            temperature=15.0,
            photon_energy=11.0,
        )
        dispatched = simulate_expanded(
            level="novice",
            eigenbands=eigenbands,
            surface_orb=surface_orb,
            ef=0.0,
            sigma=0.04,
            gamma=0.1,
            fidelity=200,
            temperature=15.0,
            photon_energy=11.0,
        )
        chex.assert_trees_all_close(
            dispatched.intensity, expected.intensity, atol=1e-12
        )

    def test_dispatch_basic_matches_direct_wrapper(self) -> None:
        """Verify that dispatching with level='basic' matches the direct wrapper.

        The test establishes the dispatch basic matches direct wrapper contract for
        expanded dispatch with the concrete values and array shapes described below.

        Notes
        -----
        1. **Run direct wrapper**:
           Call the direct wrapper with explicit parameters.

        2. **Run dispatcher**:
           Call ``simulate_expanded`` with ``level="basic"`` and the
           same parameters.

        3. **Compare**:
           Assert that both intensity arrays match to within 1e-12
           absolute tolerance.

        **Expected assertions**

        The dispatched intensity is numerically identical to the direct
        basic wrapper, confirming that ``simulate_expanded`` correctly
        routes to ``simulate_basic_expanded``.
        """
        eigenbands: Array
        surface_orb: Array
        expected: diffpes.types.ArpesSpectrum
        dispatched: diffpes.types.ArpesSpectrum

        eigenbands, surface_orb = _make_synthetic_data()
        expected = simulate_basic_expanded(
            eigenbands=eigenbands,
            surface_orb=surface_orb,
            ef=0.0,
            sigma=0.04,
            fidelity=200,
            temperature=15.0,
            photon_energy=11.0,
        )
        dispatched = simulate_expanded(
            level="basic",
            eigenbands=eigenbands,
            surface_orb=surface_orb,
            ef=0.0,
            sigma=0.04,
            fidelity=200,
            temperature=15.0,
            photon_energy=11.0,
        )
        chex.assert_trees_all_close(
            dispatched.intensity, expected.intensity, atol=1e-12
        )

    def test_dispatch_basicplus_matches_direct_wrapper(self) -> None:
        """Verify that dispatching with level='basicplus' matches the direct wrapper.

        The test establishes the dispatch basicplus matches direct wrapper contract for
        expanded dispatch with the concrete values and array shapes described below.

        Notes
        -----
        1. **Run direct wrapper**:
           Call the direct wrapper with explicit parameters.

        2. **Run dispatcher**:
           Call ``simulate_expanded`` with ``level="basicplus"`` and the
           same parameters.

        3. **Compare**:
           Assert that both intensity arrays match to within 1e-12
           absolute tolerance.

        **Expected assertions**

        The dispatched intensity is numerically identical to the direct
        basicplus wrapper, confirming that ``simulate_expanded`` correctly
        routes to ``simulate_basicplus_expanded``.
        """
        eigenbands: Array
        surface_orb: Array
        expected: diffpes.types.ArpesSpectrum
        dispatched: diffpes.types.ArpesSpectrum

        eigenbands, surface_orb = _make_synthetic_data()
        expected = simulate_basicplus_expanded(
            eigenbands=eigenbands,
            surface_orb=surface_orb,
            ef=0.0,
            sigma=0.04,
            fidelity=200,
            temperature=15.0,
            photon_energy=11.0,
        )
        dispatched = simulate_expanded(
            level="basicplus",
            eigenbands=eigenbands,
            surface_orb=surface_orb,
            ef=0.0,
            sigma=0.04,
            fidelity=200,
            temperature=15.0,
            photon_energy=11.0,
        )
        chex.assert_trees_all_close(
            dispatched.intensity, expected.intensity, atol=1e-12
        )

    def test_dispatch_advanced_matches_direct_wrapper(self) -> None:
        """Verify that dispatching with level='advanced' matches the direct wrapper.

        The test establishes the dispatch advanced matches direct wrapper contract for
        expanded dispatch with the concrete values and array shapes described below.

        Notes
        -----
        1. **Run direct wrapper**:
           Call the direct wrapper with explicit parameters.

        2. **Run dispatcher**:
           Call ``simulate_expanded`` with ``level="advanced"`` and the
           same parameters.

        3. **Compare**:
           Assert that both intensity arrays match to within 1e-12
           absolute tolerance.

        **Expected assertions**

        The dispatched intensity is numerically identical to the direct
        advanced wrapper, confirming that ``simulate_expanded`` correctly
        routes to ``simulate_advanced_expanded``.
        """
        eigenbands: Array
        surface_orb: Array
        expected: diffpes.types.ArpesSpectrum
        dispatched: diffpes.types.ArpesSpectrum

        eigenbands, surface_orb = _make_synthetic_data()
        expected = simulate_advanced_expanded(
            eigenbands=eigenbands,
            surface_orb=surface_orb,
            ef=0.0,
            sigma=0.04,
            fidelity=200,
            temperature=15.0,
            photon_energy=11.0,
            polarization="unpolarized",
            incident_theta=45.0,
            incident_phi=0.0,
            polarization_angle=0.0,
        )
        dispatched = simulate_expanded(
            level="advanced",
            eigenbands=eigenbands,
            surface_orb=surface_orb,
            ef=0.0,
            sigma=0.04,
            fidelity=200,
            temperature=15.0,
            photon_energy=11.0,
            polarization="unpolarized",
            incident_theta=45.0,
            incident_phi=0.0,
            polarization_angle=0.0,
        )
        chex.assert_trees_all_close(
            dispatched.intensity, expected.intensity, atol=1e-12
        )

    def test_dispatch_unknown_level_raises(self) -> None:
        """Verify that an unknown level raises ValueError with expected message.

        The test establishes the dispatch unknown level raises contract for expanded
        dispatch with the concrete values and array shapes described below.

        Notes
        -----
        1. **Call dispatcher with invalid level**:
           Call ``simulate_expanded(level="invalid", ...)`` with minimal
           required arguments (eigenbands, surface_orb, ef).

        2. **Check exception type**:
           Expect a ``ValueError``.

        3. **Check error message**:
           Verify the error description and one supported level in the message.

        **Expected assertions**

        The function raises ``ValueError`` with an error description and valid
        level names.
        """
        ctx: Any

        eigenbands: Array
        surface_orb: Array

        eigenbands, surface_orb = _make_synthetic_data()
        with self.assertRaises(ValueError) as ctx:
            simulate_expanded(
                level="invalid",
                eigenbands=eigenbands,
                surface_orb=surface_orb,
                ef=0.0,
            )
        self.assertIn("Unknown simulation level", str(ctx.exception))
        self.assertIn("novice", str(ctx.exception))

    def test_dispatch_soc_requires_surface_spin(self) -> None:
        """Verify that level='soc' without surface_spin raises ValueError.

        The test establishes the dispatch soc requires surface spin contract for
        expanded dispatch with the concrete values and array shapes described below.

        Notes
        -----
        1. **Setup**: Synthetic eigenbands and surface_orb (no
           surface_spin passed).
        2. **Call**: Invoke ``simulate_expanded(level="soc", ...)``
           with only the required array args and ef.
        3. **Check**: Expect ``ValueError`` and ``"surface_spin"`` in its message.

        **Expected assertions**

        Dispatching to the SOC level without providing spin data
        raises ``ValueError`` with a clear requirement for
        surface_spin.
        """
        ctx: Any

        eigenbands: Array
        surface_orb: Array

        eigenbands, surface_orb = _make_synthetic_data()
        with self.assertRaises(ValueError) as ctx:
            simulate_expanded(
                level="soc",
                eigenbands=eigenbands,
                surface_orb=surface_orb,
                ef=0.0,
            )
        self.assertIn("surface_spin", str(ctx.exception))

    def test_dispatch_soc_matches_direct_wrapper(self) -> None:
        """Verify that level='soc' with surface_spin matches simulate_soc_expanded.

        The test establishes the dispatch soc matches direct wrapper contract for
        expanded dispatch with the concrete values and array shapes described below.

        Notes
        -----
        1. **Setup**: Synthetic eigenbands and surface_orb; build a
           surface_spin array of shape (K, B, A, 6) with non-zero
           z components.
        2. **Run direct**: Call ``simulate_soc_expanded`` with the same
           scalar parameters to get the expected spectrum.
        3. **Run dispatcher**: Call ``simulate_expanded(level="soc",
           ..., surface_spin=surface_spin, ls_scale=0.01)``.
        4. **Compare**: Assert dispatched and expected intensities
           match to within 1e-12.

        **Expected assertions**

        The level-based dispatcher with ``level="soc"`` and
        surface_spin produces the same result as calling
        ``simulate_soc_expanded`` directly.
        """
        eigenbands: Array
        surface_orb: Array
        nk: int
        nb: int
        na: int
        surface_spin: Array
        expected: diffpes.types.ArpesSpectrum
        dispatched: diffpes.types.ArpesSpectrum

        eigenbands, surface_orb = _make_synthetic_data()
        nk, nb, na = (
            surface_orb.shape[0],
            surface_orb.shape[1],
            surface_orb.shape[2],
        )
        surface_spin = jnp.zeros((nk, nb, na, 6), dtype=jnp.float64)
        surface_spin = surface_spin.at[..., 4].set(0.15)
        surface_spin = surface_spin.at[..., 5].set(0.08)
        expected = simulate_soc_expanded(
            eigenbands=eigenbands,
            surface_orb=surface_orb,
            surface_spin=surface_spin,
            ef=0.0,
            sigma=0.04,
            gamma=0.1,
            fidelity=200,
            temperature=15.0,
            photon_energy=11.0,
            polarization="unpolarized",
            incident_theta=45.0,
            incident_phi=0.0,
            polarization_angle=0.0,
            ls_scale=0.01,
        )
        dispatched = simulate_expanded(
            level="soc",
            eigenbands=eigenbands,
            surface_orb=surface_orb,
            surface_spin=surface_spin,
            ef=0.0,
            sigma=0.04,
            gamma=0.1,
            fidelity=200,
            temperature=15.0,
            photon_energy=11.0,
            polarization="unpolarized",
            incident_theta=45.0,
            incident_phi=0.0,
            polarization_angle=0.0,
            ls_scale=0.01,
        )
        chex.assert_trees_all_close(
            dispatched.intensity, expected.intensity, atol=1e-12
        )


class TestSimulateSocExpanded(chex.TestCase):
    """Validate :func:`diffpes.simul.expanded.simulate_soc_expanded`.

    The tests compare manually built PyTrees with the wrapper PyTrees.
    They also compare the wrapper result with :func:`~diffpes.simul.simulate_soc`.

    :see: :func:`~diffpes.simul.simulate_soc_expanded`
    """

    def test_matches_core_soc_simulation(self) -> None:
        """Verify that the expanded SOC wrapper matches the core simulate_soc.

        The test establishes the matches core soc simulation contract for expanded soc
        wrapper with the concrete values and array shapes described below.

        Notes
        -----
        1. **Build reference manually**: Use :func:`_build_inputs` with
           surface_spin to get bands and orb_proj; build params via
           :func:`make_expanded_simulation_params` (fidelity=180,
           sigma=0.05, gamma=0.08, etc.); build pol via
           :func:`_build_polarization` (unpolarized, 45°, 0°). Call
           ``simulate_soc(bands, orb_proj, params, pol, ls_scale=0.02)``
           to get the expected spectrum.
        2. **Run expanded wrapper**: Call ``simulate_soc_expanded`` with
           the same raw arrays and scalar parameters (including
           ls_scale=0.02).
        3. **Compare**: Assert that wrapped and expected intensity
           and energy_axis match to within 1e-12.

        **Expected assertions**

        The expanded SOC wrapper produces numerically identical
        output to the core SOC simulation when given the same
        physical parameters.
        """
        eigenbands: Array
        surface_orb: Array
        nk: int
        nb: int
        na: int
        surface_spin: Array
        bands: diffpes.types.BandStructure
        soc_proj: diffpes.types.SpinOrbitalProjection
        params: diffpes.types.SimulationParams
        pol: diffpes.types.PolarizationConfig
        expected: diffpes.types.ArpesSpectrum
        wrapped: diffpes.types.ArpesSpectrum

        eigenbands, surface_orb = _make_synthetic_data()
        nk, nb, na = (
            surface_orb.shape[0],
            surface_orb.shape[1],
            surface_orb.shape[2],
        )
        surface_spin = jnp.zeros((nk, nb, na, 6), dtype=jnp.float64)
        surface_spin = surface_spin.at[..., 0].set(0.1)
        surface_spin = surface_spin.at[..., 4].set(0.2)
        from diffpes.simul.expanded import _build_inputs, _build_polarization

        bands, _ = _build_inputs(
            eigenbands=eigenbands,
            surface_orb=surface_orb,
            ef=0.0,
        )
        soc_proj = make_spin_orbital_projection(
            projections=jnp.asarray(surface_orb, dtype=jnp.float64),
            spin=surface_spin,
        )
        params = make_expanded_simulation_params(
            eigenbands=eigenbands,
            fidelity=180,
            sigma=0.05,
            gamma=0.08,
            temperature=20.0,
            photon_energy=12.0,
        )
        pol = _build_polarization(
            polarization="unpolarized",
            incident_theta=45.0,
            incident_phi=0.0,
        )
        expected = simulate_soc(bands, soc_proj, params, pol, ls_scale=0.02)
        wrapped = simulate_soc_expanded(
            eigenbands=eigenbands,
            surface_orb=surface_orb,
            surface_spin=surface_spin,
            ef=0.0,
            sigma=0.05,
            gamma=0.08,
            fidelity=180,
            temperature=20.0,
            photon_energy=12.0,
            polarization="unpolarized",
            incident_theta=45.0,
            incident_phi=0.0,
            polarization_angle=0.0,
            ls_scale=0.02,
        )
        chex.assert_trees_all_close(
            wrapped.intensity, expected.intensity, atol=1e-12
        )
        chex.assert_trees_all_close(
            wrapped.energy_axis, expected.energy_axis, atol=1e-12
        )
