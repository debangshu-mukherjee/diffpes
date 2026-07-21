"""Validate ARPES simulation spectrum functions.

Extended Summary
----------------
Validate all simulation levels in :mod:`diffpes.simul.spectrum`. Check
shapes, value constraints, and sensitivity to physical parameters.
Generate synthetic data for the tests. Require spin data for SOC tests.

"""

import chex
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any, Callable
from jaxtyping import Array, Float, jaxtyped

import diffpes
from diffpes.simul import (
    simulate_advanced,
    simulate_basic,
    simulate_basicplus,
    simulate_expert,
    simulate_novice,
    simulate_soc,
)
from diffpes.types import (
    BandStructure,
    OrbitalProjection,
    SpinOrbitalProjection,
    make_band_structure,
    make_orbital_projection,
    make_polarization_config,
    make_simulation_params,
    make_spin_orbital_projection,
)


@jaxtyped(typechecker=beartype)
def _make_synthetic_data(
    nk: int = 20,
    nb: int = 5,
    na: int = 2,
) -> tuple[BandStructure, OrbitalProjection]:
    """Generate synthetic band structure and orbital projections for testing.

    Create linearly spaced eigenvalues from -2.0 to 0.5 eV. Place
    k-points along the x-axis and set all orbital projections to 0.1.

    Parameters
    ----------
    nk : int, optional
        Number of k-points. Default is 20.
    nb : int, optional
        Number of bands. Default is 5.
    na : int, optional
        Number of atoms. Default is 2.

    Returns
    -------
    bands : BandStructure
        Band structure with linearly spaced eigenvalues, 1-D k-path,
        and Fermi energy at 0.0 eV.
    orb_proj : OrbitalProjection
        Uniform orbital projections of shape ``(nk, nb, na, 9)`` with
        all entries set to 0.1.
    """
    eigenvalues: Float[Array, "nk nb"] = jnp.linspace(
        -2.0, 0.5, nk * nb
    ).reshape(nk, nb)
    kpoints: Float[Array, "nk 3"] = jnp.zeros((nk, 3))
    kpoints = kpoints.at[:, 0].set(jnp.linspace(0.0, 1.0, nk))
    bands: BandStructure = make_band_structure(
        eigenvalues=eigenvalues,
        kpoints=kpoints,
        fermi_energy=0.0,
    )
    projections: Float[Array, "nk nb na 9"] = jnp.ones((nk, nb, na, 9)) * 0.1
    orb_proj: OrbitalProjection = make_orbital_projection(
        projections=projections
    )
    synthetic_data: tuple[BandStructure, OrbitalProjection] = (
        bands,
        orb_proj,
    )
    return synthetic_data


class TestSimulateNovice(chex.TestCase):
    """Validate :func:`diffpes.simul.spectrum.simulate_novice`.

    Verifies the novice-level simulation (Voigt broadening with uniform
    orbital weights) including output tensor shapes and non-negativity of
    the intensity map.

    :see: :func:`~diffpes.simul.simulate_novice`
    """

    def test_output_shape(self) -> None:
        """Verify that intensity and energy axis have the expected shapes.

        The test establishes the output shape contract for simulate novice with the
        concrete values and array shapes described below.

        Notes
        -----
        1. **Setup**:
           Generate synthetic data with 20 k-points and 5 bands, and
           create simulation parameters with fidelity=500 energy points.

        2. **Simulate**:
           Run ``simulate_novice`` to produce an ``ArpesSpectrum``.

        3. **Check shapes**:
           Assert that ``intensity`` is ``(20, 500)`` (k-points by energy)
           and ``energy_axis`` is ``(500,)``.

        **Expected assertions**

        The intensity shape matches ``(n_kpoints, fidelity)`` and the
        energy axis length matches ``fidelity``.
        """
        bands: diffpes.types.BandStructure
        orb_proj: diffpes.types.OrbitalProjection
        params: diffpes.types.SimulationParams
        spectrum: diffpes.types.ArpesSpectrum

        bands, orb_proj = _make_synthetic_data()
        params = make_simulation_params(fidelity=500)
        spectrum = simulate_novice(bands, orb_proj, params)
        chex.assert_shape(spectrum.intensity, (20, 500))
        chex.assert_shape(spectrum.energy_axis, (500,))

    def test_nonnegative_intensity(self) -> None:
        """Verify that all intensity values are non-negative.

        The test establishes the nonnegative intensity contract for simulate novice
        with the concrete values and array shapes described below.

        Notes
        -----
        1. **Setup**:
           Generate synthetic data and simulation parameters with
           fidelity=200.

        2. **Simulate**:
           Run ``simulate_novice`` to produce an ``ArpesSpectrum``.

        3. **Check non-negativity**:
           Assert that the minimum intensity value is at least ``-1e-10``
           (allowing for negligible floating-point undershoot).

        **Expected assertions**

        The minimum intensity is effectively non-negative, confirming that
        the Voigt convolution with Fermi-Dirac occupation does not produce
        physically impossible negative spectral weight.
        """
        bands: diffpes.types.BandStructure
        orb_proj: diffpes.types.OrbitalProjection
        params: diffpes.types.SimulationParams
        spectrum: diffpes.types.ArpesSpectrum

        bands, orb_proj = _make_synthetic_data()
        params = make_simulation_params(fidelity=200)
        spectrum = simulate_novice(bands, orb_proj, params)
        assert float(jnp.min(spectrum.intensity)) >= -1e-10


class TestSimulateBasic(chex.TestCase):
    """Validate :func:`diffpes.simul.spectrum.simulate_basic`.

    Verifies the basic-level simulation (Gaussian broadening with heuristic
    orbital weights) including output tensor shape and finiteness of all
    intensity values.

    :see: :func:`~diffpes.simul.simulate_basic`
    """

    def test_output_shape(self) -> None:
        """Verify that the intensity array has the expected shape.

        The test establishes the output shape contract for simulate basic with the
        concrete values and array shapes described below.

        Notes
        -----
        1. **Setup**:
           Generate synthetic data with 20 k-points and 5 bands, and
           create simulation parameters with fidelity=500.

        2. **Simulate**:
           Run ``simulate_basic`` to produce an ``ArpesSpectrum``.

        3. **Check shape**:
           Assert that ``intensity`` is ``(20, 500)``.

        **Expected assertions**

        The intensity shape matches ``(n_kpoints, fidelity)``.
        """
        bands: diffpes.types.BandStructure
        orb_proj: diffpes.types.OrbitalProjection
        params: diffpes.types.SimulationParams
        spectrum: diffpes.types.ArpesSpectrum

        bands, orb_proj = _make_synthetic_data()
        params = make_simulation_params(fidelity=500)
        spectrum = simulate_basic(bands, orb_proj, params)
        chex.assert_shape(spectrum.intensity, (20, 500))

    def test_finite_values(self) -> None:
        """Verify that all intensity values are finite (no NaN or Inf).

        The test establishes the finite values contract for simulate basic with the
        concrete values and array shapes described below.

        Notes
        -----
        1. **Setup**:
           Generate synthetic data and simulation parameters with
           fidelity=200.

        2. **Simulate**:
           Run ``simulate_basic`` to produce an ``ArpesSpectrum``.

        3. **Check finiteness**:
           Use ``chex.assert_tree_all_finite`` to confirm that no element
           of the intensity array is NaN or infinite.

        **Expected assertions**

        Every element of the intensity array is finite, confirming that
        the Gaussian convolution and heuristic weighting do not produce
        numerical overflow or undefined values.
        """
        bands: diffpes.types.BandStructure
        orb_proj: diffpes.types.OrbitalProjection
        params: diffpes.types.SimulationParams
        spectrum: diffpes.types.ArpesSpectrum

        bands, orb_proj = _make_synthetic_data()
        params = make_simulation_params(fidelity=200)
        spectrum = simulate_basic(bands, orb_proj, params)
        chex.assert_tree_all_finite(spectrum.intensity)


class TestSimulateBasicplus(chex.TestCase):
    """Validate :func:`diffpes.simul.spectrum.simulate_basicplus`.

    Verifies the basicplus-level simulation (Gaussian broadening with
    Yeh-Lindau photoionization cross-sections) including output tensor
    shape and sensitivity of the spectrum to changes in photon energy.

    :see: :func:`~diffpes.simul.simulate_basicplus`
    """

    def test_output_shape(self) -> None:
        """Verify that the intensity array has the expected shape.

        The test establishes the output shape contract for simulate basicplus with the
        concrete values and array shapes described below.

        Notes
        -----
        1. **Setup**:
           Generate synthetic data with 20 k-points and 5 bands, and
           create simulation parameters with fidelity=500.

        2. **Simulate**:
           Run ``simulate_basicplus`` to produce an ``ArpesSpectrum``.

        3. **Check shape**:
           Assert that ``intensity`` is ``(20, 500)``.

        **Expected assertions**

        The intensity shape matches ``(n_kpoints, fidelity)``.
        """
        bands: diffpes.types.BandStructure
        orb_proj: diffpes.types.OrbitalProjection
        params: diffpes.types.SimulationParams
        spectrum: diffpes.types.ArpesSpectrum

        bands, orb_proj = _make_synthetic_data()
        params = make_simulation_params(fidelity=500)
        spectrum = simulate_basicplus(bands, orb_proj, params)
        chex.assert_shape(spectrum.intensity, (20, 500))

    def test_yeh_lindau_affects_weights(self) -> None:
        """Verify that different photon energies produce different spectra.

        The test establishes the yeh lindau affects weights contract for simulate
        basicplus with the concrete values and array shapes described below.

        Notes
        -----
        1. **Setup**:
           Generate synthetic data and create two sets of simulation
           parameters that differ only in photon energy (20 eV vs 60 eV).

        2. **Simulate both**:
           Run ``simulate_basicplus`` with each parameter set to produce
           two spectra.

        3. **Compare**:
           Compute the total absolute difference between the two
           intensity arrays.

        **Expected assertions**

        The summed absolute intensity difference is positive. Thus,
        photon energy changes the Yeh-Lindau orbital weights.
        """
        bands: diffpes.types.BandStructure
        orb_proj: diffpes.types.OrbitalProjection
        params_low: diffpes.types.SimulationParams
        params_high: diffpes.types.SimulationParams
        spec_low: diffpes.types.ArpesSpectrum
        spec_high: diffpes.types.ArpesSpectrum
        diff: Array

        bands, orb_proj = _make_synthetic_data()
        params_low = make_simulation_params(fidelity=200, photon_energy=20.0)
        params_high = make_simulation_params(fidelity=200, photon_energy=60.0)
        spec_low = simulate_basicplus(bands, orb_proj, params_low)
        spec_high = simulate_basicplus(bands, orb_proj, params_high)
        diff = jnp.sum(jnp.abs(spec_low.intensity - spec_high.intensity))
        assert float(diff) > 0.0


class TestSimulateAdvanced(chex.TestCase):
    """Validate :func:`diffpes.simul.spectrum.simulate_advanced`.

    Verifies the advanced-level simulation (Gaussian broadening with
    Yeh-Lindau cross-sections and polarization selection rules) including
    output tensor shape and finiteness under unpolarized light conditions.

    :see: :func:`~diffpes.simul.simulate_advanced`
    """

    def test_output_shape(self) -> None:
        """Verify that intensity has the expected shape under LVP polarization.

        The test establishes the output shape contract for simulate advanced with the
        concrete values and array shapes described below.

        Notes
        -----
        1. **Setup**:
           Generate synthetic data with 20 k-points and 5 bands, create
           simulation parameters with fidelity=500, and configure linear
           vertical polarization (LVP).

        2. **Simulate**:
           Run ``simulate_advanced`` with the polarization config to
           produce an ``ArpesSpectrum``.

        3. **Check shape**:
           Assert that ``intensity`` is ``(20, 500)``.

        **Expected assertions**

        The intensity shape matches ``(n_kpoints, fidelity)``, confirming
        that the polarization pathway produces correctly shaped output.
        """
        bands: diffpes.types.BandStructure
        orb_proj: diffpes.types.OrbitalProjection
        params: diffpes.types.SimulationParams
        pol: diffpes.types.PolarizationConfig
        spectrum: diffpes.types.ArpesSpectrum

        bands, orb_proj = _make_synthetic_data()
        params = make_simulation_params(fidelity=500)
        pol = make_polarization_config(polarization_type="LVP")
        spectrum = simulate_advanced(bands, orb_proj, params, pol)
        chex.assert_shape(spectrum.intensity, (20, 500))

    def test_unpolarized(self) -> None:
        """Verify that unpolarized light produces finite intensity values.

        The test establishes the unpolarized contract for simulate advanced with the
        concrete values and array shapes described below.

        Notes
        -----
        1. **Setup**:
           Generate synthetic data, create simulation parameters with
           fidelity=200, and configure unpolarized light.

        2. **Simulate**:
           Run ``simulate_advanced`` with unpolarized configuration,
           which averages s- and p-polarization contributions.

        3. **Check finiteness**:
           Use ``chex.assert_tree_all_finite`` to confirm no NaN or Inf
           values in the intensity.

        **Expected assertions**

        All intensity values are finite, verifying that the unpolarized
        code path (averaging over orthogonal polarization vectors) does
        not introduce numerical instabilities.
        """
        bands: diffpes.types.BandStructure
        orb_proj: diffpes.types.OrbitalProjection
        params: diffpes.types.SimulationParams
        pol: diffpes.types.PolarizationConfig
        spectrum: diffpes.types.ArpesSpectrum

        bands, orb_proj = _make_synthetic_data()
        params = make_simulation_params(fidelity=200)
        pol = make_polarization_config(polarization_type="unpolarized")
        spectrum = simulate_advanced(bands, orb_proj, params, pol)
        chex.assert_tree_all_finite(spectrum.intensity)


class TestSimulateExpert(chex.TestCase):
    """Validate :func:`diffpes.simul.spectrum.simulate_expert`.

    Verify the expert simulation shape. Check finite results for
    unpolarized and circular light.

    :see: :func:`~diffpes.simul.simulate_expert`
    """

    def test_output_shape(self) -> None:
        """Verify that intensity has the expected shape under LHP polarization.

        The test establishes the output shape contract for simulate expert with the
        concrete values and array shapes described below.

        Notes
        -----
        1. **Setup**:
           Generate synthetic data with 20 k-points and 5 bands, create
           simulation parameters with fidelity=500, and configure linear
           horizontal polarization (LHP).

        2. **Simulate**:
           Run ``simulate_expert`` with the polarization config to
           produce an ``ArpesSpectrum``.

        3. **Check shape**:
           Assert that ``intensity`` is ``(20, 500)``.

        **Expected assertions**

        The intensity shape matches ``(n_kpoints, fidelity)``, confirming
        that the expert-level polarization and Voigt broadening pathway
        produces correctly shaped output.
        """
        bands: diffpes.types.BandStructure
        orb_proj: diffpes.types.OrbitalProjection
        params: diffpes.types.SimulationParams
        pol: diffpes.types.PolarizationConfig
        spectrum: diffpes.types.ArpesSpectrum

        bands, orb_proj = _make_synthetic_data()
        params = make_simulation_params(fidelity=500)
        pol = make_polarization_config(polarization_type="LHP")
        spectrum = simulate_expert(bands, orb_proj, params, pol)
        chex.assert_shape(spectrum.intensity, (20, 500))

    def test_unpolarized(self) -> None:
        """Verify that unpolarized light produces finite intensity values.

        The test establishes the unpolarized contract for simulate expert with the
        concrete values and array shapes described below.

        Notes
        -----
        1. **Setup**:
           Generate synthetic data, create simulation parameters with
           fidelity=200, and configure unpolarized light.

        2. **Simulate**:
           Run ``simulate_expert`` with unpolarized configuration,
           which averages s- and p-polarization dipole contributions.

        3. **Check finiteness**:
           Use ``chex.assert_tree_all_finite`` to confirm no NaN or Inf
           values in the intensity.

        **Expected assertions**

        All intensity values are finite, verifying that the unpolarized
        code path with Voigt broadening and dipole matrix elements does
        not introduce numerical instabilities.
        """
        bands: diffpes.types.BandStructure
        orb_proj: diffpes.types.OrbitalProjection
        params: diffpes.types.SimulationParams
        pol: diffpes.types.PolarizationConfig
        spectrum: diffpes.types.ArpesSpectrum

        bands, orb_proj = _make_synthetic_data()
        params = make_simulation_params(fidelity=200)
        pol = make_polarization_config(polarization_type="unpolarized")
        spectrum = simulate_expert(bands, orb_proj, params, pol)
        chex.assert_tree_all_finite(spectrum.intensity)

    def test_circular_polarization(self) -> None:
        """Verify that right circular polarization (RCP) produces finite values.

        The test establishes the circular polarization contract for simulate expert
        with the concrete values and array shapes described below.

        Notes
        -----
        1. **Setup**:
           Generate synthetic data, create simulation parameters with
           fidelity=200, and configure right circular polarization (RCP).

        2. **Simulate**:
           Run ``simulate_expert`` with RCP configuration, which
           builds a complex electric-field vector and evaluates dipole
           matrix elements with Voigt broadening.

        3. **Check finiteness**:
           Use ``chex.assert_tree_all_finite`` to confirm no NaN or Inf
           values in the intensity.

        **Expected assertions**

        All intensity values are finite. The complex electric field does
        not cause overflow or undefined values.
        """
        bands: diffpes.types.BandStructure
        orb_proj: diffpes.types.OrbitalProjection
        params: diffpes.types.SimulationParams
        pol: diffpes.types.PolarizationConfig
        spectrum: diffpes.types.ArpesSpectrum

        bands, orb_proj = _make_synthetic_data()
        params = make_simulation_params(fidelity=200)
        pol = make_polarization_config(polarization_type="RCP")
        spectrum = simulate_expert(bands, orb_proj, params, pol)
        chex.assert_tree_all_finite(spectrum.intensity)


@jaxtyped(typechecker=beartype)
def _make_synthetic_data_with_spin(
    nk: int = 20,
    nb: int = 5,
    na: int = 2,
) -> tuple[BandStructure, SpinOrbitalProjection]:
    """Generate synthetic band and orbital data including spin projections.

    Add a nonzero spin array to the base synthetic data. Return a
    :class:`~diffpes.types.SpinOrbitalProjection` for SOC tests.

    Parameters
    ----------
    nk : int, optional
        Number of k-points. Default is 20.
    nb : int, optional
        Number of bands. Default is 5.
    na : int, optional
        Number of atoms. Default is 2.

    Returns
    -------
    bands : BandStructure
        Band structure from :func:`_make_synthetic_data`.
    soc_proj : SpinOrbitalProjection
        Orbital projections with mandatory ``spin`` of shape
        (K, B, A, 6).
    """
    bands: BandStructure
    orb_proj: OrbitalProjection
    bands, orb_proj = _make_synthetic_data(nk=nk, nb=nb, na=na)
    spin: Float[Array, "nk nb na 6"] = jnp.zeros(
        (nk, nb, na, 6), dtype=jnp.float64
    )
    spin = spin.at[..., 0].set(0.1)
    spin = spin.at[..., 1].set(0.05)
    spin = spin.at[..., 2].set(0.0)
    spin = spin.at[..., 3].set(0.0)
    spin = spin.at[..., 4].set(0.2)
    spin = spin.at[..., 5].set(0.1)
    soc_proj: SpinOrbitalProjection = make_spin_orbital_projection(
        projections=orb_proj.projections,
        spin=spin,
    )
    synthetic_data: tuple[BandStructure, SpinOrbitalProjection] = (
        bands,
        soc_proj,
    )
    return synthetic_data


class TestSimulateSoc(chex.TestCase):
    """Validate :func:`diffpes.simul.spectrum.simulate_soc`.

    Verify that SOC requires spin data and produces the expected shapes.
    Compare zero and nonzero ``ls_scale`` values with expert results.

    :see: :func:`~diffpes.simul.simulate_soc`
    """

    def test_output_shape(self) -> None:
        """Verify SOC spectrum has intensity (K, E) and energy_axis (E).

        The test establishes the output shape contract for simulate soc with the
        concrete values and array shapes described below.

        Notes
        -----
        1. **Setup**: Build bands and orb_proj with spin via
           :func:`_make_synthetic_data_with_spin`; params with
           fidelity=100; unpolarized polarization config.
        2. **Simulate**: Run ``simulate_soc`` with ls_scale=0.01.
        3. **Check shapes**: Assert intensity shape ``(n_kpoints, 100)``
           and energy_axis shape ``(100,)``; assert all intensity values
           are finite.

        **Expected assertions**

        Output shapes match the expert convention and the spectrum
        contains no NaN or Inf.
        """
        bands: diffpes.types.BandStructure
        soc_proj: diffpes.types.SpinOrbitalProjection
        params: diffpes.types.SimulationParams
        pol: diffpes.types.PolarizationConfig
        spectrum: diffpes.types.ArpesSpectrum

        bands, soc_proj = _make_synthetic_data_with_spin()
        params = make_simulation_params(fidelity=100)
        pol = make_polarization_config(polarization_type="unpolarized")
        spectrum = simulate_soc(bands, soc_proj, params, pol, ls_scale=0.01)
        chex.assert_shape(
            spectrum.intensity, (bands.eigenvalues.shape[0], 100)
        )
        chex.assert_shape(spectrum.energy_axis, (100,))
        chex.assert_tree_all_finite(spectrum.intensity)

    def test_soc_ls_scale_zero_matches_expert(self) -> None:
        """Verify that zero ls_scale reproduces the expert intensity.

        The test establishes the soc ls scale zero matches expert contract for simulate
        soc with the concrete values and array shapes described below.

        Notes
        -----
        1. **Setup**: Synthetic data with spin; params and
           unpolarized pol_config.
        2. **Run both**: Call ``simulate_expert`` and
           ``simulate_soc(..., ls_scale=0.0)``.
        3. **Compare**: Assert intensity and energy_axis from both
           match to within 1e-12 absolute tolerance.

        **Expected assertions**

        With ls_scale=0, the SOC simulation reproduces the expert result.
        """
        bands: diffpes.types.BandStructure
        soc_proj: diffpes.types.SpinOrbitalProjection
        expert_orb: diffpes.types.OrbitalProjection
        params: diffpes.types.SimulationParams
        pol: diffpes.types.PolarizationConfig
        expert_spec: diffpes.types.ArpesSpectrum
        soc_spec: diffpes.types.ArpesSpectrum

        bands, soc_proj = _make_synthetic_data_with_spin()
        expert_orb = make_orbital_projection(
            projections=soc_proj.projections,
            spin=soc_proj.spin,
        )
        params = make_simulation_params(fidelity=100)
        pol = make_polarization_config(polarization_type="unpolarized")
        expert_spec = simulate_expert(bands, expert_orb, params, pol)
        soc_spec = simulate_soc(bands, soc_proj, params, pol, ls_scale=0.0)
        chex.assert_trees_all_close(
            expert_spec.intensity, soc_spec.intensity, atol=1e-12
        )
        chex.assert_trees_all_close(
            expert_spec.energy_axis, soc_spec.energy_axis, atol=1e-12
        )

    def test_soc_nonzero_ls_scale_differs_from_expert(self) -> None:
        """Verify that nonzero ``ls_scale`` can change the SOC intensity.

        The test establishes the soc nonzero ls scale differs from expert contract for
        simulate soc with the concrete values and array shapes described below.

        Notes
        -----
        1. **Setup**: Synthetic data with spin; params and
           unpolarized pol_config.
        2. **Run both**: Call ``simulate_expert`` and
           ``simulate_soc(..., ls_scale=0.1)``.
        3. **Compare**: Compute the maximum absolute difference
           between the two intensity arrays; assert it is greater
           than 1e-10.

        **Expected assertions**

        A nonzero ls_scale changes the intensity map. Thus, the spin
        correction affects the result.
        """
        bands: diffpes.types.BandStructure
        soc_proj: diffpes.types.SpinOrbitalProjection
        expert_orb: diffpes.types.OrbitalProjection
        params: diffpes.types.SimulationParams
        pol: diffpes.types.PolarizationConfig
        expert_spec: diffpes.types.ArpesSpectrum
        soc_spec: diffpes.types.ArpesSpectrum
        diff: Array

        bands, soc_proj = _make_synthetic_data_with_spin()
        expert_orb = make_orbital_projection(
            projections=soc_proj.projections,
            spin=soc_proj.spin,
        )
        params = make_simulation_params(fidelity=100)
        pol = make_polarization_config(polarization_type="unpolarized")
        expert_spec = simulate_expert(bands, expert_orb, params, pol)
        soc_spec = simulate_soc(bands, soc_proj, params, pol, ls_scale=0.1)
        diff = jnp.abs(soc_spec.intensity - expert_spec.intensity)
        self.assertGreater(jnp.max(diff), 1e-10)

    def test_soc_polarized_light(self) -> None:
        """Verify that SOC simulation runs with polarized light (LHP).

        The test establishes the soc polarized light contract for simulate soc with the
        concrete values and array shapes described below.

        Notes
        -----
        1. **Setup**: Build bands and spin-orbital projections. Create
           parameters with fidelity=100 and an LHP configuration. Use an
           explicit electric field for the dipole matrix elements.
        2. **Simulate**: Run ``simulate_soc`` with ls_scale=0.02.
        3. **Check shape and finiteness**: Assert intensity shape is
           ``(n_kpoints, 100)`` and all values are finite.

        **Expected assertions**

        The SOC simulation returns finite output with the expected shape.
        The polarized branch includes spin-orbit and dipole effects.
        """
        bands: diffpes.types.BandStructure
        soc_proj: diffpes.types.SpinOrbitalProjection
        params: diffpes.types.SimulationParams
        pol: diffpes.types.PolarizationConfig
        spectrum: diffpes.types.ArpesSpectrum

        bands, soc_proj = _make_synthetic_data_with_spin()
        params = make_simulation_params(fidelity=100)
        pol = make_polarization_config(
            polarization_type="LHP",
        )
        spectrum = simulate_soc(bands, soc_proj, params, pol, ls_scale=0.02)
        chex.assert_shape(
            spectrum.intensity, (bands.eigenvalues.shape[0], 100)
        )
        chex.assert_tree_all_finite(spectrum.intensity)
