"""ARPES spectrum simulation at six complexity levels (incl. spin-orbit).

Extended Summary
----------------
Provides six simulation functions of increasing physical
sophistication, from basic Voigt convolution (novice) to full
polarization-dependent dipole matrix element calculations
(expert) and spin-orbit (soc). All functions are vectorized with
``jax.vmap`` for efficient GPU execution.

Routine Listings
----------------
:func:`simulate_advanced`
    Simulate ARPES with Gaussian broadening and polarization rules.
:func:`simulate_basic`
    Simulate ARPES spectrum with Gaussian broadening and heuristic weights.
:func:`simulate_basicplus`
    Simulate ARPES with Gaussian broadening and Yeh-Lindau cross-sections.
:func:`simulate_expert`
    Simulate ARPES with Voigt broadening and dipole matrix elements.
:func:`simulate_novice`
    Simulate ARPES spectrum with Voigt broadening and uniform weights.
:func:`simulate_soc`
    Simulate ARPES with spin-orbit coupling (spin-dependent intensity).

Notes
-----
All functions accept three carriers.
Band data uses :class:`~diffpes.types.BandStructure`.
Orbital data uses :class:`~diffpes.types.OrbitalProjection`.
Settings use :class:`~diffpes.types.SimulationParams`.
Results use :class:`~diffpes.types.ArpesSpectrum`.

The ``if is_unpolarized`` branches are intentional Python-side dispatch
(config choice); they are not traced, so only one path is compiled per call.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Complex, Float, jaxtyped

from diffpes.types import (
    NON_S_ORBITAL_SLICE,
    ArpesSpectrum,
    BandStructure,
    OrbitalProjection,
    PolarizationConfig,
    ScalarFloat,
    SimulationParams,
    SpinOrbitalProjection,
    make_arpes_spectrum,
)

from .broadening import fermi_dirac, gaussian, voigt
from .crosssections import heuristic_weights, yeh_lindau_weights
from .polarization import (
    build_efield,
    build_polarization_vectors,
    dipole_matrix_elements,
    photon_wavevector,
)


@jaxtyped(typechecker=beartype)
def simulate_novice(
    bands: BandStructure,
    orb_proj: OrbitalProjection,
    params: SimulationParams,
) -> ArpesSpectrum:
    """Simulate ARPES spectrum with Voigt broadening and uniform weights.

    Entry-level simulation that convolves each band with a Voigt profile
    (combined Gaussian + Lorentzian) and applies Fermi-Dirac occupation.
    All non-s orbital projections (p, d) are summed with equal weight,
    making this the simplest model that still captures lifetime and
    instrumental broadening simultaneously.

    :see: :class:`~.test_spectrum.TestSimulateNovice`

    Implementation Logic
    --------------------
    1. **Build the energy axis**::

           energy_axis: Float[Array, " E"] = jnp.linspace(
               params.energy_min,
               params.energy_max,
               params.fidelity,
           )

       This defines the common energy grid for every k-point.

    2. **Vectorize the band accumulation**::

           intensity: Float[Array, "K E"] = jax.vmap(_single_kpoint)(
               bands.eigenvalues, weights
           )

       JAX maps the differentiable band calculation across all k-points.

    3. **Build the spectrum carrier**::

           spectrum: ArpesSpectrum = make_arpes_spectrum(
               intensity=intensity,
               energy_axis=energy_axis,
           )

       The factory validates and stores the computed arrays.

    Parameters
    ----------
    bands : BandStructure
        Electronic band structure containing eigenvalues of shape
        ``(K, B)`` and the Fermi energy.
    orb_proj : OrbitalProjection
        Orbital projections of shape ``(K, B, A, 9)`` where A is the
        number of atoms and 9 is the number of orbital channels
        (s, p_y, p_z, p_x, d_xy, d_yz, d_z2, d_xz, d_x2-y2).
    params : SimulationParams
        Simulation parameters including ``sigma``, ``gamma``,
        ``temperature``, ``energy_min``, ``energy_max``, and
        ``fidelity``.

    Returns
    -------
    spectrum : ArpesSpectrum
        Simulated ARPES intensity map with ``intensity`` of shape
        ``(K, E)`` and ``energy_axis`` of shape ``(E,)``.

    Notes
    -----
    Uses Voigt profile (combined Gaussian-Lorentzian) and sums all
    non-s orbital contributions with equal weight. This is appropriate
    when orbital cross-section data is unavailable or when a quick
    qualitative comparison with experiment is sufficient.

    See Also
    --------
    simulate_novice_expanded : Expanded variant that returns per-band
        contributions before summation.
    """
    energy_axis: Float[Array, " E"] = jnp.linspace(
        params.energy_min,
        params.energy_max,
        params.fidelity,
    )
    proj: Float[Array, "K B A 9"] = orb_proj.projections
    weights: Float[Array, "K B"] = jnp.sum(
        jnp.sum(proj[..., NON_S_ORBITAL_SLICE], axis=-1),
        axis=-1,
    )

    def _single_band(
        energy: Float[Array, " "],
        weight: Float[Array, " "],
    ) -> Float[Array, " E"]:
        """Spectral contribution of one band at one k-point (Voigt).

        Evaluates ``weight * f(E_band) * V(E; E_band, sigma, gamma)``
        where ``f`` is the Fermi-Dirac distribution and ``V`` is the
        Voigt profile. This yields the weighted spectral lineshape
        for a single band.

        Parameters
        ----------
        energy : Float[Array, " "]
            Band eigenvalue in eV.
        weight : Float[Array, " "]
            Uniform orbital weight for this band.

        Returns
        -------
        contribution : Float[Array, " E"]
            Weighted Voigt spectral contribution.
        """
        fd: Float[Array, " "] = fermi_dirac(
            energy, bands.fermi_energy, params.temperature
        )
        profile: Float[Array, " E"] = voigt(
            energy_axis, energy, params.sigma, params.gamma
        )
        contribution: Float[Array, " E"] = weight * fd * profile
        return contribution

    def _single_kpoint(
        energies: Float[Array, " B"],
        kweights: Float[Array, " B"],
    ) -> Float[Array, " E"]:
        """Sum spectral contributions over all bands at one k-point.

        Vmaps ``_single_band`` over the band axis B and sums the
        resulting ``(B, E)`` array along the band dimension.

        Parameters
        ----------
        energies : Float[Array, " B"]
            Band eigenvalues at this k-point.
        kweights : Float[Array, " B"]
            Orbital weights for each band at this k-point.

        Returns
        -------
        total : Float[Array, " E"]
            Total spectral intensity at this k-point.
        """
        contributions: Float[Array, "B E"] = jax.vmap(_single_band)(
            energies, kweights
        )
        total: Float[Array, " E"] = jnp.sum(contributions, axis=0)
        return total

    intensity: Float[Array, "K E"] = jax.vmap(_single_kpoint)(
        bands.eigenvalues, weights
    )
    spectrum: ArpesSpectrum = make_arpes_spectrum(
        intensity=intensity,
        energy_axis=energy_axis,
    )
    return spectrum


@jaxtyped(typechecker=beartype)
def simulate_basic(
    bands: BandStructure,
    orb_proj: OrbitalProjection,
    params: SimulationParams,
) -> ArpesSpectrum:
    """Simulate ARPES spectrum with Gaussian broadening and heuristic weights.

    Intermediate simulation that replaces the Voigt profile with a pure
    Gaussian and introduces energy-dependent heuristic orbital weights.
    The heuristic weights enhance p-orbital contributions below ~50 eV
    photon energy and d-orbital contributions above, providing a
    first-order approximation to photoionization cross-sections without
    requiring tabulated atomic data.

    :see: :class:`~.test_spectrum.TestSimulateBasic`

    Implementation Logic
    --------------------
    1. **Build the energy axis**::

           energy_axis: Float[Array, " E"] = jnp.linspace(
               params.energy_min,
               params.energy_max,
               params.fidelity,
           )

       This defines the common energy grid for every k-point.

    2. **Vectorize the band accumulation**::

           intensity: Float[Array, "K E"] = jax.vmap(_single_kpoint)(
               bands.eigenvalues, weights
           )

       JAX maps the differentiable Gaussian calculation across all k-points.

    3. **Build the spectrum carrier**::

           spectrum: ArpesSpectrum = make_arpes_spectrum(
               intensity=intensity,
               energy_axis=energy_axis,
           )

       The factory validates and stores the computed arrays.

    Parameters
    ----------
    bands : BandStructure
        Electronic band structure containing eigenvalues of shape
        ``(K, B)`` and the Fermi energy.
    orb_proj : OrbitalProjection
        Orbital projections of shape ``(K, B, A, 9)`` where A is the
        number of atoms and 9 is the number of orbital channels
        (s, p_y, p_z, p_x, d_xy, d_yz, d_z2, d_xz, d_x2-y2).
    params : SimulationParams
        Simulation parameters including ``sigma``, ``photon_energy``,
        ``temperature``, ``energy_min``, ``energy_max``, and
        ``fidelity``.

    Returns
    -------
    spectrum : ArpesSpectrum
        Simulated ARPES intensity map with ``intensity`` of shape
        ``(K, E)`` and ``energy_axis`` of shape ``(E,)``.

    Notes
    -----
    Uses Gaussian broadening with energy-dependent heuristic orbital
    weights (p-enhanced below 50 eV, d-enhanced above). This level is
    suitable when Yeh-Lindau cross-section tables are not available but
    some orbital selectivity is desired.
    """
    energy_axis: Float[Array, " E"] = jnp.linspace(
        params.energy_min,
        params.energy_max,
        params.fidelity,
    )
    orb_w: Float[Array, " 9"] = heuristic_weights(params.photon_energy)
    proj: Float[Array, "K B A 9"] = orb_proj.projections
    weighted_proj: Float[Array, "K B A 8"] = (
        proj[..., NON_S_ORBITAL_SLICE] * orb_w[NON_S_ORBITAL_SLICE]
    )
    weights: Float[Array, "K B"] = jnp.sum(
        jnp.sum(weighted_proj, axis=-1), axis=-1
    )

    def _single_band(
        energy: Float[Array, " "],
        weight: Float[Array, " "],
    ) -> Float[Array, " E"]:
        """Spectral contribution of one band at one k-point (Gaussian).

        Evaluates ``weight * f(E_band) * G(E; E_band, sigma)``
        where ``f`` is the Fermi-Dirac distribution and ``G`` is the
        Gaussian profile. The heuristic cross-section weighting is
        already incorporated into ``weight``.

        Parameters
        ----------
        energy : Float[Array, " "]
            Band eigenvalue in eV.
        weight : Float[Array, " "]
            Heuristic-weighted orbital projection for this band.

        Returns
        -------
        contribution : Float[Array, " E"]
            Weighted Gaussian spectral contribution.
        """
        fd: Float[Array, " "] = fermi_dirac(
            energy, bands.fermi_energy, params.temperature
        )
        profile: Float[Array, " E"] = gaussian(
            energy_axis, energy, params.sigma
        )
        contribution: Float[Array, " E"] = weight * fd * profile
        return contribution

    def _single_kpoint(
        energies: Float[Array, " B"],
        kweights: Float[Array, " B"],
    ) -> Float[Array, " E"]:
        """Sum Gaussian spectral contributions over all bands at one k-point.

        Vmaps ``_single_band`` over the band axis B and sums the
        resulting ``(B, E)`` array along the band dimension.

        Parameters
        ----------
        energies : Float[Array, " B"]
            Band eigenvalues at this k-point.
        kweights : Float[Array, " B"]
            Heuristic-weighted projections for each band.

        Returns
        -------
        total : Float[Array, " E"]
            Total spectral intensity at this k-point.
        """
        contributions: Float[Array, "B E"] = jax.vmap(_single_band)(
            energies, kweights
        )
        total: Float[Array, " E"] = jnp.sum(contributions, axis=0)
        return total

    intensity: Float[Array, "K E"] = jax.vmap(_single_kpoint)(
        bands.eigenvalues, weights
    )
    spectrum: ArpesSpectrum = make_arpes_spectrum(
        intensity=intensity,
        energy_axis=energy_axis,
    )
    return spectrum


@jaxtyped(typechecker=beartype)
def simulate_basicplus(
    bands: BandStructure,
    orb_proj: OrbitalProjection,
    params: SimulationParams,
) -> ArpesSpectrum:
    """Simulate ARPES with Gaussian broadening and Yeh-Lindau cross-sections.

    Upgrades the heuristic weights of ``simulate_basic`` to physically
    motivated Yeh-Lindau photoionization cross-sections interpolated at
    the experimental photon energy. Unlike the basic level, ALL orbital
    projections (including s) are first weighted by their respective
    cross-sections before summing the non-s channels, ensuring that
    cross-section magnitudes are correctly applied before orbital
    selection.

    :see: :class:`~.test_spectrum.TestSimulateBasicplus`

    Implementation Logic
    --------------------
    1. **Build the energy axis**::

           energy_axis: Float[Array, " E"] = jnp.linspace(
               params.energy_min,
               params.energy_max,
               params.fidelity,
           )

       This defines the common energy grid for every k-point.

    2. **Vectorize the band accumulation**::

           intensity: Float[Array, "K E"] = jax.vmap(_single_kpoint)(
               bands.eigenvalues, weights
           )

       JAX maps the weighted Gaussian calculation across all k-points.

    3. **Build the spectrum carrier**::

           spectrum: ArpesSpectrum = make_arpes_spectrum(
               intensity=intensity,
               energy_axis=energy_axis,
           )

       The factory validates and stores the computed arrays.

    Parameters
    ----------
    bands : BandStructure
        Electronic band structure containing eigenvalues of shape
        ``(K, B)`` and the Fermi energy.
    orb_proj : OrbitalProjection
        Orbital projections of shape ``(K, B, A, 9)`` where A is the
        number of atoms and 9 is the number of orbital channels
        (s, p_y, p_z, p_x, d_xy, d_yz, d_z2, d_xz, d_x2-y2).
    params : SimulationParams
        Simulation parameters including ``sigma``, ``photon_energy``,
        ``temperature``, ``energy_min``, ``energy_max``, and
        ``fidelity``.

    Returns
    -------
    spectrum : ArpesSpectrum
        Simulated ARPES intensity map with ``intensity`` of shape
        ``(K, E)`` and ``energy_axis`` of shape ``(E,)``.

    Notes
    -----
    Uses Gaussian broadening with interpolated Yeh-Lindau
    photoionization cross-section weights per orbital type. The
    cross-sections are computed from tabulated atomic data and
    interpolated to the specified photon energy.
    """
    energy_axis: Float[Array, " E"] = jnp.linspace(
        params.energy_min,
        params.energy_max,
        params.fidelity,
    )
    orb_w: Float[Array, " 9"] = yeh_lindau_weights(params.photon_energy)
    proj: Float[Array, "K B A 9"] = orb_proj.projections
    weighted_proj: Float[Array, "K B A 9"] = proj * orb_w
    weights: Float[Array, "K B"] = jnp.sum(
        jnp.sum(
            weighted_proj[..., NON_S_ORBITAL_SLICE],
            axis=-1,
        ),
        axis=-1,
    )

    def _single_band(
        energy: Float[Array, " "],
        weight: Float[Array, " "],
    ) -> Float[Array, " E"]:
        """Spectral contribution of one band (Gaussian, Yeh-Lindau weights).

        Evaluates ``weight * f(E_band) * G(E; E_band, sigma)``
        where ``f`` is the Fermi-Dirac distribution and ``G`` is the
        Gaussian profile. The Yeh-Lindau cross-section weighting is
        already incorporated into ``weight``.

        Parameters
        ----------
        energy : Float[Array, " "]
            Band eigenvalue in eV.
        weight : Float[Array, " "]
            Yeh-Lindau-weighted orbital projection for this band.

        Returns
        -------
        contribution : Float[Array, " E"]
            Weighted Gaussian spectral contribution.
        """
        fd: Float[Array, " "] = fermi_dirac(
            energy, bands.fermi_energy, params.temperature
        )
        profile: Float[Array, " E"] = gaussian(
            energy_axis, energy, params.sigma
        )
        contribution: Float[Array, " E"] = weight * fd * profile
        return contribution

    def _single_kpoint(
        energies: Float[Array, " B"],
        kweights: Float[Array, " B"],
    ) -> Float[Array, " E"]:
        """Sum Yeh-Lindau Gaussian contributions over bands at one k-point.

        Vmaps ``_single_band`` over the band axis B and sums the
        resulting ``(B, E)`` array along the band dimension.

        Parameters
        ----------
        energies : Float[Array, " B"]
            Band eigenvalues at this k-point.
        kweights : Float[Array, " B"]
            Yeh-Lindau-weighted projections for each band.

        Returns
        -------
        total : Float[Array, " E"]
            Total spectral intensity at this k-point.
        """
        contributions: Float[Array, "B E"] = jax.vmap(_single_band)(
            energies, kweights
        )
        total: Float[Array, " E"] = jnp.sum(contributions, axis=0)
        return total

    intensity: Float[Array, "K E"] = jax.vmap(_single_kpoint)(
        bands.eigenvalues, weights
    )
    spectrum: ArpesSpectrum = make_arpes_spectrum(
        intensity=intensity,
        energy_axis=energy_axis,
    )
    return spectrum


@jaxtyped(typechecker=beartype)
def simulate_advanced(
    bands: BandStructure,
    orb_proj: OrbitalProjection,
    params: SimulationParams,
    pol_config: PolarizationConfig,
) -> ArpesSpectrum:
    """Simulate ARPES with Gaussian broadening and polarization rules.

    Extends ``simulate_basicplus`` by incorporating light polarization
    dependence through dipole matrix elements. The intensity for each
    orbital channel is weighted by ``|E . d_orbital|^2`` where E is the
    electric-field polarization vector and d_orbital is the dipole
    selection vector. Supports both polarized (linear, circular) and
    unpolarized light configurations.

    :see: :class:`~.test_spectrum.TestSimulateAdvanced`

    Implementation Logic
    --------------------
    1. **Build the energy axis**::

           energy_axis: Float[Array, " E"] = jnp.linspace(
               params.energy_min,
               params.energy_max,
               params.fidelity,
           )

       This defines the common energy grid for every k-point.

    2. **Vectorize the polarized band accumulation**::

           intensity: Float[Array, "K E"] = jax.vmap(_single_kpoint)(
               bands.eigenvalues, band_intensity
           )

       JAX maps the differentiable polarization model across all k-points.

    3. **Build the spectrum carrier**::

           spectrum: ArpesSpectrum = make_arpes_spectrum(
               intensity=intensity,
               energy_axis=energy_axis,
           )

       The factory validates and stores the computed arrays.

    Parameters
    ----------
    bands : BandStructure
        Electronic band structure containing eigenvalues of shape
        ``(K, B)`` and the Fermi energy.
    orb_proj : OrbitalProjection
        Orbital projections of shape ``(K, B, A, 9)`` where A is the
        number of atoms and 9 is the number of orbital channels
        (s, p_y, p_z, p_x, d_xy, d_yz, d_z2, d_xz, d_x2-y2).
    params : SimulationParams
        Simulation parameters including ``sigma``, ``photon_energy``,
        ``temperature``, ``energy_min``, ``energy_max``, and
        ``fidelity``.
    pol_config : PolarizationConfig
        Light polarization configuration specifying
        ``polarization_type`` (``"unpolarized"``, ``"linear"``, etc.),
        incidence angles ``theta`` and ``phi``, and any additional
        polarization parameters.

    Returns
    -------
    spectrum : ArpesSpectrum
        Simulated ARPES intensity map with ``intensity`` of shape
        ``(K, E)`` and ``energy_axis`` of shape ``(E,)``.

    Notes
    -----
    Uses Gaussian broadening with Yeh-Lindau cross-sections and
    polarization-dependent orbital selection via
    ``|E . d_orbital|^2`` weighting. For unpolarized light, the s- and
    p-polarization intensities are averaged, which is exact for
    incoherent superposition of orthogonal polarization states.
    """
    energy_axis: Float[Array, " E"] = jnp.linspace(
        params.energy_min,
        params.energy_max,
        params.fidelity,
    )
    orb_w: Float[Array, " 9"] = yeh_lindau_weights(params.photon_energy)
    proj: Float[Array, "K B A 9"] = orb_proj.projections
    is_unpolarized: bool = (
        pol_config.polarization_type.lower() == "unpolarized"
    )
    if is_unpolarized:
        e_s: Float[Array, " 3"]
        e_p: Float[Array, " 3"]
        e_s, e_p = build_polarization_vectors(pol_config.theta, pol_config.phi)
        m_s: Float[Array, " 9"] = dipole_matrix_elements(
            e_s.astype(jnp.complex128)
        )
        m_p: Float[Array, " 9"] = dipole_matrix_elements(
            e_p.astype(jnp.complex128)
        )
        w_s: Float[Array, "K B A 9"] = proj * orb_w * m_s
        w_p: Float[Array, "K B A 9"] = proj * orb_w * m_p
        ws_sum: Float[Array, "K B"] = jnp.sum(
            jnp.sum(w_s[..., NON_S_ORBITAL_SLICE], axis=-1),
            axis=-1,
        )
        wp_sum: Float[Array, "K B"] = jnp.sum(
            jnp.sum(w_p[..., NON_S_ORBITAL_SLICE], axis=-1),
            axis=-1,
        )
        i_s: Float[Array, "K B"] = jnp.abs(ws_sum) ** 2
        i_p: Float[Array, "K B"] = jnp.abs(wp_sum) ** 2
        band_intensity: Float[Array, "K B"] = (i_s + i_p) / 2.0
    else:
        efield: Complex[Array, " 3"] = build_efield(pol_config)
        m_elem: Float[Array, " 9"] = dipole_matrix_elements(efield)
        weighted: Float[Array, "K B A 9"] = proj * orb_w * m_elem
        w_sum: Float[Array, "K B"] = jnp.sum(
            jnp.sum(
                weighted[..., NON_S_ORBITAL_SLICE],
                axis=-1,
            ),
            axis=-1,
        )
        band_intensity: Float[Array, "K B"] = jnp.abs(w_sum) ** 2

    def _single_band(
        energy: Float[Array, " "],
        bi: Float[Array, " "],
    ) -> Float[Array, " E"]:
        """Spectral contribution of one band (Gaussian, polarization-weighted).

        Evaluates ``bi * f(E_band) * G(E; E_band, sigma)`` where
        ``bi`` is the polarization-weighted band intensity (already
        including Yeh-Lindau cross-sections and dipole matrix element
        weighting), ``f`` is the Fermi-Dirac distribution, and ``G``
        is the Gaussian profile.

        Parameters
        ----------
        energy : Float[Array, " "]
            Band eigenvalue in eV.
        bi : Float[Array, " "]
            Polarization-weighted band intensity (|e.d|^2 weighted).

        Returns
        -------
        contribution : Float[Array, " E"]
            Weighted Gaussian spectral contribution.
        """
        fd: Float[Array, " "] = fermi_dirac(
            energy, bands.fermi_energy, params.temperature
        )
        profile: Float[Array, " E"] = gaussian(
            energy_axis, energy, params.sigma
        )
        contribution: Float[Array, " E"] = bi * fd * profile
        return contribution

    def _single_kpoint(
        energies: Float[Array, " B"],
        bi_k: Float[Array, " B"],
    ) -> Float[Array, " E"]:
        """Sum polarization-weighted Gaussian contributions over bands.

        Vmaps ``_single_band`` over the band axis B and sums the
        resulting ``(B, E)`` array along the band dimension.

        Parameters
        ----------
        energies : Float[Array, " B"]
            Band eigenvalues at this k-point.
        bi_k : Float[Array, " B"]
            Polarization-weighted band intensities at this k-point.

        Returns
        -------
        total : Float[Array, " E"]
            Total spectral intensity at this k-point.
        """
        contributions: Float[Array, "B E"] = jax.vmap(_single_band)(
            energies, bi_k
        )
        total: Float[Array, " E"] = jnp.sum(contributions, axis=0)
        return total

    intensity: Float[Array, "K E"] = jax.vmap(_single_kpoint)(
        bands.eigenvalues, band_intensity
    )
    spectrum: ArpesSpectrum = make_arpes_spectrum(
        intensity=intensity,
        energy_axis=energy_axis,
    )
    return spectrum


@jaxtyped(typechecker=beartype)
def simulate_expert(
    bands: BandStructure,
    orb_proj: OrbitalProjection,
    params: SimulationParams,
    pol_config: PolarizationConfig,
) -> ArpesSpectrum:
    """Simulate ARPES with Voigt broadening and dipole matrix elements.

    The most physically complete simulation model. Combines Voigt
    broadening (capturing both instrumental Gaussian and lifetime
    Lorentzian contributions via ``sigma`` and ``gamma``), Yeh-Lindau
    photoionization cross-sections, and full polarization-dependent
    dipole matrix element weighting. This level should be used when
    quantitative comparison with experimental spectra is required.

    :see: :class:`~.test_spectrum.TestSimulateExpert`

    Implementation Logic
    --------------------
    1. **Build the energy axis**::

           energy_axis: Float[Array, " E"] = jnp.linspace(
               params.energy_min,
               params.energy_max,
               params.fidelity,
           )

       This defines the common energy grid for every k-point.

    2. **Vectorize the polarized band accumulation**::

           intensity: Float[Array, "K E"] = jax.vmap(_single_kpoint)(
               bands.eigenvalues, band_intensity
           )

       JAX maps the differentiable Voigt model across all k-points.

    3. **Build the spectrum carrier**::

           spectrum: ArpesSpectrum = make_arpes_spectrum(
               intensity=intensity,
               energy_axis=energy_axis,
           )

       The factory validates and stores the computed arrays.

    Parameters
    ----------
    bands : BandStructure
        Electronic band structure containing eigenvalues of shape
        ``(K, B)`` and the Fermi energy.
    orb_proj : OrbitalProjection
        Orbital projections of shape ``(K, B, A, 9)`` where A is the
        number of atoms and 9 is the number of orbital channels
        (s, p_y, p_z, p_x, d_xy, d_yz, d_z2, d_xz, d_x2-y2).
    params : SimulationParams
        Simulation parameters including ``sigma``, ``gamma``,
        ``photon_energy``, ``temperature``, ``energy_min``,
        ``energy_max``, and ``fidelity``.
    pol_config : PolarizationConfig
        Light polarization configuration specifying
        ``polarization_type`` (``"unpolarized"``, ``"linear"``, etc.),
        incidence angles ``theta`` and ``phi``, and any additional
        polarization parameters.

    Returns
    -------
    spectrum : ArpesSpectrum
        Simulated ARPES intensity map with ``intensity`` of shape
        ``(K, E)`` and ``energy_axis`` of shape ``(E,)``.

    Notes
    -----
    Uses Voigt broadening with Yeh-Lindau cross-sections, polarization
    selection rules, and dipole matrix element weighting. This is the
    most physically complete model. For unpolarized light, averages s
    and p contributions. The Voigt profile is essential for accurate
    lineshape fitting where both instrumental and intrinsic broadening
    must be deconvolved.
    """
    energy_axis: Float[Array, " E"] = jnp.linspace(
        params.energy_min,
        params.energy_max,
        params.fidelity,
    )
    orb_w: Float[Array, " 9"] = yeh_lindau_weights(params.photon_energy)
    proj: Float[Array, "K B A 9"] = orb_proj.projections
    is_unpolarized: bool = (
        pol_config.polarization_type.lower() == "unpolarized"
    )
    if is_unpolarized:
        e_s: Float[Array, " 3"]
        e_p: Float[Array, " 3"]
        e_s, e_p = build_polarization_vectors(pol_config.theta, pol_config.phi)
        m_s: Float[Array, " 9"] = dipole_matrix_elements(
            e_s.astype(jnp.complex128)
        )
        m_p: Float[Array, " 9"] = dipole_matrix_elements(
            e_p.astype(jnp.complex128)
        )
        w_s: Float[Array, "K B A 9"] = proj * orb_w * m_s
        w_p: Float[Array, "K B A 9"] = proj * orb_w * m_p
        ws_sum: Float[Array, "K B"] = jnp.sum(
            jnp.sum(w_s[..., NON_S_ORBITAL_SLICE], axis=-1),
            axis=-1,
        )
        wp_sum: Float[Array, "K B"] = jnp.sum(
            jnp.sum(w_p[..., NON_S_ORBITAL_SLICE], axis=-1),
            axis=-1,
        )
        i_s: Float[Array, "K B"] = jnp.abs(ws_sum) ** 2
        i_p: Float[Array, "K B"] = jnp.abs(wp_sum) ** 2
        band_intensity: Float[Array, "K B"] = (i_s + i_p) / 2.0
    else:
        efield: Complex[Array, " 3"] = build_efield(pol_config)
        m_elem: Float[Array, " 9"] = dipole_matrix_elements(efield)
        weighted: Float[Array, "K B A 9"] = proj * orb_w * m_elem
        w_sum: Float[Array, "K B"] = jnp.sum(
            jnp.sum(
                weighted[..., NON_S_ORBITAL_SLICE],
                axis=-1,
            ),
            axis=-1,
        )
        band_intensity: Float[Array, "K B"] = jnp.abs(w_sum) ** 2

    def _single_band(
        energy: Float[Array, " "],
        bi: Float[Array, " "],
    ) -> Float[Array, " E"]:
        """Spectral contribution of one band (Voigt, polarization-weighted).

        Evaluates ``bi * f(E_band) * V(E; E_band, sigma, gamma)``
        where ``bi`` is the polarization-weighted band intensity
        (including Yeh-Lindau cross-sections and dipole matrix element
        weighting), ``f`` is the Fermi-Dirac distribution, and ``V``
        is the Voigt profile combining Gaussian (instrumental) and
        Lorentzian (lifetime) broadening.

        Parameters
        ----------
        energy : Float[Array, " "]
            Band eigenvalue in eV.
        bi : Float[Array, " "]
            Polarization-weighted band intensity (|e.d|^2 weighted).

        Returns
        -------
        contribution : Float[Array, " E"]
            Weighted Voigt spectral contribution.
        """
        fd: Float[Array, " "] = fermi_dirac(
            energy, bands.fermi_energy, params.temperature
        )
        profile: Float[Array, " E"] = voigt(
            energy_axis, energy, params.sigma, params.gamma
        )
        contribution: Float[Array, " E"] = bi * fd * profile
        return contribution

    def _single_kpoint(
        energies: Float[Array, " B"],
        bi_k: Float[Array, " B"],
    ) -> Float[Array, " E"]:
        """Sum polarization-weighted Voigt contributions over bands.

        Vmaps ``_single_band`` over the band axis B and sums the
        resulting ``(B, E)`` array along the band dimension.

        Parameters
        ----------
        energies : Float[Array, " B"]
            Band eigenvalues at this k-point.
        bi_k : Float[Array, " B"]
            Polarization-weighted band intensities at this k-point.

        Returns
        -------
        total : Float[Array, " E"]
            Total spectral intensity at this k-point.
        """
        contributions: Float[Array, "B E"] = jax.vmap(_single_band)(
            energies, bi_k
        )
        total: Float[Array, " E"] = jnp.sum(contributions, axis=0)
        return total

    intensity: Float[Array, "K E"] = jax.vmap(_single_kpoint)(
        bands.eigenvalues, band_intensity
    )
    spectrum: ArpesSpectrum = make_arpes_spectrum(
        intensity=intensity,
        energy_axis=energy_axis,
    )
    return spectrum


@jaxtyped(typechecker=beartype)
def simulate_soc(
    bands: BandStructure,
    orb_proj: SpinOrbitalProjection,
    params: SimulationParams,
    pol_config: PolarizationConfig,
    ls_scale: ScalarFloat = 0.01,
) -> ArpesSpectrum:
    """Simulate ARPES with spin-orbit coupling (spin-dependent intensity).

    Extends the expert model with a spin-orbit correction: the
    orbital-derived band intensity is modulated by the spin projection
    along the photon wavevector, enabling spin-ARPES and circular
    dichroism. Requires ``orb_proj.spin`` of shape ``(K, B, A, 6)``
    (spin up/down for x, y, z). Uses Voigt broadening and the same
    Yeh-Lindau and polarization logic as ``simulate_expert``.

    :see: :class:`~.test_spectrum.TestSimulateSoc`

    Implementation Logic
    --------------------
    1. **Compute the spin-orbit correction**::

           band_intensity_soc: Float[Array, "K B"] = band_intensity * (
               1.0 + ls_arr * spin_dot_k
           )

       This modulates the orbital intensity by the photon-aligned spin.

    2. **Vectorize the corrected band accumulation**::

           intensity: Float[Array, "K E"] = jax.vmap(_single_kpoint)(
               bands.eigenvalues, band_intensity_soc
           )

       JAX maps the differentiable spin-orbit model across all k-points.

    3. **Build the spectrum carrier**::

           spectrum: ArpesSpectrum = make_arpes_spectrum(
               intensity=intensity,
               energy_axis=energy_axis,
           )

       The factory validates and stores the computed arrays.

    Parameters
    ----------
    bands : BandStructure
        Band structure with eigenvalues and Fermi energy.
    orb_proj : SpinOrbitalProjection
        Orbital projections with mandatory spin data
        (shape ``(K, B, A, 6)``).
    params : SimulationParams
        Simulation parameters (sigma, gamma, fidelity, etc.).
    pol_config : PolarizationConfig
        Light polarization and incidence angles.
    ls_scale : ScalarFloat, optional
        Spin-orbit coupling strength for the S·k_photon correction.
        Default is 0.01.

    Returns
    -------
    spectrum : ArpesSpectrum
        Simulated ARPES intensity and energy axis.

    Notes
    -----
    The spin array has six channels (up/down for x, y, z). The net
    spin vector (Sx, Sy, Sz) per (k, band, atom) is formed by
    summing the two components for each axis, then summed over
    atoms to get a per-band spin used in the modulation
    (1 + ls_scale * S·k_photon). With ``ls_scale=0`` the result
    coincides with ``simulate_expert``.

    See Also
    --------
    simulate_expert : Same physics without spin correction.
    photon_wavevector : Builds k_photon from incidence angles.
    """
    energy_axis: Float[Array, " E"] = jnp.linspace(
        params.energy_min,
        params.energy_max,
        params.fidelity,
    )
    orb_w: Float[Array, " 9"] = yeh_lindau_weights(params.photon_energy)
    proj: Float[Array, "K B A 9"] = orb_proj.projections
    spin_raw: Float[Array, "K B A 6"] = orb_proj.spin
    is_unpolarized: bool = (
        pol_config.polarization_type.lower() == "unpolarized"
    )
    if is_unpolarized:
        e_s: Float[Array, " 3"]
        e_p: Float[Array, " 3"]
        e_s, e_p = build_polarization_vectors(pol_config.theta, pol_config.phi)
        m_s: Float[Array, " 9"] = dipole_matrix_elements(
            e_s.astype(jnp.complex128)
        )
        m_p: Float[Array, " 9"] = dipole_matrix_elements(
            e_p.astype(jnp.complex128)
        )
        w_s: Float[Array, "K B A 9"] = proj * orb_w * m_s
        w_p: Float[Array, "K B A 9"] = proj * orb_w * m_p
        ws_sum: Float[Array, "K B"] = jnp.sum(
            jnp.sum(w_s[..., NON_S_ORBITAL_SLICE], axis=-1),
            axis=-1,
        )
        wp_sum: Float[Array, "K B"] = jnp.sum(
            jnp.sum(w_p[..., NON_S_ORBITAL_SLICE], axis=-1),
            axis=-1,
        )
        i_s: Float[Array, "K B"] = jnp.abs(ws_sum) ** 2
        i_p: Float[Array, "K B"] = jnp.abs(wp_sum) ** 2
        band_intensity: Float[Array, "K B"] = (i_s + i_p) / 2.0
    else:
        efield: Complex[Array, " 3"] = build_efield(pol_config)
        m_elem: Float[Array, " 9"] = dipole_matrix_elements(efield)
        weighted: Float[Array, "K B A 9"] = proj * orb_w * m_elem
        w_sum: Float[Array, "K B"] = jnp.sum(
            jnp.sum(
                weighted[..., NON_S_ORBITAL_SLICE],
                axis=-1,
            ),
            axis=-1,
        )
        band_intensity: Float[Array, "K B"] = jnp.abs(w_sum) ** 2

    k_photon: Float[Array, " 3"] = photon_wavevector(
        pol_config.theta, pol_config.phi
    )
    spin_vec: Float[Array, "K B A 3"] = jnp.stack(
        [
            spin_raw[..., 0] + spin_raw[..., 1],
            spin_raw[..., 2] + spin_raw[..., 3],
            spin_raw[..., 4] + spin_raw[..., 5],
        ],
        axis=-1,
    )
    spin_per_band: Float[Array, "K B 3"] = jnp.sum(spin_vec, axis=-2)
    spin_dot_k: Float[Array, "K B"] = jnp.dot(spin_per_band, k_photon)
    ls_arr: Float[Array, " "] = jnp.asarray(ls_scale, dtype=jnp.float64)
    band_intensity_soc: Float[Array, "K B"] = band_intensity * (
        1.0 + ls_arr * spin_dot_k
    )

    def _single_band(
        energy: Float[Array, " "],
        bi: Float[Array, " "],
    ) -> Float[Array, " E"]:
        """Spectral contribution of one band (Voigt, SOC-corrected).

        Evaluates ``bi * f(E_band) * V(E; E_band, sigma, gamma)``
        where ``bi`` is the spin-orbit-corrected band intensity
        (including Yeh-Lindau cross-sections, dipole matrix element
        weighting, and the ``1 + ls_scale * S.k_photon`` modulation),
        ``f`` is the Fermi-Dirac distribution, and ``V`` is the Voigt
        profile.

        Parameters
        ----------
        energy : Float[Array, " "]
            Band eigenvalue in eV.
        bi : Float[Array, " "]
            SOC-corrected band intensity.

        Returns
        -------
        contribution : Float[Array, " E"]
            Weighted Voigt spectral contribution with SOC correction.
        """
        fd: Float[Array, " "] = fermi_dirac(
            energy, bands.fermi_energy, params.temperature
        )
        profile: Float[Array, " E"] = voigt(
            energy_axis, energy, params.sigma, params.gamma
        )
        contribution: Float[Array, " E"] = bi * fd * profile
        return contribution

    def _single_kpoint(
        energies: Float[Array, " B"],
        bi_k: Float[Array, " B"],
    ) -> Float[Array, " E"]:
        """Sum SOC-corrected Voigt contributions over bands at one k-point.

        Vmaps ``_single_band`` over the band axis B and sums the
        resulting ``(B, E)`` array along the band dimension.

        Parameters
        ----------
        energies : Float[Array, " B"]
            Band eigenvalues at this k-point.
        bi_k : Float[Array, " B"]
            SOC-corrected band intensities at this k-point.

        Returns
        -------
        total : Float[Array, " E"]
            Total spectral intensity at this k-point.
        """
        contributions: Float[Array, "B E"] = jax.vmap(_single_band)(
            energies, bi_k
        )
        total: Float[Array, " E"] = jnp.sum(contributions, axis=0)
        return total

    intensity: Float[Array, "K E"] = jax.vmap(_single_kpoint)(
        bands.eigenvalues, band_intensity_soc
    )
    spectrum: ArpesSpectrum = make_arpes_spectrum(
        intensity=intensity,
        energy_axis=energy_axis,
    )
    return spectrum


__all__: list[str] = [
    "simulate_advanced",
    "simulate_basic",
    "simulate_basicplus",
    "simulate_expert",
    "simulate_novice",
    "simulate_soc",
]
