"""Expanded-input workflows for ARPES simulation.

Extended Summary
----------------
Provides convenience wrappers that accept plain arrays and scalars
so that the user does not need to know about the PyTree structures
used internally by diffpes.

Routine Listings
----------------
:func:`simulate_advanced_expanded`
    Run advanced-level ARPES simulation from plain arrays.
:func:`simulate_basic_expanded`
    Run basic-level ARPES simulation from plain arrays.
:func:`simulate_basicplus_expanded`
    Run basicplus-level ARPES simulation from plain arrays.
:func:`simulate_expanded`
    Dispatch an expanded-input simulation by complexity level.
:func:`simulate_expert_expanded`
    Run expert-level ARPES simulation from plain arrays.
:func:`simulate_novice_expanded`
    Run novice-level ARPES simulation from plain arrays.
:func:`simulate_soc_expanded`
    Run SOC (spin-orbit coupling) ARPES simulation from plain arrays.

Notes
-----
Energy axes are built as
``linspace(min(eigenbands)-1, max(eigenbands)+1, fidelity)``.
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple
from jaxtyping import Array, Float, jaxtyped

from diffpes.types import (
    ArpesSpectrum,
    BandStructure,
    OrbitalProjection,
    PolarizationConfig,
    ScalarFloat,
    SimulationParams,
    SpinOrbitalProjection,
    make_band_structure,
    make_expanded_simulation_params,
    make_orbital_projection,
    make_polarization_config,
    make_spin_orbital_projection,
)

from .spectrum import (
    simulate_advanced,
    simulate_basic,
    simulate_basicplus,
    simulate_expert,
    simulate_novice,
    simulate_soc,
)


@jaxtyped(typechecker=beartype)
@jaxtyped(typechecker=beartype)
def _build_inputs(
    eigenbands: Float[Array, "K B"],
    surface_orb: Float[Array, "K B A 9"],
    ef: ScalarFloat,
) -> Tuple[BandStructure, OrbitalProjection]:
    """Convert plain arrays into core ARPES input PyTrees.

    Wraps raw eigenenergy and orbital-projection arrays into the
    :class:`~diffpes.types.BandStructure` and
    :class:`~diffpes.types.OrbitalProjection` PyTrees expected by
    the low-level simulation functions.

    Implementation Logic
    --------------------
    1. **Cast to float64**: ``eigenbands`` and ``surface_orb`` are
       promoted to ``jnp.float64`` via ``jnp.asarray`` so that all
       downstream arithmetic operates at double precision.

    2. **Synthesize k-point coordinates**: A placeholder array of
       zeros with shape ``(K, 3)`` is created. The expanded-input
       workflow does not require physical k-point coordinates because
       the simulation kernels only use the eigenvalues; the k-point
       axis merely indexes the bands.

    3. **Build PyTrees**: ``make_band_structure`` and
       ``make_orbital_projection`` assemble the validated PyTrees.

    Parameters
    ----------
    eigenbands : Float[Array, "K B"]
        Band eigenvalues in eV, with shape ``(n_kpoints, n_bands)``.
    surface_orb : Float[Array, "K B A 9"]
        Orbital projection coefficients, with shape
        ``(n_kpoints, n_bands, n_atoms, 9)``. The last axis follows
        the order ``(s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2)``.
    ef : ScalarFloat
        Fermi energy in eV.

    Returns
    -------
    bands : BandStructure
        Band-structure PyTree containing eigenvalues, placeholder
        k-points, and the Fermi energy.
    orb_proj : OrbitalProjection
        Orbital-projection PyTree wrapping ``surface_orb``.
    """
    bands_arr: Float[Array, "K B"] = jnp.asarray(eigenbands, dtype=jnp.float64)
    proj_arr: Float[Array, "K B A 9"] = jnp.asarray(
        surface_orb, dtype=jnp.float64
    )
    nkpts: int = bands_arr.shape[0]
    kpoints: Float[Array, "K 3"] = jnp.zeros((nkpts, 3), dtype=jnp.float64)
    bands: BandStructure = make_band_structure(
        eigenvalues=bands_arr,
        kpoints=kpoints,
        fermi_energy=ef,
    )
    orb_proj: OrbitalProjection = make_orbital_projection(
        projections=proj_arr,
    )
    simulation_inputs: tuple[BandStructure, OrbitalProjection] = (
        bands,
        orb_proj,
    )
    return simulation_inputs


@jaxtyped(typechecker=beartype)
def _build_polarization(
    polarization: str = "unpolarized",
    incident_theta: ScalarFloat = 45.0,
    incident_phi: ScalarFloat = 0.0,
    polarization_angle: ScalarFloat = 0.0,
) -> PolarizationConfig:
    """Create polarization config with degree-to-radian conversion.

    The expanded API accepts incident angles in **degrees** (more
    intuitive for users) while the internal
    :class:`~diffpes.types.PolarizationConfig` stores them in
    radians.  This helper performs the conversion.

    Implementation Logic
    --------------------
    1. **Convert angles**: ``incident_theta`` and ``incident_phi``
       are converted from degrees to radians via ``jnp.deg2rad``.
    2. **Pass through**: ``polarization_angle`` is already in
       radians and is forwarded without conversion.
    3. **Delegate**: Calls :func:`~diffpes.types.make_polarization_config`
       to construct the validated PyTree.

    Parameters
    ----------
    polarization : str, optional
        Polarization type: ``"s"``, ``"p"``, ``"linear"``, or
        ``"unpolarized"`` (default).
    incident_theta : ScalarFloat, optional
        Polar angle of the incident beam in **degrees**.
        Default is 45.
    incident_phi : ScalarFloat, optional
        Azimuthal angle of the incident beam in **degrees**.
        Default is 0.
    polarization_angle : ScalarFloat, optional
        Rotation angle for arbitrary linear polarization in
        **radians**. Default is 0.

    Returns
    -------
    config : PolarizationConfig
        Polarization configuration with angles in radians.

    See Also
    --------
    make_polarization_config : Lower-level factory accepting
        radians directly.
    """
    theta: Float[Array, " "] = jnp.deg2rad(
        jnp.asarray(incident_theta, dtype=jnp.float64)
    )
    phi: Float[Array, " "] = jnp.deg2rad(
        jnp.asarray(incident_phi, dtype=jnp.float64)
    )
    pol_ang: Float[Array, " "] = jnp.asarray(
        polarization_angle, dtype=jnp.float64
    )
    config: PolarizationConfig = make_polarization_config(
        theta=theta,
        phi=phi,
        polarization_angle=pol_ang,
        polarization_type=polarization,
    )
    return config


@jaxtyped(typechecker=beartype)
def simulate_novice_expanded(
    eigenbands: Float[Array, "K B"],
    surface_orb: Float[Array, "K B A 9"],
    ef: ScalarFloat,
    sigma: ScalarFloat,
    gamma: ScalarFloat,
    fidelity: int,
    temperature: ScalarFloat,
    photon_energy: ScalarFloat,
) -> ArpesSpectrum:
    """Run novice-level ARPES simulation from plain arrays.

    Simplest physical model: applies Voigt broadening (combined
    Gaussian + Lorentzian) with uniform orbital weights. All
    non-s orbitals contribute equally to the photoemission
    intensity.

    :see: :class:`~.test_expanded.TestSimulateNoviceExpanded`

    Implementation Logic
    --------------------
    1. **Build carrier inputs**::

           bands, orb_proj = _build_inputs(
               eigenbands=eigenbands,
               surface_orb=surface_orb,
               ef=ef,
           )

       This validates the raw arrays and binds the band and projection
       carriers.

    2. **Build simulation settings**::

           params = make_expanded_simulation_params(
               eigenbands=eigenbands,
               fidelity=fidelity,
               sigma=sigma,
               gamma=gamma,
               temperature=temperature,
               photon_energy=photon_energy,
           )

       This derives the energy window and preserves both Voigt widths.

    3. **Evaluate the novice model**::

           spectrum = simulate_novice(bands, orb_proj, params)

       This delegates the differentiable numerical work to the carrier API.

    Parameters
    ----------
    eigenbands : Float[Array, "K B"]
        Band eigenvalues in eV, shape ``(n_kpoints, n_bands)``.
    surface_orb : Float[Array, "K B A 9"]
        Orbital projection coefficients, shape
        ``(n_kpoints, n_bands, n_atoms, 9)``.
    ef : ScalarFloat
        Fermi energy in eV.
    sigma : ScalarFloat
        Gaussian broadening width in eV.
    gamma : ScalarFloat
        Lorentzian broadening width in eV.
    fidelity : int
        Number of points in the energy axis.
    temperature : ScalarFloat
        Electronic temperature in Kelvin for the Fermi-Dirac
        distribution.
    photon_energy : ScalarFloat
        Incident photon energy in eV.

    Returns
    -------
    spectrum : ArpesSpectrum
        Simulated ARPES spectrum containing the intensity map and
        energy axis.

    See Also
    --------
    simulate_novice : Low-level implementation accepting PyTrees.
    """
    bands: BandStructure
    orb_proj: OrbitalProjection
    bands, orb_proj = _build_inputs(
        eigenbands=eigenbands,
        surface_orb=surface_orb,
        ef=ef,
    )
    params: SimulationParams = make_expanded_simulation_params(
        eigenbands=eigenbands,
        fidelity=fidelity,
        sigma=sigma,
        gamma=gamma,
        temperature=temperature,
        photon_energy=photon_energy,
    )
    spectrum: ArpesSpectrum = simulate_novice(bands, orb_proj, params)
    return spectrum


@jaxtyped(typechecker=beartype)
def simulate_basic_expanded(
    eigenbands: Float[Array, "K B"],
    surface_orb: Float[Array, "K B A 9"],
    ef: ScalarFloat,
    sigma: ScalarFloat,
    fidelity: int,
    temperature: ScalarFloat,
    photon_energy: ScalarFloat,
) -> ArpesSpectrum:
    """Run basic-level ARPES simulation from plain arrays.

    Applies Gaussian broadening with energy-dependent heuristic
    orbital weights. The heuristic enhances p-orbital contributions
    below 50 eV photon energy and d-orbital contributions above,
    providing a rough approximation of photoionization cross-section
    effects without tabulated data.

    :see: :class:`~.test_expanded.TestSimulateBasicExpanded`

    Implementation Logic
    --------------------
    1. **Build carrier inputs**::

           bands, orb_proj = _build_inputs(
               eigenbands=eigenbands,
               surface_orb=surface_orb,
               ef=ef,
           )

       This validates the raw arrays and binds the band and projection
       carriers.

    2. **Build simulation settings**::

           params = make_expanded_simulation_params(
               eigenbands=eigenbands,
               fidelity=fidelity,
               sigma=sigma,
               temperature=temperature,
               photon_energy=photon_energy,
           )

       This derives the energy window without introducing a Lorentzian width.

    3. **Evaluate the basic model**::

           spectrum = simulate_basic(bands, orb_proj, params)

       This applies the core Gaussian model and its heuristic orbital weights.

    Parameters
    ----------
    eigenbands : Float[Array, "K B"]
        Band eigenvalues in eV, shape ``(n_kpoints, n_bands)``.
    surface_orb : Float[Array, "K B A 9"]
        Orbital projection coefficients, shape
        ``(n_kpoints, n_bands, n_atoms, 9)``.
    ef : ScalarFloat
        Fermi energy in eV.
    sigma : ScalarFloat
        Gaussian broadening width in eV.
    fidelity : int
        Number of points in the energy axis.
    temperature : ScalarFloat
        Electronic temperature in Kelvin for the Fermi-Dirac
        distribution.
    photon_energy : ScalarFloat
        Incident photon energy in eV. Determines the heuristic
        orbital weighting regime.

    Returns
    -------
    spectrum : ArpesSpectrum
        Simulated ARPES spectrum containing the intensity map and
        energy axis.

    See Also
    --------
    simulate_basic : Low-level implementation accepting PyTrees.
    """
    bands: BandStructure
    orb_proj: OrbitalProjection
    bands, orb_proj = _build_inputs(
        eigenbands=eigenbands,
        surface_orb=surface_orb,
        ef=ef,
    )
    params: SimulationParams = make_expanded_simulation_params(
        eigenbands=eigenbands,
        fidelity=fidelity,
        sigma=sigma,
        temperature=temperature,
        photon_energy=photon_energy,
    )
    spectrum: ArpesSpectrum = simulate_basic(bands, orb_proj, params)
    return spectrum


@jaxtyped(typechecker=beartype)
def simulate_basicplus_expanded(
    eigenbands: Float[Array, "K B"],
    surface_orb: Float[Array, "K B A 9"],
    ef: ScalarFloat,
    sigma: ScalarFloat,
    fidelity: int,
    temperature: ScalarFloat,
    photon_energy: ScalarFloat,
) -> ArpesSpectrum:
    """Run basicplus-level ARPES simulation from plain arrays.

    Applies Gaussian broadening with interpolated Yeh-Lindau
    photoionization cross-sections. Unlike the heuristic weights
    used at the basic level, Yeh-Lindau cross-sections are derived
    from tabulated atomic data and provide physically accurate
    orbital-dependent intensity scaling at each photon energy.

    :see: :class:`~.test_expanded.TestSimulateBasicplusExpanded`

    Implementation Logic
    --------------------
    1. **Build carrier inputs**::

           bands, orb_proj = _build_inputs(
               eigenbands=eigenbands,
               surface_orb=surface_orb,
               ef=ef,
           )

       This validates the raw arrays and binds the band and projection
       carriers.

    2. **Build simulation settings**::

           params = make_expanded_simulation_params(
               eigenbands=eigenbands,
               fidelity=fidelity,
               sigma=sigma,
               temperature=temperature,
               photon_energy=photon_energy,
           )

       This retains the photon energy used for Yeh-Lindau interpolation.

    3. **Evaluate the basicplus model**::

           spectrum = simulate_basicplus(bands, orb_proj, params)

       This delegates cross-section weighting and Gaussian broadening to the
       core.

    Parameters
    ----------
    eigenbands : Float[Array, "K B"]
        Band eigenvalues in eV, shape ``(n_kpoints, n_bands)``.
    surface_orb : Float[Array, "K B A 9"]
        Orbital projection coefficients, shape
        ``(n_kpoints, n_bands, n_atoms, 9)``.
    ef : ScalarFloat
        Fermi energy in eV.
    sigma : ScalarFloat
        Gaussian broadening width in eV.
    fidelity : int
        Number of points in the energy axis.
    temperature : ScalarFloat
        Electronic temperature in Kelvin for the Fermi-Dirac
        distribution.
    photon_energy : ScalarFloat
        Incident photon energy in eV. Used to interpolate
        Yeh-Lindau cross-section tables.

    Returns
    -------
    spectrum : ArpesSpectrum
        Simulated ARPES spectrum containing the intensity map and
        energy axis.

    See Also
    --------
    simulate_basicplus : Low-level implementation accepting PyTrees.
    """
    bands: BandStructure
    orb_proj: OrbitalProjection
    bands, orb_proj = _build_inputs(
        eigenbands=eigenbands,
        surface_orb=surface_orb,
        ef=ef,
    )
    params: SimulationParams = make_expanded_simulation_params(
        eigenbands=eigenbands,
        fidelity=fidelity,
        sigma=sigma,
        temperature=temperature,
        photon_energy=photon_energy,
    )
    spectrum: ArpesSpectrum = simulate_basicplus(bands, orb_proj, params)
    return spectrum


@jaxtyped(typechecker=beartype)
def simulate_advanced_expanded(  # noqa: PLR0913
    eigenbands: Float[Array, "K B"],
    surface_orb: Float[Array, "K B A 9"],
    ef: ScalarFloat,
    sigma: ScalarFloat,
    fidelity: int,
    temperature: ScalarFloat,
    photon_energy: ScalarFloat,
    polarization: str = "unpolarized",
    incident_theta: ScalarFloat = 45.0,
    incident_phi: ScalarFloat = 0.0,
    polarization_angle: ScalarFloat = 0.0,
) -> ArpesSpectrum:
    """Run advanced-level ARPES simulation from plain arrays.

    Builds on the basicplus level by adding polarization-dependent
    selection rules. The photoemission intensity is weighted by
    ``|e . d|^2``, where e is the light electric-field vector and d is
    the dipole matrix element for each orbital channel. For
    unpolarized light the s- and p-polarization contributions are
    averaged.

    :see: :class:`~.test_expanded.TestSimulateAdvancedExpanded`

    Implementation Logic
    --------------------
    1. **Build carrier inputs**::

           bands, orb_proj = _build_inputs(
               eigenbands=eigenbands,
               surface_orb=surface_orb,
               ef=ef,
           )

       This validates the raw arrays and binds the band and projection
       carriers.

    2. **Build simulation settings**::

           params = make_expanded_simulation_params(
               eigenbands=eigenbands,
               fidelity=fidelity,
               sigma=sigma,
               temperature=temperature,
               photon_energy=photon_energy,
           )

       This derives the Gaussian energy grid and traced physical settings.

    3. **Build polarization settings**::

           pol = _build_polarization(
               polarization=polarization,
               incident_theta=incident_theta,
               incident_phi=incident_phi,
               polarization_angle=polarization_angle,
           )

       This converts degree-valued incidence angles through the shared factory.

    4. **Evaluate the advanced model**::

           spectrum = simulate_advanced(bands, orb_proj, params, pol)

       This delegates polarization-weighted intensity to the core model.

    Parameters
    ----------
    eigenbands : Float[Array, "K B"]
        Band eigenvalues in eV, shape ``(n_kpoints, n_bands)``.
    surface_orb : Float[Array, "K B A 9"]
        Orbital projection coefficients, shape
        ``(n_kpoints, n_bands, n_atoms, 9)``.
    ef : ScalarFloat
        Fermi energy in eV.
    sigma : ScalarFloat
        Gaussian broadening width in eV.
    fidelity : int
        Number of points in the energy axis.
    temperature : ScalarFloat
        Electronic temperature in Kelvin for the Fermi-Dirac
        distribution.
    photon_energy : ScalarFloat
        Incident photon energy in eV.
    polarization : str, optional
        Polarization type: ``"s"``, ``"p"``, ``"linear"``, or
        ``"unpolarized"`` (default).
    incident_theta : ScalarFloat, optional
        Polar angle of the incident beam in degrees. Default 45.
    incident_phi : ScalarFloat, optional
        Azimuthal angle of the incident beam in degrees. Default 0.
    polarization_angle : ScalarFloat, optional
        Rotation angle for arbitrary linear polarization in
        radians. Default 0.

    Returns
    -------
    spectrum : ArpesSpectrum
        Simulated ARPES spectrum containing the intensity map and
        energy axis.

    See Also
    --------
    simulate_advanced : Low-level implementation accepting PyTrees.
    """
    bands: BandStructure
    orb_proj: OrbitalProjection
    bands, orb_proj = _build_inputs(
        eigenbands=eigenbands,
        surface_orb=surface_orb,
        ef=ef,
    )
    params: SimulationParams = make_expanded_simulation_params(
        eigenbands=eigenbands,
        fidelity=fidelity,
        sigma=sigma,
        temperature=temperature,
        photon_energy=photon_energy,
    )
    pol: PolarizationConfig = _build_polarization(
        polarization=polarization,
        incident_theta=incident_theta,
        incident_phi=incident_phi,
        polarization_angle=polarization_angle,
    )
    spectrum: ArpesSpectrum = simulate_advanced(bands, orb_proj, params, pol)
    return spectrum


@jaxtyped(typechecker=beartype)
def simulate_expert_expanded(  # noqa: PLR0913
    eigenbands: Float[Array, "K B"],
    surface_orb: Float[Array, "K B A 9"],
    ef: ScalarFloat,
    sigma: ScalarFloat,
    gamma: ScalarFloat,
    fidelity: int,
    temperature: ScalarFloat,
    photon_energy: ScalarFloat,
    polarization: str = "unpolarized",
    incident_theta: ScalarFloat = 45.0,
    incident_phi: ScalarFloat = 0.0,
    polarization_angle: ScalarFloat = 0.0,
) -> ArpesSpectrum:
    """Run expert-level ARPES simulation from plain arrays.

    Most physically complete model. Combines Voigt broadening
    (Gaussian + Lorentzian), Yeh-Lindau photoionization
    cross-sections, polarization selection rules, and full dipole
    matrix element weighting. For unpolarized light the s- and
    p-polarization contributions are averaged.

    :see: :class:`~.test_expanded.TestSimulateExpertExpanded`

    Implementation Logic
    --------------------
    1. **Build carrier inputs**::

           bands, orb_proj = _build_inputs(
               eigenbands=eigenbands,
               surface_orb=surface_orb,
               ef=ef,
           )

       This validates the raw arrays and binds the band and projection
       carriers.

    2. **Build simulation settings**::

           params = make_expanded_simulation_params(
               eigenbands=eigenbands,
               fidelity=fidelity,
               sigma=sigma,
               gamma=gamma,
               temperature=temperature,
               photon_energy=photon_energy,
           )

       This derives the energy window and preserves both Voigt widths.

    3. **Build polarization settings**::

           pol = _build_polarization(
               polarization=polarization,
               incident_theta=incident_theta,
               incident_phi=incident_phi,
               polarization_angle=polarization_angle,
           )

       This converts degree-valued incidence angles through the shared factory.

    4. **Evaluate the expert model**::

           spectrum = simulate_expert(bands, orb_proj, params, pol)

       This delegates dipole weighting and Voigt broadening to the core model.

    Parameters
    ----------
    eigenbands : Float[Array, "K B"]
        Band eigenvalues in eV, shape ``(n_kpoints, n_bands)``.
    surface_orb : Float[Array, "K B A 9"]
        Orbital projection coefficients, shape
        ``(n_kpoints, n_bands, n_atoms, 9)``.
    ef : ScalarFloat
        Fermi energy in eV.
    sigma : ScalarFloat
        Gaussian broadening width in eV.
    gamma : ScalarFloat
        Lorentzian broadening width in eV.
    fidelity : int
        Number of points in the energy axis.
    temperature : ScalarFloat
        Electronic temperature in Kelvin for the Fermi-Dirac
        distribution.
    photon_energy : ScalarFloat
        Incident photon energy in eV.
    polarization : str, optional
        Polarization type: ``"s"``, ``"p"``, ``"linear"``, or
        ``"unpolarized"`` (default).
    incident_theta : ScalarFloat, optional
        Polar angle of the incident beam in degrees. Default 45.
    incident_phi : ScalarFloat, optional
        Azimuthal angle of the incident beam in degrees. Default 0.
    polarization_angle : ScalarFloat, optional
        Rotation angle for arbitrary linear polarization in
        radians. Default 0.

    Returns
    -------
    spectrum : ArpesSpectrum
        Simulated ARPES spectrum containing the intensity map and
        energy axis.

    See Also
    --------
    simulate_expert : Low-level implementation accepting PyTrees.
    """
    bands: BandStructure
    orb_proj: OrbitalProjection
    bands, orb_proj = _build_inputs(
        eigenbands=eigenbands,
        surface_orb=surface_orb,
        ef=ef,
    )
    params: SimulationParams = make_expanded_simulation_params(
        eigenbands=eigenbands,
        fidelity=fidelity,
        sigma=sigma,
        gamma=gamma,
        temperature=temperature,
        photon_energy=photon_energy,
    )
    pol: PolarizationConfig = _build_polarization(
        polarization=polarization,
        incident_theta=incident_theta,
        incident_phi=incident_phi,
        polarization_angle=polarization_angle,
    )
    spectrum: ArpesSpectrum = simulate_expert(bands, orb_proj, params, pol)
    return spectrum


@jaxtyped(typechecker=beartype)
def simulate_soc_expanded(  # noqa: PLR0913
    eigenbands: Float[Array, "K B"],
    surface_orb: Float[Array, "K B A 9"],
    surface_spin: Float[Array, "K B A 6"],
    ef: ScalarFloat,
    sigma: ScalarFloat,
    gamma: ScalarFloat,
    fidelity: int,
    temperature: ScalarFloat,
    photon_energy: ScalarFloat,
    polarization: str = "unpolarized",
    incident_theta: ScalarFloat = 45.0,
    incident_phi: ScalarFloat = 0.0,
    polarization_angle: ScalarFloat = 0.0,
    ls_scale: ScalarFloat = 0.01,
) -> ArpesSpectrum:
    """Run SOC (spin-orbit coupling) ARPES simulation from plain arrays.

    Expert model plus spin-dependent intensity correction
    (S·k_photon). Requires spin projections of shape
    ``(n_kpoints, n_bands, n_atoms, 6)`` (up/down for x, y, z).
    Incident angles are interpreted in degrees.

    :see: :class:`~.test_expanded.TestSimulateSocExpanded`

    Implementation Logic
    --------------------
    1. **Build the band carrier**::

           bands, _projection = _build_inputs(
               eigenbands=eigenbands,
               surface_orb=surface_orb,
               ef=ef,
           )

       This validates the common raw arrays and preserves the Fermi-energy
       shift.

    2. **Attach spin-resolved projections**::

           soc_proj = make_spin_orbital_projection(
               projections=jnp.asarray(surface_orb, dtype=jnp.float64),
               spin=surface_spin,
           )

       This keeps the six spin channels explicit for the SOC correction.

    3. **Build simulation and polarization settings**::

           params = make_expanded_simulation_params(
               eigenbands=eigenbands,
               fidelity=fidelity,
               sigma=sigma,
               gamma=gamma,
               temperature=temperature,
               photon_energy=photon_energy,
           )
           pol = _build_polarization(
               polarization=polarization,
               incident_theta=incident_theta,
               incident_phi=incident_phi,
               polarization_angle=polarization_angle,
           )

       These carriers retain the traced physical parameters and the static
       mode choice.

    4. **Evaluate the SOC model**::

           spectrum = simulate_soc(
               bands, soc_proj, params, pol, ls_scale=ls_scale
           )

       This applies the spin-dependent correction only after carrier
       validation.

    Parameters
    ----------
    eigenbands : Float[Array, "K B"]
        Band eigenvalues in eV, shape ``(n_kpoints, n_bands)``.
    surface_orb : Float[Array, "K B A 9"]
        Orbital projection coefficients, shape
        ``(n_kpoints, n_bands, n_atoms, 9)``.
    surface_spin : Float[Array, "K B A 6"]
        Spin projections (required for SOC), shape
        ``(n_kpoints, n_bands, n_atoms, 6)`` with channels
        (x up, x down, y up, y down, z up, z down).
    ef : ScalarFloat
        Fermi energy in eV.
    sigma : ScalarFloat
        Gaussian broadening width in eV.
    gamma : ScalarFloat
        Lorentzian broadening width in eV.
    fidelity : int
        Number of points on the energy axis.
    temperature : ScalarFloat
        Electronic temperature in Kelvin.
    photon_energy : ScalarFloat
        Incident photon energy in eV.
    polarization : str, optional
        Polarization type (e.g. ``"unpolarized"``, ``"LHP"``).
        Default is ``"unpolarized"``.
    incident_theta : ScalarFloat, optional
        Polar angle of the incident beam in degrees. Default 45.
    incident_phi : ScalarFloat, optional
        Azimuthal angle of the incident beam in degrees. Default 0.
    polarization_angle : ScalarFloat, optional
        Rotation angle for arbitrary linear polarization in
        radians. Default 0.
    ls_scale : ScalarFloat, optional
        Spin-orbit coupling strength for the S·k_photon
        correction. Default 0.01.

    Returns
    -------
    spectrum : ArpesSpectrum
        Simulated ARPES spectrum with spin-orbit correction;
        ``intensity`` shape ``(K, E)``, ``energy_axis`` shape ``(E,)``.

    See Also
    --------
    simulate_soc : Low-level implementation accepting PyTrees.
    simulate_expert_expanded : Same pipeline without spin data.
    """
    bands: BandStructure
    _projection: OrbitalProjection
    bands, _projection = _build_inputs(
        eigenbands=eigenbands,
        surface_orb=surface_orb,
        ef=ef,
    )
    soc_proj: SpinOrbitalProjection = make_spin_orbital_projection(
        projections=jnp.asarray(surface_orb, dtype=jnp.float64),
        spin=surface_spin,
    )
    params: SimulationParams = make_expanded_simulation_params(
        eigenbands=eigenbands,
        fidelity=fidelity,
        sigma=sigma,
        gamma=gamma,
        temperature=temperature,
        photon_energy=photon_energy,
    )
    pol: PolarizationConfig = _build_polarization(
        polarization=polarization,
        incident_theta=incident_theta,
        incident_phi=incident_phi,
        polarization_angle=polarization_angle,
    )
    spectrum: ArpesSpectrum = simulate_soc(
        bands, soc_proj, params, pol, ls_scale=ls_scale
    )
    return spectrum


@jaxtyped(typechecker=beartype)
def simulate_expanded(  # noqa: PLR0913
    level: str,
    eigenbands: Float[Array, "K B"],
    surface_orb: Float[Array, "K B A 9"],
    ef: ScalarFloat = 0.0,
    sigma: ScalarFloat = 0.04,
    gamma: ScalarFloat = 0.1,
    fidelity: int = 25000,
    temperature: ScalarFloat = 15.0,
    photon_energy: ScalarFloat = 11.0,
    polarization: str = "unpolarized",
    incident_theta: ScalarFloat = 45.0,
    incident_phi: ScalarFloat = 0.0,
    polarization_angle: ScalarFloat = 0.0,
    surface_spin: Optional[Float[Array, "K B A 6"]] = None,
    ls_scale: ScalarFloat = 0.01,
) -> ArpesSpectrum:
    """Dispatch an expanded-input simulation by complexity level.

    Single entry-point that routes to one of the six simulation
    functions based on ``level``. All parameters have sensible
    defaults, so only ``level``, ``eigenbands``, and ``surface_orb``
    are required. Parameters that are unused by the selected level
    (e.g. ``gamma`` for basic/basicplus/advanced, or polarization
    settings for novice/basic/basicplus) are silently ignored.

    :see: :class:`~.test_expanded.TestSimulateExpanded`

    Implementation Logic
    --------------------
    1. **Normalize the level key**::

           level_key = level.lower()

       This makes the static dispatcher case-insensitive.

    2. **Select the matching wrapper**::

           if level_key == "novice":
               spectrum = simulate_novice_expanded(
                   eigenbands=eigenbands,
                   surface_orb=surface_orb,
                   ef=ef,
                   sigma=sigma,
                   gamma=gamma,
                   fidelity=fidelity,
                   temperature=temperature,
                   photon_energy=photon_energy,
               )

       The remaining ``elif`` branches use the same explicit forwarding
       pattern, so each level receives only its public parameters.

    3. **Reject unsupported levels**::

           msg: str = (
               "Unknown simulation level. "
               "Expected one of: novice, basic, basicplus, advanced, "
               "expert, soc."
           )
           raise ValueError(msg)

       This fails before returning a partially defined spectrum.

    4. **Return the selected spectrum**::

           return spectrum

       This preserves one annotated return variable across all valid branches.

    Parameters
    ----------
    level : str
        One of ``"novice"``, ``"basic"``, ``"basicplus"``,
        ``"advanced"``, ``"expert"``, or ``"soc"`` (case-insensitive).
    eigenbands : Float[Array, "K B"]
        Band eigenvalues in eV, shape ``(n_kpoints, n_bands)``.
    surface_orb : Float[Array, "K B A 9"]
        Orbital projection coefficients, shape
        ``(n_kpoints, n_bands, n_atoms, 9)``.
    ef : ScalarFloat, optional
        Fermi energy in eV. Default is 0.
    sigma : ScalarFloat, optional
        Gaussian broadening width in eV. Default is 0.04.
    gamma : ScalarFloat, optional
        Lorentzian broadening width in eV. Only used by novice
        and expert levels. Default is 0.1.
    fidelity : int, optional
        Number of points in the energy axis. Default is 25000.
    temperature : ScalarFloat, optional
        Electronic temperature in Kelvin. Default is 15.
    photon_energy : ScalarFloat, optional
        Incident photon energy in eV. Default is 11.
    polarization : str, optional
        Polarization type: ``"s"``, ``"p"``, ``"linear"``, or
        ``"unpolarized"``. Only used by advanced and expert.
    incident_theta : ScalarFloat, optional
        Polar angle of the incident beam in degrees. Only used
        by advanced and expert. Default is 45.
    incident_phi : ScalarFloat, optional
        Azimuthal angle of the incident beam in degrees. Only
        used by advanced and expert. Default is 0.
    polarization_angle : ScalarFloat, optional
        Rotation angle for arbitrary linear polarization in
        radians. Only used by advanced and expert. Default is 0.
    surface_spin : Optional[Float[Array, "K B A 6"]], optional
        Spin projections; required when ``level='soc'``. Default None.
    ls_scale : ScalarFloat, optional
        Spin-orbit coupling strength when ``level='soc'``.
        Default 0.01.

    Returns
    -------
    spectrum : ArpesSpectrum
        Simulated ARPES spectrum containing the intensity map and
        energy axis.

    Raises
    ------
    ValueError
        If ``level`` is not one of the six recognized levels, or
        if ``level='soc'`` and ``surface_spin`` is None.

    See Also
    --------
    simulate_novice_expanded : Voigt broadening, uniform weights.
    simulate_basic_expanded : Gaussian, heuristic weights.
    simulate_basicplus_expanded : Gaussian, Yeh-Lindau weights.
    simulate_advanced_expanded : Gaussian, Yeh-Lindau, polarization.
    simulate_expert_expanded : Voigt, Yeh-Lindau, polarization,
        dipole matrix elements.
    simulate_soc_expanded : Expert plus spin-orbit (S·k_photon) correction.
    """
    level_key: str = level.lower()
    if level_key == "novice":
        spectrum: ArpesSpectrum = simulate_novice_expanded(
            eigenbands=eigenbands,
            surface_orb=surface_orb,
            ef=ef,
            sigma=sigma,
            gamma=gamma,
            fidelity=fidelity,
            temperature=temperature,
            photon_energy=photon_energy,
        )
    elif level_key == "basic":
        spectrum = simulate_basic_expanded(
            eigenbands=eigenbands,
            surface_orb=surface_orb,
            ef=ef,
            sigma=sigma,
            fidelity=fidelity,
            temperature=temperature,
            photon_energy=photon_energy,
        )
    elif level_key == "basicplus":
        spectrum = simulate_basicplus_expanded(
            eigenbands=eigenbands,
            surface_orb=surface_orb,
            ef=ef,
            sigma=sigma,
            fidelity=fidelity,
            temperature=temperature,
            photon_energy=photon_energy,
        )
    elif level_key == "advanced":
        spectrum = simulate_advanced_expanded(
            eigenbands=eigenbands,
            surface_orb=surface_orb,
            ef=ef,
            sigma=sigma,
            fidelity=fidelity,
            temperature=temperature,
            photon_energy=photon_energy,
            polarization=polarization,
            incident_theta=incident_theta,
            incident_phi=incident_phi,
            polarization_angle=polarization_angle,
        )
    elif level_key == "expert":
        spectrum = simulate_expert_expanded(
            eigenbands=eigenbands,
            surface_orb=surface_orb,
            ef=ef,
            sigma=sigma,
            gamma=gamma,
            fidelity=fidelity,
            temperature=temperature,
            photon_energy=photon_energy,
            polarization=polarization,
            incident_theta=incident_theta,
            incident_phi=incident_phi,
            polarization_angle=polarization_angle,
        )
    elif level_key == "soc":
        if surface_spin is None:
            raise ValueError(
                "simulate_expanded(level='soc', ...) requires surface_spin."
            )
        spectrum = simulate_soc_expanded(
            eigenbands=eigenbands,
            surface_orb=surface_orb,
            surface_spin=surface_spin,
            ef=ef,
            sigma=sigma,
            gamma=gamma,
            fidelity=fidelity,
            temperature=temperature,
            photon_energy=photon_energy,
            polarization=polarization,
            incident_theta=incident_theta,
            incident_phi=incident_phi,
            polarization_angle=polarization_angle,
            ls_scale=ls_scale,
        )
    else:
        msg: str = (
            "Unknown simulation level. "
            "Expected one of: novice, basic, basicplus, advanced, expert, soc."
        )
        raise ValueError(msg)
    return spectrum


__all__: list[str] = [
    "simulate_advanced_expanded",
    "simulate_basic_expanded",
    "simulate_basicplus_expanded",
    "simulate_expanded",
    "simulate_expert_expanded",
    "simulate_novice_expanded",
    "simulate_soc_expanded",
]
