"""Simulation parameter data structures.

Extended Summary
----------------
Defines PyTree types for ARPES simulation parameters including
energy resolution, broadening widths, temperature, photon energy,
and light polarization configuration.

Routine Listings
----------------
:class:`PolarizationConfig`
    PyTree for photon polarization geometry.
:class:`SimulationParams`
    PyTree for ARPES simulation parameters.
:func:`make_polarization_config`
    Create a validated PolarizationConfig instance.
:func:`make_expanded_simulation_params`
    Build simulation parameters with auto-derived energy window.
:func:`make_simulation_params`
    Create a validated SimulationParams instance.

Notes
-----
Polarization types follow standard optics conventions:
``"LVP"`` (s-pol), ``"LHP"`` (p-pol), ``"RCP"``, ``"LCP"``,
``"LAP"`` (linear arbitrary), ``"unpolarized"``.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from .aliases import ScalarFloat, ScalarNumeric


class SimulationParams(eqx.Module):
    """PyTree for ARPES simulation parameters.

    Collects all scalar physical parameters that control an ARPES
    simulation: the energy window and its discretisation, Gaussian
    and Lorentzian broadening widths, sample temperature, and photon
    energy. These parameters are consumed by the spectral-function
    convolution and Fermi-Dirac weighting routines.

    This class is an immutable :class:`equinox.Module` PyTree. All
    float-valued fields are
    stored as JAX array children (visible to autodiff), while the
    ``fidelity`` field is a plain Python ``int`` stored as auxiliary
    data because it controls array shapes and must be known at
    compile time.

    Attributes
    ----------
    energy_min : Float[Array, " "]
        Lower bound of energy window in eV.
    energy_max : Float[Array, " "]
        Upper bound of energy window in eV.
    sigma : Float[Array, " "]
        Gaussian instrumental broadening width in eV.
    gamma : Float[Array, " "]
        Lorentzian lifetime broadening half-width in eV.
    temperature : Float[Array, " "]
        Sample temperature in Kelvin.
    photon_energy : Float[Array, " "]
        Incident photon energy in eV.
    fidelity : int
        Number of points along the energy axis.

    Notes
    -----
    Equinox derives the PyTree structure from the annotated fields.
    The ``fidelity`` field is a Python ``int`` (not a JAX array) and
    is therefore stored as auxiliary data rather than as a child. JAX
    treats auxiliary data as a compile-time constant: changing
    ``fidelity`` triggers recompilation of any ``jit``-compiled
    function that receives this PyTree. This is intentional because
    ``fidelity`` determines the length of the energy axis array and
    JAX requires static shapes.

    See Also
    --------
    make_simulation_params : Factory function with validation and
        float64 casting.
    """

    energy_min: Float[Array, " "]
    energy_max: Float[Array, " "]
    sigma: Float[Array, " "]
    gamma: Float[Array, " "]
    temperature: Float[Array, " "]
    photon_energy: Float[Array, " "]
    fidelity: int = eqx.field(static=True)


class PolarizationConfig(eqx.Module):
    """PyTree for photon polarization geometry.

    Describes the photon polarization state and incidence geometry
    for an ARPES experiment. The three angular fields define the
    light direction and polarization orientation, while the string
    ``polarization_type`` selects among standard optics conventions
    (s-pol, p-pol, circular, arbitrary linear, or unpolarized).

    This class is an immutable :class:`equinox.Module` PyTree. The three
    float-valued angular
    fields are stored as JAX array children (visible to autodiff),
    while ``polarization_type`` is a Python string stored as
    auxiliary data because JAX cannot trace strings.

    Attributes
    ----------
    theta : Float[Array, " "]
        Incident angle from surface normal in radians.
    phi : Float[Array, " "]
        In-plane azimuthal angle in radians.
    polarization_angle : Float[Array, " "]
        Arbitrary linear polarization angle in radians.
    polarization_type : str
        One of LVP, LHP, RCP, LCP, LAP, unpolarized.

    Notes
    -----
    Equinox derives the PyTree structure from the annotated fields.
    The ``polarization_type`` field is a Python string and is
    therefore stored as auxiliary data rather than as a child. JAX
    treats auxiliary data as a compile-time constant: changing
    ``polarization_type`` triggers recompilation of any
    ``jit``-compiled function that receives this PyTree. This is
    intentional because the polarization type selects different
    code branches in the matrix-element calculation.

    See Also
    --------
    make_polarization_config : Factory function with validation and
        float64 casting.
    """

    theta: Float[Array, " "]
    phi: Float[Array, " "]
    polarization_angle: Float[Array, " "]
    polarization_type: str = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def make_simulation_params(
    energy_min: ScalarNumeric = -3.0,
    energy_max: ScalarNumeric = 1.0,
    fidelity: int = 25000,
    sigma: ScalarFloat = 0.04,
    gamma: ScalarFloat = 0.1,
    temperature: ScalarFloat = 15.0,
    photon_energy: ScalarFloat = 11.0,
) -> SimulationParams:
    """Create a validated SimulationParams instance.

    Validates and normalises the inputs before constructing the
    SimulationParams PyTree. All float-valued parameters are cast to
    0-D float64 JAX arrays, while ``fidelity`` is kept as a plain
    Python int (it becomes auxiliary data in the PyTree). Sensible
    defaults are provided for a typical low-temperature ARPES
    experiment: the energy window defaults to [-3.0, 1.0] eV, which
    is wide enough for most valence-band spectra near the Fermi
    level. In higher-level simulation drivers the ``energy_min`` and
    ``energy_max`` defaults may be overridden based on the actual
    eigenband energy range supplied via an ``eigenbands`` parameter.

    Implementation Logic
    --------------------
    1. **Cast energy_min** to 0-D ``jnp.float64`` via
       ``jnp.asarray``.
    2. **Cast energy_max** to 0-D ``jnp.float64`` via
       ``jnp.asarray``.
    3. **Keep fidelity** as a Python ``int`` (not cast). It is
       stored as auxiliary data in the PyTree because it determines
       array shapes.
    4. **Cast sigma, gamma, temperature, photon_energy** each to
       0-D ``jnp.float64`` arrays via ``jnp.asarray``.
    5. **Construct** the ``SimulationParams`` Equinox module from all
       seven fields and return it.

    Parameters
    ----------
    energy_min : ScalarNumeric, optional
        Lower energy bound in eV. Default is -3.0.
    energy_max : ScalarNumeric, optional
        Upper energy bound in eV. Default is 1.0.
    fidelity : int, optional
        Energy axis resolution. Default is 25000.
    sigma : ScalarFloat, optional
        Gaussian broadening in eV. Default is 0.04.
    gamma : ScalarFloat, optional
        Lorentzian broadening in eV. Default is 0.1.
    temperature : ScalarFloat, optional
        Temperature in Kelvin. Default is 15.0.
    photon_energy : ScalarFloat, optional
        Photon energy in eV. Default is 11.0.

    Returns
    -------
    params : SimulationParams
        Validated simulation parameters.

    See Also
    --------
    SimulationParams : The PyTree class constructed by this factory.
    """
    emin_arr: Float[Array, " "] = jnp.asarray(energy_min, dtype=jnp.float64)
    emax_arr: Float[Array, " "] = jnp.asarray(energy_max, dtype=jnp.float64)
    sigma_arr: Float[Array, " "] = jnp.asarray(sigma, dtype=jnp.float64)
    gamma_arr: Float[Array, " "] = jnp.asarray(gamma, dtype=jnp.float64)
    temp_arr: Float[Array, " "] = jnp.asarray(temperature, dtype=jnp.float64)
    pe_arr: Float[Array, " "] = jnp.asarray(photon_energy, dtype=jnp.float64)

    def validate_and_create() -> SimulationParams:
        nonlocal emin_arr, gamma_arr, pe_arr, sigma_arr, temp_arr
        emin_arr = eqx.error_if(
            emin_arr,
            ~(emin_arr < emax_arr),
            "make_simulation_params: energy window",
        )
        sigma_arr = eqx.error_if(
            sigma_arr,
            ~(sigma_arr > 0.0),
            "make_simulation_params: sigma positive",
        )
        gamma_arr = eqx.error_if(
            gamma_arr,
            ~(gamma_arr > 0.0),
            "make_simulation_params: gamma positive",
        )
        temp_arr = eqx.error_if(
            temp_arr,
            ~(temp_arr > 0.0),
            "make_simulation_params: temperature positive",
        )
        pe_arr = eqx.error_if(
            pe_arr,
            ~(pe_arr > 0.0),
            "make_simulation_params: photon energy positive",
        )
        return SimulationParams(
            energy_min=emin_arr,
            energy_max=emax_arr,
            fidelity=fidelity,
            sigma=sigma_arr,
            gamma=gamma_arr,
            temperature=temp_arr,
            photon_energy=pe_arr,
        )

    params: SimulationParams = validate_and_create()
    return params


@jaxtyped(typechecker=beartype)
def make_polarization_config(
    theta: ScalarFloat = 0.7854,
    phi: ScalarFloat = 0.0,
    polarization_angle: ScalarFloat = 0.0,
    polarization_type: str = "unpolarized",
) -> PolarizationConfig:
    """Create a validated PolarizationConfig instance.

    Validates and normalises the inputs before constructing the
    PolarizationConfig PyTree. All float-valued angular parameters
    are cast to 0-D float64 JAX arrays, while
    ``polarization_type`` is kept as a plain Python string (it
    becomes auxiliary data in the PyTree).

    The default incidence angle ``theta = 0.7854`` rad corresponds
    to pi/4 (approximately 45 degrees), a common experimental
    geometry that provides balanced sensitivity to both in-plane
    and out-of-plane orbital components.

    Implementation Logic
    --------------------
    1. **Cast theta** to 0-D ``jnp.float64`` via ``jnp.asarray``.
       Default is ``0.7854`` rad = pi/4 ~ 45 degrees.
    2. **Cast phi** to 0-D ``jnp.float64`` via ``jnp.asarray``.
       Default is ``0.0`` rad.
    3. **Cast polarization_angle** to 0-D ``jnp.float64`` via
       ``jnp.asarray``. Default is ``0.0`` rad.
    4. **Keep polarization_type** as a Python string (not cast).
       It is stored as auxiliary data in the PyTree because it
       selects code branches at compile time.
    5. **Construct** the ``PolarizationConfig`` Equinox module from all
       four fields and return it.

    Parameters
    ----------
    theta : ScalarFloat, optional
        Incident angle in radians. Default is pi/4 ~ 0.7854 rad
        (45 degrees).
    phi : ScalarFloat, optional
        Azimuthal angle in radians. Default is 0.
    polarization_angle : ScalarFloat, optional
        Linear polarization angle in radians. Default is 0.
    polarization_type : str, optional
        Polarization type. Default is ``"unpolarized"``.

    Returns
    -------
    config : PolarizationConfig
        Validated polarization configuration.

    See Also
    --------
    PolarizationConfig : The PyTree class constructed by this
        factory.
    """
    theta_arr: Float[Array, " "] = jnp.asarray(theta, dtype=jnp.float64)
    phi_arr: Float[Array, " "] = jnp.asarray(phi, dtype=jnp.float64)
    pol_arr: Float[Array, " "] = jnp.asarray(
        polarization_angle, dtype=jnp.float64
    )

    def validate_and_create() -> PolarizationConfig:
        nonlocal phi_arr, pol_arr, theta_arr
        theta_arr = eqx.error_if(
            theta_arr,
            ~(jnp.isfinite(theta_arr)),
            "make_polarization_config: theta finite",
        )
        phi_arr = eqx.error_if(
            phi_arr,
            ~(jnp.isfinite(phi_arr)),
            "make_polarization_config: phi finite",
        )
        pol_arr = eqx.error_if(
            pol_arr,
            ~(jnp.isfinite(pol_arr)),
            "make_polarization_config: polarization angle finite",
        )
        return PolarizationConfig(
            theta=theta_arr,
            phi=phi_arr,
            polarization_angle=pol_arr,
            polarization_type=polarization_type,
        )

    config: PolarizationConfig = validate_and_create()
    return config


def make_expanded_simulation_params(
    eigenbands: Float[Array, "K B"],
    fidelity: int = 25000,
    sigma: ScalarFloat = 0.04,
    gamma: ScalarFloat = 0.1,
    temperature: ScalarFloat = 15.0,
    photon_energy: ScalarFloat = 11.0,
    energy_padding: ScalarFloat = 1.0,
) -> SimulationParams:
    """Build simulation parameters with auto-derived energy window.

    Constructs a :class:`~diffpes.types.SimulationParams` PyTree whose
    energy window is derived from the actual band energies rather than
    from fixed defaults.  The window spans
    ``[min(eigenbands) - energy_padding, max(eigenbands) + energy_padding]``,
    ensuring every band falls within the simulated range.

    Implementation Logic
    --------------------
    1. **Cast to float64**: ``eigenbands`` is promoted to
       ``jnp.float64`` via ``jnp.asarray``.
    2. **Derive energy bounds**: ``energy_min`` and ``energy_max``
       are computed from the global min/max of the band array plus
       symmetric padding controlled by ``energy_padding``.
    3. **Delegate**: Passes all values to
       :func:`~diffpes.types.make_simulation_params` for final
       construction and type validation.

    Parameters
    ----------
    eigenbands : Float[Array, "K B"]
        Band eigenvalues in eV, shape ``(n_kpoints, n_bands)``.
        Used only to derive ``energy_min`` and ``energy_max``.
    fidelity : int, optional
        Number of points in the energy axis. Default is 25000.
    sigma : ScalarFloat, optional
        Gaussian broadening width in eV. Default is 0.04.
    gamma : ScalarFloat, optional
        Lorentzian broadening width in eV. Default is 0.1.
    temperature : ScalarFloat, optional
        Electronic temperature in Kelvin. Default is 15.
    photon_energy : ScalarFloat, optional
        Incident photon energy in eV. Default is 11.
    energy_padding : ScalarFloat, optional
        Symmetric padding around band extrema in eV. Default is 1.

    Returns
    -------
    params : SimulationParams
        Simulation parameters with data-derived energy window.

    See Also
    --------
    make_simulation_params : Lower-level factory with explicit
        energy bounds.
    """
    bands_arr: Float[Array, "K B"] = jnp.asarray(eigenbands, dtype=jnp.float64)
    pad: Float[Array, " "] = jnp.asarray(energy_padding, dtype=jnp.float64)
    energy_min: Float[Array, " "] = jnp.min(bands_arr) - pad
    energy_max: Float[Array, " "] = jnp.max(bands_arr) + pad
    params: SimulationParams = make_simulation_params(
        energy_min=energy_min,
        energy_max=energy_max,
        fidelity=fidelity,
        sigma=sigma,
        gamma=gamma,
        temperature=temperature,
        photon_energy=photon_energy,
    )
    return params


__all__: list[str] = [
    "PolarizationConfig",
    "SimulationParams",
    "make_expanded_simulation_params",
    "make_polarization_config",
    "make_simulation_params",
]
