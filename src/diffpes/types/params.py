"""Define simulation-parameter data structures.

Extended Summary
----------------
This module defines PyTree types for ARPES simulation parameters, including
energy resolution, broadening widths, temperature, photon energy,
and light polarization configuration.

Routine Listings
----------------
:class:`PolarizationConfig`
    Store photon-polarization geometry in a JAX PyTree.
:class:`SimulationParams`
    Store ARPES simulation parameters in a JAX PyTree.
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

_POLARIZATION_TYPES: tuple[str, ...] = (
    "LVP",
    "LHP",
    "RCP",
    "LCP",
    "LAP",
    "unpolarized",
)
_MIN_FIDELITY: int = 2


class SimulationParams(eqx.Module):
    """Store ARPES simulation parameters in a JAX PyTree.

    This type collects the scalar physical parameters that control an ARPES
    simulation. These parameters define the energy window, discretization,
    broadening widths, sample temperature, and photon energy. The spectral
    function and Fermi-Dirac weighting routines use these parameters.

    This class is an immutable :class:`equinox.Module` PyTree. All
    JAX stores the float-valued fields as array children that support
    autodiff. It stores the Python ``int`` field ``fidelity`` as auxiliary
    data. This field controls array shapes and must remain concrete during
    compilation.


    :see: :class:`~.test_params.TestSimulationParams`

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
        Number of points along the energy axis (**static** -- a compile-time
        constant; changing it triggers retracing).

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
    """Store photon-polarization geometry in a JAX PyTree.

    This type describes the photon-polarization state and incidence geometry
    for an ARPES experiment. Three angular fields define the light direction
    and polarization orientation. The string ``polarization_type`` selects an
    optics convention: s-pol, p-pol, circular, arbitrary linear, or
    unpolarized.

    This class is an immutable :class:`equinox.Module` PyTree. JAX stores the
    three angular fields as array children that support autodiff. It stores
    ``polarization_type`` as auxiliary data because JAX cannot trace strings.


    :see: :class:`~.test_params.TestPolarizationConfig`

    Attributes
    ----------
    theta : Float[Array, " "]
        Incident angle from surface normal in radians.
    phi : Float[Array, " "]
        In-plane azimuthal angle in radians.
    polarization_angle : Float[Array, " "]
        Arbitrary linear polarization angle in radians.
    polarization_type : str
        One of LVP, LHP, RCP, LCP, LAP, unpolarized (**static** -- a
        compile-time constant; changing it triggers retracing).

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
def make_simulation_params(  # noqa: DOC503
    energy_min: ScalarNumeric = -3.0,
    energy_max: ScalarNumeric = 1.0,
    fidelity: int = 25000,
    sigma: ScalarFloat = 0.04,
    gamma: ScalarFloat = 0.1,
    temperature: ScalarFloat = 15.0,
    photon_energy: ScalarFloat = 11.0,
) -> SimulationParams:
    """Create a validated SimulationParams instance.

    The factory validates and normalizes the inputs before it constructs the
    SimulationParams PyTree. It casts float-valued parameters to 0-D float64
    JAX arrays. It keeps ``fidelity`` as a Python int in the PyTree auxiliary
    data.

    The defaults represent a typical low-temperature ARPES experiment. The
    energy window defaults to [-3.0, 1.0] eV. Higher-level simulation drivers
    can override this window from the supplied eigenband energy range.

    :see: :class:`~.test_params.TestMakeSimulationParams`

    Implementation Logic
    --------------------
    1. **Prepare the normalized values**::

           emin_arr = jnp.asarray(energy_min, dtype=jnp.float64)

       This expression gives the later validation steps a stable shape and
       dtype.

    2. **Apply static validation**::

           fidelity < _MIN_FIDELITY

       This predicate rejects invalid structure before JAX traces the
       numerical checks.

    3. **Apply traced validation**::

           ~jnp.isfinite(emin_arr)

       This predicate remains active during eager and compiled execution.

    4. **Return the named instance**::

           return params

       The explicit name keeps the implementation and the Returns section
       synchronized.

    Parameters
    ----------
    energy_min : ScalarNumeric, optional
        Lower energy bound in eV. Default is -3.0.
    energy_max : ScalarNumeric, optional
        Upper energy bound in eV. Default is 1.0.
    fidelity : int, optional
        Energy axis resolution (**static** -- a compile-time constant;
        changing it triggers retracing). Default is 25000.
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

    Raises
    ------
    ValueError
        If ``fidelity`` is less than two.
    EquinoxRuntimeError
        If an energy bound or physical scalar is non-finite, if
        ``energy_min >= energy_max``, or if any broadening width,
        temperature, or photon energy is not positive.

    Notes
    -----
    Static validation raises ``ValueError`` before traced construction when
    ``fidelity`` is less than two. Traced validation uses ``eqx.error_if``
    and raises ``EquinoxRuntimeError`` for non-finite values, an unordered
    energy window, or non-positive physical scalars.

    See Also
    --------
    SimulationParams : The PyTree class constructed by this factory.
    """
    if fidelity < _MIN_FIDELITY:
        msg: str = "make_simulation_params: fidelity must be at least 2"
        raise ValueError(msg)

    emin_arr: Float[Array, " "] = jnp.asarray(energy_min, dtype=jnp.float64)
    emax_arr: Float[Array, " "] = jnp.asarray(energy_max, dtype=jnp.float64)
    sigma_arr: Float[Array, " "] = jnp.asarray(sigma, dtype=jnp.float64)
    gamma_arr: Float[Array, " "] = jnp.asarray(gamma, dtype=jnp.float64)
    temp_arr: Float[Array, " "] = jnp.asarray(temperature, dtype=jnp.float64)
    pe_arr: Float[Array, " "] = jnp.asarray(photon_energy, dtype=jnp.float64)

    def validate_and_create() -> SimulationParams:
        nonlocal emax_arr, emin_arr, gamma_arr, pe_arr, sigma_arr, temp_arr
        emin_arr = eqx.error_if(
            emin_arr,
            ~jnp.isfinite(emin_arr),
            "make_simulation_params: energy_min must be finite",
        )
        emax_arr = eqx.error_if(
            emax_arr,
            ~jnp.isfinite(emax_arr),
            "make_simulation_params: energy_max must be finite",
        )
        emin_arr = eqx.error_if(
            emin_arr,
            ~(emin_arr < emax_arr),
            "make_simulation_params: energy_min must be less than energy_max",
        )
        sigma_arr = eqx.error_if(
            sigma_arr,
            ~jnp.isfinite(sigma_arr),
            "make_simulation_params: sigma must be finite",
        )
        sigma_arr = eqx.error_if(
            sigma_arr,
            ~(sigma_arr > 0.0),
            "make_simulation_params: sigma must be positive",
        )
        gamma_arr = eqx.error_if(
            gamma_arr,
            ~jnp.isfinite(gamma_arr),
            "make_simulation_params: gamma must be finite",
        )
        gamma_arr = eqx.error_if(
            gamma_arr,
            ~(gamma_arr > 0.0),
            "make_simulation_params: gamma must be positive",
        )
        temp_arr = eqx.error_if(
            temp_arr,
            ~jnp.isfinite(temp_arr),
            "make_simulation_params: temperature must be finite",
        )
        temp_arr = eqx.error_if(
            temp_arr,
            ~(temp_arr > 0.0),
            "make_simulation_params: temperature must be positive",
        )
        pe_arr = eqx.error_if(
            pe_arr,
            ~jnp.isfinite(pe_arr),
            "make_simulation_params: photon_energy must be finite",
        )
        pe_arr = eqx.error_if(
            pe_arr,
            ~(pe_arr > 0.0),
            "make_simulation_params: photon_energy must be positive",
        )
        validated_params: SimulationParams = SimulationParams(
            energy_min=emin_arr,
            energy_max=emax_arr,
            fidelity=fidelity,
            sigma=sigma_arr,
            gamma=gamma_arr,
            temperature=temp_arr,
            photon_energy=pe_arr,
        )
        return validated_params

    params: SimulationParams = validate_and_create()
    return params


@jaxtyped(typechecker=beartype)
def make_polarization_config(  # noqa: DOC503
    theta: ScalarFloat = 0.7854,
    phi: ScalarFloat = 0.0,
    polarization_angle: ScalarFloat = 0.0,
    polarization_type: str = "unpolarized",
) -> PolarizationConfig:
    """Create a validated PolarizationConfig instance.

    The factory validates and normalizes the inputs before it constructs the
    PolarizationConfig PyTree. It casts angular parameters to 0-D float64 JAX
    arrays. It keeps ``polarization_type`` as a Python string in the PyTree
    auxiliary data.

    The default incidence angle ``theta = 0.7854`` rad equals pi/4, or
    approximately 45 degrees. This common geometry provides sensitivity to
    in-plane and out-of-plane orbital components.

    :see: :class:`~.test_params.TestMakePolarizationConfig`

    Implementation Logic
    --------------------
    1. **Prepare the normalized values**::

           theta_arr = jnp.asarray(theta, dtype=jnp.float64)

       This expression gives the later validation steps a stable shape and
       dtype.

    2. **Apply static validation**::

           polarization_type not in _POLARIZATION_TYPES

       This predicate rejects invalid structure before JAX traces the
       numerical checks.

    3. **Apply traced validation**::

           ~jnp.isfinite(theta_arr)

       This predicate remains active during eager and compiled execution.

    4. **Return the named instance**::

           return config

       The explicit name keeps the implementation and the Returns section
       synchronized.

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
        Polarization type (**static** -- a compile-time constant; changing it
        triggers retracing). Default is ``"unpolarized"``.

    Returns
    -------
    config : PolarizationConfig
        Validated polarization configuration.

    Raises
    ------
    ValueError
        If ``polarization_type`` is not one of ``LVP``, ``LHP``,
        ``RCP``, ``LCP``, ``LAP``, or ``unpolarized``.
    EquinoxRuntimeError
        If any angular parameter is non-finite.

    Notes
    -----
    Static validation raises ``ValueError`` before traced construction for an
    unsupported ``polarization_type``. Traced validation uses
    ``eqx.error_if`` and raises ``EquinoxRuntimeError`` when an angle is
    non-finite.

    See Also
    --------
    PolarizationConfig : The PyTree class constructed by this
        factory.
    """
    if polarization_type not in _POLARIZATION_TYPES:
        msg: str = (
            "make_polarization_config: polarization_type must be one of "
            f"{_POLARIZATION_TYPES}"
        )
        raise ValueError(msg)

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
        validated_config: PolarizationConfig = PolarizationConfig(
            theta=theta_arr,
            phi=phi_arr,
            polarization_angle=pol_arr,
            polarization_type=polarization_type,
        )
        return validated_config

    config: PolarizationConfig = validate_and_create()
    return config


@jaxtyped(typechecker=beartype)
def make_expanded_simulation_params(  # noqa: DOC503
    eigenbands: Float[Array, "K B"],
    fidelity: int = 25000,
    sigma: ScalarFloat = 0.04,
    gamma: ScalarFloat = 0.1,
    temperature: ScalarFloat = 15.0,
    photon_energy: ScalarFloat = 11.0,
    energy_padding: ScalarFloat = 1.0,
) -> SimulationParams:
    """Build simulation parameters with auto-derived energy window.

    The factory constructs a :class:`~diffpes.types.SimulationParams` PyTree.
    It derives the energy window from the actual band energies instead of
    fixed defaults. The window spans
    ``[min(eigenbands) - energy_padding, max(eigenbands) + energy_padding]``,
    ensuring every band falls within the simulated range.

    :see: :class:`~.test_params.TestMakeExpandedSimulationParams`

    Implementation Logic
    --------------------
    1. **Prepare the normalized values**::

           bands_arr = jnp.asarray(eigenbands, dtype=jnp.float64)

       This expression gives the later validation steps a stable shape and
       dtype.

    2. **Apply static validation**::

           bands_arr.size == 0

       This predicate rejects invalid structure before JAX traces the
       numerical checks.

    3. **Apply traced validation**::

           ~jnp.all(jnp.isfinite(bands_arr))

       This predicate remains active during eager and compiled execution.

    4. **Return the named instance**::

           return params

       The explicit name keeps the implementation and the Returns section
       synchronized.

    Parameters
    ----------
    eigenbands : Float[Array, "K B"]
        Band eigenvalues in eV, shape ``(n_kpoints, n_bands)``.
        Used only to derive ``energy_min`` and ``energy_max``.
    fidelity : int, optional
        Number of points in the energy axis (**static** -- a compile-time
        constant; changing it triggers retracing). Default is 25000.
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

    Raises
    ------
    ValueError
        If ``eigenbands`` is empty, resolved statically from its shape.
    EquinoxRuntimeError
        If ``eigenbands`` or ``energy_padding`` is non-finite, or if
        ``energy_padding`` is negative under eager or compiled execution.

    Notes
    -----
    Static validation raises ``ValueError`` before tracing when the band array
    is empty. Traced validation uses ``eqx.error_if`` and raises
    ``EquinoxRuntimeError`` for non-finite bands or for padding that is
    non-finite or negative. The delegated factory applies its own documented
    static and traced validation contract.

    See Also
    --------
    make_simulation_params : Lower-level factory with explicit
        energy bounds.
    """
    bands_arr: Float[Array, "K B"] = jnp.asarray(eigenbands, dtype=jnp.float64)
    pad: Float[Array, " "] = jnp.asarray(energy_padding, dtype=jnp.float64)
    if bands_arr.size == 0:
        msg: str = "eigenbands must contain at least one value"
        raise ValueError(msg)
    bands_arr = eqx.error_if(
        bands_arr,
        ~jnp.all(jnp.isfinite(bands_arr)),
        "make_expanded_simulation_params: eigenbands finite",
    )
    pad = eqx.error_if(
        pad,
        ~jnp.isfinite(pad) | (pad < 0.0),
        "make_expanded_simulation_params: padding finite and nonnegative",
    )
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
