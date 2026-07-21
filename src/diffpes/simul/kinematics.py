r"""Compute free-electron photoemission kinematics.

Extended Summary
----------------
This module maps photon energy, binding energy, and detector angles to
photoelectron momenta. It implements the free-electron final-state model with
an inner potential. All array operations support JAX transformations.

The work function has two roles in an inverse problem. Energy referencing has
a work-function and Fermi-level offset gauge. The inner-potential relation
also depends on the work function. A photon-energy scan can reduce this gauge.

Routine Listings
----------------
:func:`detector_angles_to_kpar`
    Convert detector angles to parallel momentum.
:func:`emission_angles`
    Convert Cartesian momentum to emission angles.
:func:`final_state_k_inv_ang`
    Convert kinetic energy to final-state momentum magnitude.
:func:`kinetic_energy_ev`
    Compute the floored photoelectron kinetic energy.
:func:`kpar_to_detector_angles`
    Convert parallel momentum to detector angles.
:func:`kz_from_inner_potential`
    Compute complex out-of-plane momentum from the inner potential.

Notes
-----
Public angles use radians because they are detector-frame coordinates. The
horizontal slit uses ``Rx(ty) @ Ry(tx)``. The vertical slit uses
``Rx(tx) @ Ry(ty)``.

DiffPES maps Chinook's horizontal ``tilt.k_mesh`` angles as ``T=-tx`` and
``P=ty``. It maps ``gen_all_pol`` angles as ``theta=-tx`` and ``phi=-ty``.
For the vertical slit, the corresponding mappings are ``T=-ty, P=tx`` and
``theta=-ty, phi=-tx``. These mappings give one active detector frame.
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Bool, Complex, Float, jaxtyped

from diffpes.maths import safe_arctan2, safe_divide, safe_norm, safe_sqrt
from diffpes.types import (
    EKIN_FLOOR_EV,
    K_PREFACTOR_INV_ANG_SQRT_EV,
    TWO_ME_OVER_HBAR_SQ_INV_EV_ANG2,
    ScalarFloat,
)


@jaxtyped(typechecker=beartype)
def kinetic_energy_ev(
    photon_energy_ev: ScalarFloat,
    work_function_ev: ScalarFloat,
    binding_energy_ev: Float[Array, " ..."],
) -> Float[Array, " ..."]:
    r"""Compute the floored photoelectron kinetic energy.

    The function applies energy conservation in the three-step photoemission
    model [3]_. A physical floor defines the low-energy validity boundary.

    :see: :class:`~.test_kinematics.TestKineticEnergyEv`

    Parameters
    ----------
    photon_energy_ev : ScalarFloat
        Photon energy in eV.
    work_function_ev : ScalarFloat
        Work function in eV.
    binding_energy_ev : Float[Array, " ..."]
        Binding energies in eV. The function accepts either sign convention.

    Returns
    -------
    kinetic_energies : Float[Array, " ..."]
        Kinetic energies in eV with a lower bound of ``EKIN_FLOOR_EV``.

    Notes
    -----
    The function computes :math:`h\nu-W-|E_b|`. Values at or below the floor
    have a zero selected gradient. Values above the floor keep exact gradients.

    References
    ----------
    .. [3] A. Damascelli, Z. Hussain, and Z.-X. Shen, Rev. Mod. Phys. 75,
       473 (2003).
    """
    photon_energy_array: Float[Array, ""] = jnp.asarray(photon_energy_ev)
    work_function_array: Float[Array, ""] = jnp.asarray(work_function_ev)
    raw_kinetic_energies: Float[Array, " ..."] = (
        photon_energy_array - work_function_array - jnp.abs(binding_energy_ev)
    )
    above_floor: Bool[Array, " ..."] = raw_kinetic_energies > EKIN_FLOOR_EV
    sanitized_energies: Float[Array, " ..."] = jnp.where(
        above_floor,
        raw_kinetic_energies,
        EKIN_FLOOR_EV,
    )
    kinetic_energies: Float[Array, " ..."] = jnp.where(
        above_floor,
        sanitized_energies,
        EKIN_FLOOR_EV,
    )
    return kinetic_energies


@jaxtyped(typechecker=beartype)
def final_state_k_inv_ang(
    kinetic_energy_ev: Float[Array, " ..."],
) -> Float[Array, " ..."]:
    """Convert kinetic energy to final-state momentum magnitude.

    The function applies the free-electron dispersion. A second floor guard
    keeps direct calls finite outside the physical domain.

    :see: :class:`~.test_kinematics.TestFinalStateKInvAng`

    Parameters
    ----------
    kinetic_energy_ev : Float[Array, " ..."]
        Photoelectron kinetic energies in eV.

    Returns
    -------
    momentum_magnitudes : Float[Array, " ..."]
        Final-state momentum magnitudes in 1/Angstrom.

    Notes
    -----
    The function computes ``K_PREFACTOR_INV_ANG_SQRT_EV * sqrt(E_kin)``.
    Inputs at or below ``EKIN_FLOOR_EV`` have a zero selected gradient.
    """
    above_floor: Bool[Array, " ..."] = kinetic_energy_ev > EKIN_FLOOR_EV
    sanitized_energies: Float[Array, " ..."] = jnp.where(
        above_floor,
        kinetic_energy_ev,
        EKIN_FLOOR_EV,
    )
    physical_momenta: Float[Array, " ..."] = (
        K_PREFACTOR_INV_ANG_SQRT_EV * jnp.sqrt(sanitized_energies)
    )
    floor_momentum: float = K_PREFACTOR_INV_ANG_SQRT_EV * EKIN_FLOOR_EV**0.5
    momentum_magnitudes: Float[Array, " ..."] = jnp.where(
        above_floor,
        physical_momenta,
        floor_momentum,
    )
    return momentum_magnitudes


@jaxtyped(typechecker=beartype)
def kz_from_inner_potential(
    photon_energy_ev: ScalarFloat,
    work_function_ev: ScalarFloat,
    inner_potential_ev: ScalarFloat,
    k_par_inv_ang: Float[Array, " ..."],
) -> Tuple[Complex[Array, " ..."], Bool[Array, " ..."]]:
    r"""Compute complex out-of-plane momentum from the inner potential.

    The function implements the free-electron final-state approximation [4]_.
    Its principal complex root retains evanescent channels.

    :see: :class:`~.test_kinematics.TestKzFromInnerPotential`

    Parameters
    ----------
    photon_energy_ev : ScalarFloat
        Photon energy in eV.
    work_function_ev : ScalarFloat
        Work function in eV.
    inner_potential_ev : ScalarFloat
        Inner potential in eV.
    k_par_inv_ang : Float[Array, " ..."]
        Parallel momentum magnitudes in 1/Angstrom.

    Returns
    -------
    kz_values : Complex[Array, " ..."]
        Principal out-of-plane momenta in 1/Angstrom.
    propagating : Bool[Array, " ..."]
        Mask that identifies positive real radicands.

    Notes
    -----
    The radicand equals
    :math:`(2m_e/\hbar^2)(h\nu-W+V_0)-k_\parallel^2` above the energy floor.
    Negative radicands give positive imaginary roots. The branch point has no
    assigned derivative.

    For a propagating channel,
    :math:`\partial k_z/\partial V_0=(2\,\hbar^2/2m_e)^{-1}/k_z`.

    References
    ----------
    .. [4] A. Damascelli, Z. Hussain, and Z.-X. Shen, Rev. Mod. Phys. 75,
       473 (2003).
    """
    zero_binding: Float[Array, " ..."] = jnp.zeros_like(k_par_inv_ang)
    surface_kinetic_energies: Float[Array, " ..."] = kinetic_energy_ev(
        photon_energy_ev,
        work_function_ev,
        zero_binding,
    )
    inner_potential_array: Float[Array, ""] = jnp.asarray(inner_potential_ev)
    radicand: Float[Array, " ..."] = (
        TWO_ME_OVER_HBAR_SQ_INV_EV_ANG2
        * (surface_kinetic_energies + inner_potential_array)
        - k_par_inv_ang * k_par_inv_ang
    )
    propagating: Bool[Array, " ..."] = radicand > 0.0
    complex_radicand: Complex[Array, " ..."] = radicand.astype(jnp.complex128)
    kz_values: Complex[Array, " ..."] = jnp.sqrt(complex_radicand)
    kinematics_result: Tuple[Complex[Array, " ..."], Bool[Array, " ..."]] = (
        kz_values,
        propagating,
    )
    return kinematics_result


@jaxtyped(typechecker=beartype)
def emission_angles(
    k_cart_inv_ang: Float[Array, "... 3"],
) -> Tuple[Float[Array, " ..."], Float[Array, " ..."]]:
    """Convert Cartesian momentum to emission angles.

    The function returns the polar angle from positive z and the azimuth from
    positive x. It selects zero azimuth at normal emission.

    :see: :class:`~.test_kinematics.TestEmissionAngles`

    Parameters
    ----------
    k_cart_inv_ang : Float[Array, "... 3"]
        Cartesian momentum vectors in 1/Angstrom.

    Returns
    -------
    theta : Float[Array, " ..."]
        Polar emission angles in radians.
    phi : Float[Array, " ..."]
        Azimuthal emission angles in radians.

    Notes
    -----
    The polar angle uses ``arctan2(norm([kx, ky]), kz)``. The azimuth uses
    ``arctan2(ky, kx)``. Safe primitives give zero coordinate gradients at
    their undefined origins.
    """
    k_parallel: Float[Array, " ..."] = safe_norm(k_cart_inv_ang[..., :2])
    theta: Float[Array, " ..."] = safe_arctan2(
        k_parallel,
        k_cart_inv_ang[..., 2],
    )
    phi: Float[Array, " ..."] = safe_arctan2(
        k_cart_inv_ang[..., 1],
        k_cart_inv_ang[..., 0],
    )
    angles: Tuple[Float[Array, " ..."], Float[Array, " ..."]] = (theta, phi)
    return angles


@jaxtyped(typechecker=beartype)
def detector_angles_to_kpar(
    tx: Float[Array, " ..."],
    ty: Float[Array, " ..."],
    kinetic_energy_ev: Float[Array, " ..."],
    slit: str,
) -> Float[Array, "... 2"]:
    """Convert detector angles to parallel momentum.

    The function rotates the positive z direction with the Plan 03 detector
    convention. It broadcasts all traced inputs over their leading axes.

    :see: :class:`~.test_kinematics.TestDetectorAnglesToKpar`

    Parameters
    ----------
    tx : Float[Array, " ..."]
        First detector angles in radians.
    ty : Float[Array, " ..."]
        Second detector angles in radians.
    kinetic_energy_ev : Float[Array, " ..."]
        Photoelectron kinetic energies in eV.
    slit : str
        Static slit orientation, ``"H"`` or ``"V"``. A change causes
        retracing.

    Returns
    -------
    k_parallel : Float[Array, "... 2"]
        Cartesian parallel momenta ``(kx, ky)`` in 1/Angstrom.

    Raises
    ------
    ValueError
        If ``slit`` is not ``"H"`` or ``"V"``.

    Notes
    -----
    The horizontal slit uses ``Rx(ty) @ Ry(tx)``. The vertical slit uses
    ``Rx(tx) @ Ry(ty)``. These active rotations act on the positive z vector.
    """
    if slit not in {"H", "V"}:
        message: str = "slit must be 'H' or 'V'"
        raise ValueError(message)
    broadcast_tx: Float[Array, " ..."]
    broadcast_ty: Float[Array, " ..."]
    broadcast_energy: Float[Array, " ..."]
    broadcast_tx, broadcast_ty, broadcast_energy = jnp.broadcast_arrays(
        tx,
        ty,
        kinetic_energy_ev,
    )
    momentum_magnitudes: Float[Array, " ..."] = final_state_k_inv_ang(
        broadcast_energy
    )
    if slit == "H":
        kx: Float[Array, " ..."] = momentum_magnitudes * jnp.sin(broadcast_tx)
        ky: Float[Array, " ..."] = (
            -momentum_magnitudes
            * jnp.cos(broadcast_tx)
            * jnp.sin(broadcast_ty)
        )
    else:
        kx = momentum_magnitudes * jnp.sin(broadcast_ty)
        ky = (
            -momentum_magnitudes
            * jnp.sin(broadcast_tx)
            * jnp.cos(broadcast_ty)
        )
    k_parallel: Float[Array, "... 2"] = jnp.stack((kx, ky), axis=-1)
    return k_parallel


@jaxtyped(typechecker=beartype)
def kpar_to_detector_angles(
    k_par_inv_ang: Float[Array, "... 2"],
    kinetic_energy_ev: Float[Array, " ..."],
    slit: str,
) -> Tuple[Float[Array, " ..."], Float[Array, " ..."]]:
    """Convert parallel momentum to detector angles.

    The function gives the exact inverse detector map on the physical domain.
    This domain requires ``norm(k_parallel) < k_f``.

    :see: :class:`~.test_kinematics.TestKparToDetectorAngles`

    Parameters
    ----------
    k_par_inv_ang : Float[Array, "... 2"]
        Cartesian parallel momenta ``(kx, ky)`` in 1/Angstrom.
    kinetic_energy_ev : Float[Array, " ..."]
        Photoelectron kinetic energies in eV.
    slit : str
        Static slit orientation, ``"H"`` or ``"V"``. A change causes
        retracing.

    Returns
    -------
    tx : Float[Array, " ..."]
        First detector angles in radians.
    ty : Float[Array, " ..."]
        Second detector angles in radians.

    Raises
    ------
    ValueError
        If ``slit`` is not ``"H"`` or ``"V"``.

    Notes
    -----
    The inverse uses the positive detector-normal branch. Safe square roots
    select finite boundary values outside the open physical domain.
    """
    if slit not in {"H", "V"}:
        message: str = "slit must be 'H' or 'V'"
        raise ValueError(message)
    target_shape: tuple[int, ...] = jnp.broadcast_shapes(
        k_par_inv_ang.shape[:-1],
        kinetic_energy_ev.shape,
    )
    broadcast_k_parallel: Float[Array, "... 2"] = jnp.broadcast_to(
        k_par_inv_ang,
        (*target_shape, 2),
    )
    broadcast_energy: Float[Array, " ..."] = jnp.broadcast_to(
        kinetic_energy_ev,
        target_shape,
    )
    momentum_magnitudes: Float[Array, " ..."] = final_state_k_inv_ang(
        broadcast_energy
    )
    normalized_k_parallel: Float[Array, "... 2"] = safe_divide(
        broadcast_k_parallel,
        momentum_magnitudes[..., None],
    )
    normalized_kx: Float[Array, " ..."] = normalized_k_parallel[..., 0]
    normalized_ky: Float[Array, " ..."] = normalized_k_parallel[..., 1]
    normal_component: Float[Array, " ..."] = safe_sqrt(
        1.0 - normalized_kx * normalized_kx - normalized_ky * normalized_ky
    )
    if slit == "H":
        tx: Float[Array, " ..."] = safe_arctan2(
            normalized_kx,
            safe_sqrt(1.0 - normalized_kx * normalized_kx),
        )
        ty: Float[Array, " ..."] = safe_arctan2(
            -normalized_ky,
            normal_component,
        )
    else:
        tx = safe_arctan2(-normalized_ky, normal_component)
        ty = safe_arctan2(
            normalized_kx,
            safe_sqrt(1.0 - normalized_kx * normalized_kx),
        )
    detector_angles: Tuple[Float[Array, " ..."], Float[Array, " ..."]] = (
        tx,
        ty,
    )
    return detector_angles


__all__: list[str] = [
    "detector_angles_to_kpar",
    "emission_angles",
    "final_state_k_inv_ang",
    "kinetic_energy_ev",
    "kpar_to_detector_angles",
    "kz_from_inner_potential",
]
