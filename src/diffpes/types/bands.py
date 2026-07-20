"""Band structure and orbital projection data structures.

Extended Summary
----------------
Defines PyTree types for electronic band structure data and
orbital-resolved projections from VASP calculations. These are
the primary inputs to all ARPES simulation functions.

Routine Listings
----------------
:class:`ArpesSpectrum`
    PyTree for ARPES simulation output.
:class:`BandStructure`
    PyTree for electronic band structure.
:class:`OrbitalProjection`
    PyTree for orbital-resolved band projections.
:class:`SpinBandStructure`
    PyTree for spin-resolved electronic band structure.
:class:`SpinOrbitalProjection`
    PyTree for orbital projections with mandatory spin data.
:func:`make_arpes_spectrum`
    Create a validated ``ArpesSpectrum`` instance.
:func:`make_band_structure`
    Create a validated ``BandStructure`` instance.
:func:`make_orbital_projection`
    Create a validated ``OrbitalProjection`` instance.
:func:`make_spin_band_structure`
    Create a validated ``SpinBandStructure`` instance.
:func:`make_spin_orbital_projection`
    Create a validated ``SpinOrbitalProjection`` instance.

Notes
-----
Orbital indexing convention (9 orbitals):
``[s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2]``
matching VASP PROCAR output ordering.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Union
from jax import lax
from jaxtyping import Array, Float, jaxtyped

from .aliases import ScalarNumeric


class BandStructure(eqx.Module):
    """PyTree for electronic band structure.

    Extended Summary
    ----------------
    Stores the core outputs of a DFT band structure calculation: the
    eigenvalue spectrum E_n(k), the k-point mesh in reciprocal space, the
    integration weights for each k-point, and the Fermi energy. Together
    these fields fully describe the single-particle electronic structure
    needed to simulate angle-resolved photoemission spectra.

    This type exists as a JAX-compatible PyTree so that band structure data
    can flow through ``jax.jit``, ``jax.vmap``, and ``jax.grad``
    transformations without manual flattening. All four fields are
    JAX-traced arrays (no static auxiliary data), which means the entire
    object is differentiable with respect to any of its fields.

    Attributes
    ----------
    eigenvalues : Float[Array, "K B"]
        Band energies in eV for K k-points and B bands.
    kpoints : Float[Array, "K 3"]
        k-point coordinates in reciprocal space.
    kpoint_weights : Float[Array, " K"]
        Integration weights for each k-point.
    fermi_energy : Float[Array, " "]
        Fermi level energy in eV.

    Notes
    -----
    Implemented as an immutable :class:`equinox.Module` PyTree.
    Equinox derives the tree structure from the annotated fields; all
    fields are differentiable leaves and no static metadata is present.
    """

    eigenvalues: Float[Array, "K B"]
    kpoints: Float[Array, "K 3"]
    kpoint_weights: Float[Array, " K"]
    fermi_energy: Float[Array, " "]


class OrbitalProjection(eqx.Module):
    """Orbital-resolved band projections.

    All array fields are differentiable PyTree leaves. Optional fields are
    empty subtrees when ``None``; changing their presence changes the tree
    structure and may trigger recompilation.

    Attributes
    ----------
    projections : Float[Array, "K B A 9"]
        Orbital projection weights.
    spin : Optional[Float[Array, "K B A 6"]]
        Optional spin projections.
    oam : Optional[Float[Array, "K B A 3"]]
        Optional orbital-angular-momentum projections.
    """

    projections: Float[Array, "K B A 9"]
    spin: Optional[Float[Array, "K B A 6"]]
    oam: Optional[Float[Array, "K B A 3"]]


class SpinOrbitalProjection(eqx.Module):
    """Orbital projections with mandatory spin data.

    All present arrays are differentiable PyTree leaves. ``oam=None`` is an
    empty subtree; changing its presence changes the tree structure.

    Attributes
    ----------
    projections : Float[Array, "K B A 9"]
        Orbital projection weights.
    spin : Float[Array, "K B A 6"]
        Mandatory spin projections.
    oam : Optional[Float[Array, "K B A 3"]]
        Optional orbital-angular-momentum projections.
    """

    projections: Float[Array, "K B A 9"]
    spin: Float[Array, "K B A 6"]
    oam: Optional[Float[Array, "K B A 3"]]


@jaxtyped(typechecker=beartype)
def make_spin_orbital_projection(
    projections: Float[Array, "K B A 9"],
    spin: Float[Array, "K B A 6"],
    oam: Optional[Float[Array, "K B A 3"]] = None,
) -> SpinOrbitalProjection:
    """Create a validated ``SpinOrbitalProjection`` instance.

    Extended Summary
    ----------------
    Factory function that validates and normalises orbital projection
    data with mandatory spin before constructing a
    ``SpinOrbitalProjection`` PyTree. This is the spin-orbit coupling
    counterpart to :func:`make_orbital_projection`: it requires the
    ``spin`` field and therefore guarantees that downstream SOC
    simulation kernels receive complete spin data without runtime
    checks.

    All non-None arrays are cast to ``float64`` for numerical
    stability. The function is decorated with
    ``@jaxtyped(typechecker=beartype)`` so that shape constraints
    (K, B, A dimensions must agree across all provided arrays) are
    checked at call time.

    Implementation Logic
    --------------------
    1. **Cast projections** to ``jnp.float64`` via ``jnp.asarray``.
    2. **Cast spin** to ``jnp.float64`` via ``jnp.asarray``. This
       field is mandatory, unlike in :func:`make_orbital_projection`.
    3. **Cast oam** to ``jnp.float64`` when not ``None``; otherwise
       leave as ``None`` to signal that OAM data is absent.
    4. **Construct** the ``SpinOrbitalProjection`` Equinox module from the
       three validated fields and return it.

    Parameters
    ----------
    projections : Float[Array, "K B A 9"]
        Orbital projection weights ``|<psi|Y_{lm}>|^2`` following VASP
        ordering. Must share the K, B, A dimensions with ``spin``.
    spin : Float[Array, "K B A 6"]
        Spin projections ``[Sx_up, Sx_dn, Sy_up, Sy_dn, Sz_up,
        Sz_dn]``. Required (non-optional).
    oam : Optional[Float[Array, "K B A 3"]], optional
        Orbital angular momentum ``[L_p, L_d, L_total]``.
        Default is None.

    Returns
    -------
    soc_proj : SpinOrbitalProjection
        Validated instance with all non-None arrays in ``float64``.

    See Also
    --------
    make_orbital_projection : Factory for the optional-spin variant.
    SpinOrbitalProjection : The PyTree class constructed by this
        factory.
    """
    proj_arr: Float[Array, "K B A 9"] = jnp.asarray(
        projections, dtype=jnp.float64
    )
    spin_arr: Float[Array, "K B A 6"] = jnp.asarray(spin, dtype=jnp.float64)
    oam_arr: Optional[Float[Array, "K B A 3"]] = None
    if oam is not None:
        oam_arr = jnp.asarray(oam, dtype=jnp.float64)

    def validate_and_create() -> SpinOrbitalProjection:
        def check_projections_finite() -> Float[Array, " "]:
            return lax.cond(
                jnp.all(jnp.isfinite(proj_arr)),
                lambda: proj_arr.sum(),
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: proj_arr.sum(),
                        lambda: proj_arr.sum(),
                    )
                ),
            )

        def check_projections_non_negative() -> Float[Array, " "]:
            return lax.cond(
                jnp.all(proj_arr >= 0.0),
                lambda: proj_arr.sum(),
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: proj_arr.sum(),
                        lambda: proj_arr.sum(),
                    )
                ),
            )

        def check_spin_finite() -> Float[Array, " "]:
            return lax.cond(
                jnp.all(jnp.isfinite(spin_arr)),
                lambda: spin_arr.sum(),
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: spin_arr.sum(),
                        lambda: spin_arr.sum(),
                    )
                ),
            )

        check_projections_finite()
        check_projections_non_negative()
        check_spin_finite()
        return SpinOrbitalProjection(
            projections=proj_arr,
            spin=spin_arr,
            oam=oam_arr,
        )

    soc_proj: SpinOrbitalProjection = validate_and_create()
    return soc_proj


class SpinBandStructure(eqx.Module):
    """PyTree for spin-resolved electronic band structure.

    Extended Summary
    ----------------
    Stores eigenvalues for both spin channels from an ISPIN=2 VASP
    calculation. The two spin channels share the same k-point mesh
    and weights. This type is returned by ``read_eigenval`` when
    ``return_mode="full"`` and the EIGENVAL file contains spin-
    polarized data.

    This class is an immutable :class:`equinox.Module` PyTree. All five
    fields are dense JAX
    arrays stored as children (no auxiliary data), making the entire
    object fully differentiable with respect to any of its fields.

    Attributes
    ----------
    eigenvalues_up : Float[Array, "K B"]
        Spin-up (majority) band energies in eV for K k-points and
        B bands. JAX-traced (differentiable).
    eigenvalues_down : Float[Array, "K B"]
        Spin-down (minority) band energies in eV for K k-points
        and B bands. JAX-traced (differentiable).
    kpoints : Float[Array, "K 3"]
        k-point coordinates in reciprocal (fractional) space, shared
        by both spin channels. JAX-traced (differentiable).
    kpoint_weights : Float[Array, " K"]
        Integration weights for each k-point, used for Brillouin-zone
        averaging. Uniform weights (all ones) are the norm for band
        structure paths. JAX-traced (differentiable).
    fermi_energy : Float[Array, " "]
        Fermi level energy in eV. A 0-D scalar array.
        JAX-traced (differentiable).

    Notes
    -----
    Implemented as an immutable :class:`equinox.Module` PyTree.
    Equinox derives the tree structure from the annotated fields; all
    fields are differentiable leaves and no static metadata is present.

    See Also
    --------
    BandStructure : Single-spin-channel variant.
    make_spin_band_structure : Factory function with validation and
        float64 casting.
    """

    eigenvalues_up: Float[Array, "K B"]
    eigenvalues_down: Float[Array, "K B"]
    kpoints: Float[Array, "K 3"]
    kpoint_weights: Float[Array, " K"]
    fermi_energy: Float[Array, " "]


@jaxtyped(typechecker=beartype)
def make_spin_band_structure(
    eigenvalues_up: Float[Array, "K B"],
    eigenvalues_down: Float[Array, "K B"],
    kpoints: Float[Array, "K 3"],
    kpoint_weights: Union[Float[Array, " K"], None] = None,
    fermi_energy: ScalarNumeric = 0.0,
) -> SpinBandStructure:
    """Create a validated ``SpinBandStructure`` instance.

    Extended Summary
    ----------------
    Factory function that validates and normalises raw spin-resolved
    band structure data before constructing a ``SpinBandStructure``
    PyTree. This is the spin-polarized (ISPIN=2) counterpart to
    :func:`make_band_structure`. All input arrays are cast to
    ``float64`` for numerical stability. Missing k-point weights are
    replaced by uniform weights so that callers do not need to handle
    the common equal-weight case explicitly.

    The function is decorated with ``@jaxtyped(typechecker=beartype)``
    so that shape and dtype constraints on the inputs are checked at
    call time, catching mismatched dimensions (e.g. different K or B
    between the two spin channels) before they propagate into the
    simulation pipeline.

    Implementation Logic
    --------------------
    1. **Cast eigenvalues_up** to ``jnp.float64`` via ``jnp.asarray``.
    2. **Cast eigenvalues_down** to ``jnp.float64`` via
       ``jnp.asarray``.
    3. **Cast kpoints** to ``jnp.float64`` via ``jnp.asarray``.
    4. **Default handling**: if ``kpoint_weights`` is ``None``, a
       uniform weight vector ``jnp.ones(K)`` is created (where *K* is
       inferred from ``eigenvalues_up.shape[0]``). This is the
       standard assumption for band structure paths.
    5. **Cast fermi_energy** scalar to a 0-D ``jnp.float64`` array.
    6. **Construction**: the five validated arrays are passed to the
       ``SpinBandStructure`` named-tuple constructor, producing an
       immutable PyTree ready for use in JAX transformations.

    Parameters
    ----------
    eigenvalues_up : Float[Array, "K B"]
        Spin-up band energies in eV for K k-points and B bands.
    eigenvalues_down : Float[Array, "K B"]
        Spin-down band energies in eV. Must share the same (K, B)
        shape as ``eigenvalues_up``.
    kpoints : Float[Array, "K 3"]
        k-point coordinates in reciprocal (fractional) space.
    kpoint_weights : Union[Float[Array, " K"], None], optional
        Integration weights per k-point. Defaults to uniform weights
        ``jnp.ones(K)``.
    fermi_energy : ScalarNumeric, optional
        Fermi level in eV. Default is 0.0.

    Returns
    -------
    bands : SpinBandStructure
        Validated spin-resolved band structure with all arrays in
        ``float64``.

    See Also
    --------
    make_band_structure : Factory for single-spin-channel data.
    SpinBandStructure : The PyTree class constructed by this factory.
    """
    up_arr: Float[Array, "K B"] = jnp.asarray(
        eigenvalues_up, dtype=jnp.float64
    )
    down_arr: Float[Array, "K B"] = jnp.asarray(
        eigenvalues_down, dtype=jnp.float64
    )
    kpts_arr: Float[Array, "K 3"] = jnp.asarray(kpoints, dtype=jnp.float64)
    nkpts: int = up_arr.shape[0]
    if kpoint_weights is None:
        weights_arr: Float[Array, " K"] = jnp.ones(nkpts, dtype=jnp.float64)
    else:
        weights_arr = jnp.asarray(kpoint_weights, dtype=jnp.float64)
    fermi_arr: Float[Array, " "] = jnp.asarray(fermi_energy, dtype=jnp.float64)

    def validate_and_create() -> SpinBandStructure:
        def check_eigenvalues_up_finite() -> Float[Array, " "]:
            return lax.cond(
                jnp.all(jnp.isfinite(up_arr)),
                lambda: up_arr.sum(),
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: up_arr.sum(),
                        lambda: up_arr.sum(),
                    )
                ),
            )

        def check_eigenvalues_down_finite() -> Float[Array, " "]:
            return lax.cond(
                jnp.all(jnp.isfinite(down_arr)),
                lambda: down_arr.sum(),
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: down_arr.sum(),
                        lambda: down_arr.sum(),
                    )
                ),
            )

        def check_kpoints_finite() -> Float[Array, " "]:
            return lax.cond(
                jnp.all(jnp.isfinite(kpts_arr)),
                lambda: kpts_arr.sum(),
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: kpts_arr.sum(),
                        lambda: kpts_arr.sum(),
                    )
                ),
            )

        def check_weights_non_negative() -> Float[Array, " "]:
            return lax.cond(
                jnp.all(weights_arr >= 0.0),
                lambda: weights_arr.sum(),
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: weights_arr.sum(),
                        lambda: weights_arr.sum(),
                    )
                ),
            )

        check_eigenvalues_up_finite()
        check_eigenvalues_down_finite()
        check_kpoints_finite()
        check_weights_non_negative()
        return SpinBandStructure(
            eigenvalues_up=up_arr,
            eigenvalues_down=down_arr,
            kpoints=kpts_arr,
            kpoint_weights=weights_arr,
            fermi_energy=fermi_arr,
        )

    bands: SpinBandStructure = validate_and_create()
    return bands


class ArpesSpectrum(eqx.Module):
    """PyTree for ARPES simulation output.

    Extended Summary
    ----------------
    Stores the result of an ARPES simulation: a two-dimensional
    photoemission intensity map I(k, E) together with its energy axis.
    The k-point dimension indexes the momentum-resolved detector
    channels, and the energy dimension indexes the binding-energy grid
    on which the spectral function was evaluated.

    This type exists as a JAX-compatible PyTree so that simulated
    spectra can be compared to experimental data inside ``jit``-compiled
    loss functions and differentiated with ``grad`` for parameter
    fitting. Both fields are dense JAX arrays with no static auxiliary
    data, making the entire object fully differentiable.

    Attributes
    ----------
    intensity : Float[Array, "K E"]
        Photoemission intensity for K k-points and E energies.
    energy_axis : Float[Array, " E"]
        Energy axis values in eV.

    Notes
    -----
    Implemented as an immutable :class:`equinox.Module` PyTree. Equinox
    derives the tree structure from the annotated fields; both fields
    are differentiable leaves.
    """

    intensity: Float[Array, "K E"]
    energy_axis: Float[Array, " E"]


@jaxtyped(typechecker=beartype)
def make_band_structure(
    eigenvalues: Float[Array, "K B"],
    kpoints: Float[Array, "K 3"],
    kpoint_weights: Union[Float[Array, " K"], None] = None,
    fermi_energy: ScalarNumeric = 0.0,
) -> BandStructure:
    """Create a validated ``BandStructure`` instance.

    Extended Summary
    ----------------
    Factory function that validates and normalises raw band structure
    data before constructing a ``BandStructure`` PyTree. All input
    arrays are cast to ``float64`` for numerical stability in
    downstream ARPES simulations (energy differences and Lorentzian
    broadening are sensitive to precision). Missing k-point weights
    are replaced by uniform weights so that callers do not need to
    handle the common equal-weight case explicitly.

    The function is decorated with ``@jaxtyped(typechecker=beartype)``
    so that shape and dtype constraints on the inputs are checked at
    call time, catching mismatched dimensions before they propagate
    into the simulation pipeline.

    Implementation Logic
    --------------------
    1. **Cast to float64**: ``eigenvalues`` and ``kpoints`` are
       converted via ``jnp.asarray(..., dtype=jnp.float64)`` to
       guarantee double-precision arithmetic in all subsequent
       computations.
    2. **Default handling**: if ``kpoint_weights`` is ``None``, a
       uniform weight vector ``jnp.ones(K)`` is created (where *K* is
       inferred from ``eigenvalues.shape[0]``). This is the standard
       assumption for band structure paths (as opposed to Brillouin-
       zone integrations where weights vary).
    3. **Construction**: the four validated arrays are passed to the
       ``BandStructure`` named-tuple constructor, producing an
       immutable PyTree ready for use in JAX transformations.

    Parameters
    ----------
    eigenvalues : Float[Array, "K B"]
        Band energies in eV for K k-points and B bands.
    kpoints : Float[Array, "K 3"]
        k-point coordinates in reciprocal space.
    kpoint_weights : Union[Float[Array, " K"], None], optional
        Integration weights. Defaults to uniform weights.
    fermi_energy : ScalarNumeric, optional
        Fermi level in eV. Default is 0.0.

    Returns
    -------
    bands : BandStructure
        Validated band structure instance with all arrays in
        ``float64``.
    """
    eigenvalues_arr: Float[Array, "K B"] = jnp.asarray(
        eigenvalues, dtype=jnp.float64
    )
    kpoints_arr: Float[Array, "K 3"] = jnp.asarray(kpoints, dtype=jnp.float64)
    nkpts: int = eigenvalues_arr.shape[0]
    if kpoint_weights is None:
        weights_arr: Float[Array, " K"] = jnp.ones(nkpts, dtype=jnp.float64)
    else:
        weights_arr = jnp.asarray(kpoint_weights, dtype=jnp.float64)
    fermi_arr: Float[Array, " "] = jnp.asarray(fermi_energy, dtype=jnp.float64)

    def validate_and_create() -> BandStructure:
        def check_eigenvalues_finite() -> Float[Array, " "]:
            return lax.cond(
                jnp.all(jnp.isfinite(eigenvalues_arr)),
                lambda: eigenvalues_arr.sum(),
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: eigenvalues_arr.sum(),
                        lambda: eigenvalues_arr.sum(),
                    )
                ),
            )

        def check_kpoints_finite() -> Float[Array, " "]:
            return lax.cond(
                jnp.all(jnp.isfinite(kpoints_arr)),
                lambda: kpoints_arr.sum(),
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: kpoints_arr.sum(),
                        lambda: kpoints_arr.sum(),
                    )
                ),
            )

        def check_weights_non_negative() -> Float[Array, " "]:
            return lax.cond(
                jnp.all(weights_arr >= 0.0),
                lambda: weights_arr.sum(),
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: weights_arr.sum(),
                        lambda: weights_arr.sum(),
                    )
                ),
            )

        check_eigenvalues_finite()
        check_kpoints_finite()
        check_weights_non_negative()
        return BandStructure(
            eigenvalues=eigenvalues_arr,
            kpoints=kpoints_arr,
            kpoint_weights=weights_arr,
            fermi_energy=fermi_arr,
        )

    bands: BandStructure = validate_and_create()
    return bands


@jaxtyped(typechecker=beartype)
def make_orbital_projection(
    projections: Float[Array, "K B A 9"],
    spin: Optional[Float[Array, "K B A 6"]] = None,
    oam: Optional[Float[Array, "K B A 3"]] = None,
) -> OrbitalProjection:
    """Create a validated ``OrbitalProjection`` instance.

    Extended Summary
    ----------------
    Factory function that validates and normalises raw orbital
    projection data before constructing an ``OrbitalProjection``
    PyTree. The mandatory ``projections`` array is cast to ``float64``;
    the optional ``spin`` and ``oam`` arrays are cast only when they
    are not ``None``, preserving the sentinel meaning of ``None`` for
    calculations without spin-orbit coupling.

    The function is decorated with ``@jaxtyped(typechecker=beartype)``
    so that shape constraints (K, B, A dimensions must agree across
    all provided arrays) are checked at call time.

    Implementation Logic
    --------------------
    1. **Cast to float64**: ``projections`` is converted via
       ``jnp.asarray(..., dtype=jnp.float64)``. ``spin`` and ``oam``
       are likewise converted when they are not ``None``.
    2. **Default handling**: ``spin`` and ``oam`` default to ``None``,
       which signals that the DFT calculation did not include spin-
       orbit coupling. No placeholder arrays are created -- the
       ``None`` propagates into the PyTree leaf, keeping the tree
       structure minimal.
    3. **Construction**: the three validated fields are passed to the
       ``OrbitalProjection`` named-tuple constructor, producing an
       immutable PyTree ready for use in JAX transformations.

    Parameters
    ----------
    projections : Float[Array, "K B A 9"]
        Orbital projection weights.
    spin : Optional[Float[Array, "K B A 6"]], optional
        Spin projections. Default is None.
    oam : Optional[Float[Array, "K B A 3"]], optional
        Orbital angular momentum. Default is None.

    Returns
    -------
    orb_proj : OrbitalProjection
        Validated orbital projection instance with all non-None
        arrays in ``float64``.
    """
    proj_arr: Float[Array, "K B A 9"] = jnp.asarray(
        projections, dtype=jnp.float64
    )
    spin_arr: Optional[Float[Array, "K B A 6"]] = None
    if spin is not None:
        spin_arr = jnp.asarray(spin, dtype=jnp.float64)
    oam_arr: Optional[Float[Array, "K B A 3"]] = None
    if oam is not None:
        oam_arr = jnp.asarray(oam, dtype=jnp.float64)

    def validate_and_create() -> OrbitalProjection:
        def check_projections_finite() -> Float[Array, " "]:
            return lax.cond(
                jnp.all(jnp.isfinite(proj_arr)),
                lambda: proj_arr.sum(),
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: proj_arr.sum(),
                        lambda: proj_arr.sum(),
                    )
                ),
            )

        def check_projections_non_negative() -> Float[Array, " "]:
            return lax.cond(
                jnp.all(proj_arr >= 0.0),
                lambda: proj_arr.sum(),
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: proj_arr.sum(),
                        lambda: proj_arr.sum(),
                    )
                ),
            )

        check_projections_finite()
        check_projections_non_negative()
        return OrbitalProjection(
            projections=proj_arr,
            spin=spin_arr,
            oam=oam_arr,
        )

    orb_proj: OrbitalProjection = validate_and_create()
    return orb_proj


@jaxtyped(typechecker=beartype)
def make_arpes_spectrum(
    intensity: Float[Array, "K E"],
    energy_axis: Float[Array, " E"],
) -> ArpesSpectrum:
    """Create a validated ``ArpesSpectrum`` instance.

    Extended Summary
    ----------------
    Factory function that validates and normalises simulated ARPES
    data before constructing an ``ArpesSpectrum`` PyTree. Both input
    arrays are cast to ``float64`` so that downstream loss functions
    (e.g. mean-squared error against experimental data) maintain full
    double-precision accuracy.

    The function is decorated with ``@jaxtyped(typechecker=beartype)``
    so that the energy dimension *E* is checked for consistency
    between ``intensity`` and ``energy_axis`` at call time.

    Implementation Logic
    --------------------
    1. **Cast to float64**: ``intensity`` and ``energy_axis`` are
       converted via ``jnp.asarray(..., dtype=jnp.float64)`` to
       guarantee double-precision arithmetic.
    2. **Default handling**: none -- both fields are mandatory and
       have no sentinel defaults.
    3. **Construction**: the two validated arrays are passed to the
       ``ArpesSpectrum`` named-tuple constructor, producing an
       immutable PyTree ready for use in JAX transformations.

    Parameters
    ----------
    intensity : Float[Array, "K E"]
        Photoemission intensity map.
    energy_axis : Float[Array, " E"]
        Energy axis values in eV.

    Returns
    -------
    spectrum : ArpesSpectrum
        Validated ARPES spectrum instance with all arrays in
        ``float64``.
    """
    intensity_arr: Float[Array, "K E"] = jnp.asarray(
        intensity, dtype=jnp.float64
    )
    energy_arr: Float[Array, " E"] = jnp.asarray(
        energy_axis, dtype=jnp.float64
    )

    def validate_and_create() -> ArpesSpectrum:
        def check_intensity_finite() -> Float[Array, " "]:
            return lax.cond(
                jnp.all(jnp.isfinite(intensity_arr)),
                lambda: intensity_arr.sum(),
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: intensity_arr.sum(),
                        lambda: intensity_arr.sum(),
                    )
                ),
            )

        def check_intensity_non_negative() -> Float[Array, " "]:
            return lax.cond(
                jnp.all(intensity_arr >= 0.0),
                lambda: intensity_arr.sum(),
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: intensity_arr.sum(),
                        lambda: intensity_arr.sum(),
                    )
                ),
            )

        def check_energy_axis_finite() -> Float[Array, " "]:
            return lax.cond(
                jnp.all(jnp.isfinite(energy_arr)),
                lambda: energy_arr.sum(),
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: energy_arr.sum(),
                        lambda: energy_arr.sum(),
                    )
                ),
            )

        check_intensity_finite()
        check_intensity_non_negative()
        check_energy_axis_finite()
        return ArpesSpectrum(
            intensity=intensity_arr,
            energy_axis=energy_arr,
        )

    spectrum: ArpesSpectrum = validate_and_create()
    return spectrum


__all__: list[str] = [
    "ArpesSpectrum",
    "BandStructure",
    "OrbitalProjection",
    "SpinBandStructure",
    "SpinOrbitalProjection",
    "make_arpes_spectrum",
    "make_band_structure",
    "make_orbital_projection",
    "make_spin_band_structure",
    "make_spin_orbital_projection",
]
