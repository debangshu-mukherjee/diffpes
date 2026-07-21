"""Compute photon polarization and detector-frame transformations.

Extended Summary
----------------
The module computes complex polarization vectors from photon geometry. It
also converts polarization to the spherical basis and rotates vectors through
the detector frame. Legacy orbital weights remain until plan 06 replaces
their only simulation consumers.

Routine Listings
----------------
:func:`build_efield`
    Compute electric field vector from polarization config.
:func:`build_polarization_vectors`
    Construct s- and p-polarization basis vectors.
:func:`detector_rotation`
    Build the detector-frame rotation.
:func:`dipole_matrix_elements`
    Compute dipole matrix elements for all 9 orbitals.
:func:`photon_wavevector`
    Build the unit photon wavevector from incidence angles.
:func:`polarization_from_angles`
    Construct polarization from incidence angles.
:func:`polarization_to_spherical`
    Convert Cartesian polarization to spherical components.
:func:`rotate_frame_vectors`
    Rotate a real vector across a detector-angle grid.
:func:`rotate_polarization_grid`
    Rotate polarization across a detector-angle grid.

Notes
-----
Orbital direction vectors follow VASP orbital ordering:
[s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2].
The s-orbital has zero directionality.

The horizontal detector frame uses ``Rx(ty) @ Ry(tx)``. DiffPES maps the
Chinook ``tilt.k_mesh`` angles as ``T=-tx, P=ty``. It maps the Chinook
``gen_all_pol`` angles as ``theta=-tx, phi=-ty``. The vertical frame uses
``Rx(tx) @ Ry(ty)``. Its mappings are ``T=-ty, P=tx`` and
``theta=-ty, phi=-tx``.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Complex, Float, jaxtyped

from diffpes.maths import rodrigues_rotation
from diffpes.types import (
    ORBITAL_DIRS_NORMALIZED,
    PolarizationConfig,
    ScalarFloat,
)


@jaxtyped(typechecker=beartype)
def build_polarization_vectors(
    theta: ScalarFloat,
    phi: ScalarFloat,
) -> Tuple[Float[Array, " 3"], Float[Array, " 3"]]:
    """Construct s- and p-polarization basis vectors.

    The function constructs an orthonormal pair of polarization vectors from
    the photon incidence angles. The s-polarization is perpendicular to the
    incidence plane. The p-polarization is in the incidence plane and
    perpendicular to the wavevector.

    :see: :class:`~.test_polarization.TestBuildPolarizationVectors`

    Implementation Logic
    --------------------
    1. **Construct the s-polarization vector**::

           e_s = [sin(phi), -cos(phi), 0]

       This closed form is perpendicular to the incidence plane. It is the
       normalized ``k cross z`` convention continued to normal incidence.

    2. **Construct the p-polarization vector**::

           e_p = [-cos(theta) cos(phi),
                  -cos(theta) sin(phi),
                   sin(theta)]

       This vector equals ``e_s cross k``. It is perpendicular to the photon
       direction and completes the orthonormal transverse basis.

    Parameters
    ----------
    theta : ScalarFloat
        Incident angle from surface normal in radians.
    phi : ScalarFloat
        In-plane azimuthal angle in radians.

    Returns
    -------
    e_s : Float[Array, " 3"]
        s-polarization unit vector (perpendicular to
        incidence plane).
    e_p : Float[Array, " 3"]
        p-polarization unit vector (in incidence plane,
        perpendicular to photon wavevector).

    Notes
    -----
    The direct trigonometric form has no artificial collinearity threshold.
    At normal incidence, ``phi`` fixes the otherwise free transverse-frame
    gauge. The basis is smooth in both input angles for that gauge choice.
    """
    e_s: Float[Array, " 3"] = jnp.array(
        [
            jnp.sin(phi),
            -jnp.cos(phi),
            jnp.zeros_like(jnp.asarray(phi)),
        ],
        dtype=jnp.float64,
    )
    e_p: Float[Array, " 3"] = jnp.array(
        [
            -jnp.cos(theta) * jnp.cos(phi),
            -jnp.cos(theta) * jnp.sin(phi),
            jnp.sin(theta),
        ],
        dtype=jnp.float64,
    )
    polarization_vectors: Tuple[Float[Array, " 3"], Float[Array, " 3"]] = (
        e_s,
        e_p,
    )
    return polarization_vectors


@jaxtyped(typechecker=beartype)
def photon_wavevector(
    theta: ScalarFloat,
    phi: ScalarFloat,
) -> Float[Array, " 3"]:
    """Build the unit photon wavevector from incidence angles.

    The function constructs the unit photon propagation vector from spherical
    coordinates. Theta starts at the surface normal, and phi is the azimuthal
    angle. Spin-orbit ARPES simulations use the vector in the S·k_photon
    correction for circular dichroism.

    :see: :class:`~.test_polarization.TestPhotonWavevector`

    Notes
    -----
    Form the Cartesian spherical-coordinate vector, normalize it with its
    Euclidean norm, bind the unit result to ``k_hat``, and return it.

    Parameters
    ----------
    theta : ScalarFloat
        Incident angle from surface normal in radians.
    phi : ScalarFloat
        In-plane azimuthal angle in radians.

    Returns
    -------
    k_photon : Float[Array, " 3"]
        Unit wavevector in Cartesian coordinates.

    See Also
    --------
    build_polarization_vectors : Build the same k for the s-polarization and
        p-polarization basis. Use this function only for the propagation
        direction.
    """
    k: Float[Array, " 3"] = jnp.array(
        [
            jnp.sin(theta) * jnp.cos(phi),
            jnp.sin(theta) * jnp.sin(phi),
            jnp.cos(theta),
        ],
        dtype=jnp.float64,
    )
    k_hat: Float[Array, " 3"] = k / jnp.linalg.norm(k)
    return k_hat


@jaxtyped(typechecker=beartype)
def polarization_from_angles(
    incidence_theta: ScalarFloat,
    incidence_phi: ScalarFloat,
    kind: str,
    polarization_angle: ScalarFloat = 0.0,
) -> Complex[Array, " 3"]:
    """Construct polarization from incidence angles.

    The function returns an explicit complex Cartesian vector for a standard
    polarization state. The incidence angles use the laboratory frame.

    :see: :class:`~.test_polarization.TestPolarizationFromAngles`

    Implementation Logic
    --------------------
    1. **Construct the transverse basis**::

           e_s, e_p = build_polarization_vectors(theta, phi)

       The basis is orthonormal and perpendicular to the photon direction.

    2. **Select the requested state**::

           polarization = coefficients[0] * e_s + coefficients[1] * e_p

       The static selector chooses s, p, circular, or linear coefficients.

    3. **Return the complex vector**::

           return polarization

       The result retains phase information for later coherent contraction.

    Parameters
    ----------
    incidence_theta : ScalarFloat
        Photon angle from the surface normal in radians.
    incidence_phi : ScalarFloat
        Photon azimuth in radians.
    kind : str
        Polarization kind (**static**). Use ``"s"``, ``"p"``, ``"c+"``,
        ``"c-"``, or ``"linear"``.
    polarization_angle : ScalarFloat, optional
        Linear-basis angle in radians. Default is 0.0.

    Returns
    -------
    polarization : Complex[Array, " 3"]
        Unit polarization vector in the laboratory frame.

    Raises
    ------
    ValueError
        If ``kind`` is not a supported polarization kind.

    Notes
    -----
    The ``kind`` value is static and selects a Python branch before tracing.
    JAX differentiates the result with respect to all angle arguments.

    See Also
    --------
    build_polarization_vectors : Construct the transverse real basis.
    polarization_to_spherical : Convert the result to spherical components.
    """
    if kind not in ("s", "p", "c+", "c-", "linear"):
        msg: str = (
            "polarization_from_angles: kind must be one of "
            "('s', 'p', 'c+', 'c-', 'linear')"
        )
        raise ValueError(msg)

    e_s: Float[Array, " 3"]
    e_p: Float[Array, " 3"]
    e_s, e_p = build_polarization_vectors(incidence_theta, incidence_phi)
    e_s_complex: Complex[Array, " 3"] = e_s.astype(jnp.complex128)
    e_p_complex: Complex[Array, " 3"] = e_p.astype(jnp.complex128)
    if kind == "s":
        polarization: Complex[Array, " 3"] = e_s_complex
    elif kind == "p":
        polarization = e_p_complex
    elif kind == "c+":
        polarization = (e_s_complex + 1j * e_p_complex) / jnp.sqrt(2.0)
    elif kind == "c-":
        polarization = (e_s_complex - 1j * e_p_complex) / jnp.sqrt(2.0)
    else:
        angle: Float[Array, " "] = jnp.asarray(
            polarization_angle,
            dtype=jnp.float64,
        )
        polarization = (
            jnp.cos(angle) * e_s_complex + jnp.sin(angle) * e_p_complex
        )
    return polarization


@jaxtyped(typechecker=beartype)
def polarization_to_spherical(
    polarization: Complex[Array, " 3"],
) -> Complex[Array, " 3"]:
    """Convert Cartesian polarization to spherical components.

    The result uses component order ``(q=-1, q=0, q=+1)`` and the
    Condon-Shortley phase convention.

    :see: :class:`~.test_polarization.TestPolarizationToSpherical`

    Implementation Logic
    --------------------
    1. **Read Cartesian components**::

           epsilon_x, epsilon_y, epsilon_z = polarization

       The input remains complex so the operation preserves optical phase.

    2. **Apply the spherical-basis transform**::

           epsilon_minus = (epsilon_x - 1j * epsilon_y) / sqrt(2)

       The transform follows the registered Condon-Shortley convention.

    3. **Stack the ordered result**::

           spherical = stack((epsilon_minus, epsilon_z, epsilon_plus))

       This order matches the transitions ``q = (-1, 0, +1)``.

    Parameters
    ----------
    polarization : Complex[Array, " 3"]
        Cartesian polarization vector.

    Returns
    -------
    spherical : Complex[Array, " 3"]
        Spherical components in ``(q=-1, q=0, q=+1)`` order.

    Notes
    -----
    The transform is complex-linear. It preserves the squared vector norm and
    supports JVP, VJP, and complex-step checks without a conjugation.
    """
    epsilon_x: Complex[Array, " "] = polarization[0]
    epsilon_y: Complex[Array, " "] = polarization[1]
    epsilon_z: Complex[Array, " "] = polarization[2]
    root_two: Float[Array, " "] = jnp.sqrt(jnp.asarray(2.0, dtype=jnp.float64))
    epsilon_minus: Complex[Array, " "] = (
        epsilon_x - 1j * epsilon_y
    ) / root_two
    epsilon_plus: Complex[Array, " "] = (
        -(epsilon_x + 1j * epsilon_y) / root_two
    )
    spherical: Complex[Array, " 3"] = jnp.stack(
        (epsilon_minus, epsilon_z, epsilon_plus)
    )
    return spherical


@jaxtyped(typechecker=beartype)
def detector_rotation(
    tx: ScalarFloat,
    ty: ScalarFloat,
    slit: str,
) -> Float[Array, "3 3"]:
    """Build the detector-frame rotation.

    The function composes the two analyzer-angle rotations in the order set
    by the static slit orientation.

    :see: :class:`~.test_polarization.TestDetectorRotation`

    Implementation Logic
    --------------------
    1. **Build the Cartesian axis rotations**::

           rotation_x = rodrigues_rotation(x_axis, ty)

       Rodrigues matrices retain derivatives with respect to both angles.

    2. **Compose the slit convention**::

           horizontal = rotation_x_ty @ rotation_y_tx
           vertical = rotation_x_tx @ rotation_y_ty

       Horizontal and vertical slits use the registered composition orders.

    Parameters
    ----------
    tx : ScalarFloat
        First detector angle in radians.
    ty : ScalarFloat
        Second detector angle in radians.
    slit : str
        Slit orientation (**static**). Use ``"H"`` or ``"V"``.

    Returns
    -------
    rotation : Float[Array, "3 3"]
        Proper rotation from the reference detector frame.

    Raises
    ------
    ValueError
        If ``slit`` is not ``"H"`` or ``"V"``.

    Notes
    -----
    ``"H"`` uses ``R_x(ty) R_y(tx)``. ``"V"`` uses
    ``R_x(tx) R_y(ty)``. The same matrix rotates the emitted direction,
    polarization, and spin axis.

    Chinook uses opposite raw signs for its horizontal Ty momentum and
    polarization coordinates. The declared source-coordinate mappings give
    one active DiffPES frame.
    """
    if slit not in ("H", "V"):
        msg: str = "detector_rotation: slit must be 'H' or 'V'"
        raise ValueError(msg)
    x_axis: Float[Array, " 3"] = jnp.asarray(
        [1.0, 0.0, 0.0],
        dtype=jnp.float64,
    )
    y_axis: Float[Array, " 3"] = jnp.asarray(
        [0.0, 1.0, 0.0],
        dtype=jnp.float64,
    )
    if slit == "H":
        rotation_y_tx: Float[Array, "3 3"] = rodrigues_rotation(y_axis, tx)
        rotation_x_ty: Float[Array, "3 3"] = rodrigues_rotation(x_axis, ty)
        rotation: Float[Array, "3 3"] = rotation_x_ty @ rotation_y_tx
    else:
        rotation_x_tx: Float[Array, "3 3"] = rodrigues_rotation(x_axis, tx)
        rotation_y_ty: Float[Array, "3 3"] = rodrigues_rotation(y_axis, ty)
        rotation = rotation_x_tx @ rotation_y_ty
    return rotation


@jaxtyped(typechecker=beartype)
def rotate_frame_vectors(
    vector: Float[Array, " 3"],
    tx: Float[Array, " n_tx"],
    ty: Float[Array, " n_ty"],
    slit: str,
) -> Float[Array, "n_tx n_ty 3"]:
    """Rotate a real vector across a detector-angle grid.

    The function applies each detector-frame rotation to one real laboratory
    vector. It preserves both detector axes in the output.

    :see: :class:`~.test_polarization.TestRotateFrameVectors`

    Implementation Logic
    --------------------
    1. **Map over both angle axes**::

           rotations = vmap(vmap(detector_rotation))(tx, ty)

       Nested mapping builds one rotation for every detector coordinate.

    2. **Apply each rotation**::

           rotated = rotations @ vector

       Matrix multiplication preserves the vector norm.

    Parameters
    ----------
    vector : Float[Array, " 3"]
        Real vector in the reference laboratory frame.
    tx : Float[Array, " n_tx"]
        First detector-angle axis in radians.
    ty : Float[Array, " n_ty"]
        Second detector-angle axis in radians.
    slit : str
        Slit orientation (**static**). Use ``"H"`` or ``"V"``.

    Returns
    -------
    rotated : Float[Array, "n_tx n_ty 3"]
        Rotated vector at each detector coordinate.

    Notes
    -----
    The output has fixed shape for fixed angle-axis lengths. JAX can compile
    and differentiate the two mapped angle axes without Python data loops.
    """

    def rotate_one_tx(
        tx_value: Float[Array, " "],
    ) -> Float[Array, "n_ty 3"]:
        """Rotate one vector across the second angle axis.

        Parameters
        ----------
        tx_value : Float[Array, " "]
            Fixed first detector angle in radians.

        Returns
        -------
        rotated_row : Float[Array, "n_ty 3"]
            Rotated vectors for the second angle axis.
        """

        def rotate_one_ty(
            ty_value: Float[Array, " "],
        ) -> Float[Array, " 3"]:
            """Rotate one vector at one detector coordinate.

            Parameters
            ----------
            ty_value : Float[Array, " "]
                Second detector angle in radians.

            Returns
            -------
            rotated_vector : Float[Array, " 3"]
                Rotated vector at the detector coordinate.
            """
            rotation: Float[Array, "3 3"] = detector_rotation(
                tx_value,
                ty_value,
                slit,
            )
            rotated_vector: Float[Array, " 3"] = rotation @ vector
            return rotated_vector

        rotated_row: Float[Array, "n_ty 3"] = jax.vmap(rotate_one_ty)(ty)
        return rotated_row

    rotated: Float[Array, "n_tx n_ty 3"] = jax.vmap(rotate_one_tx)(tx)
    return rotated


@jaxtyped(typechecker=beartype)
def rotate_polarization_grid(
    polarization: Complex[Array, " 3"],
    tx: Float[Array, " n_tx"],
    ty: Float[Array, " n_ty"],
    slit: str,
) -> Complex[Array, "n_tx n_ty 3"]:
    """Rotate polarization across a detector-angle grid.

    The function applies the shared detector frame to a complex polarization
    vector without reducing its phase or amplitude.

    :see: :class:`~.test_polarization.TestRotatePolarizationGrid`

    Implementation Logic
    --------------------
    1. **Map over both angle axes**::

           rotated = vmap(vmap(rotate_one))(tx, ty)

       Nested mapping applies the same frame convention at every coordinate.

    2. **Return the complex vectors**::

           return rotated

       The result retains coherent complex components for later models.

    Parameters
    ----------
    polarization : Complex[Array, " 3"]
        Complex polarization in the reference laboratory frame.
    tx : Float[Array, " n_tx"]
        First detector-angle axis in radians.
    ty : Float[Array, " n_ty"]
        Second detector-angle axis in radians.
    slit : str
        Slit orientation (**static**). Use ``"H"`` or ``"V"``.

    Returns
    -------
    rotated : Complex[Array, "n_tx n_ty 3"]
        Rotated polarization at each detector coordinate.

    Notes
    -----
    Real rotation matrices act on the complex vector components. Therefore,
    the map is complex-linear in ``polarization`` and differentiable in both
    detector-angle axes.
    """

    def rotate_one_tx(
        tx_value: Float[Array, " "],
    ) -> Complex[Array, "n_ty 3"]:
        """Rotate polarization across the second angle axis.

        Parameters
        ----------
        tx_value : Float[Array, " "]
            Fixed first detector angle in radians.

        Returns
        -------
        rotated_row : Complex[Array, "n_ty 3"]
            Rotated polarization vectors for the second angle axis.
        """

        def rotate_one_ty(
            ty_value: Float[Array, " "],
        ) -> Complex[Array, " 3"]:
            """Rotate polarization at one detector coordinate.

            Parameters
            ----------
            ty_value : Float[Array, " "]
                Second detector angle in radians.

            Returns
            -------
            rotated_vector : Complex[Array, " 3"]
                Rotated polarization at the detector coordinate.
            """
            rotation: Float[Array, "3 3"] = detector_rotation(
                tx_value,
                ty_value,
                slit,
            )
            rotated_vector: Complex[Array, " 3"] = rotation @ polarization
            return rotated_vector

        rotated_row: Complex[Array, "n_ty 3"] = jax.vmap(rotate_one_ty)(ty)
        return rotated_row

    rotated: Complex[Array, "n_tx n_ty 3"] = jax.vmap(rotate_one_tx)(tx)
    return rotated


@jaxtyped(typechecker=beartype)
def build_efield(
    config: PolarizationConfig,
) -> Complex[Array, " 3"]:
    """Compute electric field vector from polarization config.

    Constructs the complex electric field polarization vector for the
    specified photon geometry and polarization type.

    :see: :class:`~.test_polarization.TestBuildEfield`

    Implementation Logic
    --------------------
    1. **Build s- and p-polarization basis**::

           e_s, e_p = build_polarization_vectors(
               config.theta, config.phi
           )

       This computes the real orthonormal basis from the incidence angles.

    2. **Select the static polarization branch**::

           index: int = pol_index_map.get(pol_type, 5)

       The string configuration selects one branch before JAX execution.

    3. **Evaluate the selected branch with JAX**::

           efield: Complex[Array, " 3"] = jax.lax.switch(
               index, branches, operand
           )

       The JAX switch preserves differentiation through the selected field.

    Parameters
    ----------
    config : PolarizationConfig
        Polarization geometry specification.

    Returns
    -------
    efield : Complex[Array, " 3"]
        Complex electric field polarization vector.
    """
    e_s: Float[Array, " 3"]
    e_p: Float[Array, " 3"]
    e_s, e_p = build_polarization_vectors(config.theta, config.phi)
    e_s_c: Complex[Array, " 3"] = e_s.astype(jnp.complex128)
    e_p_c: Complex[Array, " 3"] = e_p.astype(jnp.complex128)
    pol_type: str = config.polarization_type.lower()
    angle: Float[Array, " "] = jnp.asarray(
        config.polarization_angle, dtype=jnp.float64
    )
    operand: Tuple[
        Complex[Array, " 3"],
        Complex[Array, " 3"],
        Float[Array, " "],
    ] = (e_s_c, e_p_c, angle)

    def branch_lvp(op: Tuple) -> Complex[Array, " 3"]:
        """Return the s-polarization basis for linear vertical polarization.

        Parameters
        ----------
        op : Tuple
            Operand (e_s_c, e_p_c, angle). The branch uses only the first
            element.

        Returns
        -------
        Complex[Array, " 3"]
            The s-polarization complex electric field vector.
        """
        branch_efield: Complex[Array, " 3"] = op[0]
        return branch_efield

    def branch_lhp(op: Tuple) -> Complex[Array, " 3"]:
        """Return the p-polarization basis for linear horizontal polarization.

        Parameters
        ----------
        op : Tuple
            Operand (e_s_c, e_p_c, angle). The branch uses only the second
            element.

        Returns
        -------
        Complex[Array, " 3"]
            The p-polarization complex electric field vector.
        """
        branch_efield: Complex[Array, " 3"] = op[1]
        return branch_efield

    def branch_lap(op: Tuple) -> Complex[Array, " 3"]:
        """Return the basis for arbitrary linear polarization.

        Parameters
        ----------
        op : Tuple
            Operand (e_s_c, e_p_c, angle). op[2] is the polarization
            angle in radians.

        Returns
        -------
        Complex[Array, " 3"]
            The linear combination of s- and p-polarization vectors.
        """
        branch_efield: Complex[Array, " 3"] = (
            jnp.cos(op[2]) * op[0] + jnp.sin(op[2]) * op[1]
        )
        return branch_efield

    def branch_rcp(op: Tuple) -> Complex[Array, " 3"]:
        """Return the right circular polarization vector.

        Parameters
        ----------
        op : Tuple
            Operand (e_s_c, e_p_c, angle). The branch uses the first two
            elements.

        Returns
        -------
        Complex[Array, " 3"]
            Right-handed circular polarization electric field.
        """
        branch_efield: Complex[Array, " 3"] = (op[0] + 1j * op[1]) / jnp.sqrt(
            2.0
        )
        return branch_efield

    def branch_lcp(op: Tuple) -> Complex[Array, " 3"]:
        """Return the left circular polarization vector.

        Parameters
        ----------
        op : Tuple
            Operand (e_s_c, e_p_c, angle). The branch uses the first two
            elements.

        Returns
        -------
        Complex[Array, " 3"]
            Left-handed circular polarization electric field.
        """
        branch_efield: Complex[Array, " 3"] = (op[0] - 1j * op[1]) / jnp.sqrt(
            2.0
        )
        return branch_efield

    def branch_default(op: Tuple) -> Complex[Array, " 3"]:
        """Return s-polarization for an unknown or unpolarized type.

        Parameters
        ----------
        op : Tuple
            Operand (e_s_c, e_p_c, angle). The branch uses only the first
            element.

        Returns
        -------
        Complex[Array, " 3"]
            The s-polarization vector as the default.
        """
        branch_efield: Complex[Array, " 3"] = op[0]
        return branch_efield

    pol_index_map: dict[str, int] = {
        "lvp": 0,
        "lhp": 1,
        "lap": 2,
        "rcp": 3,
        "lcp": 4,
    }
    index: int = pol_index_map.get(pol_type, 5)
    branches: Tuple = (
        branch_lvp,
        branch_lhp,
        branch_lap,
        branch_rcp,
        branch_lcp,
        branch_default,
    )
    efield: Complex[Array, " 3"] = jax.lax.switch(index, branches, operand)
    return efield


@jaxtyped(typechecker=beartype)
def dipole_matrix_elements(
    efield: Complex[Array, " 3"],
) -> Float[Array, " 9"]:
    """Compute dipole matrix elements for all 9 orbitals.

    Evaluates the squared modulus of the dipole transition matrix
    element for each orbital::

        M_i = |e . d_i|^2

    where e is the electric field polarization vector and d_i is
    the normalized direction vector of orbital i.

    :see: :class:`~.test_polarization.TestDipoleMatrixElements`

    Implementation Logic
    --------------------
    1. **Dot product of E-field with each orbital direction**::

           dots = ORBITAL_DIRS_NORMALIZED @ efield

       - Computes the inner product of the complex electric field
         vector with each of the 9 normalized orbital direction
         vectors (shape [9, 3] @ [3] -> [9]). The result is a
         complex-valued array of length 9.
       - The s-orbital direction vector is [0, 0, 0] (zero vector),
         so its dot product is always zero regardless of the
         E-field, reflecting the isotropic (zero directionality)
         character of the s-orbital.

    2. **Square modulus** ``|e . d|^2``::

           matrix_elements = |dots|^2

       Takes the absolute value squared of each complex dot
       product. For real-valued E-fields this reduces to the
       squared real dot product. For circular polarization
       (complex E-field) this correctly accounts for the phase.

    Parameters
    ----------
    efield : Complex[Array, " 3"]
        Electric field polarization vector.

    Returns
    -------
    matrix_elements : Float[Array, " 9"]
        ``|e dot d_orbital|^2`` for each orbital.

    Notes
    -----
    The s-orbital has a zero direction vector and therefore always
    produces a zero dipole matrix element with any polarization.
    """
    dots: Complex[Array, " 9"] = jnp.dot(
        ORBITAL_DIRS_NORMALIZED,
        efield,
    )
    matrix_elements: Float[Array, " 9"] = jnp.abs(dots) ** 2
    return matrix_elements


__all__: list[str] = [
    "build_efield",
    "build_polarization_vectors",
    "detector_rotation",
    "dipole_matrix_elements",
    "photon_wavevector",
    "polarization_from_angles",
    "polarization_to_spherical",
    "rotate_frame_vectors",
    "rotate_polarization_grid",
]
