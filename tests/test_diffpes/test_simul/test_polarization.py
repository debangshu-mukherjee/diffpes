"""Validate ARPES polarization and dipole matrix element functions.

Extended Summary
----------------
Exercise build_polarization_vectors, build_efield, and
dipole_matrix_elements. Verify the polarization basis and each field
mode. Check dipole matrix element shape and sign.

"""

import chex
import jax.numpy as jnp
from beartype.typing import Any, Callable
from jaxtyping import Array

import diffpes
from diffpes.simul import (
    build_efield,
    build_polarization_vectors,
    dipole_matrix_elements,
    photon_wavevector,
)
from diffpes.types import make_polarization_config


class TestBuildPolarizationVectors(chex.TestCase):
    """Validate :func:`diffpes.simul.polarization.build_polarization_vectors`.

    Verifies the geometric properties of the s- and p-polarization basis
    vectors, including mutual orthogonality, unit norm, and correct output
    shape for various incidence angle combinations.

    :see: :func:`~diffpes.simul.build_polarization_vectors`
    """

    def test_orthogonality(self) -> None:
        """Verify that e_s and e_p are mutually orthogonal.

        The test establishes the orthogonality contract for build polarization vectors
        with the concrete values and array shapes described below.

        Notes
        -----
        1. **Build polarization vectors**:
           Calls ``build_polarization_vectors`` with theta=pi/4, phi=0
           to produce s- and p-polarization unit vectors.

        2. **Compute dot product**:
           Takes the inner product of e_s and e_p.

        **Expected assertions**

        The dot product of e_s and e_p is zero (within tolerance 1e-10),
        confirming the two vectors are perpendicular.
        """
        theta: Array
        phi: float
        e_s: Array
        e_p: Array
        dot_product: Array

        theta = jnp.pi / 4.0
        phi = 0.0
        e_s, e_p = build_polarization_vectors(theta, phi)
        dot_product = jnp.dot(e_s, e_p)
        chex.assert_trees_all_close(dot_product, jnp.float64(0.0), atol=1e-10)

    def test_unit_vectors(self) -> None:
        """Verify that both e_s and e_p have unit norm.

        The test establishes the unit vectors contract for build polarization vectors
        with the concrete values and array shapes described below.

        Notes
        -----
        1. **Build polarization vectors**:
           Call ``build_polarization_vectors`` with theta=pi/3 and
           phi=pi/6.

        2. **Compute norms**:
           Calculates the Euclidean norm of each vector.

        **Expected assertions**

        Both vectors have unit norm within tolerance 1e-10.
        """
        theta: Array
        phi: Array
        e_s: Array
        e_p: Array

        theta = jnp.pi / 3.0
        phi = jnp.pi / 6.0
        e_s, e_p = build_polarization_vectors(theta, phi)
        chex.assert_trees_all_close(
            jnp.linalg.norm(e_s),
            jnp.float64(1.0),
            atol=1e-10,
        )
        chex.assert_trees_all_close(
            jnp.linalg.norm(e_p),
            jnp.float64(1.0),
            atol=1e-10,
        )

    def test_shape(self) -> None:
        """Verify that the output vectors have the correct 3D shape.

        The test establishes the shape contract for build polarization vectors with the
        concrete values and array shapes described below.

        Notes
        -----
        1. **Build polarization vectors**:
           Calls ``build_polarization_vectors`` with theta=0.5, phi=0.0.

        2. **Check shapes**:
           Confirms both returned arrays have shape ``(3,)``.

        **Expected assertions**

        Both e_s and e_p have shape ``(3,)``, matching the 3D Cartesian
        coordinate system.
        """
        e_s: Array
        e_p: Array

        e_s, e_p = build_polarization_vectors(0.5, 0.0)
        chex.assert_shape(e_s, (3,))
        chex.assert_shape(e_p, (3,))


class TestPhotonWavevector(chex.TestCase):
    """Validate :func:`~diffpes.simul.photon_wavevector`.

    Covers the spherical-angle convention and unit normalization for photon
    propagation along the surface normal and in the surface plane.

    :see: :func:`~diffpes.simul.photon_wavevector`
    """

    def test_matches_cardinal_incidence_directions(self) -> None:
        """Match normal and grazing incidence to Cartesian unit vectors.

        Map zero polar angle to positive z. Map a right-angle polar angle
        at zero azimuth to positive x within float64 tolerance.

        Notes
        -----
        Evaluate two scalar angle pairs. Compare both vectors with analytic
        Cartesian directions at ``atol=1e-12``. Check their unit norms.
        """
        normal: Array
        grazing: Array

        normal = photon_wavevector(0.0, 0.0)
        grazing = photon_wavevector(jnp.pi / 2.0, 0.0)
        chex.assert_trees_all_close(
            normal,
            jnp.array([0.0, 0.0, 1.0]),
            rtol=0.0,
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            grazing,
            jnp.array([1.0, 0.0, 0.0]),
            rtol=0.0,
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            jnp.linalg.norm(normal),
            jnp.float64(1.0),
            rtol=0.0,
            atol=1e-12,
        )
        chex.assert_trees_all_close(
            jnp.linalg.norm(grazing),
            jnp.float64(1.0),
            rtol=0.0,
            atol=1e-12,
        )


class TestBuildEfield(chex.TestCase):
    """Validate :func:`diffpes.simul.polarization.build_efield`.

    Verify the electric field for each polarization type. Check the LVP,
    LHP, RCP, and LCP relations.

    :see: :func:`~diffpes.simul.build_efield`
    """

    def test_lvp_equals_s(self) -> None:
        """Verify that LVP (linear vertical polarization) yields the s-polarization vector.

        The test establishes the lvp equals s contract for build efield with the
        concrete values and array shapes described below.

        Notes
        -----
        1. **Build E-field with LVP config**:
           Create an LVP configuration at theta=pi/4 and phi=0. Compute
           its electric field vector.

        2. **Build reference s-polarization vector**:
           Independently computes e_s using ``build_polarization_vectors``
           with the same angles.

        3. **Compare real parts**:
           Checks that the real part of the E-field matches e_s.

        **Expected assertions**

        The real part of the LVP E-field equals the s-polarization vector
        within tolerance 1e-10.
        """
        config: diffpes.types.PolarizationConfig
        efield: Array
        e_s: Array

        config = make_polarization_config(
            theta=jnp.pi / 4.0,
            phi=0.0,
            polarization_type="LVP",
        )
        efield = build_efield(config)
        e_s, _ = build_polarization_vectors(config.theta, config.phi)
        chex.assert_trees_all_close(
            jnp.real(efield),
            e_s.astype(jnp.float64),
            atol=1e-10,
        )

    def test_lhp_equals_p(self) -> None:
        """Verify that LHP (linear horizontal polarization) yields the p-polarization vector.

        The test establishes the lhp equals p contract for build efield with the
        concrete values and array shapes described below.

        Notes
        -----
        1. **Build E-field with LHP config**:
           Create an LHP configuration at theta=pi/4 and phi=0. Compute
           its electric field vector.

        2. **Build reference p-polarization vector**:
           Independently computes e_p using ``build_polarization_vectors``
           with the same angles.

        3. **Compare real parts**:
           Checks that the real part of the E-field matches e_p.

        **Expected assertions**

        The real part of the LHP E-field equals the p-polarization vector
        within tolerance 1e-10.
        """
        config: diffpes.types.PolarizationConfig
        efield: Array
        e_p: Array

        config = make_polarization_config(
            theta=jnp.pi / 4.0,
            phi=0.0,
            polarization_type="LHP",
        )
        efield = build_efield(config)
        _, e_p = build_polarization_vectors(config.theta, config.phi)
        chex.assert_trees_all_close(
            jnp.real(efield),
            e_p.astype(jnp.float64),
            atol=1e-10,
        )

    def test_rcp_lcp_conjugate(self) -> None:
        """Verify that RCP and LCP E-fields share the same real part.

        The test establishes the rcp lcp conjugate contract for build efield with the
        concrete values and array shapes described below.

        Notes
        -----
        1. **Build RCP and LCP E-fields**:
           Create RCP and LCP configurations at theta=pi/4 and phi=0.
           Compute both E-field vectors.

        2. **Compare real parts**:
           Use the circular-polarization definitions. Compare their common
           real part, e_s/sqrt(2).

        **Expected assertions**

        The real parts of the RCP and LCP E-fields are equal within
        tolerance 1e-10, confirming the conjugate symmetry relationship.
        """
        config_r: diffpes.types.PolarizationConfig
        config_l: diffpes.types.PolarizationConfig
        e_rcp: Array
        e_lcp: Array

        config_r = make_polarization_config(
            theta=jnp.pi / 4.0,
            phi=0.0,
            polarization_type="RCP",
        )
        config_l = make_polarization_config(
            theta=jnp.pi / 4.0,
            phi=0.0,
            polarization_type="LCP",
        )
        e_rcp = build_efield(config_r)
        e_lcp = build_efield(config_l)
        chex.assert_trees_all_close(
            jnp.real(e_rcp),
            jnp.real(e_lcp),
            atol=1e-10,
        )

    def test_lap_linear_combination(self) -> None:
        """Verify that LAP (linear arbitrary) yields cos(angle)*e_s + sin(angle)*e_p.

        The test establishes the lap linear combination contract for build efield with
        the concrete values and array shapes described below.

        Notes
        -----
        1. **Build E-field with LAP config**:
           Create an LAP configuration with polarization_angle=0.3.
           Compute its electric field.

        2. **Build e_s and e_p** with the same angles. Form the expected
           combination cos(0.3)*e_s + sin(0.3)*e_p.

        3. **Compare**: E-field matches the expected combination.

        **Expected assertions**

        LAP E-field equals the linear combination of s- and p-vectors
        at the given polarization angle.
        """
        angle: float
        config: diffpes.types.PolarizationConfig
        efield: Array
        e_s: Array
        e_p: Array
        e_s_c: Array
        e_p_c: Array
        expected: Array

        angle = 0.3
        config = make_polarization_config(
            theta=jnp.pi / 4.0,
            phi=0.0,
            polarization_type="LAP",
            polarization_angle=angle,
        )
        efield = build_efield(config)
        e_s, e_p = build_polarization_vectors(config.theta, config.phi)
        e_s_c = e_s.astype(jnp.complex128)
        e_p_c = e_p.astype(jnp.complex128)
        expected = jnp.cos(angle) * e_s_c + jnp.sin(angle) * e_p_c
        chex.assert_trees_all_close(efield, expected, atol=1e-10)

    def test_unknown_pol_type_fallback_to_s(self) -> None:
        """Verify that unknown polarization type falls back to e_s.

        The test establishes the unknown pol type fallback to s contract for build
        efield with the concrete values and array shapes described below.

        Notes
        -----
        1. **Build E-field with an unsupported type** (e.g. "unpolarized"):
           build_efield treats it as the default branch and returns e_s.

        2. **Compare** real part of E-field to e_s from
           build_polarization_vectors.

        **Expected assertions**

        The default/else branch returns the s-polarization vector.
        """
        config: diffpes.types.PolarizationConfig
        efield: Array
        e_s: Array

        config = make_polarization_config(
            theta=jnp.pi / 4.0,
            phi=0.0,
            polarization_type="unpolarized",
        )
        efield = build_efield(config)
        e_s, _ = build_polarization_vectors(config.theta, config.phi)
        chex.assert_trees_all_close(
            jnp.real(efield),
            e_s.astype(jnp.float64),
            atol=1e-10,
        )


class TestDipoleMatrixElements(chex.TestCase):
    """Validate :func:`diffpes.simul.polarization.dipole_matrix_elements`.

    Verify the output shape and the zero s-orbital element. Check px
    coupling and non-negative elements for arbitrary polarization.

    :see: :func:`~diffpes.simul.dipole_matrix_elements`
    """

    def test_shape(self) -> None:
        """Verify that the output has shape ``(9,)`` for the 9-orbital basis.

        The test establishes the shape contract for dipole matrix elements with the
        concrete values and array shapes described below.

        Notes
        -----
        1. **Create x-polarized E-field**:
           Constructs a complex E-field vector along the x-axis.

        2. **Compute matrix elements**:
           Calls ``dipole_matrix_elements`` and checks the output shape.

        **Expected assertions**

        The output array has shape ``(9,)``, one element per orbital in
        the [s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2] basis.
        """
        efield: Array
        m: Array

        efield = jnp.array([1.0, 0.0, 0.0], dtype=jnp.complex128)
        m = dipole_matrix_elements(efield)
        chex.assert_shape(m, (9,))

    def test_s_orbital_zero(self) -> None:
        """Verify that the s-orbital dipole matrix element is always zero.

        The test establishes the s orbital zero contract for dipole matrix elements
        with the concrete values and array shapes described below.

        Notes
        -----
        1. **Create x-polarized E-field**:
           Constructs a complex E-field vector along the x-axis.

        2. **Compute matrix elements**:
           Calls ``dipole_matrix_elements`` and inspects the s-orbital
           entry (index 0).

        **Expected assertions**

        The s-orbital direction vector is [0, 0, 0]. Its matrix element
        is zero within tolerance 1e-10 for any electric field.
        """
        efield: Array
        m: Array

        efield = jnp.array([1.0, 0.0, 0.0], dtype=jnp.complex128)
        m = dipole_matrix_elements(efield)
        chex.assert_trees_all_close(m[0], jnp.float64(0.0), atol=1e-10)

    def test_px_with_x_field(self) -> None:
        """Verify that an x-polarized field produces a positive px matrix element.

        The test establishes the px with x field contract for dipole matrix elements
        with the concrete values and array shapes described below.

        Notes
        -----
        1. **Create x-polarized E-field**:
           Constructs a complex E-field vector purely along the x-axis.

        2. **Compute matrix elements**:
           Calls ``dipole_matrix_elements`` and inspects the px-orbital
           entry (index 3), whose direction vector is [1, 0, 0].

        **Expected assertions**

        The px-orbital matrix element (index 3) is strictly positive,
        confirming that the x-polarized field couples to the px-orbital
        via the dipole selection rule.
        """
        efield: Array
        m: Array

        efield = jnp.array([1.0, 0.0, 0.0], dtype=jnp.complex128)
        m = dipole_matrix_elements(efield)
        chex.assert_scalar_positive(float(m[3]))

    def test_all_nonnegative(self) -> None:
        """Verify that all dipole matrix elements are non-negative for arbitrary polarization.

        The test establishes the all nonnegative contract for dipole matrix elements
        with the concrete values and array shapes described below.

        Notes
        -----
        1. **Create arbitrary normalized E-field**:
           Create a complex field with components [0.5, 0.3, 0.8].
           Normalize it to unit length.

        2. **Compute matrix elements**:
           Calls ``dipole_matrix_elements`` on the normalized field.

        3. **Check non-negativity of all elements**:
           Iterates over all 9 entries and asserts each is >= 0.

        **Expected assertions**

        Every element of the 9-element matrix element vector is non-negative,
        as expected from the squared-modulus definition
        ``|e . d|^2 >= 0``.
        """
        i: int

        efield: Array
        m: Array

        efield = jnp.array([0.5, 0.3, 0.8], dtype=jnp.complex128)
        efield = efield / jnp.linalg.norm(efield)
        m = dipole_matrix_elements(efield)
        for i in range(9):
            assert float(m[i]) >= 0.0
