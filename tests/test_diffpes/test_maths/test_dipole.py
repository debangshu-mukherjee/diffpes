"""Validate dipole matrix element assembly.

Extended Summary
----------------
The tests validate the complete matrix-element pipeline for the ARPES
photoemission intensity. The pipeline includes the single-orbital function
``dipole_matrix_element_single``, the per-orbital intensity wrapper
``dipole_intensity_orbital``, and the multi-orbital batch function
``dipole_intensities_all_orbitals``. The tests cover allowed transitions,
finite values, JIT compatibility, and nonnegative intensities. They compare
``|M|^2`` with the intensity function and verify the output shapes. They also
verify derivatives for the polarization vector and Slater exponent.

"""

import jax
import jax.numpy as jnp
import pytest
from beartype import beartype
from beartype.typing import Any, Callable
from jaxtyping import Array, Complex, Float, jaxtyped

import diffpes
from diffpes.maths import (
    dipole_intensities_all_orbitals,
    dipole_intensity_orbital,
    dipole_matrix_element_single,
)
from diffpes.radial import slater_radial
from diffpes.types import make_orbital_basis, make_slater_params


@pytest.fixture
@jaxtyped(typechecker=beartype)
def r_grid() -> Float[Array, " R"]:
    """Provide a dense radial grid for numerical integration.

    The fixture returns 5000 points from 1e-6 to 50.0 Bohr. The nonzero
    lower bound avoids the ``1/r`` singularity in Slater radial functions.
    The upper bound lets typical exponential tails decay below machine
    precision.
    """
    grid: Float[Array, " R"] = jnp.linspace(1e-6, 50.0, 5000)
    return grid


@pytest.fixture
@jaxtyped(typechecker=beartype)
def z_polarized() -> Complex[Array, " 3"]:
    """Provide a z-polarized electric-field vector.

    The fixture returns the complex polarization vector ``(0, 0, 1)``.
    This vector selects the ``q=0`` component of the dipole operator.
    It represents linear polarization along the z-axis.
    """
    polarization: Complex[Array, " 3"] = jnp.array(
        [0.0, 0.0, 1.0], dtype=jnp.complex128
    )
    return polarization


@pytest.fixture
@jaxtyped(typechecker=beartype)
def x_polarized() -> Complex[Array, " 3"]:
    """Provide an x-polarized electric-field vector.

    The fixture returns the complex polarization vector ``(1, 0, 0)``.
    This vector mixes the ``q=+1`` and ``q=-1`` spherical components.
    It represents linear polarization along the x-axis.
    """
    polarization: Complex[Array, " 3"] = jnp.array(
        [1.0, 0.0, 0.0], dtype=jnp.complex128
    )
    return polarization


class TestDipoleMatrixElementSingle:
    """Validate ``dipole_matrix_element_single``.

    The tests validate the core single-orbital dipole matrix element.
    This element combines a radial integral with Gaunt angular coupling.
    The tests verify allowed transitions, finite values, and JIT compatibility.
    They also verify autodiff gradients for the electric-field vector.

    :see: :func:`~diffpes.maths.dipole_matrix_element_single`
    """

    def test_s_orbital_nonzero(self, r_grid, z_polarized) -> None:
        """Verify the s -> p dipole transition is nonzero for z-polarization.

        The test uses an n=1, zeta=1.0 Slater 1s orbital with z-polarized light and
        k along z. The dipole selection rule allows ``l=0`` to ``l'=1``.
        With ``q=0``, the matrix element must be nonzero. The test asserts
        ``|M| > 1e-6``
        to confirm the angular (Gaunt) and radial integrals both contribute.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        R_vals: Array
        k_vec: Array
        M: Array

        R_vals = slater_radial(r_grid, 1, 1.0)
        k_vec = jnp.array([0.0, 0.0, 1.0])
        M = dipole_matrix_element_single(
            k_vec, r_grid, R_vals, 0, 0, z_polarized
        )
        assert jnp.abs(M) > 1e-6, (
            f"Expected nonzero, got |M|={float(jnp.abs(M))}"
        )

    def test_finite_output(self, r_grid, z_polarized) -> None:
        """Verify the matrix element is numerically finite for a p-orbital.

        The test uses an ``n=2``, ``zeta=2.0`` Slater 2p orbital.
        It uses the k-vector ``(1.0, 0.5, 0.3)`` and z-polarization. It asserts
        ``jnp.isfinite(M)`` to guard against NaN or Inf from the radial
        integration or angular coupling at arbitrary k-directions.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        R_vals: Array
        k_vec: Array
        M: Array

        R_vals = slater_radial(r_grid, 2, 2.0)
        k_vec = jnp.array([1.0, 0.5, 0.3])
        M = dipole_matrix_element_single(
            k_vec, r_grid, R_vals, 1, 0, z_polarized
        )
        assert jnp.isfinite(M)

    def test_jit_compatible(self, r_grid, z_polarized) -> None:
        """Verify ``dipole_matrix_element_single`` is JAX-JIT-compatible.

        The test wraps the function in ``jax.jit`` with k as the traced argument.
        The closure captures the other inputs as constants.
        The test uses an s-orbital (l=0, m=0) with k along z.  Asserts the JIT-
        compiled output is finite, confirming no Python-side control flow
        or non-JAX operations break tracing.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        R_vals: Array
        f: Callable[..., Any]
        M: Array

        R_vals = slater_radial(r_grid, 1, 1.0)
        f = jax.jit(
            lambda k: dipole_matrix_element_single(
                k, r_grid, R_vals, 0, 0, z_polarized
            )
        )
        M = f(jnp.array([0.0, 0.0, 1.0]))
        assert jnp.isfinite(M)

    def test_gradient_wrt_efield(self, r_grid) -> None:
        """Verify autodiff gradient of intensity w.r.t. polarization vector.

        The test defines ``loss = |M(E)|^2`` for an s-orbital with k along z.
        It differentiates the complete dipole pipeline for the electric-field
        vector with ``jax.grad``. It asserts
        all three gradient components are finite, confirming the Gaunt
        lookup, radial integral, and complex-valued inner product are all
        smoothly differentiable.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        R_vals: Array
        k_vec: Array
        grad: Array

        R_vals = slater_radial(r_grid, 1, 1.0)
        k_vec = jnp.array([0.0, 0.0, 1.0])

        def loss(ef):
            M: Array

            M = dipole_matrix_element_single(k_vec, r_grid, R_vals, 0, 0, ef)
            return jnp.abs(M) ** 2

        grad = jax.grad(loss)(jnp.array([0.0, 0.0, 1.0], dtype=jnp.complex128))
        assert jnp.all(jnp.isfinite(grad))


class TestDipoleIntensityOrbital:
    """Validate ``dipole_intensity_orbital``.

    Validates the single-orbital intensity function, which returns the
    modulus squared of the dipole matrix element.  Tests verify the
    physical constraint that intensity is non-negative and that the
    function is consistent with computing ``|M|^2`` directly from
    ``dipole_matrix_element_single``.

    :see: :func:`~diffpes.maths.dipole_intensity_orbital`
    """

    def test_non_negative(self, r_grid, z_polarized) -> None:
        """Verify photoemission intensity is non-negative.

        The test uses an n=2, zeta=1.5 Slater p-orbital (l=1, m=0) with an
        off-axis k-vector and z-polarization. Intensity equals ``|M|^2`` and
        must be nonnegative. The test asserts ``I >= 0.0`` as a basic
        physical sanity check.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        R_vals: Array
        k_vec: Array
        I: Array

        R_vals = slater_radial(r_grid, 2, 1.5)
        k_vec = jnp.array([0.5, 0.3, 0.8])
        I = dipole_intensity_orbital(k_vec, r_grid, R_vals, 1, 0, z_polarized)
        assert float(I) >= 0.0

    def test_equals_abs_squared(self, r_grid, z_polarized) -> None:
        """Verify intensity equals the modulus squared of the matrix element.

        The test computes ``M`` and ``I`` independently for a 1s orbital.
        It uses k along z and z-polarization. It asserts ``I == |M|^2``
        within ``1e-12``. This comparison confirms that
        ``dipole_intensity_orbital`` is a faithful wrapper around
        ``dipole_matrix_element_single``.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        R_vals: Array
        k_vec: Array
        M: Array
        I: Array

        R_vals = slater_radial(r_grid, 1, 1.0)
        k_vec = jnp.array([0.0, 0.0, 1.5])
        M = dipole_matrix_element_single(
            k_vec, r_grid, R_vals, 0, 0, z_polarized
        )
        I = dipole_intensity_orbital(k_vec, r_grid, R_vals, 0, 0, z_polarized)
        assert jnp.allclose(I, jnp.abs(M) ** 2, atol=1e-12)


class TestDipoleIntensitiesAllOrbitals:
    """Validate ``dipole_intensities_all_orbitals``.

    The batch function computes the dipole intensity for each orbital.
    The tests verify one intensity per orbital and nonnegative values.
    They also verify derivatives of the total intensity for the Slater
    exponent ``zeta``.

    :see: :func:`~diffpes.maths.dipole_intensities_all_orbitals`
    """

    def test_output_shape(self, r_grid, z_polarized) -> None:
        """Verify output has one intensity per orbital in the basis.

        The test constructs a three-orbital basis (1s, 2pz, 2px) with corresponding
        Slater exponents and computes intensities at a single k-point.
        The test asserts the output shape is ``(3,)``, matching the number of
        orbitals in the basis.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        basis: diffpes.types.OrbitalBasis
        sp: diffpes.types.SlaterParams
        k_vec: Array
        I: Array

        basis = make_orbital_basis(
            n_values=(1, 2, 2),
            l_values=(0, 1, 1),
            m_values=(0, 0, 1),
            labels=("1s", "2pz", "2px"),
        )
        sp = make_slater_params(
            zeta=jnp.array([1.0, 1.5, 1.5]),
            orbital_basis=basis,
        )
        k_vec = jnp.array([0.0, 0.0, 1.0])
        I = dipole_intensities_all_orbitals(k_vec, r_grid, sp, z_polarized)
        assert I.shape == (3,)

    def test_all_non_negative(self, r_grid, z_polarized) -> None:
        """Verify all orbital intensities are non-negative.

        The test constructs a four-orbital basis (1s, 2py, 2pz, 2px) spanning all
        three p sub-orbitals plus s, with an off-axis k-vector and
        z-polarization.  Asserts ``I >= 0`` element-wise, confirming the
        ``|M|^2`` definition holds for every orbital in the batch.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        basis: diffpes.types.OrbitalBasis
        sp: diffpes.types.SlaterParams
        k_vec: Array
        I: Array

        basis = make_orbital_basis(
            n_values=(1, 2, 2, 2),
            l_values=(0, 1, 1, 1),
            m_values=(0, -1, 0, 1),
            labels=("1s", "2py", "2pz", "2px"),
        )
        sp = make_slater_params(
            zeta=jnp.array([1.0, 1.5, 1.5, 1.5]),
            orbital_basis=basis,
        )
        k_vec = jnp.array([0.5, 0.3, 0.8])
        I = dipole_intensities_all_orbitals(k_vec, r_grid, sp, z_polarized)
        assert jnp.all(I >= 0.0)

    def test_gradient_wrt_zeta(self, r_grid, z_polarized) -> None:
        """Verify autodiff gradient of total intensity w.r.t. Slater exponent.

        The test defines ``loss = sum(I)`` for one 1s orbital. It
        differentiates the loss for ``zeta`` with ``jax.grad``. The test
        asserts that the gradient is finite. This result confirms that the
        complete forward pipeline supports reverse-mode autodiff. Inverse
        fits of orbital exponents require this property.

        Notes
        -----
        The test builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        basis: diffpes.types.OrbitalBasis
        grad: Array

        basis = make_orbital_basis(
            n_values=(1,),
            l_values=(0,),
            m_values=(0,),
            labels=("1s",),
        )

        def loss(zeta_val):
            sp: diffpes.types.SlaterParams
            k_vec: Array
            I: Array

            sp = make_slater_params(
                zeta=jnp.array([zeta_val]),
                orbital_basis=basis,
            )
            k_vec = jnp.array([0.0, 0.0, 1.0])
            I = dipole_intensities_all_orbitals(k_vec, r_grid, sp, z_polarized)
            return jnp.sum(I)

        grad = jax.grad(loss)(jnp.array(1.0))
        assert jnp.isfinite(grad), f"Gradient is {grad}"
