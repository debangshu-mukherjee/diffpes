"""Test orbital-basis and Slater-parameter carriers.

Extended Summary
----------------
Covers construction, defaults, PyTree and gradient behavior, static validation, and traced rejection contracts for the carriers defined in ``diffpes.types.radial_params``.
"""

import chex
import jax
import jax.numpy as jnp
import pytest

from diffpes.types import make_orbital_basis, make_slater_params
from tests._assertions import assert_rejects


class TestOrbitalBasis:
    """Tests for :func:`diffpes.types.make_orbital_basis`.

    Validates construction of ``OrbitalBasis`` PyTrees that describe
    the quantum-number basis (n, l, m) for Chinook tight-binding
    calculations. Covers explicit creation with provided quantum
    numbers, automatic label generation, JAX PyTree round-trip
    (flatten/unflatten) fidelity, and input validation when quantum
    number tuple lengths are mismatched.
    """

    def test_creation(self) -> None:
        """Verify OrbitalBasis stores quantum number tuples unchanged.

        Constructs an OrbitalBasis with two orbitals: (n=1, l=0, m=0)
        and (n=2, l=1, m=0). Asserts that ``n_values`` is stored as
        ``(1, 2)`` and ``l_values`` as ``(0, 1)``, confirming the
        factory passes through the tuple auxiliary data without
        modification.
        """
        basis = make_orbital_basis(
            n_values=(1, 2),
            l_values=(0, 1),
            m_values=(0, 0),
        )
        assert basis.n_values == (1, 2)
        assert basis.l_values == (0, 1)

    def test_default_labels(self) -> None:
        """Verify auto-generated labels follow the ``orb_N`` naming convention.

        Constructs a single-orbital OrbitalBasis without supplying
        explicit labels. Asserts that ``labels`` defaults to
        ``("orb_0",)``, confirming the factory's automatic label
        generation produces zero-indexed names matching the number of
        orbitals.
        """
        basis = make_orbital_basis(
            n_values=(1,),
            l_values=(0,),
            m_values=(0,),
        )
        assert basis.labels == ("orb_0",)

    def test_pytree_flatten_unflatten(self) -> None:
        """Verify OrbitalBasis survives a JAX PyTree flatten/unflatten round-trip.

        Constructs a 3-orbital basis with explicit labels ``("s", "p", "d")``.
        Flattens via ``jax.tree_util.tree_flatten`` and reconstructs via
        ``jax.tree_util.tree_unflatten``. Asserts that ``n_values`` and
        ``labels`` on the restored object match the originals, confirming
        that the auxiliary data (tuples of ints and strings) is correctly
        encoded in the tree definition and recovered on reconstruction.
        """
        basis = make_orbital_basis(
            n_values=(1, 2, 3),
            l_values=(0, 1, 2),
            m_values=(0, 0, 0),
            labels=("s", "p", "d"),
        )
        leaves, treedef = jax.tree_util.tree_flatten(basis)
        basis2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert basis2.n_values == basis.n_values
        assert basis2.labels == basis.labels

    def test_length_mismatch_raises(self) -> None:
        """Verify ValueError is raised when quantum number tuple lengths disagree.

        Calls ``make_orbital_basis`` with ``n_values`` of length 2,
        ``l_values`` of length 1, and ``m_values`` of length 2.
        Asserts that ``ValueError`` is raised with a message matching
        ``"same length"``, confirming the factory validates that all
        three quantum number tuples have consistent lengths before
        construction.
        """
        with pytest.raises(ValueError, match="same length"):
            make_orbital_basis(
                n_values=(1, 2),
                l_values=(0,),
                m_values=(0, 0),
            )


class TestSlaterParams:
    """Tests for :func:`diffpes.types.make_slater_params`.

    Validates construction of ``SlaterParams`` PyTrees that hold
    Slater-type orbital exponents and expansion coefficients. Covers
    basic creation with shape verification, gradient flow through
    the ``zeta`` field (essential for inverse fitting), and automatic
    float64 dtype casting of input arrays.
    """

    def test_creation(self) -> None:
        """Verify SlaterParams stores zeta and default coefficients with correct shapes.

        Constructs a SlaterParams with a single orbital (zeta = [1.5])
        and a 1-orbital basis. Asserts ``zeta`` shape is (1,) and
        ``coefficients`` shape is (1, 1), confirming that the factory
        creates a default single-term expansion coefficient matrix when
        none is supplied.
        """
        basis = make_orbital_basis(
            n_values=(1,),
            l_values=(0,),
            m_values=(0,),
        )
        sp = make_slater_params(
            zeta=jnp.array([1.5]),
            orbital_basis=basis,
        )
        assert sp.zeta.shape == (1,)
        assert sp.coefficients.shape == (1, 1)

    def test_pytree_gradient(self) -> None:
        """Verify JAX gradients flow through the zeta field of SlaterParams.

        Defines a loss function ``loss(sp) = sum(zeta^2)`` and computes
        ``jax.grad(loss)`` with respect to a SlaterParams having
        ``zeta = [2.0]``. Asserts the gradient of zeta is 4.0
        (``d/d(zeta) zeta^2 = 2*zeta = 4.0``), confirming that
        SlaterParams is a valid differentiable JAX PyTree and that
        ``jax.grad`` can trace through its leaf arrays. This is
        critical for Chinook inverse fitting workflows.
        """
        basis = make_orbital_basis(
            n_values=(1,),
            l_values=(0,),
            m_values=(0,),
        )

        def loss(sp):
            return jnp.sum(sp.zeta**2)

        sp = make_slater_params(
            zeta=jnp.array([2.0]),
            orbital_basis=basis,
        )
        grad = jax.grad(loss)(sp)
        assert jnp.allclose(grad.zeta, 4.0)

    def test_float64_casting(self) -> None:
        """Verify that float32 input arrays are automatically promoted to float64.

        Constructs a SlaterParams with ``zeta`` supplied as a float32
        array. Asserts that the stored ``zeta`` has dtype ``jnp.float64``,
        confirming the factory enforces 64-bit precision regardless of
        the input dtype. This is important for numerical accuracy in
        Slater integrals and gradient-based optimization.
        """
        basis = make_orbital_basis(
            n_values=(1,),
            l_values=(0,),
            m_values=(0,),
        )
        sp = make_slater_params(
            zeta=jnp.array([1.0], dtype=jnp.float32),
            orbital_basis=basis,
        )
        assert sp.zeta.dtype == jnp.float64


class TestMakeOrbitalBasisErrors(chex.TestCase):
    """Tests for validation errors in make_orbital_basis.

    Verifies that ``make_orbital_basis`` raises ``ValueError`` when
    the quantum number arrays have mismatched lengths or when a
    ``labels`` tuple has the wrong length.
    """

    def test_length_mismatch_raises(self) -> None:
        """Verify that mismatched n_values / l_values lengths raise ValueError.

        Passes ``n_values=(1, 2)`` (length 2) with ``l_values=(0,)``
        (length 1) and asserts a ``ValueError`` is raised, covering
        the length-check guard in the factory.
        """
        with pytest.raises(ValueError, match="same length"):
            make_orbital_basis(
                n_values=(1, 2),
                l_values=(0,),
                m_values=(0,),
            )

    def test_labels_length_mismatch_raises(self) -> None:
        """Verify that a mismatched labels tuple raises ValueError.

        Passes a single-orbital basis but provides two labels, and
        asserts a ``ValueError`` matching "same length" is raised,
        covering the labels-length guard.
        """
        with pytest.raises(ValueError, match="same length"):
            make_orbital_basis(
                n_values=(1,),
                l_values=(0,),
                m_values=(0,),
                labels=("s", "extra"),
            )


class TestMakeSlaterParamsErrors(chex.TestCase):
    """Tests for validation errors and defaults in make_slater_params.

    Verifies that ``make_slater_params`` raises ``ValueError`` when
    the ``zeta`` array length does not match ``orbital_basis`` size,
    and that the default ``coefficients=None`` path creates a
    single-zeta ones array.
    """

    def test_zeta_length_mismatch_raises(self) -> None:
        """Verify that a zeta length mismatch raises ValueError.

        Creates a single-orbital basis but passes ``zeta`` of length 3,
        and asserts a ``ValueError`` matching "zeta length" is raised.
        """
        basis = make_orbital_basis(
            n_values=(1,),
            l_values=(0,),
            m_values=(0,),
        )
        zeta = jnp.ones(3, dtype=jnp.float64)
        with pytest.raises(ValueError, match="zeta length"):
            make_slater_params(zeta=zeta, orbital_basis=basis)

    def test_default_coefficients_are_ones(self) -> None:
        """Verify that coefficients=None produces a (O, 1) ones array in float64.

        Creates a 2-orbital basis with ``coefficients=None`` and asserts
        the resulting ``coefficients`` has shape ``(2, 1)``, dtype
        ``float64``, and all values equal to 1.0.
        """
        basis = make_orbital_basis(
            n_values=(1, 2),
            l_values=(0, 1),
            m_values=(0, 0),
        )
        zeta = jnp.array([1.0, 1.5], dtype=jnp.float64)
        params = make_slater_params(zeta=zeta, orbital_basis=basis)
        chex.assert_shape(params.coefficients, (2, 1))
        assert params.coefficients.dtype == jnp.float64
        chex.assert_trees_all_close(
            params.coefficients,
            jnp.ones((2, 1), dtype=jnp.float64),
        )

    def test_explicit_coefficients_are_cast_to_float64(self) -> None:
        """Verify that explicit coefficients are cast to float64.

        Creates a 2-orbital, 2-zeta basis with explicit float32 coefficients
        and asserts the stored array is float64 with the correct shape.
        This covers the ``coeff_arr = jnp.asarray(coefficients, ...)`` branch.
        """
        basis = make_orbital_basis(
            n_values=(1, 2),
            l_values=(0, 1),
            m_values=(0, 0),
        )
        zeta = jnp.array([1.0, 1.5], dtype=jnp.float64)
        coeffs = jnp.array([[0.8, 0.2], [0.6, 0.4]], dtype=jnp.float32)
        params = make_slater_params(
            zeta=zeta, orbital_basis=basis, coefficients=coeffs
        )
        chex.assert_shape(params.coefficients, (2, 2))
        assert params.coefficients.dtype == jnp.float64


def test_orbital_basis_rejects_invalid_quantum_numbers() -> None:
    """Reject principal quantum numbers below one."""
    assert_rejects(
        make_orbital_basis,
        (0,),
        (0,),
        (0,),
        match="n_values must all be at least 1",
    )


def test_slater_params_reject_nonpositive_zeta() -> None:
    """Reject nonpositive exponents and coefficient-axis mismatches."""
    basis = make_orbital_basis((1,), (0,), (0,))
    assert_rejects(
        make_slater_params,
        zeta=jnp.array([0.0]),
        orbital_basis=basis,
        match="zeta positive",
    )
    assert_rejects(
        make_slater_params,
        zeta=jnp.array([1.0]),
        orbital_basis=basis,
        coefficients=jnp.ones((2, 1)),
        match="coefficients first dimension must match",
    )
