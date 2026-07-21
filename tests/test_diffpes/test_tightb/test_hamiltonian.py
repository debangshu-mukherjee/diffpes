"""Validate the tight-binding Hamiltonian builder.

Extended Summary
----------------
The tests validate ``build_hamiltonian_k`` and the two model factories.
They cover Hermiticity, the matrix shape, JIT, analytic dispersions, and
gradients with respect to hopping parameters.

"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from beartype.typing import Any, Callable
from jaxtyping import Array

import diffpes
from diffpes.tightb import build_hamiltonian_k
from diffpes.types import (
    make_1d_chain_model,
    make_graphene_model,
)


class TestBuildHamiltonianK:
    """Validate ``build_hamiltonian_k``.

    The tests cover the Bloch Hamiltonian ``H(k)`` from hopping parameters
    and lattice vectors. They verify Hermiticity, the matrix shape, and JIT.

    :see: :func:`~diffpes.tightb.build_hamiltonian_k`
    """

    def test_hermitian(self) -> None:
        """Verify H(k) satisfies Hermiticity at the specified k-point.

        The graphene Hamiltonian satisfies ``H = H^dag``. This property
        permits a real eigenspectrum.

        Notes
        -----
        The test computes ``H`` at ``k=(0.3, 0.2, 0.0)``. It compares ``H``
        with its conjugate transpose at an absolute tolerance of 1e-12.
        """
        model: diffpes.types.TBModel
        k: Array
        H: Array

        model = make_graphene_model()
        k = jnp.array([0.3, 0.2, 0.0])
        H = build_hamiltonian_k(
            k,
            model.hopping_params,
            model.hopping_indices,
            model.n_orbitals,
            model.lattice_vectors,
        )
        assert jnp.allclose(H, H.conj().T, atol=1e-12)

    def test_correct_shape(self) -> None:
        """Verify H(k) has shape (n_orbitals, n_orbitals).

        The Hamiltonian has one row and one column for each model orbital.

        Notes
        -----
        The test computes the two-orbital graphene Hamiltonian at one k-point.
        It compares the output shape with ``(2, 2)``.
        """
        model: diffpes.types.TBModel
        k: Array
        H: Array

        model = make_graphene_model()
        k = jnp.array([0.1, 0.2, 0.0])
        H = build_hamiltonian_k(
            k,
            model.hopping_params,
            model.hopping_indices,
            model.n_orbitals,
            model.lattice_vectors,
        )
        assert H.shape == (2, 2)

    def test_jit_compatible(self) -> None:
        """Verify JAX compiles ``build_hamiltonian_k``.

        The compiled builder accepts a traced k-point and produces finite
        matrix elements.

        Notes
        -----
        The test compiles the builder for the 1D chain model with ``jax.jit``.
        It checks the result at ``k=(0.25, 0, 0)`` for finite values.
        """
        model: diffpes.types.TBModel
        f: Callable[..., Any]
        H: Array

        model = make_1d_chain_model()
        f = jax.jit(
            lambda k: build_hamiltonian_k(
                k,
                model.hopping_params,
                model.hopping_indices,
                model.n_orbitals,
                model.lattice_vectors,
            )
        )
        H = f(jnp.array([0.25, 0.0, 0.0]))
        assert jnp.all(jnp.isfinite(H))


class TestMake1DChainModel:
    """Validate the 1D chain tight-binding model factory.

    The tests compare the model with its analytic cosine dispersion.
    They also verify the eigenvalue interval ``[-2|t|, 2|t|]``.

    :see: :func:`~diffpes.types.make_1d_chain_model`
    """

    def test_cosine_dispersion(self) -> None:
        """Verify E(k) = 2t*cos(2*pi*k) for the 1D chain.

        The single-band model with ``t=-1.0`` has the dispersion
        ``E(k) = -2*cos(2*pi*k)``.

        Notes
        -----
        The test diagonalizes 101 k-points in the Brillouin zone
        ``[-0.5, 0.5]``. It compares all eigenvalues at an absolute tolerance
        of 1e-10.
        """
        model: diffpes.types.TBModel
        kpoints: Array
        diag: diffpes.types.DiagonalizedBands
        expected: Array

        model = make_1d_chain_model(t=-1.0)
        kpoints = jnp.linspace(-0.5, 0.5, 101)[:, None] * jnp.array(
            [[1.0, 0.0, 0.0]]
        )

        from diffpes.tightb.diagonalize import diagonalize_tb

        diag = diagonalize_tb(model, kpoints)
        expected = -2.0 * jnp.cos(2.0 * jnp.pi * jnp.linspace(-0.5, 0.5, 101))
        assert jnp.allclose(diag.eigenvalues[:, 0], expected, atol=1e-10)

    def test_eigenvalue_range(self) -> None:
        """Verify eigenvalue bandwidth spans [-2|t|, 2|t|].

        A chain with ``t=-1.5`` has eigenvalue limits of -3.0 and 3.0 eV.
        The corresponding bandwidth is 6.0 eV.

        Notes
        -----
        The test diagonalizes 201 k-points across the Brillouin zone. It
        compares both extrema at an absolute tolerance of 0.05 eV.
        """
        model: diffpes.types.TBModel
        kpoints: Array
        diag: diffpes.types.DiagonalizedBands

        model = make_1d_chain_model(t=-1.5)
        kpoints = jnp.linspace(-0.5, 0.5, 201)[:, None] * jnp.array(
            [[1.0, 0.0, 0.0]]
        )

        from diffpes.tightb.diagonalize import diagonalize_tb

        diag = diagonalize_tb(model, kpoints)
        assert float(diag.eigenvalues.min()) == pytest.approx(-3.0, abs=0.05)
        assert float(diag.eigenvalues.max()) == pytest.approx(3.0, abs=0.05)


class TestMakeGrapheneModel:
    """Validate the graphene tight-binding model factory.

    The tests use analytic properties of the graphene pi-band structure.
    They cover the Gamma point, the Dirac point, and hopping gradients.

    :see: :func:`~diffpes.types.make_graphene_model`
    """

    def test_gamma_point(self) -> None:
        """Verify graphene has Gamma-point eigenvalues of +/-3|t|.

        The two-orbital model with ``t=-2.7`` eV has eigenvalues of -8.1
        and 8.1 eV at the Gamma point.

        Notes
        -----
        The test diagonalizes the model at ``Gamma=(0, 0, 0)``. It compares
        both eigenvalues at an absolute tolerance of 0.01 eV.
        """
        model: diffpes.types.TBModel
        Gamma: Array
        diag: diffpes.types.DiagonalizedBands
        evals: Array

        model = make_graphene_model(t=-2.7)
        Gamma = jnp.array([[0.0, 0.0, 0.0]])

        from diffpes.tightb.diagonalize import diagonalize_tb

        diag = diagonalize_tb(model, Gamma)
        evals = jnp.sort(diag.eigenvalues[0])
        assert float(evals[0]) == pytest.approx(-8.1, abs=0.01)
        assert float(evals[1]) == pytest.approx(8.1, abs=0.01)

    def test_k_point_dirac(self) -> None:
        """Verify the Dirac point at K=(2/3, 1/3, 0) has zero-energy eigenvalues.

        The off-diagonal Bloch sum vanishes at this k-point. Both eigenvalues
        consequently equal zero.

        Notes
        -----
        The test diagonalizes the model with ``t=-2.7`` eV at the Dirac
        point. It compares both eigenvalues with zero at a tolerance of 1e-10 eV.
        """
        model: diffpes.types.TBModel
        K: Array
        diag: diffpes.types.DiagonalizedBands

        model = make_graphene_model(t=-2.7)
        K = jnp.array([[2.0 / 3, 1.0 / 3, 0.0]])

        from diffpes.tightb.diagonalize import diagonalize_tb

        diag = diagonalize_tb(model, K)
        assert jnp.allclose(diag.eigenvalues[0], 0.0, atol=1e-10)

    def test_gradient_wrt_hopping(self) -> None:
        """Verify JAX produces a finite hopping gradient for ``sum(E**2)``.

        The graphene eigenspectrum carries sensitivity to the hopping
        parameters through the Hamiltonian and the eigendecomposition.

        Notes
        -----
        The test differentiates ``sum(E**2)`` with ``jax.grad`` at
        ``k=(0.1, 0.2, 0)``. It checks every hopping gradient for a finite value.
        """
        model: diffpes.types.TBModel
        kpoints: Array
        grad: Array

        model = make_graphene_model(t=-2.7)
        kpoints = jnp.array([[0.1, 0.2, 0.0]])

        from diffpes.tightb.diagonalize import diagonalize_tb

        def loss(hop):
            m: Array
            d: diffpes.types.DiagonalizedBands

            m = eqx.tree_at(lambda item: item.hopping_params, model, hop)
            d = diagonalize_tb(m, kpoints)
            return jnp.sum(d.eigenvalues**2)

        grad = jax.grad(loss)(model.hopping_params)
        assert jnp.all(jnp.isfinite(grad))
