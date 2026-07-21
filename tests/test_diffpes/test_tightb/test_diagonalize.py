"""Validate differentiable diagonalization and the VASP adapter.

Extended Summary
----------------
The tests validate the single-k solver, the multi-k solver, and the VASP
adapter. They cover eigensystem properties, gradients, output shapes, and
the policy for phase loss.

"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from beartype.typing import Any, Callable
from jaxtyping import Array

import diffpes
from diffpes.tightb import (
    diagonalize_single_k,
    diagonalize_tb,
    vasp_to_diagonalized,
)
from diffpes.types import (
    make_1d_chain_model,
    make_band_structure,
    make_graphene_model,
    make_orbital_basis,
    make_orbital_projection,
)


class TestDiagonalizeSingleK:
    """Validate ``diagonalize_single_k``.

    The tests verify the real eigenvalues and the orthonormal eigenvectors of
    a Hermitian matrix.

    :see: :func:`~diffpes.tightb.diagonalize_single_k`
    """

    def test_eigenvalues_real(self) -> None:
        """Verify the Hermitian solver returns real eigenvalues.

        The Hermitian eigensolver produces a float64 eigenvalue array from a
        complex128 input matrix.

        Notes
        -----
        The test diagonalizes a 2x2 matrix with complex off-diagonal entries.
        It compares the eigenvalue dtype with ``jnp.float64``.
        """
        H: Array
        evals: Array
        evecs: Array

        H = jnp.array(
            [[1.0, 0.5 + 0.1j], [0.5 - 0.1j, 2.0]], dtype=jnp.complex128
        )
        evals, evecs = diagonalize_single_k(H)
        assert evals.dtype == jnp.float64

    def test_eigenvectors_orthogonal(self) -> None:
        """Verify the Hermitian solver returns orthonormal eigenvectors.

        The eigenvectors of the Hermitian matrix form a unitary
        transformation.

        Notes
        -----
        The test diagonalizes a real-symmetric 2x2 matrix. It compares
        ``U^dag U`` with the identity at an absolute tolerance of 1e-10.
        """
        H: Array
        evals: Array
        evecs: Array
        overlap: Array

        H = jnp.array([[1.0, 0.5], [0.5, 2.0]], dtype=jnp.complex128)
        evals, evecs = diagonalize_single_k(H)
        overlap = evecs.conj().T @ evecs
        assert jnp.allclose(overlap, jnp.eye(2), atol=1e-10)


class TestDiagonalizeTB:
    """Validate ``diagonalize_tb``.

    The tests use graphene and 1D chain models. They cover output shapes,
    eigenvalue order, hopping gradients, and eigenvector indexing.

    :see: :func:`~diffpes.tightb.diagonalize_tb`
    """

    def test_output_shapes(self) -> None:
        """Verify output shapes match (K, B) eigenvalues and (K, B, O) eigenvectors.

        The solver retains both k-points and both orbitals in the
        ``DiagonalizedBands`` fields.

        Notes
        -----
        The test diagonalizes a two-orbital graphene model at two k-points.
        It compares all three output shapes with their specified values.
        """
        model: diffpes.types.TBModel
        kpoints: Array
        diag: diffpes.types.DiagonalizedBands

        model = make_graphene_model()
        kpoints = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        diag = diagonalize_tb(model, kpoints)
        assert diag.eigenvalues.shape == (2, 2)
        assert diag.eigenvectors.shape == (2, 2, 2)
        assert diag.kpoints.shape == (2, 3)

    def test_eigenvalues_sorted(self) -> None:
        """Verify eigenvalues have ascending order at each k-point.

        The lower band precedes the upper band at every k-point. Downstream
        band analysis depends on this energy order.

        Notes
        -----
        The test diagonalizes the graphene model at three k-points. It compares
        the first eigenvalue with the second eigenvalue at each point.
        """
        i: int

        model: diffpes.types.TBModel
        kpoints: Array
        diag: diffpes.types.DiagonalizedBands

        model = make_graphene_model()
        kpoints = jnp.array(
            [[0.0, 0.0, 0.0], [0.1, 0.2, 0.0], [0.3, 0.1, 0.0]]
        )
        diag = diagonalize_tb(model, kpoints)
        for i in range(3):
            assert float(diag.eigenvalues[i, 0]) <= float(
                diag.eigenvalues[i, 1]
            )

    def test_differentiable(self) -> None:
        """Verify JAX differentiates the eigenvalue sum with respect to hopping parameters.

        The eigenspectrum carries hopping sensitivity through the Hamiltonian
        and the eigendecomposition.

        Notes
        -----
        The test differentiates the eigenvalue sum for a 1D chain with
        ``jax.grad``. It checks every hopping gradient for a finite value.
        """
        model: diffpes.types.TBModel
        kpoints: Array
        grad: Array

        model = make_1d_chain_model(t=-1.0)
        kpoints = jnp.array([[0.25, 0.0, 0.0]])

        def loss(hop):
            m: Array
            d: diffpes.types.DiagonalizedBands

            m = eqx.tree_at(lambda item: item.hopping_params, model, hop)
            d = diagonalize_tb(m, kpoints)
            return jnp.sum(d.eigenvalues)

        grad = jax.grad(loss)(model.hopping_params)
        assert jnp.all(jnp.isfinite(grad))

    def test_eigenvectors_shape_convention(self) -> None:
        """Verify eigenvector indexing: eigenvectors[k, band, orbital].

        The last eigenvector axis contains the orbital coefficients. Each
        vector has unit norm across this axis.

        Notes
        -----
        The test diagonalizes graphene at one k-point. It checks shape
        ``(1, 2, 2)`` and compares each norm with 1.0 at a tolerance of 1e-10.
        """
        model: diffpes.types.TBModel
        kpoints: Array
        diag: diffpes.types.DiagonalizedBands
        norms: Array

        model = make_graphene_model()
        kpoints = jnp.array([[0.0, 0.0, 0.0]])
        diag = diagonalize_tb(model, kpoints)

        assert diag.eigenvectors.shape == (1, 2, 2)

        norms = jnp.sum(jnp.abs(diag.eigenvectors[0]) ** 2, axis=1)
        assert jnp.allclose(norms, 1.0, atol=1e-10)


class TestVaspToDiagonalized:
    """Validate ``vasp_to_diagonalized``.

    The adapter uses VASP eigenvalues and approximates eigenvectors from
    orbital projections. The tests cover shapes, normalization, and the
    policy for phase loss.

    :see: :func:`~diffpes.tightb.vasp_to_diagonalized`
    """

    def test_output_shapes(self) -> None:
        """Verify output shapes match the input band structure dimensions.

        The adapter retains five k-points and three bands. Its eigenvector
        result also contains the three selected orbitals.

        Notes
        -----
        The test supplies projections for two atoms and nine channels. It
        checks eigenvalue shape ``(5, 3)`` and eigenvector shape ``(5, 3, 3)``.
        """
        K: int
        B: int
        A: int
        bands: diffpes.types.BandStructure
        orb_proj: diffpes.types.OrbitalProjection
        basis: diffpes.types.OrbitalBasis
        diag: diffpes.types.DiagonalizedBands

        K, B, A = 5, 3, 2
        bands = make_band_structure(
            eigenvalues=jnp.zeros((K, B)),
            kpoints=jnp.zeros((K, 3)),
        )
        orb_proj = make_orbital_projection(
            projections=jnp.ones((K, B, A, 9)) / 9.0
        )
        basis = make_orbital_basis(
            n_values=(1, 2, 2),
            l_values=(0, 1, 1),
            m_values=(0, 0, 1),
        )
        diag = vasp_to_diagonalized(
            bands, orb_proj, basis, phase_loss="ignore"
        )
        assert diag.eigenvalues.shape == (K, B)
        assert diag.eigenvectors.shape == (K, B, 3)

    def test_eigenvectors_normalized(self) -> None:
        """Verify the VASP projection adapter normalizes approximate eigenvectors.

        The adapter produces unit eigenvector norms from approximate orbital
        weights. This property holds when the raw weights do not sum to one.

        Notes
        -----
        The test supplies s and pz weights of 0.3 and 0.7. It compares every
        squared norm with 1.0 at an absolute tolerance of 1e-10.
        """
        K: int
        B: int
        A: int
        bands: diffpes.types.BandStructure
        proj: Array
        orb_proj: diffpes.types.OrbitalProjection
        basis: diffpes.types.OrbitalBasis
        diag: diffpes.types.DiagonalizedBands
        norms: Array

        K, B, A = 3, 2, 1
        bands = make_band_structure(
            eigenvalues=jnp.zeros((K, B)),
            kpoints=jnp.zeros((K, 3)),
        )
        proj = jnp.zeros((K, B, A, 9))
        proj = proj.at[:, :, :, 0].set(0.3)
        proj = proj.at[:, :, :, 2].set(0.7)
        orb_proj = make_orbital_projection(projections=proj)
        basis = make_orbital_basis(
            n_values=(1, 2),
            l_values=(0, 1),
            m_values=(0, 0),
        )
        diag = vasp_to_diagonalized(
            bands, orb_proj, basis, phase_loss="ignore"
        )
        norms = jnp.sum(jnp.abs(diag.eigenvectors) ** 2, axis=-1)
        assert jnp.allclose(norms, 1.0, atol=1e-10)

    def test_warns_by_default(self) -> None:
        """Verify the default phase_loss policy emits a RuntimeWarning.

        The default policy warns the user because real VASP projections do
        not contain complex eigenvector phases.

        Notes
        -----
        The test calls the adapter with a minimal input and no policy argument.
        It expects a ``RuntimeWarning`` that contains ``cannot recover complex``.
        """
        bands: diffpes.types.BandStructure
        orb_proj: diffpes.types.OrbitalProjection
        basis: diffpes.types.OrbitalBasis

        bands = make_band_structure(
            eigenvalues=jnp.zeros((1, 1)),
            kpoints=jnp.zeros((1, 3)),
        )
        orb_proj = make_orbital_projection(
            projections=jnp.ones((1, 1, 1, 9)) / 9.0
        )
        basis = make_orbital_basis(
            n_values=(1,),
            l_values=(0,),
            m_values=(0,),
        )
        with pytest.warns(RuntimeWarning, match="cannot recover complex"):
            _ = vasp_to_diagonalized(bands, orb_proj, basis)

    def test_phase_loss_error_mode_raises(self) -> None:
        """Verify phase_loss="error" raises ValueError instead of warning.

        The error policy rejects approximate eigenvectors that lack complex
        phase information.

        Notes
        -----
        The test calls the adapter with a minimal input and the error policy.
        It expects a ``ValueError`` that contains ``cannot recover complex``.
        """
        bands: diffpes.types.BandStructure
        orb_proj: diffpes.types.OrbitalProjection
        basis: diffpes.types.OrbitalBasis

        bands = make_band_structure(
            eigenvalues=jnp.zeros((1, 1)),
            kpoints=jnp.zeros((1, 3)),
        )
        orb_proj = make_orbital_projection(
            projections=jnp.ones((1, 1, 1, 9)) / 9.0
        )
        basis = make_orbital_basis(
            n_values=(1,),
            l_values=(0,),
            m_values=(0,),
        )
        with pytest.raises(ValueError, match="cannot recover complex"):
            _ = vasp_to_diagonalized(
                bands, orb_proj, basis, phase_loss="error"
            )

    def test_invalid_phase_loss_raises(self) -> None:
        """Verify an unrecognized phase_loss value raises an error.

        The type contract permits only ``warn``, ``ignore``, and ``error``.

        Notes
        -----
        The test passes ``phase_loss="bad"`` to the adapter with a minimal
        input. It expects the runtime type checker to raise an exception.
        """
        bands: diffpes.types.BandStructure
        orb_proj: diffpes.types.OrbitalProjection
        basis: diffpes.types.OrbitalBasis

        bands = make_band_structure(
            eigenvalues=jnp.zeros((1, 1)),
            kpoints=jnp.zeros((1, 3)),
        )
        orb_proj = make_orbital_projection(
            projections=jnp.ones((1, 1, 1, 9)) / 9.0
        )
        basis = make_orbital_basis(
            n_values=(1,),
            l_values=(0,),
            m_values=(0,),
        )
        with pytest.raises(Exception):
            _ = vasp_to_diagonalized(bands, orb_proj, basis, phase_loss="bad")

    def test_f_orbital_raises(self) -> None:
        """Verify an f-orbital (l=3) in the basis raises ValueError.

        The VASP PROCAR format contains only s, p, and d channels. The adapter
        rejects an f-orbital outside this set.

        Notes
        -----
        The test supplies an orbital with ``l=3`` and ``m=0``. It expects a
        ``ValueError`` that contains ``not in VASP 9-orbital set``.
        """
        bands: diffpes.types.BandStructure
        orb_proj: diffpes.types.OrbitalProjection
        basis: diffpes.types.OrbitalBasis

        bands = make_band_structure(
            eigenvalues=jnp.zeros((1, 1)),
            kpoints=jnp.zeros((1, 3)),
        )
        orb_proj = make_orbital_projection(
            projections=jnp.ones((1, 1, 1, 9)) / 9.0
        )
        basis = make_orbital_basis(
            n_values=(4,),
            l_values=(3,),
            m_values=(0,),
        )
        with pytest.raises(ValueError, match="not in VASP 9-orbital set"):
            _ = vasp_to_diagonalized(
                bands, orb_proj, basis, phase_loss="ignore"
            )
