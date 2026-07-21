"""Diagonalize bands and adapt VASP outputs.

Extended Summary
----------------
The module wraps ``jnp.linalg.eigh`` with vmap over k-points to produce
a ``DiagonalizedBands`` PyTree from a ``TBModel``. Also provides
an adapter to convert VASP ``BandStructure`` + ``OrbitalProjection``
to the same interface.

Routine Listings
----------------
:func:`diagonalize_single_k`
    Diagonalize H(k) at a single k-point.
:func:`diagonalize_tb`
    Diagonalize a TB model at all k-points.
:func:`vasp_to_diagonalized`
    Convert VASP BandStructure + OrbitalProjection to DiagonalizedBands.
"""

import warnings

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Literal
from jaxtyping import Array, Complex, Float, Int, jaxtyped

from diffpes.maths import safe_divide, safe_norm, safe_sqrt
from diffpes.types import (
    PHASE_LOSS_MESSAGE,
    BandStructure,
    DiagonalizedBands,
    OrbitalBasis,
    OrbitalProjection,
    TBModel,
    make_diagonalized_bands,
)

from .hamiltonian import build_hamiltonian_k


@jaxtyped(typechecker=beartype)
def diagonalize_single_k(
    H_k: Complex[Array, "O O"],
) -> tuple[Float[Array, " O"], Complex[Array, "O O"]]:
    """Diagonalize H(k) at a single k-point.

    The function calls the standard LAPACK-style Hermitian eigensolver
    ``jnp.linalg.eigh``. The Hamiltonian construction guarantees Hermitian
    symmetry. Therefore, ``eigh`` produces real eigenvalues and an orthonormal
    eigenvector basis.

    ``jnp.linalg.eigh`` returns eigenvalues in **ascending** order and
    eigenvectors as **columns** of the returned matrix: that is,
    ``eigenvectors[:, i]`` is the eigenvector corresponding to
    ``eigenvalues[i]``.  This column-eigenvector convention is the
    LAPACK/NumPy/JAX standard but differs from some physics textbooks
    that store eigenvectors as rows.  The caller (``diagonalize_tb``)
    transposes to a band-major layout after vmapping.

    This function contains one call to ``eigh``. Tests can isolate or replace
    this function with a custom differentiable eigensolver.

    :see: :class:`~.test_diagonalize.TestDiagonalizeSingleK`

    Parameters
    ----------
    H_k : Complex[Array, "O O"]
        Hermitian Hamiltonian matrix.

    Returns
    -------
    eigenvalues : Float[Array, " O"]
        Eigenvalues in ascending order.
    eigenvectors : Complex[Array, "O O"]
        Eigenvector columns (eigenvectors[:, i] is the i-th).

    Notes
    -----
    JAX provides analytical gradients through ``eigh`` via implicit
    differentiation of the eigenvalue equation, so
    ``jax.grad(lambda p: eigenvalues(p).sum())`` works without additional
    configuration.
    Degenerate eigenvalues can cause numerical instability in the
    backward pass; this is a known JAX limitation.
    """
    eigenvalues: Float[Array, " O"]
    eigenvectors: Complex[Array, "O O"]
    eigenvalues, eigenvectors = jnp.linalg.eigh(H_k)
    result: tuple[Float[Array, " O"], Complex[Array, "O O"]] = (
        eigenvalues,
        eigenvectors,
    )
    return result


@jaxtyped(typechecker=beartype)
def diagonalize_tb(
    tb_model: TBModel,
    kpoints: Float[Array, "K 3"],
) -> DiagonalizedBands:
    """Diagonalize a TB model at all k-points.

    The function builds H(k) for each k-point and calls ``jnp.linalg.eigh``.
    ``jax.vmap`` vectorizes both operations. JAX differentiates them with
    respect to ``tb_model.hopping_params``.

    The internal ``_build_and_diag`` closure captures the model parameters. It
    maps one k-point to its eigenvalues and eigenvectors. ``jax.vmap`` applies
    this closure along the leading k-point axis. One vectorized call computes
    the complete band structure without Python loops.

    After vmapping, the eigenvector array has shape ``(K, O, O)`` in
    the column-eigenvector convention of ``eigh``. Thus,
    ``evecs[k, :, b]`` is band ``b`` at k-point ``k``).  A transpose
    ``(0, 2, 1)`` converts this to the band-major convention
    ``(K, B, O)`` where ``evecs[k, b, :]`` gives the orbital
    coefficients of band ``b`` at k-point ``k``.  This layout matches
    the ``DiagonalizedBands.eigenvectors`` contract used by the rest
    of the ARPYES pipeline (projections, matrix elements, spectral
    function).

    The function sets the Fermi energy to 0.0 because bare tight-binding models
    have no absolute energy reference.

    :see: :class:`~.test_diagonalize.TestDiagonalizeTB`

    Parameters
    ----------
    tb_model : TBModel
        Tight-binding model.
    kpoints : Float[Array, "K 3"]
        k-points in fractional coordinates.

    Returns
    -------
    bands : DiagonalizedBands
        Diagonalized electronic structure.

    Notes
    -----
    Because ``jax.vmap`` traces the function once and broadcasts over
    the batch dimension, the number of k-points does not affect
    compilation time -- only runtime.  The full forward + backward
    pass (eigenvalues and their gradients with respect to hopping
    parameters) is differentiable end-to-end.
    """

    def _build_and_diag(k: Float[Array, " 3"]) -> tuple:
        H: Complex[Array, "O O"] = build_hamiltonian_k(
            k,
            tb_model.hopping_params,
            tb_model.hopping_indices,
            tb_model.n_orbitals,
            tb_model.lattice_vectors,
        )
        evals: Float[Array, " O"]
        evecs: Complex[Array, "O O"]
        evals, evecs = diagonalize_single_k(H)
        eigensystem: tuple[Float[Array, " O"], Complex[Array, "O O"]] = (
            evals,
            evecs,
        )
        return eigensystem

    eigenvalues: Float[Array, "K O"]
    eigenvectors: Complex[Array, "K O O"]
    eigenvalues, eigenvectors = jax.vmap(_build_and_diag)(kpoints)
    eigenvectors = jnp.transpose(eigenvectors, (0, 2, 1))

    bands: DiagonalizedBands = make_diagonalized_bands(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        kpoints=kpoints,
        fermi_energy=0.0,
    )
    return bands


@jaxtyped(typechecker=beartype)
def vasp_to_diagonalized(
    bands: BandStructure,
    orb_proj: OrbitalProjection,
    orbital_basis: OrbitalBasis,
    phase_loss: Literal["warn", "ignore", "error"] = "warn",
) -> DiagonalizedBands:
    """Convert VASP BandStructure + OrbitalProjection to DiagonalizedBands.

    VASP PROCAR gives ``|c_{k,b,orb}|^2``, not the complex
    coefficients. This adapter uses ``sqrt(|c|^2)`` with positive
    sign as an approximation. Phase information is lost.

    The adapter sums the orbital projections over atoms. It then maps them to
    the orbital basis order.

    VASP's PROCAR file provides site- and orbital-projected squared
    moduli ``|c_{k,b,atom,orb}|^2`` for each eigenstate. The VASP projection
    discards the complex phase of each coefficient. Therefore, PROCAR data
    cannot recover the true complex eigenvectors. This adapter takes
    ``sqrt(|c|^2)`` with a positive sign. The approximation produces real,
    nonnegative coefficients.

    The approximation is exact for an observable that depends only on the
    coefficient modulus. It introduces errors when an observable depends on
    relative orbital phases. Photoemission interference terms have this phase
    dependence.

    **Orbital mapping.** VASP stores projections in a fixed 9-orbital
    ordering for the s, p, and d channels::

        [s, py, pz, px, dxy, dyz, dz2, dxz, dx2 - y2]

    This order differs from some standard orders, such as the ``(l, m)``
    convention with m running from -l to +l).  The function maps from
    ``(l, m)`` quantum numbers in the ``OrbitalBasis`` to VASP's
    9-orbital column index via a lookup table.

    **Sum the atoms.** Before the orbital mapping, the adapter sums the
    projections over atom axis 2, which changes
    ``(K, B, A, 9) -> (K, B, 9)``.  This is correct when the
    ``OrbitalBasis`` describes a single composite orbital per
    ``(l, m)`` channel rather than per-atom resolution.

    **Normalize the vectors.** After the square root, the adapter normalizes
    each k-point and band eigenvector. Therefore,
    ``sum_orb |c_{k,b,orb}|^2 = 1``. A safe division selects a zero vector and
    zero gradient when all projections equal zero.

    :see: :class:`~.test_diagonalize.TestVaspToDiagonalized`

    Parameters
    ----------
    bands : BandStructure
        VASP eigenvalues and k-points.
    orb_proj : OrbitalProjection
        VASP orbital projections of shape (K, B, A, 9).
    orbital_basis : OrbitalBasis
        Quantum number metadata defining which VASP orbital
        indices to use.
    phase_loss : Literal["warn", "ignore", "error"]
        Policy for handling lost phase information:
        - ``"warn"`` (default): emit a runtime warning.
        - ``"ignore"``: proceed without a warning.
        - ``"error"``: raise ``ValueError`` and abort.

    Returns
    -------
    diag : DiagonalizedBands
        Approximate diagonalized bands.

    Raises
    ------
    ValueError
        If ``phase_loss`` is invalid, is ``"error"``, or the orbital basis
        contains a channel outside the VASP nine-orbital set.

    Notes
    -----
    The VASP 9-orbital ordering used here covers s, p, and d channels
    only. The adapter does not support f-orbital projections and raises
    ``ValueError`` for them. A PROCAR file from ``LORBIT=11`` or higher
    contains all nine channels. ``LORBIT=10`` can omit the decomposition by m
    and does not support this adapter.

    The adapter converts the resulting eigenvectors to ``complex128`` for the
    ``DiagonalizedBands`` type. Their imaginary parts remain zero.
    """
    i: int

    if phase_loss not in ("warn", "ignore", "error"):  # pragma: no cover
        msg: str = (
            "phase_loss must be one of {'warn', 'ignore', 'error'}, "
            f"got '{phase_loss}'"
        )
        raise ValueError(msg)

    if phase_loss == "error":
        raise ValueError(PHASE_LOSS_MESSAGE)
    if phase_loss == "warn":
        warnings.warn(PHASE_LOSS_MESSAGE, RuntimeWarning, stacklevel=2)

    proj_summed: Float[Array, "K B 9"] = jnp.sum(orb_proj.projections, axis=2)

    vasp_lm_to_idx: dict[tuple[int, int], int] = {
        (0, 0): 0,
        (1, -1): 1,
        (1, 0): 2,
        (1, 1): 3,
        (2, -2): 4,
        (2, -1): 5,
        (2, 0): 6,
        (2, 1): 7,
        (2, 2): 8,
    }

    n_orbs: int = len(orbital_basis.l_values)
    orbital_indices: list[int] = []
    for i in range(n_orbs):
        l_val: int = orbital_basis.l_values[i]
        m_val: int = orbital_basis.m_values[i]
        idx: int | None = vasp_lm_to_idx.get((l_val, m_val))
        if idx is None:
            msg = f"Orbital (l={l_val}, m={m_val}) not in VASP 9-orbital set"
            raise ValueError(msg)
        orbital_indices.append(idx)

    idx_arr: Int[Array, " N"] = jnp.array(orbital_indices)
    approx_c2: Float[Array, "K B N"] = proj_summed[:, :, idx_arr]
    approx_c: Float[Array, "K B N"] = safe_sqrt(approx_c2)

    norm: Float[Array, "K B 1"] = safe_norm(approx_c, axis=-1, keepdims=True)
    normalized: Float[Array, "K B N"] = safe_divide(approx_c, norm)
    eigenvectors: Complex[Array, "K B N"] = normalized.astype(jnp.complex128)

    diag: DiagonalizedBands = make_diagonalized_bands(
        eigenvalues=bands.eigenvalues,
        eigenvectors=eigenvectors,
        kpoints=bands.kpoints,
        fermi_energy=bands.fermi_energy,
    )
    return diag


__all__: list[str] = [
    "diagonalize_single_k",
    "diagonalize_tb",
    "vasp_to_diagonalized",
]
