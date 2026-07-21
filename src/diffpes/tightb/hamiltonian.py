"""Build tight-binding Hamiltonians in JAX.

Extended Summary
----------------
The module provides a minimal Slater-Koster Hamiltonian builder and convenience
functions for creating test models (graphene, 1D chain). The
Hamiltonian is fully JAX-traceable so that ``jax.grad`` can
differentiate eigenvalues with respect to hopping parameters.

Routine Listings
----------------
:func:`build_hamiltonian_k`
    Build the Bloch Hamiltonian H(k) at a single k-point.
"""

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Complex, Float, jaxtyped


@jaxtyped(typechecker=beartype)
def build_hamiltonian_k(
    k: Float[Array, " 3"],
    hopping_params: Float[Array, " H"],
    hopping_indices: tuple,
    n_orbitals: int,
    lattice_vectors: Float[Array, "3 3"],  # noqa: ARG001
) -> Complex[Array, "O O"]:
    r"""Build the Bloch Hamiltonian H(k) at a single k-point.

    .. math::

        H_{ij}(\mathbf{k}) = \sum_{\mathbf{R}} t_{ij,\mathbf{R}}
            \exp(i \mathbf{k} \cdot \mathbf{R})

    where the sum runs over lattice vectors R defined by the
    hopping indices. The function makes the result Hermitian:
    ``H = (H_raw + H_raw^dag) / 2``.

    The function computes the Bloch sum entirely in fractional coordinates.
    ``k`` uses reciprocal lattice units, and each hopping ``R`` uses direct
    lattice units. Therefore, ``k_frac . R_frac`` includes the metric. The
    Cartesian phase equals
    ``exp(2 pi i k_frac . R_frac)``. This stage does not need the lattice
    matrix. The model retains it for Cartesian conversions.

    The function iterates over each hopping in Python because the hopping count
    is small and static. For each hopping, it computes the Bloch phase. It adds
    the corresponding amplitude to ``H[orb_i, orb_j]``.

    After the accumulation, the function applies ``(H + H^dag) / 2``. A
    user-supplied hopping list can contain only the upper triangle. This
    operation fills the lower triangle symmetrically. It also corrects
    floating-point rounding that can prevent ``jnp.linalg.eigh`` from working.

    :see: :class:`~.test_hamiltonian.TestBuildHamiltonianK`

    Parameters
    ----------
    k : Float[Array, " 3"]
        k-point in fractional coordinates.
    hopping_params : Float[Array, " H"]
        Hopping amplitudes (differentiable).
    hopping_indices : tuple
        ``(orb_i, orb_j, (R_x, R_y, R_z))`` per hopping.
    n_orbitals : int
        Number of orbitals in the unit cell.
    lattice_vectors : Float[Array, "3 3"]
        Real-space lattice vectors (rows).

    Returns
    -------
    H_k : Complex[Array, "O O"]
        Hermitian Hamiltonian matrix.

    Notes
    -----
    The hopping amplitudes ``hopping_params`` are plain real floats.
    Complex spin-orbit hoppings require ``complex128`` values and no automatic
    Hermitian completion.

    The ``@jaxtyped`` and ``@beartype`` decorators detect shape and dtype
    errors when the caller invokes the function.
    """
    h_idx: int
    orb_i: int
    orb_j: int
    R_ijk: Array

    H: Complex[Array, "O O"] = jnp.zeros(
        (n_orbitals, n_orbitals), dtype=jnp.complex128
    )

    for h_idx, (orb_i, orb_j, R_ijk) in enumerate(hopping_indices):
        t: Float[Array, " "] = hopping_params[h_idx]
        R_frac: Float[Array, " 3"] = jnp.array(R_ijk, dtype=jnp.float64)
        phase: Complex[Array, " "] = jnp.exp(2j * jnp.pi * jnp.dot(k, R_frac))
        H = H.at[orb_i, orb_j].add(t * phase)

    H = (H + H.conj().T) / 2.0
    H_k: Complex[Array, "O O"] = H
    return H_k


__all__: list[str] = [
    "build_hamiltonian_k",
]
