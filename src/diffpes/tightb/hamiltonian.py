"""Tight-binding Hamiltonian builder in JAX.

Extended Summary
----------------
Provides a minimal Slater-Koster Hamiltonian builder and convenience
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
    hopping indices. The result is Hermitianized:
    ``H = (H_raw + H_raw^dag) / 2``.

    Extended Summary
    ----------------
    The Bloch sum is evaluated entirely in **fractional coordinates**.
    Because ``k`` is given in units of the reciprocal lattice vectors
    and ``R`` (encoded in each hopping triple) is given in units of the
    direct lattice vectors, the dot product ``k_frac . R_frac`` already
    absorbs the metric: the Cartesian phase would be
    ``exp(i k_cart . R_cart) = exp(2 pi i k_frac . R_frac)``, so the
    lattice-vector matrix itself is never needed at this stage (it is
    stored in the model for downstream Cartesian conversions).

    The function iterates over every hopping entry in Python (not a
    JAX scan) because the number of hoppings is typically small and
    known at trace time.  For each hopping ``(orb_i, orb_j, R_ijk)``
    the Bloch phase ``exp(2 pi i k . R)`` is computed and the
    corresponding hopping amplitude is accumulated into entry
    ``H[orb_i, orb_j]``.

    After all hoppings are accumulated the matrix is explicitly
    Hermitianized via ``(H + H^dag) / 2``.  This is necessary because
    the user-supplied hopping list may only contain the upper triangle
    (e.g. only A -> B hops); the Hermitianization symmetrically fills
    the lower triangle and also corrects floating-point round-off that
    would otherwise break ``jnp.linalg.eigh``.

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
    Complex (spin-orbit) hoppings would require promoting them to
    ``complex128`` and removing the Hermitianization short-cut.

    Because the function is decorated with ``@jaxtyped`` /
    ``@beartype``, shape and dtype errors are caught at call time
    rather than deep inside the JAX trace.
    """
    H: Complex[Array, "O O"] = jnp.zeros(
        (n_orbitals, n_orbitals), dtype=jnp.complex128
    )

    for h_idx, (orb_i, orb_j, R_ijk) in enumerate(hopping_indices):
        t: Float[Array, " "] = hopping_params[h_idx]
        R_frac: Float[Array, " 3"] = jnp.array(R_ijk, dtype=jnp.float64)
        # Bloch phase with fractional coordinates:
        # exp(2πi k_frac · R_frac), since k·R = 2π k_frac·R_frac
        phase: Complex[Array, " "] = jnp.exp(2j * jnp.pi * jnp.dot(k, R_frac))
        H = H.at[orb_i, orb_j].add(t * phase)

    # Hermitianize
    H = (H + H.conj().T) / 2.0
    H_k: Complex[Array, "O O"] = H
    return H_k


__all__: list[str] = [
    "build_hamiltonian_k",
]
