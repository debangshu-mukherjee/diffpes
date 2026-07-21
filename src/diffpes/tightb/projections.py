"""Convert eigenvectors to orbital weights.

Extended Summary
----------------
The module extracts orbital weights and coefficients from diagonalized band
structures. These functions convert complex tight-binding eigenvectors into
ARPES observables. The observables include orbital-resolved spectral weights
and complex amplitudes for coherent photoemission matrix elements.

Routine Listings
----------------
:func:`eigenvector_orbital_weights`
    Compute orbital weights from eigenvectors.
:func:`orbital_coefficients`
    Return the raw complex orbital coefficients.
"""

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Complex, Float, jaxtyped


@jaxtyped(typechecker=beartype)
def eigenvector_orbital_weights(
    eigenvectors: Complex[Array, "K B O"],
) -> Float[Array, "K B O"]:
    r"""Compute orbital weights from eigenvectors.

    For each eigenstate ``|psi_{k,b}>`` expanded in the orbital basis

    .. math::

        |\psi_{k,b}\rangle = \sum_o c_{k,b,o} |o\rangle,

    the orbital weight of orbital ``o`` in band ``b`` at k-point ``k``
    is the squared modulus of the expansion coefficient:

    .. math::

        w_{k,b,o} = |c_{k,b,o}|^2.

    This is the probability of finding the electron in orbital ``o``
    given that it occupies eigenstate ``(k, b)``.  By construction,
    normalized eigenvectors give weights that sum to 1 over orbitals for each
    ``(k, b)`` pair.

    Fat-band plots and orbital-resolved DOS use orbital weights. Photoemission
    matrix element computations also start from these weights. These
    computations also need the complex coefficients; see
    ``orbital_coefficients``.

    :see: :class:`~.test_projections.TestEigenvectorOrbitalWeights`

    Parameters
    ----------
    eigenvectors : Complex[Array, "K B O"]
        Complex orbital coefficients c_{k,b,orb}.

    Returns
    -------
    weights : Float[Array, "K B O"]
        ``|c_{k,b,orb}|^2`` per orbital.

    Notes
    -----
    The implementation uses ``jnp.abs(eigenvectors) ** 2`` rather than
    ``(eigenvectors * eigenvectors.conj()).real`` for clarity.  Both
    expressions are mathematically identical for complex arrays and
    produce the same JAX trace; the former is marginally more readable.
    """
    weights: Float[Array, "K B O"] = jnp.abs(eigenvectors) ** 2
    return weights


@jaxtyped(typechecker=beartype)
def orbital_coefficients(
    eigenvectors: Complex[Array, "K B O"],
) -> Complex[Array, "K B O"]:
    """Return the raw complex orbital coefficients.

    A full matrix element computation needs the complex coefficients
    c_{k,b,orb}, not only ``|c|^2``.

    This is an **identity function**: it returns its input unchanged.
    Its purpose gives a clear name to each call site. A pipeline can need both
    ``eigenvector_orbital_weights`` and the raw coefficients.
    Callers can write::

        weights = eigenvector_orbital_weights(evecs)
        coeffs = orbital_coefficients(evecs)

    Thus, each downstream path clearly identifies whether it uses only
    magnitudes or the full phase information.

    In the Chinook-style matrix element computation, the function multiplies
    each complex coefficient by the applicable one-electron dipole matrix
    element. It then adds the products coherently. Interference between
    orbital channels depends on the relative phases of these coefficients.
    Therefore, this computation needs the complex values, not only ``|c|^2``.

    When ``vasp_to_diagonalized`` supplies the eigenvectors, the adapter loses
    the phases. The adapter then returns real, nonnegative coefficients. In
    that regime, the coherent interference terms are approximate.

    :see: :class:`~.test_projections.TestOrbitalCoefficients`

    Parameters
    ----------
    eigenvectors : Complex[Array, "K B O"]
        Complex orbital coefficients.

    Returns
    -------
    coefficients : Complex[Array, "K B O"]
        Same as input (identity, for API clarity).

    Notes
    -----
    Because this is an identity, ``jax.grad`` through
    ``orbital_coefficients`` adds zero overhead -- the function
    compiles away entirely.
    """
    coefficients: Complex[Array, "K B O"] = eigenvectors
    return coefficients


__all__: list[str] = [
    "eigenvector_orbital_weights",
    "orbital_coefficients",
]
