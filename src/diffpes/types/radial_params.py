"""Radial wavefunction parameter data structures.

Extended Summary
----------------
Defines PyTree types for orbital basis metadata and Slater-type
radial wavefunction parameters used by the differentiable dipole
matrix element pipeline.

Routine Listings
----------------
:class:`OrbitalBasis`
    PyTree for orbital quantum number metadata.
:class:`SlaterParams`
    PyTree for Slater radial wavefunction parameters.
:func:`make_orbital_basis`
    Create a validated ``OrbitalBasis`` instance.
:func:`make_slater_params`
    Create a validated ``SlaterParams`` instance.

Notes
-----
``OrbitalBasis`` is purely static (all auxiliary data) because the
quantum numbers (n, l, m) control code paths (recurrence depths in
spherical Bessel functions and associated Legendre polynomials).
``SlaterParams`` wraps differentiable Slater exponents alongside
the static orbital basis.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional
from jaxtyping import Array, Float, jaxtyped


class OrbitalBasis(eqx.Module):
    """PyTree for orbital quantum number metadata.

    Extended Summary
    ----------------
    Describes the orbital basis set used in dipole matrix element
    calculations for the differentiable Chinook pipeline. The quantum
    numbers (n, l, m) parameterize the radial wavefunctions (via
    Slater-type orbitals) and angular parts (spherical harmonics) that
    enter the photoemission matrix element.

    All fields are static (auxiliary data) because quantum numbers
    control code paths: they determine recurrence depths in spherical
    Bessel functions and associated Legendre polynomials, and they
    index into the Gaunt coefficient table. Changing any quantum
    number alters the computational graph structure, so JAX must
    recompile when these values change.

    Attributes
    ----------
    n_values : tuple[int, ...]
        Principal quantum numbers, one per orbital. Determines the
        radial node count and the power of *r* in the Slater-type
        radial function R_nl(r) ~ r^{n-1} exp(-zeta*r). Static
        (auxiliary data, not JAX-traced).
    l_values : tuple[int, ...]
        Angular momentum quantum numbers, one per orbital (0=s, 1=p,
        2=d, 3=f). Determines the spherical harmonic Y_l^m used in
        the matrix element integral. Static (auxiliary data).
    m_values : tuple[int, ...]
        Magnetic quantum numbers, one per orbital. Ranges from -l to
        +l for each orbital. Selects the specific spherical harmonic
        component. Static (auxiliary data).
    labels : tuple[str, ...]
        Human-readable orbital labels (e.g. ``("2s", "2px", ...)``).
        Used for plotting and debugging. Static (auxiliary data).

    Notes
    -----
    Implemented as an immutable :class:`equinox.Module` PyTree.
    All fields are auxiliary data (no JAX array children) because
    changing any quantum number changes the computational graph and
    requires JIT recompilation. The children tuple is always empty.

    See Also
    --------
    SlaterParams : Wraps differentiable Slater exponents alongside
        this static basis metadata.
    make_orbital_basis : Factory function with length validation and
        default label generation.
    """

    n_values: tuple[int, ...] = eqx.field(static=True)
    l_values: tuple[int, ...] = eqx.field(static=True)
    m_values: tuple[int, ...] = eqx.field(static=True)
    labels: tuple[str, ...] = eqx.field(static=True)


class SlaterParams(eqx.Module):
    """PyTree for Slater radial wavefunction parameters.

    Extended Summary
    ----------------
    Wraps differentiable Slater exponents and linear combination
    coefficients alongside the static orbital basis metadata.
    The Slater exponents (zeta) are the primary quantities to be
    optimized in inverse fitting workflows -- ``jax.grad`` with
    respect to ``zeta`` gives the gradient of the simulated ARPES
    intensity with respect to the radial wavefunction shape.

    In a multi-zeta basis (C > 1), each orbital's radial function
    is expressed as a linear combination of C Slater-type functions
    with different exponents, weighted by the ``coefficients``
    matrix. For single-zeta (C=1), each orbital has exactly one
    exponent and a coefficient of 1.0.

    Attributes
    ----------
    zeta : Float[Array, " O"]
        Slater exponents (inverse Bohr), one per orbital. Controls
        the radial decay rate of each Slater-type orbital
        R(r) ~ r^{n-1} exp(-zeta * r). JAX-traced (differentiable)
        -- this is the primary optimization target for inverse
        fitting.
    coefficients : Float[Array, "O C"]
        Linear combination coefficients for multi-zeta expansion.
        Shape is (O, C) where O is the number of orbitals and C is
        the contraction length. For single-zeta basis sets, C=1 and
        all coefficients are 1.0. JAX-traced (differentiable).
    orbital_basis : OrbitalBasis
        Quantum numbers (n, l, m) and labels for each orbital.
        Static (auxiliary data) -- changing quantum numbers
        triggers JIT recompilation.

    Notes
    -----
    Implemented as an immutable :class:`equinox.Module` PyTree.
    ``zeta`` and ``coefficients`` are JAX array children (on the
    gradient tape), while ``orbital_basis`` is auxiliary data
    (static at trace time). This separation means that
    ``jax.grad(loss)(slater_params)`` differentiates through the
    exponents and coefficients while keeping the quantum number
    structure fixed.

    See Also
    --------
    OrbitalBasis : The static quantum number metadata nested inside.
    make_slater_params : Factory function with validation and default
        coefficient generation.
    """

    zeta: Float[Array, " O"]
    coefficients: Float[Array, "O C"]
    orbital_basis: OrbitalBasis = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def make_orbital_basis(
    n_values: tuple[int, ...],
    l_values: tuple[int, ...],
    m_values: tuple[int, ...],
    labels: Optional[tuple[str, ...]] = None,
) -> OrbitalBasis:
    """Create a validated ``OrbitalBasis`` instance.

    Extended Summary
    ----------------
    Factory function that validates quantum number tuples and
    constructs an ``OrbitalBasis`` PyTree. The three quantum number
    tuples must all have the same length (one entry per orbital).
    If ``labels`` is not provided, generic labels ``"orb_0"``,
    ``"orb_1"``, ... are generated automatically.

    Use this factory instead of the raw ``OrbitalBasis`` constructor
    to get automatic length validation and default label generation.

    Implementation Logic
    --------------------
    1. **Infer orbital count** from ``len(n_values)``.
    2. **Validate lengths**: raise ``ValueError`` if ``l_values`` or
       ``m_values`` differ in length from ``n_values``.
    3. **Default labels**: if ``labels`` is ``None``, generate
       ``("orb_0", "orb_1", ...)`` with length matching the orbital
       count.
    4. **Validate label length**: raise ``ValueError`` if ``labels``
       has a different length from the quantum number tuples.
    5. **Construct** the ``OrbitalBasis`` Equinox module and return it.

    Parameters
    ----------
    n_values : tuple[int, ...]
        Principal quantum numbers, one per orbital.
    l_values : tuple[int, ...]
        Angular momentum quantum numbers, one per orbital.
    m_values : tuple[int, ...]
        Magnetic quantum numbers, one per orbital.
    labels : tuple[str, ...], optional
        Human-readable orbital labels. Defaults to
        ``("orb_0", "orb_1", ...)``.

    Returns
    -------
    basis : OrbitalBasis
        Validated orbital basis with consistent lengths.

    Raises
    ------
    ValueError
        If ``n_values``, ``l_values``, and ``m_values`` do not all
        have the same length, or if ``labels`` has a mismatched
        length; if any principal quantum number is less than one; if
        any angular quantum number is outside ``0 <= l < n``; or if
        any magnetic quantum number is outside ``-l <= m <= l``.

    See Also
    --------
    OrbitalBasis : The PyTree class constructed by this factory.
    """
    n_orbitals: int = len(n_values)
    if len(l_values) != n_orbitals or len(m_values) != n_orbitals:
        msg: str = "n_values, l_values, m_values must have the same length"
        raise ValueError(msg)
    if labels is None:
        labels = tuple(f"orb_{i}" for i in range(n_orbitals))
    if len(labels) != n_orbitals:
        msg: str = "labels must have the same length as quantum numbers"
        raise ValueError(msg)
    if any(n < 1 for n in n_values):
        msg = "n_values must all be at least 1"
        raise ValueError(msg)
    if any(
        angular < 0 or angular >= principal
        for principal, angular in zip(n_values, l_values, strict=True)
    ):
        msg = "l_values must satisfy 0 <= l < n"
        raise ValueError(msg)
    if any(
        abs(magnetic) > angular
        for angular, magnetic in zip(l_values, m_values, strict=True)
    ):
        msg = "m_values must satisfy abs(m) <= l"
        raise ValueError(msg)
    basis: OrbitalBasis = OrbitalBasis(
        n_values=n_values,
        l_values=l_values,
        m_values=m_values,
        labels=labels,
    )
    return basis


@jaxtyped(typechecker=beartype)
def make_slater_params(
    zeta: Float[Array, " Oz"],
    orbital_basis: OrbitalBasis,
    coefficients: Optional[Float[Array, "Oc C"]] = None,
) -> SlaterParams:
    """Create a validated ``SlaterParams`` instance.

    Extended Summary
    ----------------
    Factory function that validates Slater radial wavefunction
    parameters and constructs a ``SlaterParams`` PyTree. The ``zeta``
    array length must match the number of orbitals in the provided
    ``orbital_basis``. If ``coefficients`` is not provided, a
    single-zeta basis is assumed and a column of ones is created
    (shape ``(O, 1)``).

    Use this factory instead of the raw ``SlaterParams`` constructor
    to get automatic size validation, default coefficient generation,
    and guaranteed ``float64`` dtypes.

    Implementation Logic
    --------------------
    1. **Cast zeta** to ``jnp.float64`` via ``jnp.asarray``.
    2. **Infer orbital count** from ``zeta_arr.shape[0]``.
    3. **Validate consistency**: raise ``ValueError`` if the orbital
       count from ``zeta`` does not match
       ``len(orbital_basis.n_values)``.
    4. **Default coefficients**: if ``coefficients`` is ``None``,
       create ``jnp.ones((O, 1), dtype=jnp.float64)`` for a
       single-zeta expansion. Otherwise cast the provided array to
       ``jnp.float64``.
    5. **Construct** the ``SlaterParams`` Equinox module and return it.

    Parameters
    ----------
    zeta : Float[Array, " O"]
        Slater exponents (inverse Bohr), one per orbital.
    orbital_basis : OrbitalBasis
        Orbital quantum number metadata. Must have the same number
        of orbitals as the length of ``zeta``.
    coefficients : Float[Array, "O C"], optional
        Multi-zeta linear combination coefficients. Shape ``(O, C)``
        where C is the contraction length. Defaults to ones with
        C=1 (single-zeta basis).

    Returns
    -------
    params : SlaterParams
        Validated Slater parameters with ``float64`` arrays.

    Raises
    ------
    ValueError
        If the length of ``zeta`` does not match the number of
        orbitals in ``orbital_basis``, or if the first coefficient
        dimension does not match that orbital count.
    EquinoxRuntimeError
        If ``zeta`` is non-finite or non-positive, or if
        ``coefficients`` contains a non-finite value.

    See Also
    --------
    SlaterParams : The PyTree class constructed by this factory.
    make_orbital_basis : Factory for the ``OrbitalBasis`` argument.
    """
    zeta_arr: Float[Array, " O"] = jnp.asarray(zeta, dtype=jnp.float64)
    n_orbitals: int = zeta_arr.shape[0]
    if len(orbital_basis.n_values) != n_orbitals:
        msg: str = "zeta length must match orbital_basis size"
        raise ValueError(msg)
    if coefficients is None:
        coeff_arr: Float[Array, "O C"] = jnp.ones(
            (n_orbitals, 1), dtype=jnp.float64
        )
    else:
        coeff_arr = jnp.asarray(coefficients, dtype=jnp.float64)
    if coeff_arr.shape[0] != n_orbitals:
        msg = "coefficients first dimension must match orbital_basis size"
        raise ValueError(msg)

    def validate_and_create() -> SlaterParams:
        nonlocal zeta_arr
        zeta_arr = eqx.error_if(
            zeta_arr,
            ~(jnp.all(jnp.isfinite(zeta_arr))),
            "make_slater_params: zeta finite",
        )
        zeta_arr = eqx.error_if(
            zeta_arr,
            ~(jnp.all(zeta_arr > 0.0)),
            "make_slater_params: zeta positive",
        )
        coeff_arr_checked: Float[Array, "O C"] = eqx.error_if(
            coeff_arr,
            ~(jnp.all(jnp.isfinite(coeff_arr))),
            "make_slater_params: coefficients finite",
        )
        return SlaterParams(
            zeta=zeta_arr,
            coefficients=coeff_arr_checked,
            orbital_basis=orbital_basis,
        )

    params: SlaterParams = validate_and_create()
    return params


__all__: list[str] = [
    "OrbitalBasis",
    "SlaterParams",
    "make_orbital_basis",
    "make_slater_params",
]
