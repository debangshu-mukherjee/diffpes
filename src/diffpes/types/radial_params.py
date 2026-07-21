"""Define radial-wavefunction parameter structures.

Extended Summary
----------------
This module defines PyTree types for orbital basis metadata and Slater-type
radial wavefunction parameters used by the differentiable dipole
matrix element pipeline.

Routine Listings
----------------
:class:`OrbitalBasis`
    Store orbital quantum-number metadata in a JAX PyTree.
:class:`SlaterParams`
    Store Slater radial-wavefunction parameters in a JAX PyTree.
:func:`make_orbital_basis`
    Create a validated ``OrbitalBasis`` instance.
:func:`make_slater_params`
    Create a validated ``SlaterParams`` instance.

Notes
-----
``OrbitalBasis`` contains only static auxiliary data. The quantum numbers
(n, l, m) control recurrence depths in spherical Bessel functions and
associated Legendre polynomials.
``SlaterParams`` wraps differentiable Slater exponents alongside
the static orbital basis.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional
from jaxtyping import Array, Float, jaxtyped


class OrbitalBasis(eqx.Module):
    """Store orbital quantum-number metadata in a JAX PyTree.

    This type describes the orbital basis for dipole matrix-element
    calculations for the differentiable Chinook pipeline. The quantum
    numbers (n, l, m) parameterize the radial wavefunctions (via
    Slater-type orbitals) and angular parts (spherical harmonics) that
    enter the photoemission matrix element.

    All fields contain static auxiliary data because quantum numbers control
    code paths. They determine recurrence depths in spherical Bessel functions
    and associated Legendre polynomials. They also index the Gaunt coefficient
    table. A quantum-number change alters the computational graph. JAX must
    therefore recompile after this change.


    :see: :class:`~.test_radial_params.TestOrbitalBasis`

    Attributes
    ----------
    n_values : tuple[int, ...]
        Principal quantum numbers, one per orbital. Each value determines the
        radial node count and the power of *r* in the Slater-type radial
        function R_nl(r) ~ r^{n-1} exp(-zeta*r). These values are static.
    l_values : tuple[int, ...]
        Angular momentum quantum numbers, one per orbital (0=s, 1=p,
        2=d, 3=f). Determines the spherical harmonic Y_l^m used in
        the matrix element integral (**static** -- compile-time constants;
        changing them triggers retracing).
    m_values : tuple[int, ...]
        Magnetic quantum numbers, one per orbital. Ranges from -l to
        +l for each orbital. Selects the specific spherical harmonic
        component (**static** -- compile-time constants; changing them
        triggers retracing).
    labels : tuple[str, ...]
        Human-readable orbital labels (e.g. ``("2s", "2px", ...)``).
        Used for plotting and debugging (**static** -- compile-time constants;
        changing them triggers retracing).

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
    """Store Slater radial-wavefunction parameters in a JAX PyTree.

    This type contains differentiable Slater exponents and linear-combination
    coefficients alongside the static orbital basis metadata.
    Inverse fitting workflows optimize the Slater exponents (zeta).
    ``jax.grad`` gives the gradient of the simulated ARPES intensity with
    respect to the radial wavefunction shape.

    In a multi-zeta basis (C > 1), the ``coefficients`` matrix combines C
    Slater-type functions for each orbital. These functions have different
    exponents. In a single-zeta basis (C=1), each orbital has one exponent and
    a coefficient of 1.0.


    :see: :class:`~.test_radial_params.TestSlaterParams`

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
        **Static** -- a compile-time constant; changing it triggers retracing.

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

    The factory validates quantum number tuples and
    constructs an ``OrbitalBasis`` PyTree. The three quantum number
    tuples must all have the same length (one entry per orbital).
    If ``labels`` is absent, the factory generates generic labels such as
    ``"orb_0"`` and ``"orb_1"``.

    Use this factory instead of the raw ``OrbitalBasis`` constructor
    to get automatic length validation and default label generation.

    :see: :class:`~.test_radial_params.TestMakeOrbitalBasis`

    Implementation Logic
    --------------------
    1. **Prepare the normalized values**::

           n_orbitals = len(n_values)

       This expression gives the later validation steps a stable shape and
       dtype.

    2. **Apply static validation**::

           len(l_values) != n_orbitals or len(m_values) != n_orbitals

       This predicate rejects invalid structure before JAX traces the
       numerical checks.

    3. **Return the named instance**::

           return basis

       The explicit name keeps the implementation and the Returns section
       synchronized.

    Parameters
    ----------
    n_values : tuple[int, ...]
        Principal quantum numbers (**static** -- compile-time constants;
        changing them triggers retracing), one per orbital.
    l_values : tuple[int, ...]
        Angular momentum quantum numbers (**static** -- compile-time
        constants; changing them triggers retracing), one per orbital.
    m_values : tuple[int, ...]
        Magnetic quantum numbers (**static** -- compile-time constants;
        changing them triggers retracing), one per orbital.
    labels : Optional[tuple[str, ...]], optional
        Human-readable orbital labels (**static** -- compile-time constants;
        changing them triggers retracing). Defaults to
        ``("orb_0", "orb_1", ...)``.

    Returns
    -------
    basis : OrbitalBasis
        Validated orbital basis with consistent lengths.

    Raises
    ------
    ValueError
        If the quantum-number tuples have different lengths or ``labels`` has
        a different length. The function also rejects invalid values of
        ``n``, ``l``, or ``m``.

    Notes
    -----
    Every ``OrbitalBasis`` field uses ``eqx.field(static=True)``, so the
    factory performs static validation. Invalid tuple lengths or quantum
    numbers raise ``ValueError`` before tracing. No ``eqx.error_if`` checks
    apply.

    See Also
    --------
    OrbitalBasis : The PyTree class constructed by this factory.
    """
    n_orbitals: int = len(n_values)
    if len(l_values) != n_orbitals or len(m_values) != n_orbitals:
        msg: str = "n_values, l_values, m_values must have the same length"
        raise ValueError(msg)
    resolved_labels: tuple[str, ...] = (
        tuple(f"orb_{i}" for i in range(n_orbitals))
        if labels is None
        else labels
    )
    if len(resolved_labels) != n_orbitals:
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
        labels=resolved_labels,
    )
    return basis


@jaxtyped(typechecker=beartype)
def make_slater_params(  # noqa: DOC503
    zeta: Float[Array, " Oz"],
    orbital_basis: OrbitalBasis,
    coefficients: Optional[Float[Array, "Oc C"]] = None,
) -> SlaterParams:
    """Create a validated ``SlaterParams`` instance.

    The factory validates Slater radial wavefunction
    parameters and constructs a ``SlaterParams`` PyTree. The ``zeta``
    array length must match the orbital count in ``orbital_basis``. If
    ``coefficients`` is absent, the factory uses a single-zeta basis. It
    creates a column of ones with shape ``(O, 1)``.

    Use this factory for automatic size validation and default coefficient
    generation. The factory also guarantees ``float64`` dtypes.

    :see: :class:`~.test_radial_params.TestMakeSlaterParams`

    Implementation Logic
    --------------------
    1. **Prepare the normalized values**::

           zeta_arr = jnp.asarray(zeta, dtype=jnp.float64)

       This expression gives the later validation steps a stable shape and
       dtype.

    2. **Apply static validation**::

           len(orbital_basis.n_values) != n_orbitals

       This predicate rejects invalid structure before JAX traces the
       numerical checks.

    3. **Apply traced validation**::

           ~jnp.all(jnp.isfinite(zeta_arr))

       This predicate remains active during eager and compiled execution.

    4. **Return the named instance**::

           return params

       The explicit name keeps the implementation and the Returns section
       synchronized.

    Parameters
    ----------
    zeta : Float[Array, " Oz"]
        Slater exponents (inverse Bohr), one per orbital.
    orbital_basis : OrbitalBasis
        Orbital quantum number metadata (**static** -- a compile-time
        constant; changing it triggers retracing). Must have the same number
        of orbitals as the length of ``zeta``.
    coefficients : Optional[Float[Array, "Oc C"]], optional
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
        If ``zeta`` and ``orbital_basis`` contain different orbital counts.
        The function also rejects a different orbital count in
        ``coefficients``.
    EquinoxRuntimeError
        If ``zeta`` is non-finite or non-positive, or if
        ``coefficients`` contains a non-finite value.

    Notes
    -----
    Static validation raises ``ValueError`` before traced construction when
    the orbital count differs from either leading array dimension. Traced
    validation uses ``eqx.error_if`` and raises ``EquinoxRuntimeError`` when
    an exponent is non-finite or non-positive, or a coefficient is non-finite.

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
        validated_params: SlaterParams = SlaterParams(
            zeta=zeta_arr,
            coefficients=coeff_arr_checked,
            orbital_basis=orbital_basis,
        )
        return validated_params

    params: SlaterParams = validate_and_create()
    return params


__all__: list[str] = [
    "OrbitalBasis",
    "SlaterParams",
    "make_orbital_basis",
    "make_slater_params",
]
