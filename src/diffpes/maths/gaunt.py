r"""Gaunt coefficient table for dipole transitions.

Extended Summary
----------------
Precomputes real-valued Gaunt integrals for electric dipole
transitions in real spherical harmonics:

.. math::

    G(l, m, l', m') = \int Y_l^m(\hat{r})\, r_q\, Y_{l'}^{m'}(\hat{r})\,d\Omega

where :math:`r_q` is one of the three components of the position
operator expressed in the real spherical harmonic basis (q = -1, 0, +1).

Selection rules: :math:`l' = l \pm 1` and :math:`|m' - m| \leq 1`.

The table is computed once at module load time using pure Python
(not JAX-traced) and stored as a JAX array for O(1) lookup.

Routine Listings
----------------
:obj:`GAUNT_TABLE`
    Module-level precomputed Gaunt coefficient table for l_max=4.
:func:`build_gaunt_table`
    Build the dipole Gaunt coefficient lookup table.
:func:`gaunt_lookup`
    Look up a single Gaunt coefficient from the precomputed table.
"""

import math
from functools import cache

import jax.numpy as jnp
import numpy as np
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped
from numpy import ndarray as NDArray  # noqa: N812

from diffpes.types import GAUNT_IMAG_TOL, L_MAX


def _wigner3j(j1: int, j2: int, j3: int, m1: int, m2: int, m3: int) -> float:
    r"""Evaluate Wigner 3-j symbol using the Racah formula.

    Extended Summary
    ----------------
    Computes the Wigner 3-j symbol

    .. math::

        \begin{pmatrix} j_1 & j_2 & j_3 \\ m_1 & m_2 & m_3 \end{pmatrix}

    using the explicit Racah formula, which expresses the 3-j symbol as a
    finite sum over factorials. The formula is:

    .. math::

        \begin{pmatrix} j_1 & j_2 & j_3 \\ m_1 & m_2 & m_3 \end{pmatrix}
        = (-1)^{j_1 - j_2 - m_3} \Delta(j_1, j_2, j_3) \cdot P \cdot
        \sum_t \frac{(-1)^t}{t! \cdot D(t)}

    where :math:`\Delta(j_1, j_2, j_3)` is the triangle coefficient

    .. math::

        \Delta = \sqrt{
            \frac{(j_1+j_2-j_3)!(j_1-j_2+j_3)!(-j_1+j_2+j_3)!}
                 {(j_1+j_2+j_3+1)!}
        }

    and :math:`P` is the product of square roots of magnetic-number
    factorials:

    .. math::

        P = \sqrt{(j_1+m_1)!(j_1-m_1)!(j_2+m_2)!(j_2-m_2)!(j_3+m_3)!(j_3-m_3)!}

    The summation index *t* runs over all integers for which all
    factorial arguments are non-negative, determined by ``t_min`` and
    ``t_max``.

    **Selection rules** enforced before computation:

    - :math:`m_1 + m_2 + m_3 = 0` (magnetic quantum number conservation)
    - :math:`|m_i| \le j_i` for each i (projection bounds)
    - :math:`|j_1 - j_2| \le j_3 \le j_1 + j_2` (triangle inequality)

    If any selection rule is violated, the function returns 0.0 immediately
    without computing the sum.

    Only needed for small angular momenta (l_max <= 5), so the
    factorial-based formula is efficient and exact (no floating-point
    cancellation issues at this scale).

    Parameters
    ----------
    j1 : int
        First angular momentum quantum number.
    j2 : int
        Second angular momentum quantum number.
    j3 : int
        Third angular momentum quantum number.
    m1 : int
        First magnetic quantum number.
    m2 : int
        Second magnetic quantum number.
    m3 : int
        Third magnetic quantum number.

    Returns
    -------
    value : float
        The Wigner 3-j symbol value.
    """
    t: int

    if m1 + m2 + m3 != 0:
        value: float = 0.0
        return value  # noqa: RET504 -- assign-before-return is required.
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        value = 0.0
        return value  # noqa: RET504 -- assign-before-return is required.
    if j3 < abs(j1 - j2) or j3 > j1 + j2:
        value = 0.0
        return value  # noqa: RET504 -- assign-before-return is required.

    t_min: int = max(0, j2 - j3 - m1, j1 - j3 + m2)
    t_max: int = min(j1 + j2 - j3, j1 - m1, j2 + m2)

    prefactor: float = math.sqrt(
        math.factorial(j1 + m1)
        * math.factorial(j1 - m1)
        * math.factorial(j2 + m2)
        * math.factorial(j2 - m2)
        * math.factorial(j3 + m3)
        * math.factorial(j3 - m3)
    )
    triangle: float = math.sqrt(
        math.factorial(j1 + j2 - j3)
        * math.factorial(j1 - j2 + j3)
        * math.factorial(-j1 + j2 + j3)
        / math.factorial(j1 + j2 + j3 + 1)
    )

    total: float = 0.0
    for t in range(t_min, t_max + 1):
        sign: int = (-1) ** t
        denom: int = (
            math.factorial(t)
            * math.factorial(j1 + j2 - j3 - t)
            * math.factorial(j1 - m1 - t)
            * math.factorial(j2 + m2 - t)
            * math.factorial(j3 - j2 + m1 + t)
            * math.factorial(j3 - j1 - m2 + t)
        )
        total += sign / denom

    value: float = float((-1) ** (j1 - j2 - m3) * prefactor * triangle * total)
    return value


@cache
def _complex_gaunt(
    l1: int, m1: int, l2: int, m2: int, l3: int, m3: int
) -> float:
    r"""Complex Gaunt integral for three complex spherical harmonics.

    Extended Summary
    ----------------
    Computes the Gaunt integral (three-Y integral) defined as:

    .. math::

        G(l_1,m_1; l_2,m_2; l_3,m_3)
        = \int Y_{l_1}^{m_1} Y_{l_2}^{m_2} Y_{l_3}^{m_3} \, d\Omega

    Using the standard decomposition into Wigner 3-j symbols:

    .. math::

        G = (-1)^{m_3}
            \sqrt{\frac{(2l_1+1)(2l_2+1)(2l_3+1)}{4\pi}}
            \begin{pmatrix} l_1 & l_2 & l_3 \\ 0 & 0 & 0 \end{pmatrix}
            \begin{pmatrix} l_1 & l_2 & l_3 \\ m_1 & m_2 & -m_3 \end{pmatrix}

    The first 3-j symbol enforces the parity selection rule
    :math:`l_1 + l_2 + l_3 = \text{even}`, and is evaluated first as
    an early-exit optimization. The result is cached via
    ``functools.cache`` since the same (l, m) combinations appear
    repeatedly during Gaunt table construction.

    Parameters
    ----------
    l1 : int
        First angular momentum quantum number.
    m1 : int
        First magnetic quantum number.
    l2 : int
        Second angular momentum quantum number.
    m2 : int
        Second magnetic quantum number.
    l3 : int
        Third angular momentum quantum number.
    m3 : int
        Third magnetic quantum number.

    Returns
    -------
    value : float
        The complex Gaunt integral.
    """
    w3j_000: float = _wigner3j(l1, l2, l3, 0, 0, 0)
    if w3j_000 == 0.0:
        value: float = 0.0
        return value  # noqa: RET504 -- assign-before-return is required.
    w3j_mmm: float = _wigner3j(l1, l2, l3, m1, m2, -m3)
    if w3j_mmm == 0.0:
        value = 0.0
        return value  # noqa: RET504 -- assign-before-return is required.

    prefactor: float = (-1) ** m3 * math.sqrt(
        (2 * l1 + 1) * (2 * l2 + 1) * (2 * l3 + 1) / (4.0 * math.pi)
    )
    value: float = prefactor * w3j_000 * w3j_mmm
    return value


def _real_gaunt_dipole(l: int, m: int, lp: int, mp: int, q: int) -> float:
    r"""Gaunt coefficient for real spherical harmonics with dipole operator.

    Extended Summary
    ----------------
    Computes the integral of :math:`Y_l^m(\text{real}) \cdot r_q \cdot
    Y_{l'}^{m'}(\text{real})` over the unit sphere, where
    :math:`r_q` (q = -1, 0, +1) is a dipole operator component.

    Because the Gaunt integral is natively defined for complex spherical
    harmonics, this function performs a basis transformation: each real
    harmonic is expanded in the complex basis using unitary coefficients,
    and the resulting triple sum of complex Gaunt integrals is accumulated.

    **Real-to-complex transformation:**

    .. math::

        Y_l^m(\text{real}) = \frac{1}{\sqrt{2}}
            \bigl(Y_l^m + (-1)^m Y_l^{-m}\bigr)
            \quad (m > 0)

        Y_l^0(\text{real}) = Y_l^0 \quad (m = 0)

        Y_l^m(\text{real}) = \frac{-i}{\sqrt{2}}
            \bigl((-1)^{|m|} Y_l^{|m|} - Y_l^{-|m|}\bigr)
            \quad (m < 0)

    Each real harmonic thus becomes a sum of at most two complex harmonics
    with known coefficients (computed by the nested helper
    ``_real_to_complex_coeffs``).

    **Dipole operator in complex basis:**

    The position operator component :math:`r_q` corresponds to the l=1
    real spherical harmonic :math:`Y_1^q(\text{real})`, which is itself
    expanded in the complex basis using the same transformation.

    **Assembly:**

    The full real Gaunt coefficient is:

    .. math::

        G_{\text{real}} = \sum_{\mu, \nu, \rho}
            \overline{U_{\text{final}}(m', \rho)} \cdot
            U_{\text{dip}}(q, \nu) \cdot
            U_{\text{init}}(m, \mu) \cdot
            (-1)^{\rho} \cdot
            G_{\text{complex}}(l', -\rho; 1, \nu; l, \mu)

    where :math:`U` matrices are the real-to-complex transformation
    coefficients. The conjugation on the final-state coefficients arises
    because the complex Gaunt integral involves :math:`Y_{l'}^{\rho *}`,
    and :math:`Y_l^{m *} = (-1)^m Y_l^{-m}`.

    The result is guaranteed to be real; an imaginary part exceeding
    :math:`10^{-12}` triggers a ``ValueError`` as a consistency check.

    Parameters
    ----------
    l : int
        Initial state angular momentum.
    m : int
        Initial state magnetic quantum number.
    lp : int
        Final state angular momentum.
    mp : int
        Final state magnetic quantum number.
    q : int
        Dipole component index (-1, 0, or +1).

    Returns
    -------
    value : float
        The real Gaunt coefficient for the specified quantum numbers.

    Raises
    ------
    ValueError
        If numerical basis conversion leaves a non-negligible imaginary part.
    """
    c_init: complex
    mu: int
    c_dip: complex
    nu: int
    c_final: complex
    rho: int

    sqrt2: float = math.sqrt(2.0)

    def _real_to_complex_coeffs(ll: int, mm: int) -> list[tuple[complex, int]]:
        r"""Return real-to-complex expansion coefficients.

        Computes the unitary transformation coefficients :math:`U_{m,\mu}`
        such that :math:`Y_l^m(\text{real}) = \sum_\mu U_{m,\mu} Y_l^\mu`.
        For m > 0, two terms with :math:`\mu = \pm m` contribute with
        coefficients :math:`1/\sqrt{2}` and :math:`(-1)^m/\sqrt{2}`.
        For m = 0, a single term with coefficient 1. For m < 0, two
        terms with imaginary coefficients that isolate the sine component.

        Parameters
        ----------
        ll : int
            Angular momentum (unused in the formula but kept for clarity).
        mm : int
            Magnetic quantum number of the real harmonic.

        Returns
        -------
        coeffs : list of (complex, int)
            List of ``(coefficient, mu)`` pairs.
        """
        if mm > 0:
            coeffs: list[tuple[complex, int]] = [
                (complex(1.0 / sqrt2), mm),
                (complex((-1) ** mm / sqrt2), -mm),
            ]
        elif mm == 0:
            coeffs = [(1.0 + 0.0j, 0)]
        else:
            am: int = abs(mm)
            coeffs = [
                (-1j * (-1) ** am / sqrt2, am),
                (1j / sqrt2, -am),
            ]
        return coeffs

    dip_coeffs: list[tuple[complex, int]] = _real_to_complex_coeffs(1, q)

    init_coeffs: list[tuple[complex, int]] = _real_to_complex_coeffs(l, m)
    final_coeffs: list[tuple[complex, int]] = _real_to_complex_coeffs(lp, mp)

    total: complex = 0.0 + 0.0j
    for c_init, mu in init_coeffs:
        for c_dip, nu in dip_coeffs:
            for c_final, rho in final_coeffs:
                cg: float = _complex_gaunt(lp, -rho, 1, nu, l, mu)
                coeff: complex = (
                    complex(c_final).conjugate()
                    * complex(c_dip)
                    * complex(c_init)
                )
                total += coeff * (-1) ** rho * cg

    result: float = total.real
    if abs(total.imag) > GAUNT_IMAG_TOL:  # pragma: no cover
        msg: str = f"Imaginary part {total.imag} in real Gaunt coefficient"
        raise ValueError(msg)
    return result


@jaxtyped(typechecker=beartype)
def build_gaunt_table(
    l_max: int = 4,
) -> Float[Array, "L_src M_src 3 L_dst M_dst"]:
    r"""Build the dipole Gaunt coefficient lookup table.

    Precomputes every non-zero real Gaunt coefficient for electric
    dipole transitions up to angular momentum ``l_max``. The table
    is stored as a dense 5-D NumPy array and then converted to a
    JAX array for O(1) lookup during forward-model evaluation.

    The five axes correspond to:

    - **axis 0** (``l``): initial angular momentum, size ``l_max + 1``.
    - **axis 1** (``m + l_max``): offset initial magnetic quantum number,
      size ``2 * l_max + 1``, so that m = -l_max maps to index 0.
    - **axis 2** (``q + 1``): dipole component index, size 3, for
      q in {-1, 0, +1}.
    - **axis 3** (``lp``): final angular momentum, size ``l_max + 2``
      (because the dipole operator can promote l to l + 1).
    - **axis 4** (``mp + l_max``): offset final magnetic quantum number,
      size ``2 * (l_max + 1) + 1``.

    The construction loops over all valid quantum number combinations
    respecting the dipole selection rules :math:`l' = l \pm 1` and
    calls `_real_gaunt_dipole` for each. Zero entries (forbidden
    transitions) are left at their initialized value.

    The table is indexed as
    ``GAUNT_TABLE[l, m + l_max, q + 1, lp, mp + l_max]``
    where q in {-1, 0, +1} indexes the three dipole components.

    :see: :class:`~.test_gaunt.TestBuildGauntTable`

    Implementation Logic
    --------------------
    1. **Allocate the dense coefficient table**::

           table: Float[NDArray, "L1 M1 Q L2 M2"] = np.zeros(
               (l_src_dim, m_src_dim, q_dim, l_dst_dim, m_dst_dim),
               dtype=np.float64,
           )

       The dense layout gives constant-time lookup for valid quantum numbers.

    2. **Convert the completed table to JAX**::

           gaunt_table: Float[
               Array, "L_src M_src 3 L_dst M_dst"
           ] = jnp.asarray(table, dtype=jnp.float64)

       The JAX constant can participate in differentiable forward models.

    Parameters
    ----------
    l_max : int
        Maximum angular momentum. Default 4 (s through g).

    Returns
    -------
    gaunt_table : Float[Array, "..."]
        Dense array of Gaunt coefficients.
        Shape: ``(l_max+1, 2*l_max+1, 3, l_max+2, 2*(l_max+1)+1)``.

    Notes
    -----
    This function uses pure Python / NumPy (not JAX-traced) because it
    runs once at module import time. The result is frozen as a JAX
    array constant, so it does not appear as a trainable parameter in
    any gradient computation.
    """
    l: int
    m: int
    q: int
    lp: int
    mp: int

    l_src_dim: int = l_max + 1
    m_src_dim: int = 2 * l_max + 1
    q_dim: int = 3
    l_dst_dim: int = l_max + 2
    m_dst_dim: int = 2 * (l_max + 1) + 1

    table: Float[NDArray, "L1 M1 Q L2 M2"] = np.zeros(
        (l_src_dim, m_src_dim, q_dim, l_dst_dim, m_dst_dim),
        dtype=np.float64,
    )

    for l in range(l_src_dim):
        for m in range(-l, l + 1):
            for q in (-1, 0, 1):
                for lp in (l - 1, l + 1):
                    if lp < 0 or lp >= l_dst_dim:
                        continue
                    for mp in range(-lp, lp + 1):
                        val: float = _real_gaunt_dipole(l, m, lp, mp, q)
                        table[l, m + l_max, q + 1, lp, mp + l_max] = val

    gaunt_table: Float[Array, "L_src M_src 3 L_dst M_dst"] = jnp.asarray(
        table, dtype=jnp.float64
    )
    return gaunt_table


GAUNT_TABLE: Float[Array, "..."] = build_gaunt_table(l_max=L_MAX)
"""Module-level precomputed Gaunt coefficient table for l_max=4."""


@jaxtyped(typechecker=beartype)
def gaunt_lookup(l: int, m: int, q: int, lp: int, mp: int) -> float:
    r"""Look up a single Gaunt coefficient from the precomputed table.

    Provides a convenience accessor for the module-level ``GAUNT_TABLE``
    array, converting the physical quantum numbers (l, m, q, l', m')
    into the offset indices used by the dense storage layout. The
    index mapping is:

    - ``l`` indexes axis 0 directly.
    - ``m + L_MAX`` offsets the magnetic quantum number to a
      non-negative index on axis 1.
    - ``q + 1`` maps q in {-1, 0, +1} to indices {0, 1, 2} on axis 2.
    - ``lp`` indexes axis 3 directly.
    - ``mp + L_MAX`` offsets m' on axis 4.

    The returned value is cast to a Python float for use in
    non-JAX contexts. For JAX-traced code, direct indexing into
    ``GAUNT_TABLE`` is preferred to avoid Python-level overhead.

    :see: :class:`~.test_gaunt.TestGauntLookup`

    Implementation Logic
    --------------------
    1. **Index and convert the coefficient**::

           coefficient: float = float(
               GAUNT_TABLE[l, m + L_MAX, q + 1, lp, mp + L_MAX]
           )

       The offsets map signed quantum numbers to dense array indices.

    Parameters
    ----------
    l : int
        Initial state angular momentum.
    m : int
        Initial state magnetic quantum number.
    q : int
        Dipole component (-1, 0, or +1).
    lp : int
        Final state angular momentum.
    mp : int
        Final state magnetic quantum number.

    Returns
    -------
    coefficient : float
        The Gaunt coefficient.
    """
    coefficient: float = float(
        GAUNT_TABLE[l, m + L_MAX, q + 1, lp, mp + L_MAX]
    )
    return coefficient


__all__: list[str] = [
    "GAUNT_TABLE",
    "build_gaunt_table",
    "gaunt_lookup",
]
