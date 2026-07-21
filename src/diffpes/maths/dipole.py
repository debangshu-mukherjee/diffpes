r"""Full dipole matrix element assembly.

Extended Summary
----------------
Combines radial integrals :math:`B^{l'}(k)`, Gaunt coefficients,
real spherical harmonics :math:`Y_{l'm'}(\hat{k})`, and the
polarization vector :math:`\hat{e}` to compute photoemission
dipole matrix elements from first principles:

.. math::

    M(\mathbf{k}, n, l, m) = \sum_{l', m'} B_{n,l}^{l'}(|\mathbf{k}|)
        \cdot G(l, m, l', m') \cdot Y_{l'}^{m'}(\hat{k})
        \cdot \hat{e}_{q(m'-m)}

where :math:`q = m' - m` selects the dipole component and
:math:`\hat{e}_q` is the corresponding spherical component of
the polarization vector.

Routine Listings
----------------
:func:`dipole_intensities_all_orbitals`
    Compute ``|M|^2`` for all orbitals in the basis.
:func:`dipole_intensity_orbital`
    Compute ``|M|^2`` for one orbital.
:func:`dipole_matrix_element_single`
    Compute dipole matrix element for a single orbital (n, l, m).
"""

from functools import partial

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Callable
from jaxtyping import Array, Complex, Float, Integer, jaxtyped

from diffpes.radial import radial_integral, slater_radial
from diffpes.types import OrbitalBasis, SlaterParams

from .gaunt import GAUNT_TABLE, L_MAX
from .safe import safe_arccos, safe_arctan2, safe_divide, safe_norm
from .spherical_harmonics import real_spherical_harmonic


def _cartesian_to_spherical_dipole(
    efield: Complex[Array, " 3"],
) -> Complex[Array, " 3"]:
    r"""Convert Cartesian E-field to real-harmonic dipole components.

    Maps (e_x, e_y, e_z) to (e_{q=-1}, e_{q=0}, e_{q=+1}) where
    the q index matches the real spherical harmonic convention for
    the dipole operator (l=1):

    - q = -1 corresponds to Y_1^{-1}(real) ~ sin(theta)sin(phi) ~ y
    - q =  0 corresponds to Y_1^0(real) ~ cos(theta) ~ z
    - q = +1 corresponds to Y_1^{+1}(real) ~ sin(theta)cos(phi) ~ x

    The dipole operator :math:`\hat{r}` is expanded in the real spherical
    harmonic basis for l=1. In Cartesian coordinates the three components
    of the position vector are :math:`(x, y, z)`, and they map to real
    spherical harmonics as:

    .. math::

        x = r \sin\theta \cos\phi \propto Y_1^{+1}(\text{real}) \quad (q=+1)

        y = r \sin\theta \sin\phi \propto Y_1^{-1}(\text{real}) \quad (q=-1)

        z = r \cos\theta \propto Y_1^{0}(\text{real}) \quad (q=0)

    The returned array is ordered by ascending q: ``[e_y, e_z, e_x]``,
    so that indexing with ``q_idx = q + 1`` (for q in {-1, 0, +1})
    selects the correct Cartesian component of the polarization vector.

    This is a pure permutation with no complex rotation because the
    real spherical harmonics for l=1 directly correspond to the
    Cartesian axes without mixing.

    Parameters
    ----------
    efield : Complex[Array, " 3"]
        Polarization vector ``(e_x, e_y, e_z)`` in Cartesian coordinates.

    Returns
    -------
    e_spherical : Complex[Array, " 3"]
        ``(e_{q=-1}, e_{q=0}, e_{q=+1})``.
    """
    ex: Complex[Array, ""] = efield[0]
    ey: Complex[Array, ""] = efield[1]
    ez: Complex[Array, ""] = efield[2]
    e_spherical: Complex[Array, " 3"] = jnp.array(
        [ey, ez, ex], dtype=jnp.complex128
    )
    return e_spherical


@jaxtyped(typechecker=beartype)
def dipole_matrix_element_single(
    k_vec: Float[Array, " 3"],
    r_grid: Float[Array, " R"],
    radial_values: Float[Array, " R"],
    l: int,
    m: int,
    efield: Complex[Array, " 3"],
) -> Complex[Array, " "]:
    r"""Compute dipole matrix element for a single orbital (n, l, m).

    .. math::

        M = \sum_{q} \hat{e}_q \sum_{l'} B^{l'}(|k|) \cdot
            G(l, m, l', m+q) \cdot Y_{l'}^{m+q}(\hat{k})

    where the sum is over dipole components q in {-1, 0, +1} and
    final-state angular momenta l' in {l-1, l+1}.

    This function assembles the full photoemission dipole matrix element
    by combining four ingredients:

    1. **Radial integral** :math:`B^{l'}(|k|)` -- the overlap between
       the initial radial wavefunction and the final-state spherical
       Bessel function :math:`j_{l'}(kr)`, weighted by :math:`r^3`,
       evaluated via trapezoidal quadrature on the supplied radial grid.

    2. **Gaunt coefficient** :math:`G(l, m, l', m')` -- the angular
       integral coupling initial (l, m) and final (l', m') states
       through the dipole operator, looked up from the precomputed
       ``GAUNT_TABLE``.

    3. **Real spherical harmonic** :math:`Y_{l'}^{m'}(\hat{k})` --
       the angular part of the final-state plane wave expansion,
       evaluated at the direction of the photoelectron wavevector.

    4. **Polarization component** :math:`\hat{e}_q` -- the q-th
       spherical component of the polarization vector, obtained by
       mapping Cartesian (x, y, z) to the real harmonic basis via
       `_cartesian_to_spherical_dipole`.

    The dipole selection rule :math:`l' = l \pm 1` restricts the
    final-state sum to at most two terms per q value. The magnetic
    selection rule :math:`m' = m + q` with :math:`|q| \le 1` means
    at most three q values contribute.

    Numerical stability routes the wavevector norm, normalization, polar
    angle, and azimuthal angle through the named safe-math primitives. Their
    guard conventions select zero subgradients at the zero vector and at the
    angular coordinate singularities without perturbing non-singular values.

    :see: :class:`~.test_dipole.TestDipoleMatrixElementSingle`

    Parameters
    ----------
    k_vec : Float[Array, " 3"]
        Photoelectron wavevector in Cartesian coordinates.
    r_grid : Float[Array, " R"]
        Radial grid for integration.
    radial_values : Float[Array, " R"]
        R(r) sampled on r_grid.
    l : int
        Angular momentum quantum number of the initial orbital.
    m : int
        Magnetic quantum number of the initial orbital.
    efield : Complex[Array, " 3"]
        Polarization vector in Cartesian coordinates.

    Returns
    -------
    M : Complex[Array, " "]
        Complex dipole matrix element.

    Notes
    -----
    The loop over q and l' is unrolled at Python trace time (not
    inside ``jax.lax`` control flow) because the iteration bounds
    depend on the static quantum numbers (l, m). This produces a
    fixed computation graph per (l, m) pair, which is efficient
    for JIT compilation but means different orbitals trace distinct
    XLA programs.
    """
    q_idx: int
    q: int
    lp: int

    k_mag: Float[Array, ""] = safe_norm(k_vec)
    k_hat: Float[Array, " 3"] = safe_divide(k_vec, k_mag)
    theta_k: Float[Array, ""] = safe_arccos(k_hat[2])
    phi_k: Float[Array, ""] = safe_arctan2(k_hat[1], k_hat[0])

    e_sph: Complex[Array, " 3"] = _cartesian_to_spherical_dipole(efield)

    M_total: Complex[Array, ""] = jnp.zeros((), dtype=jnp.complex128)

    for q_idx, q in enumerate((-1, 0, 1)):
        mp: int = m + q
        eq: Complex[Array, ""] = e_sph[q_idx]

        for lp in (l - 1, l + 1):
            if lp < 0 or lp > L_MAX + 1:
                continue
            if abs(mp) > lp:
                continue

            B_lp: Float[Array, ""] = radial_integral(
                k_mag, r_grid, radial_values, lp
            )

            G: Float[Array, ""] = GAUNT_TABLE[
                l, m + L_MAX, q + 1, lp, mp + L_MAX
            ]

            Y_lp_mp: Float[Array, ""] = real_spherical_harmonic(
                lp, mp, theta_k, phi_k
            )

            M_total = M_total + eq * B_lp * G * Y_lp_mp

    M: Complex[Array, ""] = M_total
    return M


@jaxtyped(typechecker=beartype)
def dipole_intensity_orbital(
    k_vec: Float[Array, " 3"],
    r_grid: Float[Array, " R"],
    radial_values: Float[Array, " R"],
    l: int,
    m: int,
    efield: Complex[Array, " 3"],
) -> Float[Array, " "]:
    r"""Compute ``|M|^2`` for one orbital.

    Computes the photoemission intensity for a single initial-state
    orbital characterized by quantum numbers (l, m) and a radial
    wavefunction sampled on a grid. The intensity is the squared
    modulus of the complex dipole matrix element:

    .. math::

        I(\mathbf{k}) = |M(\mathbf{k}, l, m)|^2

    This is a thin wrapper that calls `dipole_matrix_element_single`
    and returns :math:`|M|^2 = M \cdot M^*`. The result is real and
    non-negative by construction, and is differentiable with respect
    to all continuous inputs (k_vec, r_grid, radial_values, efield)
    through JAX automatic differentiation.

    :see: :class:`~.test_dipole.TestDipoleIntensityOrbital`

    Parameters
    ----------
    k_vec : Float[Array, " 3"]
        Photoelectron wavevector.
    r_grid : Float[Array, " R"]
        Radial grid.
    radial_values : Float[Array, " R"]
        Radial wavefunction on grid.
    l : int
        Angular momentum.
    m : int
        Magnetic quantum number.
    efield : Complex[Array, " 3"]
        Polarization vector.

    Returns
    -------
    intensity : Float[Array, " "]
        Squared modulus of the matrix element.

    Notes
    -----
    The function computes the complex matrix element first. It then applies
    the modulus squared once, after the coherent channel sum is complete.
    Gradients flow through both operations for every continuous input.
    """
    M: Complex[Array, ""] = dipole_matrix_element_single(
        k_vec, r_grid, radial_values, l, m, efield
    )
    intensity: Float[Array, ""] = jnp.abs(M) ** 2
    return intensity


@jaxtyped(typechecker=beartype)
def dipole_intensities_all_orbitals(
    k_vec: Float[Array, " 3"],
    r_grid: Float[Array, " R"],
    slater_params: SlaterParams,
    efield: Complex[Array, " 3"],
) -> Float[Array, " O"]:
    r"""Compute ``|M|^2`` for all orbitals in the basis.

    Scans over every orbital in the Slater basis set, computes its
    radial wavefunction from ``slater_params``, and evaluates the
    squared dipole matrix element. ``jax.lax.switch`` specializes one
    branch per static ``(n, l, m)`` tuple, while ``jax.lax.scan`` carries
    the differentiable exponent and coefficient arrays through the
    orbital axis.

    For each orbital *o*, the function:

    1. Extracts quantum numbers :math:`(n_o, l_o, m_o)` and Slater
       exponent :math:`\zeta_o` from ``slater_params``.
    2. Evaluates the normalized Slater radial function
       :math:`R(r) = N r^{n-1} e^{-\zeta r}` on the supplied grid.
    3. Weights by the multi-zeta coefficient
       ``slater_params.coefficients[o, 0]`` (first column for
       single-zeta bases).
    4. Calls `dipole_intensity_orbital` to compute :math:`|M|^2`.

    The results are stacked into a 1-D array of length equal to the
    number of orbitals in the basis.

    :see: :class:`~.test_dipole.TestDipoleIntensitiesAllOrbitals`

    Parameters
    ----------
    k_vec : Float[Array, " 3"]
        Photoelectron wavevector.
    r_grid : Float[Array, " R"]
        Radial grid.
    slater_params : SlaterParams
        Slater exponents and orbital basis.
    efield : Complex[Array, " 3"]
        Polarization vector.

    Returns
    -------
    intensities : Float[Array, " O"]
        ``|M|^2`` per orbital.

    Notes
    -----
    The number and quantum numbers of orbitals remain static PyTree metadata,
    so changing the basis structure retraces the function. Slater exponents
    and coefficients remain traced leaves with gradients through the scan.
    """
    basis: OrbitalBasis = slater_params.orbital_basis
    n_orbitals: int = len(basis.n_values)

    def _evaluate_orbital(
        operand: tuple[Float[Array, ""], Float[Array, ""]],
        *,
        n: int,
        l: int,
        m: int,
    ) -> Float[Array, ""]:
        """Evaluate one statically specialized orbital branch."""
        zeta_value: Float[Array, ""] = operand[0]
        coefficient: Float[Array, ""] = operand[1]
        radial_values: Float[Array, " R"] = (
            slater_radial(r_grid, n, zeta_value) * coefficient
        )
        intensity: Float[Array, ""] = dipole_intensity_orbital(
            k_vec,
            r_grid,
            radial_values,
            l,
            m,
            efield,
        )
        return intensity

    orbital_branches: tuple[
        Callable[
            [tuple[Float[Array, ""], Float[Array, ""]]],
            Float[Array, ""],
        ],
        ...,
    ] = tuple(
        partial(_evaluate_orbital, n=n, l=l, m=m)
        for n, l, m in zip(
            basis.n_values,
            basis.l_values,
            basis.m_values,
            strict=True,
        )
    )

    def _scan_orbital(
        carry: None,
        orbital_index: Integer[Array, ""],
    ) -> tuple[None, Float[Array, ""]]:
        """Evaluate one orbital while scanning traced parameter leaves."""
        operand: tuple[Float[Array, ""], Float[Array, ""]] = (
            slater_params.zeta[orbital_index],
            slater_params.coefficients[orbital_index, 0],
        )
        intensity: Float[Array, ""] = jax.lax.switch(
            orbital_index,
            orbital_branches,
            operand,
        )
        scan_output: tuple[None, Float[Array, ""]] = (carry, intensity)
        return scan_output

    orbital_indices: Integer[Array, " O"] = jnp.arange(n_orbitals)
    scan_result: tuple[None, Float[Array, " O"]] = jax.lax.scan(
        _scan_orbital,
        None,
        orbital_indices,
    )
    intensities: Float[Array, " O"] = scan_result[1]
    return intensities


__all__: list[str] = [
    "dipole_intensities_all_orbitals",
    "dipole_intensity_orbital",
    "dipole_matrix_element_single",
]
