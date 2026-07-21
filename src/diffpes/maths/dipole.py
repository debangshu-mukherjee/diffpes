r"""Assemble full dipole matrix elements.

Extended Summary
----------------
The module combines radial integrals, Gaunt coefficients, real spherical
harmonics, and the polarization vector. These quantities determine the
photoemission dipole matrix elements:

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

    The function maps (e_x, e_y, e_z) to
    (e_{q=-1}, e_{q=0}, e_{q=+1}). The q index follows the real spherical
    harmonic convention for the dipole operator with l=1:

    - q = -1 corresponds to Y_1^{-1}(real) ~ sin(theta)sin(phi) ~ y
    - q =  0 corresponds to Y_1^0(real) ~ cos(theta) ~ z
    - q = +1 corresponds to Y_1^{+1}(real) ~ sin(theta)cos(phi) ~ x

    The function expands the dipole operator :math:`\hat{r}` in the real
    spherical harmonic basis for l=1. The Cartesian position vector has
    components :math:`(x, y, z)`. These components map to real spherical
    harmonics as follows:

    .. math::

        x = r \sin\theta \cos\phi \propto Y_1^{+1}(\text{real}) \quad (q=+1)

        y = r \sin\theta \sin\phi \propto Y_1^{-1}(\text{real}) \quad (q=-1)

        z = r \cos\theta \propto Y_1^{0}(\text{real}) \quad (q=0)

    The returned array follows ascending q: ``[e_y, e_z, e_x]``. Therefore,
    ``q_idx = q + 1`` selects the corresponding Cartesian component.

    This operation is a pure permutation without a complex rotation. The real
    spherical harmonics for l=1 correspond directly to the Cartesian axes.

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

    The function assembles the full photoemission dipole matrix element from
    four quantities:

    1. **Radial integral** :math:`B^{l'}(|k|)` -- the overlap between
       the initial radial wavefunction and the final-state spherical
       Bessel function :math:`j_{l'}(kr)`. The radial integral applies the
       :math:`r^3` weight and trapezoidal quadrature.

    2. **Gaunt coefficient** :math:`G(l, m, l', m')` -- the angular
       integral that couples the initial and final states through the dipole
       operator. The function reads this coefficient from ``GAUNT_TABLE``.

    3. **Real spherical harmonic** :math:`Y_{l'}^{m'}(\hat{k})` --
       the angular part of the final-state plane wave expansion. The function
       computes it at the direction of the photoelectron wavevector.

    4. **Polarization component** :math:`\hat{e}_q` -- the q-th
       spherical component of the polarization vector. The
       `_cartesian_to_spherical_dipole` function maps the Cartesian components
       to the real harmonic basis.

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
    Python unrolls the loop over q and l' during tracing. Static quantum
    numbers determine the iteration bounds. This process produces one fixed
    computation graph for each (l, m) pair. Different orbitals therefore
    produce distinct XLA programs.
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

    The function computes the photoemission intensity for one initial-state
    orbital. The quantum numbers (l, m) and the sampled radial wavefunction
    define this orbital. The intensity is the squared modulus of the complex
    dipole matrix element:

    .. math::

        I(\mathbf{k}) = |M(\mathbf{k}, l, m)|^2

    This wrapper calls `dipole_matrix_element_single` and returns
    :math:`|M|^2 = M \cdot M^*`. The construction gives a real, nonnegative
    result. JAX differentiates the result with respect to all continuous
    inputs.

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

    The function scans every orbital in the Slater basis. It computes each
    radial wavefunction from ``slater_params`` and then computes the squared
    dipole matrix element. ``jax.lax.switch`` specializes one branch for each
    static ``(n, l, m)`` tuple. ``jax.lax.scan`` carries the differentiable
    arrays along the orbital axis.

    For each orbital *o*, the function:

    1. Extracts quantum numbers :math:`(n_o, l_o, m_o)` and Slater
       exponent :math:`\zeta_o` from ``slater_params``.
    2. Evaluates the normalized Slater radial function
       :math:`R(r) = N r^{n-1} e^{-\zeta r}` on the supplied grid.
    3. Weights by the multi-zeta coefficient
       ``slater_params.coefficients[o, 0]`` (first column for
       single-zeta bases).
    4. Calls `dipole_intensity_orbital` to compute :math:`|M|^2`.

    The function stacks the results into a one-dimensional array. Its length
    equals the number of orbitals in the basis.

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
