"""Tight-binding model and diagonalized band data structures.

Extended Summary
----------------
Defines PyTree types for tight-binding model parameters and
diagonalized electronic structure. ``DiagonalizedBands`` is the
common interface between TB-derived and VASP-derived inputs for
the differentiable forward simulator.

Routine Listings
----------------
:class:`DiagonalizedBands`
    PyTree for diagonalized electronic structure.
:class:`TBModel`
    PyTree for tight-binding model parameters (legacy).
:func:`make_diagonalized_bands`
    Create a validated ``DiagonalizedBands`` instance.
:func:`make_1d_chain_model`
    Create a 1D chain tight-binding model.
:func:`make_graphene_model`
    Create a graphene pz tight-binding model.
:func:`make_tb_model`
    Create a validated ``TBModel`` instance.

Notes
-----
``DiagonalizedBands`` has all-array children (fully differentiable).
``TBModel`` separates differentiable hopping amplitudes (children)
from static structural metadata (auxiliary data).
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Complex, Float, jaxtyped

from .aliases import ScalarFloat, ScalarNumeric
from .radial_params import OrbitalBasis, make_orbital_basis


class DiagonalizedBands(eqx.Module):
    """PyTree for diagonalized electronic structure.

    Extended Summary
    ----------------
    The common interface between VASP-derived and TB-derived inputs
    for the forward simulator ``simulate_tb_radial``. The native
    ``diffpes.tightb.diagonalize_tb`` producer constructs this PyTree
    from a ``TBModel``; the VASP adapter ``vasp_to_diagonalized``
    constructs it from VASP eigenvectors.

    By unifying both data sources into a single PyTree type, the
    differentiable forward simulator can accept inputs from either
    tight-binding models or first-principles calculations without
    code branching. The eigenvectors carry the orbital-decomposition
    information needed to compute dipole matrix elements in the
    Chinook pipeline.

    All four fields are dense JAX arrays stored as children (no
    auxiliary data), making the entire object fully differentiable.
    This enables end-to-end gradient computation from TB hopping
    parameters through diagonalization to simulated ARPES intensity.

    Attributes
    ----------
    eigenvalues : Float[Array, "K B"]
        Band energies E_n(k) in eV for K k-points and B bands.
        JAX-traced (differentiable).
    eigenvectors : Complex[Array, "K B O"]
        Complex orbital coefficients c_{k,b,orb} from the
        Hamiltonian diagonalization, where O is the number of
        orbitals in the basis. These encode the orbital character
        of each Bloch state and enter the dipole matrix element
        calculation. JAX-traced (differentiable, complex128).
    kpoints : Float[Array, "K 3"]
        k-point coordinates in reciprocal (fractional) space.
        JAX-traced (differentiable).
    fermi_energy : Float[Array, " "]
        Fermi level in eV. A 0-D scalar array.
        JAX-traced (differentiable).

    Notes
    -----
    Implemented as an immutable :class:`equinox.Module` PyTree. All four
    fields are differentiable leaves and no static metadata is present.

    See Also
    --------
    TBModel : The tight-binding model whose diagonalization produces
        a ``DiagonalizedBands``.
    make_diagonalized_bands : Factory function with validation and
        dtype casting.
    """

    eigenvalues: Float[Array, "K B"]
    eigenvectors: Complex[Array, "K B O"]
    kpoints: Float[Array, "K 3"]
    fermi_energy: Float[Array, " "]


class TBModel(eqx.Module):
    """Tight-binding parameters with differentiable and static fields.

    ``hopping_params`` and ``lattice_vectors`` are differentiable leaves.
    The connectivity, orbital count, and basis are compile-time metadata;
    changing any static field changes the PyTree definition and retraces JIT
    compiled consumers.

    Attributes
    ----------
    hopping_params : Float[Array, " H"]
        Hopping amplitudes in eV.
    lattice_vectors : Float[Array, "3 3"]
        Real-space lattice vectors in Angstrom.
    hopping_indices : tuple
        **Static.** Orbital connectivity and lattice translations.
    n_orbitals : int
        **Static.** Number of orbitals in the unit cell.
    orbital_basis : OrbitalBasis
        **Static.** Orbital quantum-number metadata.
    """

    hopping_params: Float[Array, " H"]
    lattice_vectors: Float[Array, "3 3"]
    hopping_indices: tuple = eqx.field(static=True)
    n_orbitals: int = eqx.field(static=True)
    orbital_basis: OrbitalBasis = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def make_diagonalized_bands(
    eigenvalues: Float[Array, "K B"],
    eigenvectors: Complex[Array, "K B O"],
    kpoints: Float[Array, "K 3"],
    fermi_energy: ScalarNumeric = 0.0,
) -> DiagonalizedBands:
    """Create a validated ``DiagonalizedBands`` instance.

    Extended Summary
    ----------------
    Factory function that validates and normalises diagonalized
    electronic structure data before constructing a
    ``DiagonalizedBands`` PyTree. Real-valued arrays are cast to
    ``float64``; the complex eigenvector array is cast to
    ``complex128`` to maintain full double-precision accuracy in
    the orbital decomposition.

    The function is decorated with ``@jaxtyped(typechecker=beartype)``
    so that shape constraints (K must agree across all arrays, B and
    O must be consistent) are checked at call time.

    Use this factory when constructing ``DiagonalizedBands`` from
    either TB diagonalization output or VASP eigenvector data.

    Implementation Logic
    --------------------
    1. **Cast eigenvalues** to ``jnp.float64`` via ``jnp.asarray``.
    2. **Cast eigenvectors** to ``jnp.complex128`` via
       ``jnp.asarray`` to preserve full double-precision for the
       complex orbital coefficients.
    3. **Cast kpoints** to ``jnp.float64`` via ``jnp.asarray``.
    4. **Cast fermi_energy** scalar to a 0-D ``jnp.float64`` array.
    5. **Construct** the ``DiagonalizedBands`` Equinox module from all
       four validated arrays and return it.

    Parameters
    ----------
    eigenvalues : Float[Array, "K B"]
        Band energies E_n(k) in eV for K k-points and B bands.
    eigenvectors : Complex[Array, "K B O"]
        Complex orbital coefficients c_{k,b,orb} from Hamiltonian
        diagonalization, where O is the number of orbitals.
    kpoints : Float[Array, "K 3"]
        k-point coordinates in reciprocal (fractional) space.
    fermi_energy : ScalarNumeric, optional
        Fermi level in eV. Default is 0.0.

    Returns
    -------
    bands : DiagonalizedBands
        Validated instance with ``float64`` eigenvalues/kpoints
        and ``complex128`` eigenvectors.

    See Also
    --------
    DiagonalizedBands : The PyTree class constructed by this factory.
    """
    eig_arr: Float[Array, "K B"] = jnp.asarray(eigenvalues, dtype=jnp.float64)
    vec_arr: Complex[Array, "K B O"] = jnp.asarray(
        eigenvectors, dtype=jnp.complex128
    )
    kpt_arr: Float[Array, "K 3"] = jnp.asarray(kpoints, dtype=jnp.float64)
    ef_arr: Float[Array, " "] = jnp.asarray(fermi_energy, dtype=jnp.float64)

    def validate_and_create() -> DiagonalizedBands:
        nonlocal ef_arr, eig_arr, kpt_arr
        eig_arr = eqx.error_if(
            eig_arr,
            ~(jnp.all(jnp.isfinite(eig_arr))),
            "make_diagonalized_bands: eigenvalues finite",
        )
        kpt_arr = eqx.error_if(
            kpt_arr,
            ~(jnp.all(jnp.isfinite(kpt_arr))),
            "make_diagonalized_bands: kpoints finite",
        )
        ef_arr = eqx.error_if(
            ef_arr,
            ~(jnp.isfinite(ef_arr)),
            "make_diagonalized_bands: fermi energy finite",
        )
        return DiagonalizedBands(
            eigenvalues=eig_arr,
            eigenvectors=vec_arr,
            kpoints=kpt_arr,
            fermi_energy=ef_arr,
        )

    bands: DiagonalizedBands = validate_and_create()
    return bands


@jaxtyped(typechecker=beartype)
def make_tb_model(
    hopping_params: Float[Array, " H"],
    lattice_vectors: Float[Array, "3 3"],
    hopping_indices: tuple,
    n_orbitals: int,
    orbital_basis: OrbitalBasis,
) -> TBModel:
    """Create a validated ``TBModel`` instance.

    Extended Summary
    ----------------
    Factory function that validates and normalises tight-binding
    model parameters before constructing a ``TBModel`` PyTree. The
    two differentiable arrays (``hopping_params``,
    ``lattice_vectors``) are cast to ``float64``; the three static
    fields are passed through unchanged as auxiliary data.

    The function is decorated with ``@jaxtyped(typechecker=beartype)``
    so that shape constraints on the differentiable arrays are
    checked at call time.

    Use this factory to build a ``TBModel`` from raw hopping data
    (e.g. from a Slater-Koster parameterization) before passing it
    to the Hamiltonian construction and diagonalization routines.

    Implementation Logic
    --------------------
    1. **Cast hopping_params** to ``jnp.float64`` via
       ``jnp.asarray``.
    2. **Cast lattice_vectors** to ``jnp.float64`` via
       ``jnp.asarray``.
    3. **Pass through** ``hopping_indices``, ``n_orbitals``, and
       ``orbital_basis`` unchanged -- these become auxiliary data
       in the PyTree.
    4. **Construct** the ``TBModel`` Equinox module from all five fields
       and return it.

    Parameters
    ----------
    hopping_params : Float[Array, " H"]
        Hopping amplitudes t_{ij,R} in eV, one per connection.
    lattice_vectors : Float[Array, "3 3"]
        Real-space lattice vectors as rows, in Angstroms.
    hopping_indices : tuple
        Connectivity: ``(orb_i, orb_j, (R_x, R_y, R_z))`` per
        hopping, where R_x, R_y, R_z are integer lattice translation
        indices.
    n_orbitals : int
        Number of orbitals in the unit cell.
    orbital_basis : OrbitalBasis
        Quantum number metadata for each orbital.

    Returns
    -------
    model : TBModel
        Validated tight-binding model with ``float64`` arrays and
        static structural metadata.

    See Also
    --------
    TBModel : The PyTree class constructed by this factory.
    make_orbital_basis : Factory for the ``OrbitalBasis`` argument.
    """
    hop_arr: Float[Array, " H"] = jnp.asarray(
        hopping_params, dtype=jnp.float64
    )
    lat_arr: Float[Array, "3 3"] = jnp.asarray(
        lattice_vectors, dtype=jnp.float64
    )

    def validate_and_create() -> TBModel:
        nonlocal hop_arr, lat_arr
        hop_arr = eqx.error_if(
            hop_arr,
            ~(jnp.all(jnp.isfinite(hop_arr))),
            "make_tb_model: hopping finite",
        )
        lat_arr = eqx.error_if(
            lat_arr,
            ~(jnp.all(jnp.isfinite(lat_arr))),
            "make_tb_model: lattice finite",
        )
        return TBModel(
            hopping_params=hop_arr,
            lattice_vectors=lat_arr,
            hopping_indices=hopping_indices,
            n_orbitals=n_orbitals,
            orbital_basis=orbital_basis,
        )

    model: TBModel = validate_and_create()
    return model


@jaxtyped(typechecker=beartype)
def make_1d_chain_model(
    t: ScalarFloat = -1.0,
) -> TBModel:
    r"""Create a 1D chain tight-binding model.

    Single orbital per unit cell with nearest-neighbor hopping t.

    Extended Summary
    ----------------
    The 1D chain is the simplest possible tight-binding model: one
    s-orbital per unit cell with hopping only to the two nearest
    neighbors at lattice vectors ``+a1`` and ``-a1``.  The lattice is
    set to an identity matrix (``a1 = [1, 0, 0]``, etc.) so that
    fractional and Cartesian coordinates coincide with a lattice
    constant of 1 (arbitrary units).

    The resulting band dispersion is the textbook cosine band:

    .. math::

        E(k) = 2t \cos(2 \pi k)

    with bandwidth ``|4t|``.  This model is useful as a minimal
    smoke-test for the Hamiltonian builder, diagonalizer, and
    gradient machinery.

    The hopping list contains two entries -- ``(0, 0, (+1,0,0))`` and
    ``(0, 0, (-1,0,0))`` -- which are the forward and backward
    nearest-neighbor hops of the single orbital to itself in adjacent
    unit cells.  After Hermitianization in ``build_hamiltonian_k``
    these are redundant (each is its own conjugate), so the
    on-diagonal entry receives ``2t cos(2 pi k)`` as expected.

    Parameters
    ----------
    t : ScalarFloat
        Hopping amplitude. Default -1.0.

    Returns
    -------
    model : TBModel
        1D chain model.
    """
    basis: OrbitalBasis = make_orbital_basis(
        n_values=(1,),
        l_values=(0,),
        m_values=(0,),
        labels=("s",),
    )
    hopping_indices: tuple[tuple[int, int, tuple[int, int, int]], ...] = (
        (0, 0, (1, 0, 0)),  # +R hop
        (0, 0, (-1, 0, 0)),  # -R hop
    )
    lattice: Float[Array, "3 3"] = jnp.eye(3, dtype=jnp.float64)
    model: TBModel = make_tb_model(
        hopping_params=jnp.array([t, t], dtype=jnp.float64),
        lattice_vectors=lattice,
        hopping_indices=hopping_indices,
        n_orbitals=1,
        orbital_basis=basis,
    )
    return model


@jaxtyped(typechecker=beartype)
def make_graphene_model(
    t: ScalarFloat = -2.7,
) -> TBModel:
    """Create a graphene pz tight-binding model.

    Two-orbital (A/B sublattice) model on a honeycomb lattice
    with nearest-neighbor hopping t.

    Extended Summary
    ----------------
    Graphene's honeycomb lattice has two atoms (sublattices A and B)
    per primitive cell.  The lattice vectors used here are:

    * ``a1 = (a, 0, 0)``
    * ``a2 = (a/2, a*sqrt(3)/2, 0)``
    * ``a3 = (0, 0, 10)``  (vacuum slab for 2-D periodicity)

    with ``a = 2.46`` Angstrom (the experimental graphene lattice
    constant).  The two orbitals are labeled ``A_pz`` and ``B_pz``
    with quantum numbers ``(n=2, l=1, m=0)``, representing carbon
    p_z orbitals on each sublattice.

    Each A-site atom has three nearest-neighbor B-site atoms.  In
    fractional coordinates the three A -> B hoppings connect to cells
    ``(0,0,0)``, ``(-1,0,0)``, and ``(0,-1,0)``.  The reverse B -> A
    hoppings at ``(0,0,0)``, ``(+1,0,0)``, and ``(0,+1,0)`` are
    listed explicitly so that the raw Hamiltonian matrix is already
    nearly Hermitian before the Hermitianization step in
    ``build_hamiltonian_k``.

    The resulting 2x2 Hamiltonian produces the classic Dirac-cone
    band structure with linear dispersion near the K and K' points
    and a bandwidth of ``|6t|``.

    Parameters
    ----------
    t : ScalarFloat
        Nearest-neighbor hopping. Default -2.7 eV.

    Returns
    -------
    model : TBModel
        Graphene model.

    Notes
    -----
    The default hopping value of -2.7 eV reproduces the standard
    nearest-neighbor graphene band structure commonly used in the
    literature (e.g. Castro Neto et al., Rev. Mod. Phys. 81, 109).
    The negative sign follows the convention that bonding states
    are lower in energy.
    """
    basis: OrbitalBasis = make_orbital_basis(
        n_values=(2, 2),
        l_values=(1, 1),
        m_values=(0, 0),
        labels=("A_pz", "B_pz"),
    )
    # Honeycomb lattice vectors (Angstrom)
    a: float = 2.46
    a1: Float[Array, " 3"] = jnp.array([a, 0.0, 0.0], dtype=jnp.float64)
    a2: Float[Array, " 3"] = jnp.array(
        [a / 2.0, a * jnp.sqrt(3.0) / 2.0, 0.0], dtype=jnp.float64
    )
    a3: Float[Array, " 3"] = jnp.array([0.0, 0.0, 10.0], dtype=jnp.float64)
    lattice: Float[Array, "3 3"] = jnp.stack([a1, a2, a3])

    # Three nearest-neighbor hoppings A->B
    hopping_indices: tuple[tuple[int, int, tuple[int, int, int]], ...] = (
        (0, 1, (0, 0, 0)),  # same cell
        (0, 1, (-1, 0, 0)),  # -a1
        (0, 1, (0, -1, 0)),  # -a2
        # Hermitian conjugates (B->A)
        (1, 0, (0, 0, 0)),  # same cell
        (1, 0, (1, 0, 0)),  # +a1
        (1, 0, (0, 1, 0)),  # +a2
    )
    t_val: Float[Array, " "] = jnp.asarray(t, dtype=jnp.float64)
    hopping_params: Float[Array, " H"] = jnp.array(
        [t_val, t_val, t_val, t_val, t_val, t_val], dtype=jnp.float64
    )
    model: TBModel = make_tb_model(
        hopping_params=hopping_params,
        lattice_vectors=lattice,
        hopping_indices=hopping_indices,
        n_orbitals=2,
        orbital_basis=basis,
    )
    return model


__all__: list[str] = [
    "DiagonalizedBands",
    "TBModel",
    "make_1d_chain_model",
    "make_diagonalized_bands",
    "make_graphene_model",
    "make_tb_model",
]
