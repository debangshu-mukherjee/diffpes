"""Define tight-binding model and diagonalized-band data structures.

Extended Summary
----------------
This module defines PyTree types for tight-binding model parameters and
diagonalized electronic structure. ``DiagonalizedBands`` is the
common interface between TB-derived and VASP-derived inputs for
the differentiable forward simulator.

Routine Listings
----------------
:class:`DiagonalizedBands`
    Store diagonalized electronic-structure data in a JAX PyTree.
:class:`TBModel`
    Store tight-binding parameters in a JAX PyTree.
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
    """Store diagonalized electronic-structure data in a JAX PyTree.

    This type provides the common interface between VASP-derived and
    TB-derived inputs for the forward simulator ``simulate_tb_radial``. The
    native
    ``diffpes.tightb.diagonalize_tb`` producer constructs this PyTree
    from a ``TBModel``; the VASP adapter ``vasp_to_diagonalized``
    constructs it from VASP eigenvectors.

    This single PyTree type lets the differentiable forward simulator accept
    either tight-binding or first-principles inputs without code branches.
    The eigenvectors carry the orbital-decomposition
    information needed to compute dipole matrix elements in the
    Chinook pipeline.

    All four fields are dense JAX arrays stored as children (no
    auxiliary data), making the entire object fully differentiable.
    This enables end-to-end gradient computation from TB hopping
    parameters through diagonalization to simulated ARPES intensity.


    :see: :class:`~.test_tb_model.TestDiagonalizedBands`

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
    """Store tight-binding parameters in a JAX PyTree.

    ``hopping_params`` and ``lattice_vectors`` are differentiable leaves.
    The connectivity, orbital count, and basis are compile-time metadata;
    changing any static field changes the PyTree definition and retraces JIT
    compiled consumers.


    :see: :class:`~.test_tb_model.TestTBModel`

    Attributes
    ----------
    hopping_params : Float[Array, " H"]
        Hopping amplitudes in eV.
    lattice_vectors : Float[Array, "3 3"]
        Real-space lattice vectors in Angstrom.
    hopping_indices : tuple
        Orbital connectivity and lattice translations (**static** -- a
        compile-time constant; changing it triggers retracing).
    n_orbitals : int
        Number of orbitals in the unit cell (**static** -- a compile-time
        constant; changing it triggers retracing).
    orbital_basis : OrbitalBasis
        Orbital quantum-number metadata (**static** -- a compile-time
        constant; changing it triggers retracing).
    """

    hopping_params: Float[Array, " H"]
    lattice_vectors: Float[Array, "3 3"]
    hopping_indices: tuple = eqx.field(static=True)
    n_orbitals: int = eqx.field(static=True)
    orbital_basis: OrbitalBasis = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def make_diagonalized_bands(  # noqa: DOC503
    eigenvalues: Float[Array, "Ke Be"],
    eigenvectors: Complex[Array, "Kv Bv O"],
    kpoints: Float[Array, "Kk 3"],
    fermi_energy: ScalarNumeric = 0.0,
) -> DiagonalizedBands:
    """Create a validated ``DiagonalizedBands`` instance.

    The factory validates and normalizes diagonalized
    electronic structure data before constructing a
    ``DiagonalizedBands`` PyTree. It casts real-valued arrays to ``float64``
    and the complex eigenvector array to
    ``complex128`` to maintain full double-precision accuracy in
    the orbital decomposition.

    ``@jaxtyped(typechecker=beartype)`` checks the shape constraints at call
    time. K must agree across all arrays, and B and O must be consistent.

    Use this factory when constructing ``DiagonalizedBands`` from
    either TB diagonalization output or VASP eigenvector data.

    :see: :class:`~.test_tb_model.TestMakeDiagonalizedBands`

    Implementation Logic
    --------------------
    1. **Prepare the normalized values**::

           eig_arr = jnp.asarray(eigenvalues, dtype=jnp.float64)

       This expression gives the later validation steps a stable shape and
       dtype.

    2. **Apply static validation**::

           eig_arr.shape != vec_arr.shape[:2]

       This predicate rejects invalid structure before JAX traces the
       numerical checks.

    3. **Apply traced validation**::

           ~jnp.all(jnp.isfinite(eig_arr))

       This predicate remains active during eager and compiled execution.

    4. **Return the named instance**::

           return bands

       The explicit name keeps the implementation and the Returns section
       synchronized.

    Parameters
    ----------
    eigenvalues : Float[Array, "Ke Be"]
        Band energies E_n(k) in eV for K k-points and B bands.
    eigenvectors : Complex[Array, "Kv Bv O"]
        Complex orbital coefficients c_{k,b,orb} from Hamiltonian
        diagonalization, where O is the number of orbitals.
    kpoints : Float[Array, "Kk 3"]
        k-point coordinates in reciprocal (fractional) space.
    fermi_energy : ScalarNumeric, optional
        Fermi level in eV. Default is 0.0.

    Returns
    -------
    bands : DiagonalizedBands
        Validated instance with ``float64`` eigenvalues/kpoints
        and ``complex128`` eigenvectors.

    Raises
    ------
    ValueError
        If the k-point or band dimensions disagree across the arrays.
    EquinoxRuntimeError
        If eigenvalues, eigenvectors, k-points, or the Fermi energy
        contain a non-finite value.

    Notes
    -----
    Static validation raises ``ValueError`` before traced construction when
    the k-point or band dimensions disagree. Traced validation uses
    ``eqx.error_if`` and raises ``EquinoxRuntimeError`` when any numerical
    field contains a non-finite value.

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
    if eig_arr.shape != vec_arr.shape[:2]:
        msg: str = "eigenvalues and eigenvectors must agree on K and B"
        raise ValueError(msg)
    if eig_arr.shape[0] != kpt_arr.shape[0]:
        msg: str = "eigenvalues and kpoints must agree on K"
        raise ValueError(msg)

    def validate_and_create() -> DiagonalizedBands:
        nonlocal ef_arr, eig_arr, kpt_arr, vec_arr
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
        vec_arr = eqx.error_if(
            vec_arr,
            ~(jnp.all(jnp.isfinite(vec_arr))),
            "make_diagonalized_bands: eigenvectors finite",
        )
        ef_arr = eqx.error_if(
            ef_arr,
            ~(jnp.isfinite(ef_arr)),
            "make_diagonalized_bands: fermi energy finite",
        )
        validated_bands: DiagonalizedBands = DiagonalizedBands(
            eigenvalues=eig_arr,
            eigenvectors=vec_arr,
            kpoints=kpt_arr,
            fermi_energy=ef_arr,
        )
        return validated_bands

    bands: DiagonalizedBands = validate_and_create()
    return bands


@jaxtyped(typechecker=beartype)
def make_tb_model(  # noqa: DOC503
    hopping_params: Float[Array, " H"],
    lattice_vectors: Float[Array, "3 3"],
    hopping_indices: tuple,
    n_orbitals: int,
    orbital_basis: OrbitalBasis,
) -> TBModel:
    """Create a validated ``TBModel`` instance.

    The factory validates and normalizes tight-binding
    model parameters before constructing a ``TBModel`` PyTree. The
    factory casts the two differentiable arrays to ``float64``. It passes the
    three static fields unchanged as auxiliary data.

    ``@jaxtyped(typechecker=beartype)`` checks the shape constraints on the
    differentiable arrays at call time.

    Use this factory to build a ``TBModel`` from raw hopping data. Then pass
    the model to the Hamiltonian construction and diagonalization routines.

    :see: :class:`~.test_tb_model.TestMakeTBModel`

    Implementation Logic
    --------------------
    1. **Prepare the normalized values**::

           hop_arr = jnp.asarray(hopping_params, dtype=jnp.float64)

       This expression gives the later validation steps a stable shape and
       dtype.

    2. **Apply static validation**::

           hop_arr.shape[0] != len(hopping_indices)

       This predicate rejects invalid structure before JAX traces the
       numerical checks.

    3. **Apply traced validation**::

           ~jnp.all(jnp.isfinite(hop_arr))

       This predicate remains active during eager and compiled execution.

    4. **Return the named instance**::

           return model

       The explicit name keeps the implementation and the Returns section
       synchronized.

    Parameters
    ----------
    hopping_params : Float[Array, " H"]
        Hopping amplitudes t_{ij,R} in eV, one per connection.
    lattice_vectors : Float[Array, "3 3"]
        Real-space lattice vectors as rows, in Angstroms.
    hopping_indices : tuple
        Connectivity: ``(orb_i, orb_j, (R_x, R_y, R_z))`` per
        hopping, where R_x, R_y, R_z are integer lattice translation
        indices (**static** -- a compile-time constant; changing it triggers
        retracing).
    n_orbitals : int
        Number of orbitals in the unit cell (**static** -- a compile-time
        constant; changing it triggers retracing).
    orbital_basis : OrbitalBasis
        Quantum number metadata for each orbital (**static** -- a compile-time
        constant; changing it triggers retracing).

    Returns
    -------
    model : TBModel
        Validated tight-binding model with ``float64`` arrays and
        static structural metadata.

    Raises
    ------
    ValueError
        If the hopping parameter count does not match the connectivity,
        an orbital index lies outside ``[0, n_orbitals)``, or the orbital
        basis size differs from ``n_orbitals``.
    EquinoxRuntimeError
        If hoppings or lattice vectors are non-finite, or if the lattice
        is degenerate.

    Notes
    -----
    Static validation raises ``ValueError`` before traced construction when
    hopping metadata, orbital counts, or orbital indices are inconsistent.
    Traced validation uses ``eqx.error_if`` and raises
    ``EquinoxRuntimeError`` for non-finite arrays or a degenerate lattice.

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
    if hop_arr.shape[0] != len(hopping_indices):
        msg: str = "hopping_params and hopping_indices must have equal length"
        raise ValueError(msg)
    if len(orbital_basis.n_values) != n_orbitals:
        msg: str = "orbital_basis size must match n_orbitals"
        raise ValueError(msg)
    if any(
        index < 0 or index >= n_orbitals
        for hopping in hopping_indices
        for index in hopping[:2]
    ):
        msg: str = "hopping orbital indices must be in [0, n_orbitals)"
        raise ValueError(msg)

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
        lat_arr = eqx.error_if(
            lat_arr,
            jnp.linalg.det(lat_arr) == 0.0,
            "make_tb_model: lattice non-degenerate",
        )
        validated_model: TBModel = TBModel(
            hopping_params=hop_arr,
            lattice_vectors=lat_arr,
            hopping_indices=hopping_indices,
            n_orbitals=n_orbitals,
            orbital_basis=orbital_basis,
        )
        return validated_model

    model: TBModel = validate_and_create()
    return model


@jaxtyped(typechecker=beartype)
def make_1d_chain_model(
    t: ScalarFloat = -1.0,
) -> TBModel:
    r"""Create a 1D chain tight-binding model.

    The model has one orbital in each unit cell and nearest-neighbor hopping t.

    The 1D chain has one s-orbital in each unit cell. Hopping connects only the
    two nearest neighbors at lattice vectors ``+a1`` and ``-a1``. The function
    uses an identity lattice matrix. Therefore, fractional and Cartesian
    coordinates coincide for a lattice constant of 1 in arbitrary units.

    The resulting band dispersion is the standard cosine band:

    .. math::

        E(k) = 2t \cos(2 \pi k)

    with bandwidth ``|4t|``. This model provides a minimal test for the
    Hamiltonian builder, diagonalizer, and gradient machinery.

    The hopping list contains ``(0, 0, (+1,0,0))`` and
    ``(0, 0, (-1,0,0))``. These entries connect the single orbital to itself
    in adjacent unit cells. Hermitianization in ``build_hamiltonian_k`` makes
    these entries redundant because each is its own conjugate. The diagonal
    entry therefore receives ``2t cos(2 pi k)``.

    :see: :class:`~.test_tb_model.TestMake1dChainModel`

    Implementation Logic
    --------------------
    1. **Create the orbital basis**::

           basis = make_orbital_basis(...)

       The basis contains one s orbital per unit cell.
    2. **Define nearest-neighbor hops**::

           hopping_indices = ((0, 0, (1, 0, 0)), (0, 0, (-1, 0, 0)))

       The two entries connect the orbital to both adjacent unit cells.
    3. **Construct the model**::

           model = make_tb_model(...)

       The validated factory preserves gradients through the hopping value.

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
        (0, 0, (1, 0, 0)),
        (0, 0, (-1, 0, 0)),
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

    The model has two orbitals (A/B sublattices) on a honeycomb lattice. It
    uses nearest-neighbor hopping t.

    Graphene's honeycomb lattice has two atoms (sublattices A and B)
    per primitive cell.  The lattice vectors used here are:

    * ``a1 = (a, 0, 0)``
    * ``a2 = (a/2, a*sqrt(3)/2, 0)``
    * ``a3 = (0, 0, 10)``  (vacuum slab for 2-D periodicity)

    with ``a = 2.46`` Angstrom (the experimental graphene lattice
    constant). The function labels the two orbitals ``A_pz`` and ``B_pz``.
    Their quantum numbers are ``(n=2, l=1, m=0)``. These orbitals represent
    carbon p_z orbitals on each sublattice.

    Each A-site atom has three nearest-neighbor B-site atoms.  In
    fractional coordinates the three A -> B hoppings connect to cells
    ``(0,0,0)``, ``(-1,0,0)``, and ``(0,-1,0)``.  The reverse B -> A
    hoppings use ``(0,0,0)``, ``(+1,0,0)``, and ``(0,+1,0)``. The function
    lists these reverse hoppings explicitly. The raw Hamiltonian is therefore
    nearly Hermitian before ``build_hamiltonian_k`` applies Hermitianization.

    The resulting 2x2 Hamiltonian produces the classic Dirac-cone
    band structure with linear dispersion near the K and K' points
    and a bandwidth of ``|6t|``.

    :see: :class:`~.test_tb_model.TestMakeGrapheneModel`

    Implementation Logic
    --------------------
    1. **Create the orbital basis**::

           basis: OrbitalBasis = make_orbital_basis(
               n_values=(2, 2),
               l_values=(1, 1),
               m_values=(0, 0),
               labels=("A_pz", "B_pz"),
           )

       These static quantum numbers identify the two carbon p-z orbitals.

    2. **Construct the primitive lattice**::

           a: float = 2.46
           a1 = jnp.array([a, 0.0, 0.0], dtype=jnp.float64)
           a2 = jnp.array(
               [a / 2.0, a * jnp.sqrt(3.0) / 2.0, 0.0],
               dtype=jnp.float64,
           )
           a3 = jnp.array([0.0, 0.0, 10.0], dtype=jnp.float64)
           lattice = jnp.stack([a1, a2, a3])

       The first two vectors define the honeycomb plane. The third vector
       adds vacuum normal to that plane.

    3. **Enumerate the nearest-neighbor hoppings**::

           hopping_indices = (
               (0, 1, (0, 0, 0)),
               (0, 1, (-1, 0, 0)),
               (0, 1, (0, -1, 0)),
               (1, 0, (0, 0, 0)),
               (1, 0, (1, 0, 0)),
               (1, 0, (0, 1, 0)),
           )

       The forward and reverse entries preserve the Hermitian lattice model.

    4. **Construct the validated model**::

           t_val = jnp.asarray(t, dtype=jnp.float64)
           hopping_params = jnp.array(
               [t_val, t_val, t_val, t_val, t_val, t_val],
               dtype=jnp.float64,
           )
           model = make_tb_model(
               hopping_params=hopping_params,
               lattice_vectors=lattice,
               hopping_indices=hopping_indices,
               n_orbitals=2,
               orbital_basis=basis,
           )

       The shared JAX scalar keeps one differentiable hopping value for all
       six directed bonds. The factory applies the model validation contract.

    5. **Return the named instance**::

           return model

       The explicit name keeps the implementation and the Returns section
       synchronized.

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
    nearest-neighbor graphene band structure. Castro Neto et al. use this
    value in Rev. Mod. Phys. 81, 109.
    The negative sign follows the convention that bonding states
    are lower in energy.

    The conversion with ``jnp.asarray`` keeps ``t`` on the JAX gradient tape.
    Gradients of downstream spectra therefore propagate to the shared hopping.
    """
    basis: OrbitalBasis = make_orbital_basis(
        n_values=(2, 2),
        l_values=(1, 1),
        m_values=(0, 0),
        labels=("A_pz", "B_pz"),
    )
    a: float = 2.46
    a1: Float[Array, " 3"] = jnp.array([a, 0.0, 0.0], dtype=jnp.float64)
    a2: Float[Array, " 3"] = jnp.array(
        [a / 2.0, a * jnp.sqrt(3.0) / 2.0, 0.0], dtype=jnp.float64
    )
    a3: Float[Array, " 3"] = jnp.array([0.0, 0.0, 10.0], dtype=jnp.float64)
    lattice: Float[Array, "3 3"] = jnp.stack([a1, a2, a3])

    hopping_indices: tuple[tuple[int, int, tuple[int, int, int]], ...] = (
        (0, 1, (0, 0, 0)),
        (0, 1, (-1, 0, 0)),
        (0, 1, (0, -1, 0)),
        (1, 0, (0, 0, 0)),
        (1, 0, (1, 0, 0)),
        (1, 0, (0, 1, 0)),
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
