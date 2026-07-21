"""Crystal geometry data structure for VASP crystal structures.

Extended Summary
----------------
Defines the :class:`CrystalGeometry` PyTree for representing
crystal structures parsed from VASP POSCAR files. Includes real-space
lattice vectors, reciprocal lattice, atomic coordinates, element
symbols, and atom counts per species.

Routine Listings
----------------
:class:`CrystalGeometry`
    Store VASP POSCAR crystal geometry in a JAX PyTree.
:func:`make_crystal_geometry`
    Create a validated CrystalGeometry instance.

Notes
-----
The ``symbols`` field is stored as auxiliary data in the PyTree
since JAX cannot trace Python strings. All numeric fields are
stored as JAX arrays for compatibility with JAX transformations.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Union
from jax.core import Tracer
from jaxtyping import Array, Float, Int, jaxtyped

from .aliases import ScalarNumeric

_MAX_LATTICE_CONDITION_NUMBER: float = 1e12
_MIN_SCALED_SINGULAR_VALUE: float = 1e-12


class CrystalGeometry(eqx.Module):
    """Store VASP POSCAR crystal geometry in a JAX PyTree.

    Encapsulates the full crystal structure information parsed from a
    VASP POSCAR file: real-space lattice vectors, the corresponding
    reciprocal lattice, fractional atomic coordinates, per-species
    element symbols, and per-species atom counts. Together these fields
    fully describe the periodic crystal needed for ARPES simulation.

    This class is registered as a JAX PyTree via
    reciprocal_lattice, coords, atom_counts) are stored as children
    visible to JAX tracing, while the ``symbols`` tuple of Python
    strings is stored as auxiliary data because JAX cannot trace
    Python strings.


    :see: :class:`~.test_geometry.TestCrystalGeometry`

    Attributes
    ----------
    lattice : Float[Array, "3 3"]
        Real-space lattice vectors as rows (angstroms).
    reciprocal_lattice : Float[Array, "3 3"]
        Reciprocal lattice vectors as rows (1/angstroms).
    coords : Float[Array, "N 3"]
        Fractional atomic coordinates.
    symbols : tuple[str, ...]
        Element symbols for each species (**static** -- compile-time
        constants; changing them triggers retracing).
    atom_counts : Int[Array, " S"]
        Number of atoms per species.

    Notes
    -----
    The ``symbols`` field is a tuple of Python strings declared with
    ``eqx.field(static=True)`` rather than as a traced leaf. Changing it
    triggers recompilation of any ``jit``-compiled function that
    receives this PyTree.

    See Also
    --------
    make_crystal_geometry : Factory function with validation, float64
        casting, and automatic reciprocal lattice computation.
    """

    lattice: Float[Array, "3 3"]
    reciprocal_lattice: Float[Array, "3 3"]
    coords: Float[Array, "N 3"]
    symbols: tuple[str, ...] = eqx.field(static=True)
    atom_counts: Int[Array, " S"]


def _compute_reciprocal_lattice(
    lattice: Float[Array, "3 3"],
) -> Float[Array, "3 3"]:
    r"""Compute reciprocal lattice vectors from real-space lattice.

    Derives the reciprocal lattice from the real-space lattice using
    the standard cross-product formula. The reciprocal lattice vectors
    ``b_i`` satisfy ``a_i . b_j = 2 pi delta_{ij}``.

    Implementation Logic
    --------------------
    Given real-space lattice vectors ``a1, a2, a3`` (rows of
    ``lattice``):

    1. **Extract rows**: ``a1 = lattice[0]``, ``a2 = lattice[1]``,
       ``a3 = lattice[2]``.
    2. **Compute unit-cell volume**:
       ``V = a1 . (a2 x a3)`` (scalar triple product).
    3. **Compute reciprocal vectors** via the cross-product formula:

       .. math::

           b_1 = 2\\pi \\, (a_2 \\times a_3) / V

           b_2 = 2\\pi \\, (a_3 \\times a_1) / V

           b_3 = 2\\pi \\, (a_1 \\times a_2) / V

    4. **Stack** ``[b1, b2, b3]`` into a (3, 3) array with
       reciprocal vectors as rows.

    Parameters
    ----------
    lattice : Float[Array, "3 3"]
        Real-space lattice vectors as rows.

    Returns
    -------
    reciprocal : Float[Array, "3 3"]
        Reciprocal lattice vectors as rows (units of 1/angstrom,
        with the 2 pi prefactor included).
    """
    a1: Float[Array, " 3"] = lattice[0]
    a2: Float[Array, " 3"] = lattice[1]
    a3: Float[Array, " 3"] = lattice[2]
    volume: Float[Array, " "] = jnp.dot(a1, jnp.cross(a2, a3))
    b1: Float[Array, " 3"] = 2.0 * jnp.pi * jnp.cross(a2, a3) / volume
    b2: Float[Array, " 3"] = 2.0 * jnp.pi * jnp.cross(a3, a1) / volume
    b3: Float[Array, " 3"] = 2.0 * jnp.pi * jnp.cross(a1, a2) / volume
    reciprocal: Float[Array, "3 3"] = jnp.stack([b1, b2, b3])
    return reciprocal


@jaxtyped(typechecker=beartype)
def make_crystal_geometry(  # noqa: DOC503
    lattice: Union[Float[Array, "3 3"], "list[list[ScalarNumeric]]"],
    coords: Float[Array, "N 3"],
    symbols: tuple[str, ...],
    atom_counts: Union[Int[Array, " S"], "list[int]"],
) -> CrystalGeometry:
    """Create a validated CrystalGeometry instance.

    Validates and normalises the inputs, then automatically computes
    the reciprocal lattice from the real-space lattice so the caller
    does not need to supply it. Numeric arrays are cast to the
    appropriate JAX dtypes (float64 for real-valued fields, int32
    for atom counts).

    :see: :class:`~.test_geometry.TestMakeCrystalGeometry`

    Implementation Logic
    --------------------
    1. **Prepare the normalized values**::

           lattice_arr = jnp.asarray(lattice, dtype=jnp.float64)

       This expression gives the later validation steps a stable shape and
       dtype.

    2. **Apply static validation**::

           symbols and len(symbols) != counts_arr.shape[0]

       This predicate rejects invalid structure before JAX traces the
       numerical checks.

    3. **Apply traced validation**::

           jnp.sum(counts_arr) != coords_arr.shape[0]

       This predicate remains active during eager and compiled execution.

    4. **Return the named instance**::

           return geometry

       The explicit name keeps the implementation and the Returns section
       synchronized.

    Parameters
    ----------
    lattice : Union[Float[Array, "3 3"], "list[list[ScalarNumeric]]"]
        Real-space lattice vectors as rows (angstroms).
    coords : Float[Array, "N 3"]
        Fractional atomic coordinates.
    symbols : tuple[str, ...]
        Element symbols for each species (**static** -- a compile-time
        constant; changing them triggers retracing).
    atom_counts : Union[Int[Array, " S"], "list[int]"]
        Number of atoms per species.

    Returns
    -------
    geometry : CrystalGeometry
        Validated crystal geometry instance with the reciprocal
        lattice pre-computed.

    Raises
    ------
    ValueError
        If a non-empty symbol tuple differs from the number of species,
        or if the atom counts do not sum to the number of positions. Empty
        symbols remain valid for VASP 4 files without a species line.
    EquinoxRuntimeError
        If coordinates or lattice entries are non-finite, if the
        lattice is not right-handed, or if its scaled smallest
        singular value or condition number violates the named
        numerical stability limits.

    Notes
    -----
    Static validation raises ``ValueError`` before traced construction when
    species counts disagree or concrete atom counts do not sum to positions.
    Traced validation uses ``eqx.error_if`` and raises
    ``EquinoxRuntimeError`` for inconsistent traced counts, non-finite data,
    or a lattice that fails the orientation and conditioning limits.

    See Also
    --------
    CrystalGeometry : The PyTree class constructed by this factory.
    _compute_reciprocal_lattice : Cross-product formula used to
        derive the reciprocal lattice.
    """
    lattice_arr: Float[Array, "3 3"] = jnp.asarray(lattice, dtype=jnp.float64)
    coords_arr: Float[Array, "N 3"] = jnp.asarray(coords, dtype=jnp.float64)
    counts_arr: Int[Array, " S"] = jnp.asarray(atom_counts, dtype=jnp.int32)

    if symbols and len(symbols) != counts_arr.shape[0]:
        msg: str = "make_crystal_geometry: symbols and atom_counts must agree"
        raise ValueError(msg)
    if (
        not isinstance(counts_arr, Tracer)
        and int(jnp.sum(counts_arr)) != coords_arr.shape[0]
    ):
        msg: str = "make_crystal_geometry: atom_counts must sum to positions"
        raise ValueError(msg)

    def validate_and_create() -> CrystalGeometry:
        nonlocal coords_arr, counts_arr, lattice_arr
        counts_arr = eqx.error_if(
            counts_arr,
            jnp.sum(counts_arr) != coords_arr.shape[0],
            "make_crystal_geometry: atom_counts must sum to positions",
        )
        lattice_arr = eqx.error_if(
            lattice_arr,
            ~(jnp.all(jnp.isfinite(lattice_arr))),
            "make_crystal_geometry: lattice finite",
        )
        determinant: Float[Array, " "] = jnp.linalg.det(lattice_arr)
        lattice_arr = eqx.error_if(
            lattice_arr,
            ~(determinant > 0.0),
            "make_crystal_geometry: lattice must be right-handed",
        )
        singular_values: Float[Array, " 3"] = jnp.linalg.svdvals(lattice_arr)
        largest: Float[Array, " "] = singular_values[0]
        smallest: Float[Array, " "] = singular_values[-1]
        scaled_smallest: Float[Array, " "] = smallest / largest
        lattice_arr = eqx.error_if(
            lattice_arr,
            scaled_smallest < _MIN_SCALED_SINGULAR_VALUE,
            "make_crystal_geometry: scaled singular value below limit",
        )
        condition_number: Float[Array, " "] = largest / smallest
        lattice_arr = eqx.error_if(
            lattice_arr,
            condition_number > _MAX_LATTICE_CONDITION_NUMBER,
            "make_crystal_geometry: lattice condition number exceeds limit",
        )
        coords_arr = eqx.error_if(
            coords_arr,
            ~(jnp.all(jnp.isfinite(coords_arr))),
            "make_crystal_geometry: coords finite",
        )
        reciprocal: Float[Array, "3 3"] = _compute_reciprocal_lattice(
            lattice_arr
        )
        validated_geometry: CrystalGeometry = CrystalGeometry(
            lattice=lattice_arr,
            reciprocal_lattice=reciprocal,
            coords=coords_arr,
            symbols=symbols,
            atom_counts=counts_arr,
        )
        return validated_geometry

    geometry: CrystalGeometry = validate_and_create()
    return geometry


__all__: list[str] = [
    "CrystalGeometry",
    "make_crystal_geometry",
]
