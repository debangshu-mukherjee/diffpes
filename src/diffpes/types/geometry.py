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
    PyTree for crystal geometry from VASP POSCAR.
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
from jax import lax
from jaxtyping import Array, Float, Int, jaxtyped

from .aliases import ScalarNumeric


class CrystalGeometry(eqx.Module):
    """PyTree for crystal geometry from VASP POSCAR.

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

    Attributes
    ----------
    lattice : Float[Array, "3 3"]
        Real-space lattice vectors as rows (angstroms).
    reciprocal_lattice : Float[Array, "3 3"]
        Reciprocal lattice vectors as rows (1/angstroms).
    coords : Float[Array, "N 3"]
        Fractional atomic coordinates.
    symbols : tuple[str, ...]
        Element symbols for each species. **Static** metadata; changing
        them triggers retracing.
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
def make_crystal_geometry(
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

    Implementation Logic
    --------------------
    1. **Cast lattice** to ``jnp.float64`` via ``jnp.asarray``.
       Accepts both JAX arrays and nested Python lists.
    2. **Cast coords** to ``jnp.float64`` via ``jnp.asarray``.
    3. **Cast atom_counts** to ``jnp.int32`` via ``jnp.asarray``.
       Accepts both JAX integer arrays and Python ``list[int]``.
    4. **Auto-compute reciprocal lattice** by calling
       ``_compute_reciprocal_lattice(lattice_arr)``. This derives
       ``b_i = 2 pi (a_j x a_k) / V`` from the validated
       real-space lattice.
    5. **Construct** the ``CrystalGeometry`` Equinox module from all
       five fields (including the computed reciprocal lattice) and
       return it.

    Parameters
    ----------
    lattice : Union[Float[Array, "3 3"], list]
        Real-space lattice vectors as rows (angstroms).
    coords : Float[Array, "N 3"]
        Fractional atomic coordinates.
    symbols : tuple[str, ...]
        Element symbols for each species.
    atom_counts : Union[Int[Array, " S"], list[int]]
        Number of atoms per species.

    Returns
    -------
    geometry : CrystalGeometry
        Validated crystal geometry instance with the reciprocal
        lattice pre-computed.

    See Also
    --------
    CrystalGeometry : The PyTree class constructed by this factory.
    _compute_reciprocal_lattice : Cross-product formula used to
        derive the reciprocal lattice.
    """
    lattice_arr: Float[Array, "3 3"] = jnp.asarray(lattice, dtype=jnp.float64)
    coords_arr: Float[Array, "N 3"] = jnp.asarray(coords, dtype=jnp.float64)
    counts_arr: Int[Array, " S"] = jnp.asarray(atom_counts, dtype=jnp.int32)
    reciprocal: Float[Array, "3 3"] = _compute_reciprocal_lattice(lattice_arr)

    def validate_and_create() -> CrystalGeometry:
        def check_lattice_finite() -> Float[Array, " "]:
            return lax.cond(
                jnp.all(jnp.isfinite(lattice_arr)),
                lambda: lattice_arr.sum(),
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: lattice_arr.sum(),
                        lambda: lattice_arr.sum(),
                    )
                ),
            )

        def check_lattice_nondegenerate() -> Float[Array, " "]:
            return lax.cond(
                jnp.abs(jnp.linalg.det(lattice_arr)) > 1e-10,
                lambda: lattice_arr.sum(),
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: lattice_arr.sum(),
                        lambda: lattice_arr.sum(),
                    )
                ),
            )

        def check_coords_finite() -> Float[Array, " "]:
            return lax.cond(
                jnp.all(jnp.isfinite(coords_arr)),
                lambda: coords_arr.sum(),
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: coords_arr.sum(),
                        lambda: coords_arr.sum(),
                    )
                ),
            )

        check_lattice_finite()
        check_lattice_nondegenerate()
        check_coords_finite()
        return CrystalGeometry(
            lattice=lattice_arr,
            reciprocal_lattice=reciprocal,
            coords=coords_arr,
            symbols=symbols,
            atom_counts=counts_arr,
        )

    geometry: CrystalGeometry = validate_and_create()
    return geometry


__all__: list[str] = [
    "CrystalGeometry",
    "make_crystal_geometry",
]
