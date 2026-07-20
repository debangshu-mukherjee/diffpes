"""Volumetric data structures for VASP CHGCAR files.

Extended Summary
----------------
Defines PyTree types for storing real-space volumetric grid data
parsed from VASP CHGCAR files. Two variants are provided:

- :class:`VolumetricData` -- for non-SOC calculations (ISPIN=1 or
  ISPIN=2), with an optional scalar magnetization density.
- :class:`SOCVolumetricData` -- for spin-orbit coupling calculations
  where VASP writes four grid blocks (total charge, mx, my, mz),
  storing both the scalar mz and the full 3-component magnetization
  vector.

Routine Listings
----------------
:class:`SOCVolumetricData`
    PyTree for volumetric data from SOC CHGCAR files.
:class:`VolumetricData`
    PyTree for volumetric grid data from CHGCAR.
:func:`make_soc_volumetric_data`
    Create a validated ``SOCVolumetricData`` instance.
:func:`make_volumetric_data`
    Create a validated ``VolumetricData`` instance.

Notes
-----
All real-space grid data is stored in units consistent with VASP
output: charge density in electrons per unit cell volume (not per
Angstrom^3). The ``grid_shape`` and ``symbols`` fields are stored
as auxiliary data because JAX cannot trace Python tuples of ints
or strings.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional
from jax import lax
from jaxtyping import Array, Float, Int, jaxtyped


class VolumetricData(eqx.Module):
    """PyTree for volumetric grid data from CHGCAR.

    Extended Summary
    ----------------
    Stores the charge density on a 3-D real-space grid together with
    the crystal lattice needed to interpret the grid coordinates.
    An optional magnetization density grid is included when the
    CHGCAR comes from a spin-polarized (ISPIN=2) calculation, where
    the magnetization is the difference between spin-up and spin-down
    charge densities.

    This class is an immutable :class:`equinox.Module` PyTree. Numeric
    fields (``lattice``,
    ``coords``, ``charge``, ``magnetization``, ``atom_counts``) are
    stored as children visible to JAX tracing, while ``grid_shape``
    (a Python tuple of ints) and ``symbols`` (a Python tuple of
    strings) are stored as auxiliary data because JAX cannot trace
    these types.

    Attributes
    ----------
    lattice : Float[Array, "3 3"]
        Real-space lattice vectors as rows, in Angstroms. Defines
        the unit cell geometry for interpreting grid coordinates.
        JAX-traced (differentiable).
    coords : Float[Array, "N 3"]
        Fractional atomic coordinates for all N atoms in the cell.
        JAX-traced (differentiable).
    charge : Float[Array, "Nx Ny Nz"]
        Charge density on the 3-D real-space grid, in units of
        electrons per unit cell volume (VASP convention). Nx, Ny, Nz
        are the grid dimensions along the three lattice directions.
        JAX-traced (differentiable).
    magnetization : Optional[Float[Array, "Nx Ny Nz"]]
        Scalar magnetization density (spin-up minus spin-down), or
        ``None`` for non-spin-polarized calculations (ISPIN=1).
        Same units and grid as ``charge``. JAX-traced when present.
    atom_counts : Int[Array, " S"]
        Number of atoms per species, with S = number of species.
        JAX-traced (differentiable, int32).
    grid_shape : tuple[int, int, int]
        Grid dimensions ``(Nx, Ny, Nz)``. Stored as auxiliary data
        (static) because these integers determine array shapes and
        JAX requires them at compile time.
    symbols : tuple[str, ...]
        Element symbols for each species (e.g. ``("Bi", "Se")``).
        Stored as auxiliary data (static).

    Notes
    -----
    Equinox derives the PyTree structure from the annotated fields.
    The ``grid_shape`` and ``symbols`` fields are Python tuples and
    therefore must be stored as PyTree auxiliary data. JAX treats
    auxiliary data as compile-time constants: changing them triggers
    recompilation of any ``jit``-compiled function.

    See Also
    --------
    SOCVolumetricData : Variant for spin-orbit coupling with vector
        magnetization.
    make_volumetric_data : Factory function with validation and
        float64 casting.
    """

    lattice: Float[Array, "3 3"]
    coords: Float[Array, "N 3"]
    charge: Float[Array, "Nx Ny Nz"]
    magnetization: Optional[Float[Array, "Nx Ny Nz"]]
    atom_counts: Int[Array, " S"]
    grid_shape: tuple[int, int, int] = eqx.field(static=True)
    symbols: tuple[str, ...] = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def make_volumetric_data(
    lattice: Float[Array, "3 3"],
    coords: Float[Array, "N 3"],
    charge: Float[Array, "Nx Ny Nz"],
    magnetization: Optional[Float[Array, "Nx Ny Nz"]] = None,
    grid_shape: tuple[int, int, int] = (1, 1, 1),
    symbols: tuple[str, ...] = (),
    atom_counts: Optional[Int[Array, " S"]] = None,
) -> VolumetricData:
    """Create a validated ``VolumetricData`` instance.

    Extended Summary
    ----------------
    Factory function that validates and normalises CHGCAR volumetric
    data before constructing a ``VolumetricData`` PyTree. All numeric
    arrays are cast to ``float64`` (or ``int32`` for atom counts).
    The optional ``magnetization`` field is cast only when present,
    preserving ``None`` for non-spin-polarized calculations. When
    ``atom_counts`` is ``None``, an empty int32 array is created as
    a placeholder.

    The function is decorated with ``@jaxtyped(typechecker=beartype)``
    so that shape and dtype constraints are checked at call time.

    Implementation Logic
    --------------------
    1. **Cast lattice** to ``jnp.float64`` via ``jnp.asarray``.
    2. **Cast coords** to ``jnp.float64`` via ``jnp.asarray``.
    3. **Cast charge** to ``jnp.float64`` via ``jnp.asarray``.
    4. **Cast magnetization** to ``jnp.float64`` when not ``None``;
       otherwise leave as ``None``.
    5. **Default atom_counts**: if ``None``, create an empty int32
       array ``jnp.zeros(0, dtype=jnp.int32)``; otherwise cast to
       ``jnp.int32``.
    6. **Pass through** ``grid_shape`` and ``symbols`` unchanged --
       these become auxiliary data in the PyTree.
    7. **Construct** the ``VolumetricData`` Equinox module and return it.

    Parameters
    ----------
    lattice : Float[Array, "3 3"]
        Real-space lattice vectors as rows, in Angstroms.
    coords : Float[Array, "N 3"]
        Fractional atomic coordinates.
    charge : Float[Array, "Nx Ny Nz"]
        Charge density on 3-D grid (electrons per unit cell volume).
    magnetization : Optional[Float[Array, "Nx Ny Nz"]], optional
        Magnetization density (spin-up minus spin-down).
        Default is ``None`` (non-spin-polarized).
    grid_shape : tuple[int, int, int], optional
        Grid dimensions ``(Nx, Ny, Nz)``. Default is ``(1, 1, 1)``.
    symbols : tuple[str, ...], optional
        Element symbols per species. Default is empty tuple.
    atom_counts : Optional[Int[Array, " S"]], optional
        Number of atoms per species. Default is ``None`` (replaced
        by an empty int32 array).

    Returns
    -------
    vol : VolumetricData
        Validated volumetric data with ``float64``/``int32`` arrays.

    See Also
    --------
    VolumetricData : The PyTree class constructed by this factory.
    make_soc_volumetric_data : Factory for the SOC variant with
        vector magnetization.
    """
    lattice_arr: Float[Array, "3 3"] = jnp.asarray(lattice, dtype=jnp.float64)
    coords_arr: Float[Array, "N 3"] = jnp.asarray(coords, dtype=jnp.float64)
    charge_arr: Float[Array, "Nx Ny Nz"] = jnp.asarray(
        charge, dtype=jnp.float64
    )
    mag_arr: Optional[Float[Array, "Nx Ny Nz"]] = None
    if magnetization is not None:
        mag_arr = jnp.asarray(magnetization, dtype=jnp.float64)
    if atom_counts is None:
        counts_arr: Int[Array, " S"] = jnp.zeros(0, dtype=jnp.int32)
    else:
        counts_arr = jnp.asarray(atom_counts, dtype=jnp.int32)

    def validate_and_create() -> VolumetricData:
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

        def check_charge_finite() -> Float[Array, " "]:
            return lax.cond(
                jnp.all(jnp.isfinite(charge_arr)),
                lambda: charge_arr.sum(),
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: charge_arr.sum(),
                        lambda: charge_arr.sum(),
                    )
                ),
            )

        check_lattice_finite()
        check_charge_finite()
        return VolumetricData(
            lattice=lattice_arr,
            coords=coords_arr,
            charge=charge_arr,
            magnetization=mag_arr,
            grid_shape=grid_shape,
            symbols=symbols,
            atom_counts=counts_arr,
        )

    vol: VolumetricData = validate_and_create()
    return vol


class SOCVolumetricData(eqx.Module):
    """PyTree for volumetric data from SOC CHGCAR files.

    Extended Summary
    ----------------
    Variant of :class:`VolumetricData` for spin-orbit coupling
    calculations where VASP writes 4 grid blocks in the CHGCAR:
    total charge, mx, my, mz magnetization components. The
    ``magnetization`` field holds the mz component for backward
    compatibility with ISPIN=2 consumers, while
    ``magnetization_vector`` holds the full 3-component
    magnetization vector ``(mx, my, mz)`` at each grid point.

    This class is an immutable :class:`equinox.Module` PyTree. Numeric
    fields are stored as
    children visible to JAX tracing, while ``grid_shape`` and
    ``symbols`` are stored as auxiliary data.

    Attributes
    ----------
    lattice : Float[Array, "3 3"]
        Real-space lattice vectors as rows, in Angstroms.
        JAX-traced (differentiable).
    coords : Float[Array, "N 3"]
        Fractional atomic coordinates for all N atoms.
        JAX-traced (differentiable).
    charge : Float[Array, "Nx Ny Nz"]
        Total charge density on the 3-D grid (electrons per unit
        cell volume). JAX-traced (differentiable).
    magnetization : Float[Array, "Nx Ny Nz"]
        Scalar magnetization density, specifically the mz component.
        Provided for backward compatibility with code that expects
        ISPIN=2-style scalar magnetization. JAX-traced
        (differentiable).
    magnetization_vector : Float[Array, "Nx Ny Nz 3"]
        Full vector magnetization ``(mx, my, mz)`` at each grid
        point. The last axis indexes the three Cartesian components.
        JAX-traced (differentiable).
    atom_counts : Int[Array, " S"]
        Number of atoms per species. JAX-traced (int32).
    grid_shape : tuple[int, int, int]
        Grid dimensions ``(Nx, Ny, Nz)``. Stored as auxiliary data
        (static).
    symbols : tuple[str, ...]
        Element symbols per species. Stored as auxiliary data
        (static).

    Notes
    -----
    Equinox derives the PyTree structure from the annotated fields.
    Six numeric fields are children; ``grid_shape`` and ``symbols``
    are auxiliary data. Unlike :class:`VolumetricData`, both
    ``magnetization`` and ``magnetization_vector`` are mandatory
    (non-optional) because SOC calculations always produce all four
    grid blocks.

    See Also
    --------
    VolumetricData : Non-SOC variant with optional scalar
        magnetization.
    make_soc_volumetric_data : Factory function with validation and
        float64 casting.
    """

    lattice: Float[Array, "3 3"]
    coords: Float[Array, "N 3"]
    charge: Float[Array, "Nx Ny Nz"]
    magnetization: Float[Array, "Nx Ny Nz"]
    magnetization_vector: Float[Array, "Nx Ny Nz 3"]
    atom_counts: Int[Array, " S"]
    grid_shape: tuple[int, int, int] = eqx.field(static=True)
    symbols: tuple[str, ...] = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def make_soc_volumetric_data(
    lattice: Float[Array, "3 3"],
    coords: Float[Array, "N 3"],
    charge: Float[Array, "Nx Ny Nz"],
    magnetization: Float[Array, "Nx Ny Nz"],
    magnetization_vector: Float[Array, "Nx Ny Nz 3"],
    grid_shape: tuple[int, int, int] = (1, 1, 1),
    symbols: tuple[str, ...] = (),
    atom_counts: Optional[Int[Array, " S"]] = None,
) -> SOCVolumetricData:
    """Create a validated ``SOCVolumetricData`` instance.

    Extended Summary
    ----------------
    Factory function that validates and normalises SOC CHGCAR
    volumetric data before constructing a ``SOCVolumetricData``
    PyTree. All numeric arrays are cast to ``float64`` (or ``int32``
    for atom counts). Unlike :func:`make_volumetric_data`, both
    ``magnetization`` and ``magnetization_vector`` are mandatory
    because SOC calculations always produce all four grid blocks.

    The function is decorated with ``@jaxtyped(typechecker=beartype)``
    so that shape constraints (grid dimensions must match across
    ``charge``, ``magnetization``, and ``magnetization_vector``) are
    checked at call time.

    Implementation Logic
    --------------------
    1. **Default atom_counts**: if ``None``, create an empty int32
       array ``jnp.zeros(0, dtype=jnp.int32)``; otherwise cast to
       ``jnp.int32``.
    2. **Cast all numeric fields** (``lattice``, ``coords``,
       ``charge``, ``magnetization``, ``magnetization_vector``) to
       ``jnp.float64`` via ``jnp.asarray``.
    3. **Pass through** ``grid_shape`` and ``symbols`` unchanged --
       these become auxiliary data in the PyTree.
    4. **Construct** the ``SOCVolumetricData`` Equinox module and return.

    Parameters
    ----------
    lattice : Float[Array, "3 3"]
        Real-space lattice vectors as rows, in Angstroms.
    coords : Float[Array, "N 3"]
        Fractional atomic coordinates.
    charge : Float[Array, "Nx Ny Nz"]
        Total charge density on 3-D grid (electrons per unit cell
        volume).
    magnetization : Float[Array, "Nx Ny Nz"]
        Scalar magnetization density (mz component), for backward
        compatibility with ISPIN=2 consumers.
    magnetization_vector : Float[Array, "Nx Ny Nz 3"]
        Full vector magnetization ``(mx, my, mz)`` at each grid
        point.
    grid_shape : tuple[int, int, int], optional
        Grid dimensions ``(Nx, Ny, Nz)``. Default is ``(1, 1, 1)``.
    symbols : tuple[str, ...], optional
        Element symbols per species. Default is empty tuple.
    atom_counts : Optional[Int[Array, " S"]], optional
        Number of atoms per species. Default is ``None`` (replaced
        by an empty int32 array).

    Returns
    -------
    vol : SOCVolumetricData
        Validated SOC volumetric data with ``float64``/``int32``
        arrays.

    See Also
    --------
    SOCVolumetricData : The PyTree class constructed by this factory.
    make_volumetric_data : Factory for the non-SOC variant.
    """
    if atom_counts is None:
        counts_arr: Int[Array, " S"] = jnp.zeros(0, dtype=jnp.int32)
    else:
        counts_arr = jnp.asarray(atom_counts, dtype=jnp.int32)
    soc_lattice_arr: Float[Array, "3 3"] = jnp.asarray(
        lattice, dtype=jnp.float64
    )
    soc_charge_arr: Float[Array, "Nx Ny Nz"] = jnp.asarray(
        charge, dtype=jnp.float64
    )

    def validate_and_create() -> SOCVolumetricData:
        def check_lattice_finite() -> Float[Array, " "]:
            return lax.cond(
                jnp.all(jnp.isfinite(soc_lattice_arr)),
                lambda: soc_lattice_arr.sum(),
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: soc_lattice_arr.sum(),
                        lambda: soc_lattice_arr.sum(),
                    )
                ),
            )

        def check_charge_finite() -> Float[Array, " "]:
            return lax.cond(
                jnp.all(jnp.isfinite(soc_charge_arr)),
                lambda: soc_charge_arr.sum(),
                lambda: lax.stop_gradient(
                    lax.cond(
                        False,
                        lambda: soc_charge_arr.sum(),
                        lambda: soc_charge_arr.sum(),
                    )
                ),
            )

        check_lattice_finite()
        check_charge_finite()
        return SOCVolumetricData(
            lattice=soc_lattice_arr,
            coords=jnp.asarray(coords, dtype=jnp.float64),
            charge=soc_charge_arr,
            magnetization=jnp.asarray(magnetization, dtype=jnp.float64),
            magnetization_vector=jnp.asarray(
                magnetization_vector, dtype=jnp.float64
            ),
            grid_shape=grid_shape,
            symbols=symbols,
            atom_counts=counts_arr,
        )

    vol: SOCVolumetricData = validate_and_create()
    return vol


__all__: list[str] = [
    "SOCVolumetricData",
    "VolumetricData",
    "make_soc_volumetric_data",
    "make_volumetric_data",
]
