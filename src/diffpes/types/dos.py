"""Density of states data structures.

Extended Summary
----------------
Defines the :class:`DensityOfStates` and :class:`FullDensityOfStates`
PyTrees for storing total and projected density of states from
VASP DOSCAR files.

Routine Listings
----------------
:class:`DensityOfStates`
    PyTree for density of states.
:class:`FullDensityOfStates`
    PyTree for complete density of states with spin and PDOS.
:func:`make_density_of_states`
    Create a validated DensityOfStates instance.
:func:`make_full_density_of_states`
    Create a validated ``FullDensityOfStates`` instance.

Notes
-----
All energy values are in electron-volts (eV).
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional
from jaxtyping import Array, Float, jaxtyped

from .aliases import ScalarNumeric


class DensityOfStates(eqx.Module):
    """PyTree for density of states.

    Stores total density of states (DOS) data parsed from VASP DOSCAR
    files. The DOS is represented as a pair of 1-D arrays sharing the
    same energy axis, together with a scalar Fermi energy reference.

    This class is registered as a JAX PyTree via
    through ``jax.jit``, ``jax.grad``, ``jax.vmap``, and other JAX
    transformations. All fields are JAX arrays (no auxiliary data),
    so every field participates in autodiff tracing.

    Attributes
    ----------
    energy : Float[Array, " E"]
        Energy axis in eV.
    total_dos : Float[Array, " E"]
        Total density of states.
    fermi_energy : Float[Array, " "]
        Fermi level energy in eV.

    Notes
    -----
    All three fields are JAX arrays and stored as children (no
    auxiliary data). This means every field is visible to JAX's
    tracing and transformation machinery.

    See Also
    --------
    make_density_of_states : Factory function with validation and
        float64 casting.
    """

    energy: Float[Array, " E"]
    total_dos: Float[Array, " E"]
    fermi_energy: Float[Array, " "]


@jaxtyped(typechecker=beartype)
def make_density_of_states(
    energy: Float[Array, " E"],
    total_dos: Float[Array, " E"],
    fermi_energy: ScalarNumeric = 0.0,
) -> DensityOfStates:
    """Create a validated DensityOfStates instance.

    Validates and normalises the inputs before constructing the
    DensityOfStates PyTree. All numeric inputs are cast to float64
    JAX arrays so that downstream JAX transformations operate at
    double precision without silent dtype promotion surprises.

    Implementation Logic
    --------------------
    1. **Cast energy** to ``jnp.float64`` via ``jnp.asarray``.
    2. **Cast total_dos** to ``jnp.float64`` via ``jnp.asarray``.
    3. **Cast fermi_energy** scalar to a 0-D ``jnp.float64`` array.
    4. **Construct** the ``DensityOfStates`` Equinox module from the
       three validated arrays and return it.

    Parameters
    ----------
    energy : Float[Array, " E"]
        Energy axis in eV.
    total_dos : Float[Array, " E"]
        Total density of states.
    fermi_energy : ScalarNumeric, optional
        Fermi level in eV. Default is 0.0.

    Returns
    -------
    dos : DensityOfStates
        Validated density of states instance.

    See Also
    --------
    DensityOfStates : The PyTree class constructed by this factory.
    """
    energy_arr: Float[Array, " E"] = jnp.asarray(energy, dtype=jnp.float64)
    dos_arr: Float[Array, " E"] = jnp.asarray(total_dos, dtype=jnp.float64)
    fermi_arr: Float[Array, " "] = jnp.asarray(fermi_energy, dtype=jnp.float64)

    def validate_and_create() -> DensityOfStates:
        nonlocal dos_arr, energy_arr, fermi_arr
        energy_arr = eqx.error_if(
            energy_arr,
            ~(jnp.all(jnp.isfinite(energy_arr))),
            "make_density_of_states: energy finite",
        )
        dos_arr = eqx.error_if(
            dos_arr,
            ~(jnp.all(jnp.isfinite(dos_arr))),
            "make_density_of_states: dos finite",
        )
        dos_arr = eqx.error_if(
            dos_arr,
            ~(jnp.all(dos_arr >= 0.0)),
            "make_density_of_states: dos non negative",
        )
        fermi_arr = eqx.error_if(
            fermi_arr,
            ~(jnp.isfinite(fermi_arr)),
            "make_density_of_states: fermi energy finite",
        )
        return DensityOfStates(
            energy=energy_arr,
            total_dos=dos_arr,
            fermi_energy=fermi_arr,
        )

    dos: DensityOfStates = validate_and_create()
    return dos


class FullDensityOfStates(eqx.Module):
    """PyTree for complete density of states with spin and PDOS.

    Extended Summary
    ----------------
    Stores the full DOS data from a VASP DOSCAR file including
    spin-resolved total DOS, integrated DOS, and per-atom
    site-projected DOS. Returned by ``read_doscar`` when
    ``return_mode="full"``. This is the comprehensive counterpart to
    the simpler :class:`DensityOfStates`, which only stores a single
    total-DOS channel.

    This class is registered as a JAX PyTree via
    arrays, some optional) are stored as children visible to JAX
    tracing, while ``natoms`` is a plain Python ``int`` stored as
    auxiliary data because it is a structural constant that JAX
    cannot trace.

    Attributes
    ----------
    energy : Float[Array, " E"]
        Energy axis in eV, shared by all DOS channels.
        JAX-traced (differentiable).
    total_dos_up : Float[Array, " E"]
        Total DOS for spin-up channel (or the only channel if
        ISPIN=1). Units are states/eV. JAX-traced (differentiable).
    total_dos_down : Optional[Float[Array, " E"]]
        Total DOS for spin-down channel, or ``None`` if the
        calculation is non-spin-polarized (ISPIN=1). Units are
        states/eV. JAX-traced when present.
    integrated_dos_up : Float[Array, " E"]
        Integrated (cumulative) DOS for spin-up channel. The value
        at the Fermi energy gives the number of electrons for this
        spin channel. JAX-traced (differentiable).
    integrated_dos_down : Optional[Float[Array, " E"]]
        Integrated DOS for spin-down channel, or ``None`` if
        ISPIN=1. JAX-traced when present.
    pdos : Optional[Float[Array, "A E C"]]
        Per-atom site-projected DOS. A is the number of atoms, E the
        number of energy points, and C the number of orbital columns
        (depends on LORBIT setting in VASP, typically 9 or 16).
        ``None`` if no PDOS blocks are present in the DOSCAR file.
        JAX-traced when present.
    fermi_energy : Float[Array, " "]
        Fermi level energy in eV. A 0-D scalar array.
        JAX-traced (differentiable).
    natoms : int
        Number of atoms in the unit cell. **Static** structural metadata;
        changing it triggers retracing.

    Notes
    -----
    The ``natoms`` field is a Python ``int`` declared with
    ``eqx.field(static=True)`` rather than as a traced leaf. Changing
    ``natoms`` triggers recompilation of any ``jit``-compiled
    function that receives this PyTree.

    Optional fields (``total_dos_down``, ``integrated_dos_down``,
    ``pdos``) may be ``None`` for non-spin-polarized calculations or
    DOSCAR files without projected DOS. JAX handles ``None`` leaves
    transparently.

    See Also
    --------
    DensityOfStates : Simplified single-channel variant.
    make_full_density_of_states : Factory function with validation
        and float64 casting.
    """

    energy: Float[Array, " E"]
    total_dos_up: Float[Array, " E"]
    total_dos_down: Optional[Float[Array, " E"]]
    integrated_dos_up: Float[Array, " E"]
    integrated_dos_down: Optional[Float[Array, " E"]]
    pdos: Optional[Float[Array, "A E C"]]
    fermi_energy: Float[Array, " "]
    natoms: int = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def make_full_density_of_states(
    energy: Float[Array, " E"],
    total_dos_up: Float[Array, " E"],
    integrated_dos_up: Float[Array, " E"],
    fermi_energy: ScalarNumeric = 0.0,
    total_dos_down: Optional[Float[Array, " E"]] = None,
    integrated_dos_down: Optional[Float[Array, " E"]] = None,
    pdos: Optional[Float[Array, "A E C"]] = None,
    natoms: int = 0,
) -> FullDensityOfStates:
    """Create a validated ``FullDensityOfStates`` instance.

    Extended Summary
    ----------------
    Factory function that validates and normalises full density of
    states data before constructing a ``FullDensityOfStates`` PyTree.
    All non-None numeric arrays are cast to ``float64`` for numerical
    stability. Optional fields (``total_dos_down``,
    ``integrated_dos_down``, ``pdos``) are cast only when present,
    preserving the ``None`` sentinel for non-spin-polarized or
    non-projected calculations.

    The function is decorated with ``@jaxtyped(typechecker=beartype)``
    so that the energy dimension *E* is checked for consistency
    across all provided arrays at call time.

    Use this factory when you need the complete DOSCAR output
    including spin-resolved channels and site-projected DOS. For the
    simpler case of a single total-DOS channel, prefer
    :func:`make_density_of_states` instead.

    Implementation Logic
    --------------------
    1. **Cast energy** to ``jnp.float64`` via ``jnp.asarray``.
    2. **Cast total_dos_up** to ``jnp.float64``.
    3. **Cast integrated_dos_up** to ``jnp.float64``.
    4. **Cast fermi_energy** scalar to a 0-D ``jnp.float64`` array.
    5. **Cast optional fields** (``total_dos_down``,
       ``integrated_dos_down``, ``pdos``) to ``jnp.float64`` when
       not ``None``; otherwise leave as ``None``.
    6. **Keep natoms** as a Python ``int`` (not cast). It is stored
       as auxiliary data in the resulting PyTree.
    7. **Construct** the ``FullDensityOfStates`` Equinox module from all
       fields and return it.

    Parameters
    ----------
    energy : Float[Array, " E"]
        Energy axis in eV.
    total_dos_up : Float[Array, " E"]
        Spin-up total DOS (states/eV).
    integrated_dos_up : Float[Array, " E"]
        Spin-up integrated (cumulative) DOS.
    fermi_energy : ScalarNumeric, optional
        Fermi level in eV. Default is 0.0.
    total_dos_down : Optional[Float[Array, " E"]], optional
        Spin-down total DOS. Default is None (ISPIN=1).
    integrated_dos_down : Optional[Float[Array, " E"]], optional
        Spin-down integrated DOS. Default is None.
    pdos : Optional[Float[Array, "A E C"]], optional
        Per-atom site-projected DOS with C orbital columns.
        Default is None.
    natoms : int, optional
        Number of atoms in the unit cell. Default is 0.

    Returns
    -------
    dos : FullDensityOfStates
        Validated full density of states with all non-None arrays
        in ``float64``.

    See Also
    --------
    make_density_of_states : Factory for the simplified variant.
    FullDensityOfStates : The PyTree class constructed by this
        factory.
    """
    energy_arr: Float[Array, " E"] = jnp.asarray(energy, dtype=jnp.float64)
    up_arr: Float[Array, " E"] = jnp.asarray(total_dos_up, dtype=jnp.float64)
    int_up_arr: Float[Array, " E"] = jnp.asarray(
        integrated_dos_up, dtype=jnp.float64
    )
    fermi_arr: Float[Array, " "] = jnp.asarray(fermi_energy, dtype=jnp.float64)
    down_arr: Optional[Float[Array, " E"]] = None
    if total_dos_down is not None:
        down_arr = jnp.asarray(total_dos_down, dtype=jnp.float64)
    int_down_arr: Optional[Float[Array, " E"]] = None
    if integrated_dos_down is not None:
        int_down_arr = jnp.asarray(integrated_dos_down, dtype=jnp.float64)
    pdos_arr: Optional[Float[Array, "A E C"]] = None
    if pdos is not None:
        pdos_arr = jnp.asarray(pdos, dtype=jnp.float64)

    def validate_and_create() -> FullDensityOfStates:
        nonlocal energy_arr, fermi_arr, int_up_arr, up_arr
        energy_arr = eqx.error_if(
            energy_arr,
            ~(jnp.all(jnp.isfinite(energy_arr))),
            "make_full_density_of_states: energy finite",
        )
        up_arr = eqx.error_if(
            up_arr,
            ~(jnp.all(jnp.isfinite(up_arr))),
            "make_full_density_of_states: dos up finite",
        )
        up_arr = eqx.error_if(
            up_arr,
            ~(jnp.all(up_arr >= 0.0)),
            "make_full_density_of_states: dos up non negative",
        )
        int_up_arr = eqx.error_if(
            int_up_arr,
            ~(jnp.all(jnp.isfinite(int_up_arr))),
            "make_full_density_of_states: integrated dos up finite",
        )
        fermi_arr = eqx.error_if(
            fermi_arr,
            ~(jnp.isfinite(fermi_arr)),
            "make_full_density_of_states: fermi energy finite",
        )
        return FullDensityOfStates(
            energy=energy_arr,
            total_dos_up=up_arr,
            total_dos_down=down_arr,
            integrated_dos_up=int_up_arr,
            integrated_dos_down=int_down_arr,
            pdos=pdos_arr,
            fermi_energy=fermi_arr,
            natoms=natoms,
        )

    dos: FullDensityOfStates = validate_and_create()
    return dos


__all__: list[str] = [
    "DensityOfStates",
    "FullDensityOfStates",
    "make_density_of_states",
    "make_full_density_of_states",
]
