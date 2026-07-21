"""Verify HDF5 round trips for every types-owned Equinox carrier.

Extended Summary
----------------
Exercises the introspected recursive codec over all nineteen carrier classes,
including nested modules, static tuple metadata, complex arrays, and absent
optional leaves.
"""

import tempfile
from pathlib import Path

import chex
import equinox as eqx
import jax.numpy as jnp
from beartype.typing import Any, Callable
from jaxtyping import Array

import diffpes
from diffpes.inout import load_from_h5, save_to_h5
from diffpes.types import (
    make_1d_chain_model,
    make_arpes_spectrum,
    make_band_structure,
    make_crystal_geometry,
    make_density_of_states,
    make_diagonalized_bands,
    make_full_density_of_states,
    make_kpath_info,
    make_orbital_basis,
    make_orbital_projection,
    make_polarization_config,
    make_self_energy_config,
    make_simulation_params,
    make_slater_params,
    make_soc_volumetric_data,
    make_spin_band_structure,
    make_spin_orbital_projection,
    make_volumetric_data,
    make_workflow_context,
)


def _all_carriers() -> dict[str, eqx.Module]:
    """Construct one deterministic instance of every carrier class."""
    energy: Array
    kpoints: Array
    bands: diffpes.types.BandStructure
    projections: diffpes.types.OrbitalProjection
    basis: diffpes.types.OrbitalBasis
    diagonalized: diffpes.types.DiagonalizedBands
    charge: Array

    energy = jnp.array([-1.0, 1.0], dtype=jnp.float64)
    kpoints = jnp.zeros((2, 3), dtype=jnp.float64)
    bands = make_band_structure(energy[:, None], kpoints)
    projections = make_orbital_projection(jnp.ones((2, 1, 1, 9)))
    basis = make_orbital_basis((1,), (0,), (0,), labels=("s",))
    diagonalized = make_diagonalized_bands(
        eigenvalues=energy[:, None],
        eigenvectors=jnp.ones((2, 1, 1), dtype=jnp.complex128),
        kpoints=kpoints,
    )
    charge = jnp.ones((2, 2, 2), dtype=jnp.float64)
    carriers: dict[str, eqx.Module] = {
        "arpes": make_arpes_spectrum(jnp.ones((2, 2)), energy),
        "bands": bands,
        "spin_bands": make_spin_band_structure(
            energy[:, None], energy[:, None], kpoints
        ),
        "projection": projections,
        "spin_projection": make_spin_orbital_projection(
            jnp.ones((2, 1, 1, 9)), jnp.zeros((2, 1, 1, 6))
        ),
        "dos": make_density_of_states(energy, jnp.ones(2)),
        "full_dos": make_full_density_of_states(
            energy, jnp.ones(2), jnp.arange(2.0), natoms=0
        ),
        "geometry": make_crystal_geometry(
            jnp.eye(3), jnp.zeros((1, 3)), ("X",), [1]
        ),
        "kpath": make_kpath_info(2, [0, 1], segments=1, labels=("G", "X")),
        "simulation": make_simulation_params(fidelity=16),
        "polarization": make_polarization_config(),
        "basis": basis,
        "slater": make_slater_params(jnp.ones(1), basis),
        "self_energy": make_self_energy_config(),
        "diagonalized": diagonalized,
        "tb_model": make_1d_chain_model(),
        "volumetric": make_volumetric_data(
            jnp.eye(3),
            jnp.zeros((1, 3)),
            charge,
            grid_shape=(2, 2, 2),
            symbols=("X",),
            atom_counts=jnp.ones(1, dtype=jnp.int32),
        ),
        "soc_volumetric": make_soc_volumetric_data(
            jnp.eye(3),
            jnp.zeros((1, 3)),
            charge,
            charge,
            jnp.ones((2, 2, 2, 3)),
            grid_shape=(2, 2, 2),
            symbols=("X",),
            atom_counts=jnp.ones(1, dtype=jnp.int32),
        ),
        "context": make_workflow_context(bands, projections),
    }
    return carriers


def test_all_carriers_round_trip_bitwise() -> None:
    """Round-trip every carrier with exact leaves and static metadata.

    Extended Summary
    ----------------
    Saves all nineteen deterministic carriers into one HDF5 file and reloads
    them together. Each reconstructed module must retain its exact class,
    numerical leaves, nested modules, optional ``None`` leaves, and static
    metadata.

    Notes
    -----
    Uses :func:`equinox.tree_equal` for an exact recursive comparison after a
    single multi-group save/load cycle.
    """
    temporary_directory: str
    name: str
    carrier: eqx.Module

    carriers: dict[str, eqx.Module] = _all_carriers()
    with tempfile.TemporaryDirectory() as temporary_directory:
        path: Path = Path(temporary_directory) / "all_carriers.h5"
        save_to_h5(path, **carriers)
        loaded: dict[str, eqx.Module] = load_from_h5(path)

    chex.assert_equal(set(loaded), set(carriers))
    for name, carrier in carriers.items():
        chex.assert_equal(type(loaded[name]) is type(carrier), True)
        chex.assert_equal(eqx.tree_equal(loaded[name], carrier), True)
