"""Test the two-tier validation wall for type factories."""

import jax.numpy as jnp

from diffpes.types import (
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
    make_tb_model,
    make_volumetric_data,
    make_workflow_context,
)
from tests._assertions import assert_rejects


def test_simulation_params_reject_audit_probes():
    """Reject both invalid simulation-parameter probes from the audit."""
    assert_rejects(
        make_simulation_params, sigma=-1.0, match="sigma must be positive"
    )
    assert_rejects(
        make_simulation_params,
        energy_min=5.0,
        energy_max=-5.0,
        match="energy_min must be less than energy_max",
    )


def test_polarization_config_rejects_unknown_type():
    """Reject polarization selectors outside the static allowed set."""
    assert_rejects(
        make_polarization_config,
        polarization_type="unknown",
        match="polarization_type must be one of",
    )


def test_band_factories_reject_invalid_values():
    """Reject the NaN audit probe and invalid spin-band weights."""
    assert_rejects(
        make_band_structure,
        eigenvalues=jnp.array([[jnp.nan]]),
        kpoints=jnp.zeros((1, 3)),
        match="eigenvalues finite",
    )
    assert_rejects(
        make_spin_band_structure,
        eigenvalues_up=jnp.zeros((1, 1)),
        eigenvalues_down=jnp.zeros((1, 1)),
        kpoints=jnp.zeros((1, 3)),
        kpoint_weights=jnp.array([-1.0]),
        match="weights non negative",
    )


def test_projection_factories_reject_invalid_values():
    """Reject malformed orbital axes and negative orbital probabilities."""
    assert_rejects(
        make_orbital_projection,
        projections=jnp.zeros((1, 1, 1, 8)),
        match="projections must have 9 orbital columns",
    )
    assert_rejects(
        make_spin_orbital_projection,
        projections=-jnp.ones((1, 1, 1, 9)),
        spin=jnp.zeros((1, 1, 1, 6)),
        match="projections non negative",
    )


def test_arpes_spectrum_rejects_unsorted_energy_axis():
    """Reject non-increasing ARPES energy coordinates."""
    assert_rejects(
        make_arpes_spectrum,
        intensity=jnp.zeros((1, 2)),
        energy_axis=jnp.array([0.0, 0.0]),
        match="energy axis strictly increasing",
    )


def test_dos_factories_reject_unsorted_energy_axes():
    """Reject non-increasing energy coordinates in both DOS carriers."""
    energy = jnp.array([0.0, 0.0])
    assert_rejects(
        make_density_of_states,
        energy=energy,
        total_dos=jnp.ones(2),
        match="energy strictly increasing",
    )
    assert_rejects(
        make_full_density_of_states,
        energy=energy,
        total_dos_up=jnp.ones(2),
        integrated_dos_up=jnp.ones(2),
        match="energy strictly increasing",
    )


def test_crystal_geometry_rejects_left_handed_lattice():
    """Reject a finite but left-handed real-space lattice."""
    assert_rejects(
        make_crystal_geometry,
        lattice=jnp.diag(jnp.array([1.0, 1.0, -1.0])),
        coords=jnp.zeros((1, 3)),
        symbols=("X",),
        atom_counts=[1],
        match="lattice must be right-handed",
    )


def test_kpath_rejects_unknown_mode():
    """Reject k-path mode strings outside the supported set."""
    assert_rejects(
        make_kpath_info,
        num_kpoints=2,
        label_indices=[0],
        segments=1,
        mode="unknown",
        match="mode must be one of",
    )


def test_orbital_basis_rejects_invalid_quantum_numbers():
    """Reject principal quantum numbers below one."""
    assert_rejects(
        make_orbital_basis,
        (0,),
        (0,),
        (0,),
        match="n_values must all be at least 1",
    )


def test_slater_params_reject_nonpositive_zeta():
    """Reject nonpositive exponents and coefficient-axis mismatches."""
    basis = make_orbital_basis((1,), (0,), (0,))
    assert_rejects(
        make_slater_params,
        zeta=jnp.array([0.0]),
        orbital_basis=basis,
        match="zeta positive",
    )
    assert_rejects(
        make_slater_params,
        zeta=jnp.array([1.0]),
        orbital_basis=basis,
        coefficients=jnp.ones((2, 1)),
        match="coefficients first dimension must match",
    )


def test_self_energy_rejects_unsorted_nodes():
    """Reject non-increasing and length-mismatched tabulated nodes."""
    assert_rejects(
        make_self_energy_config,
        mode="tabulated",
        coefficients=jnp.ones(2),
        energy_nodes=jnp.array([0.0, 0.0]),
        match="energy nodes strictly increasing",
    )
    assert_rejects(
        make_self_energy_config,
        mode="tabulated",
        coefficients=jnp.ones(2),
        energy_nodes=jnp.arange(3.0),
        match="energy_nodes and coefficients must have the same length",
    )


def test_tb_model_rejects_out_of_range_orbital_index():
    """Reject connectivity referencing an absent orbital."""
    basis = make_orbital_basis((1,), (0,), (0,))
    assert_rejects(
        make_tb_model,
        hopping_params=jnp.ones(1),
        lattice_vectors=jnp.eye(3),
        hopping_indices=((0, 1, (0, 0, 0)),),
        n_orbitals=1,
        orbital_basis=basis,
        match="hopping orbital indices must be in",
    )


def test_diagonalized_bands_reject_nonfinite_eigenvectors():
    """Reject non-finite eigenvectors and incompatible K/B dimensions."""
    assert_rejects(
        make_diagonalized_bands,
        eigenvalues=jnp.zeros((1, 1)),
        eigenvectors=jnp.array([[[jnp.nan + 0.0j]]]),
        kpoints=jnp.zeros((1, 3)),
        match="eigenvectors finite",
    )
    assert_rejects(
        make_diagonalized_bands,
        eigenvalues=jnp.zeros((1, 1)),
        eigenvectors=jnp.zeros((2, 1, 1), dtype=jnp.complex128),
        kpoints=jnp.zeros((1, 3)),
        match="eigenvalues and eigenvectors must agree",
    )


def test_volumetric_factories_reject_grid_mismatches():
    """Reject scalar and SOC grids inconsistent with static metadata."""
    assert_rejects(
        make_volumetric_data,
        lattice=jnp.eye(3),
        coords=jnp.zeros((1, 3)),
        charge=jnp.zeros((2, 2, 2)),
        grid_shape=(1, 1, 1),
        match="grid_shape must match charge shape",
    )
    assert_rejects(
        make_soc_volumetric_data,
        lattice=jnp.eye(3),
        coords=jnp.zeros((1, 3)),
        charge=jnp.zeros((1, 1, 1)),
        magnetization=jnp.zeros((1, 1, 1)),
        magnetization_vector=jnp.zeros((2, 1, 1, 3)),
        grid_shape=(1, 1, 1),
        match="grid_shape must match magnetization_vector spatial shape",
    )


def test_workflow_context_rejects_band_projection_mismatch():
    """Reject workflow members with inconsistent K/B dimensions."""
    bands = make_band_structure(
        eigenvalues=jnp.zeros((2, 1)), kpoints=jnp.zeros((2, 3))
    )
    projection = make_orbital_projection(jnp.zeros((1, 1, 1, 9)))
    assert_rejects(
        make_workflow_context,
        bands=bands,
        orb_proj=projection,
        match="bands and orb_proj must agree",
    )
