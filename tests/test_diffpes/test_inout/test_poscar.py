"""Validate VASP POSCAR parsing.

Covers VASP 4 and VASP 5 headers, direct and Cartesian coordinates, and selective-dynamics records.
"""

import io
import tempfile
from pathlib import Path

import chex
import jax.numpy as jnp
import pytest
from beartype.typing import TextIO

import diffpes
from diffpes.inout import (
    read_chgcar,
    read_doscar,
    read_eigenval,
    read_kpoints,
    read_poscar,
    read_procar,
)
from diffpes.types import (
    BandStructure,
    FullDensityOfStates,
    SOCVolumetricData,
    SpinBandStructure,
    SpinOrbitalProjection,
    VolumetricData,
    make_orbital_projection,
    make_spin_orbital_projection,
)

_FIXTURES_DIR: Path = Path(__file__).resolve().parent / "fixtures"


class TestReadPoscar(chex.TestCase):
    """Tests for :func:`diffpes.inout.read_poscar`.

    Covers VASP5 (species + Direct), VASP4 (Cartesian), and
    selective-dynamics POSCAR formats. Asserts lattice, coords,
    symbols, and atom_counts as appropriate.

    :see: :func:`~diffpes.inout.read_poscar`
    """

    def test_parses_vasp5_direct(self) -> None:
        """Read VASP-5 POSCAR with species and Direct coordinates and assert geometry.

        Parses the default POSCAR fixture. Asserts lattice (3,3),
        coords (6,3), symbols ("Si", "O"), and atom_counts [2, 4].
        Validates species line parsing and direct-coordinate scaling.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        path: Path
        geom: diffpes.types.CrystalGeometry

        path = _FIXTURES_DIR / "POSCAR"
        geom = read_poscar(str(path))
        chex.assert_shape(geom.lattice, (3, 3))
        chex.assert_shape(geom.coords, (6, 3))
        assert geom.symbols == ("Si", "O")
        chex.assert_trees_all_close(
            geom.atom_counts, jnp.array([2, 4], dtype=jnp.int32)
        )

    def test_parses_vasp4_cartesian(self) -> None:
        """Read VASP-4 POSCAR with Cartesian coordinates and assert geometry.

        Parses POSCAR_cartesian (no species line). Asserts coords
        shape (2, 3), empty symbols, and atom_counts [2]. Validates
        Cartesian path and single-species fallback.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        path: Path
        geom: diffpes.types.CrystalGeometry

        path = _FIXTURES_DIR / "POSCAR_cartesian"
        geom = read_poscar(str(path))
        chex.assert_shape(geom.coords, (2, 3))
        assert geom.symbols == ()
        chex.assert_trees_all_close(
            geom.atom_counts, jnp.array([2], dtype=jnp.int32)
        )

    def test_parses_selective_dynamics(self) -> None:
        """Read POSCAR with Selective dynamics line and assert coordinates.

        Parses POSCAR_selective. Asserts coords shape (1, 3) and
        first coordinate [0, 0, 0]. Validates that the selective
        dynamics line is consumed and coordinates are still read
        correctly.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        path: Path
        geom: diffpes.types.CrystalGeometry

        path = _FIXTURES_DIR / "POSCAR_selective"
        geom = read_poscar(str(path))
        chex.assert_shape(geom.coords, (1, 3))
        chex.assert_trees_all_close(
            geom.coords[0], jnp.array([0.0, 0.0, 0.0]), atol=1e-12
        )
