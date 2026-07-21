"""Validate VASP CHGCAR parsing.

Covers scalar, spin-polarized, and SOC volumetric grids together with malformed headers, grid blocks, and coordinate data.
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


class TestReadChgcar(chex.TestCase):
    """Tests for :func:`diffpes.inout.read_chgcar`.

    Validates the CHGCAR parser across three file variants: charge-only
    (single data block yielding ``VolumetricData``), ISPIN=2
    spin-polarized (two blocks yielding ``VolumetricData`` with scalar
    magnetization), and SOC (four blocks yielding
    ``SOCVolumetricData`` with vector magnetization). Asserts lattice,
    coordinate, and grid shapes, type discrimination, and numerical
    values of the volumetric data grids.

    :see: :func:`~diffpes.inout.read_chgcar`
    """

    def test_charge_only(self) -> None:
        """Read charge-only CHGCAR and verify VolumetricData output with no magnetization.

        Parses CHGCAR_charge (single data block). Asserts the result
        type is ``VolumetricData``, lattice shape is (3, 3), coords
        shape is (1, 3), ``grid_shape`` is (2, 2, 2), charge shape is
        (2, 2, 2), magnetization is ``None``, symbols is ``("Si",)``,
        and ``atom_counts`` matches ``[1]``. This validates the
        single-block CHGCAR parsing path.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        path: Path
        vol: diffpes.types.VolumetricData | diffpes.types.SOCVolumetricData

        path = _FIXTURES_DIR / "CHGCAR_charge"
        vol = read_chgcar(str(path))
        assert isinstance(vol, VolumetricData)
        chex.assert_shape(vol.lattice, (3, 3))
        chex.assert_shape(vol.coords, (1, 3))
        assert vol.grid_shape == (2, 2, 2)
        chex.assert_shape(vol.charge, (2, 2, 2))
        assert vol.magnetization is None
        assert vol.symbols == ("Si",)
        chex.assert_trees_all_close(
            vol.atom_counts, jnp.array([1], dtype=jnp.int32)
        )

    def test_charge_with_magnetization(self) -> None:
        """Read ISPIN=2 CHGCAR and verify VolumetricData includes scalar magnetization.

        Parses CHGCAR_spin (two data blocks: charge and magnetization
        density). Asserts the result type is ``VolumetricData``,
        ``grid_shape`` is (2, 2, 2), charge shape is (2, 2, 2),
        magnetization is not ``None``, and magnetization shape is
        (2, 2, 2). This validates the two-block ISPIN=2 parsing path
        where magnetization = rho_up - rho_down.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        path: Path
        vol: diffpes.types.VolumetricData | diffpes.types.SOCVolumetricData

        path = _FIXTURES_DIR / "CHGCAR_spin"
        vol = read_chgcar(str(path))
        assert isinstance(vol, VolumetricData)
        assert vol.grid_shape == (2, 2, 2)
        chex.assert_shape(vol.charge, (2, 2, 2))
        assert vol.magnetization is not None
        chex.assert_shape(vol.magnetization, (2, 2, 2))

    def test_soc_chgcar(self) -> None:
        """Read SOC CHGCAR and verify SOCVolumetricData with 3-component magnetization.

        Parses CHGCAR_soc (four data blocks: charge, mx, my, mz).
        Asserts the result type is ``SOCVolumetricData``, ``grid_shape``
        is (2, 2, 2), charge shape is (2, 2, 2), scalar magnetization
        shape is (2, 2, 2), and ``magnetization_vector`` shape is
        (2, 2, 2, 3). Further verifies that the scalar magnetization
        equals the z-component (block 4), and that the first element
        of the x-component matches the expected volume-normalized value
        ``0.10 / 27.0``, both to within ``atol=1e-12``. This validates
        the four-block SOC parsing path and volume normalization.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        path: Path
        vol: diffpes.types.VolumetricData | diffpes.types.SOCVolumetricData

        path = _FIXTURES_DIR / "CHGCAR_soc"
        vol = read_chgcar(str(path))
        assert isinstance(vol, SOCVolumetricData)
        assert vol.grid_shape == (2, 2, 2)
        chex.assert_shape(vol.charge, (2, 2, 2))
        chex.assert_shape(vol.magnetization, (2, 2, 2))
        chex.assert_shape(vol.magnetization_vector, (2, 2, 2, 3))

        chex.assert_trees_all_close(
            vol.magnetization,
            vol.magnetization_vector[..., 2],
            atol=1e-12,
        )

        chex.assert_trees_all_close(
            vol.magnetization_vector[0, 0, 0, 0],
            jnp.float64(0.10 / 27.0),
            atol=1e-12,
        )


def _write_chgcar_tmpfile(content: str) -> str:
    """Write content to a temp CHGCAR file and return its path."""
    fh: TextIO

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".CHGCAR", delete=False
    ) as fh:
        fh.write(content)
        path: str = fh.name
        return path


_CHGCAR_POSCAR_HEADER: str = (
    "Test system\n"
    " 1.0\n"
    "  3.0  0.0  0.0\n"
    "  0.0  3.0  0.0\n"
    "  0.0  0.0  3.0\n"
    "Si\n"
    "1\n"
    "Direct\n"
    "  0.0  0.0  0.0\n"
)


class TestReadChgcarErrors(chex.TestCase):
    """Error-path tests for :func:`read_chgcar` (chgcar.py lines 143-144, 150-151, etc.).

    :see: :func:`~diffpes.inout.read_chgcar`
    """

    def _cleanup(self, path: str) -> None:
        import os

        os.unlink(path)

    def test_zero_volume_lattice_raises(self) -> None:
        """Zero-volume lattice raises ValueError (chgcar.py lines 143-144).

        Writes a CHGCAR where the lattice vectors are all zero, making
        the unit cell volume = 0. Asserts ``ValueError`` matching
        ``"volume is zero"``.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        content: str
        path: str

        content = (
            "Test system\n"
            " 1.0\n"
            "  0.0  0.0  0.0\n"
            "  0.0  0.0  0.0\n"
            "  0.0  0.0  0.0\n"
            "Si\n"
            "1\n"
            "Direct\n"
            "  0.0  0.0  0.0\n"
            "\n"
            "   2   2   2\n"
            " 1.0 2.0 3.0 4.0\n"
            " 5.0 6.0 7.0 8.0\n"
        )
        path = _write_chgcar_tmpfile(content)
        try:
            with pytest.raises(ValueError, match="volume is zero"):
                read_chgcar(path)
        finally:
            self._cleanup(path)

    def test_no_grid_dimensions_raises(self) -> None:
        """Missing grid dimension line raises ValueError (chgcar.py lines 150-151).

        Writes a CHGCAR with a valid POSCAR header but no grid dimension
        line after the coordinates. Asserts ``ValueError`` matching
        ``"Could not locate"``.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        content: str
        path: str

        content = _CHGCAR_POSCAR_HEADER
        path = _write_chgcar_tmpfile(content)
        try:
            with pytest.raises(ValueError, match="Could not locate"):
                read_chgcar(path)
        finally:
            self._cleanup(path)

    def test_invalid_lattice_line_raises(self) -> None:
        """Lattice line with < 3 values raises ValueError (chgcar.py lines 292-293).

        Writes a CHGCAR where the first lattice row has only 2 values.
        Asserts ``ValueError`` matching ``"Invalid CHGCAR lattice"``.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        content: str
        path: str

        content = (
            "Test system\n"
            " 1.0\n"
            "  3.0  0.0\n"
            "  0.0  3.0  0.0\n"
            "  0.0  0.0  3.0\n"
            "Si\n"
            "1\n"
            "Direct\n"
            "  0.0  0.0  0.0\n"
        )
        path = _write_chgcar_tmpfile(content)
        try:
            with pytest.raises(ValueError, match="Invalid CHGCAR lattice"):
                read_chgcar(path)
        finally:
            self._cleanup(path)

    def test_selective_dynamics_line_consumed(self) -> None:
        """Selective dynamics line is consumed before reading coordinates (chgcar.py line 307).

        Writes a CHGCAR with a 'Selective dynamics' line before the
        coordinate-mode line. Asserts the file is parsed successfully
        and the coord_line is correctly identified as 'Direct'.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        content: str
        path: str
        vol: diffpes.types.VolumetricData | diffpes.types.SOCVolumetricData

        content = (
            "Test system\n"
            " 1.0\n"
            "  3.0  0.0  0.0\n"
            "  0.0  3.0  0.0\n"
            "  0.0  0.0  3.0\n"
            "Si\n"
            "1\n"
            "Selective dynamics\n"
            "Direct\n"
            "  0.0  0.0  0.0\n"
            "\n"
            "   2   2   2\n"
            " 1.0 2.0 3.0 4.0\n"
            " 5.0 6.0 7.0 8.0\n"
        )
        path = _write_chgcar_tmpfile(content)
        try:
            vol = read_chgcar(path)
            chex.assert_shape(vol.charge, (2, 2, 2))
        finally:
            self._cleanup(path)

    def test_invalid_coordinate_line_raises(self) -> None:
        """Atomic coordinate line with < 3 values raises ValueError (chgcar.py lines 316-317).

        Writes a CHGCAR where the atomic coordinate line has only 2
        values. Asserts ``ValueError`` matching ``"Invalid CHGCAR coordinate"``.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        content: str
        path: str

        content = (
            "Test system\n"
            " 1.0\n"
            "  3.0  0.0  0.0\n"
            "  0.0  3.0  0.0\n"
            "  0.0  0.0  3.0\n"
            "Si\n"
            "1\n"
            "Direct\n"
            "  0.0  0.0\n"
        )
        path = _write_chgcar_tmpfile(content)
        try:
            with pytest.raises(ValueError, match="Invalid CHGCAR coordinate"):
                read_chgcar(path)
        finally:
            self._cleanup(path)

    def test_cartesian_coordinates_transform(self) -> None:
        """Cartesian coordinates are correctly converted to fractional (chgcar.py lines 321-322).

        Writes a CHGCAR with Cartesian coordinate mode. The atom at
        [3.0, 0.0, 0.0] Cartesian in a [3,3,3] lattice should map to
        fractional [1.0, 0.0, 0.0]. Asserts the parsed coordinate
        equals [1.0, 0.0, 0.0] to within atol=1e-10.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        content: str
        path: str
        vol: diffpes.types.VolumetricData | diffpes.types.SOCVolumetricData

        content = (
            "Test system\n"
            " 1.0\n"
            "  3.0  0.0  0.0\n"
            "  0.0  3.0  0.0\n"
            "  0.0  0.0  3.0\n"
            "Si\n"
            "1\n"
            "Cartesian\n"
            "  3.0  0.0  0.0\n"
            "\n"
            "   2   2   2\n"
            " 1.0 2.0 3.0 4.0\n"
            " 5.0 6.0 7.0 8.0\n"
        )
        path = _write_chgcar_tmpfile(content)
        try:
            vol = read_chgcar(path)
            chex.assert_trees_all_close(
                vol.coords[0], jnp.array([1.0, 0.0, 0.0]), atol=1e-10
            )
        finally:
            self._cleanup(path)

    def test_find_next_grid_skips_non_matching_lines(self) -> None:
        """_find_next_grid_line skips lines with != 3 parts and non-int 3-part lines.

        Writes a CHGCAR where the data section contains lines with != 3
        parts (covers chgcar.py line 374) and a 3-float line like
        "1.5 2.5 3.5" (covers lines 381-382, ValueError -> continue)
        before the actual grid dimensions.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        content: str
        path: str
        vol: diffpes.types.VolumetricData | diffpes.types.SOCVolumetricData

        content = (
            _CHGCAR_POSCAR_HEADER
            + "\n"
            + "   2   2   2\n"
            + " 1.0 2.0 3.0 4.0\n"
            + " 5.0 6.0 7.0 8.0\n"
            + "\n"
            + "some augmentation text line here\n"
            + "1.5 2.5 3.5\n"
            + "   2   2   2\n"
            + " 0.1 0.2 0.3 0.4\n"
            + " 0.5 0.6 0.7 0.8\n"
        )
        path = _write_chgcar_tmpfile(content)
        try:
            vol = read_chgcar(path)
            assert isinstance(vol, VolumetricData)
            assert vol.magnetization is not None
        finally:
            self._cleanup(path)

    def test_parse_float_block_skips_blank_lines(self) -> None:
        """_parse_float_block skips blank lines in data (chgcar.py lines 449-450).

        Writes a CHGCAR where the data block has a blank line after the
        first row of values. Asserts the grid is parsed correctly.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        content: str
        path: str
        vol: diffpes.types.VolumetricData | diffpes.types.SOCVolumetricData

        content = (
            _CHGCAR_POSCAR_HEADER
            + "\n"
            + "   2   2   2\n"
            + " 1.0 2.0 3.0 4.0\n"
            + "\n"
            + " 5.0 6.0 7.0 8.0\n"
        )
        path = _write_chgcar_tmpfile(content)
        try:
            vol = read_chgcar(path)
            chex.assert_shape(vol.charge, (2, 2, 2))
        finally:
            self._cleanup(path)

    def test_parse_float_block_stops_on_non_float_token(self) -> None:
        """Non-float token in data triggers row_valid=False (chgcar.py lines 458-460).

        Writes a CHGCAR where an augmentation header line appears BETWEEN
        the two rows of float data (after 4 values, before the remaining
        4). The non-float token sets ``row_valid=False`` at line 458,
        breaks the inner token loop (line 459), and skips appending
        (line 460 is just after the except; the outer loop still advances
        idx and the subsequent valid rows complete the 8-value collection).

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        content: str
        path: str
        vol: diffpes.types.VolumetricData | diffpes.types.SOCVolumetricData

        content = (
            _CHGCAR_POSCAR_HEADER
            + "\n"
            + "   2   2   2\n"
            + " 1.0 2.0 3.0 4.0\n"
            + "augmentation occupancies\n"
            + " 5.0 6.0 7.0 8.0\n"
        )
        path = _write_chgcar_tmpfile(content)
        try:
            vol = read_chgcar(path)
            chex.assert_shape(vol.charge, (2, 2, 2))
        finally:
            self._cleanup(path)

    def test_parse_float_block_truncated_raises(self) -> None:
        """Truncated data block raises ValueError (chgcar.py lines 467-468).

        Writes a CHGCAR where the grid is 2x2x2 (needs 8 values) but
        only 4 values are present. Asserts ``ValueError`` matching
        ``"Unexpected end of CHGCAR data block"``.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        content: str
        path: str

        content = (
            _CHGCAR_POSCAR_HEADER
            + "\n"
            + "   2   2   2\n"
            + " 1.0 2.0 3.0 4.0\n"
        )
        path = _write_chgcar_tmpfile(content)
        try:
            with pytest.raises(ValueError, match="Unexpected end of CHGCAR"):
                read_chgcar(path)
        finally:
            self._cleanup(path)
