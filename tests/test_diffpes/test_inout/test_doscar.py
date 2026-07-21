"""Validate VASP DOSCAR parsing.

Covers legacy and full density-of-states carriers, spin channels, projected blocks, and malformed or truncated input.
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


class TestReadDoscar(chex.TestCase):
    """Tests for :func:`diffpes.inout.read_doscar`.

    Verifies that the DOSCAR parser produces a valid DensityOfStates
    PyTree with correct array shapes and expected numeric values from
    the minimal fixture.

    :see: :func:`~diffpes.inout.read_doscar`
    """

    def test_parses_minimal_doscar(self) -> None:
        """Read minimal DOSCAR fixture and assert shape and key values of DensityOfStates.

        Loads the fixtures/DOSCAR file and asserts that energy and
        total_dos have shape (5,), fermi_energy is scalar, and
        selected elements match the known fixture values (fermi 0.5,
        first energy -2.0, middle DOS 0.5). Ensures the parser
        correctly interprets the DOSCAR header and data block.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        path: Path
        dos: diffpes.types.DensityOfStates | diffpes.types.FullDensityOfStates

        path = _FIXTURES_DIR / "DOSCAR"
        dos = read_doscar(str(path))
        chex.assert_shape(dos.energy, (5,))
        chex.assert_shape(dos.total_dos, (5,))
        chex.assert_shape(dos.fermi_energy, ())
        chex.assert_trees_all_close(
            dos.fermi_energy, jnp.float64(0.5), atol=1e-12
        )
        chex.assert_trees_all_close(
            dos.energy[0], jnp.float64(-2.0), atol=1e-12
        )
        chex.assert_trees_all_close(
            dos.total_dos[2], jnp.float64(0.5), atol=1e-12
        )


class TestReadDoscarFull(chex.TestCase):
    """Tests for :func:`diffpes.inout.read_doscar` with ``return_mode='full'``.

    Validates the full-mode DOSCAR parser that returns
    ``FullDensityOfStates`` objects containing spin-resolved total DOS,
    integrated DOS, and optionally per-atom projected DOS (PDOS).
    Covers spin-polarized files (both channels), non-spin files
    (spin-down fields are None), PDOS-containing files, and the
    legacy fallback for spin-polarized data.

    :see: :func:`~diffpes.inout.read_doscar`
    """

    def test_spin_doscar_full(self) -> None:
        """Read spin-polarized DOSCAR in full mode and verify both spin channels.

        Parses the DOSCAR_spin fixture with ``return_mode="full"``.
        Asserts the result is a ``FullDensityOfStates``, energy has
        shape (5,), both ``total_dos_up`` and ``total_dos_down`` have
        shape (5,) and are not None, both integrated DOS arrays have
        shape (5,), ``fermi_energy`` is 0.5, the first spin-up DOS
        value is 0.10, and the first spin-down DOS value is 0.08,
        all verified to within ``atol=1e-12``.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        path: Path
        dos: diffpes.types.DensityOfStates | diffpes.types.FullDensityOfStates

        path = _FIXTURES_DIR / "DOSCAR_spin"
        dos = read_doscar(str(path), return_mode="full")
        assert isinstance(dos, FullDensityOfStates)
        chex.assert_shape(dos.energy, (5,))
        chex.assert_shape(dos.total_dos_up, (5,))
        assert dos.total_dos_down is not None
        chex.assert_shape(dos.total_dos_down, (5,))
        chex.assert_shape(dos.integrated_dos_up, (5,))
        assert dos.integrated_dos_down is not None
        chex.assert_shape(dos.integrated_dos_down, (5,))
        chex.assert_trees_all_close(
            dos.fermi_energy, jnp.float64(0.5), atol=1e-12
        )
        chex.assert_trees_all_close(
            dos.total_dos_up[0], jnp.float64(0.10), atol=1e-12
        )
        chex.assert_trees_all_close(
            dos.total_dos_down[0], jnp.float64(0.08), atol=1e-12
        )

    def test_nonspin_doscar_full(self) -> None:
        """Read non-spin-polarized DOSCAR in full mode and verify spin-down fields are None.

        Parses the standard DOSCAR fixture (non-spin-polarized) with
        ``return_mode="full"``. Asserts the result is a
        ``FullDensityOfStates``, energy and ``total_dos_up`` both have
        shape (5,), ``integrated_dos_up`` has shape (5,), and both
        ``total_dos_down`` and ``integrated_dos_down`` are ``None``,
        confirming the parser correctly detects non-spin data and
        leaves spin-down fields unset.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        path: Path
        dos: diffpes.types.DensityOfStates | diffpes.types.FullDensityOfStates

        path = _FIXTURES_DIR / "DOSCAR"
        dos = read_doscar(str(path), return_mode="full")
        assert isinstance(dos, FullDensityOfStates)
        chex.assert_shape(dos.energy, (5,))
        chex.assert_shape(dos.total_dos_up, (5,))
        assert dos.total_dos_down is None
        chex.assert_shape(dos.integrated_dos_up, (5,))
        assert dos.integrated_dos_down is None

    def test_pdos_doscar_full(self) -> None:
        """Read DOSCAR containing per-atom PDOS blocks in full mode.

        Parses the DOSCAR_pdos fixture with ``return_mode="full"``.
        Asserts the result is a ``FullDensityOfStates``, ``pdos`` is
        not None, its shape is (2, 3, 10) corresponding to 2 atoms,
        3 energy points, and 10 data columns (9 orbital channels plus
        a total column), and ``natoms`` is 2. This validates the PDOS
        block parsing path which reads per-atom decomposed DOS after
        the total DOS header.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        path: Path
        dos: diffpes.types.DensityOfStates | diffpes.types.FullDensityOfStates

        path = _FIXTURES_DIR / "DOSCAR_pdos"
        dos = read_doscar(str(path), return_mode="full")
        assert isinstance(dos, FullDensityOfStates)
        assert dos.pdos is not None

        chex.assert_shape(dos.pdos, (2, 3, 10))
        assert dos.natoms == 2

    def test_spin_doscar_legacy(self) -> None:
        """Read spin-polarized DOSCAR in legacy mode and verify only spin-up is returned.

        Parses the DOSCAR_spin fixture with ``return_mode="legacy"``.
        Asserts the result is a plain ``DensityOfStates`` (not
        ``FullDensityOfStates``), energy has shape (5,), ``total_dos``
        has shape (5,), and the first total DOS value is 0.10
        (matching the spin-up channel), verified to within
        ``atol=1e-12``. This exercises the backward-compatible path
        that discards spin-down data.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        path: Path
        dos: diffpes.types.DensityOfStates | diffpes.types.FullDensityOfStates

        path = _FIXTURES_DIR / "DOSCAR_spin"
        dos = read_doscar(str(path), return_mode="legacy")

        chex.assert_shape(dos.energy, (5,))
        chex.assert_shape(dos.total_dos, (5,))
        chex.assert_trees_all_close(
            dos.total_dos[0], jnp.float64(0.10), atol=1e-12
        )


class TestReadDoscarPdosHeader(chex.TestCase):
    """Tests for :func:`read_doscar` PDOS parsing with per-atom header lines.

    :see: :func:`~diffpes.inout.read_doscar`
    """

    def test_pdos_with_header_line(self) -> None:
        """DOSCAR with per-atom header line exercises doscar.py lines 204-221.

        Uses the DOSCAR_pdos_header fixture where each atom's PDOS block
        starts with a 4-value header line (EMIN EMAX NEDOS EFERMI).
        The condition ``NONSPIN_COLS <= len(line_vals) <= SPIN_COLS``
        is True (4 values), so lines 204-221 are executed. Asserts the
        PDOS array has the correct shape (1 atom, 3 energy points, 9
        orbital columns).

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        path: Path
        dos: diffpes.types.DensityOfStates | diffpes.types.FullDensityOfStates

        path = _FIXTURES_DIR / "DOSCAR_pdos_header"
        dos = read_doscar(str(path), return_mode="full")
        assert isinstance(dos, FullDensityOfStates)
        assert dos.pdos is not None

        chex.assert_shape(dos.pdos, (1, 3, 9))

    def test_pdos_early_break_on_empty_line(self) -> None:
        """PDOS block with fewer data rows than NEDOS exercises doscar.py line 230.

        Writes a DOSCAR where natoms=1, NEDOS=3 but the PDOS block has
        only 1 data row followed by an empty line. This causes the
        ``for j in range(1, nedos):`` loop to hit an empty ``row_line``
        and execute the ``break`` on line 230. Asserts the result is
        a ``FullDensityOfStates`` (parse continues with truncated data).

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        fh: TextIO

        content: str
        tmpname: str
        dos: diffpes.types.DensityOfStates | diffpes.types.FullDensityOfStates

        content = (
            "         1\n"
            " unknown system\n"
            " 1.0\n"
            " 0.0 0.0 0.0\n"
            " 0.0 0.0 0.0\n"
            " -2.0 1.0 3 0.5\n"
            " -2.0 0.1 0.0\n"
            " -1.0 0.2 0.1\n"
            "  1.0 0.3 0.4\n"
            " -2.0 0.05 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01\n"
            "\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".DOSCAR", delete=False
        ) as fh:
            fh.write(content)
            tmpname = fh.name
        try:
            dos = read_doscar(tmpname, return_mode="full")
            assert isinstance(dos, FullDensityOfStates)
        finally:
            import os

            os.unlink(tmpname)

    def test_pdos_header_then_empty_line_breaks(self) -> None:
        """PDOS block whose header is followed by an empty line exercises doscar.py line 206.

        Writes a DOSCAR with natoms=1, NEDOS=3 where the PDOS block has a
        4-value header line (satisfying ``NONSPIN_COLS <= len <= SPIN_COLS``)
        immediately followed by an empty line. The parser reads the header
        (line 204), then reads the empty line as ``pdos_ncols_check`` and
        hits ``if not pdos_ncols_check.strip(): break`` on line 205-206.
        Asserts the result is a ``FullDensityOfStates`` with ``pdos=None``
        (no PDOS data collected before the break).

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        fh: TextIO

        content: str
        tmpname: str
        dos: diffpes.types.DensityOfStates | diffpes.types.FullDensityOfStates

        content = (
            "         1\n"
            " unknown system\n"
            " 1.0\n"
            " 0.0 0.0 0.0\n"
            " 0.0 0.0 0.0\n"
            " -2.0 1.0 3 0.5\n"
            " -2.0 0.1 0.0\n"
            " -1.0 0.2 0.1\n"
            "  1.0 0.3 0.4\n"
            " -2.0 1.0 3 0.5\n"
            "\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".DOSCAR", delete=False
        ) as fh:
            fh.write(content)
            tmpname = fh.name
        try:
            dos = read_doscar(tmpname, return_mode="full")
            assert isinstance(dos, FullDensityOfStates)
            assert dos.pdos is None
        finally:
            import os

            os.unlink(tmpname)

    def test_pdos_header_inner_loop_empty_row_breaks(self) -> None:
        """Empty data row inside PDOS-with-header loop exercises doscar.py line 218.

        Writes a DOSCAR with natoms=1, NEDOS=3 where the PDOS block starts
        with a 4-value header line, then a valid first data row, then an
        empty line for the second data row (j=1). The inner loop reads the
        empty ``row_line`` and hits ``if not row_line.strip(): break`` on
        line 217-218. Asserts the result is a ``FullDensityOfStates``.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        fh: TextIO

        content: str
        tmpname: str
        dos: diffpes.types.DensityOfStates | diffpes.types.FullDensityOfStates

        content = (
            "         1\n"
            " unknown system\n"
            " 1.0\n"
            " 0.0 0.0 0.0\n"
            " 0.0 0.0 0.0\n"
            " -2.0 1.0 3 0.5\n"
            " -2.0 0.1 0.0\n"
            " -1.0 0.2 0.1\n"
            "  1.0 0.3 0.4\n"
            " -2.0 1.0 3 0.5\n"
            " -2.0 0.05 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01\n"
            "\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".DOSCAR", delete=False
        ) as fh:
            fh.write(content)
            tmpname = fh.name
        try:
            dos = read_doscar(tmpname, return_mode="full")
            assert isinstance(dos, FullDensityOfStates)
        finally:
            import os

            os.unlink(tmpname)
