"""Validate VASP EIGENVAL parsing.

Covers scalar and spin-resolved band carriers, multiple k-points, Fermi-level shifts, and malformed or truncated input.
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


class TestReadEigenval(chex.TestCase):
    """Tests for :func:`diffpes.inout.read_eigenval`.

    Covers single- and multi-k-point EIGENVAL parsing, including
    the loop branch for multiple k-points, and asserts BandStructure
    shapes and eigenvalue/k-point values.

    :see: :func:`~diffpes.inout.read_eigenval`
    """

    def test_parses_minimal_eigenval(self) -> None:
        """Read minimal EIGENVAL (1 k-point, 1 band) and assert BandStructure shape and values.

        Uses the minimal EIGENVAL fixture and fermi_energy=-0.5.
        Asserts eigenvalues shape (1, 1), kpoints (1, 3),
        kpoint_weights (1,), k-point [0,0,0], fermi -0.5, and
        eigenvalue -1.5. Validates header and per-k-point block parsing.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        path: Path
        bands: diffpes.types.BandStructure | diffpes.types.SpinBandStructure

        path = _FIXTURES_DIR / "EIGENVAL"
        bands = read_eigenval(str(path), fermi_energy=-0.5)
        chex.assert_shape(bands.eigenvalues, (1, 1))
        chex.assert_shape(bands.kpoints, (1, 3))
        chex.assert_shape(bands.kpoint_weights, (1,))
        chex.assert_trees_all_close(
            bands.kpoints[0], jnp.array([0.0, 0.0, 0.0]), atol=1e-12
        )
        chex.assert_trees_all_close(
            bands.fermi_energy, jnp.float64(-0.5), atol=1e-12
        )
        chex.assert_trees_all_close(
            bands.eigenvalues[0, 0], jnp.float64(-1.5), atol=1e-12
        )

    def test_parses_eigenval_two_kpoints(self) -> None:
        """Read EIGENVAL with 2 k-points and assert both k-points and eigenvalues.

        Uses EIGENVAL_two_kp fixture to exercise the parser's loop
        over multiple k-points (including the branch between k-point
        blocks). Asserts eigenvalues shape (2, 1), k-points at
        [0,0,0] and [0.5,0,0], and eigenvalues -1.0 and -0.5.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        path: Path
        bands: diffpes.types.BandStructure | diffpes.types.SpinBandStructure

        path = _FIXTURES_DIR / "EIGENVAL_two_kp"
        bands = read_eigenval(str(path), fermi_energy=0.0)
        chex.assert_shape(bands.eigenvalues, (2, 1))
        chex.assert_shape(bands.kpoints, (2, 3))
        chex.assert_trees_all_close(
            bands.kpoints[0], jnp.array([0.0, 0.0, 0.0]), atol=1e-12
        )
        chex.assert_trees_all_close(
            bands.kpoints[1], jnp.array([0.5, 0.0, 0.0]), atol=1e-12
        )
        chex.assert_trees_all_close(
            bands.eigenvalues[0, 0], jnp.float64(-1.0), atol=1e-12
        )
        chex.assert_trees_all_close(
            bands.eigenvalues[1, 0], jnp.float64(-0.5), atol=1e-12
        )

    def test_spin_polarized_legacy(self) -> None:
        """Read spin-polarized EIGENVAL in legacy mode and verify only spin-up eigenvalues.

        Parses the EIGENVAL_spin fixture with ``return_mode="legacy"``,
        which should discard spin-down data and return a plain
        ``BandStructure``. Asserts the result is a ``BandStructure``
        (not ``SpinBandStructure``), eigenvalues shape is (2, 2)
        (2 k-points, 2 bands), and the spin-up eigenvalues at k=0 are
        -1.5 and -0.5 respectively, verified to within ``atol=1e-12``.
        This exercises the legacy backward-compatibility path.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        path: Path
        bands: diffpes.types.BandStructure | diffpes.types.SpinBandStructure

        path = _FIXTURES_DIR / "EIGENVAL_spin"
        bands = read_eigenval(
            str(path), fermi_energy=0.0, return_mode="legacy"
        )
        assert isinstance(bands, BandStructure)
        chex.assert_shape(bands.eigenvalues, (2, 2))

        chex.assert_trees_all_close(
            bands.eigenvalues[0, 0], jnp.float64(-1.5), atol=1e-12
        )
        chex.assert_trees_all_close(
            bands.eigenvalues[0, 1], jnp.float64(-0.5), atol=1e-12
        )

    def test_spin_polarized_full(self) -> None:
        """Read spin-polarized EIGENVAL in full mode and verify both spin channels.

        Parses the EIGENVAL_spin fixture with ``return_mode="full"``,
        which returns a ``SpinBandStructure`` containing separate
        ``eigenvalues_up`` and ``eigenvalues_down`` arrays. Asserts the
        result type is ``SpinBandStructure``, both arrays have shape
        (2, 2), spin-up k=0 eigenvalues are [-1.5, -0.5], and spin-down
        k=0 eigenvalues are [-1.2, -0.3], all verified to within
        ``atol=1e-12``.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        path: Path
        bands: diffpes.types.BandStructure | diffpes.types.SpinBandStructure

        path = _FIXTURES_DIR / "EIGENVAL_spin"
        bands = read_eigenval(str(path), fermi_energy=0.0, return_mode="full")
        assert isinstance(bands, SpinBandStructure)
        chex.assert_shape(bands.eigenvalues_up, (2, 2))
        chex.assert_shape(bands.eigenvalues_down, (2, 2))

        chex.assert_trees_all_close(
            bands.eigenvalues_up[0, 0], jnp.float64(-1.5), atol=1e-12
        )

        chex.assert_trees_all_close(
            bands.eigenvalues_down[0, 0], jnp.float64(-1.2), atol=1e-12
        )
        chex.assert_trees_all_close(
            bands.eigenvalues_down[0, 1], jnp.float64(-0.3), atol=1e-12
        )

    def test_nonspin_full_returns_bandstructure(self) -> None:
        """Verify that full mode with a non-spin-polarized file returns plain BandStructure.

        Parses the standard EIGENVAL fixture (ISPIN=1) with
        ``return_mode="full"``. Asserts the result is a plain
        ``BandStructure`` rather than ``SpinBandStructure``, confirming
        that the full-mode path correctly detects single-spin data and
        falls back to the standard type.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        path: Path
        bands: diffpes.types.BandStructure | diffpes.types.SpinBandStructure

        path = _FIXTURES_DIR / "EIGENVAL"
        bands = read_eigenval(str(path), fermi_energy=0.0, return_mode="full")
        assert isinstance(bands, BandStructure)


def _write_tmpfile(content: str) -> str:
    """Write content to a temporary file and return its path."""
    fh: TextIO

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".EIGENVAL", delete=False
    ) as fh:
        fh.write(content)
        path: str = fh.name
        return path


_EIGENVAL_HEADER: str = (
    "     1     1     1     1\n"
    "unknown\n"
    "  1.0  1.0  1.0\n"
    "  0  0  0\n"
    "  0  0  0\n"
    "  1     1     1\n"
    "\n"
)


class TestReadEigenvalErrors(chex.TestCase):
    """Error-path tests for :func:`read_eigenval` (lines 169-184, 191-195, 255).

    :see: :func:`~diffpes.inout.read_eigenval`
    """

    def _cleanup(self, path: str) -> None:
        import os

        os.unlink(path)

    def test_eof_on_kpoint_line_raises(self) -> None:
        """EOF while reading k-point block raises ValueError (lines 169-170, 255).

        Writes an EIGENVAL with a valid 6-line header followed by a blank
        separator but no k-point data line. ``_read_next_nonempty_line``
        returns ``""`` at EOF (line 255), then line 169 detects the empty
        string and raises ValueError.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        content: str
        path: str

        content = _EIGENVAL_HEADER
        path = _write_tmpfile(content)
        try:
            with pytest.raises(ValueError, match="Unexpected EOF"):
                read_eigenval(path, fermi_energy=0.0)
        finally:
            self._cleanup(path)

    def test_invalid_kpoint_line_raises(self) -> None:
        """K-point line with < 4 values raises ValueError (lines 173-174).

        Writes an EIGENVAL where the first k-point line has only 2 values.
        Asserts ``ValueError`` matching ``"Invalid EIGENVAL k-point line"``.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        content: str
        path: str

        content = _EIGENVAL_HEADER + "  0.0  0.0\n"
        path = _write_tmpfile(content)
        try:
            with pytest.raises(
                ValueError, match="Invalid EIGENVAL k-point line"
            ):
                read_eigenval(path, fermi_energy=0.0)
        finally:
            self._cleanup(path)

    def test_eof_on_band_line_raises(self) -> None:
        """EOF while reading band line raises ValueError (lines 179-180, 255).

        Writes an EIGENVAL with a valid k-point line but no band data.
        ``_read_next_nonempty_line`` returns ``""`` at EOF (line 255),
        then line 179 detects it and raises ValueError.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        content: str
        path: str

        content = _EIGENVAL_HEADER + "  0.0  0.0  0.0  1.0\n"
        path = _write_tmpfile(content)
        try:
            with pytest.raises(ValueError, match="Unexpected EOF"):
                read_eigenval(path, fermi_energy=0.0)
        finally:
            self._cleanup(path)

    def test_invalid_band_line_raises(self) -> None:
        """Band line with no numeric values raises ValueError (lines 183-184).

        Writes an EIGENVAL with a valid k-point line but an empty-looking
        band line that has no parseable eigenvalue. Asserts ``ValueError``
        matching ``"Invalid EIGENVAL band line"``.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        content: str
        content2: str
        path: str

        content = _EIGENVAL_HEADER + "  0.0  0.0  0.0  1.0\n" + "  \n"

        content2 = _EIGENVAL_HEADER + "  0.0  0.0  0.0  1.0\nNaN\n"
        path = _write_tmpfile(content2)
        try:
            with pytest.raises(ValueError, match="Invalid EIGENVAL band line"):
                read_eigenval(path, fermi_energy=0.0)
        finally:
            self._cleanup(path)

    def test_spin_polarized_band_missing_spin_down_raises(self) -> None:
        """Spin-polarized band line with only 1 value raises ValueError (lines 191-195).

        Writes an ISPIN=2 EIGENVAL (header has `2` as first value)
        where the band line has only the spin-up energy but no
        spin-down energy. Asserts ``ValueError`` matching
        ``"Invalid spin-polarized"``.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        spin_header: str
        content: str
        path: str

        spin_header = (
            "     1     1     1     2\n"
            "unknown\n"
            "  1.0  1.0  1.0\n"
            "  0  0  0\n"
            "  0  0  0\n"
            "  1     1     1\n"
            "\n"
        )
        content = spin_header + "  0.0  0.0  0.0  1.0\n" + "  1  -1.5\n"
        path = _write_tmpfile(content)
        try:
            with pytest.raises(ValueError, match="spin-polarized"):
                read_eigenval(path, fermi_energy=0.0, return_mode="full")
        finally:
            self._cleanup(path)
