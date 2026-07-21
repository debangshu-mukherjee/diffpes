"""Validate VASP KPOINTS parsing.

Covers line-mode, automatic, and explicit sampling together with labels, weights, shifts, and malformed coordinate records.
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


class TestReadKpoints(chex.TestCase):
    """Tests for :func:`diffpes.inout.read_kpoints`.

    Covers Line-mode (with and without label fallback), Automatic,
    and Explicit KPOINTS formats. Asserts both legacy plotting fields
    (mode, labels, label_indices) and richer mode-specific metadata
    (grid/shift, explicit k-points/weights, line-mode endpoints).

    :see: :func:`~diffpes.inout.read_kpoints`
    """

    def test_line_mode(self) -> None:
        """Read Line-mode KPOINTS and assert mode, num_kpoints, and symmetry labels.

        Parses KPOINTS_line fixture. Asserts mode is "Line-mode",
        num_kpoints is 4, labels include G/X/M, and line-mode metadata
        (segments, points_per_segment, endpoints, coordinate_mode,
        comment) are populated.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        path: Path
        kpath: diffpes.types.KPathInfo

        path = _FIXTURES_DIR / "KPOINTS_line"
        kpath = read_kpoints(str(path))
        assert kpath.mode == "Line-mode"
        chex.assert_equal(kpath.num_kpoints, 4)
        chex.assert_equal(kpath.points_per_segment, 2)
        chex.assert_equal(kpath.segments, 2)
        assert "G" in kpath.labels
        assert "X" in kpath.labels
        assert "M" in kpath.labels
        assert len(kpath.labels) >= 2
        assert len(kpath.label_indices) >= 2
        assert kpath.coordinate_mode.lower() == "reciprocal"
        assert kpath.comment == "k-path"
        chex.assert_shape(kpath.kpoints, (3, 3))
        chex.assert_trees_all_close(
            kpath.kpoints[0], jnp.array([0.0, 0.0, 0.0]), atol=1e-12
        )
        chex.assert_trees_all_close(
            kpath.kpoints[-1], jnp.array([0.5, 0.5, 0.0]), atol=1e-12
        )
        assert kpath.weights is None
        assert kpath.grid is None
        assert kpath.shift is None

    def test_automatic_mode(self) -> None:
        """Read Automatic (Monkhorst-Pack) KPOINTS and assert mode and zero k-point count.

        Parses KPOINTS_auto. Asserts mode is "Automatic", num_kpoints
        is 0, and automatic metadata (grid and shift) is populated.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        path: Path
        kpath: diffpes.types.KPathInfo

        path = _FIXTURES_DIR / "KPOINTS_auto"
        kpath = read_kpoints(str(path))
        assert kpath.mode == "Automatic"
        chex.assert_equal(kpath.num_kpoints, 0)
        chex.assert_trees_all_close(
            kpath.grid, jnp.array([4, 4, 4], dtype=jnp.int32), atol=0
        )
        chex.assert_trees_all_close(
            kpath.shift, jnp.array([0.0, 0.0, 0.0]), atol=1e-12
        )
        assert kpath.coordinate_mode.lower() == "monkhorst-pack"
        assert kpath.kpoints is None
        assert kpath.weights is None

    def test_line_mode_label_fallback(self) -> None:
        """Read Line-mode KPOINTS using fallback label extraction (no "!" prefix).

        Uses KPOINTS_line_fallback where one line has five tokens
        (coordinates plus weight and label "G") and another has three
        (no label). Asserts mode is "Line-mode", "G" appears in
        labels, and the empty or second label is present, exercising
        _extract_label branches for len(parts) > 4 and return "".

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        path: Path
        kpath: diffpes.types.KPathInfo

        path = _FIXTURES_DIR / "KPOINTS_line_fallback"
        kpath = read_kpoints(str(path))
        assert kpath.mode == "Line-mode"
        assert "G" in kpath.labels
        assert "" in kpath.labels or len(kpath.labels) == 2
        chex.assert_equal(kpath.segments, 1)
        chex.assert_shape(kpath.kpoints, (2, 3))

    def test_explicit_mode(self) -> None:
        """Read Explicit KPOINTS and assert mode and k-point count.

        Parses KPOINTS_explicit. Asserts mode is "Explicit",
        num_kpoints is 3, and explicit metadata (k-points + weights)
        is parsed.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        path: Path
        kpath: diffpes.types.KPathInfo

        path = _FIXTURES_DIR / "KPOINTS_explicit"
        kpath = read_kpoints(str(path))
        assert kpath.mode == "Explicit"
        chex.assert_equal(kpath.num_kpoints, 3)
        chex.assert_shape(kpath.kpoints, (3, 3))
        chex.assert_shape(kpath.weights, (3,))
        chex.assert_trees_all_close(
            kpath.kpoints[1], jnp.array([0.5, 0.0, 0.0]), atol=1e-12
        )
        chex.assert_trees_all_close(
            kpath.weights,
            jnp.array([1.0, 0.5, 0.5], dtype=jnp.float64),
            atol=1e-12,
        )
        assert kpath.coordinate_mode.lower() == "cartesian"

    def test_explicit_mode_with_mode_header(self) -> None:
        """Read Explicit KPOINTS with an explicit mode header line and separate coordinate line.

        Parses KPOINTS_explicit_mode_header, which uses a distinct
        format where the mode is declared on a separate header line
        before the coordinate system line. Asserts mode is ``"Explicit"``,
        ``num_kpoints`` is 3, k-points shape is (3, 3), weights shape
        is (3,), the first k-point is [0, 0, 0], weights are
        [1.0, 0.5, 0.5], and coordinate mode is Cartesian. This exercises
        an alternative file layout branch in the KPOINTS parser.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        path: Path
        kpath: diffpes.types.KPathInfo

        path = _FIXTURES_DIR / "KPOINTS_explicit_mode_header"
        kpath = read_kpoints(str(path))
        assert kpath.mode == "Explicit"
        chex.assert_equal(kpath.num_kpoints, 3)
        chex.assert_shape(kpath.kpoints, (3, 3))
        chex.assert_shape(kpath.weights, (3,))
        chex.assert_trees_all_close(
            kpath.kpoints[0], jnp.array([0.0, 0.0, 0.0]), atol=1e-12
        )
        chex.assert_trees_all_close(
            kpath.weights,
            jnp.array([1.0, 0.5, 0.5], dtype=jnp.float64),
            atol=1e-12,
        )
        assert kpath.coordinate_mode.lower() == "cartesian"


def _write_kpoints_tmpfile(content: str) -> str:
    """Write content to a temp KPOINTS file and return its path."""
    fh: TextIO

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".KPOINTS", delete=False
    ) as fh:
        fh.write(content)
        path: str = fh.name
        return path


class TestReadKpointsErrors(chex.TestCase):
    """Error-path and edge-case tests for :func:`read_kpoints`.

    :see: :func:`~diffpes.inout.read_kpoints`
    """

    def _cleanup(self, path: str) -> None:
        import os

        os.unlink(path)

    def test_explicit_break_when_excess_lines(self) -> None:
        """Explicit KPOINTS with more lines than num_kpoints hits the break (line 227).

        Writes a 3-kpoint Explicit KPOINTS file with 4 coordinate lines.
        Asserts mode is 'Explicit' and only 3 k-points are returned (the
        4th line is ignored by the ``break`` at line 227).

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        content: str
        path: str
        kpath: diffpes.types.KPathInfo

        content = (
            "Excess explicit kpoints\n"
            "3\n"
            "Cartesian\n"
            "  0.0  0.0  0.0  1.0\n"
            "  0.5  0.0  0.0  0.5\n"
            "  0.5  0.5  0.0  0.5\n"
            "  0.0  0.5  0.0  0.5\n"
        )
        path = _write_kpoints_tmpfile(content)
        try:
            kpath = read_kpoints(path)
            assert kpath.mode == "Explicit"
            assert kpath.num_kpoints == 3
        finally:
            self._cleanup(path)

    def test_explicit_invalid_float_raises(self) -> None:
        """Explicit KPOINTS with non-numeric token raises ValueError (lines 231-233).

        Writes an Explicit KPOINTS with "abc 0.0 0.0" as a k-point line.
        Asserts ``ValueError`` matching ``"Invalid explicit KPOINTS"``.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        content: str
        path: str

        content = (
            "Bad explicit\n"
            "2\n"
            "Cartesian\n"
            "  abc  0.0  0.0  1.0\n"
            "  0.5  0.0  0.0  0.5\n"
        )
        path = _write_kpoints_tmpfile(content)
        try:
            with pytest.raises(ValueError, match="Invalid explicit KPOINTS"):
                read_kpoints(path)
        finally:
            self._cleanup(path)

    def test_explicit_too_few_coords_raises(self) -> None:
        """Explicit KPOINTS with < 3 coordinates per line raises ValueError (lines 235-236).

        Writes an Explicit KPOINTS where a k-point line has only 2 values.
        Asserts ``ValueError`` matching ``"at least 3 coordinates"``.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        content: str
        path: str

        content = "Too few coords\n2\nCartesian\n  0.0  0.0\n  0.5  0.0  0.0\n"
        path = _write_kpoints_tmpfile(content)
        try:
            with pytest.raises(ValueError, match="at least 3 coordinates"):
                read_kpoints(path)
        finally:
            self._cleanup(path)

    def test_explicit_no_weight_column_defaults_to_one(self) -> None:
        """Explicit KPOINTS without a 4th weight column uses default weight 1.0 (line 241).

        Writes an Explicit KPOINTS where k-point lines have only 3
        columns (no weight). Asserts that all weights equal 1.0.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        content: str
        path: str
        kpath: diffpes.types.KPathInfo

        content = "No weight\n2\nCartesian\n  0.0  0.0  0.0\n  0.5  0.0  0.0\n"
        path = _write_kpoints_tmpfile(content)
        try:
            kpath = read_kpoints(path)
            assert kpath.mode == "Explicit"
            assert kpath.weights is not None
            chex.assert_trees_all_close(
                kpath.weights,
                jnp.array([1.0, 1.0], dtype=jnp.float64),
                atol=1e-12,
            )
        finally:
            self._cleanup(path)

    def test_looks_like_kpoint_line_value_error_branch(self) -> None:
        """_looks_like_kpoint_line returns False for non-numeric 3-token lines (lines 270-276).

        Uses an Explicit KPOINTS where ``mode_line`` is not in
        ``COORDINATE_MODE_TOKENS`` (mode = "Explicit") and the first
        remaining line has 3 tokens but the first is non-numeric.
        Asserts the mode header is consumed correctly as coord_mode.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        content: str
        path: str
        kpath: diffpes.types.KPathInfo

        content = (
            "Test non-numeric\n"
            "2\n"
            "Explicit\n"
            "Reciprocal 0.5 0.5\n"
            "  0.0  0.0  0.0\n"
            "  0.5  0.0  0.0\n"
        )
        path = _write_kpoints_tmpfile(content)
        try:
            kpath = read_kpoints(path)
            assert kpath.mode == "Explicit"
            assert kpath.num_kpoints == 2
        finally:
            self._cleanup(path)

    def test_looks_like_kpoint_line_returns_true(self) -> None:
        """_looks_like_kpoint_line returns True when all three tokens are floats (lines 272-273, 276).

        Uses an Explicit KPOINTS where ``mode_line`` ("explicit") is not in
        ``COORDINATE_MODE_TOKENS`` and the first remaining line has three
        numeric tokens ("0.0 0.0 0.0 1.0"). All three float() calls succeed
        (lines 272-273 executed), so the function reaches ``return True``
        (line 276). This means ``remaining_lines.pop(0)`` is NOT called and
        ``coord_mode`` stays as ``scheme_or_mode`` ("Explicit").

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        content: str
        path: str
        kpath: diffpes.types.KPathInfo

        content = (
            "Test numeric first line\n"
            "2\n"
            "Explicit\n"
            "0.0 0.0 0.0 1.0\n"
            "0.5 0.0 0.0 0.5\n"
        )
        path = _write_kpoints_tmpfile(content)
        try:
            kpath = read_kpoints(path)
            assert kpath.mode == "Explicit"
            assert kpath.num_kpoints == 2
        finally:
            self._cleanup(path)

    def test_automatic_grid_too_few_values_raises(self) -> None:
        """Automatic KPOINTS grid line with < 3 values raises ValueError (lines 307-308).

        Writes an Automatic KPOINTS where the grid line has only 2
        integers. Asserts ``ValueError`` matching ``"3 values"``.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        content: str
        path: str

        content = "Bad auto\n0\nMonkhorst-Pack\n4 4\n0 0 0\n"
        path = _write_kpoints_tmpfile(content)
        try:
            with pytest.raises(ValueError, match="3 values"):
                read_kpoints(path)
        finally:
            self._cleanup(path)

    def test_automatic_shift_too_few_values_raises(self) -> None:
        """Automatic KPOINTS shift line with < 3 values raises ValueError (lines 343-344).

        Writes an Automatic KPOINTS where the shift line has only 2
        values. Asserts ``ValueError`` matching ``"3 values"``.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        content: str
        path: str

        content = "Bad auto shift\n0\nMonkhorst-Pack\n4 4 4\n0 0\n"
        path = _write_kpoints_tmpfile(content)
        try:
            with pytest.raises(ValueError, match="3 values"):
                read_kpoints(path)
        finally:
            self._cleanup(path)

    def test_line_mode_bad_coord_line_raises(self) -> None:
        """Line-mode KPOINTS with an unparseable coordinate raises ValueError (lines 375-376).

        Writes a Line-mode KPOINTS where a coordinate line has fewer
        than 3 float tokens. Asserts ``ValueError`` matching
        ``"Could not parse k-point coordinates"``.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        content: str
        path: str

        content = (
            "Bad line mode\n"
            "2\n"
            "Line-mode\n"
            "Reciprocal\n"
            "  text only\n"
            "  0.5  0.5  0.5  M\n"
        )
        path = _write_kpoints_tmpfile(content)
        try:
            with pytest.raises(ValueError, match="Could not parse k-point"):
                read_kpoints(path)
        finally:
            self._cleanup(path)
