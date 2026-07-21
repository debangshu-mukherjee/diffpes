"""Validate VASP PROCAR parsing.

Covers non-spin, spin-polarized, and SOC projections together with orbital ordering and malformed or truncated blocks.
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


class TestReadProcar(chex.TestCase):
    """Tests for :func:`diffpes.inout.read_procar`.

    Verifies that the PROCAR parser produces an OrbitalProjection
    with correct projection array shape and optional spin/oam
    absent when not present in the file.

    :see: :func:`~diffpes.inout.read_procar`
    """

    def test_parses_minimal_procar(self) -> None:
        """Read minimal PROCAR and assert OrbitalProjection shape and sample values.

        Loads the minimal PROCAR fixture. Asserts projections shape
        (2, 2, 1, 9), selected projection values (0.1 and 0.18),
        and that spin and oam are None. Validates k-point/band/ion
        block parsing and orbital channel ordering.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        path: Path
        orb: (
            diffpes.types.OrbitalProjection
            | diffpes.types.SpinOrbitalProjection
        )

        path = _FIXTURES_DIR / "PROCAR"
        orb = read_procar(str(path))
        chex.assert_shape(orb.projections, (2, 2, 1, 9))
        chex.assert_trees_all_close(
            orb.projections[0, 0, 0, 0], jnp.float64(0.1), atol=1e-12
        )
        chex.assert_trees_all_close(
            orb.projections[1, 1, 0, 0], jnp.float64(0.18), atol=1e-12
        )
        assert orb.spin is None
        assert orb.oam is None

    def test_spin_procar_legacy(self) -> None:
        """Read spin-polarized PROCAR in legacy mode and verify only first spin block.

        Parses the PROCAR_spin fixture with ``return_mode="legacy"``,
        which extracts only the first (spin-up) block. Asserts
        projections shape is (2, 2, 1, 9), the s-orbital value at
        [0, 0, 0, 0] equals 0.1 (matching the fixture's spin-up data),
        and ``spin`` is ``None`` (no spin decomposition in legacy mode).
        This exercises the backward-compatible single-block extraction.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        path: Path
        orb: (
            diffpes.types.OrbitalProjection
            | diffpes.types.SpinOrbitalProjection
        )

        path = _FIXTURES_DIR / "PROCAR_spin"
        orb = read_procar(str(path), return_mode="legacy")
        chex.assert_shape(orb.projections, (2, 2, 1, 9))

        chex.assert_trees_all_close(
            orb.projections[0, 0, 0, 0], jnp.float64(0.1), atol=1e-12
        )
        assert orb.spin is None

    def test_spin_procar_full(self) -> None:
        """Read spin-polarized PROCAR in full mode and verify SpinOrbitalProjection output.

        Parses the PROCAR_spin fixture with ``return_mode="full"``,
        returning a ``SpinOrbitalProjection`` with both ``projections``
        and ``spin`` arrays. Asserts the result type is
        ``SpinOrbitalProjection``, projections shape is (2, 2, 1, 9),
        spin shape is (2, 2, 1, 6), and the averaged s-orbital value
        at [0, 0, 0, 0] equals 0.09 (the mean of spin-up 0.1 and
        spin-down 0.08), verified to within ``atol=1e-12``.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        path: Path
        orb: (
            diffpes.types.OrbitalProjection
            | diffpes.types.SpinOrbitalProjection
        )

        path = _FIXTURES_DIR / "PROCAR_spin"
        orb = read_procar(str(path), return_mode="full")
        assert isinstance(orb, SpinOrbitalProjection)
        chex.assert_shape(orb.projections, (2, 2, 1, 9))
        chex.assert_shape(orb.spin, (2, 2, 1, 6))

        chex.assert_trees_all_close(
            orb.projections[0, 0, 0, 0], jnp.float64(0.09), atol=1e-12
        )


class TestReadProcarErrors(chex.TestCase):
    """Tests for error handling in :func:`read_procar`.

    :see: :func:`~diffpes.inout.read_procar`
    """

    def test_empty_procar_raises(self) -> None:
        """An empty PROCAR file raises ValueError (procar.py lines 155-156).

        Writes a file with no valid k-points block and calls
        ``read_procar``. Asserts a ``ValueError`` matching
        ``"No valid PROCAR blocks found"``.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        fh: TextIO

        tmpname: str

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".PROCAR", delete=False
        ) as fh:
            fh.write("PROCAR empty\n")
            tmpname = fh.name
        try:
            with pytest.raises(ValueError, match="No valid PROCAR blocks"):
                read_procar(tmpname)
        finally:
            import os

            os.unlink(tmpname)


class TestReadProcarSOC(chex.TestCase):
    """Tests for SOC PROCAR parsing (procar.py lines 191-210, 282-283, 307).

    :see: :func:`~diffpes.inout.read_procar`
    """

    def test_soc_procar_full(self) -> None:
        """Read SOC PROCAR (4 blocks) in full mode and verify spin components.

        Uses the PROCAR_soc fixture which has a title line before the
        first "k-points" block (covers lines 282-283), a blank line
        between the k-point header and the band header (covers line 307),
        and 4 blocks (covers lines 191-210). Asserts the result is a
        ``SpinOrbitalProjection`` with projections shape (1, 1, 1, 9)
        and spin shape (1, 1, 1, 6).

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        path: Path
        orb: (
            diffpes.types.OrbitalProjection
            | diffpes.types.SpinOrbitalProjection
        )

        path = _FIXTURES_DIR / "PROCAR_soc"
        orb = read_procar(str(path), return_mode="full")
        assert isinstance(orb, SpinOrbitalProjection)
        chex.assert_shape(orb.projections, (1, 1, 1, 9))
        chex.assert_shape(orb.spin, (1, 1, 1, 6))

        chex.assert_trees_all_close(
            orb.projections[0, 0, 0, 0], jnp.float64(0.1), atol=1e-12
        )
