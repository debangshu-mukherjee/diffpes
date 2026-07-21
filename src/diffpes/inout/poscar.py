"""VASP POSCAR file parser.

Extended Summary
----------------
Reads VASP POSCAR/CONTCAR crystal structure files and returns
a :class:`~diffpes.types.CrystalGeometry` PyTree containing lattice
vectors, atomic coordinates, element symbols, and atom counts.

Routine Listings
----------------
:func:`read_poscar`
    Parse a VASP POSCAR/CONTCAR file.

Notes
-----
Handles both direct (fractional) and Cartesian coordinate formats,
optional selective dynamics, and automatic reciprocal lattice
computation.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import TextIO
from jaxtyping import Float, jaxtyped
from numpy import ndarray as NDArray  # noqa: N812

from diffpes.types import CrystalGeometry, make_crystal_geometry


@jaxtyped(typechecker=beartype)
def read_poscar(
    filename: str = "POSCAR",
) -> CrystalGeometry:
    """Parse a VASP POSCAR/CONTCAR file.

    Reads a VASP POSCAR (or CONTCAR) file that defines the crystal
    structure: lattice vectors, element symbols, atom counts, and
    atomic coordinates in either direct (fractional) or Cartesian
    form. The function returns a :class:`~diffpes.types.CrystalGeometry`
    PyTree with coordinates always stored in fractional form and the
    lattice pre-scaled by the universal scaling factor.

    The POSCAR file format used by VASP has the following structure:

    * **Line 1**: Comment / system name (discarded).
    * **Line 2**: Universal scaling factor applied to all lattice
      vectors (and to Cartesian coordinates, if applicable).
    * **Lines 3-5**: Three rows of three floats defining the lattice
      vectors ``a``, ``b``, ``c`` in Angstroms (before scaling).
    * **Line 6**: Either element symbols (VASP >= 5) or atom counts
      (VASP 4). If non-numeric, it is the species line.
    * **Line 6 or 7**: Atom counts per species (list of integers).
    * **Next line**: Optional ``"Selective dynamics"`` flag. If
      present, the following line is the coordinate-type specifier.
    * **Coordinate-type line**: ``"Direct"`` / ``"Fractional"`` for
      fractional coordinates, or ``"Cartesian"`` / ``"Cart"`` for
      Cartesian.
    * **Coordinate lines**: ``natoms`` lines, each with at least three
      floats. Additional columns (selective-dynamics T/F flags) are
      silently ignored.

    :see: :class:`~.test_poscar.TestReadPoscar`

    Implementation Logic
    --------------------
    1. **Read and scale the lattice**::

           lattice = lattice * scaling_factor

       This applies the POSCAR scale before any coordinate conversion.

    2. **Convert Cartesian coordinates when required**::

           coords = np.linalg.solve(lattice.T, coords.T).T

       Solving against the scaled lattice produces fractional coordinates.

    3. **Return validated crystal geometry**::

           return geometry

       The result includes the reciprocal lattice and atom metadata.

    Parameters
    ----------
    filename : str, optional
        Path to POSCAR file. Default is ``"POSCAR"``.

    Returns
    -------
    geometry : CrystalGeometry
        Crystal geometry with lattice, reciprocal lattice, fractional
        coordinates, element symbols, and atom counts.

    Notes
    -----
    Coordinates are always returned in **fractional** (direct) form,
    regardless of the input format. If the input is Cartesian, the
    conversion ``frac = lattice^{-T} @ cart^T`` is performed via
    ``np.linalg.solve`` for numerical stability. The optional
    selective-dynamics flags (``T T F`` appended to coordinate lines)
    are detected and skipped but not stored. VASP-4 style files that
    lack the element-symbol line will produce an empty ``symbols``
    tuple; the caller should supply symbols from an external source
    (e.g. POTCAR) in that case.
    """
    fid: TextIO
    i: int

    path: Path = Path(filename)
    with path.open("r") as fid:
        _comment: str = fid.readline().strip()
        scale: float = float(fid.readline().strip())
        lattice: Float[NDArray, "3 3"] = np.zeros((3, 3), dtype=np.float64)
        for i in range(3):
            vals: list[float] = [float(x) for x in fid.readline().split()]
            lattice[i, :] = vals
        lattice = lattice * scale
        line: str = fid.readline().strip()
        symbols: tuple[str, ...] = ()
        if not any(c.isdigit() for c in line):
            symbols = tuple(line.split())
            line = fid.readline().strip()
        atom_counts: list[int] = [int(x) for x in line.split()]
        natoms: int = sum(atom_counts)
        line = fid.readline().strip()
        selective: bool = False
        if line[0].lower() == "s":
            selective = True  # noqa: F841
            line = fid.readline().strip()
        cartesian: bool = line[0].lower() in ("c", "k")
        coords: Float[NDArray, "N 3"] = np.zeros((natoms, 3), dtype=np.float64)
        for i in range(natoms):
            vals = [float(x) for x in fid.readline().split()[:3]]
            coords[i, :] = vals
        if cartesian:
            coords = coords * scale
            coords = np.linalg.solve(lattice.T, coords.T).T
    geometry: CrystalGeometry = make_crystal_geometry(
        lattice=jnp.asarray(lattice),
        coords=jnp.asarray(coords),
        symbols=symbols,
        atom_counts=atom_counts,
    )
    return geometry


__all__: list[str] = [
    "read_poscar",
]
