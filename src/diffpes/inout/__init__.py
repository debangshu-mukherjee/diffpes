"""VASP file parsers for ARPES simulation input.

Extended Summary
----------------
Provides parsers for VASP output files (POSCAR, EIGENVAL, KPOINTS,
DOSCAR, PROCAR, CHGCAR) that return PyTree data structures suitable
for ARPES simulations, along with HDF5 persistence, parser-adjacent
workflow helpers, and plotting utilities.

The submodules are organized as follows:

- :mod:`chgcar`
    VASP CHGCAR file parser.
- :mod:`certificate`
    Portable persistence for forward-model certificates.
- :mod:`doscar`
    VASP DOSCAR file parser.
- :mod:`eigenval`
    VASP EIGENVAL file parser.
- :mod:`hdf5`
    HDF5 serializer and deserializer for diffpes PyTrees.
- :mod:`helpers`
    Parser-adjacent workflow helpers for assembling simulation-ready arrays.
- :mod:`kpoints`
    VASP KPOINTS file parser.
- :mod:`plotting`
    Plotting utilities for ARPES spectra.
- :mod:`poscar`
    VASP POSCAR file parser.
- :mod:`procar`
    VASP PROCAR file parser.

Routine Listings
----------------
:func:`aggregate_atoms`
    Sum orbital projections over a set of atoms.
:func:`attach_certificate_h5`
    Atomically attach a certificate to an HDF5 result file.
:func:`apply_kpath_ticks`
    Apply symmetry-point ticks/labels from KPathInfo to an axis.
:func:`check_consistency`
    Check dimension agreement across parsed VASP files.
:func:`list_band_scatter_presets`
    Return supported preset names for projected band scatter plots.
:func:`load_from_h5`
    Load PyTrees from an HDF5 file.
:func:`load_certificate_h5`
    Load a certificate embedded in an HDF5 result file.
:func:`load_certificate_json`
    Load a validated forward certificate from canonical JSON.
:func:`plot_arpes_spectrum`
    Plot an ARPES intensity map from an ArpesSpectrum PyTree.
:func:`plot_arpes_with_kpath`
    Plot ARPES spectrum and annotate k-axis using KPathInfo.
:func:`plot_band_scatter_preset`
    Plot projected bands as marker-size-weighted scatter points.
:func:`plot_band_scatter_with_kpath`
    Plot projected band scatter and annotate x-axis with k-path labels.
:func:`read_chgcar`
    Parse a VASP CHGCAR file.
:func:`read_doscar`
    Parse a VASP DOSCAR file.
:func:`read_eigenval`
    Parse a VASP EIGENVAL file.
:func:`read_kpoints`
    Parse a VASP KPOINTS file.
:func:`read_poscar`
    Parse a VASP POSCAR/CONTCAR file.
:func:`read_procar`
    Parse a VASP PROCAR file.
:func:`reduce_orbitals`
    Reduce 9 orbital channels to s/p/d totals.
:func:`save_to_h5`
    Save one or more named PyTrees to an HDF5 file.
:func:`save_certificate_json`
    Atomically save a forward certificate as canonical JSON.
:func:`select_atoms`
    Extract orbital projections for a subset of atoms.

Notes
-----
All parsers use standard Python I/O (not JAX) since file
parsing is inherently sequential. They convert parsed data
to JAX arrays via factory functions.
"""

from .certificate import (
    attach_certificate_h5,
    load_certificate_h5,
    load_certificate_json,
    save_certificate_json,
)
from .chgcar import read_chgcar
from .doscar import read_doscar
from .eigenval import read_eigenval
from .hdf5 import load_from_h5, save_to_h5
from .helpers import (
    aggregate_atoms,
    check_consistency,
    reduce_orbitals,
    select_atoms,
)
from .kpoints import read_kpoints
from .plotting import (
    apply_kpath_ticks,
    list_band_scatter_presets,
    plot_arpes_spectrum,
    plot_arpes_with_kpath,
    plot_band_scatter_preset,
    plot_band_scatter_with_kpath,
)
from .poscar import read_poscar
from .procar import read_procar

__all__: list[str] = [
    "aggregate_atoms",
    "apply_kpath_ticks",
    "attach_certificate_h5",
    "check_consistency",
    "list_band_scatter_presets",
    "load_certificate_h5",
    "load_certificate_json",
    "load_from_h5",
    "plot_arpes_spectrum",
    "plot_arpes_with_kpath",
    "plot_band_scatter_preset",
    "plot_band_scatter_with_kpath",
    "read_chgcar",
    "read_doscar",
    "read_eigenval",
    "read_kpoints",
    "read_poscar",
    "read_procar",
    "reduce_orbitals",
    "save_certificate_json",
    "save_to_h5",
    "select_atoms",
]
