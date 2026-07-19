# VASP Data Ingestion

diffpes simulates ARPES from DFT output. The `diffpes.inout` package parses
the six VASP file types — POSCAR, EIGENVAL, PROCAR, KPOINTS, DOSCAR, CHGCAR
— into validated PyTrees, and `diffpes.simul` assembles them into
simulation inputs. This guide walks the full pipeline: reader signatures,
Fermi-energy handling, the orbital projection layout, one-call workflows,
and HDF5 persistence.

The readers use plain Python I/O (file parsing is inherently sequential, so
none of them is `jit`-table), but everything they *return* is a JAX PyTree
built through the `diffpes.types` factories — differentiable and
transformable from the first line of your analysis onward.

## The Readers at a Glance

| Function | Input file | Returns |
|---|---|---|
| `read_poscar(filename="POSCAR")` | POSCAR / CONTCAR | `CrystalGeometry` |
| `read_eigenval(filename="EIGENVAL", fermi_energy=0.0, return_mode="legacy")` | EIGENVAL | `BandStructure` or `SpinBandStructure` |
| `read_procar(filename="PROCAR", return_mode="legacy")` | PROCAR (`LORBIT=11/12`) | `OrbitalProjection` or `SpinOrbitalProjection` |
| `read_kpoints(filename="KPOINTS")` | KPOINTS (line / explicit / auto) | `KPathInfo` |
| `read_doscar(filename="DOSCAR", return_mode="legacy")` | DOSCAR | `DensityOfStates` or `FullDensityOfStates` |
| `read_chgcar(filename)` | CHGCAR (charge / spin / SOC) | `VolumetricData` or `SOCVolumetricData` |

The `return_mode` switch is shared by the spin-aware readers:
`"legacy"` (the default) always returns the simple, spin-agnostic type —
first spin channel only — while `"full"` returns the spin-resolved variant
when the file actually contains spin data (`ISPIN=2` or SOC) and the simple
type otherwise.

## A Worked Example on the In-Repo Fixtures

The test suite ships minimal, self-consistent VASP files under
`tests/test_diffpes/test_inout/fixtures/`. They are real parser inputs, so
they double as a runnable demonstration (run from the repository root):

```python
from diffpes.inout import (
    read_chgcar, read_doscar, read_eigenval,
    read_kpoints, read_poscar, read_procar,
)

fixtures = "tests/test_diffpes/test_inout/fixtures/"

geometry = read_poscar(fixtures + "POSCAR")
bands = read_eigenval(fixtures + "EIGENVAL", fermi_energy=0.0)
projection = read_procar(fixtures + "PROCAR")
kpath = read_kpoints(fixtures + "KPOINTS_line")
dos = read_doscar(fixtures + "DOSCAR")
volume = read_chgcar(fixtures + "CHGCAR_charge")

print(geometry.symbols, geometry.coords.shape)       # ('Si', 'O') (6, 3)
print(bands.eigenvalues.shape, bands.kpoints.shape)  # (1, 1) (1, 3)
print(projection.projections.shape)                  # (2, 2, 1, 9)
print(kpath.mode, tuple(kpath.labels))               # Line-mode ('G', 'X', 'M')
print(dos.energy.shape, float(dos.fermi_energy))     # (5,) 0.5
print(volume.charge.shape, volume.grid_shape)        # (2, 2, 2) (2, 2, 2)
```

Spin-resolved variants of each fixture exist too (`EIGENVAL_spin`,
`PROCAR_spin`, `PROCAR_soc`, `DOSCAR_spin`, `CHGCAR_spin`, `CHGCAR_soc`) —
pass `return_mode="full"` to get the `Spin*` / SOC types back. Note the
fixtures are deliberately tiny and are *not* mutually dimension-consistent
(the EIGENVAL fixture has 1 k-point, the PROCAR fixture 2), so the
cross-file workflow below is shown against a real VASP run directory.

## What Each Reader Gives You

- **`read_poscar`** returns a `CrystalGeometry` with the `lattice [3, 3]`
  pre-scaled by the universal scaling factor, the derived
  `reciprocal_lattice [3, 3]`, coordinates always converted to fractional
  form (`coords [N, 3]`), element `symbols`, and `atom_counts`. Both VASP 4
  and VASP 5+ species conventions and `Selective dynamics` blocks are
  handled.
- **`read_eigenval`** parses eigenvalues per k-point, sorting bands by
  energy within each k-point for a consistent ordering. `ISPIN=2` files
  yield a `SpinBandStructure` (both channels) under `return_mode="full"`.
- **`read_procar`** detects the block layout: 1 block (`ISPIN=1`), 2 blocks
  (`ISPIN=2`, up/down), or 4 blocks (SOC: total, Sx, Sy, Sz). In full mode
  spin data is re-encoded as 6 non-negative channels
  `[Sx+, Sx-, Sy+, Sy-, Sz+, Sz-]` on a `SpinOrbitalProjection`.
- **`read_kpoints`** handles line-mode paths (with symmetry-point labels),
  explicit lists, and automatic grids, recording which in `KPathInfo.mode`
  along with labels, per-segment counts, and grid/shift metadata.
- **`read_doscar`** returns total DOS and Fermi energy; full mode adds
  integrated DOS, spin channels, and per-atom PDOS blocks when present.
- **`read_chgcar`** parses the volumetric grid(s): charge only, charge +
  magnetization (`ISPIN=2`), or charge + 3-component magnetization vector
  (SOC, returned as `SOCVolumetricData`).

## Fermi-Energy Handling

EIGENVAL does not contain the Fermi energy; DOSCAR does. `read_eigenval`
therefore takes `fermi_energy` as an explicit argument (default `0.0`) and
stores it on the returned `BandStructure` — eigenvalues are kept as-is, and
downstream Fermi–Dirac weighting references `bands.fermi_energy`.

The workflow helper resolves this for you, in priority order: an explicit
`fermi_energy=` argument wins; otherwise the value is read from DOSCAR; if
DOSCAR is absent and no value is given, `0.0` is used (and if DOSCAR is
*required* to infer it but missing, `FileNotFoundError` is raised).

## The Orbital Projection Layout

`projections` has shape `[nkpt, nband, natom, 9]`, with the last axis in
the VASP orbital ordering:

```text
index:   0    1    2    3    4     5     6     7     8
orbital: s    py   pz   px   dxy   dyz   dz2   dxz   dx2-y2
```

The `tot` column VASP prints is not stored. Orbital families are selected
with standard zero-based, end-exclusive slices — `slice(1, 9)` for non-s,
`slice(1, 4)` for p, `slice(4, 9)` for d (see
[Expanded Wrappers and Conventions](expanded-wrappers-and-conventions.md)
for the full convention). Three helpers in `diffpes.inout` cover the common
reductions, all preserving differentiability:

```python
from diffpes.inout import aggregate_atoms, reduce_orbitals, select_atoms

surface = select_atoms(projection, [0])        # keep atom 0 (zero-based)
summed = aggregate_atoms(projection, [0])      # [K, B, 9], atom axis summed
spd = reduce_orbitals(projection.projections)  # [K, B, A, 3] s/p/d totals
```

## From Files to a Spectrum

For a real VASP run directory, `diffpes.simul` provides three tiers of
assembly. The one-call form:

```python
from diffpes.simul import run_vasp_workflow

spectrum = run_vasp_workflow(
    level="advanced",
    directory="path/to/vasp_run",   # EIGENVAL, PROCAR, DOSCAR, KPOINTS
    photon_energy=21.2,
    sigma=0.04,
    fidelity=2500,
)
```

Or split loading from simulation so the parsed context can be inspected,
subset, and reused:

```python
from diffpes.simul import load_vasp_context, prepare_projection, simulate_context

context = load_vasp_context(
    directory="path/to/vasp_run",
    procar_mode="full",       # preserve spin data when present
    check_dimensions=True,    # cross-file consistency checks
)
# context is a WorkflowContext: .bands, .orb_proj, .kpath, .dos

prepared = prepare_projection(context.orb_proj, atom_indices=[0, 1],
                              attach_oam=True)
spectrum = simulate_context(context, level="soc", photon_energy=21.2,
                            dk=0.02, normalize=True)
```

`load_vasp_context` reads EIGENVAL and PROCAR (plus DOSCAR and KPOINTS when
present), resolves the Fermi energy as described above, and — with
`check_dimensions=True` — validates that k-point and band counts agree
across files before you burn compute on an inconsistent run.
`simulate_context` then feeds `context.bands.eigenvalues` and the prepared
projections into `simulate_expanded`, optionally applying momentum
broadening (`dk`) and z-score normalization to the result.

## HDF5 Round-Trip

Parsing large PROCAR files is slow; do it once and persist the PyTrees:

```python
from diffpes.inout import load_from_h5, save_to_h5

save_to_h5(
    "vasp_run.h5",
    compression="gzip",
    bands=bands, projection=projection, geometry=geometry,
)

data = load_from_h5("vasp_run.h5")      # {"bands": ..., "projection": ..., ...}
bands_back = load_from_h5("vasp_run.h5", name="bands")
```

Each keyword becomes an HDF5 group; array fields become datasets; static
metadata (labels, modes, `grid_shape`) is stored as JSON attributes, and
the type name travels with the group so `load_from_h5` reconstructs the
exact PyTree class. Optional fields that were `None` round-trip as `None`.
Unknown group types raise on load rather than degrading silently.

## Related Reading

- [PyTree Architecture](pytree-architecture.md) — the types the readers
  return, and their validation contract.
- [Expanded Wrappers and Conventions](expanded-wrappers-and-conventions.md)
  — the `simulate_*_expanded` family that `simulate_context` dispatches to.
- [JAX Transformability and Gradients](jax-transformability-and-gradients.md)
  — what you can differentiate once the data is in PyTree form.
- API reference: {doc}`../api/inout`, {doc}`../api/simul`,
  {doc}`../api/types`.
