# diffpes

[![License](https://img.shields.io/pypi/l/diffpes.svg)](https://github.com/debangshu-mukherjee/diffpes/blob/main/LICENSE)
[![PyPI Downloads](https://static.pepy.tech/badge/diffpes)](https://pepy.tech/projects/diffpes)
[![PyPI version](https://img.shields.io/pypi/v/diffpes.svg)](https://pypi.python.org/pypi/diffpes)
[![Python Versions](https://img.shields.io/pypi/pyversions/diffpes.svg)](https://pypi.python.org/pypi/diffpes)
[![Documentation Status](https://readthedocs.org/projects/diffpes/badge/?version=latest)](https://diffpes.readthedocs.io/en/latest/?badge=latest)
[![tests](https://github.com/debangshu-mukherjee/diffpes/actions/workflows/tests.yml/badge.svg)](https://github.com/debangshu-mukherjee/diffpes/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/debangshu-mukherjee/diffpes/graph/badge.svg)](https://codecov.io/gh/debangshu-mukherjee/diffpes)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19037631.svg)](https://doi.org/10.5281/zenodo.19037631)
[![Ruff](https://img.shields.io/badge/lint%20and%20format-ruff-D7FF64?logo=ruff&logoColor=1D1D1D)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
[![jax_badge](https://tinyurl.com/mucknrvu)](https://docs.jax.dev/)
[![Lines of Code](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/debangshu-mukherjee/diffpes/main/.github/badges/loc.json)](https://github.com/debangshu-mukherjee/diffpes)

JAX-based ARPES simulation toolkit with Python-native APIs and certified
forward execution. A certified run carries bounded physics claims, provenance,
domain margins, derivative evidence, local information-flow diagnostics, and a
named assurance policy in the same differentiable PyTree as its observable.
The numerical certification path compiles and batches with JAX; portable
serialization stays at the filesystem boundary.

## Expanded-input workflows

The package includes expanded-input wrappers that let you call the
simulator with plain arrays/scalars while still running JAX kernels.

### Function mapping

- `ARPES_simulation_Novice` -> `diffpes.simul.simulate_novice_expanded`
- `ARPES_simulation_Basic` -> `diffpes.simul.simulate_basic_expanded`
- `ARPES_simulation_Basicplus` -> `diffpes.simul.simulate_basicplus_expanded`
- `ARPES_simulation_Advanced` -> `diffpes.simul.simulate_advanced_expanded`
- `ARPES_simulation_Expert` -> `diffpes.simul.simulate_expert_expanded`
- `ARPES_simulation_SOC` -> `diffpes.simul.simulate_soc_expanded`
- Dynamic dispatch by level -> `diffpes.simul.simulate_expanded`
  (use `level="soc"` with `surface_spin` for SOC)

### Notes

- Default energy-axis padding behavior:
  `min(eigenbands)-1` to `max(eigenbands)+1`.
- Incident angles for expanded wrappers are interpreted in degrees.
- Wrappers return the standard `ArpesSpectrum` PyTree.

### Python indexing conventions

Use standard Python/NumPy indexing everywhere (zero-based, end-exclusive).

- Non-s orbitals: `slice(1, 9)` -> indices 1..8
- p orbitals: `slice(1, 4)` -> indices 1..3
- d orbitals: `slice(4, 9)` -> indices 4..8

Do not use MATLAB-style indexing notation in Python code.

### Example

```python
import jax.numpy as jnp

from diffpes.simul import simulate_expanded

# [nkpt, nband]
eigenbands = jnp.linspace(-2.0, 0.5, 100).reshape(20, 5)
# [nkpt, nband, natom, 9]
surface_orb = jnp.ones((20, 5, 2, 9)) * 0.1

spectrum = simulate_expanded(
    level="advanced",
    eigenbands=eigenbands,
    surface_orb=surface_orb,
    ef=0.0,
    sigma=0.04,
    fidelity=2500,
    temperature=15.0,
    photon_energy=11.0,
    polarization="unpolarized",
    incident_theta=45.0,
    incident_phi=0.0,
    polarization_angle=0.0,
)
```

## Test coverage

Test coverage measures which lines of source code are executed
during tests. Run it with:

```bash
source .venv/bin/activate
pytest tests/ --cov=src/diffpes --cov-report=term-missing
```

To get as close to 100% as possible:

1. **Simul and types** — Already well covered. Any new branch
   (e.g. new polarization or dispatch level) should have a
   corresponding test.
2. **Expanded dispatch** — Test every `simulate_expanded(level=...)`
   branch (novice, basic, basicplus, advanced, expert, soc) and
   the unknown-level `ValueError`.
3. **HDF5** — Round-trip all PyTree types; test error paths
   (unknown type on load, missing group, unsupported type on save).
4. **VASP file readers** (`read_doscar`, `read_eigenval`, `read_kpoints`,
   `read_poscar`, `read_procar`) — Add tests that call each reader
   on minimal in-repo fixture files (e.g. under `tests/fixtures/`)
   so the parsing code paths are executed.
5. **Plotting** — Exercise the public plotting API in tests (or
   accept lower coverage for GUI-oriented code).
6. **Edge branches** — Cover optional arguments (e.g.
   `make_band_structure(..., kpoint_weights=...)`) and error
   messages so one-off branches are hit.
