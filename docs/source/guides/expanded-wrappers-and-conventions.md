# Expanded Wrappers and Conventions

The expanded-input wrappers in `diffpes.simul` let you drive the ARPES
simulator with plain arrays and scalars — no PyTree assembly required —
while still running the same JAX kernels underneath. They exist for users
migrating from script-based workflows (the historical
`ARPES_simulation_*` function family) and for quick interactive work. This
page is the reference for the wrapper family and for the argument
conventions — energy-axis padding, angle units, and array indexing — that
apply across diffpes.

## Function Mapping

Each legacy `ARPES_simulation_*` entry point corresponds to exactly one
expanded wrapper:

| Legacy name | diffpes function | Broadening | Orbital weights | Polarization |
|---|---|---|---|---|
| `ARPES_simulation_Novice` | `diffpes.simul.simulate_novice_expanded` | Voigt (`sigma` + `gamma`) | uniform | — |
| `ARPES_simulation_Basic` | `diffpes.simul.simulate_basic_expanded` | Gaussian (`sigma`) | heuristic | — |
| `ARPES_simulation_Basicplus` | `diffpes.simul.simulate_basicplus_expanded` | Gaussian (`sigma`) | Yeh–Lindau | — |
| `ARPES_simulation_Advanced` | `diffpes.simul.simulate_advanced_expanded` | Gaussian (`sigma`) | Yeh–Lindau | polarization rules |
| `ARPES_simulation_Expert` | `diffpes.simul.simulate_expert_expanded` | Voigt (`sigma` + `gamma`) | Yeh–Lindau | dipole matrix elements |
| `ARPES_simulation_SOC` | `diffpes.simul.simulate_soc_expanded` | Voigt (`sigma` + `gamma`) | Yeh–Lindau | dipole + spin-orbit (`ls_scale`) |

Every wrapper takes `eigenbands [K, B]` (eigenvalues in eV) and
`surface_orb [K, B, A, 9]` (orbital projections), plus the scalar
parameters relevant to its level, and internally builds the
`BandStructure`, `OrbitalProjection`, `SimulationParams`, and (where
applicable) `PolarizationConfig` PyTrees before calling the corresponding
`simulate_novice` … `simulate_soc` core function. The SOC wrapper
additionally requires `surface_spin [K, B, A, 6]` in the
`[Sx+, Sx-, Sy+, Sy-, Sz+, Sz-]` channel convention.

## Dynamic Dispatch: `simulate_expanded(level=...)`

A single entry point routes by level name (case-insensitive):

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
print(spectrum.intensity.shape, spectrum.energy_axis.shape)  # (20, 2500) (2500,)
```

Rules of the dispatcher:

- `level` is one of `"novice"`, `"basic"`, `"basicplus"`, `"advanced"`,
  `"expert"`, `"soc"`. Anything else raises `ValueError` listing the valid
  levels.
- Only `level`, `eigenbands`, and `surface_orb` are required; every other
  parameter has a default (`ef=0.0`, `sigma=0.04`, `gamma=0.1`,
  `fidelity=25000`, `temperature=15.0`, `photon_energy=11.0`,
  `incident_theta=45.0`).
- Parameters the selected level does not use are **silently ignored** —
  passing `gamma` to `level="basic"` or polarization settings to
  `level="novice"` is harmless.
- `level="soc"` requires `surface_spin`; omitting it raises `ValueError`.
  `ls_scale` (default `0.01`) sets the spin-orbit coupling strength.

Under `jax.jit`, mark `level`, `polarization`, and `fidelity` as static —
they are Python `str`/`int` values that select code paths and array shapes
(see [JAX Transformability and Gradients](jax-transformability-and-gradients.md)).

## The Returned `ArpesSpectrum`

All wrappers return the standard `ArpesSpectrum` PyTree from
`diffpes.types` — the same carrier the PyTree-level API and the fitting
layer use:

- `intensity` — `Float[Array, "K E"]`, the simulated photoemission
  intensity per (k-point, energy) pair.
- `energy_axis` — `Float[Array, " E"]`, the energy grid in eV, with
  `E == fidelity` points.

## Default Energy-Axis Padding

The expanded wrappers derive the energy window from the data via
`make_expanded_simulation_params`: the axis spans

```text
min(eigenbands) - 1 eV   to   max(eigenbands) + 1 eV
```

with `fidelity` evenly spaced points (`energy_padding=1.0` eV by default,
adjustable when calling `make_expanded_simulation_params` directly). For
the example above — eigenvalues spanning `[-2.0, 0.5]` eV — the axis runs
from `-3.0` to `1.5` eV. If you need a pinned window instead (for
comparing spectra across runs, say), build `SimulationParams` yourself with
`diffpes.types.make_simulation_params(energy_min=..., energy_max=..., ...)`
and use the PyTree-level `simulate_*` functions.

## Angle and Polarization Conventions

- **Incident angles are degrees at the wrapper boundary.**
  `incident_theta` (polar, from the surface normal) and `incident_phi`
  (azimuthal) are interpreted in degrees and converted to radians exactly
  once, inside the wrapper, before being stored on `PolarizationConfig`
  (whose own `theta`/`phi` fields are radians — the internal convention).
- **`polarization_angle` is radians.** It is the rotation angle for
  arbitrary linear polarization (`"LAP"`) and passes through unconverted.
- **`polarization` is a case-insensitive string**: `"LVP"` (s-pol),
  `"LHP"` (p-pol), `"RCP"`, `"LCP"`, `"LAP"`, or `"unpolarized"`.
  Unrecognized strings fall back to s-polarization in the field builder,
  so spell them carefully.
- Energies are eV, lengths Angstrom, k-vectors 1/Angstrom, temperature
  Kelvin — everywhere, no exceptions.

## Python Indexing Conventions

diffpes uses standard Python/NumPy indexing everywhere: **zero-based,
end-exclusive**. This applies to atom indices (`atom_indices=[0, 1]` means
the first two atoms), band indices, k-point indices, and orbital channels.

The 9 orbital channels of `surface_orb[..., :]` follow the VASP ordering
`[s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2]`, and the orbital families
are addressed with these slices:

| Family | Slice | Indices selected |
|---|---|---|
| non-s orbitals | `slice(1, 9)` | 1–8 (all p and d) |
| p orbitals | `slice(1, 4)` | 1, 2, 3 (py, pz, px) |
| d orbitals | `slice(4, 9)` | 4–8 (dxy, dyz, dz2, dxz, dx2-y2) |

```python
p_weight = surface_orb[..., 1:4].sum(axis=-1)   # total p character
d_weight = surface_orb[..., 4:9].sum(axis=-1)   # total d character
```

Do not use MATLAB-style one-based, end-inclusive notation anywhere — not
in code, not in comments, not in docs. If you are porting a MATLAB script
that selects "orbitals 2:9", the diffpes equivalent is `slice(1, 9)`.

## Where the Wrappers Sit in the Stack

Three tiers, from most to least manual control:

1. **PyTree level** — `simulate_novice` … `simulate_soc` and the
   first-principles `simulate_tb_radial`: you build every input PyTree
   yourself via the `diffpes.types` factories.
2. **Expanded level** (this page) — `simulate_*_expanded` /
   `simulate_expanded`: plain arrays in, PyTrees assembled for you,
   energy window auto-derived.
3. **Workflow level** — `run_vasp_workflow` and
   `load_vasp_context` + `simulate_context`: start from VASP files on
   disk; the context loader feeds `simulate_expanded` internally (see
   [VASP Data Ingestion](vasp-data-ingestion.md)).

All three return the same `ArpesSpectrum`, and all three are
differentiable in their continuous inputs — the wrappers add no
`stop_gradient`s, so `jax.grad` through `simulate_expanded` reaches
`eigenbands`, `surface_orb`, `ef`, `sigma`, and the rest exactly as it
does through the PyTree-level functions.

## Related Reading

- [PyTree Architecture](pytree-architecture.md) — the `ArpesSpectrum`,
  `SimulationParams`, and `PolarizationConfig` carriers the wrappers
  build.
- [JAX Transformability and Gradients](jax-transformability-and-gradients.md)
  — `jit`/`vmap`/`grad` through these wrappers, including which
  parameters carry gradients at each level.
- [VASP Data Ingestion](vasp-data-ingestion.md) — getting `eigenbands`
  and `surface_orb` out of EIGENVAL and PROCAR.
- API reference: {doc}`../api/simul`, {doc}`../api/types`.
