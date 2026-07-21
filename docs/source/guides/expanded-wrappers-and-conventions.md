# Expanded Wrappers and Conventions

The expanded-input wrappers in `diffpes.simul` accept plain arrays and scalars.
They build the PyTrees internally and execute the same JAX kernels. These
wrappers support quick interactive work and migration from script-based
workflows. Such workflows include the historical `ARPES_simulation_*`
function family. This page defines the wrapper family and its argument
conventions. The conventions cover energy-axis padding, angle units, and array
indexing across diffpes.

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

Every wrapper accepts `eigenbands [K, B]`, which contains eigenvalues in eV.
It also accepts `surface_orb [K, B, A, 9]`, which contains orbital
projections. Additional scalar parameters depend on the selected level. The
wrapper builds `BandStructure`, `OrbitalProjection`, and `SimulationParams`
PyTrees. Applicable wrappers also build a `PolarizationConfig` PyTree. The
wrapper then calls the corresponding `simulate_novice` … `simulate_soc` core
function. The SOC wrapper also requires `surface_spin [K, B, A, 6]`. This
array uses the `[Sx+, Sx-, Sy+, Sy-, Sz+, Sz-]` channel convention.

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
- The dispatcher requires only `level`, `eigenbands`, and `surface_orb`.
  Every other parameter has a default: `ef=0.0`, `sigma=0.04`, `gamma=0.1`,
  `fidelity=25000`, `temperature=15.0`, `photon_energy=11.0`, and
  `incident_theta=45.0`.
- The dispatcher **silently ignores** parameters that the selected level does
  not use. For example, `level="basic"` ignores `gamma`. The novice level
  ignores polarization settings.
- `level="soc"` requires `surface_spin`; omitting it raises `ValueError`.
  `ls_scale` (default `0.01`) sets the spin-orbit coupling strength.

Mark `level`, `polarization`, and `fidelity` as static under `jax.jit`.
These Python `str` and `int` values select code paths and array shapes. See
[JAX Transformability and Gradients](jax-transformability-and-gradients.md).

## The Returned `ArpesSpectrum`

All wrappers return the standard `ArpesSpectrum` PyTree from `diffpes.types`.
The PyTree-level API and fitting layer use the same carrier:

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

with `fidelity` evenly spaced points. The default `energy_padding` is 1.0 eV.
Call `make_expanded_simulation_params` directly to adjust this value. In the
preceding example, eigenvalues span `[-2.0, 0.5]` eV. Therefore, the axis spans
`[-3.0, 1.5]` eV. Build `SimulationParams` directly when comparisons require a
fixed window. Use
`diffpes.types.make_simulation_params(energy_min=..., energy_max=..., ...)`,
then call a PyTree-level `simulate_*` function.

## Angle and Polarization Conventions

- **Incident angles are degrees at the wrapper boundary.**
  The wrapper interprets `incident_theta` from the surface normal. It
  interprets `incident_phi` as the azimuthal angle. The wrapper converts both
  values to radians once. It then stores them on `PolarizationConfig`, whose
  `theta` and `phi` fields use radians.
- **`polarization_angle` is radians.** It is the rotation angle for
  arbitrary linear polarization (`"LAP"`). The wrapper does not convert it.
- **`polarization` is a case-insensitive string**: `"LVP"` (s-pol),
  `"LHP"` (p-pol), `"RCP"`, `"LCP"`, `"LAP"`, or `"unpolarized"`.
  The field builder maps unrecognized strings to s-polarization. Use an exact
  supported value.
- Energies use eV. Lengths use Angstrom, k-vectors use 1/Angstrom, and
  temperatures use Kelvin.

## Python Indexing Conventions

diffpes uses standard Python/NumPy indexing everywhere: **zero-based,
end-exclusive**. This applies to atom indices (`atom_indices=[0, 1]` means
the first two atoms), band indices, k-point indices, and orbital channels.

The 9 orbital channels of `surface_orb[..., :]` follow the VASP ordering
`[s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2]`. The following slices select
the orbital families:

| Family | Slice | Indices selected |
|---|---|---|
| non-s orbitals | `slice(1, 9)` | 1–8 (all p and d) |
| p orbitals | `slice(1, 4)` | 1, 2, 3 (py, pz, px) |
| d orbitals | `slice(4, 9)` | 4–8 (dxy, dyz, dz2, dxz, dx2-y2) |

```python
p_weight = surface_orb[..., 1:4].sum(axis=-1)   # total p character
d_weight = surface_orb[..., 4:9].sum(axis=-1)   # total d character
```

Do not use MATLAB-style one-based, end-inclusive notation in code, comments,
or documentation. When you port a MATLAB selection of "orbitals 2:9", use
the diffpes equivalent `slice(1, 9)`.

## Wrapper Layers

The API provides three tiers, from most to least manual control:

1. **PyTree level** uses `simulate_novice` … `simulate_soc` and the
   first-principles `simulate_tb_radial`. Build every input PyTree with the
   `diffpes.types` factories.
2. **Expanded level** uses `simulate_*_expanded` or `simulate_expanded`.
   Supply plain arrays. The wrapper assembles PyTrees and derives the energy
   window.
3. **Workflow level** uses `run_vasp_workflow` or `load_vasp_context` with
   `simulate_context`. Start from VASP files on disk. The context loader calls
   `simulate_expanded` internally. See
   [VASP Data Ingestion](vasp-data-ingestion.md).

All three layers return the same `ArpesSpectrum`. Their continuous inputs
remain differentiable. The wrappers add no `stop_gradient` operations.
Therefore, `jax.grad` reaches `eigenbands`, `surface_orb`, `ef`, `sigma`, and
the other continuous inputs. PyTree-level functions provide the same gradient
paths.

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
