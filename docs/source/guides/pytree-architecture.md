# PyTree Architecture

`diffpes.types` defines every structured value as a JAX PyTree. These values
include band structures, orbital projections, spectra, and simulation
parameters. This guide describes the available types and their validation.
It also describes JAX transformations, HDF5 persistence, and Equinox modules.

## The Single-Home Rule

diffpes enforces one structural rule: **all types live in `diffpes.types`**.
The `src/diffpes/types/` directory owns every PyTree, type alias, and
`make_*` factory. The consuming subpackages import these types from
`diffpes.types`. They do not define local carriers.

The inverse problem requires this rule. The fitting layer compares
`ArpesSpectrum` objects from simulations, measurements, and different
parameter sets. Therefore, one `ArpesSpectrum` must have one import surface
and one validation contract. A solver result container is also a type. Define
it in `diffpes.types`.

## The Type Inventory

The package groups types by physics domain, with one module for each family:

| Module | PyTrees | Factories |
|---|---|---|
| `types/bands.py` | `BandStructure`, `SpinBandStructure`, `OrbitalProjection`, `SpinOrbitalProjection`, `ArpesSpectrum` | `make_band_structure`, `make_spin_band_structure`, `make_orbital_projection`, `make_spin_orbital_projection`, `make_arpes_spectrum` |
| `types/params.py` | `SimulationParams`, `PolarizationConfig` | `make_simulation_params`, `make_polarization_config` |
| `types/geometry.py` | `CrystalGeometry` | `make_crystal_geometry` |
| `types/kpath.py` | `KPathInfo` | `make_kpath_info` |
| `types/dos.py` | `DensityOfStates`, `FullDensityOfStates` | `make_density_of_states`, `make_full_density_of_states` |
| `types/volumetric.py` | `VolumetricData`, `SOCVolumetricData` | `make_volumetric_data`, `make_soc_volumetric_data` |
| `types/self_energy.py` | `SelfEnergyConfig` | `make_self_energy_config` |
| `types/radial_params.py` | `OrbitalBasis`, `SlaterParams` | `make_orbital_basis`, `make_slater_params` |
| `types/tb_model.py` | `TBModel`, `DiagonalizedBands` | `make_tb_model`, `make_diagonalized_bands` |
| `types/aliases.py` | — | scalar aliases (below) |

The following carriers define important contracts:

- **`BandStructure`** contains `eigenvalues [K, B]`, `kpoints [K, 3]`,
  `kpoint_weights [K]`, `fermi_energy` (0-d). The spin-resolved variant
  `SpinBandStructure` carries `eigenvalues_up` / `eigenvalues_down` instead.
- **`OrbitalProjection`** contains `projections [K, B, A, 9]` in the VASP
  ordering, plus optional `spin [K, B, A, 6]` and `oam [K, B, A, 3]`
  channels. `SpinOrbitalProjection` uses the same shapes but requires `spin`.
  This contract distinguishes optional spin data from required SOC data.
- **`ArpesSpectrum`** contains the simulation output: `intensity [K, E]` and
  `energy_axis [E]`.
- **`SimulationParams`** contains six 0-d float arrays (`energy_min`,
  `energy_max`,
  `sigma`, `gamma`, `temperature`, `photon_energy`) plus one Python `int`,
  `fidelity`, which sets the energy-axis length.
- **`SelfEnergyConfig`** contains `coefficients [P]`, optional
  `energy_nodes [P]`, and a `mode` string. Its coefficients are
  differentiable, so `jax.grad` supports self-energy fitting.

The scalar aliases in `types/aliases.py` accept Python scalars and 0-d JAX
arrays. The aliases include `ScalarFloat`, `ScalarInteger`, `ScalarComplex`,
`ScalarBool`, `ScalarNumeric`, and `NonJaxNumber`. Therefore, APIs accept both
`sigma=0.04` and `sigma=jnp.asarray(0.04)`.

## Registration: Children vs. Auxiliary Data

Every carrier is an `eqx.Module`. Field declarations automatically define
the flatten and unflatten operations. The fields have two categories:

- **Children** are JAX arrays. JAX traces, differentiates, and batches them.
  Examples include eigenvalues, projections, intensities, and broadening
  widths.
- **Auxiliary data** is static Python metadata, declared with
  `eqx.field(static=True)`. This metadata is a compile-time constant. A change
  retriggers compilation for each `jit`-compiled function that receives the
  PyTree.

Two deliberate examples of auxiliary data:

- `SimulationParams.fidelity` is a Python `int` because it sets the energy-axis
  length. JAX requires static shapes under `jit`.
- `PolarizationConfig.polarization_type` is a Python `str` (`"LVP"`,
  `"LHP"`, `"RCP"`, `"LCP"`, `"LAP"`, `"unpolarized"`). It selects branches
  in the matrix-element calculation.

JAX does not trace these fields. A gradient for an energy-point count or a
polarization label has no physical meaning.

## Factories and Two-Tier Validation

Use the `make_*` factories to construct PyTrees. Each factory validates inputs
and casts float arrays to `float64`. Validation has two tiers:

1. **Static checks** resolve at trace time and use ordinary Python errors.
   The `@jaxtyped(typechecker=beartype)` decorator enforces shape consistency.
   For example, unequal `K` dimensions in `eigenvalues` and `kpoints` fail
   immediately. Structural checks raise `ValueError`. `make_slater_params`
   rejects a `zeta` length that differs from the orbital basis.
   `make_self_energy_config` rejects `mode="tabulated"` without
   `energy_nodes`.
2. **Traced checks** are data-dependent (finiteness, non-negativity) and
   cannot use Python `if` under `jit`. The factories use
   `equinox.error_if` to keep these checks traceable.

```python
import jax.numpy as jnp
from diffpes.types import make_band_structure

bands = make_band_structure(
    eigenvalues=jnp.linspace(-2.0, 0.5, 100).reshape(20, 5),  # [K, B]
    kpoints=jnp.zeros((20, 3)),                               # [K, 3]
    fermi_energy=0.0,
)
print(bands.eigenvalues.shape, float(bands.fermi_energy))  # (20, 5) 0.0
```

When omitted, `kpoint_weights` uses uniform weights. Every float field returns
as a `float64` JAX array because diffpes enables x64 at import. See
[JAX Transformability and Gradients](jax-transformability-and-gradients.md).

## Immutability and the jit/grad/vmap Flow

`eqx.Module` keeps PyTrees immutable, so field assignment raises an error.
Updates use functional operations:
`eqx.tree_at(lambda t: t.fermi_energy, bands, jnp.asarray(0.1))` builds a
new instance with one changed leaf. This behavior lets JAX transformations
assume that traced values do not change unexpectedly.

JAX transformations can accept a whole carrier because children are ordinary
leaves:

```python
import jax
import jax.numpy as jnp
from diffpes.simul import simulate_expanded

eigenbands = jnp.linspace(-2.0, 0.5, 100).reshape(20, 5)
surface_orb = jnp.ones((20, 5, 2, 9)) * 0.1

def peak(ef):
    spectrum = simulate_expanded(
        level="basic", eigenbands=eigenbands, surface_orb=surface_orb,
        ef=ef, sigma=0.04, fidelity=500,
        temperature=15.0, photon_energy=11.0,
    )
    return jnp.max(spectrum.intensity)

print(jax.grad(peak)(0.0))  # d(peak intensity)/d(E_F), a 0-d array
```

The gradient flows through the `ArpesSpectrum` PyTree without manual
unpacking. JAX flattens the carrier, differentiates its leaves, and
reassembles it. `vmap` uses the same process. A photon-energy batch adds a
leading batch axis to `ArpesSpectrum.intensity`.

## HDF5 Round-Trip

`diffpes.inout.hdf5` preserves any registered PyTree without data loss. Each
named PyTree becomes an HDF5 group. Array children become datasets named after
their fields. A JSON group attribute stores auxiliary data. The
`_none_fields` attribute records optional fields that contain `None`.

```python
from diffpes.inout import load_from_h5, save_to_h5

save_to_h5("run.h5", bands=bands, spectrum=spectrum)

everything = load_from_h5("run.h5")        # dict: {"bands": ..., "spectrum": ...}
bands_back = load_from_h5("run.h5", name="bands")  # single PyTree
```

The `_pytree_type` attribute stores the type name. `load_from_h5` uses
`tree_unflatten` to reconstruct the exact class and its static auxiliary data.
Unknown type names raise an error. The loader does not return a lossy array
dictionary. See
[VASP Data Ingestion](vasp-data-ingestion.md) for the full ingest-simulate-save
pipeline.

## The Equinox Module Pattern

The [contributing guide](https://github.com/debangshu-mukherjee/diffpes/blob/main/CONTRIBUTING.md)
defines the `diffpes.types` architecture. Every structured type is an
**Equinox module** (`eqx.Module`). The `eqx.field(static=True)` declaration
marks static metadata:

```python
import equinox as eqx
from jaxtyping import Array, Float

class BandStructure(eqx.Module):
    eigenvalues: Float[Array, "K B"]
    kpoints: Float[Array, "K 3"]
    kpoint_weights: Float[Array, " K"]
    fermi_energy: Float[Array, ""]
```

This pattern implements the child and auxiliary-data split without custom
registration. `eqx.Module` derives flattening operations from the field
declarations. `eqx.field(static=True)` identifies metadata such as `fidelity`
and `polarization_type`. The zero-legacy migration removed the earlier
hand-registered `NamedTuple` form. The user contract remains unchanged. Use
validated, immutable PyTrees from `diffpes.types`.

## Related Reading

- [JAX Transformability and Gradients](jax-transformability-and-gradients.md)
  describes PyTrees under `jit`, `grad`, and `vmap`. It also defines the
  gradient-correctness rules.
- [VASP Data Ingestion](vasp-data-ingestion.md) describes readers that create
  PyTrees from VASP output files.
- [Expanded Wrappers and Conventions](expanded-wrappers-and-conventions.md)
  describes plain-array functions that assemble these PyTrees.
- API reference: {doc}`../api/types`, {doc}`../api/inout`.
