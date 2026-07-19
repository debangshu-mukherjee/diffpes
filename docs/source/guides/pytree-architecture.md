# PyTree Architecture

Every structured piece of data in diffpes — a band structure, an orbital
projection, a simulated spectrum, a bundle of simulation parameters — is a
JAX PyTree defined in `diffpes.types`. This guide tours that package: what
types exist, how they are registered and validated, how they flow through
`jit`/`grad`/`vmap`, how they persist to HDF5, and where the architecture is
headed (Equinox modules).

## The Single-Home Rule

diffpes enforces one structural rule without exception: **all types live in
`diffpes.types`**. Every PyTree, every type alias, and every `make_*` factory
that builds one is defined under `src/diffpes/types/` and nowhere else. The
consuming subpackages (`simul`, `tightb`, `radial`, `maths`, `inout`,
`utils`) import their types from `diffpes.types`; they never define their
own carriers.

The reason is the inverse problem. The fitting layer compares `ArpesSpectrum`
objects produced by different paths (simulation vs. measurement, one
parameter set vs. another), so there must be exactly *one* `ArpesSpectrum` —
one import surface, one registration, one validation contract. A result
container that "feels local" to a solver is still a type, and it still goes
in `diffpes.types`.

## The Type Inventory

The package is organized by physics domain, one module per family:

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

A few carriers worth calling out:

- **`BandStructure`** — `eigenvalues [K, B]`, `kpoints [K, 3]`,
  `kpoint_weights [K]`, `fermi_energy` (0-d). The spin-resolved variant
  `SpinBandStructure` carries `eigenvalues_up` / `eigenvalues_down` instead.
- **`OrbitalProjection`** — `projections [K, B, A, 9]` in the VASP orbital
  ordering, plus optional `spin [K, B, A, 6]` and `oam [K, B, A, 3]`
  channels. `SpinOrbitalProjection` is the same shape contract but with
  `spin` *mandatory* — the type system distinguishes "spin data may exist"
  from "spin data must exist" (SOC simulation requires the latter).
- **`ArpesSpectrum`** — the simulation output: `intensity [K, E]` and
  `energy_axis [E]`.
- **`SimulationParams`** — six 0-d float arrays (`energy_min`, `energy_max`,
  `sigma`, `gamma`, `temperature`, `photon_energy`) plus one Python `int`,
  `fidelity`, which sets the energy-axis length.
- **`SelfEnergyConfig`** — `coefficients [P]`, optional `energy_nodes [P]`,
  and a `mode` string; the coefficients are differentiable, which is what
  makes self-energy *fitting* a `jax.grad` call.

The scalar aliases in `types/aliases.py` (`ScalarFloat`, `ScalarInteger`,
`ScalarComplex`, `ScalarBool`, `ScalarNumeric`, `NonJaxNumber`) are unions
accepting both Python scalars and 0-d JAX arrays, so `sigma=0.04` and
`sigma=jnp.asarray(0.04)` are both valid at every API boundary.

## Registration: Children vs. Auxiliary Data

The current implementation registers each type as a `NamedTuple` decorated
with `@register_pytree_node_class`, with hand-written `tree_flatten` /
`tree_unflatten` methods. The flatten step draws the most important line in
the whole architecture:

- **Children** are JAX arrays. They are traced, differentiated, and batched.
  Eigenvalues, projections, intensities, broadening widths — anything you
  might attach a loss to.
- **Auxiliary data** is static Python metadata. It is a compile-time
  constant: changing it retriggers compilation of any `jit`-ted function
  that receives the PyTree.

Two deliberate examples of auxiliary data:

- `SimulationParams.fidelity` is a Python `int` because it determines the
  *length* of the energy axis, and JAX requires static shapes under `jit`.
- `PolarizationConfig.polarization_type` is a Python `str` (`"LVP"`,
  `"LHP"`, `"RCP"`, `"LCP"`, `"LAP"`, `"unpolarized"`) because it selects
  code branches in the matrix-element calculation.

Neither can be traced, and neither should be: a gradient with respect to
"number of energy points" or "polarization label" is not physics.

## Factories and Two-Tier Validation

PyTrees are never constructed directly in user code — each has a `make_*`
factory that validates inputs and casts arrays to `float64`. Validation is
two-tier:

1. **Static checks** resolve at trace time and use ordinary Python errors.
   Shape consistency is enforced by the `@jaxtyped(typechecker=beartype)`
   decorator on the factory itself — passing `eigenvalues [3, 2]` with
   `kpoints [4, 3]` to `make_band_structure` fails immediately because the
   `K` dimension names disagree — while structural checks raise
   `ValueError` (e.g. `make_slater_params` rejects a `zeta` whose length
   disagrees with the orbital basis, and `make_self_energy_config` rejects
   `mode="tabulated"` without `energy_nodes`).
2. **Traced checks** are data-dependent (finiteness, non-negativity) and
   cannot use Python `if` under `jit`. The current code guards them with
   `lax.cond` constructions that keep the factory traceable.

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

`kpoint_weights` defaults to uniform weights when omitted; every float field
comes back as a `float64` JAX array because diffpes enables x64 at import
(see [JAX Transformability and Gradients](jax-transformability-and-gradients.md)).

## Immutability and the jit/grad/vmap Flow

PyTrees are immutable — as `NamedTuple`s, field assignment raises. Updates
are functional: `bands._replace(fermi_energy=jnp.asarray(0.1))` builds a new
instance. This is not a style preference; it is what makes a PyTree safe to
pass through JAX transformations, which assume values are never mutated
behind the tracer's back.

Because children are ordinary leaves, a whole carrier can be the argument of
a transformation:

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

The gradient flows *into and out of* the `ArpesSpectrum` PyTree without any
unpacking: JAX flattens it to leaves, differentiates, and reassembles.
`vmap` behaves the same way — mapping a spectrum-returning function over a
batch of photon energies yields an `ArpesSpectrum` whose `intensity` has a
leading batch axis.

## HDF5 Round-Trip

`diffpes.inout.hdf5` persists any registered PyTree losslessly. Each named
PyTree becomes an HDF5 group; array children become datasets named after
their fields; auxiliary data is stored as a JSON group attribute; `None`
optional fields are recorded in a `_none_fields` attribute.

```python
from diffpes.inout import load_from_h5, save_to_h5

save_to_h5("run.h5", bands=bands, spectrum=spectrum)

everything = load_from_h5("run.h5")        # dict: {"bands": ..., "spectrum": ...}
bands_back = load_from_h5("run.h5", name="bands")  # single PyTree
```

The type name travels in a `_pytree_type` attribute, so `load_from_h5`
reconstructs the exact class via `tree_unflatten` — including static
auxiliary data like `fidelity` and `polarization_type`. Unknown type names
raise, rather than returning a lossy dict of arrays. See
[VASP Data Ingestion](vasp-data-ingestion.md) for the full ingest-simulate-save
pipeline.

## The Codified Direction: Equinox Modules

The [contributing guide](https://github.com/debangshu-mukherjee/diffpes/blob/main/CONTRIBUTING.md)
pins the target architecture for `diffpes.types`: structured types become
**Equinox modules** (`eqx.Module`), with static metadata declared via
`eqx.field(static=True)` and traced validation via `eqx.error_if`:

```python
import equinox as eqx
from jaxtyping import Array, Float

class BandStructure(eqx.Module):
    eigenvalues: Float[Array, "K B"]
    kpoints: Float[Array, "K 3"]
    kpoint_weights: Float[Array, " K"]
    fermi_energy: Float[Array, ""]
```

This is the same children/auxiliary split described above, minus the
boilerplate: `eqx.Module` derives `tree_flatten`/`tree_unflatten` from field
declarations, `eqx.field(static=True)` replaces hand-written aux plumbing
(`fidelity`, `polarization_type`), and `eqx.error_if` replaces the `lax.cond`
guard constructions with a purpose-built primitive that raises at runtime
without breaking `jit`. The user-facing contract — immutable PyTrees built by
validating `make_*` factories, imported only from `diffpes.types` — is
unchanged by the migration; per the project's zero-legacy policy the swap
happens in place, with no compatibility shims.

Until that lands, treat what this guide describes — `NamedTuple` +
`@register_pytree_node_class` + `lax.cond` guards — as the current reality,
and the Equinox form as the standard new code is written against.

## Related Reading

- [JAX Transformability and Gradients](jax-transformability-and-gradients.md)
  — how these PyTrees behave under `jit`, `grad`, and `vmap`, and the
  gradient-correctness doctrine.
- [VASP Data Ingestion](vasp-data-ingestion.md) — the readers that produce
  these PyTrees from VASP output files.
- [Expanded Wrappers and Conventions](expanded-wrappers-and-conventions.md)
  — the plain-array entry points that assemble these PyTrees for you.
- API reference: {doc}`../api/types`, {doc}`../api/inout`.
