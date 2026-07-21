# Simulation Levels

diffpes provides six projection-based ARPES simulation levels: **novice**,
**basic**, **basicplus**, **advanced**, **expert**, and **soc**. Each level
balances physical fidelity, required inputs, and computational cost. All
levels sum and weight the bands. They also apply Fermi-Dirac occupation and
broaden the bands on an energy axis. The levels use different band weights
and lineshapes.

This guide describes each level and identifies its approximations. It also
shows the `simulate_expanded(level=...)` dispatch.

## The Common Skeleton

Every level evaluates an intensity map of the form

$$
I(\mathbf{k}, E) = \sum_{b} W_b(\mathbf{k})\;
    f\!\left(E_b(\mathbf{k}); E_F, T\right)\;
    \mathcal{L}\!\left(E - E_b(\mathbf{k})\right)
$$

with

- $E_b(\mathbf{k})$ contains band eigenvalues with shape `(K, B)` in eV.
  VASP `EIGENVAL` can supply these values.
- $W_b(\mathbf{k})$ contains weights from orbital projections with shape
  `(K, B, A, 9)`. The last axis follows the VASP orbital order
  `[s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2]`.
- $f$ is the Fermi-Dirac occupation at temperature $T$.
- $\mathcal{L}$ is a normalized Gaussian or pseudo-Voigt profile. The profile
  uses Gaussian width $\sigma$ and optional Lorentzian width $\gamma$.

See [Spectral Broadening and Self-Energy](spectral-broadening-and-self-energy.md) for the lineshapes and [ARPES Geometry and Kinematics](arpes-geometry-and-kinematics.md) for the axes and units.

## The Ladder

| Level | Lineshape | Orbital weights | Polarization | Spin | Extra inputs |
|-------|-----------|-----------------|--------------|------|--------------|
| `novice` | Voigt ($\sigma$, $\gamma$) | Uniform (non-s orbitals summed equally) | — | — | — |
| `basic` | Gaussian ($\sigma$) | Heuristic two-regime weights vs. $h\nu$ | — | — | — |
| `basicplus` | Gaussian ($\sigma$) | Interpolated Yeh-Lindau cross-sections | — | — | — |
| `advanced` | Gaussian ($\sigma$) | Yeh-Lindau $\times\; \lvert \hat{e}\cdot\hat{d}_o \rvert^2$ | yes | — | `polarization`, `incident_theta/phi` |
| `expert` | Voigt ($\sigma$, $\gamma$) | Yeh-Lindau $\times\; \lvert \hat{e}\cdot\hat{d}_o \rvert^2$ dipole weighting | yes | — | as `advanced` + `gamma` |
| `soc` | Voigt ($\sigma$, $\gamma$) | As `expert`, modulated by $1 + \lambda\, \mathbf{S}\cdot\hat{k}_{\mathrm{ph}}$ | yes | yes | `surface_spin` `(K, B, A, 6)`, `ls_scale` |

### What each level adds

**novice** uses the minimum ARPES model. Each band becomes a Voigt peak with
Fermi-Dirac occupation. The model sums all non-s orbital channels with equal
weight. Use this level for quick checks of the band geometry.

**basic** uses the `heuristic_weights` energy rule. Below 50 eV, the rule gives
the three p channels a weight of 2. Above 50 eV, it gives the five d channels
a weight of 2. This coarse approximation does not use tabulated cross-sections.

**basicplus** uses `yeh_lindau_weights`. The function linearly interpolates a
simplified Yeh-Lindau (1985) table at 20, 40, and 60 eV. It broadcasts the
subshell cross-sections to the nine-orbital basis:

```python
import diffpes

print(diffpes.simul.heuristic_weights(21.2))
# [1. 2. 2. 2. 1. 1. 1. 1. 1.]
print(diffpes.simul.yeh_lindau_weights(40.0))
# [0.08 0.9  0.9  0.9  1.5  1.5  1.5  1.5  1.5]
```

**advanced** adds polarization selection rules to the Yeh-Lindau weights.
`polarization`, `incident_theta`, and `incident_phi` define the electric field
$\hat{e}$. The angles use degrees. The model weights each orbital channel by
$\lvert \hat{e} \cdot \hat{d}_o \rvert^2$. `ORBITAL_DIRS_NORMALIZED` supplies
one fixed unit direction $\hat{d}_o$ per orbital. Unpolarized light averages
the s- and p-polarization results.

**expert** combines Voigt broadening, Yeh-Lindau cross-sections, and
polarization-dependent dipole weights. It is the most complete
projection-based level.

**soc** adds a **phenomenological spin scale** to the expert calculation. It
requires `surface_spin` projections with shape `(K, B, A, 6)`. These channels
contain the up and down components for $x$, $y$, and $z$. A spin-polarized
`PROCAR` can supply this data. The level changes each band intensity by

$$
W_b \rightarrow W_b \left( 1 + \lambda \; \mathbf{S}_b \cdot \hat{k}_{\mathrm{ph}} \right)
$$

Here, $\hat{k}_{\mathrm{ph}}$ is the photon propagation direction, and
$\lambda$ is `ls_scale`. Its default is 0.01. With `ls_scale=0`, the result
equals the expert result. This scale approximates spin-dependent intensity
asymmetries. It does not compute a matrix element.

## Scope of the Approximations

Several level components are explicit approximations:

- The **heuristic weights** in `basic` approximate atomic cross-sections. The
  simplified Yeh-Lindau table has three energy points and no element
  resolution.
- The **orbital-direction dipole model** assigns one Cartesian direction to
  each orbital. It uses $\lvert \hat{e}\cdot\hat{d} \rvert^2$ as the score.
  The model reproduces qualitative selection rules but does not compute a
  quantum-mechanical matrix element. Its zero direction gives the s orbital
  zero weight.
- The **SOC spin scale** is phenomenological. `ls_scale` is a fit parameter,
  not a derived coupling constant.

`simulate_tb_radial` provides the full matrix-element alternative. It combines
radial integrals, Gaunt coefficients, and real spherical harmonics from
{mod}`diffpes.maths` and {mod}`diffpes.radial`. See
[Matrix Elements and Polarization](matrix-elements-and-polarization.md). The
six levels remain useful because they are differentiable and low-cost. They
also require only standard VASP outputs.

## Dispatch with `simulate_expanded`

Pass a case-insensitive level name and plain arrays to `simulate_expanded`.
The function calls the matching `simulate_*_expanded` wrapper. Each level
ignores parameters that it does not use. For example, Gaussian-only levels
ignore `gamma`. Levels below `advanced` ignore polarization angles.

```python
import jax.numpy as jnp
import diffpes

# Synthetic inputs (see the geometry guide for the physical meaning)
n_k, n_b, n_atoms = 61, 2, 1
k_frac = jnp.linspace(-0.5, 0.5, n_k)
band_1 = -1.2 + 0.9 * jnp.cos(2.0 * jnp.pi * k_frac)
band_2 = -2.6 + 0.4 * jnp.cos(2.0 * jnp.pi * k_frac + jnp.pi)
eigenbands = jnp.stack([band_1, band_2], axis=1)       # (K, B) eV
surface_orb = jnp.ones((n_k, n_b, n_atoms, 9)) / 9.0   # (K, B, A, 9)

spectrum = diffpes.simul.simulate_expanded(
    level="expert",
    eigenbands=eigenbands,
    surface_orb=surface_orb,
    ef=0.0,                  # Fermi level (eV)
    sigma=0.03,              # Gaussian width (eV)
    gamma=0.05,              # Lorentzian half-width (eV) - Voigt levels only
    fidelity=2000,           # energy-axis points
    temperature=20.0,        # Kelvin
    photon_energy=40.0,      # eV
    polarization="lhp",      # p-polarized light
    incident_theta=45.0,     # degrees from surface normal
    incident_phi=0.0,        # degrees, azimuthal
)
print(spectrum.intensity.shape)    # (61, 2000)
print(spectrum.energy_axis.shape)  # (2000,)
```

The SOC level additionally needs the spin array:

```python
surface_spin = jnp.zeros((n_k, n_b, n_atoms, 6))  # (x-up, x-dn, y-up, y-dn, z-up, z-dn)

spec_soc = diffpes.simul.simulate_expanded(
    level="soc",
    eigenbands=eigenbands,
    surface_orb=surface_orb,
    surface_spin=surface_spin,
    ef=0.0,
    sigma=0.03,
    gamma=0.05,
    fidelity=2000,
    temperature=20.0,
    photon_energy=40.0,
    ls_scale=0.01,           # phenomenological spin-orbit intensity dial
)
```

An unknown `level` string, or `level="soc"` without `surface_spin`, raises a `ValueError`.

### Sensible defaults

The function requires only `level`, `eigenbands`, and `surface_orb`. The width
defaults are `sigma=0.04` and `gamma=0.1`. Other defaults include `ef=0.0`,
`fidelity=25000`, and `temperature=15.0` K. The photon energy defaults to
11.0 eV. The default uses unpolarized light at 45 degrees incidence.
`make_expanded_simulation_params` derives the energy axis from the band range.

## Choosing a Level

- Use `novice` to debug a band structure or k-path.
- Use `basicplus` for photon-energy trends without polarization. Use `basic`
  only for the two-regime heuristic.
- Use `advanced` or `expert` for polarization and geometry experiments.
- Use `soc` for spin-resolved or dichroism studies. Treat `ls_scale` as a fit
  parameter.
- Use `simulate_tb_radial` for quantitative matrix-element physics or
  gradient-based recovery of orbital parameters. See
  [Matrix Elements and Polarization](matrix-elements-and-polarization.md).

All levels are JAX-differentiable with respect to their continuous inputs.
These inputs include widths, temperature, eigenvalues, and projections. Any
level can run inside a `jax.grad` loss.

## Further Reading

- [ARPES Geometry and Kinematics](arpes-geometry-and-kinematics.md) defines the
  axes, units, and incidence-angle convention.
- [Matrix Elements and Polarization](matrix-elements-and-polarization.md)
  describes the full dipole engine and the heuristic weights.
- [Spectral Broadening and Self-Energy](spectral-broadening-and-self-energy.md)
  compares Gaussian and Voigt profiles. It also describes energy-dependent
  linewidths.
- API reference: {doc}`../api/simul` (all `simulate_*` functions), {doc}`../api/inout` (VASP parsers feeding the ladder)
