# Simulation Levels

diffpes organizes its projection-based ARPES forward models into a six-level ladder: **novice**, **basic**, **basicplus**, **advanced**, **expert**, and **soc**. Each level is a self-contained approximation tier — a deliberate trade-off between physical fidelity, required inputs, and computational cost. All six share the same skeleton (sum bands, weight them, occupy them with Fermi-Dirac, broaden them onto an energy axis) and differ only in *how the per-band weight is computed* and *which lineshape is used*.

This guide describes what each level adds, states honestly which ingredients are placeholders, and shows the single-entry-point dispatch `simulate_expanded(level=...)`.

## The Common Skeleton

Every level evaluates an intensity map of the form

$$
I(\mathbf{k}, E) = \sum_{b} W_b(\mathbf{k})\;
    f\!\left(E_b(\mathbf{k}); E_F, T\right)\;
    \mathcal{L}\!\left(E - E_b(\mathbf{k})\right)
$$

with

- $E_b(\mathbf{k})$ — band eigenvalues, shape `(K, B)` in eV (e.g. from VASP `EIGENVAL`)
- $W_b(\mathbf{k})$ — a weight built from the orbital projections `(K, B, A, 9)` (e.g. from VASP `PROCAR`), in VASP orbital order `[s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2]`
- $f$ — the Fermi-Dirac occupation at temperature $T$
- $\mathcal{L}$ — a normalized Gaussian (width $\sigma$) or pseudo-Voigt (Gaussian $\sigma$ + Lorentzian $\gamma$) profile

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

### What each rung adds

**novice** — the minimal model that still looks like ARPES: every band is a Voigt peak with Fermi-Dirac occupation. Orbital projections enter only as a total weight (all non-s channels summed with equal weight). Use it for quick sanity checks of the band geometry.

**basic** — swaps the uniform weights for an energy-regime heuristic, `heuristic_weights`: below 50 eV photon energy the three p channels get weight 2 (He-I / laser regime), above 50 eV the five d channels get weight 2 (He-II / synchrotron regime). This is a *coarse placeholder* for cross-section physics, not tabulated data.

**basicplus** — replaces the heuristic with `yeh_lindau_weights`: per-subshell photoionization cross-sections linearly interpolated from a *simplified* tabulation of Yeh & Lindau (1985) at 20/40/60 eV, broadcast to the 9-orbital basis:

```python
import diffpes

print(diffpes.simul.heuristic_weights(21.2))
# [1. 2. 2. 2. 1. 1. 1. 1. 1.]
print(diffpes.simul.yeh_lindau_weights(40.0))
# [0.08 0.9  0.9  0.9  1.5  1.5  1.5  1.5  1.5]
```

**advanced** — adds polarization selection rules on top of the Yeh-Lindau weights. The electric-field vector $\hat{e}$ is built from `polarization`, `incident_theta`, and `incident_phi` (degrees), and each orbital channel $o$ is weighted by $\lvert \hat{e} \cdot \hat{d}_o \rvert^2$ where $\hat{d}_o$ is a fixed unit direction vector per orbital (`ORBITAL_DIRS_NORMALIZED`). Unpolarized light averages the s- and p-polarization results.

**expert** — the most complete projection-based tier: Voigt broadening + Yeh-Lindau cross-sections + polarization-dependent dipole weighting, all together.

**soc** — the expert pipeline plus a **phenomenological spin dial**. It requires spin projections `surface_spin` of shape `(K, B, A, 6)` (up/down for $x$, $y$, $z$, e.g. from a spin-polarized `PROCAR`) and modulates each band's intensity by

$$
W_b \rightarrow W_b \left( 1 + \lambda \; \mathbf{S}_b \cdot \hat{k}_{\mathrm{ph}} \right)
$$

where $\hat{k}_{\mathrm{ph}}$ is the photon propagation direction and $\lambda$ = `ls_scale` (default 0.01). With `ls_scale=0` the result reduces to the expert level. This mimics spin-dependent intensity asymmetries (e.g. circular dichroism trends) but is an **intensity dial, not a matrix-element calculation**.

## Honest Framing: These Are Approximation Tiers

Several ingredients in the ladder are stated placeholders, and the documentation should not oversell them:

- The **heuristic weights** (`basic`) and the **simplified Yeh-Lindau tabulation** (three energy points per subshell, no element resolution) are rough stand-ins for real atomic cross-sections.
- The **orbital-direction dipole model** used by `advanced`/`expert` assigns each orbital a single Cartesian direction and scores it with $\lvert \hat{e}\cdot\hat{d} \rvert^2$. It reproduces qualitative selection-rule behavior (e.g. s-polarized light suppressing out-of-plane orbitals) but is not a quantum-mechanical matrix element — notably, the s-orbital gets *zero* weight at these levels because its direction vector is the zero vector.
- The **SOC spin dial** is purely phenomenological; `ls_scale` is a fitting knob, not a derived coupling constant.

These heuristics are slated for replacement by the full matrix-element engine already present in {mod}`diffpes.maths` and {mod}`diffpes.radial` — radial integrals, Gaunt coefficients, and real spherical harmonics assembled in `simulate_tb_radial` — see [Matrix Elements and Polarization](matrix-elements-and-polarization.md). The level ladder remains valuable because each tier is cheap, differentiable, and requires nothing beyond standard VASP outputs.

## Dispatch with `simulate_expanded`

`simulate_expanded` is the single entry point: pass the level name (case-insensitive) plus plain arrays, and it routes to the matching `simulate_*_expanded` wrapper. Parameters a level does not use are silently ignored (e.g. `gamma` at the Gaussian-only levels, polarization angles below `advanced`).

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

Only `level`, `eigenbands`, and `surface_orb` are required. The defaults are `ef=0.0`, `sigma=0.04`, `gamma=0.1`, `fidelity=25000`, `temperature=15.0` K, `photon_energy=11.0` eV, unpolarized light at 45° incidence. The energy axis is auto-derived as `linspace(min(eigenbands) - 1, max(eigenbands) + 1, fidelity)` via `make_expanded_simulation_params`.

## Choosing a Level

- **Debugging a band structure or k-path** → `novice` (fewest knobs, no weight surprises).
- **Photon-energy trends without polarization** → `basicplus` (skip `basic` unless you specifically want the two-regime heuristic).
- **Polarization/geometry experiments (light-polarization switching, orbital contrast)** → `advanced` or `expert`.
- **Spin-resolved or dichroism-flavored questions** → `soc`, treating `ls_scale` as a fit parameter.
- **Quantitative matrix-element physics or gradient-based recovery of orbital parameters** → step off the ladder to `simulate_tb_radial` (see [Matrix Elements and Polarization](matrix-elements-and-polarization.md)).

All levels are JAX-differentiable with respect to their continuous inputs (`sigma`, `gamma`, `temperature`, eigenvalues, projections, ...), so any tier can sit inside a `jax.grad` loss.

## Further Reading

- [ARPES Geometry and Kinematics](arpes-geometry-and-kinematics.md) — axes, units, and the incidence-angle convention
- [Matrix Elements and Polarization](matrix-elements-and-polarization.md) — the full dipole engine that supersedes the heuristic weights
- [Spectral Broadening and Self-Energy](spectral-broadening-and-self-energy.md) — Gaussian vs. Voigt, and energy-dependent linewidths
- API reference: {doc}`../api/simul` (all `simulate_*` functions), {doc}`../api/inout` (VASP parsers feeding the ladder)
