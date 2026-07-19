---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Quickstart: From Bands to a Differentiable ARPES Spectrum

This tutorial walks the shortest path through diffpes: build a synthetic
band structure, attach orbital characters, simulate an ARPES spectrum at
two fidelity levels, and then differentiate through the entire
spectrometer model with `jax.grad`. Everything runs on CPU in under a
minute and every code cell below is executed when the documentation is
built, so the outputs you see are real.

## Installation

diffpes is installed from PyPI:

```bash
pip install diffpes
```

Importing `diffpes` automatically enables 64-bit (double) precision in
JAX and configures multi-threaded CPU execution, so you do not need to
set `jax_enable_x64` yourself. We verify that below.

```{code-cell} ipython3
import diffpes
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

print(f"diffpes {diffpes.__version__}")
print(f"x64 enabled: {jax.config.jax_enable_x64}")
```

## A Synthetic Band Structure

diffpes normally consumes electronic-structure output from DFT codes
(`diffpes.inout` parses VASP `EIGENVAL` and `PROCAR` files), but the
simulation layer only needs plain arrays. That makes it easy to start
with a toy model: two cosine bands on a one-dimensional k-path, the
dispersion you would get from nearest-neighbor tight binding on a chain.

The eigenvalue array has shape `[nkpt, nband]` — the first axis indexes
points along the momentum path, the second indexes bands. Energies are
in eV, measured relative to the Fermi level, which we place at
`ef = 0.0`. The upper band is given a slight phase offset so the two
bands are not symmetry-related, which will make the matrix-element
effects easier to see later.

```{code-cell} ipython3
nkpt = 200
nband = 2
kpath = jnp.linspace(-jnp.pi, jnp.pi, nkpt)
band_lower = -1.2 - 0.8 * jnp.cos(kpath)
band_upper = -0.4 - 0.9 * jnp.cos(kpath + 0.3)
eigenbands = jnp.stack([band_lower, band_upper], axis=1)
print(f"eigenbands shape: {eigenbands.shape}")
```

A quick line plot shows the two dispersions. The upper band crosses the
Fermi level near the zone boundary, so part of it will be cut off by
Fermi-Dirac occupation in the simulated spectrum.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(kpath, eigenbands[:, 0], label="lower band")
ax.plot(kpath, eigenbands[:, 1], label="upper band")
ax.axhline(0.0, color="gray", linestyle="--", linewidth=1, label=r"$E_F$")
ax.set_xlabel(r"$k$ (rad)")
ax.set_ylabel(r"$E - E_F$ (eV)")
ax.set_title("Synthetic two-band model")
ax.legend()
plt.show()
```

## Orbital Weights

ARPES intensity is not just the band structure — each state's orbital
character determines how strongly it photoemits at a given photon
energy and light polarization. diffpes encodes this in an
orbital-projection array of shape `[nkpt, nband, natom, 9]`, matching
the layout of a VASP `PROCAR` file.

The last axis follows the VASP 9-orbital ordering, zero-based:

| Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|-------|---|---|---|---|---|---|---|---|---|
| Orbital | $s$ | $p_y$ | $p_z$ | $p_x$ | $d_{xy}$ | $d_{yz}$ | $d_{z^2}$ | $d_{xz}$ | $d_{x^2-y^2}$ |

So `weights[:, 0, 0, 2]` is the $p_z$ weight of band 0 on atom 0 at
every k-point. Below we build a one-atom model where the lower band
changes character smoothly from $p_z$ at the zone center to $p_x$ at
the zone boundary, while the upper band is a fixed 70/30 mixture of
$d_{xy}$ and $d_{x^2-y^2}$.

```{code-cell} ipython3
natom = 1
weights = jnp.zeros((nkpt, nband, natom, 9))
pz_weight = jnp.cos(kpath / 2.0) ** 2
px_weight = jnp.sin(kpath / 2.0) ** 2
weights = weights.at[:, 0, 0, 2].set(pz_weight)
weights = weights.at[:, 0, 0, 3].set(px_weight)
weights = weights.at[:, 1, 0, 4].set(0.7)
weights = weights.at[:, 1, 0, 8].set(0.3)
surface_orb = weights
print(f"surface_orb shape: {surface_orb.shape}")
```

## Simulating the Spectrum

The single entry point {func}`diffpes.simul.simulate_expanded` takes
plain arrays plus scalar experiment parameters and dispatches to one of
six fidelity levels: `novice`, `basic`, `basicplus`, `advanced`,
`expert`, and `soc`. Each level adds physics; the
[guides](../guides/index.md) describe the full ladder.

We start at `level="basic"`: Gaussian broadening (`sigma`, in eV),
Fermi-Dirac occupation at the given `temperature`, and a heuristic
orbital weighting that depends on `photon_energy`. We use 21.2 eV — the
He-I$\alpha$ line of a helium lamp, the workhorse of lab-based ARPES.

```{code-cell} ipython3
spectrum_basic = diffpes.simul.simulate_expanded(
    level="basic",
    eigenbands=eigenbands,
    surface_orb=surface_orb,
    ef=0.0,
    sigma=0.06,
    fidelity=2000,
    temperature=30.0,
    photon_energy=21.2,
)
print(f"intensity shape: {spectrum_basic.intensity.shape}")
print(f"energy axis shape: {spectrum_basic.energy_axis.shape}")
```

The result is an {class}`~diffpes.types.ArpesSpectrum` PyTree with two
fields: `intensity` of shape `[nkpt, fidelity]` — the photoemission map
$I(k, E)$ — and `energy_axis` of shape `[fidelity]`, spanning the band
extrema plus 1 eV of padding on each side.

Now the same system at `level="advanced"`, which replaces the heuristic
orbital weights with interpolated Yeh-Lindau photoionization
cross-sections and adds polarization-dependent selection rules: the
intensity of each orbital channel is weighted by
$|\hat{\varepsilon} \cdot \vec{d}\,|^2$, where $\hat{\varepsilon}$ is
the light's electric-field direction. We choose p-polarized light
incident at 45 degrees, a common experimental geometry.

```{code-cell} ipython3
spectrum_adv = diffpes.simul.simulate_expanded(
    level="advanced",
    eigenbands=eigenbands,
    surface_orb=surface_orb,
    ef=0.0,
    sigma=0.06,
    fidelity=2000,
    temperature=30.0,
    photon_energy=21.2,
    polarization="p",
    incident_theta=45.0,
)
print(f"intensity shape: {spectrum_adv.intensity.shape}")
```

Plotting both spectra side by side. The intensity array is transposed
so energy runs vertically, as on a hemispherical-analyzer detector.

```{code-cell} ipython3
energy_axis = spectrum_basic.energy_axis
extent = [
    float(kpath[0]),
    float(kpath[-1]),
    float(energy_axis[0]),
    float(energy_axis[-1]),
]
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
titles = ["basic: heuristic weights", "advanced: p-pol matrix elements"]
for ax, spec, title in zip(axes, [spectrum_basic, spectrum_adv], titles):
    ax.imshow(
        spec.intensity.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="inferno",
    )
    ax.set_title(title)
    ax.set_xlabel(r"$k$ (rad)")
axes[0].set_ylabel(r"$E - E_F$ (eV)")
axes[0].set_ylim(-2.5, 0.5)
plt.show()
```

Both panels share the same band positions — matrix elements never move
a peak, they only rescale it. What changes between the levels is the
*relative brightness* along and between the bands. In the basic
spectrum the intensity simply follows the total orbital weight, so the
lower band is roughly uniform. In the advanced spectrum the
$p_z \to p_x$ character change of the lower band couples differently to
the p-polarized field, modulating its brightness along $k$, and the
$d$-derived upper band picks up a different overall cross-section
scale. This is exactly the kind of intensity variation that, in real
experiments, is often mistaken for a change in the underlying
electronic structure — and why a forward model of the matrix elements
matters when comparing theory to data. Note also how the Fermi-Dirac
factor extinguishes both bands above $E_F$ in the two panels.

## The Differentiable Hook

Everything above — Gaussian broadening, occupation, cross-sections,
polarization weights — is composed from pure JAX operations, so the
whole spectrometer model is differentiable end to end. To demonstrate,
we define a scalar observable: the total intensity in a narrow energy
window just below the Fermi level, as a function of the experimental
resolution `sigma` and the Fermi energy `ef`.

```{code-cell} ipython3
def window_intensity(sigma, ef):
    spec = diffpes.simul.simulate_expanded(
        level="basic",
        eigenbands=eigenbands,
        surface_orb=surface_orb,
        ef=ef,
        sigma=sigma,
        fidelity=2000,
        temperature=30.0,
        photon_energy=21.2,
    )
    in_window = (spec.energy_axis > -0.3) & (spec.energy_axis < 0.0)
    return jnp.sum(spec.intensity * in_window)
```

`jax.grad` differentiates this scalar through the entire simulation —
through the Voigt/Gaussian lineshapes, the Fermi function, and the
orbital weighting — with respect to whichever argument we pick.

```{code-cell} ipython3
grad_sigma = jax.grad(window_intensity, argnums=0)(0.06, 0.0)
grad_ef = jax.grad(window_intensity, argnums=1)(0.06, 0.0)
print(f"d(window intensity)/d(sigma) = {float(grad_sigma):.1f} per eV")
print(f"d(window intensity)/d(ef)    = {float(grad_ef):.1f} per eV")
```

Both gradients are positive: broadening leaks spectral weight from the
band bottoms into the near-$E_F$ window, and raising the Fermi level
un-occupies less of the upper band inside it. These are exact
derivatives, not finite differences.

This is the point of diffpes. A spectrometer model you can
differentiate is a spectrometer model you can *invert*: wrap a
chi-squared misfit between simulated and measured spectra in
`jax.grad`, and gradient-based optimizers (the Equinox/Optimistix
stack) can recover band parameters, self-energies, or experimental
settings directly from data. The same gradients also answer design
questions — how much would the measurement improve if the resolution
were 10 meV better? — without rerunning parameter sweeps.

## Where to Go Next

- [Theory and architecture guides](../guides/index.md) — the six
  fidelity levels in detail, the dipole physics behind the `advanced`
  and `expert` levels, and which JAX transformations (`grad`, `jit`,
  `vmap`) are supported where.
- [API reference](../api/index.rst) — full function-level
  documentation, including the VASP readers in `diffpes.inout` for
  running this workflow on real DFT output.
