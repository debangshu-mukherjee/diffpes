# ARPES Geometry and Kinematics

Angle-Resolved PhotoEmission Spectroscopy (ARPES) measures the kinetic energy and emission angle of electrons ejected from a solid by monochromatic light. Because both energy and crystal momentum are conserved in the photoemission process, the measured $(E_{\mathrm{kin}}, \theta, \phi)$ distribution can be inverted into the occupied band structure $E(\mathbf{k})$. This guide covers the kinematic relations diffpes uses, the incidence-geometry conventions of its API, and how a simulated $(E, k)$ intensity map is produced.

## Unit and Sign Conventions

diffpes pins the following conventions throughout the codebase:

| Quantity | Unit | Notes |
|----------|------|-------|
| Energies ($h\nu$, $E_B$, $E_F$, $\sigma$, $\gamma$) | eV | Band eigenvalues are absolute eV; the Fermi level `ef` is passed separately |
| Wavevectors $\mathbf{k}$ | $\text{Å}^{-1}$ | Reciprocal lattice includes the $2\pi$ prefactor |
| Angles at API boundaries | degrees | `incident_theta`, `incident_phi` in the `simulate_*_expanded` family |
| Angles internally | radians | `PolarizationConfig.theta/phi`, `make_polarization_config` |
| Temperature | Kelvin | Enters through the Fermi-Dirac factor |
| Precision | float64 | `jax_enable_x64` is switched on when `diffpes` is imported |

Polar angles $\theta$ are measured from the surface normal ($z$-axis); azimuthal angles $\phi$ are measured in the surface plane from the $x$-axis.

## Energy Conservation and the Work Function

In the three-step model of photoemission, an electron bound at energy $E_B$ below the Fermi level absorbs a photon of energy $h\nu$, travels to the surface, and escapes over the surface potential barrier. Its kinetic energy in vacuum is

$$
E_{\mathrm{kin}} = h\nu - W - |E_B|
$$

where:

- $h\nu$ = photon energy (eV), e.g. 21.2 eV for He-I$\alpha$, 6-11 eV for laser ARPES
- $W$ = work function (eV), typically 4-5.5 eV — the minimum energy to remove an electron at $E_F$
- $E_B$ = binding energy relative to $E_F$ (the absolute value is used, so the sign convention of the band data does not matter)

A state can only be photoemitted if $E_{\mathrm{kin}} > 0$; diffpes clamps negative kinetic energies to zero (yielding $|\mathbf{k}| = 0$) so the forward model stays gradient-safe.

## Momentum Conservation and the $k_\parallel$ Mapping

The photon momentum is negligible at UV energies ($\sim 10^{-3}\,\text{Å}^{-1}$ at 21 eV, compared to Brillouin-zone dimensions of $\sim 1\,\text{Å}^{-1}$), so the electron's crystal momentum parallel to the surface is conserved up to a surface reciprocal lattice vector:

$$
\mathbf{k}_\parallel^{\mathrm{crystal}} = \mathbf{k}_\parallel^{\mathrm{vacuum}} + \mathbf{G}_\parallel
$$

In vacuum the photoelectron is a free electron, so its parallel momentum follows directly from the emission angle $\theta$:

$$
k_\parallel = \frac{\sqrt{2 m_e E_{\mathrm{kin}}}}{\hbar} \sin\theta
  \approx 0.5123 \, \sqrt{E_{\mathrm{kin}}\,[\mathrm{eV}]}\; \sin\theta
  \quad [\text{Å}^{-1}]
$$

The numerical prefactor comes from $\sqrt{2 m_e c^2}/(\hbar c) = \sqrt{2 \times 0.511 \times 10^6}/1973.27 \approx 0.5123$ in natural units — exactly the constants (`_ME_EV`, `_HBAR_C_EV_A`) used inside {mod}`diffpes.simul`:

```python
import jax.numpy as jnp

hv = 21.2         # He-I photon energy (eV)
w = 4.3           # work function (eV)
e_b = 0.3         # binding energy below E_F (eV)
theta_deg = 12.0  # emission angle from the surface normal

e_kin = hv - w - e_b
k_par = 0.5123 * jnp.sqrt(e_kin) * jnp.sin(jnp.deg2rad(theta_deg))
print(float(k_par))  # 0.434 1/Angstrom
```

At He-I energies the full angular window of $\pm 15°$ maps to roughly $\pm 0.5\,\text{Å}^{-1}$ — enough to cover the first Brillouin zone of most quasi-2D materials.

## Free-Electron Final State and the Inner Potential

The perpendicular momentum $k_z$ is *not* conserved across the surface. To assign a $k_z$ to a measured spectrum one conventionally assumes a **free-electron final state**: inside the solid the photoelectron behaves as a free electron whose energy is offset by the **inner potential** $V_0$ (the depth of the crystal's average potential below the vacuum level, typically 10-15 eV):

$$
k_z = \frac{\sqrt{2 m_e}}{\hbar} \sqrt{E_{\mathrm{kin}} \cos^2\theta + V_0}
$$

Scanning the photon energy scans $k_z$, which is how 3D band dispersions are mapped experimentally.

diffpes uses the free-electron final-state approximation when it needs the **magnitude** of the photoelectron wavevector — in the differentiable forward model `simulate_tb_radial` the radial matrix elements are evaluated at

$$
|\mathbf{k}| = \frac{\sqrt{2 m_e (h\nu - W - |E_B|)}}{\hbar}
$$

with the work function `work_function` as an explicit (and differentiable) parameter. The inner potential $V_0$ is not currently a model parameter: the projection-based `simulate_*` levels take band energies on a user-supplied k-path as ground truth, so no explicit $k_z$ reconstruction is performed. Keep $V_0$ in mind when comparing simulated k-paths against photon-energy-dependent experiments.

## Incidence Geometry in the API

Two distinct geometries appear in an ARPES setup, and it is worth keeping them separate:

1. **Photon incidence** — the direction the light comes from, which fixes the polarization vectors and hence the dipole matrix elements.
2. **Electron emission** — the direction the analyzer looks, which fixes $k_\parallel$ through the mapping above.

The diffpes simulation API parameterizes the **photon incidence** with two angles:

- `incident_theta` — polar angle of the photon beam from the surface normal, **in degrees** (default 45)
- `incident_phi` — azimuthal angle in the surface plane, **in degrees** (default 0)

The expanded wrappers convert these to radians before building a {class}`~diffpes.types.PolarizationConfig`. Internally, `photon_wavevector` constructs the unit propagation direction

$$
\hat{k}_{\mathrm{ph}} = (\sin\theta\cos\phi,\; \sin\theta\sin\phi,\; \cos\theta)
$$

and `build_polarization_vectors` derives the s/p polarization basis from it (see [Matrix Elements and Polarization](matrix-elements-and-polarization.md)). The electron-emission side is implicit: the k-points of the supplied band structure *are* the sampled $\mathbf{k}_\parallel$ values, so the emission-angle-to-momentum conversion is assumed to have been done when the k-path was chosen.

## What an $(E, k)$ Spectrum Is

The primary output of every diffpes simulation is an {class}`~diffpes.types.ArpesSpectrum` PyTree with two fields:

- `intensity` — a `(K, E)` array: photoemission intensity for `K` k-points along the sampled path and `E` energy grid points
- `energy_axis` — the `(E,)` energy grid in eV, spanning `[min(bands) - 1, max(bands) + 1]` by default

Schematically, each level of the simulation ladder evaluates

$$
I(\mathbf{k}, E) \;\propto\; \sum_{b} \, w_b(\mathbf{k}) \; f(E_b(\mathbf{k}); E_F, T)\; \mathcal{L}\!\left(E - E_b(\mathbf{k}); \sigma, \gamma\right)
$$

where $w_b$ is an orbital/matrix-element weight, $f$ is the Fermi-Dirac occupation, and $\mathcal{L}$ is a Gaussian or Voigt lineshape. Columns of `intensity` at fixed $\mathbf{k}$ are **energy distribution curves** (EDCs); rows at fixed $E$ are **momentum distribution curves** (MDCs).

## Worked Example

The following builds a synthetic two-band cosine dispersion (the kind of eigenvalue array you would otherwise read from a VASP `EIGENVAL` file) and simulates its ARPES map:

```python
import jax.numpy as jnp
import diffpes

# Synthetic band structure along a 1D k-path (units: eV)
n_k, n_b, n_atoms = 61, 2, 1
k_frac = jnp.linspace(-0.5, 0.5, n_k)
band_1 = -1.2 + 0.9 * jnp.cos(2.0 * jnp.pi * k_frac)
band_2 = -2.6 + 0.4 * jnp.cos(2.0 * jnp.pi * k_frac + jnp.pi)
eigenbands = jnp.stack([band_1, band_2], axis=1)        # (K, B)

# Uniform orbital projections: (K, B, A, 9) in VASP orbital order
# [s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2]
surface_orb = jnp.ones((n_k, n_b, n_atoms, 9)) / 9.0

spectrum = diffpes.simul.simulate_expanded(
    level="novice",
    eigenbands=eigenbands,
    surface_orb=surface_orb,
    ef=0.0,              # Fermi level (eV)
    sigma=0.03,          # Gaussian width (eV)
    gamma=0.05,          # Lorentzian half-width (eV)
    fidelity=2000,       # energy-axis points
    temperature=20.0,    # Kelvin
    photon_energy=21.2,  # eV (He-I)
)

print(spectrum.intensity.shape)    # (61, 2000)
print(spectrum.energy_axis.shape)  # (2000,)

# An EDC at the zone center and an MDC at E = -0.5 eV:
edc = spectrum.intensity[n_k // 2]
idx = jnp.argmin(jnp.abs(spectrum.energy_axis - (-0.5)))
mdc = spectrum.intensity[:, idx]
```

Band 1 disperses from $-2.1$ eV at the zone boundary up to $-0.3$ eV at $\Gamma$; both bands sit below $E_F = 0$, so the Fermi-Dirac factor at 20 K leaves them fully visible while cutting off any weight above zero energy.

## Where Kinematics Enters Each Simulation Path

| Simulation path | How geometry/kinematics is used |
|-----------------|--------------------------------|
| `simulate_novice` … `simulate_basicplus` | None beyond the k-path itself; intensity weights depend only on photon energy |
| `simulate_advanced`, `simulate_expert`, `simulate_soc` | `incident_theta`/`incident_phi` fix the polarization basis; the SOC level also forms $\mathbf{S}\cdot\hat{k}_{\mathrm{ph}}$ |
| `simulate_tb_radial` | Full kinematics: $|\mathbf{k}|$ from $h\nu$, $W$, and $E_B$ via the free-electron final state; the crystal k-direction is rescaled to the photoelectron momentum for the matrix-element evaluation |

## Further Reading

- [Simulation Levels](simulation-levels.md) — what physics each rung of the six-level ladder adds
- [Matrix Elements and Polarization](matrix-elements-and-polarization.md) — how the incidence angles turn into polarization vectors and dipole weights
- [Spectral Broadening and Self-Energy](spectral-broadening-and-self-energy.md) — the lineshape $\mathcal{L}$ and its physical content
- API reference: {doc}`../api/simul`, {doc}`../api/types`
