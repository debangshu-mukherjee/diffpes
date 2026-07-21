# ARPES Geometry and Kinematics

Angle-Resolved PhotoEmission Spectroscopy (ARPES) measures electron kinetic
energy and emission angle. Monochromatic light ejects these electrons from a
solid. Energy and crystal momentum conservation let researchers invert the
measured $(E_{\mathrm{kin}}, \theta, \phi)$ distribution into the occupied band
structure $E(\mathbf{k})$. This guide describes the diffpes kinematic relations
and API incidence geometry. It also explains how diffpes produces a simulated
$(E, k)$ intensity map.

## Unit and Sign Conventions

diffpes pins the following conventions throughout the codebase:

| Quantity | Unit | Notes |
|----------|------|-------|
| Energies ($h\nu$, $E_B$, $E_F$, $\sigma$, $\gamma$) | eV | Band eigenvalues use absolute eV; callers pass the Fermi level `ef` separately |
| Wavevectors $\mathbf{k}$ | $\text{Å}^{-1}$ | The reciprocal lattice includes the $2\pi$ prefactor |
| Angles at API boundaries | degrees | The `simulate_*_expanded` family accepts `incident_theta` and `incident_phi` |
| Angles internally | radians | `PolarizationConfig.theta/phi` and `make_polarization_config` use radians |
| Temperature | Kelvin | The Fermi-Dirac factor uses the temperature |
| Precision | float64 | Importing `diffpes` enables `jax_enable_x64` |

diffpes measures polar angles $\theta$ from the surface normal, or $z$-axis.
It measures azimuthal angles $\phi$ in the surface plane from the $x$-axis.

## Energy Conservation and the Work Function

In the three-step photoemission model, a bound electron absorbs a photon of
energy $h\nu$. The electron then travels to the surface and crosses the surface
potential barrier. Its kinetic energy in vacuum is

$$
E_{\mathrm{kin}} = h\nu - W - |E_B|
$$

where:

- $h\nu$ denotes photon energy in eV. He-I$\alpha$ uses 21.2 eV, while laser
  ARPES typically uses 6-11 eV.
- $W$ denotes the work function in eV. Materials typically have values from
  4-5.5 eV. This energy removes an electron at $E_F$.
- $E_B$ denotes binding energy relative to $E_F$. The equation uses its
  absolute value. Therefore, the sign convention of the band data does not
  matter.

Photoemission requires $E_{\mathrm{kin}} > 0$. diffpes clamps negative kinetic
energies to zero, which gives $|\mathbf{k}| = 0$. This guard keeps the forward
model gradient-safe.

## Momentum Conservation and the $k_\parallel$ Mapping

At 21 eV, the photon momentum is approximately
$10^{-3}\,\text{Å}^{-1}$. Brillouin-zone dimensions are approximately
$1\,\text{Å}^{-1}$. Therefore, UV experiments can neglect the photon momentum.
The electron conserves crystal momentum parallel to the surface, up to a
surface reciprocal lattice vector:

$$
\mathbf{k}_\parallel^{\mathrm{crystal}} = \mathbf{k}_\parallel^{\mathrm{vacuum}} + \mathbf{G}_\parallel
$$

In vacuum the photoelectron is a free electron, so its parallel momentum follows directly from the emission angle $\theta$:

$$
k_\parallel = \frac{\sqrt{2 m_e E_{\mathrm{kin}}}}{\hbar} \sin\theta
  \approx 0.5123 \, \sqrt{E_{\mathrm{kin}}\,[\mathrm{eV}]}\; \sin\theta
  \quad [\text{Å}^{-1}]
$$

The numerical prefactor follows from
$\sqrt{2 m_e c^2}/(\hbar c) = \sqrt{2 \times 0.511 \times 10^6}/1973.27
\approx 0.5123$ in natural units. {mod}`diffpes.simul` uses the corresponding
`_ME_EV` and `_HBAR_C_EV_A` constants:

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

At He-I energies, an angular window of $\pm 15°$ maps to approximately
$\pm 0.5\,\text{Å}^{-1}$. This range covers the first Brillouin zone of most
quasi-2D materials.

## Free-Electron Final State and the Inner Potential

The surface does *not* conserve the perpendicular momentum $k_z$. Researchers
usually assign $k_z$ with a **free-electron final state**. This model treats the
photoelectron inside the solid as a free electron. The **inner potential**
$V_0$ offsets its energy. This potential measures the average crystal
potential below the vacuum level and typically spans 10-15 eV:

$$
k_z = \frac{\sqrt{2 m_e}}{\hbar} \sqrt{E_{\mathrm{kin}} \cos^2\theta + V_0}
$$

Photon-energy scans vary $k_z$. Researchers use this variation to map 3D band
dispersions.

diffpes uses the free-electron final-state approximation to compute the
photoelectron wavevector **magnitude**. The differentiable
`simulate_tb_radial` model evaluates the radial matrix elements at

$$
|\mathbf{k}| = \frac{\sqrt{2 m_e (h\nu - W - |E_B|)}}{\hbar}
$$

The explicit `work_function` parameter remains differentiable. The current
model does not include the inner potential $V_0$. The projection-based
`simulate_*` levels use band energies on a user-supplied k-path as fixed input
data. Therefore, these levels do not reconstruct $k_z$. Consider $V_0$ when
you compare simulated k-paths with photon-energy-dependent experiments.

## Incidence Geometry in the API

An ARPES setup contains two distinct geometries:

1. **Photon incidence** specifies the light direction. This direction fixes
   the polarization vectors and dipole matrix elements.
2. **Electron emission** specifies the analyzer direction. This direction
   fixes $k_\parallel$ through the preceding mapping.

The diffpes simulation API parameterizes the **photon incidence** with two angles:

- `incident_theta` specifies the photon-beam polar angle from the surface
  normal, **in degrees**. Default 45.
- `incident_phi` specifies the surface-plane azimuthal angle, **in degrees**.
  Default 0.

The expanded wrappers convert these to radians before building a {class}`~diffpes.types.PolarizationConfig`. Internally, `photon_wavevector` constructs the unit propagation direction

$$
\hat{k}_{\mathrm{ph}} = (\sin\theta\cos\phi,\; \sin\theta\sin\phi,\; \cos\theta)
$$

`build_polarization_vectors` derives the s/p polarization basis from this
direction. See [Matrix Elements and Polarization](matrix-elements-and-polarization.md).
The API represents electron emission implicitly. The supplied band-structure
k-points define the sampled $\mathbf{k}_\parallel$ values. Users perform the
emission-angle conversion when they select the k-path.

## What an $(E, k)$ Spectrum Is

The primary output of every diffpes simulation is an {class}`~diffpes.types.ArpesSpectrum` PyTree with two fields:

- `intensity`: This `(K, E)` array stores the photoemission intensity for `K`
  k-points and `E` energy-axis points.
- `energy_axis`: This `(E,)` array stores the energy grid in eV. By default,
  it spans `[min(bands) - 1, max(bands) + 1]`.

Schematically, each simulation level evaluates

$$
I(\mathbf{k}, E) \;\propto\; \sum_{b} \, w_b(\mathbf{k}) \; f(E_b(\mathbf{k}); E_F, T)\; \mathcal{L}\!\left(E - E_b(\mathbf{k}); \sigma, \gamma\right)
$$

Here, $w_b$ is an orbital or matrix-element weight. $f$ is the Fermi-Dirac
occupation, and $\mathcal{L}$ is a Gaussian or Voigt lineshape. Each
`intensity` column at fixed $\mathbf{k}$ forms an **energy distribution curve**
(EDC). Each row at fixed $E$ forms a **momentum distribution curve** (MDC).

## Worked Example

The following example builds a synthetic two-band cosine dispersion and
simulates its ARPES map. A VASP `EIGENVAL` file can provide the equivalent
eigenvalue array:

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

Band 1 disperses from $-2.1$ eV at the zone boundary to $-0.3$ eV at
$\Gamma$. Both bands remain below $E_F = 0$. Therefore, the Fermi-Dirac factor
at 20 K preserves their weight and removes weight above zero energy.

## Kinematics by Simulation Path

| Simulation path | Geometry and kinematics behavior |
|-----------------|--------------------------------|
| `simulate_novice` … `simulate_basicplus` | The k-path supplies the geometry; photon energy controls the intensity weights |
| `simulate_advanced`, `simulate_expert`, `simulate_soc` | The incidence angles fix the polarization basis; the SOC level also forms $\mathbf{S}\cdot\hat{k}_{\mathrm{ph}}$ |
| `simulate_tb_radial` | The free-electron final state computes $|\mathbf{k}|$ from $h\nu$, $W$, and $E_B$; the model rescales the crystal direction for the matrix-element calculation |

## Further Reading

- [Simulation Levels](simulation-levels.md) — what physics each simulation level adds
- [Matrix Elements and Polarization](matrix-elements-and-polarization.md) — how the incidence angles turn into polarization vectors and dipole weights
- [Spectral Broadening and Self-Energy](spectral-broadening-and-self-energy.md) — the lineshape $\mathcal{L}$ and its physical content
- API reference: {doc}`../api/simul`, {doc}`../api/types`
