# Matrix Elements and Polarization

The photoemission intensity is not simply the band structure convolved with a lineshape: each transition is weighted by a dipole matrix element that depends on the orbital character of the initial state, the direction and polarization of the light, and the momentum of the outgoing photoelectron. diffpes implements this at two fidelities — a cheap orbital-direction heuristic used by the simulation-level ladder, and a full partial-wave dipole engine ({mod}`diffpes.maths` + {mod}`diffpes.radial`) used by `simulate_tb_radial`. This guide covers both, and states their approximations explicitly.

## Dipole Selection Rules

Within the dipole approximation the photoemission matrix element is

$$
M_{fi} = \langle \psi_f | \, \hat{e} \cdot \mathbf{r} \, | \psi_i \rangle
$$

Expanding the final state in partial waves and writing the initial state as $R_{nl}(r)\, Y_l^m(\hat{r})$, the angular integral enforces the selection rules

$$
l' = l \pm 1, \qquad m' = m + q, \quad q \in \{-1, 0, +1\}
$$

where $q$ indexes the spherical components of the polarization vector ($q = 0 \sim z$, $q = \pm 1 \sim x, y$). A p initial state therefore reaches s and d final waves; a d state reaches p and f waves. These rules are baked into the structure of the Gaunt table below.

## Real Spherical Harmonics

diffpes works in the **real** spherical harmonic basis (matching VASP's orbital-projected output). `real_spherical_harmonic` in {mod}`diffpes.maths` evaluates

$$
Y_l^m(\theta, \varphi) =
\begin{cases}
\sqrt{2}\, N_l^{|m|}\, P_l^{|m|}(\cos\theta)\, \cos(m\varphi) & m > 0 \\[2pt]
N_l^0\, P_l^0(\cos\theta) & m = 0 \\[2pt]
(-1)^{|m|}\sqrt{2}\, N_l^{|m|}\, P_l^{|m|}(\cos\theta)\, \sin(|m|\varphi) & m < 0
\end{cases}
$$

with the Condon-Shortley phase handled so that the real-harmonic values are consistent with the real-to-complex transform used in the Gaunt table. `real_spherical_harmonics_all(l_max, theta, phi)` stacks every harmonic up to `l_max` in the standard $l^2 + l + m$ ordering.

```python
import jax.numpy as jnp
from diffpes.maths import real_spherical_harmonic

theta, phi = jnp.asarray(0.7), jnp.asarray(0.3)
y20 = real_spherical_harmonic(2, 0, theta, phi)   # d_z2 angular shape
print(float(y20))  # 0.2381...
```

## Gaunt Coefficients

The angular part of the dipole integral is a **Gaunt coefficient** — the integral of three (real) spherical harmonics:

$$
G(l, m; l', m'; q) = \int Y_{l'}^{m'}(\hat{r})\; Y_1^{q}(\hat{r})\; Y_l^{m}(\hat{r})\; d\Omega
$$

diffpes precomputes all dipole-allowed real Gaunt coefficients up to `L_MAX = 4` at import time (via Wigner-3j symbols and the real-to-complex basis transform) and stores them in the dense array `GAUNT_TABLE`, indexed as `GAUNT_TABLE[l, m + L_MAX, q + 1, lp, mp + L_MAX]`. For scripting there is a convenience accessor:

```python
from diffpes.maths import gaunt_lookup

# p_z (l=1, m=0) -> d_z2 (l'=2, m'=0) via the q=0 (z) dipole component
print(gaunt_lookup(l=1, m=0, q=0, lp=2, mp=0))  # 0.2523...
# Forbidden by the selection rules -> exactly zero
print(gaunt_lookup(l=1, m=0, q=0, lp=2, mp=1))  # 0.0
```

Because the table is a frozen constant, lookups inside JIT-compiled code are O(1) array indexing and carry no gradient.

## Radial Integrals

The radial part of the matrix element couples the initial radial wavefunction to a final-state spherical Bessel wave:

$$
B^{l'}(k) = i^{\,l'} \int_0^\infty R(r)\; r^3\; j_{l'}(kr)\; dr
$$

The $r^3$ arises from the volume element $r^2\,dr$ times the dipole operator $r$; the phase $i^{l'}$ comes from the plane-wave expansion of the final state. {mod}`diffpes.radial` provides every piece:

- `spherical_bessel_jl(order, x)` — $j_l(x)$ by upward recurrence with a Taylor-limit guard at small argument
- `slater_radial(r, n, zeta)` — normalized Slater-type orbital $R(r) = N r^{n-1} e^{-\zeta r}$ (node-free; $\zeta$ is a differentiable JAX scalar)
- `hydrogenic_radial(r, n, angular_momentum, z_eff)` — exact hydrogenic $R_{nl}(r)$ with its $n - l - 1$ radial nodes via associated Laguerre recurrence
- `radial_integral(k, r, radial_values, l_prime)` — trapezoidal quadrature of the integral above on a fixed grid (radial coordinates in atomic units/Bohr)

```python
import jax.numpy as jnp
from diffpes.radial import radial_integral, slater_radial

r_grid = jnp.linspace(1e-6, 40.0, 4000)          # Bohr
radial_2p = slater_radial(r_grid, 2, jnp.asarray(1.6))
b2 = radial_integral(jnp.asarray(1.1), r_grid, radial_2p, 2)
print(b2)  # (-1.7201+0j): i^2 phase makes the l'=2 channel real and negative
```

Because $\zeta$ (or $Z_{\mathrm{eff}}$) is traced by JAX, gradients of a simulated spectrum with respect to the radial-wavefunction shape are available — the basis for inverse fitting of orbital extent.

## Polarization Vectors and `build_efield`

The light geometry is described by a {class}`~diffpes.types.PolarizationConfig` (angles in **radians** internally; the `simulate_*_expanded` wrappers accept degrees). From the incidence angles, `build_polarization_vectors` constructs the orthonormal basis $\{\hat{k}_{\mathrm{ph}}, \hat{e}_s, \hat{e}_p\}$: $\hat{e}_s \perp$ the incidence plane, $\hat{e}_p$ in-plane and $\perp \hat{k}_{\mathrm{ph}}$. `build_efield` then returns the complex field vector for the requested polarization type:

| `polarization_type` | Electric field |
|---------------------|----------------|
| `"lvp"` | $\hat{e}_s$ (linear vertical / s-pol) |
| `"lhp"` | $\hat{e}_p$ (linear horizontal / p-pol) |
| `"lap"` | $\cos\alpha\, \hat{e}_s + \sin\alpha\, \hat{e}_p$ with $\alpha$ = `polarization_angle` (radians) |
| `"rcp"` / `"lcp"` | $(\hat{e}_s \pm i \hat{e}_p)/\sqrt{2}$ (circular) |
| `"unpolarized"` | handled by the callers as the average of the s- and p-pol intensities |

```python
import jax.numpy as jnp
from diffpes.simul import build_efield
from diffpes.types import make_polarization_config

config = make_polarization_config(
    theta=jnp.deg2rad(45.0),   # radians at this API level
    phi=0.0,
    polarization_type="lhp",   # p-polarized
)
efield = build_efield(config)
print(efield)  # [-0.707+0j, 0+0j, 0.707+0j] - in-plane x and out-of-plane z
```

`photon_wavevector(theta, phi)` returns the unit propagation direction itself, used by the SOC level to form the $\mathbf{S}\cdot\hat{k}_{\mathrm{ph}}$ modulation.

## The Orbital-Direction Model (`ORBITAL_DIRS_NORMALIZED`)

The `advanced`/`expert`/`soc` levels of the simulation ladder use a deliberately cheap stand-in for the full matrix element: each of the 9 VASP orbitals is assigned a fixed unit direction $\hat{d}_o$ (`ORBITAL_DIRS_NORMALIZED`, shape `(9, 3)`; e.g. $p_x \to \hat{x}$, $d_{xz} \to (\hat{x}+\hat{z})/\sqrt{2}$, s $\to \mathbf{0}$), and the polarization weight is

$$
M_o = \left| \hat{e} \cdot \hat{d}_o \right|^2
$$

```python
from diffpes.simul import ORBITAL_DIRS_NORMALIZED, dipole_matrix_elements

weights = dipole_matrix_elements(efield)   # p-pol at 45 degrees, phi = 0
# [s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2]
# [0., 0., 0.5, 0.5, 0.25, 0.25, 0.5, ~0., 0.25]
```

The physics it captures: p-polarized light at $\phi = 0$ excites $p_z$ and $p_x$ equally, and the $d_{xz}$ channel vanishes because its direction $(\hat{x}+\hat{z})/\sqrt{2}$ is orthogonal to $\hat{e}_p = (-\hat{x}+\hat{z})/\sqrt{2}$. What it does *not* capture: k-dependence, radial physics, interference between orbital channels, and any s-orbital emission ($\hat{d}_s = \mathbf{0}$ gives $M_s = 0$ identically). Treat it as a heuristic selection-rule filter slated for replacement by the engine below.

## The Full Assembly: `dipole_matrix_element_single`

The complete partial-wave matrix element for an orbital $(n, l, m)$ combines all four ingredients:

$$
M(\mathbf{k}) = \sum_{q \in \{-1,0,+1\}} \hat{e}_q \sum_{l' = l \pm 1}
    B^{l'}(|\mathbf{k}|)\; G(l, m; l', m + q; q)\; Y_{l'}^{m+q}(\hat{k})
$$

evaluated at the **photoelectron** wavevector $\mathbf{k}$ (magnitude from free-electron kinematics, direction $\hat{k}$ toward the detector):

```python
import jax.numpy as jnp
from diffpes.maths import dipole_matrix_element_single

k_vec = jnp.array([0.3, 0.0, 1.1])   # photoelectron wavevector (1/Angstrom)
m_pz = dipole_matrix_element_single(
    k_vec, r_grid, radial_2p, 1, 0, efield,   # l=1, m=0 (p_z), p-pol field
)
print(abs(m_pz) ** 2)  # 0.0383...
```

`dipole_intensity_orbital` returns $|M|^2$ directly, and `dipole_intensities_all_orbitals` loops over a `SlaterParams` basis. The end-to-end forward model `simulate_tb_radial` sums $M_o$ over orbitals weighted by tight-binding eigenvector coefficients, $M_{kb} = \sum_o c_{kb,o} M_o$, before squaring — so interference between orbital channels *is* included at this tier:

```python
import jax.numpy as jnp
import diffpes
from diffpes.types import (
    make_diagonalized_bands, make_orbital_basis,
    make_simulation_params, make_slater_params,
)

basis = make_orbital_basis(
    n_values=(2, 2, 2), l_values=(1, 1, 1), m_values=(-1, 0, 1),
    labels=("py", "pz", "px"),
)
slater = make_slater_params(zeta=jnp.array([1.6, 1.6, 1.6]), orbital_basis=basis)

n_k, n_b = 61, 2
k_frac = jnp.linspace(-0.5, 0.5, n_k)
eigenbands = jnp.stack(
    [-1.2 + 0.9 * jnp.cos(2 * jnp.pi * k_frac),
     -2.6 + 0.4 * jnp.cos(2 * jnp.pi * k_frac + jnp.pi)], axis=1)
eigvecs = jnp.tile(jnp.eye(3, dtype=jnp.complex128)[jnp.newaxis, :n_b, :], (n_k, 1, 1))
kpts = jnp.stack([k_frac, jnp.zeros(n_k), jnp.zeros(n_k)], axis=1)

diag = make_diagonalized_bands(eigenbands, eigvecs, kpts, fermi_energy=0.0)
params = make_simulation_params(energy_min=-3.5, energy_max=0.5, fidelity=800,
                                sigma=0.03, gamma=0.05, photon_energy=21.2)
spec = diffpes.simul.simulate_tb_radial(
    diag_bands=diag, slater_params=slater, params=params,
    pol_config=config, work_function=4.3,
)
print(spec.intensity.shape)  # (61, 800)
```

**Stated approximations of this engine:** the final state is a plane wave (no scattering off the crystal potential, no surface transmission factor), the radial functions are single-zeta Slater orbitals unless you supply better ones, and the crystal k-direction is rescaled to the photoelectron momentum rather than refracted through the surface.

## Orbital Angular Momentum Output

For dichroism-style analysis, `compute_oam` converts orbital projections into the expectation value of $L_z$ per (k-point, band, atom), $\mathrm{OAM}_z = \sum_m m\, |c_m|^2$, returning p-, d-, and total contributions stacked in the last axis:

```python
import jax.numpy as jnp
from diffpes.simul import compute_oam

proj = jnp.ones((61, 2, 1, 9)) / 9.0        # (K, B, A, 9) orbital projections
oam = compute_oam(proj)                      # -> (K, B, A, 3): [p, d, total]
```

The s channel carries $m = 0$ and never contributes.

## Further Reading

- [Simulation Levels](simulation-levels.md) — where the heuristic vs. full engine sit in the level ladder
- [ARPES Geometry and Kinematics](arpes-geometry-and-kinematics.md) — how $|\mathbf{k}|$ is derived from photon energy and work function
- [Spectral Broadening and Self-Energy](spectral-broadening-and-self-energy.md) — the lineshape applied after the matrix-element weighting
- API reference: {doc}`../api/maths`, {doc}`../api/radial`, {doc}`../api/simul`
