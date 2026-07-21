# Matrix Elements and Polarization

Photoemission intensity contains more than a band structure and a lineshape.
A dipole matrix element weights each transition. This element depends on the
initial orbital, the light geometry, and the outgoing photoelectron momentum.
diffpes provides two fidelity levels. The simulation ladder uses a low-cost
orbital-direction heuristic. `simulate_tb_radial` uses the full partial-wave
engine from {mod}`diffpes.maths` and {mod}`diffpes.radial`. This guide describes
both levels and states their approximations.

## Dipole Selection Rules

Within the dipole approximation the photoemission matrix element is

$$
M_{fi} = \langle \psi_f | \, \hat{e} \cdot \mathbf{r} \, | \psi_i \rangle
$$

Expanding the final state in partial waves and writing the initial state as $R_{nl}(r)\, Y_l^m(\hat{r})$, the angular integral enforces the selection rules

$$
l' = l \pm 1, \qquad m' = m + q, \quad q \in \{-1, 0, +1\}
$$

The index $q$ identifies the spherical polarization components. The value
$q = 0$ corresponds to $z$, and $q = \pm 1$ corresponds to $x,y$. A p initial
state reaches s and d final waves. A d initial state reaches p and f waves.
The Gaunt table encodes these rules.

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

The implementation applies the Condon-Shortley phase. This choice makes the
values consistent with the real-to-complex transform in the Gaunt table.
`real_spherical_harmonics_all(l_max, theta, phi)` stacks all harmonics through
`l_max`. It uses the standard $l^2 + l + m$ ordering.

```python
import jax.numpy as jnp
from diffpes.maths import real_spherical_harmonic

theta, phi = jnp.asarray(0.7), jnp.asarray(0.3)
y20 = real_spherical_harmonic(2, 0, theta, phi)   # d_z2 angular shape
print(float(y20))  # 0.2381...
```

## Gaunt Coefficients

The angular part of the dipole integral is a **Gaunt coefficient**. This
coefficient is the integral of three real spherical harmonics:

$$
G(l, m; l', m'; q) = \int Y_{l'}^{m'}(\hat{r})\; Y_1^{q}(\hat{r})\; Y_l^{m}(\hat{r})\; d\Omega
$$

At import time, diffpes computes all dipole-allowed real Gaunt coefficients
through `L_MAX = 4`. The calculation uses Wigner-3j symbols and the
real-to-complex basis transform. diffpes stores the results in the dense
`GAUNT_TABLE` array. The index order is
`GAUNT_TABLE[l, m + L_MAX, q + 1, lp, mp + L_MAX]`. Use `gaunt_lookup` for
direct access:

```python
from diffpes.maths import gaunt_lookup

# p_z (l=1, m=0) -> d_z2 (l'=2, m'=0) via the q=0 (z) dipole component
print(gaunt_lookup(l=1, m=0, q=0, lp=2, mp=0))  # 0.2523...
# Forbidden by the selection rules -> exactly zero
print(gaunt_lookup(l=1, m=0, q=0, lp=2, mp=1))  # 0.0
```

The frozen table gives JIT-compiled code O(1) array indexing. Table lookups do
not carry gradients.

## Radial Integrals

The radial part of the matrix element couples the initial radial wavefunction to a final-state spherical Bessel wave:

$$
B^{l'}(k) = i^{\,l'} \int_0^\infty R(r)\; r^3\; j_{l'}(kr)\; dr
$$

The volume element $r^2\,dr$ and the dipole operator $r$ produce the $r^3$
factor. The final-state plane-wave expansion produces the $i^{l'}$ phase.
{mod}`diffpes.radial` provides these functions:

- `spherical_bessel_jl(order, x)` computes $j_l(x)$ by upward recurrence. A
  Taylor-limit guard handles small arguments.
- `slater_radial(r, n, zeta)` computes a normalized, node-free Slater orbital.
  The JAX scalar $\zeta$ remains differentiable.
- `hydrogenic_radial(r, n, angular_momentum, z_eff)` computes the exact
  hydrogenic $R_{nl}(r)$. An associated Laguerre recurrence creates the
  $n-l-1$ radial nodes.
- `radial_integral(k, r, radial_values, l_prime)` applies trapezoidal
  quadrature on a fixed grid. Radial coordinates use atomic units (Bohr).

```python
import jax.numpy as jnp
from diffpes.radial import radial_integral, slater_radial

r_grid = jnp.linspace(1e-6, 40.0, 4000)          # Bohr
radial_2p = slater_radial(r_grid, 2, jnp.asarray(1.6))
b2 = radial_integral(jnp.asarray(1.1), r_grid, radial_2p, 2)
print(b2)  # (-1.7201+0j): i^2 phase makes the l'=2 channel real and negative
```

JAX traces $\zeta$ or $Z_{\mathrm{eff}}$. Therefore, spectrum gradients can
describe changes in the radial wavefunction. These gradients support inverse
fitting of the orbital extent.

## Polarization Vectors and `build_efield`

A {class}`~diffpes.types.PolarizationConfig` describes the light geometry. The
configuration stores angles in **radians**. The `simulate_*_expanded` wrappers
accept degrees. `build_polarization_vectors` constructs the orthonormal basis
$\{\hat{k}_{\mathrm{ph}}, \hat{e}_s, \hat{e}_p\}$ from the incidence angles.
Here, $\hat{e}_s$ is perpendicular to the incidence plane. The vector
$\hat{e}_p$ lies in that plane and remains perpendicular to
$\hat{k}_{\mathrm{ph}}$. `build_efield` returns the requested complex field:

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

`photon_wavevector(theta, phi)` returns the unit propagation direction. The SOC
level uses this vector in the $\mathbf{S}\cdot\hat{k}_{\mathrm{ph}}$
modulation.

## The Orbital-Direction Model (`ORBITAL_DIRS_NORMALIZED`)

The `advanced`, `expert`, and `soc` levels use a low-cost approximation for the
full matrix element. The model assigns a fixed unit direction $\hat{d}_o$ to
each VASP orbital. `ORBITAL_DIRS_NORMALIZED` stores these nine directions with
shape `(9, 3)`. For example, $p_x \to \hat{x}$ and
$d_{xz} \to (\hat{x}+\hat{z})/\sqrt{2}$. The s orbital maps to $\mathbf{0}$.
The polarization weight is

$$
M_o = \left| \hat{e} \cdot \hat{d}_o \right|^2
$$

```python
from diffpes.simul import ORBITAL_DIRS_NORMALIZED, dipole_matrix_elements

weights = dipole_matrix_elements(efield)   # p-pol at 45 degrees, phi = 0
# [s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2]
# [0., 0., 0.5, 0.5, 0.25, 0.25, 0.5, ~0., 0.25]
```

The model captures qualitative selection rules. At $\phi = 0$, p-polarized
light excites $p_z$ and $p_x$ equally. The $d_{xz}$ channel vanishes because
its direction is perpendicular to $\hat{e}_p$. The model omits momentum
dependence, radial physics, orbital interference, and s-orbital emission.
Use this model only as a heuristic selection-rule filter.

## The Full Assembly: `dipole_matrix_element_single`

The complete partial-wave matrix element for an orbital $(n, l, m)$ combines all four ingredients:

$$
M(\mathbf{k}) = \sum_{q \in \{-1,0,+1\}} \hat{e}_q \sum_{l' = l \pm 1}
    B^{l'}(|\mathbf{k}|)\; G(l, m; l', m + q; q)\; Y_{l'}^{m+q}(\hat{k})
$$

The function evaluates this expression at the **photoelectron** wavevector
$\mathbf{k}$. Free-electron kinematics sets its magnitude, and the detector
sets its direction:

```python
import jax.numpy as jnp
from diffpes.maths import dipole_matrix_element_single

k_vec = jnp.array([0.3, 0.0, 1.1])   # photoelectron wavevector (1/Angstrom)
m_pz = dipole_matrix_element_single(
    k_vec, r_grid, radial_2p, 1, 0, efield,   # l=1, m=0 (p_z), p-pol field
)
print(abs(m_pz) ** 2)  # 0.0383...
```

`dipole_intensity_orbital` returns $|M|^2$ directly.
`dipole_intensities_all_orbitals` scans a `SlaterParams` basis. The
`simulate_tb_radial` model weights $M_o$ with tight-binding eigenvector
coefficients. It computes $M_{kb} = \sum_o c_{kb,o} M_o$ before squaring.
Therefore, this level includes interference between orbital channels:

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

**The engine uses these approximations:**

- The final state is a plane wave.
- The model omits scattering from the crystal potential and the surface
  transmission factor.
- The default radial functions are single-zeta Slater orbitals.
- The model rescales the crystal momentum direction to the photoelectron
  momentum. It does not model surface refraction.

## Orbital Angular Momentum Output

For dichroism analysis, `compute_oam` computes the expected $L_z$ value for
each k-point, band, and atom. It uses
$\mathrm{OAM}_z = \sum_m m\, |c_m|^2$. The final axis contains the p, d, and
total contributions:

```python
import jax.numpy as jnp
from diffpes.simul import compute_oam

proj = jnp.ones((61, 2, 1, 9)) / 9.0        # (K, B, A, 9) orbital projections
oam = compute_oam(proj)                      # -> (K, B, A, 3): [p, d, total]
```

The s channel carries $m = 0$ and never contributes.

## Further Reading

- [Simulation Levels](simulation-levels.md) locates both matrix-element models
  in the simulation levels.
- [ARPES Geometry and Kinematics](arpes-geometry-and-kinematics.md) derives
  $|\mathbf{k}|$ from photon energy and the work function.
- [Spectral Broadening and Self-Energy](spectral-broadening-and-self-energy.md)
  describes the lineshape after matrix-element weighting.
- API reference: {doc}`../api/maths`, {doc}`../api/radial`, {doc}`../api/simul`
