# Spectral Broadening and Self-Energy

An ideal band structure is a set of infinitely sharp delta functions $\delta(E - E_b(\mathbf{k}))$. A measured ARPES spectrum is not: instrumental resolution, finite quasiparticle lifetime, thermal occupation, and finite angular acceptance all reshape the signal. diffpes models each of these with a dedicated, differentiable primitive in {mod}`diffpes.simul`. This guide derives the profiles, pins their width conventions, and shows how an energy-dependent self-energy changes the lineshape.

All broadening parameters are in eV (energies), $\text{Å}^{-1}$ (momenta), and Kelvin (temperature); everything runs in float64.

## Gaussian Broadening: Instrumental Resolution

The energy resolution of beamline plus analyzer is well approximated by a Gaussian. `gaussian(energy_range, center, sigma)` evaluates the unit-area profile

$$
G(E; E_0, \sigma) = \frac{1}{\sqrt{2\pi}\,\sigma}
    \exp\!\left(-\frac{(E - E_0)^2}{2\sigma^2}\right)
$$

**Width convention:** `sigma` is the standard deviation, not the FWHM. The conversion is

$$
\mathrm{FWHM} = 2\sqrt{2\ln 2}\;\sigma \approx 2.355\,\sigma
$$

so a quoted "10 meV resolution" (FWHM) corresponds to `sigma = 0.00425`.

```python
import jax.numpy as jnp
from diffpes.simul import gaussian

energy_axis = jnp.linspace(-1.0, 1.0, 2001)
g = gaussian(energy_axis, center=0.0, sigma=0.02)
print(float(g.max()))                       # 19.947 = 1 / (sqrt(2 pi) * 0.02)
print(float(jnp.trapezoid(g, energy_axis)))  # ~1.0 (unit area)
```

Convolving each band's delta function with $G$ is exactly what the Gaussian-only simulation levels (`basic`, `basicplus`, `advanced`) do.

## Lorentzian Lifetime and the Pseudo-Voigt Profile

A quasiparticle with finite lifetime $\tau$ has a Lorentzian spectral function of half-width $\gamma = \hbar / (2\tau)$. The physical lineshape is therefore a **Voigt profile** — the convolution of the intrinsic Lorentzian with the instrumental Gaussian. A true Voigt has no closed form, so `voigt(energy_range, center, sigma, gamma)` uses the pseudo-Voigt approximation of Thompson, Cox & Hastings (1987), accurate to better than 1%:

$$
V(E) \approx \eta\, L(E; \gamma_V) + (1 - \eta)\, G(E; \sigma_V)
$$

where the effective Voigt FWHM $f_V$ is computed from the component FWHMs $f_G = 2\sqrt{2\ln 2}\,\sigma$ and $f_L = 2\gamma$ via the empirical quintic

$$
f_V = \left( f_G^5 + 2.69269 f_G^4 f_L + 2.42843 f_G^3 f_L^2
      + 4.47163 f_G^2 f_L^3 + 0.07842 f_G f_L^4 + f_L^5 \right)^{1/5}
$$

and the mixing ratio is $\eta = 1.36603\,\rho - 0.47719\,\rho^2 + 0.11116\,\rho^3$ with $\rho = f_L / f_V$, clipped to $[0, 1]$.

**Width convention:** `gamma` is the Lorentzian **half**-width at half-maximum (HWHM).

```python
from diffpes.simul import voigt

v = voigt(energy_axis, center=0.0, sigma=0.02, gamma=0.05)
print(float(v.max()))  # 5.646 - broader and lower than the pure Gaussian
```

The Voigt levels of the ladder (`novice`, `expert`, `soc`) and the forward model `simulate_tb_radial` use this profile with constant `params.sigma` / `params.gamma`.

## Fermi-Dirac Occupation

Only occupied states photoemit. Each band contribution is multiplied by the Fermi-Dirac factor at electronic temperature $T$:

$$
f(E) = \frac{1}{1 + e^{(E - E_F)/k_B T}}, \qquad k_B = 8.617 \times 10^{-5}\ \mathrm{eV/K}
$$

```python
from diffpes.simul import fermi_dirac

occ = fermi_dirac(energy=0.01, fermi_energy=0.0, temperature=20.0)
print(float(occ))  # 0.003 - 10 meV above E_F at 20 K is essentially empty
```

The thermal energy $k_B T$ is 1.3 meV at 15 K and 25 meV at room temperature — at low temperature the Fermi edge width you observe is dominated by `sigma`, not by $T$. The implementation clamps $k_B T$ to $10^{-10}$ eV so that `temperature=0.0` yields a numerically sharp step instead of a division by zero.

Note that diffpes applies $f$ at the **band center** $E_b(\mathbf{k})$, weighting each peak, rather than multiplying the final spectrum pointwise — a distinction that only matters for peaks within a few widths of $E_F$.

## Momentum Broadening

Finite angular acceptance and photon spot size smear the spectrum along $k$. `apply_momentum_broadening(intensity, k_distances, dk)` convolves each energy column of a `(K, E)` intensity map with a Gaussian of standard deviation `dk` (in $\text{Å}^{-1}$), implemented as a row-normalized dense kernel:

$$
I'(k_i, E) = \frac{\sum_j e^{-(k_i - k_j)^2 / 2\,\delta k^2}\; I(k_j, E)}
                  {\sum_j e^{-(k_i - k_j)^2 / 2\,\delta k^2}}
$$

The row normalization conserves spectral weight even for non-uniform k-point spacing, because `k_distances` are cumulative path lengths along the k-path:

```python
import jax.numpy as jnp
import diffpes
from diffpes.simul import apply_momentum_broadening

n_k = 61
k_frac = jnp.linspace(-0.5, 0.5, n_k)
eigenbands = jnp.stack(
    [-1.2 + 0.9 * jnp.cos(2 * jnp.pi * k_frac),
     -2.6 + 0.4 * jnp.cos(2 * jnp.pi * k_frac + jnp.pi)], axis=1)
surface_orb = jnp.ones((n_k, 2, 1, 9)) / 9.0

spectrum = diffpes.simul.simulate_expanded(
    level="novice", eigenbands=eigenbands, surface_orb=surface_orb,
    ef=0.0, sigma=0.03, gamma=0.05, fidelity=2000,
    temperature=20.0, photon_energy=21.2,
)

k_distances = jnp.linspace(0.0, 1.2, n_k)   # cumulative path length (1/Angstrom)
blurred = apply_momentum_broadening(spectrum.intensity, k_distances, dk=0.03)
print(blurred.shape)  # (61, 2000)
```

Typical experimental values of `dk` are 0.01-0.05 $\text{Å}^{-1}$. The kernel is fully traced by JAX, so `jax.grad` with respect to `dk` works — momentum resolution can be a fit parameter.

## Energy-Dependent Self-Energy $\Gamma(E)$

A constant `gamma` is a crude model: in real materials the imaginary part of the electron self-energy — and hence the Lorentzian width — depends on energy,

$$
\Gamma(E) = -\,\mathrm{Im}\,\Sigma(E), \qquad
A(\mathbf{k}, E) \sim \frac{1}{\pi}\,
\frac{\Gamma(E)}{\left(E - E_b(\mathbf{k})\right)^2 + \Gamma(E)^2}
$$

`evaluate_self_energy(energy, config)` evaluates $\Gamma(E)$ from a {class}`~diffpes.types.SelfEnergyConfig` in one of three modes:

| `mode` | Model | Typical use |
|--------|-------|-------------|
| `"constant"` | $\Gamma(E) = c_0$ | Equivalent to a fixed `params.gamma` |
| `"polynomial"` | $\Gamma(E) = \sum_n c_n E^n$ (coefficients highest-degree first, as in `jnp.polyval`) | Fermi-liquid $\Gamma = \Gamma_0 + \beta (E - E_F)^2$ |
| `"tabulated"` | Piecewise-linear interpolation of $(\varepsilon_i, \Gamma_i)$ nodes | Self-energies from GW or DMFT |

```python
import jax.numpy as jnp
from diffpes.simul import evaluate_self_energy
from diffpes.types import make_self_energy_config

# Fermi-liquid-like: Gamma(E) = 0.5 E^2 + 0.02 (eV), E measured from E_F
se_cfg = make_self_energy_config(
    mode="polynomial",
    coefficients=jnp.array([0.5, 0.0, 0.02]),  # [E^2, E^1, E^0]
)
energy_axis = jnp.linspace(-1.0, 1.0, 2001)
gamma_e = evaluate_self_energy(energy_axis, se_cfg)
print(float(gamma_e[1000]))  # 0.02 at E = 0 (sharp quasiparticles at E_F)
print(float(gamma_e[0]))     # 0.52 at E = -1 eV (short-lived deep states)

# Tabulated mode: nodes from a many-body calculation
se_tab = make_self_energy_config(
    mode="tabulated",
    coefficients=jnp.array([0.30, 0.10, 0.02, 0.02]),   # Gamma values (eV)
    energy_nodes=jnp.array([-3.0, -1.0, -0.1, 0.0]),    # energies (eV)
)
```

### What a self-energy does to the lineshape

For an EDC (a cut at fixed $\mathbf{k}$), the Lorentzian FWHM of a quasiparticle peak at energy $E$ is $2\Gamma(E)$, convolved with the fixed Gaussian resolution. A Fermi-liquid $\Gamma \propto (E - E_F)^2$ therefore produces the classic ARPES signature: **peaks sharpen as they approach the Fermi level** and melt into broad humps at higher binding energy. Because `coefficients` is a JAX-traced array, `jax.grad` through the simulated spectrum gives the sensitivity of every pixel to the self-energy shape — the basis for fitting $\Gamma(E)$ to measured EDC widths.

(The real part $\mathrm{Re}\,\Sigma$, which shifts and renormalizes band positions — kinks, mass enhancement — is not currently modeled; diffpes broadens the bare input bands.)

## Putting It Together

`simulate_tb_radial` accepts both an optional self-energy and optional momentum broadening; when `self_energy` is given, the Voigt Lorentzian width becomes $\Gamma(E)$ per energy-axis point instead of the constant `params.gamma`:

```python
from diffpes.types import (
    make_diagonalized_bands, make_orbital_basis,
    make_polarization_config, make_simulation_params, make_slater_params,
)

basis = make_orbital_basis(n_values=(2, 2, 2), l_values=(1, 1, 1),
                           m_values=(-1, 0, 1))
slater = make_slater_params(zeta=jnp.array([1.6, 1.6, 1.6]), orbital_basis=basis)
eigvecs = jnp.tile(jnp.eye(3, dtype=jnp.complex128)[jnp.newaxis, :2, :], (n_k, 1, 1))
kpts = jnp.stack([k_frac, jnp.zeros(n_k), jnp.zeros(n_k)], axis=1)
diag = make_diagonalized_bands(eigenbands, eigvecs, kpts, fermi_energy=0.0)

spec = diffpes.simul.simulate_tb_radial(
    diag_bands=diag,
    slater_params=slater,
    params=make_simulation_params(energy_min=-3.5, energy_max=0.5,
                                  fidelity=800, sigma=0.03, gamma=0.05),
    pol_config=make_polarization_config(theta=jnp.deg2rad(45.0)),
    work_function=4.3,
    self_energy=se_cfg,   # Gamma(E) replaces the constant gamma
    dk=0.02,              # momentum broadening (1/Angstrom)
)
print(spec.intensity.shape)  # (61, 800)
```

## Summary of Width Conventions

| Parameter | Meaning | Convention |
|-----------|---------|------------|
| `sigma` | Gaussian (instrumental) width | Standard deviation, eV; FWHM $= 2.355\,\sigma$ |
| `gamma` | Lorentzian (lifetime) width | HWHM, eV; Lorentzian FWHM $= 2\gamma$ |
| `dk` | Momentum resolution | Standard deviation, $\text{Å}^{-1}$ |
| `temperature` | Fermi-Dirac electronic temperature | Kelvin; $k_B T$ in eV internally |
| `SelfEnergyConfig.coefficients` | $\Gamma(E)$ model parameters | eV; replaces `gamma` per energy point |

## Further Reading

- [Simulation Levels](simulation-levels.md) — which levels use Gaussian vs. Voigt broadening
- [ARPES Geometry and Kinematics](arpes-geometry-and-kinematics.md) — the $(E, k)$ axes being broadened
- [Matrix Elements and Polarization](matrix-elements-and-polarization.md) — the weights applied before broadening
- API reference: {doc}`../api/simul` (`gaussian`, `voigt`, `fermi_dirac`, `apply_momentum_broadening`, `evaluate_self_energy`), {doc}`../api/types` (`make_self_energy_config`, `make_simulation_params`)
