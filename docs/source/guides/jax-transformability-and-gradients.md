# JAX Transformability and Gradients

diffpes implements a differentiable forward pipeline. The pipeline maps a band
structure to an ARPES spectrum. A loss and `jax.grad` provide derivatives for
recovering band parameters, self-energies, and experimental geometry from
measured data. This process requires predictable `jit`, `vmap`, and `grad`
behavior. This guide identifies the supported transformations. It also
provides examples and states the tested gradient correctness rules.

## The x64 Policy

Importing `diffpes` enables 64-bit precision before other diffpes modules
import JAX:

```python
jax.config.update("jax_enable_x64", True)
```

The top-level `src/diffpes/__init__.py` performs this initialization once. It
also sets the XLA flags for CPU threading. Every factory casts its arrays to
`float64`. Keep x64 enabled. Fermi-edge derivatives at 15 K and Voigt tails
span many orders of magnitude. Float32 gradients lose small sensitivities
that the Fisher information analysis requires. User-facing APIs accept angles
in degrees. The boundary converts them once, and internal functions use
radians.

## `jit`: Mark the Static Arguments

The simulators combine traced scalars with static inputs. The Python strings
`level` and `polarization` select code branches. The Python integer `fidelity`
sets the energy-axis length because JAX requires static shapes. List these
arguments in `static_argnames`:

```python
import jax
import jax.numpy as jnp
from diffpes.simul import simulate_expanded

eigenbands = jnp.linspace(-2.0, 0.5, 100).reshape(20, 5)   # [K, B]
surface_orb = jnp.ones((20, 5, 2, 9)) * 0.1                # [K, B, A, 9]

simulate_jit = jax.jit(
    simulate_expanded,
    static_argnames=("level", "fidelity", "polarization"),
)
spectrum = simulate_jit(
    level="basic", eigenbands=eigenbands, surface_orb=surface_orb,
    ef=0.0, sigma=0.04, fidelity=500, temperature=15.0, photon_energy=11.0,
)
print(spectrum.intensity.shape)  # (20, 500)
```

A new static argument value triggers recompilation. Examples include a new
`fidelity` or a different `level`. Changes to traced values do not trigger
recompilation. Such values include `sigma`, `ef`, and the eigenvalue array.
The PyTrees maintain the same distinction. `SimulationParams.fidelity` and
`PolarizationConfig.polarization_type` are auxiliary data. Therefore,
PyTree-accepting functions receive the correct static behavior. See
[PyTree Architecture](pytree-architecture.md).

## `vmap`: Parameter Sweeps

JAX can batch every traced value. One `vmap` over a closure performs a
photon-energy sweep. The result contains a batched `ArpesSpectrum.intensity`
array:

```python
def intensity_at(photon_energy):
    spec = simulate_expanded(
        level="basicplus", eigenbands=eigenbands, surface_orb=surface_orb,
        ef=0.0, sigma=0.04, fidelity=500,
        temperature=15.0, photon_energy=photon_energy,
    )
    return spec.intensity

photon_energies = jnp.array([21.2, 40.8, 60.0])   # He-I, He-II, ...
stack = jax.vmap(intensity_at)(photon_energies)
print(stack.shape)  # (3, 20, 500)
```

The same pattern batches polar angles, temperatures, or complete sets of
eigenvalues. It applies to any traced leaf in any carrier.

## `grad`: A Worked Inverse-Problem Loss

A scalar loss on `ArpesSpectrum.intensity` differentiates with respect to
any traced parameter. The following example computes regional spectrum
sensitivity to broadening, temperature, and the Fermi level:

```python
def loss(sigma, temperature, ef):
    spec = simulate_expanded(
        level="expert", eigenbands=eigenbands, surface_orb=surface_orb,
        ef=ef, sigma=sigma, gamma=0.1, fidelity=500,
        temperature=temperature, photon_energy=11.0,
    )
    return jnp.sum(spec.intensity[:, 200:300])

g_sigma, g_temp, g_ef = jax.grad(loss, argnums=(0, 1, 2))(0.04, 15.0, 0.0)
```

All three results are finite, nonzero 0-D arrays. In a fitting workflow, the
loss compares the simulation with measured data. For example, it can use
`jnp.mean((sim - meas) ** 2)`. An
[Optimistix](https://docs.kidger.site/optimistix/) solver uses the resulting
gradient.

## The Differentiability Doctrine

The project's CONTRIBUTING guide defines the following rules. The test suite
enforces them. See the *Testing / Validation* reference,
{doc}`../tests/gradients`.

### Grad-vs-finite-difference is the correctness gate

*Finite* is not *correct*. Tests require agreement between `jax.grad` and
central finite differences for every differentiable primitive:

```python
def f(sigma):
    spec = simulate_expanded(
        level="expert", eigenbands=eigenbands, surface_orb=surface_orb,
        ef=0.0, sigma=sigma, gamma=0.1, fidelity=500,
        temperature=15.0, photon_energy=11.0,
    )
    return spec.intensity[10, 250]

eps = 1e-4
autodiff = jax.grad(f)(0.04)
central = (f(0.04 + eps) - f(0.04 - eps)) / (2.0 * eps)
# autodiff and central agree to ~6 significant figures
```

A gradient that disagrees with the finite difference is a physics bug. The
forward values can still appear correct. Under the identifiability thesis,
these gradients form every row of the Fisher information matrix.

### Finite-but-zero gradients are bugs — unless the physics is flat

The tests detect zero gradients where the physics has real sensitivity.
However, a zero can also represent a physically constant response. Distinguish
these cases. The current code provides two examples:

- At the basic level, `jax.grad` of the *total* intensity with respect to
  `sigma` is approximately zero. Gaussian broadening uses a normalized
  convolution. Therefore, it conserves the sum over the complete energy axis.
  A pointwise or windowed loss shows the large local sensitivity.
- At 11 eV, `jax.grad` with respect to `photon_energy` is exactly zero for the
  Yeh–Lindau levels. The cross-section tables contain values at 20, 40, and
  60 eV. Constant extrapolation applies outside that range. The heuristic
  weights also remain piecewise constant in photon energy. Between 20 and
  60 eV, piecewise-linear interpolation gives a finite gradient. Below 20 eV,
  use `vmap` for a parameter sweep.

For an unexpected zero, first check whether a constant map connects the
parameter to the loss. Report a bug when the physics predicts a nonconstant
response.

### The double-`where` NaN guard

`jnp.where(cond, safe, unsafe)` cannot guard an unsafe branch by itself. JAX
evaluates *both* branches. An untaken branch can therefore produce `nan` and
contaminate the `where` gradient. The forward value can remain finite.
Sanitize the unsafe branch **input** with an inner `where`:

```python
def safe_sqrt(x):
    # wrong: jnp.where(x > 0, jnp.sqrt(x), 0.0) -- grad is NaN at x <= 0
    x_safe = jnp.where(x > 0.0, x, 1.0)          # inner where: fix the input
    return jnp.where(x > 0.0, jnp.sqrt(x_safe), 0.0)

print(jax.grad(safe_sqrt)(-1.0))  # 0.0, not nan
```

The simulators use this pattern for evanescent kinematics. Examples include
square roots of energy differences and normalization by sums that can vanish.

### `eigh` at degeneracies: gauge-invariant quantities only

`jnp.linalg.eigh` gradients diverge at degenerate eigenvalues. Band structures
often contain degeneracies at high-symmetry points and Kramers pairs under
SOC. Do not differentiate raw eigenvectors near a possible degeneracy.
Instead, differentiate gauge-invariant projectors and spectral weights. These
quantities cancel arbitrary phases and rotations within a degenerate
subspace:

```python
def orbital_weights(theta):
    h = jnp.array([[jnp.cos(theta), jnp.sin(theta)],
                   [jnp.sin(theta), -jnp.cos(theta)]])   # Hermitian, 2x2
    _, vecs = jnp.linalg.eigh(h)
    # |<orbital 0 | band n>|^2 -- diagonal of a projector, gauge-invariant
    return jnp.abs(vecs[0, :]) ** 2

print(jax.jacobian(orbital_weights)(0.3))  # finite away from theta = 0
```

The weights remain differentiable away from crossings. The degeneracy-aware
machinery in `diffpes.tightb` handles the crossings. A Green's-function
formulation provides another option. Represent complex *parameters* as stacked
reals under the real-ification doctrine. Reconstruct their complex form inside
the forward model. Keep complex *state*, including matrix elements and
eigenvectors, complex. Apply the modulus-squared operation late enough to
preserve interference channels.

## What Flows Gradients End-to-End Today

The following table lists live gradients through `simulate_*_expanded` and
`simulate_context`. It assumes windowed intensity losses:

| Parameter | Levels where the gradient is live |
|---|---|
| `eigenbands`, `surface_orb`, `ef`, `sigma`, `temperature` | all levels |
| `gamma` (Lorentzian width) | novice, expert, soc (Voigt levels) |
| `incident_theta`, `incident_phi`, `polarization_angle` | advanced, expert, soc |
| `surface_spin`, `ls_scale` | soc |
| `photon_energy` | piecewise: linear inside the 20–60 eV cross-section tables, flat outside; piecewise-constant at basic level |
| `level`, `polarization`, `fidelity` | static — no gradient by design |

The full tight-binding forward model `simulate_tb_radial` also supports
hopping parameters through `DiagonalizedBands`. It supports Slater radial
parameters through `SlaterParams.zeta` and its coefficients. It also supports
`SelfEnergyConfig` coefficients and the work function. JAX can trace and
differentiate the complete pipeline with respect to continuous inputs.

## Related Reading

- [PyTree Architecture](pytree-architecture.md) — why children vs.
  auxiliary data decides what is differentiable.
- [Expanded Wrappers and Conventions](expanded-wrappers-and-conventions.md)
  — the plain-array entry points used in the examples above.
- [VASP Data Ingestion](vasp-data-ingestion.md) — producing the input
  PyTrees from DFT output.
- API reference: {doc}`../api/simul`, {doc}`../api/tightb`,
  {doc}`../api/types`.
