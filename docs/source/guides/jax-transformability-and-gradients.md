# JAX Transformability and Gradients

diffpes is a differentiable instrument: the same forward pipeline that maps
a band structure to an ARPES spectrum is run in reverse — by attaching a
loss and calling `jax.grad` — to recover band parameters, self-energies, and
experimental geometry from measured data. That only works if `jit`, `vmap`,
and `grad` behave predictably everywhere. This guide states what is
supported where, shows worked examples, and spells out the gradient
correctness doctrine the codebase is tested against.

## The x64 Policy

Importing `diffpes` enables 64-bit precision *before* JAX is imported:

```python
jax.config.update("jax_enable_x64", True)
```

This happens once, in the top-level `src/diffpes/__init__.py` (together with
CPU-threading XLA flags), and every factory casts its arrays to `float64`.
Do not flip x64 off: Fermi-edge derivatives at 15 K and Voigt tails span
enough orders of magnitude that float32 gradients lose the small
sensitivities the Fisher-information analysis depends on. Angles are degrees
at user-facing API boundaries and radians internally — converted once, at
the boundary.

## `jit`: Mark the Static Arguments

The simulators mix traced scalars with genuinely static inputs: `level` and
`polarization` are Python strings that select code branches, and `fidelity`
is a Python `int` that sets the energy-axis length (JAX requires static
shapes). Name them in `static_argnames`:

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

Changing a static argument (a new `fidelity`, a different `level`) triggers
recompilation; changing a traced one (`sigma`, `ef`, the eigenvalue array
values) does not. The same split lives inside the PyTrees themselves:
`SimulationParams.fidelity` and `PolarizationConfig.polarization_type` are
auxiliary data, so PyTree-accepting functions inherit correct static
behavior automatically (see
[PyTree Architecture](pytree-architecture.md)).

## `vmap`: Parameter Sweeps

Anything traced can be batched. A photon-energy sweep is one `vmap` over a
closure, and the result is a batched `ArpesSpectrum` intensity:

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

The same pattern batches over polar angles, temperatures, or whole batches
of eigenvalue sets — any leaf, any carrier.

## `grad`: A Worked Inverse-Problem Loss

A scalar loss on `ArpesSpectrum.intensity` differentiates with respect to
any traced parameter. Here, sensitivity of a region of the spectrum to the
broadening, temperature, and Fermi level:

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

All three come back as finite, nonzero 0-d arrays. In a fitting workflow the
loss would compare against measured data (`jnp.mean((sim - meas) ** 2)`)
and the gradient would drive an
[Optimistix](https://docs.kidger.site/optimistix/) solver — but the
mechanics are exactly this.

## The Differentiability Doctrine

These are the rules of the house, codified in the project's CONTRIBUTING
guide and enforced by the test suite (see the *Testing / Validation*
reference, {doc}`../tests/gradients`).

### Grad-vs-finite-difference is the correctness gate

*Finite* is not *correct*. Every differentiable primitive is gated by
agreement between `jax.grad` and central finite differences:

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

A gradient that disagrees with the finite difference is a physics bug even
when the forward values look right — under the identifiability thesis,
every row of the Fisher information matrix is built from these gradients.

### Finite-but-zero gradients are bugs — unless the physics is flat

A zero gradient where the physics has real sensitivity is a failure mode
the tests trip on explicitly. But the converse matters too: some zeros are
honest, and you must be able to tell them apart. Two real examples from the
current code:

- `jax.grad` of the *total* intensity with respect to `sigma` is ~0 at the
  basic level: Gaussian broadening is a normalized convolution, so the sum
  over the full energy axis is conserved. A pointwise or windowed loss (as
  above) shows the true, large sensitivity.
- `jax.grad` with respect to `photon_energy` is exactly zero at 11 eV for
  the Yeh–Lindau levels: the cross-section tables are tabulated at
  20/40/60 eV with constant extrapolation outside that range, and the
  heuristic weights are piecewise-constant in photon energy by
  construction. Inside 20–60 eV the interpolation is piecewise-linear and
  the gradient is finite; below 20 eV, sweep with `vmap` instead of
  differentiating.

When you hit an unexpected zero, check whether the parameter reaches the
loss through a genuinely flat map before filing it as a bug — and if the
physics says it should not be flat, it *is* a bug.

### The double-`where` NaN guard

`jnp.where(cond, safe, unsafe)` is not enough: JAX evaluates *both*
branches, and if the unsafe branch produces `nan` on the untaken inputs,
the *gradient* of the `where` is NaN-poisoned even though the forward value
is fine. Sanitize the unsafe branch's **input** with an inner `where`:

```python
def safe_sqrt(x):
    # wrong: jnp.where(x > 0, jnp.sqrt(x), 0.0) -- grad is NaN at x <= 0
    x_safe = jnp.where(x > 0.0, x, 1.0)          # inner where: fix the input
    return jnp.where(x > 0.0, jnp.sqrt(x_safe), 0.0)

print(jax.grad(safe_sqrt)(-1.0))  # 0.0, not nan
```

This pattern appears throughout the simulators wherever kinematics can go
evanescent (square roots of energy differences, normalizations by sums that
can vanish).

### `eigh` at degeneracies: gauge-invariant quantities only

`jnp.linalg.eigh` gradients blow up at degenerate eigenvalues — exactly
where band structures like to be (high-symmetry points, Kramers pairs under
SOC). The rule: never differentiate raw eigenvectors near a possible
degeneracy. Differentiate gauge-invariant combinations instead — projectors
and spectral weights, in which the arbitrary phase (and, in a degenerate
subspace, the arbitrary rotation) cancels:

```python
def orbital_weights(theta):
    h = jnp.array([[jnp.cos(theta), jnp.sin(theta)],
                   [jnp.sin(theta), -jnp.cos(theta)]])   # Hermitian, 2x2
    _, vecs = jnp.linalg.eigh(h)
    # |<orbital 0 | band n>|^2 -- diagonal of a projector, gauge-invariant
    return jnp.abs(vecs[0, :]) ** 2

print(jax.jacobian(orbital_weights)(0.3))  # finite away from theta = 0
```

The weights are differentiable in the parameters away from crossings, and the
degeneracy-aware machinery in `diffpes.tightb` (or a Green's-function
formulation) covers the crossings themselves. Complex *parameters* follow
the real-ification doctrine — carried as stacked reals and re-complexified
inside the forward — while complex *state* (matrix elements, eigenvectors)
stays complex, with the modulus-squared applied as late as possible so that
interference channels survive.

## What Flows Gradients End-to-End Today

Through the `simulate_*_expanded` family and `simulate_context`, measured
against windowed intensity losses:

| Parameter | Levels where the gradient is live |
|---|---|
| `eigenbands`, `surface_orb`, `ef`, `sigma`, `temperature` | all levels |
| `gamma` (Lorentzian width) | novice, expert, soc (Voigt levels) |
| `incident_theta`, `incident_phi`, `polarization_angle` | advanced, expert, soc |
| `surface_spin`, `ls_scale` | soc |
| `photon_energy` | piecewise: linear inside the 20–60 eV cross-section tables, flat outside; piecewise-constant at basic level |
| `level`, `polarization`, `fidelity` | static — no gradient by design |

The full tight-binding forward model `simulate_tb_radial` extends this to
hopping parameters (via `DiagonalizedBands`), Slater radial parameters
(`SlaterParams.zeta` and coefficients), self-energy coefficients
(`SelfEnergyConfig`), and the work function — the entire pipeline is
JAX-traceable and `grad`-able in its continuous inputs.

## Related Reading

- [PyTree Architecture](pytree-architecture.md) — why children vs.
  auxiliary data decides what is differentiable.
- [Expanded Wrappers and Conventions](expanded-wrappers-and-conventions.md)
  — the plain-array entry points used in the examples above.
- [VASP Data Ingestion](vasp-data-ingestion.md) — producing the input
  PyTrees from DFT output.
- API reference: {doc}`../api/simul`, {doc}`../api/tightb`,
  {doc}`../api/types`.
