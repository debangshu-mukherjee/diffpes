# Contributing to DiffPES

Thank you for your interest in contributing to DiffPES! This guide describes how
the codebase is written — type hinting, documentation, validation, testing, and
tooling — so your contributions match the existing standards.

## Core Principle: Invertible Modularity

DiffPES is a bidirectional instrument: the same differentiable pipeline that
takes a band structure to an ARPES spectrum is run in reverse to recover band
structures, self-energies, and matrix-element parameters from measured data.
Every module is a differentiable operator, and the boundaries between modules
are the boundaries at which the inverse problem is solved — you attach a loss at
any seam and solve for what produced the data (hopping parameters, self-energy,
experimental geometry) while freezing the rest. This invertibility is the
codebase's core asset.

It rests on one invariant:

> **Reductions stay explicit, late, and differentiable. No module collapses
> information it is not forced to.**

Concretely:

- Keep matrix elements complex; apply `|·|²` as late as possible, and **never
  before a coherent sum** — dipole channels, orbital contributions, and spin
  components interfere, and circular dichroism and spin-ARPES are exactly the
  observables that die when an intermediate modulus-squared sneaks in.
- Express experimental averaging (energy/angle resolution, kz broadening,
  temperature) as explicit, differentiable operations over distributions, not
  as baked-in convolutions or hidden quadratures.
- Use `jnp.where` / `lax.cond` and continuous fields rather than discrete swaps
  or data-dependent Python control flow, so every parameter keeps a derivative.
- A gradient is part of the physics: under the identifiability thesis, every
  row of the Fisher information matrix is built from gradients, so a silently
  zero, NaN-poisoned, or conjugation-flipped gradient is a *physics* bug even
  when the forward values look right.

The failure mode is silent: when a module performs a hard, non-differentiable,
or premature reduction, the forward model still looks correct — only
invertibility breaks, and only at that one seam. Treat any such reduction as a
design smell to be justified explicitly in review, not an implementation
detail. The JAX-First rules below are the mechanics of upholding this
principle.

## Development Setup

### Prerequisites

- Python 3.12–3.14 (`requires-python = ">=3.12,<3.15"`)
- [uv](https://docs.astral.sh/uv/) (package and environment manager)
- Git
- CUDA-compatible GPU (optional, for acceleration)

### Installation for Development

1. **Clone the repository:**
   ```bash
   git clone https://github.com/debangshu-mukherjee/DiffPES.git
   cd DiffPES
   ```

2. **Install in development mode:**
   ```bash
   # Everything (docs, tests, notebooks, dev tooling)
   uv sync --extra dev

   # With CUDA support as well
   uv sync --extra dev_cuda
   ```

   The dependency groups are defined in `pyproject.toml`: `docs`, `test`,
   `notebooks`, `cuda`, `dev` (= docs + test + notebooks + tooling),
   `dev_cuda` (= dev + cuda), and `all`.

3. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

### Project Structure

```
DiffPES/
├── src/diffpes/           # Main source code
│   ├── inout/             # Data I/O (POSCAR, EIGENVAL, KPOINTS, DOSCAR,
│   │                      #   PROCAR, CHGCAR, HDF5) and plotting
│   ├── maths/             # Dipole selection rules, spherical harmonics, Gaunt
│   ├── radial/            # Radial primitives (Bessel, wavefunctions, integrals)
│   ├── simul/             # ARPES forward model (matrix elements, polarization,
│   │                      #   self-energy, broadening, resolution, spectra)
│   ├── tightb/            # Tight-binding models (Hamiltonians, diagonalization,
│   │                      #   projections)
│   ├── types/             # Equinox PyTree types, factories, and aliases
│   └── utils/             # Mathematical utilities
├── tests/                 # Test suite (mirrors src layout, see below)
└── docs/                  # Sphinx documentation
```

Each subpackage exposes its public API through `__init__.py` with an explicit
`__all__`. The top-level `src/diffpes/__init__.py` enables 64-bit precision
(`jax.config.update("jax_enable_x64", True)`) and sets CPU threading XLA flags
**before** JAX is imported. Keep import-time side effects confined to that
module.

Some infrastructure referenced here (`CHANGELOG.md`, `docs/`, `tutorials/`,
`.pre-commit-config.yaml`, CI workflows) is still landing as part of the
tooling-floor work tracked in the planning repo; the conventions below are the
standard it lands against.

## Coding Standards

### JAX-First Development

DiffPES is built on JAX for differentiable, high-performance computation. All
new code must follow JAX best practices:

**Required JAX Patterns:**
- Use `jax.lax.scan` instead of Python `for` loops over array data
- Use `jax.lax.cond` / `jnp.where` instead of data-dependent `if`/`else`
- Use `.at[].set()` for array updates instead of in-place modification
- Keep functions purely functional — no side effects, no global mutable state
- Code must remain traceable for `jit`, `grad`, `vmap`, and sharding

**Solver stack:** optimization and nonlinear solves go through
[Optimistix](https://docs.kidger.site/optimistix/) (`least_squares`,
`root_find`, `fixed_point`, `minimise`; optax optimizers via
`optimistix.OptaxMinimiser`) and linear solves through
[Lineax](https://docs.kidger.site/lineax/). Implicit differentiation uses the
built-in machinery of those libraries; hand-rolled `custom_vjp` is reserved for
primitives they cannot express (e.g. regularized `eigh` gradients).

**Differentiability rules of the house:**
- A gradient gate passes only when `jax.grad` agrees with central finite
  differences — *finite* is not *correct*, and a finite-but-zero gradient where
  the physics has real sensitivity is a failure.
- Guard the double-`jnp.where` NaN trap: when a branch can produce `nan`/`inf`,
  sanitize the *input* of the unsafe branch with an inner `where`, not just the
  output.
- Complex *parameters* are carried as stacked reals and re-complexified inside
  the forward (the real-ification doctrine); complex *state* (matrix elements,
  eigenvectors) stays complex, with `|·|²` applied late.
- `jnp.linalg.eigh` gradients blow up at degeneracies (symmetry points, Kramers
  pairs under SOC). Differentiate only gauge-invariant combinations
  (projectors, spectral functions), use the degeneracy-aware machinery in
  `diffpes.tightb`, or route through a Green's-function formulation — never
  raw eigenvector gradients at a possible degeneracy.

**Example:**
```python
# ❌ Wrong - Python loops and conditionals over array data
def bad_function(x):
    result = []
    for i in range(len(x)):
        if x[i] > 0:
            result.append(x[i] * 2)
    return jnp.array(result)


# ✅ Correct - vectorized JAX
@jaxtyped(typechecker=beartype)
def good_function(x: Float[Array, " n"]) -> Float[Array, " n"]:
    doubled_positive: Float[Array, " n"] = jnp.where(x > 0, x * 2, x)
    return doubled_positive
```

### Type Hinting with jaxtyping and beartype

Every public function is runtime-typechecked with the
`@jaxtyped(typechecker=beartype)` decorator stack and annotated with
`jaxtyping` shape/dtype specs:

```python
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple
from jaxtyping import Array, Complex, Float, jaxtyped

from diffpes.types import ArpesSpectrum, BandStructure, scalar_float


@jaxtyped(typechecker=beartype)
def simulate_spectrum(
    bands: BandStructure,
    photon_energy: scalar_float,
    temperature: Optional[scalar_float] = 15.0,
) -> ArpesSpectrum:
    """..."""
```

**Type Hinting Rules:**
- All parameters and return values are annotated; multiple returns use
  `beartype.typing.Tuple[...]`.
- Annotate intermediate variables inside function bodies too — e.g.
  `theta_rad: Float[Array, ""] = jnp.deg2rad(theta_deg)`.
- **Assign before returning.** Bind a function's result to a type-annotated
  variable and return that name, rather than returning a bare expression — so
  the returned value carries an explicit type at its definition site.
- Use descriptive dimension names in shape specs:
  `Float[Array, "nkpt nband"]`, `Complex[Array, "nkpt nband natom norb"]`,
  scalars as `Float[Array, ""]`.
- Prefer the scalar aliases from `diffpes.types` (`types/aliases.py`) for
  scalar arguments; these are unions accepting both Python scalars and 0-d JAX
  arrays.
- Import shared types from `diffpes.types`, not by re-defining them.
- Import typing constructs (`Optional`, `Union`, `Tuple`, `List`, `Dict`,
  `TypeAlias`) from `beartype.typing`, not the stdlib `typing` module.

### Custom Types and PyTrees

**All types live in `diffpes.types` — no exceptions.** Every structured data
type — every PyTree, every carrier, every type alias, **and every `make_*`
factory that builds one** — is defined under `src/diffpes/types/` and **nowhere
else**. Every other subpackage (`simul`, `tightb`, `radial`, `maths`, `inout`,
`utils`) **imports** its types from `diffpes.types`; it must **not** define its
own PyTree, container, or factory. Why: a single import surface, one
registration per type, one home for the validation contract, and no duplicate
carriers drifting across modules — which is exactly what the inverse problem
needs: the fitting layer compares `ArpesSpectrum` objects, so there must be
*one* `ArpesSpectrum`. A result/parameter container that "feels local" to a
solver or producer is **still a type**: it goes in `diffpes.types`, not beside
the function that returns it.

Structured data types are **Equinox modules** (`eqx.Module`): immutable JAX
PyTrees that flow through `jit`/`grad`/`vmap`. Static, non-array metadata
fields are declared with `eqx.field(static=True)` so they are excluded from the
differentiable leaves.

```python
import equinox as eqx
from jaxtyping import Array, Float


class BandStructure(eqx.Module):
    """JAX-compatible band structure with k-points and eigenvalues.

    :see: :class:`~.test_bands.TestBandStructure`
    ...
    """

    kpoints: Float[Array, "nkpt 3"]
    eigenvalues: Float[Array, "nkpt nband"]
    fermi_energy: Float[Array, ""]
```

### Validation Pattern for Factory Functions

Custom types are constructed through `make_*` factory functions that validate
inputs. These factories live in `diffpes.types` **next to the type they
build** (never in the consuming subpackage). Use a two-tier approach:

- **Static shape/structure checks** that can be resolved at trace time use
  plain Python `raise ValueError`.
- **Data-dependent (traced) checks** use `equinox.error_if`, which raises at
  runtime without breaking `jit`.

```python
@jaxtyped(typechecker=beartype)
def make_band_structure(
    kpoints: Float[Array, "nkpt 3"],
    eigenvalues: Float[Array, "nkpt nband"],
    fermi_energy: scalar_float,
) -> BandStructure:
    """Create a BandStructure PyTree with data validation.

    :see: :class:`~.test_bands.TestBandStructure`
    ...
    """
    kpoints = jnp.asarray(kpoints)
    eigenvalues = jnp.asarray(eigenvalues)

    if eigenvalues.shape[0] != kpoints.shape[0]:   # static -> ValueError
        raise ValueError("eigenvalues and kpoints disagree on nkpt")

    checked_eigenvalues = eqx.error_if(            # traced -> eqx.error_if
        eigenvalues,
        jnp.any(jnp.isnan(eigenvalues)),
        "eigenvalues must be finite",
    )
    band_structure: BandStructure = BandStructure(
        kpoints=kpoints,
        eigenvalues=checked_eigenvalues,
        fermi_energy=jnp.asarray(fermi_energy),
    )
    return band_structure
```

### Units, Conventions, and Indexing

- Energies in **eV**, lengths in **Angstrom**, k-vectors in **1/Angstrom**;
  angles are **degrees at user-facing API boundaries** and radians internally
  (convert once, at the boundary, into an annotated variable).
- Use standard Python/NumPy indexing everywhere (zero-based, end-exclusive):
  non-s orbitals are `slice(1, 9)`, p orbitals `slice(1, 4)`, d orbitals
  `slice(4, 9)`. Do not use MATLAB-style indexing notation, even in comments.
- Sign and phase conventions (spherical-harmonic phases, Gaunt coefficients,
  polarization vectors, rotation frames) are pinned centrally in the physics
  canon of the planning repo; when code and canon disagree, that is a bug —
  do not silently re-pin a convention locally.

### Documentation Standards

Docstrings follow the **NumPy / numpydoc convention** (enforced by Ruff's
`pydocstyle` rules and `pydoclint`, configured in `pyproject.toml`). Coverage
is checked by `interrogate` (`fail-under = 90`). Do **not** use ad-hoc section
headers — stick to the numpydoc sections below.

`pydoclint` currently sets `check-return-types = true`; jaxtyping shape
strings (e.g. `Float[Array, "nkpt nband"]`) are core signature syntax that
pydoclint cannot always parse for return-type comparison, so if it fights a
correct jaxtyping return annotation, relax that comparison in `pyproject.toml`
(rheedium's configuration is the precedent) rather than degrading the
annotation — argument order and required `Returns`/`Yields` sections stay
enforced either way.

#### Module Docstrings

Each module starts with a one-line summary, an `Extended Summary`, a
`Routine Listings` section cross-referencing every public object, and a
`Notes` section where relevant. For a package `__init__.py`, the
`Extended Summary` must list **every submodule `.py` file** (as
`- :mod:`name`` entries with a one-line description) — when you add a new
submodule, add it to that listing in the same change.

Use the correct Sphinx role in `Routine Listings`: `:func:` for functions,
`:class:` for classes/PyTrees, `:obj:` for type aliases and constants, and
`:mod:` for submodules.

**Every public object is listed in three places, and all three must agree:**

1. In its own **module**, at the **top** in the docstring's `Routine Listings`
   (the human- and Sphinx-facing API index), **and**
2. at the **bottom** in that module's `__all__` (the import-facing public
   surface), **and**
3. in the **subpackage `__init__.py`** — repeated in `__init__.py`'s **own**
   `Routine Listings` *and* `__all__`.

A symbol missing from any of the three is a defect. When you add, rename, or
remove a public function, update **all three** in the same change, and keep
the one-line summary sentence **verbatim identical** across the function
docstring and both `Routine Listings`.

**Export once, from the module that owns it — no compatibility re-exports.**
Each public symbol has exactly one canonical export path: the module that
defines it, surfaced through its own subpackage's `__init__.py`. Never add a
second export of the same symbol elsewhere — not for convenience, not to
preserve an old import location. When a symbol moves or is renamed, update
every import site and **delete** the old path in the *same* change — no shim,
no alias, no `DeprecationWarning`. The only migration record is a
`CHANGELOG.md` note. This is the project's zero-legacy policy.

#### Function and Class Docstrings

```python
@jaxtyped(typechecker=beartype)
def free_electron_kz(
    kinetic_energy: scalar_float,
    kpar: Float[Array, " n"],
    inner_potential: scalar_float,
) -> Float[Array, " n"]:
    r"""
    Calculate out-of-plane momentum in the free-electron final state.

    Computes :math:`k_z` from the photoelectron kinetic energy and
    in-plane momentum under the free-electron final-state approximation
    with inner potential :math:`V_0`.

    :see: :class:`~.test_kinematics.TestFreeElectronKz`

    Parameters
    ----------
    kinetic_energy : scalar_float
        Photoelectron kinetic energy in eV.
    kpar : Float[Array, " n"]
        In-plane momentum magnitudes in 1/Angstrom.
    inner_potential : scalar_float
        Inner potential :math:`V_0` in eV.

    Returns
    -------
    kz_values : Float[Array, " n"]
        Out-of-plane momenta in 1/Angstrom.

    Notes
    -----
    1. Form the free-electron dispersion prefactor :math:`2m/\hbar^2`.
    2. Evaluate :math:`k_z = \sqrt{(2m/\hbar^2)(E_k + V_0) - k_\parallel^2}`.
    3. Bind the result to ``kz_values`` and return it.

    See Also
    --------
    kinetic_energy_from_photon : Kinetic energy from photon energy and
        work function.
    """
```

Note how the `Returns` entry is named `kz_values` — the type-annotated
variable the body actually returns (assign-before-returning), so the
docstring, the body, and the signature agree.

**Docstring conventions:**
- Open with a single imperative summary line.
- Add a `:see:` Sphinx cross-reference linking the object to its test class
  (e.g. `:see: :class:`~.test_bessel.TestSphericalBessel``); the test class
  carries the matching back-reference, so source ↔ test links are
  bidirectional in the rendered docs.
- `Parameters` and `Returns` repeat the type and describe each item, with
  units. Name return values (numpydoc `name : type` form) after the
  **type-annotated variable actually returned** (see "Assign before
  returning"), so the docstring, the body, and the signature agree.
- **Mark static (non-traced) parameters** explicitly — arguments passed via
  `static_argnames`, Python `int`/`str`/`bool` flags that drive shape or
  control flow, and values in `eqx.field(static=True)`: e.g. *"(**static** — a
  compile-time constant; changing it triggers retracing)"*.
- Use `Notes` (often a numbered list) for the algorithm; `See Also` for
  related functions; `Attributes` for `eqx.Module` fields; `Raises` where a
  function raises.
- Use a raw string (`r"""`) when the docstring contains LaTeX/backslashes —
  matrix-element and self-energy docstrings usually do.

### Code Style

Style is enforced by Ruff (`line-length = 79`, `target-version = "py312"`,
double quotes). The active lint rule set includes `D, E, F, B, I, N, UP, ANN,
S, A, C4, PIE, PT, RET, SIM, ARG, ERA, PL`. Key conventions:

- **Variable Names**: descriptive `snake_case`; long names over abbreviations
  (`photoemission_intensity`, not `pi`). Scientific single-letter symbols
  (`G`, `L`, `S`) are permitted where they mirror the physics.
- **No inline comments for explanation**: explanations belong in docstrings.
  Comments are reserved for non-obvious rationale (the *why*, not the *what*).
- **Pure functions**: no side effects; return new data.
- **Imports**: sorted by isort (`I`); imports inside functions only to guard
  optional dependencies or platform branches.

## Testing

The test suite uses `pytest` with `chex`, `pytest-cov`, and `pytest-xdist`;
property-based tests use `hypothesis`. Tests are first-class source: the same
typing and docstring discipline as `src/` applies (every test method is
`def test_*(self) -> None:` with annotated intermediates; test docstrings
state *what* is verified — property, tolerance, units — and *how*).

### Test Layout

Tests mirror the source layout under `tests/test_diffpes/`:

```
tests/
└── test_diffpes/
    ├── test_inout/test_vasp_readers.py
    ├── test_maths/test_gaunt.py
    ├── test_radial/test_bessel.py
    ├── test_simul/...
    ├── test_tightb/test_hamiltonian.py
    └── test_types/...                 # one test_<module>.py per source module
```

- Test files are named `test_<module>.py`; test classes `Test*` (typically
  `chex.TestCase`); test functions `test_*`.
- One `Test<Symbol>` class per public symbol, carrying the `:see:`
  back-reference to the symbol under test.

### What a Test Must Validate Against

**External truths, never DiffPES's own outputs.** A verification test compares
against a closed-form result (hydrogenic radial integrals, Rashba spinors,
free-electron kinematics), a `scipy`/`sympy` reference value, or a published
number (e.g. a pinned chinook cross-check) — not against a stored output of
this package, and not against a magic number whose only provenance is the
function under test.

**Gradients are gated, not assumed.** Every differentiable primitive gets a
grad-vs-finite-difference test (central differences, stated tolerance), plus a
zero-gradient tripwire for parameters that must carry sensitivity. A test
suite that only checks forward values cannot catch a corrupted Fisher row.

**Writing tests:**
- Prefer `chex` assertions over bare `assert` for arrays:
  `chex.assert_shape`, `chex.assert_trees_all_close`,
  `chex.assert_tree_all_finite`.
- Use parameterized/table-driven cases for convention-sensitive code (signs,
  phases, orbital orderings), and `hypothesis` for invariants (unitarity,
  sum rules, gauge invariance under random eigenvector phases).
- Test JAX compatibility explicitly: `jit`, `grad`, and `vmap` paths where
  relevant.

### Running Tests

```bash
# Run the whole suite
pytest

# Run a single module / class / test
pytest tests/test_diffpes/test_radial/test_bessel.py
pytest tests/test_diffpes/test_radial/test_bessel.py::TestSphericalBessel

# Coverage
pytest tests/ --cov=src/diffpes --cov-report=term-missing
```

## Tutorial Notebooks

Tutorials live in `tutorials/` as Jupyter notebooks paired with Jupytext
percent scripts (`.ipynb` plus `.py`) so they can be edited while keeping
reviewable source diffs. **Explanation lives in markdown cells, not code
comments** — narrative, motivation, and physics belong in markdown blocks;
keep code cells comment-free. After editing a paired notebook, sync and strip
outputs before committing (the pre-commit hooks do this for you).

## Pull Request Process

### Before Submitting

```bash
# Lint and format (must match CI)
ruff check src/ tests/
ruff format src/ tests/

# Source docstring structure
pydoclint src/

# Type check
ty check

# Run all pre-commit hooks
pre-commit run --all-files

# Run the test suite
pytest
```

`ty` is the project's type checker; `pre-commit` runs ruff (check + format)
and the other hooks. If a hook modifies files, the commit aborts — re-stage
and commit again.

### PR Guidelines

1. **Branch Naming:** descriptive, e.g. `feature/slab-hamiltonian` or
   `fix/gaunt-phase-convention`.
2. **Commit Messages:** a clear summary line, then bullet points for the
   substantive changes (implementation, tests, docs).
3. **PR Description:** what the PR does, why it's needed, how to test it, and
   any breaking changes.

### Review Process

All PRs require:
- [ ] Passing CI tests
- [ ] Code review approval
- [ ] Documentation updates (if applicable)
- [ ] No merge conflicts

## Issue Guidelines

**Bug reports** include: a minimal reproducible example, expected vs actual
behavior, environment details (Python, JAX, GPU/CPU), and error messages.
For a *wrong-gradient* bug, include the finite-difference comparison — it is
as much a bug as a wrong forward value.

**Feature requests** include: the use case, proposed API (if applicable),
performance considerations, and the relationship to existing functionality.

## Development Guidelines

### Adding New Features

1. **Design Phase:**
   - Discuss the approach in an issue first; check the planning repo for an
     owning plan — most subsystems have one, with pinned conventions and gates.
   - Consider JAX constraints (tracing, shapes, purity, degeneracies) early.
   - Plan the type signatures, custom types, and public API.

2. **Implementation:**
   - **Any new type, PyTree, or `make_*` factory goes in `diffpes.types`** —
     never in the consuming subpackage; import it from there.
   - Place the code in the appropriate subpackage and export it via that
     package's `__init__.py` (`__all__`), maintaining the three-places rule.
   - Decorate with `@jaxtyped(typechecker=beartype)` and annotate fully.
   - Add numpydoc docstrings with a `:see:` cross-reference to the tests.
   - Add tests mirroring the source path under `tests/test_diffpes/`,
     including external-truth and grad-vs-finite-difference gates.

3. **Documentation:**
   - Update API documentation and `Routine Listings`.
   - Add a tutorial example if it introduces user-facing functionality.
   - Note behavior changes in `CHANGELOG.md`.

### API Evolution (zero-legacy)

The codebase carries **no compatibility layer**. When an API changes:

- **No shims, aliases, re-exports, or `DeprecationWarning`s** for old import
  paths or signatures.
- Update every call site and **delete** the old path in the *same* change;
  two implementations or import paths never ship together.
- The **only** migration record is a `CHANGELOG.md` note.
- Prefer getting the API right over preserving a wrong one — pre-1.0,
  breaking changes are allowed and expected.

### Versioning

`[project].version` in `pyproject.toml` is the **single source of truth** for
the package version (CalVer, e.g. `2026.03.01`).

## Getting Help

- **Questions:** Open a discussion or issue
- **Documentation:** Check the rendered docs (Read the Docs)

Thank you for contributing to DiffPES and advancing differentiable ARPES
simulation!
