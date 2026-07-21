# Contributing to diffpes

Thank you for your interest in contributing to diffpes! This guide describes how
the codebase is written — type hinting, documentation, validation, testing, and
tooling — so your contributions match the existing standards.

## Core Principle: Invertible Modularity

diffpes is a bidirectional instrument: the same differentiable pipeline that
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
   git clone https://github.com/debangshu-mukherjee/diffpes.git
   cd diffpes
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
diffpes/
├── src/diffpes/           # Main source code
│   ├── certify/           # JAX-native forward certification, evidence,
│   │                      #   provenance, policies, and information flow
│   ├── inout/             # Data I/O (POSCAR, EIGENVAL, KPOINTS, DOSCAR,
│   │                      #   PROCAR, CHGCAR, HDF5, certificates) and plotting
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

Most supporting infrastructure (`CHANGELOG.md`, `docs/`,
`.pre-commit-config.yaml`, CI workflows) is in place; `tutorials/` (paired
notebooks at the repo root) is still landing as part of the tooling-floor
work tracked in the planning repo.

## Coding Standards

### JAX-First Development

diffpes is built on JAX for differentiable, high-performance computation. All
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

#### Type Hinting Rules:
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
- **Cross-subpackage imports are public and go through the subpackage.**
  Whenever a file imports something from a *different* subpackage, two
  things must hold: (1) the source subpackage **exports the name
  publicly** (module `Routine Listings` + `__all__` + `__init__.py`
  re-export — the three-places rule), and (2) the importer takes it
  **from the subpackage itself, never from an individual file inside
  it** (`from diffpes.types import KB_EV_PER_K`, not
  `from diffpes.types.constants import KB_EV_PER_K`; `from
  diffpes.inout import read_procar`, not `from diffpes.inout.procar
  import read_procar`). There is no private-name exception: if another
  subpackage needs it, it is public by definition — promote it. Deep
  file-level imports are legal only *within* a subpackage (relative
  imports like `from .constants import ...`, which is how each
  `__init__.py` is built).
- **Never rename on import** (`import ... as`) for diffpes names — no
  `KB_EV_PER_K as _KB`, no `_N_ORBITALS as _NORBS`. An alias creates a
  second name for the same constant that grep, listings, and reviewers
  must chase. (Community-canonical module aliases like `jnp`/`np`/`plt`
  and the `ndarray as NDArray` casing shim for jaxtyping are the only
  exceptions.)
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
is checked by `interrogate` (`fail-under = 90`). Do **not** invent section
headers — the allowed set is the numpydoc sections plus exactly three house
extensions: `Extended Summary` and `Routine Listings` (modules and
`__init__.py`, above) and `Implementation Logic` (functions, below).

`pydoclint` sets `check-return-types = false` because jaxtyping shape strings
(e.g. `Float[Array, "nkpt nband"]`) are core signature syntax that pydoclint
cannot reliably parse for return-type comparison. Do not degrade a correct
jaxtyping annotation to satisfy that comparison. Argument types and order,
plus required `Returns`/`Yields` sections, remain enforced.

#### Prose Style: Simplified Technical English (ASD-STE100)

All prose in this repository conforms to
[ASD-STE100 Simplified Technical English](https://www.asd-ste100.org/):
every docstring, every markdown file (`README.md`, `docs/` guides,
`CHANGELOG.md`), and every tutorial markdown cell. STE exists to make
technical text impossible to misread. The rules that do the most work here:

- **Keep sentences short.** Maximum 20 words for an instruction, 25 for a
  description. One topic per sentence. One instruction per sentence.
- **Use the active voice and name the agent.** "The function computes the
  spectrum", not "the spectrum is computed".
- **Use the present tense for descriptions** and the imperative for
  instructions ("Compute the weights", "Do not tape the recursion").
- **One term, one meaning.** Use the same word for the same concept
  everywhere; do not rotate synonyms (pick "compute" and stay with it —
  do not alternate "calculate" and "evaluate" for the same operation).
  The verbatim summary-line rule already enforces this for API
  descriptions.
- **Keep the articles.** Write "the eigenvalues of the Hamiltonian", not
  telegraph-style "eigenvalues of Hamiltonian".
- **No noun clusters longer than three nouns** — break them up with
  prepositions ("the width of the Voigt profile", not "Voigt profile
  width parameter value").
- **No idioms or figures of speech.** They do not survive translation, and
  they do not survive a tired reader at a beamline at 3 a.m.
- **Technical names are exempt.** Domain terms (ARPES, PyTree, Kramers
  doublet, Gaunt coefficient, `jnp.where`) are STE technical names and
  are always permitted — used consistently, with one spelling.

Full STE dictionary compliance is a review-time discipline, not a tooling
gate; the specification is free on registration from asd-ste100.org. When
a reviewer flags a sentence as non-STE, simplify it — do not defend it.

#### Module Docstrings

Each module starts with a one-line summary, an `Extended Summary`, a
`Routine Listings` section cross-referencing every public object, and a
`Notes` section where relevant. For a package `__init__.py`, the
`Extended Summary` must list **every submodule `.py` file** (as
`- :mod:`name`` entries with a one-line description) — when you add a new
submodule, add it to that listing in the same change.

```python
# src/diffpes/radial/__init__.py
"""Differentiable radial primitives for photoemission matrix elements.

Extended Summary
----------------
This subpackage provides the radial building blocks of the matrix-element
engine: spherical Bessel functions, bound and continuum radial
wavefunctions, and the quadrature that contracts them into radial
integrals.

The submodules are organized as follows:

- :mod:`bessel`
    Spherical Bessel functions in JAX.
- :mod:`integrate`
    Radial quadrature for matrix-element integrals.
- :mod:`wavefunctions`
    Bound and continuum radial wavefunctions.

Routine Listings
----------------
:func:`radial_integral`
    Contract radial wavefunctions against the final state.
:func:`spherical_bessel_jl`
    Evaluate spherical Bessel function j_l(x).
"""
```

The one-line description under each `Routine Listings` entry is the
**verbatim** summary line of that symbol's own docstring; the `- :mod:`
description is the verbatim summary line of that submodule's docstring.

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

Every function docstring answers three questions, in Simplified Technical
English: what the function **ingests** (`Parameters`), what it **outputs**
(`Returns`), and **how the process happens inside** (`Implementation
Logic`, or `Notes` for one-formula functions). A docstring that leaves any
of the three unanswered is incomplete, independent of what pydoclint
accepts.

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

##### Section order

A function docstring uses these sections, in this order, omitting any that
do not apply:

1. Summary line
2. Extended summary (untitled prose, directly after the summary)
3. `:see:` test cross-reference
4. `Implementation Logic`
5. `Parameters`
6. `Returns` (or `Yields` for generators)
7. `Raises`
8. `Notes`
9. `References`
10. `See Also`
11. `Examples`

##### Summary line

- A **single imperative sentence** ending in a period, fitting on one line:
  "Compute normalized Gaussian broadening profile." — never "This function
  computes…", never a restatement of the parameter list.
- This exact sentence is copied **verbatim** into the module's
  `Routine Listings` and the subpackage `__init__.py`'s `Routine Listings`
  (the three-places rule). Changing it means changing all three.

##### Extended summary

- One or two short paragraphs of *what and in which regime*: the physics
  quantity computed, the approximation used, and its domain of validity
  (e.g. "using the pseudo-Voigt method of Thompson, Cox & Hastings (1987),
  accurate to better than 1% relative error"). Equations inline via
  `:math:`. The *how* does not belong here — it goes in
  `Implementation Logic`.

##### `:see:` cross-reference

- Every public object carries `:see: :class:`~.test_<module>.Test<Symbol>``
  pointing to its test class; the test class carries the matching
  back-reference. The pair is maintained together — renaming either side
  updates both.

##### `Implementation Logic` (house section)

- Required for any function whose body is more than a one-formula
  transcription; short functions may fold the algorithm into `Notes`
  instead.
- Format: **numbered bold steps**, each opening with a `::` literal block
  quoting the actual expressions, followed by indented prose explaining
  *why* that step exists:

  ```
  1. **Compute normalization factor**::

         norm_factor = sqrt(2 * pi) * sigma

     This prefactor ensures the profile integrates to unity.
  ```
- The steps must stay in sync with the body — a reviewer reads them
  side-by-side with the code. Stale Implementation Logic is a defect on the
  same footing as a stale listing.

##### `Parameters`

- One entry per signature parameter, **in signature order**, numpydoc
  `name : type` form where the type is spelled exactly as annotated
  (`Float[Array, "nkpt nband"]`, `ScalarFloat`).
- **Units on every physical quantity** ("Photon energy in eV", "In-plane
  momentum in 1/Angstrom"), and degrees vs radians stated explicitly for
  every angle.
- State defaults in prose: "Default 15.0."
- **Mark static (non-traced) parameters** — anything passed via
  `static_argnames`, Python `int`/`str`/`bool` values that drive shapes or
  control flow, and values landing in `eqx.field(static=True)`:
  *"(**static** — a compile-time constant; changing it triggers
  retracing)"*.
- PyTree arguments name their type with a `:class:` reference; do **not**
  re-document the PyTree's fields — that documentation lives once, on the
  type.

##### `Returns` / `Yields`

- Name each return value after the **type-annotated variable actually
  returned** (assign-before-returning), so the docstring, the body, and the
  signature agree; `name : type` with units. A `Tuple[...]` return gets one
  named entry per element, in order.

##### `Raises`

- Document **every explicit raise**: `ValueError` for static validation
  (with the violated condition), and `EquinoxRuntimeError` for traced
  `eqx.error_if` checks (with the runtime condition, e.g. "If ``sigma`` and
  ``gamma`` are simultaneously zero"). Do not document beartype/jaxtyping
  rejections — the typing machinery is implicit.

##### `Notes`

- Physics caveats and approximation limits that a *user* needs (validity
  ranges, convention pins by reference to the physics canon — never re-pin
  a convention locally).
- **Differentiability notes are mandatory where relevant**: which
  parameters carry gradients, how `safe_*` guards behave at boundary rays,
  and any known zero-gradient plateau (e.g. constant extrapolation outside
  a tabulation grid). A *documented* zero gradient is a stated limitation;
  an undocumented one is a bug.

##### `References`

- Numpydoc footnotes (`.. [1] Author, "Title", Journal Vol, pages
  (year).`), cited in the text as `[1]_`.
- **Footnote labels must be unique across the whole module**, not just the
  docstring: `automodule` renders every docstring of a module on one page,
  and two functions both using `.. [1]` collide. Continue numbering across
  the module (`voigt` uses `[1]`, `yeh_lindau_weights` uses `[2]`).

##### `See Also`

- Related public functions as `name : one-line description` — use the
  target's verbatim summary line where it fits.

##### `Examples`

- Doctest format, deterministic, cheap (CPU, small arrays). Ruff formats
  doctest code (`docstring-code-format`), and the rendered docs display it.

##### Class docstrings (`eqx.Module` PyTrees)

```python
class SelfEnergyConfig(eqx.Module):
    """Configure energy-dependent lifetime broadening.

    Carries the parameters of the imaginary self-energy
    :math:`\\Gamma(E)` used to build Lorentzian linewidths.

    :see: :class:`~.test_self_energy.TestSelfEnergyConfig`

    Attributes
    ----------
    gamma_0 : Float[Array, ""]
        Constant offset of :math:`\\Gamma(E)` in eV.
    mode : str
        Evaluation mode selector (**static** — stored via
        ``eqx.field(static=True)``; changing it triggers retracing).

    See Also
    --------
    make_self_energy_config : Validated factory for this type.
    """
```

- Summary line and extended summary follow the same rules as functions;
  `:see:` points at the type's test class.
- **`Attributes` documents every field** in declaration order —
  `name : type` with units — and flags every `eqx.field(static=True)` field
  as **static**.
- No `__init__` docstring: Equinox generates the constructor, and the
  construction contract is documented once, on the `make_*` factory.
  `See Also` names that factory.
- `Methods` section only if the class exposes public methods.

##### Factory (`make_*`) docstrings

- The factory's docstring **is the validation contract**: state which
  checks are static (`raise ValueError`, resolved at trace time) and which
  are traced (`eqx.error_if`, raised at runtime under `jit`), and mirror
  both in `Raises`. `Returns` names the constructed instance variable.

##### Private objects and raw strings

- `_`-prefixed helpers need at least a summary line; helpers doing real
  numerics (recurrences, quadratures, Taylor seeds) carry full `Parameters`
  / `Returns` / `Notes` — private code is exempt from the three-places
  rule, not from being understandable.
- Use a raw string (`r"""`) the moment a docstring contains a backslash —
  matrix-element and self-energy docstrings usually do.

### Code Style

Style is enforced by Ruff (`line-length = 79`, `target-version = "py312"`,
double quotes). The active lint rule set includes `D, E, F, B, I, N, UP, ANN,
S, A, C4, PIE, PT, RET, SIM, ARG, ERA, PL`. Key conventions:

- **Variable Names**: descriptive `snake_case`; long names over abbreviations
  (`photoemission_intensity`, not `pi`). Scientific single-letter symbols
  (`G`, `L`, `S`) are permitted where they mirror the physics.
- **No inline comments in `src/` unless absolutely necessary.** All
  explanation lives in the docstring, which states — in Simplified
  Technical English — what the function ingests (`Parameters`), what it
  outputs (`Returns`), and how the process happens inside
  (`Implementation Logic`). The only sanctioned comments are tooling
  directives (`# noqa: <rule>`, `# type: ignore[<rule>]`) and a one-line
  *why* that cannot live in the docstring (e.g. a workaround pinned to an
  upstream issue). A comment that narrates *what* the next line does is a
  defect — delete it or move the content into the docstring.
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
    ├── test_inout/test_chgcar.py
    ├── test_inout/test_doscar.py
    ├── test_inout/test_eigenval.py
    ├── test_inout/test_kpoints.py
    ├── test_inout/test_poscar.py
    ├── test_inout/test_procar.py
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

**External truths, never diffpes's own outputs.** A verification test compares
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

### Test Code Conventions

Tests are first-class source: the same style discipline as `src/` applies —
with a few test-specific adaptations:

- **Type-hint test bodies and helpers exactly as in `src/`.** Every test
  method is `def test_*(self) -> None:`; annotate intermediate variables; the
  assign-before-returning rule applies to any helper that returns data.
  Shared helpers carry full `jaxtyping` annotations (and
  `@jaxtyped(typechecker=beartype)` where arrays flow).
- **Document *what* and *how* on every test, class, and module (numpydoc).**
  A test's docstring is its specification, not a label. Open with the
  imperative summary line, then an `Extended Summary` stating **what** is
  verified (the property, invariant, or expected value — with units and
  tolerances), and a `Notes` section describing **how** (inputs/fixtures, the
  assertion strategy, and the `jit`/`grad`/`vmap` variant exercised). The
  **module** docstring summarises that file's coverage; each **`Test<Symbol>`
  class** docstring names the symbol under test and the scope of its cases.
- **Test docstrings are published documentation.** The test suite is rendered
  as a *Testing / Validation* reference in the Sphinx docs, so these
  docstrings are user-facing documentation of *what the library guarantees
  and how each guarantee is checked*. Write them as reader-facing prose; the
  `:see:` cross-reference makes the source ↔ test link navigable in **both**
  directions in the rendered docs.
- **No `__all__` or `Routine Listings` in test modules.** Tests are not a
  public API, so the three-places rule does **not** apply; a test module needs
  only a one-line summary + extended-summary docstring.
- **Private helpers are `_`-prefixed and local; reused fixtures go in the
  shared helper modules** (`tests/_factories.py`, `tests/_assertions.py`,
  `tests/_types.py`), not copy-pasted across files.

Example:

```python
import chex
import jax.numpy as jnp
from jaxtyping import Array, Float

import diffpes


class TestFermiFunction(chex.TestCase):
    """Validate :func:`~diffpes.simul.fermi_function`.

    Covers the Fermi-Dirac occupation across the ARPES temperature range:
    known-value accuracy at the Fermi level, shape, and finiteness.

    :see: :func:`~diffpes.simul.fermi_function`
    """

    def test_half_occupation_at_fermi_level(self) -> None:
        """Occupation at the Fermi level equals 1/2 at any temperature.

        Confirms ``fermi_function`` reproduces the analytic Fermi-Dirac
        value :math:`f(E_F) = 1/2` independent of temperature (the *what*).

        Notes
        -----
        Evaluates ``fermi_function`` at ``E = E_F = 0`` eV for temperatures
        [10, 100, 300] K and asserts shape ``(3,)``, finiteness, and
        closeness to 0.5 at ``rtol=1e-12`` (the *how*).
        """
        temperatures: Float[Array, "3"] = jnp.array([10.0, 100.0, 300.0])
        occupations: Float[Array, "3"] = diffpes.simul.fermi_function(
            energy=jnp.zeros(3),
            fermi_energy=0.0,
            temperature=temperatures,
        )

        chex.assert_shape(occupations, (3,))
        chex.assert_tree_all_finite(occupations)
        chex.assert_trees_all_close(occupations, 0.5 * jnp.ones(3), rtol=1e-12)
```

The `:see:` pair is matched: the source symbol points forward to
`Test<Symbol>`, and the test class points back to the symbol — add the
back-reference whenever you add the forward one, and renaming either side
means updating both.

### Running Tests

```bash
# Run the whole suite
pytest

# Run a single module / class / test
pytest tests/test_diffpes/test_radial/test_bessel.py
pytest tests/test_diffpes/test_radial/test_bessel.py::TestSphericalBesselJl

# Coverage
pytest tests/ --cov=src/diffpes --cov-report=term-missing
```

## Tutorial Notebooks

Tutorials live in `tutorials/` as Jupyter notebooks paired with Jupytext
percent scripts (`.ipynb` plus `.py`) so they can be edited while keeping
reviewable source diffs. **Explanation lives in markdown cells, not code
comments** — narrative, motivation, and physics belong in markdown blocks;
keep code cells comment-free. Markdown cells follow the same ASD-STE100
prose rules as all other prose in the repository. After editing a paired notebook, sync and strip
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

Two files are **generated by pre-commit hooks — do not edit them by hand**:
`.github/badges/loc.json` (the lines-of-code badge) and `requirements.txt`
(exported from `uv.lock` so the GitHub dependency graph, which does not
read `uv.lock` yet, sees a supported manifest). Both regenerate locally at
commit time; no CI job writes to the repository.

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
the package version (CalVer, e.g. `2026.06.01`; note PEP 440 normalizes it to
`2026.6.1` in built artifacts).

### Building and Releasing

Packaging is **uv end-to-end**: the build backend is `uv_build` (see
`[build-system]` in `pyproject.toml`) and releases go out with `uv publish` —
no `setuptools`, `build`, or `twine` anywhere.

```bash
# Build the sdist and wheel into dist/
uv build

# Sanity-check the artifacts
python -m zipfile -l dist/diffpes-*.whl

# Publish to PyPI (uses a PyPI API token)
UV_PUBLISH_TOKEN=<pypi-token> uv publish
```

Release checklist:

1. Bump `[project].version` (CalVer) and update `CHANGELOG.md` in the same
   commit.
2. Run the full wall (`ruff check src/ tests/`, `pydoclint src/`, `ty check`,
   `pytest`) at the release commit.
3. `uv build` from a clean tree; verify the wheel contains the full
   `diffpes/` package and the metadata carries `License-Expression: MIT`.
4. Tag the release commit (`v<version>`), then `uv publish`.

## Getting Help

- **Questions:** Open a discussion or issue
- **Documentation:** Check the rendered docs (Read the Docs)

Thank you for contributing to diffpes and advancing differentiable ARPES
simulation!
