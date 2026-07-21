# Changelog

All notable changes to diffpes are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses calendar versioning.

## [Unreleased]

### Changed

- Merged `diffpes.types.orbital_constants` and `diffpes.types.vasp_constants`
  into `diffpes.types.constants` and deleted the two modules (zero-legacy, no
  shims). All constants consumed across subpackages are now public, renamed by
  dropping their leading underscore (e.g. `_EPS` → `EPS`, `_N_ORBITALS` →
  `N_ORBITALS`, `_PHASE_LOSS_MESSAGE` → `PHASE_LOSS_MESSAGE`), and re-exported
  through `diffpes.types`; only module-internal intermediates remain private.
  `diffpes.types.constants` now imports JAX (orbital direction tables are
  device arrays) and is no longer dependency-light.
- Adopted the generalized import rule (see CONTRIBUTING): cross-subpackage
  imports must use the source subpackage's public surface
  (`from diffpes.<sub> import name`), never a file inside it; fixed the last
  offenders (`simul/workflow.py` deep imports into `diffpes.inout`).

- Scoped the Plan 01 pre-commit Ruff hooks to their source, test, and project
  metadata floor, and enabled manual CI dispatch for gate verification.
- Replaced every registered `NamedTuple` carrier with a types-owned
  `equinox.Module`, moved all carrier factories to `diffpes.types`, and
  updated HDF5 serialization for introspected array, nested-module, optional,
  and static fields. Carrier construction is now keyword-only; use
  `equinox.tree_at` instead of `NamedTuple._replace` for immutable updates.
- Consolidated declarative constants, orbital conventions, parser schemas,
  and lookup tables under `diffpes.types`.
- Moved the workflow context and its projection and DOS aliases into
  `diffpes.types`, and converted the context PyTree to an Equinox module.
- Normalized the project, repository links, documentation, and release
  surfaces to the lowercase `diffpes` name.
- Activated the two-tier factory validation wall: structural violations now
  raise `ValueError`, while traced value violations use value-threaded
  `equinox.error_if` checks that survive JIT compilation.
- Widened `read_eigenval(..., fermi_energy=...)` to `ScalarFloat` and kept
  workflow Fermi energies as traced scalar leaves instead of host floats.

### Added

- Added a tag-gated, uv-native PyPI Trusted Publishing workflow with wheel
  and source-distribution smoke tests.
- Added Equinox, Optimistix, Lineax, and Optax as the differentiable type,
  nonlinear-solver, linear-solver, and optimizer stack, following the stack
  decision adopted on 2026-07-13.
- Added Hypothesis and psutil to the test environment for property-based
  verification and runtime memory guards.
- Added a shared pytest runtime foundation with x64 enforcement, deterministic
  random keys, JAX cache cleanup, RSS leak limits, and xdist memory grouping.
- Added typed deterministic toy factories, strict numerical tree assertions,
  and an NPZ reference-comparison scaffold for the test suite.
- Added the program-wide gradient verification harness with scaled finite
  differences, Wirtinger checks, and zero-gradient tripwires.
- Added Python 3.12--3.14 GitHub Actions testing with informational Codecov
  uploads and lock-aligned Ruff and ty pre-commit hooks; install them with
  `uv run pre-commit install`.
- Added deterministic pre-refactor novice and tight-binding radial regression
  references, including standing zeta-gradient baselines and provenance.
- Added seven named gradient-safe math primitives with explicit guarded-set
  value and subgradient conventions.
- Added `pack_complex` and `unpack_complex` as the real-PyTree optimizer and
  complex-physics boundary, with a pinned JAX Wirtinger convention test.

### Removed

- Removed the unused `difftb` dependency and its broken editable
  `[tool.uv.sources]` path so diffpes installs as a standalone package.
- Retired Black, isort, jupyter-black, build, and Twine from the development
  environment; Ruff owns formatting and uv owns build and publish workflows.

### Fixed

- Restored Python 3.14 imports while beartype 0.22.9 still references the
  removed `collections.abc.ByteString` name.
- Corrected the supported Python range to `>=3.12,<3.15` and documented
  Python 3.12 support.
- Repaired project metadata by making JAX platform-independent, removing dead
  setuptools configuration, aligning interrogate with NumPy docstrings, and
  extending Ruff and runtime type-checking coverage to the test suite.
- Corrected real-to-complex Gaunt transformation coefficients to satisfy their
  complex-valued runtime type contract.
- Replaced the overflow-prone reciprocal-exponential Fermi-Dirac expression
  with a stable sigmoid, keeping values and gradients finite through the
  realistic-spectrum audit range.
- Made the Thompson-Cox-Hastings pseudo-Voigt implementation gradient-safe on
  both positive-width boundary rays and reject the undefined zero-width
  intersection eagerly and under JIT.

## [2026.03.01] - 2026-07-13

### Added

- Established the initial differentiable ARPES package release.

[unreleased]: https://github.com/debangshu-mukherjee/diffpes/compare/v2026.03.01...HEAD
[2026.03.01]: https://github.com/debangshu-mukherjee/diffpes/releases/tag/v2026.03.01
