# Changelog

All notable changes to diffpes are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses calendar versioning.

## [Unreleased]

### Changed

- Scoped the Plan 01 pre-commit Ruff hooks to their source, test, and project
  metadata floor, and enabled manual CI dispatch for gate verification.
- Replaced every registered `NamedTuple` carrier with a types-owned
  `equinox.Module`, moved all carrier factories to `diffpes.types`, and
  updated HDF5 serialization for explicit array and static fields.
- Consolidated declarative constants, orbital conventions, parser schemas,
  and lookup tables under `diffpes.types`.
- Moved the workflow context and its projection and DOS aliases into
  `diffpes.types`, and converted the context PyTree to an Equinox module.
- Normalized the project, repository links, documentation, and release
  surfaces to the lowercase `diffpes` name.
- Activated the two-tier factory validation wall: structural violations now
  raise `ValueError`, while traced value violations use value-threaded
  `equinox.error_if` checks that survive JIT compilation.

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
- Added deterministic pre-refactor novice and tight-binding radial regression
  references, including standing zeta-gradient baselines and provenance.

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

## [2026.03.01] - 2026-07-13

### Added

- Established the initial differentiable ARPES package release.

[unreleased]: https://github.com/debangshu-mukherjee/diffpes/compare/v2026.03.01...HEAD
[2026.03.01]: https://github.com/debangshu-mukherjee/diffpes/releases/tag/v2026.03.01
