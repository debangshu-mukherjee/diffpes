# Changelog

All notable changes to DiffPES are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses calendar versioning.

## [Unreleased]

### Added

- Added Equinox, Optimistix, Lineax, and Optax as the differentiable type,
  nonlinear-solver, linear-solver, and optimizer stack.

### Removed

- Removed the unused `difftb` dependency and its broken editable
  `[tool.uv.sources]` path so DiffPES installs as a standalone package.

### Fixed

- Corrected the supported Python range to `>=3.12,<3.15` and documented
  Python 3.12 support.

## [2026.03.01] - 2026-07-13

### Added

- Established the initial differentiable ARPES package release.

[unreleased]: https://github.com/debangshu-mukherjee/DiffPES/compare/v2026.03.01...HEAD
[2026.03.01]: https://github.com/debangshu-mukherjee/DiffPES/releases/tag/v2026.03.01
