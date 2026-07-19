# WP6.1 regression-reference manifest

> These files pin pre-refactor behavior, not correctness. They
> intentionally include known physics defects documented by Plan 01.
> Regenerate only with a stated physics or migration justification.

- Generation date: 2026-07-19
- Seed: `20260713`
- Device policy: CPU, JAX x64 enabled
- Platform: `Linux-5.15.0-185-generic-x86_64-with-glibc2.35`
- Python: `3.13.6`
- diffpes: `2026.6.1`
- JAX: `0.9.0.1`
- NumPy: `2.4.2`

## Factory calls

- `novice_toy`: `simulate_novice(toy_band_structure(key), toy_orbital_projection(key), toy_simulation_params(fidelity=512))`
- `tb_radial_graphene`: `simulate_tb_radial(toy_graphene_diagonalized(n_k=12)[1], toy_slater_params(), toy_simulation_params(fidelity=512), toy_polarization_config())`, plus intensity sum and zeta gradient
- `tb_radial_chain`: `simulate_tb_radial(toy_chain_diagonalized(n_k=16)[1], toy_slater_params(), toy_simulation_params(fidelity=512), toy_polarization_config())`, plus intensity sum and zeta gradient

## Artifacts

### `novice_toy.npz`

- SHA-256: `7585907bef8075904117b13506491ba488038154ff2ec331c5059a2a7ec5d56f`
- Arrays:
  - `leaf_000_intensity`: shape `(8, 512)`, dtype `float64`
  - `leaf_001_energy_axis`: shape `(512,)`, dtype `float64`

### `tb_radial_graphene.npz`

- SHA-256: `63c27089378207d8f1772c547fd9c47b674514d981dffd858fc462f5b5fa677c`
- Arrays:
  - `leaf_000_intensity`: shape `(12, 512)`, dtype `float64`
  - `leaf_001_energy_axis`: shape `(512,)`, dtype `float64`
  - `leaf_002_intensity_sum`: shape `()`, dtype `float64`
  - `leaf_003_zeta_gradient`: shape `(2,)`, dtype `float64`

### `tb_radial_chain.npz`

- SHA-256: `7ff7cac8477e5157d4752826faee4e643ffa0fe9ec5dc454869fa59b867686a2`
- Arrays:
  - `leaf_000_intensity`: shape `(16, 512)`, dtype `float64`
  - `leaf_001_energy_axis`: shape `(512,)`, dtype `float64`
  - `leaf_002_intensity_sum`: shape `()`, dtype `float64`
  - `leaf_003_zeta_gradient`: shape `(2,)`, dtype `float64`
