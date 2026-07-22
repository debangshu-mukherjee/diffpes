# WP6.1 regression-reference manifest

> These files pin deterministic behavior, not independent physics
> truth.
> The tight-binding cases were repinned for Plan 04's basis-position
> gauge and carrier-native orbital bases.
> Regenerate only with a stated physics or migration
> justification.

- Generation date: 2026-07-22
- Seed: `20260713`
- Device policy: CPU, JAX x64 enabled
- Platform: `Linux-5.15.0-185-generic-x86_64-with-glibc2.35`
- Python: `3.13.6`
- diffpes: `2026.6.4`
- JAX: `0.9.0.1`
- NumPy: `2.4.2`

## Factory calls

- `novice_toy`: `simulate_novice(toy_band_structure(key), toy_orbital_projection(key), toy_simulation_params(fidelity=512))`
- `tb_radial_graphene`: `simulate_tb_radial(toy_graphene_diagonalized(n_k=12)[1], toy_slater_params(), toy_simulation_params(fidelity=512), toy_polarization_config())`, plus intensity sum and zeta gradient
- `tb_radial_chain`: `simulate_tb_radial(toy_chain_diagonalized(n_k=16)[1], make_slater_params(zeta=[1.625], orbital_basis=bands.basis), toy_simulation_params(fidelity=512), toy_polarization_config())`, plus intensity sum and zeta gradient

## Artifacts

### `novice_toy.npz`

- SHA-256: `7585907bef8075904117b13506491ba488038154ff2ec331c5059a2a7ec5d56f`
- Arrays:
  - `leaf_000_intensity`: shape `(8, 512)`, dtype `float64`
  - `leaf_001_energy_axis`: shape `(512,)`, dtype `float64`

### `tb_radial_graphene.npz`

- SHA-256: `aab46ed26e668d05be1adaaaffa4910941835168c22db9e419f376a7a9dcaa23`
- Arrays:
  - `leaf_000_intensity`: shape `(12, 512)`, dtype `float64`
  - `leaf_001_energy_axis`: shape `(512,)`, dtype `float64`
  - `leaf_002_intensity_sum`: shape `()`, dtype `float64`
  - `leaf_003_zeta_gradient`: shape `(2,)`, dtype `float64`

### `tb_radial_chain.npz`

- SHA-256: `7ecc4c2df60129764fc0696aced30282836eb157f4686a3bf1e5f69691a60b0e`
- Arrays:
  - `leaf_000_intensity`: shape `(16, 512)`, dtype `float64`
  - `leaf_001_energy_axis`: shape `(512,)`, dtype `float64`
  - `leaf_002_intensity_sum`: shape `()`, dtype `float64`
  - `leaf_003_zeta_gradient`: shape `(1,)`, dtype `float64`
