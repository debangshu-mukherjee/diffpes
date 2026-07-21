# org.diffpes.model.arpes.tb_radial

Version: `0.1.0`.

Observable: `org.diffpes.observable.arpes.intensity`.

Implementation: `diffpes.simul.simulate_tb_radial`.

## Assumptions

- The model uses `dipole_approximation`.
- The model uses `independent_particle_initial_state`.
- The model uses `free_electron_final_state`.
- The model uses `slater_radial_basis`.
- The model uses `fermi_dirac_occupation`.
- The model uses `voigt_energy_resolution`.

## Conventions

- The model uses `org.diffpes.convention.energy.fermi_referenced_ev@1.0.0`.
- The model uses `org.diffpes.convention.length.angstrom@1.0.0`.
- The model uses `org.diffpes.convention.orbital.real_harmonics@1.0.0`.

## Domain

- `org.diffpes.domain.photon_energy.positive` uses `photon_energy_ev > 0` with `error` severity.
- `org.diffpes.domain.radial_grid.positive` uses `all(r_grid > 0)` with `error` severity.
