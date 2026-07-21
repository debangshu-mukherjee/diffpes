# Tutorials

Tutorial notebooks for diffpes. They are authored as MyST text notebooks
(Markdown files with executable code cells), so they render with live,
executed outputs in these docs while keeping reviewable plain-text sources
in git.

```{toctree}
:maxdepth: 1

quickstart
certified-forward-model
```

- [Quickstart](quickstart.md) — build a synthetic band structure, simulate
  ARPES spectra at two fidelity levels, and differentiate through the
  spectrometer model with `jax.grad`.
- [Inspect and persist a certified forward run](certified-forward-model.md) —
  read bounded claims and differentiable evidence, save canonical JSON, and
  attach the same record to an HDF5 result.

More worked, end-to-end examples are under development, such as:

- Loading VASP output and simulating a basic ARPES spectrum
- Stepping through the six simulation fidelity levels
- Polarization-dependent matrix element effects
- Gradient-based recovery of band-structure parameters from spectra

Also see the [guides](../guides/index.md) for theory and architecture
documentation, and the [API reference](../api/index.rst) for the complete
function-level documentation.
