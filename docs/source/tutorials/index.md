# Tutorials

The diffpes tutorials use MyST text notebooks. These Markdown files contain
executable code cells. The documentation build executes the cells and shows
their outputs. Git stores the reviewable plain-text sources.

```{toctree}
:maxdepth: 1

quickstart
certified-forward-model
```

- [Quickstart](quickstart.md): Build a synthetic band structure. Simulate two
  ARPES fidelity levels and differentiate the spectrometer model with
  `jax.grad`.
- [Inspect and persist a certified forward run](certified-forward-model.md):
  Read bounded claims and differentiable evidence. Save canonical JSON and
  attach the same record to an HDF5 result.

The project is developing more complete examples:

- Loading VASP output and simulating a basic ARPES spectrum
- Stepping through the six simulation fidelity levels
- Polarization-dependent matrix element effects
- Gradient-based recovery of band-structure parameters from spectra

Read the [guides](../guides/index.md) for theory and architecture. Read the
[API reference](../api/index.rst) for complete function documentation.
