# Tutorials

Tutorial notebooks for diffpes. They are authored as MyST text notebooks
(Markdown files with executable code cells), so they render with live,
executed outputs in these docs while keeping reviewable plain-text sources
in git.

```{toctree}
:maxdepth: 1

quickstart
```

- [Quickstart](quickstart.md) — build a synthetic band structure, simulate
  ARPES spectra at two fidelity levels, and differentiate through the
  spectrometer model with `jax.grad`.

More worked, end-to-end examples are under development, such as:

- Loading VASP output and simulating a basic ARPES spectrum
- Stepping through the six simulation fidelity levels
- Polarization-dependent matrix element effects
- Gradient-based recovery of band-structure parameters from spectra

Also see the [guides](../guides/index.md) for theory and architecture
documentation, and the [API reference](../api/index.rst) for the complete
function-level documentation.
