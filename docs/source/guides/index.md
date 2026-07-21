# diffpes Theory and Architecture Guides

This documentation provides comprehensive coverage of the physics and software architecture underlying diffpes, a JAX-based framework for differentiable Angle-Resolved PhotoEmission Spectroscopy (ARPES) simulation.

## Target Audience

These guides are written for **physics researchers** working with ARPES who want to understand:

- The photoemission physics implemented at each simulation level
- How electronic-structure data flows through the simulation pipeline
- The physical meaning of simulation parameters and outputs
- How differentiability enables inverse recovery of band-structure parameters

## Guide Overview

### Physics Foundations

| Guide | Description |
|-------|-------------|
| [ARPES Geometry and Kinematics](arpes-geometry-and-kinematics.md) | Photoemission geometry, energy and momentum conservation, and detector coordinates |
| [Simulation Levels](simulation-levels.md) | The six fidelity levels from Voigt convolution to polarization-dependent matrix elements |
| [Matrix Elements and Polarization](matrix-elements-and-polarization.md) | Radial integrals, Gaunt coefficients, spherical harmonics, and light-polarization effects |
| [Spectral Broadening and Self-Energy](spectral-broadening-and-self-energy.md) | Voigt profiles, resolution convolution, and self-energy models |

### Data and Architecture

| Guide | Description |
|-------|-------------|
| [PyTree Architecture](pytree-architecture.md) | Equinox data structures enabling GPU acceleration and autodiff |
| [JAX Transformability and Gradients](jax-transformability-and-gradients.md) | Which of `grad`/`vmap`/`jit` are supported where, and gradient flow through the forward model |
| [Certified Forward Models](certified-forward-models.md) | Bounded scientific claims, provenance, differentiable evidence, information flow, and portable records |
| [VASP Data Ingestion](vasp-data-ingestion.md) | Parsing POSCAR, EIGENVAL, KPOINTS, DOSCAR, PROCAR, and CHGCAR into PyTrees |
| [Expanded Wrappers and Conventions](expanded-wrappers-and-conventions.md) | The `simulate_*_expanded` wrapper family and its argument conventions |

## Quick Start

For hands-on examples, see the [tutorials](../tutorials/index.md).

## Mathematical Notation

Throughout these guides, we use:

- $\mathbf{k}$ for wavevectors (in $\text{Å}^{-1}$)
- $E_B$ for binding energy and $E_F$ for the Fermi level (in eV)
- $h\nu$ for photon energy (in eV)
- $(n, l, m)$ for orbital quantum numbers
- $\theta$ for polar emission angle
- $\phi$ for azimuthal angle
- $\Sigma(\omega)$ for the electron self-energy

Equations are rendered using LaTeX notation compatible with GitHub and MathJax.
