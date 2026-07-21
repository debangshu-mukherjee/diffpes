r"""Provide ARPES simulation functions at six complexity levels.

Extended Summary
----------------
The subpackage provides a complete ARPES simulation pipeline from pseudo-Voigt
convolution to full polarization-dependent dipole matrix element
calculations. Also exports broadening functions, cross-section
models, polarization utilities, and orbital angular momentum.

The following list describes the submodules:

- :mod:`broadening`
    Compute energy broadening functions for ARPES simulations.
- :mod:`crosssections`
    Compute photoionization cross-section weights for ARPES.
- :mod:`expanded`
    Run expanded-input workflows for ARPES simulation.
- :mod:`forward`
    Run an end-to-end differentiable ARPES forward model.
- :mod:`oam`
    Compute orbital angular momentum.
- :mod:`polarization`
    Compute photon polarization and dipole matrix elements.
- :mod:`resolution`
    Apply momentum resolution broadening to ARPES simulations.
- :mod:`self_energy`
    Evaluate energy-dependent self-energy for ARPES simulations.
- :mod:`spectrum`
    Simulate ARPES spectra at six complexity levels.
- :mod:`workflow`
    Run high-level workflows for VASP-to-ARPES simulation.

Routine Listings
----------------
:func:`apply_momentum_broadening`
    Convolve I(k, E) with a Gaussian in k-space.
:func:`build_efield`
    Compute electric field vector from polarization config.
:func:`build_polarization_vectors`
    Construct s- and p-polarization basis vectors.
:func:`compute_oam`
    Compute orbital angular momentum z-component.
:func:`dipole_matrix_elements`
    Compute dipole matrix elements for all 9 orbitals.
:func:`evaluate_self_energy`
    Evaluate the imaginary self-energy :math:`\Gamma(E)`.
:func:`fermi_dirac`
    Compute Fermi-Dirac distribution value.
:func:`gaussian`
    Compute normalized Gaussian broadening profile.
:func:`heuristic_weights`
    Compute heuristic orbital weights based on photon energy.
:func:`load_vasp_context`
    Load a simulation-ready context from VASP output files.
:func:`photon_wavevector`
    Build the unit photon wavevector from incidence angles.
:func:`prepare_projection`
    Prepare orbital projections for simulation.
:func:`run_vasp_workflow`
    Run an end-to-end VASP-to-ARPES workflow in one call.
:func:`simulate_advanced`
    Simulate ARPES with Gaussian broadening and polarization rules.
:func:`simulate_advanced_expanded`
    Run advanced-level ARPES simulation from plain arrays.
:func:`simulate_basic`
    Simulate ARPES spectrum with Gaussian broadening and heuristic weights.
:func:`simulate_basic_expanded`
    Run basic-level ARPES simulation from plain arrays.
:func:`simulate_basicplus`
    Simulate ARPES with Gaussian broadening and Yeh-Lindau cross-sections.
:func:`simulate_basicplus_expanded`
    Run basicplus-level ARPES simulation from plain arrays.
:func:`simulate_context`
    Run a level-dispatched simulation from a loaded workflow context.
:func:`simulate_expanded`
    Dispatch an expanded-input simulation by complexity level.
:func:`simulate_expert`
    Simulate ARPES with Voigt broadening and dipole matrix elements.
:func:`simulate_expert_expanded`
    Run expert-level ARPES simulation from plain arrays.
:func:`simulate_novice`
    Simulate ARPES spectrum with Voigt broadening and uniform weights.
:func:`simulate_novice_expanded`
    Run novice-level ARPES simulation from plain arrays.
:func:`simulate_soc`
    Simulate ARPES with spin-orbit coupling (spin-dependent intensity).
:func:`simulate_soc_expanded`
    Run SOC (spin-orbit coupling) ARPES simulation from plain arrays.
:func:`simulate_tb_radial`
    Run the end-to-end differentiable ARPES forward model.
:func:`voigt`
    Compute a normalized Thompson-Cox-Hastings pseudo-Voigt profile.
:func:`yeh_lindau_weights`
    Compute Yeh-Lindau cross-section weights per orbital.

Notes
-----
All simulation functions are JAX-compatible and use ``jax.vmap``
for vectorized evaluation across k-points and bands.
"""

from .broadening import fermi_dirac, gaussian, voigt
from .crosssections import heuristic_weights, yeh_lindau_weights
from .expanded import (
    simulate_advanced_expanded,
    simulate_basic_expanded,
    simulate_basicplus_expanded,
    simulate_expanded,
    simulate_expert_expanded,
    simulate_novice_expanded,
    simulate_soc_expanded,
)
from .forward import simulate_tb_radial
from .oam import compute_oam
from .polarization import (
    build_efield,
    build_polarization_vectors,
    dipole_matrix_elements,
    photon_wavevector,
)
from .resolution import apply_momentum_broadening
from .self_energy import evaluate_self_energy
from .spectrum import (
    simulate_advanced,
    simulate_basic,
    simulate_basicplus,
    simulate_expert,
    simulate_novice,
    simulate_soc,
)
from .workflow import (
    load_vasp_context,
    prepare_projection,
    run_vasp_workflow,
    simulate_context,
)

__all__: list[str] = [
    "apply_momentum_broadening",
    "build_efield",
    "build_polarization_vectors",
    "compute_oam",
    "dipole_matrix_elements",
    "evaluate_self_energy",
    "fermi_dirac",
    "gaussian",
    "heuristic_weights",
    "load_vasp_context",
    "photon_wavevector",
    "prepare_projection",
    "run_vasp_workflow",
    "simulate_advanced",
    "simulate_advanced_expanded",
    "simulate_basic",
    "simulate_basic_expanded",
    "simulate_basicplus",
    "simulate_basicplus_expanded",
    "simulate_context",
    "simulate_expanded",
    "simulate_expert",
    "simulate_expert_expanded",
    "simulate_novice",
    "simulate_novice_expanded",
    "simulate_soc",
    "simulate_soc_expanded",
    "simulate_tb_radial",
    "voigt",
    "yeh_lindau_weights",
]
