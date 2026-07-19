r"""ARPES simulation functions at six complexity levels.

Extended Summary
----------------
Provides a complete ARPES simulation pipeline from basic Voigt
convolution to full polarization-dependent dipole matrix element
calculations. Also exports broadening functions, cross-section
models, polarization utilities, and orbital angular momentum.

The submodules are organized as follows:

- :mod:`broadening`
    Energy broadening functions for ARPES simulations.
- :mod:`crosssections`
    Photoionization cross-section weights for ARPES.
- :mod:`expanded`
    Expanded-input workflows for ARPES simulation.
- :mod:`forward`
    End-to-end differentiable ARPES forward model.
- :mod:`oam`
    Orbital angular momentum calculation.
- :mod:`polarization`
    Photon polarization and dipole matrix element calculations.
- :mod:`resolution`
    Momentum resolution broadening for ARPES simulations.
- :mod:`self_energy`
    Energy-dependent self-energy evaluation for ARPES simulations.
- :mod:`spectrum`
    ARPES spectrum simulation at six complexity levels (incl. spin-orbit).
- :mod:`workflow`
    High-level workflow helpers for VASP-to-ARPES simulation.

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
:func:`make_expanded_simulation_params`
    Build simulation parameters with auto-derived energy window.
:obj:`ORBITAL_DIRS_NORMALIZED`
    Unit-normalized orbital direction vectors in VASP ordering.
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
    Compute normalized Voigt profile via pseudo-Voigt approximation.
:class:`WorkflowContext`
    Container for parsed VASP inputs used by workflow helpers.
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
    make_expanded_simulation_params,
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
    ORBITAL_DIRS_NORMALIZED,
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
    WorkflowContext,
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
    "make_expanded_simulation_params",
    "ORBITAL_DIRS_NORMALIZED",
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
    "WorkflowContext",
    "yeh_lindau_weights",
]
