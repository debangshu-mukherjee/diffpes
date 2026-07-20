"""Type definitions and factory functions for diffpes.

Extended Summary
----------------
Provides PyTree-compatible data structures and their factory
functions for representing ARPES simulation data including crystal
geometry, band structures, orbital projections, simulation
parameters, and polarization configurations. Fields that
participate in autodiff are stored as JAX array children, while
shape-determining values (e.g., ``SimulationParams.fidelity``)
and code-path selectors (e.g., ``PolarizationConfig.polarization_type``)
are stored as auxiliary data so they remain concrete at trace time
and trigger recompilation only when changed.

The submodules are organized as follows:

- :mod:`aliases`
    Scalar type aliases for JAX-compatible numeric types.
- :mod:`bands`
    Band structure and orbital projection data structures.
- :mod:`constants`
    Dependency-light numerical and physical constants used by diffpes.
- :mod:`dos`
    Density of states data structures.
- :mod:`geometry`
    Crystal geometry data structure for VASP crystal structures.
- :mod:`kpath`
    K-point path information data structure.
- :mod:`params`
    Simulation parameter data structures.
- :mod:`orbital_constants`
    Shared orbital ordering and direction conventions.
- :mod:`tables`
    Small numerical lookup tables used by simulations.
- :mod:`radial_params`
    Radial wavefunction parameter data structures.
- :mod:`self_energy`
    Self-energy configuration data structures.
- :mod:`tb_model`
    Tight-binding model and diagonalized band data structures.
- :mod:`volumetric`
    Volumetric data structures for VASP CHGCAR files.
- :mod:`vasp_constants`
    VASP parser and serialization conventions.
- :mod:`context`
    High-level VASP workflow input data structures.

Routine Listings
----------------
:class:`ArpesSpectrum`
    PyTree for ARPES simulation output.
:class:`BandStructure`
    PyTree for electronic band structure.
:obj:`BOHR_TO_ANGSTROM`
    Bohr radius in Angstrom.
:class:`CrystalGeometry`
    PyTree for crystal geometry from VASP POSCAR.
:class:`DensityOfStates`
    PyTree for density of states.
:class:`DiagonalizedBands`
    PyTree for diagonalized electronic structure.
:obj:`DosType`
    Density-of-states carrier union used by workflow contexts.
:class:`FullDensityOfStates`
    PyTree for complete density of states with spin and PDOS.
:obj:`HBAR_C_EV_A`
    Reduced Planck constant times c in eV Angstrom.
:obj:`HBAR_EV_S`
    Reduced Planck constant in eV s.
:obj:`KB_EV_PER_K`
    Boltzmann constant in eV per kelvin.
:class:`KPathInfo`
    PyTree for k-point path metadata.
:obj:`L_MAX`
    Maximum angular momentum supported by the precomputed table.
:obj:`ME_EV`
    Electron rest energy in eV.
:obj:`NonJaxNumber`
    Union of ``int``, ``float``, and ``complex``.
:obj:`ORBITAL_DIRS_NORMALIZED`
    Unit-normalized orbital directions in VASP orbital order.
:class:`OrbitalBasis`
    PyTree for orbital quantum number metadata.
:class:`OrbitalProjection`
    PyTree for orbital-resolved band projections.
:class:`PolarizationConfig`
    PyTree for photon polarization geometry.
:obj:`ProjectionType`
    Orbital-projection carrier union used by workflow contexts.
:class:`SOCVolumetricData`
    PyTree for volumetric data from SOC CHGCAR files.
:obj:`ScalarBool`
    Union of ``bool`` and ``Bool[Array, " "]``.
:obj:`ScalarComplex`
    Union of ``complex`` and ``Complex[Array, " "]``.
:obj:`ScalarFloat`
    Union of ``float`` and ``Float[Array, " "]``.
:obj:`ScalarInteger`
    Union of ``int`` and ``Int[Array, " "]``.
:obj:`ScalarNumeric`
    Union of ``int``, ``float``, ``complex``, and ``Num[Array, " "]``.
:class:`SelfEnergyConfig`
    PyTree for energy-dependent self-energy (lifetime broadening).
:class:`SimulationParams`
    PyTree for ARPES simulation parameters.
:class:`SlaterParams`
    PyTree for Slater radial wavefunction parameters.
:class:`SpinBandStructure`
    PyTree for spin-resolved electronic band structure.
:class:`SpinOrbitalProjection`
    PyTree for orbital projections with mandatory spin data.
:class:`TBModel`
    PyTree for tight-binding model parameters (legacy).
:class:`VolumetricData`
    PyTree for volumetric grid data from CHGCAR.
:class:`WorkflowContext`
    PyTree bundling parsed inputs for high-level VASP workflows.
:func:`make_arpes_spectrum`
    Create a validated ``ArpesSpectrum`` instance.
:func:`make_band_structure`
    Create a validated ``BandStructure`` instance.
:func:`make_crystal_geometry`
    Create a validated CrystalGeometry instance.
:func:`make_density_of_states`
    Create a validated DensityOfStates instance.
:func:`make_diagonalized_bands`
    Create a validated ``DiagonalizedBands`` instance.
:func:`make_expanded_simulation_params`
    Build simulation parameters with an automatically derived energy window.
:func:`make_full_density_of_states`
    Create a validated ``FullDensityOfStates`` instance.
:func:`make_graphene_model`
    Create a graphene pz tight-binding model.
:func:`make_1d_chain_model`
    Create a 1D chain tight-binding model.
:func:`make_kpath_info`
    Create a validated KPathInfo instance.
:func:`make_orbital_basis`
    Create a validated ``OrbitalBasis`` instance.
:func:`make_orbital_projection`
    Create a validated ``OrbitalProjection`` instance.
:func:`make_polarization_config`
    Create a validated PolarizationConfig instance.
:func:`make_self_energy_config`
    Create a validated ``SelfEnergyConfig`` instance.
:func:`make_simulation_params`
    Create a validated SimulationParams instance.
:func:`make_slater_params`
    Create a validated ``SlaterParams`` instance.
:func:`make_soc_volumetric_data`
    Create a validated ``SOCVolumetricData`` instance.
:func:`make_spin_band_structure`
    Create a validated ``SpinBandStructure`` instance.
:func:`make_spin_orbital_projection`
    Create a validated ``SpinOrbitalProjection`` instance.
:func:`make_tb_model`
    Create a validated ``TBModel`` instance.
:func:`make_volumetric_data`
    Create a validated ``VolumetricData`` instance.
:func:`make_workflow_context`
    Create a ``WorkflowContext`` instance from parsed inputs.

Notes
-----
All structured carriers are immutable :class:`equinox.Module` PyTrees.
Array fields remain differentiable leaves, while shape and control-flow
metadata use ``equinox.field(static=True)``.
"""

from .aliases import (
    NonJaxNumber,
    ScalarBool,
    ScalarComplex,
    ScalarFloat,
    ScalarInteger,
    ScalarNumeric,
)
from .bands import (
    ArpesSpectrum,
    BandStructure,
    OrbitalProjection,
    SpinBandStructure,
    SpinOrbitalProjection,
    make_arpes_spectrum,
    make_band_structure,
    make_orbital_projection,
    make_spin_band_structure,
    make_spin_orbital_projection,
)
from .constants import (
    BOHR_TO_ANGSTROM,
    HBAR_C_EV_A,
    HBAR_EV_S,
    KB_EV_PER_K,
    L_MAX,
    ME_EV,
)
from .context import (
    DosType,
    ProjectionType,
    WorkflowContext,
    make_workflow_context,
)
from .dos import (
    DensityOfStates,
    FullDensityOfStates,
    make_density_of_states,
    make_full_density_of_states,
)
from .geometry import (
    CrystalGeometry,
    make_crystal_geometry,
)
from .kpath import (
    KPathInfo,
    make_kpath_info,
)
from .orbital_constants import ORBITAL_DIRS_NORMALIZED
from .params import (
    PolarizationConfig,
    SimulationParams,
    make_expanded_simulation_params,
    make_polarization_config,
    make_simulation_params,
)
from .radial_params import (
    OrbitalBasis,
    SlaterParams,
    make_orbital_basis,
    make_slater_params,
)
from .self_energy import (
    SelfEnergyConfig,
    make_self_energy_config,
)
from .tb_model import (
    DiagonalizedBands,
    TBModel,
    make_1d_chain_model,
    make_diagonalized_bands,
    make_graphene_model,
    make_tb_model,
)
from .volumetric import (
    SOCVolumetricData,
    VolumetricData,
    make_soc_volumetric_data,
    make_volumetric_data,
)

__all__: list[str] = [
    "ArpesSpectrum",
    "BandStructure",
    "BOHR_TO_ANGSTROM",
    "CrystalGeometry",
    "DensityOfStates",
    "DiagonalizedBands",
    "DosType",
    "FullDensityOfStates",
    "HBAR_C_EV_A",
    "HBAR_EV_S",
    "KB_EV_PER_K",
    "KPathInfo",
    "L_MAX",
    "ME_EV",
    "NonJaxNumber",
    "ORBITAL_DIRS_NORMALIZED",
    "OrbitalBasis",
    "OrbitalProjection",
    "PolarizationConfig",
    "ProjectionType",
    "SOCVolumetricData",
    "ScalarBool",
    "ScalarComplex",
    "ScalarFloat",
    "ScalarInteger",
    "ScalarNumeric",
    "SelfEnergyConfig",
    "SimulationParams",
    "SlaterParams",
    "SpinBandStructure",
    "SpinOrbitalProjection",
    "TBModel",
    "VolumetricData",
    "WorkflowContext",
    "make_arpes_spectrum",
    "make_band_structure",
    "make_crystal_geometry",
    "make_density_of_states",
    "make_diagonalized_bands",
    "make_expanded_simulation_params",
    "make_full_density_of_states",
    "make_graphene_model",
    "make_1d_chain_model",
    "make_kpath_info",
    "make_orbital_basis",
    "make_orbital_projection",
    "make_polarization_config",
    "make_self_energy_config",
    "make_simulation_params",
    "make_slater_params",
    "make_soc_volumetric_data",
    "make_spin_band_structure",
    "make_spin_orbital_projection",
    "make_tb_model",
    "make_volumetric_data",
    "make_workflow_context",
]
