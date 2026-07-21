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
    Numerical, physical, orbital, and VASP-format constants for diffpes.
- :mod:`dos`
    Density of states data structures.
- :mod:`geometry`
    Crystal geometry data structure for VASP crystal structures.
- :mod:`kpath`
    K-point path information data structure.
- :mod:`params`
    Simulation parameter data structures.
- :mod:`tables`
    Small immutable numerical tables used by simulation routines.
- :mod:`radial_params`
    Radial wavefunction parameter data structures.
- :mod:`self_energy`
    Self-energy configuration data structures.
- :mod:`tb_model`
    Tight-binding model and diagonalized band data structures.
- :mod:`volumetric`
    Volumetric data structures for VASP CHGCAR files.
- :mod:`context`
    High-level VASP workflow input data structures.

Routine Listings
----------------
:class:`ArpesSpectrum`
    PyTree for ARPES simulation output.
:obj:`ATTR_AUX`
    HDF5 attribute name storing auxiliary PyTree data as JSON.
:obj:`ATTR_NONE`
    HDF5 attribute name listing PyTree fields stored as None.
:obj:`ATTR_TYPE`
    HDF5 attribute name storing the PyTree type name.
:obj:`BAND_LINE_MIN_VALUES`
    Minimum tokens on an EIGENVAL band line.
:obj:`BAND_LINE_SPIN_VALUES`
    Tokens on a spin-polarized EIGENVAL band line.
:obj:`BAND_NDIM`
    Expected dimensionality of band-energy arrays.
:class:`BandStructure`
    PyTree for electronic band structure.
:obj:`BOHR_TO_ANGSTROM`
    Bohr radius in Angstrom.
:obj:`COORDINATE_MODE_TOKENS`
    Recognized KPOINTS coordinate-mode tokens.
:obj:`CROSS_SECTION_ENERGIES`
    Photon-energy tabulation grid for the cross-section tables in eV.
:obj:`CROSS_SECTION_SIGMA_D`
    Yeh-Lindau d-subshell cross sections on the tabulation grid.
:obj:`CROSS_SECTION_SIGMA_P`
    Yeh-Lindau p-subshell cross sections on the tabulation grid.
:obj:`CROSS_SECTION_SIGMA_S`
    Yeh-Lindau s-subshell cross sections on the tabulation grid.
:class:`CrystalGeometry`
    PyTree for crystal geometry from VASP POSCAR.
:obj:`D_ORBITAL_SLICE`
    Slice selecting the five d orbitals.
:class:`DensityOfStates`
    PyTree for density of states.
:class:`DiagonalizedBands`
    PyTree for diagonalized electronic structure.
:obj:`DosType`
    Density-of-states carrier union used by workflow contexts.
:obj:`EIG_DOWN_INDEX`
    Column index of spin-down eigenvalues in EIGENVAL.
:obj:`EIG_UP_INDEX`
    Column index of spin-up eigenvalues in EIGENVAL.
:obj:`ENERGY_AXIS_NDIM`
    Expected dimensionality of energy-axis arrays.
:obj:`EPS`
    Epsilon floor guarding divisions and norms.
:obj:`FLOAT_TOKEN_RE`
    Compiled regex matching floating-point tokens.
:class:`FullDensityOfStates`
    PyTree for complete density of states with spin and PDOS.
:obj:`GAUNT_IMAG_TOL`
    Tolerance for discarding imaginary Gaunt residues.
:obj:`HBAR_C_EV_A`
    Reduced Planck constant times c in eV Angstrom.
:obj:`HBAR_EV_S`
    Reduced Planck constant in eV s.
:obj:`INTENSITY_NDIM`
    Expected dimensionality of intensity arrays.
:obj:`ISPIN2_BLOCKS`
    PROCAR block count for ISPIN=2 calculations.
:obj:`ISPIN_SPIN_POLARIZED`
    ISPIN value marking spin-polarized VASP runs.
:obj:`KB_EV_PER_K`
    Boltzmann constant in eV per kelvin.
:obj:`KPATH_AUX_WITH_COMMENT_LEN`
    KPathInfo auxiliary-data length including a comment.
:obj:`KPATH_AUX_WITH_COORD_MODE_LEN`
    KPathInfo auxiliary-data length including a coordinate mode.
:class:`KPathInfo`
    PyTree for k-point path metadata.
:obj:`KPOINT_LINE_VALUES`
    Tokens on an EIGENVAL k-point line.
:obj:`L_MAX`
    Maximum angular momentum supported by the precomputed table.
:obj:`LATTICE_ROWS`
    Number of lattice-vector rows in POSCAR/CHGCAR headers.
:obj:`M_D`
    Magnetic quantum numbers of the d orbitals.
:obj:`M_P`
    Magnetic quantum numbers of the p orbitals.
:func:`make_1d_chain_model`
    Create a 1D chain tight-binding model.
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
:obj:`ME_EV`
    Electron rest energy in eV.
:obj:`MIN_SUM`
    Minimum-sum floor guarding normalizations.
:obj:`N_ORBITALS`
    Number of orbitals in the VASP projection basis.
:obj:`N_SOC_MAG_BLOCKS`
    Magnetization block count in SOC CHGCAR files.
:obj:`N_SPIN_COMPONENTS`
    Spin-projection component count in PROCAR.
:obj:`N_TAYLOR`
    Taylor-series order for the Faddeeva evaluation.
:obj:`NON_S_ORBITAL_SLICE`
    Slice selecting all non-s orbitals.
:obj:`NonJaxNumber`
    Union of ``int``, ``float``, and ``complex``.
:obj:`NONSPIN_COLS`
    DOSCAR column count without spin polarization.
:obj:`NORM_EPS`
    Epsilon floor guarding eigenvector normalization.
:obj:`ORBITAL_DIRS_NORMALIZED`
    Unit-normalized orbital directions in VASP orbital order.
:obj:`ORBITAL_INDEX`
    Mapping from orbital name to VASP orbital index.
:class:`OrbitalBasis`
    PyTree for orbital quantum number metadata.
:class:`OrbitalProjection`
    PyTree for orbital-resolved band projections.
:obj:`P_ORBITAL_SLICE`
    Slice selecting the three p orbitals.
:obj:`PHASE_LOSS_MESSAGE`
    Warning text for PROCAR magnitude-only eigenvectors.
:class:`PolarizationConfig`
    PyTree for photon polarization geometry.
:obj:`PRESET_NAMES`
    Recognized band-scatter plotting preset names.
:obj:`ProjectionType`
    Orbital-projection carrier union used by workflow contexts.
:obj:`S_IDX`
    Index of the s orbital.
:obj:`SCALAR_LINE_COMPONENTS`
    Tokens on a scalar CHGCAR header line.
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
:obj:`SMALL_ARGUMENT`
    Small-argument cutoff for spherical Bessel seeds.
:obj:`SOC_BLOCKS`
    PROCAR block count for SOC calculations.
:class:`SOCVolumetricData`
    PyTree for volumetric data from SOC CHGCAR files.
:obj:`SPIN_COLS`
    DOSCAR column count with spin polarization.
:class:`SpinBandStructure`
    PyTree for spin-resolved electronic band structure.
:class:`SpinOrbitalProjection`
    PyTree for orbital projections with mandatory spin data.
:class:`TBModel`
    PyTree for tight-binding model parameters (legacy).
:class:`VolumetricData`
    PyTree for volumetric grid data from CHGCAR.
:obj:`WEIGHT_COMPONENT_COUNT`
    Tokens on a weighted k-point line.
:obj:`WEIGHT_COMPONENT_INDEX`
    Index of the weight token on a k-point line.
:class:`WorkflowContext`
    PyTree bundling parsed inputs for high-level VASP workflows.
:obj:`XYZ_COMPONENTS`
    Number of Cartesian vector components.

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
    ATTR_AUX,
    ATTR_NONE,
    ATTR_TYPE,
    BAND_LINE_MIN_VALUES,
    BAND_LINE_SPIN_VALUES,
    BAND_NDIM,
    BOHR_TO_ANGSTROM,
    COORDINATE_MODE_TOKENS,
    D_ORBITAL_SLICE,
    EIG_DOWN_INDEX,
    EIG_UP_INDEX,
    ENERGY_AXIS_NDIM,
    EPS,
    FLOAT_TOKEN_RE,
    GAUNT_IMAG_TOL,
    HBAR_C_EV_A,
    HBAR_EV_S,
    INTENSITY_NDIM,
    ISPIN2_BLOCKS,
    ISPIN_SPIN_POLARIZED,
    KB_EV_PER_K,
    KPATH_AUX_WITH_COMMENT_LEN,
    KPATH_AUX_WITH_COORD_MODE_LEN,
    KPOINT_LINE_VALUES,
    L_MAX,
    LATTICE_ROWS,
    M_D,
    M_P,
    ME_EV,
    MIN_SUM,
    N_ORBITALS,
    N_SOC_MAG_BLOCKS,
    N_SPIN_COMPONENTS,
    N_TAYLOR,
    NON_S_ORBITAL_SLICE,
    NONSPIN_COLS,
    NORM_EPS,
    ORBITAL_DIRS_NORMALIZED,
    ORBITAL_INDEX,
    P_ORBITAL_SLICE,
    PHASE_LOSS_MESSAGE,
    PRESET_NAMES,
    S_IDX,
    SCALAR_LINE_COMPONENTS,
    SMALL_ARGUMENT,
    SOC_BLOCKS,
    SPIN_COLS,
    WEIGHT_COMPONENT_COUNT,
    WEIGHT_COMPONENT_INDEX,
    XYZ_COMPONENTS,
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
from .tables import (
    CROSS_SECTION_ENERGIES,
    CROSS_SECTION_SIGMA_D,
    CROSS_SECTION_SIGMA_P,
    CROSS_SECTION_SIGMA_S,
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
    "ATTR_AUX",
    "ATTR_NONE",
    "ATTR_TYPE",
    "BAND_LINE_MIN_VALUES",
    "BAND_LINE_SPIN_VALUES",
    "BAND_NDIM",
    "BandStructure",
    "BOHR_TO_ANGSTROM",
    "COORDINATE_MODE_TOKENS",
    "CROSS_SECTION_ENERGIES",
    "CROSS_SECTION_SIGMA_D",
    "CROSS_SECTION_SIGMA_P",
    "CROSS_SECTION_SIGMA_S",
    "CrystalGeometry",
    "D_ORBITAL_SLICE",
    "DensityOfStates",
    "DiagonalizedBands",
    "DosType",
    "EIG_DOWN_INDEX",
    "EIG_UP_INDEX",
    "ENERGY_AXIS_NDIM",
    "EPS",
    "FLOAT_TOKEN_RE",
    "FullDensityOfStates",
    "GAUNT_IMAG_TOL",
    "HBAR_C_EV_A",
    "HBAR_EV_S",
    "INTENSITY_NDIM",
    "ISPIN2_BLOCKS",
    "ISPIN_SPIN_POLARIZED",
    "KB_EV_PER_K",
    "KPATH_AUX_WITH_COMMENT_LEN",
    "KPATH_AUX_WITH_COORD_MODE_LEN",
    "KPathInfo",
    "KPOINT_LINE_VALUES",
    "L_MAX",
    "LATTICE_ROWS",
    "M_D",
    "M_P",
    "make_1d_chain_model",
    "make_arpes_spectrum",
    "make_band_structure",
    "make_crystal_geometry",
    "make_density_of_states",
    "make_diagonalized_bands",
    "make_expanded_simulation_params",
    "make_full_density_of_states",
    "make_graphene_model",
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
    "ME_EV",
    "MIN_SUM",
    "N_ORBITALS",
    "N_SOC_MAG_BLOCKS",
    "N_SPIN_COMPONENTS",
    "N_TAYLOR",
    "NON_S_ORBITAL_SLICE",
    "NonJaxNumber",
    "NONSPIN_COLS",
    "NORM_EPS",
    "ORBITAL_DIRS_NORMALIZED",
    "ORBITAL_INDEX",
    "OrbitalBasis",
    "OrbitalProjection",
    "P_ORBITAL_SLICE",
    "PHASE_LOSS_MESSAGE",
    "PolarizationConfig",
    "PRESET_NAMES",
    "ProjectionType",
    "S_IDX",
    "SCALAR_LINE_COMPONENTS",
    "ScalarBool",
    "ScalarComplex",
    "ScalarFloat",
    "ScalarInteger",
    "ScalarNumeric",
    "SelfEnergyConfig",
    "SimulationParams",
    "SlaterParams",
    "SMALL_ARGUMENT",
    "SOC_BLOCKS",
    "SOCVolumetricData",
    "SPIN_COLS",
    "SpinBandStructure",
    "SpinOrbitalProjection",
    "TBModel",
    "VolumetricData",
    "WEIGHT_COMPONENT_COUNT",
    "WEIGHT_COMPONENT_INDEX",
    "WorkflowContext",
    "XYZ_COMPONENTS",
]
