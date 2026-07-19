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
- :mod:`dos`
    Density of states data structures.
- :mod:`geometry`
    Crystal geometry data structure for VASP crystal structures.
- :mod:`kpath`
    K-point path information data structure.
- :mod:`params`
    Simulation parameter data structures.
- :mod:`radial_params`
    Radial wavefunction parameter data structures.
- :mod:`self_energy`
    Self-energy configuration data structures.
- :mod:`tb_model`
    Tight-binding model and diagonalized band data structures.
- :mod:`volumetric`
    Volumetric data structures for VASP CHGCAR files.

Routine Listings
----------------
:class:`ArpesSpectrum`
    PyTree for ARPES simulation output.
:class:`BandStructure`
    PyTree for electronic band structure.
:class:`CrystalGeometry`
    PyTree for crystal geometry from VASP POSCAR.
:class:`DensityOfStates`
    PyTree for density of states.
:class:`DiagonalizedBands`
    PyTree for diagonalized electronic structure.
:class:`FullDensityOfStates`
    PyTree for complete density of states with spin and PDOS.
:class:`KPathInfo`
    PyTree for k-point path metadata.
:obj:`NonJaxNumber`
    Union of ``int``, ``float``, and ``complex``.
:class:`OrbitalBasis`
    PyTree for orbital quantum number metadata.
:class:`OrbitalProjection`
    PyTree for orbital-resolved band projections.
:class:`PolarizationConfig`
    PyTree for photon polarization geometry.
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
:func:`make_full_density_of_states`
    Create a validated ``FullDensityOfStates`` instance.
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

Notes
-----
All PyTree types use ``@register_pytree_node_class`` for
compatibility with JAX transformations (jit, grad, vmap).
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
    make_diagonalized_bands,
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
    "CrystalGeometry",
    "DensityOfStates",
    "DiagonalizedBands",
    "FullDensityOfStates",
    "KPathInfo",
    "NonJaxNumber",
    "OrbitalBasis",
    "OrbitalProjection",
    "PolarizationConfig",
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
    "make_arpes_spectrum",
    "make_band_structure",
    "make_crystal_geometry",
    "make_density_of_states",
    "make_diagonalized_bands",
    "make_full_density_of_states",
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
]
