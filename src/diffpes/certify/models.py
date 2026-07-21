"""Register built-in certified DiffPES forward models.

Extended Summary
----------------
Defines the stable scientific identity and one-PyTree executor adapter for the
current radial tight-binding ARPES forward model. Registration is explicit and
idempotent; importing DiffPES does not mutate the registry.

Routine Listings
----------------
:func:`execute_tb_radial`
    Execute the radial ARPES model from one certification input PyTree.
:func:`register_builtin_models`
    Register built-in models and information-loss transformations.
:func:`tb_radial_model_spec`
    Return the stable scientific specification for radial ARPES.
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any
from jaxtyping import Array, jaxtyped

from diffpes.simul import simulate_tb_radial
from diffpes.types import (
    TB_RADIAL_INPUT_COUNT,
    TB_RADIAL_MODEL_ID,
    TB_RADIAL_MODEL_VERSION,
    ArpesSpectrum,
    DomainResult,
    ForwardModelSpec,
    TransformationContract,
    make_convention_ref,
    make_domain_predicate,
    make_domain_result,
    make_forward_model_spec,
    make_transformation_contract,
)

from .checks import list_checks, register_check
from .registry import (
    list_models,
    list_transformations,
    register_model,
    register_transformation,
)


@jaxtyped(typechecker=beartype)
def execute_tb_radial(inputs: tuple[Any, ...]) -> ArpesSpectrum:
    """Execute the radial ARPES model from one certification input PyTree.

    The tuple order is ``(bands, radial, simulation, polarization,
    work_function, self_energy, radial_grid, momentum_width)``. Optional
    entries use ``None`` and therefore become empty JAX subtrees.

    :see: :class:`~.test_models.TestExecuteTbRadial`

    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           spectrum: ArpesSpectrum = simulate_tb_radial(
                   bands,
                   radial,
                   simulation,
                   polarization,
                   work_function=work_function,
                   self_energy=self_energy,
                   r_grid=grid,
                   dk=dk,
               )

       This expression follows the explicit validation and transformations in
       the function body. It keeps the documented output bound before return.

    Parameters
    ----------
    inputs : tuple[Any, ...]
        Eight-position radial-model input PyTree. Its tuple structure is
        **static**; a change to the structure triggers retracing.

    Returns
    -------
    spectrum : ArpesSpectrum
        Simulated ARPES intensity and axes in their declared physical units.

    Raises
    ------
    ValueError
        If ``inputs`` does not contain exactly eight entries.

    Notes
    -----
    Gradients pass through every numerical input accepted by
    :func:`~diffpes.simul.simulate_tb_radial`.
    """
    if len(inputs) != TB_RADIAL_INPUT_COUNT:
        msg: str = "tb_radial certification inputs must contain eight entries"
        raise ValueError(msg)
    bands: Any = inputs[0]
    radial: Any = inputs[1]
    simulation: Any = inputs[2]
    polarization: Any = inputs[3]
    work_function: Any = inputs[4]
    self_energy: Any = inputs[5]
    grid: Any = inputs[6]
    dk: Any = inputs[7]
    spectrum: ArpesSpectrum = simulate_tb_radial(
        bands,
        radial,
        simulation,
        polarization,
        work_function=work_function,
        self_energy=self_energy,
        r_grid=grid,
        dk=dk,
    )
    return spectrum


@jaxtyped(typechecker=beartype)
def tb_radial_model_spec() -> ForwardModelSpec:
    """Return the stable scientific specification for radial ARPES.

    The operation uses the built-in radial ARPES adapter and its declared
    conventions. It preserves JAX differentiation through numerical model
    inputs.

    :see: :class:`~.test_models.TestTbRadialModelSpec`

    Returns
    -------
    spec : ForwardModelSpec
        Versioned assumptions, conventions, domain, and differentiable paths.

    Notes
    -----
    The specification identifies the current tight-binding plus radial
    dipole model. It does not itself evaluate or certify a numerical result.
    """
    conventions: tuple = (
        make_convention_ref(
            "org.diffpes.convention.energy.fermi_referenced_ev",
            "1.0.0",
            "{}",
        ),
        make_convention_ref(
            "org.diffpes.convention.length.angstrom",
            "1.0.0",
            "{}",
        ),
        make_convention_ref(
            "org.diffpes.convention.orbital.real_harmonics",
            "1.0.0",
            "{}",
        ),
    )
    domain: tuple = (
        make_domain_predicate(
            "org.diffpes.domain.photon_energy.positive",
            "photon_energy_ev > 0",
            "eV",
            "error",
        ),
        make_domain_predicate(
            "org.diffpes.domain.radial_grid.positive",
            "all(r_grid > 0)",
            "bohr",
            "error",
        ),
    )
    spec: ForwardModelSpec = make_forward_model_spec(
        model_id=TB_RADIAL_MODEL_ID,
        model_version=TB_RADIAL_MODEL_VERSION,
        observable_id="org.diffpes.observable.arpes.intensity",
        implementation_ref="diffpes.simul.simulate_tb_radial",
        assumptions=(
            "dipole_approximation",
            "independent_particle_initial_state",
            "free_electron_final_state",
            "slater_radial_basis",
            "fermi_dirac_occupation",
            "voigt_energy_resolution",
        ),
        conventions=conventions,
        domain=domain,
        differentiable_paths=(
            "bands.eigenvalues",
            "bands.eigenvectors",
            "radial.zeta",
            "simulation.sigma",
            "simulation.gamma",
            "simulation.temperature",
            "simulation.photon_energy",
            "polarization.theta",
            "polarization.phi",
            "work_function",
        ),
        nondifferentiable_paths=(
            "simulation.fidelity",
            "polarization.polarization_type",
        ),
    )
    return spec


def _register_transformations() -> None:
    """Register built-in semantic and information-loss contracts."""
    contract: Any
    contracts: tuple[TransformationContract, ...] = (
        make_transformation_contract(
            "org.diffpes.transform.amplitude.intensity",
            "1.0.0",
            requires=("complex_photoemission_amplitude",),
            produces=("arpes_intensity",),
            preserves=("energy_reference", "momentum_coordinates"),
            destroys=("overall_matrix_element_phase",),
            invalidates_claims=("claim.amplitude.phase_recoverable",),
        ),
        make_transformation_contract(
            "org.diffpes.transform.band.incoherent_sum",
            "1.0.0",
            requires=("band_resolved_intensity",),
            produces=("summed_intensity",),
            preserves=("energy_reference", "momentum_coordinates"),
            destroys=("band_component_attribution",),
            invalidates_claims=("claim.band.attribution_preserved",),
        ),
        make_transformation_contract(
            "org.diffpes.transform.resolution.energy_voigt",
            "1.0.0",
            requires=("unbroadened_spectrum",),
            produces=("energy_broadened_spectrum",),
            preserves=("energy_reference", "momentum_coordinates"),
            introduces=("finite_energy_resolution",),
            destroys=("unresolved_energy_information",),
            invalidates_claims=("claim.spectrum.unbroadened",),
        ),
        make_transformation_contract(
            "org.diffpes.transform.normalization.zscore",
            "1.0.0",
            requires=("absolute_intensity",),
            produces=("standardized_intensity",),
            preserves=("energy_reference", "momentum_coordinates"),
            introduces=("dimensionless_standardization",),
            destroys=("absolute_intensity_calibration",),
            invalidates_claims=("claim.intensity.absolute_calibration",),
        ),
    )
    existing: set[tuple[str, str]] = {
        (item.transformation_id, item.transformation_version)
        for item in list_transformations()
    }
    for contract in contracts:
        key: tuple[str, str] = (
            contract.transformation_id,
            contract.transformation_version,
        )
        if key not in existing:
            register_transformation(contract)


def _check_positive_photon_energy(inputs: tuple[Any, ...]) -> DomainResult:
    """Check that the radial model uses a positive photon energy."""
    photon_energy: Array = inputs[2].photon_energy
    margin: Array = photon_energy
    result: DomainResult = make_domain_result(
        predicate_id="org.diffpes.domain.photon_energy.positive",
        measured=photon_energy,
        reference=jnp.zeros(()),
        residual=photon_energy,
        tolerance=jnp.zeros(()),
        margin=margin,
        passed=margin > 0.0,
        in_domain=margin > 0.0,
        severity_code=jnp.asarray(2, dtype=jnp.int32),
    )
    return result


def _check_positive_radial_grid(inputs: tuple[Any, ...]) -> DomainResult:
    """Check that every explicit radial-grid point is positive."""
    radial_grid: Any = inputs[6]
    margin: Array = (
        jnp.asarray(1e-6, dtype=jnp.float64)
        if radial_grid is None
        else jnp.min(radial_grid)
    )
    result: DomainResult = make_domain_result(
        predicate_id="org.diffpes.domain.radial_grid.positive",
        measured=margin,
        reference=jnp.zeros(()),
        residual=margin,
        tolerance=jnp.zeros(()),
        margin=margin,
        passed=margin > 0.0,
        in_domain=margin > 0.0,
        severity_code=jnp.asarray(2, dtype=jnp.int32),
    )
    return result


def _register_checks() -> None:
    """Register built-in radial-model domain checks idempotently."""
    check_id: Any
    check_fn: Any
    registered: set[str] = set(list_checks())
    checks: tuple[tuple[str, Any], ...] = (
        (
            "org.diffpes.domain.photon_energy.positive",
            _check_positive_photon_energy,
        ),
        (
            "org.diffpes.domain.radial_grid.positive",
            _check_positive_radial_grid,
        ),
    )
    for check_id, check_fn in checks:
        if check_id not in registered:
            register_check(check_id, check_fn)


@jaxtyped(typechecker=beartype)
def register_builtin_models() -> None:
    """Register built-in models and information-loss transformations.

    The operation uses the built-in radial ARPES adapter and its declared
    conventions. It preserves JAX differentiation through numerical model
    inputs.

    :see: :class:`~.test_models.TestRegisterBuiltinModels`

    Notes
    -----
    Registration is explicit and idempotent at the eager application boundary.
    Numerical model execution and domain predicates remain pure JAX programs.
    """
    existing: set[tuple[str, str]] = {
        (item.model_id, item.model_version) for item in list_models()
    }
    key: tuple[str, str] = (TB_RADIAL_MODEL_ID, TB_RADIAL_MODEL_VERSION)
    if key not in existing:
        register_model(tb_radial_model_spec(), execute_tb_radial)
    _register_transformations()
    _register_checks()


__all__: list[str] = [
    "execute_tb_radial",
    "register_builtin_models",
    "tb_radial_model_spec",
]
