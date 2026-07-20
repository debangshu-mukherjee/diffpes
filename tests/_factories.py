"""Build deterministic toy carriers for tests.

Extended Summary
----------------
Provides small, fixed-policy inputs for forward, tight-binding, and radial
tests. Random factories are deterministic for a supplied JAX key; analytic
factories use fixed grids and physical parameters. Every returned traced leaf
is checked for finiteness at construction.
"""

import chex
import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped

from diffpes.tightb import (
    diagonalize_tb,
)
from diffpes.types import (
    BandStructure,
    DiagonalizedBands,
    OrbitalProjection,
    PolarizationConfig,
    SimulationParams,
    SlaterParams,
    TBModel,
    make_1d_chain_model,
    make_band_structure,
    make_graphene_model,
    make_orbital_basis,
    make_orbital_projection,
    make_polarization_config,
    make_simulation_params,
    make_slater_params,
)


def _assert_finite(tree: object) -> None:
    """Require every numerical leaf in a toy carrier to be finite."""
    leaves: tuple[object, ...] = tuple(jax.tree.leaves(tree))
    chex.assert_tree_all_finite(leaves)


@jaxtyped(typechecker=beartype)
def toy_band_structure(
    key: PRNGKeyArray,
    n_k: int = 8,
    n_bands: int = 4,
) -> BandStructure:
    """Build a reproducible occupied-state toy band structure.

    Eigenvalues are sampled in [-2.5, 0.25] eV, safely below
    ``E_F + 0.5`` eV. This intentionally avoids the known upper-state
    ``fermi_dirac`` gradient defect until plan 02 repairs it. The supplied key
    is the entire seed policy and is never mutated.
    """
    energy_key: PRNGKeyArray
    kpoint_key: PRNGKeyArray
    energy_key, kpoint_key = jax.random.split(key)
    eigenvalues: Float[Array, "n_k n_bands"] = jax.random.uniform(
        energy_key,
        (n_k, n_bands),
        minval=-2.5,
        maxval=0.25,
        dtype=jnp.float64,
    )
    eigenvalues = jnp.sort(eigenvalues, axis=-1)
    kpoints: Float[Array, "n_k 3"] = jax.random.uniform(
        kpoint_key,
        (n_k, 3),
        minval=-0.5,
        maxval=0.5,
        dtype=jnp.float64,
    )
    bands: BandStructure = make_band_structure(
        eigenvalues=eigenvalues,
        kpoints=kpoints,
        kpoint_weights=jnp.full(n_k, 1.0 / n_k, dtype=jnp.float64),
        fermi_energy=0.0,
    )
    _assert_finite(bands)
    return bands


@jaxtyped(typechecker=beartype)
def toy_orbital_projection(
    key: PRNGKeyArray,
    n_k: int = 8,
    n_bands: int = 4,
    n_atoms: int = 2,
) -> OrbitalProjection:
    """Build reproducible normalized orbital weights.

    Positive weights are drawn from a uniform distribution using only the
    supplied key, then normalized over atom and orbital axes for each state.
    Spin and orbital-angular-momentum fields remain absent.
    """
    raw: Float[Array, "n_k n_bands n_atoms 9"] = jax.random.uniform(
        key,
        (n_k, n_bands, n_atoms, 9),
        minval=0.1,
        maxval=1.0,
        dtype=jnp.float64,
    )
    normalization: Float[Array, "n_k n_bands 1 1"] = jnp.sum(
        raw, axis=(-2, -1), keepdims=True
    )
    projections: Float[Array, "n_k n_bands n_atoms 9"] = raw / normalization
    orbital_projection: OrbitalProjection = make_orbital_projection(
        projections
    )
    _assert_finite(orbital_projection)
    return orbital_projection


@jaxtyped(typechecker=beartype)
def toy_simulation_params(fidelity: int = 512) -> SimulationParams:
    """Build fixed low-temperature simulation parameters.

    Uses an energy window of [-3, 0.5] eV, 40 meV Gaussian resolution,
    100 meV Lorentzian width, 15 K temperature, and 21.2 eV photons. No random
    seed is used because these values are an analytic fixture policy.
    """
    params: SimulationParams = make_simulation_params(
        energy_min=-3.0,
        energy_max=0.5,
        fidelity=fidelity,
        sigma=0.04,
        gamma=0.1,
        temperature=15.0,
        photon_energy=21.2,
    )
    _assert_finite(params)
    return params


@jaxtyped(typechecker=beartype)
def toy_polarization_config() -> PolarizationConfig:
    """Build a fixed p-polarized 45-degree incidence geometry.

    The analytic fixture has theta = pi/4 rad, zero azimuth, and LHP
    polarization, so it requires no random seed.
    """
    config: PolarizationConfig = make_polarization_config(
        theta=jnp.pi / 4.0,
        phi=0.0,
        polarization_angle=0.0,
        polarization_type="LHP",
    )
    _assert_finite(config)
    return config


@jaxtyped(typechecker=beartype)
def toy_graphene_diagonalized(
    n_k: int = 12,
) -> tuple[TBModel, DiagonalizedBands]:
    """Diagonalize the native graphene model on a fixed Gamma-to-K path.

    Uses the production -2.7 eV nearest-neighbor model and an
    endpoint-inclusive fractional path from Gamma to K = (1/3, 1/3, 0). No
    random seed is used.
    """
    model: TBModel = make_graphene_model()
    path_coordinate: Float[Array, " n_k"] = jnp.linspace(
        0.0, 1.0, n_k, dtype=jnp.float64
    )
    kpoints: Float[Array, "n_k 3"] = path_coordinate[:, None] * jnp.array(
        [1.0 / 3.0, 1.0 / 3.0, 0.0], dtype=jnp.float64
    )
    bands: DiagonalizedBands = diagonalize_tb(model, kpoints)
    _assert_finite((model, bands))
    result: tuple[TBModel, DiagonalizedBands] = (model, bands)
    return result


@jaxtyped(typechecker=beartype)
def toy_chain_diagonalized(
    n_k: int = 16,
) -> tuple[TBModel, DiagonalizedBands]:
    """Diagonalize the native one-dimensional chain on a fixed k-path.

    Uses the production -1 eV hopping and an endpoint-inclusive fractional
    path from -1/2 to 1/2 along kx. No random seed is used.
    """
    model: TBModel = make_1d_chain_model()
    kx: Float[Array, " n_k"] = jnp.linspace(-0.5, 0.5, n_k, dtype=jnp.float64)
    kpoints: Float[Array, "n_k 3"] = jnp.stack(
        (kx, jnp.zeros_like(kx), jnp.zeros_like(kx)), axis=-1
    )
    bands: DiagonalizedBands = diagonalize_tb(model, kpoints)
    _assert_finite((model, bands))
    result: tuple[TBModel, DiagonalizedBands] = (model, bands)
    return result


@jaxtyped(typechecker=beartype)
def toy_slater_params() -> SlaterParams:
    """Build fixed single-zeta parameters for two carbon pz orbitals.

    Both orbitals use principal quantum number 2, angular momentum 1,
    magnetic quantum number 0, and a finite positive exponent of 1.625 inverse
    Bohr. The analytic fixture policy uses no random seed.
    """
    basis = make_orbital_basis(
        n_values=(2, 2),
        l_values=(1, 1),
        m_values=(0, 0),
        labels=("A_pz", "B_pz"),
    )
    params: SlaterParams = make_slater_params(
        zeta=jnp.array([1.625, 1.625], dtype=jnp.float64),
        orbital_basis=basis,
    )
    _assert_finite(params)
    return params
