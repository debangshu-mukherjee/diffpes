"""Exercise the tight-binding radial differentiability smoke chain.

Extended Summary
----------------
Validates finite-difference-correct gradients through the diagonalized-band,
Slater-radial, polarization, and simulation-parameter carrier seams. The
eigenvector probe uses generic complex coefficients so conjugation errors
cannot hide behind real or symmetric test data.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Complex, Float

from diffpes.simul import simulate_tb_radial
from diffpes.tightb import diagonalize_tb
from diffpes.types import (
    ArpesSpectrum,
    DiagonalizedBands,
    OrbitalBasis,
    PolarizationConfig,
    SimulationParams,
    SlaterParams,
    TBModel,
    make_diagonalized_bands,
    make_graphene_model,
    make_orbital_basis,
    make_polarization_config,
    make_simulation_params,
    make_slater_params,
)
from tests._gradients import assert_grad_matches_fd, assert_nonzero_grad


def _generic_radial_fixture() -> tuple[
    DiagonalizedBands,
    SlaterParams,
    SimulationParams,
    PolarizationConfig,
    Float[Array, " R"],
]:
    """Build a lightweight radial fixture with generic complex state data."""
    eigenvectors: Complex[Array, "1 2 2"] = jnp.array(
        [
            [
                [0.73 + 0.19j, -0.28 + 0.61j],
                [0.37 - 0.52j, 0.66 + 0.24j],
            ]
        ],
        dtype=jnp.complex128,
    )
    bands: DiagonalizedBands = make_diagonalized_bands(
        eigenvalues=jnp.array([[-0.03, 0.04]], dtype=jnp.float64),
        eigenvectors=eigenvectors,
        kpoints=jnp.array([[0.12, 0.17, 0.08]], dtype=jnp.float64),
        fermi_energy=0.0,
    )
    basis: OrbitalBasis = make_orbital_basis(
        n_values=(1, 2),
        l_values=(0, 1),
        m_values=(0, 1),
        labels=("1s", "2px"),
    )
    slater: SlaterParams = make_slater_params(
        zeta=jnp.array([1.1, 1.6], dtype=jnp.float64),
        orbital_basis=basis,
    )
    params: SimulationParams = make_simulation_params(
        energy_min=-0.3,
        energy_max=0.3,
        fidelity=24,
        sigma=0.04,
        gamma=0.03,
        temperature=120.0,
        photon_energy=21.2,
    )
    polarization: PolarizationConfig = make_polarization_config(
        theta=0.7,
        phi=0.3,
        polarization_angle=0.4,
        polarization_type="LAP",
    )
    radial_grid: Float[Array, " R"] = jnp.linspace(1e-5, 20.0, 96)
    result: tuple[
        DiagonalizedBands,
        SlaterParams,
        SimulationParams,
        PolarizationConfig,
        Float[Array, " R"],
    ] = bands, slater, params, polarization, radial_grid
    return result


def _total_intensity(
    bands: DiagonalizedBands,
    slater: SlaterParams,
    params: SimulationParams,
    polarization: PolarizationConfig,
    radial_grid: Float[Array, " R"],
) -> Float[Array, ""]:
    """Evaluate the scalar smoke-chain observable."""
    spectrum: ArpesSpectrum = simulate_tb_radial(
        bands,
        slater,
        params,
        polarization,
        r_grid=radial_grid,
    )
    total: Float[Array, ""] = jnp.sum(spectrum.intensity)
    return total


class TestTBRadialCarrierGradients:
    """Validate the carrier seams of the tight-binding radial forward.

    :see: :func:`diffpes.simul.simulate_tb_radial`
    """

    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(1100)
    def test_carrier_gradient_smoke_chain(self) -> None:
        """Match FD through every requested radial-chain carrier seam.

        Extended Summary
        ----------------
        Differentiates one scalar spectrum observable jointly with respect to
        eigenvalues, generic complex eigenvectors, Slater exponents, all three
        polarization angles, and every continuous simulation parameter. Every
        selected leaf must retain finite, nonzero, FD-correct sensitivity.
        """
        bands, slater, params, polarization, radial_grid = (
            _generic_radial_fixture()
        )
        carriers: tuple[
            Float[Array, "1 2"],
            Complex[Array, "1 2 2"],
            Float[Array, " 2"],
            PolarizationConfig,
            SimulationParams,
        ] = (
            bands.eigenvalues,
            bands.eigenvectors,
            slater.zeta,
            polarization,
            params,
        )

        def loss(
            candidate: tuple[
                Float[Array, "1 2"],
                Complex[Array, "1 2 2"],
                Float[Array, " 2"],
                PolarizationConfig,
                SimulationParams,
            ],
        ) -> Float[Array, ""]:
            eigenvalues: Float[Array, "1 2"]
            eigenvectors: Complex[Array, "1 2 2"]
            zeta: Float[Array, " 2"]
            pol_candidate: PolarizationConfig
            params_candidate: SimulationParams
            (
                eigenvalues,
                eigenvectors,
                zeta,
                pol_candidate,
                params_candidate,
            ) = candidate
            updated_bands: DiagonalizedBands = eqx.tree_at(
                lambda carrier: (
                    carrier.eigenvalues,
                    carrier.eigenvectors,
                ),
                bands,
                (eigenvalues, eigenvectors),
            )
            updated_slater: SlaterParams = eqx.tree_at(
                lambda carrier: carrier.zeta,
                slater,
                zeta,
            )
            total: Float[Array, ""] = _total_intensity(
                updated_bands,
                updated_slater,
                params_candidate,
                pol_candidate,
                radial_grid,
            )
            return total

        assert_grad_matches_fd(loss, carriers, regime="stiff", modes=("rev",))
        assert_nonzero_grad(loss, carriers)


class TestTBRadialBaselineGradient:
    """Pin the fully specified pre-migration graphene gradient baseline.

    :see: :func:`diffpes.simul.simulate_tb_radial`
    """

    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(1200)
    def test_common_zeta_gradient_matches_v01_fixture(self) -> None:
        """Reproduce the tagged-v0.1 graphene/LHP common-zeta gradient.

        Extended Summary
        ----------------
        Uses the graphene fixture fully specified by the v0.1 forward tests:
        three Gamma/K/M points, carbon 2p orbitals, a 300-point energy grid,
        and a 2,000-point explicit radial grid. The expected derivative was
        independently rerun against tag ``v0.1`` before being pinned here.
        """
        model: TBModel = make_graphene_model(t=-2.7)
        kpoints: Float[Array, "3 3"] = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0 / 3.0, 1.0 / 3.0, 0.0],
                [2.0 / 3.0, 1.0 / 3.0, 0.0],
            ],
            dtype=jnp.float64,
        )
        bands: DiagonalizedBands = diagonalize_tb(model, kpoints)
        basis: OrbitalBasis = make_orbital_basis(
            n_values=(2, 2),
            l_values=(1, 1),
            m_values=(0, 0),
            labels=("A_pz", "B_pz"),
        )
        params: SimulationParams = make_simulation_params(
            energy_min=-10.0,
            energy_max=10.0,
            fidelity=300,
            sigma=0.2,
            gamma=0.2,
            temperature=30.0,
            photon_energy=21.2,
        )
        polarization: PolarizationConfig = make_polarization_config(
            polarization_type="LHP"
        )
        radial_grid: Float[Array, " 2000"] = jnp.linspace(1e-6, 30.0, 2000)

        def loss(common_zeta: Float[Array, ""]) -> Float[Array, ""]:
            slater: SlaterParams = make_slater_params(
                zeta=jnp.stack((common_zeta, common_zeta)),
                orbital_basis=basis,
            )
            total: Float[Array, ""] = _total_intensity(
                bands, slater, params, polarization, radial_grid
            )
            return total

        common_zeta: Float[Array, ""] = jnp.asarray(1.625)
        assert_grad_matches_fd(loss, common_zeta, modes=("rev",))
        gradient: Float[Array, ""] = jax.grad(loss)(common_zeta)
        expected: Float[Array, ""] = jnp.asarray(-18.895708715290983)
        assert jnp.isclose(gradient, expected, rtol=1e-6, atol=0.0)
