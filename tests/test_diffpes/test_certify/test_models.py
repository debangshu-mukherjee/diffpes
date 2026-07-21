"""Validate built-in certified model registration.

The tests cover public behavior, differentiability, validation, and stable
scientific identity in the supported certification regime.
"""

import jax.numpy as jnp
import pytest
from beartype.typing import Any

from diffpes.certify import (
    certify_forward,
    list_models,
    list_transformations,
    prepare_certification,
    register_builtin_models,
    tb_radial_model_spec,
)
from diffpes.tightb import diagonalize_tb
from diffpes.types import (
    TB_RADIAL_MODEL_ID,
    TB_RADIAL_MODEL_VERSION,
    make_1d_chain_model,
    make_execution_manifest,
    make_orbital_basis,
    make_polarization_config,
    make_simulation_params,
    make_slater_params,
)


class TestTbRadialModelSpec:
    """Verify :func:`~diffpes.certify.tb_radial_model_spec`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.tb_radial_model_spec`
    """

    def test_radial_model_spec_names_physics_and_gradients(self) -> None:
        """Declare assumptions, conventions, domains, and traced paths.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        spec: Any
        spec = tb_radial_model_spec()
        assert spec.model_id == TB_RADIAL_MODEL_ID
        assert "dipole_approximation" in spec.assumptions
        assert "radial.zeta" in spec.differentiable_paths
        assert spec.domain


class TestRegisterBuiltinModels:
    """Verify :func:`~diffpes.certify.register_builtin_models`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.register_builtin_models`
    """

    def test_builtin_registration_is_idempotent(self) -> None:
        """Register each built-in identity exactly once.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        destroyed: Any
        register_builtin_models()
        register_builtin_models()
        assert (
            sum(
                entry.model_id == TB_RADIAL_MODEL_ID for entry in list_models()
            )
            == 1
        )
        destroyed = {
            loss for entry in list_transformations() for loss in entry.destroys
        }
        assert "overall_matrix_element_phase" in destroyed
        assert "absolute_intensity_calibration" in destroyed


class TestExecuteTbRadial:
    """Verify :func:`~diffpes.certify.execute_tb_radial`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.execute_tb_radial`
    """

    @pytest.mark.rss_limit_mb(900)
    def test_radial_model_certifies_end_to_end(self) -> None:
        """Run the built-in radial ARPES model through compiled certification.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        model: Any
        kpoints: Any
        bands: Any
        basis: Any
        radial: Any
        simulation: Any
        polarization: Any
        manifest: Any
        context: Any
        inputs: Any
        result: Any
        register_builtin_models()
        model = make_1d_chain_model(t=-1.0)
        kpoints = jnp.array([[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]])
        bands = diagonalize_tb(model, kpoints)
        basis = make_orbital_basis(
            n_values=(1,),
            l_values=(0,),
            m_values=(0,),
            labels=("1s",),
        )
        radial = make_slater_params(zeta=jnp.array([1.0]), orbital_basis=basis)
        simulation = make_simulation_params(
            energy_min=-3.0,
            energy_max=3.0,
            fidelity=20,
            sigma=0.1,
            gamma=0.1,
            temperature=30.0,
            photon_energy=21.2,
        )
        polarization = make_polarization_config(polarization_type="LHP")
        manifest = make_execution_manifest(
            "radial-test",
            f"{TB_RADIAL_MODEL_ID}@{TB_RADIAL_MODEL_VERSION}",
            "1",
            "test",
            "source",
            "environment",
            "cpu",
            "f64",
            True,
            "2026-07-21T00:00:00Z",
        )
        context = prepare_certification(
            TB_RADIAL_MODEL_ID,
            TB_RADIAL_MODEL_VERSION,
            manifest,
            policy_id="org.diffpes.policy.exploratory.v1",
        )
        inputs = (
            bands,
            radial,
            simulation,
            polarization,
            jnp.asarray(4.5),
            None,
            jnp.linspace(1e-5, 20.0, 128),
            None,
        )
        result = certify_forward(context, inputs, spectrum_rank=1)
        assert result.value.intensity.shape == (2, 20)
        assert len(result.certificate.domains) == 2
        assert all(
            bool(domain.in_domain) for domain in result.certificate.domains
        )
        assert bool(result.certificate.derivatives.finite)
        assert bool(result.certificate.derivatives.fd_correct)
        assert result.certificate.information.singular_values.shape == (1,)
