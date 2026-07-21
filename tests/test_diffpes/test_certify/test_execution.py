"""Tests for compiled JAX-native certified execution.

The tests cover public behavior, differentiability, validation, and stable
scientific identity in the supported certification regime.
"""

import uuid

import jax
import jax.numpy as jnp
from beartype.typing import Any

from diffpes.certify import (
    certify_forward,
    prepare_certification,
    register_model,
    verify_certificate,
)
from diffpes.types import (
    CertificationContext,
    make_execution_manifest,
    make_forward_model_spec,
)


def _context(executor: Any) -> CertificationContext:
    suffix: Any
    model_id: Any
    spec: Any
    manifest: Any
    suffix = uuid.uuid4().hex
    model_id = f"org.diffpes.model.test{suffix}"
    spec = make_forward_model_spec(
        model_id,
        "1.0.0",
        "org.diffpes.observable.test.scalar",
        "tests.quadratic",
        differentiable_paths=("x",),
    )
    register_model(spec, executor)
    manifest = make_execution_manifest(
        f"execution-{suffix}",
        f"{model_id}@1.0.0",
        "1",
        "test",
        "source",
        "environment",
        "cpu",
        "f64",
        True,
        "2026-07-21T00:00:00Z",
    )
    return prepare_certification(
        model_id,
        "1.0.0",
        manifest,
        policy_id="org.diffpes.policy.exploratory.v1",
    )


class TestVerifyCertificate:
    """Verify :func:`~diffpes.certify.verify_certificate`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.verify_certificate`
    """

    def test_compiled_quadratic_certificate(self) -> None:
        """Produce the correct value, JVP, dependency, and spectrum.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        context: Any
        result: Any
        context = _context(lambda x: x**2)
        result = certify_forward(context, jnp.array([2.0]), spectrum_rank=1)
        assert jnp.allclose(result.value, 4.0)
        assert jnp.allclose(result.certificate.derivatives.jvp_probes, 4.0)
        assert result.certificate.dependencies.structural.tolist() == [[True]]
        assert jnp.allclose(
            result.certificate.information.singular_values, 4.0
        )
        assert bool(verify_certificate(result.certificate).structure_valid)


class TestCertifyForward:
    """Verify :func:`~diffpes.certify.certify_forward`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.certify_forward`
    """

    def test_gradient_through_result_and_information(self) -> None:
        """Differentiate both the observable and certificate evidence.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        context: Any
        value_grad: Any
        information_grad: Any
        context = _context(lambda x: x**3)
        value_grad = jax.grad(
            lambda x: jnp.sum(
                certify_forward(context, x, spectrum_rank=1).value
            )
        )(jnp.array([2.0]))
        information_grad = jax.grad(
            lambda x: jnp.sum(
                certify_forward(
                    context, x, spectrum_rank=1
                ).certificate.information.singular_values
            )
        )(jnp.array([2.0]))
        assert jnp.allclose(value_grad, 12.0)
        assert jnp.allclose(information_grad, 12.0)

    def test_vmap_certified_execution(self) -> None:
        """Batch complete certified executions with JAX VMAP.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        context: Any
        values: Any
        context = _context(lambda x: x**2)
        values = jax.vmap(
            lambda x: certify_forward(context, x, spectrum_rank=1).value
        )(jnp.array([1.0, 2.0, 3.0]))
        assert jnp.allclose(values, jnp.array([1.0, 4.0, 9.0]))


class TestPrepareCertification:
    """Verify :func:`~diffpes.certify.prepare_certification`.

    The cases cover eager resolution of stable model and policy identities.

    :see: :func:`~diffpes.certify.prepare_certification`
    """

    def test_context_binds_exact_model_identity(self) -> None:
        """Bind a registered model specification into a static context.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        context: CertificationContext = _context(lambda value: value**2)
        assert context.model.model_id in context.manifest.model_ref
        assert context.model.model_version in context.manifest.model_ref
