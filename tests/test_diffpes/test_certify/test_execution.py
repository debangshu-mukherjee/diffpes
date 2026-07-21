"""Tests for compiled JAX-native certified execution."""

import uuid

import jax
import jax.numpy as jnp

from diffpes.certify.execution import (
    certify_forward,
    prepare_certification,
    verify_certificate,
)
from diffpes.certify.registry import register_model
from diffpes.types import make_execution_manifest, make_forward_model_spec


def _context(executor):
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


class TestExecution:
    """Verify compiled values, evidence, gradients, and batching."""

    def test_compiled_quadratic_certificate(self):
        """Produce the correct value, JVP, dependency, and spectrum."""
        context = _context(lambda x: x**2)
        result = certify_forward(context, jnp.array([2.0]), spectrum_rank=1)
        assert jnp.allclose(result.value, 4.0)
        assert jnp.allclose(result.certificate.derivatives.jvp_probes, 4.0)
        assert result.certificate.dependencies.structural.tolist() == [[True]]
        assert jnp.allclose(
            result.certificate.information.singular_values, 4.0
        )
        assert bool(verify_certificate(result.certificate).structure_valid)

    def test_gradient_through_result_and_information(self):
        """Differentiate both the observable and certificate evidence."""
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

    def test_vmap_certified_execution(self):
        """Batch complete certified executions with JAX VMAP."""
        context = _context(lambda x: x**2)
        values = jax.vmap(
            lambda x: certify_forward(context, x, spectrum_rank=1).value
        )(jnp.array([1.0, 2.0, 3.0]))
        assert jnp.allclose(values, jnp.array([1.0, 4.0, 9.0]))
