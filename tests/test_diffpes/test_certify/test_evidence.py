"""Tests for differentiable certification evidence."""

import jax
import jax.numpy as jnp

from diffpes.certify.evidence import (
    derivative_evidence,
    evaluate_claim,
    evaluate_domain,
    evaluate_evidence,
)


class TestEvidence:
    """Verify continuous residuals remain available to autodiff."""

    def test_claim_retains_differentiable_margin(self):
        """Differentiate the tolerance margin beneath a Boolean claim."""

        def margin(measured):
            claim = evaluate_claim(
                "claim.test",
                "subject.test",
                "verification.test",
                measured,
                jnp.zeros(1),
                jnp.ones(1),
            )
            return claim.margin

        assert jnp.allclose(jax.grad(margin)(jnp.array([0.25])), -1.0)

    def test_domain_reports_signed_margin(self):
        """Preserve distance to the validity boundary."""
        result = evaluate_domain(
            "domain.test", jnp.array(1.2), jnp.array(1.0), jnp.array(0.5)
        )
        assert bool(result.passed)
        assert jnp.allclose(result.margin, 0.3)

    def test_external_evidence_preserves_residual(self):
        """Record numerical comparison arrays without reducing them."""
        result = evaluate_evidence(
            "evidence.test",
            "method.closed_form",
            jnp.array([1.1, 1.9]),
            jnp.array([1.0, 2.0]),
            jnp.array([0.2, 0.2]),
        )
        assert jnp.allclose(result.residual, jnp.array([0.1, -0.1]))

    def test_derivative_evidence_matches_quadratic(self):
        """Compare retained JVPs and VJPs against central differences."""
        result = derivative_evidence(
            lambda x: x**2,
            jnp.array([2.0]),
            jnp.ones((1, 1)),
            jnp.ones((1, 1)),
            input_paths=("$[0]",),
            output_projection_ids=("square",),
            scales=jnp.ones(1),
            spectrum_rank=1,
        )
        assert bool(result.finite)
        assert bool(result.fd_correct)
        assert jnp.allclose(result.jvp_probes, 4.0)
        assert jnp.allclose(result.vjp_probes, 4.0)
