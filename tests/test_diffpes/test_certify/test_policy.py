"""Tests for cumulative scientific-certification policies."""

import jax.numpy as jnp

from diffpes.certify.evidence import evaluate_claim
from diffpes.certify.policy import achieved_levels, evaluate_policy


def _claim(name, predicate, passed=True):
    measured = jnp.zeros(1) if passed else jnp.ones(1)
    return evaluate_claim(
        name,
        "subject.test",
        predicate,
        measured,
        jnp.zeros(1),
        jnp.zeros(1),
    )


class TestPolicy:
    """Verify level accumulation and policy downgrade behavior."""

    def test_exploratory_reaches_identified_and_validated(self):
        """Achieve the two exploratory levels from required claims."""
        report = evaluate_policy(
            (
                _claim("identity", "identity.model"),
                _claim("output", "output.finite"),
            ),
            "org.diffpes.policy.exploratory.v1",
        )
        assert achieved_levels(report) == ("identified", "validated")

    def test_failed_lower_level_blocks_higher_levels(self):
        """Make cumulative outcomes false above a failed identity claim."""
        report = evaluate_policy(
            (
                _claim("identity", "identity.model", passed=False),
                _claim("output", "output.finite"),
                _claim("derivative", "derivative.fd"),
                _claim("verify", "verification.closed_form"),
            )
        )
        assert not bool(jnp.any(report.achieved))
