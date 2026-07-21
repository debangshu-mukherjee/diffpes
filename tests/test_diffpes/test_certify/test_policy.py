"""Validate cumulative scientific-certification policies.

The tests cover public behavior, differentiability, validation, and stable
scientific identity in the supported certification regime.
"""

import jax.numpy as jnp
from beartype.typing import Any

from diffpes.certify import achieved_levels, evaluate_claim, evaluate_policy


def _claim(name: Any, predicate: Any, passed: Any = True) -> Any:
    measured: Any
    measured = jnp.zeros(1) if passed else jnp.ones(1)
    return evaluate_claim(
        name,
        "subject.test",
        predicate,
        measured,
        jnp.zeros(1),
        jnp.zeros(1),
    )


class TestAchievedLevels:
    """Verify :func:`~diffpes.certify.achieved_levels`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.achieved_levels`
    """

    def test_exploratory_reaches_identified_and_validated(self) -> None:
        """Achieve the two exploratory levels from required claims.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        report: Any
        report = evaluate_policy(
            (
                _claim("identity", "identity.model"),
                _claim("output", "output.finite"),
            ),
            "org.diffpes.policy.exploratory.v1",
        )
        assert achieved_levels(report) == ("identified", "validated")


class TestEvaluatePolicy:
    """Verify :func:`~diffpes.certify.evaluate_policy`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.evaluate_policy`
    """

    def test_failed_lower_level_blocks_higher_levels(self) -> None:
        """Make cumulative outcomes false above a failed identity claim.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        report: Any
        report = evaluate_policy(
            (
                _claim("identity", "identity.model", passed=False),
                _claim("output", "output.finite"),
                _claim("derivative", "derivative.fd"),
                _claim("verify", "verification.closed_form"),
            )
        )
        assert not bool(jnp.any(report.achieved))
