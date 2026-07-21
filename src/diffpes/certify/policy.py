"""Evaluate cumulative scientific-certification policies.

Extended Summary
----------------
Certification levels are derived from traced claim outcomes rather than stored
as trusted labels. Policies select required claim categories and accumulate
from identified through reproducible. A failed, unchecked, or out-of-domain
required claim prevents that level and every level above it.

Routine Listings
----------------
:func:`achieved_levels`
    Return certification level names achieved by a concrete report.
:func:`evaluate_policy`
    Derive cumulative certification outcomes from numerical claims.
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any, Iterable
from jaxtyping import Array, jaxtyped

from diffpes.types import (
    CERTIFICATION_LEVEL_IDS,
    CERTIFICATION_LEVEL_PREFIXES,
    CERTIFICATION_POLICY_IDS,
    CERTIFICATION_POLICY_LEVEL_COUNT,
    CertificationClaim,
    PolicyReport,
    make_policy_report,
)


def _required_indices(
    claims: tuple[CertificationClaim, ...], policy_id: str
) -> tuple[tuple[tuple[int, ...], ...], tuple[str, ...]]:
    """Select required claims for each cumulative policy level."""
    level_index: Any
    maximum_level: int = CERTIFICATION_POLICY_LEVEL_COUNT[policy_id]
    indices_by_level: list[tuple[int, ...]] = []
    required_ids: list[str] = []
    for level_index in range(len(CERTIFICATION_LEVEL_IDS)):
        if level_index >= maximum_level:
            indices_by_level.append(())
            continue
        prefixes: tuple[str, ...] = CERTIFICATION_LEVEL_PREFIXES[level_index]
        selected: tuple[int, ...] = tuple(
            index
            for index, claim in enumerate(claims)
            if claim.predicate_id.startswith(prefixes)
        )
        indices_by_level.append(selected)
        required_ids.extend(claims[index].claim_id for index in selected)
    result: tuple[tuple[tuple[int, ...], ...], tuple[str, ...]] = (
        tuple(indices_by_level),
        tuple(dict.fromkeys(required_ids)),
    )
    return result


@jaxtyped(typechecker=beartype)
def evaluate_policy(
    claims: Iterable[CertificationClaim],
    policy_id: str = "org.diffpes.policy.research.v1",
) -> PolicyReport:
    """Derive cumulative certification outcomes from numerical claims.

    The cumulative policy derives named levels from explicit claims. It retains
    the claim truth table as JAX arrays.

    :see: :class:`~.test_policy.TestEvaluatePolicy`

    Parameters
    ----------
    claims : Iterable[CertificationClaim]
        Numerical claims evaluated for one forward execution.
    policy_id : str
        Built-in cumulative policy identity (**static** -- a change retraces).

    Returns
    -------
    report : PolicyReport
        Traced truth table and cumulative achieved-level vector.

    Raises
    ------
    ValueError
        If ``policy_id`` is not a registered built-in policy.

    Notes
    -----
    Required claim selections are static. Boolean outcomes are JAX leaves, so
    the policy computation remains compatible with ``jit`` and ``vmap``.
    """
    level_index: Any
    indices: Any
    if policy_id not in CERTIFICATION_POLICY_IDS:
        msg: str = f"unknown certification policy: {policy_id}"
        raise ValueError(msg)
    claim_tuple: tuple[CertificationClaim, ...] = tuple(claims)
    selection: tuple[tuple[tuple[int, ...], ...], tuple[str, ...]] = (
        _required_indices(claim_tuple, policy_id)
    )
    indices_by_level: tuple[tuple[int, ...], ...] = selection[0]
    required_ids: tuple[str, ...] = selection[1]
    all_passed: Array = jnp.asarray(
        [claim.passed for claim in claim_tuple], dtype=jnp.bool_
    )
    all_checked: Array = jnp.asarray(
        [claim.checked for claim in claim_tuple], dtype=jnp.bool_
    )
    all_in_domain: Array = jnp.asarray(
        [claim.in_domain for claim in claim_tuple], dtype=jnp.bool_
    )
    valid_claim: Array = all_passed & all_checked & all_in_domain
    maximum_level: int = CERTIFICATION_POLICY_LEVEL_COUNT[policy_id]
    achieved_values: list[Array] = []
    cumulative: Array = jnp.asarray(True, dtype=jnp.bool_)
    for level_index, indices in enumerate(indices_by_level):
        if level_index >= maximum_level:
            level_passed: Array = jnp.asarray(False, dtype=jnp.bool_)
        elif not indices:
            level_passed = jnp.asarray(False, dtype=jnp.bool_)
        else:
            level_passed = jnp.all(valid_claim[jnp.asarray(indices)])
        cumulative = cumulative & level_passed
        achieved_values.append(cumulative)
    id_to_index: dict[str, int] = {
        claim.claim_id: index for index, claim in enumerate(claim_tuple)
    }
    required_indices: Array = jnp.asarray(
        [id_to_index[claim_id] for claim_id in required_ids], dtype=jnp.int32
    )
    claim_passed: Array = all_passed[required_indices]
    claim_checked: Array = all_checked[required_indices]
    claim_in_domain: Array = all_in_domain[required_indices]
    report: PolicyReport = make_policy_report(
        policy_id=policy_id,
        level_ids=CERTIFICATION_LEVEL_IDS,
        required_claim_ids=required_ids,
        claim_passed=claim_passed,
        claim_checked=claim_checked,
        claim_in_domain=claim_in_domain,
        achieved=jnp.stack(achieved_values),
    )
    return report


@jaxtyped(typechecker=beartype)
def achieved_levels(report: PolicyReport) -> tuple[str, ...]:
    """Return certification level names achieved by a concrete report.

    The cumulative policy derives named levels from explicit claims. It retains
    the claim truth table as JAX arrays.

    :see: :class:`~.test_policy.TestAchievedLevels`

    Parameters
    ----------
    report : PolicyReport
        Concrete policy report inspected at the eager boundary.

    Returns
    -------
    levels : tuple[str, ...]
        Achieved level identities in cumulative policy order.

    Notes
    -----
    This eager inspection helper converts the traced Boolean vector to a
    Python tuple and must not be called inside a compiled kernel.
    """
    levels: tuple[str, ...] = tuple(
        level
        for level, achieved in zip(
            report.level_ids, report.achieved.tolist(), strict=True
        )
        if achieved
    )
    return levels


__all__: list[str] = ["achieved_levels", "evaluate_policy"]
