"""Evaluate cumulative scientific-certification policies.

Extended Summary
----------------
Certification levels are derived from traced claim outcomes rather than stored
as trusted labels. Policies select required claim categories and accumulate
from identified through reproducible. A failed, unchecked, or out-of-domain
required claim prevents that level and every level above it.

Routine Listings
----------------
:data:`POLICY_IDS`
    Supported built-in scientific-certification policy identifiers.
:func:`achieved_levels`
    Return the names of levels achieved by a policy report.
:func:`evaluate_policy`
    Derive cumulative certification outcomes from numerical claims.
"""

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Iterable
from jaxtyping import Array

from diffpes.types.certification import (
    CertificationClaim,
    PolicyReport,
    make_policy_report,
)

LEVEL_IDS: tuple[str, ...] = (
    "identified",
    "validated",
    "differentiable",
    "verified",
    "benchmarked",
    "reproducible",
)

POLICY_IDS: tuple[str, ...] = (
    "org.diffpes.policy.exploratory.v1",
    "org.diffpes.policy.research.v1",
    "org.diffpes.policy.publication.v1",
    "org.diffpes.policy.parity.v1",
)

_LEVEL_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("identity", "semantic"),
    ("validation", "domain", "output"),
    ("derivative", "differentiability"),
    ("verification", "reference"),
    ("benchmark", "parity"),
    ("reproduction", "environment"),
)

_POLICY_LEVEL_COUNT: dict[str, int] = {
    "org.diffpes.policy.exploratory.v1": 2,
    "org.diffpes.policy.research.v1": 4,
    "org.diffpes.policy.publication.v1": 6,
    "org.diffpes.policy.parity.v1": 6,
}


def _required_indices(
    claims: tuple[CertificationClaim, ...], policy_id: str
) -> tuple[tuple[tuple[int, ...], ...], tuple[str, ...]]:
    """Select required claims for each cumulative policy level."""
    maximum_level: int = _POLICY_LEVEL_COUNT[policy_id]
    indices_by_level: list[tuple[int, ...]] = []
    required_ids: list[str] = []
    for level_index in range(len(LEVEL_IDS)):
        if level_index >= maximum_level:
            indices_by_level.append(())
            continue
        prefixes: tuple[str, ...] = _LEVEL_PREFIXES[level_index]
        selected: tuple[int, ...] = tuple(
            index
            for index, claim in enumerate(claims)
            if claim.predicate_id.startswith(prefixes)
        )
        indices_by_level.append(selected)
        required_ids.extend(claims[index].claim_id for index in selected)
    return tuple(indices_by_level), tuple(dict.fromkeys(required_ids))


@beartype
def evaluate_policy(
    claims: Iterable[CertificationClaim],
    policy_id: str = "org.diffpes.policy.research.v1",
) -> PolicyReport:
    """Derive cumulative certification outcomes from numerical claims."""
    if policy_id not in POLICY_IDS:
        msg: str = f"unknown certification policy: {policy_id}"
        raise ValueError(msg)
    claim_tuple: tuple[CertificationClaim, ...] = tuple(claims)
    indices_by_level, required_ids = _required_indices(claim_tuple, policy_id)
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
    maximum_level: int = _POLICY_LEVEL_COUNT[policy_id]
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
        level_ids=LEVEL_IDS,
        required_claim_ids=required_ids,
        claim_passed=claim_passed,
        claim_checked=claim_checked,
        claim_in_domain=claim_in_domain,
        achieved=jnp.stack(achieved_values),
    )
    return report


@beartype
def achieved_levels(report: PolicyReport) -> tuple[str, ...]:
    """Return certification level names achieved by a concrete report."""
    levels: tuple[str, ...] = tuple(
        level
        for level, achieved in zip(
            report.level_ids, report.achieved.tolist(), strict=True
        )
        if achieved
    )
    return levels


__all__: list[str] = ["POLICY_IDS", "achieved_levels", "evaluate_policy"]
