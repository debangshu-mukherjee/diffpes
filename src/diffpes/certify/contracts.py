"""Semantic contracts for composable certified transformations.

Extended Summary
----------------
Transformation contracts state the meaning required and produced by a step,
which properties are explicitly preserved, what information is destroyed,
and which prior claims cease to apply.  Composition is conservative: an
inherited property survives only when the next transformation names it in
``preserves``.

Contracts are static Equinox PyTrees.  They select and describe JAX programs;
continuous numerical checks remain in the certification carriers and kernels.

Routine Listings
----------------
:class:`TransformationContract`
    Immutable semantic and information-loss declaration.
:func:`make_transformation_contract`
    Validate and normalize a contract.
:func:`validate_composition`
    Propagate semantics, losses, and claim invalidations through a pipeline.
:func:`compose_transformations`
    Return a valid report or raise a detailed composition error.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

import equinox as eqx

_IDENTIFIER_RE: re.Pattern[str] = re.compile(
    r"^[a-z][a-z0-9]*(?:\.[a-z][a-z0-9_]*)+$"
)
_SEMVER_RE: re.Pattern[str] = re.compile(
    r"^(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


class TransformationContract(eqx.Module):
    """Static semantic contract for one registered transformation.

    Attributes
    ----------
    transformation_id : str
        Permanent reverse-DNS-like scientific identity.
    transformation_version : str
        Semantic version of the scientific transformation contract.
    requires : tuple[str, ...]
        Properties required on entry.
    produces : tuple[str, ...]
        Semantic identities of outputs produced by the step.
    preserves : tuple[str, ...]
        Inherited properties explicitly guaranteed to remain valid.
    introduces : tuple[str, ...]
        New semantic properties or assumptions introduced by the step.
    destroys : tuple[str, ...]
        Information explicitly removed or made unrecoverable.
    invalidates_claims : tuple[str, ...]
        Prior scientific claim identities that cannot survive the step.
    jax_pure : bool
        Whether the numerical transformation is a pure JAX program.
    """

    transformation_id: str = eqx.field(static=True)
    transformation_version: str = eqx.field(static=True)
    requires: tuple[str, ...] = eqx.field(static=True)
    produces: tuple[str, ...] = eqx.field(static=True)
    preserves: tuple[str, ...] = eqx.field(static=True)
    introduces: tuple[str, ...] = eqx.field(static=True)
    destroys: tuple[str, ...] = eqx.field(static=True)
    invalidates_claims: tuple[str, ...] = eqx.field(static=True)
    jax_pure: bool = eqx.field(static=True)


class CompositionReport(eqx.Module):
    """Immutable result of conservative transformation composition."""

    valid: bool = eqx.field(static=True)
    errors: tuple[str, ...] = eqx.field(static=True)
    available_semantics: tuple[str, ...] = eqx.field(static=True)
    destroyed_information: tuple[str, ...] = eqx.field(static=True)
    invalidated_claims: tuple[str, ...] = eqx.field(static=True)
    transformation_refs: tuple[str, ...] = eqx.field(static=True)


class ContractError(ValueError):
    """Report an invalid transformation contract or composition."""


def _normalize_terms(
    values: Iterable[str],
    *,
    field_name: str,
) -> tuple[str, ...]:
    """Normalize a semantic term sequence without hiding duplicates."""
    normalized = tuple(values)
    if any(
        not isinstance(value, str) or not value.strip() for value in normalized
    ):
        msg = f"{field_name} entries must be nonblank strings"
        raise ContractError(msg)
    if len(set(normalized)) != len(normalized):
        msg = f"{field_name} entries must be unique"
        raise ContractError(msg)
    return normalized


def _validate_identity(transformation_id: str, version: str) -> None:
    """Validate permanent identity and scientific semantic version."""
    if _IDENTIFIER_RE.fullmatch(transformation_id) is None:
        msg = "transformation_id must be a lowercase reverse-DNS-like ID"
        raise ContractError(msg)
    if _SEMVER_RE.fullmatch(version) is None:
        msg = "transformation_version must be a semantic version"
        raise ContractError(msg)


def make_transformation_contract(
    transformation_id: str,
    transformation_version: str,
    *,
    requires: Sequence[str] = (),
    produces: Sequence[str] = (),
    preserves: Sequence[str] = (),
    introduces: Sequence[str] = (),
    destroys: Sequence[str] = (),
    invalidates_claims: Sequence[str] = (),
    jax_pure: bool = True,
) -> TransformationContract:
    """Create a validated immutable transformation contract.

    Parameters
    ----------
    transformation_id : str
        Permanent transformation identity.
    transformation_version : str
        Scientific semantic version.
    requires, produces, preserves, introduces, destroys : Sequence[str]
        Semantic contract declarations.
    invalidates_claims : Sequence[str]
        Claims explicitly invalidated by applying the transformation.
    jax_pure : bool, optional
        Whether the numerical operation can execute as a pure JAX program.

    Returns
    -------
    contract : TransformationContract
        Validated static Equinox carrier.

    Raises
    ------
    ContractError
        If identity, version, terms, or loss declarations are inconsistent.
    """
    _validate_identity(transformation_id, transformation_version)
    normalized_requires = _normalize_terms(requires, field_name="requires")
    normalized_produces = _normalize_terms(produces, field_name="produces")
    normalized_preserves = _normalize_terms(preserves, field_name="preserves")
    normalized_introduces = _normalize_terms(
        introduces,
        field_name="introduces",
    )
    normalized_destroys = _normalize_terms(destroys, field_name="destroys")
    normalized_invalidates = _normalize_terms(
        invalidates_claims,
        field_name="invalidates_claims",
    )
    contradictory = set(normalized_preserves) & set(normalized_destroys)
    contradictory |= set(normalized_introduces) & set(normalized_destroys)
    if contradictory:
        terms = ", ".join(sorted(contradictory))
        msg = f"a contract cannot preserve/introduce and destroy: {terms}"
        raise ContractError(msg)
    return TransformationContract(
        transformation_id=transformation_id,
        transformation_version=transformation_version,
        requires=normalized_requires,
        produces=normalized_produces,
        preserves=normalized_preserves,
        introduces=normalized_introduces,
        destroys=normalized_destroys,
        invalidates_claims=normalized_invalidates,
        jax_pure=bool(jax_pure),
    )


def validate_contract(contract: TransformationContract) -> tuple[str, ...]:
    """Return structural errors for a raw or deserialized contract."""
    errors: list[str] = []
    try:
        make_transformation_contract(
            contract.transformation_id,
            contract.transformation_version,
            requires=contract.requires,
            produces=contract.produces,
            preserves=contract.preserves,
            introduces=contract.introduces,
            destroys=contract.destroys,
            invalidates_claims=contract.invalidates_claims,
            jax_pure=contract.jax_pure,
        )
    except (ContractError, TypeError) as exc:
        errors.append(str(exc))
    return tuple(errors)


def validate_composition(
    contracts: Sequence[TransformationContract],
    *,
    initial_semantics: Iterable[str] = (),
) -> CompositionReport:
    """Validate and conservatively compose transformation semantics.

    Parameters
    ----------
    contracts : Sequence[TransformationContract]
        Ordered transformation pipeline.
    initial_semantics : Iterable[str], optional
        Properties available on the pipeline inputs.

    Returns
    -------
    report : CompositionReport
        Deterministic semantics, information-loss, and claim-invalidation
        summary. Invalid composition is reported rather than raised.

    Notes
    -----
    A property not named in ``preserves`` is conservatively considered lost
    at that step. This prevents an undeclared transformation from silently
    carrying a scientific claim forward.
    """
    initial = _normalize_terms(
        initial_semantics,
        field_name="initial_semantics",
    )
    current = set(initial)
    losses: set[str] = set()
    invalidated: set[str] = set()
    errors: list[str] = []
    references: list[str] = []
    for index, contract in enumerate(contracts):
        reference = (
            f"{contract.transformation_id}@{contract.transformation_version}"
        )
        references.append(reference)
        for error in validate_contract(contract):
            errors.append(f"step {index} ({reference}): {error}")
        missing = set(contract.requires) - current
        if missing:
            details = ", ".join(sorted(missing))
            errors.append(
                f"step {index} ({reference}) is missing required semantics: "
                f"{details}"
            )
        retained = current & set(contract.preserves)
        losses.update(current - retained)
        losses.update(contract.destroys)
        current = retained
        current.difference_update(contract.destroys)
        current.update(contract.introduces)
        current.update(contract.produces)
        invalidated.update(contract.invalidates_claims)
    return CompositionReport(
        valid=not errors,
        errors=tuple(errors),
        available_semantics=tuple(sorted(current)),
        destroyed_information=tuple(sorted(losses)),
        invalidated_claims=tuple(sorted(invalidated)),
        transformation_refs=tuple(references),
    )


def compose_transformations(
    contracts: Sequence[TransformationContract],
    *,
    initial_semantics: Iterable[str] = (),
) -> CompositionReport:
    """Compose contracts and raise if any requirement is unsatisfied.

    Parameters
    ----------
    contracts : Sequence[TransformationContract]
        Ordered transformation pipeline.
    initial_semantics : Iterable[str], optional
        Properties present before the first step.

    Returns
    -------
    report : CompositionReport
        Valid composition report.

    Raises
    ------
    ContractError
        If any registered semantic requirement is missing.
    """
    report = validate_composition(
        contracts,
        initial_semantics=initial_semantics,
    )
    if not report.valid:
        raise ContractError("; ".join(report.errors))
    return report


__all__: list[str] = [
    "CompositionReport",
    "ContractError",
    "TransformationContract",
    "compose_transformations",
    "make_transformation_contract",
    "validate_composition",
    "validate_contract",
]
