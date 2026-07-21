"""Human-readable inspection of forward-model certificates.

Extended Summary
----------------
Provides deterministic, compact views of the scientific model, provenance,
claims, differentiability evidence, and policy outcome recorded by a
``ForwardCertificate``. Inspection is deliberately an eager boundary utility:
it formats already-computed certificate leaves and never evaluates a physics
claim or loads the associated forward result.

Routine Listings
----------------
:class:`CertificateDiff`
    Categorized differences between two forward certificates.
:func:`diff_certificates`
    Compare scientific, numerical, environment, and audit records.
:func:`explain_claim`
    Explain one claim and its registered evidence.
:func:`summarize_certificate`
    Render a stable compact overview of a certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy import ndarray as NDArray  # noqa: N812

from diffpes.types import ForwardCertificate

from .canonical import canonical_pytree

_ARRAY_PREVIEW_ITEMS: int = 8


@dataclass(frozen=True)
class CertificateDiff:
    """Categorized differences between two certificates.

    Attributes
    ----------
    scientific : tuple[str, ...]
        Model, semantics, inputs, transformations, and policy differences.
    numerical : tuple[str, ...]
        Claim outcome and numerical-evidence differences.
    environment : tuple[str, ...]
        Package, source, backend, precision, and environment differences.
    audit : tuple[str, ...]
        Execution identifier and audit timestamp differences.
    """

    scientific: tuple[str, ...]
    numerical: tuple[str, ...]
    environment: tuple[str, ...]
    audit: tuple[str, ...]

    @property
    def identical(self) -> bool:
        """Return whether no categorized difference was found."""
        result: bool = not any(
            (self.scientific, self.numerical, self.environment, self.audit)
        )
        return result

    @property
    def summary(self) -> str:
        """Return a one-line categorized comparison summary."""
        if self.identical:
            return "Certificates are identical."
        parts: list[str] = []
        for label, values in (
            ("scientific", self.scientific),
            ("numerical", self.numerical),
            ("environment", self.environment),
            ("audit", self.audit),
        ):
            if values:
                parts.append(f"{label}: {', '.join(values)}")
        result: str = "; ".join(parts)
        return result


def _scalar_bool(value: Any) -> bool:
    """Convert one concrete scalar array to ``bool`` for display."""
    array: NDArray = np.asarray(value)
    if array.shape != ():
        msg = (
            f"expected scalar certificate field, received shape {array.shape}"
        )
        raise ValueError(msg)
    result: bool = bool(array.item())
    return result


def _scalar_text(value: Any) -> str:
    """Format one concrete scalar numerical certificate field."""
    array: NDArray = np.asarray(value)
    if array.shape != ():
        return f"array(shape={array.shape}, dtype={array.dtype})"
    item: Any = array.item()
    if isinstance(item, float):
        return f"{item:.8g}"
    return str(item)


def _array_text(value: Any) -> str:
    """Format a bounded preview of a numerical evidence array."""
    array: NDArray = np.asarray(value)
    flat: NDArray = array.reshape(-1)
    preview: str = np.array2string(
        flat[:_ARRAY_PREVIEW_ITEMS],
        separator=", ",
        precision=8,
        suppress_small=False,
    )
    if flat.size > _ARRAY_PREVIEW_ITEMS:
        preview = f"{preview[:-1]}, ...]"
    return f"shape={array.shape}, values={preview}"


def _claim_status(claim: Any) -> str:
    """Return the bounded status label for one claim."""
    if not _scalar_bool(claim.checked):
        return "not_checked"
    if not _scalar_bool(claim.in_domain):
        return "out_of_domain"
    if _scalar_bool(claim.passed):
        return "passed"
    return "failed"


def _optional_static_tuple(value: object, name: str) -> tuple[str, ...]:
    """Read an optional tuple-valued static inspection field."""
    field_value: object = getattr(value, name, ())
    if not isinstance(field_value, tuple):
        return ()
    result: tuple[str, ...] = tuple(str(item) for item in field_value)
    return result


def _append_tuple(
    lines: list[str],
    heading: str,
    values: tuple[str, ...],
) -> None:
    """Append one heading and its static values when nonempty."""
    if not values:
        return
    lines.append(f"{heading}:")
    lines.extend(f"  - {value}" for value in values)


def summarize_certificate(certificate: ForwardCertificate) -> str:
    """Return a deterministic human-readable certificate summary.

    Parameters
    ----------
    certificate : ForwardCertificate
        Certificate to summarize. Associated result arrays are neither needed
        nor loaded.

    Returns
    -------
    summary : str
        Compact multiline summary of identity, semantics, provenance, claim
        status, and derivative diagnostics.
    """
    model: Any = certificate.model
    manifest: Any = certificate.manifest
    lines: list[str] = [
        "Forward certificate",
        f"Model: {model.model_id}@{model.model_version}",
        f"Observable: {model.observable_id}",
        f"Implementation: {model.implementation_ref}",
        f"Schema: {manifest.schema_version}",
        f"Policy: {certificate.policy_id}",
        f"Execution: {manifest.execution_id} at {manifest.started_at_utc}",
        "Environment: "
        f"package={manifest.package_version}, backend={manifest.backend}, "
        f"precision={manifest.precision_policy}, "
        f"deterministic={manifest.deterministic}",
    ]

    policy_report: Any = certificate.policy_report
    achieved_flags: tuple[bool, ...] = tuple(
        bool(item) for item in np.asarray(policy_report.achieved).tolist()
    )
    levels: tuple[str, ...] = tuple(
        level
        for level, achieved in zip(
            policy_report.level_ids,
            achieved_flags,
            strict=True,
        )
        if achieved
    )
    if levels:
        lines.append(f"Achieved levels: {', '.join(levels)}")
    unachieved: tuple[str, ...] = tuple(
        level
        for level, achieved in zip(
            policy_report.level_ids,
            achieved_flags,
            strict=True,
        )
        if not achieved
    )
    if unachieved:
        lines.append(f"Unachieved levels: {', '.join(unachieved)}")

    unmet_requirements: tuple[str, ...] = tuple(
        claim_id
        for claim_id, passed, checked, in_domain in zip(
            policy_report.required_claim_ids,
            np.asarray(policy_report.claim_passed).tolist(),
            np.asarray(policy_report.claim_checked).tolist(),
            np.asarray(policy_report.claim_in_domain).tolist(),
            strict=True,
        )
        if not (passed and checked and in_domain)
    )
    if unmet_requirements:
        lines.append(f"Unmet requirements: {', '.join(unmet_requirements)}")

    _append_tuple(lines, "Assumptions", tuple(model.assumptions))

    if model.conventions:
        lines.append("Conventions:")
        lines.extend(
            f"  - {item.convention_id}@{item.version}"
            for item in model.conventions
        )

    if certificate.artifacts:
        lines.append("Artifacts:")
        lines.extend(
            f"  - {item.artifact_id} [{item.role}; {item.media_type}]"
            for item in certificate.artifacts
        )

    if certificate.transformations:
        lines.append("Transformations:")
        for item in certificate.transformations:
            lines.append(
                f"  - {item.transformation_id}@{item.transformation_version}"
            )
            if item.destroys:
                lines.append(f"    destroys: {', '.join(item.destroys)}")
            if item.invalidates_claims:
                lines.append(
                    f"    invalidates: {', '.join(item.invalidates_claims)}"
                )

    status_counts: dict[str, int] = {
        "passed": 0,
        "failed": 0,
        "not_checked": 0,
        "out_of_domain": 0,
    }
    for claim in certificate.claims:
        status_counts[_claim_status(claim)] += 1
    lines.append(
        "Claims: "
        + ", ".join(
            f"{status}={count}" for status, count in status_counts.items()
        )
    )
    lines.extend(
        f"  - {claim.claim_id}: {_claim_status(claim)} "
        f"(margin={_scalar_text(claim.margin)})"
        for claim in certificate.claims
    )

    derivatives: Any = certificate.derivatives
    dependency_structural: NDArray = np.asarray(
        certificate.dependencies.structural
    )
    dependency_traced: NDArray = np.asarray(certificate.dependencies.traced)
    sensitivity_active: NDArray = np.asarray(certificate.sensitivities.active)
    information: Any = certificate.information
    lines.extend(
        (
            f"Derivative method: {derivatives.method}",
            "Derivative inputs: "
            + (
                ", ".join(derivatives.input_paths)
                if derivatives.input_paths
                else "none declared"
            ),
            "Derivative checks: "
            f"finite={_scalar_bool(derivatives.finite)}, "
            f"reference_agreement={_scalar_bool(derivatives.fd_correct)}",
            "Structural dependencies: "
            f"{int(np.sum(dependency_structural))}/"
            f"{dependency_structural.size} input-output pairs",
            "Locally active dependencies: "
            f"{int(np.sum(dependency_traced))}/"
            f"{dependency_traced.size} input-output pairs",
            "Locally active sensitivities: "
            f"{int(np.sum(sensitivity_active))}/"
            f"{sensitivity_active.size} projections "
            f"(threshold={_scalar_text(certificate.sensitivities.threshold)})",
            "Information diagnostics: "
            f"effective_rank={_scalar_text(information.effective_rank)}, "
            "condition_estimate="
            f"{_scalar_text(information.condition_estimate)}",
        )
    )
    summary: str = "\n".join(lines)
    return summary


def explain_claim(
    certificate: ForwardCertificate,
    claim_id: str,
) -> str:
    """Explain one claim and the numerical evidence supporting it.

    Parameters
    ----------
    certificate : ForwardCertificate
        Certificate containing the claim.
    claim_id : str
        Stable claim identifier to explain.

    Returns
    -------
    explanation : str
        Multiline explanation with status, margin, and evidence arrays.

    Raises
    ------
    KeyError
        If the certificate does not contain ``claim_id``.
    """
    matching: tuple[Any, ...] = tuple(
        claim for claim in certificate.claims if claim.claim_id == claim_id
    )
    if not matching:
        msg = f"Claim '{claim_id}' is not present in this certificate."
        raise KeyError(msg)
    claim: Any = matching[0]
    evidence_by_id: dict[str, Any] = {
        item.evidence_id: item for item in certificate.evidence
    }
    lines: list[str] = [
        f"Claim: {claim.claim_id}",
        f"Subject: {claim.subject_id}",
        f"Predicate: {claim.predicate_id}",
        f"Status: {_claim_status(claim)}",
        f"Margin: {_scalar_text(claim.margin)}",
        f"Severity code: {_scalar_text(claim.severity_code)}",
    ]
    if not claim.evidence_ids:
        lines.append("Evidence: none recorded")
    else:
        lines.append("Evidence:")
    for evidence_id in claim.evidence_ids:
        evidence: Any | None = evidence_by_id.get(evidence_id)
        if evidence is None:
            lines.append(f"  - {evidence_id}: missing from certificate")
            continue
        lines.extend(
            (
                f"  - {evidence.evidence_id}",
                f"    method: {evidence.method_id}",
                f"    source: {evidence.source_type}",
                f"    independent: {evidence.independent}",
                f"    measured: {_array_text(evidence.measured)}",
                f"    reference: {_array_text(evidence.reference)}",
                f"    residual: {_array_text(evidence.residual)}",
                f"    tolerance: {_array_text(evidence.tolerance)}",
            )
        )
    explanation: str = "\n".join(lines)
    return explanation


def _different(left: object, right: object) -> bool:
    """Return deterministic semantic inequality for supported records."""
    result: bool = canonical_pytree(left) != canonical_pytree(right)
    return result


def _field_differences(
    left: object,
    right: object,
    names: tuple[str, ...],
) -> tuple[str, ...]:
    """Return names whose values differ under canonical comparison."""
    result: tuple[str, ...] = tuple(
        name
        for name in names
        if _different(getattr(left, name), getattr(right, name))
    )
    return result


def _artifact_identity(certificate: ForwardCertificate) -> tuple[object, ...]:
    """Return meaning-bearing artifact fields without local locators."""
    result: tuple[object, ...] = tuple(
        (
            item.artifact_id,
            item.media_type,
            item.byte_checksum,
            item.content_checksum,
            item.semantic_checksum,
            item.role,
        )
        for item in certificate.artifacts
    )
    return result


def diff_certificates(
    left: ForwardCertificate,
    right: ForwardCertificate,
) -> CertificateDiff:
    """Compare two certificates by scientific meaning and record class.

    Parameters
    ----------
    left : ForwardCertificate
        Reference certificate.
    right : ForwardCertificate
        Candidate certificate.

    Returns
    -------
    difference : CertificateDiff
        Stable field-name differences grouped into scientific, numerical,
        environment, and audit categories.
    """
    scientific_names: tuple[str, ...] = (
        "model",
        "transformations",
        "domains",
        "policy_id",
        "extensions_json",
    )
    numerical_names: tuple[str, ...] = (
        "evidence",
        "claims",
        "derivatives",
        "dependencies",
        "sensitivities",
        "information",
        "policy_report",
        "certificate_checksum",
    )
    environment_names: tuple[str, ...] = (
        "schema_version",
        "package_version",
        "source_checksum",
        "environment_checksum",
        "backend",
        "precision_policy",
        "deterministic",
    )
    audit_names: tuple[str, ...] = ("execution_id", "started_at_utc")
    scientific: list[str] = list(
        _field_differences(left, right, scientific_names)
    )
    if _different(_artifact_identity(left), _artifact_identity(right)):
        scientific.append("artifacts")
    audit: list[str] = list(
        _field_differences(
            left.manifest,
            right.manifest,
            audit_names,
        )
    )
    left_locators: tuple[str | None, ...] = tuple(
        item.locator for item in left.artifacts
    )
    right_locators: tuple[str | None, ...] = tuple(
        item.locator for item in right.artifacts
    )
    if _different(left_locators, right_locators):
        audit.append("artifact_locators")
    difference = CertificateDiff(
        scientific=tuple(scientific),
        numerical=_field_differences(left, right, numerical_names),
        environment=_field_differences(
            left.manifest,
            right.manifest,
            environment_names,
        ),
        audit=tuple(audit),
    )
    return difference  # noqa: RET504


__all__: list[str] = [
    "CertificateDiff",
    "diff_certificates",
    "explain_claim",
    "summarize_certificate",
]
