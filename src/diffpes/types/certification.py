"""JAX-native carriers for certified forward-model executions.

The classes in this module separate static scientific vocabulary from traced
numerical evidence.  Identifiers, conventions, paths, and schema selections
are Equinox static fields; measurements, residuals, margins, sensitivities,
and status arrays remain ordinary JAX leaves.  This lets certification travel
through ``filter_jit``, ``vmap``, JVP, and VJP without moving bookkeeping into
the forward-model kernel.
"""

import json

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any, Optional
from jaxtyping import Array, Bool, Float, Int, jaxtyped


class ArtifactRef(eqx.Module):
    """Static identity and role of one source or derived artifact."""

    artifact_id: str = eqx.field(static=True)
    media_type: str = eqx.field(static=True)
    byte_checksum: Optional[str] = eqx.field(static=True)
    content_checksum: str = eqx.field(static=True)
    semantic_checksum: str = eqx.field(static=True)
    locator: Optional[str] = eqx.field(static=True)
    role: str = eqx.field(static=True)


class ConventionRef(eqx.Module):
    """Versioned semantic convention used by a scientific model."""

    convention_id: str = eqx.field(static=True)
    version: str = eqx.field(static=True)
    parameters_json: str = eqx.field(static=True)


class DomainPredicate(eqx.Module):
    """Static declaration of one model-domain predicate."""

    predicate_id: str = eqx.field(static=True)
    expression_id: str = eqx.field(static=True)
    units: Optional[str] = eqx.field(static=True)
    severity: str = eqx.field(static=True)


class DomainResult(eqx.Module):
    """Traced evaluation of one declared domain predicate."""

    predicate_id: str = eqx.field(static=True)
    measured: Float[Array, ""]
    reference: Float[Array, ""]
    residual: Float[Array, ""]
    tolerance: Float[Array, ""]
    margin: Float[Array, ""]
    passed: Bool[Array, ""]
    checked: Bool[Array, ""]
    in_domain: Bool[Array, ""]
    severity_code: Int[Array, ""]


class ForwardModelSpec(eqx.Module):
    """Stable scientific and differentiability identity of a forward model."""

    model_id: str = eqx.field(static=True)
    model_version: str = eqx.field(static=True)
    observable_id: str = eqx.field(static=True)
    implementation_ref: str = eqx.field(static=True)
    assumptions: tuple[str, ...] = eqx.field(static=True)
    conventions: tuple[ConventionRef, ...] = eqx.field(static=True)
    domain: tuple[DomainPredicate, ...] = eqx.field(static=True)
    differentiable_paths: tuple[str, ...] = eqx.field(static=True)
    nondifferentiable_paths: tuple[str, ...] = eqx.field(static=True)


class TransformationRecord(eqx.Module):
    """One ordered transformation and its semantic information effects."""

    transformation_id: str = eqx.field(static=True)
    transformation_version: str = eqx.field(static=True)
    parent_ids: tuple[str, ...] = eqx.field(static=True)
    output_ids: tuple[str, ...] = eqx.field(static=True)
    preserves: tuple[str, ...] = eqx.field(static=True)
    introduces: tuple[str, ...] = eqx.field(static=True)
    destroys: tuple[str, ...] = eqx.field(static=True)
    invalidates_claims: tuple[str, ...] = eqx.field(static=True)
    parameters_checksum: str = eqx.field(static=True)


class EvidenceRef(eqx.Module):
    """Numerical evidence with a static method and source identity."""

    evidence_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    artifact_refs: tuple[str, ...] = eqx.field(static=True)
    source_type: str = eqx.field(static=True)
    independent: bool = eqx.field(static=True)
    measured: Float[Array, " n_measure"]
    reference: Float[Array, " n_measure"]
    residual: Float[Array, " n_measure"]
    tolerance: Float[Array, " n_measure"]


class CertificationClaim(eqx.Module):
    """A named assurance claim and its continuous numerical evidence."""

    claim_id: str = eqx.field(static=True)
    subject_id: str = eqx.field(static=True)
    predicate_id: str = eqx.field(static=True)
    evidence_ids: tuple[str, ...] = eqx.field(static=True)
    measured: Float[Array, " n_measure"]
    reference: Float[Array, " n_measure"]
    residual: Float[Array, " n_measure"]
    tolerance: Float[Array, " n_measure"]
    passed: Bool[Array, ""]
    checked: Bool[Array, ""]
    in_domain: Bool[Array, ""]
    margin: Float[Array, ""]
    severity_code: Int[Array, ""]


class DerivativeEvidence(eqx.Module):
    """JVP, VJP, reference, and local information-spectrum evidence."""

    input_paths: tuple[str, ...] = eqx.field(static=True)
    output_projection_ids: tuple[str, ...] = eqx.field(static=True)
    method: str = eqx.field(static=True)
    scales: Float[Array, " n_input"]
    jvp_probes: Float[Array, "n_probe n_output"]
    vjp_probes: Float[Array, "n_probe n_input"]
    reference_derivatives: Float[Array, "n_probe n_deriv"]
    derivative_residuals: Float[Array, "n_probe n_deriv"]
    singular_values: Float[Array, " n_sv"]
    effective_rank: Int[Array, ""]
    condition_estimate: Float[Array, ""]
    finite: Bool[Array, ""]
    fd_correct: Bool[Array, ""]


class DependencyMap(eqx.Module):
    """Declared and JAXPR-observed structural dependency relations."""

    model_id: str = eqx.field(static=True)
    input_paths: tuple[str, ...] = eqx.field(static=True)
    output_paths: tuple[str, ...] = eqx.field(static=True)
    structural: Bool[Array, "n_output n_input"]
    traced: Bool[Array, "n_output n_input"]


class SensitivityMap(eqx.Module):
    """Scaled local sensitivities from named inputs to output projections."""

    input_paths: tuple[str, ...] = eqx.field(static=True)
    output_projection_ids: tuple[str, ...] = eqx.field(static=True)
    scales: Float[Array, " n_input"]
    sensitivities: Float[Array, "n_output n_input"]
    threshold: Float[Array, ""]
    active: Bool[Array, "n_output n_input"]


class InformationSpectrum(eqx.Module):
    """Matrix-free local information spectrum in named input coordinates."""

    input_paths: tuple[str, ...] = eqx.field(static=True)
    singular_values: Float[Array, " n_sv"]
    right_singular_vectors: Float[Array, "n_sv n_input"]
    effective_rank: Int[Array, ""]
    condition_estimate: Float[Array, ""]
    threshold: Float[Array, ""]


class ExecutionManifest(eqx.Module):
    """Static software and execution identity prepared at the I/O boundary."""

    execution_id: str = eqx.field(static=True)
    model_ref: str = eqx.field(static=True)
    schema_version: str = eqx.field(static=True)
    package_version: str = eqx.field(static=True)
    source_checksum: str = eqx.field(static=True)
    environment_checksum: str = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    precision_policy: str = eqx.field(static=True)
    deterministic: bool = eqx.field(static=True)
    started_at_utc: str = eqx.field(static=True)


class PolicyReport(eqx.Module):
    """Traced policy truth table from which achieved levels are derived."""

    policy_id: str = eqx.field(static=True)
    level_ids: tuple[str, ...] = eqx.field(static=True)
    required_claim_ids: tuple[str, ...] = eqx.field(static=True)
    claim_passed: Bool[Array, " n_claim"]
    claim_checked: Bool[Array, " n_claim"]
    claim_in_domain: Bool[Array, " n_claim"]
    achieved: Bool[Array, " n_level"]


class CertificationContext(eqx.Module):
    """Prepared static selections and references for compiled certification."""

    manifest: ExecutionManifest
    model: ForwardModelSpec
    artifacts: tuple[ArtifactRef, ...]
    transformations: tuple[TransformationRecord, ...]
    evidence: tuple[EvidenceRef, ...]
    policy_id: str = eqx.field(static=True)
    check_ids: tuple[str, ...] = eqx.field(static=True)
    input_checksums: tuple[str, ...] = eqx.field(static=True)


class ForwardCertificate(eqx.Module):
    """Complete scientific-assurance record for one forward execution."""

    manifest: ExecutionManifest
    model: ForwardModelSpec
    artifacts: tuple[ArtifactRef, ...]
    transformations: tuple[TransformationRecord, ...]
    evidence: tuple[EvidenceRef, ...]
    claims: tuple[CertificationClaim, ...]
    domains: tuple[DomainResult, ...]
    derivatives: DerivativeEvidence
    dependencies: DependencyMap
    sensitivities: SensitivityMap
    information: InformationSpectrum
    policy_report: PolicyReport
    policy_id: str = eqx.field(static=True)
    certificate_checksum: str = eqx.field(static=True)
    extensions_json: str = eqx.field(static=True)


class CertifiedResult(eqx.Module):
    """A numerical result paired with its differentiable certificate."""

    value: Any
    certificate: ForwardCertificate


class EvidenceReport(eqx.Module):
    """Offline resolution and consistency outcome for one evidence record."""

    evidence_id: str = eqx.field(static=True)
    resolved: Bool[Array, ""]
    compatible: Bool[Array, ""]
    passed: Bool[Array, ""]
    residual_norm: Float[Array, ""]


class VerificationReport(eqx.Module):
    """Offline structural, evidence, and policy verification outcome."""

    certificate_checksum: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    structure_valid: Bool[Array, ""]
    evidence_valid: Bool[Array, ""]
    policy_report: PolicyReport


class ReproductionReport(eqx.Module):
    """Numerical comparison from a deliberate forward re-execution."""

    execution_id: str = eqx.field(static=True)
    result_checksum: str = eqx.field(static=True)
    reproduced: Bool[Array, ""]
    max_abs_error: Float[Array, ""]
    max_rel_error: Float[Array, ""]
    tolerance: Float[Array, ""]


def _require_text(value: str, name: str) -> str:
    """Reject empty static vocabulary entries."""
    if not value.strip():
        raise ValueError(f"{name} must be non-empty")
    return value


def _require_optional_text(value: Optional[str], name: str) -> Optional[str]:
    """Reject an explicitly supplied empty optional string."""
    if value is not None:
        return _require_text(value, name)
    return value


def _text_tuple(values: tuple[str, ...], name: str) -> tuple[str, ...]:
    """Normalize and validate a tuple of identifiers."""
    result = tuple(_require_text(value, name) for value in values)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _json_object(value: str, name: str) -> str:
    """Require a JSON object while preserving the supplied representation."""
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} must be valid JSON") from error
    if not isinstance(decoded, dict):
        raise ValueError(f"{name} must encode a JSON object")
    return value


def _float(value: Any, name: str, ndim: int) -> Array:
    """Cast a numerical value to float64 and enforce its rank."""
    array = jnp.asarray(value, dtype=jnp.float64)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    return array


def _bool(value: Any, name: str, ndim: int) -> Array:
    """Cast a logical value to bool and enforce its rank."""
    array = jnp.asarray(value, dtype=jnp.bool_)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    return array


def _int(value: Any, name: str, ndim: int) -> Array:
    """Cast an integer value to int32 and enforce its rank."""
    array = jnp.asarray(value, dtype=jnp.int32)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    return array


def _nonnegative(array: Array, name: str) -> Array:
    """Require finite, nonnegative tolerance-like leaves under JIT."""
    return eqx.error_if(
        array,
        ~jnp.all(jnp.isfinite(array) & (array >= 0.0)),
        f"{name} must be finite and nonnegative",
    )


def _positive(array: Array, name: str) -> Array:
    """Require finite, positive scale leaves under JIT."""
    return eqx.error_if(
        array,
        ~jnp.all(jnp.isfinite(array) & (array > 0.0)),
        f"{name} must be finite and positive",
    )


@jaxtyped(typechecker=beartype)
def make_artifact_ref(
    artifact_id: str,
    media_type: str,
    byte_checksum: Optional[str],
    content_checksum: str,
    semantic_checksum: str,
    locator: Optional[str],
    role: str,
) -> ArtifactRef:
    """Create a validated artifact reference."""
    return ArtifactRef(
        artifact_id=_require_text(artifact_id, "artifact_id"),
        media_type=_require_text(media_type, "media_type"),
        byte_checksum=_require_optional_text(byte_checksum, "byte_checksum"),
        content_checksum=_require_text(content_checksum, "content_checksum"),
        semantic_checksum=_require_text(
            semantic_checksum, "semantic_checksum"
        ),
        locator=_require_optional_text(locator, "locator"),
        role=_require_text(role, "role"),
    )


@jaxtyped(typechecker=beartype)
def make_convention_ref(
    convention_id: str,
    version: str,
    parameters_json: str = "{}",
) -> ConventionRef:
    """Create a validated convention reference."""
    return ConventionRef(
        convention_id=_require_text(convention_id, "convention_id"),
        version=_require_text(version, "version"),
        parameters_json=_json_object(parameters_json, "parameters_json"),
    )


@jaxtyped(typechecker=beartype)
def make_domain_predicate(
    predicate_id: str,
    expression_id: str,
    units: Optional[str] = None,
    severity: str = "error",
) -> DomainPredicate:
    """Create a validated domain-predicate declaration."""
    return DomainPredicate(
        predicate_id=_require_text(predicate_id, "predicate_id"),
        expression_id=_require_text(expression_id, "expression_id"),
        units=_require_optional_text(units, "units"),
        severity=_require_text(severity, "severity"),
    )


@jaxtyped(typechecker=beartype)
def make_domain_result(
    predicate_id: str,
    measured: Any,
    reference: Any,
    residual: Any,
    tolerance: Any,
    margin: Any,
    passed: Any,
    checked: Any = True,
    in_domain: Any = True,
    severity_code: Any = 0,
) -> DomainResult:
    """Create one traced domain evaluation."""
    return DomainResult(
        predicate_id=_require_text(predicate_id, "predicate_id"),
        measured=_float(measured, "measured", 0),
        reference=_float(reference, "reference", 0),
        residual=_float(residual, "residual", 0),
        tolerance=_nonnegative(_float(tolerance, "tolerance", 0), "tolerance"),
        margin=_float(margin, "margin", 0),
        passed=_bool(passed, "passed", 0),
        checked=_bool(checked, "checked", 0),
        in_domain=_bool(in_domain, "in_domain", 0),
        severity_code=_int(severity_code, "severity_code", 0),
    )


@jaxtyped(typechecker=beartype)
def make_forward_model_spec(
    model_id: str,
    model_version: str,
    observable_id: str,
    implementation_ref: str,
    assumptions: tuple[str, ...] = (),
    conventions: tuple[ConventionRef, ...] = (),
    domain: tuple[DomainPredicate, ...] = (),
    differentiable_paths: tuple[str, ...] = (),
    nondifferentiable_paths: tuple[str, ...] = (),
) -> ForwardModelSpec:
    """Create a validated stable forward-model specification."""
    diff_paths = _text_tuple(differentiable_paths, "differentiable_paths")
    nondiff_paths = _text_tuple(
        nondifferentiable_paths, "nondifferentiable_paths"
    )
    overlap = set(diff_paths).intersection(nondiff_paths)
    if overlap:
        raise ValueError(
            "differentiable_paths and nondifferentiable_paths must be disjoint"
        )
    convention_ids = tuple(item.convention_id for item in conventions)
    predicate_ids = tuple(item.predicate_id for item in domain)
    _text_tuple(convention_ids, "convention ids")
    _text_tuple(predicate_ids, "domain predicate ids")
    return ForwardModelSpec(
        model_id=_require_text(model_id, "model_id"),
        model_version=_require_text(model_version, "model_version"),
        observable_id=_require_text(observable_id, "observable_id"),
        implementation_ref=_require_text(
            implementation_ref, "implementation_ref"
        ),
        assumptions=_text_tuple(assumptions, "assumptions"),
        conventions=tuple(conventions),
        domain=tuple(domain),
        differentiable_paths=diff_paths,
        nondifferentiable_paths=nondiff_paths,
    )


@jaxtyped(typechecker=beartype)
def make_transformation_record(
    transformation_id: str,
    transformation_version: str,
    parent_ids: tuple[str, ...],
    output_ids: tuple[str, ...],
    preserves: tuple[str, ...] = (),
    introduces: tuple[str, ...] = (),
    destroys: tuple[str, ...] = (),
    invalidates_claims: tuple[str, ...] = (),
    parameters_checksum: str = "none",
) -> TransformationRecord:
    """Create a validated information-aware transformation record."""
    if not output_ids:
        raise ValueError("output_ids must be non-empty")
    return TransformationRecord(
        transformation_id=_require_text(
            transformation_id, "transformation_id"
        ),
        transformation_version=_require_text(
            transformation_version, "transformation_version"
        ),
        parent_ids=_text_tuple(parent_ids, "parent_ids"),
        output_ids=_text_tuple(output_ids, "output_ids"),
        preserves=_text_tuple(preserves, "preserves"),
        introduces=_text_tuple(introduces, "introduces"),
        destroys=_text_tuple(destroys, "destroys"),
        invalidates_claims=_text_tuple(
            invalidates_claims, "invalidates_claims"
        ),
        parameters_checksum=_require_text(
            parameters_checksum, "parameters_checksum"
        ),
    )


@jaxtyped(typechecker=beartype)
def make_evidence_ref(
    evidence_id: str,
    method_id: str,
    artifact_refs: tuple[str, ...],
    source_type: str,
    independent: bool,
    measured: Any,
    reference: Any,
    residual: Any,
    tolerance: Any,
) -> EvidenceRef:
    """Create validated vector-valued numerical evidence."""
    measured_array = _float(measured, "measured", 1)
    reference_array = _float(reference, "reference", 1)
    residual_array = _float(residual, "residual", 1)
    tolerance_array = _nonnegative(
        _float(tolerance, "tolerance", 1), "tolerance"
    )
    shape = measured_array.shape
    if not (
        reference_array.shape
        == residual_array.shape
        == tolerance_array.shape
        == shape
    ):
        raise ValueError("evidence numerical arrays must have equal shapes")
    return EvidenceRef(
        evidence_id=_require_text(evidence_id, "evidence_id"),
        method_id=_require_text(method_id, "method_id"),
        artifact_refs=_text_tuple(artifact_refs, "artifact_refs"),
        source_type=_require_text(source_type, "source_type"),
        independent=independent,
        measured=measured_array,
        reference=reference_array,
        residual=residual_array,
        tolerance=tolerance_array,
    )


@jaxtyped(typechecker=beartype)
def make_certification_claim(  # noqa: PLR0913
    claim_id: str,
    subject_id: str,
    predicate_id: str,
    evidence_ids: tuple[str, ...],
    measured: Any,
    reference: Any,
    residual: Any,
    tolerance: Any,
    passed: Any,
    checked: Any = True,
    in_domain: Any = True,
    margin: Any = 0.0,
    severity_code: Any = 0,
) -> CertificationClaim:
    """Create a claim retaining both continuous and discrete evidence."""
    measured_array = _float(measured, "measured", 1)
    reference_array = _float(reference, "reference", 1)
    residual_array = _float(residual, "residual", 1)
    tolerance_array = _nonnegative(
        _float(tolerance, "tolerance", 1), "tolerance"
    )
    shape = measured_array.shape
    if not (
        reference_array.shape
        == residual_array.shape
        == tolerance_array.shape
        == shape
    ):
        raise ValueError("claim numerical arrays must have equal shapes")
    return CertificationClaim(
        claim_id=_require_text(claim_id, "claim_id"),
        subject_id=_require_text(subject_id, "subject_id"),
        predicate_id=_require_text(predicate_id, "predicate_id"),
        evidence_ids=_text_tuple(evidence_ids, "evidence_ids"),
        measured=measured_array,
        reference=reference_array,
        residual=residual_array,
        tolerance=tolerance_array,
        passed=_bool(passed, "passed", 0),
        checked=_bool(checked, "checked", 0),
        in_domain=_bool(in_domain, "in_domain", 0),
        margin=_float(margin, "margin", 0),
        severity_code=_int(severity_code, "severity_code", 0),
    )


@jaxtyped(typechecker=beartype)
def make_derivative_evidence(  # noqa: PLR0913
    input_paths: tuple[str, ...],
    output_projection_ids: tuple[str, ...],
    method: str,
    scales: Any,
    jvp_probes: Any,
    vjp_probes: Any,
    reference_derivatives: Any,
    derivative_residuals: Any,
    singular_values: Any,
    effective_rank: Any,
    condition_estimate: Any,
    finite: Any,
    fd_correct: Any,
) -> DerivativeEvidence:
    """Create validated derivative and local-information evidence."""
    paths = _text_tuple(input_paths, "input_paths")
    projections = _text_tuple(output_projection_ids, "output_projection_ids")
    scales_array = _positive(_float(scales, "scales", 1), "scales")
    jvp_array = _float(jvp_probes, "jvp_probes", 2)
    vjp_array = _float(vjp_probes, "vjp_probes", 2)
    reference_array = _float(reference_derivatives, "reference_derivatives", 2)
    residual_array = _float(derivative_residuals, "derivative_residuals", 2)
    singular_array = _nonnegative(
        _float(singular_values, "singular_values", 1), "singular_values"
    )
    if scales_array.shape[0] != len(paths):
        raise ValueError("scales length must equal input_paths length")
    if vjp_array.shape[1] != len(paths):
        raise ValueError("vjp_probes input dimension must equal input_paths")
    if jvp_array.shape[0] != vjp_array.shape[0]:
        raise ValueError("JVP and VJP probe counts must agree")
    if reference_array.shape != residual_array.shape:
        raise ValueError("reference and derivative residual shapes must agree")
    if reference_array.shape[0] != jvp_array.shape[0]:
        raise ValueError("reference derivative probe count must agree")
    return DerivativeEvidence(
        input_paths=paths,
        output_projection_ids=projections,
        method=_require_text(method, "method"),
        scales=scales_array,
        jvp_probes=jvp_array,
        vjp_probes=vjp_array,
        reference_derivatives=reference_array,
        derivative_residuals=residual_array,
        singular_values=singular_array,
        effective_rank=_int(effective_rank, "effective_rank", 0),
        condition_estimate=_float(condition_estimate, "condition_estimate", 0),
        finite=_bool(finite, "finite", 0),
        fd_correct=_bool(fd_correct, "fd_correct", 0),
    )


@jaxtyped(typechecker=beartype)
def make_dependency_map(
    model_id: str,
    input_paths: tuple[str, ...],
    output_paths: tuple[str, ...],
    structural: Any,
    traced: Any,
) -> DependencyMap:
    """Create a structural dependency map."""
    inputs = _text_tuple(input_paths, "input_paths")
    outputs = _text_tuple(output_paths, "output_paths")
    structural_array = _bool(structural, "structural", 2)
    traced_array = _bool(traced, "traced", 2)
    expected = (len(outputs), len(inputs))
    if structural_array.shape != expected or traced_array.shape != expected:
        raise ValueError(
            "dependency matrices must have shape (outputs, inputs)"
        )
    return DependencyMap(
        model_id=_require_text(model_id, "model_id"),
        input_paths=inputs,
        output_paths=outputs,
        structural=structural_array,
        traced=traced_array,
    )


@jaxtyped(typechecker=beartype)
def make_sensitivity_map(
    input_paths: tuple[str, ...],
    output_projection_ids: tuple[str, ...],
    scales: Any,
    sensitivities: Any,
    threshold: Any,
    active: Any,
) -> SensitivityMap:
    """Create a named, scaled local-sensitivity map."""
    inputs = _text_tuple(input_paths, "input_paths")
    outputs = _text_tuple(output_projection_ids, "output_projection_ids")
    scales_array = _positive(_float(scales, "scales", 1), "scales")
    sensitivities_array = _float(sensitivities, "sensitivities", 2)
    active_array = _bool(active, "active", 2)
    expected = (len(outputs), len(inputs))
    if sensitivities_array.shape != expected or active_array.shape != expected:
        raise ValueError(
            "sensitivity matrices must have shape (outputs, inputs)"
        )
    if scales_array.shape != (len(inputs),):
        raise ValueError("scales length must equal input_paths length")
    return SensitivityMap(
        input_paths=inputs,
        output_projection_ids=outputs,
        scales=scales_array,
        sensitivities=sensitivities_array,
        threshold=_nonnegative(_float(threshold, "threshold", 0), "threshold"),
        active=active_array,
    )


@jaxtyped(typechecker=beartype)
def make_information_spectrum(
    input_paths: tuple[str, ...],
    singular_values: Any,
    right_singular_vectors: Any,
    effective_rank: Any,
    condition_estimate: Any,
    threshold: Any,
) -> InformationSpectrum:
    """Create a validated local information spectrum."""
    paths = _text_tuple(input_paths, "input_paths")
    singular_array = _nonnegative(
        _float(singular_values, "singular_values", 1), "singular_values"
    )
    vectors_array = _float(right_singular_vectors, "right_singular_vectors", 2)
    if vectors_array.shape != (singular_array.shape[0], len(paths)):
        raise ValueError(
            "right_singular_vectors must have shape (singular values, inputs)"
        )
    return InformationSpectrum(
        input_paths=paths,
        singular_values=singular_array,
        right_singular_vectors=vectors_array,
        effective_rank=_int(effective_rank, "effective_rank", 0),
        condition_estimate=_float(condition_estimate, "condition_estimate", 0),
        threshold=_nonnegative(_float(threshold, "threshold", 0), "threshold"),
    )


@jaxtyped(typechecker=beartype)
def make_execution_manifest(
    execution_id: str,
    model_ref: str,
    schema_version: str,
    package_version: str,
    source_checksum: str,
    environment_checksum: str,
    backend: str,
    precision_policy: str,
    deterministic: bool,
    started_at_utc: str,
) -> ExecutionManifest:
    """Create a validated execution manifest."""
    return ExecutionManifest(
        execution_id=_require_text(execution_id, "execution_id"),
        model_ref=_require_text(model_ref, "model_ref"),
        schema_version=_require_text(schema_version, "schema_version"),
        package_version=_require_text(package_version, "package_version"),
        source_checksum=_require_text(source_checksum, "source_checksum"),
        environment_checksum=_require_text(
            environment_checksum, "environment_checksum"
        ),
        backend=_require_text(backend, "backend"),
        precision_policy=_require_text(precision_policy, "precision_policy"),
        deterministic=deterministic,
        started_at_utc=_require_text(started_at_utc, "started_at_utc"),
    )


@jaxtyped(typechecker=beartype)
def make_policy_report(
    policy_id: str,
    level_ids: tuple[str, ...],
    required_claim_ids: tuple[str, ...],
    claim_passed: Any,
    claim_checked: Any,
    claim_in_domain: Any,
    achieved: Any,
) -> PolicyReport:
    """Create a validated policy truth table."""
    levels = _text_tuple(level_ids, "level_ids")
    claims = _text_tuple(required_claim_ids, "required_claim_ids")
    passed_array = _bool(claim_passed, "claim_passed", 1)
    checked_array = _bool(claim_checked, "claim_checked", 1)
    domain_array = _bool(claim_in_domain, "claim_in_domain", 1)
    achieved_array = _bool(achieved, "achieved", 1)
    if not (
        passed_array.shape
        == checked_array.shape
        == domain_array.shape
        == (len(claims),)
    ):
        raise ValueError("policy claim arrays must match required_claim_ids")
    if achieved_array.shape != (len(levels),):
        raise ValueError("achieved must match level_ids")
    return PolicyReport(
        policy_id=_require_text(policy_id, "policy_id"),
        level_ids=levels,
        required_claim_ids=claims,
        claim_passed=passed_array,
        claim_checked=checked_array,
        claim_in_domain=domain_array,
        achieved=achieved_array,
    )


@jaxtyped(typechecker=beartype)
def make_certification_context(
    manifest: ExecutionManifest,
    model: ForwardModelSpec,
    artifacts: tuple[ArtifactRef, ...] = (),
    transformations: tuple[TransformationRecord, ...] = (),
    evidence: tuple[EvidenceRef, ...] = (),
    policy_id: str = "org.diffpes.policy.research.v1",
    check_ids: tuple[str, ...] = (),
    input_checksums: tuple[str, ...] = (),
) -> CertificationContext:
    """Create a prepared certification context."""
    expected_ref = f"{model.model_id}@{model.model_version}"
    if manifest.model_ref not in (model.model_id, expected_ref):
        raise ValueError(
            "manifest model_ref does not match model specification"
        )
    return CertificationContext(
        manifest=manifest,
        model=model,
        artifacts=tuple(artifacts),
        transformations=tuple(transformations),
        evidence=tuple(evidence),
        policy_id=_require_text(policy_id, "policy_id"),
        check_ids=_text_tuple(check_ids, "check_ids"),
        input_checksums=_text_tuple(input_checksums, "input_checksums"),
    )


def _unique_module_ids(values: tuple[Any, ...], attribute: str) -> bool:
    """Return whether a module tuple contains unique named identities."""
    identities = tuple(getattr(value, attribute) for value in values)
    return len(identities) == len(set(identities))


@jaxtyped(typechecker=beartype)
def make_forward_certificate(  # noqa: PLR0913
    manifest: ExecutionManifest,
    model: ForwardModelSpec,
    artifacts: tuple[ArtifactRef, ...],
    transformations: tuple[TransformationRecord, ...],
    evidence: tuple[EvidenceRef, ...],
    claims: tuple[CertificationClaim, ...],
    domains: tuple[DomainResult, ...],
    derivatives: DerivativeEvidence,
    dependencies: DependencyMap,
    sensitivities: SensitivityMap,
    information: InformationSpectrum,
    policy_report: PolicyReport,
    policy_id: str,
    certificate_checksum: str,
    extensions_json: str = "{}",
) -> ForwardCertificate:
    """Create and cross-validate a complete forward certificate."""
    expected_ref = f"{model.model_id}@{model.model_version}"
    if manifest.model_ref not in (model.model_id, expected_ref):
        raise ValueError(
            "manifest model_ref does not match model specification"
        )
    if policy_report.policy_id != policy_id:
        raise ValueError("policy_report policy_id does not match certificate")
    if dependencies.model_id != model.model_id:
        raise ValueError(
            "dependency model_id does not match model specification"
        )
    identity_groups = (
        (artifacts, "artifact_id"),
        (transformations, "transformation_id"),
        (evidence, "evidence_id"),
        (claims, "claim_id"),
        (domains, "predicate_id"),
    )
    for values, attribute in identity_groups:
        if not _unique_module_ids(values, attribute):
            raise ValueError(f"certificate contains duplicate {attribute}")
    return ForwardCertificate(
        manifest=manifest,
        model=model,
        artifacts=tuple(artifacts),
        transformations=tuple(transformations),
        evidence=tuple(evidence),
        claims=tuple(claims),
        domains=tuple(domains),
        derivatives=derivatives,
        dependencies=dependencies,
        sensitivities=sensitivities,
        information=information,
        policy_report=policy_report,
        policy_id=_require_text(policy_id, "policy_id"),
        certificate_checksum=_require_text(
            certificate_checksum, "certificate_checksum"
        ),
        extensions_json=_json_object(extensions_json, "extensions_json"),
    )


@jaxtyped(typechecker=beartype)
def make_certified_result(
    value: Any, certificate: ForwardCertificate
) -> CertifiedResult:
    """Pair any JAX-compatible result value with a forward certificate."""
    return CertifiedResult(value=value, certificate=certificate)


@jaxtyped(typechecker=beartype)
def make_evidence_report(
    evidence_id: str,
    resolved: Any,
    compatible: Any,
    passed: Any,
    residual_norm: Any,
) -> EvidenceReport:
    """Create an offline evidence-verification report."""
    return EvidenceReport(
        evidence_id=_require_text(evidence_id, "evidence_id"),
        resolved=_bool(resolved, "resolved", 0),
        compatible=_bool(compatible, "compatible", 0),
        passed=_bool(passed, "passed", 0),
        residual_norm=_float(residual_norm, "residual_norm", 0),
    )


@jaxtyped(typechecker=beartype)
def make_verification_report(
    certificate_checksum: str,
    policy_id: str,
    structure_valid: Any,
    evidence_valid: Any,
    policy_report: PolicyReport,
) -> VerificationReport:
    """Create an offline certificate-verification report."""
    if policy_report.policy_id != policy_id:
        raise ValueError("policy_report policy_id does not match report")
    return VerificationReport(
        certificate_checksum=_require_text(
            certificate_checksum, "certificate_checksum"
        ),
        policy_id=_require_text(policy_id, "policy_id"),
        structure_valid=_bool(structure_valid, "structure_valid", 0),
        evidence_valid=_bool(evidence_valid, "evidence_valid", 0),
        policy_report=policy_report,
    )


@jaxtyped(typechecker=beartype)
def make_reproduction_report(
    execution_id: str,
    result_checksum: str,
    reproduced: Any,
    max_abs_error: Any,
    max_rel_error: Any,
    tolerance: Any,
) -> ReproductionReport:
    """Create a report comparing a result with its re-execution."""
    return ReproductionReport(
        execution_id=_require_text(execution_id, "execution_id"),
        result_checksum=_require_text(result_checksum, "result_checksum"),
        reproduced=_bool(reproduced, "reproduced", 0),
        max_abs_error=_float(max_abs_error, "max_abs_error", 0),
        max_rel_error=_float(max_rel_error, "max_rel_error", 0),
        tolerance=_nonnegative(_float(tolerance, "tolerance", 0), "tolerance"),
    )


__all__: list[str] = [
    "ArtifactRef",
    "CertificationClaim",
    "CertificationContext",
    "CertifiedResult",
    "ConventionRef",
    "DependencyMap",
    "DerivativeEvidence",
    "DomainPredicate",
    "DomainResult",
    "EvidenceRef",
    "EvidenceReport",
    "ExecutionManifest",
    "ForwardCertificate",
    "ForwardModelSpec",
    "InformationSpectrum",
    "PolicyReport",
    "ReproductionReport",
    "SensitivityMap",
    "TransformationRecord",
    "VerificationReport",
    "make_artifact_ref",
    "make_certification_claim",
    "make_certification_context",
    "make_certified_result",
    "make_convention_ref",
    "make_dependency_map",
    "make_derivative_evidence",
    "make_domain_predicate",
    "make_domain_result",
    "make_evidence_ref",
    "make_evidence_report",
    "make_execution_manifest",
    "make_forward_certificate",
    "make_forward_model_spec",
    "make_information_spectrum",
    "make_policy_report",
    "make_reproduction_report",
    "make_sensitivity_map",
    "make_transformation_record",
    "make_verification_report",
]
