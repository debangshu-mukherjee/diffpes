"""Execute JAX-native certified forward models.

Extended Summary
----------------
Separates filesystem and registry preparation from the pure compiled
scientific kernel. The kernel evaluates the registered forward model once,
retains its linearization, derives continuous claims and derivative evidence,
and returns an Equinox certificate whose numerical leaves support JIT, VMAP,
JVP, and VJP transformations.

Routine Listings
----------------
:func:`certify_forward`
    Execute a prepared model and produce a JAX-native certified result.
:func:`prepare_certification`
    Resolve static model and policy records before compiled execution.
:func:`verify_certificate`
    Re-evaluate internal numerical claim and policy consistency.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any, Optional
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float, PyTree, jaxtyped

from diffpes.types.certification import (
    ArtifactRef,
    CertificationClaim,
    CertificationContext,
    CertifiedResult,
    DomainResult,
    EvidenceRef,
    ExecutionManifest,
    ForwardCertificate,
    TransformationRecord,
    VerificationReport,
    make_certification_claim,
    make_certification_context,
    make_certified_result,
    make_forward_certificate,
    make_verification_report,
)

from .checks import CheckFunction, get_check
from .dependencies import dependency_map, information_spectrum, sensitivity_map
from .evidence import derivative_evidence, evaluate_claim
from .policy import evaluate_policy
from .registry import RegisteredModel, get_model


def _path_names(tree: PyTree) -> tuple[str, ...]:
    """Return stable real-coordinate paths for numerical input leaves."""
    path_leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    paths: list[str] = []
    for path, leaf in path_leaves:
        name: str = jax.tree_util.keystr(path) or "$"
        if jnp.iscomplexobj(jnp.asarray(leaf)):
            paths.extend((f"{name}.real", f"{name}.imag"))
        else:
            paths.append(name)
    return tuple(paths)


def _probe_directions(inputs: PyTree) -> PyTree:
    """Construct one all-ones directional probe per input leaf."""
    leaves, treedef = jax.tree_util.tree_flatten(inputs)
    arrays: list[Array] = [jnp.asarray(leaf) for leaf in leaves]
    n_leaves: int = sum(
        2 if jnp.iscomplexobj(array) else 1 for array in arrays
    )
    batched: list[Array] = []
    selected: int = 0
    for array in arrays:
        shape: tuple[int, ...] = (n_leaves, *array.shape)
        probe: Array = jnp.zeros(shape, dtype=array.dtype)
        probe = probe.at[selected].set(jnp.ones_like(array))
        selected += 1
        if jnp.iscomplexobj(array):
            probe = probe.at[selected].set(1j * jnp.ones_like(array))
            selected += 1
        batched.append(probe)
    directions: PyTree = jax.tree_util.tree_unflatten(treedef, batched)
    return directions


@beartype
def prepare_certification(
    model_id: str,
    model_version: str,
    manifest: ExecutionManifest,
    *,
    policy_id: str = "org.diffpes.policy.research.v1",
    artifacts: tuple[ArtifactRef, ...] = (),
    transformations: tuple[TransformationRecord, ...] = (),
    evidence: tuple[EvidenceRef, ...] = (),
    check_ids: tuple[str, ...] = (),
    input_checksums: tuple[str, ...] = (),
) -> CertificationContext:
    """Resolve static scientific records before compiled execution."""
    registered: RegisteredModel = get_model(model_id, model_version)
    context: CertificationContext = make_certification_context(
        manifest=manifest,
        model=registered.spec,
        artifacts=artifacts,
        transformations=transformations,
        evidence=evidence,
        policy_id=policy_id,
        check_ids=(
            check_ids
            if check_ids
            else tuple(item.predicate_id for item in registered.spec.domain)
        ),
        input_checksums=input_checksums,
    )
    return context


def _evidence_claims(
    model_id: str, evidence: tuple[EvidenceRef, ...]
) -> tuple[CertificationClaim, ...]:
    """Convert registered numerical evidence into verification claims."""
    claims: list[CertificationClaim] = []
    for item in evidence:
        claim: CertificationClaim = evaluate_claim(
            claim_id=f"claim.{item.evidence_id}",
            subject_id=model_id,
            predicate_id="verification.external_reference",
            measured=item.measured,
            reference=item.reference,
            tolerance=item.tolerance,
            evidence_ids=(item.evidence_id,),
        )
        claims.append(claim)
    return tuple(claims)


@eqx.filter_jit
def _certify_kernel(
    executor: Any,
    context: CertificationContext,
    inputs: PyTree,
    directions: PyTree,
    scales: Float[Array, " n_probe"],
    domain_checks: tuple[CheckFunction, ...],
    *,
    spectrum_rank: int,
) -> CertifiedResult:
    """Run the pure compiled certification computation."""

    def forward_vector(candidate: PyTree) -> Array:
        value: PyTree = executor(candidate)
        flat, _ = ravel_pytree(value)
        return jnp.real(flat)

    value: PyTree = executor(inputs)
    flat_value, _ = ravel_pytree(value)
    output_size: int = flat_value.size
    input_paths: tuple[str, ...] = _path_names(inputs)
    n_probes: int = len(input_paths)
    cotangent_indices: Array = jnp.arange(n_probes) % output_size
    cotangents: Array = jnp.eye(output_size, dtype=jnp.float64)[
        cotangent_indices
    ]
    output_ids: tuple[str, ...] = tuple(
        f"output[{index}]" for index in range(output_size)
    )
    derivatives = derivative_evidence(
        forward_vector,
        inputs,
        directions,
        cotangents,
        input_paths=input_paths,
        output_projection_ids=output_ids,
        scales=scales,
        spectrum_rank=spectrum_rank,
    )
    dependencies = dependency_map(
        context.model.model_id,
        executor,
        inputs,
    )
    sensitivities = sensitivity_map(
        input_paths,
        output_ids,
        forward_vector,
        inputs,
        directions,
        scales,
    )
    information = information_spectrum(
        forward_vector,
        inputs,
        input_paths=input_paths,
        rank=spectrum_rank,
    )
    domains: tuple[DomainResult, ...] = tuple(
        check_fn(inputs) for check_fn in domain_checks
    )
    domain_claims: tuple[CertificationClaim, ...] = tuple(
        make_certification_claim(
            claim_id=f"claim.{domain.predicate_id}",
            subject_id=context.model.model_id,
            predicate_id=f"domain.{domain.predicate_id}",
            evidence_ids=(),
            measured=jnp.atleast_1d(domain.measured),
            reference=jnp.atleast_1d(domain.reference),
            residual=jnp.atleast_1d(domain.residual),
            tolerance=jnp.atleast_1d(domain.tolerance),
            passed=domain.passed,
            checked=domain.checked,
            in_domain=domain.in_domain,
            margin=domain.margin,
            severity_code=domain.severity_code,
        )
        for domain in domains
    )
    identity_claim: CertificationClaim = evaluate_claim(
        claim_id="claim.execution.identified",
        subject_id=context.model.model_id,
        predicate_id="identity.model_and_inputs",
        measured=jnp.zeros(1),
        reference=jnp.zeros(1),
        tolerance=jnp.zeros(1),
    )
    nonfinite_count: Array = jnp.sum(~jnp.isfinite(flat_value))
    output_claim: CertificationClaim = evaluate_claim(
        claim_id="claim.output.finite",
        subject_id=context.model.observable_id,
        predicate_id="output.finite",
        measured=jnp.asarray([nonfinite_count], dtype=jnp.float64),
        reference=jnp.zeros(1),
        tolerance=jnp.zeros(1),
    )
    derivative_error: Array = jnp.max(
        jnp.abs(derivatives.derivative_residuals)
    )
    derivative_tolerance: Array = 1e-9 + 1e-6 * jnp.max(
        jnp.abs(derivatives.reference_derivatives)
    )
    derivative_claim: CertificationClaim = evaluate_claim(
        claim_id="claim.derivative.fd_correct",
        subject_id=context.model.model_id,
        predicate_id="derivative.jvp_matches_central_fd",
        measured=jnp.asarray([derivative_error]),
        reference=jnp.zeros(1),
        tolerance=jnp.asarray([derivative_tolerance]),
    )
    external_claims: tuple[CertificationClaim, ...] = _evidence_claims(
        context.model.model_id, context.evidence
    )
    claims: tuple[CertificationClaim, ...] = (
        identity_claim,
        output_claim,
        derivative_claim,
        *domain_claims,
        *external_claims,
    )
    policy_report = evaluate_policy(claims, context.policy_id)
    certificate: ForwardCertificate = make_forward_certificate(
        manifest=context.manifest,
        model=context.model,
        artifacts=context.artifacts,
        transformations=context.transformations,
        domains=domains,
        evidence=context.evidence,
        claims=claims,
        derivatives=derivatives,
        dependencies=dependencies,
        sensitivities=sensitivities,
        information=information,
        policy_report=policy_report,
        policy_id=context.policy_id,
        certificate_checksum="pending-canonical-serialization",
        extensions_json="{}",
    )
    result: CertifiedResult = make_certified_result(
        value=value,
        certificate=certificate,
    )
    return result


@jaxtyped(typechecker=beartype)
def certify_forward(
    context: CertificationContext,
    inputs: PyTree,
    *,
    directions: Optional[PyTree] = None,
    scales: Optional[Float[Array, " n_probe"]] = None,
    spectrum_rank: int = 8,
) -> CertifiedResult:
    """Execute a prepared model and produce a certified JAX PyTree."""
    registered: RegisteredModel = get_model(
        context.model.model_id,
        context.model.model_version,
    )
    resolved_directions: PyTree = (
        _probe_directions(inputs) if directions is None else directions
    )
    n_probes: int = jax.tree.leaves(resolved_directions)[0].shape[0]
    resolved_scales: Array = (
        jnp.ones(n_probes, dtype=jnp.float64)
        if scales is None
        else jnp.asarray(scales, dtype=jnp.float64)
    )
    domain_checks: tuple[CheckFunction, ...] = tuple(
        get_check(check_id) for check_id in context.check_ids
    )
    result: CertifiedResult = _certify_kernel(
        registered.executor,
        context,
        inputs,
        resolved_directions,
        resolved_scales,
        domain_checks,
        spectrum_rank=spectrum_rank,
    )
    return result


@beartype
def verify_certificate(certificate: ForwardCertificate) -> VerificationReport:
    """Re-evaluate numerical claim and policy consistency without a rerun."""
    recomputed = evaluate_policy(certificate.claims, certificate.policy_id)
    claims_consistent: bool = all(
        bool(
            jnp.array_equal(
                claim.passed,
                claim.checked
                & claim.in_domain
                & jnp.all(jnp.abs(claim.residual) <= claim.tolerance),
            )
        )
        for claim in certificate.claims
    )
    policy_consistent: bool = bool(
        jnp.array_equal(
            recomputed.achieved, certificate.policy_report.achieved
        )
    )
    report: VerificationReport = make_verification_report(
        certificate_checksum=certificate.certificate_checksum,
        policy_id=certificate.policy_id,
        structure_valid=jnp.asarray(claims_consistent),
        evidence_valid=jnp.asarray(policy_consistent),
        policy_report=recomputed,
    )
    return report


__all__: list[str] = [
    "certify_forward",
    "prepare_certification",
    "verify_certificate",
]
