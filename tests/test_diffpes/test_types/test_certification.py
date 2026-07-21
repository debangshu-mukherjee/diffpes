"""Test the JAX-native certification carrier contract."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from diffpes.types import (
    CertifiedResult,
    ForwardCertificate,
    make_artifact_ref,
    make_certification_claim,
    make_certification_context,
    make_certified_result,
    make_convention_ref,
    make_dependency_map,
    make_derivative_evidence,
    make_domain_predicate,
    make_domain_result,
    make_evidence_ref,
    make_execution_manifest,
    make_forward_certificate,
    make_forward_model_spec,
    make_information_spectrum,
    make_policy_report,
    make_sensitivity_map,
    make_transformation_record,
)
from tests._assertions import assert_rejects, assert_trees_close


def _certificate() -> ForwardCertificate:
    """Build one complete, small certificate for carrier tests."""
    convention = make_convention_ref(
        "org.diffpes.convention.energy-fermi", "1", '{"unit":"eV"}'
    )
    predicate = make_domain_predicate(
        "org.diffpes.domain.energy", "closed_interval", "eV"
    )
    model = make_forward_model_spec(
        model_id="org.diffpes.model.toy",
        model_version="1.0.0",
        observable_id="org.diffpes.observable.scalar",
        implementation_ref="tests.toy",
        assumptions=("linear-response",),
        conventions=(convention,),
        domain=(predicate,),
        differentiable_paths=("parameters.x", "parameters.y"),
        nondifferentiable_paths=("configuration.mode",),
    )
    manifest = make_execution_manifest(
        execution_id="run-1",
        model_ref="org.diffpes.model.toy@1.0.0",
        schema_version="1.0",
        package_version="test",
        source_checksum="source-1",
        environment_checksum="environment-1",
        backend="cpu",
        precision_policy="float64",
        deterministic=True,
        started_at_utc="2026-07-21T00:00:00Z",
    )
    artifact = make_artifact_ref(
        artifact_id="input-1",
        media_type="application/x-diffpes-array",
        byte_checksum=None,
        content_checksum="content-1",
        semantic_checksum="semantic-1",
        locator=None,
        role="normalized-input",
    )
    transformation = make_transformation_record(
        transformation_id="org.diffpes.transform.toy",
        transformation_version="1",
        parent_ids=("input-1",),
        output_ids=("output-1",),
        preserves=("units",),
        introduces=("broadening",),
        destroys=("sharp-lines",),
        invalidates_claims=("absolute-resolution",),
        parameters_checksum="parameters-1",
    )
    evidence = make_evidence_ref(
        evidence_id="evidence-1",
        method_id="org.diffpes.method.reference",
        artifact_refs=("input-1",),
        source_type="analytic",
        independent=True,
        measured=jnp.array([1.0, 2.0]),
        reference=jnp.array([1.0, 2.0]),
        residual=jnp.zeros(2),
        tolerance=jnp.full(2, 1.0e-12),
    )
    claim = make_certification_claim(
        claim_id="claim-1",
        subject_id="output-1",
        predicate_id="reference-agreement",
        evidence_ids=("evidence-1",),
        measured=jnp.array([1.0, 2.0]),
        reference=jnp.array([1.0, 2.0]),
        residual=jnp.zeros(2),
        tolerance=jnp.full(2, 1.0e-12),
        passed=True,
        margin=1.0e-12,
    )
    domain = make_domain_result(
        predicate_id=predicate.predicate_id,
        measured=0.5,
        reference=1.0,
        residual=-0.5,
        tolerance=0.0,
        margin=0.5,
        passed=True,
    )
    derivatives = make_derivative_evidence(
        input_paths=model.differentiable_paths,
        output_projection_ids=("scalar",),
        method="jvp-vjp-fd",
        scales=jnp.ones(2),
        jvp_probes=jnp.array([[1.0], [2.0]]),
        vjp_probes=jnp.eye(2),
        reference_derivatives=jnp.eye(2),
        derivative_residuals=jnp.zeros((2, 2)),
        singular_values=jnp.array([2.0, 1.0]),
        effective_rank=2,
        condition_estimate=2.0,
        finite=True,
        fd_correct=True,
    )
    dependencies = make_dependency_map(
        model_id=model.model_id,
        input_paths=model.differentiable_paths,
        output_paths=("value",),
        structural=jnp.array([[True, True]]),
        traced=jnp.array([[True, True]]),
    )
    sensitivities = make_sensitivity_map(
        input_paths=model.differentiable_paths,
        output_projection_ids=("scalar",),
        scales=jnp.ones(2),
        sensitivities=jnp.array([[2.0, 1.0]]),
        threshold=1.0e-12,
        active=jnp.array([[True, True]]),
    )
    information = make_information_spectrum(
        input_paths=model.differentiable_paths,
        singular_values=jnp.array([2.0, 1.0]),
        right_singular_vectors=jnp.eye(2),
        effective_rank=2,
        condition_estimate=2.0,
        threshold=1.0e-12,
    )
    policy = make_policy_report(
        policy_id="org.diffpes.policy.research.v1",
        level_ids=("identified", "validated"),
        required_claim_ids=("claim-1",),
        claim_passed=jnp.array([True]),
        claim_checked=jnp.array([True]),
        claim_in_domain=jnp.array([True]),
        achieved=jnp.array([True, True]),
    )
    return make_forward_certificate(
        manifest=manifest,
        model=model,
        artifacts=(artifact,),
        transformations=(transformation,),
        evidence=(evidence,),
        claims=(claim,),
        domains=(domain,),
        derivatives=derivatives,
        dependencies=dependencies,
        sensitivities=sensitivities,
        information=information,
        policy_report=policy,
        policy_id=policy.policy_id,
        certificate_checksum="certificate-1",
        extensions_json='{"future_field":1}',
    )


def test_static_vocabulary_is_absent_from_numerical_leaves() -> None:
    """Keep identifiers static while retaining all numerical evidence."""
    certificate = _certificate()
    leaves = jax.tree.leaves(certificate)
    assert leaves
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)
    assert not any(isinstance(leaf, str) for leaf in leaves)
    assert certificate.model.model_id == "org.diffpes.model.toy"
    assert certificate.extensions_json == '{"future_field":1}'


def test_complete_result_round_trips_through_filter_jit() -> None:
    """Preserve the complete nested PyTree through compiled execution."""
    result = make_certified_result(jnp.array([2.0, 3.0]), _certificate())
    compiled = eqx.filter_jit(lambda item: item)(result)
    assert isinstance(compiled, CertifiedResult)
    assert compiled.certificate.model == result.certificate.model
    assert_trees_close(
        jax.tree.leaves(compiled),
        jax.tree.leaves(result),
        rtol=0.0,
        atol=0.0,
    )


def test_continuous_claim_evidence_is_differentiable() -> None:
    """Differentiate a smooth residual and margin through its factory."""

    def objective(value: jax.Array) -> jax.Array:
        claim = make_certification_claim(
            claim_id="claim",
            subject_id="subject",
            predicate_id="agreement",
            evidence_ids=(),
            measured=value[None],
            reference=jnp.ones(1),
            residual=(value - 1.0)[None],
            tolerance=jnp.full(1, 0.1),
            passed=jnp.abs(value - 1.0) <= 0.1,
            margin=0.1 - jnp.abs(value - 1.0),
        )
        return jnp.sum(claim.residual**2) + claim.margin

    value = jnp.asarray(1.25)
    expected = 2.0 * (value - 1.0) - 1.0
    assert_trees_close(jax.grad(objective)(value), expected)
    assert_trees_close(eqx.filter_jit(jax.grad(objective))(value), expected)


def test_domain_results_vmap_with_traced_status_and_margin() -> None:
    """Batch domain checks without concretizing status arrays."""

    def evaluate(value: jax.Array):
        margin = 1.0 - jnp.abs(value)
        return make_domain_result(
            "bounded",
            measured=value,
            reference=0.0,
            residual=value,
            tolerance=0.0,
            margin=margin,
            passed=margin >= 0.0,
            in_domain=margin >= 0.0,
        )

    result = eqx.filter_jit(jax.vmap(evaluate))(jnp.array([-0.5, 1.5]))
    assert_trees_close(result.margin, jnp.array([0.5, -0.5]))
    assert jnp.array_equal(result.passed, jnp.array([True, False]))


def test_certified_envelope_preserves_primal_jvp_and_vjp() -> None:
    """Show that attaching a certificate does not alter model derivatives."""
    certificate = _certificate()

    def ordinary(value: jax.Array) -> jax.Array:
        return jnp.sin(value) + value**2

    def certified(value: jax.Array) -> jax.Array:
        return make_certified_result(ordinary(value), certificate).value

    point = jnp.asarray(0.4)
    tangent = jnp.asarray(1.7)
    ordinary_primal, ordinary_jvp = jax.jvp(ordinary, (point,), (tangent,))
    certified_primal, certified_jvp = jax.jvp(certified, (point,), (tangent,))
    assert_trees_close(certified_primal, ordinary_primal, rtol=0.0, atol=0.0)
    assert_trees_close(certified_jvp, ordinary_jvp, rtol=0.0, atol=0.0)
    _, ordinary_pullback = jax.vjp(ordinary, point)
    _, certified_pullback = jax.vjp(certified, point)
    cotangent = jnp.asarray(2.0)
    assert_trees_close(
        certified_pullback(cotangent),
        ordinary_pullback(cotangent),
        rtol=0.0,
        atol=0.0,
    )


def test_context_cross_validates_model_identity() -> None:
    """Reject prepared contexts that combine different model identities."""
    certificate = _certificate()
    context = make_certification_context(
        certificate.manifest,
        certificate.model,
        certificate.artifacts,
        certificate.transformations,
        certificate.evidence,
        certificate.policy_id,
        ("domain", "output"),
        ("semantic-1",),
    )
    assert context.model is certificate.model
    bad_manifest = make_execution_manifest(
        execution_id=certificate.manifest.execution_id,
        model_ref="org.diffpes.model.other",
        schema_version=certificate.manifest.schema_version,
        package_version=certificate.manifest.package_version,
        source_checksum=certificate.manifest.source_checksum,
        environment_checksum=certificate.manifest.environment_checksum,
        backend=certificate.manifest.backend,
        precision_policy=certificate.manifest.precision_policy,
        deterministic=certificate.manifest.deterministic,
        started_at_utc=certificate.manifest.started_at_utc,
    )
    with pytest.raises(ValueError, match="model_ref does not match"):
        make_certification_context(bad_manifest, certificate.model)


def test_factories_reject_malformed_static_structure() -> None:
    """Reject empty identities, invalid JSON, and overlapping path classes."""
    with pytest.raises(ValueError, match="artifact_id must be non-empty"):
        make_artifact_ref("", "text/plain", None, "a", "b", None, "input")
    with pytest.raises(ValueError, match="valid JSON"):
        make_convention_ref("convention", "1", "{")
    with pytest.raises(ValueError, match="must be disjoint"):
        make_forward_model_spec(
            "model",
            "1",
            "observable",
            "implementation",
            (),
            (),
            (),
            ("x",),
            ("x",),
        )


def test_factories_reject_bad_numerical_shapes_and_tolerances() -> None:
    """Reject malformed evidence eagerly and through compiled execution."""
    with pytest.raises(ValueError, match="equal shapes"):
        make_evidence_ref(
            "evidence",
            "method",
            (),
            "analytic",
            True,
            jnp.ones(2),
            jnp.ones(1),
            jnp.ones(2),
            jnp.ones(2),
        )
    assert_rejects(
        make_domain_result,
        "domain",
        0.0,
        0.0,
        0.0,
        -1.0,
        0.0,
        False,
        match="tolerance must be finite and nonnegative",
    )


def test_certificate_rejects_cross_record_inconsistency() -> None:
    """Reject policy, dependency, and duplicate identity mismatches."""
    certificate = _certificate()
    with pytest.raises(ValueError, match="policy_report policy_id"):
        make_forward_certificate(
            certificate.manifest,
            certificate.model,
            certificate.artifacts,
            certificate.transformations,
            certificate.evidence,
            certificate.claims,
            certificate.domains,
            certificate.derivatives,
            certificate.dependencies,
            certificate.sensitivities,
            certificate.information,
            certificate.policy_report,
            "org.diffpes.policy.other.v1",
            certificate.certificate_checksum,
        )
    with pytest.raises(ValueError, match="duplicate artifact_id"):
        make_forward_certificate(
            certificate.manifest,
            certificate.model,
            certificate.artifacts * 2,
            certificate.transformations,
            certificate.evidence,
            certificate.claims,
            certificate.domains,
            certificate.derivatives,
            certificate.dependencies,
            certificate.sensitivities,
            certificate.information,
            certificate.policy_report,
            certificate.policy_id,
            certificate.certificate_checksum,
        )
