"""Tests for stable human-readable certificate inspection."""

import pytest

from diffpes.certify import (
    diff_certificates,
    explain_claim,
    summarize_certificate,
)
from tests.test_diffpes.test_inout.test_certificate import sample_certificate


def test_summary_answers_identity_provenance_claim_and_derivative_questions():
    summary = summarize_certificate(sample_certificate())

    assert "Model: org.diffpes.model.arpes.test@1.0.0" in summary
    assert "Observable: org.diffpes.observable.arpes.intensity" in summary
    assert "org.diffpes.convention.energy.fermi_referenced_ev@1.0.0" in summary
    assert "bands [initial_state; application/x-vasp-eigenval]" in summary
    assert "destroys: overall_phase" in summary
    assert "Claims: passed=1" in summary
    assert (
        "Achieved levels: identified, validated, differentiable, verified"
        in summary
    )
    assert "Unachieved levels: benchmarked, reproducible" in summary
    assert "claim.output.finite: passed (margin=0.5)" in summary
    assert "jax.linearize+jvp+vjp+central_fd" in summary
    assert "Structural dependencies: 2/2 input-output pairs" in summary
    assert "Locally active sensitivities: 2/2 projections" in summary
    assert "effective_rank=2" in summary
    assert "/private/data" not in summary


def test_explain_claim_reports_bounded_numerical_evidence():
    explanation = explain_claim(sample_certificate(), "claim.output.finite")

    assert "Status: passed" in explanation
    assert "Margin: 0.5" in explanation
    assert "reference-spectrum" in explanation
    assert "method: org.diffpes.method.reference" in explanation
    assert "independent: True" in explanation
    assert "tolerance: shape=(2,), values=[1.e-08, 1.e-08]" in explanation


def test_explain_claim_rejects_missing_id():
    with pytest.raises(KeyError, match="not present"):
        explain_claim(sample_certificate(), "claim.missing")


def test_diff_categorizes_environment_audit_and_scientific_changes():
    original = sample_certificate()
    audit_changed = sample_certificate(
        execution_id="run-002",
        started_at_utc="2026-07-21T13:00:00Z",
    )
    scientific_changed = sample_certificate(model_version="1.1.0")
    environment_manifest = sample_certificate(
        environment_checksum="crc32:canonical-1:environment:00000000"
    )

    audit = diff_certificates(original, audit_changed)
    scientific = diff_certificates(original, scientific_changed)
    environment = diff_certificates(original, environment_manifest)

    assert audit.scientific == ()
    assert audit.audit == ("execution_id", "started_at_utc")
    assert scientific.scientific == ("model",)
    assert scientific.environment == ()
    assert environment.environment == ("environment_checksum",)
    assert "audit: execution_id, started_at_utc" in audit.summary


def test_diff_identical_certificate_has_clear_summary():
    certificate = sample_certificate()
    difference = diff_certificates(certificate, certificate)

    assert difference.identical
    assert difference.summary == "Certificates are identical."
