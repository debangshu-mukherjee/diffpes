"""Tests for portable forward-certificate persistence."""

import json
import os
from pathlib import Path

import chex
import h5py
import jax.numpy as jnp
import pytest

from diffpes.inout.certificate import (
    CERTIFICATE_FORMAT,
    CertificateFormatError,
    _storage_checksum,
    attach_certificate_h5,
    load_certificate_h5,
    load_certificate_json,
    save_certificate_json,
)
from diffpes.types import (
    ForwardCertificate,
    make_artifact_ref,
    make_certification_claim,
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


def sample_certificate(
    *,
    execution_id: str = "run-001",
    started_at_utc: str = "2026-07-21T12:00:00Z",
    model_version: str = "1.0.0",
    environment_checksum: str = "crc32:canonical-1:environment:89abcdef",
    extensions_json: str = '{"project":"demo","unicode":"Å"}',
) -> ForwardCertificate:
    """Return one small, fully populated certificate test fixture."""
    convention = make_convention_ref(
        "org.diffpes.convention.energy.fermi_referenced_ev",
        "1.0.0",
    )
    predicate = make_domain_predicate(
        "org.diffpes.domain.photon_energy.positive",
        "photon_energy_ev > 0",
        units="eV",
    )
    model = make_forward_model_spec(
        model_id="org.diffpes.model.arpes.test",
        model_version=model_version,
        observable_id="org.diffpes.observable.arpes.intensity",
        implementation_ref="tests.forward",
        assumptions=("dipole_approximation",),
        conventions=(convention,),
        domain=(predicate,),
        differentiable_paths=("params.sigma", "params.temperature"),
    )
    manifest = make_execution_manifest(
        execution_id=execution_id,
        model_ref=f"{model.model_id}@{model.model_version}",
        schema_version="1.0.0",
        package_version="2026.06.02",
        source_checksum="crc32:canonical-1:source:01234567",
        environment_checksum=environment_checksum,
        backend="cpu",
        precision_policy="float64",
        deterministic=True,
        started_at_utc=started_at_utc,
    )
    artifact = make_artifact_ref(
        artifact_id="bands",
        media_type="application/x-vasp-eigenval",
        byte_checksum="crc32:canonical-1:artifact-bytes:10203040",
        content_checksum="crc32:canonical-1:normalized-content:20304050",
        semantic_checksum="crc32:canonical-1:semantic:30405060",
        locator="/private/data/EIGENVAL",
        role="initial_state",
    )
    transformation = make_transformation_record(
        transformation_id="org.diffpes.transform.amplitude.intensity",
        transformation_version="1.0.0",
        parent_ids=("amplitude",),
        output_ids=("intensity",),
        preserves=("energy_reference",),
        destroys=("overall_phase",),
        invalidates_claims=("claim.phase",),
        parameters_checksum="crc32:canonical-1:parameters:40506070",
    )
    evidence = make_evidence_ref(
        evidence_id="reference-spectrum",
        method_id="org.diffpes.method.reference",
        artifact_refs=("bands",),
        source_type="analytic_reference",
        independent=True,
        measured=jnp.array([1.0, 2.0]),
        reference=jnp.array([1.0, 2.0]),
        residual=jnp.zeros(2),
        tolerance=jnp.full(2, 1e-8),
    )
    claim = make_certification_claim(
        claim_id="claim.output.finite",
        subject_id=model.observable_id,
        predicate_id="output.finite",
        evidence_ids=(evidence.evidence_id,),
        measured=jnp.zeros(1),
        reference=jnp.zeros(1),
        residual=jnp.zeros(1),
        tolerance=jnp.zeros(1),
        passed=True,
        checked=True,
        in_domain=True,
        margin=0.5,
        severity_code=1,
    )
    domain = make_domain_result(
        predicate_id=predicate.predicate_id,
        measured=21.2,
        reference=20.0,
        residual=1.2,
        tolerance=20.0,
        margin=18.8,
        passed=True,
        checked=True,
        in_domain=True,
        severity_code=1,
    )
    derivatives = make_derivative_evidence(
        input_paths=("params.sigma", "params.temperature"),
        output_projection_ids=("total_intensity",),
        method="jax.linearize+jvp+vjp+central_fd",
        scales=jnp.array([0.05, 30.0]),
        jvp_probes=jnp.array([[1.0], [0.5]]),
        vjp_probes=jnp.array([[1.0, 0.5], [0.2, 0.1]]),
        reference_derivatives=jnp.array([[1.0], [0.5]]),
        derivative_residuals=jnp.zeros((2, 1)),
        singular_values=jnp.array([2.0, 0.25]),
        effective_rank=2,
        condition_estimate=8.0,
        finite=True,
        fd_correct=True,
    )
    dependencies = make_dependency_map(
        model_id=model.model_id,
        input_paths=("params.sigma", "params.temperature"),
        output_paths=("spectrum.intensity",),
        structural=jnp.array([[True, True]]),
        traced=jnp.array([[True, True]]),
    )
    sensitivities = make_sensitivity_map(
        input_paths=("params.sigma", "params.temperature"),
        output_projection_ids=("total_intensity",),
        scales=jnp.array([0.05, 30.0]),
        sensitivities=jnp.array([[1.0, 0.5]]),
        threshold=1e-12,
        active=jnp.array([[True, True]]),
    )
    information = make_information_spectrum(
        input_paths=("params.sigma", "params.temperature"),
        singular_values=jnp.array([2.0, 0.25]),
        right_singular_vectors=jnp.eye(2),
        effective_rank=2,
        condition_estimate=8.0,
        threshold=1e-10,
    )
    policy = make_policy_report(
        policy_id="org.diffpes.policy.research.v1",
        level_ids=(
            "identified",
            "validated",
            "differentiable",
            "verified",
            "benchmarked",
            "reproducible",
        ),
        required_claim_ids=(claim.claim_id,),
        claim_passed=jnp.array([True]),
        claim_checked=jnp.array([True]),
        claim_in_domain=jnp.array([True]),
        achieved=jnp.array([True, True, True, True, False, False]),
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
        certificate_checksum="crc32:canonical-1:certificate:50607080",
        extensions_json=extensions_json,
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_document(path: Path, document: dict) -> None:
    document["consistency_checksum"] = _storage_checksum(document)
    path.write_text(
        json.dumps(
            document,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_json_round_trip_is_byte_stable_and_lossless(tmp_path):
    certificate = sample_certificate()
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    save_certificate_json(certificate, first)
    loaded = load_certificate_json(first)
    save_certificate_json(loaded, second)

    assert first.read_bytes() == second.read_bytes()
    assert loaded.model.model_id == certificate.model.model_id
    assert json.loads(loaded.extensions_json) == json.loads(
        certificate.extensions_json
    )
    chex.assert_trees_all_equal(
        loaded.derivatives.jvp_probes,
        certificate.derivatives.jvp_probes,
    )
    document = _read_json(first)
    assert document["format"] == CERTIFICATE_FORMAT
    assert document["consistency_checksum"] == (
        "crc32:certificate-json-v1:f09dec09"
    )
    jvp_node = document["certificate"]["fields"]["derivatives"]["fields"][
        "jvp_probes"
    ]
    assert jvp_node["dtype"] == "<f8"
    assert jvp_node["shape"] == [2, 1]
    assert jvp_node["encoding"] == "base64"


def test_json_corruption_fails_consistency_check(tmp_path):
    path = tmp_path / "certificate.json"
    save_certificate_json(sample_certificate(), path)
    document = _read_json(path)
    document["certificate"]["fields"]["model"]["fields"]["model_id"] = (
        "org.diffpes.model.changed"
    )
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(CertificateFormatError, match="checksum mismatch"):
        load_certificate_json(path)


def test_unknown_schema_major_is_rejected_before_interpretation(tmp_path):
    path = tmp_path / "certificate.json"
    save_certificate_json(sample_certificate(), path)
    document = _read_json(path)
    document["schema_version"] = "2.0.0"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(CertificateFormatError, match="schema major 2"):
        load_certificate_json(path)


def test_unknown_minor_extensions_survive_round_trip(tmp_path):
    source = tmp_path / "future.json"
    restored = tmp_path / "restored.json"
    save_certificate_json(sample_certificate(), source)
    document = _read_json(source)
    document["schema_version"] = "1.1.0"
    manifest_fields = document["certificate"]["fields"]["manifest"]["fields"]
    manifest_fields["schema_version"] = "1.1.0"
    document["extensions"]["future_quantity"] = {"units": "eV", "value": 2}
    document["future_top_level"] = {"meaning": "retained"}
    document["certificate"]["fields"]["model"]["fields"][
        "future_model_field"
    ] = {"kind": "tuple", "items": ["alpha"]}
    _write_document(source, document)

    loaded = load_certificate_json(source)
    extensions = json.loads(loaded.extensions_json)
    save_certificate_json(loaded, restored)
    reloaded = load_certificate_json(restored)

    assert extensions["future_quantity"]["units"] == "eV"
    assert (
        extensions["org.diffpes.persistence.unknown_document_fields"][
            "future_top_level"
        ]["meaning"]
        == "retained"
    )
    assert (
        "certificate.model"
        in extensions["org.diffpes.persistence.unknown_module_fields"]
    )
    assert json.loads(reloaded.extensions_json) == extensions


def test_current_minor_rejects_unknown_structural_field(tmp_path):
    path = tmp_path / "certificate.json"
    save_certificate_json(sample_certificate(), path)
    document = _read_json(path)
    document["unexpected"] = True
    _write_document(path, document)

    with pytest.raises(CertificateFormatError, match="unknown current-schema"):
        load_certificate_json(path)


def test_json_write_failure_keeps_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "certificate.json"
    path.write_bytes(b"previous contents")

    def fail_replace(source, destination):
        del source, destination
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        save_certificate_json(sample_certificate(), path)

    assert path.read_bytes() == b"previous contents"
    assert list(tmp_path.iterdir()) == [path]


def test_hdf5_attach_preserves_results_and_round_trips(tmp_path):
    path = tmp_path / "spectrum.h5"
    certificate = sample_certificate()
    with h5py.File(path, "w") as file:
        file.create_dataset(
            "spectrum/intensity", data=jnp.arange(6).reshape(2, 3)
        )

    attach_certificate_h5(path, "spectrum", certificate)
    loaded = load_certificate_h5(path, "spectrum")

    assert loaded.model.model_id == certificate.model.model_id
    chex.assert_trees_all_equal(
        loaded.information.singular_values,
        certificate.information.singular_values,
    )
    with h5py.File(path, "r") as file:
        chex.assert_trees_all_equal(
            file["spectrum/intensity"][()],
            jnp.arange(6).reshape(2, 3),
        )
        group = file["_diffpes_certificates/spectrum"]
        assert group.attrs["model_id"] == certificate.model.model_id


def test_hdf5_replace_and_missing_name(tmp_path):
    path = tmp_path / "spectrum.h5"
    attach_certificate_h5(path, "spectrum", sample_certificate())
    newer = sample_certificate(execution_id="run-002")
    attach_certificate_h5(path, "spectrum", newer)

    assert (
        load_certificate_h5(path, "spectrum").manifest.execution_id
        == "run-002"
    )
    with pytest.raises(KeyError, match="missing"):
        load_certificate_h5(path, "missing")


def test_hdf5_corruption_and_index_disagreement_fail_closed(tmp_path):
    path = tmp_path / "spectrum.h5"
    attach_certificate_h5(path, "spectrum", sample_certificate())
    with h5py.File(path, "a") as file:
        group = file["_diffpes_certificates/spectrum"]
        group.attrs["model_version"] = "99.0.0"

    with pytest.raises(CertificateFormatError, match="index mismatch"):
        load_certificate_h5(path, "spectrum")


def test_hdf5_corrupt_authoritative_json_fails_closed(tmp_path):
    path = tmp_path / "spectrum.h5"
    attach_certificate_h5(path, "spectrum", sample_certificate())
    with h5py.File(path, "a") as file:
        dataset = file["_diffpes_certificates/spectrum/canonical_json"]
        damaged = dataset[()]
        damaged[0] ^= 1
        dataset[...] = damaged

    with pytest.raises(CertificateFormatError, match="valid UTF-8 JSON"):
        load_certificate_h5(path, "spectrum")


def test_hdf5_write_failure_keeps_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "spectrum.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("sentinel", data=jnp.array([1.0, 2.0]))
    previous = path.read_bytes()

    def fail_write(*args, **kwargs):
        del args, kwargs
        raise OSError("simulated HDF5 write failure")

    monkeypatch.setattr(
        "diffpes.inout.certificate._write_h5_record",
        fail_write,
    )
    with pytest.raises(OSError, match="simulated HDF5 write failure"):
        attach_certificate_h5(path, "spectrum", sample_certificate())

    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("name", ["", ".", "..", "nested/name", "bad\x00name"])
def test_hdf5_name_rejects_path_components(tmp_path, name):
    with pytest.raises(ValueError, match="one nonblank group component"):
        attach_certificate_h5(
            tmp_path / "result.h5", name, sample_certificate()
        )
