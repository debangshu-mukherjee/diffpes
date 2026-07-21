"""Tests for provenance DAG validation and information propagation."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from diffpes.certify.provenance import (
    ProvenanceError,
    build_provenance,
    effective_information,
    invalidated_claims,
    lineage,
    validate_provenance,
)
from diffpes.types.certification import make_transformation_record


def _record(
    transformation_id,
    parents,
    outputs,
    *,
    preserves=(),
    introduces=(),
    destroys=(),
    invalidates=(),
):
    return make_transformation_record(
        transformation_id=transformation_id,
        transformation_version="1.0.0",
        parent_ids=parents,
        output_ids=outputs,
        preserves=preserves,
        introduces=introduces,
        destroys=destroys,
        invalidates_claims=invalidates,
        parameters_checksum="none",
    )


def test_dag_is_topological_and_propagates_losses_and_claims() -> None:
    """Propagate inherited loss and claim invalidation to terminal results."""
    amplitude = _record(
        "org.diffpes.transform.test.amplitude",
        ("bands",),
        ("amplitude",),
        preserves=("energy-axis", "absolute-scale", "phase"),
        introduces=("matrix-element-amplitude",),
    )
    intensity = _record(
        "org.diffpes.transform.test.intensity",
        ("amplitude",),
        ("intensity",),
        preserves=("energy-axis", "absolute-scale"),
        introduces=("intensity-observable",),
        destroys=("phase",),
        invalidates=("org.diffpes.claim.phase.resolved",),
    )
    normalization = _record(
        "org.diffpes.transform.test.normalize",
        ("intensity",),
        ("normalized",),
        preserves=("energy-axis", "intensity-observable"),
        introduces=("dimensionless-scale",),
        destroys=("absolute-scale",),
        invalidates=("org.diffpes.claim.intensity.absolute",),
    )
    graph = build_provenance(
        (normalization, intensity, amplitude),
        external_inputs={"bands": ("energy-axis", "absolute-scale", "phase")},
    )
    report = validate_provenance(graph)
    assert report.valid, report.errors
    assert graph.topological_order == ("amplitude", "intensity", "normalized")
    state = effective_information(graph, "normalized")
    assert state.active_semantics == (
        "dimensionless-scale",
        "energy-axis",
        "intensity-observable",
    )
    assert set(state.destroyed_information) == {
        "absolute-scale",
        "matrix-element-amplitude",
        "phase",
    }
    assert invalidated_claims(graph, "normalized") == (
        "org.diffpes.claim.intensity.absolute",
        "org.diffpes.claim.phase.resolved",
    )
    assert lineage(graph, "normalized") == (
        "amplitude",
        "bands",
        "intensity",
    )


def test_cycle_is_rejected_and_remains_inspectable_in_nonstrict_mode() -> None:
    """Detect a multi-step cycle without hanging topological analysis."""
    first = _record(
        "org.diffpes.transform.test.cycle_a",
        ("b",),
        ("a",),
    )
    second = _record(
        "org.diffpes.transform.test.cycle_b",
        ("a",),
        ("b",),
    )
    with pytest.raises(ProvenanceError, match="cycle"):
        build_provenance((first, second))
    graph = build_provenance((first, second), strict=False)
    report = validate_provenance(graph)
    assert not report.valid
    assert any("cycle" in error for error in report.errors)


def test_missing_parent_and_multiple_producer_are_rejected() -> None:
    """Require every non-source parent and every output producer to be unique."""
    missing = _record(
        "org.diffpes.transform.test.missing",
        ("not-declared",),
        ("result",),
    )
    with pytest.raises(ProvenanceError, match="missing parents"):
        build_provenance((missing,))

    first = _record("org.diffpes.transform.test.one", (), ("same",))
    second = _record("org.diffpes.transform.test.two", (), ("same",))
    with pytest.raises(ProvenanceError, match="multiple transformations"):
        build_provenance((first, second))


def test_unused_external_input_is_rejected() -> None:
    """Expose required inputs that were declared but never consumed."""
    source = _record("org.diffpes.transform.test.source", (), ("result",))
    with pytest.raises(ProvenanceError, match="not consumed"):
        build_provenance((source,), external_inputs=("unused",))


def test_source_record_and_multiple_outputs_are_supported() -> None:
    """Allow explicit source nodes and one transformation with several outputs."""
    source = _record(
        "org.diffpes.transform.test.source_multi",
        (),
        ("left", "right"),
        introduces=("parsed",),
    )
    graph = build_provenance((source,))
    report = validate_provenance(graph)
    assert report.valid
    assert report.roots == ("left", "right")
    assert report.terminal_outputs == ("left", "right")
    assert effective_information(graph, "left").active_semantics == ("parsed",)


def test_unknown_node_inspection_is_rejected() -> None:
    """Avoid silently returning empty state for a misspelled artifact ID."""
    source = _record("org.diffpes.transform.test.known", (), ("known",))
    graph = build_provenance((source,))
    with pytest.raises(KeyError, match="unknown provenance node"):
        effective_information(graph, "unknown")
    with pytest.raises(KeyError, match="unknown provenance node"):
        lineage(graph, "unknown")


@given(
    st.lists(st.integers(min_value=0, max_value=31), min_size=1, max_size=12)
)
def test_generated_acyclic_graphs_validate(parent_choices) -> None:
    """Validate generated DAGs independently of their supplied record order."""
    records = []
    for index, choice in enumerate(parent_choices):
        parents = () if index == 0 else (f"node-{choice % index}",)
        records.append(
            _record(
                f"org.diffpes.transform.generated.step_{index}",
                parents,
                (f"node-{index}",),
                preserves=("axis",),
                introduces=(f"stage-{index}",),
            )
        )
    graph = build_provenance(tuple(reversed(records)))
    report = validate_provenance(graph)
    assert report.valid, report.errors
    assert set(graph.topological_order) == {
        f"node-{index}" for index in range(len(records))
    }
