"""Artifact lineage and semantic information-loss graphs.

Extended Summary
----------------
This module builds a deterministic directed acyclic graph from immutable
``TransformationRecord`` carriers.  It rejects missing parents, cycles,
multiple producers, and unused declared inputs.  Semantic properties,
information losses, and claim invalidations are propagated to every output.

Propagation is deliberately conservative: inherited semantics remain active
only when a transformation explicitly lists them in ``preserves``.  A missing
preservation declaration therefore becomes a visible information loss rather
than an accidental scientific promise.

Routine Listings
----------------
:func:`build_provenance`
    Validate records, topologically order them, and propagate information.
:func:`validate_provenance`
    Re-evaluate the complete graph structure and derived state.
:func:`effective_information`
    Inspect semantics and losses at one graph output.
:func:`lineage`
    Return the transitive artifact ancestry of one output.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable, Mapping, Sequence

import equinox as eqx

from diffpes.types.certification import TransformationRecord

from .checksums import checksum_pytree


class InformationState(eqx.Module):
    """Effective semantic state attached to one artifact or result node."""

    node_id: str = eqx.field(static=True)
    active_semantics: tuple[str, ...] = eqx.field(static=True)
    destroyed_information: tuple[str, ...] = eqx.field(static=True)
    invalidated_claims: tuple[str, ...] = eqx.field(static=True)


class ProvenanceGraph(eqx.Module):
    """Immutable validated lineage graph and propagated semantic state."""

    records: tuple[TransformationRecord, ...]
    external_inputs: tuple[str, ...] = eqx.field(static=True)
    initial_semantics: tuple[tuple[str, tuple[str, ...]], ...] = eqx.field(
        static=True
    )
    topological_order: tuple[str, ...] = eqx.field(static=True)
    information: tuple[InformationState, ...]
    validation_errors: tuple[str, ...] = eqx.field(static=True)
    graph_checksum: str = eqx.field(static=True)


class ProvenanceReport(eqx.Module):
    """Structural and semantic validation report for a provenance graph."""

    valid: bool = eqx.field(static=True)
    errors: tuple[str, ...] = eqx.field(static=True)
    roots: tuple[str, ...] = eqx.field(static=True)
    terminal_outputs: tuple[str, ...] = eqx.field(static=True)
    orphaned_inputs: tuple[str, ...] = eqx.field(static=True)
    topological_order: tuple[str, ...] = eqx.field(static=True)
    graph_checksum: str = eqx.field(static=True)


class ProvenanceError(ValueError):
    """Report an invalid artifact-lineage or semantic graph."""


class _Analysis(eqx.Module):
    """Internal complete analysis before construction or comparison."""

    ordered_records: tuple[TransformationRecord, ...]
    topological_order: tuple[str, ...] = eqx.field(static=True)
    information: tuple[InformationState, ...]
    errors: tuple[str, ...] = eqx.field(static=True)
    roots: tuple[str, ...] = eqx.field(static=True)
    terminal_outputs: tuple[str, ...] = eqx.field(static=True)
    orphaned_inputs: tuple[str, ...] = eqx.field(static=True)


def _normalize_terms(
    values: Iterable[str],
    *,
    field_name: str,
) -> tuple[str, ...]:
    """Validate and deterministically order one identifier/property set."""
    normalized = tuple(values)
    if any(
        not isinstance(value, str) or not value.strip() for value in normalized
    ):
        msg = f"{field_name} entries must be nonblank strings"
        raise ProvenanceError(msg)
    if len(set(normalized)) != len(normalized):
        msg = f"{field_name} entries must be unique"
        raise ProvenanceError(msg)
    return tuple(sorted(normalized))


def _normalize_external_inputs(
    external_inputs: Mapping[str, Iterable[str]] | Iterable[str],
) -> tuple[tuple[str, ...], tuple[tuple[str, tuple[str, ...]], ...]]:
    """Normalize external node identities and their input semantics."""
    if isinstance(external_inputs, Mapping):
        node_ids = _normalize_terms(
            external_inputs.keys(),
            field_name="external_inputs",
        )
        semantic_pairs = tuple(
            (
                node_id,
                _normalize_terms(
                    external_inputs[node_id],
                    field_name=f"initial semantics for {node_id}",
                ),
            )
            for node_id in node_ids
        )
    else:
        node_ids = _normalize_terms(
            external_inputs,
            field_name="external_inputs",
        )
        semantic_pairs = tuple((node_id, ()) for node_id in node_ids)
    return node_ids, semantic_pairs


def _record_key(
    record: TransformationRecord,
    original_index: int,
) -> tuple[str, str, tuple[str, ...], int]:
    """Return a deterministic ordering key for one transformation execution."""
    return (
        record.transformation_id,
        record.transformation_version,
        record.output_ids,
        original_index,
    )


def _record_errors(  # noqa: PLR0912
    records: Sequence[TransformationRecord],
    external_inputs: tuple[str, ...],
) -> tuple[
    list[str],
    dict[str, int],
    set[str],
    tuple[str, ...],
]:
    """Collect local record and node-identity errors."""
    errors: list[str] = []
    producer: dict[str, int] = {}
    consumed: set[str] = set()
    all_outputs: list[str] = []
    for index, record in enumerate(records):
        reference = (
            f"{record.transformation_id}@{record.transformation_version}"
        )
        if not record.transformation_id.strip():
            errors.append(f"record {index} has a blank transformation ID")
        if not record.transformation_version.strip():
            errors.append(f"record {index} has a blank transformation version")
        if not record.output_ids:
            errors.append(f"record {index} ({reference}) has no outputs")
        if len(set(record.parent_ids)) != len(record.parent_ids):
            errors.append(f"record {index} ({reference}) repeats a parent")
        if len(set(record.output_ids)) != len(record.output_ids):
            errors.append(f"record {index} ({reference}) repeats an output")
        overlap = set(record.parent_ids) & set(record.output_ids)
        if overlap:
            details = ", ".join(sorted(overlap))
            errors.append(
                f"record {index} ({reference}) directly consumes its own "
                f"output: {details}"
            )
        contradictory = set(record.preserves) & set(record.destroys)
        contradictory |= set(record.introduces) & set(record.destroys)
        if contradictory:
            details = ", ".join(sorted(contradictory))
            errors.append(
                f"record {index} ({reference}) both retains and destroys: "
                f"{details}"
            )
        for output_id in record.output_ids:
            if not output_id.strip():
                errors.append(
                    f"record {index} ({reference}) has a blank output"
                )
                continue
            if output_id in producer:
                errors.append(
                    f"multiple transformations produce output {output_id!r}"
                )
            else:
                producer[output_id] = index
            all_outputs.append(output_id)
        consumed.update(record.parent_ids)
    collision = set(external_inputs) & set(all_outputs)
    if collision:
        details = ", ".join(sorted(collision))
        errors.append(
            f"external inputs are also produced internally: {details}"
        )
    known_nodes = set(external_inputs) | set(all_outputs)
    for index, record in enumerate(records):
        missing = set(record.parent_ids) - known_nodes
        if missing:
            details = ", ".join(sorted(missing))
            errors.append(f"record {index} has missing parents: {details}")
    orphaned = (
        tuple(sorted(set(external_inputs) - consumed)) if records else ()
    )
    if orphaned:
        errors.append(
            "declared external inputs are not consumed: " + ", ".join(orphaned)
        )
    return errors, producer, consumed, orphaned


def _topological_indices(
    records: Sequence[TransformationRecord],
    producer: Mapping[str, int],
) -> tuple[tuple[int, ...], bool]:
    """Return deterministic Kahn ordering and whether a cycle remains."""
    dependencies: list[set[int]] = []
    children: list[set[int]] = [set() for _ in records]
    for index, record in enumerate(records):
        parents = {
            producer[parent]
            for parent in record.parent_ids
            if parent in producer and producer[parent] != index
        }
        dependencies.append(parents)
        for parent_index in parents:
            children[parent_index].add(index)
    indegree = [len(parents) for parents in dependencies]
    ready: list[tuple[tuple[str, str, tuple[str, ...], int], int]] = []
    for index, degree in enumerate(indegree):
        if degree == 0:
            heapq.heappush(ready, (_record_key(records[index], index), index))
    ordered: list[int] = []
    while ready:
        _, index = heapq.heappop(ready)
        ordered.append(index)
        for child in sorted(children[index]):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(
                    ready,
                    (_record_key(records[child], child), child),
                )
    has_cycle = len(ordered) != len(records)
    if has_cycle:
        remaining = sorted(
            set(range(len(records))) - set(ordered),
            key=lambda index: _record_key(records[index], index),
        )
        ordered.extend(remaining)
    return tuple(ordered), has_cycle


def _propagate_information(
    ordered_indices: Sequence[int],
    records: Sequence[TransformationRecord],
    initial_semantics: tuple[tuple[str, tuple[str, ...]], ...],
    *,
    has_cycle: bool,
) -> tuple[InformationState, ...]:
    """Propagate semantics only when the graph admits a complete ordering."""
    state: dict[str, InformationState] = {
        node_id: InformationState(
            node_id=node_id,
            active_semantics=semantics,
            destroyed_information=(),
            invalidated_claims=(),
        )
        for node_id, semantics in initial_semantics
    }
    if has_cycle:
        return tuple(state[node_id] for node_id, _ in initial_semantics)
    for index in ordered_indices:
        record = records[index]
        parent_states = tuple(
            state[parent_id]
            for parent_id in record.parent_ids
            if parent_id in state
        )
        inherited = {
            term
            for parent_state in parent_states
            for term in parent_state.active_semantics
        }
        losses = {
            term
            for parent_state in parent_states
            for term in parent_state.destroyed_information
        }
        invalidated = {
            claim
            for parent_state in parent_states
            for claim in parent_state.invalidated_claims
        }
        retained = inherited & set(record.preserves)
        losses.update(inherited - retained)
        losses.update(record.destroys)
        retained.difference_update(record.destroys)
        retained.update(record.introduces)
        invalidated.update(record.invalidates_claims)
        for output_id in record.output_ids:
            state[output_id] = InformationState(
                node_id=output_id,
                active_semantics=tuple(sorted(retained)),
                destroyed_information=tuple(sorted(losses)),
                invalidated_claims=tuple(sorted(invalidated)),
            )
    return tuple(state[node_id] for node_id in sorted(state))


def _analyze(
    records: Sequence[TransformationRecord],
    external_inputs: tuple[str, ...],
    initial_semantics: tuple[tuple[str, tuple[str, ...]], ...],
) -> _Analysis:
    """Perform deterministic structural analysis and semantic propagation."""
    errors, producer, consumed, orphaned = _record_errors(
        records,
        external_inputs,
    )
    ordered_indices, has_cycle = _topological_indices(records, producer)
    if has_cycle:
        errors.append("provenance graph contains a cycle")
    ordered_records = tuple(records[index] for index in ordered_indices)
    output_order = tuple(
        output_id
        for record in ordered_records
        for output_id in record.output_ids
    )
    information = _propagate_information(
        ordered_indices,
        records,
        initial_semantics,
        has_cycle=has_cycle,
    )
    all_outputs = set(producer)
    roots = tuple(
        sorted(
            set(external_inputs)
            | {
                output_id
                for record in records
                if not record.parent_ids
                for output_id in record.output_ids
            }
        )
    )
    terminal_outputs = tuple(sorted(all_outputs - consumed))
    return _Analysis(
        ordered_records=ordered_records,
        topological_order=output_order,
        information=information,
        errors=tuple(errors),
        roots=roots,
        terminal_outputs=terminal_outputs,
        orphaned_inputs=orphaned,
    )


def build_provenance(
    records: Sequence[TransformationRecord],
    *,
    external_inputs: Mapping[str, Iterable[str]] | Iterable[str] = (),
    strict: bool = True,
) -> ProvenanceGraph:
    """Build a deterministic provenance DAG and propagate information.

    Parameters
    ----------
    records : Sequence[TransformationRecord]
        Transformation executions linking parent and output artifact IDs.
    external_inputs : Mapping[str, Iterable[str]] or Iterable[str], optional
        Known source-node IDs. A mapping additionally declares the semantic
        properties initially available on each source.
    strict : bool, optional
        Raise :class:`ProvenanceError` for any invalid graph. ``False`` is
        useful to construct an inspectable rejected graph.

    Returns
    -------
    graph : ProvenanceGraph
        Immutable graph with propagated per-node semantic state.

    Raises
    ------
    ProvenanceError
        If strict validation detects a cycle, missing/duplicate node, unused
        input, or contradictory information declaration.
    """
    normalized_records = tuple(records)
    input_ids, semantic_pairs = _normalize_external_inputs(external_inputs)
    analysis = _analyze(normalized_records, input_ids, semantic_pairs)
    if strict and analysis.errors:
        raise ProvenanceError("; ".join(analysis.errors))
    checksum = checksum_pytree(
        (normalized_records, semantic_pairs),
        record_kind="provenance",
    )
    return ProvenanceGraph(
        records=normalized_records,
        external_inputs=input_ids,
        initial_semantics=semantic_pairs,
        topological_order=analysis.topological_order,
        information=analysis.information,
        validation_errors=analysis.errors,
        graph_checksum=checksum,
    )


def validate_provenance(graph: ProvenanceGraph) -> ProvenanceReport:
    """Independently re-evaluate graph structure and derived state.

    Parameters
    ----------
    graph : ProvenanceGraph
        Graph returned by :func:`build_provenance` or deserialized elsewhere.

    Returns
    -------
    report : ProvenanceReport
        Complete deterministic validation result.
    """
    analysis = _analyze(
        graph.records,
        graph.external_inputs,
        graph.initial_semantics,
    )
    errors = list(analysis.errors)
    expected_checksum = checksum_pytree(
        (graph.records, graph.initial_semantics),
        record_kind="provenance",
    )
    if graph.graph_checksum != expected_checksum:
        errors.append("provenance consistency checksum does not match records")
    if graph.topological_order != analysis.topological_order:
        errors.append(
            "stored topological order does not match graph structure"
        )
    if graph.information != analysis.information:
        errors.append("stored information propagation does not match records")
    return ProvenanceReport(
        valid=not errors,
        errors=tuple(errors),
        roots=analysis.roots,
        terminal_outputs=analysis.terminal_outputs,
        orphaned_inputs=analysis.orphaned_inputs,
        topological_order=analysis.topological_order,
        graph_checksum=expected_checksum,
    )


def effective_information(
    graph: ProvenanceGraph,
    output_id: str,
) -> InformationState:
    """Return propagated semantics, losses, and invalidations for one node.

    Raises
    ------
    KeyError
        If ``output_id`` is absent from the graph.
    """
    for state in graph.information:
        if state.node_id == output_id:
            return state
    msg = f"unknown provenance node: {output_id}"
    raise KeyError(msg)


def invalidated_claims(
    graph: ProvenanceGraph,
    output_id: str,
) -> tuple[str, ...]:
    """Return every claim invalidated at or upstream of one output."""
    return effective_information(graph, output_id).invalidated_claims


def lineage(graph: ProvenanceGraph, output_id: str) -> tuple[str, ...]:
    """Return the transitive parent-node lineage of one output.

    Raises
    ------
    KeyError
        If ``output_id`` is absent from the graph.
    """
    producer: dict[str, TransformationRecord] = {
        produced: record
        for record in graph.records
        for produced in record.output_ids
    }
    known = set(graph.external_inputs) | set(producer)
    if output_id not in known:
        msg = f"unknown provenance node: {output_id}"
        raise KeyError(msg)
    ancestors: set[str] = set()
    pending = [output_id]
    while pending:
        current = pending.pop()
        record = producer.get(current)
        if record is None:
            continue
        for parent_id in record.parent_ids:
            if parent_id not in ancestors:
                ancestors.add(parent_id)
                pending.append(parent_id)
    return tuple(sorted(ancestors))


__all__: list[str] = [
    "InformationState",
    "ProvenanceError",
    "ProvenanceGraph",
    "ProvenanceReport",
    "build_provenance",
    "effective_information",
    "invalidated_claims",
    "lineage",
    "validate_provenance",
]
