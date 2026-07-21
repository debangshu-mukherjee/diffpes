"""Tests for non-security scientific-record checksums."""

import jax.numpy as jnp
import pytest

from diffpes.certify.checksums import (
    CHECKSUM_ALGORITHM,
    artifact_ref,
    checksum_bytes,
    checksum_chunks,
    checksum_pytree,
    parse_checksum,
    result_checksum,
    semantic_checksum,
)


def test_checksum_is_typed_versioned_and_streaming_stable() -> None:
    """Record bookkeeping context and remain invariant to chunk boundaries."""
    whole = checksum_bytes(b"abcdef", record_kind="result")
    streamed = checksum_chunks((b"ab", b"c", b"def"), record_kind="result")
    assert whole == streamed
    algorithm, version, kind, value = parse_checksum(whole)
    assert algorithm == CHECKSUM_ALGORITHM == "crc32"
    assert version == "1"
    assert kind == "result"
    assert len(value) == 8


def test_record_kind_and_one_bit_change_checksum() -> None:
    """Detect accidental byte disagreement and distinguish record purposes."""
    original = checksum_bytes(b"\x00", record_kind="result")
    changed = checksum_bytes(b"\x01", record_kind="result")
    semantic = checksum_bytes(b"\x00", record_kind="semantic")
    assert original != changed
    assert original != semantic


def test_semantic_and_numerical_declarations_affect_identity() -> None:
    """Distinguish equal arrays with different units or precision contracts."""
    value = jnp.array([1.0, 2.0])
    assert semantic_checksum(value, {"unit": "eV"}) != semantic_checksum(
        value, {"unit": "hartree"}
    )
    assert result_checksum(value, {"precision": "float64"}) != result_checksum(
        value, {"precision": "float32"}
    )


def test_streaming_pytree_matches_equal_jax_values() -> None:
    """Produce stable normalized-content bookkeeping for equal carriers."""
    left = checksum_pytree(jnp.arange(8), record_kind="normalized-content")
    right = checksum_pytree(jnp.arange(8), record_kind="normalized-content")
    assert left == right


def test_artifact_ref_separates_byte_content_and_semantics(tmp_path) -> None:
    """Record exact source bytes separately from parsed scientific meaning."""
    path = tmp_path / "bands.dat"
    path.write_bytes(b"1 2 3\n")
    normalized = jnp.array([1.0, 2.0, 3.0])
    reference = artifact_ref(
        path,
        normalized,
        role="initial-bands",
        semantics={"unit": "eV", "axis": "band"},
    )
    assert parse_checksum(reference.byte_checksum)[2] == "artifact-bytes"
    assert (
        parse_checksum(reference.content_checksum)[2] == "normalized-content"
    )
    assert parse_checksum(reference.semantic_checksum)[2] == "semantic"
    assert reference.role == "initial-bands"
    assert reference.locator == str(path)


@pytest.mark.parametrize("record_kind", ["", "Result", "two words", "a/b"])
def test_invalid_record_kind_is_rejected(record_kind) -> None:
    """Reject record labels that would make the checksum format ambiguous."""
    with pytest.raises(ValueError, match="record_kind"):
        checksum_bytes(b"value", record_kind=record_kind)


def test_invalid_checksum_record_is_rejected() -> None:
    """Refuse incomplete and hand-written checksum forms."""
    with pytest.raises(ValueError, match="format"):
        parse_checksum("crc32:1234")
