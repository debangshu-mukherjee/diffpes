"""Compute non-security consistency checksums for scientific records.

Extended Summary
----------------
Checksums in this module detect accidental disagreement between canonical
records.  They are ordinary CRC32 bookkeeping values and provide no evidence
of authorship, authenticity, physical validity, numerical correctness, or
reproducibility. Certification policy must not treat a checksum as a
scientific claim.

Every returned string records the algorithm, canonicalization version, record
kind, and checksum value. The functions process large carrier and file
payloads in bounded chunks.

Routine Listings
----------------
:func:`checksum_bytes`
    Return a non-security consistency checksum for ``data``.
:func:`checksum_chunks`
    Compute a consistency checksum over consecutive byte chunks.
:func:`checksum_file`
    Stream exact file bytes into a consistency checksum.
:func:`checksum_pytree`
    Stream a canonical carrier into a consistency checksum.
:func:`parse_checksum`
    Parse and validate one checksum string.
:func:`artifact_ref`
    Build separate byte, normalized-content, and semantic identities.
:func:`semantic_checksum`
    Identify content together with its declared scientific meaning.
:func:`result_checksum`
    Identify a result under a declared numerical configuration.
"""

from __future__ import annotations

import re
import zlib
from collections.abc import Iterable
from pathlib import Path

from beartype import beartype
from beartype.typing import TYPE_CHECKING, Any
from jaxtyping import jaxtyped

from diffpes.types import (
    CANONICAL_PYTREE_VERSION,
    CHECKSUM_ALGORITHM,
    CHECKSUM_FILE_CHUNK_BYTES,
    CHECKSUM_PATTERN,
    CHECKSUM_RECORD_KIND_PATTERN,
    make_artifact_ref,
)

from .canonical import iter_canonical_pytree_chunks

if TYPE_CHECKING:
    from diffpes.types import ArtifactRef


def _validate_record_kind(record_kind: str) -> None:
    """Reject ambiguous or unstable checksum record-kind labels."""
    if CHECKSUM_RECORD_KIND_PATTERN.fullmatch(record_kind) is None:
        msg: str = (
            "record_kind must start with a lowercase letter and contain "
            "only lowercase letters, digits, and hyphens"
        )
        raise ValueError(msg)


def _format_checksum(value: int, *, record_kind: str) -> str:
    """Format a CRC32 value with its bookkeeping context."""
    checksum: str = (
        f"{CHECKSUM_ALGORITHM}:canonical-{CANONICAL_PYTREE_VERSION}:"
        f"{record_kind}:{value & 0xFFFFFFFF:08x}"
    )
    return checksum


@jaxtyped(typechecker=beartype)
def checksum_chunks(
    chunks: Iterable[bytes | memoryview],
    *,
    record_kind: str,
) -> str:
    """Compute a consistency checksum over consecutive byte chunks.

    The CRC32 value detects accidental disagreement in canonical scientific
    records. It is bookkeeping evidence, not cryptographic authentication.

    :see: :class:`~.test_checksums.TestChecksumChunks`


    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           checksum: str = _format_checksum(value, record_kind=record_kind)

       The function validates and transforms the inputs before it binds the
       documented output.

    Parameters
    ----------
    chunks : Iterable[bytes | memoryview]
        Consecutive byte-like pieces of one record.
    record_kind : str
        Stable description, such as ``"normalized-content"`` or ``"result"``.

    Returns
    -------
    checksum : str
        Versioned, typed CRC32 consistency checksum.
    """
    chunk: Any
    _validate_record_kind(record_kind)
    value: int = 0
    for chunk in chunks:
        value = zlib.crc32(chunk, value)
    checksum: str = _format_checksum(value, record_kind=record_kind)
    return checksum


@jaxtyped(typechecker=beartype)
def checksum_bytes(data: bytes, *, record_kind: str) -> str:
    """Return a non-security consistency checksum for ``data``.

    The CRC32 value detects accidental disagreement in canonical scientific
    records. It is bookkeeping evidence, not cryptographic authentication.

    :see: :class:`~.test_checksums.TestChecksumBytes`


    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           checksum: str = checksum_chunks((data,), record_kind=record_kind)

       The function validates and transforms the inputs before it binds the
       documented output.

    Parameters
    ----------
    data : bytes
        Exact record bytes.
    record_kind : str
        Stable kind distinguishing otherwise equal byte payloads.

    Returns
    -------
    checksum : str
        Typed CRC32 bookkeeping value.
    """
    checksum: str = checksum_chunks((data,), record_kind=record_kind)
    return checksum


@jaxtyped(typechecker=beartype)
def checksum_pytree(tree: object, *, record_kind: str) -> str:
    """Stream a canonical carrier into a consistency checksum.

    The CRC32 value detects accidental disagreement in canonical scientific
    records. It is bookkeeping evidence, not cryptographic authentication.

    :see: :class:`~.test_checksums.TestChecksumPytree`


    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           checksum: str = checksum_chunks(chunks, record_kind=record_kind)

       The function validates and transforms the inputs before it binds the
       documented output.

    Parameters
    ----------
    tree : object
        Supported carrier or nested scientific PyTree.
    record_kind : str
        Stable record-kind label.

    Returns
    -------
    checksum : str
        Typed CRC32 bookkeeping value.
    """
    chunks: Iterable[bytes | memoryview] = iter_canonical_pytree_chunks(tree)
    checksum: str = checksum_chunks(chunks, record_kind=record_kind)
    return checksum


@jaxtyped(typechecker=beartype)
def checksum_file(path: str | Path, *, record_kind: str) -> str:
    """Stream exact file bytes into a consistency checksum.

    The CRC32 value detects accidental disagreement in canonical scientific
    records. It is bookkeeping evidence, not cryptographic authentication.

    :see: :class:`~.test_checksums.TestChecksumFile`


    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           checksum: str = checksum_chunks(chunks(), record_kind=record_kind)

       The function validates and transforms the inputs before it binds the
       documented output.

    Parameters
    ----------
    path : str | Path
        Existing regular file.
    record_kind : str
        Stable record-kind label.

    Returns
    -------
    checksum : str
        Typed CRC32 bookkeeping value.
    """
    source: Path = Path(path)

    def chunks() -> Iterable[bytes]:
        stream: Any
        chunk: Any
        with source.open("rb") as stream:
            while chunk := stream.read(CHECKSUM_FILE_CHUNK_BYTES):
                yield chunk

    checksum: str = checksum_chunks(chunks(), record_kind=record_kind)
    return checksum


@jaxtyped(typechecker=beartype)
def parse_checksum(checksum: str) -> tuple[str, str, str, str]:
    """Parse and validate one checksum string.

    The CRC32 value detects accidental disagreement in canonical scientific
    records. It is bookkeeping evidence, not cryptographic authentication.

    :see: :class:`~.test_checksums.TestParseChecksum`


    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           parsed: tuple[str, str, str, str] = (
                   CHECKSUM_ALGORITHM,
                   match.group("canonical"),
                   match.group("kind"),
                   match.group("value"),
               )

       The function validates and transforms the inputs before it binds the
       documented output.

    Parameters
    ----------
    checksum : str
        Value produced by this module.

    Returns
    -------
    parsed : tuple[str, str, str, str]
        Algorithm, canonicalization version, record kind, and hexadecimal
        value.

    Raises
    ------
    ValueError
        If ``checksum`` is not in the current explicit format.
    """
    match: re.Match[str] | None = CHECKSUM_PATTERN.fullmatch(checksum)
    if match is None:
        msg: str = "invalid DiffPES consistency-checksum format"
        raise ValueError(msg)
    parsed: tuple[str, str, str, str] = (
        CHECKSUM_ALGORITHM,
        match.group("canonical"),
        match.group("kind"),
        match.group("value"),
    )
    return parsed


@jaxtyped(typechecker=beartype)
def semantic_checksum(
    value: object,
    semantics: object,
) -> str:
    """Identify content together with its declared scientific meaning.

    The CRC32 value detects accidental disagreement in canonical scientific
    records. It is bookkeeping evidence, not cryptographic authentication.

    :see: :class:`~.test_checksums.TestSemanticChecksum`


    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           checksum: str = checksum_pytree(payload, record_kind="semantic")

       The function validates and transforms the inputs before it binds the
       documented output.

    Parameters
    ----------
    value : object
        Normalized scientific content.
    semantics : object
        Units, axes, frames, conventions, schema, and other meaning-bearing
        declarations.

    Returns
    -------
    checksum : str
        Non-security semantic consistency checksum.
    """
    payload: tuple[str, object, object] = (
        "org.diffpes.semantic-record.v1",
        value,
        semantics,
    )
    checksum: str = checksum_pytree(payload, record_kind="semantic")
    return checksum


@jaxtyped(typechecker=beartype)
def result_checksum(value: object, numerical: object) -> str:
    """Identify a result under a declared numerical configuration.

    The CRC32 value detects accidental disagreement in canonical scientific
    records. It is bookkeeping evidence, not cryptographic authentication.

    :see: :class:`~.test_checksums.TestResultChecksum`


    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           checksum: str = checksum_pytree(payload, record_kind="result")

       The function validates and transforms the inputs before it binds the
       documented output.

    Parameters
    ----------
    value : object
        Result carrier.
    numerical : object
        Precision, backend-independent tolerance semantics, and any other
        numerical configuration that defines result identity.

    Returns
    -------
    checksum : str
        Non-security result consistency checksum.
    """
    payload: tuple[str, object, object] = (
        "org.diffpes.result-record.v1",
        value,
        numerical,
    )
    checksum: str = checksum_pytree(payload, record_kind="result")
    return checksum


@jaxtyped(typechecker=beartype)
def artifact_ref(
    path: str | Path,
    normalized: object,
    *,
    role: str,
    media_type: str = "application/octet-stream",
    semantics: object | None = None,
    artifact_id: str | None = None,
) -> ArtifactRef:
    """Build separate byte, normalized-content, and semantic identities.

    The CRC32 value detects accidental disagreement in canonical scientific
    records. It is bookkeeping evidence, not cryptographic authentication.

    :see: :class:`~.test_checksums.TestArtifactRef`


    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           reference: ArtifactRef = make_artifact_ref(
                   artifact_id=resolved_id,
                   media_type=media_type,
                   byte_checksum=byte_value,
                   content_checksum=content_value,
                   semantic_checksum=semantic_value,
                   locator=str(source),
                   role=role,
               )

       The function validates and transforms the inputs before it binds the
       documented output.

    Parameters
    ----------
    path : str | Path
        Source artifact with a record of its exact bytes.
    normalized : object
        Parsed, normalized scientific carrier derived from the source.
    role : str
        Scientific role of this artifact in the execution.
    media_type : str, optional
        Declared media type of the source bytes.
    semantics : object | None, optional
        Meaning-bearing declarations. By default, the role and normalized
        carrier type form the minimal semantic descriptor.
    artifact_id : str | None, optional
        Stable caller-owned identity. By default, the function derives a local
        identity from the byte checksum value.

    Returns
    -------
    reference : ArtifactRef
        Immutable certification carrier with three deliberately separate
        consistency checksums.
    """
    source: Path = Path(path)
    byte_value: str = checksum_file(source, record_kind="artifact-bytes")
    content_value: str = checksum_pytree(
        normalized,
        record_kind="normalized-content",
    )
    descriptor: object
    if semantics is None:
        descriptor = {
            "carrier_type": (
                f"{type(normalized).__module__}."
                f"{type(normalized).__qualname__}"
            ),
            "role": role,
        }
    else:
        descriptor = semantics
    semantic_value: str = semantic_checksum(normalized, descriptor)
    if artifact_id is None:
        checksum_value: str = parse_checksum(byte_value)[3]
        resolved_id: str = f"artifact-{checksum_value}"
    else:
        resolved_id = artifact_id
    reference: ArtifactRef = make_artifact_ref(
        artifact_id=resolved_id,
        media_type=media_type,
        byte_checksum=byte_value,
        content_checksum=content_value,
        semantic_checksum=semantic_value,
        locator=str(source),
        role=role,
    )
    return reference


__all__: list[str] = [
    "artifact_ref",
    "checksum_bytes",
    "checksum_chunks",
    "checksum_file",
    "checksum_pytree",
    "parse_checksum",
    "result_checksum",
    "semantic_checksum",
]
