"""Non-security consistency checksums for scientific records.

Extended Summary
----------------
Checksums in this module detect accidental disagreement between canonical
records.  They are ordinary CRC32 bookkeeping values and provide no evidence
of authorship, authenticity, physical validity, numerical correctness, or
reproducibility.  Certification policy is forbidden from treating a checksum
as a scientific claim.

Every returned string records the algorithm, canonicalization version, record
kind, and checksum value.  Large carrier and file payloads are processed in
bounded chunks.

Routine Listings
----------------
:func:`checksum_bytes`
    Compute a typed checksum for an in-memory byte record.
:func:`checksum_pytree`
    Stream the canonical representation of a supported carrier.
:func:`artifact_ref`
    Describe exact source bytes and normalized scientific content separately.
:func:`semantic_checksum`
    Include scientific units, axes, frames, and conventions in an identity.
:func:`result_checksum`
    Include the declared numerical identity of a result.
"""

from __future__ import annotations

import re
import zlib
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from diffpes.types.certification import make_artifact_ref

from .canonical import (
    CANONICAL_PYTREE_VERSION,
    CanonicalChunk,
    canonical_pytree,
    iter_canonical_pytree_chunks,
)

if TYPE_CHECKING:
    from diffpes.types.certification import ArtifactRef

type SemanticDescriptor = object
type NumericalIdentity = object

CHECKSUM_ALGORITHM: str = "crc32"
CHECKSUM_FORMAT_VERSION: str = "1"
_CHECKSUM_RE: re.Pattern[str] = re.compile(
    r"^crc32:canonical-(?P<canonical>[0-9]+):"
    r"(?P<kind>[a-z][a-z0-9-]*):(?P<value>[0-9a-f]{8})$"
)
_RECORD_KIND_RE: re.Pattern[str] = re.compile(r"^[a-z][a-z0-9-]*$")
_FILE_CHUNK_BYTES: int = 1024 * 1024


def _validate_record_kind(record_kind: str) -> None:
    """Reject ambiguous or unstable checksum record-kind labels."""
    if _RECORD_KIND_RE.fullmatch(record_kind) is None:
        msg = (
            "record_kind must start with a lowercase letter and contain "
            "only lowercase letters, digits, and hyphens"
        )
        raise ValueError(msg)


def _format_checksum(value: int, *, record_kind: str) -> str:
    """Format a CRC32 value with its bookkeeping context."""
    return (
        f"{CHECKSUM_ALGORITHM}:canonical-{CANONICAL_PYTREE_VERSION}:"
        f"{record_kind}:{value & 0xFFFFFFFF:08x}"
    )


def checksum_chunks(
    chunks: Iterable[CanonicalChunk],
    *,
    record_kind: str,
) -> str:
    """Compute a consistency checksum over consecutive byte chunks.

    Parameters
    ----------
    chunks : Iterable[CanonicalChunk]
        Consecutive byte-like pieces of one record.
    record_kind : str
        Stable description such as ``"normalized-content"`` or ``"result"``.

    Returns
    -------
    checksum : str
        Versioned, typed CRC32 consistency checksum.
    """
    _validate_record_kind(record_kind)
    value = 0
    for chunk in chunks:
        value = zlib.crc32(chunk, value)
    return _format_checksum(value, record_kind=record_kind)


def checksum_bytes(data: bytes, *, record_kind: str) -> str:
    """Return a non-security consistency checksum for ``data``.

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
    return checksum_chunks((data,), record_kind=record_kind)


def checksum_pytree(tree: object, *, record_kind: str) -> str:
    """Stream a canonical carrier into a consistency checksum.

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
    chunks = iter_canonical_pytree_chunks(tree)
    return checksum_chunks(chunks, record_kind=record_kind)


def checksum_file(path: str | Path, *, record_kind: str) -> str:
    """Stream exact file bytes into a consistency checksum.

    Parameters
    ----------
    path : str or Path
        Existing regular file.
    record_kind : str
        Stable record-kind label.

    Returns
    -------
    checksum : str
        Typed CRC32 bookkeeping value.
    """
    source = Path(path)

    def chunks() -> Iterable[bytes]:
        with source.open("rb") as stream:
            while chunk := stream.read(_FILE_CHUNK_BYTES):
                yield chunk

    return checksum_chunks(chunks(), record_kind=record_kind)


def parse_checksum(checksum: str) -> tuple[str, str, str, str]:
    """Parse and validate one checksum string.

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
    match = _CHECKSUM_RE.fullmatch(checksum)
    if match is None:
        msg = "invalid DiffPES consistency-checksum format"
        raise ValueError(msg)
    return (
        CHECKSUM_ALGORITHM,
        match.group("canonical"),
        match.group("kind"),
        match.group("value"),
    )


def semantic_checksum(
    value: object,
    semantics: SemanticDescriptor,
) -> str:
    """Identify content together with its declared scientific meaning.

    Parameters
    ----------
    value : object
        Normalized scientific content.
    semantics : SemanticDescriptor
        Units, axes, frames, conventions, schema, and other meaning-bearing
        declarations.

    Returns
    -------
    checksum : str
        Non-security semantic consistency checksum.
    """
    payload = ("org.diffpes.semantic-record.v1", value, semantics)
    return checksum_pytree(payload, record_kind="semantic")


def result_checksum(value: object, numerical: NumericalIdentity) -> str:
    """Identify a result under a declared numerical configuration.

    Parameters
    ----------
    value : object
        Result carrier.
    numerical : NumericalIdentity
        Precision, backend-independent tolerance semantics, and any other
        numerical configuration that defines result identity.

    Returns
    -------
    checksum : str
        Non-security result consistency checksum.
    """
    payload = ("org.diffpes.result-record.v1", value, numerical)
    return checksum_pytree(payload, record_kind="result")


def artifact_ref(
    path: str | Path,
    normalized: object,
    *,
    role: str,
    media_type: str = "application/octet-stream",
    semantics: SemanticDescriptor | None = None,
    artifact_id: str | None = None,
) -> ArtifactRef:
    """Build separate byte, normalized-content, and semantic identities.

    Parameters
    ----------
    path : str or Path
        Source artifact whose exact bytes are recorded.
    normalized : object
        Parsed, normalized scientific carrier derived from the source.
    role : str
        Scientific role of this artifact in the execution.
    media_type : str, optional
        Declared media type of the source bytes.
    semantics : SemanticDescriptor, optional
        Meaning-bearing declarations. By default, the role and normalized
        carrier type form the minimal semantic descriptor.
    artifact_id : str, optional
        Stable caller-owned identity. By default a local identity is derived
        from the byte checksum value for bookkeeping convenience.

    Returns
    -------
    reference : ArtifactRef
        Immutable certification carrier with three deliberately separate
        consistency checksums.
    """
    source = Path(path)
    byte_value = checksum_file(source, record_kind="artifact-bytes")
    content_value = checksum_pytree(
        normalized,
        record_kind="normalized-content",
    )
    descriptor: SemanticDescriptor
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
    semantic_value = semantic_checksum(normalized, descriptor)
    if artifact_id is None:
        checksum_value = parse_checksum(byte_value)[3]
        resolved_id = f"artifact-{checksum_value}"
    else:
        resolved_id = artifact_id
    return make_artifact_ref(
        artifact_id=resolved_id,
        media_type=media_type,
        byte_checksum=byte_value,
        content_checksum=content_value,
        semantic_checksum=semantic_value,
        locator=str(source),
        role=role,
    )


def canonical_checksum_bytes(tree: object) -> bytes:
    """Return canonical bytes used by :func:`checksum_pytree` for diagnostics.

    This convenience helper is intended for golden tests and inspection. It
    assembles the complete record and should not be used for large arrays.
    """
    return canonical_pytree(tree)


__all__: list[str] = [
    "CHECKSUM_ALGORITHM",
    "CHECKSUM_FORMAT_VERSION",
    "NumericalIdentity",
    "SemanticDescriptor",
    "artifact_ref",
    "canonical_checksum_bytes",
    "checksum_bytes",
    "checksum_chunks",
    "checksum_file",
    "checksum_pytree",
    "parse_checksum",
    "result_checksum",
    "semantic_checksum",
]
