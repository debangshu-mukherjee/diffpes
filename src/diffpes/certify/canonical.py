"""Canonical scientific-record representations for certification.

Extended Summary
----------------
This module turns the deliberately small set of values accepted by the
certification layer into deterministic byte records.  The representation is
typed: lists and tuples differ, array dtype and shape are retained, and Python
scalars are distinct from zero-dimensional arrays.  Text is normalized to NFC
and numerical arrays use little-endian C-order bytes.

Canonicalization is bookkeeping at the Python/JAX boundary.  It must never be
called from a traced numerical kernel and it contributes no physical or
numerical certification claim.

Routine Listings
----------------
:func:`canonical_json`
    Encode JSON-like data while retaining Python container and scalar types.
:func:`canonical_pytree`
    Encode a supported carrier or PyTree as one canonical byte string.
:func:`iter_canonical_pytree_chunks`
    Stream the same representation without assembling large array payloads.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import math
import struct
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

type JsonScalar = None | bool | int | float | str
type JsonValue = (
    JsonScalar
    | list[JsonValue]
    | tuple[JsonValue, ...]
    | Mapping[str, JsonValue]
)
type CanonicalChunk = bytes | memoryview

CANONICAL_JSON_VERSION: str = "1"
CANONICAL_PYTREE_VERSION: str = "1"
_JSON_PREFIX: bytes = b"DIFFPES-CANONICAL-JSON-V1\x00"
_PYTREE_PREFIX: bytes = b"DIFFPES-CANONICAL-PYTREE-V1\x00"
_DEFAULT_ARRAY_CHUNK_BYTES: int = 1024 * 1024
_SUPPORTED_ARRAY_KINDS: frozenset[str] = frozenset({"b", "i", "u", "f", "c"})


class CanonicalizationError(ValueError):
    """Report a value that has no stable certification representation."""


def _normalize_text(value: str) -> str:
    """Return the NFC-normalized form of ``value``."""
    return unicodedata.normalize("NFC", value)


def _length(value: int) -> bytes:
    """Encode a nonnegative record length as an unsigned 64-bit integer."""
    if value < 0:
        msg = "canonical record lengths must be nonnegative"
        raise CanonicalizationError(msg)
    return struct.pack(">Q", value)


def _text_record(tag: bytes, value: str) -> bytes:
    """Encode a tagged normalized UTF-8 text record."""
    encoded = _normalize_text(value).encode("utf-8")
    return tag + _length(len(encoded)) + encoded


def _qualified_name(value_type: type[Any]) -> str:
    """Return a stable module-qualified type name."""
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _float_bits(value: float) -> str:
    """Return exact IEEE-754 binary64 bits as lowercase hexadecimal."""
    if not math.isfinite(value):
        msg = "canonical records reject NaN and infinite floats"
        raise CanonicalizationError(msg)
    return struct.pack(">d", value).hex()


def _json_node(value: object) -> object:  # noqa: PLR0911
    """Convert JSON-like input to an explicitly typed JSON tree."""
    if value is None:
        return {"$none": True}
    if isinstance(value, bool):
        return {"$bool": value}
    if isinstance(value, int):
        return {"$int": str(value)}
    if isinstance(value, float):
        return {"$float64": _float_bits(value)}
    if isinstance(value, str):
        return {"$str": _normalize_text(value)}
    if isinstance(value, tuple):
        return {"$tuple": [_json_node(item) for item in value]}
    if isinstance(value, list):
        return {"$list": [_json_node(item) for item in value]}
    if isinstance(value, Mapping):
        normalized: list[tuple[str, object]] = []
        seen: set[str] = set()
        for key, item in value.items():
            if not isinstance(key, str):
                msg = "canonical JSON mappings require string keys"
                raise CanonicalizationError(msg)
            normalized_key = _normalize_text(key)
            if normalized_key in seen:
                msg = "mapping keys collide after Unicode normalization"
                raise CanonicalizationError(msg)
            seen.add(normalized_key)
            normalized.append((normalized_key, item))
        normalized.sort(key=lambda pair: pair[0].encode("utf-8"))
        return {
            "$map": [
                [{"$str": key}, _json_node(item)] for key, item in normalized
            ]
        }
    msg = f"unsupported canonical JSON value: {type(value)!r}"
    raise CanonicalizationError(msg)


def canonical_json(value: object) -> bytes:
    """Return deterministic typed JSON bytes for ``value``.

    Parameters
    ----------
    value : JsonValue
        JSON-like data. Mapping keys must be strings. Tuples are accepted and
        deliberately remain distinguishable from lists.

    Returns
    -------
    encoded : bytes
        Versioned canonical UTF-8 JSON record.

    Raises
    ------
    CanonicalizationError
        If a key is not text, normalized keys collide, a float is nonfinite,
        or a value is outside the supported vocabulary.
    """
    node = _json_node(value)
    payload = json.dumps(
        node,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _JSON_PREFIX + payload


def _is_array(value: object) -> bool:
    """Return whether ``value`` exposes a concrete NumPy/JAX array protocol."""
    if isinstance(value, np.ndarray | np.generic):
        return True
    return all(
        hasattr(value, attr) for attr in ("__array__", "dtype", "shape")
    )


def _canonical_array(value: object) -> np.ndarray:
    """Materialize one array in the canonical dtype and memory order."""
    try:
        array = np.asarray(value)
    except Exception as exc:
        msg = (
            "canonicalization requires concrete arrays and cannot consume "
            "a JAX tracer"
        )
        raise CanonicalizationError(msg) from exc
    if array.dtype.kind not in _SUPPORTED_ARRAY_KINDS:
        msg = f"unsupported array dtype for canonicalization: {array.dtype}"
        raise CanonicalizationError(msg)
    if array.dtype.kind in {"f", "c"} and not bool(np.all(np.isfinite(array))):
        msg = "canonical records reject arrays containing NaN or infinity"
        raise CanonicalizationError(msg)
    dtype = array.dtype.newbyteorder("<")
    return np.asarray(array, dtype=dtype, order="C")


def _iter_array_chunks(
    value: object,
    *,
    chunk_bytes: int,
) -> Iterator[CanonicalChunk]:
    """Yield a canonical typed header followed by bounded array chunks."""
    array = _canonical_array(value)
    dtype_text = array.dtype.str.encode("ascii")
    yield b"A" + _length(len(dtype_text)) + dtype_text
    yield _length(array.ndim)
    for dimension in array.shape:
        yield _length(int(dimension))
    yield _length(array.nbytes)
    if array.nbytes == 0:
        return
    payload = memoryview(array).cast("B")
    for offset in range(0, array.nbytes, chunk_bytes):
        yield payload[offset : offset + chunk_bytes]


def _iter_mapping_chunks(
    value: Mapping[object, object],
    *,
    chunk_bytes: int,
) -> Iterator[CanonicalChunk]:
    """Yield a mapping sorted by normalized text keys."""
    normalized: list[tuple[str, object]] = []
    seen: set[str] = set()
    for key, item in value.items():
        if not isinstance(key, str):
            msg = "canonical PyTree mappings require string keys"
            raise CanonicalizationError(msg)
        normalized_key = _normalize_text(key)
        if normalized_key in seen:
            msg = "mapping keys collide after Unicode normalization"
            raise CanonicalizationError(msg)
        seen.add(normalized_key)
        normalized.append((normalized_key, item))
    normalized.sort(key=lambda pair: pair[0].encode("utf-8"))
    yield b"M" + _length(len(normalized))
    for key, item in normalized:
        yield _text_record(b"K", key)
        yield from _iter_value_chunks(item, chunk_bytes=chunk_bytes)


def _iter_dataclass_chunks(
    value: object,
    *,
    chunk_bytes: int,
) -> Iterator[CanonicalChunk]:
    """Yield dataclass or Equinox fields in declaration order."""
    fields = dataclasses.fields(value)
    yield _text_record(b"O", _qualified_name(type(value)))
    yield _length(len(fields))
    for field in fields:
        yield _text_record(b"K", field.name)
        yield from _iter_value_chunks(
            getattr(value, field.name),
            chunk_bytes=chunk_bytes,
        )


def _iter_sequence_chunks(
    value: Sequence[object],
    *,
    tag: bytes,
    chunk_bytes: int,
) -> Iterator[CanonicalChunk]:
    """Yield one tagged list or tuple record."""
    yield tag + _length(len(value))
    for item in value:
        yield from _iter_value_chunks(item, chunk_bytes=chunk_bytes)


def _iter_value_chunks(  # noqa: PLR0912
    value: object,
    *,
    chunk_bytes: int,
) -> Iterator[CanonicalChunk]:
    """Yield canonical chunks for one supported value."""
    if value is None:
        yield b"N"
    elif isinstance(value, bool):
        yield b"B\x01" if value else b"B\x00"
    elif isinstance(value, int):
        yield _text_record(b"I", str(value))
    elif isinstance(value, float):
        yield b"F" + bytes.fromhex(_float_bits(value))
    elif isinstance(value, complex):
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            msg = "canonical records reject nonfinite complex values"
            raise CanonicalizationError(msg)
        yield b"C" + struct.pack(">dd", value.real, value.imag)
    elif isinstance(value, str):
        yield _text_record(b"S", value)
    elif isinstance(value, bytes):
        yield b"Y" + _length(len(value)) + value
    elif isinstance(value, Path):
        yield _text_record(b"P", value.as_posix())
    elif isinstance(value, enum.Enum):
        yield _text_record(b"E", _qualified_name(type(value)))
        yield from _iter_value_chunks(value.value, chunk_bytes=chunk_bytes)
    elif _is_array(value):
        yield from _iter_array_chunks(value, chunk_bytes=chunk_bytes)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        yield from _iter_dataclass_chunks(value, chunk_bytes=chunk_bytes)
    elif isinstance(value, tuple):
        yield from _iter_sequence_chunks(
            value,
            tag=b"T",
            chunk_bytes=chunk_bytes,
        )
    elif isinstance(value, list):
        yield from _iter_sequence_chunks(
            value,
            tag=b"L",
            chunk_bytes=chunk_bytes,
        )
    elif isinstance(value, Mapping):
        yield from _iter_mapping_chunks(value, chunk_bytes=chunk_bytes)
    else:
        msg = f"unsupported canonical PyTree value: {type(value)!r}"
        raise CanonicalizationError(msg)


def iter_canonical_pytree_chunks(
    tree: object,
    *,
    chunk_bytes: int = _DEFAULT_ARRAY_CHUNK_BYTES,
) -> Iterator[CanonicalChunk]:
    """Yield canonical carrier bytes in bounded chunks.

    Parameters
    ----------
    tree : object
        A supported scientific carrier or nested PyTree-like value.
    chunk_bytes : int, optional
        Maximum payload bytes yielded for each numerical-array chunk.

    Yields
    ------
    chunk : CanonicalChunk
        Consecutive chunks of the canonical representation.

    Raises
    ------
    ValueError
        If ``chunk_bytes`` is not positive.
    CanonicalizationError
        If the tree contains an unsupported or nonfinite value.
    """
    if chunk_bytes <= 0:
        msg = "chunk_bytes must be positive"
        raise ValueError(msg)
    yield _PYTREE_PREFIX
    yield from _iter_value_chunks(tree, chunk_bytes=chunk_bytes)


def canonical_pytree(tree: object) -> bytes:
    """Return canonical bytes for a supported carrier or PyTree.

    Parameters
    ----------
    tree : object
        Supported nested scientific content. Equinox modules are represented
        through their dataclass fields, including static metadata.

    Returns
    -------
    encoded : bytes
        Complete versioned canonical record.

    Notes
    -----
    Use :func:`iter_canonical_pytree_chunks` when only a streaming checksum is
    required for a large array.
    """
    return b"".join(iter_canonical_pytree_chunks(tree))


__all__: list[str] = [
    "CANONICAL_JSON_VERSION",
    "CANONICAL_PYTREE_VERSION",
    "CanonicalChunk",
    "CanonicalizationError",
    "JsonScalar",
    "JsonValue",
    "canonical_json",
    "canonical_pytree",
    "iter_canonical_pytree_chunks",
]
