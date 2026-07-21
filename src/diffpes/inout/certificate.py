"""Portable persistence for forward-model certificates.

Extended Summary
----------------
Stores ``ForwardCertificate`` PyTrees as deterministic, transparent JSON and
embeds those exact bytes in HDF5 result files. Numerical leaves are represented
losslessly with dtype, shape, byte order, and base64-encoded canonical bytes.
The storage checksum is CRC32 bookkeeping for accidental mismatch detection;
it is not authentication and contributes no scientific certification claim.

Routine Listings
----------------
:obj:`CERTIFICATE_FORMAT`
    Stable portable forward-certificate format identifier.
:obj:`CERTIFICATE_SCHEMA_MAJOR`
    Supported portable certificate schema major version.
:obj:`CERTIFICATE_SCHEMA_MINOR`
    Current portable certificate schema minor version.
:class:`CertificateFormatError`
    Report malformed, inconsistent, or unsupported certificate records.
:func:`attach_certificate_h5`
    Atomically attach canonical certificate bytes to an HDF5 file.
:func:`load_certificate_h5`
    Load and validate an HDF5-embedded certificate.
:func:`load_certificate_json`
    Load and validate a canonical JSON certificate.
:func:`save_certificate_json`
    Atomically write a canonical JSON certificate.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import os
import re
import shutil
import tempfile
import unicodedata
import zlib
from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import h5py
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Union
from numpy import ndarray as NDArray  # noqa: N812

from diffpes.types import (
    ArtifactRef,
    CertificationClaim,
    ConventionRef,
    DependencyMap,
    DerivativeEvidence,
    DomainPredicate,
    DomainResult,
    EvidenceRef,
    ExecutionManifest,
    ForwardCertificate,
    ForwardModelSpec,
    InformationSpectrum,
    PolicyReport,
    SensitivityMap,
    TransformationRecord,
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

CERTIFICATE_FORMAT: str = "org.diffpes.forward-certificate"
CERTIFICATE_SCHEMA_MAJOR: int = 1
CERTIFICATE_SCHEMA_MINOR: int = 0
H5_CERTIFICATE_GROUP: str = "_diffpes_certificates"

_ARRAY_KINDS: frozenset[str] = frozenset({"b", "i", "u", "f", "c"})
_DOCUMENT_KEYS: frozenset[str] = frozenset(
    {
        "format",
        "schema_version",
        "certificate",
        "extensions",
        "consistency_checksum",
    }
)
_REQUIRED_DOCUMENT_KEYS: frozenset[str] = _DOCUMENT_KEYS
_SCHEMA_RE: re.Pattern[str] = re.compile(
    r"^(?P<major>0|[1-9][0-9]*)"
    r"(?:\.(?P<minor>0|[1-9][0-9]*))?"
    r"(?:\.(?:0|[1-9][0-9]*))?$"
)

_MODULE_FACTORIES: dict[type[Any], Callable[..., Any]] = {
    ArtifactRef: make_artifact_ref,
    CertificationClaim: make_certification_claim,
    ConventionRef: make_convention_ref,
    DependencyMap: make_dependency_map,
    DerivativeEvidence: make_derivative_evidence,
    DomainPredicate: make_domain_predicate,
    DomainResult: make_domain_result,
    EvidenceRef: make_evidence_ref,
    ExecutionManifest: make_execution_manifest,
    ForwardCertificate: make_forward_certificate,
    ForwardModelSpec: make_forward_model_spec,
    InformationSpectrum: make_information_spectrum,
    PolicyReport: make_policy_report,
    SensitivityMap: make_sensitivity_map,
    TransformationRecord: make_transformation_record,
}
_MODULE_TYPES: dict[str, type[Any]] = {
    module_type.__name__: module_type for module_type in _MODULE_FACTORIES
}


class CertificateFormatError(ValueError):
    """Report a malformed, inconsistent, or unsupported certificate record."""


def _normalize_text(value: str) -> str:
    """Return NFC-normalized certificate text."""
    return unicodedata.normalize("NFC", value)


def _normalize_json_value(value: Any) -> Any:
    """Normalize an extension value and reject non-JSON/nonfinite data."""
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            msg = "certificate JSON rejects NaN and infinite values"
            raise CertificateFormatError(msg)
        return value
    if isinstance(value, str):
        return _normalize_text(value)
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                msg = "certificate JSON object keys must be strings"
                raise CertificateFormatError(msg)
            normalized_key: str = _normalize_text(key)
            if normalized_key in normalized:
                msg = "certificate keys collide after Unicode normalization"
                raise CertificateFormatError(msg)
            normalized[normalized_key] = _normalize_json_value(item)
        return normalized
    msg = f"unsupported certificate JSON value: {type(value)!r}"
    raise CertificateFormatError(msg)


def _json_bytes(value: Mapping[str, Any], *, newline: bool) -> bytes:
    """Encode a normalized mapping as deterministic UTF-8 JSON."""
    normalized: Any = _normalize_json_value(value)
    encoded: bytes = json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if newline:
        encoded += b"\n"
    return encoded


def _storage_checksum(document: Mapping[str, Any]) -> str:
    """Return the non-security CRC32 of a document without its checksum."""
    payload: dict[str, Any] = dict(document)
    payload.pop("consistency_checksum", None)
    value: int = zlib.crc32(_json_bytes(payload, newline=False))
    return f"crc32:certificate-json-v1:{value & 0xFFFFFFFF:08x}"


def _encode_array(value: object) -> dict[str, Any]:
    """Encode one concrete numerical leaf without decimal conversion."""
    try:
        array: NDArray = np.asarray(value)
    except Exception as exc:
        msg = "certificate persistence requires concrete, non-traced arrays"
        raise CertificateFormatError(msg) from exc
    if array.dtype.kind not in _ARRAY_KINDS:
        msg = f"unsupported certificate array dtype: {array.dtype}"
        raise CertificateFormatError(msg)
    if array.dtype.kind in {"f", "c"} and not bool(np.all(np.isfinite(array))):
        msg = "certificate persistence rejects nonfinite numerical leaves"
        raise CertificateFormatError(msg)
    canonical_dtype: np.dtype[Any] = array.dtype.newbyteorder("<")
    canonical: NDArray = np.asarray(
        array,
        dtype=canonical_dtype,
        order="C",
    )
    payload: str = base64.b64encode(canonical.tobytes(order="C")).decode(
        "ascii"
    )
    result: dict[str, Any] = {
        "kind": "array",
        "dtype": canonical.dtype.str,
        "shape": list(canonical.shape),
        "byte_order": "little",
        "order": "C",
        "encoding": "base64",
        "data": payload,
    }
    return result


def _is_array(value: object) -> bool:
    """Return whether a value exposes a concrete numerical array protocol."""
    if isinstance(value, np.ndarray | np.generic):
        return True
    attributes: tuple[str, ...] = ("__array__", "dtype", "shape")
    return all(hasattr(value, attr) for attr in attributes)


def _encode_value(  # noqa: PLR0911
    value: object,
    *,
    root: bool = False,
) -> Any:
    """Encode one supported carrier field into the transparent schema."""
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            msg = "certificate persistence rejects nonfinite scalars"
            raise CertificateFormatError(msg)
        return value
    if isinstance(value, str):
        return _normalize_text(value)
    if _is_array(value):
        return _encode_array(value)
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [_encode_value(item) for item in value],
        }
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [_encode_value(item) for item in value],
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            msg = "certificate mappings require string keys"
            raise CertificateFormatError(msg)
        return {
            "kind": "mapping",
            "items": {key: _encode_value(item) for key, item in value.items()},
        }
    value_type: type[Any] = type(value)
    if value_type in _MODULE_FACTORIES and is_dataclass(value):
        encoded_fields: dict[str, Any] = {}
        for field in fields(value):
            if root and field.name == "extensions_json":
                continue
            encoded_fields[field.name] = _encode_value(
                getattr(value, field.name)
            )
        return {
            "kind": "module",
            "type": value_type.__name__,
            "fields": encoded_fields,
        }
    msg = f"unsupported certificate field value: {type(value)!r}"
    raise CertificateFormatError(msg)


def _parse_extensions(certificate: ForwardCertificate) -> dict[str, Any]:
    """Parse and normalize the certificate extension object."""
    try:
        value: Any = json.loads(
            certificate.extensions_json,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, TypeError) as exc:
        msg = "certificate extensions_json must encode a JSON object"
        raise CertificateFormatError(msg) from exc
    if not isinstance(value, dict):
        msg = "certificate extensions_json must encode a JSON object"
        raise CertificateFormatError(msg)
    normalized: Any = _normalize_json_value(value)
    return normalized


def _certificate_document(certificate: ForwardCertificate) -> dict[str, Any]:
    """Build the complete portable document for one certificate."""
    schema_version: str = certificate.manifest.schema_version
    _parse_schema_version(schema_version)
    document: dict[str, Any] = {
        "format": CERTIFICATE_FORMAT,
        "schema_version": schema_version,
        "certificate": _encode_value(certificate, root=True),
        "extensions": _parse_extensions(certificate),
    }
    document["consistency_checksum"] = _storage_checksum(document)
    return document


def _reject_json_constant(value: str) -> None:
    """Reject JSON's non-standard NaN and Infinity tokens."""
    msg = f"certificate JSON contains invalid constant {value!r}"
    raise CertificateFormatError(msg)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object while rejecting duplicate names."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            msg = f"duplicate certificate JSON key: {key!r}"
            raise CertificateFormatError(msg)
        result[key] = value
    return result


def _read_document(data: bytes) -> dict[str, Any]:
    """Parse, structurally validate, and checksum one JSON document."""
    try:
        decoded: Any = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        msg = "certificate is not valid UTF-8 JSON"
        raise CertificateFormatError(msg) from exc
    if not isinstance(decoded, dict):
        msg = "certificate document must be a JSON object"
        raise CertificateFormatError(msg)
    missing: frozenset[str] = _REQUIRED_DOCUMENT_KEYS - decoded.keys()
    if missing:
        msg = f"certificate document is missing fields: {sorted(missing)}"
        raise CertificateFormatError(msg)
    if decoded["format"] != CERTIFICATE_FORMAT:
        msg = f"unsupported certificate format: {decoded['format']!r}"
        raise CertificateFormatError(msg)
    _, minor = _parse_schema_version(decoded["schema_version"])
    extra: frozenset[str] = decoded.keys() - _DOCUMENT_KEYS
    if extra and minor <= CERTIFICATE_SCHEMA_MINOR:
        msg = f"unknown current-schema certificate fields: {sorted(extra)}"
        raise CertificateFormatError(msg)
    expected_checksum: str = _storage_checksum(decoded)
    if decoded["consistency_checksum"] != expected_checksum:
        msg = "certificate consistency checksum mismatch"
        raise CertificateFormatError(msg)
    extensions: Any = decoded["extensions"]
    if not isinstance(extensions, dict):
        msg = "certificate extensions must be a JSON object"
        raise CertificateFormatError(msg)
    normalized: Any = _normalize_json_value(decoded)
    return normalized


def _parse_schema_version(value: object) -> tuple[int, int]:
    """Parse a schema version and reject unsupported major versions."""
    if not isinstance(value, str):
        msg = "certificate schema_version must be a string"
        raise CertificateFormatError(msg)
    match: re.Match[str] | None = _SCHEMA_RE.fullmatch(value)
    if match is None:
        msg = f"invalid certificate schema version: {value!r}"
        raise CertificateFormatError(msg)
    major: int = int(match.group("major"))
    minor_text: str | None = match.group("minor")
    minor: int = 0 if minor_text is None else int(minor_text)
    if major != CERTIFICATE_SCHEMA_MAJOR:
        msg = (
            f"unsupported certificate schema major {major}; "
            f"reader supports {CERTIFICATE_SCHEMA_MAJOR}.x"
        )
        raise CertificateFormatError(msg)
    return major, minor


def _decode_array(node: Mapping[str, Any]) -> Any:
    """Decode and validate one losslessly represented numerical leaf."""
    required: frozenset[str] = frozenset(
        {"kind", "dtype", "shape", "byte_order", "order", "encoding", "data"}
    )
    if node.keys() != required:
        msg = "array record has missing or unknown fields"
        raise CertificateFormatError(msg)
    if node["byte_order"] != "little" or node["order"] != "C":
        msg = "certificate arrays require little-endian C-order storage"
        raise CertificateFormatError(msg)
    if node["encoding"] != "base64":
        msg = "unsupported certificate array encoding"
        raise CertificateFormatError(msg)
    try:
        dtype: np.dtype[Any] = np.dtype(node["dtype"])
    except TypeError as exc:
        msg = "invalid certificate array dtype"
        raise CertificateFormatError(msg) from exc
    if dtype.kind not in _ARRAY_KINDS or dtype.byteorder == ">":
        msg = f"unsupported certificate array dtype: {dtype}"
        raise CertificateFormatError(msg)
    shape_value: Any = node["shape"]
    if not isinstance(shape_value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0
        for item in shape_value
    ):
        msg = "certificate array shape must contain nonnegative integers"
        raise CertificateFormatError(msg)
    shape: tuple[int, ...] = tuple(shape_value)
    data_value: Any = node["data"]
    if not isinstance(data_value, str):
        msg = "certificate array data must be base64 text"
        raise CertificateFormatError(msg)
    try:
        payload: bytes = base64.b64decode(data_value, validate=True)
    except (binascii.Error, ValueError) as exc:
        msg = "certificate array contains invalid base64 data"
        raise CertificateFormatError(msg) from exc
    count: int = math.prod(shape)
    expected_bytes: int = count * dtype.itemsize
    if len(payload) != expected_bytes:
        msg = "certificate array byte length does not match dtype and shape"
        raise CertificateFormatError(msg)
    array: NDArray = np.frombuffer(payload, dtype=dtype).reshape(shape)
    if dtype.kind in {"f", "c"} and not bool(np.all(np.isfinite(array))):
        msg = "certificate persistence rejects nonfinite numerical leaves"
        raise CertificateFormatError(msg)
    result: Any = jnp.asarray(array.copy())
    return result


def _decode_value(
    node: Any,
    *,
    schema_minor: int,
    extensions: dict[str, Any],
    path: str,
) -> Any:
    """Decode one schema node through the registered carrier factories."""
    if node is None or isinstance(node, bool | int | float | str):
        return node
    if not isinstance(node, dict):
        msg = f"invalid certificate node at {path}"
        raise CertificateFormatError(msg)
    kind: Any = node.get("kind")
    if kind == "array":
        return _decode_array(node)
    if kind in {"tuple", "list"}:
        if node.keys() != {"kind", "items"} or not isinstance(
            node["items"], list
        ):
            msg = f"invalid {kind} record at {path}"
            raise CertificateFormatError(msg)
        values: list[Any] = [
            _decode_value(
                item,
                schema_minor=schema_minor,
                extensions=extensions,
                path=f"{path}[{index}]",
            )
            for index, item in enumerate(node["items"])
        ]
        return tuple(values) if kind == "tuple" else values
    if kind == "mapping":
        if node.keys() != {"kind", "items"} or not isinstance(
            node["items"], dict
        ):
            msg = f"invalid mapping record at {path}"
            raise CertificateFormatError(msg)
        return {
            key: _decode_value(
                item,
                schema_minor=schema_minor,
                extensions=extensions,
                path=f"{path}.{key}",
            )
            for key, item in node["items"].items()
        }
    if kind != "module":
        msg = f"unknown certificate node kind at {path}: {kind!r}"
        raise CertificateFormatError(msg)
    return _decode_module(
        node,
        schema_minor=schema_minor,
        extensions=extensions,
        path=path,
    )


def _record_unknown_fields(
    extensions: dict[str, Any],
    path: str,
    values: dict[str, Any],
) -> None:
    """Retain fields introduced by a newer compatible minor schema."""
    key: str = "org.diffpes.persistence.unknown_module_fields"
    existing: Any = extensions.setdefault(key, {})
    if not isinstance(existing, dict):
        msg = f"reserved extension key {key!r} must contain an object"
        raise CertificateFormatError(msg)
    existing[path] = values


def _decode_module(
    node: Mapping[str, Any],
    *,
    schema_minor: int,
    extensions: dict[str, Any],
    path: str,
) -> Any:
    """Decode one whitelisted Equinox carrier via its validation factory."""
    if node.keys() != {"kind", "type", "fields"}:
        msg = f"invalid module record at {path}"
        raise CertificateFormatError(msg)
    type_name: Any = node["type"]
    encoded_fields: Any = node["fields"]
    if not isinstance(type_name, str) or type_name not in _MODULE_TYPES:
        msg = f"unsupported certificate module type at {path}: {type_name!r}"
        raise CertificateFormatError(msg)
    if not isinstance(encoded_fields, dict):
        msg = f"module fields must be an object at {path}"
        raise CertificateFormatError(msg)
    module_type: type[Any] = _MODULE_TYPES[type_name]
    expected_names: set[str] = {field.name for field in fields(module_type)}
    if module_type is ForwardCertificate:
        expected_encoded: set[str] = expected_names - {"extensions_json"}
    else:
        expected_encoded = expected_names
    provided_names: set[str] = set(encoded_fields)
    missing: set[str] = expected_encoded - provided_names
    if missing:
        msg = f"module {type_name} is missing fields: {sorted(missing)}"
        raise CertificateFormatError(msg)
    unknown: set[str] = provided_names - expected_encoded
    if unknown and schema_minor <= CERTIFICATE_SCHEMA_MINOR:
        msg = f"module {type_name} has unknown fields: {sorted(unknown)}"
        raise CertificateFormatError(msg)
    if unknown:
        _record_unknown_fields(
            extensions,
            path,
            {name: encoded_fields[name] for name in sorted(unknown)},
        )
    values: dict[str, Any] = {
        name: _decode_value(
            encoded_fields[name],
            schema_minor=schema_minor,
            extensions=extensions,
            path=f"{path}.{name}",
        )
        for name in expected_encoded
    }
    if module_type is ForwardCertificate:
        values["extensions_json"] = json.dumps(
            _normalize_json_value(extensions),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    factory: Callable[..., Any] = _MODULE_FACTORIES[module_type]
    try:
        result: Any = factory(**values)
    except (TypeError, ValueError) as exc:
        msg = f"invalid {type_name} data at {path}: {exc}"
        raise CertificateFormatError(msg) from exc
    return result


def _certificate_from_document(document: dict[str, Any]) -> ForwardCertificate:
    """Construct a validated certificate from a parsed document."""
    _, minor = _parse_schema_version(document["schema_version"])
    extensions: dict[str, Any] = dict(document["extensions"])
    extra: dict[str, Any] = {
        key: value
        for key, value in document.items()
        if key not in _DOCUMENT_KEYS
    }
    if extra:
        extensions["org.diffpes.persistence.unknown_document_fields"] = extra
    decoded: Any = _decode_value(
        document["certificate"],
        schema_minor=minor,
        extensions=extensions,
        path="certificate",
    )
    if not isinstance(decoded, ForwardCertificate):
        msg = "certificate document root is not a ForwardCertificate"
        raise CertificateFormatError(msg)
    if decoded.manifest.schema_version != document["schema_version"]:
        msg = "document and manifest schema versions disagree"
        raise CertificateFormatError(msg)
    return decoded


def _atomic_write(path: Path, data: bytes) -> None:
    """Write bytes through a same-directory temporary and atomic replace."""
    path.parent.mkdir(parents=False, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary: Path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@beartype
def save_certificate_json(
    certificate: ForwardCertificate,
    path: Union[str, Path],
) -> None:
    """Atomically save a forward certificate as canonical JSON.

    Parameters
    ----------
    certificate : ForwardCertificate
        Validated scientific-assurance record to persist.
    path : str or Path
        Destination JSON path. Its parent directory must already exist.

    Raises
    ------
    CertificateFormatError
        If a value cannot be represented in the current portable schema.
    """
    document: dict[str, Any] = _certificate_document(certificate)
    data: bytes = _json_bytes(document, newline=True)
    _atomic_write(Path(path), data)


@beartype
def load_certificate_json(path: Union[str, Path]) -> ForwardCertificate:
    """Load a validated forward certificate from canonical JSON.

    Parameters
    ----------
    path : str or Path
        Source JSON path.

    Returns
    -------
    certificate : ForwardCertificate
        Reconstructed certificate with numerical leaves restored as JAX
        arrays.

    Raises
    ------
    CertificateFormatError
        If the record is malformed, inconsistent, or uses an unknown major
        schema.
    """
    data: bytes = Path(path).read_bytes()
    document: dict[str, Any] = _read_document(data)
    certificate: ForwardCertificate = _certificate_from_document(document)
    return certificate


def _validate_h5_name(name: str) -> None:
    """Reject ambiguous or path-like HDF5 certificate names."""
    if not name or name in {".", ".."} or "/" in name or "\x00" in name:
        msg = "HDF5 certificate name must be one nonblank group component"
        raise ValueError(msg)


def _write_h5_record(
    path: Path,
    name: str,
    data: bytes,
    certificate: ForwardCertificate,
) -> None:
    """Write one exact JSON record and its convenience index attributes."""
    document: dict[str, Any] = _read_document(data)
    with h5py.File(path, "a") as file:
        root: h5py.Group = file.require_group(H5_CERTIFICATE_GROUP)
        if name in root:
            del root[name]
        group: h5py.Group = root.create_group(name)
        group.create_dataset(
            "canonical_json",
            data=np.frombuffer(data, dtype=np.uint8),
            compression="gzip",
            shuffle=True,
            fletcher32=True,
        )
        group.attrs["format"] = CERTIFICATE_FORMAT
        group.attrs["schema_version"] = certificate.manifest.schema_version
        group.attrs["model_id"] = certificate.model.model_id
        group.attrs["model_version"] = certificate.model.model_version
        group.attrs["policy_id"] = certificate.policy_id
        group.attrs["execution_id"] = certificate.manifest.execution_id
        group.attrs["consistency_checksum"] = document["consistency_checksum"]
        file.flush()


@beartype
def attach_certificate_h5(
    path: Union[str, Path],
    name: str,
    certificate: ForwardCertificate,
) -> None:
    """Atomically attach a certificate to an HDF5 result file.

    The complete file is updated through a same-directory temporary and
    ``os.replace``. Existing numerical result groups are preserved.

    Parameters
    ----------
    path : str or Path
        Existing HDF5 result path, or a path for a new HDF5 container.
    name : str
        Single-component result name used under the certificate index group.
    certificate : ForwardCertificate
        Certificate associated with the named result.
    """
    _validate_h5_name(name)
    destination: Path = Path(path)
    destination.parent.mkdir(parents=False, exist_ok=True)
    document: dict[str, Any] = _certificate_document(certificate)
    data: bytes = _json_bytes(document, newline=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary: Path = Path(temporary_name)
    try:
        if destination.exists():
            shutil.copy2(destination, temporary)
        _write_h5_record(temporary, name, data, certificate)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@beartype
def load_certificate_h5(
    path: Union[str, Path],
    name: str,
) -> ForwardCertificate:
    """Load a certificate embedded in an HDF5 result file.

    Parameters
    ----------
    path : str or Path
        HDF5 result path.
    name : str
        Certificate name supplied to :func:`attach_certificate_h5`.

    Returns
    -------
    certificate : ForwardCertificate
        Reconstructed and validated certificate.

    Raises
    ------
    KeyError
        If the certificate group or named record is absent.
    CertificateFormatError
        If the exact JSON bytes or HDF5 convenience index are inconsistent.
    """
    _validate_h5_name(name)
    source: Path = Path(path)
    with h5py.File(source, "r") as file:
        if H5_CERTIFICATE_GROUP not in file:
            msg = f"No certificates found in {source}"
            raise KeyError(msg)
        root: h5py.Group = file[H5_CERTIFICATE_GROUP]
        if name not in root:
            msg = f"Certificate '{name}' not found in {source}"
            raise KeyError(msg)
        group: h5py.Group = root[name]
        if "canonical_json" not in group:
            msg = "HDF5 certificate record has no canonical_json dataset"
            raise CertificateFormatError(msg)
        stored: NDArray = np.asarray(group["canonical_json"][()])
        if stored.dtype != np.dtype(np.uint8) or stored.ndim != 1:
            msg = "HDF5 canonical_json dataset must be one-dimensional uint8"
            raise CertificateFormatError(msg)
        data: bytes = stored.tobytes()
        document: dict[str, Any] = _read_document(data)
        certificate: ForwardCertificate = _certificate_from_document(document)
        expected_attrs: dict[str, str] = {
            "format": CERTIFICATE_FORMAT,
            "schema_version": certificate.manifest.schema_version,
            "model_id": certificate.model.model_id,
            "model_version": certificate.model.model_version,
            "policy_id": certificate.policy_id,
            "execution_id": certificate.manifest.execution_id,
            "consistency_checksum": document["consistency_checksum"],
        }
        for key, expected in expected_attrs.items():
            actual: Any = group.attrs.get(key)
            if actual is None or str(actual) != expected:
                msg = f"HDF5 certificate index mismatch for {key!r}"
                raise CertificateFormatError(msg)
    return certificate


__all__: list[str] = [
    "CERTIFICATE_FORMAT",
    "CERTIFICATE_SCHEMA_MAJOR",
    "CERTIFICATE_SCHEMA_MINOR",
    "CertificateFormatError",
    "attach_certificate_h5",
    "load_certificate_h5",
    "load_certificate_json",
    "save_certificate_json",
]
