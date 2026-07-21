"""Deterministic registries for certified models and transformations.

Extended Summary
----------------
The registry binds permanent scientific identities to model executors and
semantic transformation contracts.  Entries are stored as sorted immutable
tuples, duplicate identities are rejected, and callers receive frozen
snapshots.  Registration order therefore cannot alter lookup, listing, or the
registry consistency checksum.

Registry checksums are non-security bookkeeping values.  They identify an
accidental record mismatch and never establish scientific validity or the
identity of an author.

Routine Listings
----------------
:func:`register_model`
    Bind one immutable model specification to a JAX-compatible executor.
:func:`get_model`
    Resolve an exact model identity and semantic version.
:func:`register_transformation`
    Add one immutable semantic transformation contract.
:func:`validate_registry`
    Recompute all structural and checksum invariants.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from typing import Any

import equinox as eqx

from diffpes.types.certification import ForwardModelSpec

from .checksums import checksum_pytree
from .contracts import TransformationContract, validate_contract

_IDENTIFIER_RE: re.Pattern[str] = re.compile(
    r"^[a-z][a-z0-9]*(?:\.[a-z][a-z0-9_]*)+$"
)
_SEMVER_RE: re.Pattern[str] = re.compile(
    r"^(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


class RegisteredModel(eqx.Module):
    """Frozen binding between a scientific model spec and its executor."""

    spec: ForwardModelSpec
    executor: Callable[..., Any] = eqx.field(static=True)
    registration_checksum: str = eqx.field(static=True)


class RegisteredTransformation(eqx.Module):
    """Frozen registered transformation and its consistency checksum."""

    contract: TransformationContract
    registration_checksum: str = eqx.field(static=True)


class RegistrySnapshot(eqx.Module):
    """Immutable deterministic snapshot of all current registry entries."""

    models: tuple[RegisteredModel, ...]
    transformations: tuple[RegisteredTransformation, ...]
    checksum: str = eqx.field(static=True)


class RegistryReport(eqx.Module):
    """Structural validation result for one registry snapshot."""

    valid: bool = eqx.field(static=True)
    errors: tuple[str, ...] = eqx.field(static=True)
    model_count: int = eqx.field(static=True)
    transformation_count: int = eqx.field(static=True)
    checksum: str = eqx.field(static=True)
    frozen: bool = eqx.field(static=True)


class RegistryError(ValueError):
    """Report an invalid, duplicate, missing, or frozen registry entry."""


_MODELS: tuple[RegisteredModel, ...] = ()
_TRANSFORMATIONS: tuple[RegisteredTransformation, ...] = ()
_FROZEN: bool = False
_LOCK = threading.RLock()


def _model_key(entry: RegisteredModel) -> tuple[str, str]:
    """Return a sortable exact scientific model identity."""
    return entry.spec.model_id, entry.spec.model_version


def _transformation_key(
    entry: RegisteredTransformation,
) -> tuple[str, str]:
    """Return a sortable exact transformation identity."""
    contract = entry.contract
    return contract.transformation_id, contract.transformation_version


def _validate_model_spec(spec: ForwardModelSpec) -> None:
    """Validate the registry-facing identity fields of a model spec."""
    if _IDENTIFIER_RE.fullmatch(spec.model_id) is None:
        msg = "model_id must be a lowercase reverse-DNS-like ID"
        raise RegistryError(msg)
    if _SEMVER_RE.fullmatch(spec.model_version) is None:
        msg = "model_version must be a semantic version"
        raise RegistryError(msg)
    if _IDENTIFIER_RE.fullmatch(spec.observable_id) is None:
        msg = "observable_id must be a lowercase reverse-DNS-like ID"
        raise RegistryError(msg)
    if not spec.implementation_ref.strip():
        msg = "implementation_ref must be nonblank"
        raise RegistryError(msg)


def _ensure_open() -> None:
    """Reject mutation once an application has frozen registration."""
    if _FROZEN:
        msg = "the certification registry is frozen"
        raise RegistryError(msg)


def register_model(
    spec: ForwardModelSpec,
    executor: Callable[..., Any],
) -> None:
    """Register an exact model identity once.

    Parameters
    ----------
    spec : ForwardModelSpec
        Immutable scientific model specification.
    executor : Callable[..., Any]
        Pure model implementation. Numerical certification later verifies its
        actual JAX behavior; registration only requires it to be callable.

    Raises
    ------
    RegistryError
        If the identity is invalid, already registered, or the registry is
        frozen.
    TypeError
        If ``executor`` is not callable.
    """
    global _MODELS  # noqa: PLW0603
    _validate_model_spec(spec)
    if not callable(executor):
        msg = "model executor must be callable"
        raise TypeError(msg)
    checksum = checksum_pytree(spec, record_kind="model-registration")
    entry = RegisteredModel(
        spec=spec,
        executor=executor,
        registration_checksum=checksum,
    )
    key = _model_key(entry)
    with _LOCK:
        _ensure_open()
        if any(_model_key(existing) == key for existing in _MODELS):
            msg = f"duplicate model identity: {key[0]}@{key[1]}"
            raise RegistryError(msg)
        _MODELS = tuple(sorted((*_MODELS, entry), key=_model_key))


def register_transformation(contract: TransformationContract) -> None:
    """Register an exact transformation contract once.

    Parameters
    ----------
    contract : TransformationContract
        Validated semantic and information-loss declaration.

    Raises
    ------
    RegistryError
        If the contract is invalid, duplicated, or the registry is frozen.
    """
    global _TRANSFORMATIONS  # noqa: PLW0603
    errors = validate_contract(contract)
    if errors:
        raise RegistryError("; ".join(errors))
    checksum = checksum_pytree(
        contract,
        record_kind="transformation-registration",
    )
    entry = RegisteredTransformation(
        contract=contract,
        registration_checksum=checksum,
    )
    key = _transformation_key(entry)
    with _LOCK:
        _ensure_open()
        if any(
            _transformation_key(existing) == key
            for existing in _TRANSFORMATIONS
        ):
            msg = f"duplicate transformation identity: {key[0]}@{key[1]}"
            raise RegistryError(msg)
        _TRANSFORMATIONS = tuple(
            sorted((*_TRANSFORMATIONS, entry), key=_transformation_key)
        )


def get_model(model_id: str, model_version: str) -> RegisteredModel:
    """Resolve an exact registered model.

    Raises
    ------
    KeyError
        If the exact ID and semantic version are absent.
    """
    key = (model_id, model_version)
    with _LOCK:
        for entry in _MODELS:
            if _model_key(entry) == key:
                return entry
    msg = f"unknown model identity: {model_id}@{model_version}"
    raise KeyError(msg)


def get_transformation(
    transformation_id: str,
    transformation_version: str,
) -> RegisteredTransformation:
    """Resolve an exact registered transformation contract.

    Raises
    ------
    KeyError
        If the exact ID and semantic version are absent.
    """
    key = (transformation_id, transformation_version)
    with _LOCK:
        for entry in _TRANSFORMATIONS:
            if _transformation_key(entry) == key:
                return entry
    msg = (
        "unknown transformation identity: "
        f"{transformation_id}@{transformation_version}"
    )
    raise KeyError(msg)


def list_models() -> tuple[ForwardModelSpec, ...]:
    """Return model specifications in deterministic identity order."""
    with _LOCK:
        return tuple(entry.spec for entry in _MODELS)


def list_registered_models() -> tuple[RegisteredModel, ...]:
    """Return an immutable deterministic snapshot including executors."""
    with _LOCK:
        return _MODELS


def list_transformations() -> tuple[TransformationContract, ...]:
    """Return transformation contracts in deterministic identity order."""
    with _LOCK:
        return tuple(entry.contract for entry in _TRANSFORMATIONS)


def _registry_checksum(
    models: tuple[RegisteredModel, ...],
    transformations: tuple[RegisteredTransformation, ...],
) -> str:
    """Identify sorted entry contents without serializing callables."""
    payload = (
        tuple(
            (
                entry.spec.model_id,
                entry.spec.model_version,
                entry.registration_checksum,
            )
            for entry in models
        ),
        tuple(
            (
                entry.contract.transformation_id,
                entry.contract.transformation_version,
                entry.registration_checksum,
            )
            for entry in transformations
        ),
    )
    return checksum_pytree(payload, record_kind="registry")


def registry_snapshot() -> RegistrySnapshot:
    """Return one internally consistent immutable registry snapshot."""
    with _LOCK:
        models = _MODELS
        transformations = _TRANSFORMATIONS
        checksum = _registry_checksum(models, transformations)
    return RegistrySnapshot(
        models=models,
        transformations=transformations,
        checksum=checksum,
    )


def freeze_registry() -> RegistrySnapshot:
    """Prevent later registration and return the final immutable snapshot."""
    global _FROZEN  # noqa: PLW0603
    with _LOCK:
        _FROZEN = True
        return registry_snapshot()


def validate_registry() -> RegistryReport:
    """Recompute registry structure and consistency checksums.

    Returns
    -------
    report : RegistryReport
        Validation errors, entry counts, deterministic checksum, and frozen
        state. The checksum has bookkeeping meaning only.
    """
    with _LOCK:
        models = _MODELS
        transformations = _TRANSFORMATIONS
        frozen = _FROZEN
    errors: list[str] = []
    model_keys = tuple(_model_key(entry) for entry in models)
    transformation_keys = tuple(
        _transformation_key(entry) for entry in transformations
    )
    if model_keys != tuple(sorted(model_keys)):
        errors.append("model entries are not deterministically sorted")
    if len(set(model_keys)) != len(model_keys):
        errors.append("model identities are not unique")
    if transformation_keys != tuple(sorted(transformation_keys)):
        errors.append(
            "transformation entries are not deterministically sorted"
        )
    if len(set(transformation_keys)) != len(transformation_keys):
        errors.append("transformation identities are not unique")
    for entry in models:
        try:
            _validate_model_spec(entry.spec)
        except RegistryError as exc:
            errors.append(str(exc))
        expected = checksum_pytree(
            entry.spec,
            record_kind="model-registration",
        )
        if expected != entry.registration_checksum:
            errors.append(
                f"model registration mismatch: "
                f"{entry.spec.model_id}@{entry.spec.model_version}"
            )
    for entry in transformations:
        errors.extend(validate_contract(entry.contract))
        expected = checksum_pytree(
            entry.contract,
            record_kind="transformation-registration",
        )
        if expected != entry.registration_checksum:
            contract = entry.contract
            errors.append(
                "transformation registration mismatch: "
                f"{contract.transformation_id}@"
                f"{contract.transformation_version}"
            )
    checksum = _registry_checksum(models, transformations)
    return RegistryReport(
        valid=not errors,
        errors=tuple(errors),
        model_count=len(models),
        transformation_count=len(transformations),
        checksum=checksum,
        frozen=frozen,
    )


__all__: list[str] = [
    "RegisteredModel",
    "RegisteredTransformation",
    "RegistryError",
    "RegistryReport",
    "RegistrySnapshot",
    "freeze_registry",
    "get_model",
    "get_transformation",
    "list_models",
    "list_registered_models",
    "list_transformations",
    "register_model",
    "register_transformation",
    "registry_snapshot",
    "validate_registry",
]
