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
    Register an exact model identity once.
:func:`get_model`
    Resolve an exact registered model.
:func:`register_transformation`
    Register an exact transformation contract once.
:func:`get_transformation`
    Resolve an exact registered transformation contract.
:func:`list_models`
    Return model specifications in deterministic identity order.
:func:`list_registered_models`
    Return an immutable deterministic snapshot including executors.
:func:`list_transformations`
    Return transformation contracts in deterministic identity order.
:func:`registry_snapshot`
    Return one internally consistent immutable registry snapshot.
:func:`freeze_registry`
    Prevent later registration and return the final immutable snapshot.
:func:`validate_registry`
    Recompute registry structure and consistency checksums.
"""

from __future__ import annotations

import threading
from functools import cache

from beartype import beartype
from beartype.typing import Any, Callable
from jaxtyping import jaxtyped

from diffpes.types import (
    CERTIFICATION_IDENTIFIER_PATTERN,
    CERTIFICATION_SEMVER_PATTERN,
    ForwardModelSpec,
    RegisteredModel,
    RegisteredTransformation,
    RegistryReport,
    RegistrySnapshot,
    TransformationContract,
    make_registered_model,
    make_registered_transformation,
    make_registry_report,
    make_registry_snapshot,
)

from .checksums import checksum_pytree
from .contracts import validate_contract


class _RegistryState:
    """Mutable process-local state hidden behind immutable public snapshots."""

    def __init__(self) -> None:
        self.models: tuple[RegisteredModel, ...] = ()
        self.transformations: tuple[RegisteredTransformation, ...] = ()
        self.frozen = False
        self.lock = threading.RLock()


@cache
def _registry_state() -> _RegistryState:
    """Return the lazily initialized process-local registry state."""
    state: _RegistryState = _RegistryState()
    return state


def _model_key(entry: RegisteredModel) -> tuple[str, str]:
    """Return a sortable exact scientific model identity."""
    key: tuple[str, str] = (entry.spec.model_id, entry.spec.model_version)
    return key


def _transformation_key(
    entry: RegisteredTransformation,
) -> tuple[str, str]:
    """Return a sortable exact transformation identity."""
    contract: TransformationContract = entry.contract
    key: tuple[str, str] = (
        contract.transformation_id,
        contract.transformation_version,
    )
    return key


def _validate_model_spec(spec: ForwardModelSpec) -> None:
    """Validate the registry-facing identity fields of a model spec."""
    if CERTIFICATION_IDENTIFIER_PATTERN.fullmatch(spec.model_id) is None:
        msg: str = "model_id must be a lowercase reverse-DNS-like ID"
        raise ValueError(msg)
    if CERTIFICATION_SEMVER_PATTERN.fullmatch(spec.model_version) is None:
        msg: str = "model_version must be a semantic version"
        raise ValueError(msg)
    if CERTIFICATION_IDENTIFIER_PATTERN.fullmatch(spec.observable_id) is None:
        msg: str = "observable_id must be a lowercase reverse-DNS-like ID"
        raise ValueError(msg)
    if not spec.implementation_ref.strip():
        msg: str = "implementation_ref must be nonblank"
        raise ValueError(msg)


def _ensure_open() -> None:
    """Reject mutation once an application has frozen registration."""
    if _registry_state().frozen:
        msg: str = "the certification registry is frozen"
        raise ValueError(msg)


@jaxtyped(typechecker=beartype)
def register_model(
    spec: ForwardModelSpec,
    executor: Callable[..., Any],
) -> None:
    """Register an exact model identity once.

    The process-local registry uses exact scientific identities and
    deterministic ordering. Freezing prevents later mutation of the selected
    programs.

    :see: :class:`~.test_registry.TestRegisterModel`


    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           state.models = tuple(sorted((*state.models, entry), key=_model_key))

       This expression follows the explicit validation and transformations in
       the function body. It keeps the documented output bound before return.

    Parameters
    ----------
    spec : ForwardModelSpec
        Immutable scientific model specification.
    executor : Callable[..., Any]
        Pure model implementation. Certification later verifies its JAX
        behavior. Registration only requires a callable.

    Raises
    ------
    ValueError
        If the identity is invalid, already registered, or the registry is
        frozen.
    """
    _validate_model_spec(spec)
    checksum: str = checksum_pytree(
        spec,
        record_kind="model-registration",
    )
    entry: RegisteredModel = make_registered_model(
        spec=spec,
        executor=executor,
        registration_checksum=checksum,
    )
    key: tuple[str, str] = _model_key(entry)
    state: _RegistryState = _registry_state()
    with state.lock:
        _ensure_open()
        if any(_model_key(existing) == key for existing in state.models):
            msg: str = f"duplicate model identity: {key[0]}@{key[1]}"
            raise ValueError(msg)
        state.models = tuple(sorted((*state.models, entry), key=_model_key))


@jaxtyped(typechecker=beartype)
def register_transformation(contract: TransformationContract) -> None:
    """Register an exact transformation contract once.

    The process-local registry uses exact scientific identities and
    deterministic ordering. Freezing prevents later mutation of the selected
    programs.

    :see: :class:`~.test_registry.TestRegisterTransformation`


    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           state.transformations = tuple(
                       sorted(
                           (*state.transformations, entry),
                           key=_transformation_key,
                       )
                   )

       This expression follows the explicit validation and transformations in
       the function body. It keeps the documented output bound before return.

    Parameters
    ----------
    contract : TransformationContract
        Validated semantic and information-loss declaration.

    Raises
    ------
    ValueError
        If the contract is invalid, duplicated, or the registry is frozen.
    """
    errors: tuple[str, ...] = validate_contract(contract)
    if errors:
        msg: str = "; ".join(errors)
        raise ValueError(msg)
    checksum: str = checksum_pytree(
        contract,
        record_kind="transformation-registration",
    )
    entry: RegisteredTransformation = make_registered_transformation(
        contract=contract,
        registration_checksum=checksum,
    )
    key: tuple[str, str] = _transformation_key(entry)
    state: _RegistryState = _registry_state()
    with state.lock:
        _ensure_open()
        if any(
            _transformation_key(existing) == key
            for existing in state.transformations
        ):
            msg: str = f"duplicate transformation identity: {key[0]}@{key[1]}"
            raise ValueError(msg)
        state.transformations = tuple(
            sorted(
                (*state.transformations, entry),
                key=_transformation_key,
            )
        )


@jaxtyped(typechecker=beartype)
def get_model(model_id: str, model_version: str) -> RegisteredModel:
    """Resolve an exact registered model.

    The process-local registry uses exact scientific identities and
    deterministic ordering. Freezing prevents later mutation of the selected
    programs.

    :see: :class:`~.test_registry.TestGetModel`


    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           msg: str = f"unknown model identity: {model_id}@{model_version}"

       This expression follows the explicit validation and transformations in
       the function body. It keeps the documented output bound before return.

    Parameters
    ----------
    model_id : str
        Permanent scientific model identifier.
    model_version : str
        Exact semantic model version.

    Returns
    -------
    model : RegisteredModel
        Immutable registered model binding.

    Raises
    ------
    KeyError
        If the exact ID and semantic version are absent.
    """
    entry: Any
    key: tuple[str, str] = (model_id, model_version)
    state: _RegistryState = _registry_state()
    with state.lock:
        for entry in state.models:
            if _model_key(entry) == key:
                model: RegisteredModel = entry
                return model
    msg: str = f"unknown model identity: {model_id}@{model_version}"
    raise KeyError(msg)


@jaxtyped(typechecker=beartype)
def get_transformation(
    transformation_id: str,
    transformation_version: str,
) -> RegisteredTransformation:
    """Resolve an exact registered transformation contract.

    The process-local registry uses exact scientific identities and
    deterministic ordering. Freezing prevents later mutation of the selected
    programs.

    :see: :class:`~.test_registry.TestGetTransformation`


    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           msg: str = (
                   "unknown transformation identity: "
                   f"{transformation_id}@{transformation_version}"
               )

       This expression follows the explicit validation and transformations in
       the function body. It keeps the documented output bound before return.

    Parameters
    ----------
    transformation_id : str
        Permanent transformation identifier.
    transformation_version : str
        Exact semantic transformation version.

    Returns
    -------
    transformation : RegisteredTransformation
        Immutable registered transformation binding.

    Raises
    ------
    KeyError
        If the exact ID and semantic version are absent.
    """
    entry: Any
    key: tuple[str, str] = (transformation_id, transformation_version)
    state: _RegistryState = _registry_state()
    with state.lock:
        for entry in state.transformations:
            if _transformation_key(entry) == key:
                transformation: RegisteredTransformation = entry
                return transformation
    msg: str = (
        "unknown transformation identity: "
        f"{transformation_id}@{transformation_version}"
    )
    raise KeyError(msg)


@jaxtyped(typechecker=beartype)
def list_models() -> tuple[ForwardModelSpec, ...]:
    """Return model specifications in deterministic identity order.

    The process-local registry uses exact scientific identities and
    deterministic ordering. Freezing prevents later mutation of the selected
    programs.

    :see: :class:`~.test_registry.TestListModels`

    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           models: tuple[ForwardModelSpec, ...] = tuple(
                       entry.spec for entry in state.models
                   )

       This expression follows the explicit validation and transformations in
       the function body. It keeps the documented output bound before return.

    Returns
    -------
    models : tuple[ForwardModelSpec, ...]
        Immutable sorted model specifications without executor callables.
    """
    state: _RegistryState = _registry_state()
    with state.lock:
        models: tuple[ForwardModelSpec, ...] = tuple(
            entry.spec for entry in state.models
        )
    return models


@jaxtyped(typechecker=beartype)
def list_registered_models() -> tuple[RegisteredModel, ...]:
    """Return an immutable deterministic snapshot including executors.

    The process-local registry uses exact scientific identities and
    deterministic ordering. Freezing prevents later mutation of the selected
    programs.

    :see: :class:`~.test_registry.TestListRegisteredModels`

    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           models: tuple[RegisteredModel, ...] = state.models

       This expression follows the explicit validation and transformations in
       the function body. It keeps the documented output bound before return.

    Returns
    -------
    models : tuple[RegisteredModel, ...]
        Immutable sorted model bindings including static executor callables.
    """
    state: _RegistryState = _registry_state()
    with state.lock:
        models: tuple[RegisteredModel, ...] = state.models
    return models


@jaxtyped(typechecker=beartype)
def list_transformations() -> tuple[TransformationContract, ...]:
    """Return transformation contracts in deterministic identity order.

    The process-local registry uses exact scientific identities and
    deterministic ordering. Freezing prevents later mutation of the selected
    programs.

    :see: :class:`~.test_registry.TestListTransformations`

    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           transformations: tuple[TransformationContract, ...] = tuple(
                       entry.contract for entry in state.transformations
                   )

       This expression follows the explicit validation and transformations in
       the function body. It keeps the documented output bound before return.

    Returns
    -------
    transformations : tuple[TransformationContract, ...]
        Immutable sorted semantic and information-loss contracts.
    """
    state: _RegistryState = _registry_state()
    with state.lock:
        transformations: tuple[TransformationContract, ...] = tuple(
            entry.contract for entry in state.transformations
        )
    return transformations


def _registry_checksum(
    models: tuple[RegisteredModel, ...],
    transformations: tuple[RegisteredTransformation, ...],
) -> str:
    """Identify sorted entry contents without serializing callables."""
    payload: tuple[tuple[tuple[str, str, str], ...], ...] = (
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
    checksum: str = checksum_pytree(payload, record_kind="registry")
    return checksum


@jaxtyped(typechecker=beartype)
def registry_snapshot() -> RegistrySnapshot:
    """Return one internally consistent immutable registry snapshot.

    The process-local registry uses exact scientific identities and
    deterministic ordering. Freezing prevents later mutation of the selected
    programs.

    :see: :class:`~.test_registry.TestRegistrySnapshot`

    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           snapshot: RegistrySnapshot = make_registry_snapshot(
                   models=models,
                   transformations=transformations,
                   checksum=checksum,
               )

       This expression follows the explicit validation and transformations in
       the function body. It keeps the documented output bound before return.

    Returns
    -------
    snapshot : RegistrySnapshot
        Models, transformations, and their non-security consistency checksum.
    """
    state: _RegistryState = _registry_state()
    with state.lock:
        models: tuple[RegisteredModel, ...] = state.models
        transformations: tuple[RegisteredTransformation, ...] = (
            state.transformations
        )
        checksum: str = _registry_checksum(models, transformations)
    snapshot: RegistrySnapshot = make_registry_snapshot(
        models=models,
        transformations=transformations,
        checksum=checksum,
    )
    return snapshot


@jaxtyped(typechecker=beartype)
def freeze_registry() -> RegistrySnapshot:
    """Prevent later registration and return the final immutable snapshot.

    The process-local registry uses exact scientific identities and
    deterministic ordering. Freezing prevents later mutation of the selected
    programs.

    :see: :class:`~.test_registry.TestFreezeRegistry`

    Returns
    -------
    snapshot : RegistrySnapshot
        Final immutable contents after mutation is disabled.

    Notes
    -----
    Freezing is process-local eager registry control and is never invoked from
    a traced numerical kernel.
    """
    state: _RegistryState = _registry_state()
    with state.lock:
        state.frozen = True
        snapshot: RegistrySnapshot = registry_snapshot()
    return snapshot


@jaxtyped(typechecker=beartype)
def validate_registry() -> RegistryReport:
    """Recompute registry structure and consistency checksums.

    The process-local registry uses exact scientific identities and
    deterministic ordering. Freezing prevents later mutation of the selected
    programs.

    :see: :class:`~.test_registry.TestValidateRegistry`


    Implementation Logic
    --------------------
    1. **Bind the documented output**::

           report: RegistryReport = make_registry_report(
                   valid=not errors,
                   errors=tuple(errors),
                   model_count=len(models),
                   transformation_count=len(transformations),
                   checksum=checksum,
                   frozen=frozen,
               )

       This expression follows the explicit validation and transformations in
       the function body. It keeps the documented output bound before return.

    Returns
    -------
    report : RegistryReport
        Validation errors, entry counts, deterministic checksum, and frozen
        state. The checksum has bookkeeping meaning only.
    """
    entry: Any
    state: _RegistryState = _registry_state()
    with state.lock:
        models: tuple[RegisteredModel, ...] = state.models
        transformations: tuple[RegisteredTransformation, ...] = (
            state.transformations
        )
        frozen: bool = state.frozen
    errors: list[str] = []
    model_keys: tuple[tuple[str, str], ...] = tuple(
        _model_key(entry) for entry in models
    )
    transformation_keys: tuple[tuple[str, str], ...] = tuple(
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
    exc: ValueError
    for entry in models:
        try:
            _validate_model_spec(entry.spec)
        except ValueError as exc:
            errors.append(str(exc))
        expected: str = checksum_pytree(
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
            contract: TransformationContract = entry.contract
            errors.append(
                "transformation registration mismatch: "
                f"{contract.transformation_id}@"
                f"{contract.transformation_version}"
            )
    checksum: str = _registry_checksum(models, transformations)
    report: RegistryReport = make_registry_report(
        valid=not errors,
        errors=tuple(errors),
        model_count=len(models),
        transformation_count=len(transformations),
        checksum=checksum,
        frozen=frozen,
    )
    return report


__all__: list[str] = [
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
