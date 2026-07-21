"""Tests for deterministic immutable certification registries."""

import pytest

from diffpes.certify.contracts import make_transformation_contract
from diffpes.certify.registry import (
    RegistryError,
    get_model,
    get_transformation,
    list_models,
    list_transformations,
    register_model,
    register_transformation,
    registry_snapshot,
    validate_registry,
)
from diffpes.types.certification import make_forward_model_spec


def _model_spec(name: str):
    return make_forward_model_spec(
        model_id=f"org.diffpes.model.registry_test.{name}",
        model_version="1.0.0",
        observable_id="org.diffpes.observable.arpes.intensity",
        implementation_ref=f"tests.registry:{name}",
        differentiable_paths=("parameters.scale",),
    )


def test_models_are_sorted_and_resolved_independent_of_registration_order() -> (
    None
):
    """Expose deterministic immutable snapshots after reverse-order inserts."""
    zulu = _model_spec("zulu")
    alpha = _model_spec("alpha")
    register_model(zulu, lambda value: value)
    register_model(alpha, lambda value: value)

    ids = tuple(spec.model_id for spec in list_models())
    assert ids == tuple(sorted(ids))
    assert get_model(alpha.model_id, alpha.model_version).spec is alpha
    snapshot = registry_snapshot()
    assert snapshot.models == tuple(
        sorted(
            snapshot.models,
            key=lambda item: (item.spec.model_id, item.spec.model_version),
        )
    )
    with pytest.raises(TypeError):
        snapshot.models[0] = snapshot.models[-1]


def test_duplicate_model_identity_is_rejected_even_for_same_spec() -> None:
    """Prevent import order from replacing an existing scientific identity."""
    spec = _model_spec("duplicate")
    register_model(spec, lambda value: value)
    with pytest.raises(RegistryError, match="duplicate model identity"):
        register_model(spec, lambda value: value)


def test_transformation_registry_is_sorted_and_rejects_duplicates() -> None:
    """Apply the same append-only identity rule to semantic contracts."""
    zulu = make_transformation_contract(
        "org.diffpes.transform.registry_test.zulu",
        "1.0.0",
        produces=("zulu-output",),
    )
    alpha = make_transformation_contract(
        "org.diffpes.transform.registry_test.alpha",
        "1.0.0",
        produces=("alpha-output",),
    )
    register_transformation(zulu)
    register_transformation(alpha)
    identities = tuple(
        contract.transformation_id for contract in list_transformations()
    )
    assert identities == tuple(sorted(identities))
    assert (
        get_transformation(alpha.transformation_id, "1.0.0").contract is alpha
    )
    with pytest.raises(RegistryError, match="duplicate transformation"):
        register_transformation(alpha)


def test_registry_report_recomputes_structural_consistency() -> None:
    """Report a stable checksum and successful internal validation."""
    report = validate_registry()
    assert report.valid, report.errors
    assert report.model_count >= 0
    assert report.transformation_count >= 0
    assert report.checksum.startswith("crc32:canonical-1:registry:")


def test_unknown_registry_entries_raise_key_error() -> None:
    """Require an exact scientific identity and version on lookup."""
    with pytest.raises(KeyError, match="unknown model"):
        get_model("org.diffpes.model.registry_test.absent", "1.0.0")
    with pytest.raises(KeyError, match="unknown transformation"):
        get_transformation(
            "org.diffpes.transform.registry_test.absent",
            "1.0.0",
        )


@pytest.mark.parametrize(
    ("model_id", "model_version"),
    [("invalid", "1.0.0"), ("org.diffpes.model.registry_test.bad", "v1")],
)
def test_invalid_model_identity_is_rejected(model_id, model_version) -> None:
    """Enforce permanent reverse-DNS IDs and semantic model versions."""
    spec = make_forward_model_spec(
        model_id=model_id,
        model_version=model_version,
        observable_id="org.diffpes.observable.arpes.intensity",
        implementation_ref="tests.registry:invalid",
    )
    with pytest.raises(RegistryError):
        register_model(spec, lambda value: value)
