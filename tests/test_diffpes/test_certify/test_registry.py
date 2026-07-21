"""Validate deterministic immutable certification registries.

The tests cover public behavior, differentiability, validation, and stable
scientific identity in the supported certification regime.
"""

import subprocess
import sys

import pytest
from beartype.typing import Any

from diffpes.certify import (
    freeze_registry,
    get_model,
    get_transformation,
    list_models,
    list_registered_models,
    list_transformations,
    register_model,
    register_transformation,
    registry_snapshot,
    validate_registry,
)
from diffpes.types import make_forward_model_spec, make_transformation_contract


def _model_spec(name: str) -> Any:
    return make_forward_model_spec(
        model_id=f"org.diffpes.model.registry_test.{name}",
        model_version="1.0.0",
        observable_id="org.diffpes.observable.arpes.intensity",
        implementation_ref=f"tests.registry:{name}",
        differentiable_paths=("parameters.scale",),
    )


class TestValidateRegistry:
    """Verify :func:`~diffpes.certify.validate_registry`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.validate_registry`
    """

    def test_snapshot_is_structurally_valid(self) -> None:
        """Verify a snapshot satisfies ordering and checksum invariants.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        Recomputes validation against the current process-local registry.
        """
        assert validate_registry().valid

    def test_registry_report_recomputes_structural_consistency(self) -> None:
        """Report a stable checksum and successful internal validation.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        report: Any
        report = validate_registry()
        assert report.valid, report.errors
        assert report.model_count >= 0
        assert report.transformation_count >= 0
        assert report.checksum.startswith("crc32:canonical-1:registry:")


class TestRegistrySnapshot:
    """Verify :func:`~diffpes.certify.registry_snapshot`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.registry_snapshot`
    """

    def test_models_are_sorted_and_resolved_independent_of_registration_order(
        self,
    ) -> None:
        """Expose deterministic immutable snapshots after reverse-order inserts.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        zulu: Any
        alpha: Any
        ids: Any
        snapshot: Any
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


class TestRegisterModel:
    """Verify :func:`~diffpes.certify.register_model`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.register_model`
    """

    def test_duplicate_model_identity_is_rejected_even_for_same_spec(
        self,
    ) -> None:
        """Prevent import order from replacing an existing scientific identity.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        spec: Any
        spec = _model_spec("duplicate")
        register_model(spec, lambda value: value)
        with pytest.raises(ValueError, match="duplicate model identity"):
            register_model(spec, lambda value: value)

    @pytest.mark.parametrize(
        ("model_id", "model_version"),
        [("invalid", "1.0.0"), ("org.diffpes.model.registry_test.bad", "v1")],
    )
    def test_invalid_model_identity_is_rejected(
        self, model_id: Any, model_version: Any
    ) -> None:
        """Enforce permanent reverse-DNS IDs and semantic model versions.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        spec: Any
        spec = make_forward_model_spec(
            model_id=model_id,
            model_version=model_version,
            observable_id="org.diffpes.observable.arpes.intensity",
            implementation_ref="tests.registry:invalid",
        )
        with pytest.raises(ValueError):
            register_model(spec, lambda value: value)


class TestRegisterTransformation:
    """Verify :func:`~diffpes.certify.register_transformation`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.register_transformation`
    """

    def test_transformation_registry_is_sorted_and_rejects_duplicates(
        self,
    ) -> None:
        """Apply the same append-only identity rule to semantic contracts.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        zulu: Any
        alpha: Any
        identities: Any
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
            get_transformation(alpha.transformation_id, "1.0.0").contract
            is alpha
        )
        with pytest.raises(ValueError, match="duplicate transformation"):
            register_transformation(alpha)


class TestGetModel:
    """Verify :func:`~diffpes.certify.get_model`.

    The cases cover the public behavior in the supported certification regime.

    :see: :func:`~diffpes.certify.get_model`
    """

    def test_unknown_registry_entries_raise_key_error(self) -> None:
        """Require an exact scientific identity and version on lookup.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        with pytest.raises(KeyError, match="unknown model"):
            get_model("org.diffpes.model.registry_test.absent", "1.0.0")
        with pytest.raises(KeyError, match="unknown transformation"):
            get_transformation(
                "org.diffpes.transform.registry_test.absent",
                "1.0.0",
            )


class TestGetTransformation:
    """Verify :func:`~diffpes.certify.get_transformation`.

    The cases cover exact lookup of a registered semantic contract.

    :see: :func:`~diffpes.certify.get_transformation`
    """

    def test_registered_contract_is_resolved_exactly(self) -> None:
        """Resolve one transformation by its permanent identity and version.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        contract: Any
        resolved: Any
        contract = make_transformation_contract(
            "org.diffpes.transform.registry_test.lookup",
            "1.0.0",
            produces=("lookup-output",),
        )
        register_transformation(contract)
        resolved = get_transformation(contract.transformation_id, "1.0.0")
        assert resolved.contract is contract


class TestListModels:
    """Verify :func:`~diffpes.certify.list_models`.

    The cases cover stable ordering without exposing executor callables.

    :see: :func:`~diffpes.certify.list_models`
    """

    def test_model_specs_are_sorted(self) -> None:
        """Return model specifications in permanent identity order.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        models: Any
        models = list_models()
        identities: tuple[tuple[str, str], ...] = tuple(
            (model.model_id, model.model_version) for model in models
        )
        assert identities == tuple(sorted(identities))


class TestListRegisteredModels:
    """Verify :func:`~diffpes.certify.list_registered_models`.

    The cases cover stable ordering of model specifications and executors.

    :see: :func:`~diffpes.certify.list_registered_models`
    """

    def test_registered_bindings_are_sorted(self) -> None:
        """Return complete registered bindings in permanent identity order.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        models: Any
        models = list_registered_models()
        identities: tuple[tuple[str, str], ...] = tuple(
            (model.spec.model_id, model.spec.model_version) for model in models
        )
        assert identities == tuple(sorted(identities))


class TestListTransformations:
    """Verify :func:`~diffpes.certify.list_transformations`.

    The cases cover stable ordering of transformation contracts.

    :see: :func:`~diffpes.certify.list_transformations`
    """

    def test_contracts_are_sorted(self) -> None:
        """Return transformation contracts in permanent identity order.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        contracts: Any
        contracts = list_transformations()
        identities: tuple[tuple[str, str], ...] = tuple(
            (item.transformation_id, item.transformation_version)
            for item in contracts
        )
        assert identities == tuple(sorted(identities))


class TestFreezeRegistry:
    """Verify :func:`~diffpes.certify.freeze_registry`.

    The case isolates the process-global freeze operation in a child process.

    :see: :func:`~diffpes.certify.freeze_registry`
    """

    def test_freeze_rejects_later_registration(self) -> None:
        """Reject registry mutation after an application freezes its snapshot.

        The case uses explicit inputs in the supported certification regime.
        It checks the public result or the documented failure state.

        Notes
        -----
        The test compares the result with explicit numerical or structural assertions.
        """
        program: str = """
from diffpes.certify import freeze_registry, register_model
from diffpes.types import make_forward_model_spec
freeze_registry()
spec = make_forward_model_spec(
    'org.diffpes.model.registry_test.frozen',
    '1.0.0',
    'org.diffpes.observable.test.result',
    'tests.registry:frozen',
)
try:
    register_model(spec, lambda value: value)
except ValueError as exc:
    assert 'frozen' in str(exc)
else:
    raise AssertionError('registration unexpectedly succeeded')
"""
        completed: subprocess.CompletedProcess[str] = subprocess.run(
            [sys.executable, "-c", program],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
