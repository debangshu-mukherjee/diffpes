"""Tests for semantic transformation contracts."""

import pytest

from diffpes.certify.contracts import (
    ContractError,
    compose_transformations,
    make_transformation_contract,
    validate_composition,
)


def test_composition_propagates_information_loss_and_invalidations() -> None:
    """Carry explicit semantics and all downstream losses conservatively."""
    amplitude = make_transformation_contract(
        "org.diffpes.transform.matrix_element.amplitude",
        "1.0.0",
        requires=("energy-axis", "orbital-phase"),
        produces=("amplitude",),
        preserves=("energy-axis", "orbital-phase"),
        introduces=("dipole-approximation",),
    )
    intensity = make_transformation_contract(
        "org.diffpes.transform.observable.intensity",
        "1.0.0",
        requires=("amplitude",),
        produces=("intensity",),
        preserves=("energy-axis",),
        destroys=("orbital-phase",),
        invalidates_claims=("org.diffpes.claim.phase.resolved",),
    )
    report = compose_transformations(
        (amplitude, intensity),
        initial_semantics=("energy-axis", "orbital-phase"),
    )
    assert report.valid
    assert report.available_semantics == ("energy-axis", "intensity")
    assert set(report.destroyed_information) == {
        "amplitude",
        "dipole-approximation",
        "orbital-phase",
    }
    assert report.invalidated_claims == ("org.diffpes.claim.phase.resolved",)


def test_missing_requirement_is_reported_and_strict_composition_raises() -> (
    None
):
    """Refuse a transformation whose parent semantics do not satisfy it."""
    contract = make_transformation_contract(
        "org.diffpes.transform.resolution.energy",
        "1.0.0",
        requires=("energy-axis",),
        produces=("broadened-intensity",),
    )
    report = validate_composition(
        (contract,), initial_semantics=("intensity",)
    )
    assert not report.valid
    assert "energy-axis" in report.errors[0]
    with pytest.raises(ContractError, match="energy-axis"):
        compose_transformations((contract,), initial_semantics=("intensity",))


def test_contradictory_contract_is_rejected() -> None:
    """Reject a property declared both preserved and destroyed."""
    with pytest.raises(ContractError, match="preserve/introduce and destroy"):
        make_transformation_contract(
            "org.diffpes.transform.invalid.example",
            "1.0.0",
            preserves=("absolute-intensity",),
            destroys=("absolute-intensity",),
        )


@pytest.mark.parametrize(
    ("identifier", "version"),
    [("not-an-id", "1.0.0"), ("org.diffpes.transform.valid", "version")],
)
def test_contract_identity_and_version_are_validated(
    identifier, version
) -> None:
    """Require stable reverse-DNS identities and semantic versions."""
    with pytest.raises(ContractError):
        make_transformation_contract(identifier, version)
