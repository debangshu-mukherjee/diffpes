"""Tests for centralized diffpes constants and orbital conventions."""

from types import MappingProxyType

import jax.numpy as jnp

from diffpes.types import (
    COORDINATE_MODE_TOKENS,
    CROSS_SECTION_ENERGIES,
    CROSS_SECTION_SIGMA_D,
    CROSS_SECTION_SIGMA_P,
    CROSS_SECTION_SIGMA_S,
    D_ORBITAL_SLICE,
    L_MAX,
    N_ORBITALS,
    ORBITAL_DIRS_NORMALIZED,
    ORBITAL_INDEX,
    P_ORBITAL_SLICE,
)


def test_orbital_conventions_have_one_consistent_ordering() -> None:
    """Keep shared VASP orbital indices, slices, and directions aligned."""
    assert isinstance(ORBITAL_INDEX, MappingProxyType)
    assert len(ORBITAL_INDEX) == N_ORBITALS
    assert P_ORBITAL_SLICE == slice(1, 4)
    assert D_ORBITAL_SLICE == slice(4, 9)
    assert ORBITAL_DIRS_NORMALIZED.shape == (N_ORBITALS, 3)
    assert jnp.allclose(ORBITAL_DIRS_NORMALIZED[0], jnp.zeros(3))


def test_lookup_tables_and_tokens_are_immutable_conventions() -> None:
    """Keep cross-section rows aligned and parser tokens immutable."""
    assert isinstance(COORDINATE_MODE_TOKENS, frozenset)
    assert L_MAX == 4
    assert COORDINATE_MODE_TOKENS == {
        "cartesian",
        "direct",
        "fractional",
        "reciprocal",
    }
    assert CROSS_SECTION_ENERGIES.shape == (3,)
    for values in (
        CROSS_SECTION_SIGMA_S,
        CROSS_SECTION_SIGMA_P,
        CROSS_SECTION_SIGMA_D,
    ):
        assert values.shape == CROSS_SECTION_ENERGIES.shape
