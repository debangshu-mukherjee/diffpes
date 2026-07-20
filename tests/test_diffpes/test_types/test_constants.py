"""Tests for centralized diffpes constants and orbital conventions."""

from types import MappingProxyType

import jax.numpy as jnp

from diffpes.types import (
    CROSS_SECTION_ENERGIES,
    CROSS_SECTION_SIGMA_D,
    CROSS_SECTION_SIGMA_P,
    CROSS_SECTION_SIGMA_S,
    L_MAX,
    ORBITAL_DIRS_NORMALIZED,
)
from diffpes.types.orbital_constants import (
    _D_ORBITAL_SLICE,
    _N_ORBITALS,
    _ORBITAL_INDEX,
    _P_ORBITAL_SLICE,
)
from diffpes.types.vasp_constants import _COORDINATE_MODE_TOKENS


def test_orbital_conventions_have_one_consistent_ordering() -> None:
    """Keep shared VASP orbital indices, slices, and directions aligned."""
    assert isinstance(_ORBITAL_INDEX, MappingProxyType)
    assert len(_ORBITAL_INDEX) == _N_ORBITALS
    assert _P_ORBITAL_SLICE == slice(1, 4)
    assert _D_ORBITAL_SLICE == slice(4, 9)
    assert ORBITAL_DIRS_NORMALIZED.shape == (_N_ORBITALS, 3)
    assert jnp.allclose(ORBITAL_DIRS_NORMALIZED[0], jnp.zeros(3))


def test_lookup_tables_and_tokens_are_immutable_conventions() -> None:
    """Keep cross-section rows aligned and parser tokens immutable."""
    assert isinstance(_COORDINATE_MODE_TOKENS, frozenset)
    assert L_MAX == 4
    assert _COORDINATE_MODE_TOKENS == {
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
