"""Validate centralized parser, orbital, and tabulated-data constants.

The cases check immutable VASP conventions and alignment of the bundled
photoionization cross-section rows against independent shape relationships.
"""

from types import MappingProxyType

import chex
import jax
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


class TestOrbitalConstants:
    """Validate the shared VASP orbital-ordering constants.

    Indices, slices, and direction rows must describe one nine-orbital basis
    with the scalar orbital first.

    :see: :data:`~diffpes.types.ORBITAL_INDEX`
    :see: :data:`~diffpes.types.ORBITAL_DIRS_NORMALIZED`
    """

    def test_share_one_orbital_ordering(self) -> None:
        """Keep orbital indices, slices, and direction rows aligned.

        The check verifies nine immutable indices, the standard p and d slices,
        and a zero direction for the scalar orbital.

        Notes
        -----
        Compares the public mapping and slice constants exactly, then uses Chex
        for the direction-table shape and scalar-orbital row.
        """
        assert isinstance(ORBITAL_INDEX, MappingProxyType)
        assert len(ORBITAL_INDEX) == N_ORBITALS
        assert P_ORBITAL_SLICE == slice(1, 4)
        assert D_ORBITAL_SLICE == slice(4, 9)
        chex.assert_shape(ORBITAL_DIRS_NORMALIZED, (N_ORBITALS, 3))
        chex.assert_trees_all_close(ORBITAL_DIRS_NORMALIZED[0], jnp.zeros(3))


class TestParserConstants:
    """Validate immutable parser tokens and angular-momentum bounds.

    Coordinate selectors and the maximum supported orbital angular momentum
    are pinned shared conventions rather than parser-local values.

    :see: :data:`~diffpes.types.COORDINATE_MODE_TOKENS`
    :see: :data:`~diffpes.types.L_MAX`
    """

    def test_tokens_are_immutable_conventions(self) -> None:
        """Preserve the coordinate-token set and maximum angular momentum.

        The check expects a frozen four-token convention and ``L_MAX=4``.

        Notes
        -----
        Compares the public token container type and contents plus the integer
        angular-momentum bound against independent literal values.
        """
        expected_tokens: frozenset[str] = frozenset(
            {"cartesian", "direct", "fractional", "reciprocal"}
        )

        assert isinstance(COORDINATE_MODE_TOKENS, frozenset)
        assert COORDINATE_MODE_TOKENS == expected_tokens
        assert L_MAX == 4


class TestCrossSectionTables:
    """Validate alignment of the bundled photoionization tables.

    Every s-, p-, and d-channel cross section must provide one value at each
    shared photon-energy node.

    :see: :data:`~diffpes.types.CROSS_SECTION_ENERGIES`
    """

    def test_channel_rows_share_energy_shape(self) -> None:
        """Align every cross-section channel with the energy coordinates.

        The check verifies three energy nodes and an equal one-dimensional
        shape for every angular-momentum channel.

        Notes
        -----
        Iterates over the three independent public channel arrays and uses Chex
        to compare each shape with the energy table.
        """
        channels: tuple[jax.Array, ...] = (
            CROSS_SECTION_SIGMA_S,
            CROSS_SECTION_SIGMA_P,
            CROSS_SECTION_SIGMA_D,
        )

        chex.assert_shape(CROSS_SECTION_ENERGIES, (3,))
        values: jax.Array
        for values in channels:
            chex.assert_shape(values, CROSS_SECTION_ENERGIES.shape)
