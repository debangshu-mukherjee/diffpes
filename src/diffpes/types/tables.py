"""Store small immutable numerical tables for simulation routines.

Extended Summary
----------------
This module stores shared physical data as JAX arrays. Jitted simulation
kernels can use these arrays directly.
The module currently holds the simplified Yeh-Lindau photoionization
cross-section tabulation: the photon-energy grid and the s, p, and
d subshell cross sections evaluated on it. Like
:mod:`diffpes.types.constants`, this module imports JAX so its
tables are device arrays consumable directly inside jitted kernels.

Routine Listings
----------------
:obj:`CROSS_SECTION_ENERGIES`
    Photon-energy tabulation grid for the cross-section tables in eV.
:obj:`CROSS_SECTION_SIGMA_D`
    Yeh-Lindau d-subshell cross sections on the tabulation grid.
:obj:`CROSS_SECTION_SIGMA_P`
    Yeh-Lindau p-subshell cross sections on the tabulation grid.
:obj:`CROSS_SECTION_SIGMA_S`
    Yeh-Lindau s-subshell cross sections on the tabulation grid.

Notes
-----
The values use a simplified Yeh-Lindau tabulation at 20, 40, and 60 eV.
Consumers interpolate with constant extrapolation outside the grid. A grid
extension changes each consumer's photon-energy sensitivity. Rerun the
gradient and finite-difference gates after this change.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float

CROSS_SECTION_ENERGIES: Float[Array, " 3"] = jnp.array(
    [20.0, 40.0, 60.0], dtype=jnp.float64
)
CROSS_SECTION_SIGMA_S: Float[Array, " 3"] = jnp.array(
    [0.1, 0.08, 0.06], dtype=jnp.float64
)
CROSS_SECTION_SIGMA_P: Float[Array, " 3"] = jnp.array(
    [0.6, 0.9, 1.1], dtype=jnp.float64
)
CROSS_SECTION_SIGMA_D: Float[Array, " 3"] = jnp.array(
    [2.0, 1.5, 1.2], dtype=jnp.float64
)

__all__: list[str] = [
    "CROSS_SECTION_ENERGIES",
    "CROSS_SECTION_SIGMA_D",
    "CROSS_SECTION_SIGMA_P",
    "CROSS_SECTION_SIGMA_S",
]
