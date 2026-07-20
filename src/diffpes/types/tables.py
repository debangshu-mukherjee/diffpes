"""Small immutable numerical tables used by simulation routines."""

import jax.numpy as jnp
from jaxtyping import Array, Float

_CROSS_SECTION_ENERGIES: Float[Array, " 3"] = jnp.array(
    [20.0, 40.0, 60.0], dtype=jnp.float64
)
_CROSS_SECTION_SIGMA_S: Float[Array, " 3"] = jnp.array(
    [0.1, 0.08, 0.06], dtype=jnp.float64
)
_CROSS_SECTION_SIGMA_P: Float[Array, " 3"] = jnp.array(
    [0.6, 0.9, 1.1], dtype=jnp.float64
)
_CROSS_SECTION_SIGMA_D: Float[Array, " 3"] = jnp.array(
    [2.0, 1.5, 1.2], dtype=jnp.float64
)

__all__: list[str] = []
