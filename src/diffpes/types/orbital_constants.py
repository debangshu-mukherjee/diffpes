"""Orbital ordering conventions and immutable lookup arrays."""

from types import MappingProxyType

import jax.numpy as jnp
from jaxtyping import Array, Float

_S_IDX: int = 0
_P_ORBITAL_SLICE: slice = slice(1, 4)
_D_ORBITAL_SLICE: slice = slice(4, 9)
_NON_S_ORBITAL_SLICE: slice = slice(1, 9)
_N_ORBITALS: int = 9

_ORBITAL_INDEX = MappingProxyType(
    {
        "s": 0,
        "py": 1,
        "pz": 2,
        "px": 3,
        "dxy": 4,
        "dyz": 5,
        "dz2": 6,
        "dxz": 7,
        "dx2y2": 8,
    }
)

_M_P: Float[Array, " 3"] = jnp.array([1.0, 0.0, -1.0], dtype=jnp.float64)
_M_D: Float[Array, " 5"] = jnp.array(
    [-2.0, -1.0, 0.0, 1.0, 2.0], dtype=jnp.float64
)

_ORBITAL_DIRS: Float[Array, "9 3"] = jnp.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, -1.0, 0.0],
    ],
    dtype=jnp.float64,
)
_ORBITAL_NORMS: Float[Array, " 9"] = jnp.where(
    jnp.linalg.norm(_ORBITAL_DIRS, axis=1) > 0.0,
    jnp.linalg.norm(_ORBITAL_DIRS, axis=1),
    1.0,
)
ORBITAL_DIRS_NORMALIZED: Float[Array, "9 3"] = (
    _ORBITAL_DIRS / _ORBITAL_NORMS[:, jnp.newaxis]
)

__all__: list[str] = ["ORBITAL_DIRS_NORMALIZED"]
