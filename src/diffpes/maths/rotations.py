r"""Construct differentiable three-dimensional rotations.

Extended Summary
----------------
This module provides one shared Rodrigues primitive for detector, crystal,
polarization, and spin frames. The function accepts a traced axis and angle.
It keeps both values available to JAX transformations.

Routine Listings
----------------
:func:`rodrigues_rotation`
    Construct a rotation matrix with Rodrigues' formula.

Notes
-----
The function uses active rotations on column vectors. A zero axis gives the
identity matrix and a zero selected gradient.
"""

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from diffpes.types import ScalarFloat

from .safe import safe_divide, safe_norm


@jaxtyped(typechecker=beartype)
def rodrigues_rotation(
    axis: Float[Array, "3"],
    angle: ScalarFloat,
) -> Float[Array, "3 3"]:
    r"""Construct a rotation matrix with Rodrigues' formula.

    The function safely normalizes the rotation axis. It then constructs an
    active rotation matrix for column vectors.

    :see: :class:`~.test_rotations.TestRodriguesRotation`

    Parameters
    ----------
    axis : Float[Array, "3"]
        Rotation axis in Cartesian coordinates.
    angle : ScalarFloat
        Active rotation angle in radians.

    Returns
    -------
    rotation : Float[Array, "3 3"]
        Active Cartesian rotation matrix.

    Notes
    -----
    The function uses
    :math:`R = I + \sin(\alpha)[\hat n]_\times +
    (1 - \cos(\alpha))[\hat n]_\times^2`.
    The safe normalization gives the identity matrix for a zero axis.
    """
    angle_array: Float[Array, ""] = jnp.asarray(angle)
    axis_norm: Float[Array, ""] = safe_norm(axis)
    normalized_axis: Float[Array, "3"] = safe_divide(axis, axis_norm)
    axis_x: Float[Array, ""] = normalized_axis[0]
    axis_y: Float[Array, ""] = normalized_axis[1]
    axis_z: Float[Array, ""] = normalized_axis[2]
    zero: Float[Array, ""] = jnp.zeros_like(axis_x)
    skew: Float[Array, "3 3"] = jnp.stack(
        (
            jnp.stack((zero, -axis_z, axis_y)),
            jnp.stack((axis_z, zero, -axis_x)),
            jnp.stack((-axis_y, axis_x, zero)),
        )
    )
    identity: Float[Array, "3 3"] = jnp.eye(3, dtype=axis.dtype)
    skew_squared: Float[Array, "3 3"] = skew @ skew
    rotation: Float[Array, "3 3"] = (
        identity
        + jnp.sin(angle_array) * skew
        + (1.0 - jnp.cos(angle_array)) * skew_squared
    )
    return rotation


__all__: list[str] = ["rodrigues_rotation"]
