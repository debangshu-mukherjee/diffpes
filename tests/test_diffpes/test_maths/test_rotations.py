"""Validate differentiable Cartesian rotations.

Extended Summary
----------------
The tests compare Rodrigues matrices with SciPy rotations. They check
orthogonality, orientation, fixed axes, JAX transforms, and gradients.
"""

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from scipy.spatial.transform import Rotation

from diffpes.maths.rotations import rodrigues_rotation
from tests._gradients import gradient_gate


class TestRodriguesRotation(chex.TestCase):
    """Validate :func:`~diffpes.maths.rodrigues_rotation`.

    The tests cover external values, rotation invariants, the zero-axis guard,
    JIT, vmap, and gradients in the axis and angle.

    :see: :func:`~diffpes.maths.rodrigues_rotation`
    """

    def test_matches_scipy_rotation(self) -> None:
        """Match SciPy rotation matrices to relative tolerance 1e-12.

        The test normalizes three external axes. SciPy then constructs each
        active matrix from the corresponding rotation vector.

        Notes
        -----
        Compare each float64 matrix with ``Rotation.from_rotvec``. Use
        ``rtol=1e-12`` and ``atol=1e-12``.
        """
        cases: tuple[tuple[list[float], float], ...] = (
            ([1.0, 0.0, 0.0], 0.37),
            ([0.0, -2.0, 0.0], -0.82),
            ([1.2, -0.7, 2.1], 1.13),
        )
        axis_values: list[float]
        angle_value: float
        for axis_values, angle_value in cases:
            with self.subTest(axis=axis_values, angle=angle_value):
                axis: Float[Array, "3"] = jnp.asarray(
                    axis_values,
                    dtype=jnp.float64,
                )
                axis_numpy: np.ndarray = np.asarray(
                    axis_values,
                    dtype=np.float64,
                )
                rotation_vector: np.ndarray = (
                    angle_value * axis_numpy / np.linalg.norm(axis_numpy)
                )
                expected: Float[Array, "3 3"] = jnp.asarray(
                    Rotation.from_rotvec(rotation_vector).as_matrix()
                )
                actual: Float[Array, "3 3"] = rodrigues_rotation(
                    axis,
                    angle_value,
                )
                chex.assert_trees_all_close(
                    actual,
                    expected,
                    rtol=1e-12,
                    atol=1e-12,
                )

    def test_preserves_rotation_invariants(self) -> None:
        """Verify orthogonality, orientation, and the fixed-axis property.

        A proper rotation has ``R @ R.T = I`` and determinant one. It also
        leaves its normalized axis unchanged.

        Notes
        -----
        Use a generic axis and a 0.73 radian angle. Check each invariant at
        absolute tolerance ``1e-14``.
        """
        axis: Float[Array, "3"] = jnp.array([1.4, -0.2, 0.9])
        normalized_axis: Float[Array, "3"] = axis / jnp.linalg.norm(axis)
        rotation: Float[Array, "3 3"] = rodrigues_rotation(axis, 0.73)
        chex.assert_trees_all_close(
            rotation @ rotation.T,
            jnp.eye(3),
            rtol=0.0,
            atol=1e-14,
        )
        chex.assert_trees_all_close(
            jnp.linalg.det(rotation),
            jnp.array(1.0),
            rtol=0.0,
            atol=1e-14,
        )
        chex.assert_trees_all_close(
            rotation @ normalized_axis,
            normalized_axis,
            rtol=0.0,
            atol=1e-14,
        )

    def test_zero_axis_returns_identity_with_finite_gradient(self) -> None:
        """Verify the identity value and finite gradient at a zero axis.

        The safe axis normalization selects a zero subgradient at the origin.
        Thus, the matrix remains finite for this boundary input.

        Notes
        -----
        JIT the matrix calculation. Differentiate its sum with respect to the
        three axis components.
        """
        axis: Float[Array, "3"] = jnp.zeros(3)
        rotation: Float[Array, "3 3"] = jax.jit(rodrigues_rotation)(
            axis,
            0.4,
        )
        gradient: Float[Array, "3"] = jax.grad(
            lambda candidate: jnp.sum(rodrigues_rotation(candidate, 0.4))
        )(axis)
        chex.assert_trees_all_close(
            rotation,
            jnp.eye(3),
            rtol=0.0,
            atol=0.0,
        )
        chex.assert_trees_all_close(
            gradient,
            jnp.zeros(3),
            rtol=0.0,
            atol=0.0,
        )
        chex.assert_tree_all_finite((rotation, gradient))

    def test_jit_and_vmap_match_eager_results(self) -> None:
        """Verify consistent matrices under JIT and vmap.

        The test uses one generic axis and five traced angles. It compares the
        batched matrices with an eager stack.

        Notes
        -----
        JIT a vmapped Rodrigues call over the angle axis. Compare all entries
        at ``rtol=1e-14``.
        """
        axis: Float[Array, "3"] = jnp.array([0.4, -0.8, 1.3])
        angles: Float[Array, " 5"] = jnp.linspace(-0.7, 0.9, 5)
        vmapped: Float[Array, "5 3 3"] = jax.jit(
            jax.vmap(lambda angle: rodrigues_rotation(axis, angle))
        )(angles)
        eager: Float[Array, "5 3 3"] = jnp.stack(
            tuple(rodrigues_rotation(axis, angle) for angle in angles)
        )
        chex.assert_trees_all_close(
            vmapped,
            eager,
            rtol=1e-14,
            atol=1e-14,
        )

    def test_axis_and_angle_gradients_match_finite_differences(self) -> None:
        """Match axis and angle gradients with central finite differences.

        A generic matrix contraction depends on both traced inputs. The shared
        harness checks forward mode, reverse mode, and central differences.

        Notes
        -----
        Contract the matrix with asymmetric weights. Require nonzero gradient
        norms for the axis leaf and the angle leaf.
        """
        axis: Float[Array, "3"] = jnp.array([0.7, -1.1, 0.3])
        angle: Float[Array, ""] = jnp.array(0.61)
        weights: Float[Array, "3 3"] = jnp.array(
            [
                [0.2, -0.7, 1.1],
                [1.4, 0.5, -0.3],
                [-0.9, 0.8, 0.6],
            ]
        )

        def loss(
            parameters: tuple[Float[Array, "3"], Float[Array, ""]],
        ) -> Float[Array, ""]:
            candidate_axis: Float[Array, "3"]
            candidate_angle: Float[Array, ""]
            candidate_axis, candidate_angle = parameters
            value: Float[Array, ""] = jnp.sum(
                rodrigues_rotation(candidate_axis, candidate_angle) * weights
            )
            return value

        gradient_gate(loss, (axis, angle), regime="smooth")
