"""Verify the repository dependency and runtime foundation.

The tests in this module establish that the differentiable solver stack is
installed, DiffPES enables JAX 64-bit precision before numerical work, and
Equinox modules retain their structure through JAX PyTree operations.
"""

import diffpes  # noqa: I001 -- DiffPES must configure JAX before stack imports.

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax
import optax
import optimistix
from jaxtyping import Array, Float


class TestStack(chex.TestCase):
    """Validate the differentiable runtime stack and its JAX contracts.

    Covers import availability for Equinox, Optimistix, Lineax, and Optax,
    the package-wide float64 configuration, and Equinox PyTree reconstruction.
    """

    def test_stack_imports(self) -> None:
        """Preserve stack imports, x64 precision, and PyTree structure.

        Confirms that every library selected for types and solvers imports in
        the DiffPES runtime, scalar JAX arrays default to float64, and a native
        Equinox module round-trips through JAX tree flattening.

        Notes
        -----
        Imports the runtime packages at module collection after DiffPES,
        constructs a scalar Equinox linear layer with a fixed key, and checks
        the reconstructed module type and leaves exactly.
        """
        runtime_modules: tuple[object, ...] = (
            eqx,
            optimistix,
            lineax,
            optax,
        )
        module_names: tuple[str, ...] = tuple(
            module.__name__ for module in runtime_modules
        )
        precision_probe: Float[Array, ""] = jnp.zeros(())
        linear_module: eqx.Module = eqx.nn.Linear(
            "scalar",
            "scalar",
            key=jax.random.PRNGKey(0),
        )
        flattened: tuple[list[Array], jax.tree_util.PyTreeDef] = (
            jax.tree_util.tree_flatten(linear_module)
        )
        leaves: list[Array]
        tree_definition: jax.tree_util.PyTreeDef
        leaves, tree_definition = flattened
        reconstructed: eqx.Module = jax.tree_util.tree_unflatten(
            tree_definition,
            leaves,
        )
        reconstructed_leaves: list[Array] = jax.tree_util.tree_leaves(
            reconstructed
        )

        chex.assert_equal(
            module_names,
            ("equinox", "optimistix", "lineax", "optax"),
        )
        chex.assert_equal(precision_probe.dtype, jnp.float64)
        self.assertIsInstance(reconstructed, eqx.Module)
        chex.assert_trees_all_equal(reconstructed_leaves, leaves)
