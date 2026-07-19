"""Verify the repository dependency and runtime foundation.

The tests in this module establish that the differentiable solver stack is
installed, DiffPES enables JAX 64-bit precision before numerical work, and
Equinox modules retain their structure through JAX PyTree operations.
"""

import re
import tomllib
from pathlib import Path

# ruff: noqa: I001 -- DiffPES must configure JAX before stack imports.
import diffpes

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax
import optax
import optimistix
from beartype.typing import Any
from jaxtyping import Array, Float


class TestMetadata(chex.TestCase):
    """Validate the install, tooling, and Python metadata contract.

    Covers standalone dependency purity, unconditional JAX installation,
    supported Python versions, uv-build ownership, and test-tooling scope.
    """

    def test_project_metadata(self) -> None:
        """Keep project metadata consistent with the standalone test floor.

        Confirms that retired dependencies and configuration are absent, JAX
        has one unconditional runtime constraint, Python 3.12 is supported,
        and Ruff, pytest, and interrogate use the program-wide settings.

        Notes
        -----
        Parses ``pyproject.toml`` with the standard-library TOML reader and
        compares its declarative values against the WP2.2 metadata contract.
        """
        project_file: Path = (
            Path(__file__).resolve().parents[1] / "pyproject.toml"
        )
        configuration: dict[str, Any] = tomllib.loads(project_file.read_text())
        project: dict[str, Any] = configuration["project"]
        runtime_dependencies: list[str] = project["dependencies"]
        optional_dependencies: dict[str, list[str]] = project[
            "optional-dependencies"
        ]
        dependency_groups: tuple[str, ...] = tuple(
            runtime_dependencies
        ) + tuple(
            dependency
            for group in optional_dependencies.values()
            for dependency in group
        )
        retired_names: tuple[str, ...] = (
            "difftb",
            "black",
            "isort",
            "twine",
        )
        jax_constraints: list[str] = [
            dependency
            for dependency in runtime_dependencies
            if re.match(r"^jax(?:\[|[<>=!~]|$)", dependency) is not None
        ]
        tool_configuration: dict[str, Any] = configuration["tool"]
        pytest_options: dict[str, Any] = tool_configuration["pytest"][
            "ini_options"
        ]
        ruff_configuration: dict[str, Any] = tool_configuration["ruff"]
        interrogate_configuration: dict[str, Any] = tool_configuration[
            "interrogate"
        ]

        self.assertFalse(
            any(
                retired_name in dependency.lower()
                for retired_name in retired_names
                for dependency in dependency_groups
            )
        )
        self.assertEqual(jax_constraints, ["jax>=0.7.0"])
        self.assertTrue(project["requires-python"].startswith(">="))
        self.assertIn(
            "Programming Language :: Python :: 3.12",
            project["classifiers"],
        )
        self.assertNotIn("setuptools", tool_configuration)
        self.assertNotIn("style", interrogate_configuration)
        self.assertIn("tests/**/*.py", ruff_configuration["include"])
        self.assertEqual(
            pytest_options["addopts"],
            "-n auto --dist loadgroup "
            "--jaxtyping-packages=diffpes,beartype.beartype",
        )


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
