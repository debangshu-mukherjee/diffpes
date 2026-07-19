"""Verify the repository dependency and runtime foundation.

The tests in this module establish that the differentiable solver stack is
installed, DiffPES enables JAX 64-bit precision before numerical work, and
Equinox modules retain their structure through JAX PyTree operations.
"""

import hashlib
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
import pytest
from beartype.typing import Any
from jaxtyping import Array, Float, PRNGKeyArray

from _assertions import assert_tree_finite, assert_trees_close
from _factories import (
    toy_band_structure,
    toy_chain_diagonalized,
    toy_graphene_diagonalized,
    toy_orbital_projection,
    toy_polarization_config,
    toy_simulation_params,
    toy_slater_params,
)
from diffpes.types import (
    BandStructure,
    DiagonalizedBands,
    OrbitalProjection,
    PolarizationConfig,
    SimulationParams,
    SlaterParams,
    TBModel,
)


class TestConftest:
    """Validate the shared pytest numerical and resource contracts.

    Covers the x64 session invariant, stable node-derived random keys, and the
    RSS leak guard's failure behavior in an isolated pytest subprocess.
    """

    def test_x64_and_rng_key(
        self,
        request: pytest.FixtureRequest,
        rng_key: PRNGKeyArray,
    ) -> None:
        """Keep x64 precision and random keys stable across workers.

        Confirms the default scalar dtype is float64 and independently derives
        the expected SHA-256 seed for this node ID to verify the fixture key.

        Notes
        -----
        Hashes the fully qualified pytest node ID, converts its first four
        bytes to an integer, and compares the resulting typed JAX key exactly.
        """
        precision_probe: Float[Array, ""] = jnp.zeros(())
        digest: bytes = hashlib.sha256(request.node.nodeid.encode()).digest()
        expected_seed: int = int.from_bytes(digest[:4], byteorder="big")
        expected_key: PRNGKeyArray = jax.random.key(expected_seed)

        chex.assert_equal(precision_probe.dtype, jnp.float64)
        chex.assert_trees_all_equal(rng_key, expected_key)

    def test_rss_leak_guard_trips(self, pytester: pytest.Pytester) -> None:
        """Reject a retained allocation larger than its marked RSS limit.

        Confirms the real plugin reports a teardown error when a test retains
        more than 100 MiB rather than merely simulating the guard arithmetic.

        Notes
        -----
        Copies the repository conftest into an isolated pytest subprocess,
        touches a retained 160 MiB byte array page-by-page, and requires the
        guard's measured-RSS diagnostic and teardown error.
        """
        conftest_path: Path = Path(__file__).with_name("conftest.py")
        pytester.makeconftest(conftest_path.read_text())
        pytester.makepyfile(
            """
            import pytest

            _RETAINED = []

            @pytest.mark.rss_limit_mb(100)
            def test_retained_allocation():
                allocation = bytearray(160 * 1024 * 1024)
                for offset in range(0, len(allocation), 4096):
                    allocation[offset] = 1
                _RETAINED.append(allocation)
            """
        )
        result: Any = pytester.runpytest_subprocess(
            "-q",
            "-n",
            "0",
        )

        result.assert_outcomes(passed=1, errors=1)
        result.stdout.fnmatch_lines(["*retained*MiB RSS*limit is 100.0 MiB*"])


class TestHelpers:
    """Validate deterministic shared factories and assertion wrappers.

    Covers every WP3.2 factory's declared carrier, shape, finite leaves, and
    fixed-seed reproducibility using the shared strict assertion functions.
    """

    def test_factories_and_assertions(self, rng_key: PRNGKeyArray) -> None:
        """Build finite, correctly shaped, reproducible toy carriers.

        Confirms all seven factories return their declared production types,
        random factories repeat bit-for-bit for one key, and analytic
        tight-binding paths expose the requested number of k-points.

        Notes
        -----
        Builds reduced-size carriers, checks dimensions with Chex, verifies
        every leaf is finite, and compares repeated random trees at zero
        relative and absolute tolerance.
        """
        bands: BandStructure = toy_band_structure(rng_key, n_k=5, n_bands=3)
        repeated_bands: BandStructure = toy_band_structure(
            rng_key,
            n_k=5,
            n_bands=3,
        )
        projections: OrbitalProjection = toy_orbital_projection(
            rng_key,
            n_k=5,
            n_bands=3,
            n_atoms=2,
        )
        repeated_projections: OrbitalProjection = toy_orbital_projection(
            rng_key,
            n_k=5,
            n_bands=3,
            n_atoms=2,
        )
        simulation: SimulationParams = toy_simulation_params(fidelity=64)
        polarization: PolarizationConfig = toy_polarization_config()
        graphene_model: TBModel
        graphene_bands: DiagonalizedBands
        graphene_model, graphene_bands = toy_graphene_diagonalized(n_k=6)
        chain_model: TBModel
        chain_bands: DiagonalizedBands
        chain_model, chain_bands = toy_chain_diagonalized(n_k=7)
        slater: SlaterParams = toy_slater_params()
        all_carriers: tuple[object, ...] = (
            bands,
            projections,
            simulation,
            polarization,
            graphene_model,
            graphene_bands,
            chain_model,
            chain_bands,
            slater,
        )

        assert isinstance(bands, BandStructure)
        assert isinstance(projections, OrbitalProjection)
        assert isinstance(simulation, SimulationParams)
        assert isinstance(polarization, PolarizationConfig)
        assert isinstance(graphene_model, TBModel)
        assert isinstance(graphene_bands, DiagonalizedBands)
        assert isinstance(chain_model, TBModel)
        assert isinstance(chain_bands, DiagonalizedBands)
        assert isinstance(slater, SlaterParams)
        chex.assert_shape(bands.eigenvalues, (5, 3))
        chex.assert_shape(projections.projections, (5, 3, 2, 9))
        chex.assert_shape(graphene_bands.kpoints, (6, 3))
        chex.assert_shape(chain_bands.kpoints, (7, 3))
        chex.assert_shape(slater.zeta, (2,))
        assert_tree_finite(all_carriers)
        assert_trees_close(bands, repeated_bands, rtol=0.0, atol=0.0)
        assert_trees_close(
            projections,
            repeated_projections,
            rtol=0.0,
            atol=0.0,
        )


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
