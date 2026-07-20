"""Verify the repository dependency and runtime foundation.

The tests in this module establish that the differentiable solver stack is
installed, diffpes enables JAX 64-bit precision before numerical work, and
Equinox modules retain their structure through JAX PyTree operations.
"""

import ast
import hashlib
import re
import tomllib
from pathlib import Path

# ruff: noqa: I001 -- diffpes must configure JAX before stack imports.
import diffpes

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax
import optax
import optimistix
import pytest
import yaml
import numpy as np
from beartype import beartype
from beartype.typing import Any
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped

from tests._assertions import (
    assert_matches_reference,
    assert_tree_finite,
    assert_trees_close,
)
from tests._factories import (
    toy_band_structure,
    toy_chain_diagonalized,
    toy_graphene_diagonalized,
    toy_orbital_projection,
    toy_polarization_config,
    toy_simulation_params,
    toy_slater_params,
)
from diffpes.types import (
    ArpesSpectrum,
    BandStructure,
    DiagonalizedBands,
    OrbitalProjection,
    PolarizationConfig,
    SimulationParams,
    SlaterParams,
    TBModel,
    make_slater_params,
)
from diffpes.simul import simulate_novice, simulate_tb_radial


@jaxtyped(typechecker=beartype)
def _tb_reference_payload(
    bands: DiagonalizedBands,
    slater: SlaterParams,
) -> tuple[ArpesSpectrum, Float[Array, ""], Float[Array, " n_orbitals"]]:
    """Recompute one tight-binding radial reference and zeta gradient."""
    params: SimulationParams = toy_simulation_params(fidelity=512)
    polarization: PolarizationConfig = toy_polarization_config()
    spectrum: ArpesSpectrum = simulate_tb_radial(
        bands,
        slater,
        params,
        polarization,
    )

    def intensity_sum(zeta: Float[Array, " n_orbitals"]) -> Float[Array, ""]:
        varied_slater: SlaterParams = make_slater_params(
            zeta=zeta,
            orbital_basis=slater.orbital_basis,
            coefficients=slater.coefficients,
        )
        varied_spectrum: ArpesSpectrum = simulate_tb_radial(
            bands,
            varied_slater,
            params,
            polarization,
        )
        total: Float[Array, ""] = jnp.sum(varied_spectrum.intensity)
        return total

    total_intensity: Float[Array, ""] = jnp.sum(spectrum.intensity)
    zeta_gradient: Float[Array, " n_orbitals"] = jax.grad(intensity_sum)(
        slater.zeta
    )
    payload: tuple[
        ArpesSpectrum,
        Float[Array, ""],
        Float[Array, " n_orbitals"],
    ] = (spectrum, total_intensity, zeta_gradient)
    return payload


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
            "diff" + "tb",
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


class TestCI(chex.TestCase):
    """Validate the continuous-integration workflow from WP5.1.

    Covers workflow syntax, push and pull-request triggers, and the complete
    supported-Python matrix declared by the package metadata.
    """

    def test_workflow_matrix(self) -> None:
        """Exercise CI on every supported Python minor version.

        Confirms the workflow exists, parses as YAML, runs for pushes and pull
        requests, and tests Python 3.12, 3.13, and 3.14 exactly.

        Notes
        -----
        Loads the checked-in workflow using PyYAML and compares its declarative
        triggers and test matrix with the WP5.1 external configuration truth.
        """
        repository_root: Path = Path(__file__).resolve().parents[1]
        workflow_path: Path = repository_root / ".github/workflows/tests.yml"
        workflow: dict[str, Any] = yaml.safe_load(workflow_path.read_text())
        triggers: list[str] = workflow["on"]
        python_versions: list[str] = workflow["jobs"]["test"]["strategy"][
            "matrix"
        ]["python-version"]

        self.assertTrue(workflow_path.is_file())
        chex.assert_equal(triggers, ["push", "pull_request"])
        chex.assert_equal(python_versions, ["3.12", "3.13", "3.14"])

    def test_pypi_release_workflow(self) -> None:
        """Publish matching version tags through trusted PyPI identity.

        Confirms the dedicated release workflow is tag-only, uses the protected
        ``pypi`` environment with job-scoped OIDC permission, smoke-tests both
        distribution formats, and requires uv trusted publishing.

        Notes
        -----
        Parses the workflow as YAML and inspects its trigger, permissions, and
        executable commands without contacting PyPI or minting credentials.
        """
        repository_root: Path = Path(__file__).resolve().parents[1]
        workflow_path: Path = repository_root / ".github/workflows/release.yml"
        workflow: dict[str, Any] = yaml.safe_load(workflow_path.read_text())
        publish_job: dict[str, Any] = workflow["jobs"]["publish"]
        run_commands: tuple[str, ...] = tuple(
            step["run"] for step in publish_job["steps"] if "run" in step
        )
        combined_commands: str = "\n".join(run_commands)

        chex.assert_equal(workflow["on"]["push"]["tags"], ["v*"])
        chex.assert_equal(publish_job["environment"]["name"], "pypi")
        chex.assert_equal(
            publish_job["permissions"],
            {"contents": "read", "id-token": "write"},
        )
        self.assertIn("uv build --no-sources", combined_commands)
        self.assertIn("--with dist/*.whl", combined_commands)
        self.assertIn("--with dist/*.tar.gz", combined_commands)
        self.assertIn(
            "uv publish --trusted-publishing always dist/*",
            combined_commands,
        )


class TestRegressionReferences(chex.TestCase):
    """Validate the pre-refactor forward baselines from WP6.1.

    Covers fixed-seed novice and tight-binding radial spectra, the standing
    zeta-gradient regression, archive metadata, and manifest checksums.
    """

    @pytest.mark.big_mem
    @pytest.mark.rss_limit_mb(1200)
    def test_forward_replay_and_manifest(self) -> None:
        """Replay all reference artifacts within their pinned tolerances.

        Confirms spectrum arrays reproduce at relative tolerance ``1e-12``,
        the zeta gradients reproduce at least as strictly as their ``1e-9``
        contract, and every committed archive matches its manifest SHA-256.

        Notes
        -----
        Rebuilds all three CPU/x64 factory pipelines with seed 20260713,
        compares their PyTree leaf order through the shared NPZ loader, then
        checks declared shapes, float64 dtypes, and artifact digests.
        """
        reference_directory: Path = (
            Path(__file__).parent / "test_diffpes" / "_reference_data"
        )
        manifest: str = (reference_directory / "MANIFEST.md").read_text()
        key: PRNGKeyArray = jax.random.key(20260713)
        novice: ArpesSpectrum = simulate_novice(
            toy_band_structure(key),
            toy_orbital_projection(key),
            toy_simulation_params(fidelity=512),
        )
        graphene_model: TBModel
        graphene_bands: DiagonalizedBands
        graphene_model, graphene_bands = toy_graphene_diagonalized(n_k=12)
        chain_model: TBModel
        chain_bands: DiagonalizedBands
        chain_model, chain_bands = toy_chain_diagonalized(n_k=16)
        del graphene_model, chain_model
        slater: SlaterParams = toy_slater_params()
        graphene_payload: tuple[
            ArpesSpectrum,
            Float[Array, ""],
            Float[Array, " n_orbitals"],
        ] = _tb_reference_payload(graphene_bands, slater)
        chain_payload: tuple[
            ArpesSpectrum,
            Float[Array, ""],
            Float[Array, " n_orbitals"],
        ] = _tb_reference_payload(chain_bands, slater)

        assert_matches_reference(novice, "novice_toy", rtol=1e-12)
        assert_matches_reference(
            graphene_payload,
            "tb_radial_graphene",
            rtol=1e-12,
        )
        assert_matches_reference(
            chain_payload,
            "tb_radial_chain",
            rtol=1e-12,
        )
        chex.assert_shape(novice.intensity, (8, 512))
        chex.assert_shape(graphene_payload[0].intensity, (12, 512))
        chex.assert_shape(chain_payload[0].intensity, (16, 512))
        actual_dtypes: tuple[jnp.dtype, ...] = tuple(
            array.dtype
            for array in (
                novice.intensity,
                graphene_payload[0].intensity,
                chain_payload[0].intensity,
                graphene_payload[2],
                chain_payload[2],
            )
        )
        chex.assert_equal(actual_dtypes, (jnp.float64,) * 5)

        for artifact_name in (
            "novice_toy",
            "tb_radial_graphene",
            "tb_radial_chain",
        ):
            artifact_path: Path = reference_directory / f"{artifact_name}.npz"
            digest: str = hashlib.sha256(
                artifact_path.read_bytes()
            ).hexdigest()
            self.assertIn(f"SHA-256: `{digest}`", manifest)
            with np.load(artifact_path, allow_pickle=False) as archive:
                self.assertTrue(
                    all(
                        array.dtype == np.float64 for array in archive.values()
                    )
                )


class TestRepositoryArchitecture(chex.TestCase):
    """Enforce centralized Equinox carriers, factories, and constants."""

    @staticmethod
    def _production_modules() -> tuple[tuple[Path, ast.Module], ...]:
        """Parse every production Python module in deterministic order."""
        source_root: Path = Path(__file__).resolve().parents[1] / "src/diffpes"
        modules: tuple[tuple[Path, ast.Module], ...] = tuple(
            (path, ast.parse(path.read_text(encoding="utf-8")))
            for path in sorted(source_root.rglob("*.py"))
        )
        return modules

    def test_legacy_pytree_carriers_are_forbidden(self) -> None:
        """Reject NamedTuple and manual JAX PyTree registration machinery."""
        violations: list[str] = []
        for path, module in self._production_modules():
            for node in ast.walk(module):
                if isinstance(node, ast.ClassDef):
                    bases: set[str] = {
                        ast.unparse(base) for base in node.bases
                    }
                    methods: set[str] = {
                        child.name
                        for child in node.body
                        if isinstance(child, ast.FunctionDef)
                    }
                    if "NamedTuple" in bases or methods & {
                        "tree_flatten",
                        "tree_unflatten",
                    }:
                        violations.append(f"{path}:{node.lineno}:{node.name}")
                if isinstance(node, ast.Call):
                    called: str = ast.unparse(node.func)
                    if called.endswith("register_pytree_node_class"):
                        violations.append(f"{path}:{node.lineno}:{called}")
        self.assertEqual(violations, [])

    def test_all_production_carriers_are_types_equinox_modules(self) -> None:
        """Keep every public production carrier under ``diffpes.types``."""
        violations: list[str] = []
        for path, module in self._production_modules():
            in_types: bool = path.parent.name == "types"
            for node in module.body:
                if not isinstance(node, ast.ClassDef) or node.name.startswith(
                    "_"
                ):
                    continue
                bases: set[str] = {ast.unparse(base) for base in node.bases}
                if not in_types or "eqx.Module" not in bases:
                    violations.append(f"{path}:{node.lineno}:{node.name}")
        self.assertEqual(violations, [])

    def test_make_factories_are_types_owned(self) -> None:
        """Forbid carrier-building ``make_*`` factories outside types."""
        violations: list[str] = []
        for path, module in self._production_modules():
            if path.parent.name == "types":
                continue
            for node in module.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith("make_"):
                        violations.append(f"{path}:{node.lineno}:{node.name}")
        self.assertEqual(violations, [])

    def test_declarative_constants_are_types_owned(self) -> None:
        """Allow only generated/runtime module state outside types."""
        allowed: dict[str, set[str]] = {
            "__init__.py": {"__version__"},
            "inout/hdf5.py": {"_PYTREE_REGISTRY"},
            "maths/gaunt.py": {"GAUNT_TABLE"},
            "utils/math.py": {"_W_POLY"},
        }
        violations: list[str] = []
        for path, module in self._production_modules():
            if path.parent.name == "types":
                continue
            relative_path: str = path.as_posix().split("/src/diffpes/", 1)[1]
            for node in module.body:
                names: list[str] = []
                if isinstance(node, ast.Assign):
                    names = [
                        target.id
                        for target in node.targets
                        if isinstance(target, ast.Name)
                    ]
                elif isinstance(node, ast.AnnAssign) and isinstance(
                    node.target, ast.Name
                ):
                    names = [node.target.id]
                for name in names:
                    if name == "__all__" or name in allowed.get(
                        relative_path, set()
                    ):
                        continue
                    violations.append(f"{path}:{node.lineno}:{name}")
        self.assertEqual(violations, [])


class TestStack(chex.TestCase):
    """Validate the differentiable runtime stack and its JAX contracts.

    Covers import availability for Equinox, Optimistix, Lineax, and Optax,
    the package-wide float64 configuration, and Equinox PyTree reconstruction.
    """

    def test_stack_imports(self) -> None:
        """Preserve stack imports, x64 precision, and PyTree structure.

        Confirms that every library selected for types and solvers imports in
        the diffpes runtime, scalar JAX arrays default to float64, and a native
        Equinox module round-trips through JAX tree flattening.

        Notes
        -----
        Imports the runtime packages at module collection after diffpes,
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
