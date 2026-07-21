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
        chex.assert_equal(
            triggers,
            ["push", "pull_request", "workflow_dispatch"],
        )
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

        artifact_name: str
        archive: Any
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
    """Enforce the production architecture rules from CONTRIBUTING.

    Covers carrier and factory ownership, import boundaries, public runtime
    type checking, explicit returns, package listings, and zero-legacy exports.
    """

    @staticmethod
    def _production_modules() -> tuple[tuple[Path, ast.Module], ...]:
        """Parse every production Python module in deterministic order."""
        source_root: Path = Path(__file__).resolve().parents[1] / "src/diffpes"
        modules: tuple[tuple[Path, ast.Module], ...] = tuple(
            (path, ast.parse(path.read_text(encoding="utf-8")))
            for path in sorted(source_root.rglob("*.py"))
        )
        return modules

    @staticmethod
    def _test_modules() -> tuple[tuple[Path, ast.Module], ...]:
        """Parse every test Python module in deterministic order."""
        test_root: Path = Path(__file__).resolve().parents[1] / "tests"
        modules: tuple[tuple[Path, ast.Module], ...] = tuple(
            (path, ast.parse(path.read_text(encoding="utf-8")))
            for path in sorted(test_root.rglob("*.py"))
        )
        return modules

    @staticmethod
    def _literal_exports(module: ast.Module) -> set[str]:
        """Return literal names from one module-level ``__all__``."""
        exports: set[str] = set()
        node: ast.stmt
        for node in module.body:
            value: ast.expr | None = None
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            ):
                value = node.value
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "__all__"
            ):
                value = node.value
            if isinstance(value, (ast.List, ast.Tuple)):
                exports = {
                    entry.value
                    for entry in value.elts
                    if isinstance(entry, ast.Constant)
                    and isinstance(entry.value, str)
                }
        return exports

    @staticmethod
    def _routine_listing_summaries(docstring: str) -> dict[str, str]:
        """Return public names and summaries from one Routine Listings block."""
        summaries: dict[str, str] = {}
        lines: list[str] = docstring.splitlines()
        index: int
        line: str
        for index, line in enumerate(lines):
            match: re.Match[str] | None = re.match(
                r":(?:func|class|obj):`(?:~[^`]*\.)?([^`]+)`",
                line.strip(),
            )
            if match is None:
                continue
            summary: str = ""
            if index + 1 < len(lines) and lines[index + 1].startswith("    "):
                summary = lines[index + 1].strip()
            summaries[match.group(1)] = summary
        return summaries

    def test_legacy_pytree_carriers_are_forbidden(self) -> None:
        """Reject legacy PyTree carrier and registration machinery.

        Confirms production carriers do not use ``NamedTuple`` or manual JAX
        flattening hooks instead of the project Equinox carrier contract.

        Notes
        -----
        Parses every production class and call expression, then reports the
        source location of each forbidden base, method, or registration call.
        """
        violations: list[str] = []
        path: Path
        module: ast.Module
        node: ast.AST
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
        """Keep every public carrier under ``diffpes.types``.

        Confirms public production classes are Equinox modules and have the
        types subpackage as their single architectural owner.

        Notes
        -----
        Parses each public class declaration and compares its direct bases and
        source directory with the carrier ownership rule.
        """
        violations: list[str] = []
        path: Path
        module: ast.Module
        node: ast.stmt
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
        """Forbid ``make_*`` factories outside ``diffpes.types``.

        Confirms consumers cannot create a second construction contract for a
        public carrier in another production subpackage.

        Notes
        -----
        Scans top-level production callables and reports each ``make_*`` name
        whose module is not owned by the types subpackage.
        """
        violations: list[str] = []
        path: Path
        module: ast.Module
        node: ast.stmt
        for path, module in self._production_modules():
            if path.parent.name == "types":
                continue
            for node in module.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith("make_"):
                        violations.append(f"{path}:{node.lineno}:{node.name}")
        self.assertEqual(violations, [])

    def test_declarative_constants_are_types_owned(self) -> None:
        """Keep declarative constants under ``diffpes.types``.

        Confirms non-types modules contain only explicitly approved generated
        or runtime state in addition to their public export lists.

        Notes
        -----
        Parses module-level assignments and compares them with the narrow
        allowlist for version, registry, generated table, and polynomial data.
        """
        allowed: dict[str, set[str]] = {
            "__init__.py": {"__version__"},
            "inout/hdf5.py": {"_PYTREE_REGISTRY"},
            "maths/gaunt.py": {"GAUNT_TABLE"},
            "utils/math.py": {"_W_POLY"},
        }
        violations: list[str] = []
        path: Path
        module: ast.Module
        node: ast.stmt
        name: str
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

    def test_type_aliases_are_types_owned(self) -> None:
        """Keep every production type alias under ``diffpes.types``.

        Confirms PEP 695 declarations and legacy ``TypeAlias`` annotations do
        not create local type vocabularies in consuming subpackages.

        Notes
        -----
        Parses module-level declarations and reports the exact source location
        of each alias found outside the types subpackage.
        """
        violations: list[str] = []
        path: Path
        module: ast.Module
        node: ast.stmt
        for path, module in self._production_modules():
            if path.parent.name == "types":
                continue
            for node in module.body:
                if isinstance(node, ast.TypeAlias):
                    alias_name: str = ast.unparse(node.name)
                    violations.append(f"{path}:{node.lineno}:{alias_name}")
                elif isinstance(node, ast.AnnAssign) and ast.unparse(
                    node.annotation
                ).endswith("TypeAlias"):
                    target_name: str = ast.unparse(node.target)
                    violations.append(f"{path}:{node.lineno}:{target_name}")
        self.assertEqual(violations, [])

    def test_public_functions_are_runtime_typechecked(self) -> None:
        """Require the project decorator on every public production function.

        Confirms public module-level callables use the exact
        ``@jaxtyped(typechecker=beartype)`` stack required by CONTRIBUTING.

        Notes
        -----
        Compares normalized decorator syntax through the AST and reports each
        missing function with its source line.
        """
        violations: list[str] = []
        required_decorator: str = "jaxtyped(typechecker=beartype)"
        path: Path
        module: ast.Module
        node: ast.stmt
        for path, module in self._production_modules():
            for node in module.body:
                if not isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) or node.name.startswith("_"):
                    continue
                decorators: set[str] = {
                    ast.unparse(decorator) for decorator in node.decorator_list
                }
                if required_decorator not in decorators:
                    violations.append(f"{path}:{node.lineno}:{node.name}")
        self.assertEqual(violations, [])

    def test_functions_assign_before_returning(self) -> None:
        """Require production functions to return annotated names.

        Confirms each value-returning path binds its result before returning,
        including paths in private and nested helpers.

        Notes
        -----
        Walks each public function while excluding nested function bodies and
        reports non-name return expressions by source line.
        """

        class ReturnVisitor(ast.NodeVisitor):
            """Collect bare returns without descending into nested callables."""

            def __init__(self, root: ast.FunctionDef | ast.AsyncFunctionDef):
                self.root: ast.FunctionDef | ast.AsyncFunctionDef = root
                self.violations: list[int] = []

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                """Visit only the requested root function body."""
                if node is self.root:
                    self.generic_visit(node)

            def visit_AsyncFunctionDef(
                self, node: ast.AsyncFunctionDef
            ) -> None:
                """Visit only the requested asynchronous root function body."""
                if node is self.root:
                    self.generic_visit(node)

            def visit_Lambda(self, node: ast.Lambda) -> None:
                """Exclude lambda expression bodies from the outer contract."""
                del node

            def visit_Return(self, node: ast.Return) -> None:
                """Record returns whose value is not a bound local name."""
                if node.value is not None and not isinstance(
                    node.value, ast.Name
                ):
                    self.violations.append(node.lineno)

        violations: list[str] = []
        path: Path
        module: ast.Module
        node: ast.AST
        for path, module in self._production_modules():
            for node in ast.walk(module):
                if not isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    continue
                visitor: ReturnVisitor = ReturnVisitor(node)
                visitor.visit(node)
                violations.extend(
                    f"{path}:{line}:{node.name}" for line in visitor.violations
                )
        self.assertEqual(violations, [])

    def test_function_intermediates_are_annotated(self) -> None:
        """Require explicit types for production intermediate variables.

        Confirms assignment, loop, context, walrus, and exception targets have
        an annotation in that function scope while respecting ``nonlocal``.

        Notes
        -----
        Walks one callable scope at a time, excludes nested callables and
        throwaway ``_`` bindings, and reports each unannotated local target.
        """

        class AssignmentVisitor(ast.NodeVisitor):
            """Collect annotations and assignments in one callable scope."""

            def __init__(self, root: ast.FunctionDef | ast.AsyncFunctionDef):
                self.root: ast.FunctionDef | ast.AsyncFunctionDef = root
                self.annotated: set[str] = set()
                self.nonlocal_names: set[str] = set()
                self.assignments: list[tuple[int, str]] = []

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                """Visit only the requested root function body."""
                if node is self.root:
                    self.generic_visit(node)

            def visit_AsyncFunctionDef(
                self, node: ast.AsyncFunctionDef
            ) -> None:
                """Visit only the requested asynchronous root function body."""
                if node is self.root:
                    self.generic_visit(node)

            def visit_Lambda(self, node: ast.Lambda) -> None:
                """Exclude lambda expression scopes from the outer contract."""
                del node

            def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
                """Record a directly annotated local name."""
                if isinstance(node.target, ast.Name):
                    self.annotated.add(node.target.id)
                self.generic_visit(node)

            def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
                """Record names whose annotations belong to an outer scope."""
                self.nonlocal_names.update(node.names)

            def _record_target(self, target: ast.expr, line: int) -> None:
                """Record stored names within one assignment-like target."""
                candidate: ast.AST
                for candidate in ast.walk(target):
                    if (
                        isinstance(candidate, ast.Name)
                        and isinstance(candidate.ctx, ast.Store)
                        and candidate.id != "_"
                    ):
                        self.assignments.append((line, candidate.id))

            def visit_Assign(self, node: ast.Assign) -> None:
                """Record plain local-name assignment targets."""
                target: ast.expr
                for target in node.targets:
                    self._record_target(target, node.lineno)
                self.generic_visit(node)

            def visit_For(self, node: ast.For) -> None:
                """Record an ordinary loop target."""
                self._record_target(node.target, node.lineno)
                self.generic_visit(node)

            def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
                """Record an asynchronous loop target."""
                self._record_target(node.target, node.lineno)
                self.generic_visit(node)

            def visit_With(self, node: ast.With) -> None:
                """Record context-manager binding targets."""
                item: ast.withitem
                for item in node.items:
                    if item.optional_vars is not None:
                        self._record_target(item.optional_vars, node.lineno)
                self.generic_visit(node)

            def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
                """Record asynchronous context-manager binding targets."""
                item: ast.withitem
                for item in node.items:
                    if item.optional_vars is not None:
                        self._record_target(item.optional_vars, node.lineno)
                self.generic_visit(node)

            def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
                """Record an assignment-expression target."""
                self._record_target(node.target, node.lineno)
                self.generic_visit(node)

            def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
                """Record an exception-handler binding."""
                if node.name is not None and node.name != "_":
                    self.assignments.append((node.lineno, node.name))
                self.generic_visit(node)

        violations: list[str] = []
        path: Path
        module: ast.Module
        node: ast.AST
        for path, module in self._production_modules():
            for node in ast.walk(module):
                if not isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    continue
                visitor: AssignmentVisitor = AssignmentVisitor(node)
                visitor.visit(node)
                violations.extend(
                    f"{path}:{line}:{node.name}:{name}"
                    for line, name in visitor.assignments
                    if name not in visitor.annotated
                    and name not in visitor.nonlocal_names
                )
        self.assertEqual(violations, [])

    def test_cross_subpackage_imports_use_public_surfaces(self) -> None:
        """Forbid deep imports across production subpackage boundaries.

        Confirms consumers import through ``diffpes.<subpackage>`` instead of
        reaching into another subpackage's implementation file.

        Notes
        -----
        Compares each absolute DiffPES import with the importing file's owning
        subpackage and reports cross-boundary modules deeper than one level.
        """
        violations: list[str] = []
        path: Path
        module: ast.Module
        node: ast.AST
        for path, module in self._production_modules():
            relative_parts: tuple[str, ...] = tuple(
                path.as_posix().split("/src/diffpes/", 1)[1].split("/")
            )
            owner: str = relative_parts[0]
            for node in ast.walk(module):
                if not isinstance(node, ast.ImportFrom) or node.level != 0:
                    continue
                imported_module: str = node.module or ""
                parts: list[str] = imported_module.split(".")
                if (
                    len(parts) > 2
                    and parts[0] == "diffpes"
                    and parts[1] != owner
                ):
                    violations.append(
                        f"{path}:{node.lineno}:{imported_module}"
                    )
        self.assertEqual(violations, [])

    def test_diffpes_imports_are_not_renamed(self) -> None:
        """Forbid aliases for names imported from DiffPES surfaces.

        Confirms each internal DiffPES name has one spelling at every consumer
        and excludes reviewer-hostile private aliases for shared constants.

        Notes
        -----
        Inspects absolute DiffPES imports and reports every ``as`` binding;
        canonical third-party aliases such as ``jnp`` are outside this scan.
        """
        violations: list[str] = []
        path: Path
        module: ast.Module
        node: ast.AST
        imported_name: ast.alias
        for path, module in self._production_modules():
            for node in ast.walk(module):
                if isinstance(node, ast.ImportFrom) and (
                    node.module or ""
                ).startswith("diffpes"):
                    for imported_name in node.names:
                        if imported_name.asname is not None:
                            violations.append(
                                f"{path}:{node.lineno}:{imported_name.name}"
                            )
                elif isinstance(node, ast.Import):
                    for imported_name in node.names:
                        if (
                            imported_name.name.startswith("diffpes")
                            and imported_name.asname is not None
                        ):
                            violations.append(
                                f"{path}:{node.lineno}:{imported_name.name}"
                            )
        self.assertEqual(violations, [])

    def test_typing_constructs_use_beartype_typing(self) -> None:
        """Forbid production imports from the standard typing module.

        Confirms runtime-visible typing constructs come from
        ``beartype.typing`` as required by the package type-checking contract.

        Notes
        -----
        Reports both ``import typing`` and ``from typing import ...`` at their
        production source locations.
        """
        violations: list[str] = []
        path: Path
        module: ast.Module
        node: ast.stmt
        for path, module in self._production_modules():
            for node in module.body:
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module == "typing"
                ):
                    violations.append(f"{path}:{node.lineno}:from typing")
                elif isinstance(node, ast.Import) and any(
                    imported_name.name == "typing"
                    for imported_name in node.names
                ):
                    violations.append(f"{path}:{node.lineno}:import typing")
        self.assertEqual(violations, [])

    def test_package_docstrings_list_every_submodule(self) -> None:
        """Keep package ``Extended Summary`` submodule lists exact.

        Confirms each package docstring contains one ``- :mod:`` entry for
        every sibling module. Each entry repeats that module's summary line.

        Notes
        -----
        Compares filenames and summary lines with the Sphinx module roles and
        descriptions parsed from each production ``__init__.py`` docstring.
        """
        violations: list[str] = []
        path: Path
        module: ast.Module
        for path, module in self._production_modules():
            if path.name != "__init__.py":
                continue
            actual_modules: set[str] = {
                sibling.stem
                for sibling in path.parent.glob("*.py")
                if sibling.name != "__init__.py"
            }
            module_docstring: str = (
                ast.get_docstring(module, clean=False) or ""
            )
            listed_modules: set[str] = set(
                re.findall(r"- :mod:`([^`]+)`", module_docstring)
            )
            listed_descriptions: dict[str, str] = dict(
                re.findall(
                    r"(?m)^- :mod:`([^`]+)`\n    ([^\n]+)$",
                    module_docstring,
                )
            )
            if actual_modules != listed_modules:
                violations.append(
                    f"{path}: missing={sorted(actual_modules - listed_modules)} "
                    f"stale={sorted(listed_modules - actual_modules)}"
                )
            sibling: Path
            for sibling in path.parent.glob("*.py"):
                if sibling.name == "__init__.py":
                    continue
                sibling_module: ast.Module = ast.parse(sibling.read_text())
                sibling_docstring: str = (
                    ast.get_docstring(sibling_module, clean=False) or ""
                )
                sibling_summary: str = sibling_docstring.splitlines()[0]
                if listed_descriptions.get(sibling.stem) != sibling_summary:
                    violations.append(
                        f"{path}: submodule summary mismatch: {sibling.stem}"
                    )
        self.assertEqual(violations, [])

    def test_public_api_uses_three_place_documentation(self) -> None:
        """Keep exports and summaries synchronized in all three locations.

        Confirms each public definition is exported and each module and
        subpackage surface lists exactly the same names and summary sentences.

        Notes
        -----
        Parses literal export lists and Sphinx Routine Listings, then compares
        defining docstrings, module entries, and subpackage entries verbatim.
        """
        parsed_modules: tuple[tuple[Path, ast.Module], ...] = (
            self._production_modules()
        )
        module_records: dict[
            Path, tuple[ast.Module, set[str], dict[str, str]]
        ] = {}
        violations: list[str] = []
        path: Path
        module: ast.Module
        name: str
        for path, module in parsed_modules:
            module_docstring: str = (
                ast.get_docstring(module, clean=False) or ""
            )
            exports: set[str] = self._literal_exports(module)
            listings: dict[str, str] = self._routine_listing_summaries(
                module_docstring
            )
            module_records[path] = (module, exports, listings)
            if path.name == "__init__.py":
                continue
            public_definitions: dict[str, str] = {
                node.name: (ast.get_docstring(node) or "").splitlines()[0]
                for node in module.body
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                )
                and not node.name.startswith("_")
            }
            missing_exports: set[str] = set(public_definitions) - exports
            if missing_exports:
                violations.append(
                    f"{path}: public definitions missing from __all__: "
                    f"{sorted(missing_exports)}"
                )
            if exports != set(listings):
                violations.append(
                    f"{path}: __all__/listing mismatch: "
                    f"missing={sorted(exports - set(listings))}, "
                    f"stale={sorted(set(listings) - exports)}"
                )
            for name in exports & set(listings) & set(public_definitions):
                if public_definitions[name] != listings[name]:
                    violations.append(f"{path}: summary mismatch: {name}")

        source_root: Path = Path(__file__).resolve().parents[1] / "src/diffpes"
        package_path: Path
        for package_path in sorted(source_root.iterdir()):
            init_path: Path = package_path / "__init__.py"
            if not package_path.is_dir() or init_path not in module_records:
                continue
            package_module: ast.Module
            package_exports: set[str]
            package_listings: dict[str, str]
            package_module, package_exports, package_listings = module_records[
                init_path
            ]
            del package_module
            submodule_exports: set[str] = set()
            submodule_summaries: dict[str, str] = {}
            for path, (_, exports, listings) in module_records.items():
                if path.parent == package_path and path.name != "__init__.py":
                    submodule_exports.update(exports)
                    submodule_summaries.update(listings)
            if package_exports != submodule_exports:
                violations.append(
                    f"{init_path}: package/module export mismatch: "
                    f"missing={sorted(submodule_exports - package_exports)}, "
                    f"extra={sorted(package_exports - submodule_exports)}"
                )
            if package_exports != set(package_listings):
                violations.append(
                    f"{init_path}: __all__/listing mismatch: "
                    f"missing={sorted(package_exports - set(package_listings))}, "
                    f"stale={sorted(set(package_listings) - package_exports)}"
                )
            for name in (
                package_exports
                & set(package_listings)
                & set(submodule_summaries)
            ):
                if package_listings[name] != submodule_summaries[name]:
                    violations.append(f"{init_path}: summary mismatch: {name}")
        self.assertEqual(violations, [])

    def test_public_docstrings_follow_house_process_format(self) -> None:
        """Keep public source docstrings on the house process format.

        Extended Summary
        ----------------
        Confirms functions and classes use untitled extended prose. Every
        public function must explain its process in Notes or literal steps.

        Notes
        -----
        Parses source docstrings and checks each numbered bold logic step for
        the required double-colon heading and an indented literal expression.
        """
        violations: list[str] = []
        path: Path
        module: ast.Module
        node: ast.stmt
        for path, module in self._production_modules():
            if path.name == "__init__.py":
                continue
            for node in module.body:
                if not isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ) or node.name.startswith("_"):
                    continue
                docstring: str = ast.get_docstring(node) or ""
                if "\nExtended Summary\n" in docstring:
                    violations.append(
                        f"{path}:{node.lineno}:{node.name}: titled extended "
                        "summary"
                    )
                summary_end: int = docstring.find("\n")
                see_position: int = docstring.find("\n:see:")
                extended_summary: str = docstring[
                    summary_end:see_position
                ].strip()
                if summary_end < 0 or see_position < 0 or not extended_summary:
                    violations.append(
                        f"{path}:{node.lineno}:{node.name}: no untitled "
                        "extended summary before :see:"
                    )
                if isinstance(node, ast.ClassDef):
                    continue
                has_logic: bool = "\nImplementation Logic\n" in docstring
                has_notes: bool = "\nNotes\n" in docstring
                if not has_logic and not has_notes:
                    violations.append(
                        f"{path}:{node.lineno}:{node.name}: no process section"
                    )
                    continue
                if not has_logic:
                    continue
                section_match: re.Match[str] | None = re.search(
                    r"(?ms)^Implementation Logic\n-+\n"
                    r"(.*?)(?=^[A-Z][A-Za-z ]+\n-+\n|\Z)",
                    docstring,
                )
                if section_match is None:
                    violations.append(
                        f"{path}:{node.lineno}:{node.name}: malformed logic "
                        "section"
                    )
                    continue
                logic_section: str = section_match.group(1)
                step_headings: list[str] = re.findall(
                    r"(?m)^\d+\. \*\*[^\n]+\*\*:+$", logic_section
                )
                valid_headings: list[str] = re.findall(
                    r"(?m)^\d+\. \*\*[^\n]+\*\*::$", logic_section
                )
                literal_steps: list[str] = re.findall(
                    r"(?m)^\d+\. \*\*[^\n]+\*\*::\n\n {7}\S",
                    logic_section,
                )
                if (
                    not valid_headings
                    or step_headings != valid_headings
                    or len(literal_steps) != len(valid_headings)
                ):
                    violations.append(
                        f"{path}:{node.lineno}:{node.name}: "
                        f"steps={len(step_headings)}, "
                        f"valid={len(valid_headings)}, "
                        f"literal={len(literal_steps)}"
                    )
        self.assertEqual(violations, [])

    def test_public_objects_have_symbol_owned_tests(self) -> None:
        """Require one reciprocal ``Test<Symbol>`` class per public object.

        Confirms every public production function and class links to its exact
        symbol-owned class in the mirrored test module and that class links back.

        Notes
        -----
        Normalizes underscores and capitalization so scientific abbreviations
        remain flexible while generic multi-symbol test classes are rejected.
        """
        repository_root: Path = Path(__file__).resolve().parents[1]
        source_root: Path = repository_root / "src/diffpes"
        tests_root: Path = repository_root / "tests/test_diffpes"
        violations: list[str] = []
        source_path: Path
        source_module: ast.Module
        node: ast.stmt
        for source_path, source_module in self._production_modules():
            if source_path.name == "__init__.py":
                continue
            relative_path: Path = source_path.relative_to(source_root)
            subpackage: str = relative_path.parts[0]
            test_path: Path = (
                tests_root / f"test_{subpackage}" / f"test_{source_path.name}"
            )
            test_classes: dict[str, ast.ClassDef] = {}
            if test_path.is_file():
                test_module: ast.Module = ast.parse(
                    test_path.read_text(encoding="utf-8")
                )
                test_classes = {
                    node.name: node
                    for node in test_module.body
                    if isinstance(node, ast.ClassDef)
                }
            for node in source_module.body:
                if not isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ) or node.name.startswith("_"):
                    continue
                source_docstring: str = ast.get_docstring(node) or ""
                targets: list[str] = re.findall(
                    r":see:\s+:class:`[^`]*\.(Test\w+)`",
                    source_docstring,
                )
                if len(targets) != 1:
                    violations.append(
                        f"{source_path}:{node.lineno}:{node.name}: "
                        f"test targets={targets}"
                    )
                    continue
                target_name: str = targets[0]
                expected_normalized: str = "test" + re.sub(
                    r"[^a-z0-9]", "", node.name.lower()
                )
                actual_normalized: str = re.sub(
                    r"[^a-z0-9]", "", target_name.lower()
                )
                if actual_normalized != expected_normalized:
                    violations.append(
                        f"{source_path}:{node.lineno}:{node.name}: "
                        f"target={target_name}"
                    )
                    continue
                test_class: ast.ClassDef | None = test_classes.get(target_name)
                if test_class is None:
                    violations.append(
                        f"{source_path}:{node.lineno}:{node.name}: "
                        f"missing {test_path}:{target_name}"
                    )
                    continue
                class_docstring: str = ast.get_docstring(test_class) or ""
                reciprocal_name: str = f"diffpes.{subpackage}.{node.name}"
                if reciprocal_name not in class_docstring:
                    violations.append(
                        f"{test_path}:{test_class.lineno}:{target_name}: "
                        f"missing {reciprocal_name}"
                    )
        self.assertEqual(violations, [])

    def test_test_docstrings_specify_what_and_how(self) -> None:
        """Require complete reader-facing specifications on every test.

        Confirms each test module has an extended summary and every test
        callable has ``-> None``, extended what prose, and a how-focused Notes.

        Notes
        -----
        Parses published test docstrings and reports missing structural parts;
        semantic prose quality remains a review responsibility.
        """
        violations: list[str] = []
        path: Path
        module: ast.Module
        node: ast.AST
        for path, module in self._test_modules():
            module_docstring: str = ast.get_docstring(module) or ""
            if (
                path.name != "__init__.py"
                and len(
                    [
                        line
                        for line in module_docstring.splitlines()
                        if line.strip()
                    ]
                )
                < 2
            ):
                violations.append(f"{path}: module extended summary")
            for node in ast.walk(module):
                if not isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) or not node.name.startswith("test_"):
                    continue
                if not (
                    isinstance(node.returns, ast.Constant)
                    and node.returns.value is None
                ):
                    violations.append(
                        f"{path}:{node.lineno}:{node.name}: -> None"
                    )
                docstring: str = ast.get_docstring(node) or ""
                before_notes: str = docstring.split("Notes\n", 1)[0]
                if (
                    len(
                        [
                            line
                            for line in before_notes.splitlines()
                            if line.strip()
                        ]
                    )
                    < 2
                ):
                    violations.append(
                        f"{path}:{node.lineno}:{node.name}: extended summary"
                    )
                if "\nNotes\n" not in docstring:
                    violations.append(
                        f"{path}:{node.lineno}:{node.name}: Notes"
                    )
        self.assertEqual(violations, [])

    def test_test_intermediates_are_annotated(self) -> None:
        """Require explicit types for intermediate variables in tests.

        Confirms assignment, loop, context, walrus, and exception targets in
        test callables carry a type annotation in their own scope.

        Notes
        -----
        Excludes nested callables, legal ``nonlocal`` reassignments, and the
        throwaway ``_`` name while reporting every other local target.
        """

        class TestAssignmentVisitor(ast.NodeVisitor):
            """Collect annotations and assignments in one test callable."""

            def __init__(self, root: ast.FunctionDef | ast.AsyncFunctionDef):
                self.root: ast.FunctionDef | ast.AsyncFunctionDef = root
                self.annotated: set[str] = set()
                self.nonlocal_names: set[str] = set()
                self.assignments: list[tuple[int, str]] = []

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                """Visit only the requested root function body."""
                if node is self.root:
                    self.generic_visit(node)

            def visit_AsyncFunctionDef(
                self, node: ast.AsyncFunctionDef
            ) -> None:
                """Visit only the requested asynchronous root function body."""
                if node is self.root:
                    self.generic_visit(node)

            def visit_Lambda(self, node: ast.Lambda) -> None:
                """Exclude lambda expression scopes from the outer contract."""
                del node

            def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
                """Record a directly annotated local name."""
                if isinstance(node.target, ast.Name):
                    self.annotated.add(node.target.id)
                self.generic_visit(node)

            def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
                """Record names whose annotations belong to an outer scope."""
                self.nonlocal_names.update(node.names)

            def _record_target(self, target: ast.expr, line: int) -> None:
                """Record stored names within one assignment-like target."""
                candidate: ast.AST
                for candidate in ast.walk(target):
                    if (
                        isinstance(candidate, ast.Name)
                        and isinstance(candidate.ctx, ast.Store)
                        and candidate.id != "_"
                    ):
                        self.assignments.append((line, candidate.id))

            def visit_Assign(self, node: ast.Assign) -> None:
                """Record plain local-name assignment targets."""
                target: ast.expr
                for target in node.targets:
                    self._record_target(target, node.lineno)
                self.generic_visit(node)

            def visit_For(self, node: ast.For) -> None:
                """Record an ordinary loop target."""
                self._record_target(node.target, node.lineno)
                self.generic_visit(node)

            def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
                """Record an asynchronous loop target."""
                self._record_target(node.target, node.lineno)
                self.generic_visit(node)

            def visit_With(self, node: ast.With) -> None:
                """Record context-manager binding targets."""
                item: ast.withitem
                for item in node.items:
                    if item.optional_vars is not None:
                        self._record_target(item.optional_vars, node.lineno)
                self.generic_visit(node)

            def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
                """Record asynchronous context-manager binding targets."""
                item: ast.withitem
                for item in node.items:
                    if item.optional_vars is not None:
                        self._record_target(item.optional_vars, node.lineno)
                self.generic_visit(node)

            def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
                """Record an assignment-expression target."""
                self._record_target(node.target, node.lineno)
                self.generic_visit(node)

            def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
                """Record an exception-handler binding."""
                if node.name is not None and node.name != "_":
                    self.assignments.append((node.lineno, node.name))
                self.generic_visit(node)

        violations: list[str] = []
        path: Path
        module: ast.Module
        node: ast.AST
        for path, module in self._test_modules():
            for node in ast.walk(module):
                if not isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    continue
                visitor: TestAssignmentVisitor = TestAssignmentVisitor(node)
                visitor.visit(node)
                violations.extend(
                    f"{path}:{line}:{node.name}:{name}"
                    for line, name in visitor.assignments
                    if name not in visitor.annotated
                    and name not in visitor.nonlocal_names
                )
        self.assertEqual(violations, [])

    def test_public_symbols_have_one_owning_subpackage(self) -> None:
        """Forbid compatibility re-exports across subpackage surfaces.

        Confirms a public name appears in exactly one non-root subpackage
        ``__all__`` so moves cannot leave aliases or secondary import paths.

        Notes
        -----
        Reads literal ``__all__`` entries from each first-level subpackage and
        reports names claimed by more than one owner.
        """
        owners: dict[str, list[str]] = {}
        path: Path
        module: ast.Module
        node: ast.stmt
        entry: ast.expr
        for path, module in self._production_modules():
            if path.name != "__init__.py" or path.parent.name == "diffpes":
                continue
            for node in module.body:
                target: ast.expr | None = None
                if isinstance(node, ast.Assign) and len(node.targets) == 1:
                    target = node.targets[0]
                elif isinstance(node, ast.AnnAssign):
                    target = node.target
                if (
                    not isinstance(target, ast.Name)
                    or target.id != "__all__"
                    or not isinstance(node.value, (ast.List, ast.Tuple))
                ):
                    continue
                for entry in node.value.elts:
                    if isinstance(entry, ast.Constant) and isinstance(
                        entry.value, str
                    ):
                        owners.setdefault(entry.value, []).append(
                            path.parent.name
                        )
        violations: list[str] = [
            f"{name}:{sorted(subpackages)}"
            for name, subpackages in sorted(owners.items())
            if len(subpackages) > 1
        ]
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
