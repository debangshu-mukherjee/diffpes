"""Generate deterministic WP6.1 forward-model regression references.

Extended Summary
----------------
Builds the three CPU/x64 behavioral baselines required before the Equinox
migration. The archives deliberately pin current behavior rather than physics
truth and use deterministic ZIP metadata so unchanged arrays produce identical
files and SHA-256 digests.
"""

import hashlib
import io
import platform
import sys
import zipfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import diffpes

_REPOSITORY_ROOT: Path = Path(__file__).resolve().parents[1]
_TESTS_DIRECTORY: Path = _REPOSITORY_ROOT / "tests"
sys.path.insert(0, str(_REPOSITORY_ROOT))

from diffpes.simul import simulate_novice, simulate_tb_radial  # noqa: E402
from diffpes.types import (  # noqa: E402
    ArpesSpectrum,
    DiagonalizedBands,
    SlaterParams,
    make_slater_params,
)
from tests._factories import (  # noqa: E402
    toy_band_structure,
    toy_chain_diagonalized,
    toy_graphene_diagonalized,
    toy_orbital_projection,
    toy_polarization_config,
    toy_simulation_params,
    toy_slater_params,
)

_REFERENCE_DIRECTORY: Path = (
    _TESTS_DIRECTORY / "test_diffpes" / "_reference_data"
)
_SEED: int = 20260713
_FIXED_ZIP_TIME: tuple[int, int, int, int, int, int] = (2026, 7, 19, 0, 0, 0)


def _spectrum_arrays(spectrum: ArpesSpectrum) -> dict[str, np.ndarray]:
    """Convert one spectrum to named NumPy reference arrays."""
    arrays: dict[str, np.ndarray] = {
        "leaf_000_intensity": np.asarray(spectrum.intensity),
        "leaf_001_energy_axis": np.asarray(spectrum.energy_axis),
    }
    return arrays


def _tb_payload(
    bands: DiagonalizedBands,
    slater: SlaterParams,
) -> dict[str, np.ndarray]:
    """Evaluate one tight-binding radial spectrum and its zeta gradient."""
    params = toy_simulation_params(fidelity=512)
    polarization = toy_polarization_config()
    spectrum: ArpesSpectrum = simulate_tb_radial(
        bands,
        slater,
        params,
        polarization,
    )

    def intensity_sum(zeta: jax.Array) -> jax.Array:
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
        total: jax.Array = jnp.sum(varied_spectrum.intensity)
        return total

    zeta_gradient: jax.Array = jax.grad(intensity_sum)(slater.zeta)
    arrays: dict[str, np.ndarray] = _spectrum_arrays(spectrum)
    arrays["leaf_002_intensity_sum"] = np.asarray(jnp.sum(spectrum.intensity))
    arrays["leaf_003_zeta_gradient"] = np.asarray(zeta_gradient)
    return arrays


def build_payloads() -> dict[str, dict[str, np.ndarray]]:
    """Build all three fixed-seed WP6.1 reference payloads."""
    key: jax.Array = jax.random.key(_SEED)
    novice_spectrum: ArpesSpectrum = simulate_novice(
        toy_band_structure(key),
        toy_orbital_projection(key),
        toy_simulation_params(fidelity=512),
    )
    _, graphene_bands = toy_graphene_diagonalized(n_k=12)
    _, chain_bands = toy_chain_diagonalized(n_k=16)
    slater: SlaterParams = toy_slater_params()
    payloads: dict[str, dict[str, np.ndarray]] = {
        "novice_toy": _spectrum_arrays(novice_spectrum),
        "tb_radial_graphene": _tb_payload(graphene_bands, slater),
        "tb_radial_chain": _tb_payload(chain_bands, slater),
    }
    return payloads


def _write_deterministic_npz(
    path: Path,
    arrays: dict[str, np.ndarray],
) -> None:
    """Write an NPZ with stable member order, timestamps, and permissions."""
    with zipfile.ZipFile(path, mode="w") as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.save(buffer, arrays[name], allow_pickle=False)
            member = zipfile.ZipInfo(f"{name}.npy", _FIXED_ZIP_TIME)
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o100644 << 16
            archive.writestr(member, buffer.getvalue())


def _sha256(path: Path) -> str:
    """Calculate the SHA-256 digest of one generated artifact."""
    digest: str = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def _manifest(payloads: dict[str, dict[str, np.ndarray]]) -> str:
    """Render provenance, array metadata, and hashes for all artifacts."""
    lines: list[str] = [
        "# WP6.1 regression-reference manifest",
        "",
        "> These files pin pre-refactor behavior, not correctness. They",
        "> intentionally include known physics defects documented by Plan 01.",
        "> Regenerate only with a stated physics or migration justification.",
        "",
        "- Generation date: 2026-07-19",
        f"- Seed: `{_SEED}`",
        "- Device policy: CPU, JAX x64 enabled",
        f"- Platform: `{platform.platform()}`",
        f"- Python: `{platform.python_version()}`",
        f"- DiffPES: `{diffpes.__version__}`",
        f"- JAX: `{jax.__version__}`",
        f"- NumPy: `{np.__version__}`",
        "",
        "## Factory calls",
        "",
        "- `novice_toy`: `simulate_novice(toy_band_structure(key), "
        "toy_orbital_projection(key), "
        "toy_simulation_params(fidelity=512))`",
        "- `tb_radial_graphene`: `simulate_tb_radial("
        "toy_graphene_diagonalized(n_k=12)[1], toy_slater_params(), "
        "toy_simulation_params(fidelity=512), "
        "toy_polarization_config())`, plus intensity sum and zeta gradient",
        "- `tb_radial_chain`: `simulate_tb_radial("
        "toy_chain_diagonalized(n_k=16)[1], toy_slater_params(), "
        "toy_simulation_params(fidelity=512), "
        "toy_polarization_config())`, plus intensity sum and zeta gradient",
        "",
        "## Artifacts",
        "",
    ]
    for artifact_name, arrays in payloads.items():
        artifact_path: Path = _REFERENCE_DIRECTORY / f"{artifact_name}.npz"
        lines.extend(
            [
                f"### `{artifact_name}.npz`",
                "",
                f"- SHA-256: `{_sha256(artifact_path)}`",
                "- Arrays:",
            ]
        )
        for array_name in sorted(arrays):
            array: np.ndarray = arrays[array_name]
            lines.append(
                f"  - `{array_name}`: shape `{array.shape}`, dtype "
                f"`{array.dtype}`"
            )
        lines.append("")
    manifest: str = "\n".join(lines)
    return manifest


def main() -> None:
    """Generate, verify, and document the deterministic references."""
    _REFERENCE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    first_payloads: dict[str, dict[str, np.ndarray]] = build_payloads()
    second_payloads: dict[str, dict[str, np.ndarray]] = build_payloads()
    for artifact_name, first_arrays in first_payloads.items():
        second_arrays: dict[str, np.ndarray] = second_payloads[artifact_name]
        for array_name, first_array in first_arrays.items():
            np.testing.assert_allclose(
                first_array,
                second_arrays[array_name],
                rtol=1e-12,
                atol=0.0,
            )
        artifact_path: Path = _REFERENCE_DIRECTORY / f"{artifact_name}.npz"
        _write_deterministic_npz(artifact_path, first_arrays)
    manifest_path: Path = _REFERENCE_DIRECTORY / "MANIFEST.md"
    manifest_path.write_text(_manifest(first_payloads) + "\n")


if __name__ == "__main__":
    main()
