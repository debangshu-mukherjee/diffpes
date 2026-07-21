"""Certify differentiable DiffPES forward-model executions.

Extended Summary
----------------
The certification package combines static scientific identity and provenance
with numerical evidence computed through the actual JAX forward program.
Canonical records and CRC32 checksums provide non-security bookkeeping at the
I/O boundary; they do not establish physical validity.  Runtime claims,
derivative checks, dependency maps, sensitivities, and information spectra are
JAX-native numerical carriers.

The submodules are organized as follows:

- :mod:`canonical`
    Deterministic scientific-record representations.
- :mod:`checksums`
    Streaming non-security consistency checksums.
- :mod:`checks`
    Stable pure-JAX scientific predicate registry.
- :mod:`contracts`
    Composable semantic and information-loss contracts.
- :mod:`registry`
    Stable model and transformation identities.
- :mod:`provenance`
    Artifact lineage and claim-invalidation propagation.
- :mod:`dependencies`
    JAXPR dependencies, JVP/VJP sensitivities, and information spectra.
- :mod:`evidence`
    Continuous numerical evidence and derivative checks.
- :mod:`policy`
    Declarative cumulative certification levels.
- :mod:`execution`
    Prepared and compiled certified forward execution.
- :mod:`inspect`
    Human-readable certificate summaries and differences.
"""

from .canonical import (
    CANONICAL_JSON_VERSION,
    CANONICAL_PYTREE_VERSION,
    CanonicalizationError,
    canonical_json,
    canonical_pytree,
    iter_canonical_pytree_chunks,
)
from .checks import CheckFunction, get_check, list_checks, register_check
from .checksums import (
    CHECKSUM_ALGORITHM,
    CHECKSUM_FORMAT_VERSION,
    artifact_ref,
    checksum_bytes,
    checksum_chunks,
    checksum_file,
    checksum_pytree,
    parse_checksum,
    result_checksum,
    semantic_checksum,
)
from .contracts import (
    CompositionReport,
    ContractError,
    TransformationContract,
    compose_transformations,
    make_transformation_contract,
    validate_composition,
    validate_contract,
)
from .dependencies import (
    dependency_map,
    information_spectrum,
    linearized_forward,
    sensitivity_map,
)
from .evidence import (
    derivative_evidence,
    evaluate_claim,
    evaluate_domain,
    evaluate_evidence,
)
from .execution import (
    certify_forward,
    prepare_certification,
    verify_certificate,
)
from .inspect import (
    CertificateDiff,
    diff_certificates,
    explain_claim,
    summarize_certificate,
)
from .models import (
    TB_RADIAL_MODEL_ID,
    TB_RADIAL_MODEL_VERSION,
    execute_tb_radial,
    register_builtin_models,
    tb_radial_model_spec,
)
from .policy import POLICY_IDS, achieved_levels, evaluate_policy
from .provenance import (
    InformationState,
    ProvenanceError,
    ProvenanceGraph,
    ProvenanceReport,
    build_provenance,
    effective_information,
    invalidated_claims,
    lineage,
    validate_provenance,
)
from .registry import (
    RegisteredModel,
    RegisteredTransformation,
    RegistryError,
    RegistryReport,
    RegistrySnapshot,
    freeze_registry,
    get_model,
    get_transformation,
    list_models,
    list_registered_models,
    list_transformations,
    register_model,
    register_transformation,
    registry_snapshot,
    validate_registry,
)

__all__: list[str] = [
    "CANONICAL_JSON_VERSION",
    "CANONICAL_PYTREE_VERSION",
    "CHECKSUM_ALGORITHM",
    "CHECKSUM_FORMAT_VERSION",
    "CheckFunction",
    "POLICY_IDS",
    "CanonicalizationError",
    "CertificateDiff",
    "CompositionReport",
    "ContractError",
    "InformationState",
    "ProvenanceError",
    "ProvenanceGraph",
    "ProvenanceReport",
    "RegisteredModel",
    "RegisteredTransformation",
    "RegistryError",
    "RegistryReport",
    "RegistrySnapshot",
    "TB_RADIAL_MODEL_ID",
    "TB_RADIAL_MODEL_VERSION",
    "TransformationContract",
    "achieved_levels",
    "artifact_ref",
    "build_provenance",
    "canonical_json",
    "canonical_pytree",
    "certify_forward",
    "checksum_bytes",
    "checksum_chunks",
    "checksum_file",
    "checksum_pytree",
    "compose_transformations",
    "dependency_map",
    "derivative_evidence",
    "diff_certificates",
    "effective_information",
    "evaluate_claim",
    "evaluate_domain",
    "evaluate_evidence",
    "evaluate_policy",
    "execute_tb_radial",
    "explain_claim",
    "freeze_registry",
    "get_check",
    "get_model",
    "get_transformation",
    "information_spectrum",
    "invalidated_claims",
    "iter_canonical_pytree_chunks",
    "lineage",
    "linearized_forward",
    "list_models",
    "list_checks",
    "list_registered_models",
    "list_transformations",
    "make_transformation_contract",
    "parse_checksum",
    "prepare_certification",
    "register_model",
    "register_builtin_models",
    "register_check",
    "register_transformation",
    "registry_snapshot",
    "result_checksum",
    "semantic_checksum",
    "sensitivity_map",
    "summarize_certificate",
    "tb_radial_model_spec",
    "validate_composition",
    "validate_contract",
    "validate_provenance",
    "validate_registry",
    "verify_certificate",
]
