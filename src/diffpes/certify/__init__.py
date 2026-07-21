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
    Canonical scientific-record representations for certification.
- :mod:`checksums`
    Non-security consistency checksums for scientific records.
- :mod:`checks`
    Register pure JAX scientific certification checks.
- :mod:`contracts`
    Semantic contracts for composable certified transformations.
- :mod:`registry`
    Deterministic registries for certified models and transformations.
- :mod:`provenance`
    Artifact lineage and semantic information-loss graphs.
- :mod:`dependencies`
    Trace differentiable information flow through forward models.
- :mod:`evidence`
    Build differentiable evidence for certified forward models.
- :mod:`policy`
    Evaluate cumulative scientific-certification policies.
- :mod:`execution`
    Execute JAX-native certified forward models.
- :mod:`inspect`
    Human-readable inspection of forward-model certificates.
- :mod:`models`
    Register built-in certified DiffPES forward models.

Routine Listings
----------------
:func:`achieved_levels`
    Return certification level names achieved by a concrete report.
:func:`artifact_ref`
    Build separate byte, normalized-content, and semantic identities.
:func:`build_provenance`
    Build a deterministic provenance DAG and propagate information.
:func:`canonical_json`
    Return deterministic typed JSON bytes for ``value``.
:func:`canonical_pytree`
    Return canonical bytes for a supported carrier or PyTree.
:func:`certify_forward`
    Execute a prepared model and produce a certified JAX PyTree.
:func:`checksum_bytes`
    Return a non-security consistency checksum for ``data``.
:func:`checksum_chunks`
    Compute a consistency checksum over consecutive byte chunks.
:func:`checksum_file`
    Stream exact file bytes into a consistency checksum.
:func:`checksum_pytree`
    Stream a canonical carrier into a consistency checksum.
:func:`compose_transformations`
    Compose contracts and raise if any requirement is unsatisfied.
:func:`dependency_map`
    Trace leaf-level structural and local numerical dependencies.
:func:`derivative_evidence`
    Compare JAX information flow with batched finite differences.
:func:`diff_certificates`
    Compare two certificates by scientific meaning and record class.
:func:`effective_information`
    Return propagated semantics, losses, and invalidations for one node.
:func:`evaluate_claim`
    Evaluate a numerical claim and preserve residual and margin leaves.
:func:`evaluate_domain`
    Evaluate a symmetric domain predicate around a reference value.
:func:`evaluate_evidence`
    Compare measured values with an external numerical reference.
:func:`evaluate_policy`
    Derive cumulative certification outcomes from numerical claims.
:func:`execute_tb_radial`
    Execute the radial ARPES model from one certification input PyTree.
:func:`explain_claim`
    Explain one claim and the numerical evidence supporting it.
:func:`freeze_registry`
    Prevent later registration and return the final immutable snapshot.
:func:`get_check`
    Resolve a registered JAX certification check.
:func:`get_model`
    Resolve an exact registered model.
:func:`get_transformation`
    Resolve an exact registered transformation contract.
:func:`information_spectrum`
    Estimate the leading local information spectrum matrix-free.
:func:`invalidated_claims`
    Return every claim invalidated at or upstream of one output.
:func:`iter_canonical_pytree_chunks`
    Yield canonical carrier bytes in bounded chunks.
:func:`lineage`
    Return the transitive parent-node lineage of one output.
:func:`linearized_forward`
    Evaluate a forward model and retain its JVP linearization.
:func:`list_checks`
    List registered checks in deterministic identity order.
:func:`list_models`
    Return model specifications in deterministic identity order.
:func:`list_registered_models`
    Return an immutable deterministic snapshot including executors.
:func:`list_transformations`
    Return transformation contracts in deterministic identity order.
:func:`parse_checksum`
    Parse and validate one checksum string.
:func:`prepare_certification`
    Resolve static scientific records before compiled execution.
:func:`register_builtin_models`
    Register built-in models and information-loss transformations.
:func:`register_check`
    Register a stable predicate identity and pure JAX callable.
:func:`register_model`
    Register an exact model identity once.
:func:`register_transformation`
    Register an exact transformation contract once.
:func:`registry_snapshot`
    Return one internally consistent immutable registry snapshot.
:func:`result_checksum`
    Identify a result under a declared numerical configuration.
:func:`semantic_checksum`
    Identify content together with its declared scientific meaning.
:func:`sensitivity_map`
    Measure scaled JVP sensitivities for a batch of tangent directions.
:func:`summarize_certificate`
    Return a deterministic human-readable certificate summary.
:func:`tb_radial_model_spec`
    Return the stable scientific specification for radial ARPES.
:func:`validate_composition`
    Validate and conservatively compose transformation semantics.
:func:`validate_contract`
    Return structural errors for a raw or deserialized contract.
:func:`validate_provenance`
    Independently re-evaluate graph structure and derived state.
:func:`validate_registry`
    Recompute registry structure and consistency checksums.
:func:`verify_certificate`
    Re-evaluate numerical claim and policy consistency without a rerun.
"""

from .canonical import (
    canonical_json,
    canonical_pytree,
    iter_canonical_pytree_chunks,
)
from .checks import get_check, list_checks, register_check
from .checksums import (
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
    compose_transformations,
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
from .inspect import diff_certificates, explain_claim, summarize_certificate
from .models import (
    execute_tb_radial,
    register_builtin_models,
    tb_radial_model_spec,
)
from .policy import achieved_levels, evaluate_policy
from .provenance import (
    build_provenance,
    effective_information,
    invalidated_claims,
    lineage,
    validate_provenance,
)
from .registry import (
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
