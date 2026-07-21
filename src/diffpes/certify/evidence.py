"""Build differentiable evidence for certified forward models.

Extended Summary
----------------
Turns continuous JAX measurements into explicit evidence and claim PyTrees.
Boolean outcomes are derived only after residuals and margins are retained, so
optimization and experiment-design code can differentiate the meaningful
quantities beneath a certification threshold.

Routine Listings
----------------
:func:`derivative_evidence`
    Compare JVP derivatives with batched central differences and VJPs.
:func:`evaluate_claim`
    Evaluate a numerical claim while preserving its continuous residual.
:func:`evaluate_domain`
    Evaluate a validity-domain predicate and retain its signed margin.
:func:`evaluate_evidence`
    Compare measured values with an external numerical reference.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Callable
from jaxtyping import Array, Float, PyTree, jaxtyped

from diffpes.types.certification import (
    CertificationClaim,
    DerivativeEvidence,
    DomainResult,
    EvidenceRef,
    make_certification_claim,
    make_derivative_evidence,
    make_domain_result,
    make_evidence_ref,
)

from .dependencies import information_spectrum


def _as_vector(value: Array) -> Array:
    """Flatten a numerical value to a one-dimensional JAX array."""
    vector: Array = jnp.ravel(jnp.asarray(value, dtype=jnp.float64))
    return vector


@jaxtyped(typechecker=beartype)
def evaluate_evidence(
    evidence_id: str,
    method_id: str,
    measured: Array,
    reference: Array,
    tolerance: Array,
    *,
    artifact_refs: tuple[str, ...] = (),
    source_type: str = "external_reference",
    independent: bool = True,
) -> EvidenceRef:
    """Compare measured values with an external numerical reference."""
    measured_array: Array = _as_vector(measured)
    reference_array: Array = _as_vector(reference)
    tolerance_array: Array = jnp.broadcast_to(
        _as_vector(tolerance), measured_array.shape
    )
    if measured_array.shape != reference_array.shape:
        msg: str = "measured and reference evidence arrays must agree"
        raise ValueError(msg)
    residual: Array = measured_array - reference_array
    evidence: EvidenceRef = make_evidence_ref(
        evidence_id=evidence_id,
        method_id=method_id,
        artifact_refs=artifact_refs,
        source_type=source_type,
        independent=independent,
        measured=measured_array,
        reference=reference_array,
        residual=residual,
        tolerance=tolerance_array,
    )
    return evidence


@jaxtyped(typechecker=beartype)
def evaluate_claim(
    claim_id: str,
    subject_id: str,
    predicate_id: str,
    measured: Array,
    reference: Array,
    tolerance: Array,
    *,
    evidence_ids: tuple[str, ...] = (),
    checked: bool = True,
    in_domain: bool = True,
    severity_code: int = 1,
) -> CertificationClaim:
    """Evaluate a numerical claim and preserve residual and margin leaves."""
    measured_array: Array = _as_vector(measured)
    reference_array: Array = _as_vector(reference)
    tolerance_array: Array = jnp.broadcast_to(
        _as_vector(tolerance), measured_array.shape
    )
    if measured_array.shape != reference_array.shape:
        msg: str = "claim measured and reference arrays must agree"
        raise ValueError(msg)
    residual: Array = measured_array - reference_array
    component_margin: Array = tolerance_array - jnp.abs(residual)
    margin: Array = jnp.min(component_margin)
    checked_array: Array = jnp.asarray(checked, dtype=jnp.bool_)
    domain_array: Array = jnp.asarray(in_domain, dtype=jnp.bool_)
    passed: Array = (
        checked_array & domain_array & jnp.all(component_margin >= 0)
    )
    claim: CertificationClaim = make_certification_claim(
        claim_id=claim_id,
        subject_id=subject_id,
        predicate_id=predicate_id,
        evidence_ids=evidence_ids,
        measured=measured_array,
        reference=reference_array,
        residual=residual,
        tolerance=tolerance_array,
        passed=passed,
        checked=checked_array,
        in_domain=domain_array,
        margin=margin,
        severity_code=jnp.asarray(severity_code, dtype=jnp.int32),
    )
    return claim


@jaxtyped(typechecker=beartype)
def evaluate_domain(
    predicate_id: str,
    measured: Array,
    reference: Array,
    tolerance: Array,
    *,
    checked: bool = True,
    severity_code: int = 1,
) -> DomainResult:
    """Evaluate a symmetric domain predicate around a reference value."""
    measured_array: Array = jnp.asarray(measured, dtype=jnp.float64)
    reference_array: Array = jnp.asarray(reference, dtype=jnp.float64)
    tolerance_array: Array = jnp.asarray(tolerance, dtype=jnp.float64)
    residual: Array = measured_array - reference_array
    margin: Array = tolerance_array - jnp.abs(residual)
    checked_array: Array = jnp.asarray(checked, dtype=jnp.bool_)
    passed: Array = checked_array & (margin >= 0)
    result: DomainResult = make_domain_result(
        predicate_id=predicate_id,
        measured=measured_array,
        reference=reference_array,
        residual=residual,
        tolerance=tolerance_array,
        margin=margin,
        passed=passed,
        checked=checked_array,
        in_domain=passed,
        severity_code=jnp.asarray(severity_code, dtype=jnp.int32),
    )
    return result


def _perturb(inputs: PyTree, direction: PyTree, step: Array) -> PyTree:
    """Add one scaled tangent PyTree to an input PyTree."""
    perturbed: PyTree = jax.tree.map(
        lambda value, tangent: value + step * tangent,
        inputs,
        direction,
    )
    return perturbed


@jaxtyped(typechecker=beartype)
def derivative_evidence(
    forward_fn: Callable[[PyTree], Float[Array, " n_output"]],
    inputs: PyTree,
    directions: PyTree,
    cotangents: Float[Array, "n_cotangent n_output"],
    *,
    input_paths: tuple[str, ...],
    output_projection_ids: tuple[str, ...],
    scales: Float[Array, " n_probe"],
    step: float = 6e-6,
    spectrum_rank: int = 8,
) -> DerivativeEvidence:
    """Compare JAX information flow with batched finite differences.

    The tangent tree has a leading probe axis on every leaf. The function
    reuses one ``jax.linearize`` result for all JVP probes, evaluates central
    differences with ``vmap``, and applies one retained VJP to all output
    cotangents. It additionally records a matrix-free local information
    spectrum.
    """
    output, pushforward = jax.linearize(forward_fn, inputs)
    del output
    jvp_values: Array = jax.vmap(pushforward)(directions)

    def finite_difference(direction: PyTree, local_step: Array) -> Array:
        plus: Array = forward_fn(_perturb(inputs, direction, local_step))
        minus: Array = forward_fn(_perturb(inputs, direction, -local_step))
        derivative: Array = (plus - minus) / (2.0 * local_step)
        return derivative

    steps: Array = step * jnp.maximum(jnp.abs(scales), 1e-3)
    reference: Array = jax.vmap(finite_difference)(directions, steps)
    residuals: Array = jvp_values - reference
    _, pullback = jax.vjp(forward_fn, inputs)

    def pull_one(cotangent: Array) -> Array:
        pulled: PyTree = pullback(cotangent)[0]
        values: list[Array] = []
        for leaf in jax.tree.leaves(pulled):
            array: Array = jnp.asarray(leaf)
            values.append(jnp.linalg.norm(jnp.ravel(jnp.real(array))))
            if jnp.iscomplexobj(array):
                values.append(jnp.linalg.norm(jnp.ravel(jnp.imag(array))))
        summaries: Array = jnp.asarray(values)
        return summaries

    vjp_values: Array = jax.vmap(pull_one)(cotangents)
    spectrum = information_spectrum(
        forward_fn,
        inputs,
        input_paths=input_paths,
        rank=spectrum_rank,
    )
    finite: Array = (
        jnp.all(jnp.isfinite(jvp_values))
        & jnp.all(jnp.isfinite(vjp_values))
        & jnp.all(jnp.isfinite(reference))
    )
    fd_correct: Array = jnp.allclose(
        jvp_values,
        reference,
        rtol=1e-6,
        atol=1e-9,
    )
    evidence: DerivativeEvidence = make_derivative_evidence(
        input_paths=input_paths,
        output_projection_ids=output_projection_ids,
        method="jax.linearize+jvp+vjp+central_fd",
        scales=scales,
        jvp_probes=jnp.reshape(jvp_values, (jvp_values.shape[0], -1)),
        vjp_probes=jnp.reshape(vjp_values, (vjp_values.shape[0], -1)),
        reference_derivatives=jnp.reshape(reference, (reference.shape[0], -1)),
        derivative_residuals=jnp.reshape(residuals, (residuals.shape[0], -1)),
        singular_values=spectrum.singular_values,
        effective_rank=spectrum.effective_rank,
        condition_estimate=spectrum.condition_estimate,
        finite=finite,
        fd_correct=fd_correct,
    )
    return evidence


__all__: list[str] = [
    "derivative_evidence",
    "evaluate_claim",
    "evaluate_domain",
    "evaluate_evidence",
]
