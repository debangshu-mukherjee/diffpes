"""Trace differentiable information flow through forward models.

Extended Summary
----------------
Provides JAX-native structural dependency, local sensitivity, retained
linearization, and matrix-free information-spectrum tools used by certified
forward executions. Structural flow is read from typed JAXPR, while numerical
flow is measured with JVP and VJP operations on the actual forward program.

Routine Listings
----------------
:func:`dependency_map`
    Trace leaf-level structural and local numerical dependencies.
:func:`information_spectrum`
    Estimate the leading singular spectrum without materializing a Jacobian.
:func:`linearized_forward`
    Evaluate a forward model and retain its reusable linearization.
:func:`sensitivity_map`
    Measure scaled JVP sensitivity from named inputs to output projections.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any, Callable, Optional
from jaxtyping import Array, Float, PyTree, jaxtyped

from diffpes.types.certification import (
    DependencyMap,
    InformationSpectrum,
    SensitivityMap,
    make_dependency_map,
    make_information_spectrum,
    make_sensitivity_map,
)
from diffpes.utils import pack_complex, unpack_complex


def _path_names(tree: PyTree) -> tuple[str, ...]:
    """Return stable JAX key-path names for all leaves."""
    path_leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    names: tuple[str, ...] = tuple(
        jax.tree_util.keystr(path) or "$" for path, _ in path_leaves
    )
    return names


def _structural_dependencies(
    forward_fn: Callable[[PyTree], PyTree], inputs: PyTree
) -> tuple[PyTree, Array]:
    """Propagate input-leaf dependency sets through a closed JAXPR."""
    output: PyTree = jax.eval_shape(forward_fn, inputs)
    closed = jax.make_jaxpr(forward_fn)(inputs)
    jaxpr = closed.jaxpr
    n_inputs: int = len(jaxpr.invars)
    dependency: dict[Any, frozenset[int]] = {
        var: frozenset((index,)) for index, var in enumerate(jaxpr.invars)
    }
    for constvar in jaxpr.constvars:
        dependency[constvar] = frozenset()

    def variable_dependencies(variable: Any) -> frozenset[int]:
        try:
            result: frozenset[int] = dependency.get(variable, frozenset())
        except TypeError:
            result = frozenset()
        return result

    for equation in jaxpr.eqns:
        incoming: frozenset[int] = frozenset().union(
            *(variable_dependencies(variable) for variable in equation.invars)
        )
        for variable in equation.outvars:
            dependency[variable] = incoming
    rows: list[Array] = []
    for variable in jaxpr.outvars:
        indices: frozenset[int] = dependency.get(variable, frozenset())
        rows.append(
            jnp.asarray(
                [index in indices for index in range(n_inputs)],
                dtype=jnp.bool_,
            )
        )
    structural: Array = jnp.stack(rows, axis=0)
    return output, structural


def _leaf_direction(inputs: PyTree, leaf_index: int) -> PyTree:
    """Build an all-ones tangent for one input leaf."""
    leaves, treedef = jax.tree_util.tree_flatten(inputs)
    tangent_leaves: list[Array] = [jnp.zeros_like(leaf) for leaf in leaves]
    tangent_leaves[leaf_index] = jnp.ones_like(leaves[leaf_index])
    tangent: PyTree = jax.tree_util.tree_unflatten(treedef, tangent_leaves)
    return tangent


@jaxtyped(typechecker=beartype)
def linearized_forward(
    forward_fn: Callable[[PyTree], PyTree], inputs: PyTree
) -> tuple[PyTree, Callable[[PyTree], PyTree]]:
    """Evaluate a forward model and retain its JVP linearization.

    Parameters
    ----------
    forward_fn : Callable
        Pure JAX forward function accepting one input PyTree.
    inputs : PyTree
        Evaluation point for the forward model.

    Returns
    -------
    output : PyTree
        Forward-model value at ``inputs``.
    pushforward : Callable
        Reusable linear map from input tangents to output tangents.
    """
    output, pushforward = jax.linearize(forward_fn, inputs)
    return output, pushforward


@jaxtyped(typechecker=beartype)
def dependency_map(
    model_id: str,
    forward_fn: Callable[[PyTree], PyTree],
    inputs: PyTree,
    *,
    threshold: float = 1e-12,
) -> DependencyMap:
    """Trace leaf-level structural and local numerical dependencies.

    Structural dependencies come from typed JAXPR variable flow. ``traced``
    records whether an all-ones tangent for an input leaf produces a local
    output-leaf response above ``threshold``. A false traced entry is local
    evidence only and is never interpreted as global independence.
    """
    abstract_output, structural = _structural_dependencies(forward_fn, inputs)
    output, pushforward = linearized_forward(forward_fn, inputs)
    del output
    n_inputs: int = len(jax.tree.leaves(inputs))
    numerical_rows: list[Array] = []
    for index in range(n_inputs):
        tangent: PyTree = _leaf_direction(inputs, index)
        response: PyTree = pushforward(tangent)
        activity: Array = jnp.asarray(
            [
                jnp.linalg.norm(jnp.ravel(jnp.asarray(leaf))) > threshold
                for leaf in jax.tree.leaves(response)
            ],
            dtype=jnp.bool_,
        )
        numerical_rows.append(activity)
    traced: Array = jnp.stack(numerical_rows, axis=0).T
    result: DependencyMap = make_dependency_map(
        model_id=model_id,
        input_paths=_path_names(inputs),
        output_paths=_path_names(abstract_output),
        structural=structural,
        traced=traced,
    )
    return result


@jaxtyped(typechecker=beartype)
def sensitivity_map(
    input_paths: tuple[str, ...],
    output_projection_ids: tuple[str, ...],
    forward_fn: Callable[[PyTree], Array],
    inputs: PyTree,
    directions: PyTree,
    scales: Float[Array, " n_input"],
    *,
    threshold: float = 1e-12,
) -> SensitivityMap:
    """Measure scaled JVP sensitivities for a batch of tangent directions.

    ``directions`` has the same tree structure as ``inputs`` and a leading
    probe axis on every leaf. The flattened output of ``forward_fn`` must
    correspond to ``output_projection_ids``.
    """
    _, pushforward = linearized_forward(forward_fn, inputs)
    responses: Array = jax.vmap(pushforward)(directions)
    response_array: Array = jnp.reshape(responses, (responses.shape[0], -1))
    scaled: Array = (response_array * scales[:, None]).T
    active: Array = jnp.abs(scaled) > threshold
    result: SensitivityMap = make_sensitivity_map(
        input_paths=input_paths,
        output_projection_ids=output_projection_ids,
        scales=scales,
        sensitivities=scaled,
        threshold=threshold,
        active=active,
    )
    return result


def _deterministic_subspace(size: int, rank: int, dtype: Any) -> Array:
    """Construct a deterministic full-rank starting subspace."""
    rows: Array = jnp.arange(1, size + 1, dtype=dtype)[:, None]
    cols: Array = jnp.arange(1, rank + 1, dtype=dtype)[None, :]
    initial: Array = jnp.sin(rows * cols) + jnp.cos(rows * (cols + 0.5))
    orthogonal, _ = jnp.linalg.qr(initial, mode="reduced")
    return orthogonal


def _element_paths(tree: PyTree) -> tuple[str, ...]:
    """Expand leaf paths to one stable name per scalar parameter."""
    path_leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    names: list[str] = []
    for path, leaf in path_leaves:
        base: str = jax.tree_util.keystr(path) or "$"
        array: Array = jnp.asarray(leaf)
        size: int = array.size
        components: tuple[str, ...] = (
            ("real", "imag") if jnp.iscomplexobj(array) else ("",)
        )
        for index in range(size):
            indexed: str = base if size == 1 else f"{base}[{index}]"
            names.extend(
                indexed if not component else f"{indexed}.{component}"
                for component in components
            )
    return tuple(names)


def _ravel_real_pytree(
    tree: PyTree,
) -> tuple[Array, Callable[[Array], PyTree]]:
    """Ravel a numerical PyTree in independent real coordinates."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    arrays: list[Array] = [jnp.asarray(leaf) for leaf in leaves]
    parts: list[Array] = []
    for array in arrays:
        if jnp.iscomplexobj(array):
            parts.append(jnp.ravel(pack_complex(array)))
        elif jnp.issubdtype(array.dtype, jnp.inexact):
            parts.append(jnp.ravel(array))
        else:
            msg: str = "information inputs must contain inexact array leaves"
            raise TypeError(msg)
    flat: Array = jnp.concatenate(parts) if parts else jnp.zeros(0)

    def unravel(vector: Array) -> PyTree:
        offset: int = 0
        rebuilt: list[Array] = []
        for array in arrays:
            size: int = array.size
            if jnp.iscomplexobj(array):
                count: int = 2 * size
                packed: Array = jnp.reshape(
                    vector[offset : offset + count], (*array.shape, 2)
                )
                rebuilt.append(unpack_complex(packed).astype(array.dtype))
            else:
                count = size
                rebuilt.append(
                    jnp.reshape(
                        vector[offset : offset + count], array.shape
                    ).astype(array.dtype)
                )
            offset += count
        result: PyTree = jax.tree_util.tree_unflatten(treedef, rebuilt)
        return result

    return flat, unravel


@jaxtyped(typechecker=beartype)
def information_spectrum(
    forward_fn: Callable[[PyTree], PyTree],
    inputs: PyTree,
    *,
    input_paths: Optional[tuple[str, ...]] = None,
    output_weights: Optional[Float[Array, " n_output"]] = None,
    rank: int = 8,
    iterations: int = 8,
    threshold: float = 1e-10,
) -> InformationSpectrum:
    """Estimate the leading local information spectrum matrix-free.

    Computes leading eigenpairs of ``J.T @ W @ J`` through retained JVP and
    VJP maps. The dense pixel-by-parameter Jacobian is never materialized.
    Singular values are square roots of the nonnegative information
    eigenvalues.
    """
    flat_inputs, unravel_inputs = _ravel_real_pytree(inputs)

    def flat_forward(flat: Array) -> Array:
        output: PyTree = forward_fn(unravel_inputs(flat))
        flat_output, _ = _ravel_real_pytree(output)
        return flat_output

    flat_output, pushforward = jax.linearize(flat_forward, flat_inputs)
    _, pullback = jax.vjp(flat_forward, flat_inputs)
    weights: Array = (
        jnp.ones_like(flat_output)
        if output_weights is None
        else jnp.asarray(output_weights, dtype=flat_output.real.dtype)
    )
    if weights.shape != flat_output.shape:
        msg: str = "output_weights must match the flattened forward output"
        raise ValueError(msg)
    effective_rank_limit: int = min(rank, flat_inputs.size, flat_output.size)
    if effective_rank_limit < 1:
        msg = "rank must be positive for non-empty inputs and outputs"
        raise ValueError(msg)

    def normal(vector: Array) -> Array:
        response: Array = pushforward(vector)
        cotangent: Array = weights * response
        pulled: Array = pullback(cotangent)[0]
        return jnp.real(pulled)

    def apply_columns(matrix: Array) -> Array:
        applied: Array = jax.vmap(normal, in_axes=1, out_axes=1)(matrix)
        return applied

    subspace: Array = _deterministic_subspace(
        flat_inputs.size, effective_rank_limit, flat_inputs.real.dtype
    )

    def iteration(_: Array, basis: Array) -> Array:
        updated: Array = apply_columns(basis)
        orthogonal, _ = jnp.linalg.qr(updated, mode="reduced")
        return orthogonal

    subspace = jax.lax.fori_loop(0, iterations, iteration, subspace)
    projected: Array = subspace.T @ apply_columns(subspace)
    eigenvalues, eigenvectors_small = jnp.linalg.eigh(projected)
    order: Array = jnp.argsort(eigenvalues)[::-1]
    eigenvalues = jnp.maximum(eigenvalues[order], 0.0)
    right_vectors: Array = (subspace @ eigenvectors_small[:, order]).T
    singular_values: Array = jnp.sqrt(eigenvalues)
    active: Array = singular_values > threshold
    effective_rank: Array = jnp.sum(active, dtype=jnp.int32)
    largest: Array = singular_values[0]
    smallest_active: Array = jnp.min(
        jnp.where(active, singular_values, jnp.inf)
    )
    condition: Array = jnp.where(
        effective_rank > 0,
        largest / smallest_active,
        jnp.inf,
    )
    paths: tuple[str, ...] = (
        _element_paths(inputs) if input_paths is None else input_paths
    )
    if len(paths) != flat_inputs.size:
        paths = _element_paths(inputs)
    result: InformationSpectrum = make_information_spectrum(
        input_paths=paths,
        singular_values=singular_values,
        right_singular_vectors=right_vectors,
        effective_rank=effective_rank,
        condition_estimate=condition,
        threshold=threshold,
    )
    return result


__all__: list[str] = [
    "dependency_map",
    "information_spectrum",
    "linearized_forward",
    "sensitivity_map",
]
