"""Provide the program-wide differentiability gate.

Extended Summary
----------------
Every differentiability gate in the diffpes plan series calls
``gradient_gate`` or ``assert_grad_matches_fd`` from this module. Ad-hoc finite
differences in a gate are a review-blocking defect. A failure of this harness
on a physics function is a physics failure: the Fisher matrix is built from
the gradients this module certifies, and a spuriously zero parameter-Jacobian
column removes the corresponding Fisher row and column.

The harness combines JAX's randomized directional checks with scale-aware,
elementwise central differences and explicit zero-gradient tripwires. Complex
leaves follow JAX's complex-to-real Wirtinger convention.

Notes
-----
The finite-difference scaling follows Nocedal and Wright, *Numerical
Optimization*, section 8.1. The complex-gradient convention follows Martins
et al. (2003) and the JAX advanced-autodiff cookbook.
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Any, Callable, Optional
from jax import test_util
from jaxtyping import Array, Complex, Float, PRNGKeyArray, PyTree, jaxtyped

from diffpes.types import ScalarFloat
from tests._assertions import assert_tree_finite
from tests._types import GradRegime, ScalarLoss

RTOL_LADDER: dict[GradRegime, float] = {
    "smooth": 1e-6,
    "stiff": 1e-5,
    "singular": 1e-4,
}
EPS_F64: float = 2.220446049250313e-16


@jaxtyped(typechecker=beartype)
def fd_step(
    theta: Float[Array, "..."], *, scale_floor: ScalarFloat = 1e-3
) -> Float[Array, "..."]:
    """Calculate a scale-aware central-finite-difference step.

    Uses ``EPS_F64**(1/3) * maximum(abs(theta), scale_floor)`` elementwise,
    which balances truncation and round-off error for central differences.
    """
    step: Float[Array, "..."] = EPS_F64 ** (1.0 / 3.0) * jnp.maximum(
        jnp.abs(theta), scale_floor
    )
    return step


def _path_name(path: tuple[object, ...]) -> str:
    """Render a JAX key path in the stable tree-path notation."""
    path_name: str = jax.tree_util.keystr(path)
    return path_name


def _as_jax_arrays(tree: PyTree) -> PyTree:
    """Normalize numerical-check inputs before runtime type validation."""
    normalized: PyTree = jax.tree.map(jnp.asarray, tree)
    return normalized


def _central_leaf_grad(
    jitted_fn: ScalarLoss,
    treedef: jax.tree_util.PyTreeDef,
    leaves: list[Array],
    leaf_index: int,
    scale_floor: ScalarFloat,
) -> Array:
    """Differentiate one numerical leaf by elementwise central differences."""
    leaf: Array = jnp.asarray(leaves[leaf_index])
    steps: Array = fd_step(jnp.real(leaf), scale_floor=scale_floor)
    flat_leaf: Array = jnp.ravel(leaf)
    flat_steps: Array = jnp.ravel(steps)
    basis: Array = jnp.eye(flat_leaf.size, dtype=flat_leaf.dtype)

    def evaluate(delta: Array) -> Float[Array, ""]:
        perturbed_leaves: list[Array] = list(leaves)
        perturbed_leaves[leaf_index] = jnp.reshape(
            flat_leaf + delta, leaf.shape
        )
        perturbed_tree: PyTree = jax.tree_util.tree_unflatten(
            treedef, perturbed_leaves
        )
        value: Float[Array, ""] = jitted_fn(perturbed_tree)
        return value

    real_deltas: Array = basis * flat_steps[:, None]
    real_gradient: Array = jax.vmap(
        lambda delta: (evaluate(delta) - evaluate(-delta)) / (2.0 * flat_steps)
    )(real_deltas)
    real_diagonal: Array = jnp.diag(real_gradient)
    if not jnp.issubdtype(leaf.dtype, jnp.complexfloating):
        gradient: Array = jnp.reshape(real_diagonal, leaf.shape)
        return gradient

    imaginary_deltas: Array = 1j * basis * flat_steps[:, None]
    imaginary_gradient: Array = jax.vmap(
        lambda delta: (evaluate(delta) - evaluate(-delta)) / (2.0 * flat_steps)
    )(imaginary_deltas)
    imaginary_diagonal: Array = jnp.diag(imaginary_gradient)
    gradient = jnp.reshape(real_diagonal - 1j * imaginary_diagonal, leaf.shape)
    return gradient


@jaxtyped(typechecker=beartype)
def central_fd_grad(
    fn: ScalarLoss, theta: PyTree, *, scale_floor: ScalarFloat = 1e-3
) -> PyTree:
    """Calculate an elementwise central-FD gradient over a numerical PyTree.

    Real leaves use symmetric perturbations with :func:`fd_step`. Complex
    leaves separately perturb real and imaginary components and combine them
    as ``d/dRe - 1j*d/dIm``, matching JAX's complex-to-real convention. The
    cost is two forward evaluations per real parameter and four per complex
    parameter, so this helper is restricted to toy-model gates.
    """
    leaves: list[Any]
    treedef: jax.tree_util.PyTreeDef
    leaves, treedef = jax.tree_util.tree_flatten(theta)
    array_leaves: list[Array] = [jnp.asarray(leaf) for leaf in leaves]
    jitted_fn: ScalarLoss = jax.jit(fn)
    gradient_leaves: list[Array] = [
        _central_leaf_grad(
            jitted_fn, treedef, array_leaves, index, scale_floor
        )
        for index in range(len(array_leaves))
    ]
    gradient: PyTree = jax.tree_util.tree_unflatten(treedef, gradient_leaves)
    return gradient


@jaxtyped(typechecker=beartype)
def assert_grad_matches_fd(
    fn: ScalarLoss,
    theta: PyTree,
    *,
    regime: GradRegime = "smooth",
    atol: Optional[ScalarFloat] = None,
    scale_floor: ScalarFloat = 1e-3,
    modes: tuple[str, ...] = ("fwd", "rev"),
) -> None:
    """Assert autodiff agrees with directional and elementwise FD checks.

    The relative tolerance is selected from :data:`RTOL_LADDER`. If ``atol``
    is omitted, it is estimated from the central-FD round-off floor
    ``EPS_F64**(2/3) * max(1, abs(fn(theta))) / median(h)``. Failures identify
    the exact PyTree leaf path and largest absolute discrepancy.
    """
    step_leaves: list[Array] = [
        fd_step(jnp.real(jnp.asarray(leaf)), scale_floor=scale_floor)
        for leaf in jax.tree.leaves(theta)
    ]
    median_step: Float[Array, ""] = jnp.median(
        jnp.concatenate([jnp.ravel(step) for step in step_leaves])
    )
    relative_tolerance: float = RTOL_LADDER[regime]
    value: Float[Array, ""] = fn(theta)
    absolute_tolerance: ScalarFloat = (
        EPS_F64 ** (2.0 / 3.0) * jnp.maximum(1.0, jnp.abs(value)) / median_step
        if atol is None
        else atol
    )

    def checked_fn(candidate: PyTree) -> Float[Array, ""]:
        normalized: PyTree = _as_jax_arrays(candidate)
        checked_value: Float[Array, ""] = fn(normalized)
        return checked_value

    test_util.check_grads(
        checked_fn,
        (theta,),
        order=1,
        modes=modes,
        eps=float(median_step),
        atol=float(absolute_tolerance),
        rtol=relative_tolerance,
    )
    automatic: PyTree = jax.grad(fn)(theta)
    finite_difference: PyTree = central_fd_grad(
        fn, theta, scale_floor=scale_floor
    )
    automatic_paths: list[tuple[tuple[object, ...], Array]]
    automatic_treedef: jax.tree_util.PyTreeDef
    automatic_paths, automatic_treedef = jax.tree_util.tree_flatten_with_path(
        automatic
    )
    finite_leaves: list[Array]
    finite_treedef: jax.tree_util.PyTreeDef
    finite_leaves, finite_treedef = jax.tree_util.tree_flatten(
        finite_difference
    )
    if automatic_treedef != finite_treedef:
        raise AssertionError("autodiff and finite-difference trees differ")
    path: tuple[object, ...]
    actual: Array
    expected: Array
    for (path, actual), expected in zip(
        automatic_paths, finite_leaves, strict=True
    ):
        tolerance: Array = absolute_tolerance + relative_tolerance * jnp.abs(
            expected
        )
        difference: Array = jnp.abs(actual - expected)
        if not bool(jnp.all(difference <= tolerance)):
            message: str = (
                f"gradient mismatch at {_path_name(path)}: "
                f"max_abs_error={float(jnp.max(difference)):.6e}, "
                f"atol={float(absolute_tolerance):.6e}, "
                f"rtol={relative_tolerance:.6e}"
            )
            raise AssertionError(message)


@jaxtyped(typechecker=beartype)
def assert_nonzero_grad(
    fn: ScalarLoss,
    theta: PyTree,
    *,
    sensitive_paths: Optional[tuple[str, ...]] = None,
    min_norm: ScalarFloat = 1e-12,
) -> None:
    """Assert every selected gradient leaf has physically useful sensitivity.

    By default every leaf is checked. ``sensitive_paths`` selects exact JAX
    key-path strings, and each selected leaf must have Euclidean norm strictly
    greater than ``min_norm``.
    """
    gradient: PyTree = jax.grad(fn)(theta)
    path_leaves: list[tuple[tuple[object, ...], Array]]
    path_leaves, _ = jax.tree_util.tree_flatten_with_path(gradient)
    available_paths: set[str] = {_path_name(path) for path, _ in path_leaves}
    selected_paths: set[str] = (
        available_paths if sensitive_paths is None else set(sensitive_paths)
    )
    missing_paths: set[str] = selected_paths - available_paths
    if missing_paths:
        message: str = (
            f"unknown sensitive gradient paths: {sorted(missing_paths)}"
        )
        raise ValueError(message)
    path: tuple[object, ...]
    leaf: Array
    for path, leaf in path_leaves:
        path_name: str = _path_name(path)
        if path_name in selected_paths:
            norm: Float[Array, ""] = jnp.linalg.norm(jnp.ravel(leaf))
            if not bool(norm > min_norm):
                message = (
                    f"gradient at {path_name} has norm {float(norm):.6e}; "
                    f"required > {float(min_norm):.6e}"
                )
                raise AssertionError(message)


@jaxtyped(typechecker=beartype)
def gradient_gate(
    fn: ScalarLoss,
    theta: PyTree,
    *,
    regime: GradRegime = "smooth",
    sensitive_paths: Optional[tuple[str, ...]] = None,
    **kwargs: Any,
) -> None:
    """Run finite, finite-difference, and nonzero gradient checks together.

    This is the single entry point used by sibling-plan differentiability
    gates. Keyword arguments are forwarded to :func:`assert_grad_matches_fd`.
    """
    gradient: PyTree = jax.grad(fn)(theta)
    assert_tree_finite(gradient)
    assert_grad_matches_fd(fn, theta, regime=regime, **kwargs)
    assert_nonzero_grad(fn, theta, sensitive_paths=sensitive_paths)


@jaxtyped(typechecker=beartype)
def random_generic_complex(
    key: PRNGKeyArray,
    shape: tuple[int, ...],
    *,
    scale: ScalarFloat = 1.0,
) -> Complex[Array, "..."]:
    """Generate generic complex data with asymmetric independent components.

    Real and imaginary components use independent normal draws at scales
    ``scale`` and ``0.7 * scale``. The asymmetry prevents conjugation errors
    from passing accidentally through Hermitian or equal-component inputs.
    """
    real_key: PRNGKeyArray
    imaginary_key: PRNGKeyArray
    real_key, imaginary_key = jax.random.split(key)
    real_part: Float[Array, "..."] = scale * jax.random.normal(real_key, shape)
    imaginary_part: Float[Array, "..."] = (
        0.7 * scale * jax.random.normal(imaginary_key, shape)
    )
    values: Complex[Array, "..."] = real_part + 1j * imaginary_part
    return values


@jaxtyped(typechecker=beartype)
def complex_step_derivative(
    fn: Callable[[Float[Array, "..."]], Float[Array, "..."]],
    x: Float[Array, "..."],
    *,
    h: ScalarFloat = 1e-20,
) -> Float[Array, "..."]:
    """Calculate a complex-step derivative for a holomorphic sub-block.

    Evaluates ``imag(fn(x + 1j*h)) / h``. An identically real result raises
    because it signals a non-holomorphic operation such as conjugation,
    absolute value, or real-part extraction. This method is never valid across
    a modulus-squared operation.
    """
    complex_value: Array = fn(x.astype(jnp.complex128) + 1j * h)
    imaginary_part: Array = jnp.imag(complex_value)
    if bool(jnp.all(imaginary_part == 0.0)):
        message: str = (
            "complex-step output has zero imaginary part; the function may "
            "be non-holomorphic"
        )
        raise ValueError(message)
    derivative: Float[Array, "..."] = imaginary_part / h
    return derivative
