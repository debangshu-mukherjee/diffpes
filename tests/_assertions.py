"""Provide shared assertions for numerical test trees.

Extended Summary
----------------
Wraps Chex's PyTree assertions with the strict defaults used throughout the
test suite and loads behavioral regression arrays from the reference-data
directory established by WP6.1.
"""

from pathlib import Path

import chex
import jax
import numpy as np
from beartype import beartype
from jaxtyping import PyTree, jaxtyped

_REFERENCE_DIRECTORY: Path = (
    Path(__file__).parent / "test_diffpes" / "_reference_data"
)


@jaxtyped(typechecker=beartype)
def assert_trees_close(
    actual: PyTree,
    desired: PyTree,
    *,
    rtol: float = 1e-12,
    atol: float = 0.0,
) -> None:
    """Assert that corresponding numerical PyTree leaves are close.

    Uses Chex tree comparison with a relative tolerance of ``1e-12`` and zero
    absolute tolerance by default, preserving sensitivity near zero while
    allowing harmless floating-point reduction-order variation.
    """
    chex.assert_trees_all_close(actual, desired, rtol=rtol, atol=atol)


@jaxtyped(typechecker=beartype)
def assert_tree_finite(tree: PyTree) -> None:
    """Assert that every numerical leaf in a PyTree is finite.

    Delegates recursively to Chex so nested production carriers and ordinary
    array collections share one NaN and infinity gate.
    """
    leaves: tuple[object, ...] = tuple(jax.tree.leaves(tree))
    chex.assert_tree_all_finite(leaves)


@jaxtyped(typechecker=beartype)
def assert_matches_reference(
    tree: PyTree,
    name: str,
    *,
    rtol: float = 1e-12,
) -> None:
    """Compare numerical leaves with a named WP6.1 NPZ reference.

    Loads ``tests/test_diffpes/_reference_data/<name>.npz`` and compares the
    archive arrays in stored order with the numerical leaves of ``tree``.
    ``name`` must be a bare filename stem, preventing paths outside the pinned
    reference directory. WP6.1 supplies the archives and their manifest.

    Parameters
    ----------
    tree : PyTree
        Numerical production carrier or nested collection to compare.
    name : str
        Bare filename stem of the NPZ artifact.
    rtol : float, optional
        Relative comparison tolerance. Default is ``1e-12``.

    Raises
    ------
    ValueError
        If ``name`` is not a bare filename stem or the leaf count differs from
        the archive array count.
    """
    if Path(name).name != name or Path(name).suffix:
        message: str = "name must be a bare reference filename stem"
        raise ValueError(message)
    reference_path: Path = _REFERENCE_DIRECTORY / f"{name}.npz"
    with np.load(reference_path, allow_pickle=False) as archive:
        desired_leaves: tuple[np.ndarray, ...] = tuple(
            archive[key] for key in archive.files
        )
    actual_leaves: tuple[object, ...] = tuple(jax.tree.leaves(tree))
    if len(actual_leaves) != len(desired_leaves):
        message = (
            "reference leaf count differs: "
            f"actual={len(actual_leaves)}, desired={len(desired_leaves)}"
        )
        raise ValueError(message)
    assert_trees_close(actual_leaves, desired_leaves, rtol=rtol)
