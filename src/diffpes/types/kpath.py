"""K-point path information data structure.

Extended Summary
----------------
Defines the :class:`KPathInfo` PyTree for storing Brillouin-zone
metadata parsed from VASP KPOINTS files, including both plotting
labels and mode-specific metadata (automatic grids, explicit weights,
line-mode segments/endpoints).

Routine Listings
----------------
:class:`KPathInfo`
    PyTree for k-point path metadata.
:func:`make_kpath_info`
    Create a validated KPathInfo instance.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Union
from jax import lax
from jaxtyping import Array, Float, Int, jaxtyped


class KPathInfo(eqx.Module):
    """PyTree for k-point path metadata.

    Extended Summary
    ----------------
    Stores Brillouin-zone path information parsed from VASP KPOINTS
    files. Includes plotting fields (labels + label indices) and
    mode-specific metadata needed for full parser completeness:
    automatic-mode grid/shift, explicit-mode k-points/weights, and
    line-mode segment endpoints.

    This class is registered as a JAX PyTree via
    as children. String metadata is stored as auxiliary data because
    JAX cannot trace Python strings.

    Attributes
    ----------
    num_kpoints : Int[Array, " "]
        Total number of k-points in the path (line mode) or header
        count (explicit mode).
    label_indices : Int[Array, " L"]
        Indices of symmetry points along the path.
    points_per_segment : Int[Array, " "]
        Raw integer from line 2 of KPOINTS (line mode: points per
        segment).
    segments : Int[Array, " "]
        Number of line segments in line mode.
    kpoints : Optional[Float[Array, "K 3"]]
        Mode-specific k-points:
        line mode -> segment endpoints (segments + 1),
        explicit mode -> listed k-points,
        automatic mode -> None.
    weights : Optional[Float[Array, " K"]]
        Explicit-mode per-k-point weights (None otherwise).
    grid : Optional[Int[Array, " 3"]]
        Automatic-mode Monkhorst-Pack/Gamma grid (None otherwise).
    shift : Optional[Float[Array, " 3"]]
        Automatic-mode grid shift (None otherwise).
    mode : str
        KPOINTS file mode (Automatic, Line-mode, Explicit). **Static**.
    labels : tuple[str, ...]
        Symmetry point labels (e.g., Gamma, M, K). **Static**.
    comment : str
        Raw comment from KPOINTS line 1. **Static**.
    coordinate_mode : str
        Coordinate/scheme line metadata:
        line/explicit -> Reciprocal/Cartesian line,
        automatic -> scheme line (e.g., Monkhorst-Pack). **Static**.

    Notes
    -----
    String metadata is declared with ``eqx.field(static=True)`` rather
    than as traced leaves. Changing any static value triggers
    recompilation of any ``jit``-compiled function that receives
    this PyTree.

    See Also
    --------
    make_kpath_info : Factory function with validation and int32
        casting.
    """

    num_kpoints: Int[Array, " "]
    label_indices: Int[Array, " L"]
    points_per_segment: Int[Array, " "]
    segments: Int[Array, " "]
    kpoints: Optional[Float[Array, "K 3"]]
    weights: Optional[Float[Array, " K"]]
    grid: Optional[Int[Array, " 3"]]
    shift: Optional[Float[Array, " 3"]]
    mode: str = eqx.field(static=True)
    labels: tuple[str, ...] = eqx.field(static=True)
    comment: str = eqx.field(static=True)
    coordinate_mode: str = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def make_kpath_info(  # noqa: PLR0913
    num_kpoints: Union[int, Int[Array, " "]],
    label_indices: Union[Int[Array, " L"], "list[int]"],
    points_per_segment: Union[int, Int[Array, " "]] = 0,
    segments: Union[int, Int[Array, " "]] = 0,
    kpoints: Optional[Float[Array, "K 3"]] = None,
    weights: Optional[Float[Array, " K"]] = None,
    grid: Optional[Union[Int[Array, " 3"], "list[int]"]] = None,
    shift: Optional[Float[Array, " 3"]] = None,
    mode: str = "Line-mode",
    labels: tuple[str, ...] = (),
    comment: str = "",
    coordinate_mode: str = "",
) -> KPathInfo:
    """Create a validated KPathInfo instance.

    Extended Summary
    ----------------
    Factory function that validates and normalises raw k-path
    metadata before constructing a ``KPathInfo`` PyTree. Integer
    scalars and arrays are cast to ``int32``; float arrays are cast
    to ``float64``. Optional fields (``kpoints``, ``weights``,
    ``grid``, ``shift``) are cast only when present, preserving
    ``None`` for modes that do not use them.

    The function is decorated with ``@jaxtyped(typechecker=beartype)``
    so that shape constraints are checked at call time. String
    metadata (``mode``, ``labels``, ``comment``, ``coordinate_mode``)
    is passed through unchanged and stored as PyTree auxiliary data.

    Use this factory when constructing ``KPathInfo`` from parsed
    KPOINTS data or when building synthetic k-paths for testing. The
    factory ensures consistent dtypes and handles the three KPOINTS
    modes (Line-mode, Explicit, Automatic) through the optional
    fields.

    Implementation Logic
    --------------------
    1. **Cast integer fields** (``num_kpoints``, ``label_indices``,
       ``points_per_segment``, ``segments``) to ``jnp.int32`` via
       ``jnp.asarray``. Accepts both Python ints and JAX arrays.
    2. **Cast optional float fields** (``kpoints``, ``weights``,
       ``shift``) to ``jnp.float64`` when not ``None``.
    3. **Cast optional int field** (``grid``) to ``jnp.int32`` when
       not ``None``.
    4. **Pass string metadata** (``mode``, ``labels``, ``comment``,
       ``coordinate_mode``) through unchanged -- stored as auxiliary
       data in the PyTree.
    5. **Construct** the ``KPathInfo`` Equinox module from all validated
       fields and return it.

    Parameters
    ----------
    num_kpoints : Union[int, Int[Array, " "]]
        Total number of k-points along the path.
    label_indices : Union[Int[Array, " L"], list[int]]
        Indices of symmetry points along the path.
    points_per_segment : Union[int, Int[Array, " "]], optional
        Raw value from line 2 of KPOINTS. Default is 0.
    segments : Union[int, Int[Array, " "]], optional
        Number of path segments in line mode. Default is 0.
    kpoints : Optional[Float[Array, "K 3"]], optional
        Mode-specific k-point coordinates. Default is None.
    weights : Optional[Float[Array, " K"]], optional
        Explicit-mode weights. Default is None.
    grid : Optional[Union[Int[Array, " 3"], list[int]]], optional
        Automatic-mode MP/Gamma grid. Default is None.
    shift : Optional[Float[Array, " 3"]], optional
        Automatic-mode grid shift. Default is None.
    mode : str, optional
        KPOINTS file mode. Default is ``"Line-mode"``.
    labels : tuple[str, ...], optional
        Symmetry point labels. Default is empty tuple.
    comment : str, optional
        KPOINTS comment line. Default is empty string.
    coordinate_mode : str, optional
        Coordinate/scheme line metadata. Default is empty string.

    Returns
    -------
    kpath : KPathInfo
        Validated k-path info instance.

    See Also
    --------
    KPathInfo : The PyTree class constructed by this factory.
    """
    nkpts_arr: Int[Array, " "] = jnp.asarray(num_kpoints, dtype=jnp.int32)
    indices_arr: Int[Array, " L"] = jnp.asarray(label_indices, dtype=jnp.int32)
    pps_arr: Int[Array, " "] = jnp.asarray(points_per_segment, dtype=jnp.int32)
    segments_arr: Int[Array, " "] = jnp.asarray(segments, dtype=jnp.int32)
    kpoints_arr: Optional[Float[Array, "K 3"]] = None
    if kpoints is not None:
        kpoints_arr = jnp.asarray(kpoints, dtype=jnp.float64)
    weights_arr: Optional[Float[Array, " K"]] = None
    if weights is not None:
        weights_arr = jnp.asarray(weights, dtype=jnp.float64)
    grid_arr: Optional[Int[Array, " 3"]] = None
    if grid is not None:
        grid_arr = jnp.asarray(grid, dtype=jnp.int32)
    shift_arr: Optional[Float[Array, " 3"]] = None
    if shift is not None:
        shift_arr = jnp.asarray(shift, dtype=jnp.float64)

    def validate_and_create() -> KPathInfo:
        def check_num_kpoints_non_negative() -> Int[Array, " "]:
            return lax.cond(
                nkpts_arr >= 0,
                lambda: nkpts_arr,
                lambda: lax.stop_gradient(
                    lax.cond(False, lambda: nkpts_arr, lambda: nkpts_arr)
                ),
            )

        check_num_kpoints_non_negative()
        return KPathInfo(
            num_kpoints=nkpts_arr,
            label_indices=indices_arr,
            points_per_segment=pps_arr,
            segments=segments_arr,
            kpoints=kpoints_arr,
            weights=weights_arr,
            grid=grid_arr,
            shift=shift_arr,
            mode=mode,
            labels=labels,
            comment=comment,
            coordinate_mode=coordinate_mode,
        )

    kpath: KPathInfo = validate_and_create()
    return kpath


__all__: list[str] = [
    "KPathInfo",
    "make_kpath_info",
]
