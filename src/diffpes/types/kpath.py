"""Define the k-point path information data structure.

Extended Summary
----------------
This module defines the :class:`KPathInfo` PyTree for Brillouin-zone metadata
from VASP KPOINTS files. The metadata includes plotting labels, automatic
grids, explicit weights, and line-mode segments.

Routine Listings
----------------
:class:`KPathInfo`
    Store k-point path metadata in a JAX PyTree.
:func:`make_kpath_info`
    Create a validated KPathInfo instance.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Union
from jax.core import Tracer
from jaxtyping import Array, Float, Int, jaxtyped

_KPATH_MODES: tuple[str, ...] = ("Automatic", "Line-mode", "Explicit")


class KPathInfo(eqx.Module):
    """Store k-point path metadata in a JAX PyTree.

    This type stores Brillouin-zone path information from VASP KPOINTS files.
    It includes plotting labels and label indices. Mode-specific metadata
    includes an automatic grid and shift, explicit k-points and weights, and
    line-mode segment endpoints.

    JAX stores numerical fields as PyTree children. It stores string metadata
    as auxiliary data because JAX cannot trace Python strings.


    :see: :class:`~.test_kpath.TestKPathInfo`

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
        KPOINTS file mode (Automatic, Line-mode, Explicit; **static** -- a
        compile-time constant; changing it triggers retracing).
    labels : tuple[str, ...]
        Symmetry point labels (e.g., Gamma, M, K; **static** -- compile-time
        constants; changing them triggers retracing).
    comment : str
        Raw comment from KPOINTS line 1 (**static** -- a compile-time
        constant; changing it triggers retracing).
    coordinate_mode : str
        Coordinate/scheme line metadata:
        line/explicit -> Reciprocal/Cartesian line,
        automatic -> scheme line (e.g., Monkhorst-Pack; **static** -- a
        compile-time constant; changing it triggers retracing).

    Notes
    -----
    ``eqx.field(static=True)`` declares string metadata as auxiliary data
    instead of traced leaves. Changing any static value triggers
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
def make_kpath_info(  # noqa: DOC503, PLR0913
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

    The factory validates and normalizes raw k-path
    metadata before constructing a ``KPathInfo`` PyTree. Integer
    The factory casts integer scalars and arrays to ``int32``. It casts float
    arrays to ``float64``. The factory casts optional fields only when they
    are present. Thus, ``None`` continues to identify modes that omit them.

    ``@jaxtyped(typechecker=beartype)`` checks shape constraints at call time.
    The factory passes string metadata unchanged and stores it as PyTree
    auxiliary data.

    Use this factory when constructing ``KPathInfo`` from parsed
    KPOINTS data or when building synthetic k-paths for testing. The
    factory ensures consistent dtypes and handles the three KPOINTS
    modes (Line-mode, Explicit, Automatic) through the optional
    fields.

    :see: :class:`~.test_kpath.TestMakeKPathInfo`

    Implementation Logic
    --------------------
    1. **Prepare the normalized values**::

           nkpts_arr = jnp.asarray(num_kpoints, dtype=jnp.int32)

       This expression gives the later validation steps a stable shape and
       dtype.

    2. **Apply static validation**::

           mode not in _KPATH_MODES

       This predicate rejects invalid structure before JAX traces the
       numerical checks.

    3. **Apply traced validation**::

           ~(nkpts_arr >= 0)

       This predicate remains active during eager and compiled execution.

    4. **Return the named instance**::

           return kpath

       The explicit name keeps the implementation and the Returns section
       synchronized.

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
        KPOINTS file mode (**static** -- a compile-time constant; changing it
        triggers retracing). Default is ``"Line-mode"``.
    labels : tuple[str, ...], optional
        Symmetry point labels (**static** -- compile-time constants; changing
        them triggers retracing). Default is empty tuple.
    comment : str, optional
        KPOINTS comment line (**static** -- a compile-time constant; changing
        it triggers retracing). Default is empty string.
    coordinate_mode : str, optional
        Coordinate/scheme line metadata (**static** -- a compile-time
        constant; changing it triggers retracing). Default is empty string.

    Returns
    -------
    kpath : KPathInfo
        Validated k-path info instance.

    Raises
    ------
    ValueError
        If ``mode`` has an unsupported value. The function also rejects
        incomplete or inconsistent line-mode metadata.
    EquinoxRuntimeError
        If ``num_kpoints`` is negative, a traced line-mode segment count is
        less than one, or supplied ``kpoints``, ``weights``, or ``shift``
        values are non-finite.

    Notes
    -----
    Static validation raises ``ValueError`` before traced construction for an
    unsupported mode or inconsistent line-mode metadata. Traced validation
    uses ``eqx.error_if`` and raises ``EquinoxRuntimeError`` for negative
    k-point counts, invalid traced segment counts, and non-finite optional
    floating-point arrays.

    See Also
    --------
    KPathInfo : The PyTree class constructed by this factory.
    """
    if mode not in _KPATH_MODES:
        msg: str = f"make_kpath_info: mode must be one of {_KPATH_MODES}"
        raise ValueError(msg)
    if mode == "Line-mode" and len(label_indices) < 1:
        msg: str = "make_kpath_info: at least one label index is required"
        raise ValueError(msg)
    if mode == "Line-mode" and labels and len(labels) != len(label_indices):
        msg: str = "make_kpath_info: labels and label_indices must agree"
        raise ValueError(msg)

    nkpts_arr: Int[Array, " "] = jnp.asarray(num_kpoints, dtype=jnp.int32)
    indices_arr: Int[Array, " L"] = jnp.asarray(label_indices, dtype=jnp.int32)
    pps_arr: Int[Array, " "] = jnp.asarray(points_per_segment, dtype=jnp.int32)
    segments_arr: Int[Array, " "] = jnp.asarray(segments, dtype=jnp.int32)
    if (
        mode == "Line-mode"
        and not isinstance(segments_arr, Tracer)
        and int(segments_arr) < 1
    ):
        msg: str = "make_kpath_info: line mode requires at least one segment"
        raise ValueError(msg)
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
        nonlocal kpoints_arr, nkpts_arr, segments_arr, shift_arr, weights_arr
        nkpts_arr = eqx.error_if(
            nkpts_arr,
            ~(nkpts_arr >= 0),
            "make_kpath_info: num kpoints non negative",
        )
        if mode == "Line-mode":
            segments_arr = eqx.error_if(
                segments_arr,
                segments_arr < 1,
                "make_kpath_info: line mode requires at least one segment",
            )
        if kpoints_arr is not None:
            kpoints_arr = eqx.error_if(
                kpoints_arr,
                ~jnp.all(jnp.isfinite(kpoints_arr)),
                "make_kpath_info: kpoints must be finite",
            )
        if weights_arr is not None:
            weights_arr = eqx.error_if(
                weights_arr,
                ~jnp.all(jnp.isfinite(weights_arr)),
                "make_kpath_info: weights must be finite",
            )
        if shift_arr is not None:
            shift_arr = eqx.error_if(
                shift_arr,
                ~jnp.all(jnp.isfinite(shift_arr)),
                "make_kpath_info: shift must be finite",
            )
        validated_kpath: KPathInfo = KPathInfo(
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
        return validated_kpath

    kpath: KPathInfo = validate_and_create()
    return kpath


__all__: list[str] = [
    "KPathInfo",
    "make_kpath_info",
]
