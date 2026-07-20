"""HDF5 serializer and deserializer for diffpes PyTrees.

Extended Summary
----------------
Provides functions for saving and loading diffpes Equinox PyTree objects
to and from HDF5 files via ``h5py``. Each module's named array fields
become HDF5 datasets, and static metadata is
stored as HDF5 group attributes in JSON format.

Routine Listings
----------------
:func:`load_from_h5`
    Load PyTrees from an HDF5 file.
:func:`save_to_h5`
    Save one or more named PyTrees to an HDF5 file.

Notes
-----
All nineteen types-owned Equinox carriers are supported. Serialization
metadata is derived from dataclass fields: non-static fields are stored as
datasets or recursive module groups, while ``eqx.field(static=True)`` values
are encoded as tuple-preserving JSON.
"""

import json
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path

import equinox as eqx
import h5py
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Any, Optional, Union
from jaxtyping import Shaped
from numpy import ndarray as NDArray  # noqa: N812

from diffpes.types import (
    ArpesSpectrum,
    BandStructure,
    CrystalGeometry,
    DensityOfStates,
    DiagonalizedBands,
    FullDensityOfStates,
    KPathInfo,
    OrbitalBasis,
    OrbitalProjection,
    PolarizationConfig,
    SelfEnergyConfig,
    SimulationParams,
    SlaterParams,
    SOCVolumetricData,
    SpinBandStructure,
    SpinOrbitalProjection,
    TBModel,
    VolumetricData,
    WorkflowContext,
)
from diffpes.types.vasp_constants import (
    _ATTR_AUX,
    _ATTR_NONE,
    _ATTR_TYPE,
)


@dataclass(frozen=True)
class _PyTreeMeta:
    """Serialization metadata for a registered PyTree type.

    Stores the class reference, the ordered names of JAX-traced
    array field names, static metadata field names, and encoder/decoder
    callables for converting static metadata to and from JSON.

    Attributes
    ----------
    cls : Any
        The Equinox module class.
    children_fields : tuple[str, ...]
        Ordered field names of JAX array children.
    static_fields : tuple[str, ...]
        Ordered field names of static Equinox metadata.
    """

    cls: Any  # Equinox module class
    children_fields: tuple[str, ...]
    static_fields: tuple[str, ...]


def _pytree_classes() -> tuple[type[eqx.Module], ...]:
    """Return the complete carrier class set used by the codec."""
    classes: tuple[type[eqx.Module], ...] = (
        ArpesSpectrum,
        BandStructure,
        CrystalGeometry,
        DensityOfStates,
        DiagonalizedBands,
        FullDensityOfStates,
        KPathInfo,
        OrbitalBasis,
        OrbitalProjection,
        PolarizationConfig,
        SelfEnergyConfig,
        SimulationParams,
        SlaterParams,
        SOCVolumetricData,
        SpinBandStructure,
        SpinOrbitalProjection,
        TBModel,
        VolumetricData,
        WorkflowContext,
    )
    return classes


def _encode_static(value: Any) -> Any:  # noqa: ANN401
    """Encode nested static Equinox metadata without losing tuple types."""
    if isinstance(value, tuple):
        return {"__tuple__": [_encode_static(item) for item in value]}
    if isinstance(value, eqx.Module) and is_dataclass(value):
        return {
            "__module__": type(value).__name__,
            "fields": {
                field.name: _encode_static(getattr(value, field.name))
                for field in fields(value)
            },
        }
    if isinstance(value, list):
        return [_encode_static(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _encode_static(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def _decode_static(value: Any) -> Any:  # noqa: ANN401
    """Decode tuple-preserving and nested-module static metadata."""
    if isinstance(value, dict) and "__tuple__" in value:
        return tuple(_decode_static(item) for item in value["__tuple__"])
    if isinstance(value, dict) and "__module__" in value:
        class_name: str = str(value["__module__"])
        module_class: type[eqx.Module] = _PYTREE_REGISTRY[class_name].cls
        module_fields: dict[str, Any] = {
            str(name): _decode_static(item)
            for name, item in value["fields"].items()
        }
        return module_class(**module_fields)
    if isinstance(value, list):
        return [_decode_static(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _decode_static(item) for key, item in value.items()}
    return value


def _decode_aux_data(type_name: str, value: Any) -> Any:  # noqa: ANN401
    """Decode current static metadata and supported legacy HDF5 aux data.

    Plan 02's pre-migration codec wrote plain JSON lists for tuple-valued
    metadata. Current files use explicit tuple tags. The three conversions
    below retain read compatibility with those pinned files without restoring
    a per-carrier write registry.
    """
    decoded: Any = _decode_static(value)
    if isinstance(value, dict) or value is None:
        return decoded
    if type_name == "CrystalGeometry":
        return tuple(str(item) for item in value)
    if type_name == "KPathInfo":
        return (
            str(value[0]),
            tuple(str(item) for item in value[1]),
            str(value[2]),
            str(value[3]),
        )
    if type_name in {"SOCVolumetricData", "VolumetricData"}:
        return (
            tuple(int(item) for item in value[0]),
            tuple(str(item) for item in value[1]),
        )
    return decoded


def _module_meta(module_class: type[eqx.Module]) -> _PyTreeMeta:
    """Build serialization metadata from Equinox dataclass fields."""
    module_fields = fields(module_class)
    children_fields: tuple[str, ...] = tuple(
        field.name
        for field in module_fields
        if not bool(field.metadata.get("static", False))
    )
    static_fields: tuple[str, ...] = tuple(
        field.name
        for field in module_fields
        if bool(field.metadata.get("static", False))
    )
    return _PyTreeMeta(
        cls=module_class,
        children_fields=children_fields,
        static_fields=static_fields,
    )


_PYTREE_REGISTRY: dict[str, _PyTreeMeta] = {
    cls.__name__: _module_meta(cls) for cls in _pytree_classes()
}


@beartype
def _dataset_write_kwargs(
    data: Shaped[NDArray, "..."],
    compression: Optional[str],
    compression_opts: Any,  # noqa: ANN401
    shuffle: bool,
    fletcher32: bool,
    chunks: Optional[Union[bool, tuple[int, ...]]],
) -> dict[str, Any]:
    """Build ``h5py.create_dataset`` keyword arguments for one child array.

    Extended Summary
    ----------------
    HDF5 storage filters (compression, shuffle, checksums) and chunking
    are only valid for datasets with non-scalar dataspace. This helper
    inspects the array dimensionality and returns the appropriate
    keyword dictionary for ``h5py.Group.create_dataset``.

    Implementation Logic
    --------------------
    1. For scalar datasets (``data.ndim == 0``), return an empty dict
       since HDF5 filter/chunk flags are invalid for scalar dataspace.
    2. For non-scalar datasets, conditionally include each supported
       storage flag (``compression``, ``compression_opts``, ``shuffle``,
       ``fletcher32``, ``chunks``) only when the corresponding argument
       is not ``None`` / ``False``.

    Parameters
    ----------
    data : np.ndarray
        The NumPy array to be written. Its ``ndim`` determines whether
        filters are applicable.
    compression : Optional[str]
        HDF5 compression filter name (e.g. ``"gzip"``, ``"lzf"``).
    compression_opts : Any
        Compression-specific options (e.g. gzip level 1-9).
    shuffle : bool
        Whether to enable the HDF5 byte-shuffle filter.
    fletcher32 : bool
        Whether to enable the Fletcher32 checksum filter.
    chunks : Optional[Union[bool, tuple[int, ...]]]
        Chunking policy: ``True`` for auto-chunking, or an explicit
        chunk shape tuple.

    Returns
    -------
    dict[str, Any]
        Keyword arguments to pass to ``h5py.Group.create_dataset``.
        Empty dict for scalar datasets.
    """
    if data.ndim == 0:
        return {}

    kwargs: dict[str, Any] = {}
    if compression is not None:
        kwargs["compression"] = compression
    if compression_opts is not None:
        kwargs["compression_opts"] = compression_opts
    if shuffle:
        kwargs["shuffle"] = True
    if fletcher32:
        kwargs["fletcher32"] = True
    if chunks is not None:
        kwargs["chunks"] = chunks
    return kwargs


@beartype
def save_to_h5(
    path: Union[str, Path],
    /,
    *,
    compression: Optional[str] = None,
    compression_opts: Any = None,  # noqa: ANN401
    shuffle: bool = False,
    fletcher32: bool = False,
    chunks: Optional[Union[bool, tuple[int, ...]]] = None,
    **pytrees: Any,  # noqa: ANN401
) -> None:
    """Save one or more named PyTrees to an HDF5 file.

    Serializes each keyword-argument PyTree into a named HDF5
    group. JAX array fields become HDF5 datasets (named by the
    Equinox field name), and static metadata is stored as a
    JSON-encoded group attribute.

    Implementation Logic
    --------------------
    1. **Validate inputs**:
       Ensure at least one PyTree is provided.

    2. **Iterate over keyword arguments**:
       For each ``(group_name, pytree)`` pair:

       a. Look up ``type(pytree).__name__`` in
          ``_PYTREE_REGISTRY`` to obtain serialization metadata.

       b. Read child arrays and static metadata by registered field name.

       c. Create an HDF5 group named ``group_name``.

       d. Store the type name as ``_pytree_type`` attribute and
          the JSON-encoded aux_data as ``_aux_data_json``.

       e. For each child field: if the value is ``None``
          (Optional field), record the field name in
          ``_none_fields``; otherwise create an HDF5 dataset
          from ``numpy.asarray(child)`` with optional storage
          flags (compression/chunk/checksum) for non-scalar
          datasets.

       f. Store the ``_none_fields`` list as a JSON attribute.

    Parameters
    ----------
    path : Union[str, Path]
        File path for the HDF5 file to create.
    compression : Optional[str], optional
        HDF5 compression filter name (e.g. ``"gzip"``, ``"lzf"``).
        Applied to non-scalar datasets only.
    compression_opts : Any, optional
        Compression options passed through to h5py (e.g. gzip level).
        Must be ``None`` when ``compression`` is ``None``.
    shuffle : bool, optional
        If True, enable HDF5 shuffle filter on non-scalar datasets.
    fletcher32 : bool, optional
        If True, enable HDF5 Fletcher32 checksum on non-scalar datasets.
    chunks : Optional[Union[bool, tuple[int, ...]]], optional
        Chunking policy for non-scalar datasets. ``True`` enables
        auto-chunking, or provide an explicit chunk-shape tuple.
    **pytrees : PyTree
        Named PyTree instances. Each keyword argument name
        becomes an HDF5 group name.

    Raises
    ------
    ValueError
        If no PyTrees are provided.
    ValueError
        If ``compression_opts`` is provided without ``compression``.
    TypeError
        If a PyTree's class is not in the registry.

    Notes
    -----
    Scalar datasets (shape ``()``) are always written without HDF5
    filter/chunk flags because those options are invalid for scalar
    dataspace in HDF5.
    """
    if not pytrees:
        msg = "At least one PyTree must be provided."
        raise ValueError(msg)
    if compression is None and compression_opts is not None:
        msg = "compression_opts requires compression to be set."
        raise ValueError(msg)

    def _write_module(
        grp: h5py.Group,
        pytree: Any,  # noqa: ANN401
    ) -> None:
        """Write one Equinox module, recursively storing module children."""
        type_name: str = type(pytree).__name__
        if type_name not in _PYTREE_REGISTRY:
            msg = f"Unsupported PyTree type: {type_name}"
            raise TypeError(msg)
        meta: _PyTreeMeta = _PYTREE_REGISTRY[type_name]
        static_values: tuple[Any, ...] = tuple(
            getattr(pytree, field_name) for field_name in meta.static_fields
        )
        aux_data: Any = None
        if len(static_values) == 1:
            aux_data = static_values[0]
        elif static_values:
            aux_data = static_values
        grp.attrs[_ATTR_TYPE] = type_name
        grp.attrs[_ATTR_AUX] = json.dumps(_encode_static(aux_data))

        none_fields: list[str] = []
        for field_name in meta.children_fields:
            child: Any = getattr(pytree, field_name)
            if child is None:
                none_fields.append(field_name)
            elif isinstance(child, eqx.Module):
                child_group: h5py.Group = grp.create_group(field_name)
                _write_module(child_group, child)
            else:
                child_arr: Shaped[NDArray, "..."] = np.asarray(child)
                ds_kwargs: dict[str, Any] = _dataset_write_kwargs(
                    data=child_arr,
                    compression=compression,
                    compression_opts=compression_opts,
                    shuffle=shuffle,
                    fletcher32=fletcher32,
                    chunks=chunks,
                )
                grp.create_dataset(field_name, data=child_arr, **ds_kwargs)
        grp.attrs[_ATTR_NONE] = json.dumps(none_fields)

    file_path: Path = Path(path)
    with h5py.File(file_path, "w") as f:
        for group_name, pytree in pytrees.items():
            grp: h5py.Group = f.create_group(group_name)
            _write_module(grp, pytree)


@beartype
def load_from_h5(
    path: Union[str, Path],
    name: Optional[str] = None,
) -> Any:  # noqa: ANN401
    """Load PyTrees from an HDF5 file.

    Deserializes HDF5 groups back into diffpes PyTree objects
    by reading datasets as JAX arrays and reconstructing the
    Equinox module with keyword arguments.

    Implementation Logic
    --------------------
    1. **Open the HDF5 file** for reading.

    2. **Select groups to load**:
       If ``name`` is provided, load only that group. If
       ``name`` is ``None``, load all top-level groups.

    3. **For each group**:

       a. Read ``_pytree_type`` attribute and look up the class
          in ``_PYTREE_REGISTRY``.

       b. Read ``_aux_data_json`` attribute and decode it via
          the type-specific ``aux_decoder``.

       c. Read ``_none_fields`` attribute (defaulting to empty
          list).

       d. For each children field name: if the name appears in
          ``_none_fields``, set the child to ``None``;
          otherwise read the HDF5 dataset and convert to a JAX
          array via ``jnp.asarray``.

       e. Reconstruct the Equinox module with keyword arguments.

    Parameters
    ----------
    path : Union[str, Path]
        File path to the HDF5 file to read.
    name : Optional[str], optional
        Name of a specific group to load. If ``None``, all
        groups are loaded and returned as a dict.

    Returns
    -------
    result : PyTree or dict[str, PyTree]
        A single PyTree if ``name`` is given, otherwise a dict
        mapping group names to PyTree instances.

    Raises
    ------
    KeyError
        If ``name`` is specified but does not exist in the file.
    TypeError
        If a group's ``_pytree_type`` is not in the registry.
    """
    file_path: Path = Path(path)

    def _load_group(
        grp: h5py.Group,
    ) -> Any:  # noqa: ANN401
        type_name: str = str(grp.attrs[_ATTR_TYPE])
        if type_name not in _PYTREE_REGISTRY:
            msg = f"Unknown PyTree type: {type_name}"
            raise TypeError(msg)

        meta: _PyTreeMeta = _PYTREE_REGISTRY[type_name]
        aux_json: Any = json.loads(str(grp.attrs[_ATTR_AUX]))
        aux_data: Any = _decode_aux_data(type_name, aux_json)

        none_fields: list[str] = json.loads(str(grp.attrs[_ATTR_NONE]))

        children: list[Any] = []
        for field_name in meta.children_fields:
            if field_name in none_fields:
                children.append(None)
            elif isinstance(grp[field_name], h5py.Group):
                children.append(_load_group(grp[field_name]))
            else:
                arr: Shaped[NDArray, "..."] = grp[field_name][()]
                children.append(jnp.asarray(arr))

        constructor_fields: dict[str, Any] = dict(
            zip(meta.children_fields, children, strict=True)
        )
        static_values: tuple[Any, ...]
        if not meta.static_fields:
            static_values = ()
        elif len(meta.static_fields) == 1:
            static_values = (aux_data,)
        else:
            static_values = tuple(aux_data)
        constructor_fields.update(
            zip(meta.static_fields, static_values, strict=True)
        )
        loaded: Any = meta.cls(**constructor_fields)
        return loaded

    with h5py.File(file_path, "r") as f:
        if name is not None:
            if name not in f:
                msg = f"Group '{name}' not found in {path}"
                raise KeyError(msg)
            return _load_group(f[name])

        result: dict[str, Any] = {}
        for group_name in f:
            result[group_name] = _load_group(f[group_name])
        return result


__all__: list[str] = [
    "load_from_h5",
    "save_to_h5",
]
