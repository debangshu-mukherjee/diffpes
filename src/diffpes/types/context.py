"""Structured inputs for high-level VASP simulation workflows.

Extended Summary
----------------
Defines the Equinox container that bundles parsed VASP outputs for the
high-level simulation workflow.  Keeping this carrier in :mod:`diffpes.types`
provides one canonical import surface for every diffpes PyTree.

Routine Listings
----------------
:class:`WorkflowContext`
    Parsed band, projection, k-path, and density-of-states inputs.
:func:`make_workflow_context`
    Create a workflow context from parsed inputs.
:obj:`DosType`
    Supported density-of-states containers.
:obj:`ProjectionType`
    Supported orbital-projection containers.
"""

import equinox as eqx
from beartype import beartype
from beartype.typing import Optional, TypeAlias, Union
from jaxtyping import jaxtyped

from .bands import BandStructure, OrbitalProjection, SpinOrbitalProjection
from .dos import DensityOfStates, FullDensityOfStates
from .kpath import KPathInfo

ProjectionType: TypeAlias = Union[OrbitalProjection, SpinOrbitalProjection]
DosType: TypeAlias = Union[DensityOfStates, FullDensityOfStates]


class WorkflowContext(eqx.Module):
    """Parsed VASP inputs used by high-level workflow helpers.

    Extended Summary
    ----------------
    Bundles the electronic bands and orbital projections required by an
    ARPES simulation with optional k-path and density-of-states metadata.
    Every non-``None`` field is itself an Equinox PyTree, so the complete
    context can pass through JAX transformations as one immutable object.

    Attributes
    ----------
    bands : BandStructure
        Band eigenvalues and k-point coordinates from EIGENVAL.
    orb_proj : ProjectionType
        Orbital projections from PROCAR, optionally including spin.
    kpath : Optional[KPathInfo]
        Parsed KPOINTS metadata when available.
    dos : Optional[DosType]
        Parsed DOSCAR data when available.
    """

    bands: BandStructure
    orb_proj: ProjectionType
    kpath: Optional[KPathInfo]
    dos: Optional[DosType]


@jaxtyped(typechecker=beartype)
def make_workflow_context(
    bands: BandStructure,
    orb_proj: ProjectionType,
    kpath: Optional[KPathInfo] = None,
    dos: Optional[DosType] = None,
) -> WorkflowContext:
    """Create a workflow context from parsed VASP inputs.

    Parameters
    ----------
    bands : BandStructure
        Band eigenvalues and k-point coordinates.
    orb_proj : ProjectionType
        Orbital projections, optionally including spin channels.
    kpath : Optional[KPathInfo], optional
        Parsed k-path metadata. Default is ``None``.
    dos : Optional[DosType], optional
        Parsed density-of-states data. Default is ``None``.

    Returns
    -------
    WorkflowContext
        Immutable Equinox container holding the supplied inputs.
    """
    context: WorkflowContext = WorkflowContext(
        bands=bands,
        orb_proj=orb_proj,
        kpath=kpath,
        dos=dos,
    )
    return context


__all__: list[str] = [
    "DosType",
    "ProjectionType",
    "WorkflowContext",
    "make_workflow_context",
]
