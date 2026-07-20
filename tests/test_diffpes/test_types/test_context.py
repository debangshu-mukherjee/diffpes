"""Test workflow-context construction and validation.

Extended Summary
----------------
Covers dimensional consistency checks for the workflow carrier defined in ``diffpes.types.context`` in eager and compiled execution.
"""

import jax.numpy as jnp

from diffpes.types import (
    make_band_structure,
    make_orbital_projection,
    make_workflow_context,
)
from tests._assertions import assert_rejects


def test_workflow_context_rejects_band_projection_mismatch() -> None:
    """Reject workflow members with inconsistent K/B dimensions."""
    bands = make_band_structure(
        eigenvalues=jnp.zeros((2, 1)), kpoints=jnp.zeros((2, 3))
    )
    projection = make_orbital_projection(jnp.zeros((1, 1, 1, 9)))
    assert_rejects(
        make_workflow_context,
        bands=bands,
        orb_proj=projection,
        match="bands and orb_proj must agree",
    )
