"""Test k-path metadata construction and validation.

Extended Summary
----------------
Covers field storage and mode validation for the carrier defined in ``diffpes.types.kpath``.
"""

import chex

from diffpes.types import make_kpath_info
from tests._assertions import assert_rejects


class TestMakeKPathInfo(chex.TestCase):
    """Tests for :func:`diffpes.types.kpath.make_kpath_info`.

    Verifies correct construction of ``KPathInfo`` PyTrees including
    output shape validation for label indices, and correct storage of
    the ``mode`` string and ``labels`` tuple as auxiliary data.
    """

    def test_basic_creation(self) -> None:
        """Verify that a KPathInfo is created with correct fields and shapes.

        Test Logic
        ----------
        1. **Construct k-path info**:
           Call ``make_kpath_info`` with 100 k-points, three label
           indices (start, midpoint, end), ``"Line-mode"`` mode, and
           symmetry labels ``("G", "M", "K")``.

        2. **Assert label_indices shape**:
           Check that ``label_indices`` has shape (3,), matching the
           three supplied indices.

        3. **Assert auxiliary fields**:
           Confirm that ``mode`` is ``"Line-mode"`` and ``labels`` is
           ``("G", "M", "K")``.

        Asserts
        -------
        ``label_indices`` has the expected 1-D shape and string fields
        are stored unchanged as auxiliary data.
        """
        kpath = make_kpath_info(
            num_kpoints=100,
            label_indices=[0, 49, 99],
            segments=2,
            mode="Line-mode",
            labels=("G", "M", "K"),
        )
        chex.assert_shape(kpath.label_indices, (3,))
        chex.assert_equal(kpath.mode, "Line-mode")
        chex.assert_equal(kpath.labels, ("G", "M", "K"))


def test_kpath_rejects_unknown_mode() -> None:
    """Reject k-path mode strings outside the supported set."""
    assert_rejects(
        make_kpath_info,
        num_kpoints=2,
        label_indices=[0],
        segments=1,
        mode="unknown",
        match="mode must be one of",
    )
