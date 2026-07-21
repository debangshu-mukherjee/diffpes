"""Validate k-path metadata storage and mode consistency.

The cases cover immutable plotting metadata and rejection of unsupported
KPOINTS mode selectors in ``diffpes.types.kpath``.
"""

import chex

from diffpes.types import KPathInfo, make_kpath_info
from tests._assertions import assert_rejects


class TestKPathInfo:
    """Validate :class:`~diffpes.types.KPathInfo` field storage.

    The carrier must preserve line-mode labels and their integer indices.

    :see: :class:`~diffpes.types.KPathInfo`
    """

    def test_stores_line_mode_metadata(self) -> None:
        """Preserve line-mode indices, labels, and mode text.

        The check verifies the three label positions for a 100-point path and
        the corresponding static symmetry labels.

        Notes
        -----
        Constructs the carrier through its validated factory, checks the array
        shape with Chex, and compares the static metadata exactly.
        """
        kpath: KPathInfo = make_kpath_info(
            num_kpoints=100,
            label_indices=[0, 49, 99],
            segments=2,
            mode="Line-mode",
            labels=("G", "M", "K"),
        )

        chex.assert_shape(kpath.label_indices, (3,))
        chex.assert_equal(kpath.mode, "Line-mode")
        chex.assert_equal(kpath.labels, ("G", "M", "K"))


class TestMakeKPathInfo:
    """Validate :func:`~diffpes.types.make_kpath_info`.

    The factory must reject mode strings outside the supported parser
    conventions.

    :see: :func:`~diffpes.types.make_kpath_info`
    """

    def test_rejects_unknown_mode(self) -> None:
        """Reject a k-path mode outside the supported static set.

        The check isolates the mode selector contract before any optional
        mode-specific arrays are interpreted.

        Notes
        -----
        Supplies otherwise valid minimal metadata with ``mode="unknown"`` and
        matches the allowed-mode diagnostic.
        """
        assert_rejects(
            make_kpath_info,
            num_kpoints=2,
            label_indices=[0],
            segments=1,
            mode="unknown",
            match="mode must be one of",
        )
