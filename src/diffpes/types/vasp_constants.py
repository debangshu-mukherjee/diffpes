"""Declarative VASP parser and serialization conventions."""

import re

_LATTICE_ROWS: int = 3
_XYZ_COMPONENTS: int = 3
_SCALAR_LINE_COMPONENTS: int = 3
_N_SOC_MAG_BLOCKS: int = 3

_NONSPIN_COLS: int = 3
_SPIN_COLS: int = 5

_ISPIN_SPIN_POLARIZED: int = 2
_KPOINT_LINE_VALUES: int = 4
_BAND_LINE_MIN_VALUES: int = 2
_BAND_LINE_SPIN_VALUES: int = 3
_EIG_UP_INDEX: int = 1
_EIG_DOWN_INDEX: int = 2

_ATTR_TYPE: str = "_pytree_type"
_ATTR_AUX: str = "_aux_data_json"
_ATTR_NONE: str = "_none_fields"
_KPATH_AUX_WITH_COMMENT_LEN: int = 3
_KPATH_AUX_WITH_COORD_MODE_LEN: int = 4

_FLOAT_TOKEN_RE: re.Pattern[str] = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)
_WEIGHT_COMPONENT_INDEX: int = 3
_WEIGHT_COMPONENT_COUNT: int = 4
_COORDINATE_MODE_TOKENS: frozenset[str] = frozenset(
    {"cartesian", "reciprocal", "direct", "fractional"}
)

_INTENSITY_NDIM: int = 2
_ENERGY_AXIS_NDIM: int = 1
_BAND_NDIM: int = 2
_PRESET_NAMES: tuple[str, ...] = (
    "s",
    "py",
    "pz",
    "px",
    "p",
    "dxy",
    "dyz",
    "dz2",
    "dxz",
    "dx2y2",
    "d",
    "non_s",
    "total",
    "spin_x_up",
    "spin_x_down",
    "spin_y_up",
    "spin_y_down",
    "spin_z_up",
    "spin_z_down",
    "spin_x",
    "spin_y",
    "spin_z",
    "oam_p",
    "oam_d",
    "oam_total",
    "oam_abs_total",
)

_N_SPIN_COMPONENTS: int = 6
_ISPIN2_BLOCKS: int = 2
_SOC_BLOCKS: int = 4

_PHASE_LOSS_MESSAGE: str = (
    "vasp_to_diagonalized uses sqrt(|c|^2) magnitudes from PROCAR and cannot "
    "recover complex eigenvector phases. Matrix elements that depend on phase "
    "interference are approximate."
)

__all__: list[str] = []
