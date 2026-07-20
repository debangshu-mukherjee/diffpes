r"""Dependency-light numerical and physical constants used by diffpes.

Extended Summary
----------------
Single home for the scalar constants shared across subpackages, kept
free of JAX imports so it can be imported anywhere without side
effects. Physical constants (CODATA values in eV-based units: the
Bohr-to-Angstrom conversion, :math:`\hbar c` in eV Angstrom,
:math:`\hbar` in eV s, :math:`k_B` in eV/K, and the electron rest
energy in eV) fix the unit system stated in the conventions charter —
energies in eV, lengths in Angstrom. Numerical guards (epsilon floors,
minimum-sum thresholds, the small-argument cutoff for spherical Bessel
seeds, and the Faddeeva Taylor-series order) centralize the tolerances
that keep kernels finite and differentiable near singular points.
The physical constants and :obj:`L_MAX` are public API;
underscore-prefixed numerical guards are private implementation
details.

Routine Listings
----------------
:obj:`BOHR_TO_ANGSTROM`
    Bohr radius in Angstrom.
:obj:`HBAR_C_EV_A`
    Reduced Planck constant times c in eV Angstrom.
:obj:`HBAR_EV_S`
    Reduced Planck constant in eV s.
:obj:`KB_EV_PER_K`
    Boltzmann constant in eV per kelvin.
:obj:`L_MAX`
    Maximum angular momentum supported by the precomputed table.
:obj:`ME_EV`
    Electron rest energy in eV.

Notes
-----
Changing a private guard value here changes the numerical behavior of
every consumer at once; treat edits as physics changes requiring the
grad-vs-finite-difference gates to be rerun, not as free refactors.
"""

_EPS: float = 1e-12
_GAUNT_IMAG_TOL: float = 1e-12
_MIN_SUM: float = 1e-30
_NORM_EPS: float = 1e-12
_N_TAYLOR: int = 64
_SMALL_ARGUMENT: float = 1e-8

BOHR_TO_ANGSTROM: float = 0.529177
HBAR_C_EV_A: float = 1973.269804
HBAR_EV_S: float = 6.582119569e-16
KB_EV_PER_K: float = 8.617333e-5
L_MAX: int = 4
ME_EV: float = 510998.95

__all__: list[str] = [
    "BOHR_TO_ANGSTROM",
    "HBAR_C_EV_A",
    "HBAR_EV_S",
    "KB_EV_PER_K",
    "L_MAX",
    "ME_EV",
]
