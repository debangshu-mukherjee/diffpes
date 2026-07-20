"""Utility functions for ARPES simulations.

Extended Summary
----------------
Provides mathematical utilities used throughout the ARPES simulation
pipeline. The Faddeeva function (complex error function) is implemented
via a 64-term Taylor series and is used by the Voigt broadening profile.
Z-score normalization is provided for preprocessing spectra before
comparison with experiment. Complex packing functions provide the sanctioned
real-valued optimizer boundary for complex physics parameters.

The submodules are organized as follows:

- :mod:`math`
    Mathematical utility functions for ARPES simulations.

Routine Listings
----------------
:func:`faddeeva`
    Evaluate the Faddeeva function w(z) = exp(-z^2) erfc(-iz).
:func:`pack_complex`
    Pack complex parameters as stacked real values.
:func:`unpack_complex`
    Unpack stacked real parameters into complex values.
:func:`zscore_normalize`
    Apply z-score normalization (zero-mean, unit-variance).

Notes
-----
All functions are JAX-compatible and support JIT compilation and
automatic differentiation. The Faddeeva implementation uses
``jax.lax.scan`` for the coefficient recurrence (no Python for loops).
"""

from .math import faddeeva, pack_complex, unpack_complex, zscore_normalize

__all__: list[str] = [
    "faddeeva",
    "pack_complex",
    "unpack_complex",
    "zscore_normalize",
]
