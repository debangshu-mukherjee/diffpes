"""Differentiable ARPES simulations in JAX.

Extended Summary
----------------
A comprehensive toolkit for Angle-Resolved PhotoEmission
Spectroscopy (ARPES) simulations using JAX for automatic
differentiation and GPU acceleration. The package is built
around a bidirectional thesis: the same differentiable physics
that maps electronic structure forward to ARPES spectra also
supports gradient-based inverse recovery of band-structure
parameters from measured spectra. Supports six levels of
physical sophistication from basic Gaussian convolution to full
polarization-dependent dipole matrix element calculations.

Routine Listings
----------------
:mod:`inout`
    VASP file parsers for ARPES simulation input.
:mod:`maths`
    Angular matrix elements for dipole photoemission.
:mod:`radial`
    Differentiable radial primitives for ARPES matrix elements.
:mod:`simul`
    ARPES simulation functions at six complexity levels.
:mod:`tightb`
    Provide native tight-binding tools and ARPES-side adapters.
:mod:`types`
    Type definitions and factory functions for diffpes.
:mod:`utils`
    Utility functions for ARPES simulations.

Examples
--------
>>> import diffpes
>>> bands = diffpes.inout.read_eigenval("EIGENVAL", fermi_energy=-1.5)
>>> orb = diffpes.inout.read_procar("PROCAR")
>>> params = diffpes.types.make_simulation_params(sigma=0.04)
>>> spectrum = diffpes.simul.simulate_basic(bands, orb, params)

Notes
-----
All computations are JAX-compatible and support automatic
differentiation for gradient-based optimization of ARPES
simulation parameters. 64-bit precision is enabled at import,
and XLA CPU threading flags are set before the JAX import so
CPU execution uses multi-threaded kernels.
"""

import collections.abc
import os
from importlib.metadata import version

if not hasattr(collections.abc, "ByteString"):
    setattr(  # noqa: B010 -- Python 3.14 compatibility for beartype 0.22.9.
        collections.abc,
        "ByteString",
        collections.abc.Buffer,
    )

os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=0",
)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

from . import inout, maths, radial, simul, tightb, types, utils  # noqa: E402

__version__: str = version("diffpes")

__all__: list[str] = [
    "inout",
    "maths",
    "radial",
    "simul",
    "tightb",
    "types",
    "utils",
]
