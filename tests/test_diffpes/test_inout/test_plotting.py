"""Tests for ARPES plotting utilities.

Extended Summary
----------------
Exercises the plotting module's public API: plot_arpes_spectrum,
apply_kpath_ticks, plot_arpes_with_kpath, and projected-band scatter
helpers. Tests cover successful
rendering with default and custom options, validation of spectrum array
shapes and compatibility, reuse of an existing axis, color limits and
colorbar, and edge cases such as empty k-path labels. All logic is
documented in the docstrings of each test class and test method.

"""

import chex
import equinox as eqx
import jax.numpy as jnp
import matplotlib
import pytest
from beartype import beartype
from beartype.typing import Any, Callable
from jaxtyping import Array, Float, jaxtyped

import diffpes

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

from diffpes.inout import (
    apply_kpath_ticks,
    list_band_scatter_presets,
    plot_arpes_spectrum,
    plot_arpes_with_kpath,
    plot_band_scatter_preset,
    plot_band_scatter_with_kpath,
)
from diffpes.types import (
    ArpesSpectrum,
    BandStructure,
    KPathInfo,
    OrbitalProjection,
    make_arpes_spectrum,
    make_band_structure,
    make_kpath_info,
    make_orbital_projection,
    make_spin_orbital_projection,
)


def _unvalidated_kpath(
    label_indices: list[int], labels: tuple[str, ...]
) -> KPathInfo:
    """Build malformed legacy k-path metadata for plotting edge tests."""
    kpath: KPathInfo = KPathInfo(
        num_kpoints=jnp.asarray(60, dtype=jnp.int32),
        label_indices=jnp.asarray(label_indices, dtype=jnp.int32),
        points_per_segment=jnp.asarray(0, dtype=jnp.int32),
        segments=jnp.asarray(0, dtype=jnp.int32),
        kpoints=None,
        weights=None,
        grid=None,
        shift=None,
        mode="Line-mode",
        labels=labels,
        comment="",
        coordinate_mode="",
    )
    return kpath


@jaxtyped(typechecker=beartype)
def _make_spectrum(nk: int = 20, ne: int = 120) -> ArpesSpectrum:
    """Build a minimal ArpesSpectrum for plotting tests.

    Creates a valid ArpesSpectrum with intensity shape (nk, ne) and
    energy_axis length ne, so that plot functions receive consistent
    test data without reading files.

    Parameters
    ----------
    nk : int, optional
        Number of k-points (first dimension of intensity). Default 20.
    ne : int, optional
        Number of energy points (second dimension of intensity and
        length of energy_axis). Default 120.

    Returns
    -------
    spectrum : ArpesSpectrum
        PyTree with intensity (nk, ne) and energy_axis (ne,).
    """
    intensity: Float[Array, "nk ne"] = jnp.linspace(
        0.0, 1.0, nk * ne, dtype=jnp.float64
    ).reshape(nk, ne)
    energy_axis: Float[Array, " ne"] = jnp.linspace(-2.0, 0.5, ne)
    spectrum: ArpesSpectrum = make_arpes_spectrum(
        intensity=intensity,
        energy_axis=energy_axis,
    )
    return spectrum


class TestPlotArpesSpectrum(chex.TestCase):
    """Tests for :func:`diffpes.inout.plot_arpes_spectrum`.

    Covers default plotting, optional color limits and colorbar,
    validation of spectrum array dimensions and shape compatibility,
    and reuse of a user-provided matplotlib axis.

    :see: :func:`~diffpes.inout.plot_arpes_spectrum`
    """

    def test_returns_expected_objects(self) -> None:
        """Plot with default options returns figure, axis, and image with correct shape and labels.

        Builds a spectrum and calls plot_arpes_spectrum with colorbar=False.
        Asserts that the returned image array has shape (E, K) i.e. (120, 20)
        after transpose, and that the axis has the default xlabel, ylabel,
        and title. The figure is closed after assertions to avoid leaking
        resources.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        spectrum: diffpes.types.ArpesSpectrum
        fig: Figure
        ax: Axes
        image: AxesImage

        spectrum = _make_spectrum()
        fig, ax, image = plot_arpes_spectrum(spectrum, colorbar=False)
        chex.assert_equal(image.get_array().shape, (120, 20))
        chex.assert_equal(ax.get_xlabel(), "k-point index")
        chex.assert_equal(ax.get_ylabel(), "Energy (eV)")
        chex.assert_equal(ax.get_title(), "Simulated ARPES Spectrum")
        plt.close(fig)

    def test_with_clim_and_colorbar(self) -> None:
        """Passing clim and colorbar=True applies color limits and adds a colorbar.

        Calls plot_arpes_spectrum with colorbar=True and clim=(0.0, 0.5).
        Asserts that the image's color limits are set to (0.0, 0.5) via
        get_clim(), ensuring the clim and colorbar code paths are exercised.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        spectrum: diffpes.types.ArpesSpectrum
        fig: Figure
        ax: Axes
        image: AxesImage

        spectrum = _make_spectrum()
        fig, ax, image = plot_arpes_spectrum(
            spectrum, colorbar=True, clim=(0.0, 0.5)
        )
        chex.assert_equal(image.get_clim(), (0.0, 0.5))
        plt.close(fig)

    def test_validation_rejects_wrong_intensity_ndim(self) -> None:
        """_prepare_plot_arrays raises ValueError when intensity is not 2D.

        Constructs an ArpesSpectrum with 1D intensity (bypassing the
        factory's type checks) and calls plot_arpes_spectrum. Expects
        a ValueError whose message indicates that spectrum.intensity
        must have shape (K, E). This validates the dimension check
        inside the plotting pipeline.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        spectrum: ArpesSpectrum = eqx.tree_at(
            lambda candidate: candidate.intensity,
            _make_spectrum(nk=5, ne=10),
            jnp.linspace(0.0, 1.0, 10),
        )
        with pytest.raises(
            ValueError, match="Expected spectrum.intensity to have shape"
        ):
            plot_arpes_spectrum(spectrum, colorbar=False)

    def test_validation_rejects_wrong_energy_axis_ndim(self) -> None:
        """_prepare_plot_arrays raises ValueError when energy_axis is not 1D.

        Constructs an ArpesSpectrum with 2D energy_axis and calls
        plot_arpes_spectrum. Expects a ValueError whose message
        indicates that spectrum.energy_axis must have shape (E,).

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        spectrum: ArpesSpectrum = eqx.tree_at(
            lambda candidate: candidate.energy_axis,
            _make_spectrum(nk=5, ne=10),
            jnp.zeros((10, 2)),
        )
        with pytest.raises(
            ValueError, match="Expected spectrum.energy_axis to have shape"
        ):
            plot_arpes_spectrum(spectrum, colorbar=False)

    def test_validation_rejects_shape_mismatch(self) -> None:
        """_prepare_plot_arrays raises ValueError when intensity and energy_axis lengths disagree.

        Uses intensity of shape (5, 10) and energy_axis of length 7, so
        intensity.shape[1] != energy_axis.shape[0]. Expects a ValueError
        with a message about incompatible shapes.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        spectrum: ArpesSpectrum = eqx.tree_at(
            lambda candidate: candidate.energy_axis,
            _make_spectrum(nk=5, ne=10),
            jnp.linspace(-1.0, 1.0, 7),
        )
        with pytest.raises(ValueError, match="Incompatible shapes"):
            plot_arpes_spectrum(spectrum, colorbar=False)

    def test_uses_existing_axis(self) -> None:
        """When ax is provided, the same figure and axis are returned and used for plotting.

        Creates a figure and axis with plt.subplots(), then passes ax to
        plot_arpes_spectrum. Asserts that the returned figure and axis
        are the same objects as those passed in, and that the image was
        drawn on the provided axis (no new figure created).

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        spectrum: diffpes.types.ArpesSpectrum
        fig: Figure
        ax: Axes
        out_fig: Figure
        out_ax: Axes

        spectrum = _make_spectrum(nk=10, ne=40)
        fig, ax = plt.subplots()
        out_fig, out_ax, _ = plot_arpes_spectrum(
            spectrum, ax=ax, colorbar=False
        )
        chex.assert_equal(out_fig is fig, True)
        chex.assert_equal(out_ax is ax, True)
        plt.close(fig)


class TestApplyKpathTicks(chex.TestCase):
    """Tests for :func:`diffpes.inout.apply_kpath_ticks`.

    Covers application of symmetry-point ticks and labels to an axis,
    behaviour when the number of labels is less than the number of
    label indices, and the early-return path when there are no labels.

    :see: :func:`~diffpes.inout.apply_kpath_ticks`
    """

    def test_sets_ticks_and_labels(self) -> None:
        """apply_kpath_ticks sets x-axis ticks and labels from KPathInfo.

        Builds a KPathInfo with four label indices and four labels
        (G, M, K, G). Applies apply_kpath_ticks to a fresh axis and
        asserts that the x-tick labels after the call are exactly
        ["G", "M", "K", "G"], confirming that indices and labels
        are applied in order.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        fig: Figure
        ax: Axes
        kpath: diffpes.types.KPathInfo
        labels: list[str]

        fig, ax = plt.subplots()
        kpath = make_kpath_info(
            num_kpoints=60,
            label_indices=[0, 19, 39, 59],
            segments=3,
            labels=("G", "M", "K", "G"),
        )
        apply_kpath_ticks(ax, kpath)
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        chex.assert_equal(labels, ["G", "M", "K", "G"])
        plt.close(fig)

    def test_handles_label_index_mismatch(self) -> None:
        """When labels are fewer than label_indices, only the available labels are used.

        Provides four label indices but only two labels (G, M). Asserts
        that the axis ends up with exactly two tick labels, ["G", "M"],
        so that the truncation logic (min(len(indices), len(labels)))
        is exercised and no index error occurs.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        fig: Figure
        ax: Axes
        kpath: diffpes.types.KPathInfo
        labels: list[str]

        fig, ax = plt.subplots()
        kpath = _unvalidated_kpath([0, 19, 39, 59], ("G", "M"))
        apply_kpath_ticks(ax, kpath)
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        chex.assert_equal(labels, ["G", "M"])
        plt.close(fig)

    def test_empty_labels_returns_ax_unchanged(self) -> None:
        """When KPathInfo has no labels, apply_kpath_ticks returns the axis without setting ticks.

        Builds a KPathInfo with empty label_indices and empty labels so
        that n_labels is zero. Asserts that the return value is the
        same axis object and that the early-return path (no tick/label
        setting) is taken without error.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        fig: Figure
        ax: Axes
        kpath: diffpes.types.KPathInfo
        out: Array

        fig, ax = plt.subplots()
        kpath = _unvalidated_kpath([], ())
        out = apply_kpath_ticks(ax, kpath)
        chex.assert_equal(out is ax, True)
        plt.close(fig)


class TestPlotArpesWithKpath(chex.TestCase):
    """Tests for :func:`diffpes.inout.plot_arpes_with_kpath`.

    Covers the combined workflow of plotting an ARPES spectrum and
    annotating the k-axis with symmetry labels from KPathInfo.

    :see: :func:`~diffpes.inout.plot_arpes_with_kpath`
    """

    def test_combined_plot(self) -> None:
        """plot_arpes_with_kpath produces a spectrum image and applies k-path ticks and labels.

        Builds a spectrum and a KPathInfo with three symmetry points.
        Calls plot_arpes_with_kpath and asserts that the image array
        has the expected shape (120, 20), that the x-tick labels are
        ("G", "M", "K"), and that the x-axis label is "Momentum (k)",
        confirming that both the spectrum plot and the k-path
        annotation are applied correctly.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        spectrum: diffpes.types.ArpesSpectrum
        kpath: diffpes.types.KPathInfo
        fig: Figure
        ax: Axes
        image: AxesImage
        labels: list[str]

        spectrum = _make_spectrum()
        kpath = make_kpath_info(
            num_kpoints=20,
            label_indices=[0, 9, 19],
            segments=2,
            labels=("G", "M", "K"),
        )
        fig, ax, image = plot_arpes_with_kpath(
            spectrum=spectrum,
            kpath=kpath,
            colorbar=False,
        )
        chex.assert_equal(image.get_array().shape, (120, 20))
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        chex.assert_equal(labels, ["G", "M", "K"])
        chex.assert_equal(ax.get_xlabel(), "Momentum (k)")
        plt.close(fig)


@jaxtyped(typechecker=beartype)
def _make_band_and_projection(
    nk: int = 12,
    nb: int = 3,
    na: int = 2,
) -> tuple[BandStructure, OrbitalProjection]:
    """Build minimal band/projection inputs for band-scatter tests."""
    eigen: Float[Array, "nk nb"] = jnp.linspace(
        -1.2, 0.8, nk * nb, dtype=jnp.float64
    ).reshape(nk, nb)
    kx: Float[Array, " nk"] = jnp.linspace(0.0, 1.0, nk, dtype=jnp.float64)
    kpoints: Float[Array, "nk 3"] = jnp.stack(
        [kx, jnp.zeros_like(kx), jnp.zeros_like(kx)],
        axis=1,
    )
    bands: BandStructure = make_band_structure(
        eigenvalues=eigen,
        kpoints=kpoints,
        fermi_energy=0.15,
    )

    projections: Float[Array, "nk nb na 9"] = (
        jnp.ones((nk, nb, na, 9), dtype=jnp.float64) * 0.05
    )
    projections = projections.at[..., 1:4].set(0.2)
    spin: Float[Array, "nk nb na 6"] = jnp.zeros(
        (nk, nb, na, 6), dtype=jnp.float64
    )
    spin = spin.at[: nk // 2, ..., 4].set(0.2)
    spin = spin.at[nk // 2 :, ..., 5].set(0.3)
    orbital_projection: OrbitalProjection = make_orbital_projection(
        projections=projections,
        spin=spin,
    )
    result: tuple[BandStructure, OrbitalProjection] = (
        bands,
        orbital_projection,
    )
    return result


class TestListBandScatterPresets(chex.TestCase):
    """Validate :func:`~diffpes.inout.list_band_scatter_presets`.

    Covers the stable public names for orbital, spin, and orbital-angular-
    momentum scatter modes.

    :see: :func:`~diffpes.inout.list_band_scatter_presets`
    """

    def test_returns_each_preset_family(self) -> None:
        """Return at least one name from each supported preset family.

        The result must expose orbital, signed-spin, and OAM choices so callers
        can build selection controls without reading private tables.

        Notes
        -----
        Calls the listing function once and checks representative public names
        in the returned immutable tuple.
        """
        presets: tuple[str, ...]

        presets = list_band_scatter_presets()
        assert "p" in presets
        assert "spin_z" in presets
        assert "oam_total" in presets


class TestPlotBandScatterPreset(chex.TestCase):
    """Tests for projected-band scatter plotting presets.

    :see: :func:`~diffpes.inout.plot_band_scatter_preset`
    """

    def test_lists_presets(self) -> None:
        """list_band_scatter_presets returns known keys.

        This case establishes the lists presets contract for plot band scatter with the
        concrete values and array shapes described below.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        presets: tuple[str, ...]

        presets = list_band_scatter_presets()
        assert "p" in presets
        assert "d" in presets
        assert "spin_z" in presets
        assert "oam_total" in presets

    def test_orbital_preset_plot(self) -> None:
        """Orbital preset renders a scatter with one point per (k, band).

        This case establishes the orbital preset plot contract for plot band scatter
        with the concrete values and array shapes described below.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        bands: diffpes.types.BandStructure
        orb: diffpes.types.OrbitalProjection
        fig: Figure
        ax: Axes
        scatter: PathCollection

        bands, orb = _make_band_and_projection(nk=10, nb=4, na=2)
        fig, ax, scatter = plot_band_scatter_preset(
            bands=bands,
            orb_proj=orb,
            preset="p",
            colorbar=False,
        )
        chex.assert_equal(scatter.get_offsets().shape[0], 40)
        chex.assert_equal(ax.get_xlabel(), "Momentum (k)")
        chex.assert_equal(ax.get_ylabel(), "Energy (eV)")
        plt.close(fig)

    def test_signed_spin_preset_with_colorbar(self) -> None:
        """Signed spin preset can render with colorbar.

        This case establishes the signed spin preset with colorbar contract for plot
        band scatter with the concrete values and array shapes described below.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        bands: diffpes.types.BandStructure
        orb: diffpes.types.OrbitalProjection
        fig: Figure

        bands, orb = _make_band_and_projection(nk=8, nb=2, na=2)
        fig, _, _ = plot_band_scatter_preset(
            bands=bands,
            orb_proj=orb,
            preset="spin_z",
            colorbar=True,
        )

        chex.assert_equal(len(fig.axes), 2)
        plt.close(fig)

    def test_spin_preset_requires_spin_data(self) -> None:
        """Spin presets raise when projection has no spin field.

        This case establishes the spin preset requires spin data contract for plot band
        scatter with the concrete values and array shapes described below.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        bands: diffpes.types.BandStructure
        orb: diffpes.types.OrbitalProjection
        no_spin: diffpes.types.OrbitalProjection

        bands, orb = _make_band_and_projection(nk=8, nb=2, na=1)
        no_spin = make_orbital_projection(
            projections=orb.projections, spin=None
        )
        with pytest.raises(ValueError, match="requires spin data"):
            plot_band_scatter_preset(
                bands=bands,
                orb_proj=no_spin,
                preset="spin_z",
            )

    def test_band_scatter_with_kpath(self) -> None:
        """K-path wrapper applies symmetry labels on projected-band scatter.

        This case establishes the band scatter with kpath contract for plot band scatter
        with the concrete values and array shapes described below.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        bands: diffpes.types.BandStructure
        orb: diffpes.types.OrbitalProjection
        kpath: diffpes.types.KPathInfo
        fig: Figure
        ax: Axes
        scatter: PathCollection
        labels: list[str]

        bands, orb = _make_band_and_projection(nk=10, nb=3, na=2)
        kpath = make_kpath_info(
            num_kpoints=10,
            label_indices=[0, 4, 9],
            segments=2,
            labels=("G", "M", "K"),
        )
        fig, ax, scatter = plot_band_scatter_with_kpath(
            bands=bands,
            orb_proj=orb,
            kpath=kpath,
            preset="d",
            colorbar=False,
        )
        chex.assert_equal(scatter.get_offsets().shape[0], 30)
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        chex.assert_equal(labels, ["G", "M", "K"])
        plt.close(fig)


class TestPlotBandScatterWithKpath(chex.TestCase):
    """Validate :func:`~diffpes.inout.plot_band_scatter_with_kpath`.

    Covers composition of projected-band marker weights with line-mode
    symmetry labels on the shared momentum axis.

    :see: :func:`~diffpes.inout.plot_band_scatter_with_kpath`
    """

    def test_applies_labels_to_projected_bands(self) -> None:
        """Apply all supplied symmetry labels to the scatter axis.

        A ten-point, three-band fixture must produce thirty offsets and retain
        the three requested high-symmetry labels in order.

        Notes
        -----
        Builds deterministic band and projection carriers, applies a three-label
        ``KPathInfo``, and checks the collection size and rendered tick text.
        """
        bands: diffpes.types.BandStructure
        orb: diffpes.types.OrbitalProjection
        kpath: diffpes.types.KPathInfo
        fig: Figure
        ax: Axes
        scatter: PathCollection
        labels: list[str]

        bands, orb = _make_band_and_projection(nk=10, nb=3, na=2)
        kpath = make_kpath_info(
            num_kpoints=10,
            label_indices=[0, 4, 9],
            segments=2,
            labels=("G", "M", "K"),
        )
        fig, ax, scatter = plot_band_scatter_with_kpath(
            bands=bands,
            orb_proj=orb,
            kpath=kpath,
            preset="d",
            colorbar=False,
        )
        chex.assert_equal(scatter.get_offsets().shape[0], 30)
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        chex.assert_equal(labels, ["G", "M", "K"])
        plt.close(fig)


class TestPlotBandScatterEdgeCases(chex.TestCase):
    """Edge-case tests for the band-scatter plotting helpers.

    Covers remaining uncovered lines in plotting.py:
    - Lines 449-450: ``_prepare_band_arrays`` ValueError for wrong ndim
    - Lines 463-464: ``_subset_atom_axis`` with explicit atom_indices
    - Lines 497-498: orbital-index preset ("s")
    - Line 502: "total" preset
    - Line 528: spin_channel preset ("spin_z_up")
    - Line 541: OAM array subsetting when oam is not None
    - Lines 551-553: OAM preset requires OAM data (oam is None → ValueError)
    - Lines 554-556: oam_component preset branch
    - Lines 562-564: unknown preset → ValueError
    - Lines 644-648: weight shape mismatch → ValueError
    - Line 667: ``fig = ax.figure`` when ax is provided

    :see: :func:`~diffpes.inout.plot_band_scatter_preset`
    """

    def _make_bands_1d(self, nk=4, nb=2):
        """Build BandStructure with 1D eigenvalues (bypassing factory)."""
        valid_bands: BandStructure = make_band_structure(
            eigenvalues=jnp.zeros((nk, nb), dtype=jnp.float64),
            kpoints=jnp.zeros((nk, 3), dtype=jnp.float64),
            kpoint_weights=jnp.zeros(nk, dtype=jnp.float64),
            fermi_energy=jnp.float64(0.0),
        )
        malformed_bands: BandStructure = eqx.tree_at(
            lambda candidate: candidate.eigenvalues,
            valid_bands,
            jnp.zeros(nk * nb, dtype=jnp.float64),
        )
        return malformed_bands

    def _make_orb_with_spin_and_oam(self, nk=4, nb=2, na=1):
        """Build OrbitalProjection with spin and OAM attached."""
        proj: Array
        spin: Array
        oam: Array

        proj = jnp.ones((nk, nb, na, 9), dtype=jnp.float64) * 0.1
        spin = jnp.zeros((nk, nb, na, 6), dtype=jnp.float64)
        spin = spin.at[..., 0].set(0.3)
        spin = spin.at[..., 4].set(0.2)
        oam = jnp.ones((nk, nb, na, 3), dtype=jnp.float64) * 0.05
        return make_orbital_projection(projections=proj, spin=spin, oam=oam)

    def test_prepare_band_arrays_wrong_ndim_raises(self) -> None:
        """``_prepare_band_arrays`` with 1D eigenvalues raises ValueError (lines 449-450).

        Constructs a BandStructure with 1D eigenvalues (bypassing the
        factory), then calls ``plot_band_scatter_preset``. Asserts a
        ``ValueError`` matching ``"shape (K, B)"``.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        bands: diffpes.types.BandStructure
        proj: Array
        orb: diffpes.types.OrbitalProjection

        bands = self._make_bands_1d(nk=4, nb=2)
        proj = jnp.ones((4, 2, 1, 9), dtype=jnp.float64) * 0.1
        orb = make_orbital_projection(projections=proj)
        with pytest.raises(ValueError, match="shape"):
            plot_band_scatter_preset(bands=bands, orb_proj=orb, preset="p")

    def test_subset_atom_axis_with_indices(self) -> None:
        """Passing atom_indices calls ``_subset_atom_axis`` (lines 463-464).

        Plots the "p" preset with ``atom_indices=[0]`` so that
        ``_subset_atom_axis`` is called with a non-None index array.
        Asserts the scatter point count equals nk * nb.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        nk: int
        nb: int
        na: int
        eigen: Array
        kpoints: Array
        bands: diffpes.types.BandStructure
        proj: Array
        orb: diffpes.types.OrbitalProjection
        fig: Figure
        ax: Axes
        scatter: PathCollection

        nk, nb, na = 6, 2, 2
        eigen = jnp.linspace(-1.0, 0.5, nk * nb, dtype=jnp.float64).reshape(
            nk, nb
        )
        kpoints = jnp.zeros((nk, 3), dtype=jnp.float64)
        bands = make_band_structure(eigenvalues=eigen, kpoints=kpoints)
        proj = jnp.ones((nk, nb, na, 9), dtype=jnp.float64) * 0.1
        orb = make_orbital_projection(projections=proj)
        fig, ax, scatter = plot_band_scatter_preset(
            bands=bands,
            orb_proj=orb,
            preset="p",
            atom_indices=[0],
            colorbar=False,
        )
        chex.assert_equal(scatter.get_offsets().shape[0], nk * nb)
        plt.close(fig)

    def test_s_orbital_preset(self) -> None:
        """'s' preset uses ORBITAL_INDEX branch (lines 497-498).

        Calls ``plot_band_scatter_preset`` with ``preset='s'``. Asserts
        that the scatter renders without error and the point count is
        correct (exercises the ``ORBITAL_INDEX[key]`` branch).

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        nk: int
        nb: int
        eigen: Array
        bands: diffpes.types.BandStructure
        proj: Array
        orb: diffpes.types.OrbitalProjection
        fig: Figure
        ax: Axes
        scatter: PathCollection

        nk, nb = 6, 2
        eigen = jnp.linspace(-1.0, 0.5, nk * nb, dtype=jnp.float64).reshape(
            nk, nb
        )
        bands = make_band_structure(
            eigenvalues=eigen,
            kpoints=jnp.zeros((nk, 3), dtype=jnp.float64),
        )
        proj = jnp.ones((nk, nb, 1, 9), dtype=jnp.float64) * 0.1
        orb = make_orbital_projection(projections=proj)
        fig, ax, scatter = plot_band_scatter_preset(
            bands=bands, orb_proj=orb, preset="s", colorbar=False
        )
        chex.assert_equal(scatter.get_offsets().shape[0], nk * nb)
        plt.close(fig)

    def test_total_preset(self) -> None:
        """'total' preset sums all orbital channels (line 502).

        Calls ``plot_band_scatter_preset`` with ``preset='total'``.
        Asserts the scatter renders and the point count is correct
        (exercises the ``elif key == 'total'`` branch at line 501-502).

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        nk: int
        nb: int
        eigen: Array
        bands: diffpes.types.BandStructure
        proj: Array
        orb: diffpes.types.OrbitalProjection
        fig: Figure
        ax: Axes
        scatter: PathCollection

        nk, nb = 4, 3
        eigen = jnp.linspace(-1.0, 0.5, nk * nb, dtype=jnp.float64).reshape(
            nk, nb
        )
        bands = make_band_structure(
            eigenvalues=eigen,
            kpoints=jnp.zeros((nk, 3), dtype=jnp.float64),
        )
        proj = jnp.ones((nk, nb, 1, 9), dtype=jnp.float64) * 0.05
        orb = make_orbital_projection(projections=proj)
        fig, ax, scatter = plot_band_scatter_preset(
            bands=bands, orb_proj=orb, preset="total", colorbar=False
        )
        chex.assert_equal(scatter.get_offsets().shape[0], nk * nb)
        plt.close(fig)

    def test_spin_channel_preset(self) -> None:
        """'spin_z_up' preset uses the spin_channel branch (line 528).

        Calls ``plot_band_scatter_preset`` with ``preset='spin_z_up'``.
        This exercises the ``if key in spin_channel`` branch (line 527-528).
        Asserts the scatter renders without error.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        nk: int
        nb: int
        eigen: Array
        bands: diffpes.types.BandStructure
        proj: Array
        spin: Array
        orb: diffpes.types.OrbitalProjection
        fig: Figure
        ax: Axes
        scatter: PathCollection

        nk, nb = 4, 2
        eigen = jnp.linspace(-1.0, 0.5, nk * nb, dtype=jnp.float64).reshape(
            nk, nb
        )
        bands = make_band_structure(
            eigenvalues=eigen,
            kpoints=jnp.zeros((nk, 3), dtype=jnp.float64),
        )
        proj = jnp.ones((nk, nb, 1, 9), dtype=jnp.float64) * 0.1
        spin = jnp.ones((nk, nb, 1, 6), dtype=jnp.float64) * 0.1
        orb = make_orbital_projection(projections=proj, spin=spin)
        fig, ax, scatter = plot_band_scatter_preset(
            bands=bands, orb_proj=orb, preset="spin_z_up", colorbar=False
        )
        chex.assert_equal(scatter.get_offsets().shape[0], nk * nb)
        plt.close(fig)

    def test_oam_preset_with_oam_data(self) -> None:
        """OAM preset with OAM data exercises oam_arr subsetting and component branch.

        Calls ``plot_band_scatter_preset`` with ``preset='oam_total'``
        and an OrbitalProjection that has ``oam`` data. This exercises
        line 541 (``oam_arr = _subset_atom_axis(...)``) and lines
        554-556 (``if key in oam_component: weights = ...``).

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        nk: int
        nb: int
        eigen: Array
        bands: diffpes.types.BandStructure
        orb: diffpes.types.OrbitalProjection
        fig: Figure
        ax: Axes
        scatter: PathCollection

        nk, nb = 4, 2
        eigen = jnp.linspace(-1.0, 0.5, nk * nb, dtype=jnp.float64).reshape(
            nk, nb
        )
        bands = make_band_structure(
            eigenvalues=eigen,
            kpoints=jnp.zeros((nk, 3), dtype=jnp.float64),
        )
        orb = self._make_orb_with_spin_and_oam(nk=nk, nb=nb, na=1)
        fig, ax, scatter = plot_band_scatter_preset(
            bands=bands, orb_proj=orb, preset="oam_total", colorbar=False
        )
        chex.assert_equal(scatter.get_offsets().shape[0], nk * nb)
        plt.close(fig)

    def test_oam_abs_total_preset(self) -> None:
        """'oam_abs_total' preset uses ``np.abs`` on OAM component 2 (lines 557-559).

        Calls ``plot_band_scatter_preset`` with ``preset='oam_abs_total'``
        and an OrbitalProjection with OAM data present. This exercises
        the ``elif key == 'oam_abs_total'`` branch at lines 557-559,
        which computes ``np.sum(np.abs(oam_arr[..., 2]), axis=2)`` and
        sets ``signed = False``. Asserts the scatter renders without error.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        nk: int
        nb: int
        eigen: Array
        bands: diffpes.types.BandStructure
        orb: diffpes.types.OrbitalProjection
        fig: Figure
        ax: Axes
        scatter: PathCollection

        nk, nb = 4, 2
        eigen = jnp.linspace(-1.0, 0.5, nk * nb, dtype=jnp.float64).reshape(
            nk, nb
        )
        bands = make_band_structure(
            eigenvalues=eigen,
            kpoints=jnp.zeros((nk, 3), dtype=jnp.float64),
        )
        orb = self._make_orb_with_spin_and_oam(nk=nk, nb=nb, na=1)
        fig, ax, scatter = plot_band_scatter_preset(
            bands=bands, orb_proj=orb, preset="oam_abs_total", colorbar=False
        )
        chex.assert_equal(scatter.get_offsets().shape[0], nk * nb)
        plt.close(fig)

    def test_oam_preset_without_oam_data_raises(self) -> None:
        """OAM preset without OAM data raises ValueError (lines 551-553).

        Calls ``plot_band_scatter_preset`` with ``preset='oam_p'`` but
        provides an OrbitalProjection with ``oam=None``. Asserts
        ``ValueError`` matching ``"requires OAM data"``.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        nk: int
        nb: int
        eigen: Array
        bands: diffpes.types.BandStructure
        proj: Array
        orb: diffpes.types.OrbitalProjection

        nk, nb = 4, 2
        eigen = jnp.linspace(-1.0, 0.5, nk * nb, dtype=jnp.float64).reshape(
            nk, nb
        )
        bands = make_band_structure(
            eigenvalues=eigen,
            kpoints=jnp.zeros((nk, 3), dtype=jnp.float64),
        )
        proj = jnp.ones((nk, nb, 1, 9), dtype=jnp.float64) * 0.1
        orb = make_orbital_projection(projections=proj)
        with pytest.raises(ValueError, match="requires OAM data"):
            plot_band_scatter_preset(bands=bands, orb_proj=orb, preset="oam_p")

    def test_unknown_preset_raises(self) -> None:
        """Unknown preset string raises ValueError (lines 562-564).

        Calls ``plot_band_scatter_preset`` with an unrecognized preset
        name. Asserts ``ValueError`` matching ``"Unknown preset"``.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        nk: int
        nb: int
        eigen: Array
        bands: diffpes.types.BandStructure
        proj: Array
        orb: diffpes.types.OrbitalProjection

        nk, nb = 4, 2
        eigen = jnp.linspace(-1.0, 0.5, nk * nb, dtype=jnp.float64).reshape(
            nk, nb
        )
        bands = make_band_structure(
            eigenvalues=eigen,
            kpoints=jnp.zeros((nk, 3), dtype=jnp.float64),
        )
        proj = jnp.ones((nk, nb, 1, 9), dtype=jnp.float64) * 0.1
        orb = make_orbital_projection(projections=proj)
        with pytest.raises(ValueError, match="Unknown preset"):
            plot_band_scatter_preset(
                bands=bands, orb_proj=orb, preset="not_a_real_preset"
            )

    def test_weight_shape_mismatch_raises(self) -> None:
        """Weights shape != eigenvalues shape raises ValueError (lines 644-648).

        Creates a BandStructure with shape (4, 2) but an OrbitalProjection
        with shape (4, 3, 1, 9), so the "p" preset produces weights of
        shape (4, 3) != (4, 2). Asserts ``ValueError`` matching
        ``"Preset weights must have shape"``.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        nk: int
        nb_bands: int
        nb_proj: int
        eigen: Array
        bands: diffpes.types.BandStructure
        proj: Array
        orb: diffpes.types.OrbitalProjection

        nk = 4
        nb_bands = 2
        nb_proj = 3
        eigen = jnp.linspace(
            -1.0, 0.5, nk * nb_bands, dtype=jnp.float64
        ).reshape(nk, nb_bands)
        bands = make_band_structure(
            eigenvalues=eigen,
            kpoints=jnp.zeros((nk, 3), dtype=jnp.float64),
        )
        proj = jnp.ones((nk, nb_proj, 1, 9), dtype=jnp.float64) * 0.1
        orb = make_orbital_projection(projections=proj)
        with pytest.raises(ValueError, match="Preset weights must have shape"):
            plot_band_scatter_preset(bands=bands, orb_proj=orb, preset="p")

    def test_uses_provided_ax(self) -> None:
        """When ax is provided, ``fig = ax.figure`` is used (line 667).

        Creates a figure and axis, passes them to ``plot_band_scatter_preset``,
        and asserts the returned figure is the same object, confirming
        that line 667 (``fig = ax.figure``) is executed instead of
        ``fig, ax = plt.subplots()``.

        Notes
        -----
        Builds the inputs in the test body and checks the stated property with the documented numerical or structural assertions."""
        nk: int
        nb: int
        eigen: Array
        bands: diffpes.types.BandStructure
        proj: Array
        orb: diffpes.types.OrbitalProjection
        fig0: Figure
        ax0: Axes
        out_fig: Figure
        out_ax: Axes

        nk, nb = 6, 2
        eigen = jnp.linspace(-1.0, 0.5, nk * nb, dtype=jnp.float64).reshape(
            nk, nb
        )
        bands = make_band_structure(
            eigenvalues=eigen,
            kpoints=jnp.zeros((nk, 3), dtype=jnp.float64),
        )
        proj = jnp.ones((nk, nb, 1, 9), dtype=jnp.float64) * 0.1
        orb = make_orbital_projection(projections=proj)
        fig0, ax0 = plt.subplots()
        out_fig, out_ax, _ = plot_band_scatter_preset(
            bands=bands, orb_proj=orb, preset="p", ax=ax0, colorbar=False
        )
        chex.assert_equal(out_fig is fig0, True)
        plt.close(fig0)
