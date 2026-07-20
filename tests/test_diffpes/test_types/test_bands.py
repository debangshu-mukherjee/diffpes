"""Test band, projection, and ARPES spectrum carriers and factories.

Extended Summary
----------------
Covers construction, PyTree round trips, eager and compiled execution, gradient-transparent validation, and rejection contracts for the carriers defined in ``diffpes.types.bands``.
"""

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from diffpes.types import (
    make_arpes_spectrum,
    make_band_structure,
    make_orbital_projection,
    make_spin_band_structure,
    make_spin_orbital_projection,
)
from tests._assertions import assert_rejects


class TestSpinOrbitalProjectionWithOAM:
    """Tests for make_spin_orbital_projection with explicit OAM data.

    Exercises the branch in ``make_spin_orbital_projection`` where
    ``oam`` is not None, covering the OAM array casting code path.
    """

    def test_with_oam_stores_array(self) -> None:
        """Verify that providing oam creates a float64 OAM array in the result.

        Constructs a SpinOrbitalProjection with explicit OAM and asserts
        the ``oam`` field is not None and has the correct shape.
        """
        K, B, A = 2, 3, 1
        proj = jnp.ones((K, B, A, 9), dtype=jnp.float64)
        spin = jnp.zeros((K, B, A, 6), dtype=jnp.float64)
        oam = jnp.ones((K, B, A, 3), dtype=jnp.float64) * 0.5
        sop = make_spin_orbital_projection(
            projections=proj, spin=spin, oam=oam
        )
        assert sop.oam is not None
        assert sop.oam.shape == (K, B, A, 3)
        assert sop.oam.dtype == jnp.float64


class TestBandsPyTree:
    """PyTree round-trip tests for SpinOrbitalProjection and SpinBandStructure.

    Exercises the ``tree_flatten`` / ``tree_unflatten`` methods for
    the spin-aware band and projection types.
    """

    def test_spin_orbital_projection_pytree_round_trip(self) -> None:
        """Verify SpinOrbitalProjection survives a JAX PyTree flatten/unflatten round-trip."""
        K, B, A = 3, 4, 2
        proj = jnp.ones((K, B, A, 9), dtype=jnp.float64)
        spin = jnp.zeros((K, B, A, 6), dtype=jnp.float64)
        sop = make_spin_orbital_projection(projections=proj, spin=spin)
        leaves, treedef = jax.tree_util.tree_flatten(sop)
        sop2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert jnp.allclose(sop2.projections, sop.projections)
        assert jnp.allclose(sop2.spin, sop.spin)
        assert sop2.oam is None

    def test_spin_band_structure_pytree_round_trip(self) -> None:
        """Verify SpinBandStructure survives a JAX PyTree flatten/unflatten round-trip."""
        K, B = 5, 3
        evals_up = jnp.zeros((K, B), dtype=jnp.float64)
        evals_down = jnp.ones((K, B), dtype=jnp.float64)
        kpoints = jnp.zeros((K, 3), dtype=jnp.float64)
        sbs = make_spin_band_structure(
            eigenvalues_up=evals_up,
            eigenvalues_down=evals_down,
            kpoints=kpoints,
            fermi_energy=-1.0,
        )
        leaves, treedef = jax.tree_util.tree_flatten(sbs)
        sbs2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert jnp.allclose(sbs2.eigenvalues_up, sbs.eigenvalues_up)
        assert jnp.allclose(sbs2.eigenvalues_down, sbs.eigenvalues_down)
        assert jnp.allclose(sbs2.fermi_energy, sbs.fermi_energy)


class TestMakeBandStructure(chex.TestCase):
    """Tests for :func:`diffpes.types.bands.make_band_structure`.

    Verifies correct construction of ``BandStructure`` PyTrees including
    output shape validation under both JIT and eager modes, default
    uniform k-point weight generation, and automatic type conversion of
    the ``fermi_energy`` scalar to a JAX array.
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_basic_creation(self) -> None:
        """Verify that a BandStructure is created with correct field shapes.

        Test Logic
        ----------
        1. **Construct band structure**:
           Create zero-filled eigenvalues (10 k-points, 5 bands) and
           k-points arrays, then call the factory via ``self.variant``
           to test both JIT-compiled and eager execution paths.

        2. **Assert shapes**:
           Check that ``eigenvalues`` is (10, 5), ``kpoints`` is (10, 3),
           ``kpoint_weights`` is (10,), and ``fermi_energy`` is scalar.

        Asserts
        -------
        All output fields have the expected shapes, confirming that the
        factory correctly allocates default weights and casts the Fermi
        energy to a 0-D array.
        """
        nk, nb = 10, 5
        eigenvalues = jnp.zeros((nk, nb))
        kpoints = jnp.zeros((nk, 3))
        var_fn = self.variant(make_band_structure)
        bands = var_fn(
            eigenvalues=eigenvalues,
            kpoints=kpoints,
            fermi_energy=0.0,
        )
        chex.assert_shape(bands.eigenvalues, (nk, nb))
        chex.assert_shape(bands.kpoints, (nk, 3))
        chex.assert_shape(bands.kpoint_weights, (nk,))
        chex.assert_shape(bands.fermi_energy, ())

    @chex.variants(with_jit=True, without_jit=True)
    def test_default_weights(self) -> None:
        """Verify that omitting kpoint_weights produces uniform weights.

        Test Logic
        ----------
        1. **Construct without explicit weights**:
           Call ``make_band_structure`` with 8 k-points and no
           ``kpoint_weights`` argument.

        2. **Compare to expected uniform vector**:
           Build an expected weight vector of all ones (float64) with
           length matching the number of k-points.

        Asserts
        -------
        ``bands.kpoint_weights`` is element-wise close to a uniform
        ``ones(8)`` vector, confirming the default-weight logic.
        """
        nk = 8
        eigenvalues = jnp.zeros((nk, 3))
        kpoints = jnp.zeros((nk, 3))
        var_fn = self.variant(make_band_structure)
        bands = var_fn(eigenvalues=eigenvalues, kpoints=kpoints)
        expected = jnp.ones(nk, dtype=jnp.float64)
        chex.assert_trees_all_close(bands.kpoint_weights, expected)

    @chex.variants(with_jit=True, without_jit=True)
    def test_type_conversion(self) -> None:
        """Verify that the fermi_energy Python float is cast to a JAX array.

        Test Logic
        ----------
        1. **Pass a Python float**:
           Supply ``fermi_energy=-1.5`` as a plain Python float to the
           factory function.

        2. **Check result type**:
           Inspect whether ``bands.fermi_energy`` is an instance of
           ``jax.Array``, confirming the factory performed the cast.

        Asserts
        -------
        ``bands.fermi_energy`` is an instance of ``jax.Array``, not a
        raw Python float, verifying the float64 conversion logic.
        """
        eigenvalues = jnp.ones((4, 2))
        kpoints = jnp.zeros((4, 3))
        var_fn = self.variant(make_band_structure)
        bands = var_fn(
            eigenvalues=eigenvalues,
            kpoints=kpoints,
            fermi_energy=-1.5,
        )
        chex.assert_equal(isinstance(bands.fermi_energy, jax.Array), True)

    @chex.variants(with_jit=True, without_jit=True)
    def test_explicit_kpoint_weights(self) -> None:
        """Verify that explicit kpoint_weights are stored correctly.

        Test Logic
        ----------
        1. **Construct with explicit weights**:
           Call ``make_band_structure`` with a non-uniform
           ``kpoint_weights`` array (e.g. length 6).

        2. **Assert stored weights**:
           Check that ``bands.kpoint_weights`` matches the supplied
           array (cast to float64).

        Asserts
        -------
        The else branch (kpoint_weights is not None) is exercised and
        the supplied weights are preserved.
        """
        nk, nb = 6, 2
        eigenvalues = jnp.zeros((nk, nb))
        kpoints = jnp.zeros((nk, 3))
        weights = jnp.linspace(0.5, 1.5, nk, dtype=jnp.float64)
        var_fn = self.variant(make_band_structure)
        bands = var_fn(
            eigenvalues=eigenvalues,
            kpoints=kpoints,
            kpoint_weights=weights,
            fermi_energy=0.0,
        )
        chex.assert_trees_all_close(bands.kpoint_weights, weights, atol=1e-12)

    def test_nonfinite_eigenvalues_raise(self) -> None:
        """Reject non-finite eigenvalues eagerly and under JIT.

        Asserts
        -------
        The value-threaded ``eqx.error_if`` check raises in both execution
        modes instead of silently returning a poisoned carrier.
        """
        eigenvalues = jnp.array([[jnp.nan]], dtype=jnp.float64)
        kpoints = jnp.zeros((1, 3), dtype=jnp.float64)

        for under_jit in (False, True):
            with self.subTest(under_jit=under_jit):
                factory = (
                    eqx.filter_jit(make_band_structure)
                    if under_jit
                    else make_band_structure
                )
                with pytest.raises(RuntimeError, match="eigenvalues finite"):
                    factory(eigenvalues=eigenvalues, kpoints=kpoints)

    def test_validation_is_gradient_transparent(self) -> None:
        """Preserve valid values and their gradients through validation.

        Asserts
        -------
        The validated eigenvalue leaf is bitwise equal to its input and its
        gradient matches direct array construction.
        """
        eigenvalues = jnp.array([[0.25, -0.5]], dtype=jnp.float64)
        kpoints = jnp.zeros((1, 3), dtype=jnp.float64)

        def validated_sum(values):
            bands = make_band_structure(eigenvalues=values, kpoints=kpoints)
            result = jnp.sum(bands.eigenvalues)
            return result

        bands = make_band_structure(eigenvalues=eigenvalues, kpoints=kpoints)
        chex.assert_trees_all_equal(bands.eigenvalues, eigenvalues)
        chex.assert_trees_all_equal(
            jax.grad(validated_sum)(eigenvalues),
            jax.grad(jnp.sum)(eigenvalues),
        )


class TestMakeOrbitalProjection(chex.TestCase):
    """Tests for :func:`diffpes.types.bands.make_orbital_projection`.

    Verifies correct construction of ``OrbitalProjection`` PyTrees including
    output shape validation, default ``None`` handling for optional spin and
    OAM fields, and proper shape storage when spin data is provided.
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_basic_creation(self) -> None:
        """Verify that an OrbitalProjection is created with correct defaults.

        Test Logic
        ----------
        1. **Construct with projections only**:
           Create a zero-filled projections array with shape
           (10 k-points, 5 bands, 2 atoms, 9 orbitals) and call the
           factory without providing optional ``spin`` or ``oam``.

        2. **Assert projection shape**:
           Check that ``orb.projections`` has shape (10, 5, 2, 9).

        3. **Assert optional fields are None**:
           Confirm that both ``orb.spin`` and ``orb.oam`` are ``None``
           when not supplied.

        Asserts
        -------
        Projections shape is correct and optional fields default to
        ``None``, confirming the factory's sentinel-preserving logic.
        """
        nk, nb, na = 10, 5, 2
        proj = jnp.zeros((nk, nb, na, 9))
        var_fn = self.variant(make_orbital_projection)
        orb = var_fn(projections=proj)
        chex.assert_shape(orb.projections, (nk, nb, na, 9))
        chex.assert_equal(orb.spin, None)
        chex.assert_equal(orb.oam, None)

    @chex.variants(with_jit=True, without_jit=True)
    def test_with_spin(self) -> None:
        """Verify that spin projections are stored with the correct shape.

        Test Logic
        ----------
        1. **Construct with spin data**:
           Create projection and spin arrays with compatible leading
           dimensions (4 k-points, 3 bands, 1 atom) and pass both to
           the factory.

        2. **Assert spin shape**:
           Check that ``orb.spin`` has shape (4, 3, 1, 6), matching the
           6 spin-projection channels (up/down for x, y, z).

        Asserts
        -------
        ``orb.spin`` has the expected 4-D shape, confirming that the
        factory correctly casts and stores the optional spin array.
        """
        nk, nb, na = 4, 3, 1
        proj = jnp.ones((nk, nb, na, 9))
        spin = jnp.zeros((nk, nb, na, 6))
        var_fn = self.variant(make_orbital_projection)
        orb = var_fn(projections=proj, spin=spin)
        chex.assert_shape(orb.spin, (nk, nb, na, 6))


class TestMakeArpesSpectrum(chex.TestCase):
    """Tests for :func:`diffpes.types.bands.make_arpes_spectrum`.

    Verifies correct construction of ``ArpesSpectrum`` PyTrees including
    output shape validation for the 2-D intensity map and 1-D energy axis
    under both JIT and eager execution modes.
    """

    @chex.variants(with_jit=True, without_jit=True)
    def test_basic_creation(self) -> None:
        """Verify that an ArpesSpectrum is created with correct field shapes.

        Test Logic
        ----------
        1. **Construct spectrum**:
           Create a zero-filled intensity map (10 k-points, 100 energy
           points) and a linearly spaced energy axis, then call the
           factory via ``self.variant`` to test both JIT and eager modes.

        2. **Assert shapes**:
           Check that ``intensity`` is (10, 100) and ``energy_axis`` is
           (100,).

        Asserts
        -------
        Both output fields have the expected shapes, confirming that the
        factory correctly casts inputs to float64 arrays and preserves
        dimensionality.
        """
        nk, ne = 10, 100
        intensity = jnp.zeros((nk, ne))
        energy_axis = jnp.linspace(-3.0, 1.0, ne)
        var_fn = self.variant(make_arpes_spectrum)
        spec = var_fn(
            intensity=intensity,
            energy_axis=energy_axis,
        )
        chex.assert_shape(spec.intensity, (nk, ne))
        chex.assert_shape(spec.energy_axis, (ne,))


def test_band_factories_reject_invalid_values() -> None:
    """Reject the NaN audit probe and invalid spin-band weights."""
    assert_rejects(
        make_band_structure,
        eigenvalues=jnp.array([[jnp.nan]]),
        kpoints=jnp.zeros((1, 3)),
        match="eigenvalues finite",
    )
    assert_rejects(
        make_spin_band_structure,
        eigenvalues_up=jnp.zeros((1, 1)),
        eigenvalues_down=jnp.zeros((1, 1)),
        kpoints=jnp.zeros((1, 3)),
        kpoint_weights=jnp.array([-1.0]),
        match="weights non negative",
    )


def test_projection_factories_reject_invalid_values() -> None:
    """Reject malformed orbital axes and negative orbital probabilities."""
    assert_rejects(
        make_orbital_projection,
        projections=jnp.zeros((1, 1, 1, 8)),
        match="projections must have 9 orbital columns",
    )
    assert_rejects(
        make_spin_orbital_projection,
        projections=-jnp.ones((1, 1, 1, 9)),
        spin=jnp.zeros((1, 1, 1, 6)),
        match="projections non negative",
    )


def test_arpes_spectrum_rejects_unsorted_energy_axis() -> None:
    """Reject non-increasing ARPES energy coordinates."""
    assert_rejects(
        make_arpes_spectrum,
        intensity=jnp.zeros((1, 2)),
        energy_axis=jnp.array([0.0, 0.0]),
        match="energy axis strictly increasing",
    )
