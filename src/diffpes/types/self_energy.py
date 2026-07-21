"""Self-energy configuration data structures.

Extended Summary
----------------
Defines the PyTree type for energy-dependent self-energy
(lifetime broadening) used by the differentiable forward simulator.

Routine Listings
----------------
:class:`SelfEnergyConfig`
    Store energy-dependent self-energy data in a JAX PyTree.
:func:`make_self_energy_config`
    Create a validated ``SelfEnergyConfig`` instance.

Notes
-----
The self-energy coefficients are differentiable (JAX children)
while the mode string is static (auxiliary data) because it
selects different code branches.
"""

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional
from jaxtyping import Array, Float, jaxtyped

from .aliases import ScalarFloat


class SelfEnergyConfig(eqx.Module):
    """Store energy-dependent self-energy data in a JAX PyTree.

    Models the imaginary part of the electronic self-energy
    Im[Sigma(E)] as a function of binding energy. In the forward
    ARPES simulator this replaces the scalar Lorentzian half-width
    ``gamma`` with an energy-dependent broadening that captures
    quasiparticle lifetime effects more realistically.

    Three modes are supported, selected by the ``mode`` string:

    - **constant**: a single scalar broadening ``gamma`` applied
      uniformly at all energies. Equivalent to the standard
      ``SimulationParams.gamma``.
    - **polynomial**: Im[Sigma(E)] = a0 + a1*E + a2*E^2 + ...,
      with coefficients stored in ``coefficients``.
    - **tabulated**: Im[Sigma] is specified at discrete energy nodes
      and interpolated between them. Requires ``energy_nodes``.

    The ``coefficients`` array is the primary differentiable
    quantity: ``jax.grad`` with respect to these coefficients gives
    the sensitivity of the simulated ARPES spectrum to the
    self-energy shape, enabling inverse fitting of lifetime
    broadening from experimental data.


    :see: :class:`~.test_self_energy.TestSelfEnergyConfig`

    Attributes
    ----------
    coefficients : Float[Array, " P"]
        Parameters for the self-energy model. JAX-traced
        (differentiable).

        - mode="constant": P=1, ``[gamma]`` in eV.
        - mode="polynomial": P=degree+1, ``[a0, a1, ...]`` where
          Im[Sigma(E)] = sum_i a_i * E^i.
        - mode="tabulated": P=N, ``[gamma_1, ..., gamma_N]`` in eV
          at the corresponding ``energy_nodes``.
    energy_nodes : Optional[Float[Array, " P"]]
        Energy grid (eV) for tabulated mode. Must have the same
        length P as ``coefficients`` in tabulated mode. ``None``
        for constant and polynomial modes. JAX-traced when present.
    mode : str
        One of ``"constant"``, ``"polynomial"``, ``"tabulated"``.
        **Static** -- a compile-time constant; changing it triggers retracing.

    Notes
    -----
    Implemented as an immutable :class:`equinox.Module` PyTree.
    ``coefficients`` and ``energy_nodes`` are children (on the
    gradient tape); ``mode`` is auxiliary data (compile-time
    constant). Changing ``mode`` triggers JIT recompilation because
    it alters the computation graph.

    See Also
    --------
    make_self_energy_config : Factory function with mode validation
        and default coefficient generation.
    """

    coefficients: Float[Array, " P"]
    energy_nodes: Optional[Float[Array, " P"]]
    mode: str = eqx.field(static=True)


@jaxtyped(typechecker=beartype)
def make_self_energy_config(  # noqa: DOC503
    gamma: ScalarFloat = 0.1,
    mode: str = "constant",
    coefficients: Optional[Float[Array, " Pc"]] = None,
    energy_nodes: Optional[Float[Array, " Pn"]] = None,
) -> SelfEnergyConfig:
    """Create a validated ``SelfEnergyConfig`` instance.

    Factory function that validates inputs and constructs a
    ``SelfEnergyConfig`` PyTree. Performs mode validation (only
    ``"constant"``, ``"polynomial"``, ``"tabulated"`` are accepted)
    and ensures that ``energy_nodes`` is provided when required by
    the tabulated mode.

    The convenience parameter ``gamma`` provides a shortcut for the
    common constant-broadening case: when ``coefficients`` is
    ``None``, a single-element array ``[gamma]`` is created
    automatically.

    :see: :class:`~.test_self_energy.TestMakeSelfEnergyConfig`

    Implementation Logic
    --------------------
    1. **Prepare the normalized values**::

           nodes_arr = None

       This expression gives the later validation steps a stable shape and
       dtype.

    2. **Apply static validation**::

           mode not in ('constant', 'polynomial', 'tabulated')

       This predicate rejects invalid structure before JAX traces the
       numerical checks.

    3. **Apply traced validation**::

           ~jnp.all(jnp.isfinite(coeff_arr))

       This predicate remains active during eager and compiled execution.

    4. **Return the named instance**::

           return config

       The explicit name keeps the implementation and the Returns section
       synchronized.

    Parameters
    ----------
    gamma : ScalarFloat, optional
        Constant broadening in eV. Used as the sole coefficient when
        ``coefficients`` is ``None`` and mode is ``"constant"``.
        Default is 0.1.
    mode : str, optional
        Self-energy model. One of ``"constant"``, ``"polynomial"``,
        ``"tabulated"`` (**static** -- a compile-time constant; changing it
        triggers retracing). Default is ``"constant"``.
    coefficients : Optional[Float[Array, " Pc"]], optional
        Explicit model coefficients. If ``None``, defaults to
        ``[gamma]``. For polynomial mode, these are the polynomial
        coefficients ``[a0, a1, ...]``. For tabulated mode, these
        are the broadening values at each energy node.
    energy_nodes : Optional[Float[Array, " Pn"]], optional
        Energy grid (eV) for tabulated mode. Must have the same
        length as ``coefficients`` in tabulated mode. Ignored for
        other modes. Default is ``None``.

    Returns
    -------
    config : SelfEnergyConfig
        Validated self-energy configuration with ``float64`` arrays.

    Raises
    ------
    ValueError
        If ``mode`` is not one of the three supported strings, or if
        ``energy_nodes`` is present for a non-tabulated mode, absent for
        tabulated mode, or has a different length from ``coefficients``.
    EquinoxRuntimeError
        If coefficients are non-finite or tabulated energy nodes are
        non-finite or not strictly increasing.

    Notes
    -----
    Static validation raises ``ValueError`` before traced construction for an
    unsupported mode, an invalid node-mode combination, or unequal tabulated
    lengths. Traced validation uses ``eqx.error_if`` and raises
    ``EquinoxRuntimeError`` for non-finite coefficients or for tabulated nodes
    that are non-finite or not strictly increasing.

    See Also
    --------
    SelfEnergyConfig : The PyTree class constructed by this factory.
    """
    if mode not in ("constant", "polynomial", "tabulated"):
        msg: str = (
            "mode must be 'constant', 'polynomial',"
            f" or 'tabulated', got '{mode}'"
        )
        raise ValueError(msg)
    coeff_arr: Float[Array, " P"]
    if coefficients is None:
        coeff_arr = jnp.asarray([gamma], dtype=jnp.float64)
    else:
        coeff_arr = jnp.asarray(coefficients, dtype=jnp.float64)
    nodes_arr: Optional[Float[Array, " P"]] = None
    if energy_nodes is not None:
        nodes_arr = jnp.asarray(energy_nodes, dtype=jnp.float64)
    if mode == "tabulated" and nodes_arr is None:
        msg: str = "energy_nodes required for tabulated mode"
        raise ValueError(msg)
    if mode != "tabulated" and nodes_arr is not None:
        msg: str = "energy_nodes are only valid for tabulated mode"
        raise ValueError(msg)
    if nodes_arr is not None and nodes_arr.shape[0] != coeff_arr.shape[0]:
        msg: str = "energy_nodes and coefficients must have the same length"
        raise ValueError(msg)

    def validate_and_create() -> SelfEnergyConfig:
        nonlocal coeff_arr, nodes_arr
        coeff_arr = eqx.error_if(
            coeff_arr,
            ~(jnp.all(jnp.isfinite(coeff_arr))),
            "make_self_energy_config: coefficients finite",
        )
        if nodes_arr is not None:
            nodes_arr = eqx.error_if(
                nodes_arr,
                ~(jnp.all(jnp.isfinite(nodes_arr))),
                "make_self_energy_config: energy nodes finite",
            )
            nodes_arr = eqx.error_if(
                nodes_arr,
                ~(jnp.all(jnp.diff(nodes_arr) > 0.0)),
                "make_self_energy_config: energy nodes strictly increasing",
            )
        validated_config: SelfEnergyConfig = SelfEnergyConfig(
            coefficients=coeff_arr,
            energy_nodes=nodes_arr,
            mode=mode,
        )
        return validated_config

    config: SelfEnergyConfig = validate_and_create()
    return config


__all__: list[str] = [
    "SelfEnergyConfig",
    "make_self_energy_config",
]
