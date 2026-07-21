r"""Evaluate energy-dependent self-energy for ARPES simulations.

Extended Summary
----------------
Evaluates the imaginary part of the electronic self-energy
(lifetime broadening) as a function of energy. The self-energy
determines the Lorentzian component of the spectral linewidth
and is critical for modelling correlated materials where
quasiparticle lifetimes vary strongly with energy.

The function supports three parametric models:

- **constant**: Uniform broadening at all energies.
- **polynomial**: Polynomial expansion in energy, for example Fermi-liquid
  quadratic dependence).
- **tabulated**: Piecewise-linear interpolation from user-supplied
  (energy, gamma) pairs, for example from GW or DMFT computations.

All modes are JAX-differentiable with respect to the model
coefficients, enabling gradient-based fitting of self-energy
parameters to experimental linewidths.

Routine Listings
----------------
:func:`evaluate_self_energy`
    Evaluate the imaginary self-energy :math:`\Gamma(E)`.

Notes
-----
The mode string controls Python dispatch outside JAX tracing. Therefore, JAX
compiles one code path for each invocation.
"""

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from diffpes.types import SelfEnergyConfig


@jaxtyped(typechecker=beartype)
def evaluate_self_energy(
    energy: Float[Array, " ..."],
    config: SelfEnergyConfig,
) -> Float[Array, " ..."]:
    r"""Evaluate the imaginary self-energy :math:`\Gamma(E)`.

    The function computes an energy-dependent Lorentzian broadening width from
    the specified self-energy model. The result replaces ``params.gamma`` in
    the Voigt profile when the lifetime changes with energy. This behavior can
    occur near the Fermi level or in correlated materials.

    The imaginary part of the electron self-energy determines the
    quasiparticle lifetime and thus the Lorentzian component of the
    spectral linewidth. The function supports three parametric models:

    - **constant**: A single scalar broadening applied uniformly at
      all energies. Equivalent to using ``params.gamma`` directly.
    - **polynomial**: :math:`\Gamma(E) = \sum_n c_n E^n`. Store
      ``coefficients`` in descending degree order for ``jnp.polyval``. This
      model can represent the quadratic
      Fermi-liquid self-energy :math:`\Gamma \propto (E - E_F)^2`.
    - **tabulated**: Piecewise-linear interpolation of user-supplied
      :math:`(\varepsilon_i, \Gamma_i)` pairs via ``jnp.interp``.
      Suitable for self-energies obtained from many-body calculations
      such as GW or DMFT.

    :see: :class:`~.test_self_energy.TestEvaluateSelfEnergy`

    Implementation Logic
    --------------------
    1. **Read the static mode**::

           mode = config.mode

       The Python string selects one computation while JAX traces its arrays.
    2. **Compute the selected self-energy**::

           result = jnp.broadcast_to(config.coefficients[0], energy.shape)
           result = jnp.polyval(config.coefficients, energy)
           result = jnp.interp(
               energy, config.energy_nodes, config.coefficients
           )

       Each branch returns a width array with the same shape as ``energy``.
    3. **Reject an unknown mode**::

           raise ValueError(msg)

       This static error prevents an unsupported model from entering a trace.

    Parameters
    ----------
    energy : Float[Array, " ..."]
        Energy values in eV at which to evaluate the self-energy.
        Any shape. The output has the same shape.
    config : SelfEnergyConfig
        Self-energy model specification containing:

        - ``mode`` : str -- ``"constant"``, ``"polynomial"``, or
          ``"tabulated"``.
        - ``coefficients`` : array -- Model parameters (scalar for
          constant, polynomial coefficients for polynomial, gamma
          values for tabulated).
        - ``energy_nodes`` : array (only for ``"tabulated"``) --
          Energy grid for the tabulated gamma values.

    Returns
    -------
    result : Float[Array, " ..."]
        Energy-dependent Lorentzian broadening in eV, same shape
        as ``energy``.

    Raises
    ------
    ValueError
        If ``config.mode`` is not one of ``"constant"``,
        ``"polynomial"``, or ``"tabulated"``.

    Notes
    -----
    All three modes are fully JAX-differentiable with respect to
    ``config.coefficients``. The ``"constant"`` and ``"polynomial"``
    modes are also differentiable with respect to ``energy``. The
    ``"tabulated"`` mode uses ``jnp.interp`` which has limited
    gradient support (piecewise-constant gradients).
    """
    mode: str = config.mode

    if mode == "constant":
        result: Float[Array, " ..."] = jnp.broadcast_to(
            config.coefficients[0], energy.shape
        )
        return result
    if mode == "polynomial":
        result: Float[Array, " ..."] = jnp.polyval(config.coefficients, energy)
        return result
    if mode == "tabulated":
        assert config.energy_nodes is not None
        result: Float[Array, " ..."] = jnp.interp(
            energy, config.energy_nodes, config.coefficients
        )
        return result
    msg: str = f"Unknown self-energy mode: {mode}"
    raise ValueError(msg)


__all__: list[str] = ["evaluate_self_energy"]
