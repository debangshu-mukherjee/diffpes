"""Define shared test-side type aliases.

Extended Summary
----------------
Centralizes annotations used by the differentiability test harness. Numerical
aliases are imported from :mod:`diffpes.types` or :mod:`jaxtyping`; this module
does not duplicate production type definitions.
"""

from beartype.typing import Callable, Literal, TypeAlias
from jaxtyping import Array, Float

GradRegime: TypeAlias = Literal["smooth", "stiff", "singular"]
ScalarLoss: TypeAlias = Callable[..., Float[Array, ""]]
