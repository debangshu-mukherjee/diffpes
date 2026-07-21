"""Register pure JAX scientific certification checks.

Extended Summary
----------------
Connects stable predicate identities to pure functions evaluated inside the
compiled certification kernel. Registration selects programs statically;
every check returns a :class:`~diffpes.types.DomainResult` with a continuous
margin and a derived Boolean outcome.

Routine Listings
----------------
:func:`get_check`
    Resolve a registered JAX certification check.
:func:`list_checks`
    List registered checks in deterministic identity order.
:func:`register_check`
    Register a stable predicate identity and pure JAX callable.
"""

from beartype import beartype
from beartype.typing import Callable
from jaxtyping import PyTree

from diffpes.types import DomainResult

CheckFunction = Callable[[PyTree], DomainResult]

_CHECKS: tuple[tuple[str, CheckFunction], ...] = ()


@beartype
def register_check(check_id: str, check_fn: CheckFunction) -> None:
    """Register a stable predicate identity and pure JAX callable."""
    global _CHECKS  # noqa: PLW0603
    if not check_id or not check_id.startswith("org.diffpes."):
        msg: str = "check_id must be a stable org.diffpes identifier"
        raise ValueError(msg)
    if any(existing_id == check_id for existing_id, _ in _CHECKS):
        msg = f"duplicate certification check: {check_id}"
        raise ValueError(msg)
    _CHECKS = tuple(sorted((*_CHECKS, (check_id, check_fn))))


@beartype
def get_check(check_id: str) -> CheckFunction:
    """Resolve a registered JAX certification check."""
    for existing_id, check_fn in _CHECKS:
        if existing_id == check_id:
            return check_fn
    raise KeyError(f"unknown certification check: {check_id}")


@beartype
def list_checks() -> tuple[str, ...]:
    """List registered checks in deterministic identity order."""
    checks: tuple[str, ...] = tuple(check_id for check_id, _ in _CHECKS)
    return checks


__all__: list[str] = [
    "CheckFunction",
    "get_check",
    "list_checks",
    "register_check",
]
