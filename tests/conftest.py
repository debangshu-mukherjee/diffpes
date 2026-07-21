"""Provide the shared pytest runtime foundation.

Extended Summary
----------------
The autouse fixtures in this module enforce the numerical and resource
contracts shared by the complete test suite. They verify JAX 64-bit mode,
clear compilation caches between tests, and reject excessive retained memory.
The collection hook serializes memory-intensive tests under pytest-xdist, and
the random-key fixture supplies stable, worker-independent test randomness.
"""

import hashlib

# ruff: noqa: I001 -- diffpes must configure JAX before JAX is imported.
import diffpes  # noqa: F401 -- Import activates the package-wide x64 contract.

import jax
import jax.numpy as jnp
import psutil
import pytest
from beartype import beartype
from beartype.typing import Iterator
from jaxtyping import PRNGKeyArray, jaxtyped

pytest_plugins: tuple[str, ...] = ("pytester",)

RSS_LEAK_LIMIT_MB: float = 500.0
_BYTES_PER_MEBIBYTE: int = 1024**2


@pytest.fixture(autouse=True, scope="session")
def assert_x64_enabled() -> Iterator[None]:
    """Require JAX 64-bit precision for the complete test session.

    Yields
    ------
    None
        Control to the test session after validating the default JAX dtype.

    Notes
    -----
    Asserts that importing diffpes selected ``jax.numpy.float64`` as the
    default scalar dtype before the suite begins.
    """
    actual_dtype: jnp.dtype = jnp.zeros(()).dtype
    assert actual_dtype == jnp.float64, (
        "diffpes tests require JAX 64-bit mode; "
        f"the default scalar dtype is {actual_dtype}."
    )
    yield


@pytest.fixture(autouse=True)
def rss_leak_guard(request: pytest.FixtureRequest) -> Iterator[None]:
    """Reject process memory growth beyond the test-specific RSS limit.

    The default limit is 500 MiB. ``@pytest.mark.rss_limit_mb(limit)`` may be
    used for a test whose expected retained allocation needs a different bound.
    JAX caches are cleared both before the baseline and before the final
    measurement, so compiled executables do not hide application-level leaks.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Pytest request carrying the current test node and its markers.

    Yields
    ------
    None
        Control to the test after recording its baseline RSS.
    """
    limit_marker: pytest.Mark | None = request.node.get_closest_marker(
        "rss_limit_mb"
    )
    limit_mb: float = RSS_LEAK_LIMIT_MB
    if limit_marker is not None:
        if len(limit_marker.args) != 1:
            pytest.fail("rss_limit_mb requires exactly one numeric argument")
        limit_mb = float(limit_marker.args[0])

    process: psutil.Process = psutil.Process()
    jax.clear_caches()
    rss_before: int = process.memory_info().rss
    yield
    jax.clear_caches()
    rss_after: int = process.memory_info().rss
    growth_mb: float = (rss_after - rss_before) / _BYTES_PER_MEBIBYTE
    assert growth_mb <= limit_mb, (
        f"{request.node.nodeid} retained {growth_mb:.1f} MiB RSS; "
        f"limit is {limit_mb:.1f} MiB"
    )


@pytest.fixture(autouse=True)
def clear_jax_caches(rss_leak_guard: None) -> Iterator[None]:
    """Clear JAX caches after every test and before its RSS measurement.

    Parameters
    ----------
    rss_leak_guard : None
        Dependency that orders cache clearing before the guard's teardown.

    Yields
    ------
    None
        Control to the test before clearing all JAX compilation caches.
    """
    yield
    jax.clear_caches()


@jaxtyped(typechecker=beartype)
def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Serialize memory-intensive tests on one pytest-xdist worker.

    Parameters
    ----------
    config : pytest.Config
        Active pytest configuration. Required by the pytest hook contract.
    items : list[pytest.Item]
        Collected test items to augment with xdist grouping metadata.
    """
    del config
    xdist_group: pytest.MarkDecorator = pytest.mark.xdist_group("big_mem")
    item: pytest.Item
    for item in items:
        if item.get_closest_marker("big_mem") is not None:
            item.add_marker(xdist_group)


@pytest.fixture
@jaxtyped(typechecker=beartype)
def rng_key(request: pytest.FixtureRequest) -> PRNGKeyArray:
    """Create a deterministic random key from the current test node ID.

    Hashing the fully qualified node ID makes the seed stable across Python
    processes, repeated runs, and xdist workers while keeping distinct tests
    statistically independent.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Pytest request identifying the current test node.

    Returns
    -------
    PRNGKeyArray
        Typed JAX random key derived from the first 32 hash bits.
    """
    digest: bytes = hashlib.sha256(request.node.nodeid.encode()).digest()
    seed: int = int.from_bytes(digest[:4], byteorder="big")
    key: PRNGKeyArray = jax.random.key(seed)
    return key
