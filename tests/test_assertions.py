"""Test the shared assertion helpers."""

import equinox as eqx
import jax.numpy as jnp
import pytest

from tests._assertions import assert_rejects


def test_assert_rejects_eager_and_jit():
    """Accept a value-threaded rejection in eager and compiled execution."""

    def reject_negative(value):
        checked = eqx.error_if(value, value < 0, "value must be non-negative")
        return checked

    assert_rejects(
        reject_negative,
        jnp.asarray(-1.0),
        match="value must be non-negative",
    )


def test_assert_rejects_fails_for_passing_callable():
    """Fail when the callable unexpectedly accepts invalid test data."""

    def accept(value):
        return value

    with pytest.raises(AssertionError, match="DID NOT RAISE"):
        assert_rejects(
            accept, jnp.asarray(-1.0), match="unused", under_jit=False
        )
