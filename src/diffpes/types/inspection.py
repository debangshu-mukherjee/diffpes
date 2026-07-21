"""Types-owned records produced by certificate inspection.

Extended Summary
----------------
Inspection records categorize scientific, numerical, environment, and audit
differences without evaluating any new physical claim.

Routine Listings
----------------
:class:`CertificateDiff`
    Store categorized differences between two forward certificates.
:func:`make_certificate_diff`
    Construct a validated certificate-difference record.
"""

import equinox as eqx
from beartype import beartype
from jaxtyping import jaxtyped


class CertificateDiff(eqx.Module):
    """Store categorized differences between two forward certificates.

    Differences are grouped by scientific meaning, numerical evidence,
    execution environment, and audit metadata without re-running the model.

    :see: :class:`~.test_inspection.TestCertificateDiff`

    Attributes
    ----------
    scientific : tuple[str, ...]
        Differing scientific fields (**static** -- compile-time constants;
        changing them triggers retracing).
    numerical : tuple[str, ...]
        Differing numerical-evidence fields (**static** -- compile-time
        constants; changing them triggers retracing).
    environment : tuple[str, ...]
        Differing execution-environment fields (**static** -- compile-time
        constants; changing them triggers retracing).
    audit : tuple[str, ...]
        Differing audit fields (**static** -- compile-time constants; changing
        them triggers retracing).

    Notes
    -----
    Inspection compares persisted metadata only. This carrier has no numerical
    leaves and does not reevaluate or differentiate a forward model.

    See Also
    --------
    make_certificate_diff : Construct a validated certificate-difference
        record.
    """

    scientific: tuple[str, ...] = eqx.field(static=True)
    numerical: tuple[str, ...] = eqx.field(static=True)
    environment: tuple[str, ...] = eqx.field(static=True)
    audit: tuple[str, ...] = eqx.field(static=True)

    @property
    @jaxtyped(typechecker=beartype)
    def identical(self) -> bool:
        """Return whether no categorized difference was found.

        :see: :class:`~.test_inspection.TestCertificateDiff`

        Returns
        -------
        identical : bool
            Whether every difference category is empty.
        """
        identical: bool = not any(
            (self.scientific, self.numerical, self.environment, self.audit)
        )
        return identical

    @property
    @jaxtyped(typechecker=beartype)
    def summary(self) -> str:
        """Return a one-line categorized comparison summary.

        :see: :class:`~.test_inspection.TestCertificateDiff`

        Returns
        -------
        summary : str
            Human-readable list of nonempty difference categories.
        """
        if self.identical:
            summary: str = "Certificates are identical."
            return summary
        parts: list[str] = []
        label: str
        values: tuple[str, ...]
        for label, values in (
            ("scientific", self.scientific),
            ("numerical", self.numerical),
            ("environment", self.environment),
            ("audit", self.audit),
        ):
            if values:
                parts.append(f"{label}: {', '.join(values)}")
        summary = "; ".join(parts)
        return summary  # noqa: RET504 -- assign-before-return is required.


def _difference_names(value: tuple[str, ...], name: str) -> tuple[str, ...]:
    """Validate one immutable sequence of differing field names."""
    if not isinstance(value, tuple) or any(
        not isinstance(item, str) or not item for item in value
    ):
        msg: str = f"{name} must be a tuple of nonempty field names"
        raise ValueError(msg)
    return value


@jaxtyped(typechecker=beartype)
def make_certificate_diff(  # noqa: DOC502
    *,
    scientific: tuple[str, ...] = (),
    numerical: tuple[str, ...] = (),
    environment: tuple[str, ...] = (),
    audit: tuple[str, ...] = (),
) -> CertificateDiff:
    """Construct a validated certificate-difference record.

    Validate and freeze field names in each comparison category.

    :see: :class:`~.test_inspection.TestMakeCertificateDiff`

    Implementation Logic
    --------------------
    1. **Validate category names**::

           scientific=_difference_names(scientific, "scientific")

       Require immutable tuples of nonempty field names in every category.
    2. **Construct the difference**::

           difference = CertificateDiff(...)

       Bind and return the categorized comparison carrier.

    Parameters
    ----------
    scientific : tuple[str, ...]
        Differing scientific fields (**static** -- compile-time constants;
        changing them triggers retracing). Default is empty.
    numerical : tuple[str, ...]
        Differing numerical fields (**static** -- compile-time constants;
        changing them triggers retracing). Default is empty.
    environment : tuple[str, ...]
        Differing environment fields (**static** -- compile-time constants;
        changing them triggers retracing). Default is empty.
    audit : tuple[str, ...]
        Differing audit fields (**static** -- compile-time constants; changing
        them triggers retracing). Default is empty.

    Returns
    -------
    difference : CertificateDiff
        Validated immutable certificate difference.

    Raises
    ------
    ValueError
        If a category is not a tuple of nonempty field names.

    Notes
    -----
    Validation is static and does not introduce a gradient path.
    """
    difference: CertificateDiff = CertificateDiff(
        scientific=_difference_names(scientific, "scientific"),
        numerical=_difference_names(numerical, "numerical"),
        environment=_difference_names(environment, "environment"),
        audit=_difference_names(audit, "audit"),
    )
    return difference


__all__: list[str] = ["CertificateDiff", "make_certificate_diff"]
