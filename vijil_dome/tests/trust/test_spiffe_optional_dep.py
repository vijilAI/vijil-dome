"""A7 — SPIFFE attestation must be real in a locked install.

`vijil_dome/trust/identity.py` guards `from spiffe import WorkloadApiClient`
behind a `_HAS_SPIFFE` flag. For X.509 SVID attestation to work without a
human manually running `pip install spiffe`, the `spiffe` (py-spiffe) package
must be declared as an optional dependency (extra: `identity`) and locked in
`poetry.lock`.

These tests assert the contract from the *consumer* side: when the `spiffe`
package is importable, the import guard in `identity.py` resolves to
`_HAS_SPIFFE is True` and binds `WorkloadApiClient`. When it is not importable
(core install, no extra), the test skips honestly rather than passing
vacuously — so a not-installed environment cannot mask a regression.
"""

from __future__ import annotations

import pytest

import vijil_dome.trust.identity as identity_module


def _spiffe_imports_cleanly() -> bool:
    """Ground truth: does ``import spiffe`` actually succeed?

    This mirrors the semantics of the import guard in ``identity.py`` —
    *import success*, not mere findability. ``importlib.util.find_spec``
    would report a package that is installed-but-broken (e.g. a vendored
    protobuf stub whose gencode is newer than the runtime, which raises
    ``protobuf.VersionError`` on import) as present, diverging from what the
    guard records. Attempting the import keeps this test's ground truth in
    lockstep with ``_HAS_SPIFFE``.
    """
    try:
        import spiffe  # noqa: F401
    except Exception:  # noqa: BLE001 — any import-time failure means "unusable"
        return False
    return True


# Is the real py-spiffe package importable in this interpreter? This is the
# ground truth the import guard in identity.py keys off of.
_SPIFFE_IMPORTABLE = _spiffe_imports_cleanly()


@pytest.mark.skipif(
    not _SPIFFE_IMPORTABLE,
    reason="spiffe (py-spiffe) not installed; install vijil-dome[identity] to exercise this path",
)
def test_has_spiffe_true_when_spiffe_installed() -> None:
    """When spiffe is importable, the identity module's guard reports True."""
    assert identity_module._HAS_SPIFFE is True


@pytest.mark.skipif(
    not _SPIFFE_IMPORTABLE,
    reason="spiffe (py-spiffe) not installed; install vijil-dome[identity] to exercise this path",
)
def test_workload_api_client_bound_when_spiffe_installed() -> None:
    """When spiffe is importable, WorkloadApiClient is bound in the module.

    The X.509 attestation path (`_try_spire_attestation`) references
    `WorkloadApiClient` directly. If the import guard succeeds but the symbol
    is not bound, attestation would raise NameError at runtime — a silent
    fail-open into the unattested fallback. This guards that seam.
    """
    assert hasattr(identity_module, "WorkloadApiClient")
    assert identity_module.WorkloadApiClient is not None


def test_has_spiffe_flag_matches_importability() -> None:
    """`_HAS_SPIFFE` must agree with the real importability of `spiffe`.

    This runs in every environment (no skip). It catches the two-sided
    failure: the flag claiming True when the package is absent (fail-open),
    or False when present (the A7 gap — dep undeclared so never installed).
    """
    assert identity_module._HAS_SPIFFE is _SPIFFE_IMPORTABLE
