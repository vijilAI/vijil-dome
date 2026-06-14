"""Tests for AgentIdentity — SPIFFE + API key fallback."""

from __future__ import annotations

import importlib
import logging
import sys
from unittest.mock import patch

import pytest

from vijil_dome.trust.identity import AgentIdentity

# ---------------------------------------------------------------------------
# 1. test_from_api_key — creates identity, not attested, has api_key
# ---------------------------------------------------------------------------


def test_from_api_key() -> None:
    identity = AgentIdentity.from_api_key("vj-test-key-abc123")

    assert identity.api_key == "vj-test-key-abc123"
    assert identity.is_attested() is False
    assert identity.spiffe_id is None


# ---------------------------------------------------------------------------
# 2. test_spire_unavailable_returns_unattested — nonexistent socket → not attested
# ---------------------------------------------------------------------------


def test_spire_unavailable_returns_unattested() -> None:
    identity = AgentIdentity(spire_socket="/nonexistent/path/to/socket.sock")

    assert identity.is_attested() is False
    assert identity.spiffe_id is None


# ---------------------------------------------------------------------------
# 3. test_mtls_context_without_spire_raises — raises RuntimeError
# ---------------------------------------------------------------------------


def test_mtls_context_without_spire_raises() -> None:
    identity = AgentIdentity.from_api_key("vj-key-xyz")

    with pytest.raises(RuntimeError):
        identity.mtls_context()


# ---------------------------------------------------------------------------
# 4. test_auth_header_with_api_key — returns Bearer header
# ---------------------------------------------------------------------------


def test_auth_header_with_api_key() -> None:
    identity = AgentIdentity.from_api_key("vj-secret-key")

    header = identity.auth_header()

    assert header == {"authorization": "Bearer vj-secret-key"}


# ---------------------------------------------------------------------------
# 5. JWT-SVID via delegate — tests the delegate attestation path
# ---------------------------------------------------------------------------


def test_delegate_url_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """AgentIdentity reads delegate URL from VIJIL_IDENTITY_DELEGATE_URL env var."""
    monkeypatch.setenv("VIJIL_IDENTITY_DELEGATE_URL", "http://delegate:8080")
    identity = AgentIdentity(
        spire_socket="/nonexistent/socket",
        agent_name="test-agent",
    )
    # Delegate URL is set but attestation fails (no real service) — still unattested
    assert identity._delegate_url == "http://delegate:8080"
    assert not identity.is_attested()


def test_delegate_attestation_with_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Successful delegate attestation sets spiffe_id and jwt_svid."""

    class MockResponse:
        status_code = 200
        def raise_for_status(self) -> None:
            pass
        def json(self) -> dict:
            return {
                "jwt_svid": "eyJ0eXAiOiJKV1QiLCJhbGciOiJub25lIn0.eyJzdWIiOiJzcGlmZmU6Ly92aWppbC5haS9ucy9tYW5hZ2VkL2FnZW50L3Rlc3QiLCJhdWQiOlsidmlqaWwiXX0.",
                "spiffe_id": "spiffe://vijil.ai/ns/managed/agent/test",
                "expires_at": "2026-04-10T12:00:00Z",
                "trust_domain": "vijil.ai",
            }

    def mock_post(*args, **kwargs):
        return MockResponse()

    monkeypatch.setattr("httpx.post", mock_post)

    identity = AgentIdentity(
        spire_socket="/nonexistent/socket",
        delegate_url="http://delegate:8080",
        agent_name="test",
    )

    assert identity.is_attested()
    assert identity.spiffe_id == "spiffe://vijil.ai/ns/managed/agent/test"
    assert identity.jwt_svid is not None
    assert identity.jwt_svid.startswith("eyJ")


def test_jwt_svid_auth_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """JWT-SVID identity returns Bearer header with the JWT token."""
    class MockResponse:
        status_code = 200
        def raise_for_status(self) -> None:
            pass
        def json(self) -> dict:
            return {
                "jwt_svid": "jwt-token-value",
                "spiffe_id": "spiffe://vijil.ai/ns/managed/agent/test",
                "expires_at": "2026-04-10T12:00:00Z",
                "trust_domain": "vijil.ai",
            }

    monkeypatch.setattr("httpx.post", lambda *a, **kw: MockResponse())

    identity = AgentIdentity(
        spire_socket="/nonexistent/socket",
        delegate_url="http://delegate:8080",
        agent_name="test",
    )

    assert identity.auth_header() == {"authorization": "Bearer jwt-token-value"}


def test_delegate_failure_falls_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """If delegate service is unreachable, identity falls through to unattested."""
    def mock_post_fail(*args, **kwargs):
        raise ConnectionError("delegate unreachable")

    monkeypatch.setattr("httpx.post", mock_post_fail)

    identity = AgentIdentity(
        spire_socket="/nonexistent/socket",
        delegate_url="http://delegate:8080",
        agent_name="test",
    )

    assert not identity.is_attested()
    assert identity.spiffe_id is None


# DOME-168: the spiffe import guard distinguishes "absent" (silent degrade) from
# "installed-but-broken" (loud WARNING degrade). It must never crash `import vijil_dome`.
_IDENTITY_LOGGER = "vijil_dome.trust.identity"


class _RaisingSpiffe:
    """A fake `spiffe` module whose import (`from spiffe import ...`) raises ``exc``."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def __getattr__(self, name: str) -> object:
        raise self._exc


@pytest.fixture
def _restore_identity() -> object:
    yield
    import vijil_dome.trust.identity as identity_mod

    importlib.reload(identity_mod)  # restore the real import state for the rest of the suite


def test_spiffe_absent_degrades_silently(
    _restore_identity: object, caplog: pytest.LogCaptureFixture
) -> None:
    """spiffe genuinely absent (ModuleNotFoundError naming spiffe) is the expected base install:
    degrade quietly, no warning."""
    import vijil_dome.trust.identity as identity_mod

    absent = ModuleNotFoundError("No module named 'spiffe'", name="spiffe")
    with patch.dict(sys.modules, {"spiffe": _RaisingSpiffe(absent)}):
        with caplog.at_level(logging.WARNING):
            importlib.reload(identity_mod)

    assert identity_mod._HAS_SPIFFE is False
    assert not [r for r in caplog.records if r.name == _IDENTITY_LOGGER]  # silent


@pytest.mark.parametrize(
    "exc",
    [
        ModuleNotFoundError("No module named 'grpc'", name="grpc"),  # missing transitive dep
        ImportError("cannot import name 'WorkloadApiClient'", name="spiffe"),  # symbol gone
        RuntimeError("protobuf gencode older than runtime (VersionError)"),  # spiffe>=0.2.4, protobuf<6
    ],
)
def test_spiffe_installed_but_broken_degrades_loudly(
    _restore_identity: object, caplog: pytest.LogCaptureFixture, exc: Exception
) -> None:
    """An installed-but-unusable spiffe — transitive dep missing, symbol gone, or protobuf
    VersionError — must degrade with a loud WARNING, never crash `import vijil_dome`."""
    import vijil_dome.trust.identity as identity_mod

    with patch.dict(sys.modules, {"spiffe": _RaisingSpiffe(exc)}):
        with caplog.at_level(logging.WARNING):
            importlib.reload(identity_mod)  # must NOT raise

    assert identity_mod._HAS_SPIFFE is False
    assert any(
        r.levelno == logging.WARNING and r.name == _IDENTITY_LOGGER for r in caplog.records
    )
