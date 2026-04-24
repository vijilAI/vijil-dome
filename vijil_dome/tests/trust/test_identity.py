"""Tests for AgentIdentity — SPIFFE + API key fallback."""

from __future__ import annotations

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
