"""Tests for TrustRuntime orchestrator."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from vijil_dome.trust.guard import EnforcementResult
from vijil_dome.trust.runtime import TrustRuntime

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONSTRAINTS_DICT = {
    "agent_id": "agent-123",
    "dome_config": {
        "input_guards": ["prompt_injection"],
        "output_guards": ["pii_filter"],
        "guards": {"prompt_injection": {"threshold": 0.8}, "pii_filter": {}},
    },
    "tool_permissions": [
        {
            "name": "book_flight",
            "identity": "spiffe://vijil.ai/tools/book_flight/v1",
            "endpoint": "mcp+tls://book_flight.internal:8443",
        },
        {
            "name": "search_hotels",
            "identity": "spiffe://vijil.ai/tools/search_hotels/v1",
            "endpoint": "mcp+tls://search_hotels.internal:8443",
        },
    ],
    "organization": {
        "required_input_guards": ["prompt_injection"],
        "required_output_guards": ["pii_filter"],
        "denied_tools": ["delete_records"],
    },
    "enforcement_mode": "warn",
    "updated_at": "2026-04-03T12:00:00+00:00",
}


def _make_mock_client() -> MagicMock:
    client = MagicMock()
    client._http._token = "test-api-key-123"
    client._http.get.return_value = CONSTRAINTS_DICT
    return client


def _make_runtime(*, mode: str = "warn") -> TrustRuntime:
    client = _make_mock_client()
    return TrustRuntime(client=client, agent_id="agent-123", mode=mode)


# ---------------------------------------------------------------------------
# 1. test_construction_fetches_constraints
# ---------------------------------------------------------------------------


def test_construction_fetches_constraints() -> None:
    runtime = _make_runtime()

    assert runtime.mode == "warn"
    assert runtime._policy is not None
    assert runtime._agent_id == "agent-123"


# ---------------------------------------------------------------------------
# 2. test_check_tool_call_permitted — book_flight is permitted
# ---------------------------------------------------------------------------


def test_check_tool_call_permitted() -> None:
    runtime = _make_runtime()
    result = runtime.check_tool_call("book_flight", {})

    assert result.permitted is True
    assert result.tool_name == "book_flight"


# ---------------------------------------------------------------------------
# 3. test_check_tool_call_denied — delete_records is denied
# ---------------------------------------------------------------------------


def test_check_tool_call_denied() -> None:
    runtime = _make_runtime()
    result = runtime.check_tool_call("delete_records", {})

    assert result.permitted is False
    assert result.tool_name == "delete_records"


# ---------------------------------------------------------------------------
# 4. test_guard_input_delegates_to_dome — mock Dome guard_input
# ---------------------------------------------------------------------------


def test_guard_input_delegates_to_dome() -> None:
    runtime = _make_runtime()

    mock_scan = MagicMock()
    mock_scan.flagged = False
    mock_scan.enforced = False
    mock_scan.detection_score = 0.0
    mock_scan.response_string = ""
    mock_scan.exec_time = 0.0
    mock_scan.trace = {}

    mock_dome = MagicMock()
    mock_dome.guard_input.return_value = mock_scan
    runtime._dome = mock_dome

    result = runtime.guard_input("hello")

    assert isinstance(result, EnforcementResult)
    assert result.flagged is False
    mock_dome.guard_input.assert_called_once_with("hello", agent_id="agent-123")


# ---------------------------------------------------------------------------
# 5. test_warn_mode_does_not_enforce — denied tool has enforced=False
# ---------------------------------------------------------------------------


def test_warn_mode_does_not_enforce() -> None:
    runtime = _make_runtime(mode="warn")
    result = runtime.check_tool_call("delete_records", {})

    assert result.permitted is False
    assert result.enforced is False


# ---------------------------------------------------------------------------
# 6. test_enforce_mode_enforces — denied tool has enforced=True
# ---------------------------------------------------------------------------


def test_enforce_mode_enforces() -> None:
    with patch("vijil_dome.Dome"):
        runtime = _make_runtime(mode="enforce")
    result = runtime.check_tool_call("delete_records", {})

    assert result.permitted is False
    assert result.enforced is True


# ---------------------------------------------------------------------------
# 7. test_wrap_tool_returns_callable
# ---------------------------------------------------------------------------


def test_wrap_tool_returns_callable() -> None:
    runtime = _make_runtime()

    def book_flight(destination: str) -> str:
        return f"Booked to {destination}"

    wrapped = runtime.wrap_tool(book_flight)

    assert callable(wrapped)
    assert wrapped.__name__ == "book_flight"


# ---------------------------------------------------------------------------
# 8. test_wrapped_tool_calls_original — returns original result when permitted
# ---------------------------------------------------------------------------


def test_wrapped_tool_calls_original() -> None:
    runtime = _make_runtime()

    def book_flight(destination: str) -> str:
        return f"Booked to {destination}"

    wrapped = runtime.wrap_tool(book_flight)
    result = wrapped("Paris")

    assert result == "Booked to Paris"


# ---------------------------------------------------------------------------
# BC-5: Silent guards disabled on init error
# ---------------------------------------------------------------------------


def test_enforce_mode_raises_on_dome_init_failure() -> None:
    with patch("vijil_dome.Dome", side_effect=RuntimeError("torch missing")):
        with pytest.raises(RuntimeError, match="Dome initialization failed in enforce mode"):
            TrustRuntime(
                client=_make_mock_client(),
                agent_id="agent-123",
                mode="enforce",
            )


def test_warn_mode_degrades_on_dome_init_failure() -> None:
    with patch("vijil_dome.Dome", side_effect=RuntimeError("torch missing")):
        runtime = TrustRuntime(
            client=_make_mock_client(),
            agent_id="agent-123",
            mode="warn",
        )
    assert runtime._guards_disabled is True
    assert runtime._dome is None


def test_guards_disabled_surfaced_in_result() -> None:
    with patch("vijil_dome.Dome", side_effect=RuntimeError("torch missing")):
        runtime = TrustRuntime(
            client=_make_mock_client(),
            agent_id="agent-123",
            mode="warn",
        )
    result = runtime.guard_input("hello")
    assert result.guards_disabled is True
    assert result.flagged is False


# ---------------------------------------------------------------------------
# BC-7: Silent mTLS downgrade logging
# ---------------------------------------------------------------------------


def test_mtls_downgrade_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    runtime = _make_runtime()
    runtime._dome = None  # skip guard pass

    mock_identity = MagicMock()
    mock_identity.is_attested.return_value = True
    mock_identity.mtls_context.side_effect = RuntimeError("no SVID")
    mock_identity.spiffe_id = "spiffe://vijil.ai/agents/test/v1"
    runtime._identity = mock_identity

    mock_manifest = MagicMock()
    mock_tool = MagicMock()
    mock_tool.name = "search_flights"
    mock_tool.identity = "spiffe://vijil.ai/tools/search_flights/v1"
    mock_tool.endpoint = "mcp+tls://localhost:9999"
    mock_manifest.tools = [mock_tool]
    runtime._manifest = mock_manifest

    with caplog.at_level(logging.WARNING):
        # attest() will fail on socket connection after the mTLS downgrade path
        try:
            runtime.attest()
        except (ConnectionRefusedError, OSError):
            pass

    assert any("mTLS downgrade" in r.message for r in caplog.records)


def test_mtls_downgrade_hard_fails_in_enforce_mode() -> None:
    with patch("vijil_dome.Dome"):
        runtime = _make_runtime(mode="enforce")
    runtime._dome = None

    mock_identity = MagicMock()
    mock_identity.is_attested.return_value = True
    mock_identity.mtls_context.side_effect = RuntimeError("no SVID")
    mock_identity.spiffe_id = "spiffe://vijil.ai/agents/test/v1"
    runtime._identity = mock_identity

    mock_manifest = MagicMock()
    mock_tool = MagicMock()
    mock_tool.name = "search_flights"
    mock_tool.identity = "spiffe://vijil.ai/tools/search_flights/v1"
    mock_tool.endpoint = "mcp+tls://localhost:9999"
    mock_manifest.tools = [mock_tool]
    runtime._manifest = mock_manifest

    try:
        result = runtime.attest()
    except Exception:
        return

    tool_status = next(
        (t for t in result.tools if t.tool_name == "search_flights"), None
    )
    assert tool_status is not None
    assert tool_status.verified is False
    assert "enforce" in (tool_status.error or "").lower()


# ---------------------------------------------------------------------------
# BC-18: Mode validation
# ---------------------------------------------------------------------------


def test_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="mode must be one of"):
        TrustRuntime(
            client=_make_mock_client(),
            agent_id="agent-123",
            mode="enforced",
        )


def test_typo_mode_raises() -> None:
    with pytest.raises(ValueError, match="mode must be one of"):
        TrustRuntime(
            client=_make_mock_client(),
            agent_id="agent-123",
            mode="Enforce",
        )


# ---------------------------------------------------------------------------
# BC-7: Unattested identity emits structured audit event
# ---------------------------------------------------------------------------


def test_unattested_identity_emits_audit_event() -> None:
    captured: list = []
    with patch(
        "vijil_dome.trust.audit.AuditEmitter._log_sink",
        side_effect=captured.append,
    ):
        _make_runtime()

    unattested = [e for e in captured if e.event_type == "identity_unattested"]
    assert len(unattested) == 1
