"""Tests for TrustRuntime orchestrator."""

from __future__ import annotations

from unittest.mock import MagicMock

from vijil_dome.trust.guard import GuardResult
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

    assert isinstance(result, GuardResult)
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
