"""A1 (DOME-164): the MAC decision is threaded with the agent's attested identity.

Observe-only: ``identity_verified`` + ``agent_spiffe_id`` reflect real attestation state, but
permit/deny behavior is unchanged (enforcement lands later in A2/A3 behind a mode).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from vijil_dome.trust.constraints import AgentConstraints
from vijil_dome.trust.policy import ToolPolicy
from vijil_dome.trust.runtime import TrustRuntime

_SPIFFE = "spiffe://vijil.ai/org/team-1/agent/agent-1"


def _constraints(*, enforcement_mode: str = "enforce") -> AgentConstraints:
    return AgentConstraints.model_validate(
        {
            "agent_id": "agent-1",
            "dome_config": {"input_guards": [], "output_guards": [], "guards": {}},
            "tool_permissions": [
                {"name": "book_flight", "identity": "spiffe://vijil.ai/tools/book_flight/v1",
                 "endpoint": "mcp+tls://book_flight.internal:8443"},
            ],
            "organization": {
                "required_input_guards": [], "required_output_guards": [],
                "denied_tools": ["charge_card"],
            },
            "enforcement_mode": enforcement_mode,
            "updated_at": "2026-04-03T12:00:00+00:00",
        }
    )


def _runtime() -> TrustRuntime:
    client = MagicMock()
    client._http._token = "test-api-key"  # api-key path -> unattested
    client._http.get.return_value = _constraints().model_dump(mode="json")
    return TrustRuntime(client=client, agent_id="agent-1", mode="enforce")


# --- ToolPolicy.check threads identity ---------------------------------------


def test_check_records_attested_identity_on_permit() -> None:
    result = ToolPolicy(_constraints()).check("book_flight", {}, spiffe_id=_SPIFFE, attested=True)
    assert result.permitted is True
    assert result.identity_verified is True
    assert result.agent_spiffe_id == _SPIFFE


def test_check_records_identity_on_deny_too() -> None:
    result = ToolPolicy(_constraints()).check("charge_card", {}, spiffe_id=_SPIFFE, attested=True)
    assert result.permitted is False  # denied tool
    assert result.identity_verified is True  # identity still recorded on a deny
    assert result.agent_spiffe_id == _SPIFFE


def test_check_unattested_is_not_verified() -> None:
    result = ToolPolicy(_constraints()).check("book_flight", {}, spiffe_id=None, attested=False)
    assert result.identity_verified is False
    assert result.agent_spiffe_id is None


def test_identity_does_not_change_permit_deny() -> None:
    # Observe-only: attestation must not alter the permit/deny outcome in A1.
    policy = ToolPolicy(_constraints())
    assert policy.check("book_flight", {}, attested=False).permitted is True
    assert policy.check("book_flight", {}, attested=True).permitted is True
    assert policy.check("charge_card", {}, attested=True).permitted is False


# --- TrustRuntime.check_tool_call threads its own identity -------------------


def test_check_tool_call_reflects_attested_runtime_identity() -> None:
    runtime = _runtime()
    runtime._identity = MagicMock()
    runtime._identity.is_attested.return_value = True
    runtime._identity.spiffe_id = _SPIFFE
    result = runtime.check_tool_call("book_flight", {})
    assert result.identity_verified is True
    assert result.agent_spiffe_id == _SPIFFE


def test_check_tool_call_unattested_runtime_identity_not_verified() -> None:
    result = _runtime().check_tool_call("book_flight", {})  # api-key -> unattested
    assert result.identity_verified is False


# --- the audit event carries the attested identity (A6 folded into A1) -------


def _attested_runtime_with_audit_capture() -> tuple[TrustRuntime, list]:
    runtime = _runtime()
    runtime._identity = MagicMock()
    runtime._identity.is_attested.return_value = True
    runtime._identity.spiffe_id = _SPIFFE
    events: list = []
    runtime._audit._sink = events.append
    return runtime, events


def test_audit_event_carries_spiffe_id_on_permit() -> None:
    runtime, events = _attested_runtime_with_audit_capture()
    runtime.check_tool_call("book_flight", {})  # permitted tool
    assert events[-1].event_type == "tool_mac"
    assert events[-1].attributes["agent_spiffe_id"] == _SPIFFE
    assert events[-1].attributes["identity_verified"] is True
    assert events[-1].attributes["permitted"] is True


def test_audit_event_carries_spiffe_id_on_deny() -> None:
    runtime, events = _attested_runtime_with_audit_capture()
    runtime.check_tool_call("charge_card", {})  # denied tool
    assert events[-1].attributes["agent_spiffe_id"] == _SPIFFE
    assert events[-1].attributes["permitted"] is False


# --- identity is threaded on every deny path, incl. allowed_actions ----------


def test_check_allowed_actions_deny_threads_identity() -> None:
    constraints = AgentConstraints.model_validate(
        {
            "agent_id": "agent-1",
            "dome_config": {"input_guards": [], "output_guards": [], "guards": {}},
            "tool_permissions": [
                {"name": "book_flight", "identity": "spiffe://vijil.ai/tools/book_flight/v1",
                 "endpoint": "mcp+tls://book_flight.internal:8443", "allowed_actions": ["read"]},
            ],
            "organization": {"required_input_guards": [], "required_output_guards": [], "denied_tools": []},
            "enforcement_mode": "enforce",
            "updated_at": "2026-04-03T12:00:00+00:00",
        }
    )
    result = ToolPolicy(constraints).check(
        "book_flight", {"action": "write"}, spiffe_id=_SPIFFE, attested=True
    )
    assert result.permitted is False  # action not in allowed_actions
    assert result.identity_verified is True
    assert result.agent_spiffe_id == _SPIFFE


def test_check_defaults_to_unattested_when_identity_kwargs_omitted() -> None:
    # Omitting the identity kwargs defaults to unattested / no spiffe_id.
    result = ToolPolicy(_constraints()).check("book_flight", {})
    assert result.identity_verified is False
    assert result.agent_spiffe_id is None
