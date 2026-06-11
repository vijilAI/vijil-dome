"""B3 (DOME-163): enforcement-alive heartbeat.

A wrapped agent emits a signed-later 'I am enforcing' beacon so a registered
agent that goes dark or silently downgrades is detectable downstream.

Scope of B3: the event shape (``Heartbeat`` model) + emit
(``AuditEmitter.emit_heartbeat`` and ``TrustRuntime.emit_heartbeat``).
Periodic scheduling and SVID-signing are a follow-up.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from vijil_dome.trust.audit import AuditEmitter, AuditEvent, Heartbeat
from vijil_dome.trust.constraints import AgentConstraints
from vijil_dome.trust.runtime import TrustRuntime

_SPIFFE = "spiffe://vijil.ai/org/team-1/agent/agent-1"


# ---------------------------------------------------------------------------
# 1. Heartbeat pydantic model carries the four enforcement-liveness fields
# ---------------------------------------------------------------------------


def test_heartbeat_model_fields() -> None:
    hb = Heartbeat(
        configured_mode="enforce",
        guards_constructed=True,
        detector_reachable=True,
        attested=True,
        agent_spiffe_id=_SPIFFE,
    )
    assert hb.configured_mode == "enforce"
    assert hb.guards_constructed is True
    assert hb.detector_reachable is True
    assert hb.attested is True
    assert hb.agent_spiffe_id == _SPIFFE


def test_heartbeat_unattested_spiffe_is_none() -> None:
    # An unattested agent emits a beacon with attested=False and agent_spiffe_id=None.
    hb = Heartbeat(
        configured_mode="warn",
        guards_constructed=False,
        detector_reachable=False,
        attested=False,
        agent_spiffe_id=None,
    )
    assert hb.attested is False
    assert hb.agent_spiffe_id is None


# ---------------------------------------------------------------------------
# 2. AuditEmitter.emit_heartbeat emits an 'enforcement_heartbeat' event
# ---------------------------------------------------------------------------


def test_emit_heartbeat_event_shape() -> None:
    events: list[AuditEvent] = []
    emitter = AuditEmitter(agent_id="agent-hb", sink=events.append)

    emitter.emit_heartbeat(
        configured_mode="enforce",
        guards_constructed=True,
        detector_reachable=True,
        attested=True,
        agent_spiffe_id=_SPIFFE,
    )

    assert len(events) == 1
    event = events[0]
    assert event.event_type == "enforcement_heartbeat"
    assert event.agent_id == "agent-hb"
    assert event.attributes["configured_mode"] == "enforce"
    assert event.attributes["guards_constructed"] is True
    assert event.attributes["detector_reachable"] is True
    assert event.attributes["attested"] is True
    assert event.attributes["agent_spiffe_id"] == _SPIFFE


def test_emit_heartbeat_unattested_carries_none_spiffe() -> None:
    events: list[AuditEvent] = []
    emitter = AuditEmitter(agent_id="agent-hb", sink=events.append)

    emitter.emit_heartbeat(
        configured_mode="warn",
        guards_constructed=False,
        detector_reachable=False,
        attested=False,
        agent_spiffe_id=None,
    )

    assert events[-1].event_type == "enforcement_heartbeat"
    assert events[-1].attributes["agent_spiffe_id"] is None
    assert events[-1].attributes["attested"] is False
    assert events[-1].attributes["guards_constructed"] is False
    assert events[-1].attributes["detector_reachable"] is False


# ---------------------------------------------------------------------------
# 3. TrustRuntime.emit_heartbeat gathers live runtime state and emits it
# ---------------------------------------------------------------------------


def _constraints() -> AgentConstraints:
    return AgentConstraints.model_validate(
        {
            "agent_id": "agent-1",
            "dome_config": {"input_guards": [], "output_guards": [], "guards": {}},
            "tool_permissions": [],
            "organization": {
                "required_input_guards": [],
                "required_output_guards": [],
                "denied_tools": [],
            },
            "enforcement_mode": "enforce",
            "updated_at": "2026-04-03T12:00:00+00:00",
        }
    )


def _runtime(*, mode: str = "enforce") -> TrustRuntime:
    client = MagicMock()
    client._http._token = "test-api-key"  # api-key path -> unattested
    client._http.get.return_value = _constraints().model_dump(mode="json")
    return TrustRuntime(client=client, agent_id="agent-1", mode=mode)


def test_runtime_emit_heartbeat_reports_effective_mode() -> None:
    events: list[AuditEvent] = []
    runtime = _runtime(mode="enforce")
    runtime._audit._sink = events.append

    runtime.emit_heartbeat()

    assert events[-1].event_type == "enforcement_heartbeat"
    assert events[-1].attributes["configured_mode"] == "enforce"


def test_runtime_emit_heartbeat_reports_attested_spiffe_id() -> None:
    events: list[AuditEvent] = []
    runtime = _runtime()
    runtime._identity = MagicMock()
    runtime._identity.is_attested.return_value = True
    runtime._identity.spiffe_id = _SPIFFE
    runtime._audit._sink = events.append

    runtime.emit_heartbeat()

    assert events[-1].attributes["agent_spiffe_id"] == _SPIFFE


def test_runtime_emit_heartbeat_no_guards_means_hooks_not_attached() -> None:
    # No Dome guards configured (empty input/output guards) -> _dome is None.
    events: list[AuditEvent] = []
    runtime = _runtime()
    runtime._audit._sink = events.append

    runtime.emit_heartbeat()

    assert events[-1].attributes["guards_constructed"] is False


def test_runtime_emit_heartbeat_detector_unreachable_when_guards_disabled() -> None:
    # A starved/failed detector backend sets _guards_disabled; the beacon must
    # report detector_reachable=False so a silent downgrade is detectable.
    events: list[AuditEvent] = []
    runtime = _runtime()
    runtime._guards_disabled = True
    runtime._audit._sink = events.append

    runtime.emit_heartbeat()

    assert events[-1].attributes["detector_reachable"] is False


def test_runtime_emit_heartbeat_no_guards_means_detector_not_reachable() -> None:
    # varunc1996 review of #245: with no guards configured (_dome is None) and guards NOT
    # disabled, detector_reachable must be False — "no detector configured" must not read as
    # "detector reachable" (the B4 reconciler keys on it). A real reachability probe is DOME-169.
    events: list[AuditEvent] = []
    runtime = _runtime()  # empty guards -> _dome is None, _guards_disabled stays False
    runtime._audit._sink = events.append

    runtime.emit_heartbeat()

    assert events[-1].attributes["detector_reachable"] is False


def test_runtime_emit_heartbeat_returns_heartbeat_model() -> None:
    runtime = _runtime()
    hb = runtime.emit_heartbeat()
    assert isinstance(hb, Heartbeat)
    assert hb.configured_mode == "enforce"
