"""Tests for AuditEmitter and AuditEvent."""

from __future__ import annotations

from vijil_dome.trust.audit import AuditEmitter, AuditEvent

# ---------------------------------------------------------------------------
# 1. emit_guard — verify event_type and agent_id
# ---------------------------------------------------------------------------


def test_emit_guard_event() -> None:
    events: list[AuditEvent] = []
    emitter = AuditEmitter(agent_id="agent-abc", sink=events.append)

    emitter.emit_guard("input", flagged=True, score=0.91, exec_time_ms=42.0)

    assert len(events) == 1
    event = events[0]
    assert event.event_type == "guard"
    assert event.agent_id == "agent-abc"
    assert event.attributes["direction"] == "input"
    assert event.attributes["flagged"] is True
    assert event.attributes["score"] == 0.91
    assert event.attributes["exec_time_ms"] == 42.0


# ---------------------------------------------------------------------------
# 2. emit_tool_mac — verify event_type is "tool_mac"
# ---------------------------------------------------------------------------


def test_emit_tool_mac_event() -> None:
    events: list[AuditEvent] = []
    emitter = AuditEmitter(agent_id="agent-xyz", sink=events.append)

    emitter.emit_tool_mac("search_web", permitted=True, identity_verified=True)

    assert len(events) == 1
    event = events[0]
    assert event.event_type == "tool_mac"
    assert event.agent_id == "agent-xyz"
    assert event.attributes["tool_name"] == "search_web"
    assert event.attributes["permitted"] is True
    assert event.attributes["identity_verified"] is True


# ---------------------------------------------------------------------------
# 3. emit_attestation — verify event_type is "attestation"
# ---------------------------------------------------------------------------


def test_emit_attestation_event() -> None:
    events: list[AuditEvent] = []
    emitter = AuditEmitter(agent_id="agent-007", sink=events.append)

    emitter.emit_attestation(all_verified=False, tool_count=3)

    assert len(events) == 1
    event = events[0]
    assert event.event_type == "attestation"
    assert event.agent_id == "agent-007"
    assert event.attributes["all_verified"] is False
    assert event.attributes["tool_count"] == 3
