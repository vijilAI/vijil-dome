"""End-to-end identity-MAC enforcement through TrustRuntime (not just ToolPolicy).

varunc1996 review of #248: A2/A3 tests drive ToolPolicy.check directly; no test
exercises check_tool_call / wrap_tool raising PermissionError on an
identity-grounds denial in enforce mode — the integration that would catch
mode-vs-enforced flag drift.

These tests confirm:
- check_tool_call returns a denied result with enforced=True
- wrap_tool raises PermissionError in enforce mode
- audit emits svid_keyed_unattested when an SVID-keyed policy is evaluated for
  an unattested caller under the default "warn" (makes the residual exposure
  visible).
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from vijil_dome.trust.audit import AuditEvent
from vijil_dome.trust.runtime import TrustRuntime

_SUBJECT = "spiffe://vijil.ai/org/team-1/agent/agent-1"
_OTHER = "spiffe://vijil.ai/org/team-1/agent/agent-99"


def _constraints(*, unattested_tool_policy: str = "deny", enforcement_mode: str = "enforce") -> dict:
    return {
        "agent_id": _SUBJECT,
        "dome_config": {"input_guards": [], "output_guards": [], "guards": {}},
        "tool_permissions": [
            {
                "name": "book_flight",
                "identity": "spiffe://vijil.ai/tools/book_flight/v1",
                "endpoint": "mcp+tls://book_flight.internal:8443",
            }
        ],
        "organization": {"required_input_guards": [], "required_output_guards": [], "denied_tools": []},
        "enforcement_mode": enforcement_mode,
        "unattested_tool_policy": unattested_tool_policy,
        "updated_at": datetime.now(tz=timezone.utc).isoformat(),
    }


def _runtime(c: dict, *, sid: str | None, att: bool) -> TrustRuntime:
    rt = TrustRuntime(agent_id="agent-1", mode="enforce", constraints=c)
    rt._identity = SimpleNamespace(is_attested=lambda: att, spiffe_id=sid)
    return rt


def _tool(name: str = "book_flight"):
    def fn(**kwargs): return "ok"  # noqa: E704
    fn.__name__ = name
    return fn


# ---------------------------------------------------------------------------
# check_tool_call e2e — A2 unattested + deny policy
# ---------------------------------------------------------------------------


def test_check_tool_call_a2_deny_returns_enforced_denied() -> None:
    result = _runtime(_constraints(unattested_tool_policy="deny"), sid=None, att=False).check_tool_call(
        "book_flight", {}
    )
    assert result.permitted is False
    assert result.enforced is True  # enforced=True because enforcement_mode is "enforce"
    assert "attest" in (result.error or "").lower()


# ---------------------------------------------------------------------------
# wrap_tool e2e — raises PermissionError on identity-grounds denial
# ---------------------------------------------------------------------------


def test_wrap_tool_a2_deny_raises_permission_error() -> None:
    rt = _runtime(_constraints(unattested_tool_policy="deny"), sid=None, att=False)
    with pytest.raises(PermissionError, match="book_flight"):
        rt.wrap_tool(_tool("book_flight"))()


def test_wrap_tool_a3_svid_mismatch_raises_permission_error() -> None:
    rt = _runtime(_constraints(), sid=_OTHER, att=True)
    with pytest.raises(PermissionError, match="book_flight"):
        rt.wrap_tool(_tool("book_flight"))()


def test_wrap_tool_a3_svid_match_permits() -> None:
    rt = _runtime(_constraints(), sid=_SUBJECT, att=True)
    assert rt.wrap_tool(_tool("book_flight"))() == "ok"


def test_wrap_tool_warn_mode_does_not_raise() -> None:
    c = _constraints(unattested_tool_policy="deny", enforcement_mode="warn")
    rt = TrustRuntime(agent_id="agent-1", mode="warn", constraints=c)
    rt._identity = SimpleNamespace(is_attested=lambda: False, spiffe_id=None)
    # Both constraint enforcement_mode and runtime mode are "warn" -> enforced=False -> no raise.
    result = rt.check_tool_call("book_flight", {})
    assert result.permitted is False
    assert result.enforced is False
    # wrap_tool must NOT raise when enforced=False
    rt.wrap_tool(_tool("book_flight"))()


# ---------------------------------------------------------------------------
# Audit: svid_keyed_unattested event when SVID-keyed policy + unattested + warn
# ---------------------------------------------------------------------------


def test_svid_keyed_unattested_audit_event_emitted() -> None:
    events: list[AuditEvent] = []
    rt = _runtime(_constraints(unattested_tool_policy="warn"), sid=None, att=False)
    rt._audit._sink = events.append

    result = rt.check_tool_call("book_flight", {})

    assert result.permitted is True  # default warn -> permitted (DOME-166 will flip)
    svid_events = [e for e in events if e.event_type == "svid_keyed_unattested"]
    assert len(svid_events) == 1
    assert svid_events[0].attributes["tool_name"] == "book_flight"
    assert svid_events[0].attributes["policy_subject"] == _SUBJECT


def test_svid_keyed_unattested_not_emitted_when_attested() -> None:
    events: list[AuditEvent] = []
    rt = _runtime(_constraints(unattested_tool_policy="warn"), sid=_SUBJECT, att=True)
    rt._audit._sink = events.append

    rt.check_tool_call("book_flight", {})

    assert not any(e.event_type == "svid_keyed_unattested" for e in events)
