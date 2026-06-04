"""A2 (DOME-163): fail-closed on an unattested identity, gated by ``unattested_tool_policy``.

The flag controls the DECISION (deny an unattested agent's tool calls); ``enforcement_mode``
controls whether that deny is ENFORCED (the existing pattern). Default is ``warn`` so existing
enforce-mode-but-unattested flows are not silently bricked; ``deny`` is the secure opt-in.
"""

from __future__ import annotations

from vijil_dome.trust.constraints import AgentConstraints
from vijil_dome.trust.policy import ToolPolicy

_SPIFFE = "spiffe://vijil.ai/org/team-1/agent/agent-1"


def _constraints(*, enforcement_mode: str = "enforce", unattested_tool_policy: str = "warn") -> AgentConstraints:
    return AgentConstraints.model_validate(
        {
            "agent_id": "agent-1",
            "dome_config": {"input_guards": [], "output_guards": [], "guards": {}},
            "tool_permissions": [
                {"name": "book_flight", "identity": "spiffe://vijil.ai/tools/book_flight/v1",
                 "endpoint": "mcp+tls://book_flight.internal:8443"},
            ],
            "organization": {"required_input_guards": [], "required_output_guards": [], "denied_tools": []},
            "enforcement_mode": enforcement_mode,
            "unattested_tool_policy": unattested_tool_policy,
            "updated_at": "2026-04-03T12:00:00+00:00",
        }
    )


def test_unattested_denied_under_deny_policy() -> None:
    result = ToolPolicy(_constraints(unattested_tool_policy="deny")).check(
        "book_flight", {}, spiffe_id=None, attested=False
    )
    assert result.permitted is False
    assert "attest" in (result.error or "").lower()
    assert result.enforced is True  # enforce mode -> the deny is enforced


def test_attested_permitted_under_deny_policy() -> None:
    result = ToolPolicy(_constraints(unattested_tool_policy="deny")).check(
        "book_flight", {}, spiffe_id=_SPIFFE, attested=True
    )
    assert result.permitted is True
    assert result.identity_verified is True


def test_unattested_permitted_under_warn_policy_backcompat() -> None:
    # Default warn-policy preserves today's behavior: unattested still gets the normal tool check.
    result = ToolPolicy(_constraints(unattested_tool_policy="warn")).check(
        "book_flight", {}, spiffe_id=None, attested=False
    )
    assert result.permitted is True


def test_unattested_deny_decision_not_enforced_in_warn_mode() -> None:
    # warn enforcement mode computes the deny but does not enforce it (logged, not blocked).
    result = ToolPolicy(_constraints(enforcement_mode="warn", unattested_tool_policy="deny")).check(
        "book_flight", {}, spiffe_id=None, attested=False
    )
    assert result.permitted is False  # policy decides deny
    assert result.enforced is False  # warn mode does not enforce


def test_default_unattested_policy_is_warn() -> None:
    # Omitting the field defaults to warn (no unattested deny) — non-bricking.
    constraints = AgentConstraints.model_validate(
        {
            "agent_id": "agent-1",
            "dome_config": {"input_guards": [], "output_guards": [], "guards": {}},
            "tool_permissions": [],
            "organization": {"required_input_guards": [], "required_output_guards": [], "denied_tools": []},
            "enforcement_mode": "enforce",
            "updated_at": "2026-04-03T12:00:00+00:00",
        }
    )
    assert constraints.unattested_tool_policy == "warn"
