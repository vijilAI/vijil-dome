"""Tests for ToolPolicy MAC enforcement."""

from __future__ import annotations

from vijil_dome.trust.constraints import AgentConstraints, ToolPermission
from vijil_dome.trust.policy import ToolPolicy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_permission(name: str, identity: str | None = None) -> ToolPermission:
    return ToolPermission(
        name=name,
        identity=identity or f"spiffe://vijil.ai/tools/{name}/v1",
        endpoint=f"mcp+tls://{name}.internal:8443",
        allowed_actions=["search", "book"],
    )


def _make_agent_constraints(
    *,
    enforcement_mode: str = "enforce",
    tool_names: list[str] | None = None,
    denied_tools: list[str] | None = None,
) -> AgentConstraints:
    tool_names = tool_names or ["book_flight", "search_hotels"]
    denied_tools = denied_tools or ["charge_card"]
    return AgentConstraints.model_validate(
        {
            "agent_id": "spiffe://vijil.ai/agents/travel-agent/v1",
            "dome_config": {
                "input_guards": ["prompt_injection"],
                "output_guards": ["pii_filter"],
                "guards": {"prompt_injection": {"threshold": 0.8}, "pii_filter": {}},
            },
            "tool_permissions": [
                {
                    "name": n,
                    "identity": f"spiffe://vijil.ai/tools/{n}/v1",
                    "endpoint": f"mcp+tls://{n}.internal:8443",
                }
                for n in tool_names
            ],
            "organization": {
                "required_input_guards": ["prompt_injection"],
                "required_output_guards": ["pii_filter"],
                "denied_tools": denied_tools,
            },
            "enforcement_mode": enforcement_mode,
            "updated_at": "2026-04-03T12:00:00+00:00",
        }
    )


# ---------------------------------------------------------------------------
# 1. test_permitted_tool — book_flight is in permissions → permitted
# ---------------------------------------------------------------------------


def test_permitted_tool() -> None:
    policy = ToolPolicy(_make_agent_constraints())
    result = policy.check("book_flight")

    assert result.permitted is True
    assert result.tool_name == "book_flight"
    assert result.policy_permitted is True
    assert result.enforced is False
    assert result.error is None


# ---------------------------------------------------------------------------
# 2. test_denied_tool_not_in_permissions — delete_all_records not in permissions → denied
# ---------------------------------------------------------------------------


def test_denied_tool_not_in_permissions() -> None:
    policy = ToolPolicy(_make_agent_constraints())
    result = policy.check("delete_all_records")

    assert result.permitted is False
    assert result.tool_name == "delete_all_records"
    assert result.policy_permitted is False
    assert result.error == "not in agent permissions"


# ---------------------------------------------------------------------------
# 3. test_org_denied_tool — charge_card in org denied_tools → denied with org error
# ---------------------------------------------------------------------------


def test_org_denied_tool() -> None:
    policy = ToolPolicy(_make_agent_constraints())
    result = policy.check("charge_card")

    assert result.permitted is False
    assert result.tool_name == "charge_card"
    assert result.policy_permitted is False
    assert result.error == "denied by organization constraints"


# ---------------------------------------------------------------------------
# 4. test_warn_mode_still_reports_denied — denied but enforced=False in warn mode
# ---------------------------------------------------------------------------


def test_warn_mode_still_reports_denied() -> None:
    policy = ToolPolicy(_make_agent_constraints(enforcement_mode="warn"))
    result = policy.check("delete_all_records")

    assert result.permitted is False
    assert result.enforced is False


# ---------------------------------------------------------------------------
# 5. test_enforce_mode_sets_enforced — denied with enforced=True in enforce mode
# ---------------------------------------------------------------------------


def test_enforce_mode_sets_enforced() -> None:
    policy = ToolPolicy(_make_agent_constraints(enforcement_mode="enforce"))
    result = policy.check("delete_all_records")

    assert result.permitted is False
    assert result.enforced is True


# ---------------------------------------------------------------------------
# 6. test_get_permission_returns_entry — returns ToolPermission with correct identity
# ---------------------------------------------------------------------------


def test_get_permission_returns_entry() -> None:
    policy = ToolPolicy(_make_agent_constraints())
    perm = policy.get_permission("book_flight")

    assert perm is not None
    assert perm.name == "book_flight"
    assert perm.identity == "spiffe://vijil.ai/tools/book_flight/v1"


# ---------------------------------------------------------------------------
# 7. test_get_permission_returns_none_for_unknown — returns None
# ---------------------------------------------------------------------------


def test_get_permission_returns_none_for_unknown() -> None:
    policy = ToolPolicy(_make_agent_constraints())
    perm = policy.get_permission("unknown_tool")

    assert perm is None
