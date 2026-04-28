"""Tests for AgentConstraints and related constraint models."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from vijil_dome.trust.constraints import (
    AgentConstraints,
    DomeGuardConfig,
    OrganizationConstraints,
    ToolPermission,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_permission(name: str = "flights") -> ToolPermission:
    return ToolPermission(
        name=name,
        identity=f"spiffe://vijil.ai/tools/{name}/v1",
        endpoint=f"mcp+tls://{name}.internal:8443",
        allowed_actions=["search", "book"],
    )


def _make_dome_config() -> DomeGuardConfig:
    return DomeGuardConfig(
        input_guards=["prompt_injection", "pii_detector"],
        output_guards=["sensitive_data_filter"],
        guards={
            "prompt_injection": {"threshold": 0.8},
            "pii_detector": {"entities": ["EMAIL", "SSN"]},
            "sensitive_data_filter": {"mode": "redact"},
        },
    )


def _make_org_constraints() -> OrganizationConstraints:
    return OrganizationConstraints(
        required_input_guards=["prompt_injection"],
        required_output_guards=["sensitive_data_filter"],
        denied_tools=["raw_sql", "shell_exec"],
        max_model_tier="gpt-4o",
    )


def _make_agent_constraints(**kwargs: object) -> AgentConstraints:
    defaults: dict[str, object] = {
        "agent_id": "spiffe://vijil.ai/agents/travel-agent/v1",
        "dome_config": _make_dome_config(),
        "tool_permissions": [
            _make_tool_permission("flights"),
            _make_tool_permission("hotels"),
        ],
        "organization": _make_org_constraints(),
        "enforcement_mode": "enforce",
        "updated_at": datetime(2026, 4, 3, 12, 0, 0, tzinfo=UTC),
    }
    defaults.update(kwargs)
    return AgentConstraints.model_validate(defaults)


# ---------------------------------------------------------------------------
# 1. test_parse_constraints — parse a full JSON-like dict into AgentConstraints
# ---------------------------------------------------------------------------


def test_parse_constraints() -> None:
    raw = {
        "agent_id": "spiffe://vijil.ai/agents/travel-agent/v1",
        "dome_config": {
            "input_guards": ["prompt_injection"],
            "output_guards": ["pii_filter"],
            "guards": {
                "prompt_injection": {"threshold": 0.9},
                "pii_filter": {"mode": "block"},
            },
        },
        "tool_permissions": [
            {
                "name": "flights",
                "identity": "spiffe://vijil.ai/tools/flights/v1",
                "endpoint": "mcp+tls://flights.internal:8443",
                "allowed_actions": ["search"],
            }
        ],
        "organization": {
            "required_input_guards": ["prompt_injection"],
            "required_output_guards": ["pii_filter"],
            "denied_tools": ["shell_exec"],
            "max_model_tier": None,
        },
        "enforcement_mode": "warn",
        "updated_at": "2026-04-03T12:00:00+00:00",
    }

    constraints = AgentConstraints.model_validate(raw)

    assert constraints.agent_id == "spiffe://vijil.ai/agents/travel-agent/v1"
    assert constraints.enforcement_mode == "warn"
    assert constraints.dome_config.input_guards == ["prompt_injection"]
    assert constraints.dome_config.output_guards == ["pii_filter"]
    assert constraints.dome_config.guards["prompt_injection"] == {"threshold": 0.9}
    assert len(constraints.tool_permissions) == 1
    assert constraints.tool_permissions[0].name == "flights"
    assert constraints.tool_permissions[0].allowed_actions == ["search"]
    assert constraints.organization.denied_tools == ["shell_exec"]
    assert constraints.organization.max_model_tier is None
    assert constraints.updated_at == datetime(2026, 4, 3, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# 2. test_tool_permitted_in_permissions — check tool names are in permissions
# ---------------------------------------------------------------------------


def test_tool_permitted_in_permissions() -> None:
    constraints = _make_agent_constraints()

    permitted_names = {p.name for p in constraints.tool_permissions}

    assert "flights" in permitted_names
    assert "hotels" in permitted_names
    assert "raw_sql" not in permitted_names


def test_tool_permission_without_allowed_actions() -> None:
    """allowed_actions is optional — None means unrestricted."""
    perm = ToolPermission(
        name="hotels",
        identity="spiffe://vijil.ai/tools/hotels/v1",
        endpoint="mcp+tls://hotels.internal:8443",
    )
    assert perm.allowed_actions is None


def test_tool_permission_spiffe_id_preserved() -> None:
    perm = _make_tool_permission("cars")
    assert perm.identity == "spiffe://vijil.ai/tools/cars/v1"
    assert perm.endpoint == "mcp+tls://cars.internal:8443"


# ---------------------------------------------------------------------------
# 3. test_org_denied_tools — verify denied_tools list
# ---------------------------------------------------------------------------


def test_org_denied_tools() -> None:
    constraints = _make_agent_constraints()

    denied = constraints.organization.denied_tools

    assert "raw_sql" in denied
    assert "shell_exec" in denied
    assert "flights" not in denied


def test_org_denied_tools_empty() -> None:
    org = OrganizationConstraints(
        required_input_guards=[],
        required_output_guards=[],
        denied_tools=[],
    )
    assert org.denied_tools == []


def test_org_max_model_tier_optional() -> None:
    org = OrganizationConstraints(
        required_input_guards=["prompt_injection"],
        required_output_guards=[],
        denied_tools=[],
    )
    assert org.max_model_tier is None


def test_enforcement_mode_rejects_invalid() -> None:
    with pytest.raises(ValidationError):
        _make_agent_constraints(enforcement_mode="invalid_mode")


def test_enforcement_mode_warn_accepted() -> None:
    constraints = _make_agent_constraints(enforcement_mode="warn")
    assert constraints.enforcement_mode == "warn"


def test_enforcement_mode_enforce_accepted() -> None:
    constraints = _make_agent_constraints(enforcement_mode="enforce")
    assert constraints.enforcement_mode == "enforce"
