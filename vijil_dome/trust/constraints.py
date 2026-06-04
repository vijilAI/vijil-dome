"""Agent constraint models — Dome config, tool permissions, org policy."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class ToolPermission(BaseModel):
    """Permission entry for a single tool an agent is allowed to call."""

    name: str
    identity: str  # SPIFFE ID of the tool, e.g. spiffe://vijil.ai/tools/flights/v1
    endpoint: str  # Tool endpoint, e.g. mcp+tls://flights.internal:8443
    allowed_actions: list[str] | None = None  # None means all actions permitted


class DomeGuardConfig(BaseModel):
    """Dome guardrail configuration for an agent."""

    input_guards: list[str]
    output_guards: list[str]
    guards: dict[str, dict[str, object]]  # guard name → detector config


class OrganizationConstraints(BaseModel):
    """Organization-level policy constraints applied to all agents."""

    required_input_guards: list[str]
    required_output_guards: list[str]
    denied_tools: list[str]
    max_model_tier: str | None = None


class AgentConstraints(BaseModel):
    """Full constraint set for a single agent instance."""

    agent_id: str
    dome_config: DomeGuardConfig
    tool_permissions: list[ToolPermission]
    organization: OrganizationConstraints
    enforcement_mode: Literal["warn", "enforce"]
    # How to treat an UNATTESTED agent's tool calls. "warn" (default) preserves today's behavior
    # (the normal tool check applies) so existing enforce-mode-but-unattested flows are not
    # bricked; "deny" fails closed — an unattested agent is denied regardless of tool. The flag
    # sets the DECISION; enforcement_mode decides whether that deny is enforced.
    unattested_tool_policy: Literal["warn", "deny"] = "warn"
    updated_at: datetime
