"""ToolPolicy — Mandatory Access Control enforcement for agent tool calls."""

from __future__ import annotations

from vijil_dome.trust.models import TrustModel
from vijil_dome.trust.constraints import AgentConstraints, ToolPermission


class ToolCallResult(TrustModel):
    """Result of a MAC check for a single tool call."""

    permitted: bool
    tool_name: str
    identity_verified: bool = False
    policy_permitted: bool = False
    enforced: bool = False
    error: str | None = None


class ToolPolicy:
    """Mandatory Access Control policy derived from AgentConstraints.

    Checks whether an agent is permitted to call a named tool by:
    1. Rejecting tools denied at the organization level.
    2. Rejecting tools absent from the agent's explicit permission list.
    3. Honouring the enforcement mode — enforced=True only when the call
       would be blocked AND the mode is "enforce".
    """

    def __init__(self, constraints: AgentConstraints) -> None:
        self._permissions: dict[str, ToolPermission] = {
            p.name: p for p in constraints.tool_permissions
        }
        self._denied_tools: frozenset[str] = frozenset(
            constraints.organization.denied_tools
        )
        self._enforcement_mode: str = constraints.enforcement_mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, tool_name: str) -> ToolCallResult:
        """Return a ToolCallResult reflecting the MAC decision for *tool_name*."""
        if tool_name in self._denied_tools:
            return self._deny(tool_name, "denied by organization constraints")

        if tool_name not in self._permissions:
            return self._deny(tool_name, "not in agent permissions")

        return ToolCallResult(
            permitted=True,
            tool_name=tool_name,
            policy_permitted=True,
        )

    def get_permission(self, tool_name: str) -> ToolPermission | None:
        """Return the ToolPermission entry for *tool_name*, or None if absent."""
        return self._permissions.get(tool_name)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _deny(self, tool_name: str, error: str) -> ToolCallResult:
        enforced = self._enforcement_mode == "enforce"
        return ToolCallResult(
            permitted=False,
            tool_name=tool_name,
            policy_permitted=False,
            enforced=enforced,
            error=error,
        )
