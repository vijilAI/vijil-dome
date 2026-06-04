"""ToolPolicy — Mandatory Access Control enforcement for agent tool calls."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from vijil_dome.trust.constraints import AgentConstraints, ToolPermission


class ToolCallResult(BaseModel):
    """Result of a MAC check for a single tool call."""

    permitted: bool
    tool_name: str
    args: dict[str, Any] | None = None
    identity_verified: bool = False
    agent_spiffe_id: str | None = None
    policy_permitted: bool = False
    enforced: bool = False
    error: str | None = None


class ToolPolicy:
    """Mandatory Access Control policy derived from AgentConstraints.

    Checks whether an agent is permitted to call a named tool by:
    1. Rejecting tools denied at the organization level.
    2. Rejecting tools absent from the agent's explicit permission list.
    3. Checking allowed_actions if the permission restricts specific actions.
    4. Honouring the enforcement mode — enforced=True only when the call
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
        self._unattested_tool_policy: str = constraints.unattested_tool_policy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        *,
        spiffe_id: str | None = None,
        attested: bool = False,
    ) -> ToolCallResult:
        """Return a ToolCallResult reflecting the MAC decision.

        Args:
            tool_name: The tool being called.
            args: The arguments passed to the tool. Carried on the result for
                potential downstream audit/debugging (not emitted today); future
                versions may enforce parameter-level constraints (e.g., allowed
                parameter ranges, required fields).
            spiffe_id: The calling agent's SPIFFE ID, recorded on the result for
                audit. Threaded for downstream identity-keyed policy (A2/A3); the
                permit/deny outcome is NOT keyed on it here. It is recorded even
                when ``attested`` is False, so consumers MUST gate on
                ``identity_verified`` before keying any decision on it.
            attested: Whether the agent's identity is attested. Recorded as
                ``identity_verified``. When ``unattested_tool_policy`` is "deny", an
                unattested agent is denied here (fail-closed); enforcement_mode then
                decides whether that deny is enforced or only logged.
        """
        # A2: fail-closed on an unattested identity when the operator opted into "deny".
        # An unattested agent is denied regardless of which tool it calls.
        if not attested and self._unattested_tool_policy == "deny":
            return self._deny(
                tool_name,
                "agent identity is not attested",
                args,
                spiffe_id=spiffe_id,
                attested=attested,
            )

        if tool_name in self._denied_tools:
            return self._deny(
                tool_name, "denied by organization constraints", args,
                spiffe_id=spiffe_id, attested=attested,
            )

        perm = self._permissions.get(tool_name)
        if perm is None:
            return self._deny(
                tool_name, "not in agent permissions", args,
                spiffe_id=spiffe_id, attested=attested,
            )

        # Check allowed_actions if the permission restricts them.
        # Fail-closed: missing or None action is denied when allowed_actions is set.
        if perm.allowed_actions is not None and args is not None:
            action = args.get("action")
            if action is None or action not in perm.allowed_actions:
                return self._deny(
                    tool_name,
                    f"action {action!r} not in allowed_actions {perm.allowed_actions}",
                    args,
                    spiffe_id=spiffe_id,
                    attested=attested,
                )

        return ToolCallResult(
            permitted=True,
            tool_name=tool_name,
            args=args,
            identity_verified=attested,
            agent_spiffe_id=spiffe_id,
            policy_permitted=True,
        )

    def get_permission(self, tool_name: str) -> ToolPermission | None:
        """Return the ToolPermission entry for *tool_name*, or None if absent."""
        return self._permissions.get(tool_name)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _deny(
        self,
        tool_name: str,
        error: str,
        args: dict[str, Any] | None = None,
        *,
        spiffe_id: str | None = None,
        attested: bool = False,
    ) -> ToolCallResult:
        enforced = self._enforcement_mode == "enforce"
        return ToolCallResult(
            permitted=False,
            tool_name=tool_name,
            args=args,
            identity_verified=attested,
            agent_spiffe_id=spiffe_id,
            policy_permitted=False,
            enforced=enforced,
            error=error,
        )
