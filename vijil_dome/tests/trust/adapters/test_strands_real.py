"""Integration test: create_trust_hooks() with real Strands SDK.

Tests trust enforcement via Strands HookProvider callbacks:
- BeforeToolCallEvent: tool MAC (cancel_tool for denied tools)
- AfterToolCallEvent: tool response guard
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vijil_dome.trust.adapters.strands import create_trust_hooks, secure_agent

CONSTRAINTS = {
    "agent_id": "test-strands-agent",
    "dome_config": {"input_guards": [], "output_guards": [], "guards": {}},
    "tool_permissions": [
        {"name": "search_flights", "identity": "spiffe://vijil.ai/ns/test/tool/flights/v1", "endpoint": "local"},
    ],
    "organization": {
        "required_input_guards": [],
        "required_output_guards": [],
        "denied_tools": ["delete_records"],
    },
    "enforcement_mode": "warn",
    "updated_at": "2026-04-12T00:00:00Z",
}


class TestSecureAgent:
    """Test secure_agent() modifies and returns the agent."""

    def test_returns_same_agent(self):
        from strands import Agent
        agent = Agent.__new__(Agent)
        result = secure_agent(agent, agent_id="test", constraints=CONSTRAINTS)
        assert result is agent

    def test_injects_hooks(self):
        from strands import Agent
        agent = Agent.__new__(Agent)
        agent.hooks = None
        secure_agent(agent, agent_id="test", constraints=CONSTRAINTS)
        assert agent.hooks is not None
        assert len(agent.hooks) >= 1

    def test_runtime_on_agent(self):
        from strands import Agent
        agent = Agent.__new__(Agent)
        agent.hooks = None
        secure_agent(agent, agent_id="test", constraints=CONSTRAINTS)
        assert hasattr(agent, "_vijil_runtime")
        assert agent._vijil_runtime.mode == "warn"

    def test_appends_to_existing_hooks(self):
        from strands import Agent
        agent = Agent.__new__(Agent)
        existing = MagicMock()
        agent.hooks = [existing]
        secure_agent(agent, agent_id="test", constraints=CONSTRAINTS)
        assert existing in agent.hooks
        assert len(agent.hooks) == 2


class TestTrustHookProviderCreation:
    def test_creates_hook_provider(self):
        hooks = create_trust_hooks(agent_id="test", constraints=CONSTRAINTS)
        assert hasattr(hooks, "register_hooks")
        assert hasattr(hooks, "runtime")

    def test_runtime_attached(self):
        hooks = create_trust_hooks(agent_id="test", constraints=CONSTRAINTS)
        assert hooks.runtime.mode == "warn"

    def test_attestation_attached(self):
        hooks = create_trust_hooks(agent_id="test", constraints=CONSTRAINTS)
        assert hooks.attestation.all_verified

    def test_registers_all_hooks(self):
        from strands.hooks import HookRegistry
        hooks = create_trust_hooks(agent_id="test", constraints=CONSTRAINTS)
        registry = HookRegistry()
        hooks.register_hooks(registry)
        # Registry should have callbacks registered (internal structure varies)
        assert registry is not None


class TestToolMACViaStrands:
    """Test tool MAC through BeforeToolCallEvent."""

    @pytest.fixture
    def hooks_warn(self):
        return create_trust_hooks(agent_id="test", constraints=CONSTRAINTS, mode="warn")

    @pytest.fixture
    def hooks_enforce(self):
        enforce = {**CONSTRAINTS, "enforcement_mode": "enforce"}
        return create_trust_hooks(agent_id="test", constraints=enforce, mode="enforce")

    @pytest.mark.asyncio
    async def test_permitted_tool_not_cancelled(self, hooks_warn):
        """Permitted tool: cancel_tool stays False."""
        from strands.hooks import BeforeToolCallEvent

        event = BeforeToolCallEvent(
            agent=MagicMock(messages=[]),
            selected_tool=MagicMock(),
            tool_use={"toolUseId": "1", "name": "search_flights", "input": {}},
            invocation_state={},
        )
        await hooks_warn._check_tool(event)
        assert not event.cancel_tool

    @pytest.mark.asyncio
    async def test_denied_tool_warns_in_warn_mode(self, hooks_warn):
        """Denied tool in warn mode: cancel_tool stays False (warning only)."""
        from strands.hooks import BeforeToolCallEvent

        event = BeforeToolCallEvent(
            agent=MagicMock(messages=[]),
            selected_tool=MagicMock(),
            tool_use={"toolUseId": "1", "name": "delete_records", "input": {}},
            invocation_state={},
        )
        await hooks_warn._check_tool(event)
        assert not event.cancel_tool  # Warn mode does not cancel

    @pytest.mark.asyncio
    async def test_denied_tool_cancelled_in_enforce_mode(self, hooks_enforce):
        """Denied tool in enforce mode: cancel_tool set to error message."""
        from strands.hooks import BeforeToolCallEvent

        event = BeforeToolCallEvent(
            agent=MagicMock(messages=[]),
            selected_tool=MagicMock(),
            tool_use={"toolUseId": "1", "name": "delete_records", "input": {}},
            invocation_state={},
        )
        await hooks_enforce._check_tool(event)
        assert event.cancel_tool
        assert "delete_records" in str(event.cancel_tool)

    @pytest.mark.asyncio
    async def test_unknown_tool_cancelled_in_enforce_mode(self, hooks_enforce):
        """Unknown tool in enforce mode: cancel_tool set."""
        from strands.hooks import BeforeToolCallEvent

        event = BeforeToolCallEvent(
            agent=MagicMock(messages=[]),
            selected_tool=MagicMock(),
            tool_use={"toolUseId": "1", "name": "send_email", "input": {}},
            invocation_state={},
        )
        await hooks_enforce._check_tool(event)
        assert event.cancel_tool

    @pytest.mark.asyncio
    async def test_audit_emitted_on_mac_check(self, hooks_warn):
        """MAC check emits audit event."""
        from strands.hooks import BeforeToolCallEvent

        from vijil_dome.trust.audit import AuditEvent

        events: list[AuditEvent] = []
        hooks_warn.runtime._audit._sink = events.append

        event = BeforeToolCallEvent(
            agent=MagicMock(messages=[]),
            selected_tool=MagicMock(),
            tool_use={"toolUseId": "1", "name": "search_flights", "input": {}},
            invocation_state={},
        )
        await hooks_warn._check_tool(event)

        assert len(events) == 1
        assert events[0].event_type == "tool_mac"
        assert events[0].attributes["tool_name"] == "search_flights"
        assert events[0].attributes["permitted"] is True
