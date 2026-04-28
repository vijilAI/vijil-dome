"""Integration test: secure_agent() with a real Google ADK Agent.

Tests trust enforcement via ADK callbacks:
- before_model_callback: input guard
- after_model_callback: output guard
- before_tool_callback: tool MAC
- after_tool_callback: tool response guard
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from vijil_dome.trust.adapters.adk import secure_agent

# Local constraints — no Console needed
CONSTRAINTS = {
    "agent_id": "test-adk-agent",
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
    "updated_at": "2026-04-10T00:00:00Z",
}


@pytest.fixture
def adk_agent():
    """Create a real ADK Agent with a simple tool."""
    from google.adk import Agent

    def search_flights(destination: str) -> dict[str, Any]:
        """Search for flights to a destination."""
        return {"flights": [{"airline": "TestAir", "destination": destination, "price": 299}]}

    agent = Agent(
        name="travel_agent",
        model="gemini-2.0-flash",
        instruction="You are a helpful travel agent.",
        tools=[search_flights],
    )
    return agent


class TestSecureAgentCallbackInjection:
    """Test that secure_agent() injects the right callbacks."""

    def test_injects_callbacks(self, adk_agent):
        """secure_agent adds all four callback types."""
        secure_agent(adk_agent, agent_id="test", constraints=CONSTRAINTS)

        assert adk_agent.before_model_callback is not None
        assert adk_agent.after_model_callback is not None
        assert adk_agent.before_tool_callback is not None
        assert adk_agent.after_tool_callback is not None

    def test_runtime_attached(self, adk_agent):
        """TrustRuntime is accessible from the agent."""
        secure_agent(adk_agent, agent_id="test", constraints=CONSTRAINTS)

        assert hasattr(adk_agent, "_vijil_runtime")
        assert adk_agent._vijil_runtime.mode == "warn"

    def test_attestation_attached(self, adk_agent):
        """Attestation result is accessible from the agent."""
        secure_agent(adk_agent, agent_id="test", constraints=CONSTRAINTS)

        assert hasattr(adk_agent, "_vijil_attestation")
        assert adk_agent._vijil_attestation.all_verified  # No manifest = vacuously true

    def test_returns_same_agent(self, adk_agent):
        """secure_agent modifies in place and returns the same object."""
        result = secure_agent(adk_agent, agent_id="test", constraints=CONSTRAINTS)
        assert result is adk_agent

    def test_appends_to_existing_callbacks(self, adk_agent):
        """If the agent already has callbacks, new ones are appended."""
        existing_callback = MagicMock()
        adk_agent.before_model_callback = existing_callback

        secure_agent(adk_agent, agent_id="test", constraints=CONSTRAINTS)

        assert isinstance(adk_agent.before_model_callback, list)
        assert existing_callback in adk_agent.before_model_callback
        assert len(adk_agent.before_model_callback) == 2


class TestToolMACViaCallback:
    """Test tool MAC enforcement through the before_tool_callback."""

    def test_permitted_tool_returns_none(self, adk_agent):
        """Permitted tool: callback returns None (allow execution)."""
        secure_agent(adk_agent, agent_id="test", constraints=CONSTRAINTS, mode="warn")

        callback = adk_agent.before_tool_callback
        mock_tool = MagicMock()
        mock_tool.name = "search_flights"
        mock_context = MagicMock()

        result = callback(mock_tool, {"destination": "Paris"}, mock_context)
        assert result is None  # None means "allow"

    def test_denied_tool_warns_in_warn_mode(self, adk_agent):
        """Denied tool in warn mode: callback returns None (allow with warning)."""
        secure_agent(adk_agent, agent_id="test", constraints=CONSTRAINTS, mode="warn")

        callback = adk_agent.before_tool_callback
        mock_tool = MagicMock()
        mock_tool.name = "delete_records"
        mock_context = MagicMock()

        result = callback(mock_tool, {}, mock_context)
        assert result is None  # Warn mode allows

    def test_denied_tool_blocks_in_enforce_mode(self, adk_agent):
        """Denied tool in enforce mode: callback returns error dict."""
        enforce_constraints = {**CONSTRAINTS, "enforcement_mode": "enforce"}
        secure_agent(adk_agent, agent_id="test", constraints=enforce_constraints, mode="enforce")

        callback = adk_agent.before_tool_callback
        mock_tool = MagicMock()
        mock_tool.name = "delete_records"
        mock_context = MagicMock()

        result = callback(mock_tool, {}, mock_context)
        assert result is not None
        assert "error" in result
        assert "delete_records" in result["error"]

    def test_unknown_tool_denied(self, adk_agent):
        """Tool not in permissions is denied."""
        enforce_constraints = {**CONSTRAINTS, "enforcement_mode": "enforce"}
        secure_agent(adk_agent, agent_id="test", constraints=enforce_constraints, mode="enforce")

        callback = adk_agent.before_tool_callback
        mock_tool = MagicMock()
        mock_tool.name = "send_email"
        mock_context = MagicMock()

        result = callback(mock_tool, {}, mock_context)
        assert result is not None
        assert "error" in result


class TestInputGuardViaCallback:
    """Test input guard enforcement through before_model_callback."""

    def test_unflagged_input_passes(self, adk_agent):
        """Clean input: callback returns None (allow)."""
        secure_agent(adk_agent, agent_id="test", constraints=CONSTRAINTS)

        callback = adk_agent.before_model_callback
        mock_context = MagicMock()

        # Simulate LlmRequest with user content
        from google.genai import types
        mock_request = MagicMock()
        mock_request.contents = [
            types.Content(role="user", parts=[types.Part(text="Book a flight to Paris")])
        ]

        result = callback(mock_context, mock_request)
        assert result is None  # No Dome = passthrough = None
