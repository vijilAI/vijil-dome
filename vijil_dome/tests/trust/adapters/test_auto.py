"""Tests for the unified secure_agent() dispatcher."""

import operator
from typing import Annotated, Any, TypedDict

import pytest

from vijil_dome.trust.adapters.auto import _detect_framework, secure_agent


class SimpleState(TypedDict):
    messages: Annotated[list[dict[str, Any]], operator.add]


CONSTRAINTS = {
    "agent_id": "test",
    "dome_config": {"input_guards": [], "output_guards": [], "guards": {}},
    "tool_permissions": [],
    "organization": {"required_input_guards": [], "required_output_guards": [], "denied_tools": []},
    "enforcement_mode": "warn",
    "updated_at": "2026-04-12T00:00:00Z",
}


class TestDetectFramework:
    def test_detects_langgraph(self):
        from langgraph.graph import StateGraph
        graph = StateGraph(SimpleState)
        assert _detect_framework(graph) == "langgraph"

    def test_detects_adk(self):
        from google.adk import Agent
        agent = Agent(name="test_agent", model="gemini-2.0-flash", instruction="test")
        assert _detect_framework(agent) == "adk"

    def test_detects_strands(self):
        from strands import Agent
        agent = Agent.__new__(Agent)  # Avoid full init (needs model config)
        assert _detect_framework(agent) == "strands"

    def test_unknown_raises(self):
        assert _detect_framework("not an agent") == "unknown"


class TestSecureAgentDispatch:
    def test_dispatches_langgraph(self):
        from langgraph.graph import END, StateGraph

        graph = StateGraph(SimpleState)
        graph.add_node("noop", lambda s: {"messages": []})
        graph.set_entry_point("noop")
        graph.add_edge("noop", END)

        result = secure_agent(graph, agent_id="test", constraints=CONSTRAINTS)
        # Returns a SecureGraph
        assert hasattr(result, "invoke")
        assert hasattr(result, "runtime")

    def test_dispatches_adk(self):
        from google.adk import Agent
        agent = Agent(name="test_agent", model="gemini-2.0-flash", instruction="test")

        result = secure_agent(agent, agent_id="test", constraints=CONSTRAINTS)
        # Returns the same agent (modified in place)
        assert result is agent
        assert hasattr(agent, "_vijil_runtime")

    def test_dispatches_strands(self):
        from strands import Agent
        agent = Agent.__new__(Agent)

        result = secure_agent(agent, agent_id="test", constraints=CONSTRAINTS)
        # Returns the same agent (modified in place)
        assert result is agent
        assert hasattr(agent, "_vijil_runtime")

    def test_unknown_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported agent type"):
            secure_agent("not an agent", agent_id="test", constraints=CONSTRAINTS)

    def test_importable_from_vijil(self):
        from vijil import secure_agent as sa
        assert callable(sa)
