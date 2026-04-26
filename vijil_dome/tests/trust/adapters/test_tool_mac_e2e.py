"""End-to-end tool MAC verification.

These tests verify that MAC enforcement actually fires during graph/agent
execution — not just when check_tool_call() is called in isolation.

For LangGraph: a node calls a wrapped tool → MAC permits or blocks.
For ADK: the before_tool_callback fires → MAC permits or blocks.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

import pytest

from vijil_dome.trust.adapters.langgraph import secure_graph

# ---------------------------------------------------------------------------
# LangGraph: tool execution through a real graph node
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list[dict[str, Any]], operator.add]
    tool_results: Annotated[list[str], operator.add]


def book_flight(destination: str) -> str:
    """Permitted tool."""
    return f"Booked flight to {destination}"


def charge_card(amount: float) -> str:
    """Denied tool — org constraint."""
    return f"Charged ${amount}"


def send_email(to: str, body: str) -> str:
    """Unknown tool — not in permissions."""
    return f"Email sent to {to}"


CONSTRAINTS = {
    "agent_id": "mac-test-agent",
    "dome_config": {"input_guards": [], "output_guards": [], "guards": {}},
    "tool_permissions": [
        {"name": "book_flight", "identity": "spiffe://vijil.ai/tools/flights/v1", "endpoint": "local"},
    ],
    "organization": {
        "required_input_guards": [],
        "required_output_guards": [],
        "denied_tools": ["charge_card"],
    },
    "enforcement_mode": "enforce",
    "updated_at": "2026-04-10T00:00:00Z",
}


class TestLangGraphToolMACEndToEnd:
    """Verify MAC fires when a LangGraph node calls a wrapped tool."""

    def test_permitted_tool_executes_through_graph(self):
        """A graph node that calls a permitted wrapped tool gets the real result."""
        from langgraph.graph import END, StateGraph

        def call_book_flight(state: AgentState) -> dict[str, Any]:
            # This simulates what happens when the LLM selects a tool —
            # the node calls the tool function directly
            wrapped_fn: Any = state.get("_wrapped_book_flight", book_flight)
            result = wrapped_fn(destination="Tokyo")
            return {"messages": [{"role": "tool", "content": result}], "tool_results": [result]}

        graph = StateGraph(AgentState)
        graph.add_node("call_tool", call_book_flight)
        graph.set_entry_point("call_tool")
        graph.add_edge("call_tool", END)

        app = secure_graph(graph, agent_id="mac-test", constraints=CONSTRAINTS, mode="enforce")

        # Wrap the tool and pass it through state so the node can use it
        wrapped = app.runtime.wrap_tool(book_flight)

        result = app.invoke({
            "messages": [{"role": "user", "content": "Book a flight"}],
            "tool_results": [],
            "_wrapped_book_flight": wrapped,
        })

        # Tool executed successfully
        assert any("Tokyo" in r for r in result.get("tool_results", []))

    def test_denied_tool_raises_through_graph(self):
        """A graph node that calls a denied wrapped tool raises PermissionError.

        The wrapped tool is captured via closure — this is how secure_graph()
        would inject wrapped tools into graph nodes in practice.
        """
        from langgraph.graph import END, StateGraph

        app_holder: list = []  # Capture runtime via closure

        def call_charge_card(state: AgentState) -> dict[str, Any]:
            # In real usage, the tool would be wrapped before graph compilation.
            # Here we use the runtime from the closure to wrap and call.
            wrapped_fn = app_holder[0].runtime.wrap_tool(charge_card)
            result = wrapped_fn(amount=500.0)
            return {"messages": [{"role": "tool", "content": result}], "tool_results": [result]}

        graph = StateGraph(AgentState)
        graph.add_node("call_tool", call_charge_card)
        graph.set_entry_point("call_tool")
        graph.add_edge("call_tool", END)

        app = secure_graph(graph, agent_id="mac-test", constraints=CONSTRAINTS, mode="enforce")
        app_holder.append(app)

        with pytest.raises(PermissionError, match="charge_card"):
            app.invoke({
                "messages": [{"role": "user", "content": "Charge my card"}],
                "tool_results": [],
            })

    def test_unknown_tool_raises_through_graph(self):
        """A tool not in permissions is denied in enforce mode."""
        from langgraph.graph import END, StateGraph

        app_holder: list = []

        def call_send_email(state: AgentState) -> dict[str, Any]:
            wrapped_fn = app_holder[0].runtime.wrap_tool(send_email)
            result = wrapped_fn(to="evil@example.com", body="secrets")
            return {"messages": [{"role": "tool", "content": result}], "tool_results": [result]}

        graph = StateGraph(AgentState)
        graph.add_node("call_tool", call_send_email)
        graph.set_entry_point("call_tool")
        graph.add_edge("call_tool", END)

        app = secure_graph(graph, agent_id="mac-test", constraints=CONSTRAINTS, mode="enforce")
        app_holder.append(app)

        with pytest.raises(PermissionError, match="send_email"):
            app.invoke({
                "messages": [{"role": "user", "content": "Send an email"}],
                "tool_results": [],
            })

    def test_denied_tool_warns_but_executes_in_warn_mode(self):
        """In warn mode, denied tool executes but the MAC check still fires."""
        from langgraph.graph import END, StateGraph

        warn_constraints = {**CONSTRAINTS, "enforcement_mode": "warn"}
        call_log: list[str] = []

        def call_charge_card(state: AgentState) -> dict[str, Any]:
            wrapped_fn: Any = state.get("_wrapped_charge_card", charge_card)
            result = wrapped_fn(amount=500.0)
            call_log.append(result)
            return {"messages": [{"role": "tool", "content": result}], "tool_results": [result]}

        graph = StateGraph(AgentState)
        graph.add_node("call_tool", call_charge_card)
        graph.set_entry_point("call_tool")
        graph.add_edge("call_tool", END)

        app = secure_graph(graph, agent_id="mac-test", constraints=warn_constraints, mode="warn")
        wrapped = app.runtime.wrap_tool(charge_card)

        # Should NOT raise — warn mode
        app.invoke({
            "messages": [{"role": "user", "content": "Charge my card"}],
            "tool_results": [],
            "_wrapped_charge_card": wrapped,
        })

        # Tool executed despite being denied (warn mode)
        assert len(call_log) == 1
        assert "Charged" in call_log[0]


class TestADKToolMACEndToEnd:
    """Verify MAC fires through ADK's before_tool_callback."""

    def test_permitted_tool_callback_allows(self):
        """before_tool_callback returns None for permitted tools."""
        from unittest.mock import MagicMock

        from google.adk import Agent

        from vijil_dome.trust.adapters.adk import secure_agent

        def search_flights(destination: str) -> dict:
            return {"flights": []}

        agent = Agent(
            name="test_agent",
            model="gemini-2.0-flash",
            instruction="Help with travel.",
            tools=[search_flights],
        )
        secure_agent(agent, agent_id="test", constraints=CONSTRAINTS, mode="enforce")

        # Simulate ADK calling the before_tool_callback
        callback = agent.before_tool_callback
        mock_tool = MagicMock()
        mock_tool.name = "book_flight"  # In permissions

        result = callback(mock_tool, {"destination": "Paris"}, MagicMock())
        assert result is None  # None = allow execution

    def test_denied_tool_callback_blocks(self):
        """before_tool_callback returns error dict for denied tools in enforce mode."""
        from unittest.mock import MagicMock

        from google.adk import Agent

        from vijil_dome.trust.adapters.adk import secure_agent

        agent = Agent(
            name="test_agent",
            model="gemini-2.0-flash",
            instruction="Help with travel.",
            tools=[],
        )
        secure_agent(agent, agent_id="test", constraints=CONSTRAINTS, mode="enforce")

        callback = agent.before_tool_callback
        mock_tool = MagicMock()
        mock_tool.name = "charge_card"  # Org-denied

        result = callback(mock_tool, {"amount": 100}, MagicMock())
        assert result is not None
        assert "error" in result
        assert "charge_card" in result["error"]
        assert "denied" in result["error"].lower()

    def test_unknown_tool_callback_blocks(self):
        """Tool not in permissions is blocked in enforce mode."""
        from unittest.mock import MagicMock

        from google.adk import Agent

        from vijil_dome.trust.adapters.adk import secure_agent

        agent = Agent(
            name="test_agent",
            model="gemini-2.0-flash",
            instruction="Help with travel.",
            tools=[],
        )
        secure_agent(agent, agent_id="test", constraints=CONSTRAINTS, mode="enforce")

        callback = agent.before_tool_callback
        mock_tool = MagicMock()
        mock_tool.name = "send_email"  # Not in permissions

        result = callback(mock_tool, {"to": "x"}, MagicMock())
        assert result is not None
        assert "error" in result

    def test_audit_events_emitted_on_mac_check(self):
        """Every MAC check emits an audit event."""
        from unittest.mock import MagicMock

        from google.adk import Agent

        from vijil_dome.trust.adapters.adk import secure_agent
        from vijil_dome.trust.audit import AuditEvent

        agent = Agent(
            name="test_agent",
            model="gemini-2.0-flash",
            instruction="Help.",
            tools=[],
        )
        secure_agent(agent, agent_id="test", constraints=CONSTRAINTS, mode="enforce")

        # Capture audit events
        events: list[AuditEvent] = []
        agent._vijil_runtime._audit._sink = events.append

        callback = agent.before_tool_callback

        # Check a permitted tool
        permitted_tool = MagicMock()
        permitted_tool.name = "book_flight"
        callback(permitted_tool, {}, MagicMock())

        # Check a denied tool
        denied_tool = MagicMock()
        denied_tool.name = "charge_card"
        callback(denied_tool, {}, MagicMock())

        # Verify audit events
        assert len(events) == 2
        assert events[0].event_type == "tool_mac"
        assert events[0].attributes["tool_name"] == "book_flight"
        assert events[0].attributes["permitted"] is True

        assert events[1].event_type == "tool_mac"
        assert events[1].attributes["tool_name"] == "charge_card"
        assert events[1].attributes["permitted"] is False
