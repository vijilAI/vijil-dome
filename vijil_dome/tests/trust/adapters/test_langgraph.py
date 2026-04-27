"""Tests for the LangGraph adapter — SecureGraph and secure_graph factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vijil_dome.trust.guard import EnforcementResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _unflagged_guard() -> EnforcementResult:
    return EnforcementResult(
        flagged=False,
        enforced=False,
        score=0.0,
        guarded_response=None,
        exec_time_ms=0.0,
        trace=[],
    )


def _flagged_guard(*, enforced: bool = True) -> EnforcementResult:
    return EnforcementResult(
        flagged=True,
        enforced=enforced,
        score=0.95,
        guarded_response="I cannot help with that request.",
        exec_time_ms=1.2,
        trace=[],
    )


@pytest.fixture()
def mock_runtime() -> MagicMock:
    rt = MagicMock()
    rt.guard_input.return_value = _unflagged_guard()
    rt.guard_output.return_value = _unflagged_guard()
    rt.attest.return_value = MagicMock(all_verified=True)
    rt.mode = "warn"
    rt.wrap_tools.side_effect = lambda tools: tools  # identity
    return rt


@pytest.fixture()
def mock_graph() -> MagicMock:
    """A mock compiled LangGraph (no .compile method)."""
    graph = MagicMock()
    # Remove compile so it looks like an already-compiled graph
    del graph.compile
    graph.invoke.return_value = {"messages": ["The weather is sunny."]}
    graph.stream.return_value = iter([
        {"messages": ["chunk1"]},
        {"messages": ["chunk2"]},
    ])
    return graph


# ---------------------------------------------------------------------------
# 1. test_secure_graph_wraps_invoke
# ---------------------------------------------------------------------------


def test_secure_graph_wraps_invoke(mock_runtime: MagicMock, mock_graph: MagicMock) -> None:
    """invoke() should call guard_input, delegate to graph.invoke, then guard_output."""
    from vijil_dome.trust.adapters.langgraph import SecureGraph

    sg = SecureGraph(graph=mock_graph, runtime=mock_runtime)

    result = sg.invoke({"messages": ["What is the weather?"]})

    # graph.invoke was called with the original input
    mock_graph.invoke.assert_called_once_with({"messages": ["What is the weather?"]}, None)

    # Guards were called
    mock_runtime.guard_input.assert_called_once_with("What is the weather?")
    mock_runtime.guard_output.assert_called_once_with("The weather is sunny.")

    # Result is the graph's output
    assert result == {"messages": ["The weather is sunny."]}


# ---------------------------------------------------------------------------
# 2. test_invoke_blocked_in_enforce_mode
# ---------------------------------------------------------------------------


def test_invoke_blocked_in_enforce_mode(mock_runtime: MagicMock, mock_graph: MagicMock) -> None:
    """When guard_input flags and enforces, graph.invoke must NOT be called."""
    from vijil_dome.trust.adapters.langgraph import SecureGraph

    mock_runtime.mode = "enforce"
    mock_runtime.guard_input.return_value = _flagged_guard(enforced=True)

    sg = SecureGraph(graph=mock_graph, runtime=mock_runtime)

    result = sg.invoke({"messages": ["How to hack a server?"]})

    # graph.invoke must NOT have been called
    mock_graph.invoke.assert_not_called()

    # Result contains the guarded response
    assert result["messages"] == ["I cannot help with that request."]


# ---------------------------------------------------------------------------
# 3. test_stream_yields_chunks
# ---------------------------------------------------------------------------


def test_stream_yields_chunks(mock_runtime: MagicMock, mock_graph: MagicMock) -> None:
    """stream() should yield all chunks from the underlying graph.stream."""
    from vijil_dome.trust.adapters.langgraph import SecureGraph

    sg = SecureGraph(graph=mock_graph, runtime=mock_runtime)

    chunks = list(sg.stream({"messages": ["Tell me a story."]}))

    assert len(chunks) == 2
    assert chunks[0] == {"messages": ["chunk1"]}
    assert chunks[1] == {"messages": ["chunk2"]}

    # Input was guarded
    mock_runtime.guard_input.assert_called_once_with("Tell me a story.")


# ---------------------------------------------------------------------------
# 4. test_invoke_output_guard_enforced
# ---------------------------------------------------------------------------


def test_invoke_output_guard_enforced(mock_runtime: MagicMock, mock_graph: MagicMock) -> None:
    """When guard_output flags and enforces, the guarded response replaces graph output."""
    from vijil_dome.trust.adapters.langgraph import SecureGraph

    mock_runtime.mode = "enforce"
    mock_runtime.guard_output.return_value = _flagged_guard(enforced=True)

    sg = SecureGraph(graph=mock_graph, runtime=mock_runtime)

    result = sg.invoke({"messages": ["Tell me something."]})

    # graph.invoke WAS called (input passed)
    mock_graph.invoke.assert_called_once()

    # But output was replaced with guarded response
    assert result["messages"] == ["I cannot help with that request."]


# ---------------------------------------------------------------------------
# 5. test_secure_graph_factory_compiles_uncompiled
# ---------------------------------------------------------------------------


def test_secure_graph_factory_compiles_uncompiled() -> None:
    """secure_graph() should call .compile() on an uncompiled StateGraph."""
    from vijil_dome.trust.adapters.langgraph import SecureGraph, secure_graph

    uncompiled = MagicMock()
    compiled = MagicMock()
    del compiled.compile  # compiled graph has no .compile
    compiled.invoke.return_value = {"messages": ["ok"]}
    uncompiled.compile.return_value = compiled

    with patch("vijil_dome.trust.adapters.langgraph.TrustRuntime") as mock_rt:
        rt_instance = MagicMock()
        rt_instance.attest.return_value = MagicMock(all_verified=True)
        rt_instance.mode = "warn"
        rt_instance.wrap_tools.side_effect = lambda tools: tools
        mock_rt.return_value = rt_instance

        sg = secure_graph(
            uncompiled,
            client=MagicMock(),
            agent_id="agent-1",
            mode="warn",
        )

    # .compile() was called on the uncompiled graph
    uncompiled.compile.assert_called_once()

    assert isinstance(sg, SecureGraph)


# ---------------------------------------------------------------------------
# 6. test_secure_graph_factory_uses_compiled_directly
# ---------------------------------------------------------------------------


def test_secure_graph_factory_uses_compiled_directly() -> None:
    """secure_graph() should use an already-compiled graph without calling .compile()."""
    from vijil_dome.trust.adapters.langgraph import SecureGraph, secure_graph

    compiled = MagicMock(spec=[])  # empty spec — no .compile attribute
    compiled.invoke = MagicMock(return_value={"messages": ["ok"]})
    compiled.stream = MagicMock()

    with patch("vijil_dome.trust.adapters.langgraph.TrustRuntime") as mock_rt:
        rt_instance = MagicMock()
        rt_instance.attest.return_value = MagicMock(all_verified=True)
        rt_instance.mode = "warn"
        rt_instance.wrap_tools.side_effect = lambda tools: tools
        mock_rt.return_value = rt_instance

        sg = secure_graph(
            compiled,
            client=MagicMock(),
            agent_id="agent-2",
        )

    assert isinstance(sg, SecureGraph)


# ---------------------------------------------------------------------------
# 7. test_invoke_handles_dict_message
# ---------------------------------------------------------------------------


def test_invoke_handles_dict_message(mock_runtime: MagicMock, mock_graph: MagicMock) -> None:
    """invoke() should extract 'content' from a dict message."""
    from vijil_dome.trust.adapters.langgraph import SecureGraph

    sg = SecureGraph(graph=mock_graph, runtime=mock_runtime)

    sg.invoke({"messages": [{"role": "user", "content": "Hello!"}]})

    mock_runtime.guard_input.assert_called_once_with("Hello!")
    mock_graph.invoke.assert_called_once()


# ---------------------------------------------------------------------------
# 8. test_attestation_available_at_construction
# ---------------------------------------------------------------------------


def test_attestation_available_at_construction(mock_runtime: MagicMock, mock_graph: MagicMock) -> None:
    """SecureGraph should call runtime.attest() and expose the result."""
    from vijil_dome.trust.adapters.langgraph import SecureGraph

    sg = SecureGraph(graph=mock_graph, runtime=mock_runtime)

    mock_runtime.attest.assert_called_once()
    assert sg.attestation.all_verified is True
