"""secure_agent and all three adapters forward heartbeat params (DOME-169 Task 3.1).

The adapters' only job here is to thread ``heartbeat_interval`` and
``beacon_signer`` into ``TrustRuntime`` — the scheduler behaviour itself is
covered by the runtime tests. A capturing fake runtime records the constructor
kwargs and aborts early, so these tests verify the forwarding without the
framework graph/callback machinery. The corrected scope includes LangGraph,
which does NOT get the params for free (its ``**compile_kwargs`` route to
``graph.compile()``, not to ``TrustRuntime``).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from vijil_dome.trust.signing import UnwiredBeaconSigner

_SIGNER = UnwiredBeaconSigner()
_INTERVAL = 0.5


class _Captured(Exception):
    """Raised by the fake runtime to capture ctor kwargs and abort early."""


class _CapturingRuntime:
    last_kwargs: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        _CapturingRuntime.last_kwargs = dict(kwargs)
        raise _Captured


def _assert_forwarded() -> None:
    captured = _CapturingRuntime.last_kwargs
    assert captured["heartbeat_interval"] == _INTERVAL
    assert captured["beacon_signer"] is _SIGNER


def test_langgraph_secure_graph_forwards(monkeypatch: pytest.MonkeyPatch) -> None:
    from vijil_dome.trust.adapters import langgraph as lg

    monkeypatch.setattr(lg, "TrustRuntime", _CapturingRuntime)
    with pytest.raises(_Captured):
        lg.secure_graph(
            SimpleNamespace(),
            agent_id="a",
            constraints={},
            heartbeat_interval=_INTERVAL,
            beacon_signer=_SIGNER,
        )
    _assert_forwarded()


def test_adk_secure_agent_forwards(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("google.adk")
    from vijil_dome.trust.adapters import adk

    monkeypatch.setattr(adk, "TrustRuntime", _CapturingRuntime)
    with pytest.raises(_Captured):
        adk.secure_agent(
            SimpleNamespace(),
            agent_id="a",
            constraints={},
            heartbeat_interval=_INTERVAL,
            beacon_signer=_SIGNER,
        )
    _assert_forwarded()


def test_strands_secure_agent_forwards(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("strands")
    from vijil_dome.trust.adapters import strands

    monkeypatch.setattr(strands, "TrustRuntime", _CapturingRuntime)
    with pytest.raises(_Captured):
        strands.secure_agent(
            SimpleNamespace(),
            agent_id="a",
            constraints={},
            heartbeat_interval=_INTERVAL,
            beacon_signer=_SIGNER,
        )
    _assert_forwarded()


def test_auto_secure_agent_forwards_to_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    from vijil_dome.trust.adapters import auto

    captured: dict[str, Any] = {}

    def _fake_secure_graph(graph: Any, **kwargs: Any) -> str:
        captured.update(kwargs)
        return "wrapped"

    monkeypatch.setattr(auto, "_detect_framework", lambda agent: "langgraph")
    monkeypatch.setattr(
        "vijil_dome.trust.adapters.langgraph.secure_graph", _fake_secure_graph
    )
    result = auto.secure_agent(
        SimpleNamespace(),
        agent_id="a",
        constraints={},
        heartbeat_interval=_INTERVAL,
        beacon_signer=_SIGNER,
    )
    assert result == "wrapped"
    assert captured["heartbeat_interval"] == _INTERVAL
    assert captured["beacon_signer"] is _SIGNER
