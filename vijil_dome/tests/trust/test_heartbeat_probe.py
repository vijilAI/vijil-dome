"""Real, bounded detector-reachability probe for the heartbeat (DOME-169 Task 2.1).

``detector_reachable`` must mean "the detector backend actually responds," not
merely "a Dome instance was constructed." The guard engine catches a failing
detector internally and records it in ``ScanResult.errored_methods`` rather than
raising, so the probe reads reachability from that field — across whichever
guardrail (input and/or output) is actually configured — and caches the result
for a TTL so the real detectors run at most once per heartbeat cadence.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from vijil_dome.trust.audit import AuditEvent
from vijil_dome.trust.constraints import AgentConstraints
from vijil_dome.trust.runtime import TrustRuntime


def _constraints() -> AgentConstraints:
    return AgentConstraints.model_validate(
        {
            "agent_id": "agent-1",
            "dome_config": {"input_guards": [], "output_guards": [], "guards": {}},
            "tool_permissions": [],
            "organization": {
                "required_input_guards": [],
                "required_output_guards": [],
                "denied_tools": [],
            },
            "enforcement_mode": "enforce",
            "updated_at": "2026-04-03T12:00:00+00:00",
        }
    )


def _runtime() -> TrustRuntime:
    client = MagicMock()
    client._http._token = "test-api-key"  # api-key path -> unattested
    client._http.get.return_value = _constraints().model_dump(mode="json")
    return TrustRuntime(client=client, agent_id="agent-1", mode="warn")


class _StubScanResult:
    def __init__(self, errored_methods: list[str] | None = None) -> None:
        self.errored_methods = errored_methods or []


class _StubDome:
    """A Dome stand-in mirroring the real guard contract.

    The real Dome records a failing detector in ``errored_methods`` and returns
    a normal ``ScanResult`` — it does NOT raise. The stub does the same so the
    tests exercise the contract production actually honors. ``raises`` simulates
    the rarer non-detector failure path.
    """

    def __init__(
        self,
        *,
        errored: bool = False,
        raises: bool = False,
        has_input: bool = True,
        has_output: bool = False,
    ) -> None:
        self.calls = 0
        self._errored = errored
        self._raises = raises
        self.input_guardrail = object() if has_input else None
        self.output_guardrail = object() if has_output else None

    def _scan(self, label: str) -> _StubScanResult:
        self.calls += 1
        if self._raises:
            raise RuntimeError("guard engine crashed")
        return _StubScanResult(errored_methods=[label] if self._errored else [])

    def guard_input(self, text: str) -> _StubScanResult:
        return self._scan("pi-detector")

    def guard_output(self, text: str) -> _StubScanResult:
        return self._scan("toxicity-detector")


def test_probe_false_when_no_guards_configured() -> None:
    runtime = _runtime()  # empty guards -> _dome is None
    assert runtime._probe_detector_reachable() is False


def test_probe_false_when_guards_disabled() -> None:
    runtime = _runtime()
    runtime._dome = _StubDome()
    runtime._guards_disabled = True
    assert runtime._probe_detector_reachable() is False


def test_probe_true_when_backend_responds() -> None:
    runtime = _runtime()
    runtime._dome = _StubDome(errored=False)
    assert runtime._probe_detector_reachable() is True


def test_probe_false_when_detector_errors() -> None:
    # The key case: the real Dome does NOT raise on a dead backend — it records
    # the detector in errored_methods. A probe that only caught exceptions would
    # report this dead backend as reachable.
    runtime = _runtime()
    runtime._dome = _StubDome(errored=True)
    assert runtime._probe_detector_reachable() is False


def test_probe_false_when_backend_raises() -> None:
    # Defensive: a non-detector failure (engine crash) also reads as unreachable.
    runtime = _runtime()
    runtime._dome = _StubDome(raises=True)
    assert runtime._probe_detector_reachable() is False


def test_probe_uses_output_guardrail_when_only_output_configured() -> None:
    # An output-only config: guard_input is a no-op, so the probe must exercise
    # the output guardrail or it would falsely report reachable.
    runtime = _runtime()
    runtime._dome = _StubDome(has_input=False, has_output=True, errored=False)
    assert runtime._probe_detector_reachable() is True


def test_probe_false_when_output_only_detector_errors() -> None:
    runtime = _runtime()
    runtime._dome = _StubDome(has_input=False, has_output=True, errored=True)
    assert runtime._probe_detector_reachable() is False


def test_probe_false_when_dome_has_no_guardrails() -> None:
    runtime = _runtime()
    runtime._dome = _StubDome(has_input=False, has_output=False)
    assert runtime._probe_detector_reachable() is False


def test_probe_caches_within_ttl() -> None:
    runtime = _runtime()
    stub = _StubDome()
    runtime._dome = stub
    assert runtime._probe_detector_reachable() is True
    assert runtime._probe_detector_reachable() is True
    assert stub.calls == 1  # second call served from cache


def test_probe_reruns_after_ttl_expires() -> None:
    runtime = _runtime()
    stub = _StubDome()
    runtime._dome = stub
    fake_now = [0.0]
    runtime._clock = lambda: fake_now[0]
    runtime._detector_probe_ttl_s = 10.0

    assert runtime._probe_detector_reachable() is True  # probe at t=0
    fake_now[0] = 5.0
    assert runtime._probe_detector_reachable() is True  # within TTL -> cached
    assert stub.calls == 1
    fake_now[0] = 20.0
    assert runtime._probe_detector_reachable() is True  # TTL expired -> re-probe
    assert stub.calls == 2


def test_emit_heartbeat_distinguishes_constructed_from_reachable() -> None:
    # The core semantic: a built-but-unreachable backend reports
    # guards_constructed=True AND detector_reachable=False.
    events: list[AuditEvent] = []
    runtime = _runtime()
    runtime._dome = _StubDome(errored=True)
    runtime._audit._sink = events.append

    runtime.emit_heartbeat()

    assert events[-1].attributes["guards_constructed"] is True
    assert events[-1].attributes["detector_reachable"] is False
