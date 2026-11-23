"""Remote detectors judge the score against the model's own threshold.

Each Vijil detector model ships its operating point in a vijil_detector.json
manifest, and the inference servers return it as ``details["threshold"]``.
Until VIJIL-1670 the config parser wrote 0.5 into every remote detector, so
that threshold never applied: ``prompt-harmfulness-fast``, swept to flag at
0.95, flagged at 0.5 when remote and at 0.95 in-process.

The rule, shared with dome-go (VIJIL-1666):

- a threshold the operator configured wins;
- otherwise ``details["threshold"]`` applies;
- a ``details["threshold"]`` that is present but invalid is an error, never a
  silent fall back to a default;
- a server that sends none (one predating manifests) keeps 0.5.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from vijil_dome.detectors.detection_api import DetectorResult
from vijil_dome.detectors.methods import remote_method
from vijil_dome.detectors.methods.remote_method import RemoteDetectionMethod
from vijil_dome.types import DomePayload

DETECTOR = "prompt-harmfulness-fast"


class _FakeDispatcher:
    def __init__(self, score: float, details: dict[str, Any]) -> None:
        self.score = score
        self.details = details
        self.configs: list[dict[str, Any]] = []

    async def detect(self, payload: DomePayload, detectors: list[Any]) -> list[DetectorResult]:
        self.configs.extend(inv.config for inv in detectors)
        return [DetectorResult(detector_name=DETECTOR, score=self.score, details=self.details)]


def _run(method: RemoteDetectionMethod, dispatcher: _FakeDispatcher,
         monkeypatch: pytest.MonkeyPatch) -> tuple[bool, dict[str, Any]]:
    monkeypatch.setattr(remote_method, "_get_dispatcher", lambda: dispatcher)
    return asyncio.run(method.detect(DomePayload(text="some text")))


@pytest.mark.parametrize(("score", "flagged"), [(0.90, False), (0.95, True), (0.97, True)])
def test_server_threshold_applies_when_none_is_configured(
    score: float, flagged: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    hit, data = _run(RemoteDetectionMethod(DETECTOR), _FakeDispatcher(score, {"threshold": 0.95}),
                     monkeypatch)
    assert "error" not in data
    assert hit is flagged


def test_configured_threshold_overrides_the_servers(monkeypatch: pytest.MonkeyPatch) -> None:
    hit, _ = _run(RemoteDetectionMethod(DETECTOR, threshold=0.5),
                  _FakeDispatcher(0.90, {"threshold": 0.95}), monkeypatch)
    assert hit is True


@pytest.mark.parametrize("bad", ["0.95", 1.5, 0.0, -0.1, True, float("nan"), None])
def test_invalid_server_threshold_raises_never_a_default(
    bad: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A raise, not a soft error: detect_with_time records it as label="error",
    # so it reaches errored_methods under every policy (see the next test).
    with pytest.raises(ValueError, match="threshold"):
        _run(RemoteDetectionMethod(DETECTOR), _FakeDispatcher(0.99, {"threshold": bad}),
             monkeypatch)


def test_invalid_server_threshold_is_recorded_as_an_errored_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(remote_method, "_get_dispatcher",
                        lambda: _FakeDispatcher(0.99, {"threshold": "0.95"}))
    method = RemoteDetectionMethod(DETECTOR)
    timed = asyncio.run(method.detect_with_time(DomePayload(text="some text")))
    assert timed.hit is False
    assert timed.result.get("label") == "error"


def test_threshold_of_exactly_one_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    hit, data = _run(RemoteDetectionMethod(DETECTOR), _FakeDispatcher(1.0, {"threshold": 1}),
                     monkeypatch)
    assert "error" not in data
    assert hit is True


def test_server_without_a_threshold_keeps_the_legacy_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hit, data = _run(RemoteDetectionMethod(DETECTOR), _FakeDispatcher(0.6, {}), monkeypatch)
    assert "error" not in data
    assert hit is True


def test_threshold_is_never_sent_to_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    dispatcher = _FakeDispatcher(0.1, {"threshold": 0.95})
    _run(RemoteDetectionMethod(DETECTOR, threshold=0.7), dispatcher, monkeypatch)
    assert all("threshold" not in config for config in dispatcher.configs)


def test_config_parser_does_not_inject_a_default_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vijil_dome.guardrails import config_parser

    monkeypatch.setenv("DOME_INFERENCE_URL", "http://localhost:8000")
    unset = config_parser.create_detector_for_guard(DETECTOR, "moderation", {})
    configured = config_parser.create_detector_for_guard(DETECTOR, "moderation", {"threshold": 0.7})
    assert isinstance(unset, RemoteDetectionMethod)
    assert isinstance(configured, RemoteDetectionMethod)
    assert unset._threshold is None
    assert configured._threshold == 0.7


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, True, float("nan")])
def test_configured_threshold_outside_zero_to_one_is_refused(bad: Any) -> None:
    # 0 would flag everything and anything above 1 would flag nothing; both
    # silently disable the detector, so the config is refused instead.
    with pytest.raises(ValueError, match="threshold"):
        RemoteDetectionMethod(DETECTOR, threshold=bad)


def test_config_parser_refuses_an_explicit_null_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    from vijil_dome.guardrails import config_parser

    monkeypatch.setenv("DOME_INFERENCE_URL", "http://localhost:8000")
    with pytest.raises(ValueError, match="null"):
        config_parser.create_detector_for_guard(DETECTOR, "moderation", {"threshold": None})

