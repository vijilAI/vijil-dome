"""Tests for GuardResult and related models."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vijil_dome.trust.guard import DetectorTrace, GuardResult, GuardTrace

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detector(name: str = "toxicity", hit: bool = False, score: float = 0.1) -> DetectorTrace:
    return DetectorTrace(
        detector_name=name,
        hit=hit,
        score=score,
        exec_time_ms=12.5,
    )


def _guard(name: str = "input_guard", detectors: list[DetectorTrace] | None = None) -> GuardTrace:
    return GuardTrace(
        guard_name=name,
        detectors=detectors or [_detector()],
    )


def _make_scan_result(
    *,
    flagged: bool = False,
    enforced: bool = False,
    detection_score: float = 0.0,
    response_string: str | None = None,
    exec_time: float = 0.05,
    trace: dict | None = None,
) -> SimpleNamespace:
    """Return a duck-typed object matching Dome's ScanResult shape."""
    return SimpleNamespace(
        flagged=flagged,
        enforced=enforced,
        detection_score=detection_score,
        response_string=response_string,
        exec_time=exec_time,
        trace=trace or {},
    )


# ---------------------------------------------------------------------------
# 1. Unflagged result construction
# ---------------------------------------------------------------------------


def test_unflagged_result() -> None:
    result = GuardResult(
        flagged=False,
        enforced=False,
        score=0.05,
        guarded_response=None,
        exec_time_ms=48.3,
        trace=[],
    )

    assert result.flagged is False
    assert result.enforced is False
    assert result.score == pytest.approx(0.05)
    assert result.guarded_response is None
    assert result.exec_time_ms == pytest.approx(48.3)
    assert result.trace == []


# ---------------------------------------------------------------------------
# 2. Flagged + enforced result with nested trace
# ---------------------------------------------------------------------------


def test_flagged_enforced_result() -> None:
    det = _detector(name="prompt_injection", hit=True, score=0.92)
    guard = _guard(name="input_guard", detectors=[det])

    result = GuardResult(
        flagged=True,
        enforced=True,
        score=0.92,
        guarded_response="Request blocked.",
        exec_time_ms=75.0,
        trace=[guard],
    )

    assert result.flagged is True
    assert result.enforced is True
    assert result.score == pytest.approx(0.92)
    assert result.guarded_response == "Request blocked."
    assert len(result.trace) == 1

    g = result.trace[0]
    assert g.guard_name == "input_guard"
    assert len(g.detectors) == 1

    d = g.detectors[0]
    assert d.detector_name == "prompt_injection"
    assert d.hit is True
    assert d.score == pytest.approx(0.92)


# ---------------------------------------------------------------------------
# 3. from_scan_result — unflagged (no response string)
# ---------------------------------------------------------------------------


def test_from_scan_result_unflagged() -> None:
    scan = _make_scan_result(
        flagged=False,
        enforced=False,
        detection_score=0.03,
        response_string="should be ignored when not flagged",
        exec_time=0.042,
        trace={},
    )

    result = GuardResult.from_scan_result(scan)

    assert result.flagged is False
    assert result.enforced is False
    assert result.score == pytest.approx(0.03)
    # guarded_response omitted when not flagged
    assert result.guarded_response is None
    assert result.exec_time_ms == pytest.approx(42.0)
    assert result.trace == []


# ---------------------------------------------------------------------------
# 4. from_scan_result — flagged with nested trace
# ---------------------------------------------------------------------------


def test_from_scan_result_flagged_with_trace() -> None:
    scan = _make_scan_result(
        flagged=True,
        enforced=True,
        detection_score=0.88,
        response_string="Blocked.",
        exec_time=0.110,
        trace={
            "input_guard": {
                "prompt_injection": SimpleNamespace(
                    hit=True,
                    result={"score": 0.88},
                    exec_time=0.110,
                )
            }
        },
    )

    result = GuardResult.from_scan_result(scan)

    assert result.flagged is True
    assert result.enforced is True
    assert result.score == pytest.approx(0.88)
    assert result.guarded_response == "Blocked."
    assert result.exec_time_ms == pytest.approx(110.0)

    assert len(result.trace) == 1
    g = result.trace[0]
    assert g.guard_name == "input_guard"
    assert len(g.detectors) == 1

    d = g.detectors[0]
    assert d.detector_name == "prompt_injection"
    assert d.hit is True
    assert d.score == pytest.approx(0.88)
    assert d.exec_time_ms == pytest.approx(110.0)


# ---------------------------------------------------------------------------
# 5. from_scan_result — empty trace dict
# ---------------------------------------------------------------------------


def test_from_scan_result_empty_trace() -> None:
    scan = _make_scan_result(
        flagged=False,
        enforced=False,
        detection_score=0.0,
        exec_time=0.010,
        trace={},
    )

    result = GuardResult.from_scan_result(scan)

    assert result.trace == []
    assert result.exec_time_ms == pytest.approx(10.0)
