"""Tests for guardrail infrastructure fixes: BC-17 (PII logging), BC-2 (timeout), BC-3 (executor lifecycle)."""

from __future__ import annotations

import asyncio
import logging

import pytest

from vijil_dome.detectors import DetectionMethod, DetectionResult
from vijil_dome.guardrails import Guard, Guardrail
from vijil_dome.types import DomePayload


# ---------------------------------------------------------------------------
# Mock detectors
# ---------------------------------------------------------------------------


class MockCleanDetector(DetectionMethod):
    """Always returns clean (no hit)."""

    async def detect(self, dome_input: DomePayload) -> DetectionResult:
        return (False, {"response_string": dome_input.query_string})


class MockHangingDetector(DetectionMethod):
    """Sleeps for a long time to simulate a hanging detector."""

    def __init__(self, delay: float = 30.0):
        self.delay = delay

    async def detect(self, dome_input: DomePayload) -> DetectionResult:
        await asyncio.sleep(self.delay)
        return (False, {"response_string": dome_input.query_string})


# ---------------------------------------------------------------------------
# BC-17: PII logging — INFO level must not contain raw payload
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_info_log_does_not_contain_payload(caplog: pytest.LogCaptureFixture) -> None:
    guard = Guard(
        guard_name="test_guard",
        detector_list=[MockCleanDetector()],
        run_in_parallel=False,
    )
    guardrail = Guardrail(level="input", guard_list=[guard], run_in_parallel=False)

    secret = "super-secret-payload-xyz-12345"
    with caplog.at_level(logging.DEBUG, logger="vijil.dome"):
        await guardrail.async_scan(secret)

    info_messages = " ".join(r.message for r in caplog.records if r.levelno == logging.INFO)
    assert secret not in info_messages


@pytest.mark.asyncio
async def test_debug_log_is_redacted_with_env_flag(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VIJIL_LOG_PAYLOADS", "1")

    guard = Guard(
        guard_name="test_guard",
        detector_list=[MockCleanDetector()],
        run_in_parallel=False,
    )
    guardrail = Guardrail(level="input", guard_list=[guard], run_in_parallel=False)

    secret = "debug-visible-payload-abc-67890"
    with caplog.at_level(logging.DEBUG, logger="vijil.dome"):
        await guardrail.async_scan(secret)

    debug_messages = " ".join(r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG)
    assert "Scan payload logging enabled (redacted)" in debug_messages
    assert secret not in debug_messages


# ---------------------------------------------------------------------------
# BC-2: Sequential guard timeout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sequential_guard_timeout() -> None:
    hanging = MockHangingDetector(delay=30.0)
    guard = Guard(
        guard_name="timeout_guard",
        detector_list=[hanging],
        run_in_parallel=False,
    )

    result = await guard.sequential_guard("test input")

    assert result.exec_time < 6.0
    detector_name = type(hanging).__name__
    assert detector_name in result.details
    assert result.details[detector_name].result.get("error") == "Detection method timed out"
    assert result.triggered is False


@pytest.mark.asyncio
async def test_sequential_batch_guard_timeout() -> None:
    hanging = MockHangingDetector(delay=30.0)
    guard = Guard(
        guard_name="timeout_guard",
        detector_list=[hanging],
        run_in_parallel=False,
    )

    result = await guard.sequential_guard_batch(["input one", "input two"])

    assert len(result.items) == 2
    for item in result.items:
        detector_name = type(hanging).__name__
        assert detector_name in item.details
        assert item.details[detector_name].result.get("error") == "Detection method timed out"
        assert item.triggered is False


# ---------------------------------------------------------------------------
# BC-3: ThreadPoolExecutor lifecycle
# ---------------------------------------------------------------------------


def test_guardrail_close_shuts_down_executor() -> None:
    guard = Guard(guard_name="g", detector_list=[MockCleanDetector()])
    guardrail = Guardrail(level="input", guard_list=[guard])

    assert guardrail.executor is not None
    guardrail.close()
    with pytest.raises(RuntimeError):
        guardrail.executor.submit(lambda: None)


def test_guardrail_context_manager() -> None:
    guard = Guard(guard_name="g", detector_list=[MockCleanDetector()])
    with Guardrail(level="input", guard_list=[guard]) as guardrail:
        assert guardrail.executor is not None
        executor = guardrail.executor

    with pytest.raises(RuntimeError):
        executor.submit(lambda: None)


# ---------------------------------------------------------------------------
# BC-1: Detector error visibility — errored_methods surfaced in results
# ---------------------------------------------------------------------------


class MockErrorDetector(DetectionMethod):
    """Simulates a detector that errors internally and returns label='error'."""

    async def detect(self, dome_input: DomePayload) -> DetectionResult:
        return (False, {
            "label": "error",
            "error": "torch not available",
            "response_string": dome_input.query_string,
            "score": 0.0,
        })


class MockExceptionDetector(DetectionMethod):
    """Raises an unhandled exception during detection."""

    async def detect(self, dome_input: DomePayload) -> DetectionResult:
        raise RuntimeError("model failed to load")


class MockTriggeringDetector(DetectionMethod):
    """Always triggers."""

    async def detect(self, dome_input: DomePayload) -> DetectionResult:
        return (True, {"response_string": "Blocked", "score": 0.95})


@pytest.mark.asyncio
async def test_errored_detector_surfaced_in_result() -> None:
    guard = Guard(
        guard_name="test_guard",
        detector_list=[MockErrorDetector()],
        run_in_parallel=False,
    )
    guardrail = Guardrail(level="input", guard_list=[guard], run_in_parallel=False)

    result = await guardrail.async_scan("test input")

    assert result.flagged is False
    assert "MockErrorDetector" in result.errored_methods


@pytest.mark.asyncio
async def test_errored_and_triggered_coexist() -> None:
    guard = Guard(
        guard_name="test_guard",
        detector_list=[MockErrorDetector(), MockTriggeringDetector()],
        run_in_parallel=False,
        early_exit=False,
    )
    guardrail = Guardrail(level="input", guard_list=[guard], run_in_parallel=False)

    result = await guardrail.async_scan("test input")

    assert result.flagged is True
    assert "MockTriggeringDetector" in result.triggered_methods
    assert "MockErrorDetector" in result.errored_methods


@pytest.mark.asyncio
async def test_clean_scan_has_empty_errored_methods() -> None:
    guard = Guard(
        guard_name="test_guard",
        detector_list=[MockCleanDetector()],
        run_in_parallel=False,
    )
    guardrail = Guardrail(level="input", guard_list=[guard], run_in_parallel=False)

    result = await guardrail.async_scan("test input")

    assert result.flagged is False
    assert result.errored_methods == []
    assert result.triggered_methods == []


@pytest.mark.asyncio
async def test_exception_detector_surfaced_in_errored_methods() -> None:
    guard = Guard(
        guard_name="test_guard",
        detector_list=[MockExceptionDetector()],
        run_in_parallel=False,
    )
    guardrail = Guardrail(level="input", guard_list=[guard], run_in_parallel=False)

    result = await guardrail.async_scan("test input")

    assert result.flagged is False
    assert "MockExceptionDetector" in result.errored_methods


@pytest.mark.asyncio
async def test_timeout_detector_surfaced_in_errored_methods() -> None:
    hanging = MockHangingDetector(delay=30.0)
    guard = Guard(
        guard_name="test_guard",
        detector_list=[hanging],
        run_in_parallel=False,
    )
    guardrail = Guardrail(level="input", guard_list=[guard], run_in_parallel=False)

    result = await guardrail.async_scan("test input")

    assert result.flagged is False
    assert "MockHangingDetector" in result.errored_methods


@pytest.mark.asyncio
async def test_cancelled_detector_not_in_errored_methods() -> None:
    """Cancelled tasks (e.g. from early_exit) should not appear as errored."""
    guard = Guard(
        guard_name="test_guard",
        detector_list=[MockTriggeringDetector(), MockHangingDetector(delay=30.0)],
        run_in_parallel=True,
        early_exit=True,
    )
    guardrail = Guardrail(level="input", guard_list=[guard], run_in_parallel=False)

    result = await guardrail.async_scan("test input")

    assert result.flagged is True
    assert "MockHangingDetector" not in result.errored_methods
