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
async def test_debug_log_contains_payload_with_env_flag(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VIJIL_LOG_PAYLOADS", "1")

    # Re-import to pick up the env var change
    import vijil_dome.guardrails as guardrails_mod
    monkeypatch.setattr(guardrails_mod, "_LOG_PAYLOADS", True)

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
    assert secret in debug_messages


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
    assert guardrail.executor._shutdown is True


def test_guardrail_context_manager() -> None:
    guard = Guard(guard_name="g", detector_list=[MockCleanDetector()])
    with Guardrail(level="input", guard_list=[guard]) as guardrail:
        assert guardrail.executor is not None
        executor = guardrail.executor

    assert executor._shutdown is True
