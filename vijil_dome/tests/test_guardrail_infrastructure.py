"""Tests for guardrail infrastructure fixes: BC-17 (PII logging), BC-2 (timeout), BC-3 (executor lifecycle)."""

from __future__ import annotations

import asyncio
import logging
import pathlib

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


# ---------------------------------------------------------------------------
# Detector contract: detect_with_time catches exceptions from detect()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_detect_with_time_returns_error_label_on_exception() -> None:
    """Base class detect_with_time must return label='error' when detect() raises."""
    detector = MockExceptionDetector()
    result = await detector.detect_with_time("test input")

    assert result.hit is False
    assert result.result["label"] == "error"
    assert "model failed to load" in result.result["error"]
    assert result.result["response_string"] == "test input"


@pytest.mark.asyncio
async def test_detect_batch_with_time_returns_error_labels_on_exception() -> None:
    """Base class detect_batch_with_time must return label='error' for all items."""
    detector = MockExceptionDetector()
    result = await detector.detect_batch_with_time(["input one", "input two"])

    assert len(result.results) == 2
    for r in result.results:
        assert r.hit is False
        assert r.result["label"] == "error"
        assert "model failed to load" in r.result["error"]


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
    assert "test_guard:MockErrorDetector" in result.errored_methods


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
    assert "test_guard:MockErrorDetector" in result.errored_methods


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
    assert "test_guard:MockExceptionDetector" in result.errored_methods


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
    assert "test_guard:MockHangingDetector" in result.errored_methods


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


# ---------------------------------------------------------------------------
# BC-9: Score clamping
# ---------------------------------------------------------------------------


class OverScoreDetector(DetectionMethod):
    async def detect(self, dome_input: DomePayload) -> DetectionResult:
        return (True, {"response_string": "blocked", "score": 1.5})


class NegativeScoreDetector(DetectionMethod):
    async def detect(self, dome_input: DomePayload) -> DetectionResult:
        return (True, {"response_string": "blocked", "score": -0.5})


@pytest.mark.asyncio
async def test_score_clamped_to_one() -> None:
    guard = Guard(
        guard_name="test",
        detector_list=[OverScoreDetector()],
        run_in_parallel=False,
    )
    result = await guard.sequential_guard("test input")
    assert result.detection_score <= 1.0


@pytest.mark.asyncio
async def test_negative_score_clamped_to_zero() -> None:
    guard = Guard(
        guard_name="test",
        detector_list=[NegativeScoreDetector()],
        run_in_parallel=False,
    )
    result = await guard.sequential_guard("test input")
    assert result.detection_score >= 0.0


# ---------------------------------------------------------------------------
# DOME-163 B1: Fail-CLOSED on detector-unreachable
#
# Today the legacy content-guard path fails OPEN: a detector that errors,
# times out, or raises (e.g. the EKS inference server is unreachable)
# synthesizes hit=False, so the guard treats the error as a clean pass.
# The on_error="fail_closed" option must make an errored detector BLOCK
# in enforce mode, while the default on_error="fail_open" preserves the
# back-compatible silent-pass behavior.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_guard_fail_closed_sequential_blocks_on_error() -> None:
    """A detector that errors must BLOCK when the guard is fail_closed."""
    guard = Guard(
        guard_name="test_guard",
        detector_list=[MockErrorDetector()],
        run_in_parallel=False,
        on_error="fail_closed",
    )

    result = await guard.sequential_guard("test input")

    assert result.triggered is True
    assert "MockErrorDetector" in result.errored_methods


@pytest.mark.asyncio
async def test_guard_fail_closed_parallel_blocks_on_error() -> None:
    """Same fail-closed semantics on the parallel guard path."""
    guard = Guard(
        guard_name="test_guard",
        detector_list=[MockErrorDetector()],
        run_in_parallel=True,
        on_error="fail_closed",
    )
    guardrail = Guardrail(level="input", guard_list=[guard], run_in_parallel=False)

    result = await guardrail.async_scan("test input")

    assert result.flagged is True
    assert "test_guard:MockErrorDetector" in result.errored_methods


@pytest.mark.asyncio
async def test_guard_fail_closed_blocks_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A detector timeout (slow-loris the inference server) must BLOCK.

    The detector timeout is monkeypatched to a few ms so the test exercises the
    timeout path without forcing the real 5s DETECTOR_TIMEOUT_SECONDS wait in CI.
    """
    monkeypatch.setattr("vijil_dome.guardrails.DETECTOR_TIMEOUT_SECONDS", 0.05)
    guard = Guard(
        guard_name="test_guard",
        detector_list=[MockHangingDetector(delay=1.0)],
        run_in_parallel=False,
        on_error="fail_closed",
    )

    result = await guard.sequential_guard("test input")

    assert result.triggered is True
    assert "MockHangingDetector" in result.errored_methods


@pytest.mark.asyncio
async def test_guard_fail_closed_blocks_on_exception() -> None:
    """A detector that raises (e.g. ConnectError) must BLOCK when fail_closed."""
    guard = Guard(
        guard_name="test_guard",
        detector_list=[MockExceptionDetector()],
        run_in_parallel=False,
        on_error="fail_closed",
    )

    result = await guard.sequential_guard("test input")

    assert result.triggered is True
    assert "MockExceptionDetector" in result.errored_methods


@pytest.mark.asyncio
async def test_guard_fail_open_default_preserves_pass_on_error() -> None:
    """Back-compat: the default (fail_open) still passes an errored detector."""
    guard = Guard(
        guard_name="test_guard",
        detector_list=[MockErrorDetector()],
        run_in_parallel=False,
    )

    result = await guard.sequential_guard("test input")

    assert result.triggered is False
    assert "MockErrorDetector" in result.errored_methods


@pytest.mark.asyncio
async def test_guard_fail_closed_clean_detector_passes() -> None:
    """fail_closed must NOT block a clean (non-errored) detector."""
    guard = Guard(
        guard_name="test_guard",
        detector_list=[MockCleanDetector()],
        run_in_parallel=False,
        on_error="fail_closed",
    )

    result = await guard.sequential_guard("test input")

    assert result.triggered is False
    assert result.errored_methods == []


@pytest.mark.asyncio
async def test_guard_fail_closed_real_hit_takes_precedence() -> None:
    """A genuine hit alongside an error stays a hit (not laundered to error-only)."""
    guard = Guard(
        guard_name="test_guard",
        detector_list=[MockTriggeringDetector(), MockErrorDetector()],
        run_in_parallel=False,
        early_exit=False,
        on_error="fail_closed",
    )

    result = await guard.sequential_guard("test input")

    assert result.triggered is True
    assert "MockTriggeringDetector" in result.triggered_methods
    assert "MockErrorDetector" in result.errored_methods


@pytest.mark.asyncio
async def test_guardrail_fail_closed_blocks_when_a_guard_errors() -> None:
    """A clean guard plus an errored guard must BLOCK at the guardrail level."""
    clean_guard = Guard(
        guard_name="clean",
        detector_list=[MockCleanDetector()],
        run_in_parallel=False,
        on_error="fail_closed",
    )
    error_guard = Guard(
        guard_name="errored",
        detector_list=[MockErrorDetector()],
        run_in_parallel=False,
        on_error="fail_closed",
    )
    guardrail = Guardrail(
        level="input",
        guard_list=[clean_guard, error_guard],
        run_in_parallel=False,
        early_exit=False,
    )

    result = await guardrail.async_scan("test input")

    assert result.flagged is True
    assert "errored:MockErrorDetector" in result.errored_methods


@pytest.mark.asyncio
async def test_guardrail_fail_closed_blocks_when_guard_task_raises() -> None:
    """If a guard task itself raises (parallel path), fail_closed must BLOCK
    rather than silently dropping the guard and passing."""

    class ExplodingGuard(Guard):
        async def async_scan(self, *args, **kwargs):  # type: ignore[override]
            raise RuntimeError("guard exploded")

    exploding = ExplodingGuard(
        guard_name="boom",
        detector_list=[MockCleanDetector()],
        run_in_parallel=False,
    )
    guardrail = Guardrail(
        level="input",
        guard_list=[exploding],
        run_in_parallel=True,
        on_error="fail_closed",
    )

    result = await guardrail.async_scan("test input")

    assert result.flagged is True
    # The crashed guard must be named in errored_methods, not vanish (review #247).
    assert "boom:<guard-task-raised>" in result.errored_methods


@pytest.mark.asyncio
async def test_guardrail_fail_open_default_passes_when_guard_task_raises() -> None:
    """Back-compat: default fail_open still drops a raising guard and passes."""

    class ExplodingGuard(Guard):
        async def async_scan(self, *args, **kwargs):  # type: ignore[override]
            raise RuntimeError("guard exploded")

    exploding = ExplodingGuard(
        guard_name="boom",
        detector_list=[MockCleanDetector()],
        run_in_parallel=False,
    )
    guardrail = Guardrail(
        level="input",
        guard_list=[exploding],
        run_in_parallel=True,
    )

    result = await guardrail.async_scan("test input")

    assert result.flagged is False


def test_config_parser_wires_on_error_to_guard_and_guardrail() -> None:
    """on_error from the config dict must reach both the Guardrail and its Guards."""
    from vijil_dome.guardrails.config_parser import create_guardrail

    config = {
        "input-guards": ["my-guard"],
        "input-on-error": "fail_closed",
        "my-guard": {
            "type": "security",
            "methods": ["encoding-heuristics"],
        },
    }
    # encoding-heuristics is a pure-heuristic local detector (no model
    # weights, no network) — we only need construction, not a scan.
    guardrail = create_guardrail("input", config)

    assert guardrail.on_error == "fail_closed"
    assert all(g.on_error == "fail_closed" for g in guardrail.guard_list)


def test_create_guardrail_default_on_error_applies_when_config_omits_it() -> None:
    """DOME-167: enforce mode passes default_on_error="fail_closed"; it applies only when the
    config omits on-error. An explicit config on-error still wins, and with no default the
    back-compatible fail_open is preserved."""
    from vijil_dome.guardrails.config_parser import create_guardrail

    config = {
        "input-guards": ["my-guard"],
        "my-guard": {"type": "security", "methods": ["encoding-heuristics"]},
    }

    # enforce mode supplies fail_closed → applies, propagates to child guards
    enforced = create_guardrail("input", config, default_on_error="fail_closed")
    assert enforced.on_error == "fail_closed"
    assert all(g.on_error == "fail_closed" for g in enforced.guard_list)

    # back-compat: no default supplied → fail_open
    assert create_guardrail("input", config).on_error == "fail_open"

    # explicit config on-error beats the enforce default
    explicit = {**config, "input-on-error": "fail_open"}
    assert create_guardrail("input", explicit, default_on_error="fail_closed").on_error == "fail_open"


def test_config_parser_invalid_on_error_raises() -> None:
    """A typo in on-error must fail loud, not silently default to fail_open."""
    from vijil_dome.guardrails.config_parser import create_guardrail

    config = {
        "input-guards": ["my-guard"],
        "input-on-error": "fail-closed",  # wrong: hyphen instead of underscore
        "my-guard": {
            "type": "security",
            "methods": ["encoding-heuristics"],
        },
    }
    with pytest.raises(ValueError, match="Invalid on-error value"):
        create_guardrail("input", config)


def test_config_parser_per_guard_on_error_overrides_guardrail(
    tmp_path: pathlib.Path,
) -> None:
    """A per-guard on-error wins over the guardrail-level default, and the
    TOML round-trip preserves on-error end to end."""
    from vijil_dome.guardrails.config_parser import convert_toml_to_guardrails

    toml_text = (
        "[guardrail]\n"
        'input-guards = ["strict", "lenient"]\n'
        'input-on-error = "fail_closed"\n'
        "\n"
        "[strict]\n"
        'type = "security"\n'
        'methods = ["encoding-heuristics"]\n'
        "\n"
        "[lenient]\n"
        'type = "security"\n'
        'on-error = "fail_open"\n'
        'methods = ["encoding-heuristics"]\n'
    )
    toml_path = tmp_path / "config.toml"
    toml_path.write_text(toml_text)

    input_guardrail, output_guardrail, *_ = convert_toml_to_guardrails(str(toml_path))

    assert input_guardrail.on_error == "fail_closed"
    by_name = {g.guard_name: g.on_error for g in input_guardrail.guard_list}
    assert by_name["strict"] == "fail_closed"  # inherits guardrail default
    assert by_name["lenient"] == "fail_open"  # per-guard override wins
    assert output_guardrail.on_error == "fail_open"  # untouched default
