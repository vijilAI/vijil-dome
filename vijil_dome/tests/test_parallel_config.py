import pytest
import asyncio
from vijil_dome import Dome, create_dome_config
from vijil_dome.guardrails import Guard, Guardrail
from vijil_dome.detectors import DetectionMethod, DetectionResult


class MockSlowDetector(DetectionMethod):
    """A mock detector that takes time to complete and does not trigger."""

    def __init__(self, delay: float = 0.5, name: str = "SlowDetector"):
        self.delay = delay
        self._name = name
        self.was_executed = False
        self.was_cancelled = False

    async def detect(self, query_string: str) -> DetectionResult:
        try:
            self.was_executed = True
            await asyncio.sleep(self.delay)
            return (False, {"response_string": query_string})
        except asyncio.CancelledError:
            self.was_cancelled = True
            raise


class MockTriggeringDetector(DetectionMethod):
    """A mock detector that triggers immediately."""

    def __init__(self, delay: float = 0.0, name: str = "TriggeringDetector"):
        self.delay = delay
        self._name = name
        self.was_executed = False

    async def detect(self, query_string: str) -> DetectionResult:
        self.was_executed = True
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        return (True, {"response_string": "Blocked by mock detector"})


PARALLEL_CONFIG = {
    "input-guards": [
        "security-input-guard",
        "moderation-input-guard",
        "privacy-input-guard",
    ],
    "output-guards": ["moderation-output-guard"],
    "input-early-exit": False,
    "output-early-exit": False,
    "input-run-parallel": True,  # ✅ Run input guards in parallel (correct attribute name)
    "output-run-parallel": True,  # ✅ Run output guards in parallel
    "security-input-guard": {
        "type": "security",
        "early-exit": False,
        "run-parallel": True,  # ✅ Run detectors in this guard in parallel (uses thread pool)
        "methods": ["prompt-injection-mbert", "encoding-heuristics"],
    },
    "moderation-input-guard": {
        "type": "moderation",
        "early-exit": False,
        "run-parallel": True,  # ✅ Run detectors in parallel
        "methods": ["moderation-flashtext"],
    },
    "moderation-output-guard": {
        "type": "moderation",
        "early-exit": False,
        "run-parallel": True,  # ✅ Run detectors in parallel (DeBERTa needs thread pool)
        "methods": ["moderation-deberta", "moderation-flashtext"],
    },
    "privacy-input-guard": {
        "type": "privacy",
        "run-parallel": True,  # ✅ Run detectors in parallel (Presidio needs thread pool)
        "methods": ["privacy-presidio"],
    },
}


@pytest.mark.asyncio
async def test_dome_parallel_config():
    dome = Dome(dome_config=create_dome_config(PARALLEL_CONFIG))
    input_str = (
        "this is an input prompt that gets run through a bunch of detectors in parallel"
    )
    await dome.async_guard_input(input_str)
    output_str = "this is an output prompt that gets run through a bunch of detectors in parallel"
    await dome.async_guard_output(output_str)


@pytest.mark.asyncio
async def test_guard_parallel_early_exit():
    """Test that early exit works correctly in parallel Guard execution.

    This test verifies that when a detector triggers in parallel mode with
    early_exit=True, pending detectors are cancelled and the guard returns
    immediately without crashing.
    """
    # Create detectors: one that triggers quickly, one that's slow
    triggering_detector = MockTriggeringDetector(delay=0.0)
    slow_detector = MockSlowDetector(delay=1.0)

    # Create a guard with early_exit=True and run_in_parallel=True
    guard = Guard(
        guard_name="test-guard",
        detector_list=[slow_detector, triggering_detector],
        early_exit=True,
        run_in_parallel=True,
    )

    # Run the guard
    import concurrent.futures

    executor = concurrent.futures.ThreadPoolExecutor()
    result = await guard.parallel_guard("test query", executor)

    # Verify the guard was triggered
    assert result.triggered is True
    assert "Blocked by mock detector" in result.response

    # The triggering detector should have been executed
    assert triggering_detector.was_executed is True


@pytest.mark.asyncio
async def test_guard_parallel_early_exit_cancels_pending():
    """Test that pending tasks are cancelled when early exit triggers."""
    # Create detectors: triggering one completes first, slow one should be cancelled
    triggering_detector = MockTriggeringDetector(delay=0.05)
    slow_detector = MockSlowDetector(delay=2.0)

    guard = Guard(
        guard_name="test-guard",
        detector_list=[slow_detector, triggering_detector],
        early_exit=True,
        run_in_parallel=True,
    )

    import concurrent.futures

    executor = concurrent.futures.ThreadPoolExecutor()

    # Time the execution - it should be fast due to early exit
    import time

    start = time.time()
    result = await guard.parallel_guard("test query", executor)
    elapsed = time.time() - start

    # Should complete much faster than the slow detector's delay
    assert elapsed < 1.0
    assert result.triggered is True


@pytest.mark.asyncio
async def test_guardrail_parallel_early_exit():
    """Test that early exit works correctly in parallel Guardrail execution.

    This test verifies that when a guard triggers in parallel mode with
    early_exit=True, pending guards are cancelled and the guardrail returns
    immediately without crashing.
    """
    # Create two guards: one that triggers, one that's slow
    triggering_detector = MockTriggeringDetector(delay=0.0)
    slow_detector = MockSlowDetector(delay=1.0)

    triggering_guard = Guard(
        guard_name="triggering-guard",
        detector_list=[triggering_detector],
        early_exit=True,
        run_in_parallel=False,
    )

    slow_guard = Guard(
        guard_name="slow-guard",
        detector_list=[slow_detector],
        early_exit=True,
        run_in_parallel=False,
    )

    # Create guardrail with early_exit=True and run_in_parallel=True
    guardrail = Guardrail(
        level="input",
        guard_list=[slow_guard, triggering_guard],
        early_exit=True,
        run_in_parallel=True,
    )

    result = await guardrail.async_scan("test query")

    # Verify the guardrail was flagged
    assert result.flagged is True
    # Check for the new default blocked message pattern (DOME-44)
    # which uses LLM-safe refusal phrases recognized by Diamond
    assert "I'm not able to" in result.guardrail_response_message


@pytest.mark.asyncio
async def test_guardrail_parallel_early_exit_cancels_pending():
    """Test that pending guard tasks are cancelled when early exit triggers."""
    triggering_detector = MockTriggeringDetector(delay=0.05)
    slow_detector = MockSlowDetector(delay=2.0)

    triggering_guard = Guard(
        guard_name="triggering-guard",
        detector_list=[triggering_detector],
        early_exit=True,
        run_in_parallel=False,
    )

    slow_guard = Guard(
        guard_name="slow-guard",
        detector_list=[slow_detector],
        early_exit=True,
        run_in_parallel=False,
    )

    guardrail = Guardrail(
        level="output",
        guard_list=[slow_guard, triggering_guard],
        early_exit=True,
        run_in_parallel=True,
    )

    import time

    start = time.time()
    result = await guardrail.async_scan("test query")
    elapsed = time.time() - start

    # Should complete much faster than the slow guard's delay
    assert elapsed < 1.0
    assert result.flagged is True


@pytest.mark.asyncio
async def test_guard_parallel_no_early_exit():
    """Test that without early exit, all detectors run to completion."""
    triggering_detector = MockTriggeringDetector(delay=0.0)
    slow_detector = MockSlowDetector(delay=0.1)

    guard = Guard(
        guard_name="test-guard",
        detector_list=[slow_detector, triggering_detector],
        early_exit=False,
        run_in_parallel=True,
    )

    import concurrent.futures

    executor = concurrent.futures.ThreadPoolExecutor()
    result = await guard.parallel_guard("test query", executor)

    # Both detectors should have been executed
    assert triggering_detector.was_executed is True
    assert slow_detector.was_executed is True
    assert result.triggered is True


@pytest.mark.asyncio
async def test_guardrail_parallel_no_early_exit():
    """Test that without early exit, all guards run to completion."""
    triggering_detector = MockTriggeringDetector(delay=0.0)
    slow_detector = MockSlowDetector(delay=0.1)

    triggering_guard = Guard(
        guard_name="triggering-guard",
        detector_list=[triggering_detector],
        early_exit=True,
        run_in_parallel=False,
    )

    slow_guard = Guard(
        guard_name="slow-guard",
        detector_list=[slow_detector],
        early_exit=True,
        run_in_parallel=False,
    )

    guardrail = Guardrail(
        level="input",
        guard_list=[slow_guard, triggering_guard],
        early_exit=False,
        run_in_parallel=True,
    )

    result = await guardrail.async_scan("test query")

    # Both guards should have been executed
    assert triggering_detector.was_executed is True
    assert slow_detector.was_executed is True
    assert result.flagged is True

