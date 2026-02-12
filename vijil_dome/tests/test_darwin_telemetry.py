"""Tests for Darwin-compatible telemetry instrumentation.

Verifies that instrument_dome() emits Darwin span attributes
(detection.label, detection.score, detection.method, team.id, agent.id)
alongside the existing split metrics.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from vijil_dome.guardrails import GuardrailResult, GuardResult
from vijil_dome.detectors import DetectionTimingResult
from vijil_dome.integrations.vijil.telemetry import (
    _extract_detection_score,
    _extract_detection_method,
    _set_darwin_span_attributes,
)


# --- Fixtures: realistic GuardrailResult objects ---


def _make_detection(hit: bool, score: float) -> DetectionTimingResult:
    return DetectionTimingResult(
        hit=hit,
        result={"score": score, "label": "flagged" if hit else "clean"},
        exec_time=0.05,
    )


def _make_guard_result(
    triggered: bool, detections: dict[str, DetectionTimingResult]
) -> GuardResult:
    return GuardResult(
        triggered=triggered,
        details=detections,
        exec_time=0.1,
        response="blocked" if triggered else "",
    )


def _make_guardrail_result(
    flagged: bool, guard_details: dict[str, GuardResult]
) -> GuardrailResult:
    return GuardrailResult(
        flagged=flagged,
        guard_exec_details=guard_details,
        exec_time=0.2,
        guardrail_response_message="blocked" if flagged else "",
    )


@pytest.fixture
def flagged_guardrail_result() -> GuardrailResult:
    """A GuardrailResult where prompt-injection guard triggered with score 0.95."""
    return _make_guardrail_result(
        flagged=True,
        guard_details={
            "prompt-injection": _make_guard_result(
                triggered=True,
                detections={
                    "deberta-v3": _make_detection(hit=True, score=0.95),
                    "mbert": _make_detection(hit=False, score=0.15),
                },
            ),
            "toxicity": _make_guard_result(
                triggered=False,
                detections={
                    "deberta-toxicity": _make_detection(hit=False, score=0.02),
                },
            ),
        },
    )


@pytest.fixture
def clean_guardrail_result() -> GuardrailResult:
    """A GuardrailResult where nothing was flagged."""
    return _make_guardrail_result(
        flagged=False,
        guard_details={
            "prompt-injection": _make_guard_result(
                triggered=False,
                detections={
                    "deberta-v3": _make_detection(hit=False, score=0.1),
                },
            ),
        },
    )


# --- Tests: _extract_detection_score ---


class TestExtractDetectionScore:
    def test_returns_max_score_from_triggered_guard(
        self, flagged_guardrail_result: GuardrailResult
    ):
        score = _extract_detection_score(flagged_guardrail_result)
        assert score == 0.95

    def test_returns_zero_when_nothing_flagged(
        self, clean_guardrail_result: GuardrailResult
    ):
        score = _extract_detection_score(clean_guardrail_result)
        assert score == 0.0

    def test_handles_guard_result_directly(self):
        guard = _make_guard_result(
            triggered=True,
            detections={"det1": _make_detection(hit=True, score=0.7)},
        )
        score = _extract_detection_score(guard)
        assert score == 0.7

    def test_handles_missing_score_key(self):
        guard = _make_guard_result(
            triggered=True,
            detections={
                "det1": DetectionTimingResult(
                    hit=True, result={"label": "flagged"}, exec_time=0.01
                )
            },
        )
        score = _extract_detection_score(guard)
        assert score == 0.0


# --- Tests: _extract_detection_method ---


class TestExtractDetectionMethod:
    def test_returns_first_triggered_guard_name(
        self, flagged_guardrail_result: GuardrailResult
    ):
        method = _extract_detection_method(flagged_guardrail_result)
        assert method == "prompt-injection"

    def test_returns_unknown_when_nothing_triggered(
        self, clean_guardrail_result: GuardrailResult
    ):
        method = _extract_detection_method(clean_guardrail_result)
        assert method == "unknown"

    def test_returns_detector_name_for_guard_result(self):
        guard = _make_guard_result(
            triggered=True,
            detections={
                "not-hit": _make_detection(hit=False, score=0.1),
                "the-hit": _make_detection(hit=True, score=0.9),
            },
        )
        method = _extract_detection_method(guard)
        assert method == "the-hit"


# --- Tests: _set_darwin_span_attributes ---


class TestSetDarwinSpanAttributes:
    def test_sets_all_attributes_on_flagged_result(
        self, flagged_guardrail_result: GuardrailResult
    ):
        span = MagicMock()
        _set_darwin_span_attributes(
            span,
            flagged_guardrail_result,
            agent_id="agent-123",
            team_id="team-456",
        )

        span.set_attribute.assert_any_call("team.id", "team-456")
        span.set_attribute.assert_any_call("agent.id", "agent-123")
        span.set_attribute.assert_any_call("detection.label", "flagged")
        span.set_attribute.assert_any_call("detection.score", 0.95)
        span.set_attribute.assert_any_call("detection.method", "prompt-injection")

    def test_sets_clean_label_on_unflagged_result(
        self, clean_guardrail_result: GuardrailResult
    ):
        span = MagicMock()
        _set_darwin_span_attributes(span, clean_guardrail_result)

        span.set_attribute.assert_any_call("detection.label", "clean")
        span.set_attribute.assert_any_call("detection.score", 0.0)
        span.set_attribute.assert_any_call("detection.method", "unknown")

    def test_skips_team_and_agent_when_not_provided(
        self, flagged_guardrail_result: GuardrailResult
    ):
        span = MagicMock()
        _set_darwin_span_attributes(span, flagged_guardrail_result)

        # Should NOT have called set_attribute with team.id or agent.id
        call_keys = [call.args[0] for call in span.set_attribute.call_args_list]
        assert "team.id" not in call_keys
        assert "agent.id" not in call_keys


# --- Tests: _add_darwin_detection_spans integration ---


class TestAddDarwinDetectionSpans:
    """Test that _add_darwin_detection_spans wraps scan correctly."""

    def test_sync_scan_emits_darwin_span(self, flagged_guardrail_result):
        from vijil_dome.integrations.instrumentation.otel_instrumentation import (
            _add_darwin_detection_spans,
        )

        # Create a mock guardrail with scan methods
        guardrail = MagicMock()
        guardrail.scan = MagicMock(return_value=flagged_guardrail_result)

        # Create a mock tracer that returns a context-managed span
        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)

        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span

        # Apply the wrapper
        _add_darwin_detection_spans(guardrail, mock_tracer, "dome-input")

        # Call the wrapped scan
        result = guardrail.scan("test input", agent_id="a1", team_id="t1")

        # Verify span was created
        mock_tracer.start_as_current_span.assert_called_once_with("dome-detection")

        # Verify Darwin attributes were set
        mock_span.set_attribute.assert_any_call("dome.guardrail", "dome-input")
        mock_span.set_attribute.assert_any_call("detection.label", "flagged")
        mock_span.set_attribute.assert_any_call("detection.score", 0.95)
        mock_span.set_attribute.assert_any_call("agent.id", "a1")
        mock_span.set_attribute.assert_any_call("team.id", "t1")

        # Verify the result is passed through
        assert result is flagged_guardrail_result

    @pytest.mark.asyncio
    async def test_async_scan_emits_darwin_span(self, flagged_guardrail_result):
        from vijil_dome.integrations.instrumentation.otel_instrumentation import (
            _add_darwin_detection_spans,
        )

        # Create a mock guardrail with async scan
        guardrail = MagicMock()

        async def mock_async_scan(*args, **kwargs):
            return flagged_guardrail_result

        guardrail.async_scan = mock_async_scan

        # Create a mock tracer
        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)

        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span

        # Apply the wrapper
        _add_darwin_detection_spans(guardrail, mock_tracer, "dome-output")

        # Call the wrapped async scan
        result = await guardrail.async_scan("test output", agent_id="a2", team_id="t2")

        # Verify span was created
        mock_tracer.start_as_current_span.assert_called_once_with("dome-detection")
        mock_span.set_attribute.assert_any_call("dome.guardrail", "dome-output")
        mock_span.set_attribute.assert_any_call("detection.label", "flagged")

        assert result is flagged_guardrail_result
