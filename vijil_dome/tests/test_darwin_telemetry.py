"""Tests for Darwin-compatible telemetry instrumentation.

Verifies that:
- GuardResult and GuardrailResult expose detection_score and triggered_methods
- instrument_dome() emits Darwin span attributes (detection.label,
  detection.score, detection.methods, team.id, agent.id) alongside
  the existing split metrics.
"""

import importlib.util
from unittest.mock import MagicMock

import pytest

from vijil_dome.guardrails import GuardrailResult, GuardResult
from vijil_dome.detectors import DetectionTimingResult
from vijil_dome.integrations.vijil.telemetry import _set_darwin_span_attributes


# --- Fixtures: realistic GuardrailResult objects with native fields ---


def _make_detection(hit: bool, score: float) -> DetectionTimingResult:
    return DetectionTimingResult(
        hit=hit,
        result={"score": score, "label": "flagged" if hit else "clean"},
        exec_time=0.05,
    )


def _make_guard_result(
    triggered: bool,
    detections: dict[str, DetectionTimingResult],
    detection_score: float = 0.0,
    triggered_methods: list[str] | None = None,
) -> GuardResult:
    return GuardResult(
        triggered=triggered,
        details=detections,
        exec_time=0.1,
        response="blocked" if triggered else "",
        detection_score=detection_score,
        triggered_methods=triggered_methods or [],
    )


def _make_guardrail_result(
    flagged: bool,
    guard_details: dict[str, GuardResult],
    detection_score: float = 0.0,
    triggered_methods: list[str] | None = None,
) -> GuardrailResult:
    return GuardrailResult(
        flagged=flagged,
        guard_exec_details=guard_details,
        exec_time=0.2,
        guardrail_response_message="blocked" if flagged else "",
        detection_score=detection_score,
        triggered_methods=triggered_methods or [],
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
                detection_score=0.95,
                triggered_methods=["deberta-v3"],
            ),
            "toxicity": _make_guard_result(
                triggered=False,
                detections={
                    "deberta-toxicity": _make_detection(hit=False, score=0.02),
                },
            ),
        },
        detection_score=0.95,
        triggered_methods=["deberta-v3"],
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


# --- Tests: native fields on GuardResult ---


class TestGuardResultNativeFields:
    def test_detection_score_defaults_to_zero(self):
        guard = _make_guard_result(
            triggered=False,
            detections={"det": _make_detection(hit=False, score=0.1)},
        )
        assert guard.detection_score == 0.0

    def test_triggered_methods_defaults_empty(self):
        guard = _make_guard_result(
            triggered=False,
            detections={"det": _make_detection(hit=False, score=0.1)},
        )
        assert guard.triggered_methods == []

    def test_stores_score_and_methods(self):
        guard = _make_guard_result(
            triggered=True,
            detections={"deberta": _make_detection(hit=True, score=0.9)},
            detection_score=0.9,
            triggered_methods=["deberta"],
        )
        assert guard.detection_score == 0.9
        assert guard.triggered_methods == ["deberta"]

    def test_multiple_triggered_methods(self):
        guard = _make_guard_result(
            triggered=True,
            detections={
                "det-a": _make_detection(hit=True, score=0.8),
                "det-b": _make_detection(hit=True, score=0.6),
            },
            detection_score=0.8,
            triggered_methods=["det-a", "det-b"],
        )
        assert len(guard.triggered_methods) == 2
        assert "det-a" in guard.triggered_methods
        assert "det-b" in guard.triggered_methods


# --- Tests: native fields on GuardrailResult ---


class TestGuardrailResultNativeFields:
    def test_detection_score_defaults_to_zero(self, clean_guardrail_result):
        assert clean_guardrail_result.detection_score == 0.0

    def test_triggered_methods_defaults_empty(self, clean_guardrail_result):
        assert clean_guardrail_result.triggered_methods == []

    def test_flagged_result_carries_score(self, flagged_guardrail_result):
        assert flagged_guardrail_result.detection_score == 0.95

    def test_flagged_result_carries_methods(self, flagged_guardrail_result):
        assert "deberta-v3" in flagged_guardrail_result.triggered_methods


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
        span.set_attribute.assert_any_call("detection.method", "deberta-v3")
        span.set_attribute.assert_any_call("detection.methods", ["deberta-v3"])

    def test_sets_clean_label_on_unflagged_result(
        self, clean_guardrail_result: GuardrailResult
    ):
        span = MagicMock()
        _set_darwin_span_attributes(span, clean_guardrail_result)

        span.set_attribute.assert_any_call("detection.label", "clean")
        span.set_attribute.assert_any_call("detection.score", 0.0)

        # No methods should be set when nothing triggered
        call_keys = [call.args[0] for call in span.set_attribute.call_args_list]
        assert "detection.method" not in call_keys
        assert "detection.methods" not in call_keys

    def test_skips_team_and_agent_when_not_provided(
        self, flagged_guardrail_result: GuardrailResult
    ):
        span = MagicMock()
        _set_darwin_span_attributes(span, flagged_guardrail_result)

        call_keys = [call.args[0] for call in span.set_attribute.call_args_list]
        assert "team.id" not in call_keys
        assert "agent.id" not in call_keys

    def test_perturbation_changing_score_does_not_affect_label(
        self, flagged_guardrail_result: GuardrailResult
    ):
        """Changing detection_score must not flip detection.label."""
        low_score_result = flagged_guardrail_result.model_copy(
            update={"detection_score": 0.01}
        )
        span = MagicMock()
        _set_darwin_span_attributes(span, low_score_result)

        span.set_attribute.assert_any_call("detection.label", "flagged")
        span.set_attribute.assert_any_call("detection.score", 0.01)


_otel_available = importlib.util.find_spec("opentelemetry") is not None


# --- Tests: _add_darwin_detection_spans integration ---


@pytest.mark.skipif(not _otel_available, reason="opentelemetry not installed")
class TestAddDarwinDetectionSpans:
    """Test that _add_darwin_detection_spans wraps scan correctly."""

    def test_sync_scan_emits_darwin_span(self, flagged_guardrail_result):
        from vijil_dome.integrations.instrumentation.otel_instrumentation import (
            _add_darwin_detection_spans,
        )

        guardrail = MagicMock()
        guardrail.scan = MagicMock(return_value=flagged_guardrail_result)

        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)

        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span

        _add_darwin_detection_spans(guardrail, mock_tracer, "dome-input")
        result = guardrail.scan("test input", agent_id="a1", team_id="t1")

        mock_tracer.start_as_current_span.assert_called_once_with("dome-detection")
        mock_span.set_attribute.assert_any_call("dome.guardrail", "dome-input")
        mock_span.set_attribute.assert_any_call("detection.label", "flagged")
        mock_span.set_attribute.assert_any_call("detection.score", 0.95)
        mock_span.set_attribute.assert_any_call("agent.id", "a1")
        mock_span.set_attribute.assert_any_call("team.id", "t1")

        assert result is flagged_guardrail_result

    @pytest.mark.asyncio
    async def test_async_scan_emits_darwin_span(self, flagged_guardrail_result):
        from vijil_dome.integrations.instrumentation.otel_instrumentation import (
            _add_darwin_detection_spans,
        )

        guardrail = MagicMock()

        async def mock_async_scan(*args, **kwargs):
            return flagged_guardrail_result

        guardrail.async_scan = mock_async_scan

        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)

        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span

        _add_darwin_detection_spans(guardrail, mock_tracer, "dome-output")
        result = await guardrail.async_scan("test output", agent_id="a2", team_id="t2")

        mock_tracer.start_as_current_span.assert_called_once_with("dome-detection")
        mock_span.set_attribute.assert_any_call("dome.guardrail", "dome-output")
        mock_span.set_attribute.assert_any_call("detection.label", "flagged")

        assert result is flagged_guardrail_result
