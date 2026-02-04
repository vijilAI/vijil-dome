# Copyright 2025 Vijil, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# vijil and vijil-dome are trademarks owned by Vijil Inc.

"""Tests for Darwin-compatible telemetry instrumentation.

These tests verify that the Darwin telemetry module emits the correct
span attributes and metrics that Darwin's TelemetryDetectionAdapter
can query to trigger evolution mutations.
"""

import pytest
from unittest.mock import MagicMock, patch
from contextlib import contextmanager

from vijil_dome.guardrails import GuardrailResult, GuardResult
from vijil_dome.detectors import DetectionTimingResult
from vijil_dome.integrations.vijil.telemetry import (
    _extract_detection_score,
    _extract_detection_method,
    _set_darwin_span_attributes,
    darwin_trace,
    DARWIN_METRIC_PREFIX,
)


class MockSpan:
    """Mock OTEL span for testing attribute setting."""

    def __init__(self):
        self.attributes = {}

    def set_attribute(self, key: str, value):
        self.attributes[key] = value


class MockTracer:
    """Mock OTEL tracer for testing darwin_trace decorator."""

    def __init__(self):
        self.spans = []
        self._current_span = None

    @contextmanager
    def start_as_current_span(self, name: str):
        span = MockSpan()
        span.name = name
        self.spans.append(span)
        self._current_span = span
        yield span
        self._current_span = None


@pytest.fixture
def mock_tracer():
    return MockTracer()


@pytest.fixture
def flagged_guardrail_result():
    """Create a flagged GuardrailResult with detection details."""
    detection_result = DetectionTimingResult(
        hit=True,
        result={
            "score": 0.92,
            "response_string": "Blocked by prompt injection detector",
        },
        exec_time=45.0,
    )
    guard_result = GuardResult(
        triggered=True,
        details={"PromptInjectionDeberta": detection_result},
        exec_time=0.05,
        response="Guard:prompt-injection Blocked by prompt injection detector",
    )
    return GuardrailResult(
        flagged=True,
        guardrail_response_message="Blocked by input guardrail",
        exec_time=0.06,
        guard_exec_details={"prompt-injection": guard_result},
    )


@pytest.fixture
def clean_guardrail_result():
    """Create a clean (not flagged) GuardrailResult."""
    detection_result = DetectionTimingResult(
        hit=False,
        result={"score": 0.15, "response_string": "What is the weather?"},
        exec_time=30.0,
    )
    guard_result = GuardResult(
        triggered=False,
        details={"PromptInjectionDeberta": detection_result},
        exec_time=0.03,
        response="What is the weather?",
    )
    return GuardrailResult(
        flagged=False,
        guardrail_response_message="What is the weather?",
        exec_time=0.04,
        guard_exec_details={"prompt-injection": guard_result},
    )


class TestExtractDetectionScore:
    """Tests for _extract_detection_score function."""

    def test_extracts_score_from_flagged_guardrail_result(self, flagged_guardrail_result):
        score = _extract_detection_score(flagged_guardrail_result)
        assert score == 0.92

    def test_returns_zero_for_clean_result(self, clean_guardrail_result):
        # Clean results return 0.0 because we only extract scores from
        # triggered guards - Darwin only cares about flagged detection scores
        score = _extract_detection_score(clean_guardrail_result)
        assert score == 0.0

    def test_returns_max_score_from_multiple_guards(self):
        """When multiple guards trigger, return the highest score."""
        detection1 = DetectionTimingResult(
            hit=True, result={"score": 0.75}, exec_time=20.0
        )
        detection2 = DetectionTimingResult(
            hit=True, result={"score": 0.95}, exec_time=25.0
        )
        guard1 = GuardResult(
            triggered=True,
            details={"Detector1": detection1},
            exec_time=0.02,
            response="Blocked",
        )
        guard2 = GuardResult(
            triggered=True,
            details={"Detector2": detection2},
            exec_time=0.03,
            response="Blocked",
        )
        result = GuardrailResult(
            flagged=True,
            guardrail_response_message="Blocked",
            exec_time=0.05,
            guard_exec_details={"guard1": guard1, "guard2": guard2},
        )

        score = _extract_detection_score(result)
        assert score == 0.95

    def test_handles_guard_result_directly(self):
        """Test with GuardResult instead of GuardrailResult."""
        detection = DetectionTimingResult(
            hit=True, result={"score": 0.88}, exec_time=15.0
        )
        guard_result = GuardResult(
            triggered=True,
            details={"ToxicityDetector": detection},
            exec_time=0.02,
            response="Blocked",
        )

        score = _extract_detection_score(guard_result)
        assert score == 0.88


class TestExtractDetectionMethod:
    """Tests for _extract_detection_method function."""

    def test_extracts_guard_name_from_flagged_result(self, flagged_guardrail_result):
        method = _extract_detection_method(flagged_guardrail_result)
        assert method == "prompt-injection"

    def test_returns_unknown_for_clean_result(self, clean_guardrail_result):
        method = _extract_detection_method(clean_guardrail_result)
        assert method == "unknown"

    def test_returns_first_triggered_guard(self):
        """When multiple guards trigger, return the first one found."""
        detection = DetectionTimingResult(
            hit=True, result={"score": 0.90}, exec_time=20.0
        )
        guard1 = GuardResult(
            triggered=True,
            details={"Detector": detection},
            exec_time=0.02,
            response="Blocked",
        )
        guard2 = GuardResult(
            triggered=True,
            details={"Detector": detection},
            exec_time=0.03,
            response="Blocked",
        )
        result = GuardrailResult(
            flagged=True,
            guardrail_response_message="Blocked",
            exec_time=0.05,
            guard_exec_details={"jailbreak": guard1, "toxicity": guard2},
        )

        method = _extract_detection_method(result)
        # Should be one of the triggered guards (dict iteration order)
        assert method in ["jailbreak", "toxicity"]


class TestSetDarwinSpanAttributes:
    """Tests for _set_darwin_span_attributes function."""

    def test_sets_all_darwin_attributes_on_flagged_result(
        self, flagged_guardrail_result
    ):
        span = MockSpan()

        _set_darwin_span_attributes(
            span,
            flagged_guardrail_result,
            agent_id="agent-test-123",
            team_id="team-abc-456",
        )

        assert span.attributes["team.id"] == "team-abc-456"
        assert span.attributes["agent.id"] == "agent-test-123"
        assert span.attributes["detection.label"] == "flagged"
        assert span.attributes["detection.score"] == 0.92
        assert span.attributes["detection.method"] == "prompt-injection"

    def test_sets_clean_label_on_clean_result(self, clean_guardrail_result):
        span = MockSpan()

        _set_darwin_span_attributes(span, clean_guardrail_result)

        assert span.attributes["detection.label"] == "clean"

    def test_omits_missing_context_ids(self, flagged_guardrail_result):
        span = MockSpan()

        _set_darwin_span_attributes(span, flagged_guardrail_result)

        assert "team.id" not in span.attributes
        assert "agent.id" not in span.attributes
        assert "detection.label" in span.attributes
        assert "detection.score" in span.attributes


class TestDarwinTraceDecorator:
    """Tests for darwin_trace decorator."""

    def test_creates_span_with_correct_name(self, mock_tracer, flagged_guardrail_result):
        @darwin_trace(mock_tracer, "input-guardrail.scan")
        def scan_function(text, agent_id=None, team_id=None):
            return flagged_guardrail_result

        result = scan_function("test input", agent_id="agent-1", team_id="team-1")

        assert len(mock_tracer.spans) == 1
        assert mock_tracer.spans[0].name == "input-guardrail.scan"

    def test_sets_generic_attributes(self, mock_tracer, flagged_guardrail_result):
        @darwin_trace(mock_tracer, "test.scan")
        def scan_function(text, agent_id=None):
            return flagged_guardrail_result

        scan_function("hello world", agent_id="agent-x")

        span = mock_tracer.spans[0]
        assert "function.args" in span.attributes
        assert "function.kwargs" in span.attributes
        assert "function.result" in span.attributes

    def test_sets_darwin_attributes_for_guardrail_result(
        self, mock_tracer, flagged_guardrail_result
    ):
        @darwin_trace(mock_tracer, "input-guardrail.scan")
        def scan_function(text, agent_id=None, team_id=None):
            return flagged_guardrail_result

        scan_function("test", agent_id="agent-1", team_id="team-1")

        span = mock_tracer.spans[0]
        assert span.attributes["detection.label"] == "flagged"
        assert span.attributes["detection.score"] == 0.92
        assert span.attributes["detection.method"] == "prompt-injection"
        assert span.attributes["agent.id"] == "agent-1"
        assert span.attributes["team.id"] == "team-1"

    def test_skips_darwin_attributes_for_non_guardrail_result(self, mock_tracer):
        @darwin_trace(mock_tracer, "other.function")
        def other_function(text):
            return {"result": "not a guardrail result"}

        other_function("test")

        span = mock_tracer.spans[0]
        # Should have generic attributes but NOT Darwin-specific ones
        assert "function.result" in span.attributes
        assert "detection.label" not in span.attributes
        assert "detection.score" not in span.attributes

    @pytest.mark.asyncio
    async def test_works_with_async_functions(
        self, mock_tracer, flagged_guardrail_result
    ):
        @darwin_trace(mock_tracer, "async.scan")
        async def async_scan(text, agent_id=None, team_id=None):
            return flagged_guardrail_result

        result = await async_scan("test", agent_id="agent-async", team_id="team-async")

        assert len(mock_tracer.spans) == 1
        span = mock_tracer.spans[0]
        assert span.attributes["detection.label"] == "flagged"
        assert span.attributes["agent.id"] == "agent-async"


class TestDarwinMetricPrefix:
    """Tests for metric naming convention."""

    def test_darwin_metric_prefix_is_dome(self):
        assert DARWIN_METRIC_PREFIX == "dome"
