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

import pytest
from pydantic import ValidationError

from vijil_dome import Dome, DomePayload
from vijil_dome.Dome import DomeConfig, ScanResult
from vijil_dome.detectors import DetectionMethod, DetectionResult
from vijil_dome.guardrails import Guard, Guardrail


# ---------------------------------------------------------------------------
# DomePayload unit tests
# ---------------------------------------------------------------------------

class TestDomePayloadCreation:
    def test_create_from_text(self):
        p = DomePayload(text="hello")
        assert p.query_string == "hello"
        assert p.text == "hello"
        assert p.prompt is None
        assert p.response is None

    def test_create_from_prompt_response(self):
        p = DomePayload(prompt="what is 2+2?", response="4")
        assert "Input: what is 2+2?" in p.query_string
        assert "Output: 4" in p.query_string
        assert p.text is None

    def test_create_prompt_only(self):
        p = DomePayload(prompt="just a question")
        assert p.query_string == "Input: just a question"

    def test_create_response_only(self):
        p = DomePayload(response="just an answer")
        assert p.query_string == "Output: just an answer"

    def test_query_string_multiline_format(self):
        p = DomePayload(prompt="hello", response="world")
        assert p.query_string == "Input: hello\nOutput: world"


class TestDomePayloadValidation:
    def test_empty_raises(self):
        with pytest.raises(ValidationError):
            DomePayload()

    def test_text_with_prompt_raises(self):
        with pytest.raises(ValidationError):
            DomePayload(text="hi", prompt="hello")

    def test_text_with_response_raises(self):
        with pytest.raises(ValidationError):
            DomePayload(text="hi", response="hello")

    def test_text_with_both_raises(self):
        with pytest.raises(ValidationError):
            DomePayload(text="hi", prompt="hello", response="world")


class TestDomePayloadCoerce:
    def test_coerce_from_string(self):
        p = DomePayload.coerce("hello")
        assert isinstance(p, DomePayload)
        assert p.text == "hello"
        assert p.query_string == "hello"

    def test_coerce_from_dome_payload(self):
        original = DomePayload(prompt="q", response="a")
        result = DomePayload.coerce(original)
        assert result is original

    def test_coerce_invalid_type_raises(self):
        with pytest.raises(TypeError):
            DomePayload.coerce(123)

    def test_coerce_none_raises(self):
        with pytest.raises(TypeError):
            DomePayload.coerce(None)


# ---------------------------------------------------------------------------
# Mock detectors for pipeline tests
# ---------------------------------------------------------------------------

class NoopDetector(DetectionMethod):
    """Always-safe detector for testing pipeline plumbing."""
    async def detect(self, dome_input: DomePayload) -> DetectionResult:
        return False, {
            "type": type(self),
            "response_string": dome_input.query_string,
        }


class FlagDetector(DetectionMethod):
    """Always-flags detector for testing pipeline plumbing."""
    async def detect(self, dome_input: DomePayload) -> DetectionResult:
        return True, {
            "type": type(self),
            "response_string": "BLOCKED",
            "score": 1.0,
        }


class PayloadCapturingDetector(DetectionMethod):
    """Captures the DomePayload for inspection in tests."""
    def __init__(self):
        self.last_dome_input = None

    async def detect(self, dome_input: DomePayload) -> DetectionResult:
        self.last_dome_input = dome_input
        return False, {
            "type": type(self),
            "response_string": dome_input.query_string,
        }


def _make_dome(detector, level="input"):
    guard = Guard("test-guard", [detector], early_exit=True)
    input_guardrail = Guardrail("input", [guard] if level == "input" else [])
    output_guardrail = Guardrail("output", [guard] if level == "output" else [])
    config = DomeConfig(
        input_guardrail=input_guardrail,
        output_guardrail=output_guardrail,
    )
    return Dome(dome_config=config)


# ---------------------------------------------------------------------------
# Dome backward compatibility: str input
# ---------------------------------------------------------------------------

class TestDomeBackwardCompatString:
    def test_guard_input_string(self):
        dome = _make_dome(NoopDetector())
        result = dome.guard_input("hello world")
        assert isinstance(result, ScanResult)
        assert result.is_safe()
        assert result.response_string == "hello world"

    @pytest.mark.asyncio
    async def test_async_guard_input_string(self):
        dome = _make_dome(NoopDetector())
        result = await dome.async_guard_input("hello world")
        assert result.is_safe()
        assert result.response_string == "hello world"

    def test_guard_output_string(self):
        dome = _make_dome(NoopDetector(), level="output")
        result = dome.guard_output("hello world")
        assert result.is_safe()
        assert result.response_string == "hello world"

    @pytest.mark.asyncio
    async def test_async_guard_output_string(self):
        dome = _make_dome(NoopDetector(), level="output")
        result = await dome.async_guard_output("hello world")
        assert result.is_safe()

    def test_guard_input_string_flagged(self):
        dome = _make_dome(FlagDetector())
        result = dome.guard_input("anything")
        assert not result.is_safe()
        assert result.enforced


# ---------------------------------------------------------------------------
# Dome with DomePayload: structured input
# ---------------------------------------------------------------------------

class TestDomeStructuredInput:
    def test_guard_input_dome_payload(self):
        dome = _make_dome(NoopDetector())
        result = dome.guard_input(DomePayload(prompt="question", response="answer"))
        assert result.is_safe()
        assert "Input: question" in result.response_string
        assert "Output: answer" in result.response_string

    def test_guard_output_dome_payload(self):
        dome = _make_dome(NoopDetector(), level="output")
        result = dome.guard_output(DomePayload(prompt="q", response="a"))
        assert result.is_safe()

    def test_guard_input_flagged_payload(self):
        dome = _make_dome(FlagDetector())
        result = dome.guard_input(DomePayload(prompt="bad", response="stuff"))
        assert not result.is_safe()
        assert result.enforced

    def test_detector_receives_dome_payload(self):
        detector = PayloadCapturingDetector()
        dome = _make_dome(detector)
        dome.guard_input(DomePayload(prompt="hello", response="world"))
        assert detector.last_dome_input is not None
        assert isinstance(detector.last_dome_input, DomePayload)
        assert detector.last_dome_input.prompt == "hello"
        assert detector.last_dome_input.response == "world"

    def test_detector_receives_text_payload_from_string(self):
        detector = PayloadCapturingDetector()
        dome = _make_dome(detector)
        dome.guard_input("plain string")
        assert detector.last_dome_input is not None
        assert detector.last_dome_input.text == "plain string"
        assert detector.last_dome_input.prompt is None

    @pytest.mark.asyncio
    async def test_async_guard_input_dome_payload(self):
        dome = _make_dome(NoopDetector())
        result = await dome.async_guard_input(DomePayload(prompt="q", response="a"))
        assert result.is_safe()

    @pytest.mark.asyncio
    async def test_async_guard_output_dome_payload(self):
        dome = _make_dome(NoopDetector(), level="output")
        result = await dome.async_guard_output(DomePayload(prompt="q", response="a"))
        assert result.is_safe()


# ---------------------------------------------------------------------------
# Batch tests
# ---------------------------------------------------------------------------

class TestDomeBatchWithPayload:
    def test_guard_input_batch_strings(self):
        dome = _make_dome(NoopDetector())
        inputs = ["hello", "world"]
        result = dome.guard_input_batch(inputs)
        assert result.all_safe()
        assert result[0].response_string == "hello"
        assert result[1].response_string == "world"

    def test_guard_input_batch_payloads(self):
        dome = _make_dome(NoopDetector())
        inputs = [
            DomePayload(text="plain"),
            DomePayload(prompt="q", response="a"),
        ]
        result = dome.guard_input_batch(inputs)
        assert result.all_safe()
        assert result[0].response_string == "plain"
        assert "Input: q" in result[1].response_string

    def test_guard_input_batch_mixed(self):
        dome = _make_dome(NoopDetector())
        inputs = ["plain string", DomePayload(prompt="q", response="a")]
        result = dome.guard_input_batch(inputs)
        assert result.all_safe()
        assert result[0].response_string == "plain string"

    @pytest.mark.asyncio
    async def test_async_guard_input_batch_payloads(self):
        dome = _make_dome(NoopDetector())
        inputs = [DomePayload(prompt="q1", response="a1"), "plain"]
        result = await dome.async_guard_input_batch(inputs)
        assert result.all_safe()


# ---------------------------------------------------------------------------
# Empty guardrail tests
# ---------------------------------------------------------------------------

class TestDomeEmptyGuardrail:
    def test_no_guardrail_string(self):
        config = {"input-guards": [], "output-guards": []}
        dome = Dome(dome_config=config)
        result = dome.guard_input("hello")
        assert result.is_safe()
        assert result.response_string == "hello"

    def test_no_guardrail_payload(self):
        config = {"input-guards": [], "output-guards": []}
        dome = Dome(dome_config=config)
        result = dome.guard_input(DomePayload(prompt="q", response="a"))
        assert result.is_safe()
        assert "Input: q" in result.response_string
