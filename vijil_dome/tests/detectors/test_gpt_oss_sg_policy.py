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
import os
from pathlib import Path

from vijil_dome.detectors import (
    POLICY_GPT_OSS_SAFEGUARD,
    DetectionFactory,
    DetectionCategory,
)
from vijil_dome.detectors.methods.gpt_oss_safeguard_policy import PolicyGptOssSafeguard


@pytest.mark.asyncio
async def test_policy_detector_spam_detection():
    """Test spam detection with policy file"""
    if not os.getenv("NEBIUS_API_KEY"):
        pytest.skip("NEBIUS_API_KEY not set")

    policy_file = (
        Path(__file__).parent.parent.parent
        / "detectors"
        / "policies"
        / "spam_policy.md"
    )

    # Create detector via factory
    detector = await DetectionFactory.get_detect_with_time(
        DetectionCategory.Generic,
        POLICY_GPT_OSS_SAFEGUARD,
        policy_file=str(policy_file)
    )

    # Test cases: (text, expected_spam)
    test_cases = [
        ("How do I reset my password?", False),
        ("BUY NOW!!! CLICK HERE FOR FREE MONEY $$$", True),
        ("Hi Sam, here is the report you requested", False),
        ("JOIN NOW JOIN NOW JOIN NOW", True),
    ]

    for text, expected_spam in test_cases:
        result = await detector(text)
        assert result.hit == expected_spam, f"Failed for: {text}"
        assert result.result["model"] == "openai/gpt-oss-120b"
        assert result.result["hub"] == "nebius"


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
])
async def test_policy_detector_with_models(model_name):
    """Test detector with both 120B and 20B models"""
    if not os.getenv("NEBIUS_API_KEY"):
        pytest.skip("NEBIUS_API_KEY not set")

    policy_file = (
        Path(__file__).parent.parent.parent
        / "detectors"
        / "policies"
        / "spam_policy.md"
    )

    detector = PolicyGptOssSafeguard(
        policy_file=str(policy_file),
        model_name=model_name
    )

    # Test safe content
    result = await detector.detect("How do I upload a file?")
    assert not result[0]
    assert result[1]["model"] == model_name

    # Test spam content
    result = await detector.detect("BUY NOW CLICK HERE!!!")
    assert result[0]
    assert result[1]["model"] == model_name


@pytest.mark.asyncio
async def test_policy_detector_direct_instantiation():
    """Test direct instantiation"""
    if not os.getenv("NEBIUS_API_KEY"):
        pytest.skip("NEBIUS_API_KEY not set")

    policy_file = (
        Path(__file__).parent.parent.parent
        / "detectors"
        / "policies"
        / "spam_policy.md"
    )

    detector = PolicyGptOssSafeguard(
        policy_file=str(policy_file),
        hub_name="nebius",
        model_name="openai/gpt-oss-120b",
        reasoning_effort="medium",
        timeout=60,
        max_retries=3
    )

    result = await detector.detect("What is the capital of France?")
    assert not result[0]
    assert result[1]["hub"] == "nebius"
    assert result[1]["policy_source"] == str(policy_file)


@pytest.mark.asyncio
async def test_policy_file_not_found():
    """Test that non-existent policy file raises FileNotFoundError"""
    with pytest.raises(FileNotFoundError):
        PolicyGptOssSafeguard(policy_file="/nonexistent/path/policy.md")


def test_invalid_reasoning_effort():
    """Test invalid reasoning effort raises ValueError"""
    policy_file = (
        Path(__file__).parent.parent.parent
        / "detectors"
        / "policies"
        / "spam_policy.md"
    )

    with pytest.raises(ValueError, match="reasoning_effort must be one of"):
        PolicyGptOssSafeguard(
            policy_file=str(policy_file),
            reasoning_effort="invalid"
        )


def test_response_parsing_harmony_format():
    """Test parsing of Harmony format response"""
    policy_file = (
        Path(__file__).parent.parent.parent
        / "detectors"
        / "policies"
        / "spam_policy.md"
    )

    detector = PolicyGptOssSafeguard(policy_file=str(policy_file))

    # Test with full Harmony format
    harmony_response = """<reasoning>
The content contains repetitive all-caps text matching spam pattern.
</reasoning><output>
1
</output>"""

    is_violation, reasoning = detector._parse_response(harmony_response)
    assert is_violation is True
    assert "spam pattern" in reasoning

    # Test safe response
    harmony_response = """<reasoning>
The content is a normal question.
</reasoning><output>
0
</output>"""

    is_violation, reasoning = detector._parse_response(harmony_response)
    assert is_violation is False


def test_response_parsing_no_tags():
    """Test parsing when no Harmony tags present"""
    policy_file = (
        Path(__file__).parent.parent.parent
        / "detectors"
        / "policies"
        / "spam_policy.md"
    )

    detector = PolicyGptOssSafeguard(policy_file=str(policy_file))

    # Test fallback parsing
    is_violation, reasoning = detector._parse_response("1")
    assert is_violation is True

    is_violation, reasoning = detector._parse_response("0")
    assert is_violation is False


@pytest.mark.asyncio
async def test_metadata_in_result():
    """Test that result includes proper metadata"""
    if not os.getenv("NEBIUS_API_KEY"):
        pytest.skip("NEBIUS_API_KEY not set")

    policy_file = (
        Path(__file__).parent.parent.parent
        / "detectors"
        / "policies"
        / "spam_policy.md"
    )

    detector = PolicyGptOssSafeguard(policy_file=str(policy_file))
    result = await detector.detect("Hello world")

    # Check required metadata fields
    assert "model_response" in result[1]
    assert "model" in result[1]
    assert "hub" in result[1]
    assert "policy_source" in result[1]
    assert "reasoning_effort" in result[1]

    assert result[1]["model"] == "openai/gpt-oss-120b"
    assert result[1]["hub"] == "nebius"
    assert result[1]["reasoning_effort"] == "medium"


@pytest.mark.asyncio
async def test_custom_timeout_and_retries():
    """Test custom configuration"""
    policy_file = (
        Path(__file__).parent.parent.parent
        / "detectors"
        / "policies"
        / "spam_policy.md"
    )

    detector = PolicyGptOssSafeguard(
        policy_file=str(policy_file),
        timeout=120,
        max_retries=5
    )

    assert detector.timeout == 120
    assert detector.max_retries == 5


def test_reasoning_directive_appending():
    """Test that reasoning directive is appended correctly"""
    policy_file = (
        Path(__file__).parent.parent.parent
        / "detectors"
        / "policies"
        / "spam_policy.md"
    )

    detector = PolicyGptOssSafeguard(
        policy_file=str(policy_file),
        reasoning_effort="high"
    )

    # Check reasoning directive was appended
    assert "Reasoning: high" in detector.policy

    # Test with policy that already has directive
    policy_with_directive = "# Policy\n\nReasoning: medium"
    result = detector._build_system_message(policy_with_directive)
    # Should not duplicate
    assert result.count("Reasoning:") == 1
