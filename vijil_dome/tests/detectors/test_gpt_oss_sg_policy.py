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

import os
from pathlib import Path

import pytest

from vijil_dome.detectors import (
    POLICY_GPT_OSS_SAFEGUARD,
    DetectionCategory,
    DetectionFactory,
)
from vijil_dome.detectors.methods.gpt_oss_safeguard_policy import (
    OutputFormat,
    PolicyGptOssSafeguard,
)

SPAM_POLICY_FILE = (
    Path(__file__).parent.parent.parent
    / "detectors"
    / "policies"
    / "spam_policy.md"
)


def _has_groq_key() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


@pytest.mark.asyncio
async def test_policy_file_not_found():
    with pytest.raises(FileNotFoundError):
        PolicyGptOssSafeguard(policy_file="/nonexistent/path/policy.md")


def test_invalid_reasoning_effort():
    with pytest.raises(ValueError, match="reasoning_effort must be one of"):
        PolicyGptOssSafeguard(
            policy_file=str(SPAM_POLICY_FILE),
            reasoning_effort="invalid",  # type: ignore[arg-type]
        )


def test_invalid_output_format():
    with pytest.raises(ValueError, match="output_format must be one of"):
        PolicyGptOssSafeguard(
            policy_file=str(SPAM_POLICY_FILE),
            output_format="invalid",  # type: ignore[arg-type]
        )


def test_response_parsing_harmony_format():
    detector = PolicyGptOssSafeguard(policy_file=str(SPAM_POLICY_FILE))

    harmony_response = """<reasoning>
The content contains repetitive all-caps text matching spam pattern.
</reasoning><output>
1
</output>"""

    is_violation, reasoning = detector._parse_response(harmony_response)
    assert is_violation is True
    assert "spam pattern" in reasoning

    harmony_response = """<reasoning>
The content is a normal question.
</reasoning><output>
0
</output>"""

    is_violation, _reasoning = detector._parse_response(harmony_response)
    assert is_violation is False


def test_response_parsing_no_tags():
    detector = PolicyGptOssSafeguard(policy_file=str(SPAM_POLICY_FILE))
    is_violation, _reasoning = detector._parse_response("1")
    assert is_violation is True
    is_violation, _reasoning = detector._parse_response("0")
    assert is_violation is False


@pytest.mark.asyncio
async def test_custom_timeout_and_retries():
    detector = PolicyGptOssSafeguard(
        policy_file=str(SPAM_POLICY_FILE),
        timeout=120,
        max_retries=5,
    )
    assert detector.timeout == 120
    assert detector.max_retries == 5


def test_reasoning_directive_appending():
    detector = PolicyGptOssSafeguard(
        policy_file=str(SPAM_POLICY_FILE),
        reasoning_effort="high",
    )
    assert "Reasoning: high" in detector.policy

    policy_with_directive = "# Policy\n\nReasoning: medium"
    result = detector._build_system_message(policy_with_directive)
    assert result.count("Reasoning:") == 1


@pytest.mark.asyncio
async def test_policy_detector_spam_detection_factory():
    # Integration no-op: pass cleanly if API key is not configured.
    if not _has_groq_key():
        return

    detector = await DetectionFactory.get_detect_with_time(
        DetectionCategory.Generic,
        POLICY_GPT_OSS_SAFEGUARD,
        policy_file=str(SPAM_POLICY_FILE),
    )

    test_cases = [
        ("How do I reset my password?", False),
        ("BUY NOW!!! CLICK HERE FOR FREE MONEY $$$", True),
        ("Hi Sam, here is the report you requested", False),
        ("JOIN NOW JOIN NOW JOIN NOW", True),
    ]

    for text, expected_spam in test_cases:
        result = await detector(text)
        assert result.hit == expected_spam, f"Failed for: {text}"
        assert result.result["config"]["model"] == "openai/gpt-oss-safeguard-20b"
        assert result.result["config"]["hub"] == "groq"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    ["openai/gpt-oss-safeguard-120b", "openai/gpt-oss-safeguard-20b"],
)
async def test_policy_detector_models(model_name: str):
    if not _has_groq_key():
        return

    detector = PolicyGptOssSafeguard(
        policy_file=str(SPAM_POLICY_FILE),
        model_name=model_name,
    )

    result = await detector.detect("How do I upload a file?")
    assert not result[0]
    assert result[1]["config"]["model"] == model_name

    result = await detector.detect("BUY NOW CLICK HERE!!!")
    assert result[0]
    assert result[1]["config"]["model"] == model_name


@pytest.mark.asyncio
async def test_policy_detector_direct_instantiation():
    if not _has_groq_key():
        return

    detector = PolicyGptOssSafeguard(
        policy_file=str(SPAM_POLICY_FILE),
        hub_name="groq",
        model_name="openai/gpt-oss-safeguard-120b",
        reasoning_effort="medium",
        timeout=60,
        max_retries=3,
    )

    result = await detector.detect("What is the capital of France?")
    assert not result[0]
    assert result[1]["config"]["hub"] == "groq"
    assert result[1]["config"]["policy_source"] == str(SPAM_POLICY_FILE)


@pytest.mark.asyncio
async def test_metadata_in_result():
    if not _has_groq_key():
        return

    detector = PolicyGptOssSafeguard(policy_file=str(SPAM_POLICY_FILE))
    result = await detector.detect("Hello world")

    assert "model_response" in result[1]
    assert "config" in result[1]
    assert "parsed_output" in result[1]
    assert result[1]["config"]["model"] == "openai/gpt-oss-safeguard-20b"
    assert result[1]["config"]["hub"] == "groq"
    assert result[1]["config"]["reasoning_effort"] == "medium"


@pytest.mark.asyncio
@pytest.mark.parametrize("reasoning_effort", ["low", "medium", "high"])
@pytest.mark.parametrize("output_format", ["binary", "policy_ref", "with_rationale"])
async def test_policy_detector_config_matrix(
    reasoning_effort: str,
    output_format: OutputFormat,
):
    # Integration no-op: pass cleanly if API key is not configured.
    if not _has_groq_key():
        return

    detector = PolicyGptOssSafeguard(
        policy_file=str(SPAM_POLICY_FILE),
        hub_name="groq",
        model_name="openai/gpt-oss-safeguard-20b",
        reasoning_effort=reasoning_effort,  # type: ignore[arg-type]
        output_format=output_format,
    )

    flagged, metadata = await detector.detect(
        "User request: Need campaign advice\n"
        "Agent response: BUY NOW!!! CLICK HERE FOR FREE MONEY $$$"
    )

    assert isinstance(flagged, bool)
    assert isinstance(metadata, dict)
    assert metadata["config"]["reasoning_effort"] == reasoning_effort
    assert metadata["config"]["output_format"] == output_format
    assert metadata["config"]["hub"] == "groq"
    assert metadata["config"]["model"] == "openai/gpt-oss-safeguard-20b"
    assert metadata["config"]["policy_source"] == str(SPAM_POLICY_FILE)
    assert isinstance(metadata["parsed_output"], dict)
