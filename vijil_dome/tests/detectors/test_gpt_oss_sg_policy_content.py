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

from vijil_dome.detectors.methods.gpt_oss_safeguard_policy import PolicyGptOssSafeguard


def test_policy_content_parameter():
    """Test that PolicyGptOssSafeguard accepts policy_content parameter"""
    policy_content = """# Test Policy

## INSTRUCTIONS
Check if input contains "test-violation".
Return ONLY: 0 or 1

## VIOLATES (1)
- Contains "test-violation"

## SAFE (0)
- Normal text

Content: [INPUT]
Answer:"""

    detector = PolicyGptOssSafeguard(policy_content=policy_content)
    assert detector.policy_source == "inline_content"
    assert "# Test Policy" in detector.policy
    assert "Reasoning: medium" in detector.policy


def test_policy_content_vs_file_exclusive():
    """Test that policy_content and policy_file are mutually exclusive"""
    policy_file = (
        Path(__file__).parent.parent.parent
        / "detectors"
        / "policies"
        / "spam_policy.md"
    )

    policy_content = "# Test Policy"

    # Should raise if both provided
    with pytest.raises(ValueError, match="Cannot specify both"):
        PolicyGptOssSafeguard(
            policy_file=str(policy_file),
            policy_content=policy_content
        )


def test_policy_content_or_file_required():
    """Test that at least one of policy_file or policy_content is required"""
    with pytest.raises(ValueError, match="Either"):
        PolicyGptOssSafeguard()


def test_policy_content_with_reasoning_effort():
    """Test policy_content with different reasoning effort levels"""
    policy_content = "# Test Policy\n\nContent here"

    for effort in ["low", "medium", "high"]:
        detector = PolicyGptOssSafeguard(
            policy_content=policy_content,
            reasoning_effort=effort
        )
        assert f"Reasoning: {effort}" in detector.policy


def test_policy_content_preserves_existing_reasoning():
    """Test that policy_content with existing Reasoning directive is preserved"""
    policy_content = """# Test Policy

Reasoning: high"""

    detector = PolicyGptOssSafeguard(
        policy_content=policy_content,
        reasoning_effort="medium"  # Should be ignored
    )
    # Should not duplicate Reasoning directive
    assert detector.policy.count("Reasoning:") == 1
    assert "Reasoning: high" in detector.policy


@pytest.mark.asyncio
async def test_policy_content_detection():
    """Test detection with policy_content (if API key available)"""
    if not os.getenv("NEBIUS_API_KEY"):
        pytest.skip("NEBIUS_API_KEY not set")

    policy_content = """# Test Policy

## INSTRUCTIONS
Check if input contains "blockme".
Return ONLY: 0 or 1

## VIOLATES (1)
- Contains "blockme"

## SAFE (0)
- Normal text

Content: [INPUT]
Answer:"""

    detector = PolicyGptOssSafeguard(
        policy_content=policy_content,
        model_name="openai/gpt-oss-120b"
    )

    # Test safe input
    result = await detector.detect("What is the weather?")
    assert not result[0]
    assert result[1]["policy_source"] == "inline_content"

    # Test violation
    result = await detector.detect("This is a blockme query")
    # Note: Actual result depends on model, but should have metadata
    assert "model" in result[1]
    assert result[1]["policy_source"] == "inline_content"


def test_backward_compatibility_with_file():
    """Test that existing policy_file usage still works"""
    policy_file = (
        Path(__file__).parent.parent.parent
        / "detectors"
        / "policies"
        / "spam_policy.md"
    )

    detector = PolicyGptOssSafeguard(policy_file=str(policy_file))
    assert detector.policy_source == str(policy_file)
    assert "# Spam Detection Policy" in detector.policy
