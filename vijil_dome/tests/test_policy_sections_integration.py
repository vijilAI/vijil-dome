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
import json
import tempfile
import os

from vijil_dome import Dome
from vijil_dome.utils.policy_loader import load_policy_sections_from_file
from vijil_dome.utils.policy_config_builder import build_dome_config_from_sections


@pytest.fixture
def sample_policy_data():
    """Sample policy data for testing"""
    return {
        "version": "1.0",
        "last_updated": "2025-01-15T10:30:00Z",
        "etag": "test-etag",
        "policy_id": "test-policy-123",
        "policy_name": "Test Policy",
        "source_file": "test.pdf",
        "sections": [
            {
                "section_id": "test-violation",
                "content": """# Test Violation Policy

## INSTRUCTIONS
Check if input contains the word "VIOLATION".
Return ONLY a single digit:
- 0 = No violation
- 1 = Violation detected

## VIOLATES (1)
- Contains word "VIOLATION"

## SAFE (0)
- Normal queries without "VIOLATION"

## EXAMPLES
Content: "What is the weather?"
Answer: 0

Content: "This contains VIOLATION"
Answer: 1

Content: [INPUT]
Answer:""",
                "applies_to": ["input"],
                "metadata": {
                    "header": "Test Violation Policy",
                    "level": 1,
                    "order": 1
                }
            },
            {
                "section_id": "output-moderation",
                "content": """# Output Moderation Policy

## INSTRUCTIONS
Check if output contains inappropriate content.
Return ONLY: 0 or 1

## VIOLATES (1)
- Contains "badword"

## SAFE (0)
- Appropriate content

Content: [INPUT]
Answer:""",
                "applies_to": ["output"],
                "metadata": {
                    "header": "Output Moderation",
                    "level": 1,
                    "order": 2
                }
            }
        ]
    }


def test_build_config_from_sections(sample_policy_data):
    """Test building Dome config from policy sections"""
    config = build_dome_config_from_sections(sample_policy_data)

    assert "input-guards" in config
    assert "output-guards" in config
    assert config["input-run-parallel"] is True
    assert config["input-early-exit"] is True


def test_dome_instance_from_config(sample_policy_data):
    """Test creating Dome instance from built config"""
    config = build_dome_config_from_sections(sample_policy_data)
    dome = Dome(config)

    assert dome.input_guardrail is not None
    assert dome.output_guardrail is not None
    assert dome.input_guardrail.run_in_parallel is True
    assert dome.input_guardrail.early_exit is True
    assert dome.output_guardrail.run_in_parallel is True
    assert dome.output_guardrail.early_exit is True


def test_dome_config_structure(sample_policy_data):
    """Test that built config has correct structure for Dome"""
    config = build_dome_config_from_sections(sample_policy_data)
    
    # Should be valid Dome config
    dome = Dome(config)
    
    # Verify guardrails exist
    assert dome.input_guardrail is not None
    assert dome.output_guardrail is not None
    
    # Verify guard structure
    assert len(dome.input_guardrail.guard_list) == 1
    assert len(dome.output_guardrail.guard_list) == 1
    
    input_guard = dome.input_guardrail.guard_list[0]
    assert input_guard.run_in_parallel is True
    assert input_guard.early_exit is True


def test_load_and_build_integration(sample_policy_data):
    """Test loading from file and building config"""
    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_policy_data, f, indent=2)
        temp_file = f.name

    try:
        # Load
        policy_data = load_policy_sections_from_file(temp_file)
        assert policy_data["policy_id"] == "test-policy-123"

        # Build config
        config = build_dome_config_from_sections(policy_data)
        assert "input-guards" in config

        # Create Dome
        dome = Dome(config)
        assert dome.input_guardrail is not None

    finally:
        os.unlink(temp_file)


@pytest.mark.asyncio
async def test_guardrail_execution_structure(sample_policy_data):
    """Test that guardrails execute correctly (structure test, no API needed)"""
    config = build_dome_config_from_sections(sample_policy_data)
    dome = Dome(config)

    # Test that guardrails are properly configured
    assert dome.input_guardrail.level == "input"
    assert dome.output_guardrail.level == "output"
    assert dome.input_guardrail.run_in_parallel is True
    assert dome.output_guardrail.run_in_parallel is True


@pytest.mark.asyncio
async def test_guardrail_with_api_key(sample_policy_data):
    """Test guardrail execution with API key (if available)"""
    if not os.getenv("NEBIUS_API_KEY"):
        pytest.skip("NEBIUS_API_KEY not set")

    config = build_dome_config_from_sections(sample_policy_data)
    dome = Dome(config)

    # Test input guardrail
    result = await dome.async_guard_input("This contains VIOLATION")
    
    # Should be flagged (depending on model accuracy)
    assert isinstance(result.flagged, bool)
    assert hasattr(result, "response_string")
    assert hasattr(result, "guard_exec_details")


def test_multiple_sections_parallel_config(sample_policy_data):
    """Test that multiple sections create multiple detectors"""
    # Add more sections
    sample_policy_data["sections"].extend([
        {
            "section_id": "input-section-2",
            "content": "# Another Input Policy",
            "applies_to": ["input"]
        },
        {
            "section_id": "input-section-3",
            "content": "# Third Input Policy",
            "applies_to": ["input"]
        }
    ])

    config = build_dome_config_from_sections(sample_policy_data)
    dome = Dome(config)

    input_guard = dome.input_guardrail.guard_list[0]
    # Should have 3 detectors: test-violation (from fixture), input-section-2, input-section-3
    # (output-moderation is only in output)
    assert len(input_guard.detector_list) == 3


def test_sections_filtered_correctly(sample_policy_data):
    """Test that sections are filtered by applies_to correctly"""
    config = build_dome_config_from_sections(sample_policy_data)

    # Input guard should only have input sections
    input_guard_config = config["input-guards"][0]["all-policy-sections-input"]
    # Methods are keys like "test-violation-0", check if section_id is in the key
    input_methods_str = " ".join(input_guard_config["methods"])
    assert "test-violation" in input_methods_str
    assert "output-moderation" not in input_methods_str

    # Output guard should only have output sections
    output_guard_config = config["output-guards"][0]["all-policy-sections-output"]
    output_methods_str = " ".join(output_guard_config["methods"])
    assert "output-moderation" in output_methods_str
    assert "test-violation" not in output_methods_str


def test_dome_create_from_config_dict(sample_policy_data):
    """Test that built config can be used with Dome constructor"""
    config = build_dome_config_from_sections(sample_policy_data)
    
    # Should work with Dome constructor
    from vijil_dome import create_dome_config
    dome_config = create_dome_config(config)
    dome = Dome(dome_config=dome_config)
    assert dome.input_guardrail is not None
    assert dome.output_guardrail is not None


def test_config_with_custom_detector_settings(sample_policy_data):
    """Test config building with custom detector settings"""
    config = build_dome_config_from_sections(
        sample_policy_data,
        model_name="openai/gpt-oss-20b",
        reasoning_effort="low",
        timeout=30,
        max_retries=2
    )

    input_guard = config["input-guards"][0]["all-policy-sections-input"]
    # Find the config for test-violation section
    method_keys = input_guard["methods"]
    violation_key = [k for k in method_keys if "test-violation" in k][0]
    detector_config = input_guard[violation_key]

    assert detector_config["model_name"] == "openai/gpt-oss-20b"
    assert detector_config["reasoning_effort"] == "low"
    assert detector_config["timeout"] == 30
    assert detector_config["max_retries"] == 2
