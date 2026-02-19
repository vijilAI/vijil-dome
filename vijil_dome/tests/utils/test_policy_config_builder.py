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

from vijil_dome.utils.policy_config_builder import build_dome_config_from_sections


@pytest.fixture
def sample_policy_data():
    """Sample policy data with multiple sections"""
    return {
        "version": "1.0",
        "policy_id": "test-policy-123",
        "policy_name": "Test Policy",
        "sections": [
            {
                "section_id": "input-section-1",
                "content": "# Input Policy 1\n\nContent here",
                "applies_to": ["input"],
                "metadata": {"header": "Input Policy 1"}
            },
            {
                "section_id": "input-section-2",
                "content": "# Input Policy 2\n\nContent here",
                "applies_to": ["input"],
            },
            {
                "section_id": "output-section-1",
                "content": "# Output Policy 1\n\nContent here",
                "applies_to": ["output"],
            },
            {
                "section_id": "both-section-1",
                "content": "# Both Policy\n\nContent here",
                "applies_to": ["input", "output"],
            }
        ]
    }


def test_build_dome_config_from_sections_basic(sample_policy_data):
    """Test basic config building"""
    config = build_dome_config_from_sections(sample_policy_data)

    assert "input-guards" in config
    assert "output-guards" in config
    assert config["input-run-parallel"] is True
    assert config["input-early-exit"] is True
    assert config["output-run-parallel"] is True
    assert config["output-early-exit"] is True


def test_build_dome_config_filters_input_sections(sample_policy_data):
    """Test that only input sections are in input-guards"""
    config = build_dome_config_from_sections(sample_policy_data)

    input_guard = config["input-guards"][0]["all-policy-sections-input"]
    methods = input_guard["methods"]

    # Methods should be unique keys for each section (section_id-index)
    # Should have 3 methods (input-section-1, input-section-2, both-section-1)
    assert len(methods) == 3
    
    # Verify detector configs exist for each method and use correct detector method
    for method_key in methods:
        assert method_key in input_guard
        detector_config = input_guard[method_key]
        assert detector_config["method"] == "policy-gpt-oss-safeguard"
    
    # Should NOT include: output-section-1
    output_guard = config["output-guards"][0]["all-policy-sections-output"]
    output_methods = output_guard["methods"]
    assert "output-section-1" in str(output_methods) or any("output-section-1" in m for m in output_methods)


def test_build_dome_config_filters_output_sections(sample_policy_data):
    """Test that only output sections are in output-guards"""
    config = build_dome_config_from_sections(sample_policy_data)

    output_guard = config["output-guards"][0]["all-policy-sections-output"]
    methods = output_guard["methods"]

    # Methods are keys like "output-section-1-0", check if section_id is in the key
    output_methods_str = " ".join(methods)
    assert "output-section-1" in output_methods_str
    assert "both-section-1" in output_methods_str
    # Should NOT include: input-only sections
    assert "input-section-1" not in output_methods_str
    assert "input-section-2" not in output_methods_str


def test_build_dome_config_detector_config(sample_policy_data):
    """Test that detector configs are created correctly"""
    config = build_dome_config_from_sections(
        sample_policy_data,
        model_name="openai/gpt-oss-20b",
        reasoning_effort="high"
    )

    input_guard = config["input-guards"][0]["all-policy-sections-input"]
    # Find the config for input-section-1 (it will have a key like "input-section-1-0")
    method_keys = input_guard["methods"]
    section_1_key = [k for k in method_keys if "input-section-1" in k][0]
    detector_config = input_guard[section_1_key]

    assert detector_config["method"] == "policy-gpt-oss-safeguard"
    assert detector_config["policy_content"] == "# Input Policy 1\n\nContent here"
    assert detector_config["section_id"] == "input-section-1"
    assert detector_config["model_name"] == "openai/gpt-oss-20b"
    assert detector_config["reasoning_effort"] == "high"
    assert detector_config["hub_name"] == "nebius"
    assert detector_config["timeout"] == 60  # default
    assert detector_config["max_retries"] == 3  # default


def test_build_dome_config_parallel_settings(sample_policy_data):
    """Test parallel and early-exit settings"""
    config = build_dome_config_from_sections(
        sample_policy_data,
        run_parallel=True,
        early_exit=True
    )

    input_guard = config["input-guards"][0]["all-policy-sections-input"]
    assert input_guard["run-parallel"] is True
    assert input_guard["early-exit"] is True
    assert config["input-run-parallel"] is True
    assert config["input-early-exit"] is True


def test_build_dome_config_custom_settings(sample_policy_data):
    """Test custom parallel/early-exit settings"""
    config = build_dome_config_from_sections(
        sample_policy_data,
        run_parallel=False,
        early_exit=False
    )

    input_guard = config["input-guards"][0]["all-policy-sections-input"]
    assert input_guard["run-parallel"] is False
    assert input_guard["early-exit"] is False
    assert config["input-run-parallel"] is False
    assert config["input-early-exit"] is False


def test_build_dome_config_optional_parameters(sample_policy_data):
    """Test optional detector parameters"""
    config = build_dome_config_from_sections(
        sample_policy_data,
        timeout=120,
        max_retries=5,
        api_key="test-key"
    )

    input_guard = config["input-guards"][0]["all-policy-sections-input"]
    # Find the config for input-section-1
    method_keys = input_guard["methods"]
    section_1_key = [k for k in method_keys if "input-section-1" in k][0]
    detector_config = input_guard[section_1_key]

    assert detector_config["timeout"] == 120
    assert detector_config["max_retries"] == 5
    assert detector_config["api_key"] == "test-key"


def test_build_dome_config_includes_metadata(sample_policy_data):
    """Test that section metadata is included in detector config"""
    config = build_dome_config_from_sections(sample_policy_data)

    input_guard = config["input-guards"][0]["all-policy-sections-input"]
    # Find the config for input-section-1
    method_keys = input_guard["methods"]
    section_1_key = [k for k in method_keys if "input-section-1" in k][0]
    detector_config = input_guard[section_1_key]

    # Should include header from metadata if present
    if "metadata" in sample_policy_data["sections"][0] and "header" in sample_policy_data["sections"][0]["metadata"]:
        assert detector_config["header"] == "Input Policy 1"


def test_build_dome_config_only_input_sections():
    """Test config with only input sections"""
    policy_data = {
        "sections": [
            {
                "section_id": "input-only",
                "content": "# Input",
                "applies_to": ["input"]
            }
        ]
    }

    config = build_dome_config_from_sections(policy_data)

    assert "input-guards" in config
    assert "output-guards" not in config


def test_build_dome_config_only_output_sections():
    """Test config with only output sections"""
    policy_data = {
        "sections": [
            {
                "section_id": "output-only",
                "content": "# Output",
                "applies_to": ["output"]
            }
        ]
    }

    config = build_dome_config_from_sections(policy_data)

    assert "output-guards" in config
    assert "input-guards" not in config


def test_build_dome_config_missing_sections():
    """Test that missing sections raises error"""
    with pytest.raises(ValueError, match="sections"):
        build_dome_config_from_sections({"version": "1.0"})


def test_build_dome_config_empty_sections():
    """Test that empty sections raises error"""
    with pytest.raises(ValueError, match="non-empty"):
        build_dome_config_from_sections({"sections": []})


def test_build_dome_config_guard_type_generic(sample_policy_data):
    """Test that guard type is set to generic"""
    config = build_dome_config_from_sections(sample_policy_data)

    input_guard = config["input-guards"][0]["all-policy-sections-input"]
    assert input_guard["type"] == "generic"

    output_guard = config["output-guards"][0]["all-policy-sections-output"]
    assert output_guard["type"] == "generic"
