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

from typing import Dict, Any, List, Optional

# The registered detector method name for PolicyGptOssSafeguard
POLICY_GPT_OSS_SAFEGUARD_METHOD = "policy-gpt-oss-safeguard"


def build_dome_config_from_sections(
    policy_data: Dict[str, Any],
    model_name: str = "openai/gpt-oss-120b",
    reasoning_effort: str = "medium",
    hub_name: str = "nebius",
    timeout: Optional[int] = 60,
    max_retries: Optional[int] = 3,
    api_key: Optional[str] = None,
    early_exit: bool = True,
    run_parallel: bool = True,
) -> Dict[str, Any]:
    """
    Build a Dome guardrail config dictionary from policy sections.

    Creates a config where each policy section becomes a detector that runs
    in parallel with early exit enabled.

    Args:
        policy_data: Policy data dictionary with 'sections' array
        model_name: LLM model to use for all detectors (default: "openai/gpt-oss-120b")
        reasoning_effort: Reasoning depth - "low", "medium", "high" (default: "medium")
        hub_name: LLM hub to use (default: "nebius")
        timeout: Request timeout in seconds (default: 60)
        max_retries: Maximum retry attempts (default: 3)
        api_key: API key for the hub (optional, uses env var if not provided)
        early_exit: Enable early exit when first violation detected (default: True)
        run_parallel: Run detectors in parallel (default: True)

    Returns:
        Dome config dictionary ready for Dome(config)

    Raises:
        ValueError: If policy_data structure is invalid
    """
    if "sections" not in policy_data:
        raise ValueError("policy_data must contain 'sections' array")

    sections = policy_data["sections"]
    if not isinstance(sections, list) or len(sections) == 0:
        raise ValueError("'sections' must be a non-empty array")

    # Filter sections by applies_to
    input_sections = [
        s for s in sections if isinstance(s.get("applies_to"), list) and "input" in s["applies_to"]
    ]
    output_sections = [
        s for s in sections if isinstance(s.get("applies_to"), list) and "output" in s["applies_to"]
    ]

    # Build config dictionary
    config: Dict[str, Any] = {}

    # Get policy_id if available
    policy_id = policy_data.get("policy_id")

    # Build input guard config
    if input_sections:
        input_guard_config = _build_guard_config(
            input_sections,
            "all-policy-sections-input",
            model_name=model_name,
            reasoning_effort=reasoning_effort,
            hub_name=hub_name,
            timeout=timeout,
            max_retries=max_retries,
            api_key=api_key,
            early_exit=early_exit,
            run_parallel=run_parallel,
            policy_id=policy_id,
        )
        config["input-guards"] = [{"all-policy-sections-input": input_guard_config}]
        config["input-run-parallel"] = run_parallel
        config["input-early-exit"] = early_exit

    # Build output guard config
    if output_sections:
        output_guard_config = _build_guard_config(
            output_sections,
            "all-policy-sections-output",
            model_name=model_name,
            reasoning_effort=reasoning_effort,
            hub_name=hub_name,
            timeout=timeout,
            max_retries=max_retries,
            api_key=api_key,
            early_exit=early_exit,
            run_parallel=run_parallel,
            policy_id=policy_id,
        )
        config["output-guards"] = [{"all-policy-sections-output": output_guard_config}]
        config["output-run-parallel"] = run_parallel
        config["output-early-exit"] = early_exit

    return config


def _build_guard_config(
    sections: List[Dict[str, Any]],
    guard_name: str,
    model_name: str,
    reasoning_effort: str,
    hub_name: str,
    timeout: Optional[int],
    max_retries: Optional[int],
    api_key: Optional[str],
    early_exit: bool,
    run_parallel: bool,
    policy_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a guard configuration dictionary for a list of sections.

    Args:
        sections: List of section dictionaries
        guard_name: Name for the guard
        model_name: Model name for detectors
        reasoning_effort: Reasoning effort level
        hub_name: Hub name
        timeout: Request timeout
        max_retries: Max retries
        api_key: API key
        early_exit: Enable early exit
        run_parallel: Run in parallel

    Returns:
        Guard config dictionary
    """
    guard_config: Dict[str, Any] = {
        "type": "generic",
        "methods": [],
        "run-parallel": run_parallel,
        "early-exit": early_exit,
    }

    # Add detector config for each section
    # Each section uses the same detector method but with unique config keys
    for i, section in enumerate(sections):
        section_id = section["section_id"]
        # Create unique config key for each section
        config_key = f"{section_id}-{i}"
        guard_config["methods"].append(config_key)

        # Build detector config
        detector_config: Dict[str, Any] = {
            # Specify the actual detector method name
            "method": POLICY_GPT_OSS_SAFEGUARD_METHOD,
            "policy_content": section["content"],
            "section_id": section_id,
            "model_name": model_name,
            "reasoning_effort": reasoning_effort,
            "hub_name": hub_name,
        }

        # Add optional parameters only if provided
        if timeout is not None:
            detector_config["timeout"] = timeout
        if max_retries is not None:
            detector_config["max_retries"] = max_retries
        if api_key is not None:
            detector_config["api_key"] = api_key

        # Add section metadata if available
        if "metadata" in section and isinstance(section["metadata"], dict):
            metadata = section["metadata"]
            if "header" in metadata:
                detector_config["header"] = metadata["header"]
        if policy_id:
            detector_config["policy_id"] = policy_id

        # Store config with unique key
        guard_config[config_key] = detector_config

    return guard_config
