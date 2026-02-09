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

"""
Example: Policy-based Content Classification with GPT-OSS-Safeguard

This example demonstrates how to use the PolicyGptOssSafeguard detector
for custom policy-based content moderation using OpenAI's gpt-oss-safeguard
model via Groq.

Prerequisites:
- Set GROQ_API_KEY environment variable
- Create a policy file (see vijil_dome/detectors/policies/spam_policy.md for example)

The detector supports:
- Custom policy files defining violation criteria
- Two model variants: openai/gpt-oss-safeguard-120b (more accurate) and
  openai/gpt-oss-safeguard-20b (faster)
- Three output formats:
  - "binary": Returns 0/1 only
  - "policy_ref": Returns JSON with violation + policy_category
  - "with_rationale": Returns JSON with full reasoning
- Configurable reasoning effort: low, medium, high
"""

import asyncio
import os
from pathlib import Path

from vijil_dome.detectors import (
    POLICY_GPT_OSS_SAFEGUARD,
    DetectionCategory,
    DetectionFactory,
)
from vijil_dome.detectors.methods.gpt_oss_safeguard_policy import (
    OutputFormat,
    PolicyGptOssSafeguard,
)


async def example_factory_usage():
    print("\n" + "=" * 60)
    print("Example 1: Factory Pattern Usage")
    print("=" * 60)

    policy_file = (
        Path(__file__).parent.parent.parent
        / "detectors"
        / "policies"
        / "spam_policy.md"
    )

    detector = await DetectionFactory.get_detect_with_time(
        DetectionCategory.Generic,
        POLICY_GPT_OSS_SAFEGUARD,
        policy_file=str(policy_file),
    )

    test_cases = [
        "How do I reset my password?",
        "BUY NOW!!! CLICK HERE FOR FREE MONEY $$$",
        "Can you help me with my account settings?",
    ]

    for text in test_cases:
        result = await detector(text)
        print(f"\nInput: {text}")
        print(f"Violation: {result.hit}")
        print(f"Execution time: {result.exec_time}ms")
        print(f"Model: {result.result['config']['model']}")


async def example_direct_instantiation():
    print("\n" + "=" * 60)
    print("Example 2: Direct Instantiation")
    print("=" * 60)

    policy_file = (
        Path(__file__).parent.parent.parent
        / "detectors"
        / "policies"
        / "spam_policy.md"
    )

    detector = PolicyGptOssSafeguard(
        policy_file=str(policy_file),
        hub_name="groq",
        model_name="openai/gpt-oss-safeguard-20b",
        output_format="policy_ref",
        reasoning_effort="high",
        timeout=90,
        max_retries=3,
    )

    test_text = "JOIN NOW!!! LIMITED TIME OFFER!!! CLICK HERE!!!"
    result = await detector.detect(test_text)

    print(f"\nInput: {test_text}")
    print(f"Violation detected: {result[0]}")
    print(f"Config: {result[1]['config']}")
    print(f"Parsed output: {result[1]['parsed_output']}")


async def example_different_models():
    print("\n" + "=" * 60)
    print("Example 3: Model Comparison (120B vs 20B)")
    print("=" * 60)

    policy_file = (
        Path(__file__).parent.parent.parent
        / "detectors"
        / "policies"
        / "spam_policy.md"
    )

    test_text = "Congratulations! You've won a FREE iPhone! Click now to claim!"
    for model_name in ["openai/gpt-oss-safeguard-120b", "openai/gpt-oss-safeguard-20b"]:
        print(f"\n--- Testing with {model_name} ---")
        detector = PolicyGptOssSafeguard(
            policy_file=str(policy_file),
            model_name=model_name,
            reasoning_effort="medium",
        )
        result = await detector.detect(test_text)
        print(f"Violation: {result[0]}")
        print(f"Model: {result[1]['config']['model']}")


async def example_output_formats():
    print("\n" + "=" * 60)
    print("Example 4: Output Format Comparison")
    print("=" * 60)

    policy_file = (
        Path(__file__).parent.parent.parent
        / "detectors"
        / "policies"
        / "spam_policy.md"
    )

    test_text = "BUY NOW!!! FREE MONEY!!! CLICK HERE!!!"
    formats: list[OutputFormat] = ["binary", "policy_ref", "with_rationale"]
    for fmt in formats:
        print(f"\n--- Output format: {fmt} ---")
        detector = PolicyGptOssSafeguard(
            policy_file=str(policy_file),
            output_format=fmt,
            reasoning_effort="medium",
        )
        result = await detector.detect(test_text)
        print(f"Violation: {result[0]}")
        print(f"Parsed output: {result[1]['parsed_output']}")


async def example_with_dome_integration():
    print("\n" + "=" * 60)
    print("Example 5: Direct Usage Pattern (Recommended)")
    print("=" * 60)

    policy_file = (
        Path(__file__).parent.parent.parent
        / "detectors"
        / "policies"
        / "spam_policy.md"
    )

    print("\nNote: The Dome config parser currently supports these guard types:")
    print("  - security")
    print("  - moderation")
    print("  - privacy")
    print("  - integrity")
    print("\nDetectionCategory.Generic is not yet mapped in Dome config.")
    print("Recommended: Use direct detector instantiation (Examples 1-4 above).\n")

    detector = PolicyGptOssSafeguard(
        policy_file=str(policy_file),
        model_name="openai/gpt-oss-safeguard-20b",
        reasoning_effort="medium",
    )

    test_cases = [
        "What is your refund policy?",
        "BUY NOW BUY NOW BUY NOW!!!",
    ]

    for text in test_cases:
        result = await detector.detect(text)
        print(f"\nInput: {text}")
        print(f"Violation: {result[0]}")
        print(f"Model: {result[1]['config']['model']}")


async def main():
    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY environment variable not set")
        print("Please set your Groq API key before running this example:")
        print("  export GROQ_API_KEY='your-api-key-here'")
        return

    print("\n" + "=" * 60)
    print("GPT-OSS-Safeguard Policy Detector Examples")
    print("=" * 60)

    try:
        await example_factory_usage()
        await example_direct_instantiation()
        await example_different_models()
        await example_output_formats()
        await example_with_dome_integration()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
    except Exception as exc:
        print(f"\nError running examples: {exc}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
