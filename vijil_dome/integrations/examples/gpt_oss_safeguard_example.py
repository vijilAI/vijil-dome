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
model via the Nebius gateway.

Prerequisites:
- Set GROQ_API_KEY environment variable
- Create a policy file (see vijil_dome/detectors/policies/spam_policy.md for example)

The detector supports:
- Custom policy files defining violation criteria
- Two model variants: openai/gpt-oss-safeguard-120b (more accurate) and openai/gpt-oss-safeguard-20b (faster)
- Configurable reasoning effort: low, medium, high
"""

import os
import asyncio
from pathlib import Path

from vijil_dome.detectors import (
    POLICY_GPT_OSS_SAFEGUARD,
    DetectionFactory,
    DetectionCategory,
)
from vijil_dome.detectors.methods.gpt_oss_safeguard_policy import PolicyGptOssSafeguard


async def example_factory_usage():
    """Example 1: Using DetectionFactory to create detector"""
    print("\n" + "="*60)
    print("Example 1: Factory Pattern Usage")
    print("="*60)

    # Path to policy file
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

    # Test with various inputs
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
        print(f"Model: {result.result['model']}")


async def example_direct_instantiation():
    """Example 2: Direct instantiation with custom parameters"""
    print("\n" + "="*60)
    print("Example 2: Direct Instantiation")
    print("="*60)

    policy_file = (
        Path(__file__).parent.parent.parent
        / "detectors"
        / "policies"
        / "spam_policy.md"
    )

    # Create detector with custom configuration
    detector = PolicyGptOssSafeguard(
        policy_file=str(policy_file),
        hub_name="groq",
        model_name="openai/gpt-oss-safeguard-20b",
        reasoning_effort="high",  # Use high reasoning for better accuracy
        timeout=90,
        max_retries=3
    )

    test_text = "JOIN NOW!!! LIMITED TIME OFFER!!! CLICK HERE!!!"
    result = await detector.detect(test_text)

    print(f"\nInput: {test_text}")
    print(f"Violation detected: {result[0]}")
    print(f"Model response: {result[1]['model_response'][:200]}...")
    print(f"Policy source: {result[1]['policy_source']}")
    print(f"Reasoning effort: {result[1]['reasoning_effort']}")


async def example_different_models():
    """Example 3: Comparing 120B vs 20B models"""
    print("\n" + "="*60)
    print("Example 3: Model Comparison (120B vs 20B)")
    print("="*60)

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
            reasoning_effort="medium"
        )

        result = await detector.detect(test_text)
        print(f"Violation: {result[0]}")
        print(f"Model: {result[1]['model']}")


async def example_reasoning_efforts():
    """Example 4: Different reasoning effort levels"""
    print("\n" + "="*60)
    print("Example 4: Reasoning Effort Levels")
    print("="*60)

    policy_file = (
        Path(__file__).parent.parent.parent
        / "detectors"
        / "policies"
        / "spam_policy.md"
    )

    test_text = "Special discount available - contact us for details"

    for effort in ["low", "medium", "high"]:
        print(f"\n--- Reasoning effort: {effort} ---")

        detector = PolicyGptOssSafeguard(
            policy_file=str(policy_file),
            reasoning_effort=effort
        )

        result = await detector.detect(test_text)
        print(f"Violation: {result[0]}")
        print(f"Reasoning effort: {result[1]['reasoning_effort']}")


async def example_with_dome_integration():
    """Example 5: Direct Usage Recommendation (Dome config limitation)"""
    print("\n" + "="*60)
    print("Example 5: Direct Usage Pattern (Recommended)")
    print("="*60)

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

    # Demonstrate direct usage pattern
    detector = PolicyGptOssSafeguard(
        policy_file=str(policy_file),
        model_name="openai/gpt-oss-safeguard-20b",
        reasoning_effort="medium"
    )

    test_cases = [
        "What is your refund policy?",
        "BUY NOW BUY NOW BUY NOW!!!",
    ]

    for text in test_cases:
        result = await detector.detect(text)
        print(f"\nInput: {text}")
        print(f"Violation: {result[0]}")
        print(f"Model: {result[1]['model']}")


async def main():
    """Run all examples"""
    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY environment variable not set")
        print("Please set your Groq API key before running this example:")
        print("  export GROQ_API_KEY='your-api-key-here'")
        return

    print("\n" + "="*60)
    print("GPT-OSS-Safeguard Policy Detector Examples")
    print("="*60)

    try:
        # Run examples
        await example_factory_usage()
        await example_direct_instantiation()
        await example_different_models()
        await example_reasoning_efforts()
        await example_with_dome_integration()

        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
