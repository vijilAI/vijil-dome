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
Example: Policy Sections from S3 Integration

This example demonstrates how to load policy sections from S3 JSON files,
cache them locally, and use Dome to check inputs/outputs against all
sections in parallel with early exit.

Prerequisites:
- Set AWS credentials (via environment variables, IAM role, or AWS credentials file)
- Set NEBIUS_API_KEY environment variable (for PolicyGptOssSafeguard detector)
- Have policy sections JSON file in S3 at: teams/{team_id}/policies/{policy_id}/sections.json

The example shows:
- Loading policy sections from S3 (with automatic caching)
- Building Dome config from sections
- Using convenience factory methods
- Testing guardrails and identifying which sections triggered violations
"""

import asyncio
import os
from vijil_dome import Dome
from vijil_dome.utils.policy_loader import load_policy_sections_from_s3, load_policy_sections_from_file
from vijil_dome.utils.policy_config_builder import build_dome_config_from_sections


async def example_convenience_method_by_ids():
    """Example 1: Using convenience method with team_id and policy_id"""
    print("\n" + "="*60)
    print("Example 1: Convenience Method with team_id/policy_id")
    print("="*60)

    # Check for API key
    if not os.getenv("NEBIUS_API_KEY"):
        print("WARNING: NEBIUS_API_KEY not set. Policy detectors will fail.")
        print("Set it with: export NEBIUS_API_KEY='your-api-key'")
        return

    try:
        # Create Dome instance using convenience method
        # This constructs the S3 path: teams/{team_id}/policies/{policy_id}/sections.json
        dome = Dome.create_from_s3_policy_by_ids(
            bucket="my-policy-bucket",  # Replace with your bucket
            team_id="550e8400-e29b-41d4-a716-446655440000",
            policy_id="123e4567-e89b-12d3-a456-426614174000",
            model_name="openai/gpt-oss-120b",
            reasoning_effort="medium"
        )

        print("✓ Dome instance created successfully")
        print(f"✓ Input guardrail: {dome.input_guardrail is not None}")
        print(f"✓ Output guardrail: {dome.output_guardrail is not None}")

        # Test input guardrail
        test_inputs = [
            "What is your refund policy?",
            "My SSN is 123-45-6789",  # Should trigger PII policy
            "How can I reset my password?",
        ]

        print("\nTesting input guardrails:")
        for test_input in test_inputs:
            result = await dome.async_guard_input(test_input)
            print(f"\nInput: '{test_input}'")
            print(f"Flagged: {result.flagged}")
            if result.flagged:
                print(f"Response: {result.response_string}")
                # Check which sections triggered
                for guard_name, guard_res in result.guard_exec_details.items():
                    for detector_name, det_res in guard_res.details.items():
                        if det_res.hit:
                            section_id = det_res.result.get("section_id", detector_name)
                            print(f"  → Triggered by section: {section_id}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


async def example_convenience_method_full_path():
    """Example 2: Using convenience method with full S3 key path"""
    print("\n" + "="*60)
    print("Example 2: Convenience Method with Full S3 Path")
    print("="*60)

    if not os.getenv("NEBIUS_API_KEY"):
        print("WARNING: NEBIUS_API_KEY not set.")
        return

    try:
        dome = Dome.create_from_s3_policy(
            bucket="my-policy-bucket",
            key="teams/550e8400-e29b-41d4-a716-446655440000/policies/123e4567-e89b-12d3-a456-426614174000/sections.json",
            model_name="openai/gpt-oss-120b",
            reasoning_effort="medium"
        )

        print("✓ Dome instance created successfully")

        # Test output guardrail
        test_outputs = [
            "I'd be happy to help you with that!",
            "You're an idiot and I hate you!",  # Should trigger moderation policy
            "Here's the information you requested.",
        ]

        print("\nTesting output guardrails:")
        for test_output in test_outputs:
            result = await dome.async_guard_output(test_output)
            print(f"\nOutput: '{test_output[:50]}...'")
            print(f"Flagged: {result.flagged}")
            if result.flagged:
                print(f"Response: {result.response_string}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


async def example_manual_steps():
    """Example 3: Manual steps - load, build config, create Dome"""
    print("\n" + "="*60)
    print("Example 3: Manual Steps")
    print("="*60)

    if not os.getenv("NEBIUS_API_KEY"):
        print("WARNING: NEBIUS_API_KEY not set.")
        return

    try:
        # Step 1: Load policy sections from S3
        print("Step 1: Loading policy sections from S3...")
        policy_data = load_policy_sections_from_s3(
            bucket="my-policy-bucket",
            key="teams/550e8400-e29b-41d4-a716-446655440000/policies/123e4567-e89b-12d3-a456-426614174000/sections.json"
        )
        print(f"✓ Loaded {len(policy_data['sections'])} sections")
        print(f"  Policy: {policy_data.get('policy_name', 'Unknown')}")
        print(f"  Policy ID: {policy_data.get('policy_id', 'Unknown')}")

        # Step 2: Build Dome config
        print("\nStep 2: Building Dome config...")
        config = build_dome_config_from_sections(
            policy_data,
            model_name="openai/gpt-oss-120b",
            reasoning_effort="medium"
        )
        print(f"✓ Config built")
        print(f"  Input guards: {len(config.get('input-guards', []))}")
        print(f"  Output guards: {len(config.get('output-guards', []))}")

        # Step 3: Create Dome instance
        print("\nStep 3: Creating Dome instance...")
        dome = Dome(config)
        print("✓ Dome instance created")

        # Step 4: Use guardrails
        print("\nStep 4: Testing guardrails...")
        result = await dome.async_guard_input("Test query")
        print(f"✓ Test completed - Flagged: {result.flagged}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


async def example_local_file():
    """Example 4: Loading from local file (for testing/development)"""
    print("\n" + "="*60)
    print("Example 4: Loading from Local File")
    print("="*60)

    if not os.getenv("NEBIUS_API_KEY"):
        print("WARNING: NEBIUS_API_KEY not set.")
        return

    try:
        # Create a sample policy JSON file for testing
        sample_policy = {
            "version": "1.0",
            "last_updated": "2025-01-15T10:30:00Z",
            "etag": "test-etag",
            "policy_id": "test-policy-123",
            "policy_name": "Test Policy",
            "source_file": "test_policy.pdf",
            "sections": [
                {
                    "section_id": "test-section-1",
                    "content": "# Test Policy\n\n## INSTRUCTIONS\nCheck if input contains test violations.\nReturn ONLY a single digit:\n- 0 = No violation\n- 1 = Violation detected\n\n## VIOLATES (1)\n- Contains word 'test-violation'\n\n## SAFE (0)\n- Normal queries\n\nContent: [INPUT]\nAnswer:",
                    "applies_to": ["input"],
                    "metadata": {
                        "header": "Test Section",
                        "level": 1,
                        "order": 1
                    }
                }
            ]
        }

        import json
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_policy, f, indent=2)
            temp_file = f.name

        try:
            # Load from local file
            print(f"Loading from local file: {temp_file}")
            policy_data = load_policy_sections_from_file(temp_file)
            print(f"✓ Loaded {len(policy_data['sections'])} sections")

            # Build config and create Dome
            config = build_dome_config_from_sections(
                policy_data,
                model_name="openai/gpt-oss-120b",
                reasoning_effort="medium"
            )
            dome = Dome(config)

            # Test
            result = await dome.async_guard_input("This is a test-violation query")
            print(f"\nTest result:")
            print(f"  Input: 'This is a test-violation query'")
            print(f"  Flagged: {result.flagged}")
            if result.flagged:
                print(f"  Response: {result.response_string}")

        finally:
            # Cleanup
            os.unlink(temp_file)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


async def example_cache_behavior():
    """Example 5: Demonstrating cache behavior"""
    print("\n" + "="*60)
    print("Example 5: Cache Behavior")
    print("="*60)

    if not os.getenv("NEBIUS_API_KEY"):
        print("WARNING: NEBIUS_API_KEY not set.")
        return

    try:
        bucket = "my-policy-bucket"
        key = "teams/550e8400-e29b-41d4-a716-446655440000/policies/123e4567-e89b-12d3-a456-426614174000/sections.json"

        print("First load (will download from S3):")
        import time
        start = time.time()
        policy_data1 = load_policy_sections_from_s3(bucket, key)
        elapsed1 = time.time() - start
        print(f"  Time: {elapsed1:.2f}s")
        print(f"  Sections: {len(policy_data1['sections'])}")

        print("\nSecond load (should use cache):")
        start = time.time()
        policy_data2 = load_policy_sections_from_s3(bucket, key)
        elapsed2 = time.time() - start
        print(f"  Time: {elapsed2:.2f}s")
        print(f"  Sections: {len(policy_data2['sections'])}")
        print(f"  Cache speedup: {elapsed1/elapsed2:.1f}x faster")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("Policy Sections from S3 Integration Examples")
    print("="*60)

    # Note: These examples require actual S3 access and API keys
    # Uncomment the examples you want to run:

    # await example_convenience_method_by_ids()
    # await example_convenience_method_full_path()
    # await example_manual_steps()
    # await example_local_file()
    # await example_cache_behavior()

    print("\n" + "="*60)
    print("Note: Examples are commented out. Uncomment to run.")
    print("Make sure to:")
    print("  1. Set NEBIUS_API_KEY environment variable")
    print("  2. Configure AWS credentials")
    print("  3. Update bucket/team_id/policy_id values")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
