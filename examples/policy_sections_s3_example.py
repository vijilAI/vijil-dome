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
# vijil and vijil-dome are trademarks of Vijil Inc.

"""
Example: Policy Sections Detector with S3 Support

This example demonstrates how to use PolicySectionsDetector via config-driven
approach to load policy sections from S3, cache them locally, and check
inputs/outputs against all sections in parallel with early exit.

Prerequisites:
- Set AWS credentials (via environment variables, IAM role, or AWS credentials file)
- Set NEBIUS_API_KEY environment variable (for PolicyGptOssSafeguard detector)
- Have policy sections JSON file in S3 at: teams/{team_id}/policies/{policy_id}/sections.json
- See vijil_dome/detectors/policies/sample_policy_sections.json for format reference

Rationale:
Policies should be split into 400-600 token sections for optimal performance.
Each section is evaluated independently by PolicyGptOssSafeguard, allowing
parallel execution with fast fail. This is more efficient than dumping everything
into a single policy violation detector.

The example shows:
- Config-driven approach with policy_s3_bucket and policy_s3_key
- Rate limiting with max_parallel_sections
- S3 authentication parameters
- Testing guardrails and identifying which sections triggered violations
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from vijil_dome import Dome
from vijil_dome.utils.policy_loader import load_policy_sections_from_file


async def example_s3_config():
    """Example 1: Using PolicySectionsDetector with S3 via config"""
    print("\n" + "="*60)
    print("Example 1: PolicySectionsDetector with S3 Config")
    print("="*60)

    if not os.getenv("NEBIUS_API_KEY"):
        print("WARNING: NEBIUS_API_KEY not set. Policy detectors will fail.")
        print("Set it with: export NEBIUS_API_KEY='your-api-key'")
        return

    try:
        # Config-driven approach - S3 loading happens in detector
        config = {
            "input-guards": ["policy-input"],
            "policy-input": {
                "type": "policy",
                "method": "policy-sections",
                "policy_s3_bucket": "my-policy-bucket",  # Replace with your bucket
                "policy_s3_key": "teams/550e8400-e29b-41d4-a716-446655440000/policies/123e4567-e89b-12d3-a456-426614174000/sections.json",
                "applies_to": "input",
                "max_parallel_sections": 10,  # Rate limiting
                "model_name": "openai/gpt-oss-120b",
                "reasoning_effort": "medium",
                "hub_name": "nebius",
                "timeout": 60,
                "max_retries": 3,
                # S3 auth params (optional - uses boto3 defaults if not provided)
                # "aws_access_key_id": "...",
                # "aws_secret_access_key": "...",
                # "aws_session_token": "...",
                # "region_name": "...",
            }
        }

        dome = Dome(config)
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
                for guard_name, guard_res in result.trace.items():
                    for detector_name, det_res in guard_res.details.items():
                        if det_res.hit:
                            section_info = det_res.result.get("sections", [])
                            violating_section = det_res.result.get("violating_section")
                            if violating_section:
                                print(f"  → Triggered by section: {violating_section.get('section_id')}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


async def example_rate_limiting():
    """Example 2: Demonstrating max_parallel_sections for rate limiting"""
    print("\n" + "="*60)
    print("Example 2: Rate Limiting with max_parallel_sections")
    print("="*60)

    if not os.getenv("NEBIUS_API_KEY"):
        print("WARNING: NEBIUS_API_KEY not set.")
        return

    try:
        config = {
            "input-guards": ["policy-input"],
            "policy-input": {
                "type": "policy",
                "method": "policy-sections",
                "policy_s3_bucket": "my-policy-bucket",
                "policy_s3_key": "teams/team-123/policies/policy-456/sections.json",
                "applies_to": "input",
                "max_parallel_sections": 5,  # Limit to 5 concurrent LLM calls
                "model_name": "openai/gpt-oss-120b",
                "reasoning_effort": "medium",
                "hub_name": "nebius",
            }
        }

        dome = Dome(config)
        print("✓ Dome instance created with max_parallel_sections=5")
        print("  This limits concurrent LLM calls to avoid rate limits")
        print("  Sections will be processed in batches of 5")

        result = await dome.async_guard_input("Test query")
        print(f"\nTest completed - Flagged: {result.flagged}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


async def example_local_file():
    """Example 3: Loading from local file (for testing/development)"""
    print("\n" + "="*60)
    print("Example 3: Loading from Local File")
    print("="*60)

    if not os.getenv("NEBIUS_API_KEY"):
        print("WARNING: NEBIUS_API_KEY not set.")
        return

    try:
        # Load sample sections file
        sample_file = Path(__file__).parent.parent / "vijil_dome" / "detectors" / "policies" / "sample_policy_sections.json"
        
        if not sample_file.exists():
            print(f"Sample file not found: {sample_file}")
            return

        policy_data = load_policy_sections_from_file(str(sample_file))
        print(f"✓ Loaded {len(policy_data['sections'])} sections from {sample_file}")

        # Create config with direct sections
        # Note: PolicySectionsDetector accepts policy_sections directly
        # For this example, we'll use the config builder approach
        from vijil_dome.utils.policy_config_builder import build_dome_config_from_sections

        config = build_dome_config_from_sections(
            policy_data,
            model_name="openai/gpt-oss-120b",
            reasoning_effort="medium"
        )

        dome = Dome(config)
        print("✓ Dome instance created")

        # Test
        result = await dome.async_guard_input("This is a test-violation query")
        print(f"\nTest result:")
        print(f"  Input: 'This is a test-violation query'")
        print(f"  Flagged: {result.flagged}")
        if result.flagged:
            print(f"  Response: {result.response_string}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


async def example_s3_auth():
    """Example 4: Using S3 authentication parameters"""
    print("\n" + "="*60)
    print("Example 4: S3 Authentication Parameters")
    print("="*60)

    if not os.getenv("NEBIUS_API_KEY"):
        print("WARNING: NEBIUS_API_KEY not set.")
        return

    try:
        # Config with explicit S3 auth params
        config = {
            "input-guards": ["policy-input"],
            "policy-input": {
                "type": "policy",
                "method": "policy-sections",
                "policy_s3_bucket": "my-policy-bucket",
                "policy_s3_key": "teams/team-123/policies/policy-456/sections.json",
                "applies_to": "input",
                "model_name": "openai/gpt-oss-120b",
                "reasoning_effort": "medium",
                "hub_name": "nebius",
                # S3 auth params
                "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
                "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
                "region_name": os.getenv("AWS_REGION", "us-east-1"),
            }
        }

        dome = Dome(config)
        print("✓ Dome instance created with explicit S3 credentials")
        print("  Note: If not provided, boto3 uses default credentials")

        result = await dome.async_guard_input("Test query")
        print(f"\nTest completed - Flagged: {result.flagged}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("Policy Sections Detector Examples")
    print("="*60)

    # Run examples (uncomment to run)
    # await example_s3_config()
    # await example_rate_limiting()
    # await example_local_file()
    # await example_s3_auth()

    print("\n" + "="*60)
    print("Note: Examples are commented out. Uncomment to run.")
    print("Make sure to:")
    print("  1. Set NEBIUS_API_KEY environment variable")
    print("  2. Configure AWS credentials")
    print("  3. Update bucket/team_id/policy_id values in examples")
    print("  4. See vijil_dome/detectors/policies/sample_policy_sections.json for format")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
