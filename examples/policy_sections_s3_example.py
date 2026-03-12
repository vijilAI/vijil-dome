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
Example: Policy Guardrail Modes

This example demonstrates three different modes for running policy guardrails:

1. **Full Policy Mode** - Single detector with entire policy
   - Best for small policies (<400 tokens)
   - Uses PolicyGptOssSafeguard directly
   - Single LLM call per query

2. **Chunked Parallel Mode** - Multiple sections evaluated in parallel
   - Best for large policies split into 400-600 token sections
   - Uses PolicySectionsDetector with use_rag=False
   - All sections evaluated in parallel batches with fast fail
   - More efficient than dumping everything into a single detector

3. **RAG/VectorDB Mode** - Retrieval-first approach with FAISS
   - Best for very large policies (100+ sections)
   - Uses PolicySectionsDetector with use_rag=True
   - Retrieves only relevant sections using FAISS similarity search
   - Evaluates only retrieved sections, reducing LLM calls

Prerequisites:
- Set AWS credentials (via environment variables, IAM role, or AWS credentials file)
- Set OPENAI_API_KEY or GROQ_API_KEY environment variable (for PolicyGptOssSafeguard detector)
- For S3 examples: Set POLICY_S3_BUCKET and POLICY_S3_KEY_PREFIX environment variables
  Example:
    export POLICY_S3_BUCKET="my-policy-bucket"
    export POLICY_S3_KEY_PREFIX="teams/team-123/policies/policy-456"
- For RAG mode: Have FAISS index and section_ids.json in same S3 directory
- See vijil_dome/detectors/policies/sample_policy_sections.json for format reference
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Load .env file if it exists
try:
    from dotenv import load_dotenv  # type: ignore[import-untyped]
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from vijil_dome import Dome
from vijil_dome.utils.policy_loader import load_policy_sections_from_file

# S3 Configuration - Set these via environment variables
# Example: export POLICY_S3_BUCKET="my-policy-bucket"
#          export POLICY_S3_KEY_PREFIX="teams/team-123/policies/policy-456"
POLICY_S3_BUCKET = os.getenv("POLICY_S3_BUCKET", "my-policy-bucket")
POLICY_S3_KEY_PREFIX = os.getenv("POLICY_S3_KEY_PREFIX", "teams/team-123/policies/policy-456")
# Normalize prefix to remove trailing slash to avoid double slashes
POLICY_S3_KEY_PREFIX = POLICY_S3_KEY_PREFIX.rstrip("/")

# Warn if using default values (for public repo safety)
if POLICY_S3_BUCKET == "my-policy-bucket" or POLICY_S3_KEY_PREFIX == "teams/team-123/policies/policy-456":
    print(
        "Using default S3 bucket/key values. Set POLICY_S3_BUCKET and POLICY_S3_KEY_PREFIX "
        "environment variables to use your own S3 resources."
    )


async def example_full_policy_mode():
    """Example 1: Full Policy Mode - Single detector with entire policy"""
    print("""\

============================================================
Example 1: Full Policy Mode
============================================================
Best for: Small policies (<400 tokens)
Mode: Single detector evaluates entire policy in one LLM call
""")

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY or GROQ_API_KEY not set.")
        print("Set it with: export OPENAI_API_KEY='your-api-key'")
        print("  or: export GROQ_API_KEY='your-api-key'")
        return
    
    print(f"Using API key: {'OPENAI_API_KEY' if os.getenv('OPENAI_API_KEY') else 'GROQ_API_KEY'}")

    try:
        # Create a sample policy file for demonstration
        sample_policy = """# Customer Support Policy

## INSTRUCTIONS
Classify whether an AI agent response violates customer support guidelines.

## VIOLATES (1)
- Shares internal pricing, discount codes, or employee-only information
- Makes promises about refunds or credits without authorization
- Provides legal, medical, or financial advice
- Uses aggressive, dismissive, or unprofessional language

## SAFE (0)
- Answers product questions using public documentation
- Directs customers to appropriate support channels
- Provides general troubleshooting steps
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(sample_policy)
            policy_path = f.name

        try:
            # Full Policy Mode config - uses type: "generic" with policy-gpt-oss-safeguard
            config = {
                "input-guards": ["policy-check"],
                "output-guards": [],
                "policy-check": {
                    "type": "generic",
                    "methods": ["policy-gpt-oss-safeguard"],
                    "policy-gpt-oss-safeguard": {
                        "policy_file": policy_path,
                        "hub_name": "groq",
                        "model_name": "openai/gpt-oss-safeguard-20b",
                        "output_format": "policy_ref",
                        "reasoning_effort": "medium",
                    },
                },
            }

            print("Creating Dome instance (Full Policy Mode)...")
            dome = Dome(config)
            print("✓ Dome instance created successfully (Full Policy Mode)")
            print(f"✓ Input guardrail: {dome.input_guardrail is not None}")
            print(f"✓ Output guardrail: {dome.output_guardrail is not None}")

            # Test
            test_inputs = [
                "User request: Can I get a discount?\nAgent response: Sure! Use code EMPLOYEE50 for 50% off.",
                "User request: How do I reset my password?\nAgent response: You can reset it at our help center.",
            ]

            print("\nTesting input guardrails:")
            for i, test_input in enumerate(test_inputs, 1):
                print(f"\n[{i}/{len(test_inputs)}] Testing input: '{test_input[:50]}...'")
                result = await dome.async_guard_input(test_input)
                print(f"  Result: {'FLAGGED' if result.flagged else 'ALLOWED'}")
                if result.flagged:
                    print(f"  Response: {result.response_string[:100]}...")

        finally:
            os.unlink(policy_path)

    except Exception as e:
        print(f"Error in example: {e}", exc_info=True)


async def example_chunked_parallel_mode():
    """Example 2: Chunked Parallel Mode - All sections evaluated in parallel"""
    print("""\

============================================================
Example 2: Chunked Parallel Mode
============================================================
Best for: Large policies split into 400-600 token sections
Mode: All sections evaluated in parallel batches with fast fail
""")

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY or GROQ_API_KEY not set.")
        return
    
    print(f"Using API key: {'OPENAI_API_KEY' if os.getenv('OPENAI_API_KEY') else 'GROQ_API_KEY'}")
    print("Checking AWS credentials...")
    aws_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_profile = os.getenv("AWS_PROFILE")
    if aws_key:
        print("AWS credentials found: Access Key ID set")
    elif aws_profile:
        print(f"AWS credentials found: Profile '{aws_profile}'")
    else:
        print("No AWS credentials found. Will use boto3 defaults (IAM role, etc.)")

    try:
        # Chunked Parallel Mode config - uses type: "policy" with policy-sections
        # use_rag=False (default) means all sections are evaluated
        config = {
            "input-guards": ["policy-input"],
            "policy-input": {
                "type": "policy",
                "methods": ["policy-sections"],
                "policy-sections": {
                    "policy_s3_bucket": POLICY_S3_BUCKET,
                    "policy_s3_key": f"{POLICY_S3_KEY_PREFIX}/sections.json",
                    "applies_to": "input",
                    "use_rag": False,  # Explicitly disable RAG - evaluate all sections
                    "max_parallel_sections": 10,  # Rate limiting: max concurrent LLM calls
                    "model_name": "openai/gpt-oss-safeguard-20b",
                    "reasoning_effort": "medium",
                    "hub_name": "groq",
                    "timeout": 60,
                    "max_retries": 3,
                    # S3 auth params - pass explicitly to avoid profile conflicts
                    "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
                    "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
                    "aws_session_token": os.getenv("AWS_SESSION_TOKEN"),
                    "region_name": os.getenv("AWS_REGION"),
                },
            }
        }

        print("Creating Dome instance (Chunked Parallel Mode)...")
        dome = Dome(config)
        print("✓ Dome instance created successfully (Chunked Parallel Mode)")
        print(f"✓ Input guardrail: {dome.input_guardrail is not None}")
        print("✓ All sections will be evaluated in parallel batches of 10")

        # Test input guardrail
        test_inputs = [
            "What is your refund policy?",
            "My SSN is 123-45-6789",  # Should trigger PII policy
            "How can I reset my password?",
        ]

        print("\nTesting input guardrails:")
        for i, test_input in enumerate(test_inputs, 1):
            print(f"\n[{i}/{len(test_inputs)}] Testing input: '{test_input}'")
            result = await dome.async_guard_input(test_input)
            print(f"  Result: {'FLAGGED' if result.flagged else 'ALLOWED'}")
            if result.flagged:
                print(f"  Response: {result.response_string[:100]}...")
                # Check which sections triggered
                for guard_name, guard_res in result.trace.items():
                    for detector_name, det_res in guard_res.details.items():
                        if det_res.hit:
                            violating_section = det_res.result.get("violating_section")
                            if violating_section:
                                print(f"  → Triggered by section: {violating_section.get('section_id')}")

    except Exception as e:
        print(f"Error in example: {e}", exc_info=True)


async def example_rag_mode():
    """Example 3: RAG/VectorDB Mode - Retrieval-first approach with FAISS"""
    print("""\

============================================================
Example 3: RAG/VectorDB Mode
============================================================
Best for: Very large policies (100+ sections)
Mode: Retrieves only relevant sections using FAISS, then evaluates
""")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set (required for RAG mode).")
        print("Set it with: export OPENAI_API_KEY='your-api-key'")
        return
    
    print("Using OPENAI_API_KEY for RAG mode")
    print("Checking AWS credentials for S3 access...")
    aws_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_profile = os.getenv("AWS_PROFILE")
    if aws_key:
        print("AWS credentials found: Access Key ID set")
    elif aws_profile:
        print(f"AWS credentials found: Profile '{aws_profile}'")
    else:
        print("No AWS credentials found. Will use boto3 defaults (IAM role, etc.)")

    try:
        # RAG Mode config - uses type: "policy" with policy-sections
        # use_rag=True enables retrieval-first approach
        config = {
            "input-guards": ["policy-input"],
            "policy-input": {
                "type": "policy",
                "methods": ["policy-sections"],
                "policy-sections": {
                    "policy_s3_bucket": POLICY_S3_BUCKET,
                    "policy_s3_key": f"{POLICY_S3_KEY_PREFIX}/sections.json",
                    "applies_to": "input",
                    "use_rag": True,  # Enable RAG - retrieve relevant sections first
                    "faiss_s3_key": f"{POLICY_S3_KEY_PREFIX}/faiss.index",
                    "section_ids_s3_key": f"{POLICY_S3_KEY_PREFIX}/section_ids.json",
                    "top_k": 5,  # Retrieve top 5 most relevant sections
                    "similarity_threshold": 0.7,  # Minimum similarity score to include
                    "embedding_model": "text-embedding-ada-002",  # OpenAI embedding model
                    "embedding_engine": "OpenAI",  # Embedding engine
                    "model_name": "openai/gpt-oss-safeguard-20b",
                    "reasoning_effort": "medium",
                    "hub_name": "groq",
                    "timeout": 60,
                    "max_retries": 3,
                    # S3 auth params - pass explicitly to avoid profile conflicts
                    "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
                    "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
                    "aws_session_token": os.getenv("AWS_SESSION_TOKEN"),
                    "region_name": os.getenv("AWS_REGION"),
                },
            }
        }

        print("Creating Dome instance (RAG Mode)...")
        print("  Loading FAISS index and section IDs from S3...")
        dome = Dome(config)
        print("✓ Dome instance created successfully (RAG Mode)")
        print(f"✓ Input guardrail: {dome.input_guardrail is not None}")
        print("✓ RAG will retrieve top 5 relevant sections before evaluation")

        # Test input guardrail
        test_inputs = [
            "What is your refund policy?",
            "My credit card number is 4532-1234-5678-9010",  # Should trigger PII policy
            "How can I contact customer support?",
        ]

        print("\nTesting input guardrails:")
        for i, test_input in enumerate(test_inputs, 1):
            print(f"\n[{i}/{len(test_inputs)}] Testing input: '{test_input}'")
            result = await dome.async_guard_input(test_input)
            print(f"  Result: {'FLAGGED' if result.flagged else 'ALLOWED'}")
            if result.flagged:
                print(f"  Response: {result.response_string[:100]}...")
                # Check RAG retrieval info
                for guard_name, guard_res in result.trace.items():
                    for detector_name, det_res in guard_res.details.items():
                        if det_res.hit:
                            rag_info = det_res.result.get("rag_info")
                            if rag_info:
                                print(f"  → RAG retrieved {rag_info.get('retrieved_sections')} sections")
                                print(f"  → Total sections: {rag_info.get('total_sections')}")

    except Exception as e:
        print(f"Error in example: {e}", exc_info=True)


async def example_rate_limiting():
    """Example 4: Rate limiting with max_parallel_sections"""
    print("""\

============================================================
Example 4: Rate Limiting with max_parallel_sections
============================================================
Demonstrates how to limit concurrent LLM calls to avoid rate limits
""")

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY or GROQ_API_KEY not set.")
        return
    
    print(f"Using API key: {'OPENAI_API_KEY' if os.getenv('OPENAI_API_KEY') else 'GROQ_API_KEY'}")
    print("Checking AWS credentials...")
    aws_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_profile = os.getenv("AWS_PROFILE")
    if aws_key:
        print("AWS credentials found: Access Key ID set")
    elif aws_profile:
        print(f"AWS credentials found: Profile '{aws_profile}'")
    else:
        print("No AWS credentials found. Will use boto3 defaults (IAM role, etc.)")

    try:
        config = {
            "input-guards": ["policy-input"],
            "policy-input": {
                "type": "policy",
                "methods": ["policy-sections"],
                "policy-sections": {
                    "policy_s3_bucket": POLICY_S3_BUCKET,
                    "policy_s3_key": f"{POLICY_S3_KEY_PREFIX}/sections.json",
                    "applies_to": "input",
                    "use_rag": False,
                    "max_parallel_sections": 5,  # Limit to 5 concurrent LLM calls
                    "model_name": "openai/gpt-oss-safeguard-20b",
                    "reasoning_effort": "medium",
                    "hub_name": "groq",
                    # S3 auth params - pass explicitly to avoid profile conflicts
                    "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
                    "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
                    "aws_session_token": os.getenv("AWS_SESSION_TOKEN"),
                    "region_name": os.getenv("AWS_REGION"),
                },
            }
        }

        print("Creating Dome instance with max_parallel_sections=5...")
        dome = Dome(config)
        print("✓ Dome instance created with max_parallel_sections=5")
        print("  This limits concurrent LLM calls to avoid rate limits")
        print("  Sections will be processed in batches of 5")

        print("\nTesting with sample query...")
        result = await dome.async_guard_input("Test query")
        print(f"Test completed - Result: {'FLAGGED' if result.flagged else 'ALLOWED'}")

    except Exception as e:
        print(f"Error in example: {e}", exc_info=True)


async def example_s3_auth():
    """Example 5: Using S3 authentication parameters"""
    print("""\

============================================================
Example 5: S3 Authentication Parameters
============================================================
Demonstrates explicit S3 credential configuration
""")

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY or GROQ_API_KEY not set.")
        return
    
    print(f"Using API key: {'OPENAI_API_KEY' if os.getenv('OPENAI_API_KEY') else 'GROQ_API_KEY'}")
    print("Checking AWS credentials...")
    aws_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_profile = os.getenv("AWS_PROFILE")
    if aws_key:
        print("AWS credentials found: Access Key ID set")
    elif aws_profile:
        print(f"AWS credentials found: Profile '{aws_profile}'")
    else:
        print("No AWS credentials found. Will use boto3 defaults (IAM role, etc.)")

    try:
        # Config with explicit S3 auth params
        config = {
            "input-guards": ["policy-input"],
            "policy-input": {
                "type": "policy",
                "methods": ["policy-sections"],
                "policy-sections": {
                    "policy_s3_bucket": POLICY_S3_BUCKET,
                    "policy_s3_key": f"{POLICY_S3_KEY_PREFIX}/sections.json",
                    "applies_to": "input",
                    "use_rag": False,
                    "model_name": "openai/gpt-oss-safeguard-20b",
                    "reasoning_effort": "medium",
                    "hub_name": "groq",
                    # S3 auth params
                    "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
                    "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
                    "region_name": os.getenv("AWS_REGION", "us-east-1"),
                },
            }
        }

        print("Creating Dome instance with explicit S3 credentials...")
        dome = Dome(config)
        print("✓ Dome instance created with explicit S3 credentials")
        print("  Note: If not provided, boto3 uses default credentials")

        print("\nTesting with sample query...")
        result = await dome.async_guard_input("Test query")
        print(f"Test completed - Result: {'FLAGGED' if result.flagged else 'ALLOWED'}")

    except Exception as e:
        print(f"Error in example: {e}", exc_info=True)


async def example_local_file():
    """Example 6: Loading from local file (for testing/development)"""
    print("""\

============================================================
Example 6: Loading from Local File
============================================================
Demonstrates using local policy sections file for testing
""")

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY or GROQ_API_KEY not set.")
        return
    
    print(f"Using API key: {'OPENAI_API_KEY' if os.getenv('OPENAI_API_KEY') else 'GROQ_API_KEY'}")
    print("Checking AWS credentials...")
    aws_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_profile = os.getenv("AWS_PROFILE")
    if aws_key:
        print("AWS credentials found: Access Key ID set")
    elif aws_profile:
        print(f"AWS credentials found: Profile '{aws_profile}'")
    else:
        print("No AWS credentials found. Will use boto3 defaults (IAM role, etc.)")

    try:
        # Load sample sections file
        sample_file = Path(__file__).parent.parent / "vijil_dome" / "detectors" / "policies" / "sample_policy_sections.json"
        
        if not sample_file.exists():
            print(f"Sample file not found: {sample_file}")
            return

        print(f"Loading policy sections from {sample_file}...")
        policy_data = load_policy_sections_from_file(str(sample_file))
        print(f"✓ Loaded {len(policy_data['sections'])} sections from {sample_file.name}")

        # Create config with direct sections using config builder
        from vijil_dome.utils.policy_config_builder import build_dome_config_from_sections

        print("Building Dome config from sections...")
        config = build_dome_config_from_sections(
            policy_data,
            model_name="openai/gpt-oss-safeguard-20b",
            reasoning_effort="medium",
            hub_name="groq",
        )

        print("Creating Dome instance...")
        dome = Dome(config)
        print("✓ Dome instance created")

        # Test
        test_query = "This is a test-violation query"
        print(f"\nTesting with query: '{test_query}'")
        result = await dome.async_guard_input(test_query)
        print("\nTest result:")
        print(f"  Input: '{test_query}'")
        print(f"  Result: {'FLAGGED' if result.flagged else 'ALLOWED'}")
        if result.flagged:
            print(f"  Response: {result.response_string[:100]}...")

    except Exception as e:
        print(f"Error in example: {e}", exc_info=True)


async def main():
    """Run all examples"""
    print("""\
============================================================
Policy Guardrail Examples - All Three Modes
============================================================

This example demonstrates three policy guardrail modes:
  1. Full Policy Mode - Single detector with entire policy
  2. Chunked Parallel Mode - All sections evaluated in parallel
  3. RAG/VectorDB Mode - Retrieval-first with FAISS

============================================================
""")

    # Run examples (comment out any you don't want to run)
    await example_full_policy_mode()
    await example_chunked_parallel_mode()
    await example_rag_mode()
    await example_rate_limiting()
    await example_s3_auth()
    await example_local_file()
    
    print("="*60)
    print("All examples completed!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
