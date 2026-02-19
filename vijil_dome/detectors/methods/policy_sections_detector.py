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

import asyncio
import logging
from typing import Optional, List, Dict, Any, Union

from vijil_dome.detectors import (
    DetectionCategory,
    DetectionResult,
    DetectionMethod,
    register_method,
    POLICY_SECTIONS,
)
from vijil_dome.detectors.methods.gpt_oss_safeguard_policy import PolicyGptOssSafeguard
from vijil_dome.utils.policy_loader import (
    load_policy_sections_from_s3,
    validate_policy_json,
)

logger = logging.getLogger("vijil.dome")


@register_method(DetectionCategory.Policy, POLICY_SECTIONS)
class PolicySectionsDetector(DetectionMethod):
    """
    Policy sections detector that loads policy sections from S3 or accepts them directly,
    creates multiple PolicyGptOssSafeguard instances (one per section), and runs them
    in parallel batches with fast fail.

    This detector is optimized for policies that are split into 400-600 token sections
    for optimal performance. Each section is evaluated independently, and the detector
    fails fast on the first violation detected.

    Example with S3:
        detector = PolicySectionsDetector(
            policy_s3_bucket="my-bucket",
            policy_s3_key="teams/team-123/policies/policy-456/sections.json",
            applies_to="input",
            max_parallel_sections=10,
            model_name="openai/gpt-oss-120b",
            reasoning_effort="medium"
        )

    Example with direct sections:
        detector = PolicySectionsDetector(
            policy_sections=[
                {
                    "section_id": "section-1",
                    "content": "# Policy\n\n## INSTRUCTIONS\n...",
                    "applies_to": ["input"],
                }
            ],
            applies_to="input",
            model_name="openai/gpt-oss-120b",
        )
    """

    def __init__(
        self,
        # S3 loading (mutually exclusive with policy_sections)
        policy_s3_bucket: Optional[str] = None,
        policy_s3_key: Optional[str] = None,
        # Direct sections (mutually exclusive with S3 params)
        policy_sections: Optional[List[Dict[str, Any]]] = None,
        # Filtering
        applies_to: Union[str, List[str]] = "input",
        # Rate limiting
        max_parallel_sections: Optional[int] = None,
        # PolicyGptOssSafeguard params
        model_name: str = "openai/gpt-oss-120b",
        reasoning_effort: str = "medium",
        hub_name: str = "nebius",
        timeout: Optional[int] = 60,
        max_retries: Optional[int] = 3,
        api_key: Optional[str] = None,
        # S3 auth params (optional)
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        region_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the policy sections detector.

        Args:
            policy_s3_bucket: S3 bucket name (required if policy_sections not provided)
            policy_s3_key: S3 object key (required if policy_sections not provided)
            policy_sections: Direct list of section dicts (required if S3 params not provided)
            applies_to: Filter sections by applies_to - "input", "output", or ["input", "output"] (default: "input")
            max_parallel_sections: Maximum number of sections to run in parallel (default: None = no limit)
            model_name: LLM model identifier (default: "openai/gpt-oss-120b")
            reasoning_effort: Reasoning depth - "low", "medium", "high" (default: "medium")
            hub_name: LLM hub to use (default: "nebius")
            timeout: Request timeout in seconds (default: 60)
            max_retries: Maximum retry attempts (default: 3)
            api_key: API key for the hub (optional, uses env var if not provided)
            aws_access_key_id: AWS access key (optional, uses boto3 defaults if not provided)
            aws_secret_access_key: AWS secret key (optional)
            aws_session_token: AWS session token (optional)
            region_name: AWS region (optional, uses boto3 defaults if not provided)
            cache_dir: Local cache directory (optional, defaults to ~/.cache/vijil-dome/policies/)

        Raises:
            ValueError: If neither S3 params nor policy_sections provided, or both provided
            ValueError: If sections structure is invalid
        """
        # Validate mutual exclusivity
        has_s3 = policy_s3_bucket is not None and policy_s3_key is not None
        has_sections = policy_sections is not None

        if not has_s3 and not has_sections:
            raise ValueError(
                "Either policy_s3_bucket/policy_s3_key or policy_sections must be provided"
            )
        if has_s3 and has_sections:
            raise ValueError(
                "Cannot specify both S3 params and policy_sections"
            )

        # Load sections
        if has_s3:
            policy_data = load_policy_sections_from_s3(
                bucket=policy_s3_bucket,
                key=policy_s3_key,
                cache_dir=cache_dir,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                region_name=region_name,
            )
            sections = policy_data.get("sections", [])
        else:
            # Validate direct sections structure
            if not isinstance(policy_sections, list) or len(policy_sections) == 0:
                raise ValueError("policy_sections must be a non-empty list")
            # Create a minimal policy_data structure for validation
            policy_data = {"sections": policy_sections}
            validate_policy_json(policy_data)
            sections = policy_sections

        # Normalize applies_to to list
        if isinstance(applies_to, str):
            applies_to_list = [applies_to]
        else:
            applies_to_list = applies_to

        # Filter sections by applies_to
        filtered_sections = []
        for section in sections:
            section_applies_to = section.get("applies_to", [])
            if not isinstance(section_applies_to, list):
                section_applies_to = [section_applies_to]
            # Check if any applies_to value matches
            if any(applies in section_applies_to for applies in applies_to_list):
                filtered_sections.append(section)

        if len(filtered_sections) == 0:
            raise ValueError(
                f"No sections found matching applies_to={applies_to_list}"
            )

        # Create PolicyGptOssSafeguard instances for each section
        self.detectors: List[PolicyGptOssSafeguard] = []
        self.section_metadata: List[Dict[str, Any]] = []

        for section in filtered_sections:
            detector = PolicyGptOssSafeguard(
                policy_content=section["content"],
                model_name=model_name,
                reasoning_effort=reasoning_effort,
                hub_name=hub_name,
                timeout=timeout,
                max_retries=max_retries,
                api_key=api_key,
            )
            self.detectors.append(detector)
            # Store metadata for result aggregation
            self.section_metadata.append({
                "section_id": section.get("section_id"),
                "header": section.get("metadata", {}).get("header") if isinstance(section.get("metadata"), dict) else None,
                "policy_id": policy_data.get("policy_id"),
            })

        self.max_parallel_sections = max_parallel_sections
        self.blocked_response_string = f"Method:{POLICY_SECTIONS}"

    async def detect(self, query_string: str) -> DetectionResult:
        """
        Detect policy violations by running all section detectors in parallel batches.

        Args:
            query_string: Content to classify

        Returns:
            DetectionResult with violation status and aggregated metadata
        """
        if len(self.detectors) == 0:
            return False, {"error": "No detectors configured"}

        # If max_parallel_sections is set, process in batches
        if self.max_parallel_sections and self.max_parallel_sections > 0:
            return await self._detect_batched(query_string)
        else:
            return await self._detect_all_parallel(query_string)

    async def _detect_all_parallel(self, query_string: str) -> DetectionResult:
        """Run all detectors in parallel with fast fail."""
        # Create tasks with metadata
        task_data = [
            {
                "task": asyncio.create_task(detector.detect_with_time(query_string)),
                "section_idx": idx,
            }
            for idx, detector in enumerate(self.detectors)
        ]
        tasks = [td["task"] for td in task_data]

        violation_detected = False
        violation_result = None
        all_results = []

        while tasks:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                if task.cancelled():
                    continue
                try:
                    result = task.result()
                    # Find the section index for this task
                    section_idx = next(
                        td["section_idx"] for td in task_data if td["task"] == task
                    )
                    section_meta = self.section_metadata[section_idx]

                    all_results.append({
                        "section_id": section_meta["section_id"],
                        "header": section_meta["header"],
                        "hit": result.hit,
                        "result": result.result,
                        "exec_time": result.exec_time,
                    })

                    if result.hit:
                        violation_detected = True
                        violation_result = result
                        # Cancel remaining tasks
                        for p in pending:
                            p.cancel()
                        break
                except Exception as e:
                    logger.warning(f"Detector task raised exception: {e}")
                    continue

            if violation_detected:
                break

            # Update task_data to only include pending tasks
            task_data = [td for td in task_data if td["task"] in pending]
            tasks = pending

        if violation_detected and violation_result:
            return True, {
                "violation": True,
                "sections": all_results,
                "violating_section": all_results[-1] if all_results else None,
                "model": violation_result.result.get("model"),
                "hub": violation_result.result.get("hub"),
                "response_string": self.blocked_response_string,
            }

        return False, {
            "violation": False,
            "sections": all_results,
            "response_string": query_string,
        }

    async def _detect_batched(self, query_string: str) -> DetectionResult:
        """Run detectors in batches with fast fail."""
        violation_detected = False
        all_results = []
        violation_result = None

        # Process in batches
        for batch_start in range(0, len(self.detectors), self.max_parallel_sections):
            batch_end = min(batch_start + self.max_parallel_sections, len(self.detectors))
            batch_detectors = self.detectors[batch_start:batch_end]
            batch_metadata = self.section_metadata[batch_start:batch_end]

            # Create tasks with metadata
            batch_task_data = [
                {
                    "task": asyncio.create_task(detector.detect_with_time(query_string)),
                    "task_idx": idx,
                }
                for idx, detector in enumerate(batch_detectors)
            ]
            tasks = [td["task"] for td in batch_task_data]

            batch_results = []
            batch_violation = False

            while tasks:
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    if task.cancelled():
                        continue
                    try:
                        result = task.result()
                        # Find the task index for this task
                        task_idx = next(
                            td["task_idx"] for td in batch_task_data if td["task"] == task
                        )
                        section_meta = batch_metadata[task_idx]

                        batch_results.append({
                            "section_id": section_meta["section_id"],
                            "header": section_meta["header"],
                            "hit": result.hit,
                            "result": result.result,
                            "exec_time": result.exec_time,
                        })

                        if result.hit:
                            batch_violation = True
                            violation_result = result
                            # Cancel remaining tasks in this batch
                            for p in pending:
                                p.cancel()
                            break
                    except Exception as e:
                        logger.warning(f"Detector task raised exception: {e}")
                        continue

                if batch_violation:
                    break

                # Update batch_task_data to only include pending tasks
                batch_task_data = [td for td in batch_task_data if td["task"] in pending]
                tasks = pending

            all_results.extend(batch_results)

            if batch_violation:
                violation_detected = True
                break  # Fast fail - don't process remaining batches

        if violation_detected and violation_result:
            return True, {
                "violation": True,
                "sections": all_results,
                "violating_section": next(
                    (r for r in all_results if r["hit"]), None
                ),
                "model": violation_result.result.get("model"),
                "hub": violation_result.result.get("hub"),
                "response_string": self.blocked_response_string,
            }

        return False, {
            "violation": False,
            "sections": all_results,
            "response_string": query_string,
        }
