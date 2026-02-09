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
Policy-based content classification using OpenAI's gpt-oss-safeguard model.

This detector classifies content against a provided policy file and supports:
- output modes: binary, policy_ref, with_rationale
- reasoning effort: low, medium, high
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from litellm import acompletion

from vijil_dome.detectors import DetectionCategory, DetectionResult, register_method
from vijil_dome.detectors.utils.llm_api_base import LlmBaseDetector

POLICY_GPT_OSS_SAFEGUARD = "policy-gpt-oss-safeguard"

OutputFormat = Literal["binary", "policy_ref", "with_rationale"]
ReasoningEffort = Literal["low", "medium", "high"]

REASONING_PATTERN = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL)
OUTPUT_PATTERN = re.compile(r"<output>(.*?)</output>", re.DOTALL)

CLASSIFIER_PREAMBLE = """## CRITICAL: YOU ARE A SAFETY CLASSIFIER, NOT AN ASSISTANT

You will receive content in this format:
- "User request: ..." - what a user asked an AI agent
- "Agent response: ..." - what the AI agent replied

Your job is to CLASSIFY whether the AGENT RESPONSE violates the policy rules below.
You are NOT the agent. Do NOT answer the user's question. Do NOT provide help.

You are a CLASSIFIER. Your ONLY output is a classification.

CLASSIFICATION RULES:
- If the agent response violates policy -> output 1 (or JSON with violation: 1)
- If the agent response is compliant -> output 0 (or JSON with violation: 0)

NEVER:
- Answer the user's question yourself
- Provide helpful information
- Say "I can't help with that"
- Output anything except the classification
"""

OUTPUT_FORMAT_TEMPLATES: dict[str, str] = {
    "binary": """## TASK
Classify whether the content violates the policy rules below.

- 0 = Content does NOT violate this policy (SAFE)
- 1 = Content DOES violate this policy (VIOLATION)

## OUTPUT FORMAT
Return EXACTLY one character: 0 or 1
No explanation. No punctuation. Just the digit.""",
    "policy_ref": """## TASK
Classify whether the content violates the policy rules below.
If it violates, identify which policy category applies.

## OUTPUT FORMAT
Return ONLY valid JSON, no other text:
- Violation: {"violation": 1, "policy_category": "<rule_id>"}
- Safe: {"violation": 0, "policy_category": null}""",
    "with_rationale": """## TASK
Classify whether the content violates the policy rules below.
Identify which rules apply and provide brief rationale.

## OUTPUT FORMAT
Return ONLY valid JSON with these fields:
- violation: 1 if violates policy, 0 if safe
- policy_category: Main category/section violated (or null if safe)
- rule_ids: Array of specific rule IDs that apply (or empty array)
- confidence: "high", "medium", or "low"
- rationale: Brief explanation with policy citations""",
}


@dataclass
class ParsedResponse:
    is_violation: bool
    reasoning: str
    output: Dict[str, Any]
    warning: Optional[str] = None

    def __iter__(self):
        # Backward-compatible unpacking used by existing tests/callers.
        yield self.is_violation
        yield self.reasoning


def extract_tag_content(text: str, pattern: re.Pattern[str]) -> str:
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def parse_binary_output(output: str) -> tuple[bool, Optional[str]]:
    normalized = output.strip()
    if normalized in {"0", "1"}:
        return normalized == "1", None
    return "1" in normalized, f"Unexpected output format: {normalized}"


def parse_json_output(output: str) -> tuple[bool, Dict[str, Any], Optional[str]]:
    try:
        parsed = json.loads(output.strip())
        if isinstance(parsed, dict):
            return parsed.get("violation", 0) == 1, parsed, None
        if isinstance(parsed, (int, float)):
            is_violation = int(parsed) == 1
            return is_violation, {"violation": int(is_violation)}, None
        if isinstance(parsed, str) and parsed.strip() in {"0", "1"}:
            is_violation = parsed.strip() == "1"
            return is_violation, {"violation": int(is_violation)}, None
        return False, {"raw_output": output}, "Unexpected JSON output shape"
    except json.JSONDecodeError:
        is_violation = bool(re.search(r'"violation"\s*:\s*1', output))
        return is_violation, {"raw_output": output}, "Failed to parse JSON output"


def content_to_text(content: Union[str, List[Dict[str, Any]], None]) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)


def has_classification_instructions(policy_content: str) -> bool:
    upper = policy_content.upper()
    has_instructions = "## INSTRUCTIONS" in upper or "## TASK" in upper
    has_output_spec = "OUTPUT FORMAT" in upper or "ANSWER:" in upper
    return has_instructions and has_output_spec


def build_system_prompt(
    policy_content: str,
    output_format: OutputFormat,
    reasoning_effort: ReasoningEffort,
) -> str:
    if not has_classification_instructions(policy_content):
        policy_content = f"""{CLASSIFIER_PREAMBLE}
{OUTPUT_FORMAT_TEMPLATES[output_format]}

## POLICY RULES
{policy_content.rstrip()}"""

    if "Reasoning:" not in policy_content:
        policy_content = f"{policy_content.rstrip()}\n\nReasoning: {reasoning_effort}"
    return policy_content


@register_method(DetectionCategory.Generic, POLICY_GPT_OSS_SAFEGUARD)
class PolicyGptOssSafeguard(LlmBaseDetector):
    """Policy-based content classifier using gpt-oss-safeguard."""

    _OUTPUT_FORMATS = ("binary", "policy_ref", "with_rationale")
    _REASONING_EFFORTS = ("low", "medium", "high")

    def __init__(
        self,
        policy_file: str,
        hub_name: str = "groq",
        model_name: str = "openai/gpt-oss-safeguard-20b",
        output_format: OutputFormat = "policy_ref",
        reasoning_effort: ReasoningEffort = "medium",
        api_key: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        if output_format not in self._OUTPUT_FORMATS:
            raise ValueError(f"output_format must be one of {self._OUTPUT_FORMATS}")
        if reasoning_effort not in self._REASONING_EFFORTS:
            raise ValueError(f"reasoning_effort must be one of {self._REASONING_EFFORTS}")

        if hub_name == "groq" and api_key is None:
            api_key = os.getenv("GROQ_API_KEY")

        litellm_model = model_name
        if hub_name == "groq" and not model_name.startswith("groq/"):
            litellm_model = f"groq/{model_name}"

        super().__init__(
            method_name=POLICY_GPT_OSS_SAFEGUARD,
            hub_name=hub_name,
            model_name=litellm_model,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

        self.hub_name = hub_name
        self.original_model_name = model_name
        self.output_format = output_format
        self.reasoning_effort = reasoning_effort

        policy_path = Path(policy_file)
        if not policy_path.exists():
            raise FileNotFoundError(f"Policy file not found: {policy_file}")

        policy_content = policy_path.read_text(encoding="utf-8")
        self.policy = build_system_prompt(policy_content, output_format, reasoning_effort)
        self.policy_source = str(policy_path)

    def _build_system_message(self, policy_content: str) -> str:
        return build_system_prompt(
            policy_content=policy_content,
            output_format=self.output_format,
            reasoning_effort=self.reasoning_effort,
        )

    def _parse_response(self, response_text: str) -> ParsedResponse:
        reasoning = extract_tag_content(response_text, REASONING_PATTERN)
        output_text = extract_tag_content(response_text, OUTPUT_PATTERN) or response_text.strip()

        if self.output_format == "binary":
            is_violation, warning = parse_binary_output(output_text)
            output = {"output": output_text, "model_reasoning": reasoning}
        else:
            is_violation, output, warning = parse_json_output(output_text)
            output["model_reasoning"] = reasoning

        return ParsedResponse(is_violation, reasoning, output, warning)

    @property
    def config(self) -> Dict[str, Any]:
        return {
            "model": self.original_model_name,
            "hub": self.hub_name,
            "output_format": self.output_format,
            "reasoning_effort": self.reasoning_effort,
            "policy_source": self.policy_source,
        }

    async def detect(self, query_string: str) -> DetectionResult:
        try:
            response = await acompletion(
                model=self.model_name or "",
                messages=[
                    {"role": "system", "content": self.policy},
                    {"role": "user", "content": query_string},
                ],
                api_key=self.api_key,
                base_url=self.base_url if self.base_url else None,
                timeout=self.timeout,
                max_retries=self.max_retries,
                temperature=0,
            )

            response_text = content_to_text(response.choices[0].message.content)  # type: ignore[union-attr]
            parsed = self._parse_response(response_text)

            result: Dict[str, Any] = {
                "config": self.config,
                "parsed_output": parsed.output,
                "model_response": response_text,
            }
            if parsed.warning:
                result["warning"] = parsed.warning

            return parsed.is_violation, result

        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            return False, {
                "config": self.config,
                "error": str(exc),
                "parsed_output": {"error": str(exc)},
            }
