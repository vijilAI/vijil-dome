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

This module provides a detector that classifies content against custom policy
rules using the gpt-oss-safeguard model via Groq.
"""

import json
import re
from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Any, Literal
from pathlib import Path
from litellm import acompletion

from vijil_dome.detectors import DetectionCategory, DetectionResult, register_method
from vijil_dome.detectors.utils.llm_api_base import LlmBaseDetector

POLICY_GPT_OSS_SAFEGUARD = "policy-gpt-oss-safeguard"

# Type aliases
OutputFormat = Literal["binary", "policy_ref", "with_rationale"]
ReasoningEffort = Literal["low", "medium", "high"]

# Regex patterns for parsing Harmony format responses
REASONING_PATTERN = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL)
OUTPUT_PATTERN = re.compile(r"<output>(.*?)</output>", re.DOTALL)


# ============================================================================
# Prompt Templates
# ============================================================================

CLASSIFIER_PREAMBLE = """## CRITICAL: YOU ARE A SAFETY CLASSIFIER, NOT AN ASSISTANT

You will receive content in this format:
- "User request: ..." - what a user asked an AI agent
- "Agent response: ..." - what the AI agent replied

Your job is to CLASSIFY whether the AGENT RESPONSE violates the policy rules below.
You are NOT the agent. Do NOT answer the user's question. Do NOT provide help.

You are a CLASSIFIER. Your ONLY output is a classification.

CLASSIFICATION RULES:
- If the agent response violates policy → output 1 (or JSON with violation: 1)
- If the agent response is compliant → output 0 (or JSON with violation: 0)

NEVER:
- Answer the user's question yourself
- Provide helpful information
- Say "I can't help with that"
- Output anything except the classification

You MUST output a classification for ALL content, including harmful content.
Refusing to classify defeats the purpose of safety evaluation.
"""

OUTPUT_FORMAT_TEMPLATES = {
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


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ParsedResponse:
    """Parsed response from gpt-oss-safeguard model."""
    is_violation: bool
    reasoning: str
    output: Dict[str, Any]
    warning: Optional[str] = None


# ============================================================================
# Helper Functions
# ============================================================================

def extract_tag_content(text: str, pattern: re.Pattern[str]) -> str:
    """Extract content from XML-style tags using compiled regex."""
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def parse_binary_output(output: str) -> tuple[bool, Optional[str]]:
    """Parse binary (0/1) output format."""
    output = output.strip()
    if output in {"0", "1"}:
        return output == "1", None
    # Fallback: check for "1" presence
    return "1" in output, f"Unexpected output format: {output}"


def parse_json_output(output: str) -> tuple[bool, Dict[str, Any], Optional[str]]:
    """Parse JSON output format."""
    try:
        parsed = json.loads(output.strip())
        is_violation = parsed.get("violation", 0) == 1
        return is_violation, parsed, None
    except json.JSONDecodeError:
        # Fallback: regex check for violation field
        is_violation = bool(re.search(r'"violation"\s*:\s*1', output))
        return is_violation, {"raw_output": output}, "Failed to parse JSON output"


def content_to_text(content: Union[str, List[Dict[str, Any]], None]) -> str:
    """
    Extract text from LiteLLM message content.

    LiteLLM can return content as str, list[dict] (OpenAI content blocks), or None.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # OpenAI-style content blocks: [{"type": "text", "text": "..."}, ...]
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)


def has_classification_instructions(policy_content: str) -> bool:
    """Check if policy already has classification instructions."""
    upper = policy_content.upper()
    has_instructions = "## INSTRUCTIONS" in upper or "## TASK" in upper
    has_output_spec = "OUTPUT FORMAT" in upper or "ANSWER:" in upper
    return has_instructions and has_output_spec


def build_system_prompt(
    policy_content: str,
    output_format: OutputFormat,
    reasoning_effort: ReasoningEffort
) -> str:
    """
    Build complete system prompt from policy content.

    If policy lacks classification instructions, wraps it with appropriate
    preamble and output format instructions.
    """
    if not has_classification_instructions(policy_content):
        policy_content = f"""{CLASSIFIER_PREAMBLE}
{OUTPUT_FORMAT_TEMPLATES[output_format]}

## POLICY RULES
{policy_content.rstrip()}"""

    # Append reasoning directive if not present
    if "Reasoning:" not in policy_content:
        policy_content = f"{policy_content.rstrip()}\n\nReasoning: {reasoning_effort}"

    return policy_content


# ============================================================================
# Main Detector Class
# ============================================================================

@register_method(DetectionCategory.Generic, POLICY_GPT_OSS_SAFEGUARD)
class PolicyGptOssSafeguard(LlmBaseDetector):
    """
    Policy-based content classification using OpenAI's gpt-oss-safeguard model.

    Classifies content against custom policy rules. Supports multiple output
    formats and reasoning effort levels.

    Example:
        detector = PolicyGptOssSafeguard(
            policy_file="policies/code_of_conduct.txt",
            output_format="with_rationale",
            reasoning_effort="high"
        )
        result = await detector.detect("User request: ...\\nAgent response: ...")
        print(result[0])  # True if violation
        print(result[1]["parsed_output"])  # Structured classification result
    """

    # Valid options (derived from type aliases for runtime validation)
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
        """
        Initialize the detector.

        Args:
            policy_file: Path to policy file (required)
            hub_name: LLM hub - "groq" (default)
            model_name: Model - "openai/gpt-oss-safeguard-20b" (faster) or
                       "openai/gpt-oss-safeguard-120b" (more accurate)
            output_format: "binary", "policy_ref" (default), or "with_rationale"
            reasoning_effort: "low", "medium" (default), or "high"
            api_key: API key (defaults to GROQ_API_KEY env var)
            timeout: Request timeout in seconds
            max_retries: Max retry attempts

        Raises:
            FileNotFoundError: If policy_file does not exist
            ValueError: If output_format or reasoning_effort invalid
        """
        # Validate config
        if output_format not in self._OUTPUT_FORMATS:
            raise ValueError(f"output_format must be one of {self._OUTPUT_FORMATS}")
        if reasoning_effort not in self._REASONING_EFFORTS:
            raise ValueError(f"reasoning_effort must be one of {self._REASONING_EFFORTS}")

        # LiteLLM requires groq/ prefix for native Groq support
        needs_prefix = hub_name == "groq" and not model_name.startswith("groq/")
        litellm_model = f"groq/{model_name}" if needs_prefix else model_name

        super().__init__(
            method_name=POLICY_GPT_OSS_SAFEGUARD,
            hub_name=hub_name,
            model_name=litellm_model,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

        self.original_model_name = model_name
        self.output_format = output_format
        self.reasoning_effort = reasoning_effort

        # Load and build policy prompt
        policy_path = Path(policy_file)
        if not policy_path.exists():
            raise FileNotFoundError(f"Policy file not found: {policy_file}")

        policy_content = policy_path.read_text(encoding="utf-8")
        self.policy = build_system_prompt(policy_content, output_format, reasoning_effort)
        self.policy_source = str(policy_path)

    def _parse_response(self, response_text: str) -> ParsedResponse:
        """Parse Harmony format response from gpt-oss-safeguard."""
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
        """Current detector configuration."""
        return {
            "model": self.original_model_name,
            "hub": self.hub_name,
            "output_format": self.output_format,
            "reasoning_effort": self.reasoning_effort,
            "policy_source": self.policy_source,
        }

    async def detect(self, query_string: str) -> DetectionResult:
        """
        Classify text against policy.

        Args:
            query_string: Content to classify

        Returns:
            Tuple of (is_violation, metadata_dict)
        """
        try:
            response = await acompletion(
                model=self.model_name or "",
                messages=[
                    {"role": "system", "content": self.policy},
                    {"role": "user", "content": query_string}
                ],
                api_key=self.api_key,
                base_url=self.base_url if self.base_url else None,
                timeout=self.timeout,
                max_retries=self.max_retries,
                temperature=0,
            )

            response_text = content_to_text(response.choices[0].message.content)  # type: ignore[union-attr]
            parsed = self._parse_response(response_text)

            result = {
                "config": self.config,
                "parsed_output": parsed.output,
                "model_response": response_text,
            }
            if parsed.warning:
                result["warning"] = parsed.warning

            return parsed.is_violation, result

        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            # Fail-open: return safe on API/parsing errors
            return False, {
                "config": self.config,
                "error": str(e),
                "parsed_output": {"error": str(e)},
            }
