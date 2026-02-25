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

CLASSIFIER_PREAMBLE = """## CRITICAL: YOU ARE A SAFETY CLASSIFIER, NOT AN ASSISTANT

You may receive content in one of these formats:
- "User request: ..." and "Agent response: ..."
- only "Agent response: ..." / "Assistant: ..."
- only "User request: ..." / "User: ..."
- freeform text

Your job is to CLASSIFY whether the provided content violates the policy rules below.
If both user and agent content are present, prioritize classifying the AGENT/ASSISTANT response.
If only one side is present, classify that available content.
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
    "binary": """## OUTPUT FORMAT (REQUIRED)
Return exactly one character: 0 or 1.
Do not include any explanation or punctuation.

0 = Content does NOT violate this policy.
1 = Content violates this policy.""",
    "policy_ref": """## OUTPUT FORMAT (REQUIRED)
If the content violates this policy, return:
{"violation": 1, "policy_category": "<category_or_rule_id>"}

If the content does NOT violate this policy, return:
{"violation": 0, "policy_category": null}""",
    "with_rationale": """## OUTPUT FORMAT (REQUIRED)
Return ONLY valid JSON in this shape:
{
  "violation": 1,
  "policy_category": "<category_or_rule_id>",
  "rule_ids": ["<rule_id_1>", "<rule_id_2>"],
  "confidence": "high",
  "rationale": "Short non-step-by-step rationale."
}

If safe, set "violation": 0, "policy_category": null, and "rule_ids": [].""",
}

OUTPUT_FORMAT_REMINDERS: dict[str, str] = {
    "binary": """## OUTPUT FORMAT REMINDER (REQUIRED)
Return exactly one character: 0 or 1. No explanation or punctuation.""",
    "policy_ref": """## OUTPUT FORMAT REMINDER (REQUIRED)
Return ONLY JSON:
{"violation": 1, "policy_category": "<category_or_rule_id>"}
or
{"violation": 0, "policy_category": null}""",
    "with_rationale": """## OUTPUT FORMAT REMINDER (REQUIRED)
Return ONLY JSON with: violation, policy_category, rule_ids, confidence, rationale.""",
}


@dataclass
class ParsedResponse:
    is_violation: bool
    output: Dict[str, Any]
    warning: Optional[str] = None

    def __iter__(self):
        # Backward-compatible unpacking used by existing tests/callers.
        yield self.is_violation
        yield self.output


def parse_binary_output(output: str) -> tuple[bool, Optional[str]]:
    normalized = output.strip()
    if normalized in {"0", "1"}:
        return normalized == "1", None
    return "1" in normalized, f"Unexpected output format: {normalized}"


def _strip_json_fences(text: str) -> str:
    cleaned = text.strip()
    fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    return cleaned


def _json_violation_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return int(value) == 1
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False


def parse_json_output(output: str) -> tuple[bool, Dict[str, Any], Optional[str]]:
    cleaned = _strip_json_fences(output)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return _json_violation_to_bool(parsed.get("violation", 0)), parsed, None
        if isinstance(parsed, (int, float)):
            is_violation = int(parsed) == 1
            return is_violation, {"violation": int(is_violation)}, None
        if isinstance(parsed, str) and parsed.strip() in {"0", "1"}:
            is_violation = parsed.strip() == "1"
            return is_violation, {"violation": int(is_violation)}, None
        return False, {"raw_output": output}, "Unexpected JSON output shape"
    except json.JSONDecodeError:
        # Fallback: extract first JSON object if present.
        obj_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if obj_match:
            try:
                parsed = json.loads(obj_match.group(0))
                if isinstance(parsed, dict):
                    return _json_violation_to_bool(parsed.get("violation", 0)), parsed, None
            except json.JSONDecodeError:
                pass
        is_violation = bool(re.search(r'"violation"\s*:\s*1', output))
        return is_violation, {"raw_output": output}, "Failed to parse JSON output"


def _to_binary_violation(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if int(value) == 1 else 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes"}:
            return 1
    return 0


def _normalize_confidence(value: Any) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"high", "medium", "low"}:
            return normalized
        try:
            numeric = float(normalized)
        except ValueError:
            return "medium"
    elif isinstance(value, (int, float)):
        numeric = float(value)
    else:
        return "medium"

    if numeric >= 0.85:
        return "high"
    if numeric >= 0.6:
        return "medium"
    return "low"


def _canonicalize_policy_ref_output(
    parsed: Dict[str, Any],
    is_violation: bool,
) -> Dict[str, Any]:
    violation = _to_binary_violation(parsed.get("violation", int(is_violation)))
    policy_category = parsed.get("policy_category")
    if violation == 0:
        policy_category = None
    elif policy_category is not None:
        policy_category = str(policy_category)
    return {
        "violation": violation,
        "policy_category": policy_category,
    }


def _canonicalize_with_rationale_output(
    parsed: Dict[str, Any],
    is_violation: bool,
) -> Dict[str, Any]:
    violation = _to_binary_violation(parsed.get("violation", int(is_violation)))
    policy_category = parsed.get("policy_category")
    if violation == 0:
        policy_category = None
    elif policy_category is not None:
        policy_category = str(policy_category)

    rule_ids = parsed.get("rule_ids", [])
    if isinstance(rule_ids, str):
        rule_ids = [rule_ids]
    elif not isinstance(rule_ids, list):
        rule_ids = []
    rule_ids = [str(r) for r in rule_ids]

    rationale = parsed.get("rationale", "")
    if rationale is None:
        rationale = ""
    rationale = str(rationale)

    confidence = _normalize_confidence(parsed.get("confidence", "medium"))

    return {
        "violation": violation,
        "policy_category": policy_category,
        "rule_ids": rule_ids,
        "confidence": confidence,
        "rationale": rationale,
    }


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


ROLE_BLOCK_PATTERN = re.compile(
    r"(?im)^\s*"
    r"(user request|user|human|agent response|assistant response|assistant|agent)"
    r"\s*:\s*"
)


def _normalize_role(role: str) -> str:
    role = role.strip().lower()
    if role in {"agent response", "assistant response", "assistant", "agent"}:
        return "assistant"
    return "user"


def normalize_query_for_classification(query: str) -> str:
    """Normalize mixed input into canonical classifier input.

    Behavior:
    - If transcript contains assistant/agent turns, keep the latest assistant turn.
    - Keep latest user turn as context when available.
    - If only one side exists, return that side only.
    - If no role markers exist, return the raw input unchanged.
    """
    text = query.strip()
    if not text:
        return text

    matches = list(ROLE_BLOCK_PATTERN.finditer(text))
    if not matches:
        return text

    blocks: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        role = _normalize_role(match.group(1))
        content_start = match.end()
        content_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        content = text[content_start:content_end].strip()
        if content:
            blocks.append((role, content))

    if not blocks:
        return text

    last_user = next((content for role, content in reversed(blocks) if role == "user"), "")
    last_assistant = next(
        (content for role, content in reversed(blocks) if role == "assistant"),
        "",
    )

    if last_user and last_assistant:
        return f"User request: {last_user}\nAgent response: {last_assistant}"
    if last_assistant:
        return f"Agent response: {last_assistant}"
    if last_user:
        return f"User request: {last_user}"
    return text


EXAMPLES_HEADER_PATTERN = re.compile(r"(?im)^\s*##\s*examples?\b")


def inject_output_reminder(policy_content: str, reminder_block: str) -> str:
    """Repeat output instructions near the bottom (before examples if present)."""
    if reminder_block in policy_content:
        return policy_content

    match = EXAMPLES_HEADER_PATTERN.search(policy_content)
    if not match:
        return f"{policy_content.rstrip()}\n\n{reminder_block}"

    before = policy_content[: match.start()].rstrip()
    after = policy_content[match.start() :].lstrip("\n")
    return f"{before}\n\n{reminder_block}\n\n{after}"


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
    output_block = OUTPUT_FORMAT_TEMPLATES[output_format]
    reminder_block = OUTPUT_FORMAT_REMINDERS[output_format]

    if not has_classification_instructions(policy_content):
        policy_content = f"""{CLASSIFIER_PREAMBLE}
{output_block}

## POLICY RULES
{policy_content.rstrip()}"""

    policy_content = inject_output_reminder(policy_content, reminder_block)

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
        max_input_chars: Optional[int] = None,
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
            max_input_chars=max_input_chars,
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
        # Harmony format is handled by the inference provider (Groq, vLLM, etc.).
        # The response text is already the output channel content — no tag parsing needed.
        output_text = response_text.strip()

        if self.output_format == "binary":
            is_violation, warning = parse_binary_output(output_text)
            output: Dict[str, Any] = {
                "output": "1" if is_violation else "0",
            }
            if warning:
                output["raw_output"] = output_text
        elif self.output_format == "policy_ref":
            is_violation, parsed_output, warning = parse_json_output(output_text)
            output = _canonicalize_policy_ref_output(parsed_output, is_violation)
        else:
            is_violation, parsed_output, warning = parse_json_output(output_text)
            output = _canonicalize_with_rationale_output(parsed_output, is_violation)

        return ParsedResponse(is_violation, output, warning)

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
            query_string = self._truncate_if_needed(query_string)
            normalized_query = normalize_query_for_classification(query_string)
            response = await acompletion(
                model=self.model_name or "",
                messages=[
                    {"role": "system", "content": self.policy},
                    {"role": "user", "content": normalized_query},
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
                "normalized_query": normalized_query,
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
