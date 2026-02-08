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

from typing import Optional, Union, List, Dict, Any
from pathlib import Path
from litellm import acompletion

from vijil_dome.detectors import DetectionCategory, DetectionResult, register_method
from vijil_dome.detectors.utils.llm_api_base import LlmBaseDetector

POLICY_GPT_OSS_SAFEGUARD = "policy-gpt-oss-safeguard"


@register_method(DetectionCategory.Generic, POLICY_GPT_OSS_SAFEGUARD)
class PolicyGptOssSafeguard(LlmBaseDetector):
    """
    Policy-based content classification using OpenAI's gpt-oss-safeguard model.

    This detector uses custom policy prompts to classify content based on
    user-defined rules and criteria. The policy file defines what constitutes
    a violation and the expected output format.

    Example:
        detector = PolicyGptOssSafeguard(
            policy_file="vijil_dome/detectors/policies/spam_policy.md",
            model_name="openai/gpt-oss-120b",
            reasoning_effort="medium"
        )

        result = await detector.detect("Check this out: example.com")
        print(result.hit)  # True if policy violation detected
        print(result.reason)  # Explanation from model
    """

    def __init__(
        self,
        policy_file: str,
        hub_name: str = "groq",
        model_name: str = "openai/gpt-oss-safeguard-20b",
        reasoning_effort: str = "medium",
        api_key: Optional[str] = None,
        timeout: Optional[int] = 60,
        max_retries: Optional[int] = 3,
    ):
        """
        Initialize the policy-based GPT-OSS-Safeguard detector.

        Args:
            policy_file: Path to policy markdown file (required)
            hub_name: LLM hub to use (default: "groq")
            model_name: Model identifier - options:
                       - "openai/gpt-oss-safeguard-20b" (default, faster)
                       - "openai/gpt-oss-safeguard-120b" (most accurate)
            reasoning_effort: Reasoning depth - "low", "medium", "high" (default: "medium")
            api_key: API key for the hub (defaults to GROQ_API_KEY env var)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts

        Raises:
            FileNotFoundError: If policy_file path does not exist
            ValueError: If reasoning_effort not in [low, medium, high]
        """
        super().__init__(
            method_name=POLICY_GPT_OSS_SAFEGUARD,
            hub_name=hub_name,
            model_name=model_name,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

        # Store original model name for metadata
        self.original_model_name = model_name

        # Validate reasoning_effort
        valid_efforts = ["low", "medium", "high"]
        if reasoning_effort not in valid_efforts:
            raise ValueError(
                f"reasoning_effort must be one of {valid_efforts}, got: {reasoning_effort}"
            )
        self.reasoning_effort = reasoning_effort

        # Load policy from file
        policy_path = Path(policy_file)
        if not policy_path.exists():
            raise FileNotFoundError(f"Policy file not found: {policy_file}")

        with open(policy_path, 'r', encoding='utf-8') as f:
            policy_content = f.read()

        # Build system message with reasoning directive
        self.policy = self._build_system_message(policy_content)
        self.policy_source = str(policy_path)

    def _build_system_message(self, policy_content: str) -> str:
        """
        Build system message with policy content + reasoning directive.

        Format (from PDF):
        {policy_content}

        Reasoning: {low|medium|high}

        Args:
            policy_content: Raw content from policy file

        Returns:
            Complete system message with reasoning directive appended
        """
        # Check if policy already has Reasoning directive
        if "Reasoning:" in policy_content:
            # User already specified it in policy file - use as-is
            return policy_content
        else:
            # Append reasoning directive
            return f"{policy_content.rstrip()}\n\nReasoning: {self.reasoning_effort}"

    def _content_to_text(self, content: Union[str, List[Dict[str, Any]], None]) -> str:
        """
        Extract text from LiteLLM message content.

        LiteLLM can return content as str, list[dict] (OpenAI content blocks), or None.

        Args:
            content: Raw content from llm_response.choices[0].message.content

        Returns:
            Extracted text string
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
        # Fallback: stringify unknown types
        return str(content)

    def _parse_response(self, response_text: str) -> tuple[bool, str]:
        """
        Parse Harmony format response from gpt-oss-safeguard.

        Expected format:
        <reasoning>Model's reasoning process...</reasoning><output>0 or 1</output>

        Args:
            response_text: Raw response from the model

        Returns:
            Tuple of (is_violation, reasoning)
        """
        # Extract <output> content
        if "<output>" in response_text and "</output>" in response_text:
            output_start = response_text.find("<output>") + len("<output>")
            output_end = response_text.find("</output>")
            output = response_text[output_start:output_end].strip()
        else:
            # Fallback: use entire response if no tags
            output = response_text.strip()

        # Extract <reasoning> content (for auditability)
        reasoning = ""
        if "<reasoning>" in response_text and "</reasoning>" in response_text:
            reasoning_start = response_text.find("<reasoning>") + len("<reasoning>")
            reasoning_end = response_text.find("</reasoning>")
            reasoning = response_text[reasoning_start:reasoning_end].strip()

        # Strict violation check: exact match for "0" or "1"
        output_clean = output.strip()
        if output_clean in {"0", "1"}:
            is_violation = (output_clean == "1")
        else:
            # Handle unexpected output gracefully
            # Check if "1" appears as primary indicator (original loose logic as fallback)
            is_violation = "1" in output_clean
            # Log warning in reasoning
            reasoning = f"Warning: Unexpected output format '{output_clean}'. " + (reasoning or "")

        return is_violation, reasoning or output

    async def detect(self, text: str) -> DetectionResult:
        """
        Detect policy violations in the given text.

        Args:
            text: Content to classify

        Returns:
            DetectionResult with violation status and metadata
        """
        try:
            llm_response = await acompletion(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.policy},
                    {"role": "user", "content": text}
                ],
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
                temperature=0,  # Deterministic classification
            )

            response_text = self._content_to_text(llm_response.choices[0].message.content)
            is_violation, reasoning = self._parse_response(response_text)

            return is_violation, {
                "model_response": response_text,
                "model": self.original_model_name,
                "hub": self.hub_name,
                "policy_source": self.policy_source,
                "reasoning_effort": self.reasoning_effort,
                "reasoning": reasoning,
            }

        except Exception as e:
            # Fail-open: return safe result on error
            return False, {
                "error": str(e),
                "model": self.original_model_name,
                "reasoning": f"Error during detection: {str(e)}",
            }
