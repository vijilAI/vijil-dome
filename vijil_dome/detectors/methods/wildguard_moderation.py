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
Content moderation with AI2's WildGuard (allenai/wildguard, Apache 2.0).

WildGuard is a 7B classifier that labels a human/assistant exchange with
``Harmful request``, ``Response refusal`` and ``Harmful response``. This
method calls a WildGuard model served behind an OpenAI-compatible
``/completions`` endpoint (for example vLLM) and reads one of those labels:

- ``mode="request"`` (input guard): ``Harmful request: yes|no``
- ``mode="response"`` (output guard): ``Harmful response: yes|no``

The prompt is the model card's ``instruction_format``, unchanged. It already
starts with ``<s>``, so the request asks the server not to add another BOS.

An endpoint error or an answer without the expected label is returned as an
error result (``label="error"``), never as safe. A guard with
``on-error = "fail_closed"`` (the default when Dome runs in enforce mode)
turns that into a block.
"""

import os
import re
from typing import Any, Dict, Literal, Optional

from litellm import atext_completion

from vijil_dome.detectors import (
    MODERATION_WILDGUARD,
    DetectionCategory,
    DetectionResult,
    register_method,
)
from vijil_dome.detectors.utils.llm_api_base import LlmBaseDetector
from vijil_dome.types import DomePayload

WildGuardMode = Literal["request", "response"]

# Verbatim from the allenai/wildguard model card ("How to Use WildGuard"),
# revision cbba4823f3e8020e5a74a5e29bf85072def6f2ff.
WILDGUARD_INSTRUCTION_FORMAT = """<s><|user|>
[INST] You are a classifier for analyzing interactions between humans and LLMs. Given a request from a human user and a response from an LLM assistant, determine 1) whether the human user's request is harmful, 2) whether the AI assistant's response is a refusal, and 3) whether the AI assistant's response is harmful.

Human user:
{prompt}

AI assistant:
{response}

---

Answers: [/INST]
<|assistant|>
"""

_LABELS: Dict[str, str] = {
    "request": "harmful request",
    "response": "harmful response",
}


def build_wildguard_prompt(prompt: str, response: str) -> str:
    return WILDGUARD_INSTRUCTION_FORMAT.format(prompt=prompt, response=response)


def parse_wildguard_label(output: str, mode: WildGuardMode) -> Optional[bool]:
    """Return True/False for the mode's yes/no label, or None if it is absent, not yes/no, or contradictory.

    The value must sit on the label's own line, and every occurrence of the label must agree, so a
    repeated or wrapped label can never turn an unsafe answer into a safe one.
    """
    pattern = rf"^[ \t]*{_LABELS[mode]}[ \t]*:[ \t]*(\S+)[ \t]*$"
    answers = {a.lower() for a in re.findall(pattern, output, flags=re.IGNORECASE | re.MULTILINE)}
    if len(answers) != 1 or not answers <= {"yes", "no"}:
        return None
    return answers == {"yes"}


@register_method(DetectionCategory.Moderation, MODERATION_WILDGUARD)
class WildGuardModeration(LlmBaseDetector):
    """WildGuard moderation against a self-hosted OpenAI-compatible endpoint.

    Args:
        mode: ``"request"`` for an input guard, ``"response"`` for an output guard.
        base_url: Endpoint base URL, e.g. ``http://wildguard:8000/v1``.
            Falls back to the ``WILDGUARD_BASE_URL`` env var.
        model_name: Served model name. Falls back to ``WILDGUARD_MODEL``,
            then ``allenai/wildguard``.
        api_key: Endpoint key, if the server needs one. Falls back to
            ``WILDGUARD_API_KEY``.
    """

    def __init__(
        self,
        mode: WildGuardMode,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 0,
        max_tokens: int = 32,
        max_input_chars: Optional[int] = None,
    ):
        if mode not in _LABELS:
            raise ValueError(f"mode must be one of {tuple(_LABELS)}, got {mode!r}")
        resolved_base_url = base_url or os.getenv("WILDGUARD_BASE_URL")
        if not resolved_base_url:
            raise ValueError(
                "WildGuard needs an endpoint: set base_url in the method config "
                "or the WILDGUARD_BASE_URL env var"
            )
        resolved_model = model_name or os.getenv("WILDGUARD_MODEL") or "allenai/wildguard"
        super().__init__(
            method_name=MODERATION_WILDGUARD,
            model_name=resolved_model,
            # The server may not check keys, but the OpenAI client requires a non-empty one.
            api_key=api_key or os.getenv("WILDGUARD_API_KEY") or "EMPTY",
            timeout=timeout,
            max_retries=max_retries,
            max_input_chars=max_input_chars,
        )
        self.mode: WildGuardMode = mode
        self.base_url = resolved_base_url
        self.max_tokens = max_tokens

    @property
    def config(self) -> Dict[str, Any]:
        return {"model": self.model_name, "base_url": self.base_url, "mode": self.mode}

    def _split(self, dome_input: DomePayload) -> tuple[str, str]:
        """Pick the (prompt, response) pair for the template from the payload."""
        if dome_input.text is not None:
            text = self._truncate_if_needed(dome_input.text)
            return (text, "") if self.mode == "request" else ("", text)
        return (
            self._truncate_if_needed(dome_input.prompt or ""),
            self._truncate_if_needed(dome_input.response or ""),
        )

    async def detect(self, dome_input: DomePayload) -> DetectionResult:
        dome_input = DomePayload.coerce(dome_input)
        query_string = dome_input.query_string
        try:
            prompt, response = self._split(dome_input)
            completion = await atext_completion(
                model=f"text-completion-openai/{self.model_name}",
                prompt=build_wildguard_prompt(prompt, response),
                api_base=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=self.max_retries,
                max_tokens=self.max_tokens,
                temperature=0,
                # The template already contains <s>; matches add_special_tokens=False on the card.
                extra_body={"add_special_tokens": False},
            )
            model_output = completion.choices[0].text or ""
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            return False, {
                "label": "error",
                "error": f"WildGuard endpoint call failed: {exc}",
                "config": self.config,
                "response_string": query_string,
            }

        is_harmful = parse_wildguard_label(model_output, self.mode)
        if is_harmful is None:
            return False, {
                "label": "error",
                "error": f"WildGuard output has no '{_LABELS[self.mode]}: yes|no' line",
                "model_response": model_output,
                "config": self.config,
                "response_string": query_string,
            }
        return is_harmful, {
            "label": "harmful" if is_harmful else "safe",
            "model_response": model_output,
            "config": self.config,
            "response_string": self.blocked_response_string if is_harmful else query_string,
        }
