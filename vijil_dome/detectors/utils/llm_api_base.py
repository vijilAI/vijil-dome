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

import os
from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import replace
from litellm import ModelResponse

from vijil_dome.detectors import DetectionResult, DetectionMethod


class LlmBaseDetector(DetectionMethod, ABC):
    """
    Abstract base class for detection models using the Async OpenAI Client for LLM API calls.
    Subclasses can implement their own detection logic and prompt engineering
    """

    def __init__(
        self,
        method_name: str,
        hub_name: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        supported_hubs = [None, "openai", "together"]
        self.model_name = model_name
        self.hub_name = hub_name
        if self.hub_name not in supported_hubs:
            raise ValueError(
                f"Unsupported hub name: {self.hub_name}. Supported hubs are: {supported_hubs}"
            )
        if self.hub_name == "together":
            self.base_url = "https://api.together.xyz/v1"
            if api_key is None:
                api_key = os.getenv("TOGETHERAI_API_KEY")
        else:
            self.base_url = None
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.blocked_response_string = f"Method:{method_name}"

    @abstractmethod
    async def detect(self, query_string: str) -> DetectionResult:
        pass


class LlmBaseDetectorWithContext(LlmBaseDetector):
    """
    Abstract base class for context-dependent detection models using the Async OpenAI Client for LLM API calls.
    Subclasses can implement their own detection logic and prompt engineering
    """

    def __init__(
        self,
        method_name: str,
        hub_name: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        context: Optional[str] = None,
    ) -> None:
        super().__init__(
            method_name=method_name,
            hub_name=hub_name,
            model_name=model_name,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.context = context

    # Add additional context to the existing context
    def add_context(self, addition_context: str) -> None:
        if self.context is None:
            self.context = addition_context
        else:
            self.context += addition_context

    # Replace the existing context with new context
    def update_context(self, new_context: str) -> None:
        self.context = new_context


def parse_litellm_response(llm_response: ModelResponse) -> dict:
    return {
        "id": llm_response.id,
        "created": llm_response.created,
        "model": llm_response.model,
        "object": llm_response.object,
        "system_fingerprint": llm_response.system_fingerprint,
        "choices": [
            {
                "finish_reason": choice.finish_reason,
                "index": choice.index,
                "message": {
                    "content": choice.message.content,
                    "role": choice.message.role,
                    "tool_calls": choice.message.tool_calls,
                    "function_call": choice.message.function_call,
                    "provider_specific_fields": choice.message.provider_specific_fields,
                    # "annotations": choice.message.annotations,
                },
            }
            for choice in llm_response.choices
        ],
        "usage": {
            "completion_tokens": llm_response.usage.completion_tokens,
            "prompt_tokens": llm_response.usage.prompt_tokens,
            "total_tokens": llm_response.usage.total_tokens,
        },
    }
