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

from litellm import completion as litellm_completion

from vijil_dome.detectors import (
    MODERATION_LLM,
    SECURITY_LLM,
    HALLUCINATION_LLM,
    FACTCHECK_LLM,
    GENERIC_LLM,
    DetectionCategory,
    DetectionResult,
    register_method,
)
from vijil_dome.detectors.utils.llm_api_base import (
    LlmBaseDetector,
    LlmBaseDetectorWithContext,
    parse_litellm_response,
)
from vijil_dome.detectors.utils.prompt_templates.modifiable_prompts import (
    format_custom_llm_classifier_prompt,
    format_llm_hallucination_prompt,
    format_llm_factcheck_prompt,
)
from typing import Optional
from string import Template


# A generic LLM detector - specify your sys prompt and trigger word
@register_method(DetectionCategory.Generic, GENERIC_LLM)
class GenericLLMDetector(LlmBaseDetector):
    def __init__(
        self,
        sys_prompt_template: str,
        trigger_word_list: list[str],
        hub_name: str = "openai",
        model_name: str = "gpt-4-turbo",
        api_key: Optional[str] = None,
    ):
        super().__init__(
            method_name=GENERIC_LLM,
            hub_name=hub_name,
            model_name=model_name,
            api_key=api_key,
        )
        self.sys_prompt_template = Template(sys_prompt_template)
        self.trigger_word_list = trigger_word_list

    async def detect(self, query_string: str) -> DetectionResult:
        llm_prompt = self.sys_prompt_template.substitute(query_string=query_string)
        llm_response = litellm_completion(
            model=self.model_name,
            messages=[{"role": "system", "content": llm_prompt}],
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        print(f"llm_response: {llm_response}")
        if not llm_response.choices or len(llm_response.choices) == 0:
            raise ValueError("No results returned from OpenAI")
        llm_dec = llm_response.choices[0].message.content
        flagged = False
        for trigger in self.trigger_word_list:
            if trigger in llm_dec.lower():
                flagged = True
                break
        return flagged, {
            "type": type(self),
            "response": parse_litellm_response(llm_response),
            "response_string": self.blocked_response_string
            if flagged
            else query_string,
        }


@register_method(DetectionCategory.Moderation, MODERATION_LLM)
class LlmModerations(LlmBaseDetector):
    def __init__(
        self,
        hub_name: str = "openai",
        model_name: str = "gpt-4-turbo",
        api_key: Optional[str] = None,
    ):
        super().__init__(
            method_name=MODERATION_LLM,
            hub_name=hub_name,
            model_name=model_name,
            api_key=api_key,
        )

    async def detect(self, query_string: str) -> DetectionResult:
        content = format_custom_llm_classifier_prompt(
            "user",
            query_string,
            injection=False,
            toxic=True,
            pii=False,
        )
        llm_response = litellm_completion(
            model=self.model_name,
            messages=[{"role": "system", "content": content}],
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        if not llm_response.choices or len(llm_response.choices) == 0:
            raise ValueError("No moderation results returned from OpenAI")
        llm_dec = llm_response.choices[0].message.content
        flagged = "unsafe" in llm_dec.lower()
        return flagged, {
            "type": type(self),
            "response": parse_litellm_response(llm_response),
            "response_string": self.blocked_response_string
            if flagged
            else query_string,
        }


@register_method(DetectionCategory.Security, SECURITY_LLM)
class LlmSecurity(LlmBaseDetector):
    start_count = 0
    completion_count = 0

    def __init__(
        self,
        hub_name: str = "openai",
        model_name: str = "gpt-4-turbo",
        api_key: Optional[str] = None,
    ):
        super().__init__(
            method_name=SECURITY_LLM,
            hub_name=hub_name,
            model_name=model_name,
            api_key=api_key,
        )

    async def detect(self, query_string: str) -> DetectionResult:
        content = format_custom_llm_classifier_prompt(
            "user",
            query_string,
            injection=True,
            toxic=False,
            pii=False,
        )
        llm_response = litellm_completion(
            model=self.model_name,
            messages=[{"role": "system", "content": content}],
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        if not llm_response.choices or len(llm_response.choices) == 0:
            raise ValueError("No moderation results returned from OpenAI")
        llm_dec = llm_response.choices[0].message.content
        flagged = "unsafe" in llm_dec.lower()
        return flagged, {
            "type": type(self),
            "response": parse_litellm_response(llm_response),
            "response_string": self.blocked_response_string
            if flagged
            else query_string,
        }


@register_method(DetectionCategory.Integrity, HALLUCINATION_LLM)
class LlmHallucination(LlmBaseDetectorWithContext):
    def __init__(
        self,
        hub_name: str = "openai",
        model_name: str = "gpt-4-turbo",
        api_key: Optional[str] = None,
        context: Optional[str] = None,
    ):
        super().__init__(
            method_name=HALLUCINATION_LLM,
            hub_name=hub_name,
            model_name=model_name,
            api_key=api_key,
            context=context,
        )

    async def detect(self, query_string: str) -> DetectionResult:
        if self.context is None:
            raise ValueError(
                "No context provided for LLM-Based Hallucination detection!"
            )
        content = format_llm_hallucination_prompt(self.context, query_string)
        llm_response = litellm_completion(
            model=self.model_name,
            messages=[{"role": "system", "content": content}],
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        if not llm_response.choices or len(llm_response.choices) == 0:
            raise ValueError("No moderation results returned from OpenAI")
        llm_dec = llm_response.choices[0].message.content
        flagged = "no" in llm_dec.lower()
        return flagged, {
            "type": type(self),
            "response": parse_litellm_response(llm_response),
            "response_string": self.blocked_response_string
            if flagged
            else query_string,
        }


@register_method(DetectionCategory.Integrity, FACTCHECK_LLM)
class LlmFactcheck(LlmBaseDetectorWithContext):
    def __init__(
        self,
        hub_name: str = "openai",
        model_name: str = "gpt-4-turbo",
        api_key: Optional[str] = None,
        context: Optional[str] = None,
    ):
        super().__init__(
            method_name=FACTCHECK_LLM,
            hub_name=hub_name,
            model_name=model_name,
            api_key=api_key,
            context=context,
        )

    async def detect(self, query_string: str) -> DetectionResult:
        if self.context is None:
            raise ValueError("No context provided for LLM-Based fact-check")
        content = format_llm_factcheck_prompt(self.context, query_string)
        llm_response = litellm_completion(
            model=self.model_name,
            messages=[{"role": "system", "content": content}],
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        if not llm_response.choices or len(llm_response.choices) == 0:
            raise ValueError("No moderation results returned from OpenAI")
        llm_dec = llm_response.choices[0].message.content
        flagged = "no" in llm_dec.lower()
        return flagged, {
            "type": type(self),
            "response": parse_litellm_response(llm_response),
            "response_string": self.blocked_response_string
            if flagged
            else query_string,
        }
