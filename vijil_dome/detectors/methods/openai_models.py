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

from typing import Optional
from litellm import moderation as litellm_moderation

from vijil_dome.detectors import (
    MODERATION_OPENAI,
    DetectionCategory,
    DetectionResult,
    register_method,
)
from vijil_dome.detectors.utils.llm_api_base import (
    LlmBaseDetector,
)


@register_method(DetectionCategory.Moderation, MODERATION_OPENAI)
class OpenAIModerations(LlmBaseDetector):
    def __init__(self, score_threshold_dict: Optional[dict] = None):
        super().__init__(
            method_name=MODERATION_OPENAI,
            hub_name="openai",
            model_name="text-moderation-latest",
        )

        # Support for using custom thresholds per moderation category
        # See: https://platform.openai.com/docs/guides/moderation/overview for a list of categories
        self.score_threshold_dict = score_threshold_dict

    async def detect(self, query_string: str) -> DetectionResult:
        oai_response = litellm_moderation(input=query_string)
        oai_moderation = oai_response.results
        if not oai_moderation or len(oai_moderation) == 0:
            raise ValueError("No moderation results returned from OpenAI")

        if self.score_threshold_dict is None:
            flagged = oai_moderation[0].flagged
        else:
            flagged = False
            for category in self.score_threshold_dict:
                if len(oai_moderation) == 0:
                    raise ValueError("No moderation results returned from OpenAI")
                moderation = oai_moderation[0]
                if moderation.category_scores is None:
                    raise ValueError("No category scores found in moderation results")
                if category not in moderation.category_scores:
                    raise ValueError(
                        f"Category {category} not found in moderation results"
                    )
                score = moderation.category_scores[category]
                if score > self.score_threshold_dict[category]:
                    flagged = True
                    break
        return flagged, {
            "type": type(self),
            "response": {
                "id": oai_response.id,
                "model": oai_response.model,
                "results": [{
                    "categories": result.categories,
                    "category_scores": result.category_scores,
                    "flagged": result.flagged
                } for result in oai_response.results]
            },
            "response_string": self.blocked_response_string
            if flagged
            else query_string,
        }
