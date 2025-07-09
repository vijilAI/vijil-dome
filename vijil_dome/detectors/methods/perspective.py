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
from typing import Optional, Dict
try:
    from googleapiclient import discovery
except ImportError as e:
    raise ImportError(
        "Google extra is required for PerspectiveAPI. "
        "Please install it with 'pip install vijil-dome[google]'."
    ) from e
from vijil_dome.detectors import (
    DetectionMethod,
    register_method,
    DetectionCategory,
    DetectionResult,
    MODERATION_PERSPECTIVE,
)


@register_method(DetectionCategory.Moderation, MODERATION_PERSPECTIVE)
class PerspectiveAPI(DetectionMethod):
    def __init__(
        self,
        api_key: Optional[str] = None,
        attributes: Dict = {"TOXICITY": {}},
        score_threshold: Dict = {"TOXICITY": 0.5},
    ):
        self.api_key = (
            api_key if api_key is not None else os.getenv("PERSPECTIVE_API_KEY")
        )
        self.client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=self.api_key,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )
        self.attributes = attributes
        self.score_threshold = score_threshold
        self.blocked_response_string = f"Method:{MODERATION_PERSPECTIVE}"

    async def detect(self, query_string: str) -> DetectionResult:
        # Handle empty query string gracefully
        if not len(query_string):
            return False, {"type": type(self), "response": None}
        perspective_response = (
            self.client.comments()
            .analyze(
                body={
                    "comment": {"text": query_string},
                    "requestedAttributes": {attr: {} for attr in self.attributes},
                }
            )
            .execute()
        )
        flagged = False
        for attr in self.attributes:
            attr_score = perspective_response["attributeScores"][attr]["summaryScore"][
                "value"
            ]
            if attr in self.score_threshold:
                if attr_score > self.score_threshold[attr]:
                    flagged = True
        return flagged, {
            "type": type(self),
            "response": perspective_response,
            "response_string": self.blocked_response_string
            if flagged
            else query_string,
        }
