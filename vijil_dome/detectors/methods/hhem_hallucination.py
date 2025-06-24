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

import logging
from vijil_dome.detectors import (
    HHEM,
    register_method,
    DetectionCategory,
    DetectionResult,
)
from vijil_dome.detectors.utils.hf_model import HFBaseModelWithContext
import torch
import numpy as np

logger = logging.getLogger("vijil.dome")


@register_method(DetectionCategory.Integrity, HHEM)
class HhemHallucinationModel(HFBaseModelWithContext):
    # Note: Queries with a score below the factual_consistency_score_threshold are flagged
    def __init__(
        self,
        context: str = "",
        factual_consistency_score_threshold: float = 0.5,
        trust_remote_code: bool = True,
    ):
        try:
            super().__init__(
                "vectara/hallucination_evaluation_model",
                tokenizer_name="google/flan-t5-base",
                trust_remote_code=trust_remote_code,
            )
            self.context = context
            # Put the model into eval mode when initialized
            self.model.eval()
            self.score_threshold = factual_consistency_score_threshold
            self.blocked_response_string = f"Method:{HHEM}"

        except Exception as e:
            logger.error(f"Failed to initialize HHEM model: {str(e)}")
            raise

    # Flagged if contradiction is found
    async def detect(self, query_string: str) -> DetectionResult:
        with torch.no_grad():
            prediction = self.model.predict(
                [(self.context if self.context else "", query_string)]
            )
        scores = prediction.cpu().detach().numpy().flatten()
        flagged = scores[0] < self.score_threshold
        return bool(flagged), {
            "type": type(self),
            "predictions": prediction,
            "response_string": self.blocked_response_string
            if flagged
            else query_string,
        }
