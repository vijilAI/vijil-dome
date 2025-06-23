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
    FACTCHECK_ROBERTA,
    register_method,
    DetectionCategory,
    DetectionResult,
)
from vijil_dome.detectors.utils.hf_model import HFBaseModelWithContext
import torch

logger = logging.getLogger("vijil.dome")


# Note: This model evaluates if the query contradicts established context
# If the query is untrue, but is not contradicted by the context, it will NOT be flagged
@register_method(DetectionCategory.Integrity, FACTCHECK_ROBERTA)
class RobertaFactCheckModel(HFBaseModelWithContext):
    """
    A concrete implementation of HFBaseModel using a roberta model fine tuned for fact-checking
    """

    def __init__(self, context: str = ""):
        try:
            super().__init__("Dzeniks/roberta-fact-check")
            self.context = context
            # Put the model into eval mode when initialized
            self.model.eval()
            self.blocked_response_string = f"Method:{FACTCHECK_ROBERTA}"

        except Exception as e:
            logger.error(f"Failed to initialize RoBERTa model: {str(e)}")
            raise

    # Flagged if contradiction is found
    async def detect(self, query_string: str) -> DetectionResult:
        if self.context is None:
            raise ValueError("No context is available for roberta fact-check!")
        encoded_dict = self.tokenizer.encode_plus(
            query_string, self.context, return_tensors="pt"
        )
        with torch.no_grad():
            prediction = self.model(**encoded_dict)
        flagged = torch.argmax(prediction[0]).item()
        return bool(flagged), {
            "type": type(self),
            "response_string": self.blocked_response_string
            if flagged
            else query_string,
        }
