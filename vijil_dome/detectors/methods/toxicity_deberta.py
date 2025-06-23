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
import torch
import os
from vijil_dome.detectors import (
    MODERATION_DEBERTA,
    register_method,
    DetectionCategory,
    DetectionResult,
)
from vijil_dome.detectors.utils.hf_model import HFBaseModel
from transformers import pipeline
from typing import Optional

logger = logging.getLogger("vijil.dome")


@register_method(DetectionCategory.Moderation, MODERATION_DEBERTA)
class ToxicityDeberta(HFBaseModel):
    """
    https://huggingface.co/cooperleong00/deberta-v3-large_toxicity-scorer
    """

    def __init__(self, truncation=True, max_length=208, device: Optional[str] = None):
        try:
            model_path = os.path.join(
                os.path.dirname(__file__), "models", "deberta-toxicity"
            )
            if os.path.exists(model_path):
                super().__init__(model_path, local_files_only=True)
            else:
                super().__init__("cooperleong00/deberta-v3-large_toxicity-scorer")
            self.classifier = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                truncation=truncation,
                max_length=max_length,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if device is None
                else torch.device(device),
            )
            self.blocked_response_string = f"Method:{MODERATION_DEBERTA}"
            self.run_in_executor = True
            logger.info("Initialized Toxicity Model")
        except Exception as e:
            logger.error(f"Failed to initialize Deberta toxicity model: {str(e)}")
            raise

    def sync_detect(self, query_string: str) -> DetectionResult:
        pred = self.classifier(query_string)
        flagged = pred[0]["label"] == "LABEL_1"
        return flagged, {
            "type": type(self),
            "predictions": pred,
            "response_string": self.blocked_response_string
            if flagged
            else query_string,
        }

    async def detect(self, query_string: str) -> DetectionResult:
        logger.info("Detecting using Deberta Toxicity Model...")
        return self.sync_detect(query_string)
