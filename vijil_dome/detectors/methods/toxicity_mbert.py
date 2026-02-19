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
from vijil_dome.detectors import (
    MODERATION_MBERT,
    register_method,
    DetectionCategory,
    DetectionResult,
)
from transformers import pipeline
from vijil_dome.detectors.utils.hf_model import HFBaseModel
from typing import Optional

logger = logging.getLogger("vijil.dome")


@register_method(DetectionCategory.Moderation, MODERATION_MBERT)
class MBertToxicContentModel(HFBaseModel):
    """
    Vijil Finetuned ModernBERT model for toxic content detection.
    https://huggingface.co/vijil/vijil_dome_toxic_content_detection
    """

    def __init__(
        self,
        score_threshold: float = 0.5,
        truncation: bool = True,
        max_length: int = 512,
    ):
        try:
            super().__init__(
                model_name="vijil/vijil_dome_toxic_content_detection",
                tokenizer_name="answerdotai/ModernBERT-base",
            )

            self.score_threshold = score_threshold
            self.classifier = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                truncation=truncation,
                max_length=max_length,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )
            self.response_string = f"Method:{MODERATION_MBERT}"
            self.run_in_executor = True
            logger.info("Initialized Vijil MBert toxic content model..")
        except Exception as e:
            logger.error(
                f"Failed to initialize MBert toxic content model: {str(e)}"
            )
            raise

    def sync_detect(
        self, query_string: str, agent_id: Optional[str] = None
    ) -> DetectionResult:
        pred = self.classifier(query_string)
        if pred[0]["label"] in ("toxic", "LABEL_1", 1, "1"):
            toxic_score = pred[0]["score"]
        else:
            toxic_score = 1.0 - pred[0]["score"]
        flagged = toxic_score >= self.score_threshold
        return flagged, {
            "type": type(self),
            "score": toxic_score,
            "predictions": pred,
            "response_string": self.response_string if flagged else query_string,
        }

    async def detect(self, query_string: str) -> DetectionResult:
        logger.info(f"Detecting using {self.__class__.__name__}...")
        return self.sync_detect(query_string)
