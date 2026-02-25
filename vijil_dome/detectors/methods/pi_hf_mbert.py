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
    PI_MBERT,
    register_method,
    DetectionCategory,
    DetectionResult,
    BatchDetectionResult,
)
from transformers import pipeline
from vijil_dome.detectors.utils.hf_model import HFBaseModel
from vijil_dome.detectors.utils.sliding_window import chunk_text
from typing import List, Optional

logger = logging.getLogger("vijil.dome")


@register_method(DetectionCategory.Security, PI_MBERT)
class MBertPromptInjectionModel(HFBaseModel):
    """
    Vijil Finetuned MBERT model
    """

    def __init__(
        self,
        score_threshold: float = 0.5,
        truncation: bool = True,
        max_length: int = 8192,
        window_stride: int = 4096,
    ):
        try:
            super().__init__(
                model_name="vijil/vijil_dome_prompt_injection_detection",
                tokenizer_name="answerdotai/ModernBERT-base",
            )

            self.score_threshold = score_threshold
            self.max_length = max_length
            self.window_stride = window_stride
            self.classifier = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                truncation=truncation,
                max_length=max_length,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )
            self.response_string = f"Method:{PI_MBERT}"
            self.run_in_executor = True
            logger.info("Initialized Vijil Mbert model..")
        except Exception as e:
            logger.error(f"Failed to initialize MBert model: {str(e)}")
            raise

    def _extract_injection_score(self, item):
        if item["label"] in (1, "1", "LABEL_1"):
            return item["score"]
        return 1.0 - item["score"]

    def sync_detect(
        self, query_string: str, agent_id: Optional[str] = None
    ) -> DetectionResult:
        chunks = chunk_text(
            query_string, self.tokenizer, self.max_length, self.window_stride
        )
        num_windows = len(chunks)

        if num_windows == 1:
            pred = self.classifier(query_string)
            injection_score = self._extract_injection_score(pred[0])
            flagged = injection_score >= self.score_threshold
            return flagged, {
                "type": type(self),
                "score": injection_score,
                "predictions": pred,
                "response_string": self.response_string if flagged else query_string,
                "num_windows": 1,
            }

        # Multi-window: batch all chunks, any-positive with max score
        all_preds = self.classifier(chunks)
        max_score = 0.0
        for pred in all_preds:
            item = pred[0] if isinstance(pred, list) else pred
            score = self._extract_injection_score(item)
            if score > max_score:
                max_score = score

        flagged = max_score >= self.score_threshold
        return flagged, {
            "type": type(self),
            "score": max_score,
            "predictions": all_preds,
            "response_string": self.response_string if flagged else query_string,
            "num_windows": num_windows,
        }

    async def detect(self, query_string: str) -> DetectionResult:
        logger.info(f"Detecting using {self.__class__.__name__}...")
        return self.sync_detect(query_string)

    async def detect_batch(self, inputs: List[str]) -> BatchDetectionResult:
        # Phase 1: chunk each input, build flat list + per-input ranges
        flat_chunks = []
        ranges = []
        for text in inputs:
            chunks = chunk_text(
                text, self.tokenizer, self.max_length, self.window_stride
            )
            start = len(flat_chunks)
            flat_chunks.extend(chunks)
            ranges.append((start, len(flat_chunks)))

        # Phase 2: single pipeline call on all chunks
        all_preds = self.classifier(flat_chunks)

        # Phase 3: re-aggregate per input using any-positive with max score
        results = []
        for query_string, (start, end) in zip(inputs, ranges):
            chunk_preds = all_preds[start:end]
            num_windows = end - start
            max_score = 0.0
            for pred in chunk_preds:
                item = pred[0] if isinstance(pred, list) else pred
                score = self._extract_injection_score(item)
                if score > max_score:
                    max_score = score
            flagged = max_score >= self.score_threshold
            results.append((flagged, {
                "type": type(self),
                "score": max_score,
                "predictions": chunk_preds,
                "response_string": self.response_string if flagged else query_string,
                "num_windows": num_windows,
            }))
        return results
