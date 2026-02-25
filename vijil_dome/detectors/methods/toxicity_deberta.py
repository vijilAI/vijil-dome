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
    BatchDetectionResult,
)
from vijil_dome.detectors.utils.hf_model import HFBaseModel
from vijil_dome.detectors.utils.sliding_window import chunk_text
from transformers import pipeline
from typing import List, Optional

logger = logging.getLogger("vijil.dome")


@register_method(DetectionCategory.Moderation, MODERATION_DEBERTA)
class ToxicityDeberta(HFBaseModel):
    """
    https://huggingface.co/cooperleong00/deberta-v3-large_toxicity-scorer
    """

    def __init__(
        self,
        truncation=True,
        max_length=208,
        window_stride: int = 104,
        device: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        truncation:
            Whether to truncate inputs exceeding *max_length*.
        max_length:
            Maximum tokens per window. This model's effective limit is
            208 tokens, so the sliding window activates for most
            non-trivial inputs.
        window_stride:
            Step size in tokens between sliding windows for inputs that
            exceed *max_length*. Default 104 (half of *max_length*).
            A smaller stride increases overlap and detection
            thoroughness at the cost of speed.
        device:
            Torch device string (e.g. ``"cpu"``, ``"cuda:0"``). If
            *None*, CUDA is used when available.
        """
        try:
            model_path = os.path.join(
                os.path.dirname(__file__), "models", "deberta-toxicity"
            )
            if os.path.exists(model_path):
                super().__init__(model_path, local_files_only=True)
            else:
                super().__init__("cooperleong00/deberta-v3-large_toxicity-scorer")
            self.max_length = max_length
            self.window_stride = window_stride
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

    def sync_detect(
        self, query_string: str, agent_id: Optional[str] = None
    ) -> DetectionResult:
        chunks = chunk_text(
            query_string, self.tokenizer, self.max_length, self.window_stride
        )
        num_windows = len(chunks)

        if num_windows == 1:
            pred = self.classifier(query_string)
            flagged = pred[0]["label"] == "LABEL_1"
            return flagged, {
                "type": type(self),
                "predictions": pred,
                "response_string": self.blocked_response_string
                if flagged
                else query_string,
                "num_windows": 1,
            }

        # Multi-window: batch all chunks, any-positive aggregation
        all_preds = self.classifier(chunks)
        flagged = False
        for window_pred in all_preds:
            item = window_pred[0] if isinstance(window_pred, list) else window_pred
            if item["label"] == "LABEL_1":
                flagged = True
                break

        return flagged, {
            "type": type(self),
            "predictions": all_preds,
            "response_string": self.blocked_response_string
            if flagged
            else query_string,
            "num_windows": num_windows,
        }

    async def detect(self, query_string: str) -> DetectionResult:
        logger.info("Detecting using Deberta Toxicity Model...")
        return self.sync_detect(query_string)

    async def detect_batch(self, inputs: List[str]) -> BatchDetectionResult:
        # Phase 1: chunk each input, build flat list + per-input ranges
        flat_chunks: List[str] = []
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

        # Phase 3: re-aggregate per input using any-positive
        results = []
        for query_string, (start, end) in zip(inputs, ranges):
            chunk_preds = all_preds[start:end]
            num_windows = end - start
            flagged = False
            for pred in chunk_preds:
                item = pred[0] if isinstance(pred, list) else pred
                if item["label"] == "LABEL_1":
                    flagged = True
                    break
            results.append((flagged, {
                "type": type(self),
                "predictions": chunk_preds,
                "response_string": self.blocked_response_string
                if flagged
                else query_string,
                "num_windows": num_windows,
            }))
        return results
