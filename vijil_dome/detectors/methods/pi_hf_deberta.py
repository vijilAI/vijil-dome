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
import os

import torch
from transformers import pipeline

from vijil_dome.detectors import (
    PI_DEBERTA_FINETUNED_11122024,
    PI_DEBERTA_V3_BASE,
    BatchDetectionResult,
    DetectionCategory,
    DetectionResult,
    register_method,
)
from vijil_dome.detectors.utils.hf_model import HFBaseModel
from vijil_dome.detectors.utils.sliding_window import chunk_text
from vijil_dome.types import DomePayload

logger = logging.getLogger("vijil.dome")


class BaseDebertaPromptInjectionModel(HFBaseModel):
    """
    Base class for DeBERTa-based prompt injection detection models.
    """

    def __init__(
        self,
        model_identifier: str,
        response_method: str,
        model_dir: str = "deberta-prompt-injection",
        truncation: bool = True,
        max_length: int = 512,
        window_stride: int = 448,
    ):
        """
        Parameters
        ----------
        model_identifier:
            HuggingFace model name or path.
        response_method:
            Method name used in blocked response strings.
        model_dir:
            Local model directory name under ``methods/models/``.
        truncation:
            Whether to truncate inputs exceeding *max_length*.
        max_length:
            Maximum tokens per window (DeBERTa supports up to 512).
        window_stride:
            Step size in tokens between sliding windows for inputs that
            exceed *max_length*. Default 448 (64-token overlap).
        """
        try:
            model_path = os.path.join(
                os.path.dirname(__file__),
                "models",
                model_dir,
            )
            if os.path.exists(model_path):
                super().__init__(model_path, local_files_only=True)
            else:
                super().__init__(model_identifier)

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
            self.response_string = f"Method:{response_method}"
            self.run_in_executor = True
            logger.info("Initialized security model..")
        except Exception as e:
            logger.error(f"Failed to initialize DeBERTa model: {e!s}")
            raise

    def sync_detect(
        self,
        dome_input: DomePayload,
        agent_id: str | None = None,
        team_id: str | None = None,
        user_id: str | None = None,
    ) -> DetectionResult:
        dome_input = DomePayload.coerce(dome_input)
        query_string = dome_input.query_string
        chunks = chunk_text(
            query_string, self.tokenizer, self.max_length, self.window_stride
        )
        num_windows = len(chunks)

        if num_windows == 1:
            pred = self.classifier(query_string)
            flagged = pred[0]["label"] != "SAFE"
            return flagged, {
                "type": type(self),
                "predictions": pred,
                "response_string": self.response_string if flagged else query_string,
                "num_windows": 1,
            }

        # Multi-window: batch all chunks through pipeline, any-positive aggregation
        all_preds = self.classifier(chunks, batch_size=self.max_batch_concurrency)
        flagged = False
        for window_pred in all_preds:
            item = window_pred[0] if isinstance(window_pred, list) else window_pred
            if item["label"] != "SAFE":
                flagged = True
                break

        return flagged, {
            "type": type(self),
            "predictions": all_preds,
            "response_string": self.response_string if flagged else query_string,
            "num_windows": num_windows,
        }

    async def detect(self, dome_input: DomePayload) -> DetectionResult:
        dome_input = DomePayload.coerce(dome_input)
        logger.info(f"Detecting using {self.__class__.__name__}...")
        return self.sync_detect(dome_input)

    async def detect_batch(self, inputs: list[str | DomePayload]) -> BatchDetectionResult:
        dome_inputs = [DomePayload.coerce(x) for x in inputs]
        # Phase 1: chunk each input, build flat list + per-input ranges
        flat_chunks: list[str] = []
        ranges = []
        for di in dome_inputs:
            query_string = di.query_string
            chunks = chunk_text(
                query_string, self.tokenizer, self.max_length, self.window_stride
            )
            start = len(flat_chunks)
            flat_chunks.extend(chunks)
            ranges.append((start, len(flat_chunks)))

        # Phase 2: pipeline call on all chunks (batched)
        all_preds = self.classifier(flat_chunks, batch_size=self.max_batch_concurrency)

        # Phase 3: re-aggregate per input using any-positive
        results = []
        for dome_item, (start, end) in zip(dome_inputs, ranges):
            query_string = dome_item.query_string
            chunk_preds = all_preds[start:end]
            num_windows = end - start
            flagged = False
            for pred in chunk_preds:
                item = pred[0] if isinstance(pred, list) else pred
                if item["label"] != "SAFE":
                    flagged = True
                    break
            results.append((flagged, {
                "type": type(self),
                "predictions": chunk_preds,
                "response_string": self.response_string if flagged else query_string,
                "num_windows": num_windows,
            }))
        return results


@register_method(DetectionCategory.Security, PI_DEBERTA_V3_BASE)
class DebertaPromptInjectionModel(BaseDebertaPromptInjectionModel):
    """
    https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2
    """

    def __init__(
        self,
        truncation: bool = True,
        max_length: int = 512,
        window_stride: int = 448,
    ):
        super().__init__(
            model_identifier="protectai/deberta-v3-base-prompt-injection-v2",
            response_method=PI_DEBERTA_V3_BASE,
            truncation=truncation,
            max_length=max_length,
            window_stride=window_stride,
        )


@register_method(DetectionCategory.Security, PI_DEBERTA_FINETUNED_11122024)
class DebertaTuned60PromptInjectionModel(BaseDebertaPromptInjectionModel):
    """
    https://huggingface.co/vijil/pi_deberta_finetuned_11122024
    """

    def __init__(
        self,
        truncation: bool = True,
        max_length: int = 512,
        window_stride: int = 448,
    ):
        super().__init__(
            model_identifier="vijil/pi_deberta_finetuned_11122024",
            response_method=PI_DEBERTA_FINETUNED_11122024,
            truncation=truncation,
            max_length=max_length,
            window_stride=window_stride,
        )

