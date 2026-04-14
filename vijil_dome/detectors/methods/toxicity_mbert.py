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

"""
Toxicity / moderation detection using Vijil's ModernBERT fine-tune — three modes:

  fast (default): ModernBERT binary classifier (<5ms, self-hosted)
  safeguard:      GPT-OSS-Safeguard-20B via Groq API (~200ms, high accuracy)
  hybrid:         ModernBERT first, Safeguard on low-confidence (~5ms avg)
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import httpx
import torch
from transformers import pipeline

from vijil_dome.detectors import (
    MODERATION_MBERT,
    MODERATION_MBERT_SAFEGUARD,
    MODERATION_MBERT_HYBRID,
    register_method,
    DetectionCategory,
    DetectionMethod,
    DetectionResult,
    BatchDetectionResult,
)
from vijil_dome.detectors.utils.hf_model import HFBaseModel
from vijil_dome.detectors.utils.sliding_window import chunk_text
from vijil_dome.types import DomePayload

logger = logging.getLogger("vijil.dome")

MODERATION_SAFEGUARD_SYSTEM_PROMPT = (
    "Evaluate if this text contains toxic, harmful, or abusive content. "
    "Toxic content includes: hate speech, threats of violence, harassment, "
    "bullying, sexually explicit material, self-harm encouragement, "
    "dangerous instructions, or content that demeans individuals or groups. "
    "Reply with exactly one word: safe or unsafe"
)

DEFAULT_SAFEGUARD_TEMPERATURE = 0.0
# gpt-oss-safeguard-20b is a reasoning model and reserves a chunk of the
# token budget for internal reasoning before emitting any assistant content.
# A small cap (e.g. 8) lets the reasoning alone blow the budget, hit
# finish_reason=length, and return an empty `content` — which silently
# classifies everything as safe. Keep this generous.
DEFAULT_SAFEGUARD_MAX_TOKENS = 2000
DEFAULT_SAFEGUARD_MAX_INPUT_CHARS = 400_000

GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"


# ----------------------------------------------------------------------
# Fast mode (original detector, unchanged interface)
# ----------------------------------------------------------------------


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
        max_length: int = 8192,
        window_stride: int = 4096,
    ):
        """
        Parameters
        ----------
        score_threshold:
            Toxicity probability above which input is flagged.
        truncation:
            Whether to truncate inputs exceeding *max_length*.
        max_length:
            Maximum tokens per window. ModernBERT natively supports up
            to 8192 tokens, so sliding windows only activate for very
            long inputs.
        window_stride:
            Step size in tokens between sliding windows for inputs that
            exceed *max_length*. Default 4096 (half of *max_length*).
        """
        try:
            super().__init__(
                model_name="vijil/vijil_dome_toxic_content_detection",
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
            self.response_string = f"Method:{MODERATION_MBERT}"
            self.run_in_executor = True
            logger.info("Initialized Vijil MBert toxic content model..")
        except Exception as e:
            logger.error(
                f"Failed to initialize MBert toxic content model: {str(e)}"
            )
            raise

    def _extract_toxic_score(self, item):
        if item["label"] in ("toxic", "LABEL_1", 1, "1"):
            return item["score"]
        return 1.0 - item["score"]

    # ------------------------------------------------------------------
    # Score-only classification (used by Hybrid)
    # ------------------------------------------------------------------

    def _classify(self, dome_input: DomePayload) -> Tuple[float, dict]:
        """Run ModernBERT classification on a single DomePayload.

        Returns ``(toxic_score, best_prediction_item)`` before any
        flagging decision, so the hybrid class can inspect confidence.
        """
        query_string = dome_input.query_string
        chunks = chunk_text(
            query_string, self.tokenizer, self.max_length, self.window_stride
        )
        if len(chunks) == 1:
            pred = self.classifier(query_string)
            item = pred[0]
            return self._extract_toxic_score(item), item

        all_preds = self.classifier(chunks, batch_size=self.max_batch_concurrency)
        best_score = 0.0
        best_item: dict = {}
        for window_pred in all_preds:
            item = window_pred[0] if isinstance(window_pred, list) else window_pred  # type: ignore[assignment]
            score = self._extract_toxic_score(item)
            if score > best_score:
                best_score = score
                best_item = item
        return best_score, best_item

    def _classify_batch(
        self, dome_inputs: List[DomePayload]
    ) -> List[Tuple[float, dict]]:
        """Run ModernBERT classification on many DomePayloads in one call.

        All chunks from all payloads are flattened into a single pipeline
        call for efficient batching, then re-aggregated per payload.
        """
        flat_chunks: List[str] = []
        ranges = []
        for di in dome_inputs:
            query_string = di.query_string
            chunks = chunk_text(
                query_string, self.tokenizer, self.max_length, self.window_stride
            )
            start = len(flat_chunks)
            flat_chunks.extend(chunks)
            ranges.append((start, len(flat_chunks)))

        all_preds = self.classifier(flat_chunks, batch_size=self.max_batch_concurrency)

        results = []
        for start, end in ranges:
            chunk_preds = all_preds[start:end]
            best_score = 0.0
            best_item: dict = {}
            for pred in chunk_preds:
                item = pred[0] if isinstance(pred, list) else pred
                score = self._extract_toxic_score(item)
                if score > best_score:
                    best_score = score
                    best_item = item
            results.append((best_score, best_item))
        return results

    # ------------------------------------------------------------------
    # Public detection interface (unchanged)
    # ------------------------------------------------------------------

    def sync_detect(
        self,
        dome_input: DomePayload,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> DetectionResult:
        dome_input = DomePayload.coerce(dome_input)
        query_string = dome_input.query_string
        chunks = chunk_text(
            query_string, self.tokenizer, self.max_length, self.window_stride
        )
        num_windows = len(chunks)

        if num_windows == 1:
            pred = self.classifier(query_string)
            toxic_score = self._extract_toxic_score(pred[0])
            flagged = toxic_score >= self.score_threshold
            return flagged, {
                "type": type(self),
                "score": toxic_score,
                "predictions": pred,
                "response_string": self.response_string if flagged else query_string,
                "num_windows": 1,
            }

        # Multi-window: batch all chunks, any-positive with max score
        all_preds = self.classifier(chunks, batch_size=self.max_batch_concurrency)
        max_score = 0.0
        for window_pred in all_preds:
            item = window_pred[0] if isinstance(window_pred, list) else window_pred
            score = self._extract_toxic_score(item)
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

    async def detect(self, dome_input: DomePayload) -> DetectionResult:
        dome_input = DomePayload.coerce(dome_input)
        logger.info(f"Detecting using {self.__class__.__name__}...")
        return self.sync_detect(dome_input)

    async def detect_batch(self, inputs: List[Union[str, DomePayload]]) -> BatchDetectionResult:
        dome_inputs = [DomePayload.coerce(x) for x in inputs]
        # Phase 1: chunk each input, build flat list + per-input ranges
        flat_chunks: List[str] = []
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

        # Phase 3: re-aggregate per input using any-positive with max score
        results = []
        for dome_item, (start, end) in zip(dome_inputs, ranges):
            query_string = dome_item.query_string
            chunk_preds = all_preds[start:end]
            num_windows = end - start
            max_score = 0.0
            for pred in chunk_preds:
                item = pred[0] if isinstance(pred, list) else pred
                score = self._extract_toxic_score(item)
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


# ----------------------------------------------------------------------
# Safeguard mode (API-only)
# ----------------------------------------------------------------------


@register_method(DetectionCategory.Moderation, MODERATION_MBERT_SAFEGUARD)
class ModerationMbertSafeguard(DetectionMethod):
    """
    Toxicity / moderation detection using GPT-OSS-Safeguard-20B via Groq.
    ~200ms latency, high accuracy, requires GROQ_API_KEY.

    Does not load ModernBERT \u2014 this mode is API-only.
    """

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        groq_model: str = "openai/gpt-oss-safeguard-20b",
        temperature: float = DEFAULT_SAFEGUARD_TEMPERATURE,
        max_tokens: int = DEFAULT_SAFEGUARD_MAX_TOKENS,
        timeout_seconds: float = 10.0,
        max_input_chars: Optional[int] = DEFAULT_SAFEGUARD_MAX_INPUT_CHARS,
        **kwargs,
    ):
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")
        self.groq_model = groq_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.max_input_chars = max_input_chars
        self.response_string = f"Method:{MODERATION_MBERT_SAFEGUARD}"
        self.run_in_executor = False
        if not self.groq_api_key:
            logger.warning(
                f"GROQ_API_KEY not set \u2014 {MODERATION_MBERT_SAFEGUARD} will fail at runtime"
            )
        logger.info("Initialized moderation mbert detector (Safeguard mode)")

    def _truncate_if_needed(self, text: str) -> str:
        if self.max_input_chars is not None and len(text) > self.max_input_chars:
            logger.warning(
                "moderation-mbert-safeguard input truncated from %d to %d chars",
                len(text),
                self.max_input_chars,
            )
            return text[: self.max_input_chars]
        return text

    def _groq_payload(self, query_string: str) -> dict:
        return {
            "model": self.groq_model,
            "messages": [
                {"role": "system", "content": MODERATION_SAFEGUARD_SYSTEM_PROMPT},
                {"role": "user", "content": query_string},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "reasoning_effort": "low",
        }

    async def _call_safeguard(
        self, client: httpx.AsyncClient, query_string: str
    ) -> str:
        resp = await client.post(
            GROQ_CHAT_COMPLETIONS_URL,
            headers={
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json",
            },
            json=self._groq_payload(query_string),
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"].get("content", "").strip().lower()

    async def detect(self, dome_input: DomePayload) -> DetectionResult:
        dome_input = DomePayload.coerce(dome_input)
        query_string = self._truncate_if_needed(dome_input.query_string)
        logger.info("Detecting toxicity using Safeguard...")
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                content = await self._call_safeguard(client, query_string)
            flagged = "unsafe" in content
            return flagged, {
                "type": type(self),
                "detector": MODERATION_MBERT_SAFEGUARD,
                "score": 1.0 if flagged else 0.0,
                "label": "toxic" if flagged else "safe",
                "safeguard_verdict": content,
                "response_string": self.response_string if flagged else query_string,
            }
        except Exception as e:
            logger.error(f"Moderation Safeguard API call failed: {e}")
            return False, {
                "type": type(self),
                "detector": MODERATION_MBERT_SAFEGUARD,
                "score": 0.0,
                "label": "error",
                "error": str(e),
                "response_string": query_string,
            }

    async def detect_batch(
        self, inputs: List[Union[str, DomePayload]]
    ) -> BatchDetectionResult:
        return await self._gather_with_concurrency(
            [self.detect(DomePayload.coerce(item)) for item in inputs]
        )


# ----------------------------------------------------------------------
# Hybrid mode (fast + Safeguard escalation)
# ----------------------------------------------------------------------


@register_method(DetectionCategory.Moderation, MODERATION_MBERT_HYBRID)
class ModerationMbertHybrid(MBertToxicContentModel):
    """
    Hybrid toxicity / moderation detection: ModernBERT first, Safeguard on
    low-confidence. ~5ms average, near-100% accuracy, API cost only on
    uncertain examples.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.85,
        groq_api_key: Optional[str] = None,
        groq_model: str = "openai/gpt-oss-safeguard-20b",
        temperature: float = DEFAULT_SAFEGUARD_TEMPERATURE,
        max_tokens: int = DEFAULT_SAFEGUARD_MAX_TOKENS,
        timeout_seconds: float = 10.0,
        max_input_chars: Optional[int] = DEFAULT_SAFEGUARD_MAX_INPUT_CHARS,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.confidence_threshold = confidence_threshold
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")
        self.groq_model = groq_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.max_input_chars = max_input_chars
        self.response_string = f"Method:{MODERATION_MBERT_HYBRID}"
        self.run_in_executor = False  # async for Safeguard fallback

        if not self.groq_api_key:
            logger.warning(
                f"GROQ_API_KEY not set \u2014 {MODERATION_MBERT_HYBRID} will fall back to fast-only"
            )
        logger.info(
            "Initialized moderation mbert detector "
            f"(hybrid mode, confidence_threshold={confidence_threshold})"
        )

    # ------------------------------------------------------------------
    # Payload builders
    # ------------------------------------------------------------------

    def _fast_payload(
        self,
        dome_input: DomePayload,
        score: float,
        prediction: dict,
        stage: str,
    ) -> Dict:
        flagged = score >= self.score_threshold
        return {
            "type": type(self),
            "detector": MODERATION_MBERT_HYBRID,
            "stage": stage,
            "score": score,
            "prediction": prediction,
            "threshold": self.score_threshold,
            "confidence": max(score, 1.0 - score),
            "label": "toxic" if flagged else "safe",
            "response_string": (
                self.response_string if flagged else dome_input.query_string
            ),
        }

    def _safeguard_payload(
        self,
        dome_input: DomePayload,
        content: str,
        fast_score: float,
    ) -> Dict:
        flagged = "unsafe" in content
        return {
            "type": type(self),
            "detector": MODERATION_MBERT_HYBRID,
            "stage": "safeguard",
            "score": 1.0 if flagged else 0.0,
            "fast_score": fast_score,
            "fast_confidence": max(fast_score, 1.0 - fast_score),
            "label": "toxic" if flagged else "safe",
            "safeguard_verdict": content,
            "response_string": (
                self.response_string if flagged else dome_input.query_string
            ),
        }

    def _truncate_if_needed(self, text: str) -> str:
        if self.max_input_chars is not None and len(text) > self.max_input_chars:
            logger.warning(
                "moderation-mbert-hybrid escalation input truncated from %d to %d chars",
                len(text),
                self.max_input_chars,
            )
            return text[: self.max_input_chars]
        return text

    # ------------------------------------------------------------------
    # Safeguard escalation
    # ------------------------------------------------------------------

    async def _escalate(
        self,
        client: httpx.AsyncClient,
        dome_input: DomePayload,
        fast_score: float,
        prediction: dict,
    ) -> DetectionResult:
        """Run Safeguard on one low-confidence item, with graceful fallback."""
        query_string = self._truncate_if_needed(dome_input.query_string)
        try:
            resp = await client.post(
                GROQ_CHAT_COMPLETIONS_URL,
                headers={
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.groq_model,
                    "messages": [
                        {"role": "system", "content": MODERATION_SAFEGUARD_SYSTEM_PROMPT},
                        {"role": "user", "content": query_string},
                    ],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "reasoning_effort": "low",
                },
            )
            resp.raise_for_status()
            content = (
                resp.json()["choices"][0]["message"].get("content", "").strip().lower()
            )
            payload = self._safeguard_payload(dome_input, content, fast_score)
            return "unsafe" in content, payload
        except Exception as e:
            logger.error(
                f"Moderation Safeguard escalation failed: {e}, using fast result"
            )
            payload = self._fast_payload(
                dome_input, fast_score, prediction, stage="fast-fallback"
            )
            payload["error"] = str(e)
            return fast_score >= self.score_threshold, payload

    # ------------------------------------------------------------------
    # Public detection interface
    # ------------------------------------------------------------------

    async def detect(self, dome_input: DomePayload) -> DetectionResult:
        dome_input = DomePayload.coerce(dome_input)
        logger.info("Detecting toxicity using hybrid mode...")

        # Stage 1: Fast ModernBERT classification in a thread executor
        loop = asyncio.get_running_loop()
        score, prediction = await loop.run_in_executor(
            None, self._classify, dome_input
        )
        confidence = max(score, 1.0 - score)
        logger.debug(
            "moderation-mbert-hybrid fast-stage prediction=%s score=%.3f",
            prediction,
            score,
        )

        # Confident enough -> return the fast result immediately.
        if confidence >= self.confidence_threshold:
            flagged = score >= self.score_threshold
            return flagged, self._fast_payload(
                dome_input, score, prediction, stage="fast"
            )

        # Low confidence -> escalate to Safeguard if we have a key.
        if not self.groq_api_key:
            flagged = score >= self.score_threshold
            return flagged, self._fast_payload(
                dome_input, score, prediction, stage="fast-fallback"
            )

        logger.info(
            f"Low confidence ({confidence:.2f} < {self.confidence_threshold}), "
            "escalating to Safeguard..."
        )
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            return await self._escalate(client, dome_input, score, prediction)

    async def detect_batch(
        self, inputs: List[Union[str, DomePayload]]
    ) -> BatchDetectionResult:
        """Batched hybrid detection.

        Runs the fast ModernBERT stage on the whole batch in one pipeline
        call, then escalates any low-confidence items to Safeguard
        concurrently via ``asyncio.gather``.
        """
        dome_inputs = [DomePayload.coerce(x) for x in inputs]

        # Stage 1: batched fast classification in a thread executor.
        loop = asyncio.get_running_loop()
        scored = await loop.run_in_executor(
            None, self._classify_batch, dome_inputs
        )

        # Partition into "confident enough" vs "escalate".
        results: List[Optional[DetectionResult]] = [None] * len(dome_inputs)
        escalate_indices: List[int] = []
        for idx, (dome_input, (score, prediction)) in enumerate(
            zip(dome_inputs, scored)
        ):
            confidence = max(score, 1.0 - score)
            if confidence >= self.confidence_threshold:
                flagged = score >= self.score_threshold
                results[idx] = (
                    flagged,
                    self._fast_payload(dome_input, score, prediction, stage="fast"),
                )
            elif not self.groq_api_key:
                flagged = score >= self.score_threshold
                results[idx] = (
                    flagged,
                    self._fast_payload(
                        dome_input, score, prediction, stage="fast-fallback"
                    ),
                )
            else:
                escalate_indices.append(idx)

        # Stage 2: concurrent Safeguard escalation for low-confidence items.
        if escalate_indices:
            logger.info(
                "moderation-mbert-hybrid escalating %d/%d low-confidence items",
                len(escalate_indices),
                len(dome_inputs),
            )
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                semaphore = asyncio.Semaphore(self.max_batch_concurrency)

                async def run_one(i: int) -> DetectionResult:
                    async with semaphore:
                        score, prediction = scored[i]
                        return await self._escalate(
                            client, dome_inputs[i], score, prediction
                        )

                escalated = await asyncio.gather(
                    *(run_one(i) for i in escalate_indices)
                )
                for i, r in zip(escalate_indices, escalated):
                    results[i] = r

        return [r for r in results if r is not None]
