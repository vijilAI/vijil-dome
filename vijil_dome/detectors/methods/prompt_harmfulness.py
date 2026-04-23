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
Prompt Harmfulness Detection — Two modes:

  fast:      ModernBERT binary classifier (<5ms, F1=0.986)
  safeguard: GPT-OSS-Safeguard-20B via API (~200ms, ~100% accuracy)

Distilled from GPT-OSS-Safeguard-20B with a 16-category harmfulness
policy validated against 11 major AI safety taxonomies (MLCommons,
LlamaGuard 3, OpenAI, Anthropic, HarmBench, Do-Not-Answer, NIST AI
600-1, EU AI Act, Google DeepMind, WildGuard).

Detects prompts that attempt to elicit harmful, dangerous, or illegal
content from AI agents. Covers: Violence, CBRN, Weapons, Cybercrime,
Criminal activity, Self-harm, Child exploitation, Sexual content,
Non-consensual intimate content, Hate speech, Harassment, Privacy
violations, IP violations, Deceptive content, Election interference,
Regulatory evasion.

This detector fills the gap between toxicity detection (catches surface
language like profanity and slurs) and prompt injection detection
(catches adversarial system-prompt overrides). A polite request for
dangerous information passes both of those filters but is caught here.

Input format:
    This detector classifies the user's prompt only. It runs as an
    input guard — the agent's response is not needed. When given a
    ``DomePayload`` with both ``prompt`` and ``response`` set, the
    detector uses only the ``prompt`` field and emits a warning that
    the response is ignored.
"""

import logging
import os
from typing import List, Optional, Union

import httpx

try:
    import torch
    from transformers import pipeline
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from vijil_dome.detectors import (
    PROMPT_HARMFULNESS_FAST,
    PROMPT_HARMFULNESS_SAFEGUARD,
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

HARMFULNESS_SAFEGUARD_SYSTEM_PROMPT = (
    "Evaluate if this user prompt attempts to elicit harmful, dangerous, "
    "or illegal content from an AI agent. A prompt is unsafe if it "
    "requests, instructs, or manipulates the agent toward producing any "
    "of the following: (1) Violence, (2) CBRN threats, (3) Weapons and "
    "explosives, (4) Cybercrime, (5) Criminal activity, (6) Self-harm, "
    "(7) Child exploitation, (8) Sexual content, (9) Non-consensual "
    "intimate content, (10) Hate speech, (11) Harassment, (12) Privacy "
    "violations, (13) IP violations, (14) Deceptive content, "
    "(15) Election interference, (16) Regulatory evasion. "
    "A prompt is safe if it discusses these topics in an educational, "
    "journalistic, research, or defensive context without requesting "
    "actionable harmful output. "
    "Reply with exactly one word: safe or unsafe"
)

DEFAULT_SAFEGUARD_TEMPERATURE = 0.0
DEFAULT_SAFEGUARD_MAX_TOKENS = 2000
DEFAULT_SAFEGUARD_MAX_INPUT_CHARS = 400_000
DEFAULT_SAFEGUARD_API_KEY_NAME = "GROQ_API_KEY"
DEFAULT_SAFEGUARD_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_SAFEGUARD_MODEL = "openai/gpt-oss-safeguard-20b"


# ----------------------------------------------------------------------
# Fast mode
# ----------------------------------------------------------------------


@register_method(DetectionCategory.Moderation, PROMPT_HARMFULNESS_FAST)
class PromptHarmfulnessFast(HFBaseModel):
    """
    Fast prompt harmfulness detection using distilled ModernBERT.
    <5ms latency, F1=0.986, zero API cost. Self-hosted.
    """

    def __init__(
        self,
        model_name: str = "vijil/prompt-harmfulness-detector",
        tokenizer_name: str = "answerdotai/ModernBERT-base",
        score_threshold: float = 0.95,
        max_length: int = 512,
        window_stride: int = 256,
    ):
        if not _HAS_TORCH:
            raise ImportError(
                f"{PROMPT_HARMFULNESS_FAST} requires 'torch' and 'transformers'. "
                "Install with: pip install vijil-dome[local]"
            )
        try:
            super().__init__(
                model_name=model_name,
                tokenizer_name=tokenizer_name,
            )
            self.score_threshold = score_threshold
            self.max_length = max_length
            self.window_stride = window_stride
            self.classifier = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                truncation=True,
                max_length=max_length,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )
            self.response_string = f"Method:{PROMPT_HARMFULNESS_FAST}"
            self.run_in_executor = True
            logger.info("Initialized prompt harmfulness detector (ModernBERT)")
        except Exception as e:
            logger.error(f"Failed to initialize prompt harmfulness detector: {e}")
            raise

    def _extract_harmful_score(self, item: dict) -> float:
        """Extract the harmful-class probability from classifier output."""
        if item["label"] in (1, "1", "LABEL_1", "biased", "harmful"):
            return item["score"]
        return 1.0 - item["score"]

    @staticmethod
    def _extract_prompt_text(dome_input: DomePayload) -> str:
        """Extract only the prompt/text portion — ignore the response.

        This detector classifies the user's *prompt*, not the agent's
        response. Using ``query_string`` would include both sides when
        a structured ``DomePayload(prompt=..., response=...)`` is
        provided, which could cause false positives driven by the
        response text.
        """
        if dome_input.response is not None:
            logger.warning(
                "prompt-harmfulness detector received a DomePayload with "
                "a response field. This detector classifies prompts only; "
                "the response will be ignored."
            )
        if dome_input.prompt is not None:
            return dome_input.prompt
        return dome_input.text or ""

    def sync_detect(
        self,
        dome_input: DomePayload,
        agent_id: Optional[str] = None,
    ) -> DetectionResult:
        dome_input = DomePayload.coerce(dome_input)
        query_string = self._extract_prompt_text(dome_input)

        chunks = chunk_text(
            query_string, self.tokenizer, self.max_length, self.window_stride
        )
        num_windows = len(chunks)

        if num_windows == 1:
            pred = self.classifier(query_string)
            item = pred[0]
            harmful_score = self._extract_harmful_score(item)
            flagged = harmful_score >= self.score_threshold
            return flagged, {
                "type": type(self),
                "detector": PROMPT_HARMFULNESS_FAST,
                "score": harmful_score,
                "prediction": item,
                "threshold": self.score_threshold,
                "label": "harmful" if flagged else "safe",
                "confidence": max(harmful_score, 1.0 - harmful_score),
                "response_string": self.response_string if flagged else query_string,
                "num_windows": 1,
            }

        # Multi-window: batch all chunks, any-positive with max score
        all_preds = self.classifier(chunks, batch_size=self.max_batch_concurrency)
        max_score = 0.0
        best_pred: dict = {}
        for pred in all_preds:  # type: ignore[assignment]
            item = pred[0] if isinstance(pred, list) else pred  # type: ignore[assignment]
            score = self._extract_harmful_score(item)
            if score > max_score:
                max_score = score
                best_pred = item

        flagged = max_score >= self.score_threshold
        return flagged, {
            "type": type(self),
            "detector": PROMPT_HARMFULNESS_FAST,
            "score": max_score,
            "prediction": best_pred,
            "threshold": self.score_threshold,
            "label": "harmful" if flagged else "safe",
            "confidence": max(max_score, 1.0 - max_score),
            "response_string": self.response_string if flagged else query_string,
            "num_windows": num_windows,
        }

    async def detect(self, dome_input: DomePayload) -> DetectionResult:
        dome_input = DomePayload.coerce(dome_input)
        logger.info("Detecting prompt harmfulness using ModernBERT...")
        return self.sync_detect(dome_input)

    async def detect_batch(
        self, inputs: List[Union[str, DomePayload]]
    ) -> BatchDetectionResult:
        dome_inputs = [DomePayload.coerce(x) for x in inputs]

        # Phase 1: chunk each input, build flat list + per-input ranges
        flat_chunks: List[str] = []
        ranges = []
        for di in dome_inputs:
            query_string = self._extract_prompt_text(di)
            chunks = chunk_text(
                query_string, self.tokenizer, self.max_length, self.window_stride
            )
            start = len(flat_chunks)
            flat_chunks.extend(chunks)
            ranges.append((start, len(flat_chunks)))

        # Phase 2: pipeline call on all chunks (batched)
        all_preds = self.classifier(
            flat_chunks, batch_size=self.max_batch_concurrency
        )

        # Phase 3: re-aggregate per input using max score
        results: BatchDetectionResult = []
        for dome_input, (start, end) in zip(dome_inputs, ranges):
            query_string = self._extract_prompt_text(dome_input)
            chunk_preds = all_preds[start:end]
            num_windows = end - start
            max_score = 0.0
            best_pred: dict = {}
            for pred in chunk_preds:
                item = pred[0] if isinstance(pred, list) else pred  # type: ignore[assignment]
                score = self._extract_harmful_score(item)
                if score > max_score:
                    max_score = score
                    best_pred = item
            flagged = max_score >= self.score_threshold
            results.append((flagged, {
                "type": type(self),
                "detector": PROMPT_HARMFULNESS_FAST,
                "score": max_score,
                "prediction": best_pred,
                "threshold": self.score_threshold,
                "label": "harmful" if flagged else "safe",
                "confidence": max(max_score, 1.0 - max_score),
                "response_string": self.response_string if flagged else query_string,
                "num_windows": num_windows,
            }))
        return results


# ----------------------------------------------------------------------
# Safeguard mode (API-only)
# ----------------------------------------------------------------------


@register_method(DetectionCategory.Moderation, PROMPT_HARMFULNESS_SAFEGUARD)
class PromptHarmfulnessSafeguard(DetectionMethod):
    """
    Accurate prompt harmfulness detection backed by an OpenAI-compatible
    chat completions endpoint. Defaults to GPT-OSS-Safeguard-20B on Groq.

    Does not load ModernBERT — this mode is API-only.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_key_name: str = DEFAULT_SAFEGUARD_API_KEY_NAME,
        base_url: str = DEFAULT_SAFEGUARD_BASE_URL,
        model: str = DEFAULT_SAFEGUARD_MODEL,
        temperature: float = DEFAULT_SAFEGUARD_TEMPERATURE,
        max_tokens: int = DEFAULT_SAFEGUARD_MAX_TOKENS,
        reasoning_effort: Optional[str] = "low",
        timeout_seconds: float = 10.0,
        max_input_chars: Optional[int] = DEFAULT_SAFEGUARD_MAX_INPUT_CHARS,
        **kwargs,
    ):
        self.api_key_name = api_key_name
        self.api_key = api_key or os.environ.get(api_key_name, "")
        self.base_url = base_url.rstrip("/")
        self.chat_completions_url = f"{self.base_url}/chat/completions"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = timeout_seconds
        self.max_input_chars = max_input_chars
        self.response_string = f"Method:{PROMPT_HARMFULNESS_SAFEGUARD}"
        self.run_in_executor = False
        if not self.api_key:
            logger.warning(
                f"{api_key_name} not set — {PROMPT_HARMFULNESS_SAFEGUARD} will fail at runtime"
            )
        logger.info(
            "Initialized prompt harmfulness detector "
            f"(Safeguard mode, model={model}, base_url={self.base_url})"
        )

    def _truncate_if_needed(self, text: str) -> str:
        if self.max_input_chars is not None and len(text) > self.max_input_chars:
            logger.warning(
                "prompt-harmfulness-safeguard input truncated from %d to %d chars",
                len(text),
                self.max_input_chars,
            )
            return text[: self.max_input_chars]
        return text

    def _build_payload(self, query_string: str) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": HARMFULNESS_SAFEGUARD_SYSTEM_PROMPT},
                {"role": "user", "content": query_string},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        return payload

    async def detect(self, dome_input: DomePayload) -> DetectionResult:
        dome_input = DomePayload.coerce(dome_input)
        if dome_input.response is not None:
            logger.warning(
                "prompt-harmfulness-safeguard received a DomePayload with "
                "a response field. This detector classifies prompts only; "
                "the response will be ignored."
            )
        prompt_text = dome_input.prompt if dome_input.prompt is not None else (dome_input.text or "")
        query_string = self._truncate_if_needed(prompt_text)
        logger.info("Detecting prompt harmfulness using Safeguard...")
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                resp = await client.post(
                    self.chat_completions_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=self._build_payload(query_string),
                )
                resp.raise_for_status()
                content = (
                    resp.json()["choices"][0]["message"]
                    .get("content", "")
                    .strip()
                    .lower()
                )
                flagged = "unsafe" in content
                return flagged, {
                    "type": type(self),
                    "detector": PROMPT_HARMFULNESS_SAFEGUARD,
                    "score": 1.0 if flagged else 0.0,
                    "label": "harmful" if flagged else "safe",
                    "safeguard_verdict": content,
                    "response_string": (
                        self.response_string if flagged else query_string
                    ),
                }
        except Exception as e:
            logger.error(f"Safeguard API call failed: {e}")
            return False, {
                "type": type(self),
                "detector": PROMPT_HARMFULNESS_SAFEGUARD,
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
