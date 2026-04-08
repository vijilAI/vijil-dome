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
EEOC Stereotype Detection — Three modes:

  fast:     ModernBERT binary classifier (<5ms, F1=0.923)
  safeguard: GPT-OSS-Safeguard-20B via Groq API (~200ms, ~100% accuracy)
  hybrid:   ModernBERT first, Safeguard on low-confidence (~5ms avg)

Distilled from GPT-OSS-Safeguard-20B with a custom EEOC discrimination
policy. Detects stereotypes and harmful assumptions about EEOC protected
classes: Race/Color, Sex/Gender, Religion, National Origin, Age (40+),
Disability.

Why "Stereotype" and not "Bias":
    These detectors flag stereotyping, harmful generalizations, and
    differential treatment expressed within a *single* prompt or response.
    They do NOT detect counterfactual bias — i.e., whether the same prompt
    produces different outputs when only the protected class is varied.
    Counterfactual bias requires comparing two prompt-response pairs and
    is out of scope for this detector class.

Input format:
    The underlying ModernBERT model was trained on `prompt [SEP] response`
    pairs. The current Dome `DetectionMethod.detect()` interface accepts a
    single string, so this detector treats the input as the "prompt" half
    of the pair (with an empty response). This means:

    - On input guards: works as trained (the agent input is the prompt).
    - On output guards: out-of-distribution. The agent response is fed in
      where a prompt is expected.

    An F1-delta study comparing input-only, output-only, and paired
    evaluation is pending. Until that lands, prefer input guards. See
    PR #163 review thread for context.
"""

import asyncio
import logging
import os
from typing import List, Optional

import httpx
import torch
from transformers import pipeline

from vijil_dome.detectors import (
    STEREOTYPE_EEOC_FAST,
    STEREOTYPE_EEOC_SAFEGUARD,
    STEREOTYPE_EEOC_HYBRID,
    register_method,
    DetectionCategory,
    DetectionResult,
    BatchDetectionResult,
)
from vijil_dome.detectors.utils.hf_model import HFBaseModel

logger = logging.getLogger("vijil.dome")

SAFEGUARD_SYSTEM_PROMPT = (
    "Evaluate if this AI response discriminates against EEOC protected classes. "
    "Protected classes: Race/Color, Sex/Gender/Sexual Orientation, Religion, "
    "National Origin, Age (40+), Disability. "
    "A response is unsafe if it stereotypes, shows differential treatment, "
    "or reinforces harmful assumptions about any protected class. "
    "Reply with exactly one word: safe or unsafe"
)

# Safeguard call defaults — single-word verdict, deterministic, low cost.
DEFAULT_SAFEGUARD_TEMPERATURE = 0.0
DEFAULT_SAFEGUARD_MAX_TOKENS = 8


class StereotypeEEOCBase(HFBaseModel):
    """Base class for EEOC stereotype detection using ModernBERT."""

    def __init__(
        self,
        model_name: str = "vijil/stereotype-eeoc-detector",
        tokenizer_name: str = "vijil/stereotype-eeoc-detector",
        score_threshold: float = 0.5,
        max_length: int = 512,
    ):
        try:
            super().__init__(
                model_name=model_name,
                tokenizer_name=tokenizer_name,
            )
            self.score_threshold = score_threshold
            self.max_length = max_length
            self.classifier = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                truncation=True,
                max_length=max_length,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )
            self.run_in_executor = True
            logger.info("Initialized EEOC stereotype detector (ModernBERT)")
        except Exception as e:
            logger.error(f"Failed to initialize EEOC stereotype detector: {e}")
            raise

    def _extract_stereotype_score(self, item: dict) -> float:
        """Extract the positive-class probability from classifier output."""
        if item["label"] in (1, "1", "LABEL_1", "biased", "stereotyped"):
            return item["score"]
        return 1.0 - item["score"]

    def _classify(self, prompt: str, response: str) -> tuple[float, dict]:
        """Run ModernBERT classification on a prompt+response pair."""
        text = f"{prompt} [SEP] {response}"
        pred = self.classifier(text)
        item = pred[0]
        return self._extract_stereotype_score(item), item


@register_method(DetectionCategory.Moderation, STEREOTYPE_EEOC_FAST)
class StereotypeEEOCFast(StereotypeEEOCBase):
    """
    Fast EEOC stereotype detection using distilled ModernBERT.
    <5ms latency, F1=0.923, zero API cost. Self-hosted.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.response_string = f"Method:{STEREOTYPE_EEOC_FAST}"

    def sync_detect(
        self, query_string: str, agent_id: Optional[str] = None
    ) -> DetectionResult:
        # Treat the entire query_string as the "prompt" half of the
        # trained pair. See module docstring for caveats.
        stereotype_score, prediction = self._classify(query_string, "")
        flagged = stereotype_score >= self.score_threshold
        logger.debug(
            "stereotype-eeoc-fast prediction=%s score=%.3f", prediction, stereotype_score
        )

        return flagged, {
            "type": type(self),
            "detector": STEREOTYPE_EEOC_FAST,
            "score": stereotype_score,
            "prediction": prediction,
            "threshold": self.score_threshold,
            "label": "stereotyped" if flagged else "neutral",
            "confidence": max(stereotype_score, 1.0 - stereotype_score),
            "response_string": self.response_string if flagged else query_string,
        }

    async def detect(self, query_string: str) -> DetectionResult:
        logger.info(f"Detecting EEOC stereotype using {self.__class__.__name__}...")
        return self.sync_detect(query_string)

    async def detect_batch(self, inputs: List[str]) -> BatchDetectionResult:
        results = []
        texts = [f"{inp} [SEP] " for inp in inputs]
        all_preds = self.classifier(texts)
        for query_string, pred in zip(inputs, all_preds):
            item = pred[0] if isinstance(pred, list) else pred
            stereotype_score = self._extract_stereotype_score(item)
            flagged = stereotype_score >= self.score_threshold
            results.append((flagged, {
                "type": type(self),
                "detector": STEREOTYPE_EEOC_FAST,
                "score": stereotype_score,
                "prediction": item,
                "threshold": self.score_threshold,
                "label": "stereotyped" if flagged else "neutral",
                "confidence": max(stereotype_score, 1.0 - stereotype_score),
                "response_string": self.response_string if flagged else query_string,
            }))
        return results


@register_method(DetectionCategory.Moderation, STEREOTYPE_EEOC_SAFEGUARD)
class StereotypeEEOCSafeguard:
    """
    Accurate EEOC stereotype detection using GPT-OSS-Safeguard-20B via Groq.
    ~200ms latency, ~100% accuracy, requires GROQ_API_KEY.

    Does not load ModernBERT — this mode is API-only.
    """

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        groq_model: str = "openai/gpt-oss-safeguard-20b",
        temperature: float = DEFAULT_SAFEGUARD_TEMPERATURE,
        max_tokens: int = DEFAULT_SAFEGUARD_MAX_TOKENS,
        timeout_seconds: float = 10.0,
        **kwargs,
    ):
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")
        self.groq_model = groq_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.response_string = f"Method:{STEREOTYPE_EEOC_SAFEGUARD}"
        self.run_in_executor = False  # async HTTP, no executor needed
        if not self.groq_api_key:
            logger.warning(
                f"GROQ_API_KEY not set — {STEREOTYPE_EEOC_SAFEGUARD} will fail at runtime"
            )
        logger.info("Initialized EEOC stereotype detector (Safeguard mode)")

    async def detect(self, query_string: str) -> DetectionResult:
        logger.info("Detecting EEOC stereotype using Safeguard...")
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.groq_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.groq_model,
                        "messages": [
                            {"role": "system", "content": SAFEGUARD_SYSTEM_PROMPT},
                            {"role": "user", "content": query_string},
                        ],
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                        "reasoning_effort": "low",
                    },
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"].get("content", "").strip().lower()
                flagged = "unsafe" in content

                return flagged, {
                    "type": type(self),
                    "detector": STEREOTYPE_EEOC_SAFEGUARD,
                    "score": 1.0 if flagged else 0.0,
                    "label": "stereotyped" if flagged else "neutral",
                    "safeguard_verdict": content,
                    "response_string": self.response_string if flagged else query_string,
                }
        except Exception as e:
            logger.error(f"Safeguard API call failed: {e}")
            return False, {
                "type": type(self),
                "detector": STEREOTYPE_EEOC_SAFEGUARD,
                "score": 0.0,
                "label": "error",
                "error": str(e),
                "response_string": query_string,
            }


@register_method(DetectionCategory.Moderation, STEREOTYPE_EEOC_HYBRID)
class StereotypeEEOCHybrid(StereotypeEEOCBase):
    """
    Hybrid EEOC stereotype detection: ModernBERT first, Safeguard on
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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.confidence_threshold = confidence_threshold
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")
        self.groq_model = groq_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.response_string = f"Method:{STEREOTYPE_EEOC_HYBRID}"
        self.run_in_executor = False  # async for Safeguard fallback

        if not self.groq_api_key:
            logger.warning(
                f"GROQ_API_KEY not set — {STEREOTYPE_EEOC_HYBRID} will fall back to fast-only"
            )
        logger.info(
            "Initialized EEOC stereotype detector "
            f"(hybrid mode, confidence_threshold={confidence_threshold})"
        )

    async def detect(self, query_string: str) -> DetectionResult:
        logger.info("Detecting EEOC stereotype using hybrid mode...")

        # Stage 1: Fast ModernBERT classification.
        loop = asyncio.get_running_loop()
        stereotype_score, prediction = await loop.run_in_executor(
            None, self._classify, query_string, ""
        )
        confidence = max(stereotype_score, 1.0 - stereotype_score)
        logger.debug(
            "stereotype-eeoc-hybrid fast-stage prediction=%s score=%.3f",
            prediction,
            stereotype_score,
        )

        # If confident enough, return immediately.
        if confidence >= self.confidence_threshold:
            flagged = stereotype_score >= self.score_threshold
            return flagged, {
                "type": type(self),
                "detector": STEREOTYPE_EEOC_HYBRID,
                "stage": "fast",
                "score": stereotype_score,
                "prediction": prediction,
                "threshold": self.score_threshold,
                "confidence": confidence,
                "label": "stereotyped" if flagged else "neutral",
                "response_string": self.response_string if flagged else query_string,
            }

        # Stage 2: Low confidence — escalate to Safeguard.
        if not self.groq_api_key:
            # No API key — fall back to fast result.
            flagged = stereotype_score >= self.score_threshold
            return flagged, {
                "type": type(self),
                "detector": STEREOTYPE_EEOC_HYBRID,
                "stage": "fast-fallback",
                "score": stereotype_score,
                "prediction": prediction,
                "threshold": self.score_threshold,
                "confidence": confidence,
                "label": "stereotyped" if flagged else "neutral",
                "response_string": self.response_string if flagged else query_string,
            }

        logger.info(
            f"Low confidence ({confidence:.2f} < {self.confidence_threshold}), "
            "escalating to Safeguard..."
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.groq_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.groq_model,
                        "messages": [
                            {"role": "system", "content": SAFEGUARD_SYSTEM_PROMPT},
                            {"role": "user", "content": query_string},
                        ],
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                        "reasoning_effort": "low",
                    },
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"].get("content", "").strip().lower()
                flagged = "unsafe" in content

                return flagged, {
                    "type": type(self),
                    "detector": STEREOTYPE_EEOC_HYBRID,
                    "stage": "safeguard",
                    "score": 1.0 if flagged else 0.0,
                    "fast_score": stereotype_score,
                    "fast_confidence": confidence,
                    "label": "stereotyped" if flagged else "neutral",
                    "safeguard_verdict": content,
                    "response_string": self.response_string if flagged else query_string,
                }
        except Exception as e:
            logger.error(f"Safeguard escalation failed: {e}, using fast result")
            flagged = stereotype_score >= self.score_threshold
            return flagged, {
                "type": type(self),
                "detector": STEREOTYPE_EEOC_HYBRID,
                "stage": "fast-fallback",
                "score": stereotype_score,
                "prediction": prediction,
                "threshold": self.score_threshold,
                "confidence": confidence,
                "label": "stereotyped" if flagged else "neutral",
                "error": str(e),
                "response_string": self.response_string if flagged else query_string,
            }
