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

  fast:      ModernBERT binary classifier (<5ms, F1=0.923)
  safeguard: GPT-OSS-Safeguard-20B via Groq API (~200ms, ~100% accuracy)
  hybrid:    ModernBERT first, Safeguard on low-confidence (~5ms avg)

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

Input format and chunking:
    The underlying ModernBERT model was trained on `prompt [SEP] response`
    pairs at ``max_length`` 512 tokens. When a caller passes a structured
    ``DomePayload`` with both ``prompt`` and ``response`` fields set, the
    detector reconstructs the training format exactly. When only ``text``
    is set, the detector treats it as the prompt half with an empty
    response.

    Inputs longer than ``max_length`` are split into multiple chunks
    anchored on a **[SEP]-centered** window:

      - The center chunk holds the last ``usable/2`` tokens of the prompt
        and the first ``usable/2`` tokens of the response, joined by
        ``[SEP]``. Anchoring on the prompt/response boundary preserves
        the training-format signal. If one side is shorter than its
        half-budget, the unused tokens are donated to the other side.
      - Any prompt tokens that fall **before** the center's prompt tail
        are sliced into additional prompt-only chunks of the form
        ``"<chunk> [SEP] "`` (empty response half).
      - Any response tokens that fall **after** the center's response
        head are sliced into additional response-only chunks of the
        form ``" [SEP] <chunk>"`` (empty prompt half).

    All chunks from a payload are classified in one batched pipeline
    call. The payload's final score is the **max** across chunks — any
    chunk flagging the input drives the overall flag — and the winning
    chunk's raw prediction is surfaced on the result payload.
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import httpx

try:
    import torch
    from transformers import pipeline
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from vijil_dome.detectors import (
    STEREOTYPE_EEOC_FAST,
    STEREOTYPE_EEOC_SAFEGUARD,
    STEREOTYPE_EEOC_HYBRID,
    STEREOTYPE_EEOC_REMOTE,
    register_method,
    DetectionCategory,
    DetectionMethod,
    DetectionResult,
    BatchDetectionResult,
)
from vijil_dome.detectors.utils.hf_model import HFBaseModel
from vijil_dome.types import DomePayload

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
# gpt-oss-safeguard-20b is a reasoning model and reserves a chunk of the
# token budget for internal reasoning before emitting any assistant content.
# A small cap (e.g. 8) lets the reasoning alone blow the budget, hit
# finish_reason=length, and return an empty `content` — which silently
# classifies everything as safe. Keep this generous.
DEFAULT_SAFEGUARD_MAX_TOKENS = 2000

# GPT-OSS-Safeguard-20B has a ~130K token context window. We don't load
# the tokenizer in Safeguard mode (API-only), so we cap in characters
# instead. At a conservative ~3 chars/token for English + code/symbols,
# 400K chars comfortably fits under the token cap while leaving headroom
# for the system prompt and the 8-token verdict response. Callers can
# pass ``max_input_chars=None`` to disable the cap.
DEFAULT_SAFEGUARD_MAX_INPUT_CHARS = 400_000

# Token budget reservations for the " [SEP] " string we insert between
# prompt and response. ModernBERT's tokenizer turns this into ~4 tokens;
# we reserve 6 to be safe.
_SEP_STRING_TOKEN_BUDGET = 6
# Model special-token overhead (CLS + SEP emitted by the tokenizer).
_SPECIAL_TOKEN_OVERHEAD = 2

# The safeguard endpoint is OpenAI-compatible. Groq is the default, but
# callers can point at any OpenAI-compatible `/chat/completions` endpoint
# (local vLLM, Together, Fireworks, OpenAI itself, ...) by overriding
# `base_url`, `model`, and `api_key_name` on the detector.
DEFAULT_SAFEGUARD_API_KEY_NAME = "GROQ_API_KEY"
DEFAULT_SAFEGUARD_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_SAFEGUARD_MODEL = "openai/gpt-oss-safeguard-20b"

# Default model name on Vijil's inference endpoint (matches HuggingFace).
DEFAULT_VIJIL_INFERENCE_STEREOTYPE_MODEL = "vijil/stereotype-eeoc-detector"


class StereotypeEEOCBase(HFBaseModel):
    """Base class for EEOC stereotype detection using ModernBERT."""

    def __init__(
        self,
        model_name: str = "vijil/stereotype-eeoc-detector",
        # Load the tokenizer from the upstream ModernBERT repo rather than
        # from the fine-tuned weights. The v2 model's tokenizer_config.json
        # references a wrapper class ("TokenizersBackend") that transformers
        # cannot resolve at load time. The vocabulary is identical to
        # answerdotai/ModernBERT-base (we fine-tuned on top), so loading the
        # upstream tokenizer produces the same token ids without the config
        # compatibility issue. This mirrors how pi_hf_mbert handles the
        # same situation.
        tokenizer_name: str = "answerdotai/ModernBERT-base",
        score_threshold: float = 0.5,
        max_length: int = 512,
    ):
        if not _HAS_TORCH:
            raise ImportError(
                f"{STEREOTYPE_EEOC_FAST} requires 'torch' and 'transformers'. "
                "Install with: pip install vijil-dome[local]"
            )
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

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _extract_stereotype_score(self, item: dict) -> float:
        """Extract the positive-class probability from classifier output."""
        if item["label"] in (1, "1", "LABEL_1", "biased", "stereotyped"):
            return item["score"]
        return 1.0 - item["score"]

    @staticmethod
    def _split_payload(dome_input: DomePayload) -> Tuple[str, str]:
        """Pull ``(prompt_text, response_text)`` from a DomePayload.

        Structured prompt/response fields win over the legacy text field.
        When only ``text`` is set, the full text becomes the prompt side
        with an empty response — matching how we handled single-string
        inputs before DomePayload existed.
        """
        if dome_input.prompt is not None or dome_input.response is not None:
            return dome_input.prompt or "", dome_input.response or ""
        return dome_input.text or "", ""

    def _decode(self, ids: List[int]) -> str:
        """Decode a list of token ids, returning an empty string when empty."""
        if not ids:
            return ""
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def _build_chunks(
        self, prompt_text: str, response_text: str
    ) -> List[str]:
        """Construct one or more classifier-ready chunks.

        The primary chunk is the ``[SEP]``-centered window (last
        ``prompt_keep`` tokens of the prompt, first ``response_keep``
        tokens of the response). Any prompt tokens that fall before the
        center's prompt tail are emitted as additional prompt-only
        chunks shaped ``"<chunk> [SEP] "``. Any response tokens that
        fall after the center's response head are emitted as
        response-only chunks shaped ``" [SEP] <chunk>"``. Chunks are
        returned in natural reading order::

            [prompt_chunk_0, ..., prompt_chunk_k, center, response_chunk_0, ...]

        When the combined prompt+response already fits inside
        ``max_length``, the original strings are joined without any
        tokenizer round-tripping (preserves exact formatting) and a
        single-element list is returned.
        """
        # Fast path — the full string fits, no chunking needed.
        if not self._exceeds_budget(prompt_text, response_text):
            return [f"{prompt_text} [SEP] {response_text}"]

        usable = self.max_length - _SPECIAL_TOKEN_OVERHEAD - _SEP_STRING_TOKEN_BUDGET
        if usable <= 0:
            raise ValueError(
                f"max_length ({self.max_length}) is too small for a single chunk "
                f"(need > {_SPECIAL_TOKEN_OVERHEAD + _SEP_STRING_TOKEN_BUDGET})"
            )

        prompt_ids = self.tokenizer.encode(
            prompt_text, add_special_tokens=False, verbose=False
        )
        response_ids = self.tokenizer.encode(
            response_text, add_special_tokens=False, verbose=False
        )

        # Split budget 50/50 for the center chunk, then donate unused
        # tokens from the shorter side to the longer side.
        half = usable // 2
        prompt_keep = min(len(prompt_ids), half)
        response_keep = min(len(response_ids), half)
        leftover = usable - prompt_keep - response_keep
        if leftover > 0 and len(prompt_ids) > prompt_keep:
            donation = min(leftover, len(prompt_ids) - prompt_keep)
            prompt_keep += donation
            leftover -= donation
        if leftover > 0 and len(response_ids) > response_keep:
            donation = min(leftover, len(response_ids) - response_keep)
            response_keep += donation

        prompt_tail_ids = prompt_ids[-prompt_keep:] if prompt_keep > 0 else []
        response_head_ids = response_ids[:response_keep]

        center_chunk = (
            f"{self._decode(prompt_tail_ids)} [SEP] {self._decode(response_head_ids)}"
        )

        # Prompt tokens before the center tail → prompt-only flank chunks.
        prompt_head_len = len(prompt_ids) - prompt_keep
        prompt_head_ids = prompt_ids[:prompt_head_len] if prompt_head_len > 0 else []
        prompt_chunks: List[str] = []
        for start in range(0, len(prompt_head_ids), usable):
            chunk_ids = prompt_head_ids[start : start + usable]
            prompt_chunks.append(f"{self._decode(chunk_ids)} [SEP] ")

        # Response tokens after the center head → response-only flank chunks.
        response_tail_ids = response_ids[response_keep:]
        response_chunks: List[str] = []
        for start in range(0, len(response_tail_ids), usable):
            chunk_ids = response_tail_ids[start : start + usable]
            response_chunks.append(f" [SEP] {self._decode(chunk_ids)}")

        logger.debug(
            "stereotype-eeoc chunking: prompt %d tokens → %d prompt chunks + center tail %d; "
            "response %d tokens → center head %d + %d response chunks",
            len(prompt_ids),
            len(prompt_chunks),
            prompt_keep,
            len(response_ids),
            response_keep,
            len(response_chunks),
        )
        return prompt_chunks + [center_chunk] + response_chunks

    def _exceeds_budget(self, prompt_text: str, response_text: str) -> bool:
        """Cheap pre-check: do prompt+response exceed the usable budget?"""
        usable = self.max_length - _SPECIAL_TOKEN_OVERHEAD - _SEP_STRING_TOKEN_BUDGET
        if usable <= 0:
            return True
        total = len(
            self.tokenizer.encode(
                prompt_text, add_special_tokens=False, verbose=False
            )
        ) + len(
            self.tokenizer.encode(
                response_text, add_special_tokens=False, verbose=False
            )
        )
        return total > usable

    def _build_chunks_for_payload(self, dome_input: DomePayload) -> List[str]:
        """End-to-end: DomePayload → list of classifier-ready chunks."""
        prompt_text, response_text = self._split_payload(dome_input)
        return self._build_chunks(prompt_text, response_text)

    def _aggregate(self, preds) -> Tuple[float, dict]:
        """Reduce per-chunk predictions to a single (score, item) pair.

        Policy: the chunk with the highest stereotype score wins. Any
        chunk flagging the input drives the overall flag, and the
        raw winning prediction is surfaced so callers can inspect what
        triggered the flag.
        """
        best_score = -1.0
        best_item: dict = {}
        for pred in preds:
            item = pred[0] if isinstance(pred, list) else pred
            score = self._extract_stereotype_score(item)
            if score > best_score:
                best_score = score
                best_item = item
        return best_score, best_item

    def _classify(self, dome_input: DomePayload) -> Tuple[float, dict]:
        """Run ModernBERT classification on a single DomePayload.

        When the payload exceeds ``max_length`` the chunker emits
        multiple chunks; all are scored in one pipeline call and
        ``_aggregate`` reduces them to a single (score, prediction)
        using the max-score policy.
        """
        chunks = self._build_chunks_for_payload(dome_input)
        preds = self.classifier(chunks, batch_size=self.max_batch_concurrency)
        return self._aggregate(preds)

    # ------------------------------------------------------------------
    # Batched fast-stage (used by Fast and Hybrid)
    # ------------------------------------------------------------------

    def _classify_batch(
        self, dome_inputs: List[DomePayload]
    ) -> List[Tuple[float, dict]]:
        """Run ModernBERT classification on many DomePayloads in one call.

        Each input may produce one or more chunks (see ``_build_chunks``).
        All chunks from all payloads are flattened into a single list and
        sent to ``self.classifier`` in one call so the HF pipeline can
        batch the forward passes — substantially faster than a per-item
        loop. Results are then sliced back per payload using recorded
        offsets and reduced via ``_aggregate``.
        """
        per_payload_chunks = [
            self._build_chunks_for_payload(d) for d in dome_inputs
        ]
        flat_chunks: List[str] = []
        offsets: List[Tuple[int, int]] = []
        for chunks in per_payload_chunks:
            start = len(flat_chunks)
            flat_chunks.extend(chunks)
            offsets.append((start, len(flat_chunks)))

        all_preds = self.classifier(
            flat_chunks, batch_size=self.max_batch_concurrency
        )
        return [self._aggregate(all_preds[start:end]) for start, end in offsets]


# ----------------------------------------------------------------------
# Fast mode
# ----------------------------------------------------------------------


@register_method(DetectionCategory.Moderation, STEREOTYPE_EEOC_FAST)
class StereotypeEEOCFast(StereotypeEEOCBase):
    """
    Fast EEOC stereotype detection using distilled ModernBERT.
    <5ms latency, F1=0.923, zero API cost. Self-hosted.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.response_string = f"Method:{STEREOTYPE_EEOC_FAST}"

    def _build_payload(
        self,
        dome_input: DomePayload,
        stereotype_score: float,
        prediction: dict,
    ) -> Dict:
        flagged = stereotype_score >= self.score_threshold
        return {
            "type": type(self),
            "detector": STEREOTYPE_EEOC_FAST,
            "score": stereotype_score,
            "prediction": prediction,
            "threshold": self.score_threshold,
            "label": "stereotyped" if flagged else "neutral",
            "confidence": max(stereotype_score, 1.0 - stereotype_score),
            "response_string": self.response_string if flagged else dome_input.query_string,
        }

    def sync_detect(
        self,
        dome_input: DomePayload,
        agent_id: Optional[str] = None,
    ) -> DetectionResult:
        dome_input = DomePayload.coerce(dome_input)
        stereotype_score, prediction = self._classify(dome_input)
        flagged = stereotype_score >= self.score_threshold
        logger.debug(
            "stereotype-eeoc-fast prediction=%s score=%.3f",
            prediction,
            stereotype_score,
        )
        return flagged, self._build_payload(dome_input, stereotype_score, prediction)

    async def detect(self, dome_input: DomePayload) -> DetectionResult:
        logger.info(f"Detecting EEOC stereotype using {self.__class__.__name__}...")
        return self.sync_detect(DomePayload.coerce(dome_input))

    async def detect_batch(
        self, inputs: List[Union[str, DomePayload]]
    ) -> BatchDetectionResult:
        dome_inputs = [DomePayload.coerce(x) for x in inputs]
        scored = self._classify_batch(dome_inputs)
        results: BatchDetectionResult = []
        for dome_input, (score, prediction) in zip(dome_inputs, scored):
            flagged = score >= self.score_threshold
            results.append((flagged, self._build_payload(dome_input, score, prediction)))
        return results


# ----------------------------------------------------------------------
# Safeguard mode (API-only)
# ----------------------------------------------------------------------


@register_method(DetectionCategory.Moderation, STEREOTYPE_EEOC_SAFEGUARD)
class StereotypeEEOCSafeguard(DetectionMethod):
    """
    Accurate EEOC stereotype detection backed by an OpenAI-compatible chat
    completions endpoint. Defaults to GPT-OSS-Safeguard-20B on Groq
    (~200ms, ~100% accuracy), but the endpoint, model, and credential
    name are all overridable so callers can point at any OpenAI-style
    deployment (local vLLM, Together, Fireworks, OpenAI proper, ...).

    Does not load ModernBERT — this mode is API-only.

    Oversize inputs are **truncated** (not chunked) to ``max_input_chars``
    before being sent to the endpoint, matching the convention used by
    other API-backed detectors in this library (see ``LlmBaseDetector``).
    The default is intentionally generous — GPT-OSS-Safeguard-20B has a
    ~130K token context window, so truncation should be a rare
    safety-net rather than something callers hit routinely.
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
        self.response_string = f"Method:{STEREOTYPE_EEOC_SAFEGUARD}"
        self.run_in_executor = False  # async HTTP, no executor needed
        if not self.api_key:
            logger.warning(
                f"{api_key_name} not set — {STEREOTYPE_EEOC_SAFEGUARD} will fail at runtime"
            )
        logger.info(
            "Initialized EEOC stereotype detector "
            f"(Safeguard mode, model={model}, base_url={self.base_url})"
        )

    def _truncate_if_needed(self, text: str) -> str:
        """Cap ``text`` at ``self.max_input_chars`` to stay under the
        Safeguard model's context window. No-op when the cap is ``None``."""
        if self.max_input_chars is not None and len(text) > self.max_input_chars:
            logger.warning(
                "stereotype-eeoc-safeguard input truncated from %d to %d chars",
                len(text),
                self.max_input_chars,
            )
            return text[: self.max_input_chars]
        return text

    def _build_payload(self, query_string: str) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SAFEGUARD_SYSTEM_PROMPT},
                {"role": "user", "content": query_string},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        return payload

    async def _call_safeguard(
        self, client: httpx.AsyncClient, query_string: str
    ) -> str:
        resp = await client.post(
            self.chat_completions_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=self._build_payload(query_string),
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"].get("content", "").strip().lower()

    async def detect(self, dome_input: DomePayload) -> DetectionResult:
        dome_input = DomePayload.coerce(dome_input)
        query_string = self._truncate_if_needed(dome_input.query_string)
        logger.info("Detecting EEOC stereotype using Safeguard...")
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                content = await self._call_safeguard(client, query_string)
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

    async def detect_batch(
        self, inputs: List[Union[str, DomePayload]]
    ) -> BatchDetectionResult:
        """Run Safeguard on a batch, capped by ``self.max_batch_concurrency``.

        Mirrors ``LlmBaseDetector.detect_batch`` — each input becomes
        one independent API call, and the inherited
        ``_gather_with_concurrency`` helper bounds in-flight requests
        at the detector's ``max_batch_concurrency``.
        """
        return await self._gather_with_concurrency(
            [self.detect(DomePayload.coerce(item)) for item in inputs]
        )


# ----------------------------------------------------------------------
# Remote mode (Vijil inference endpoint, no local model)
# ----------------------------------------------------------------------


@register_method(DetectionCategory.Moderation, STEREOTYPE_EEOC_REMOTE)
class StereotypeEEOCRemote(DetectionMethod):
    """EEOC stereotype detection via Vijil's remote inference service.

    Calls Vijil's hosted ModernBERT endpoint — no local model, no torch.
    Suitable for lightweight deployments where the inference server is
    co-located or latency is acceptable.

    This detector only works with Vijil's inference endpoint — do not
    point it at arbitrary OpenAI-compatible services.
    """

    def __init__(
        self,
        vijil_inference_url: str,
        vijil_inference_model: Optional[str] = None,
        vijil_inference_api_key: Optional[str] = None,
        score_threshold: float = 0.5,
        timeout_seconds: float = 10.0,
    ):
        from vijil_dome.detectors.utils.vijil_inference import VijilInferenceClient

        self.score_threshold = score_threshold
        self.response_string = f"Method:{STEREOTYPE_EEOC_REMOTE}"
        self.run_in_executor = False
        self._vijil_client = VijilInferenceClient(
            base_url=vijil_inference_url,
            model=vijil_inference_model or DEFAULT_VIJIL_INFERENCE_STEREOTYPE_MODEL,
            api_key=vijil_inference_api_key,
            timeout_seconds=timeout_seconds,
        )
        logger.info(
            "Initialized EEOC stereotype detector (remote mode, "
            f"model={self._vijil_client.model}, url={self._vijil_client.base_url})"
        )

    async def detect(self, dome_input: DomePayload) -> DetectionResult:
        dome_input = DomePayload.coerce(dome_input)
        query_string = dome_input.query_string
        logger.info("Detecting EEOC stereotype using Vijil remote inference...")
        try:
            async with httpx.AsyncClient(
                timeout=self._vijil_client.timeout_seconds
            ) as client:
                score = await self._vijil_client.classify(client, query_string)
            flagged = score >= self.score_threshold
            return flagged, {
                "type": type(self),
                "detector": STEREOTYPE_EEOC_REMOTE,
                "score": score,
                "label": "stereotyped" if flagged else "neutral",
                "response_string": self.response_string if flagged else query_string,
            }
        except Exception as e:
            logger.error(f"Vijil remote inference failed: {e}")
            return False, {
                "type": type(self),
                "detector": STEREOTYPE_EEOC_REMOTE,
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


@register_method(DetectionCategory.Moderation, STEREOTYPE_EEOC_HYBRID)
class StereotypeEEOCHybrid(StereotypeEEOCBase):
    """
    Hybrid EEOC stereotype detection: fast stage first (local ModernBERT
    or Vijil remote inference), Safeguard on low-confidence.

    When ``vijil_inference_url`` is provided the fast stage calls Vijil's
    hosted inference endpoint instead of loading ModernBERT locally,
    eliminating the torch/transformers dependency for the fast stage.
    """

    def __init__(
        self,
        vijil_inference_url: Optional[str] = None,
        vijil_inference_model: Optional[str] = None,
        vijil_inference_api_key: Optional[str] = None,
        confidence_threshold: float = 0.85,
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
        self._use_vijil_inference = vijil_inference_url is not None

        if self._use_vijil_inference:
            from vijil_dome.detectors.utils.vijil_inference import VijilInferenceClient
            assert vijil_inference_url is not None
            self.score_threshold = kwargs.get("score_threshold", 0.5)
            self._vijil_client = VijilInferenceClient(
                base_url=vijil_inference_url,
                model=vijil_inference_model or DEFAULT_VIJIL_INFERENCE_STEREOTYPE_MODEL,
                api_key=vijil_inference_api_key,
            )
        else:
            super().__init__(**kwargs)

        self.confidence_threshold = confidence_threshold
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
        self.response_string = f"Method:{STEREOTYPE_EEOC_HYBRID}"
        self.run_in_executor = False  # async for Safeguard fallback

        if not self.api_key:
            logger.warning(
                f"{api_key_name} not set — {STEREOTYPE_EEOC_HYBRID} will fall back to fast-only"
            )
        fast_mode = (
            f"vijil-inference ({self._vijil_client.base_url})"
            if self._use_vijil_inference
            else "local"
        )
        logger.info(
            "Initialized EEOC stereotype detector "
            f"(hybrid mode, fast={fast_mode}, "
            f"confidence_threshold={confidence_threshold}, "
            f"safeguard_model={model}, safeguard_url={self.base_url})"
        )

    # Fast-stage payload builder — shared between detect and detect_batch.
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
            "detector": STEREOTYPE_EEOC_HYBRID,
            "stage": stage,
            "score": score,
            "prediction": prediction,
            "threshold": self.score_threshold,
            "confidence": max(score, 1.0 - score),
            "label": "stereotyped" if flagged else "neutral",
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
            "detector": STEREOTYPE_EEOC_HYBRID,
            "stage": "safeguard",
            "score": 1.0 if flagged else 0.0,
            "fast_score": fast_score,
            "fast_confidence": max(fast_score, 1.0 - fast_score),
            "label": "stereotyped" if flagged else "neutral",
            "safeguard_verdict": content,
            "response_string": (
                self.response_string if flagged else dome_input.query_string
            ),
        }

    def _truncate_if_needed(self, text: str) -> str:
        """Cap ``text`` at ``self.max_input_chars`` before sending to
        Safeguard. Mirrors ``StereotypeEEOCSafeguard._truncate_if_needed``
        so the escalation path honors the same ~130K token cap."""
        if self.max_input_chars is not None and len(text) > self.max_input_chars:
            logger.warning(
                "stereotype-eeoc-hybrid escalation input truncated from %d to %d chars",
                len(text),
                self.max_input_chars,
            )
            return text[: self.max_input_chars]
        return text

    async def _escalate(
        self,
        client: httpx.AsyncClient,
        dome_input: DomePayload,
        fast_score: float,
        prediction: dict,
    ) -> DetectionResult:
        """Run Safeguard on one low-confidence item, with graceful fallback."""
        query_string = self._truncate_if_needed(dome_input.query_string)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SAFEGUARD_SYSTEM_PROMPT},
                {"role": "user", "content": query_string},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        try:
            resp = await client.post(
                self.chat_completions_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            content = (
                resp.json()["choices"][0]["message"].get("content", "").strip().lower()
            )
            payload = self._safeguard_payload(dome_input, content, fast_score)
            return "unsafe" in content, payload
        except Exception as e:
            logger.error(f"Safeguard escalation failed: {e}, using fast result")
            payload = self._fast_payload(
                dome_input, fast_score, prediction, stage="fast-fallback"
            )
            payload["error"] = str(e)
            return fast_score >= self.score_threshold, payload

    # ------------------------------------------------------------------
    # Vijil remote inference helpers (hybrid fast stage)
    # ------------------------------------------------------------------

    async def _classify_vijil(
        self, client: httpx.AsyncClient, dome_input: DomePayload
    ) -> Tuple[float, dict]:
        """Classify a single input via Vijil's remote inference endpoint."""
        score = await self._vijil_client.classify(client, dome_input.query_string)
        return score, {}

    async def _classify_batch_vijil(
        self, dome_inputs: List[DomePayload]
    ) -> List[Tuple[float, dict]]:
        """Classify a batch via Vijil's remote inference endpoint."""
        async with httpx.AsyncClient(
            timeout=self._vijil_client.timeout_seconds
        ) as client:
            semaphore = asyncio.Semaphore(self.max_batch_concurrency)

            async def _one(di: DomePayload) -> Tuple[float, dict]:
                async with semaphore:
                    try:
                        score = await self._vijil_client.classify(
                            client, di.query_string
                        )
                        return score, {}
                    except Exception as e:
                        logger.error(f"Vijil inference batch item failed: {e}")
                        return 0.0, {"error": str(e)}

            return list(await asyncio.gather(*(_one(di) for di in dome_inputs)))

    # ------------------------------------------------------------------
    # Public detection interface
    # ------------------------------------------------------------------

    async def detect(self, dome_input: DomePayload) -> DetectionResult:
        dome_input = DomePayload.coerce(dome_input)
        logger.info("Detecting EEOC stereotype using hybrid mode...")

        # Stage 1: fast classification (remote or local)
        if self._use_vijil_inference:
            try:
                async with httpx.AsyncClient(
                    timeout=self._vijil_client.timeout_seconds
                ) as client:
                    score, prediction = await self._classify_vijil(client, dome_input)
            except Exception as e:
                logger.error(f"Vijil inference failed in hybrid fast stage: {e}")
                return False, {
                    "type": type(self),
                    "detector": STEREOTYPE_EEOC_HYBRID,
                    "stage": "vijil-inference-error",
                    "score": 0.0,
                    "label": "error",
                    "error": str(e),
                    "response_string": dome_input.query_string,
                }
        else:
            loop = asyncio.get_running_loop()
            score, prediction = await loop.run_in_executor(
                None, self._classify, dome_input
            )

        confidence = max(score, 1.0 - score)
        logger.debug(
            "stereotype-eeoc-hybrid fast-stage prediction=%s score=%.3f",
            prediction,
            score,
        )

        if confidence >= self.confidence_threshold:
            flagged = score >= self.score_threshold
            return flagged, self._fast_payload(dome_input, score, prediction, stage="fast")

        if not self.api_key:
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

        Runs the fast stage on the whole batch (remote or local), then
        escalates any low-confidence items to Safeguard concurrently.
        """
        dome_inputs = [DomePayload.coerce(x) for x in inputs]

        # Stage 1: batched fast classification (remote or local).
        if self._use_vijil_inference:
            scored = await self._classify_batch_vijil(dome_inputs)
        else:
            loop = asyncio.get_running_loop()
            scored = await loop.run_in_executor(
                None, self._classify_batch, dome_inputs
            )

        # Partition into "confident enough" vs "escalate".
        results: List[Optional[DetectionResult]] = [None] * len(dome_inputs)
        escalate_indices: List[int] = []
        for idx, (dome_input, (score, prediction)) in enumerate(zip(dome_inputs, scored)):
            confidence = max(score, 1.0 - score)
            if confidence >= self.confidence_threshold:
                flagged = score >= self.score_threshold
                results[idx] = (
                    flagged,
                    self._fast_payload(dome_input, score, prediction, stage="fast"),
                )
            elif not self.api_key:
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
                "stereotype-eeoc-hybrid escalating %d/%d low-confidence items",
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
