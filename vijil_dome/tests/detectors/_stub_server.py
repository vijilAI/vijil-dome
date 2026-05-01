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

"""Stub inference server for the thin-client end-to-end tests.

Test-only scaffolding — never imported by the shipped library. Implements
``POST /v1/detect`` with heuristic scoring (regex pattern matches, not
real model inference) so ``test_thin_client_e2e.py`` can drive the full
Dome → dispatcher → HTTP → server → result pipeline without spinning up
the real inference stack.

Used as a sibling import from ``test_thin_client_e2e.py``::

    from _stub_server import app

Requires fastapi + uvicorn (Poetry dev group).
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from vijil_dome.detectors.detection_api import (
    DetectRequest,
    DetectResponse,
    DetectorInvocation,
    DetectorResult,
)

logger = logging.getLogger(__name__)

VERSION = "stub-1.0.0"

app = FastAPI(title="Vijil Inference Stub", version=VERSION)


# ---------------------------------------------------------------------------
# Heuristic detectors — simple pattern matching, no models
# ---------------------------------------------------------------------------

# Prompt injection patterns
_PI_PATTERNS = [
    r"ignore\s+(previous|above|all)\s+(instructions|prompts)",
    r"you\s+are\s+now\s+(acting|pretending)",
    r"system\s*prompt",
    r"disregard\s+(your|the)\s+(rules|instructions)",
    r"jailbreak",
    r"DAN\s+mode",
]

# Toxicity patterns
_TOXIC_PATTERNS = [
    r"\b(kill|murder|attack|destroy|hurt)\b.*\b(you|them|people)\b",
    r"\b(hate|despise)\b.*\b(you|them|everyone)\b",
    r"\b(stupid|idiot|moron|dumb)\b",
]

# PII patterns
_PII_PATTERNS = {
    "CREDIT_CARD": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "PHONE": r"\b\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "API_KEY": r"\b(sk_live_|sk_test_|api_key|api_secret|whsec_)\w+\b",
}

# Harmful prompt patterns
_HARMFUL_PATTERNS = [
    r"\b(how\s+to|help\s+me)\s+(hack|steal|fraud|launder|exploit)\b",
    r"\b(make\s+a\s+bomb|create\s+malware|forge\s+documents)\b",
    r"\b(money\s+laundering|identity\s+theft|tax\s+evasion)\b",
]

# Stereotype patterns
_STEREOTYPE_PATTERNS = [
    r"\b(women|men|blacks|whites|asians|muslims|jews)\b.*(always|never|can't|shouldn't)\b",
    r"\b(all|every)\s+(women|men|immigrants|elderly)\b",
]


def _score_patterns(text: str, patterns: list[str]) -> float:
    """Score text against regex patterns. Returns 0.0-1.0."""
    text_lower = text.lower()
    matches = sum(1 for p in patterns if re.search(p, text_lower, re.IGNORECASE))
    if matches == 0:
        return 0.0
    return min(1.0, 0.3 + matches * 0.25)


def _detect_pii(text: str) -> tuple[float, dict[str, Any]]:
    """Detect PII patterns. Returns (score, details)."""
    found_types: list[str] = []
    for entity_type, pattern in _PII_PATTERNS.items():
        if re.search(pattern, text):
            found_types.append(entity_type)
    if not found_types:
        return 0.0, {"entity_types": [], "count": 0}
    score = min(1.0, 0.5 + len(found_types) * 0.15)
    return score, {"entity_types": found_types, "count": len(found_types)}


def _run_detector(invocation: DetectorInvocation, input_text: str, prompt_text: str | None) -> DetectorResult:
    """Run a single stub detector and return a result.

    ``input_text`` is the flattened query string (``payload.query_string``).
    ``prompt_text`` is ``payload.prompt`` when ``payload.response`` is also
    set — used by detectors that need the upstream input as context
    (hallucination, fact-check) separate from the response under review.
    """
    start = time.monotonic()
    name = invocation.detector_name
    threshold = invocation.config.get("threshold", 0.5)

    # LLM-detector → semantic category (matches inference#30 server)
    _LLM_CATEGORIES = {
        "security-llm": "security",
        "moderation-prompt-engineering": "moderation",
        "hallucination-llm": "hallucination",
        "fact-check-llm": "fact-check",
        "generic-llm": "generic",
        "policy-gpt-oss-safeguard": "policy",
    }

    # Route to appropriate heuristic. Names match dome's canonical
    # registry and the inference server's @register_detector decorators.
    if name in ("prompt-injection-mbert",
                "prompt-injection-deberta-v3-base",
                "prompt-injection-deberta-finetuned-11122024"):
        score = _score_patterns(input_text, _PI_PATTERNS)
        category = "prompt_injection"
        details: dict[str, Any] = {}
    elif name in ("moderation-mbert", "moderation-deberta"):
        score = _score_patterns(input_text, _TOXIC_PATTERNS)
        category = "moderation"
        details = {}
    elif name == "privacy-presidio":
        score, details = _detect_pii(input_text)
        category = "privacy"
    elif name == "stereotype-eeoc-fast":
        score = _score_patterns(input_text, _STEREOTYPE_PATTERNS)
        category = "stereotype"
        details = {}
    elif name == "prompt-harmfulness-fast":
        score = _score_patterns(input_text, _HARMFUL_PATTERNS)
        category = "prompt_harmfulness"
        details = {}
    elif name in _LLM_CATEGORIES:
        # LLM-based: combine multiple pattern sets
        pi_score = _score_patterns(input_text, _PI_PATTERNS)
        toxic_score = _score_patterns(input_text, _TOXIC_PATTERNS)
        harmful_score = _score_patterns(input_text, _HARMFUL_PATTERNS)
        score = max(pi_score, toxic_score, harmful_score)
        category = _LLM_CATEGORIES[name]
        details = {"model": "stub", "reasoning": "heuristic pattern match"}
    else:
        # Unknown detector — return safe with a warning
        latency = (time.monotonic() - start) * 1000
        return DetectorResult(
            detector_name=name,
            is_flagged=False,
            score=0.0,
            category="unknown",
            details={"warning": f"Unknown detector: {name}"},
            latency_ms=latency,
        )

    latency = (time.monotonic() - start) * 1000
    return DetectorResult(
        detector_name=name,
        is_flagged=score >= threshold,
        score=round(score, 4),
        category=category,
        details=details,
        latency_ms=round(latency, 2),
    )


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------


@app.post("/v1/detect")
async def detect(request: DetectRequest) -> JSONResponse:
    """Batch detection endpoint — matches the inference server contract."""
    start = time.monotonic()

    # Flatten DomePayload to (input_text, prompt_text) at the dispatch
    # boundary. Mirror of inference-server/server.py:detect — see the
    # KNOWN FIDELITY GAP note there about prompt-harmfulness-fast and
    # stereotype-eeoc-fast (deferred to a follow-up; this stub adopts
    # the same flatten so behavior matches the real server).
    input_text = request.payload.query_string
    prompt_text = request.payload.prompt if request.payload.response else None

    results = [
        _run_detector(inv, input_text, prompt_text)
        for inv in request.detectors
    ]

    total_latency = (time.monotonic() - start) * 1000
    response = DetectResponse(
        results=results,
        server_version=VERSION,
        total_latency_ms=round(total_latency, 2),
    )
    return JSONResponse(content=response.model_dump())


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy", "version": VERSION, "mode": "stub"}


@app.get("/v1/detectors")
async def list_detectors() -> dict[str, list[str]]:
    """List available detectors in this stub server.

    Mirrors the canonical set the real inference server (vijil-inference#30)
    advertises. Mode (``stub`` vs production) is reported by ``/health``.
    """
    return {
        "detectors": [
            "prompt-injection-mbert",
            "prompt-injection-deberta-v3-base",
            "prompt-injection-deberta-finetuned-11122024",
            "moderation-mbert",
            "moderation-deberta",
            "stereotype-eeoc-fast",
            "prompt-harmfulness-fast",
            "privacy-presidio",
            "security-llm",
            "moderation-prompt-engineering",
            "hallucination-llm",
            "fact-check-llm",
            "generic-llm",
            "policy-gpt-oss-safeguard",
        ],
    }
