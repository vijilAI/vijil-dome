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
Client for Vijil's hosted ModernBERT inference endpoints.

This module provides a lightweight HTTP client for Vijil's remote model
inference service.  The endpoint follows an OpenAI-compatible
``/v1/chat/completions`` shape but returns a **single float score** as
the assistant message content — it is NOT a general-purpose OpenAI proxy
and must only be pointed at a Vijil inference deployment.

Usage::

    client = VijilInferenceClient(
        base_url="https://inference.example.vijil.ai",
        model="vijil/vijil_dome_prompt_injection_detection",
    )
    async with httpx.AsyncClient(timeout=10.0) as http:
        score = await client.classify(http, "some input text")
"""

import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger("vijil.dome")

DEFAULT_VIJIL_INFERENCE_API_KEY_NAME = "VIJIL_INFERENCE_API_KEY"


class VijilInferenceClient:
    """HTTP client for Vijil's hosted model inference service.

    This client is tightly coupled to Vijil's inference endpoint — it
    expects the response ``content`` to be a single float score.  Do not
    point it at arbitrary OpenAI-compatible endpoints; use only with
    Vijil-operated inference deployments.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        api_key_name: str = DEFAULT_VIJIL_INFERENCE_API_KEY_NAME,
        timeout_seconds: float = 10.0,
        max_tokens: int = 32,
    ):
        self.base_url = base_url.rstrip("/")
        self.completions_url = f"{self.base_url}/v1/chat/completions"
        self.model = model
        self.api_key = api_key or os.environ.get(api_key_name, "")
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens

    def _build_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _build_payload(self, text: str) -> dict:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": self.max_tokens,
        }

    async def classify(self, client: httpx.AsyncClient, text: str) -> float:
        """Send *text* to the Vijil inference endpoint and return the score.

        Raises ``ValueError`` if the response content cannot be parsed
        as a float, or ``httpx.HTTPStatusError`` on non-2xx responses.
        """
        resp = await client.post(
            self.completions_url,
            headers=self._build_headers(),
            json=self._build_payload(text),
        )
        resp.raise_for_status()
        content = (
            resp.json()["choices"][0]["message"].get("content", "").strip()
        )
        try:
            return float(content)
        except (ValueError, TypeError):
            logger.error(
                "Vijil inference returned non-numeric content for model %s: %r "
                "(expected a float score). Defaulting to 0.0.",
                self.model,
                content,
            )
            return 0.0
