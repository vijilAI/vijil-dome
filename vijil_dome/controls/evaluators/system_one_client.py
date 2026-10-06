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

"""Client for TypeSafe's "System One" typed-decision API (Jev).

Jev returns calibrated typed answers instead of generated text. This client
only implements the ``noul`` (boolean-probability) primitive, which is what
:class:`~vijil_dome.controls.evaluators.policy_rule_judge.PolicyRuleJudge`
needs: "did this turn violate rule X" is a yes/no question per rule, posed
in one batched call against the same state.

Reachable either directly at TypeSafe (``api.typesafe.ai``) or, as deployed
here, through OpenRouter's TypeSafe-compatible shim at the same path —
"switch the base URL only". See
``docs/policy-enforcement/2026-10-06-policy-enforcement-scoping.md`` (§3.5)
for how this was confirmed against TypeSafe's and OpenRouter's own docs.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1/systemone"
DEFAULT_MODEL = "typesafe/jev-1.13"


class SystemOneError(RuntimeError):
    """Raised when the System One API returns an error or an unusable response."""


class SystemOneClient:
    """Minimal async client for System One's ``noul`` primitive.

    Parameters
    ----------
    api_key:
        OpenRouter (or TypeSafe) API key. Defaults to the
        ``OPENROUTER_API_KEY`` environment variable.
    base_url:
        Defaults to OpenRouter's TypeSafe-compatible endpoint. Pass
        ``"https://api.typesafe.ai/v1/systemone"`` to call TypeSafe directly.
    model:
        Defaults to ``typesafe/jev-1.13``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url
        self.model = model
        self.timeout = timeout

    async def ask_noul_batch(
        self,
        state: str,
        questions: dict[str, dict[str, Any]],
    ) -> dict[str, float]:
        """Pose a batch of Noul (yes/no) questions against one state.

        Parameters
        ----------
        state:
            The text being evaluated (e.g. the current agent turn).
        questions:
            Maps an arbitrary question key (we use the policy rule's
            ``rule_id``) to ``{"instructions": str, "criteria": {"true": str,
            "false": str}}`` (``criteria`` optional).

        Returns
        -------
        dict mapping each question key to its ``noul`` probability
        (0.0 = strong no, 1.0 = strong yes). Keys the caller asked for that
        are missing from the response are omitted, not defaulted — callers
        should treat a missing key as "unknown", not "no violation".

        Raises
        ------
        SystemOneError:
            On a missing API key, a non-2xx response, or a response that
            doesn't contain an ``answers`` object. Deliberately not caught
            here — ``ControlEngine`` already has fail-open/fail-closed
            semantics per control, keyed on exactly this kind of exception.
        """
        if not self.api_key:
            raise SystemOneError(
                "No API key configured for System One. Set OPENROUTER_API_KEY "
                "or pass api_key= explicitly."
            )
        if not questions:
            return {}

        payload: dict[str, Any] = {
            "state": state,
            "model": self.model,
            "questions": {
                key: {"type": "noul", **q} for key, q in questions.items()
            },
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
        except httpx.HTTPError as exc:
            raise SystemOneError(f"System One request failed: {exc}") from exc

        if response.status_code >= 400:
            raise SystemOneError(
                f"System One returned HTTP {response.status_code}: {response.text[:500]}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise SystemOneError(f"System One returned non-JSON response: {exc}") from exc

        answers = data.get("answers")
        if not isinstance(answers, dict):
            raise SystemOneError(
                f"System One response missing 'answers' object: {data!r}"
            )

        results: dict[str, float] = {}
        for key, answer in answers.items():
            if isinstance(answer, dict) and "noul" in answer:
                try:
                    results[key] = float(answer["noul"])
                except (TypeError, ValueError):
                    continue
        return results
