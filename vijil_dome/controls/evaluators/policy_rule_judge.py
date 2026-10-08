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

"""Judges a turn against a batch of Console-extracted policy rules.

Registered as a *control evaluator* (``@register_evaluator``), not a Dome
``DetectionMethod`` wired through the ``dome:`` bridge. That choice is
deliberate: ``ControlEngine._get_evaluator`` caches evaluator instances by
evaluator *name* only (``vijil_dome/controls/engine.py``), and
``DomeBridgeEvaluator`` additionally caches the underlying detector instance
by whatever ``detector_kwargs`` it was first constructed with
(``vijil_dome/controls/evaluators/dome_bridge.py``). Two Controls that both
reference ``dome:policy-rule-judge`` but carry *different* rule sets (e.g.
one per ``consequence.action`` bucket) would silently share one
bridge-cached detector built from whichever Control's rules happened to
construct it first — every other bucket would judge against the wrong
rules.

A plain registered :class:`Evaluator`, by contrast, receives ``config`` fresh
on *every* call (see ``RegexEvaluator`` for the same pattern) — only the
stateless evaluator object itself is shared across Controls, never the
per-call config. That's what we need: the rules list lives in each Control's
own ``evaluator.config``, not in constructor state.
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any

from vijil_dome.controls.evaluators import register_evaluator
from vijil_dome.controls.evaluators.base import Evaluator, EvaluatorResult
from vijil_dome.controls.evaluators.system_one_client import SystemOneClient, SystemOneError

logger = logging.getLogger(__name__)

DEFAULT_VIOLATION_THRESHOLD = 0.5
DEFAULT_BATCH_SIZE = 20


def _build_question(rule: dict[str, Any]) -> dict[str, Any]:
    natural_language = rule.get("natural_language", "")
    rule_id = rule.get("rule_id", "?")
    question: dict[str, Any] = {
        "instructions": (
            f"Policy rule {rule_id}: {natural_language}\n\n"
            "Does the turn below violate this rule?"
        ),
        "criteria": {
            "true": "The turn violates this rule.",
            "false": "The turn complies with this rule, or this rule does not apply.",
        },
    }
    return question


def _chunk(rules: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    if size <= 0:
        return [rules]
    return [rules[i : i + size] for i in range(0, len(rules), size)]


def _validate_rule_ids(rules: list[dict[str, Any]]) -> None:
    """Raise if any rule is missing a ``rule_id``, or two rules share one.

    ``_score_all`` keys both its per-batch question dict and its merged
    score dict by ``rule_id`` -- a duplicate (or a missing id, which
    defaults to the same ``"?"`` placeholder for every rule lacking one)
    silently collapses two distinct rules onto one score. ``_validate_scores``
    then sees a "valid" score for that id and applies it to every rule
    sharing it, so a duplicated prohibition can go unjudged even with
    ``on_error: fail_closed``. Catching this before scoring, rather than
    trying to detect it after the fact, is the only way to guarantee every
    rule actually got its own judgment.
    """
    seen: set[str] = set()
    bad: list[str] = []
    for rule in rules:
        rule_id = rule.get("rule_id")
        if not rule_id or not isinstance(rule_id, str):
            bad.append(f"missing rule_id (rule={rule!r})")
        elif rule_id in seen:
            bad.append(f"duplicate rule_id={rule_id!r}")
        else:
            seen.add(rule_id)
    if bad:
        raise SystemOneError(f"Invalid rules config: {', '.join(bad)}")


def _validate_threshold(threshold: Any) -> float:
    """Raise if ``violation_threshold`` isn't a finite number in [0, 1].

    An out-of-range threshold (e.g. > 1.0) makes ``score >= threshold``
    false for every possible score, so a deny Control would silently never
    flag a violation -- with no exception raised, ``on_error: fail_closed``
    never gets a chance to apply. Validating up front, the same as
    ``_validate_scores`` does for scores, ensures bad config fails the
    engine's error policy instead of failing open.
    """
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not math.isfinite(threshold)
        or not (0.0 <= threshold <= 1.0)
    ):
        raise SystemOneError(
            f"Invalid violation_threshold: {threshold!r} (must be a finite number in [0, 1])"
        )
    return float(threshold)


def _validate_scores(
    rules: list[dict[str, Any]], noul_scores: dict[str, float]
) -> None:
    """Raise if System One didn't return a usable score for every rule.

    A missing, non-finite, or out-of-[0,1]-range score previously fell
    through to "not violated" (``score is not None and score >= threshold``),
    which bypasses ``ControlEngine``'s ``on_error`` fail_open/fail_closed
    entirely -- a judge failure on a ``block``-bucket Control would
    silently *allow* instead of failing closed. Raising here instead lets
    the engine's existing per-Control error policy decide, the same as
    any other evaluator failure.
    """
    bad: list[str] = []
    for rule in rules:
        rule_id = rule.get("rule_id", "?")
        score = noul_scores.get(rule_id)
        if score is None or not math.isfinite(score) or not (0.0 <= score <= 1.0):
            bad.append(f"{rule_id}={score!r}")
    if bad:
        raise SystemOneError(
            f"System One returned missing or invalid scores for: {', '.join(bad)}"
        )


@register_evaluator("policy-rule-judge")
class PolicyRuleJudge(Evaluator):
    """Judges a turn against a batch of policy rules using Jev/System One.

    Config keys (set per-Control by Console's ``PolicyGapCompiler`` — see
    the scoping doc §5A/§5C):
        rules (list[dict]): Each entry carries at least ``rule_id`` and
            ``natural_language``; ``rule_type``/``action``/``target``/
            ``consequence`` are echoed back in ``EvaluatorResult.metadata``
            for traceability (e.g. the OTel ``dome-control`` span) but don't
            affect the judging itself.
        violation_threshold (float): Noul probability at/above which a rule
            counts as violated. Default ``0.5``.
        batch_size (int): Max rules per System One call. Default ``20`` —
            keeps each call well inside Jev's 32k-token context window.
            Batches run concurrently.
        model (str): Overrides :data:`SystemOneClient`'s default model.
        api_key (str): Overrides ``OPENROUTER_API_KEY``.
    """

    async def evaluate(
        self, value: Any, config: dict[str, Any]
    ) -> EvaluatorResult:
        rules: list[dict[str, Any]] = config.get("rules") or []
        if not rules:
            return EvaluatorResult(matched=False, message="No rules configured")
        _validate_rule_ids(rules)

        text = str(value) if value is not None else ""
        threshold = _validate_threshold(config.get("violation_threshold", DEFAULT_VIOLATION_THRESHOLD))
        batch_size = config.get("batch_size", DEFAULT_BATCH_SIZE)

        client_kwargs: dict[str, Any] = {}
        if "model" in config:
            client_kwargs["model"] = config["model"]
        if "api_key" in config:
            client_kwargs["api_key"] = config["api_key"]
        client = SystemOneClient(**client_kwargs)

        noul_scores = await self._score_all(client, text, rules, batch_size)
        _validate_scores(rules, noul_scores)

        verdicts: dict[str, dict[str, Any]] = {}
        violated_rule_ids: list[str] = []
        for rule in rules:
            rule_id = rule.get("rule_id", "?")
            score = noul_scores.get(rule_id)
            violated = score is not None and score >= threshold
            if violated:
                violated_rule_ids.append(rule_id)
            verdicts[rule_id] = {
                "noul": score,
                "violated": violated,
                "rule_type": rule.get("rule_type"),
                "action": rule.get("action"),
                "target": rule.get("target"),
                "natural_language": rule.get("natural_language"),
                "consequence": rule.get("consequence"),
            }

        matched = bool(violated_rule_ids)
        scored = [s for s in noul_scores.values() if s is not None]
        if matched:
            confidence = max(
                noul_scores[rid] for rid in violated_rule_ids if noul_scores.get(rid) is not None
            )
            message = f"Violated rule(s): {', '.join(violated_rule_ids)}"
        else:
            confidence = 1.0 - max(scored) if scored else 1.0
            message = "No rule violations detected"

        return EvaluatorResult(
            matched=matched,
            confidence=max(0.0, min(1.0, confidence)),
            message=message,
            metadata={
                "violated_rule_ids": violated_rule_ids,
                "rule_verdicts": verdicts,
            },
        )

    async def _score_all(
        self,
        client: SystemOneClient,
        text: str,
        rules: list[dict[str, Any]],
        batch_size: int,
    ) -> dict[str, float]:
        batches = _chunk(rules, batch_size)
        batch_results = await asyncio.gather(
            *(
                client.ask_noul_batch(
                    text,
                    {rule.get("rule_id", "?"): _build_question(rule) for rule in batch},
                )
                for batch in batches
            )
        )
        merged: dict[str, float] = {}
        for result in batch_results:
            merged.update(result)
        return merged
