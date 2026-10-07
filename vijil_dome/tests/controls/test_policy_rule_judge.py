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

from unittest.mock import AsyncMock, patch

import pytest

from vijil_dome.controls.engine import ControlEngine
from vijil_dome.controls.evaluators import resolve_evaluator
from vijil_dome.controls.evaluators.policy_rule_judge import PolicyRuleJudge
from vijil_dome.controls.evaluators.system_one_client import SystemOneError
from vijil_dome.controls.models import Step

RULE_BLOCK = {
    "rule_id": "PRIV-001",
    "rule_type": "prohibition",
    "natural_language": "The agent must not share a user's SSN.",
    "action": "share_data",
    "target": "ssn",
    "consequence": {"action": "block", "severity": "critical"},
}
RULE_WARN = {
    "rule_id": "BRAND-002",
    "rule_type": "recommendation",
    "natural_language": "The agent should use a friendly tone.",
    "action": "respond",
    "consequence": {"action": "warn", "severity": "low"},
}


def _patch_scores(scores: dict[str, float]):
    return patch.object(
        PolicyRuleJudge,
        "_score_all",
        AsyncMock(return_value=scores),
    )


@pytest.mark.asyncio
async def test_resolves_via_registry():
    evaluator = resolve_evaluator("policy-rule-judge")
    assert isinstance(evaluator, PolicyRuleJudge)


@pytest.mark.asyncio
async def test_no_rules_configured_does_not_match():
    evaluator = PolicyRuleJudge()
    result = await evaluator.evaluate("some turn", {"rules": []})
    assert result.matched is False
    assert "No rules" in result.message


@pytest.mark.asyncio
async def test_violation_above_threshold_matches():
    evaluator = PolicyRuleJudge()
    with _patch_scores({"PRIV-001": 0.9}):
        result = await evaluator.evaluate(
            "Here is the user's SSN: 123-45-6789",
            {"rules": [RULE_BLOCK]},
        )
    assert result.matched is True
    assert result.confidence == pytest.approx(0.9)
    assert "PRIV-001" in result.metadata["violated_rule_ids"]
    assert result.metadata["rule_verdicts"]["PRIV-001"]["violated"] is True
    assert result.metadata["rule_verdicts"]["PRIV-001"]["consequence"] == {
        "action": "block",
        "severity": "critical",
    }


@pytest.mark.asyncio
async def test_score_below_threshold_does_not_match():
    evaluator = PolicyRuleJudge()
    with _patch_scores({"PRIV-001": 0.1}):
        result = await evaluator.evaluate(
            "The weather is nice today",
            {"rules": [RULE_BLOCK]},
        )
    assert result.matched is False
    assert result.metadata["violated_rule_ids"] == []
    assert result.confidence == pytest.approx(0.9)  # 1 - max(scores)


@pytest.mark.asyncio
async def test_custom_threshold_is_respected():
    evaluator = PolicyRuleJudge()
    with _patch_scores({"PRIV-001": 0.6}):
        result = await evaluator.evaluate(
            "borderline text",
            {"rules": [RULE_BLOCK], "violation_threshold": 0.8},
        )
    assert result.matched is False

    with _patch_scores({"PRIV-001": 0.6}):
        result = await evaluator.evaluate(
            "borderline text",
            {"rules": [RULE_BLOCK], "violation_threshold": 0.5},
        )
    assert result.matched is True


@pytest.mark.asyncio
async def test_multiple_rules_only_violated_ones_listed():
    evaluator = PolicyRuleJudge()
    with _patch_scores({"PRIV-001": 0.95, "BRAND-002": 0.1}):
        result = await evaluator.evaluate(
            "Here is the SSN, have a nice day",
            {"rules": [RULE_BLOCK, RULE_WARN]},
        )
    assert result.matched is True
    assert result.metadata["violated_rule_ids"] == ["PRIV-001"]
    assert result.metadata["rule_verdicts"]["BRAND-002"]["violated"] is False


@pytest.mark.asyncio
async def test_missing_score_raises_instead_of_silently_passing():
    """A missing score must not fall through to "not violated" -- that
    would bypass ControlEngine's on_error entirely (a judge failure on a
    block-bucket Control should fail_closed, not silently allow)."""
    evaluator = PolicyRuleJudge()
    with _patch_scores({}):  # System One dropped this rule_id from its answers
        with pytest.raises(SystemOneError, match="PRIV-001"):
            await evaluator.evaluate("text", {"rules": [RULE_BLOCK]})


@pytest.mark.asyncio
async def test_non_finite_score_raises():
    evaluator = PolicyRuleJudge()
    with _patch_scores({"PRIV-001": float("nan")}):
        with pytest.raises(SystemOneError, match="PRIV-001"):
            await evaluator.evaluate("text", {"rules": [RULE_BLOCK]})


@pytest.mark.asyncio
async def test_out_of_range_score_raises():
    evaluator = PolicyRuleJudge()
    with _patch_scores({"PRIV-001": 1.5}):
        with pytest.raises(SystemOneError, match="PRIV-001"):
            await evaluator.evaluate("text", {"rules": [RULE_BLOCK]})


@pytest.mark.asyncio
async def test_out_of_range_threshold_raises_instead_of_never_matching():
    """A violation_threshold > 1.0 makes score >= threshold false for every
    possible score -- a deny Control would then silently never flag a
    violation, with no exception for on_error: fail_closed to catch."""
    evaluator = PolicyRuleJudge()
    with _patch_scores({"PRIV-001": 1.0}):  # max possible score
        with pytest.raises(SystemOneError, match="violation_threshold"):
            await evaluator.evaluate(
                "text", {"rules": [RULE_BLOCK], "violation_threshold": 1.1}
            )


@pytest.mark.asyncio
async def test_non_finite_threshold_raises():
    evaluator = PolicyRuleJudge()
    with _patch_scores({"PRIV-001": 1.0}):
        with pytest.raises(SystemOneError, match="violation_threshold"):
            await evaluator.evaluate(
                "text", {"rules": [RULE_BLOCK], "violation_threshold": float("nan")}
            )


@pytest.mark.asyncio
async def test_duplicate_rule_id_raises_instead_of_silently_sharing_a_score():
    """Two rules sharing a rule_id would collapse onto one score in
    _score_all's batch dict, then both appear "validly scored" to
    _validate_scores even though only one was actually judged."""
    evaluator = PolicyRuleJudge()
    duplicate = {**RULE_WARN, "rule_id": "PRIV-001"}
    with pytest.raises(SystemOneError, match="duplicate"):
        await evaluator.evaluate("text", {"rules": [RULE_BLOCK, duplicate]})


@pytest.mark.asyncio
async def test_missing_rule_id_raises():
    evaluator = PolicyRuleJudge()
    no_id_rule = {k: v for k, v in RULE_BLOCK.items() if k != "rule_id"}
    with pytest.raises(SystemOneError, match="missing rule_id"):
        await evaluator.evaluate("text", {"rules": [no_id_rule]})


def _block_control(on_error: str) -> dict:
    return {
        "name": "block-bucket",
        "condition": {
            "selector": "output",
            "evaluator": {"name": "policy-rule-judge", "config": {"rules": [RULE_BLOCK]}},
        },
        "action": {"decision": "deny", "on_error": on_error},
    }


@pytest.mark.asyncio
async def test_judge_failure_is_fail_closed_for_a_deny_control():
    """A block-bucket Control's own on_error=fail_closed must win when the
    judge itself fails (not just when it judges "no violation")."""
    engine = ControlEngine()
    engine.load_controls([_block_control("fail_closed")])
    with _patch_scores({}):  # judge raises SystemOneError
        result = await engine.evaluate(
            Step(type="llm", name="output", output="anything"), stage="post"
        )
    assert result.action == "deny"
    assert result.matches[0].error is not None


@pytest.mark.asyncio
async def test_judge_failure_is_fail_open_when_configured():
    engine = ControlEngine()
    engine.load_controls([_block_control("fail_open")])
    with _patch_scores({}):  # judge raises SystemOneError
        result = await engine.evaluate(
            Step(type="llm", name="output", output="anything"), stage="post"
        )
    assert result.action == "allow"
    assert result.matches[0].error is not None


@pytest.mark.asyncio
async def test_batches_rules_across_concurrent_calls():
    evaluator = PolicyRuleJudge()
    rules = [
        {**RULE_BLOCK, "rule_id": f"R-{i}"} for i in range(5)
    ]
    seen_batches: list[list[str]] = []

    async def fake_ask_noul_batch(text, questions):
        seen_batches.append(list(questions.keys()))
        return {rid: 0.0 for rid in questions}

    with patch(
        "vijil_dome.controls.evaluators.policy_rule_judge.SystemOneClient.ask_noul_batch",
        AsyncMock(side_effect=fake_ask_noul_batch),
    ):
        result = await evaluator.evaluate(
            "text", {"rules": rules, "batch_size": 2}
        )

    assert result.matched is False
    assert len(seen_batches) == 3  # ceil(5/2)
    assert sum(len(b) for b in seen_batches) == 5
