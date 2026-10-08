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

"""EvaluatorResult.metadata/confidence threading into ControlMatch.

A leaf condition's full EvaluatorResult (not just .matched) needs to reach
ControlMatch so instrumentation (the dome-control OTel span) can surface
evaluator-specific details like PolicyRuleJudge's violated rule_ids.
"""

from __future__ import annotations

import pytest

from vijil_dome.controls.engine import ControlEngine
from vijil_dome.controls.models import Control, Step


def _control_with_condition(name: str, condition: dict, decision: str = "deny") -> Control:
    return Control.model_validate(
        {
            "name": name,
            "condition": condition,
            "action": {"decision": decision},
        }
    )


@pytest.mark.asyncio
async def test_leaf_match_carries_evaluator_metadata_and_confidence():
    control = _control_with_condition(
        "regex-ssn",
        {
            "selector": "input",
            "evaluator": {"name": "regex", "config": {"pattern": r"\d{3}-\d{2}-\d{4}"}},
        },
    )
    engine = ControlEngine([control])
    step = Step(type="llm", name="chat", input="my ssn is 123-45-6789")

    result = await engine.evaluate(step, stage="pre")

    assert result.action == "deny"
    assert len(result.matches) == 1
    match = result.matches[0]
    assert match.triggered is True
    assert match.confidence == pytest.approx(1.0)
    assert match.metadata["pattern"] == r"\d{3}-\d{2}-\d{4}"


@pytest.mark.asyncio
async def test_non_triggered_leaf_still_carries_metadata():
    control = _control_with_condition(
        "regex-ssn",
        {
            "selector": "input",
            "evaluator": {"name": "regex", "config": {"pattern": r"\d{3}-\d{2}-\d{4}"}},
        },
    )
    engine = ControlEngine([control])
    step = Step(type="llm", name="chat", input="nothing sensitive here")

    result = await engine.evaluate(step, stage="pre")

    assert result.action == "allow"
    assert len(result.matches) == 1
    assert result.matches[0].triggered is False
    assert "using_re2" in result.matches[0].metadata


@pytest.mark.asyncio
async def test_composite_condition_leaves_metadata_empty():
    """Composite and/or/not conditions keep the pre-existing bool-only path --
    no regression, metadata enrichment is leaf-only for now."""
    control = _control_with_condition(
        "composite",
        {
            "or": [
                {
                    "selector": "input",
                    "evaluator": {"name": "regex", "config": {"pattern": "a"}},
                },
                {
                    "selector": "input",
                    "evaluator": {"name": "regex", "config": {"pattern": "b"}},
                },
            ]
        },
    )
    engine = ControlEngine([control])
    step = Step(type="llm", name="chat", input="abc")

    result = await engine.evaluate(step, stage="pre")

    assert result.matches[0].triggered is True
    assert result.matches[0].metadata == {}
    assert result.matches[0].confidence == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_control_action_message_overrides_evaluator_message():
    control = Control.model_validate(
        {
            "name": "regex-ssn",
            "condition": {
                "selector": "input",
                "evaluator": {"name": "regex", "config": {"pattern": r"\d{3}-\d{2}-\d{4}"}},
            },
            "action": {"decision": "deny", "message": "custom block message"},
        }
    )
    engine = ControlEngine([control])
    step = Step(type="llm", name="chat", input="123-45-6789")

    result = await engine.evaluate(step, stage="pre")

    assert result.matches[0].message == "custom block message"
