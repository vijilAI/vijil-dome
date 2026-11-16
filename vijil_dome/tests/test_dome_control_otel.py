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

"""Tests for the dome-control OTel span (VijilDome / Controls path).

Mirrors test_otel_instrumentation.py's MagicMock-tracer style used for
instrument_dome's idempotency tests.
"""

from __future__ import annotations

from typing import Literal
from unittest.mock import MagicMock

import pytest

from vijil_dome.controls.models import (
    Control,
    ControlAction,
    ControlMatch,
    EvaluationResult,
    Step,
)
from vijil_dome.core import VijilDome
from vijil_dome.integrations.instrumentation.otel_instrumentation import (
    _set_dome_control_span_attributes,
    instrument_vijil_dome,
)


def _ssn_block_control(name: str = "block-ssn") -> Control:
    return Control.model_validate(
        {
            "name": name,
            "condition": {
                "selector": "output",
                "evaluator": {"name": "regex", "config": {"pattern": r"\d{3}-\d{2}-\d{4}"}},
            },
            "action": {"decision": "deny", "message": "blocked"},
            "annotations": {"vijil.ai/source-policy-id": "POLICY-1"},
        }
    )


@pytest.mark.asyncio
async def test_triggered_control_emits_dome_control_span():
    dome = VijilDome(
        policy=[_ssn_block_control()], agent_id="agent-1", team_id="team-1", enforce=False
    )
    tracer = MagicMock()
    instrument_vijil_dome(dome, tracer=tracer)

    await dome.async_guard_output("here is 123-45-6789")

    tracer.start_as_current_span.assert_called_once_with("dome-control")
    span = tracer.start_as_current_span.return_value.__enter__.return_value
    set_calls = {c.args[0]: c.args[1] for c in span.set_attribute.call_args_list}
    assert set_calls["control.name"] == "block-ssn"
    assert set_calls["control.decision"] == "deny"
    assert set_calls["team.id"] == "team-1"
    assert set_calls["agent.id"] == "agent-1"
    assert set_calls["policy.id"] == "POLICY-1"


@pytest.mark.asyncio
async def test_non_triggered_control_emits_no_span():
    dome = VijilDome(
        policy=[_ssn_block_control()], agent_id="agent-1", enforce=False
    )
    tracer = MagicMock()
    instrument_vijil_dome(dome, tracer=tracer)

    await dome.async_guard_output("nothing sensitive here")

    tracer.start_as_current_span.assert_not_called()


@pytest.mark.asyncio
async def test_instrument_vijil_dome_is_idempotent():
    dome = VijilDome(policy=[_ssn_block_control()], enforce=False)
    tracer = MagicMock()

    instrument_vijil_dome(dome, tracer=tracer)
    original_evaluate = dome.engine.evaluate
    instrument_vijil_dome(dome, tracer=tracer)

    assert dome.engine.evaluate is original_evaluate


def test_span_attributes_surface_policy_rule_judge_metadata():
    control = Control.model_validate(
        {
            "name": "block-privacy",
            "condition": {
                "selector": "output",
                "evaluator": {"name": "policy-rule-judge", "config": {}},
            },
            "action": {"decision": "deny"},
            "annotations": {"vijil.ai/source-policy-id": "POLICY-9"},
        }
    )
    dome = VijilDome(policy=[control], agent_id="agent-9", user_id="user-9")

    match = ControlMatch(
        control_name="block-privacy",
        triggered=True,
        action=ControlAction(decision="deny"),
        confidence=0.93,
        metadata={
            "violated_rule_ids": ["PRIV-001"],
            "rule_verdicts": {
                "PRIV-001": {
                    "noul": 0.93,
                    "violated": True,
                    "rule_type": "prohibition",
                    "consequence": {"action": "block", "severity": "critical"},
                },
                "BRAND-002": {"noul": 0.1, "violated": False},
            },
        },
    )

    span = MagicMock()
    _set_dome_control_span_attributes(span, match, dome, Step(type="llm", name="t", output="x"), "post")

    set_calls = {c.args[0]: c.args[1] for c in span.set_attribute.call_args_list}
    assert set_calls["policy.id"] == "POLICY-9"
    assert set_calls["rule.ids"] == ["PRIV-001"]
    assert set_calls["consequence.action"] == "block"
    assert set_calls["consequence.severity"] == "critical"
    assert set_calls["rule.type"] == "prohibition"
    assert "BRAND-002" in set_calls["rule.verdicts"]
    assert set_calls["user.id"] == "user-9"


def _regex_control(selector: str) -> Control:
    return Control.model_validate(
        {
            "name": "match-secret",
            "condition": {
                "selector": selector,
                "evaluator": {"name": "regex", "config": {"pattern": "secret"}},
            },
            "action": {"decision": "deny"},
        }
    )


async def _fire_with_result(selector: str, step: Step, stage: Literal["pre", "post"]) -> tuple[dict, EvaluationResult]:
    dome = VijilDome(policy=[_regex_control(selector)], enforce=False)
    tracer = MagicMock()
    instrument_vijil_dome(dome, tracer=tracer)
    result = await dome.engine.evaluate(step, stage)
    tracer.start_as_current_span.assert_called_once_with("dome-control")
    span = tracer.start_as_current_span.return_value.__enter__.return_value
    return {c.args[0]: c.args[1] for c in span.set_attribute.call_args_list}, result


async def _fire(selector: str, step: Step, stage: Literal["pre", "post"]) -> dict:
    attrs, _ = await _fire_with_result(selector, step, stage)
    return attrs


@pytest.mark.asyncio
async def test_pre_stage_span_records_input_only():
    attrs = await _fire("input", Step(type="llm", name="t", input="my secret"), "pre")

    assert attrs["control.stage"] == "pre"
    assert attrs["control.input"] == "my secret"
    assert "control.output" not in attrs


@pytest.mark.asyncio
async def test_post_stage_span_records_output_only():
    step = Step(type="llm", name="t", input="question", output="a secret answer")
    attrs = await _fire("output", step, "post")

    assert attrs["control.stage"] == "post"
    assert attrs["control.output"] == "a secret answer"
    assert "control.input" not in attrs


@pytest.mark.asyncio
async def test_structured_input_is_recorded_as_json():
    attrs = await _fire("input", Step(type="llm", name="t", input={"msg": "secret"}), "pre")

    assert attrs["control.input"] == '{"msg": "secret"}'


@pytest.mark.asyncio
async def test_long_input_is_truncated():
    attrs = await _fire("input", Step(type="llm", name="t", input="secret" + "a" * 3000), "pre")

    assert attrs["control.input"].endswith("...<truncated>")
    assert len(attrs["control.input"]) == 2048


@pytest.mark.asyncio
async def test_non_ascii_input_is_recorded_unescaped():
    attrs = await _fire("input", Step(type="llm", name="t", input={"msg": "secret café 秘密"}), "pre")

    assert attrs["control.input"] == '{"msg": "secret café 秘密"}'


def _circular_input() -> dict:
    payload: dict = {"msg": "secret"}
    payload["self"] = payload
    return payload


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [{("a", "b"): 1, "msg": "secret"}, _circular_input()],
    ids=["tuple-key", "circular"],
)
async def test_unencodable_input_still_returns_the_evaluation(payload: dict):
    attrs, result = await _fire_with_result("input", Step(type="llm", name="t", input=payload), "pre")

    assert isinstance(attrs["control.input"], str)
    assert "secret" in attrs["control.input"]
    assert any(m.triggered for m in result.matches)
