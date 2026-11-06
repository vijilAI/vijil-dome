"""Tests for precondition gates and ``Control.skip_on_missing_condition``.

Vijil Console compiles conditional policy rules (``unless user.role == admin``)
into controls shaped like::

    condition: and: [ <precondition leaf on context.<field>>, <judge leaf> ]
    skip_on_missing_condition: false

A precondition leaf (``precondition: true``) that cannot be judged -- its
selector does not resolve, or its evaluator errors -- is decided by the
control's ``skip_on_missing_condition`` rather than being a plain non-match:
False (default) enforces the control anyway, True skips it.
"""

from __future__ import annotations

from typing import Any

import pytest

try:
    import celpy  # noqa: F401

    _HAS_CEL = True
except ImportError:
    _HAS_CEL = False

pytestmark = pytest.mark.skipif(not _HAS_CEL, reason="cel-python not installed")

from vijil_dome.controls.engine import ControlEngine  # noqa: E402
from vijil_dome.controls.evaluators import register_evaluator  # noqa: E402
from vijil_dome.controls.evaluators.base import Evaluator, EvaluatorResult  # noqa: E402
from vijil_dome.controls.models import Control, Step  # noqa: E402


@register_evaluator("test-gate-always-match")
class _AlwaysMatch(Evaluator):
    """Stands in for the judge so only the gate decides the outcome."""

    async def evaluate(self, value: Any, config: dict[str, Any]) -> EvaluatorResult:
        return EvaluatorResult(matched=True, confidence=1.0, message="match")


@register_evaluator("test-gate-raises")
class _Raises(Evaluator):
    async def evaluate(self, value: Any, config: dict[str, Any]) -> EvaluatorResult:
        raise RuntimeError("boom")


@register_evaluator("test-gate-schema-error")
class _SchemaError(Evaluator):
    """Mimics JsonSchemaEvaluator reporting an unusable schema via metadata."""

    async def evaluate(self, value: Any, config: dict[str, Any]) -> EvaluatorResult:
        return EvaluatorResult(matched=False, message="bad schema", metadata={"schema_error": "invalid"})


def _custom_gate(evaluator: str) -> dict[str, Any]:
    return {"selector": "context.x", "evaluator": {"name": evaluator, "config": {}}, "precondition": True}


def _gate(field: str, expression: str, *, precondition: bool = True) -> dict[str, Any]:
    return {
        "selector": f"context.{field}",
        "evaluator": {"name": "cel", "config": {"expression": expression}},
        "precondition": precondition,
    }


def _control(gate: dict[str, Any], **extra: Any) -> Control:
    judge = {"selector": "output", "evaluator": {"name": "test-gate-always-match", "config": {}}}
    return Control.model_validate(
        {
            "name": "gated",
            "scope": {"stages": ["post"]},
            "condition": {"and": [gate, judge], "early_exit": True},
            "action": {"decision": "deny"},
            **extra,
        }
    )


async def _enforced(gate: dict[str, Any], context: dict[str, Any], **extra: Any) -> bool:
    step = Step(type="llm", name="n", output="hi", context=context)
    result = await ControlEngine([_control(gate, **extra)]).evaluate(step, stage="post")
    # The default on_error is fail_closed, so an engine exception also yields
    # "deny". Assert none happened so a pass means the gate itself decided.
    assert [m.error for m in result.matches if m.error] == []
    return result.action == "deny"


SMS = _gate("channel", 'value == "sms"')  # "when channel == sms"


def test_skip_on_missing_condition_defaults_to_false():
    assert _control(SMS).skip_on_missing_condition is False


class TestResolvedContext:
    """When the context is present the flag is irrelevant."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("skip", [False, True])
    async def test_condition_holds_enforces(self, skip):
        assert await _enforced(SMS, {"channel": "sms"}, skip_on_missing_condition=skip) is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("skip", [False, True])
    async def test_condition_fails_skips(self, skip):
        assert await _enforced(SMS, {"channel": "web"}, skip_on_missing_condition=skip) is False


class TestMissingContext:
    @pytest.mark.asyncio
    async def test_missing_field_enforces_by_default(self):
        assert await _enforced(SMS, {}) is True

    @pytest.mark.asyncio
    async def test_missing_field_skips_when_flag_set(self):
        assert await _enforced(SMS, {}, skip_on_missing_condition=True) is False

    @pytest.mark.asyncio
    async def test_nested_missing_field(self):
        gate = _gate("user.role", 'value == "admin"')
        assert await _enforced(gate, {"user": {}}) is True
        assert await _enforced(gate, {"user": {}}, skip_on_missing_condition=True) is False


class TestUnevaluableContext:
    @pytest.mark.asyncio
    async def test_cel_type_error_enforces_by_default(self):
        gate = _gate("amount", "value > 100")
        assert await _enforced(gate, {"amount": "abc"}) is True

    @pytest.mark.asyncio
    async def test_cel_type_error_skips_when_flag_set(self):
        gate = _gate("amount", "value > 100")
        assert await _enforced(gate, {"amount": "abc"}, skip_on_missing_condition=True) is False


class TestEvaluatorFailures:
    """Raised errors and error-metadata results count as unresolved context,
    not as the control's on_error policy or a plain non-match."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("evaluator", ["test-gate-raises", "test-gate-schema-error"])
    async def test_enforces_by_default(self, evaluator):
        assert await _enforced(_custom_gate(evaluator), {"x": 1}) is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("evaluator", ["test-gate-raises", "test-gate-schema-error"])
    async def test_skips_when_flag_set_even_if_fail_closed(self, evaluator):
        # action.on_error defaults to fail_closed; the flag must win for preconditions.
        assert await _enforced(_custom_gate(evaluator), {"x": 1}, skip_on_missing_condition=True) is False


class TestFlagOnlyAffectsPreconditions:
    @pytest.mark.asyncio
    async def test_plain_leaf_missing_is_still_a_non_match(self):
        """Without precondition: true, a missing selector is a non-match in both modes."""
        plain = _gate("channel", 'value == "sms"', precondition=False)
        assert await _enforced(plain, {}) is False
        assert await _enforced(plain, {}, skip_on_missing_condition=True) is False

    @pytest.mark.asyncio
    async def test_judge_leaf_missing_output_is_unaffected(self):
        """The flag must not turn a missing judge input into a match."""
        step = Step(type="llm", name="n", output=None, context={"channel": "sms"})
        step_missing = step.model_copy(update={"output": None})
        control = Control.model_validate(
            {
                "name": "c",
                "scope": {"stages": ["post"]},
                "condition": {"selector": "context.nope", "evaluator": {"name": "test-gate-always-match", "config": {}}},
                "action": {"decision": "deny"},
            }
        )
        result = await ControlEngine([control]).evaluate(step_missing, stage="post")
        assert result.action == "allow"


class TestComposites:
    @pytest.mark.asyncio
    async def test_or_gate_enforces_when_any_condition_holds(self):
        gate = {"or": [_gate("channel", 'value == "sms"'), _gate("amount", "value > 100")]}
        assert await _enforced(gate, {"amount": 500}, skip_on_missing_condition=True) is True
        assert await _enforced(gate, {"amount": 5, "channel": "web"}, skip_on_missing_condition=True) is False
