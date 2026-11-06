"""Contract tests for gating a control on ``context.<field>`` conditions.

Vijil Console compiles conditional policy rules (``unless user.role == admin``)
into controls shaped like::

    and: [ <gate>, <judge leaf> ]  (early_exit)

where the gate is, depending on its "skip on missing condition" setting:

* skip=True:   ``leaf(H)``            -- enforce only if the condition holds
* skip=False:  ``not(leaf(not H))``   -- enforce unless the condition is disproven

Both forms rely on two engine behaviors that are not otherwise pinned down:

1. A leaf whose selector does not resolve is a non-match (not an error).
2. A CEL evaluation error (e.g. comparing a string to a number) is a non-match.

If either changes, conditional policy rules would silently flip between
enforced and skipped, so they are asserted here.
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


def _cel_leaf(field: str, expression: str) -> dict[str, Any]:
    return {"selector": f"context.{field}", "evaluator": {"name": "cel", "config": {"expression": expression}}}


def _control(gate: dict[str, Any]) -> Control:
    judge = {"selector": "output", "evaluator": {"name": "test-gate-always-match", "config": {}}}
    return Control.model_validate(
        {
            "name": "gated",
            "scope": {"stages": ["post"]},
            "condition": {"and": [gate, judge], "early_exit": True},
            "action": {"decision": "deny"},
        }
    )


async def _enforced(gate: dict[str, Any], context: dict[str, Any]) -> bool:
    step = Step(type="llm", name="n", output="hi", context=context)
    result = await ControlEngine([_control(gate)]).evaluate(step, stage="post")
    return result.action == "deny"


def _skip_gate(holds: str, field: str) -> dict[str, Any]:
    return _cel_leaf(field, holds)


def _enforce_gate(holds: str, field: str) -> dict[str, Any]:
    return {"not": _cel_leaf(field, f"!({holds})")}


HOLDS = 'value == "sms"'  # "when channel == sms"


class TestSkipOnMissingGate:
    @pytest.mark.asyncio
    async def test_condition_holds_enforces(self):
        assert await _enforced(_skip_gate(HOLDS, "channel"), {"channel": "sms"}) is True

    @pytest.mark.asyncio
    async def test_condition_fails_skips(self):
        assert await _enforced(_skip_gate(HOLDS, "channel"), {"channel": "web"}) is False

    @pytest.mark.asyncio
    async def test_missing_field_skips(self):
        assert await _enforced(_skip_gate(HOLDS, "channel"), {}) is False

    @pytest.mark.asyncio
    async def test_cel_type_error_skips(self):
        assert await _enforced(_skip_gate("value > 100", "amount"), {"amount": "abc"}) is False


class TestEnforceOnMissingGate:
    @pytest.mark.asyncio
    async def test_condition_holds_enforces(self):
        assert await _enforced(_enforce_gate(HOLDS, "channel"), {"channel": "sms"}) is True

    @pytest.mark.asyncio
    async def test_condition_fails_skips(self):
        assert await _enforced(_enforce_gate(HOLDS, "channel"), {"channel": "web"}) is False

    @pytest.mark.asyncio
    async def test_missing_field_enforces(self):
        assert await _enforced(_enforce_gate(HOLDS, "channel"), {}) is True

    @pytest.mark.asyncio
    async def test_nested_missing_field_enforces(self):
        assert await _enforced(_enforce_gate('value == "admin"', "user.role"), {"user": {}}) is True

    @pytest.mark.asyncio
    async def test_cel_type_error_enforces(self):
        assert await _enforced(_enforce_gate("value > 100", "amount"), {"amount": "abc"}) is True
