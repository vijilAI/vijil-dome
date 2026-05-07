"""Tests for control data models."""

import pytest
from pydantic import ValidationError

from vijil_dome.controls.models import (
    ConditionNode,
    Control,
    ControlAction,
    ControlMatch,
    ControlScope,
    EvaluationResult,
    EvaluatorRef,
    Step,
    SteeringContext,
)


class TestStep:
    def test_minimal(self):
        s = Step(type="llm", name="chat")
        assert s.type == "llm"
        assert s.name == "chat"
        assert s.input is None
        assert s.output is None
        assert s.context == {}

    def test_full(self):
        s = Step(
            type="tool",
            name="search",
            input={"query": "test"},
            output={"results": []},
            context={"user_role": "admin"},
        )
        assert s.type == "tool"
        assert s.input == {"query": "test"}
        assert s.context["user_role"] == "admin"


class TestControlScope:
    def test_defaults(self):
        scope = ControlScope()
        assert scope.step_types is None
        assert scope.step_names is None
        assert scope.step_name_regex is None
        assert scope.stages is None

    def test_specific(self):
        scope = ControlScope(
            step_types=["tool"],
            step_names=["transfer_funds"],
            stages=["pre"],
        )
        assert scope.step_types == ["tool"]
        assert scope.step_names == ["transfer_funds"]


class TestConditionNode:
    def test_leaf(self):
        node = ConditionNode(
            selector="input.text",
            evaluator=EvaluatorRef(name="regex", config={"pattern": ".*"}),
        )
        assert node.is_leaf()
        assert not node.is_composite()

    def test_and_composite(self):
        node = ConditionNode(
            and_=[
                ConditionNode(
                    selector="input",
                    evaluator=EvaluatorRef(name="regex", config={"pattern": "a"}),
                ),
                ConditionNode(
                    selector="input",
                    evaluator=EvaluatorRef(name="regex", config={"pattern": "b"}),
                ),
            ]
        )
        assert node.is_composite()
        assert not node.is_leaf()
        assert len(node.and_) == 2

    def test_not_composite(self):
        node = ConditionNode(
            not_=ConditionNode(
                selector="context.role",
                evaluator=EvaluatorRef(name="list", config={"values": ["admin"]}),
            )
        )
        assert node.is_composite()
        assert node.not_ is not None

    def test_mutual_exclusivity_rejects_leaf_and_composite(self):
        with pytest.raises(ValidationError, match="cannot have both"):
            ConditionNode(
                selector="input",
                evaluator=EvaluatorRef(name="regex", config={"pattern": ".*"}),
                and_=[
                    ConditionNode(
                        selector="input",
                        evaluator=EvaluatorRef(name="regex", config={"pattern": "x"}),
                    )
                ],
            )

    def test_empty_and_rejected(self):
        with pytest.raises(ValidationError, match="at least one"):
            ConditionNode.model_validate({"and": []})

    def test_empty_or_rejected(self):
        with pytest.raises(ValidationError, match="at least one"):
            ConditionNode.model_validate({"or": []})

    def test_from_dict_with_aliases(self):
        data = {
            "and": [
                {"selector": "input", "evaluator": {"name": "regex", "config": {"pattern": "x"}}},
                {
                    "not": {
                        "selector": "context.role",
                        "evaluator": {"name": "list", "config": {"values": ["admin"]}},
                    }
                },
            ]
        }
        node = ConditionNode.model_validate(data)
        assert node.and_ is not None
        assert len(node.and_) == 2
        assert node.and_[1].not_ is not None


class TestControl:
    def test_from_dict(self):
        data = {
            "name": "block-ssn",
            "scope": {"stages": ["post"]},
            "condition": {
                "selector": "output",
                "evaluator": {"name": "regex", "config": {"pattern": r"\d{3}-\d{2}-\d{4}"}},
            },
            "action": {"decision": "deny", "message": "SSN detected"},
        }
        ctrl = Control.model_validate(data)
        assert ctrl.name == "block-ssn"
        assert ctrl.enabled is True
        assert ctrl.priority == 100
        assert ctrl.action.decision == "deny"
        assert ctrl.condition.is_leaf()

    def test_steer_action(self):
        ctrl = Control(
            name="require-2fa",
            condition=ConditionNode(
                selector="input.amount",
                evaluator=EvaluatorRef(name="cel", config={"expression": "value > 10000"}),
            ),
            action=ControlAction(
                decision="steer",
                steering_context=SteeringContext(message="Require 2FA"),
            ),
        )
        assert ctrl.action.decision == "steer"
        assert ctrl.action.steering_context.message == "Require 2FA"


class TestControlAnnotations:
    def test_annotations_roundtrip(self):
        data = {
            "name": "annotated",
            "condition": {
                "selector": "input",
                "evaluator": {"name": "regex", "config": {"pattern": ".*"}},
            },
            "action": {"decision": "deny"},
            "annotations": {
                "vijil.ai/trust-delta": {"dimension": "security", "delta": 0.15},
                "galileo.com/scoring": "strict",
            },
        }
        ctrl = Control.model_validate(data)
        assert ctrl.annotations["vijil.ai/trust-delta"]["delta"] == 0.15
        assert ctrl.annotations["galileo.com/scoring"] == "strict"

        dumped = ctrl.model_dump()
        assert dumped["annotations"] == data["annotations"]

    def test_annotations_default_empty(self):
        ctrl = Control(
            name="no-ann",
            condition=ConditionNode(
                selector="input",
                evaluator=EvaluatorRef(name="regex", config={"pattern": "x"}),
            ),
            action=ControlAction(decision="deny"),
        )
        assert ctrl.annotations == {}

    def test_annotations_structured_values(self):
        ctrl = Control.model_validate({
            "name": "structured",
            "condition": {
                "selector": "input",
                "evaluator": {"name": "regex", "config": {"pattern": "x"}},
            },
            "action": {"decision": "deny"},
            "annotations": {
                "vendor/nested": {"list": [1, 2, 3], "flag": True, "score": 0.95},
            },
        })
        ann = ctrl.annotations["vendor/nested"]
        assert ann["list"] == [1, 2, 3]
        assert ann["flag"] is True
        assert ann["score"] == 0.95

    def test_unknown_fields_preserved(self):
        """With extra='allow', unknown top-level fields survive parsing."""
        data = {
            "name": "with-extras",
            "condition": {
                "selector": "input",
                "evaluator": {"name": "regex", "config": {"pattern": "x"}},
            },
            "action": {"decision": "deny"},
            "custom_vendor_field": "some-value",
        }
        ctrl = Control.model_validate(data)
        assert ctrl.model_extra.get("custom_vendor_field") == "some-value"


class TestEvaluationResult:
    def test_defaults(self):
        r = EvaluationResult()
        assert r.permitted is True
        assert r.action == "allow"
        assert r.matches == []

    def test_denied(self):
        r = EvaluationResult(
            action="deny",
            matches=[
                ControlMatch(
                    control_name="test",
                    triggered=True,
                    action=ControlAction(decision="deny"),
                )
            ],
        )
        assert r.permitted is False
        assert r.action == "deny"

    def test_permitted_computed_from_action(self):
        assert EvaluationResult(action="allow").permitted is True
        assert EvaluationResult(action="steer").permitted is True
        assert EvaluationResult(action="deny").permitted is False
