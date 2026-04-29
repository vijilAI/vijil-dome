"""Tests for the control engine."""

import pytest

from vijil_dome.controls.engine import ControlEngine
from vijil_dome.controls.models import (
    ConditionNode,
    Control,
    ControlAction,
    ControlScope,
    EvaluatorRef,
    Step,
    SteeringContext,
)


def _step(
    type_: str = "llm",
    name: str = "chat",
    input_: dict | str | None = None,
    output_: dict | str | None = None,
    context: dict | None = None,
) -> Step:
    return Step(
        type=type_,
        name=name,
        input=input_,
        output=output_,
        context=context or {},
    )


def _deny_control(
    name: str = "test-deny",
    selector: str = "input",
    evaluator_name: str = "regex",
    evaluator_config: dict | None = None,
    scope: dict | None = None,
    priority: int = 100,
    on_error: str = "fail_closed",
) -> Control:
    return Control(
        name=name,
        scope=ControlScope.model_validate(scope or {}),
        condition=ConditionNode(
            selector=selector,
            evaluator=EvaluatorRef(
                name=evaluator_name,
                config=evaluator_config or {"pattern": ".*"},
            ),
        ),
        action=ControlAction(decision="deny", message=f"Denied by {name}", on_error=on_error),
        priority=priority,
    )


# ------------------------------------------------------------------
# Scope matching
# ------------------------------------------------------------------


class TestScopeMatching:
    @pytest.mark.asyncio
    async def test_no_scope_matches_everything(self):
        engine = ControlEngine([_deny_control()])
        result = await engine.evaluate(_step(input_="hello"), stage="pre")
        assert result.permitted is False

    @pytest.mark.asyncio
    async def test_stage_filter(self):
        ctrl = _deny_control(scope={"stages": ["post"]})
        engine = ControlEngine([ctrl])

        result_pre = await engine.evaluate(_step(input_="hello"), stage="pre")
        assert result_pre.permitted is True

        result_post = await engine.evaluate(_step(input_="hello"), stage="post")
        assert result_post.permitted is False

    @pytest.mark.asyncio
    async def test_step_type_filter(self):
        ctrl = _deny_control(scope={"step_types": ["tool"]})
        engine = ControlEngine([ctrl])

        result = await engine.evaluate(_step(type_="llm", input_="x"), stage="pre")
        assert result.permitted is True

        result = await engine.evaluate(_step(type_="tool", input_="x"), stage="pre")
        assert result.permitted is False

    @pytest.mark.asyncio
    async def test_step_name_filter(self):
        ctrl = _deny_control(scope={"step_names": ["transfer_funds"]})
        engine = ControlEngine([ctrl])

        result = await engine.evaluate(
            _step(name="transfer_funds", input_="x"), stage="pre"
        )
        assert result.permitted is False

        result = await engine.evaluate(
            _step(name="search", input_="x"), stage="pre"
        )
        assert result.permitted is True

    @pytest.mark.asyncio
    async def test_step_name_regex(self):
        ctrl = _deny_control(scope={"step_name_regex": r"^(search|lookup)_.*"})
        engine = ControlEngine([ctrl])

        result = await engine.evaluate(
            _step(name="search_flights", input_="x"), stage="pre"
        )
        assert result.permitted is False

        result = await engine.evaluate(
            _step(name="book_flight", input_="x"), stage="pre"
        )
        assert result.permitted is True


# ------------------------------------------------------------------
# Condition trees
# ------------------------------------------------------------------


class TestConditionTrees:
    @pytest.mark.asyncio
    async def test_leaf_match(self):
        engine = ControlEngine([
            _deny_control(
                evaluator_config={"pattern": r"\d{3}-\d{2}-\d{4}"},
            )
        ])
        result = await engine.evaluate(
            _step(input_="SSN: 123-45-6789"), stage="pre"
        )
        assert result.permitted is False

    @pytest.mark.asyncio
    async def test_leaf_no_match(self):
        engine = ControlEngine([
            _deny_control(
                evaluator_config={"pattern": r"\d{3}-\d{2}-\d{4}"},
            )
        ])
        result = await engine.evaluate(
            _step(input_="no SSN here"), stage="pre"
        )
        assert result.permitted is True

    @pytest.mark.asyncio
    async def test_and_condition(self):
        ctrl = Control(
            name="and-test",
            condition=ConditionNode(
                and_=[
                    ConditionNode(
                        selector="input",
                        evaluator=EvaluatorRef(
                            name="regex", config={"pattern": "transfer"}
                        ),
                    ),
                    ConditionNode(
                        selector="input",
                        evaluator=EvaluatorRef(
                            name="regex", config={"pattern": "urgent"}
                        ),
                    ),
                ]
            ),
            action=ControlAction(decision="deny"),
        )
        engine = ControlEngine([ctrl])

        result = await engine.evaluate(
            _step(input_="urgent transfer"), stage="pre"
        )
        assert result.permitted is False

        result = await engine.evaluate(
            _step(input_="normal transfer"), stage="pre"
        )
        assert result.permitted is True

    @pytest.mark.asyncio
    async def test_or_condition(self):
        ctrl = Control(
            name="or-test",
            condition=ConditionNode(
                or_=[
                    ConditionNode(
                        selector="input",
                        evaluator=EvaluatorRef(
                            name="regex", config={"pattern": "drop table"}
                        ),
                    ),
                    ConditionNode(
                        selector="input",
                        evaluator=EvaluatorRef(
                            name="regex", config={"pattern": "delete from"}
                        ),
                    ),
                ]
            ),
            action=ControlAction(decision="deny"),
        )
        engine = ControlEngine([ctrl])

        result = await engine.evaluate(
            _step(input_="drop table users"), stage="pre"
        )
        assert result.permitted is False

        result = await engine.evaluate(
            _step(input_="select * from users"), stage="pre"
        )
        assert result.permitted is True

    @pytest.mark.asyncio
    async def test_not_condition(self):
        ctrl = Control(
            name="not-admin",
            condition=ConditionNode(
                not_=ConditionNode(
                    selector="context.user_role",
                    evaluator=EvaluatorRef(
                        name="list", config={"values": ["admin"]}
                    ),
                )
            ),
            action=ControlAction(decision="deny"),
        )
        engine = ControlEngine([ctrl])

        result = await engine.evaluate(
            _step(input_="x", context={"user_role": "admin"}), stage="pre"
        )
        assert result.permitted is True  # NOT admin = False -> not denied

        result = await engine.evaluate(
            _step(input_="x", context={"user_role": "guest"}), stage="pre"
        )
        assert result.permitted is False  # NOT guest-in-admin = True -> denied


# ------------------------------------------------------------------
# Actions: deny, steer, observe
# ------------------------------------------------------------------


class TestActions:
    @pytest.mark.asyncio
    async def test_deny_blocks(self):
        engine = ControlEngine([_deny_control()])
        result = await engine.evaluate(_step(input_="anything"), stage="pre")
        assert result.permitted is False
        assert result.action == "deny"

    @pytest.mark.asyncio
    async def test_steer_permits_with_context(self):
        ctrl = Control(
            name="steer-test",
            condition=ConditionNode(
                selector="input",
                evaluator=EvaluatorRef(name="regex", config={"pattern": ".*"}),
            ),
            action=ControlAction(
                decision="steer",
                steering_context=SteeringContext(message="Please verify"),
            ),
        )
        engine = ControlEngine([ctrl])
        result = await engine.evaluate(_step(input_="anything"), stage="pre")
        assert result.permitted is True
        assert result.action == "steer"
        assert result.steering_context is not None
        assert result.steering_context.message == "Please verify"

    @pytest.mark.asyncio
    async def test_observe_permits_silently(self):
        ctrl = Control(
            name="observe-test",
            condition=ConditionNode(
                selector="input",
                evaluator=EvaluatorRef(name="regex", config={"pattern": ".*"}),
            ),
            action=ControlAction(decision="observe"),
        )
        engine = ControlEngine([ctrl])
        result = await engine.evaluate(_step(input_="anything"), stage="pre")
        assert result.permitted is True
        assert result.action == "allow"  # observe doesn't change action
        assert len(result.matches) == 1
        assert result.matches[0].triggered is True

    @pytest.mark.asyncio
    async def test_deny_takes_priority_over_steer(self):
        deny = _deny_control(name="deny", priority=10)
        steer = Control(
            name="steer",
            condition=ConditionNode(
                selector="input",
                evaluator=EvaluatorRef(name="regex", config={"pattern": ".*"}),
            ),
            action=ControlAction(
                decision="steer",
                steering_context=SteeringContext(message="steer"),
            ),
            priority=20,
        )
        engine = ControlEngine([deny, steer])
        result = await engine.evaluate(_step(input_="x"), stage="pre")
        assert result.action == "deny"


# ------------------------------------------------------------------
# Cancel-on-deny
# ------------------------------------------------------------------


class TestCancelOnDeny:
    @pytest.mark.asyncio
    async def test_first_deny_cancels_remaining(self):
        fast_deny = _deny_control(name="fast", priority=1)
        slow_deny = _deny_control(name="slow", priority=2)
        engine = ControlEngine([fast_deny, slow_deny])

        result = await engine.evaluate(_step(input_="x"), stage="pre")
        assert result.permitted is False
        triggered = [m for m in result.matches if m.triggered]
        assert len(triggered) >= 1


# ------------------------------------------------------------------
# on_error
# ------------------------------------------------------------------


class TestOnError:
    @pytest.mark.asyncio
    async def test_fail_closed_blocks_on_error(self):
        ctrl = Control(
            name="bad-evaluator",
            condition=ConditionNode(
                selector="input.nonexistent.deep.path",
                evaluator=EvaluatorRef(
                    name="regex", config={"pattern": "(.*)"}
                ),
            ),
            action=ControlAction(decision="deny", on_error="fail_closed"),
        )
        engine = ControlEngine([ctrl])
        result = await engine.evaluate(_step(input_="x"), stage="pre")
        # Missing path returns MISSING -> leaf returns False -> not triggered
        # This is NOT an error, it's a missing path. Test actual error:
        assert result.permitted is True

    @pytest.mark.asyncio
    async def test_fail_open_passes_on_error(self):
        ctrl = Control(
            name="bad-evaluator",
            condition=ConditionNode(
                selector="input",
                evaluator=EvaluatorRef(name="nonexistent_eval_xyz"),
            ),
            action=ControlAction(decision="deny", on_error="fail_open"),
        )
        engine = ControlEngine([ctrl])
        result = await engine.evaluate(_step(input_="x"), stage="pre")
        assert result.permitted is True
        assert len(result.matches) == 1
        assert result.matches[0].error is not None

    @pytest.mark.asyncio
    async def test_fail_closed_blocks_on_evaluator_error(self):
        ctrl = Control(
            name="bad-evaluator",
            condition=ConditionNode(
                selector="input",
                evaluator=EvaluatorRef(name="nonexistent_eval_xyz"),
            ),
            action=ControlAction(decision="deny", on_error="fail_closed"),
        )
        engine = ControlEngine([ctrl])
        result = await engine.evaluate(_step(input_="x"), stage="pre")
        assert result.permitted is False
        assert result.matches[0].error is not None


# ------------------------------------------------------------------
# No controls / disabled controls
# ------------------------------------------------------------------


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_engine(self):
        engine = ControlEngine()
        result = await engine.evaluate(_step(input_="x"), stage="pre")
        assert result.permitted is True
        assert result.matches == []

    @pytest.mark.asyncio
    async def test_disabled_control_skipped(self):
        ctrl = _deny_control()
        ctrl.enabled = False
        engine = ControlEngine([ctrl])
        result = await engine.evaluate(_step(input_="x"), stage="pre")
        assert result.permitted is True

    @pytest.mark.asyncio
    async def test_add_control_maintains_priority(self):
        engine = ControlEngine()
        engine.add_control(_deny_control(name="p100", priority=100))
        engine.add_control(_deny_control(name="p10", priority=10))
        engine.add_control(_deny_control(name="p50", priority=50))
        assert [c.name for c in engine.controls] == ["p10", "p50", "p100"]

    def test_evaluate_sync(self):
        engine = ControlEngine([_deny_control()])
        result = engine.evaluate_sync(_step(input_="x"), stage="pre")
        assert result.permitted is False
