"""Tests for the control engine."""

from __future__ import annotations

from typing import Literal

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
    type_: Literal["tool", "llm"] = "llm",
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
    on_error: Literal["fail_open", "fail_closed"] = "fail_closed",
) -> Control:
    return Control(
        name=name,
        scope=ControlScope.model_validate(scope or {}),
        condition=ConditionNode.model_validate(
            {
                "selector": selector,
                "evaluator": {
                    "name": evaluator_name,
                    "config": evaluator_config or {"pattern": ".*"},
                },
            }
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
        assert len(triggered) == 1

    @pytest.mark.asyncio
    async def test_slow_evaluator_cancelled(self):
        """Verify a slow deny control is actually cancelled by a fast one."""
        import asyncio
        from vijil_dome.controls.evaluators import register_evaluator
        from vijil_dome.controls.evaluators.base import Evaluator, EvaluatorResult

        slow_completed = False

        @register_evaluator("_test_slow")
        class SlowEvaluator(Evaluator):
            async def evaluate(self, value, config):
                nonlocal slow_completed
                await asyncio.sleep(5)
                slow_completed = True
                return EvaluatorResult(matched=True)

        fast = _deny_control(name="fast", priority=1)
        slow = _deny_control(
            name="slow", priority=2,
            evaluator_name="_test_slow",
            evaluator_config={},
        )
        engine = ControlEngine([fast, slow])

        result = await engine.evaluate(_step(input_="x"), stage="pre")
        assert result.permitted is False
        assert not slow_completed


# ------------------------------------------------------------------
# Early exit (short-circuit)
# ------------------------------------------------------------------


class TestEarlyExit:
    @pytest.mark.asyncio
    async def test_and_early_exit_stops_on_first_false(self):
        """With early_exit, and_ should not evaluate remaining children after False."""
        from vijil_dome.controls.evaluators import register_evaluator
        from vijil_dome.controls.evaluators.base import Evaluator, EvaluatorResult

        second_called = False

        @register_evaluator("_test_track_call")
        class TrackCallEvaluator(Evaluator):
            async def evaluate(self, value, config):
                nonlocal second_called
                second_called = True
                return EvaluatorResult(matched=True)

        ctrl = Control(
            name="and-early",
            condition=ConditionNode.model_validate({
                "and": [
                    {"selector": "input", "evaluator": {"name": "regex", "config": {"pattern": "NOMATCH"}}},
                    {"selector": "input", "evaluator": {"name": "_test_track_call", "config": {}}},
                ],
                "early_exit": True,
            }),
            action=ControlAction(decision="deny"),
        )
        engine = ControlEngine([ctrl])
        result = await engine.evaluate(_step(input_="hello"), stage="pre")
        assert result.action == "allow"
        assert not second_called

    @pytest.mark.asyncio
    async def test_or_early_exit_stops_on_first_true(self):
        """With early_exit, or_ should not evaluate remaining children after True."""
        from vijil_dome.controls.evaluators import register_evaluator
        from vijil_dome.controls.evaluators.base import Evaluator, EvaluatorResult

        second_called = False

        @register_evaluator("_test_track_call_2")
        class TrackCallEvaluator2(Evaluator):
            async def evaluate(self, value, config):
                nonlocal second_called
                second_called = True
                return EvaluatorResult(matched=True)

        ctrl = Control(
            name="or-early",
            condition=ConditionNode.model_validate({
                "or": [
                    {"selector": "input", "evaluator": {"name": "regex", "config": {"pattern": ".*"}}},
                    {"selector": "input", "evaluator": {"name": "_test_track_call_2", "config": {}}},
                ],
                "early_exit": True,
            }),
            action=ControlAction(decision="deny"),
        )
        engine = ControlEngine([ctrl])
        result = await engine.evaluate(_step(input_="hello"), stage="pre")
        assert result.action == "deny"
        assert not second_called

    @pytest.mark.asyncio
    async def test_and_without_early_exit_runs_all(self):
        """Default (no early_exit) evaluates all children in parallel."""
        ctrl = Control(
            name="and-parallel",
            condition=ConditionNode.model_validate({
                "and": [
                    {"selector": "input", "evaluator": {"name": "regex", "config": {"pattern": "hello"}}},
                    {"selector": "input", "evaluator": {"name": "regex", "config": {"pattern": "hello"}}},
                ],
            }),
            action=ControlAction(decision="deny"),
        )
        engine = ControlEngine([ctrl])
        result = await engine.evaluate(_step(input_="hello"), stage="pre")
        assert result.action == "deny"

    @pytest.mark.asyncio
    async def test_early_exit_from_yaml_style_dict(self):
        """early_exit can be set in a policy dict (YAML/JSON friendly)."""
        ctrl = Control.model_validate({
            "name": "policy-early",
            "condition": {
                "and": [
                    {"selector": "input", "evaluator": {"name": "regex", "config": {"pattern": ".*"}}},
                ],
                "early_exit": True,
            },
            "action": {"decision": "deny"},
        })
        assert ctrl.condition.early_exit is True


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


# ------------------------------------------------------------------
# ReDoS protection (step_name_regex)
# ------------------------------------------------------------------


class TestReDoSProtection:
    @pytest.mark.asyncio
    async def test_step_name_regex_matching(self):
        ctrl = Control(
            name="regex-scope",
            scope=ControlScope(step_name_regex=r"^chat_.*"),
            condition=ConditionNode(
                selector="input",
                evaluator=EvaluatorRef(name="regex", config={"pattern": ".*"}),
            ),
            action=ControlAction(decision="deny"),
        )
        engine = ControlEngine([ctrl])

        result = await engine.evaluate(_step(name="chat_v2", input_="x"), stage="pre")
        assert result.action == "deny"

        result = await engine.evaluate(_step(name="search", input_="x"), stage="pre")
        assert result.action == "allow"

    @pytest.mark.asyncio
    async def test_re2_fallback_warning(self, caplog, monkeypatch):
        import vijil_dome.controls.engine as eng_mod

        monkeypatch.setattr(eng_mod, "_SCOPE_HAS_RE2", False)

        ctrl = Control(
            name="regex-scope",
            scope=ControlScope(step_name_regex=r"^chat_.*"),
            condition=ConditionNode(
                selector="input",
                evaluator=EvaluatorRef(name="regex", config={"pattern": ".*"}),
            ),
            action=ControlAction(decision="deny"),
        )
        engine = ControlEngine([ctrl])

        import logging
        with caplog.at_level(logging.WARNING, logger="vijil_dome.controls.engine"):
            result = await engine.evaluate(
                _step(name="chat_v2", input_="x"), stage="pre"
            )

        assert result.action == "deny"
        assert any("re2 not installed" in msg for msg in caplog.messages)


# ------------------------------------------------------------------
# Thread safety
# ------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_add_and_evaluate(self):
        import asyncio
        import threading

        engine = ControlEngine([_deny_control(name="initial")])
        errors: list[Exception] = []

        def add_controls():
            try:
                for i in range(50):
                    engine.add_control(_deny_control(name=f"added-{i}"))
            except Exception as exc:
                errors.append(exc)

        def run_evaluates():
            try:
                loop = asyncio.new_event_loop()
                for _ in range(50):
                    loop.run_until_complete(
                        engine.evaluate(_step(input_="x"), stage="pre")
                    )
                loop.close()
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=add_controls)
        t2 = threading.Thread(target=run_evaluates)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert errors == []
