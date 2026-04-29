"""Tests for the @control() decorator."""

import pytest

from vijil_dome.controls.decorator import control, _extract_input, _safe_bind
from vijil_dome.controls.engine import ControlEngine
from vijil_dome.controls.errors import ControlSteerError, ControlViolationError
from vijil_dome.controls.models import (
    ConditionNode,
    Control,
    ControlAction,
    EvaluatorRef,
    SteeringContext,
)


def _deny_on_pattern(
    pattern: str = ".*",
    stage: str | None = None,
    selector: str = "input",
) -> Control:
    scope = {"stages": [stage]} if stage else {}
    return Control(
        name="test-deny",
        scope=scope,
        condition=ConditionNode(
            selector=selector,
            evaluator=EvaluatorRef(name="regex", config={"pattern": pattern}),
        ),
        action=ControlAction(decision="deny", message="Blocked"),
    )


def _steer_on_pattern(pattern: str = ".*") -> Control:
    return Control(
        name="test-steer",
        condition=ConditionNode(
            selector="input",
            evaluator=EvaluatorRef(name="regex", config={"pattern": pattern}),
        ),
        action=ControlAction(
            decision="steer",
            steering_context=SteeringContext(message="Please rephrase"),
        ),
    )


# ------------------------------------------------------------------
# Core decorator behavior
# ------------------------------------------------------------------


class TestControlDecoratorAsync:
    @pytest.mark.asyncio
    async def test_deny_raises(self):
        engine = ControlEngine([_deny_on_pattern()])

        @control(engine=engine)
        async def chat(msg: str) -> str:
            return f"echo: {msg}"

        with pytest.raises(ControlViolationError):
            await chat("anything")

    @pytest.mark.asyncio
    async def test_allow_passes(self):
        engine = ControlEngine([_deny_on_pattern(r"BLOCKED")])

        @control(engine=engine)
        async def chat(msg: str) -> str:
            return f"echo: {msg}"

        result = await chat("hello")
        assert result == "echo: hello"

    @pytest.mark.asyncio
    async def test_steer_raises(self):
        engine = ControlEngine([_steer_on_pattern()])

        @control(engine=engine)
        async def chat(msg: str) -> str:
            return f"echo: {msg}"

        with pytest.raises(ControlSteerError) as exc_info:
            await chat("anything")
        assert exc_info.value.steering_context.message == "Please rephrase"

    @pytest.mark.asyncio
    async def test_shadow_mode_logs_but_passes(self):
        engine = ControlEngine([_deny_on_pattern()])

        @control(engine=engine, enforce=False)
        async def chat(msg: str) -> str:
            return f"echo: {msg}"

        result = await chat("anything")
        assert result == "echo: anything"

    @pytest.mark.asyncio
    async def test_post_stage_deny(self):
        engine = ControlEngine([
            _deny_on_pattern(r"secret", stage="post", selector="output")
        ])

        @control(engine=engine)
        async def chat(msg: str) -> str:
            return "here is a secret"

        with pytest.raises(ControlViolationError):
            await chat("tell me")

    @pytest.mark.asyncio
    async def test_post_stage_allow(self):
        engine = ControlEngine([
            _deny_on_pattern(r"secret", stage="post", selector="output")
        ])

        @control(engine=engine)
        async def chat(msg: str) -> str:
            return "nothing here"

        result = await chat("tell me")
        assert result == "nothing here"


class TestControlDecoratorSync:
    def test_deny_raises(self):
        engine = ControlEngine([_deny_on_pattern()])

        @control(engine=engine)
        def chat(msg: str) -> str:
            return f"echo: {msg}"

        with pytest.raises(ControlViolationError):
            chat("anything")

    def test_allow_passes(self):
        engine = ControlEngine([_deny_on_pattern(r"BLOCKED")])

        @control(engine=engine)
        def chat(msg: str) -> str:
            return f"echo: {msg}"

        result = chat("hello")
        assert result == "echo: hello"


class TestControlDecoratorWithPolicy:
    @pytest.mark.asyncio
    async def test_policy_from_list_of_dicts(self):
        policy = [
            {
                "name": "block-ssn",
                "condition": {
                    "selector": "input",
                    "evaluator": {
                        "name": "regex",
                        "config": {"pattern": r"\d{3}-\d{2}-\d{4}"},
                    },
                },
                "action": {"decision": "deny", "message": "SSN detected"},
            }
        ]

        @control(policy=policy)
        async def chat(msg: str) -> str:
            return f"echo: {msg}"

        with pytest.raises(ControlViolationError):
            await chat("My SSN is 123-45-6789")

        result = await chat("No PII here")
        assert result == "echo: No PII here"

    @pytest.mark.asyncio
    async def test_step_name_defaults_to_func_name(self):
        engine = ControlEngine([
            Control(
                name="name-check",
                scope={"step_names": ["my_function"]},
                condition=ConditionNode(
                    selector="input",
                    evaluator=EvaluatorRef(
                        name="regex", config={"pattern": ".*"}
                    ),
                ),
                action=ControlAction(decision="deny"),
            )
        ])

        @control(engine=engine)
        async def my_function(msg: str) -> str:
            return msg

        with pytest.raises(ControlViolationError):
            await my_function("test")

    @pytest.mark.asyncio
    async def test_custom_step_name(self):
        engine = ControlEngine([
            Control(
                name="name-check",
                scope={"step_names": ["custom_name"]},
                condition=ConditionNode(
                    selector="input",
                    evaluator=EvaluatorRef(
                        name="regex", config={"pattern": ".*"}
                    ),
                ),
                action=ControlAction(decision="deny"),
            )
        ])

        @control(engine=engine, step_name="custom_name")
        async def my_function(msg: str) -> str:
            return msg

        with pytest.raises(ControlViolationError):
            await my_function("test")


# ------------------------------------------------------------------
# Input extraction heuristics
# ------------------------------------------------------------------


class TestInputExtraction:
    def test_single_param_returns_value_directly(self):
        def fn(msg: str): ...
        result = _extract_input(fn, ("hello",), {}, "llm")
        assert result == "hello"

    def test_well_known_name_message(self):
        def fn(message: str, temperature: float = 0.7): ...
        result = _extract_input(fn, ("hello",), {}, "llm")
        assert result == "hello"

    def test_well_known_name_query(self):
        def fn(query: str, limit: int = 10): ...
        result = _extract_input(fn, ("search term",), {}, "llm")
        assert result == "search term"

    def test_well_known_name_prompt(self):
        def fn(prompt: str, model: str = "gpt-4"): ...
        result = _extract_input(fn, ("tell me a joke",), {}, "llm")
        assert result == "tell me a joke"

    def test_well_known_name_input(self):
        def fn(input: str, ctx: dict = None): ...
        result = _extract_input(fn, ("data",), {}, "llm")
        assert result == "data"

    def test_first_string_fallback(self):
        def fn(count: int, payload: str): ...
        result = _extract_input(fn, (5, "the text"), {}, "llm")
        assert result == "the text"

    def test_multiple_params_no_known_name_returns_dict(self):
        def fn(count: int, limit: int): ...
        result = _extract_input(fn, (5, 10), {}, "llm")
        assert result == {"count": 5, "limit": 10}

    def test_tool_always_returns_dict(self):
        def fn(query: str, limit: int = 10): ...
        result = _extract_input(fn, ("search",), {}, "tool")
        assert result == {"query": "search", "limit": 10}

    def test_tool_single_param_still_dict(self):
        def fn(url: str): ...
        result = _extract_input(fn, ("https://example.com",), {}, "tool")
        assert result == {"url": "https://example.com"}

    def test_self_is_filtered(self):
        class MyAgent:
            def chat(self, message: str):
                ...

        agent = MyAgent()
        result = _extract_input(
            MyAgent.chat, (agent, "hello"), {}, "llm"
        )
        assert result == "hello"

    def test_self_filtered_from_tool(self):
        class MyAgent:
            def search(self, query: str, limit: int = 5):
                ...

        agent = MyAgent()
        result = _extract_input(
            MyAgent.search, (agent, "test"), {}, "tool"
        )
        assert result == {"query": "test", "limit": 5}
        assert "self" not in result

    def test_none_well_known_skipped(self):
        def fn(message: str | None = None, fallback: str = "default"): ...
        result = _extract_input(fn, (), {}, "llm")
        assert result == "default"

    def test_kwargs_work(self):
        def fn(message: str, temperature: float = 0.7): ...
        result = _extract_input(fn, (), {"message": "hi"}, "llm")
        assert result == "hi"

    def test_stringify_fallback_no_params(self):
        def fn(): ...
        result = _extract_input(fn, (), {}, "llm")
        assert result == ""


# ------------------------------------------------------------------
# Explicit mappers
# ------------------------------------------------------------------


class TestExplicitMappers:
    @pytest.mark.asyncio
    async def test_input_mapper(self):
        engine = ControlEngine([_deny_on_pattern(r"BLOCKED")])

        @control(
            engine=engine,
            input_mapper=lambda msg, **kw: msg.upper(),
        )
        async def chat(msg: str) -> str:
            return f"echo: {msg}"

        # "blocked" uppercased to "BLOCKED" by mapper, should trigger deny
        with pytest.raises(ControlViolationError):
            await chat("blocked")

    @pytest.mark.asyncio
    async def test_input_mapper_extracts_nested(self):
        engine = ControlEngine([_deny_on_pattern(r"secret")])

        @control(
            engine=engine,
            input_mapper=lambda request: request["body"]["text"],
        )
        async def process(request: dict) -> str:
            return "done"

        with pytest.raises(ControlViolationError):
            await process({"body": {"text": "this is a secret"}})

        result = await process({"body": {"text": "safe content"}})
        assert result == "done"

    @pytest.mark.asyncio
    async def test_output_mapper(self):
        engine = ControlEngine([
            _deny_on_pattern(r"SENSITIVE", stage="post", selector="output")
        ])

        @control(
            engine=engine,
            output_mapper=lambda out: out.get("text", ""),
        )
        async def generate(msg: str) -> dict:
            return {"text": "SENSITIVE data", "metadata": {}}

        with pytest.raises(ControlViolationError):
            await generate("tell me")

    @pytest.mark.asyncio
    async def test_context_mapper(self):
        # Control that denies non-admin users
        ctrl = Control(
            name="admin-only",
            condition=ConditionNode(
                not_=ConditionNode(
                    selector="context.user_role",
                    evaluator=EvaluatorRef(
                        name="list", config={"values": ["admin"]}
                    ),
                )
            ),
            action=ControlAction(decision="deny", message="Admin only"),
        )
        engine = ControlEngine([ctrl])

        @control(
            engine=engine,
            context_mapper=lambda msg, user=None: {"user_role": user or "guest"},
        )
        async def admin_action(msg: str, user: str = None) -> str:
            return "done"

        # Guest should be denied
        with pytest.raises(ControlViolationError):
            await admin_action("do thing", user="guest")

        # Admin should pass
        result = await admin_action("do thing", user="admin")
        assert result == "done"

    @pytest.mark.asyncio
    async def test_all_mappers_together(self):
        ctrl = Control(
            name="check-all",
            condition=ConditionNode(
                and_=[
                    ConditionNode(
                        selector="input",
                        evaluator=EvaluatorRef(
                            name="regex", config={"pattern": "DANGER"}
                        ),
                    ),
                    ConditionNode(
                        not_=ConditionNode(
                            selector="context.role",
                            evaluator=EvaluatorRef(
                                name="list", config={"values": ["admin"]}
                            ),
                        )
                    ),
                ]
            ),
            action=ControlAction(decision="deny"),
        )
        engine = ControlEngine([ctrl])

        @control(
            engine=engine,
            input_mapper=lambda text, **kw: text.upper(),
            context_mapper=lambda text, role="user", **kw: {"role": role},
        )
        async def action(text: str, role: str = "user") -> str:
            return "ok"

        # "danger" uppercased + non-admin = deny
        with pytest.raises(ControlViolationError):
            await action("danger", role="user")

        # admin bypasses even with DANGER input
        result = await action("danger", role="admin")
        assert result == "ok"

    def test_sync_mappers(self):
        engine = ControlEngine([_deny_on_pattern(r"MAPPED")])

        @control(
            engine=engine,
            input_mapper=lambda x: "MAPPED",
        )
        def process(x: str) -> str:
            return x

        with pytest.raises(ControlViolationError):
            process("anything")


# ------------------------------------------------------------------
# Method decoration (self handling)
# ------------------------------------------------------------------


class TestMethodDecoration:
    @pytest.mark.asyncio
    async def test_method_self_not_in_input(self):
        engine = ControlEngine([_deny_on_pattern(r"blocked")])

        class Bot:
            @control(engine=engine)
            async def chat(self, message: str) -> str:
                return f"echo: {message}"

        bot = Bot()
        result = await bot.chat("hello")
        assert result == "echo: hello"

        with pytest.raises(ControlViolationError):
            await bot.chat("blocked")

    @pytest.mark.asyncio
    async def test_method_tool_self_not_in_input(self):
        # Use a selector that checks for "self" key in input dict
        ctrl = Control(
            name="check-no-self",
            condition=ConditionNode(
                selector="input.self",
                evaluator=EvaluatorRef(
                    name="regex", config={"pattern": ".*"}
                ),
            ),
            action=ControlAction(decision="deny", message="self leaked"),
        )
        engine = ControlEngine([ctrl])

        class Agent:
            @control(engine=engine, step_type="tool")
            async def search(self, query: str) -> str:
                return "results"

        agent = Agent()
        # Should NOT deny — self should be filtered from the input dict
        result = await agent.search("test query")
        assert result == "results"
