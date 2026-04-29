"""Tests for the VijilDome class."""

import pytest

from vijil_dome.controls.errors import ControlSteerError, ControlViolationError
from vijil_dome.controls.models import (
    ConditionNode,
    Control,
    ControlAction,
    EvaluatorRef,
    SteeringContext,
)
from vijil_dome.core import VijilDome


def _ssn_deny_policy() -> list[dict]:
    return [
        {
            "name": "block-ssn",
            "scope": {"stages": ["pre"]},
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


def _output_ssn_deny_policy() -> list[dict]:
    return [
        {
            "name": "block-ssn-output",
            "scope": {"stages": ["post"]},
            "condition": {
                "selector": "output",
                "evaluator": {
                    "name": "regex",
                    "config": {"pattern": r"\d{3}-\d{2}-\d{4}"},
                },
            },
            "action": {"decision": "deny", "message": "SSN in output"},
        }
    ]


class TestGuardInput:
    def test_deny_blocks(self):
        dome = VijilDome(policy=_ssn_deny_policy())
        with pytest.raises(ControlViolationError):
            dome.guard_input("SSN: 123-45-6789")

    def test_allow_passes(self):
        dome = VijilDome(policy=_ssn_deny_policy())
        result = dome.guard_input("No PII here")
        assert result.permitted is True

    def test_shadow_mode(self):
        dome = VijilDome(policy=_ssn_deny_policy(), enforce=False)
        result = dome.guard_input("SSN: 123-45-6789")
        assert result.permitted is False
        assert result.action == "deny"

    @pytest.mark.asyncio
    async def test_async_deny(self):
        dome = VijilDome(policy=_ssn_deny_policy())
        with pytest.raises(ControlViolationError):
            await dome.async_guard_input("SSN: 123-45-6789")

    @pytest.mark.asyncio
    async def test_async_allow(self):
        dome = VijilDome(policy=_ssn_deny_policy())
        result = await dome.async_guard_input("No PII")
        assert result.permitted is True


class TestGuardOutput:
    def test_deny_blocks(self):
        dome = VijilDome(policy=_output_ssn_deny_policy())
        with pytest.raises(ControlViolationError):
            dome.guard_output("SSN: 123-45-6789")

    def test_allow_passes(self):
        dome = VijilDome(policy=_output_ssn_deny_policy())
        result = dome.guard_output("Clean output")
        assert result.permitted is True

    @pytest.mark.asyncio
    async def test_async_deny(self):
        dome = VijilDome(policy=_output_ssn_deny_policy())
        with pytest.raises(ControlViolationError):
            await dome.async_guard_output("SSN: 123-45-6789")


class TestGuardToolCall:
    def test_tool_deny(self):
        policy = [
            {
                "name": "block-transfer",
                "scope": {"step_types": ["tool"], "step_names": ["transfer_funds"]},
                "condition": {
                    "selector": "input",
                    "evaluator": {"name": "regex", "config": {"pattern": ".*"}},
                },
                "action": {"decision": "deny", "message": "Transfers blocked"},
            }
        ]
        dome = VijilDome(policy=policy)
        with pytest.raises(ControlViolationError):
            dome.guard_tool_call("transfer_funds", {"amount": 1000})

    def test_tool_allow_different_name(self):
        policy = [
            {
                "name": "block-transfer",
                "scope": {"step_types": ["tool"], "step_names": ["transfer_funds"]},
                "condition": {
                    "selector": "input",
                    "evaluator": {"name": "regex", "config": {"pattern": ".*"}},
                },
                "action": {"decision": "deny"},
            }
        ]
        dome = VijilDome(policy=policy)
        result = dome.guard_tool_call("search_flights", {"query": "NYC"})
        assert result.permitted is True

    @pytest.mark.asyncio
    async def test_async_tool_deny(self):
        policy = [
            {
                "name": "block-transfer",
                "scope": {"step_types": ["tool"], "step_names": ["transfer_funds"]},
                "condition": {
                    "selector": "input",
                    "evaluator": {"name": "regex", "config": {"pattern": ".*"}},
                },
                "action": {"decision": "deny"},
            }
        ]
        dome = VijilDome(policy=policy)
        with pytest.raises(ControlViolationError):
            await dome.async_guard_tool_call("transfer_funds", {"amount": 1000})


class TestSteer:
    def test_steer_raises(self):
        policy = [
            {
                "name": "require-2fa",
                "condition": {
                    "selector": "input",
                    "evaluator": {"name": "regex", "config": {"pattern": ".*"}},
                },
                "action": {
                    "decision": "steer",
                    "steering_context": {"message": "Require 2FA"},
                },
            }
        ]
        dome = VijilDome(policy=policy)
        with pytest.raises(ControlSteerError) as exc_info:
            dome.guard_input("test")
        assert exc_info.value.steering_context.message == "Require 2FA"


class TestEmptyPolicy:
    def test_no_policy(self):
        dome = VijilDome()
        result = dome.guard_input("anything")
        assert result.permitted is True

    def test_agent_id(self):
        dome = VijilDome(agent_id="test-bot")
        assert dome.agent_id == "test-bot"


class TestDecoratorMethod:
    @pytest.mark.asyncio
    async def test_control_decorator(self):
        dome = VijilDome(policy=_ssn_deny_policy())

        @dome.control()
        async def chat(msg: str) -> str:
            return f"echo: {msg}"

        with pytest.raises(ControlViolationError):
            await chat("SSN: 123-45-6789")

        result = await chat("hello")
        assert result == "echo: hello"


class TestPolicyFromControlObjects:
    def test_control_objects(self):
        controls = [
            Control(
                name="block-all",
                condition=ConditionNode(
                    selector="input",
                    evaluator=EvaluatorRef(
                        name="regex", config={"pattern": ".*"}
                    ),
                ),
                action=ControlAction(decision="deny"),
            )
        ]
        dome = VijilDome(policy=controls)
        with pytest.raises(ControlViolationError):
            dome.guard_input("test")
