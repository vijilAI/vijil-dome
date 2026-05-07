"""Tests for the VijilDome class."""

import logging

import pytest

from vijil_dome.controls.errors import ControlSteerError, ControlViolationError
from vijil_dome.controls.models import (
    ConditionNode,
    Control,
    ControlAction,
    EvaluatorRef,
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


class TestShadowModeLogging:
    def test_deny_shadow_logs_warning(self, caplog):
        dome = VijilDome(policy=_ssn_deny_policy(), enforce=False)
        with caplog.at_level(logging.WARNING, logger="vijil_dome.controls.errors"):
            result = dome.guard_input("SSN: 123-45-6789")
        assert result.action == "deny"
        assert any("[shadow]" in r.message for r in caplog.records)
        assert any("deny" in r.message.lower() for r in caplog.records)

    def test_steer_shadow_logs_warning(self, caplog):
        policy = [
            {
                "name": "steer-test",
                "condition": {
                    "selector": "input",
                    "evaluator": {"name": "regex", "config": {"pattern": ".*"}},
                },
                "action": {
                    "decision": "steer",
                    "steering_context": {"message": "Please rephrase"},
                },
            }
        ]
        dome = VijilDome(policy=policy, enforce=False)
        with caplog.at_level(logging.WARNING, logger="vijil_dome.controls.errors"):
            result = dome.guard_input("anything")
        assert result.action == "steer"
        assert any("[shadow]" in r.message for r in caplog.records)
        assert any("steer" in r.message.lower() for r in caplog.records)

    def test_allow_shadow_no_warning(self, caplog):
        dome = VijilDome(policy=_ssn_deny_policy(), enforce=False)
        with caplog.at_level(logging.WARNING, logger="vijil_dome.controls.errors"):
            result = dome.guard_input("no PII here")
        assert result.action == "allow"
        assert not any("[shadow]" in r.message for r in caplog.records)


class TestEdgeCaseInputs:
    def test_empty_string_input(self):
        dome = VijilDome(policy=_ssn_deny_policy())
        result = dome.guard_input("")
        assert result.permitted is True

    def test_whitespace_input(self):
        dome = VijilDome(policy=_ssn_deny_policy())
        result = dome.guard_input("   ")
        assert result.permitted is True

    def test_dict_input(self):
        dome = VijilDome(policy=_ssn_deny_policy())
        result = dome.guard_input({"message": "no PII"})
        assert result.permitted is True

    def test_no_controls_allows_everything(self):
        dome = VijilDome()
        result = dome.guard_input("")
        assert result.permitted is True

    def test_empty_output(self):
        dome = VijilDome(policy=_output_ssn_deny_policy())
        result = dome.guard_output("")
        assert result.permitted is True

    def test_falsy_selector_zero(self):
        """A selector resolving to 0 should still be evaluated, not skipped."""
        policy = [
            {
                "name": "check-amount",
                "condition": {
                    "selector": "input.amount",
                    "evaluator": {
                        "name": "regex",
                        "config": {"pattern": "^0$"},
                    },
                },
                "action": {"decision": "deny", "message": "Zero amount"},
            }
        ]
        dome = VijilDome(policy=policy)
        with pytest.raises(ControlViolationError):
            dome.guard_input({"amount": 0})

    def test_falsy_selector_empty_string(self):
        """A selector resolving to '' should still be evaluated."""
        policy = [
            {
                "name": "check-name",
                "condition": {
                    "selector": "input.name",
                    "evaluator": {
                        "name": "regex",
                        "config": {"pattern": "^$"},
                    },
                },
                "action": {"decision": "deny", "message": "Empty name"},
            }
        ]
        dome = VijilDome(policy=policy)
        with pytest.raises(ControlViolationError):
            dome.guard_input({"name": ""})

    def test_falsy_selector_false(self):
        """A selector resolving to False should still be evaluated."""
        policy = [
            {
                "name": "check-verified",
                "condition": {
                    "selector": "input.verified",
                    "evaluator": {
                        "name": "regex",
                        "config": {"pattern": "^False$"},
                    },
                },
                "action": {"decision": "deny", "message": "Not verified"},
            }
        ]
        dome = VijilDome(policy=policy)
        with pytest.raises(ControlViolationError):
            dome.guard_input({"verified": False})

    def test_falsy_selector_empty_list(self):
        """A selector resolving to [] should still be evaluated."""
        policy = [
            {
                "name": "check-items",
                "condition": {
                    "selector": "input.items",
                    "evaluator": {
                        "name": "regex",
                        "config": {"pattern": r"^\[\]$"},
                    },
                },
                "action": {"decision": "deny", "message": "Empty items"},
            }
        ]
        dome = VijilDome(policy=policy)
        with pytest.raises(ControlViolationError):
            dome.guard_input({"items": []})
