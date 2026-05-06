"""AgentControl policy compatibility tests.

These tests load AC-format policies verbatim and verify they parse, load,
and evaluate correctly through the Dome control engine.
"""

import pytest

from vijil_dome.controls.engine import ControlEngine
from vijil_dome.controls.errors import ControlSteerError, ControlViolationError
from vijil_dome.controls.models import Control, Step


# ------------------------------------------------------------------
# AC example policies (verbatim from their repo/docs)
# ------------------------------------------------------------------

AC_BLOCK_SSN = {
    "name": "block-ssn-output",
    "description": "Block SSN patterns in output to prevent PII leakage",
    "enabled": True,
    "execution": "server",
    "scope": {"step_types": ["llm"], "stages": ["post"]},
    "condition": {
        "selector": {"path": "output"},
        "evaluator": {
            "name": "regex",
            "config": {"pattern": r"\b\d{3}-\d{2}-\d{4}\b", "flags": []},
        },
    },
    "action": {"decision": "deny"},
    "tags": ["pii", "ssn", "output-filter"],
}

AC_BLOCK_SQL_KEYWORDS = {
    "name": "block-sql-keywords",
    "description": "Block dangerous SQL operations in input",
    "enabled": True,
    "execution": "server",
    "scope": {"step_types": ["llm"], "stages": ["pre"]},
    "condition": {
        "selector": {"path": "input"},
        "evaluator": {
            "name": "list",
            "config": {
                "values": ["DROP", "DELETE", "TRUNCATE", "ALTER", "GRANT"],
                "logic": "any",
                "match_on": "match",
                "match_mode": "contains",
                "case_sensitive": False,
            },
        },
    },
    "action": {"decision": "deny"},
    "tags": ["sql-injection", "input-filter", "security"],
}

AC_BLOCK_SANCTIONED_COUNTRIES = {
    "name": "block-sanctioned-countries",
    "description": "Block transfers to OFAC sanctioned countries",
    "enabled": True,
    "execution": "server",
    "scope": {
        "step_types": ["tool"],
        "step_names": ["process_wire_transfer"],
        "stages": ["pre"],
    },
    "condition": {
        "selector": {"path": "input.destination_country"},
        "evaluator": {
            "name": "list",
            "config": {
                "values": ["north korea", "iran", "syria", "cuba", "crimea"],
                "logic": "any",
                "match_mode": "contains",
                "case_sensitive": False,
            },
        },
    },
    "action": {"decision": "deny"},
}

AC_BLOCK_FRAUD_RISK = {
    "name": "block-fraud-risk",
    "description": "Block transactions with high fraud risk scores",
    "enabled": True,
    "execution": "server",
    "scope": {
        "step_types": ["tool"],
        "step_names": ["process_wire_transfer"],
        "stages": ["pre"],
    },
    "condition": {
        "selector": {"path": "input"},
        "evaluator": {
            "name": "json",
            "config": {
                "field_constraints": {
                    "fraud_score": {"type": "number", "max": 0.8},
                },
            },
        },
    },
    "action": {"decision": "deny"},
}

AC_OBSERVE_NEW_RECIPIENTS = {
    "name": "observe-new-recipients",
    "description": "Observe transfers to new recipients for audit review",
    "enabled": True,
    "execution": "server",
    "scope": {
        "step_types": ["tool"],
        "step_names": ["process_wire_transfer"],
        "stages": ["pre"],
    },
    "condition": {
        "selector": {"path": "input.recipient"},
        "evaluator": {
            "name": "list",
            "config": {
                "values": ["John Smith", "Acme Corp", "Global Suppliers Inc"],
                "logic": "any",
                "match_on": "no_match",
                "match_mode": "exact",
                "case_sensitive": False,
            },
        },
    },
    "action": {"decision": "observe"},
}

AC_LARGE_TRANSFER_STEER = {
    "name": "large-transfer-2fa",
    "description": "Require verification for large transfers",
    "enabled": True,
    "execution": "server",
    "scope": {
        "step_types": ["tool"],
        "step_names": ["process_wire_transfer"],
        "stages": ["pre"],
    },
    "condition": {
        "selector": {"path": "input"},
        "evaluator": {
            "name": "json",
            "config": {
                # AC example uses oneOf with "match on violation" semantics.
                # Added "required" for case 2 so standard JSON Schema correctly
                # rejects {amount: 50000} without verified_2fa.
                "json_schema": {
                    "type": "object",
                    "oneOf": [
                        {
                            "properties": {
                                "amount": {
                                    "type": "number",
                                    "exclusiveMaximum": 10000,
                                }
                            }
                        },
                        {
                            "properties": {
                                "amount": {
                                    "type": "number",
                                    "minimum": 10000,
                                },
                                "verified_2fa": {"const": True},
                            },
                            "required": ["verified_2fa"],
                        },
                    ],
                }
            },
        },
    },
    "action": {
        "decision": "steer",
        "steering_context": {
            "message": "Large transfer requires identity verification via 2FA",
        },
    },
}

AC_COMPOSITE_AND_NOT = {
    "name": "risk-admin-check",
    "description": "Block high-risk actions by non-admin users",
    "enabled": True,
    "execution": "server",
    "scope": {"stages": ["pre"]},
    "condition": {
        "and": [
            {
                "selector": {"path": "context.risk_level"},
                "evaluator": {
                    "name": "list",
                    "config": {"values": ["high", "critical"]},
                },
            },
            {
                "not": {
                    "selector": {"path": "context.user_role"},
                    "evaluator": {
                        "name": "list",
                        "config": {"values": ["admin", "security"]},
                    },
                }
            },
        ],
    },
    "action": {"decision": "deny"},
}

AC_PROD_CONTROL = {
    "name": "prod-block-keywords",
    "description": "Reject restricted input keywords on prod.",
    "enabled": True,
    "execution": "server",
    "scope": {"step_types": ["llm"], "stages": ["pre"]},
    "condition": {
        "selector": {"path": "input"},
        "evaluator": {
            "name": "list",
            "config": {
                "values": ["DROP TABLE", "rm -rf", "sudo"],
                "logic": "any",
                "match_on": "match",
                "match_mode": "contains",
                "case_sensitive": False,
            },
        },
    },
    "action": {"decision": "deny"},
    "tags": ["env-bound", "prod-only"],
}


# ------------------------------------------------------------------
# Model parsing tests
# ------------------------------------------------------------------


class TestACPolicyParsing:
    """Verify AC policies parse into our Control model correctly."""

    @pytest.mark.parametrize(
        "policy",
        [
            AC_BLOCK_SSN,
            AC_BLOCK_SQL_KEYWORDS,
            AC_BLOCK_SANCTIONED_COUNTRIES,
            AC_BLOCK_FRAUD_RISK,
            AC_OBSERVE_NEW_RECIPIENTS,
            AC_LARGE_TRANSFER_STEER,
            AC_COMPOSITE_AND_NOT,
            AC_PROD_CONTROL,
        ],
        ids=lambda p: p["name"],
    )
    def test_parses_without_error(self, policy):
        ctrl = Control.model_validate(policy)
        assert ctrl.name == policy["name"]
        assert ctrl.enabled is True

    def test_selector_normalized_from_dict(self):
        ctrl = Control.model_validate(AC_BLOCK_SSN)
        assert ctrl.condition.selector == "output"
        assert isinstance(ctrl.condition.selector, str)

    def test_description_preserved(self):
        ctrl = Control.model_validate(AC_BLOCK_SSN)
        assert ctrl.description == "Block SSN patterns in output to prevent PII leakage"

    def test_tags_preserved(self):
        ctrl = Control.model_validate(AC_BLOCK_SSN)
        assert ctrl.tags == ["pii", "ssn", "output-filter"]

    def test_execution_field_ignored(self):
        ctrl = Control.model_validate(AC_BLOCK_SSN)
        assert not hasattr(ctrl, "execution")

    def test_composite_selectors_normalized(self):
        ctrl = Control.model_validate(AC_COMPOSITE_AND_NOT)
        leaf = ctrl.condition.and_[0]
        assert leaf.selector == "context.risk_level"
        not_leaf = ctrl.condition.and_[1].not_
        assert not_leaf.selector == "context.user_role"

    def test_steering_context_parsed(self):
        ctrl = Control.model_validate(AC_LARGE_TRANSFER_STEER)
        assert ctrl.action.steering_context is not None
        assert "2FA" in ctrl.action.steering_context.message


# ------------------------------------------------------------------
# Engine evaluation tests
# ------------------------------------------------------------------


class TestACPolicyEvaluation:
    """Run AC policies through the engine and verify behavior."""

    @pytest.mark.asyncio
    async def test_ssn_blocked_in_output(self):
        engine = ControlEngine()
        engine.load_controls([AC_BLOCK_SSN])

        step = Step(
            type="llm",
            name="chat",
            output="Your SSN is 123-45-6789",
        )
        result = await engine.evaluate(step, stage="post")
        assert result.action == "deny"
        assert not result.permitted

    @pytest.mark.asyncio
    async def test_ssn_clean_output_passes(self):
        engine = ControlEngine()
        engine.load_controls([AC_BLOCK_SSN])

        step = Step(type="llm", name="chat", output="Your account is active")
        result = await engine.evaluate(step, stage="post")
        assert result.action == "allow"
        assert result.permitted

    @pytest.mark.asyncio
    async def test_ssn_not_checked_on_pre_stage(self):
        engine = ControlEngine()
        engine.load_controls([AC_BLOCK_SSN])

        step = Step(
            type="llm",
            name="chat",
            input="My SSN is 123-45-6789",
        )
        result = await engine.evaluate(step, stage="pre")
        assert result.action == "allow"

    @pytest.mark.asyncio
    async def test_sql_keywords_blocked(self):
        engine = ControlEngine()
        engine.load_controls([AC_BLOCK_SQL_KEYWORDS])

        step = Step(type="llm", name="query", input="DROP TABLE users")
        result = await engine.evaluate(step, stage="pre")
        assert result.action == "deny"

    @pytest.mark.asyncio
    async def test_sql_safe_query_passes(self):
        engine = ControlEngine()
        engine.load_controls([AC_BLOCK_SQL_KEYWORDS])

        step = Step(
            type="llm",
            name="query",
            input="SELECT * FROM users WHERE id = 1",
        )
        result = await engine.evaluate(step, stage="pre")
        assert result.action == "allow"

    @pytest.mark.asyncio
    async def test_sanctioned_country_blocked(self):
        engine = ControlEngine()
        engine.load_controls([AC_BLOCK_SANCTIONED_COUNTRIES])

        step = Step(
            type="tool",
            name="process_wire_transfer",
            input={"destination_country": "iran", "amount": 5000},
        )
        result = await engine.evaluate(step, stage="pre")
        assert result.action == "deny"

    @pytest.mark.asyncio
    async def test_allowed_country_passes(self):
        engine = ControlEngine()
        engine.load_controls([AC_BLOCK_SANCTIONED_COUNTRIES])

        step = Step(
            type="tool",
            name="process_wire_transfer",
            input={"destination_country": "germany", "amount": 5000},
        )
        result = await engine.evaluate(step, stage="pre")
        assert result.action == "allow"

    @pytest.mark.asyncio
    async def test_fraud_score_too_high_blocked(self):
        engine = ControlEngine()
        engine.load_controls([AC_BLOCK_FRAUD_RISK])

        step = Step(
            type="tool",
            name="process_wire_transfer",
            input={"fraud_score": 0.95, "amount": 1000},
        )
        result = await engine.evaluate(step, stage="pre")
        assert result.action == "deny"

    @pytest.mark.asyncio
    async def test_fraud_score_acceptable_passes(self):
        engine = ControlEngine()
        engine.load_controls([AC_BLOCK_FRAUD_RISK])

        step = Step(
            type="tool",
            name="process_wire_transfer",
            input={"fraud_score": 0.3, "amount": 1000},
        )
        result = await engine.evaluate(step, stage="pre")
        assert result.action == "allow"

    @pytest.mark.asyncio
    async def test_new_recipient_observed(self):
        engine = ControlEngine()
        engine.load_controls([AC_OBSERVE_NEW_RECIPIENTS])

        step = Step(
            type="tool",
            name="process_wire_transfer",
            input={"recipient": "Unknown Corp", "amount": 500},
        )
        result = await engine.evaluate(step, stage="pre")
        assert result.action == "observe" or (
            result.action == "allow"
            and any(m.triggered for m in result.matches)
        )

    @pytest.mark.asyncio
    async def test_known_recipient_not_observed(self):
        engine = ControlEngine()
        engine.load_controls([AC_OBSERVE_NEW_RECIPIENTS])

        step = Step(
            type="tool",
            name="process_wire_transfer",
            input={"recipient": "John Smith", "amount": 500},
        )
        result = await engine.evaluate(step, stage="pre")
        assert not any(m.triggered for m in result.matches)

    @pytest.mark.asyncio
    async def test_large_transfer_steered(self):
        engine = ControlEngine()
        engine.load_controls([AC_LARGE_TRANSFER_STEER])

        step = Step(
            type="tool",
            name="process_wire_transfer",
            input={"amount": 50000},
        )
        result = await engine.evaluate(step, stage="pre")
        assert result.action == "steer"
        assert result.steering_context is not None
        assert "2FA" in result.steering_context.message

    @pytest.mark.asyncio
    async def test_large_transfer_with_2fa_passes(self):
        engine = ControlEngine()
        engine.load_controls([AC_LARGE_TRANSFER_STEER])

        step = Step(
            type="tool",
            name="process_wire_transfer",
            input={"amount": 50000, "verified_2fa": True},
        )
        result = await engine.evaluate(step, stage="pre")
        assert result.action == "allow"

    @pytest.mark.asyncio
    async def test_small_transfer_passes(self):
        engine = ControlEngine()
        engine.load_controls([AC_LARGE_TRANSFER_STEER])

        step = Step(
            type="tool",
            name="process_wire_transfer",
            input={"amount": 500},
        )
        result = await engine.evaluate(step, stage="pre")
        assert result.action == "allow"

    @pytest.mark.asyncio
    async def test_composite_and_not_denies_non_admin(self):
        engine = ControlEngine()
        engine.load_controls([AC_COMPOSITE_AND_NOT])

        step = Step(
            type="llm",
            name="action",
            input="do something",
            context={"risk_level": "high", "user_role": "viewer"},
        )
        result = await engine.evaluate(step, stage="pre")
        assert result.action == "deny"

    @pytest.mark.asyncio
    async def test_composite_and_not_allows_admin(self):
        engine = ControlEngine()
        engine.load_controls([AC_COMPOSITE_AND_NOT])

        step = Step(
            type="llm",
            name="action",
            input="do something",
            context={"risk_level": "high", "user_role": "admin"},
        )
        result = await engine.evaluate(step, stage="pre")
        assert result.action == "allow"

    @pytest.mark.asyncio
    async def test_composite_and_not_allows_low_risk(self):
        engine = ControlEngine()
        engine.load_controls([AC_COMPOSITE_AND_NOT])

        step = Step(
            type="llm",
            name="action",
            input="do something",
            context={"risk_level": "low", "user_role": "viewer"},
        )
        result = await engine.evaluate(step, stage="pre")
        assert result.action == "allow"

    @pytest.mark.asyncio
    async def test_prod_keywords_blocked(self):
        engine = ControlEngine()
        engine.load_controls([AC_PROD_CONTROL])

        step = Step(type="llm", name="shell", input="sudo rm -rf /")
        result = await engine.evaluate(step, stage="pre")
        assert result.action == "deny"

    @pytest.mark.asyncio
    async def test_prod_safe_input_passes(self):
        engine = ControlEngine()
        engine.load_controls([AC_PROD_CONTROL])

        step = Step(type="llm", name="shell", input="ls -la")
        result = await engine.evaluate(step, stage="pre")
        assert result.action == "allow"


# ------------------------------------------------------------------
# Multi-control evaluation (full AC policy set)
# ------------------------------------------------------------------


class TestACMultiControlPolicy:
    """Test loading multiple AC controls together as a single policy."""

    @pytest.fixture
    def engine(self):
        engine = ControlEngine()
        engine.load_controls([
            AC_BLOCK_SSN,
            AC_BLOCK_SQL_KEYWORDS,
            AC_BLOCK_SANCTIONED_COUNTRIES,
            AC_OBSERVE_NEW_RECIPIENTS,
            AC_LARGE_TRANSFER_STEER,
            AC_PROD_CONTROL,
        ])
        return engine

    @pytest.mark.asyncio
    async def test_all_controls_loaded(self, engine):
        assert len(engine.controls) == 6

    @pytest.mark.asyncio
    async def test_deny_takes_priority_over_steer(self, engine):
        """When both deny and steer could fire, deny wins."""
        engine_mixed = ControlEngine()
        engine_mixed.load_controls([
            AC_BLOCK_SANCTIONED_COUNTRIES,
            AC_LARGE_TRANSFER_STEER,
        ])

        step = Step(
            type="tool",
            name="process_wire_transfer",
            input={
                "destination_country": "iran",
                "amount": 50000,
            },
        )
        result = await engine_mixed.evaluate(step, stage="pre")
        assert result.action == "deny"

    @pytest.mark.asyncio
    async def test_scope_isolation(self, engine):
        """Controls scoped to specific step names don't fire on other steps."""
        step = Step(type="tool", name="search_flights", input={"query": "NYC"})
        result = await engine.evaluate(step, stage="pre")
        assert result.action == "allow"


# ------------------------------------------------------------------
# Unsupported evaluator tests
# ------------------------------------------------------------------


class TestACUnsupportedEvaluators:
    """Verify clear errors for AC evaluators we don't support."""

    @pytest.mark.asyncio
    async def test_galileo_luna2_gives_clear_error(self):
        engine = ControlEngine()
        engine.load_controls([{
            "name": "galileo-toxicity",
            "condition": {
                "selector": "input",
                "evaluator": {"name": "galileo.luna2", "config": {}},
            },
            "action": {"decision": "deny"},
        }])

        step = Step(type="llm", name="chat", input="test")
        result = await engine.evaluate(step, stage="pre")
        error_match = next(
            (m for m in result.matches if m.error), None
        )
        assert error_match is not None
        assert "galileo" in error_match.error.lower()
        assert "Dome" in error_match.error

    @pytest.mark.asyncio
    async def test_cisco_gives_clear_error(self):
        engine = ControlEngine()
        engine.load_controls([{
            "name": "cisco-defense",
            "condition": {
                "selector": "input",
                "evaluator": {"name": "cisco.ai_defense", "config": {}},
            },
            "action": {"decision": "deny"},
        }])

        step = Step(type="llm", name="chat", input="test")
        result = await engine.evaluate(step, stage="pre")
        error_match = next(
            (m for m in result.matches if m.error), None
        )
        assert error_match is not None
        assert "cisco" in error_match.error.lower()

    @pytest.mark.asyncio
    async def test_sql_evaluator_gives_clear_error(self):
        engine = ControlEngine()
        engine.load_controls([{
            "name": "sql-check",
            "condition": {
                "selector": "input",
                "evaluator": {
                    "name": "sql",
                    "config": {"blocked_operations": ["DROP"]},
                },
            },
            "action": {"decision": "deny"},
        }])

        step = Step(type="llm", name="query", input="SELECT 1")
        result = await engine.evaluate(step, stage="pre")
        error_match = next(
            (m for m in result.matches if m.error), None
        )
        assert error_match is not None
        assert "sql" in error_match.error.lower()

    @pytest.mark.asyncio
    async def test_unsupported_evaluator_fail_closed_denies(self):
        """Unsupported evaluator with fail_closed should deny."""
        engine = ControlEngine()
        engine.load_controls([{
            "name": "galileo-strict",
            "condition": {
                "selector": "input",
                "evaluator": {"name": "galileo.luna2", "config": {}},
            },
            "action": {"decision": "deny", "on_error": "fail_closed"},
        }])

        step = Step(type="llm", name="chat", input="test")
        result = await engine.evaluate(step, stage="pre")
        assert result.action == "deny"

    @pytest.mark.asyncio
    async def test_unsupported_evaluator_fail_open_passes(self):
        """Unsupported evaluator with fail_open should allow."""
        engine = ControlEngine()
        engine.load_controls([{
            "name": "galileo-lenient",
            "condition": {
                "selector": "input",
                "evaluator": {"name": "galileo.luna2", "config": {}},
            },
            "action": {"decision": "deny", "on_error": "fail_open"},
        }])

        step = Step(type="llm", name="chat", input="test")
        result = await engine.evaluate(step, stage="pre")
        assert result.action == "allow"


# ------------------------------------------------------------------
# VijilDome integration with AC policies
# ------------------------------------------------------------------


class TestACPolicyWithVijilDome:
    """Test AC policies through the VijilDome class."""

    def test_guard_input_with_ac_policy(self):
        from vijil_dome.core import VijilDome

        dome = VijilDome(policy=[AC_BLOCK_SQL_KEYWORDS, AC_PROD_CONTROL])

        result = dome.guard_input("SELECT * FROM users")
        assert result.permitted

        with pytest.raises(ControlViolationError):
            dome.guard_input("DROP TABLE users")

    def test_guard_output_with_ac_policy(self):
        from vijil_dome.core import VijilDome

        dome = VijilDome(policy=[AC_BLOCK_SSN])

        result = dome.guard_output("Your account is active")
        assert result.permitted

        with pytest.raises(ControlViolationError):
            dome.guard_output("SSN: 123-45-6789")

    def test_guard_tool_call_with_ac_policy(self):
        from vijil_dome.core import VijilDome

        dome = VijilDome(
            policy=[AC_BLOCK_SANCTIONED_COUNTRIES, AC_LARGE_TRANSFER_STEER]
        )

        result = dome.guard_tool_call(
            "process_wire_transfer",
            {"destination_country": "germany", "amount": 500},
        )
        assert result.permitted

        with pytest.raises(ControlViolationError):
            dome.guard_tool_call(
                "process_wire_transfer",
                {"destination_country": "iran", "amount": 500},
            )

    def test_steer_with_ac_policy(self):
        from vijil_dome.core import VijilDome

        dome = VijilDome(policy=[AC_LARGE_TRANSFER_STEER])

        with pytest.raises(ControlSteerError) as exc_info:
            dome.guard_tool_call(
                "process_wire_transfer",
                {"amount": 50000},
            )
        assert "2FA" in exc_info.value.steering_context.message

    def test_shadow_mode_with_ac_policy(self):
        from vijil_dome.core import VijilDome

        dome = VijilDome(policy=[AC_BLOCK_SQL_KEYWORDS], enforce=False)
        result = dome.guard_input("DROP TABLE users")
        assert result.action == "deny"
