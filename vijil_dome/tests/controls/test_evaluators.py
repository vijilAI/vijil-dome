"""Tests for built-in evaluators."""

import pytest

from vijil_dome.controls.evaluators import (
    list_evaluators,
    register_evaluator,
    resolve_evaluator,
)
from vijil_dome.controls.evaluators.base import Evaluator, EvaluatorResult


# ------------------------------------------------------------------
# Regex evaluator
# ------------------------------------------------------------------


class TestRegexEvaluator:
    @pytest.fixture
    def evaluator(self):
        return resolve_evaluator("regex")

    @pytest.mark.asyncio
    async def test_single_pattern_match(self, evaluator):
        result = await evaluator.evaluate(
            "my SSN is 123-45-6789",
            {"pattern": r"\d{3}-\d{2}-\d{4}"},
        )
        assert result.matched is True

    @pytest.mark.asyncio
    async def test_single_pattern_no_match(self, evaluator):
        result = await evaluator.evaluate(
            "hello world",
            {"pattern": r"\d{3}-\d{2}-\d{4}"},
        )
        assert result.matched is False

    @pytest.mark.asyncio
    async def test_multiple_patterns(self, evaluator):
        result = await evaluator.evaluate(
            "DROP TABLE users",
            {"patterns": [r"(?i)drop\s+table", r"(?i)delete\s+from"]},
        )
        assert result.matched is True

    @pytest.mark.asyncio
    async def test_case_insensitive_flag(self, evaluator):
        result = await evaluator.evaluate(
            "HELLO",
            {"pattern": "hello", "flags": "i"},
        )
        assert result.matched is True

    @pytest.mark.asyncio
    async def test_negate(self, evaluator):
        result = await evaluator.evaluate(
            "safe text",
            {"pattern": r"\d{3}-\d{2}-\d{4}", "negate": True},
        )
        assert result.matched is True

    @pytest.mark.asyncio
    async def test_no_pattern(self, evaluator):
        result = await evaluator.evaluate("text", {})
        assert result.matched is False

    @pytest.mark.asyncio
    async def test_none_value(self, evaluator):
        result = await evaluator.evaluate(
            None,
            {"pattern": ".*"},
        )
        assert result.matched is True  # str(None) = "None", .* matches


# ------------------------------------------------------------------
# List evaluator
# ------------------------------------------------------------------


class TestListEvaluator:
    @pytest.fixture
    def evaluator(self):
        return resolve_evaluator("list")

    @pytest.mark.asyncio
    async def test_exact_match(self, evaluator):
        result = await evaluator.evaluate(
            "admin",
            {"values": ["admin", "superuser"]},
        )
        assert result.matched is True

    @pytest.mark.asyncio
    async def test_exact_no_match(self, evaluator):
        result = await evaluator.evaluate(
            "guest",
            {"values": ["admin", "superuser"]},
        )
        assert result.matched is False

    @pytest.mark.asyncio
    async def test_contains_mode(self, evaluator):
        result = await evaluator.evaluate(
            "the quick brown fox",
            {"values": ["quick", "slow"], "match_mode": "contains"},
        )
        assert result.matched is True

    @pytest.mark.asyncio
    async def test_starts_with(self, evaluator):
        result = await evaluator.evaluate(
            "admin_user",
            {"values": ["admin"], "match_mode": "starts_with"},
        )
        assert result.matched is True

    @pytest.mark.asyncio
    async def test_ends_with(self, evaluator):
        result = await evaluator.evaluate(
            "super_admin",
            {"values": ["admin"], "match_mode": "ends_with"},
        )
        assert result.matched is True

    @pytest.mark.asyncio
    async def test_case_insensitive(self, evaluator):
        result = await evaluator.evaluate(
            "ADMIN",
            {"values": ["admin"], "case_sensitive": False},
        )
        assert result.matched is True

    @pytest.mark.asyncio
    async def test_negate(self, evaluator):
        result = await evaluator.evaluate(
            "guest",
            {"values": ["admin"], "negate": True},
        )
        assert result.matched is True

    @pytest.mark.asyncio
    async def test_logic_all(self, evaluator):
        result = await evaluator.evaluate(
            "admin superuser",
            {"values": ["admin", "superuser"], "match_mode": "contains", "logic": "all"},
        )
        assert result.matched is True

    @pytest.mark.asyncio
    async def test_empty_values(self, evaluator):
        result = await evaluator.evaluate("text", {"values": []})
        assert result.matched is False


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------


class TestRegistry:
    def test_list_evaluators_includes_builtins(self):
        names = list_evaluators()
        assert "regex" in names
        assert "list" in names

    def test_resolve_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown evaluator"):
            resolve_evaluator("nonexistent_evaluator_xyz")

    def test_custom_evaluator_registration(self):
        @register_evaluator("test_custom_eval_123")
        class MyEval(Evaluator):
            async def evaluate(self, value, config):
                return EvaluatorResult(matched=True)

        ev = resolve_evaluator("test_custom_eval_123")
        assert isinstance(ev, MyEval)

    def test_register_non_evaluator_raises(self):
        with pytest.raises(TypeError, match="must subclass Evaluator"):

            @register_evaluator("bad")
            class NotAnEvaluator:
                pass
