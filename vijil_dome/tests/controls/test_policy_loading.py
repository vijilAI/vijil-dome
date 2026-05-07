"""Tests for file-based policy loading (JSON and YAML)."""

from __future__ import annotations

import json

import pytest

from vijil_dome.controls.engine import ControlEngine
from vijil_dome.controls.decorator import control
from vijil_dome.controls.errors import ControlViolationError
from vijil_dome.controls.models import Step

try:
    import yaml  # type: ignore[import-untyped]

    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


POLICY_DICTS = [
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


class TestLoadFromJsonFile:
    def test_json_flat_list(self, tmp_path):
        path = tmp_path / "policy.json"
        path.write_text(json.dumps(POLICY_DICTS))

        engine = ControlEngine()
        engine.load_controls_from_file(str(path))

        assert len(engine.controls) == 1
        assert engine.controls[0].name == "block-ssn"

    def test_json_wrapped_in_controls_key(self, tmp_path):
        path = tmp_path / "policy.json"
        path.write_text(json.dumps({"controls": POLICY_DICTS}))

        engine = ControlEngine()
        engine.load_controls_from_file(str(path))

        assert len(engine.controls) == 1

    @pytest.mark.asyncio
    async def test_loaded_policy_evaluates(self, tmp_path):
        path = tmp_path / "policy.json"
        path.write_text(json.dumps(POLICY_DICTS))

        engine = ControlEngine()
        engine.load_controls_from_file(str(path))

        step = Step(type="llm", name="chat", input="SSN: 123-45-6789")
        result = await engine.evaluate(step, stage="pre")
        assert result.action == "deny"

        step_clean = Step(type="llm", name="chat", input="no PII here")
        result_clean = await engine.evaluate(step_clean, stage="pre")
        assert result_clean.action == "allow"


@pytest.mark.skipif(not _HAS_YAML, reason="PyYAML not installed")
class TestLoadFromYamlFile:
    def test_yaml_flat_list(self, tmp_path):
        path = tmp_path / "policy.yaml"
        path.write_text(yaml.dump(POLICY_DICTS))

        engine = ControlEngine()
        engine.load_controls_from_file(str(path))

        assert len(engine.controls) == 1
        assert engine.controls[0].name == "block-ssn"

    def test_yaml_wrapped_in_controls_key(self, tmp_path):
        path = tmp_path / "policy.yml"
        path.write_text(yaml.dump({"controls": POLICY_DICTS}))

        engine = ControlEngine()
        engine.load_controls_from_file(str(path))

        assert len(engine.controls) == 1

    @pytest.mark.asyncio
    async def test_yaml_policy_evaluates(self, tmp_path):
        path = tmp_path / "policy.yaml"
        path.write_text(yaml.dump(POLICY_DICTS))

        engine = ControlEngine()
        engine.load_controls_from_file(str(path))

        step = Step(type="llm", name="chat", input="SSN: 123-45-6789")
        result = await engine.evaluate(step, stage="pre")
        assert result.action == "deny"


class TestLoadFromUnsupportedFile:
    def test_toml_rejected(self, tmp_path):
        path = tmp_path / "policy.toml"
        path.write_text("[controls]")

        engine = ControlEngine()
        with pytest.raises(ValueError, match="Unsupported policy file format"):
            engine.load_controls_from_file(str(path))

    def test_txt_rejected(self, tmp_path):
        path = tmp_path / "policy.txt"
        path.write_text("{}")

        engine = ControlEngine()
        with pytest.raises(ValueError, match="Unsupported policy file format"):
            engine.load_controls_from_file(str(path))


class TestDecoratorWithPolicyFile:
    @pytest.mark.asyncio
    async def test_control_with_json_policy_file(self, tmp_path):
        path = tmp_path / "policy.json"
        path.write_text(json.dumps(POLICY_DICTS))

        @control(policy=str(path))
        async def chat(msg: str) -> str:
            return f"echo: {msg}"

        with pytest.raises(ControlViolationError):
            await chat("SSN: 123-45-6789")

        result = await chat("hello")
        assert result == "echo: hello"

    @pytest.mark.skipif(not _HAS_YAML, reason="PyYAML not installed")
    @pytest.mark.asyncio
    async def test_control_with_yaml_policy_file(self, tmp_path):
        path = tmp_path / "policy.yaml"
        path.write_text(yaml.dump(POLICY_DICTS))

        @control(policy=str(path))
        async def chat(msg: str) -> str:
            return f"echo: {msg}"

        with pytest.raises(ControlViolationError):
            await chat("SSN: 123-45-6789")
