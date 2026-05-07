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


class TestLoadFromTomlFile:
    def test_toml_native_controls(self, tmp_path):
        """TOML with a [[controls]] array loads like JSON/YAML."""
        toml_text = """\
[[controls]]
name = "block-ssn"

[controls.condition]
selector = "input"

[controls.condition.evaluator]
name = "regex"

[controls.condition.evaluator.config]
pattern = '\\d{3}-\\d{2}-\\d{4}'

[controls.action]
decision = "deny"
message = "SSN detected"
"""
        path = tmp_path / "policy.toml"
        path.write_text(toml_text)

        engine = ControlEngine()
        engine.load_controls_from_file(str(path))
        assert len(engine.controls) == 1
        assert engine.controls[0].name == "block-ssn"

    def test_toml_dome_guardrail_translation(self, tmp_path):
        """Legacy Dome [guardrail] TOML is translated to controls."""
        toml_text = """\
[guardrail]
input-guards = ["prompt-injection"]

[prompt-injection]
type = "security"
methods = ["prompt-injection-deberta-v3-base"]
"""
        path = tmp_path / "guards.toml"
        path.write_text(toml_text)

        engine = ControlEngine()
        engine.load_controls_from_file(str(path))

        assert len(engine.controls) == 1
        ctrl = engine.controls[0]
        assert ctrl.name == "prompt-injection"
        assert ctrl.scope.stages == ["pre"]
        assert ctrl.condition.evaluator.name == "dome:prompt-injection-deberta-v3-base"
        assert ctrl.action.decision == "deny"
        assert ctrl.annotations["vijil.ai/source"] == "dome-toml"
        assert ctrl.annotations["vijil.ai/category"] == "security"

    def test_toml_dome_multi_method_guard(self, tmp_path):
        """Guard with multiple methods produces OR condition."""
        toml_text = """\
[guardrail]
input-guards = ["security-guard"]

[security-guard]
type = "security"
methods = ["prompt-injection-deberta-v3-base", "prompt-injection-mbert"]
"""
        path = tmp_path / "multi.toml"
        path.write_text(toml_text)

        engine = ControlEngine()
        engine.load_controls_from_file(str(path))

        ctrl = engine.controls[0]
        assert ctrl.condition.or_ is not None
        assert len(ctrl.condition.or_) == 2

    def test_toml_dome_output_guard(self, tmp_path):
        """Output guards get stage=post and selector=output."""
        toml_text = """\
[guardrail]
output-guards = ["toxicity"]

[toxicity]
type = "moderation"
methods = ["moderation-deberta"]
"""
        path = tmp_path / "output.toml"
        path.write_text(toml_text)

        engine = ControlEngine()
        engine.load_controls_from_file(str(path))

        ctrl = engine.controls[0]
        assert ctrl.scope.stages == ["post"]
        assert ctrl.condition.selector == "output"

    def test_toml_missing_sections_raises(self, tmp_path):
        """TOML without 'controls' or 'guardrail' raises ValueError."""
        path = tmp_path / "empty.toml"
        path.write_text('[metadata]\nname = "test"\n')

        engine = ControlEngine()
        with pytest.raises(ValueError, match="must contain either"):
            engine.load_controls_from_file(str(path))


class TestLoadFromUnsupportedFile:
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
