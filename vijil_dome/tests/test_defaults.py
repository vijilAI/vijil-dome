"""BC-12: Centralized model/hub constants in defaults.py."""

from __future__ import annotations

import pytest


class TestCentralizedConstants:
    """Model/hub defaults should be centralized and overridable."""

    def test_defaults_module_has_constants(self) -> None:
        import os
        from vijil_dome.defaults import (
            DEFAULT_LLM_HUB,
            DEFAULT_LLM_MODEL,
            DEFAULT_SAFEGUARD_MODEL,
        )

        expected_model = os.environ.get("VIJIL_LLM_MODEL") or "gpt-4-turbo"
        expected_hub = os.environ.get("VIJIL_LLM_HUB") or "openai"
        expected_safeguard = (
            os.environ.get("VIJIL_SAFEGUARD_MODEL") or "openai/gpt-oss-safeguard-20b"
        )
        assert DEFAULT_LLM_MODEL == expected_model
        assert DEFAULT_LLM_HUB == expected_hub
        assert DEFAULT_SAFEGUARD_MODEL == expected_safeguard

    def test_llm_detectors_use_default_constant(self) -> None:
        """LLM detector __init__ defaults should reference centralized constants."""
        import inspect
        from vijil_dome.defaults import DEFAULT_LLM_MODEL
        from vijil_dome.detectors.methods.llm_models import LlmSecurity

        sig = inspect.signature(LlmSecurity.__init__)
        default = sig.parameters["model_name"].default
        assert default == DEFAULT_LLM_MODEL

    def test_env_var_overrides_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """VIJIL_LLM_MODEL env var should override the default."""
        monkeypatch.setenv("VIJIL_LLM_MODEL", "claude-3-opus")
        import importlib
        import vijil_dome.defaults
        importlib.reload(vijil_dome.defaults)
        assert vijil_dome.defaults.DEFAULT_LLM_MODEL == "claude-3-opus"
        monkeypatch.delenv("VIJIL_LLM_MODEL")
        importlib.reload(vijil_dome.defaults)
