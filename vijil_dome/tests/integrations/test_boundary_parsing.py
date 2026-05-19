"""BC-10/BC-11: External boundary JSON parsing and exception chaining."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestExternalBoundaryParsing:
    """External API responses must be guarded against malformed JSON."""

    def test_evaluate_catches_json_decode_error(self) -> None:
        from vijil_dome.integrations.vijil.evaluate import get_config_from_vijil_agent

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.side_effect = ValueError("not json")

        with patch("httpx.get", return_value=mock_response):
            with pytest.raises(ValueError, match="Invalid response"):
                get_config_from_vijil_agent("token", "agent-1")

    def test_evaluate_preserves_exception_chain(self) -> None:
        import httpx
        from vijil_dome.integrations.vijil.evaluate import (
            get_config_from_vijil_evaluation,
        )

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404", request=MagicMock(), response=MagicMock()
        )

        with patch("httpx.post", return_value=mock_response):
            with pytest.raises(ConnectionError) as exc_info:
                get_config_from_vijil_evaluation("token", "eval-1")
            assert exc_info.value.__cause__ is not None

    def test_safeguard_parse_no_regex_fallback_violation(self) -> None:
        from vijil_dome.detectors.methods.gpt_oss_safeguard_policy import (
            parse_json_output,
        )

        is_violation, data, error = parse_json_output(
            'some garbage with "violation": 1 in it'
        )
        assert is_violation is False
        assert error is not None


class TestNarrowExceptions:
    """Bare except: and broad Exception catches should be narrowed."""

    def test_metrics_catches_exception_not_bare(self) -> None:
        import ast

        from vijil_dome.guardrails.instrumentation import metrics

        source = Path(metrics.__file__).read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                pytest.fail(f"Bare 'except:' at line {node.lineno} in metrics.py")

    def test_config_parser_chains_exceptions(self) -> None:
        import vijil_dome.guardrails.config_parser as mod

        source = Path(mod.__file__).read_text()
        assert "from e" in source
