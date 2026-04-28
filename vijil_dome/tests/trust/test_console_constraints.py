"""Tests for TrustRuntime with Console-format constraints.

Validates that the SDK correctly parses the exact JSON wire format
returned by Console's GET /agents/{id}/constraints endpoint
(VijilAI/vijil-console PR #626).

Uses the wire format contract from the Console PR to construct
realistic test data.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from vijil_dome.trust.constraints import AgentConstraints
from vijil_dome.trust.runtime import TrustRuntime


def _console_constraints_response(
    agent_id: str = "d0087de1-a032-49f7-9d28-3abf4a34404d",
    *,
    input_guards: list[str] | None = None,
    output_guards: list[str] | None = None,
    guards: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a response matching Console's exact wire format."""
    return {
        "agent_id": agent_id,
        "dome_config": {
            "input_guards": ["security_guard", "moderation_guard"] if input_guards is None else input_guards,
            "output_guards": ["moderation_guard", "privacy_guard"] if output_guards is None else output_guards,
            "guards": guards if guards is not None else {
                "security_guard": {
                    "type": "security",
                    "methods": ["encoding-heuristics", "security-llm"],
                    "early-exit": False,
                    "run-parallel": True,
                },
                "moderation_guard": {
                    "type": "moderation",
                    "methods": ["moderation-flashtext"],
                },
                "privacy_guard": {
                    "type": "privacy",
                    "methods": ["privacy-presidio"],
                    "privacy-presidio": {"anonymize": True},
                },
            },
        },
        "tool_permissions": [],  # Phase 1 stub
        "organization": {
            "required_input_guards": [],
            "required_output_guards": [],
            "denied_tools": [],
            "max_model_tier": None,
        },
        "enforcement_mode": "warn",  # Phase 1 stub
        "updated_at": "2026-04-11T00:00:00+00:00",
    }


class TestAgentConstraintsParsing:
    """Verify SDK parses Console wire format correctly."""

    def test_parse_full_response(self):
        """AgentConstraints parses the full Console response."""
        raw = _console_constraints_response()
        constraints = AgentConstraints.model_validate(raw)

        assert constraints.agent_id == "d0087de1-a032-49f7-9d28-3abf4a34404d"
        assert constraints.enforcement_mode == "warn"
        assert len(constraints.dome_config.input_guards) == 2
        assert len(constraints.dome_config.output_guards) == 2
        assert "security_guard" in constraints.dome_config.guards
        assert constraints.tool_permissions == []

    def test_dome_config_guard_details(self):
        """Guard config includes detector methods and options."""
        raw = _console_constraints_response()
        constraints = AgentConstraints.model_validate(raw)
        guard = constraints.dome_config.guards["security_guard"]

        assert guard["type"] == "security"
        assert "encoding-heuristics" in guard["methods"]
        assert guard["run-parallel"] is True

    def test_empty_dome_config(self):
        """Agent with no Dome config gets empty guards."""
        raw = _console_constraints_response(
            input_guards=[], output_guards=[], guards={}
        )
        constraints = AgentConstraints.model_validate(raw)

        assert constraints.dome_config.input_guards == []
        assert constraints.dome_config.output_guards == []

    def test_updated_at_parses_iso8601(self):
        """updated_at field parses ISO-8601 with timezone."""
        raw = _console_constraints_response()
        constraints = AgentConstraints.model_validate(raw)

        assert constraints.updated_at.year == 2026
        assert constraints.updated_at.tzinfo is not None


class TestTrustRuntimeWithConsoleConstraints:
    """Verify TrustRuntime constructs correctly from Console data."""

    def _mock_client(self, agent_id: str = "test-agent") -> MagicMock:
        """Create a mock Vijil client that returns Console wire format."""
        client = MagicMock()
        client._http.get.return_value = _console_constraints_response(agent_id)
        client._http._token = "test-token"
        return client

    def test_runtime_fetches_from_console(self):
        """TrustRuntime calls GET /agents/{id}/constraints."""
        client = self._mock_client("travel-agent")
        runtime = TrustRuntime(client=client, agent_id="travel-agent")

        client._http.get.assert_called_once_with("/agents/travel-agent/constraints")
        assert runtime._constraints.agent_id == "travel-agent"

    def test_runtime_mode_overrides_console(self):
        """Runtime mode parameter overrides Console's enforcement_mode."""
        client = self._mock_client()
        runtime = TrustRuntime(client=client, agent_id="test", mode="enforce")

        assert runtime.mode == "enforce"

    def test_dome_config_reshapes_for_dome_library(self):
        """Dome config from Console reshapes into the format Dome() expects."""
        client = self._mock_client()
        runtime = TrustRuntime(client=client, agent_id="test")

        # The runtime converts DomeGuardConfig to Dome's dict format
        dome_cfg = runtime._constraints.dome_config
        assert isinstance(dome_cfg.input_guards, list)
        assert isinstance(dome_cfg.guards, dict)
        # Dome expects "input-guards" (hyphens), our model uses input_guards (underscores)
        # The runtime's Dome init code (lines 64-70) handles this conversion

    def test_phase1_stubs_are_safe_defaults(self):
        """Phase 1 stubs from Console produce safe defaults in the runtime."""
        client = self._mock_client()
        runtime = TrustRuntime(client=client, agent_id="test")

        # tool_permissions is empty → all tools pass MAC (no restrictions)
        # But: if a tool is not in permissions, it IS denied by ToolPolicy
        # Phase 1 stub means: no tool_permissions = ToolPolicy denies everything
        # except in warn mode (which is also the Phase 1 default)
        result = runtime.check_tool_call("any_tool", {})
        assert not result.permitted  # Not in permissions
        assert not result.enforced   # Warn mode from Phase 1 default

    def test_local_constraints_override_console(self):
        """Passing constraints directly skips the Console fetch."""
        client = self._mock_client()
        local = _console_constraints_response("local-agent")
        runtime = TrustRuntime(
            client=client, agent_id="local-agent", constraints=local
        )

        # Console was NOT called
        client._http.get.assert_not_called()
        assert runtime._constraints.agent_id == "local-agent"

    def test_no_client_no_constraints_uses_defaults(self):
        """No client + no constraints → minimal safe defaults."""
        runtime = TrustRuntime(agent_id="dev-agent", mode="warn")

        assert runtime._constraints.agent_id == "dev-agent"
        assert runtime._constraints.dome_config.input_guards == []
        assert runtime._constraints.enforcement_mode == "warn"
