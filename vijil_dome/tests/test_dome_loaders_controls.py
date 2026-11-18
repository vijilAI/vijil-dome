import sys
import warnings
from unittest.mock import MagicMock, patch

import httpx
import pytest

from vijil_dome import Dome
from vijil_dome.core import VijilDome
from vijil_dome.utils.api_config_loader import DEFAULT_CONSOLE_API_BASE_URL

GUARDS = {
    "input-guards": ["in-guard"],
    "output-guards": [],
    "in-guard": {"type": "security", "methods": ["encoding-heuristics"]},
}
REGEX_DENY = {
    "name": "block-secret",
    "scope": {"stages": ["post"]},
    "action": {"decision": "deny"},
    "condition": {
        "selector": "output",
        "evaluator": {"name": "regex", "config": {"pattern": "SECRET-123"}},
    },
}
BODY = {**GUARDS, "controls": [REGEX_DENY]}

# The package re-exports the Dome class under the module's name, so reach the
# module itself to patch names it imported.
dome_module = sys.modules["vijil_dome.Dome"]


def _from_api(config):
    with patch(
        "vijil_dome.utils.api_config_loader.load_dome_config_from_api",
        return_value=(config, "tok", 0.0),
    ):
        return Dome.create_from_api("cid", "secret", "ag", "tm")


def _from_s3(config):
    with patch.object(dome_module, "load_dome_config_from_s3", return_value=config):
        return Dome.create_from_s3("bucket", team_id="tm", agent_id="ag")


def _guard_names(dome):
    return [g.guard_name for g in dome.input_guardrail.guard_list]


def test_dome_create_from_api_warns_on_controls():
    with pytest.warns(DeprecationWarning, match="does not enforce controls"):
        _from_api(BODY)


def test_dome_create_from_s3_warns_on_controls():
    with pytest.warns(DeprecationWarning, match="does not enforce controls"):
        _from_s3(BODY)


def test_dome_loaders_silent_without_controls():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _from_api({**GUARDS, "controls": []})
        _from_s3(GUARDS)


def test_dome_create_from_api_result_unchanged():
    with pytest.warns(DeprecationWarning):
        with_controls = _from_api(BODY)
    without_controls = _from_api(GUARDS)
    assert _guard_names(with_controls) == _guard_names(without_controls)
    assert with_controls.output_guardrail.guard_list == []


def _ok_response(config_body):
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.raise_for_status.side_effect = None
    response.json.return_value = {"id": "c1", "config_body": config_body}
    return response


@patch("vijil_dome.utils.api_config_loader.httpx.get")
def test_create_from_vijil_agent_fetches_active_config(mock_get):
    mock_get.return_value = _ok_response(dict(BODY))
    with pytest.warns(DeprecationWarning):
        v = Dome.create_from_vijil_agent("ag", "key", team_id="tm")

    args, kwargs = mock_get.call_args
    assert (
        args[0]
        == f"{DEFAULT_CONSOLE_API_BASE_URL}/v1/agent-configurations/ag/dome-configs/active"
    )
    assert kwargs["params"] == {"team_id": "tm"}
    assert kwargs["headers"]["Authorization"] == "Bearer key"
    assert isinstance(v, VijilDome) and v.team_id == "tm"
    assert v.input_guardrail is not None
    assert len(v.engine.controls) == 1


@patch("vijil_dome.utils.api_config_loader.httpx.get")
def test_create_from_vijil_agent_uses_given_base_url(mock_get):
    mock_get.return_value = _ok_response(dict(GUARDS))
    with pytest.warns(DeprecationWarning):
        Dome.create_from_vijil_agent(
            "ag", "key", "https://console.example.com", team_id="tm"
        )
    assert mock_get.call_args.args[0].startswith("https://console.example.com/v1/")


@patch("vijil_dome.utils.api_config_loader.httpx.get")
def test_create_from_vijil_agent_warns_deprecated(mock_get):
    mock_get.return_value = _ok_response(dict(GUARDS))
    with pytest.warns(DeprecationWarning, match="VijilDome.create_from_api"):
        Dome.create_from_vijil_agent("ag", "key", team_id="tm")


def test_create_from_vijil_agent_requires_team_id():
    with pytest.raises(TypeError):
        Dome.create_from_vijil_agent("ag", "key")  # type: ignore[call-arg]


def test_controls_warning_logged_once_per_loader(caplog):
    dome_module._loaders_warned_about_controls.clear()
    with pytest.warns(DeprecationWarning):
        _from_api(BODY)
        _from_api(BODY)
        _from_s3(BODY)
    logged = [r.getMessage() for r in caplog.records if r.name == "vijil_dome.Dome"]
    assert len(logged) == 2
    assert "Dome.create_from_api loaded a config with 1 controls" in logged[0]
    assert "Use VijilDome.create_from_s3 to enforce" in logged[1]
