# Copyright 2025 Vijil, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# vijil and vijil-dome are trademarks owned by Vijil Inc.

"""Tests for API-based Dome config loading and change detection."""

import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

from vijil_dome.utils.api_config_loader import (
    api_config_has_changed,
    exchange_api_key_for_token,
    load_dome_config_from_api,
)

BASE_URL = "https://console-api.example.com"
CLIENT_ID = "vk_abc123"
CLIENT_SECRET = "s3cr3t"
AGENT_ID = "agent-xyz"
TEAM_ID = "team-abc"

SAMPLE_CONTROLS_DOC = {
    "id": "config-1",
    "updated_at": 1000,
    "controls": [
        {
            "name": "deny-control",
            "scope": {"stages": ["post"]},
            "action": {"decision": "deny"},
            "condition": {
                "selector": "output",
                "evaluator": {"name": "policy-rule-judge", "config": {"rules": []}},
            },
        }
    ],
}


def _envelope(config_body, config_id="config-1", updated_at=1000):
    return {
        "id": config_id,
        "status": "active",
        "agent_id": AGENT_ID,
        "config_body": config_body,
        "updated_at": updated_at,
    }


def _mock_response(status_code=200, json_body=None):
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.json.return_value = json_body or {}
    if status_code >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=response
        )
    else:
        response.raise_for_status.side_effect = None
    return response


# ---------------------------------------------------------------------------
# exchange_api_key_for_token
# ---------------------------------------------------------------------------


@patch("vijil_dome.utils.api_config_loader.httpx.post")
def test_exchange_api_key_for_token_success(mock_post):
    mock_post.return_value = _mock_response(
        200, {"access_token": "jwt-123", "expires_in": 3600}
    )

    token, expires_at = exchange_api_key_for_token(BASE_URL, CLIENT_ID, CLIENT_SECRET)

    assert token == "jwt-123"
    assert expires_at > time.time()
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert call_kwargs.args[0] == f"{BASE_URL}/v1/auth/token"
    assert call_kwargs.kwargs["json"] == {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    }


@patch("vijil_dome.utils.api_config_loader.httpx.post")
def test_exchange_api_key_for_token_invalid_raises(mock_post):
    mock_post.return_value = _mock_response(401, {"detail": "invalid credentials"})

    with pytest.raises(ValueError, match="Failed to exchange"):
        exchange_api_key_for_token(BASE_URL, CLIENT_ID, CLIENT_SECRET)


# ---------------------------------------------------------------------------
# load_dome_config_from_api — fetch + unwrap
# ---------------------------------------------------------------------------


@patch("vijil_dome.utils.api_config_loader.httpx.get")
@patch("vijil_dome.utils.api_config_loader.httpx.post")
def test_load_config_unwraps_config_body(mock_post, mock_get, tmp_path):
    mock_post.return_value = _mock_response(
        200, {"access_token": "jwt-123", "expires_in": 3600}
    )
    mock_get.return_value = _mock_response(200, _envelope(SAMPLE_CONTROLS_DOC))

    config, token, expires_at = load_dome_config_from_api(
        base_url=BASE_URL,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        agent_id=AGENT_ID,
        team_id=TEAM_ID,
        cache_dir=str(tmp_path),
    )

    assert config == SAMPLE_CONTROLS_DOC
    assert token == "jwt-123"
    assert expires_at > time.time()
    get_call = mock_get.call_args
    assert get_call.args[0] == f"{BASE_URL}/v1/agent-configurations/{AGENT_ID}/dome-configs/active"
    assert get_call.kwargs["params"] == {"team_id": TEAM_ID}
    assert get_call.kwargs["headers"]["Authorization"] == "Bearer jwt-123"


@patch("vijil_dome.utils.api_config_loader.httpx.get")
@patch("vijil_dome.utils.api_config_loader.httpx.post")
def test_load_config_404_raises_clear_error(mock_post, mock_get, tmp_path):
    mock_post.return_value = _mock_response(
        200, {"access_token": "jwt-123", "expires_in": 3600}
    )
    mock_get.return_value = _mock_response(404, {"detail": "Not Found"})

    with pytest.raises(ValueError, match="No active .applied. Dome config"):
        load_dome_config_from_api(
            base_url=BASE_URL,
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            agent_id=AGENT_ID,
            team_id=TEAM_ID,
            cache_dir=str(tmp_path),
        )


@patch("vijil_dome.utils.api_config_loader.httpx.get")
@patch("vijil_dome.utils.api_config_loader.httpx.post")
def test_load_config_401_retries_with_fresh_token(mock_post, mock_get, tmp_path):
    """A rejected token (e.g. revoked key mid-flight) triggers one re-exchange + retry."""
    mock_post.return_value = _mock_response(
        200, {"access_token": "jwt-fresh", "expires_in": 3600}
    )
    mock_get.side_effect = [
        _mock_response(401, {"detail": "unauthorized"}),
        _mock_response(200, _envelope(SAMPLE_CONTROLS_DOC)),
    ]

    config, token, _ = load_dome_config_from_api(
        base_url=BASE_URL,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        agent_id=AGENT_ID,
        team_id=TEAM_ID,
        cache_dir=str(tmp_path),
        access_token="jwt-stale",
        token_expires_at=time.time() + 3600,
    )

    assert config == SAMPLE_CONTROLS_DOC
    assert token == "jwt-fresh"
    assert mock_get.call_count == 2
    assert mock_post.call_count == 1


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


@patch("vijil_dome.utils.api_config_loader.httpx.get")
@patch("vijil_dome.utils.api_config_loader.httpx.post")
def test_cache_hit_skips_both_calls(mock_post, mock_get, tmp_path):
    # First call populates the cache.
    mock_post.return_value = _mock_response(
        200, {"access_token": "jwt-123", "expires_in": 3600}
    )
    mock_get.return_value = _mock_response(200, _envelope(SAMPLE_CONTROLS_DOC))

    load_dome_config_from_api(
        base_url=BASE_URL,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        agent_id=AGENT_ID,
        team_id=TEAM_ID,
        cache_dir=str(tmp_path),
        cache_ttl_seconds=3600,
    )
    mock_post.reset_mock()
    mock_get.reset_mock()

    config, _, _ = load_dome_config_from_api(
        base_url=BASE_URL,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        agent_id=AGENT_ID,
        team_id=TEAM_ID,
        cache_dir=str(tmp_path),
        cache_ttl_seconds=3600,
    )

    assert config == SAMPLE_CONTROLS_DOC
    mock_post.assert_not_called()
    mock_get.assert_not_called()


@patch("vijil_dome.utils.api_config_loader.httpx.get")
@patch("vijil_dome.utils.api_config_loader.httpx.post")
def test_expired_cache_refetches(mock_post, mock_get, tmp_path):
    mock_post.return_value = _mock_response(
        200, {"access_token": "jwt-123", "expires_in": 3600}
    )
    mock_get.return_value = _mock_response(200, _envelope(SAMPLE_CONTROLS_DOC))

    load_dome_config_from_api(
        base_url=BASE_URL,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        agent_id=AGENT_ID,
        team_id=TEAM_ID,
        cache_dir=str(tmp_path),
        cache_ttl_seconds=0,
    )
    mock_get.reset_mock()

    config, _, _ = load_dome_config_from_api(
        base_url=BASE_URL,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        agent_id=AGENT_ID,
        team_id=TEAM_ID,
        cache_dir=str(tmp_path),
        cache_ttl_seconds=0,
    )

    assert config == SAMPLE_CONTROLS_DOC
    mock_get.assert_called_once()


# ---------------------------------------------------------------------------
# api_config_has_changed
# ---------------------------------------------------------------------------


@patch("vijil_dome.utils.api_config_loader.httpx.get")
@patch("vijil_dome.utils.api_config_loader.httpx.post")
def test_config_has_changed_false(mock_post, mock_get, tmp_path):
    mock_post.return_value = _mock_response(
        200, {"access_token": "jwt-123", "expires_in": 3600}
    )
    mock_get.return_value = _mock_response(200, _envelope(SAMPLE_CONTROLS_DOC))

    result = api_config_has_changed(
        SAMPLE_CONTROLS_DOC,
        base_url=BASE_URL,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        agent_id=AGENT_ID,
        team_id=TEAM_ID,
        cache_dir=str(tmp_path),
    )
    assert result is False


@patch("vijil_dome.utils.api_config_loader.httpx.get")
@patch("vijil_dome.utils.api_config_loader.httpx.post")
def test_config_has_changed_true(mock_post, mock_get, tmp_path):
    mock_post.return_value = _mock_response(
        200, {"access_token": "jwt-123", "expires_in": 3600}
    )
    new_doc = {"id": "config-2", "updated_at": 2000, "controls": []}
    mock_get.return_value = _mock_response(200, _envelope(new_doc, config_id="config-2"))

    result = api_config_has_changed(
        SAMPLE_CONTROLS_DOC,
        base_url=BASE_URL,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        agent_id=AGENT_ID,
        team_id=TEAM_ID,
        cache_dir=str(tmp_path),
    )
    assert result is True


# ---------------------------------------------------------------------------
# VijilDome.create_from_api
# ---------------------------------------------------------------------------


@patch("vijil_dome.utils.api_config_loader.httpx.get")
@patch("vijil_dome.utils.api_config_loader.httpx.post")
def test_vijildome_create_from_api(mock_post, mock_get, tmp_path):
    from vijil_dome.core import VijilDome

    mock_post.return_value = _mock_response(
        200, {"access_token": "jwt-123", "expires_in": 3600}
    )
    mock_get.return_value = _mock_response(200, _envelope(SAMPLE_CONTROLS_DOC))

    dome = VijilDome.create_from_api(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        agent_id=AGENT_ID,
        team_id=TEAM_ID,
        base_url=BASE_URL,
        cache_dir=str(tmp_path),
    )

    assert dome.agent_id == AGENT_ID
    assert dome.team_id == TEAM_ID
    assert dome._api_config_dict == SAMPLE_CONTROLS_DOC
    assert dome._api_access_token == "jwt-123"


@patch("vijil_dome.utils.api_config_loader.httpx.get")
@patch("vijil_dome.utils.api_config_loader.httpx.post")
def test_vijildome_config_has_changed_true(mock_post, mock_get, tmp_path):
    from vijil_dome.core import VijilDome

    mock_post.return_value = _mock_response(
        200, {"access_token": "jwt-123", "expires_in": 3600}
    )
    mock_get.return_value = _mock_response(200, _envelope(SAMPLE_CONTROLS_DOC))

    dome = VijilDome.create_from_api(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        agent_id=AGENT_ID,
        team_id=TEAM_ID,
        base_url=BASE_URL,
        cache_dir=str(tmp_path),
    )

    new_doc = {"id": "config-2", "updated_at": 2000, "controls": []}
    mock_get.return_value = _mock_response(200, _envelope(new_doc, config_id="config-2"))

    assert dome.config_has_changed() is True


def test_vijildome_config_has_changed_not_from_s3_or_api():
    from vijil_dome.core import VijilDome

    dome = VijilDome(policy=[])
    with pytest.raises(ValueError, match="created via VijilDome.create_from_s3"):
        dome.config_has_changed()
