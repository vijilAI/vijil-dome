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

"""Tests for S3-based Dome config loading and change detection."""

import io
import json
import time
from unittest.mock import MagicMock, patch

import pytest

from vijil_dome.utils.config_loader import (
    build_s3_config_key,
    config_has_changed,
    load_dome_config_from_s3,
)

SAMPLE_CONFIG = {
    "input-guards": [
        {
            "security_default": {
                "type": "security",
                "methods": ["prompt-injection-mbert"],
            }
        }
    ],
    "output-guards": [],
}

BUCKET = "my-dome-bucket"
TEAM_ID = "team-abc"
AGENT_ID = "agent-xyz"
EXPECTED_KEY = "teams/team-abc/agents/agent-xyz/dome/config.json"


# ---------------------------------------------------------------------------
# build_s3_config_key
# ---------------------------------------------------------------------------


def test_build_s3_config_key():
    assert build_s3_config_key("t1", "a1") == "teams/t1/agents/a1/dome/config.json"


# ---------------------------------------------------------------------------
# Key resolution
# ---------------------------------------------------------------------------


def test_load_requires_key_or_ids():
    with pytest.raises(ValueError, match="Either 'key' or both"):
        load_dome_config_from_s3(BUCKET)


def test_load_requires_both_ids():
    with pytest.raises(ValueError, match="Either 'key' or both"):
        load_dome_config_from_s3(BUCKET, team_id=TEAM_ID)


# ---------------------------------------------------------------------------
# load_dome_config_from_s3 — download path
# ---------------------------------------------------------------------------


@patch("vijil_dome.utils.config_loader._create_s3_client")
def test_load_config_from_s3_with_ids(mock_create_client, tmp_path):
    """Config is downloaded using constructed key from team_id + agent_id."""
    body = io.BytesIO(json.dumps(SAMPLE_CONFIG).encode())
    mock_client = MagicMock()
    mock_client.get_object.return_value = {"Body": body, "ETag": '"etag123"'}
    mock_create_client.return_value = mock_client

    result = load_dome_config_from_s3(
        BUCKET,
        team_id=TEAM_ID,
        agent_id=AGENT_ID,
        cache_dir=str(tmp_path),
    )

    assert result == SAMPLE_CONFIG
    mock_client.get_object.assert_called_once_with(Bucket=BUCKET, Key=EXPECTED_KEY)


@patch("vijil_dome.utils.config_loader._create_s3_client")
def test_load_config_from_s3_with_explicit_key(mock_create_client, tmp_path):
    """Explicit key overrides team_id/agent_id."""
    custom_key = "custom/path/config.json"
    body = io.BytesIO(json.dumps(SAMPLE_CONFIG).encode())
    mock_client = MagicMock()
    mock_client.get_object.return_value = {"Body": body, "ETag": '"etag456"'}
    mock_create_client.return_value = mock_client

    result = load_dome_config_from_s3(
        BUCKET,
        key=custom_key,
        cache_dir=str(tmp_path),
    )

    assert result == SAMPLE_CONFIG
    mock_client.get_object.assert_called_once_with(Bucket=BUCKET, Key=custom_key)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


@patch("vijil_dome.utils.config_loader._create_s3_client")
def test_cache_hit_skips_s3(mock_create_client, tmp_path):
    """Within TTL, cached config is used without contacting S3."""
    # Pre-populate cache
    from vijil_dome.utils.config_loader import _get_config_cache_dir

    cache_path = _get_config_cache_dir(str(tmp_path), EXPECTED_KEY)
    (cache_path / "config.json").write_text(json.dumps(SAMPLE_CONFIG))
    (cache_path / "metadata.json").write_text(json.dumps({"etag": "etag123"}))

    mock_client = MagicMock()
    mock_create_client.return_value = mock_client

    result = load_dome_config_from_s3(
        BUCKET,
        team_id=TEAM_ID,
        agent_id=AGENT_ID,
        cache_dir=str(tmp_path),
        cache_ttl_seconds=3600,
    )

    assert result == SAMPLE_CONFIG
    mock_client.get_object.assert_not_called()
    mock_client.head_object.assert_not_called()


@patch("vijil_dome.utils.config_loader._create_s3_client")
def test_etag_match_reuses_cache(mock_create_client, tmp_path):
    """Expired TTL but matching ETag reuses cache without downloading."""
    from vijil_dome.utils.config_loader import _get_config_cache_dir

    cache_path = _get_config_cache_dir(str(tmp_path), EXPECTED_KEY)
    config_file = cache_path / "config.json"
    config_file.write_text(json.dumps(SAMPLE_CONFIG))
    (cache_path / "metadata.json").write_text(json.dumps({"etag": "etag-same"}))

    # Make cache appear old
    old_time = time.time() - 600
    import os

    os.utime(config_file, (old_time, old_time))

    mock_client = MagicMock()
    mock_client.head_object.return_value = {"ETag": '"etag-same"'}
    mock_create_client.return_value = mock_client

    result = load_dome_config_from_s3(
        BUCKET,
        team_id=TEAM_ID,
        agent_id=AGENT_ID,
        cache_dir=str(tmp_path),
        cache_ttl_seconds=300,
    )

    assert result == SAMPLE_CONFIG
    mock_client.head_object.assert_called_once()
    mock_client.get_object.assert_not_called()


@patch("vijil_dome.utils.config_loader._create_s3_client")
def test_etag_mismatch_redownloads(mock_create_client, tmp_path):
    """Expired TTL and different ETag triggers fresh download."""
    from vijil_dome.utils.config_loader import _get_config_cache_dir

    cache_path = _get_config_cache_dir(str(tmp_path), EXPECTED_KEY)
    config_file = cache_path / "config.json"
    config_file.write_text(json.dumps({"old": "config"}))
    (cache_path / "metadata.json").write_text(json.dumps({"etag": "etag-old"}))

    old_time = time.time() - 600
    import os

    os.utime(config_file, (old_time, old_time))

    body = io.BytesIO(json.dumps(SAMPLE_CONFIG).encode())
    mock_client = MagicMock()
    mock_client.head_object.return_value = {"ETag": '"etag-new"'}
    mock_client.get_object.return_value = {"Body": body, "ETag": '"etag-new"'}
    mock_create_client.return_value = mock_client

    result = load_dome_config_from_s3(
        BUCKET,
        team_id=TEAM_ID,
        agent_id=AGENT_ID,
        cache_dir=str(tmp_path),
        cache_ttl_seconds=300,
    )

    assert result == SAMPLE_CONFIG
    mock_client.get_object.assert_called_once()


# ---------------------------------------------------------------------------
# config_has_changed
# ---------------------------------------------------------------------------


@patch("vijil_dome.utils.config_loader._create_s3_client")
def test_config_has_changed_false(mock_create_client, tmp_path):
    """Returns False when S3 config matches local."""
    body = io.BytesIO(json.dumps(SAMPLE_CONFIG).encode())
    mock_client = MagicMock()
    mock_client.get_object.return_value = {"Body": body, "ETag": '"e1"'}
    mock_create_client.return_value = mock_client

    result = config_has_changed(
        SAMPLE_CONFIG,
        BUCKET,
        team_id=TEAM_ID,
        agent_id=AGENT_ID,
        cache_dir=str(tmp_path),
    )
    assert result is False


@patch("vijil_dome.utils.config_loader._create_s3_client")
def test_config_has_changed_true(mock_create_client, tmp_path):
    """Returns True when S3 config differs from local."""
    different_config = {"input-guards": [], "output-guards": []}
    body = io.BytesIO(json.dumps(different_config).encode())
    mock_client = MagicMock()
    mock_client.get_object.return_value = {"Body": body, "ETag": '"e2"'}
    mock_create_client.return_value = mock_client

    result = config_has_changed(
        SAMPLE_CONFIG,
        BUCKET,
        team_id=TEAM_ID,
        agent_id=AGENT_ID,
        cache_dir=str(tmp_path),
    )
    assert result is True


# ---------------------------------------------------------------------------
# Dome.create_from_s3
# ---------------------------------------------------------------------------


@patch("vijil_dome.utils.config_loader._create_s3_client")
def test_dome_create_from_s3(mock_create_client, tmp_path):
    """Factory method creates a working Dome instance."""
    from vijil_dome import Dome

    body = io.BytesIO(json.dumps(SAMPLE_CONFIG).encode())
    mock_client = MagicMock()
    mock_client.get_object.return_value = {"Body": body, "ETag": '"etag-factory"'}
    mock_create_client.return_value = mock_client

    dome = Dome.create_from_s3(
        BUCKET,
        team_id=TEAM_ID,
        agent_id=AGENT_ID,
        cache_dir=str(tmp_path),
    )

    assert dome.input_guardrail is not None
    assert dome._s3_bucket == BUCKET
    assert dome._s3_key == EXPECTED_KEY
    assert dome._s3_config_dict == SAMPLE_CONFIG


@patch("vijil_dome.utils.config_loader._create_s3_client")
def test_dome_create_from_s3_explicit_key(mock_create_client, tmp_path):
    """Factory method works with explicit key."""
    from vijil_dome import Dome

    custom_key = "my/custom/config.json"
    body = io.BytesIO(json.dumps(SAMPLE_CONFIG).encode())
    mock_client = MagicMock()
    mock_client.get_object.return_value = {"Body": body, "ETag": '"etag-custom"'}
    mock_create_client.return_value = mock_client

    dome = Dome.create_from_s3(
        BUCKET,
        key=custom_key,
        cache_dir=str(tmp_path),
    )

    assert dome._s3_key == custom_key


# ---------------------------------------------------------------------------
# Dome.config_has_changed (instance method)
# ---------------------------------------------------------------------------


@patch("vijil_dome.utils.config_loader._create_s3_client")
def test_dome_config_has_changed_false(mock_create_client, tmp_path):
    """Instance method returns False when config unchanged."""
    from vijil_dome import Dome

    config_bytes = json.dumps(SAMPLE_CONFIG).encode()

    mock_client = MagicMock()
    # First call for create_from_s3
    mock_client.get_object.return_value = {
        "Body": io.BytesIO(config_bytes),
        "ETag": '"etag-same"',
    }
    mock_create_client.return_value = mock_client

    dome = Dome.create_from_s3(
        BUCKET, team_id=TEAM_ID, agent_id=AGENT_ID, cache_dir=str(tmp_path)
    )

    # Second call for config_has_changed (cache_ttl_seconds=0 forces recheck)
    mock_client.get_object.return_value = {
        "Body": io.BytesIO(config_bytes),
        "ETag": '"etag-same"',
    }

    assert dome.config_has_changed() is False


@patch("vijil_dome.utils.config_loader._create_s3_client")
def test_dome_config_has_changed_true(mock_create_client, tmp_path):
    """Instance method returns True when config changed on S3."""
    from vijil_dome import Dome

    mock_client = MagicMock()
    mock_client.get_object.return_value = {
        "Body": io.BytesIO(json.dumps(SAMPLE_CONFIG).encode()),
        "ETag": '"etag-v1"',
    }
    mock_create_client.return_value = mock_client

    dome = Dome.create_from_s3(
        BUCKET, team_id=TEAM_ID, agent_id=AGENT_ID, cache_dir=str(tmp_path)
    )

    new_config = {"input-guards": [], "output-guards": []}
    mock_client.get_object.return_value = {
        "Body": io.BytesIO(json.dumps(new_config).encode()),
        "ETag": '"etag-v2"',
    }

    assert dome.config_has_changed() is True


def test_dome_config_has_changed_not_from_s3():
    """Raises ValueError for Dome not created from S3."""
    from vijil_dome import Dome

    dome = Dome(dome_config=SAMPLE_CONFIG)
    with pytest.raises(ValueError, match="only available for Dome instances"):
        dome.config_has_changed()
