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

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from vijil_dome.utils.s3_utils import create_s3_client as _create_s3_client

logger = logging.getLogger("vijil.dome")

S3_CONFIG_KEY_TEMPLATE = "teams/{team_id}/agents/{agent_id}/dome/config.json"


def build_s3_config_key(team_id: str, agent_id: str) -> str:
    """Construct the standard S3 key for a Dome config.

    Args:
        team_id: Team identifier.
        agent_id: Agent identifier.

    Returns:
        S3 object key in the form ``teams/{team_id}/agents/{agent_id}/dome/config.json``.
    """
    return S3_CONFIG_KEY_TEMPLATE.format(team_id=team_id, agent_id=agent_id)


def _resolve_key(
    key: Optional[str] = None,
    team_id: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> str:
    """Resolve the S3 key from explicit key or team_id + agent_id."""
    if key is not None:
        return key
    if team_id is not None and agent_id is not None:
        return build_s3_config_key(team_id, agent_id)
    raise ValueError(
        "Either 'key' or both 'team_id' and 'agent_id' must be provided."
    )


def _get_config_cache_dir(cache_dir: Optional[str], s3_key: str) -> Path:
    """Get the cache directory for a given S3 key."""
    if cache_dir:
        base = Path(cache_dir)
    else:
        cache_home = os.getenv("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
        base = Path(cache_home) / "vijil-dome" / "configs"
    key_hash = hashlib.sha256(s3_key.encode()).hexdigest()[:16]
    cache_path = base / key_hash
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def load_dome_config_from_s3(
    bucket: str,
    key: Optional[str] = None,
    team_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    cache_dir: Optional[str] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    region_name: Optional[str] = None,
    cache_ttl_seconds: int = 300,
) -> Dict[str, Any]:
    """Load a Dome config dict from S3 with local caching.

    The S3 key can be provided directly via *key*, or constructed
    automatically from *team_id* and *agent_id* using the standard path
    ``teams/{team_id}/agents/{agent_id}/dome/config.json``.

    Cache behaviour mirrors :func:`vijil_dome.utils.policy_loader.load_policy_sections_from_s3`:
    TTL-based fast path, then ETag comparison, with graceful fallback to
    stale cache on S3 errors.

    Args:
        bucket: S3 bucket name.
        key: Full S3 object key (overrides team_id/agent_id).
        team_id: Team identifier (used with agent_id to build the key).
        agent_id: Agent identifier (used with team_id to build the key).
        cache_dir: Override local cache directory.
        aws_access_key_id: AWS access key (optional).
        aws_secret_access_key: AWS secret key (optional).
        aws_session_token: AWS session token (optional).
        region_name: AWS region (optional).
        cache_ttl_seconds: Seconds before re-checking S3 (default 300).

    Returns:
        Parsed Dome config dictionary.

    Raises:
        ImportError: If boto3 is not installed.
        ValueError: If key resolution fails or JSON is invalid.
    """
    import importlib.util

    if importlib.util.find_spec("boto3") is None:
        raise ImportError(
            "boto3 is required for S3 access. Install it with: pip install boto3"
        )

    s3_key = _resolve_key(key, team_id, agent_id)
    cache_path = _get_config_cache_dir(cache_dir, s3_key)
    config_json_path = cache_path / "config.json"
    metadata_json_path = cache_path / "metadata.json"

    s3_client = _create_s3_client(
        aws_access_key_id, aws_secret_access_key, aws_session_token, region_name
    )

    # Check cache validity
    cache_valid = False
    if config_json_path.exists() and metadata_json_path.exists():
        cache_age = time.time() - config_json_path.stat().st_mtime
        if cache_age < cache_ttl_seconds:
            logger.info(
                "Using cached config (age: %.0fs < TTL: %ds)", cache_age, cache_ttl_seconds
            )
            cache_valid = True
        else:
            try:
                with open(metadata_json_path, "r", encoding="utf-8") as f:
                    cached_metadata = json.load(f)

                s3_head = s3_client.head_object(Bucket=bucket, Key=s3_key)
                s3_etag = s3_head.get("ETag", "").strip('"')
                cached_etag = cached_metadata.get("etag", "")

                if s3_etag and cached_etag and s3_etag == cached_etag:
                    cache_valid = True
                    # Touch the file to reset TTL window
                    config_json_path.touch()
                    logger.info("Cache ETag match – reusing cached config")
            except Exception as e:
                logger.warning("Cache validation failed, will re-download: %s", e)

    if cache_valid:
        with open(config_json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Download from S3
    logger.info("Downloading config from s3://%s/%s", bucket, s3_key)
    s3_object = s3_client.get_object(Bucket=bucket, Key=s3_key)
    config_bytes = s3_object["Body"].read()
    config_dict = json.loads(config_bytes.decode("utf-8"))

    if not isinstance(config_dict, dict):
        raise ValueError("Dome config must be a JSON object (dict)")

    # Write cache
    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    metadata = {
        "etag": s3_object.get("ETag", "").strip('"'),
        "s3_bucket": bucket,
        "s3_key": s3_key,
    }
    with open(metadata_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Cached config to %s", config_json_path)
    return config_dict


def config_has_changed(
    local_config: Dict[str, Any],
    bucket: str,
    key: Optional[str] = None,
    team_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    config_id: Optional[str] = None,
    cache_dir: Optional[str] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    region_name: Optional[str] = None,
) -> bool:
    """Check whether the S3 config differs from *local_config*.

    When *config_id* is provided, the remote config's ``"id"`` field is
    compared first as a fast path — if the IDs match, the configs are
    assumed identical and a full deep comparison is skipped.

    Forces an ETag check (``cache_ttl_seconds=0``) so the comparison is
    always against the latest S3 version.

    Args:
        local_config: The config dict to compare against.
        bucket: S3 bucket name.
        key: Full S3 key (overrides team_id/agent_id).
        team_id: Team identifier.
        agent_id: Agent identifier.
        config_id: The ``id`` of the local config for fast comparison.
        cache_dir: Override local cache directory.
        aws_access_key_id: AWS access key (optional).
        aws_secret_access_key: AWS secret key (optional).
        aws_session_token: AWS session token (optional).
        region_name: AWS region (optional).

    Returns:
        ``True`` if the S3 config is different from *local_config*, ``False`` otherwise.
    """
    remote_config = load_dome_config_from_s3(
        bucket=bucket,
        key=key,
        team_id=team_id,
        agent_id=agent_id,
        cache_dir=cache_dir,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
        region_name=region_name,
        cache_ttl_seconds=0,
    )

    # Fast path: if both configs have an "id" field, compare IDs only
    remote_id = remote_config.get("id")
    if config_id is not None and remote_id is not None:
        return config_id != remote_id

    # Fallback: deep comparison
    return remote_config != local_config
