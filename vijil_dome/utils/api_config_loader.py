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

"""Load a Dome config directly from Console's API, instead of S3.

Two-hop flow, both legs already exist on Console today (nothing new added
there): exchange a team API key (``client_id``/``client_secret``, minted via
Console's existing "API Keys" page) for a short-lived JWT via
``POST /v1/auth/token``, then call the existing
``GET /{agent_id}/dome-configs/active`` endpoint with that JWT. That endpoint
only ever returns a config with ``status = active`` -- never a pending draft
-- so this can't accidentally start enforcing a config nobody has reviewed
and applied yet.

Caching mirrors :mod:`vijil_dome.utils.config_loader`'s S3 TTL behaviour:
a local on-disk cache, re-checked after *cache_ttl_seconds*. There's no
native ETag from this JSON endpoint, so freshness is instead checked via
the config's own ``id`` (set by Console on every write) -- cheaper than a
full diff, same idea as ``config_has_changed``'s S3 fast path.
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import httpx

logger = logging.getLogger("vijil.dome")

DEFAULT_CONSOLE_API_BASE_URL = "https://console-api.vijil.ai"

# Re-exchange the JWT once fewer than this many seconds remain on it, rather
# than waiting for a 401 -- avoids a doomed request on a token that's about
# to expire mid-flight.
_TOKEN_REFRESH_MARGIN_SECONDS = 60


def _config_cache_key(base_url: str, team_id: str, agent_id: str) -> str:
    return f"{base_url}|{team_id}|{agent_id}"


def _get_config_cache_dir(cache_dir: Optional[str], cache_key: str) -> Path:
    if cache_dir:
        base = Path(cache_dir)
    else:
        import os

        cache_home = os.getenv("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
        base = Path(cache_home) / "vijil-dome" / "configs"
    key_hash = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
    cache_path = base / key_hash
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def exchange_api_key_for_token(
    base_url: str,
    client_id: str,
    client_secret: str,
    timeout: float = 10.0,
) -> Tuple[str, float]:
    """Exchange a Console API key for a short-lived JWT.

    Returns:
        Tuple of (access_token, expires_at_epoch_seconds).

    Raises:
        ValueError: If the exchange fails (invalid/revoked key, network error).
    """
    url = f"{base_url.rstrip('/')}/v1/auth/token"
    try:
        response = httpx.post(
            url,
            json={"client_id": client_id, "client_secret": client_secret},
            timeout=timeout,
        )
        response.raise_for_status()
    except httpx.HTTPError as e:
        raise ValueError(f"Failed to exchange API key for a Console token: {e}") from e

    body = response.json()
    access_token = body["access_token"]
    expires_in = body.get("expires_in", 0)
    return access_token, time.time() + expires_in


def _fetch_active_config(
    base_url: str,
    agent_id: str,
    team_id: str,
    access_token: str,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v1/agent-configurations/{agent_id}/dome-configs/active"
    response = httpx.get(
        url,
        params={"team_id": team_id},
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=timeout,
    )
    if response.status_code == 404:
        raise ValueError(
            f"No active (applied) Dome config for agent {agent_id}. "
            "A config must be generated and Applied on the Protect tab first."
        )
    response.raise_for_status()
    envelope = response.json()
    # The endpoint returns an AgentDomeConfigResponse envelope (id, status,
    # config_body, timestamps, ...) -- the actual controls/guards policy
    # document, same shape S3's config.json is, lives in config_body.
    #
    # A present-but-empty config_body ({}) is a legitimate "no controls
    # configured yet" state. But a *missing* or non-dict config_body is a
    # malformed response -- `or {}` would silently coerce that into the
    # same empty-policy state and cache it, so VijilDome.create_from_api()
    # would install zero controls instead of surfacing a failed load.
    raw_config_body = envelope.get("config_body")
    if raw_config_body is None or not isinstance(raw_config_body, dict):
        raise ValueError(
            f"Console returned a malformed config_body for agent {agent_id}'s "
            f"active Dome config (got {raw_config_body!r}); refusing to silently "
            "install an empty policy."
        )
    # Keep the envelope's own id/updated_at on it too (mirroring config_body's
    # own embedded "id", which Console already sets to match) so change
    # detection has something to compare even if a config_body were ever
    # missing one.
    config_body = raw_config_body
    config_body.setdefault("id", envelope.get("id"))
    config_body.setdefault("updated_at", envelope.get("updated_at"))
    return config_body


def load_dome_config_from_api(
    base_url: str,
    client_id: str,
    client_secret: str,
    agent_id: str,
    team_id: str,
    cache_dir: Optional[str] = None,
    cache_ttl_seconds: int = 300,
    access_token: Optional[str] = None,
    token_expires_at: Optional[float] = None,
) -> Tuple[Dict[str, Any], str, float]:
    """Load the active Dome config for an agent from Console's API, with local caching.

    Args:
        base_url: Console API base URL (e.g. ``https://console-api.vijil.ai``).
        client_id: API key client ID, minted on Console's API Keys page.
        client_secret: API key client secret.
        agent_id: The agent whose active config to fetch.
        team_id: The team the agent belongs to.
        cache_dir: Override local cache directory.
        cache_ttl_seconds: Seconds before re-checking Console (default 300).
        access_token: A still-valid token from a prior call, to skip exchange.
        token_expires_at: Expiry (epoch seconds) for *access_token*.

    Returns:
        Tuple of (config dict, access_token, token_expires_at) -- the token is
        returned so callers can cache and reuse it across calls.
    """
    cache_key = _config_cache_key(base_url, team_id, agent_id)
    cache_path = _get_config_cache_dir(cache_dir, cache_key)
    config_json_path = cache_path / "config.json"
    metadata_json_path = cache_path / "metadata.json"

    if config_json_path.exists():
        cache_age = time.time() - config_json_path.stat().st_mtime
        if cache_age < cache_ttl_seconds:
            logger.info(
                "Using cached config (age: %.0fs < TTL: %ds)", cache_age, cache_ttl_seconds
            )
            try:
                with open(config_json_path, "r", encoding="utf-8") as f:
                    return (
                        json.load(f),
                        access_token or "",
                        token_expires_at or 0.0,
                    )
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning("Corrupt cache file %s, will re-fetch: %s", config_json_path, e)

    if access_token is None or token_expires_at is None or (
        token_expires_at - time.time() < _TOKEN_REFRESH_MARGIN_SECONDS
    ):
        access_token, token_expires_at = exchange_api_key_for_token(
            base_url, client_id, client_secret
        )

    try:
        config_dict = _fetch_active_config(base_url, agent_id, team_id, access_token)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            # Token rejected outright (e.g. revoked key mid-flight) -- one retry
            # with a freshly exchanged token before giving up.
            access_token, token_expires_at = exchange_api_key_for_token(
                base_url, client_id, client_secret
            )
            config_dict = _fetch_active_config(base_url, agent_id, team_id, access_token)
        else:
            raise ValueError(f"Failed to fetch Dome config from Console: {e}") from e

    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    metadata = {
        "config_id": config_dict.get("id"),
        "updated_at": config_dict.get("updated_at"),
        "base_url": base_url,
        "team_id": team_id,
        "agent_id": agent_id,
    }
    with open(metadata_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Cached config to %s", config_json_path)
    return config_dict, access_token, token_expires_at


def api_config_has_changed(
    local_config: Dict[str, Any],
    base_url: str,
    client_id: str,
    client_secret: str,
    agent_id: str,
    team_id: str,
    cache_dir: Optional[str] = None,
) -> bool:
    """Check whether Console's active config differs from *local_config*.

    Forces a live fetch (``cache_ttl_seconds=0``) so the comparison is always
    against the latest version. Compares by ``id`` first (fast path, same as
    the S3 loader's ``config_has_changed``); falls back to a deep comparison
    if either config lacks an ``id``.
    """
    remote_config, _, _ = load_dome_config_from_api(
        base_url=base_url,
        client_id=client_id,
        client_secret=client_secret,
        agent_id=agent_id,
        team_id=team_id,
        cache_dir=cache_dir,
        cache_ttl_seconds=0,
    )

    local_id = local_config.get("id")
    remote_id = remote_config.get("id")
    if local_id is not None and remote_id is not None:
        return local_id != remote_id

    return remote_config != local_config
