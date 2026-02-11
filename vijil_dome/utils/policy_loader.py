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

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger("vijil.dome")


def get_default_cache_dir() -> Path:
    """Get the default cache directory for policy files."""
    cache_home = os.getenv("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    cache_dir = Path(cache_home) / "vijil-dome" / "policies"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def load_policy_sections_from_file(file_path: str) -> Dict[str, Any]:
    """
    Load policy sections JSON from a local file.

    Args:
        file_path: Path to local JSON file

    Returns:
        Parsed policy data dictionary

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If JSON is invalid or structure is invalid
    """
    policy_path = Path(file_path)
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy file not found: {file_path}")

    with open(policy_path, 'r', encoding='utf-8') as f:
        policy_data = json.load(f)

    validate_policy_json(policy_data)
    return policy_data


def validate_policy_json(policy_data: Dict[str, Any]) -> None:
    """
    Validate that policy JSON has required structure.

    Args:
        policy_data: Policy data dictionary to validate

    Raises:
        ValueError: If structure is invalid
    """
    if not isinstance(policy_data, dict):
        raise ValueError("Policy data must be a dictionary")

    if "sections" not in policy_data:
        raise ValueError("Policy data must contain 'sections' array")

    if not isinstance(policy_data["sections"], list):
        raise ValueError("'sections' must be an array")

    if len(policy_data["sections"]) == 0:
        raise ValueError("'sections' array cannot be empty")

    # Validate each section
    for i, section in enumerate(policy_data["sections"]):
        if not isinstance(section, dict):
            raise ValueError(f"Section {i} must be a dictionary")

        if "section_id" not in section:
            raise ValueError(f"Section {i} must have 'section_id' field")

        if "content" not in section:
            raise ValueError(f"Section {i} must have 'content' field")

        if "applies_to" not in section:
            raise ValueError(f"Section {i} must have 'applies_to' field")

        if not isinstance(section["applies_to"], list):
            raise ValueError(f"Section {i} 'applies_to' must be an array")

        if len(section["applies_to"]) == 0:
            raise ValueError(f"Section {i} 'applies_to' cannot be empty")

        valid_applies_to = ["input", "output"]
        for applies in section["applies_to"]:
            if applies not in valid_applies_to:
                raise ValueError(
                    f"Section {i} 'applies_to' contains invalid value '{applies}'. "
                    f"Must be one of {valid_applies_to}"
                )


def load_policy_sections_from_s3(
    bucket: str,
    key: str,
    cache_dir: Optional[str] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    region_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load policy sections JSON from S3 with local caching.

    Downloads from S3 and caches locally. On subsequent calls, checks cache
    validity using ETag and timestamp before downloading again.

    Args:
        bucket: S3 bucket name
        key: S3 object key (path)
        cache_dir: Local cache directory (defaults to ~/.cache/vijil-dome/policies/)
        aws_access_key_id: AWS access key (optional, uses boto3 defaults if not provided)
        aws_secret_access_key: AWS secret key (optional)
        aws_session_token: AWS session token (optional)
        region_name: AWS region (optional, uses boto3 defaults if not provided)

    Returns:
        Parsed policy data dictionary

    Raises:
        ImportError: If boto3 is not installed
        ValueError: If JSON structure is invalid
        Exception: For S3 access errors
    """
    try:
        import boto3
    except ImportError:
        raise ImportError(
            "boto3 is required for S3 access. Install it with: pip install boto3"
        )

    # Determine cache directory
    if cache_dir:
        cache_path = Path(cache_dir)
    else:
        cache_path = get_default_cache_dir()

    # Create cache directory structure
    # Use policy_id from key if available, otherwise use hash of key
    policy_id = _extract_policy_id_from_key(key)
    if not policy_id:
        # Fallback: use hash of key as cache identifier
        import hashlib
        policy_id = hashlib.md5(key.encode()).hexdigest()

    policy_cache_dir = cache_path / policy_id
    policy_cache_dir.mkdir(parents=True, exist_ok=True)

    policy_json_path = policy_cache_dir / "policy.json"
    metadata_json_path = policy_cache_dir / "metadata.json"

    # Check if cache exists and is valid
    cache_valid = False
    if policy_json_path.exists() and metadata_json_path.exists():
        try:
            with open(metadata_json_path, 'r', encoding='utf-8') as f:
                cached_metadata = json.load(f)

            # Create S3 client to check object metadata
            s3_client = _create_s3_client(
                aws_access_key_id, aws_secret_access_key, aws_session_token, region_name
            )
            s3_object = s3_client.head_object(Bucket=bucket, Key=key)

            s3_etag = s3_object.get("ETag", "").strip('"')
            cached_etag = cached_metadata.get("etag", "")

            # Check if ETag matches (indicates file hasn't changed)
            if s3_etag and cached_etag and s3_etag == cached_etag:
                cache_valid = True
                logger.info(f"Using cached policy from {policy_json_path}")
        except Exception as e:
            logger.warning(f"Failed to validate cache, will re-download: {e}")

    # Load from cache if valid
    if cache_valid:
        with open(policy_json_path, 'r', encoding='utf-8') as f:
            policy_data = json.load(f)
        validate_policy_json(policy_data)
        return policy_data

    # Download from S3
    logger.info(f"Downloading policy from s3://{bucket}/{key}")
    s3_client = _create_s3_client(
        aws_access_key_id, aws_secret_access_key, aws_session_token, region_name
    )
    s3_object = s3_client.get_object(Bucket=bucket, Key=key)
    policy_json_bytes = s3_object["Body"].read()
    policy_data = json.loads(policy_json_bytes.decode("utf-8"))

    validate_policy_json(policy_data)

    # Save to cache
    with open(policy_json_path, 'w', encoding='utf-8') as f:
        json.dump(policy_data, f, indent=2, ensure_ascii=False)

    # Save metadata
    metadata = {
        "etag": s3_object.get("ETag", "").strip('"'),
        "last_updated": policy_data.get("last_updated"),
        "source_file": policy_data.get("source_file"),
        "s3_bucket": bucket,
        "s3_key": key,
    }
    with open(metadata_json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Cached policy to {policy_json_path}")
    return policy_data


def _create_s3_client(
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    region_name: Optional[str] = None,
):
    """Create boto3 S3 client with optional credentials."""
    import boto3

    client_kwargs = {}
    if aws_access_key_id:
        client_kwargs["aws_access_key_id"] = aws_access_key_id
    if aws_secret_access_key:
        client_kwargs["aws_secret_access_key"] = aws_secret_access_key
    if aws_session_token:
        client_kwargs["aws_session_token"] = aws_session_token
    if region_name:
        client_kwargs["region_name"] = region_name

    if client_kwargs:
        return boto3.client("s3", **client_kwargs)
    else:
        # Use boto3 default credentials (env vars, IAM role, etc.)
        return boto3.client("s3")


def _extract_policy_id_from_key(key: str) -> Optional[str]:
    """
    Extract policy_id from S3 key path.

    Expected format: teams/{team_id}/policies/{policy_id}/sections.json
    Returns policy_id if found, None otherwise.
    """
    # Try to extract from standard path structure
    parts = key.split("/")
    try:
        # Look for pattern: .../policies/{policy_id}/sections.json
        if "policies" in parts:
            policies_idx = parts.index("policies")
            if policies_idx + 1 < len(parts):
                policy_id = parts[policies_idx + 1]
                # Validate it looks like a UUID or valid identifier
                if policy_id and policy_id != "sections.json":
                    return policy_id
    except (ValueError, IndexError):
        pass
    return None
