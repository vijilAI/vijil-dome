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
import time
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from vijil_dome.utils.s3_utils import create_s3_client as _create_s3_client

logger = logging.getLogger("vijil.dome")


def get_default_cache_dir() -> Path:
    """Get the default cache directory for FAISS files."""
    cache_home = os.getenv("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    cache_dir = Path(cache_home) / "vijil-dome" / "faiss"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _is_cache_valid(
    file_path: Path,
    metadata_path: Path,
    cache_ttl_seconds: int,
    s3_client,
    bucket: str,
    key: str,
) -> bool:
    """Check if cache is valid based on TTL and ETag.
    
    Args:
        file_path: Path to cached file
        metadata_path: Path to metadata file
        cache_ttl_seconds: Cache TTL in seconds
        s3_client: boto3 S3 client
        bucket: S3 bucket name
        key: S3 object key
        
    Returns:
        True if cache is valid, False otherwise
    """
    if not file_path.exists() or not metadata_path.exists():
        return False
    
    try:
        # Check cache age
        cache_age = time.time() - file_path.stat().st_mtime
        if cache_age < cache_ttl_seconds:
            # Cache is fresh, skip S3 check
            logger.info(f"Using cached file (age: {cache_age:.0f}s < {cache_ttl_seconds}s)")
            return True
        
        # Cache is old, check if S3 file changed
        try:
            s3_head = s3_client.head_object(Bucket=bucket, Key=key)
            s3_etag = s3_head.get("ETag", "").strip('"')
            
            # Load cached metadata
            with open(metadata_path, 'r') as f:
                cached_metadata = json.load(f)
            
            cached_etag = cached_metadata.get("etag", "")
            if s3_etag == cached_etag:
                logger.info(f"Cache valid (ETag match), using cached file")
                # Update cache timestamp
                file_path.touch()
                return True
        except Exception as e:
            logger.warning(f"Failed to validate cache, will re-download: {e}")
            return False
    except Exception as e:
        logger.warning(f"Failed to validate cache, will re-download: {e}")
        return False
    
    return False


def load_faiss_index_from_s3(
    bucket: str,
    key: str,
    cache_dir: Optional[str] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    region_name: Optional[str] = None,
    cache_ttl_seconds: int = 3600,
) -> str:
    """
    Load FAISS index file from S3 with local caching.
    
    Downloads from S3 and caches locally. On subsequent calls, checks cache
    validity using file modification time before downloading again.
    
    Args:
        bucket: S3 bucket name
        key: S3 object key (path to faiss.index file)
        cache_dir: Local cache directory (defaults to ~/.cache/vijil-dome/faiss/)
        aws_access_key_id: AWS access key (optional, uses boto3 defaults if not provided)
        aws_secret_access_key: AWS secret key (optional)
        aws_session_token: AWS session token (optional)
        region_name: AWS region (optional, uses boto3 defaults if not provided)
        cache_ttl_seconds: Cache TTL in seconds - skip S3 head_object check if cache is newer (default: 3600 = 1 hour)
        
    Returns:
        Path to local cached FAISS index file
        
    Raises:
        ImportError: If boto3 is not installed
        Exception: For S3 access errors
    """
    import importlib.util
    if importlib.util.find_spec("boto3") is None:
        raise ImportError(
            "boto3 is not installed. Install it with: pip install boto3"
        )
    
    if cache_dir is None:
        cache_dir = str(get_default_cache_dir())
    
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Create cache filename from bucket and key
    cache_key = f"{bucket}/{key}".replace("/", "_").replace(" ", "_")
    index_file_path = cache_path / f"{cache_key}.index"
    metadata_file_path = cache_path / f"{cache_key}.metadata.json"
    
    # Check cache validity
    s3_client = _create_s3_client(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
        region_name=region_name,
    )
    
    cache_valid = _is_cache_valid(
        index_file_path,
        metadata_file_path,
        cache_ttl_seconds,
        s3_client,
        bucket,
        key,
    )
    
    if cache_valid:
        return str(index_file_path)
    
    # Download from S3 if cache invalid
    logger.info(f"Downloading FAISS index from s3://{bucket}/{key}")
    try:
        s3_object = s3_client.get_object(Bucket=bucket, Key=key)
        index_bytes = s3_object["Body"].read()
        
        # Save to cache
        with open(index_file_path, 'wb') as f:
            f.write(index_bytes)
        
        # Save metadata
        metadata = {
            "etag": s3_object.get("ETag", "").strip('"'),
            "s3_bucket": bucket,
            "s3_key": key,
            "last_updated": time.time(),
        }
        with open(metadata_file_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Cached FAISS index to {index_file_path}")
    except Exception as e:
        if index_file_path.exists():
            logger.warning(f"S3 download failed, using stale cache: {e}")
            return str(index_file_path)
        raise
    
    return str(index_file_path)


def load_section_ids_from_s3(
    bucket: str,
    key: str,
    cache_dir: Optional[str] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    region_name: Optional[str] = None,
    cache_ttl_seconds: int = 3600,
) -> Dict[str, str]:
    """
    Load section_ids.json mapping from S3 with local caching.
    
    The section_ids.json file maps FAISS index positions to section IDs.
    Expected format: {"0": "section-1", "1": "section-2", ...}
    
    Args:
        bucket: S3 bucket name
        key: S3 object key (path to section_ids.json file)
        cache_dir: Local cache directory (defaults to ~/.cache/vijil-dome/faiss/)
        aws_access_key_id: AWS access key (optional, uses boto3 defaults if not provided)
        aws_secret_access_key: AWS secret key (optional)
        aws_session_token: AWS session token (optional)
        region_name: AWS region (optional, uses boto3 defaults if not provided)
        cache_ttl_seconds: Cache TTL in seconds (default: 3600 = 1 hour)
        
    Returns:
        Dictionary mapping FAISS index positions (as strings) to section IDs
        
    Raises:
        ImportError: If boto3 is not installed
        ValueError: If JSON structure is invalid
        Exception: For S3 access errors
    """
    import importlib.util
    if importlib.util.find_spec("boto3") is None:
        raise ImportError(
            "boto3 is not installed. Install it with: pip install boto3"
        )
    
    if cache_dir is None:
        cache_dir = str(get_default_cache_dir())
    
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Create cache filename from bucket and key
    cache_key = f"{bucket}/{key}".replace("/", "_").replace(" ", "_")
    json_file_path = cache_path / f"{cache_key}.json"
    metadata_file_path = cache_path / f"{cache_key}.metadata.json"
    
    # Check cache validity
    cache_valid = False
    if json_file_path.exists() and metadata_file_path.exists():
        try:
            # Check cache age
            cache_age = time.time() - json_file_path.stat().st_mtime
            if cache_age < cache_ttl_seconds:
                # Cache is fresh, skip S3 check
                logger.info(f"Using cached section_ids.json (age: {cache_age:.0f}s < {cache_ttl_seconds}s)")
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                # Apply normalization/validation (same as download path)
                if isinstance(cached_data, list):
                    if len(cached_data) > 0:
                        if isinstance(cached_data[0], str):
                            cached_data = {
                                str(i): section_id for i, section_id in enumerate(cached_data)
                            }
                        elif isinstance(cached_data[0], dict):
                            cached_data = {
                                str(i): item.get("section_id", item.get("id", str(i)))
                                for i, item in enumerate(cached_data)
                            }
                        else:
                            logger.warning(f"Cached data is list but cannot convert, will re-download")
                            cache_valid = False
                    else:
                        cached_data = {}
                elif isinstance(cached_data, dict) and "data" in cached_data:
                    cached_data = cached_data["data"]
                # Validate structure
                if isinstance(cached_data, dict):
                    return cached_data
                else:
                    logger.warning(f"Cached data is not a dict (got {type(cached_data).__name__}), will re-download")
                    cache_valid = False
            
            # Cache is old, check if S3 file changed
            try:
                s3_head = s3_client.head_object(Bucket=bucket, Key=key)
                s3_etag = s3_head.get("ETag", "").strip('"')
                
                # Load cached metadata
                with open(metadata_file_path, 'r') as f:
                    cached_metadata = json.load(f)
                
                cached_etag = cached_metadata.get("etag", "")
                if s3_etag == cached_etag:
                    cache_valid = True
                    logger.info(f"Cache valid (ETag match), using cached section_ids.json")
                    # Update cache timestamp
                    json_file_path.touch()
                    with open(json_file_path, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                        # Handle list format (convert to dict)
                        if isinstance(cached_data, list):
                            if len(cached_data) > 0:
                                if isinstance(cached_data[0], str):
                                    cached_data = {
                                        str(i): section_id for i, section_id in enumerate(cached_data)
                                    }
                                elif isinstance(cached_data[0], dict):
                                    cached_data = {
                                        str(i): item.get("section_id", item.get("id", str(i)))
                                        for i, item in enumerate(cached_data)
                                    }
                                else:
                                    logger.warning(f"Cached data is list but cannot convert, will re-download")
                                    cache_valid = False
                            else:
                                cached_data = {}
                        elif isinstance(cached_data, dict) and "data" in cached_data:
                            cached_data = cached_data["data"]
                        # Validate cached data structure
                        if not isinstance(cached_data, dict):
                            logger.warning(f"Cached data is not a dict (got {type(cached_data).__name__}), will re-download")
                            cache_valid = False
                        elif cache_valid:
                            return cached_data
            except Exception as e:
                logger.warning(f"Failed to validate cache, will re-download: {e}")
        except Exception as e:
            logger.warning(f"Failed to validate cache, will re-download: {e}")
    
    # Download from S3 if cache invalid
    if not cache_valid:
        logger.info(f"Downloading section_ids.json from s3://{bucket}/{key}")
        try:
            s3_object = s3_client.get_object(Bucket=bucket, Key=key)
            json_bytes = s3_object["Body"].read()
            json_str = json_bytes.decode("utf-8")
            section_ids_data = json.loads(json_str)
            
            # Handle case where JSON might be wrapped in a list or have a "data" key
            if isinstance(section_ids_data, list):
                # Convert list to dict - handle both formats:
                # 1. List of strings: ["section-1", "section-2", ...] -> {"0": "section-1", "1": "section-2", ...}
                # 2. List of objects: [{"section_id": "section-1"}, ...] -> {"0": "section-1", ...}
                if len(section_ids_data) > 0:
                    if isinstance(section_ids_data[0], str):
                        # Simple list of strings - map index to section ID
                        section_ids_data = {
                            str(i): section_id for i, section_id in enumerate(section_ids_data)
                        }
                    elif isinstance(section_ids_data[0], dict):
                        # List of objects - extract section_id field
                        section_ids_data = {
                            str(i): item.get("section_id", item.get("id", str(i)))
                            for i, item in enumerate(section_ids_data)
                        }
                    else:
                        raise ValueError(
                            f"section_ids.json is a list but cannot be converted to dict. "
                            f"Expected list of strings or list of objects with 'section_id' field. "
                            f"Content preview: {json_str[:500]}"
                        )
                else:
                    # Empty list - create empty dict
                    section_ids_data = {}
            elif isinstance(section_ids_data, dict) and "data" in section_ids_data:
                # Handle wrapped format: {"data": {...}}
                section_ids_data = section_ids_data["data"]
            
            # Validate structure
            if not isinstance(section_ids_data, dict):
                raise ValueError(
                    f"section_ids.json must be a dictionary, got {type(section_ids_data).__name__}. "
                    f"Content preview: {json_str[:500]}"
                )
            
            # Validate all values are strings (section IDs)
            for k, v in section_ids_data.items():
                if not isinstance(v, str):
                    raise ValueError(f"Section ID at index {k} must be a string, got {type(v)}")
            
            # Save to cache
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(section_ids_data, f, indent=2, ensure_ascii=False)
            
            # Save metadata
            metadata = {
                "etag": s3_object.get("ETag", "").strip('"'),
                "s3_bucket": bucket,
                "s3_key": key,
                "last_updated": time.time(),
            }
            with open(metadata_file_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Cached section_ids.json to {json_file_path}")
            return section_ids_data
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in section_ids.json: {e}")
        except Exception as e:
            if json_file_path.exists():
                logger.warning(f"S3 download failed, using stale cache: {e}")
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            raise
    
    # Should not reach here, but return cached file if exists
    if json_file_path.exists():
        with open(json_file_path, 'r', encoding='utf-8') as f:
            cached_data = json.load(f)
            # Handle list format (convert to dict)
            if isinstance(cached_data, list):
                if len(cached_data) > 0:
                    if isinstance(cached_data[0], str):
                        cached_data = {
                            str(i): section_id for i, section_id in enumerate(cached_data)
                        }
                    elif isinstance(cached_data[0], dict):
                        cached_data = {
                            str(i): item.get("section_id", item.get("id", str(i)))
                            for i, item in enumerate(cached_data)
                        }
                    else:
                        raise ValueError(
                            f"Cached section_ids.json is list but cannot convert. "
                            f"Please delete cache file: {json_file_path}"
                        )
                else:
                    cached_data = {}
            elif isinstance(cached_data, dict) and "data" in cached_data:
                cached_data = cached_data["data"]
            if not isinstance(cached_data, dict):
                raise ValueError(
                    f"Cached section_ids.json is not a dict (got {type(cached_data).__name__}). "
                    f"Please delete cache file: {json_file_path}"
                )
            return cached_data
    
    raise FileNotFoundError(f"Could not load section_ids.json from S3 or cache")
