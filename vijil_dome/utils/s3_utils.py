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

import os
from typing import Optional


def create_s3_client(
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    region_name: Optional[str] = None,
):
    """Create boto3 S3 client with optional credentials.
    
    This is a shared utility for creating S3 clients with proper credential handling.
    It avoids ProfileNotFound errors when AWS_PROFILE is set but doesn't exist.
    
    Args:
        aws_access_key_id: AWS access key (optional, uses env var if not provided)
        aws_secret_access_key: AWS secret key (optional, uses env var if not provided)
        aws_session_token: AWS session token (optional, uses env var if not provided)
        region_name: AWS region (optional, uses env var if not provided)
        
    Returns:
        boto3 S3 client
        
    Raises:
        ImportError: If boto3 is not installed
    """
    try:
        import boto3
    except ImportError:
        raise ImportError(
            "boto3 is not installed. Install it with: pip install boto3"
        )

    # Collect credentials - prefer explicit params, fall back to env vars
    access_key = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
    session_token = aws_session_token or os.getenv("AWS_SESSION_TOKEN")
    aws_region = region_name or os.getenv("AWS_REGION")
    
    # If we have explicit credentials (from params or env vars), create client directly
    # This avoids ProfileNotFound errors when AWS_PROFILE is set but doesn't exist
    if access_key and secret_key:
        client_kwargs = {
            "aws_access_key_id": access_key,
            "aws_secret_access_key": secret_key,
        }
        if session_token:
            client_kwargs["aws_session_token"] = session_token
        if aws_region:
            client_kwargs["region_name"] = aws_region
        
        # Temporarily unset AWS_PROFILE to avoid ProfileNotFound errors
        # boto3/botocore will try to read it even with explicit credentials
        original_profile = os.environ.pop("AWS_PROFILE", None)
        original_default_profile = os.environ.pop("AWS_DEFAULT_PROFILE", None)
        try:
            # Create client with explicit credentials
            # By unsetting AWS_PROFILE, boto3 won't try to load the non-existent profile
            client = boto3.client("s3", **client_kwargs)
            return client
        finally:
            # Restore AWS_PROFILE if it was set
            if original_profile is not None:
                os.environ["AWS_PROFILE"] = original_profile
            if original_default_profile is not None:
                os.environ["AWS_DEFAULT_PROFILE"] = original_default_profile
    
    # No explicit credentials - try to use boto3 defaults
    # Handle profile that might not exist
    profile_name = os.getenv("AWS_PROFILE")
    if profile_name:
        try:
            # Try to create session with profile
            session = boto3.Session(profile_name=profile_name)
            return session.client("s3")
        except Exception:
            # Profile doesn't exist or is invalid, fall back to default
            pass
    
    return boto3.client("s3")
