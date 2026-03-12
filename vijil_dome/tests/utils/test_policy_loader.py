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

import pytest
import json
import tempfile
import os
from unittest.mock import patch

from vijil_dome.utils.policy_loader import (
    load_policy_sections_from_file,
    validate_policy_json,
    load_policy_sections_from_s3,
)


@pytest.fixture
def valid_policy_data():
    """Sample valid policy data"""
    return {
        "version": "1.0",
        "last_updated": "2025-01-15T10:30:00Z",
        "etag": "test-etag-123",
        "policy_id": "test-policy-123",
        "policy_name": "Test Policy",
        "source_file": "test.pdf",
        "sections": [
            {
                "section_id": "section-1",
                "content": "# Test Section\n\nContent here",
                "applies_to": ["input"],
                "metadata": {
                    "header": "Test Section",
                    "level": 1,
                    "order": 1
                }
            },
            {
                "section_id": "section-2",
                "content": "# Another Section",
                "applies_to": ["output"],
                "metadata": {
                    "header": "Another Section",
                    "level": 1,
                    "order": 2
                }
            },
            {
                "section_id": "section-3",
                "content": "# Both Section",
                "applies_to": ["input", "output"],
            }
        ]
    }


def test_validate_policy_json_valid(valid_policy_data):
    """Test validation with valid policy data"""
    validate_policy_json(valid_policy_data)
    # Should not raise


def test_validate_policy_json_missing_sections():
    """Test validation fails with missing sections"""
    with pytest.raises(ValueError, match="sections"):
        validate_policy_json({"version": "1.0"})


def test_validate_policy_json_sections_not_array():
    """Test validation fails when sections is not an array"""
    with pytest.raises(ValueError, match="sections.*array"):
        validate_policy_json({"sections": "not-an-array"})


def test_validate_policy_json_empty_sections():
    """Test validation fails with empty sections array"""
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_policy_json({"sections": []})


def test_validate_policy_json_missing_section_id(valid_policy_data):
    """Test validation fails with missing section_id"""
    valid_policy_data["sections"][0].pop("section_id")
    with pytest.raises(ValueError, match="section_id"):
        validate_policy_json(valid_policy_data)


def test_validate_policy_json_missing_content(valid_policy_data):
    """Test validation fails with missing content"""
    valid_policy_data["sections"][0].pop("content")
    with pytest.raises(ValueError, match="content"):
        validate_policy_json(valid_policy_data)


def test_validate_policy_json_missing_applies_to(valid_policy_data):
    """Test validation fails with missing applies_to"""
    valid_policy_data["sections"][0].pop("applies_to")
    with pytest.raises(ValueError, match="applies_to"):
        validate_policy_json(valid_policy_data)


def test_validate_policy_json_applies_to_not_array(valid_policy_data):
    """Test validation fails when applies_to is not an array"""
    valid_policy_data["sections"][0]["applies_to"] = "not-an-array"
    with pytest.raises(ValueError, match="applies_to.*array"):
        validate_policy_json(valid_policy_data)


def test_validate_policy_json_empty_applies_to(valid_policy_data):
    """Test validation fails with empty applies_to"""
    valid_policy_data["sections"][0]["applies_to"] = []
    with pytest.raises(ValueError, match="applies_to.*cannot be empty"):
        validate_policy_json(valid_policy_data)


def test_validate_policy_json_invalid_applies_to_value(valid_policy_data):
    """Test validation fails with invalid applies_to value"""
    valid_policy_data["sections"][0]["applies_to"] = ["invalid"]
    with pytest.raises(ValueError, match="invalid value"):
        validate_policy_json(valid_policy_data)


def test_validate_policy_json_valid_applies_to_values(valid_policy_data):
    """Test validation passes with valid applies_to values"""
    # Test input only
    valid_policy_data["sections"][0]["applies_to"] = ["input"]
    validate_policy_json(valid_policy_data)

    # Test output only
    valid_policy_data["sections"][0]["applies_to"] = ["output"]
    validate_policy_json(valid_policy_data)

    # Test both
    valid_policy_data["sections"][0]["applies_to"] = ["input", "output"]
    validate_policy_json(valid_policy_data)


def test_load_policy_sections_from_file(valid_policy_data):
    """Test loading policy from local file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(valid_policy_data, f, indent=2)
        temp_file = f.name

    try:
        loaded = load_policy_sections_from_file(temp_file)
        assert loaded["policy_id"] == "test-policy-123"
        assert len(loaded["sections"]) == 3
        assert loaded["sections"][0]["section_id"] == "section-1"
    finally:
        os.unlink(temp_file)


def test_load_policy_sections_from_file_not_found():
    """Test loading from non-existent file raises FileNotFoundError"""
    with pytest.raises(FileNotFoundError):
        load_policy_sections_from_file("/nonexistent/path/policy.json")


def test_load_policy_sections_from_file_invalid_json():
    """Test loading invalid JSON raises error"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write("invalid json content")
        temp_file = f.name

    try:
        with pytest.raises(json.JSONDecodeError):
            load_policy_sections_from_file(temp_file)
    finally:
        os.unlink(temp_file)


def test_load_policy_sections_from_file_validates_structure(valid_policy_data):
    """Test that loaded file is validated"""
    # Remove required field
    valid_policy_data["sections"][0].pop("section_id")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(valid_policy_data, f)
        temp_file = f.name

    try:
        with pytest.raises(ValueError, match="section_id"):
            load_policy_sections_from_file(temp_file)
    finally:
        os.unlink(temp_file)


def test_load_policy_sections_from_s3_missing_boto3():
    """Test that missing boto3 raises ImportError"""
    # Temporarily remove boto3 from sys.modules if present
    import sys
    boto3_backup = sys.modules.pop('boto3', None)
    boto3_client_backup = sys.modules.pop('boto3.client', None)
    
    try:
        # Mock the import to raise ImportError
        with patch.dict('sys.modules', {'boto3': None}):
            with pytest.raises(ImportError, match="boto3"):
                load_policy_sections_from_s3("bucket", "key")
    finally:
        # Restore boto3 if it was there
        if boto3_backup:
            sys.modules['boto3'] = boto3_backup
        if boto3_client_backup:
            sys.modules['boto3.client'] = boto3_client_backup


def test_extract_policy_id_from_key():
    """Test policy ID extraction from S3 key"""
    from vijil_dome.utils.policy_loader import _extract_policy_id_from_key

    # Standard format: teams/{team_id}/policies/{policy_id}/sections.json
    key = "teams/team-123/policies/policy-456/sections.json"
    assert _extract_policy_id_from_key(key) == "policy-456"

    # Different format: policies/{policy_id}/sections.json
    key = "policies/policy-789/sections.json"
    assert _extract_policy_id_from_key(key) == "policy-789"

    # No policy ID found
    key = "some/other/path/file.json"
    assert _extract_policy_id_from_key(key) is None

    # Edge case: sections.json in path but not policy ID
    key = "policies/sections.json"
    assert _extract_policy_id_from_key(key) is None
