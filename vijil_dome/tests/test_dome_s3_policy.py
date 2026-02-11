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
import sys
from unittest.mock import Mock, patch

from vijil_dome import Dome


@pytest.fixture
def sample_policy_data():
    """Sample policy data"""
    return {
        "version": "1.0",
        "policy_id": "test-policy-123",
        "sections": [
            {
                "section_id": "section-1",
                "content": "# Test Policy",
                "applies_to": ["input"]
            }
        ]
    }


def test_create_from_s3_policy_by_ids(sample_policy_data):
    """Test create_from_s3_policy_by_ids convenience method"""
    dome_module = sys.modules['vijil_dome.Dome']
    
    with patch.object(dome_module, 'load_policy_sections_from_s3', return_value=sample_policy_data) as mock_load_s3, \
         patch.object(dome_module, 'build_dome_config_from_sections', return_value={"input-guards": [], "output-guards": []}):
        
        dome = Dome.create_from_s3_policy_by_ids(
            bucket="test-bucket",
            team_id="team-123",
            policy_id="policy-456"
        )

        # Verify load was called with correct key
        mock_load_s3.assert_called_once()
        call_kwargs = mock_load_s3.call_args[1]
        assert call_kwargs["bucket"] == "test-bucket"
        assert "team-123" in str(mock_load_s3.call_args)
        assert "policy-456" in str(mock_load_s3.call_args)
        
        assert dome.input_guardrail is not None


def test_create_from_s3_policy_full_path(sample_policy_data):
    """Test create_from_s3_policy with full S3 key path"""
    dome_module = sys.modules['vijil_dome.Dome']
    
    with patch.object(dome_module, 'load_policy_sections_from_s3') as mock_load_s3, \
         patch.object(dome_module, 'build_dome_config_from_sections') as mock_build_config:
        
        mock_load_s3.return_value = sample_policy_data
        mock_config = {"input-guards": [], "output-guards": []}
        mock_build_config.return_value = mock_config
        
        dome = Dome.create_from_s3_policy(
            bucket="test-bucket",
            key="custom/path/to/sections.json"
        )

        mock_load_s3.assert_called_once()
        call_kwargs = mock_load_s3.call_args[1]
        assert call_kwargs["bucket"] == "test-bucket"
        assert call_kwargs["key"] == "custom/path/to/sections.json"
        
        assert dome.input_guardrail is not None


def test_create_from_s3_policy_with_detector_kwargs(sample_policy_data):
    """Test create_from_s3_policy with custom detector kwargs"""
    dome_module = sys.modules['vijil_dome.Dome']
    
    with patch.object(dome_module, 'load_policy_sections_from_s3') as mock_load_s3, \
         patch.object(dome_module, 'build_dome_config_from_sections') as mock_build_config:
        
        mock_load_s3.return_value = sample_policy_data
        mock_build_config.return_value = {"input-guards": [], "output-guards": []}
        
        dome = Dome.create_from_s3_policy_by_ids(
            bucket="test-bucket",
            team_id="team-123",
            policy_id="policy-456",
            model_name="openai/gpt-oss-20b",
            reasoning_effort="high",
            timeout=120
        )

        # Verify kwargs were passed to build_config
        call_kwargs = mock_build_config.call_args[1]
        assert call_kwargs["model_name"] == "openai/gpt-oss-20b"
        assert call_kwargs["reasoning_effort"] == "high"
        assert call_kwargs["timeout"] == 120
        assert dome.input_guardrail is not None


def test_create_from_s3_policy_parallel_settings(sample_policy_data):
    """Test create_from_s3_policy with custom parallel settings"""
    dome_module = sys.modules['vijil_dome.Dome']
    
    with patch.object(dome_module, 'load_policy_sections_from_s3') as mock_load_s3, \
         patch.object(dome_module, 'build_dome_config_from_sections') as mock_build_config:
        
        mock_load_s3.return_value = sample_policy_data
        mock_build_config.return_value = {"input-guards": [], "output-guards": []}
        
        dome = Dome.create_from_s3_policy_by_ids(
            bucket="test-bucket",
            team_id="team-123",
            policy_id="policy-456",
            run_parallel=False,
            early_exit=False
        )

        # Verify kwargs were passed
        call_kwargs = mock_build_config.call_args[1]
        assert call_kwargs["run_parallel"] is False
        assert call_kwargs["early_exit"] is False
        
        assert dome.input_guardrail is not None


def test_create_from_s3_policy_aws_credentials(sample_policy_data):
    """Test create_from_s3_policy with AWS credentials"""
    dome_module = sys.modules['vijil_dome.Dome']
    
    with patch.object(dome_module, 'load_policy_sections_from_s3') as mock_load_s3, \
         patch.object(dome_module, 'build_dome_config_from_sections') as mock_build_config:
        
        mock_load_s3.return_value = sample_policy_data
        mock_build_config.return_value = {"input-guards": [], "output-guards": []}
        
        Dome.create_from_s3_policy(
            bucket="test-bucket",
            key="test-key",
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
            region_name="us-east-1"
        )

        # Verify credentials were passed to loader
        call_kwargs = mock_load_s3.call_args[1]
        assert call_kwargs["aws_access_key_id"] == "test-key"
        assert call_kwargs["aws_secret_access_key"] == "test-secret"
        assert call_kwargs["region_name"] == "us-east-1"


def test_create_from_s3_policy_cache_dir(sample_policy_data):
    """Test create_from_s3_policy with custom cache directory"""
    import tempfile
    dome_module = sys.modules['vijil_dome.Dome']
    
    with patch.object(dome_module, 'load_policy_sections_from_s3') as mock_load_s3, \
         patch.object(dome_module, 'build_dome_config_from_sections') as mock_build_config:
        
        mock_load_s3.return_value = sample_policy_data
        mock_build_config.return_value = {"input-guards": [], "output-guards": []}
        
        # Use tempfile for cache directory instead of /custom/cache/path
        with tempfile.TemporaryDirectory() as temp_dir:
            Dome.create_from_s3_policy(
                bucket="test-bucket",
                key="test-key",
                cache_dir=temp_dir
            )

            call_kwargs = mock_load_s3.call_args[1]
            assert call_kwargs["cache_dir"] == temp_dir


def test_create_from_s3_policy_client_parameter(sample_policy_data):
    """Test create_from_s3_policy with OpenAI client parameter"""
    from openai import OpenAI
    dome_module = sys.modules['vijil_dome.Dome']
    
    with patch.object(dome_module, 'load_policy_sections_from_s3') as mock_load_s3, \
         patch.object(dome_module, 'build_dome_config_from_sections') as mock_build_config:
        
        mock_load_s3.return_value = sample_policy_data
        mock_build_config.return_value = {"input-guards": [], "output-guards": []}
        mock_client = Mock(spec=OpenAI)
        
        Dome.create_from_s3_policy(
            bucket="test-bucket",
            key="test-key",
            client=mock_client
        )

        # Verify function was called successfully
        mock_load_s3.assert_called_once()
