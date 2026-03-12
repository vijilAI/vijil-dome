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
from unittest.mock import patch

from vijil_dome import Dome


@pytest.fixture
def sample_policy_data():
    """Sample policy data"""
    return {
        "version": "1.0",
        "policy_id": "test-policy-123",
        "policy_name": "Test Policy",
        "sections": [
            {
                "section_id": "section-1",
                "content": "# Test Policy\n\n## INSTRUCTIONS\nThis is a test policy section.",
                "applies_to": ["input"],
                "metadata": {
                    "header": "Test Policy",
                    "level": 1,
                }
            },
            {
                "section_id": "section-2",
                "content": "# Another Section\n\nMore content here.",
                "applies_to": ["input", "output"],
                "metadata": {
                    "header": "Another Section",
                    "level": 1,
                }
            }
        ]
    }


def test_dome_with_s3_policy_config(sample_policy_data):
    """Test creating Dome with S3 policy config"""
    with patch('vijil_dome.detectors.methods.policy_sections_detector.load_policy_sections_from_s3', return_value=sample_policy_data):
        config = {
            "input-guards": ["policy-input"],
            "policy-input": {
                "type": "policy",
                "methods": ["policy-sections"],
                "policy-sections": {
                    "policy_s3_bucket": "test-bucket",
                    "policy_s3_key": "teams/team-123/policies/policy-456/sections.json",
                    "applies_to": "input",
                    "max_parallel_sections": 10,
                    "model_name": "gpt-4o",
                    "reasoning_effort": "medium",
                    "hub_name": "openai",
                    "timeout": 60,
                    "max_retries": 3,
                }
            }
        }
        
        dome = Dome(config)
        assert dome.input_guardrail is not None
        # Note: output_guardrail may exist if sections have applies_to=["input", "output"]


def test_dome_with_s3_policy_aws_credentials(sample_policy_data):
    """Test Dome config with AWS credentials"""
    with patch('vijil_dome.detectors.methods.policy_sections_detector.load_policy_sections_from_s3', return_value=sample_policy_data) as mock_load:
        config = {
            "input-guards": ["policy-input"],
            "policy-input": {
                "type": "policy",
                "methods": ["policy-sections"],
                "policy-sections": {
                    "policy_s3_bucket": "test-bucket",
                    "policy_s3_key": "test-key",
                    "aws_access_key_id": "test-key",
                    "aws_secret_access_key": "test-secret",
                    "aws_session_token": "test-token",
                    "region_name": "us-east-1",
                }
            }
        }
        
        dome = Dome(config)
        
        # Verify credentials were passed to loader
        mock_load.assert_called_once()
        call_kwargs = mock_load.call_args[1]
        assert call_kwargs["bucket"] == "test-bucket"
        assert call_kwargs["key"] == "test-key"
        assert call_kwargs["aws_access_key_id"] == "test-key"
        assert call_kwargs["aws_secret_access_key"] == "test-secret"
        assert call_kwargs["aws_session_token"] == "test-token"
        assert call_kwargs["region_name"] == "us-east-1"
        
        assert dome.input_guardrail is not None


def test_dome_with_s3_policy_cache_dir(sample_policy_data):
    """Test Dome config with custom cache directory"""
    import tempfile
    with patch('vijil_dome.detectors.methods.policy_sections_detector.load_policy_sections_from_s3', return_value=sample_policy_data) as mock_load:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "input-guards": ["policy-input"],
                "policy-input": {
                    "type": "policy",
                    "methods": ["policy-sections"],
                    "policy-sections": {
                        "policy_s3_bucket": "test-bucket",
                        "policy_s3_key": "test-key",
                        "cache_dir": temp_dir,
                    }
                }
            }
            
            dome = Dome(config)
            
            call_kwargs = mock_load.call_args[1]
            assert call_kwargs["cache_dir"] == temp_dir
            assert dome.input_guardrail is not None


def test_dome_with_s3_policy_detector_params(sample_policy_data):
    """Test Dome config with custom detector parameters"""
    with patch('vijil_dome.detectors.methods.policy_sections_detector.load_policy_sections_from_s3', return_value=sample_policy_data):
        config = {
            "input-guards": ["policy-input"],
            "policy-input": {
                "type": "policy",
                "methods": ["policy-sections"],
                "policy-sections": {
                    "policy_s3_bucket": "test-bucket",
                    "policy_s3_key": "test-key",
                    "model_name": "gpt-4o",
                    "reasoning_effort": "high",
                    "hub_name": "openai",
                    "timeout": 120,
                    "max_retries": 5,
                    "max_parallel_sections": 5,
                }
            }
        }
        
        dome = Dome(config)
        assert dome.input_guardrail is not None


def test_dome_with_s3_policy_rag_config(sample_policy_data):
    """Test Dome config with RAG parameters"""
    with patch('vijil_dome.detectors.methods.policy_sections_detector.load_policy_sections_from_s3', return_value=sample_policy_data), \
         patch('vijil_dome.detectors.methods.policy_sections_detector.load_faiss_index_from_s3', return_value="/tmp/test.index"), \
         patch('vijil_dome.detectors.methods.policy_sections_detector.load_section_ids_from_s3', return_value={"0": "section-1", "1": "section-2"}):
        
        config = {
            "input-guards": ["policy-input"],
            "policy-input": {
                "type": "policy",
                "methods": ["policy-sections"],
                "policy-sections": {
                    "policy_s3_bucket": "test-bucket",
                    "policy_s3_key": "test-key",
                    "use_rag": True,
                    "faiss_s3_key": "test-key/faiss.index",
                    "section_ids_s3_key": "test-key/section_ids.json",
                    "top_k": 5,
                    "similarity_threshold": 0.7,
                    "embedding_model": "text-embedding-ada-002",
                    "embedding_engine": "OpenAI",
                }
            }
        }
        
        dome = Dome(config)
        assert dome.input_guardrail is not None


def test_dome_with_direct_policy_sections():
    """Test Dome config with direct policy sections (no S3)"""
    policy_sections = [
        {
            "section_id": "section-1",
            "content": "# Test Policy\n\n## INSTRUCTIONS\nThis is a test.",
            "applies_to": ["input"],
            "metadata": {
                "header": "Test Policy",
            }
        }
    ]
    
    config = {
        "input-guards": ["policy-input"],
        "policy-input": {
            "type": "policy",
            "methods": ["policy-sections"],
            "policy-sections": {
                "policy_sections": policy_sections,
                "applies_to": "input",
                "hub_name": "openai",  # Explicitly set hub_name
            }
        }
    }
    
    dome = Dome(config)
    assert dome.input_guardrail is not None
