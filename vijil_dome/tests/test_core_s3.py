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

from unittest.mock import patch

import pytest

from vijil_dome.core import VijilDome

SAMPLE_CONFIG = {
    "input-guards": [],
    "output-guards": [],
    "controls": [
        {
            "name": "block-ssn-sharing",
            "condition": {
                "selector": "output",
                "evaluator": {"name": "policy-rule-judge", "config": {"rules": []}},
            },
            "action": {"decision": "deny"},
        }
    ],
}


def test_create_from_s3_loads_controls_key():
    with patch(
        "vijil_dome.utils.config_loader.load_dome_config_from_s3",
        return_value=SAMPLE_CONFIG,
    ) as mock_load:
        dome = VijilDome.create_from_s3(
            bucket="my-bucket", team_id="team-1", agent_id="agent-1"
        )

    mock_load.assert_called_once()
    assert len(dome.engine.controls) == 1
    assert dome.engine.controls[0].name == "block-ssn-sharing"


def test_create_from_s3_missing_controls_key_yields_empty_engine():
    with patch(
        "vijil_dome.utils.config_loader.load_dome_config_from_s3",
        return_value={"input-guards": [], "output-guards": []},
    ):
        dome = VijilDome.create_from_s3(
            bucket="my-bucket", team_id="team-1", agent_id="agent-1"
        )

    assert dome.engine.controls == []


def test_config_has_changed_requires_s3_origin():
    dome = VijilDome(policy=[])
    with pytest.raises(ValueError, match="created via VijilDome.create_from_s3"):
        dome.config_has_changed()


def test_config_has_changed_delegates_to_config_loader():
    with patch(
        "vijil_dome.utils.config_loader.load_dome_config_from_s3",
        return_value=SAMPLE_CONFIG,
    ):
        dome = VijilDome.create_from_s3(
            bucket="my-bucket", team_id="team-1", agent_id="agent-1"
        )

    with patch(
        "vijil_dome.utils.config_loader.config_has_changed", return_value=True
    ) as mock_changed:
        assert dome.config_has_changed() is True

    mock_changed.assert_called_once()
    call_kwargs = mock_changed.call_args.kwargs
    assert call_kwargs["bucket"] == "my-bucket"
    assert call_kwargs["key"] == "teams/team-1/agents/agent-1/dome/config.json"
    assert call_kwargs["local_config"] == SAMPLE_CONFIG
