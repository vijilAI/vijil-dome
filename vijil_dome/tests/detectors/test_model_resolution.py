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

"""Detector weights resolve from the local S3-synced directory, or not at all.

The property under test is a negative one: no code path reaches the
HuggingFace Hub. That matters for air-gapped and egress-restricted
deployments, where a silent fallback turns a misconfigured mount into an
outbound network call rather than an error.
"""

import json
from pathlib import Path

import pytest

from vijil_dome.detectors.utils import hf_model
from vijil_dome.detectors.utils.hf_model import (
    ModelNotAvailableError,
    resolve_model_path,
)


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    """Point MODEL_CACHE_DIR at an empty temp dir for the duration of a test."""
    monkeypatch.setattr(hf_model, "MODEL_CACHE_DIR", str(tmp_path))
    return tmp_path


def _make_model(root: Path, repo_id: str) -> Path:
    """Create a directory that looks like a synced model (has config.json)."""
    d = root / repo_id
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"model_type": "bert"}))
    return d


class TestResolvesLocally:
    def test_resolves_a_synced_model_to_its_local_path(self, model_dir):
        expected = _make_model(model_dir, "vijil/vijil_dome_prompt_injection_detection")
        got = resolve_model_path("vijil/vijil_dome_prompt_injection_detection")
        assert got == str(expected)

    def test_resolves_a_third_party_base_model(self, model_dir):
        # Base tokenizers and encoders are not under the vijil/ prefix, but
        # they are loaded through the same resolver — the mBERT detectors pass
        # answerdotai/ModernBERT-base as tokenizer_name. If the sync only
        # covered vijil/, this is the call that would reach the Hub.
        expected = _make_model(model_dir, "answerdotai/ModernBERT-base")
        assert resolve_model_path("answerdotai/ModernBERT-base") == str(expected)

    def test_absolute_paths_pass_through_untouched(self, model_dir):
        d = _make_model(model_dir, "vijil/some-detector")
        assert resolve_model_path(str(d)) == str(d)


class TestRefusesToReachTheHub:
    def test_missing_model_raises_instead_of_returning_a_hub_id(self, model_dir):
        # The old behaviour returned the repo id so from_pretrained would
        # download it. Returning the id at all is the bug: it is what made an
        # unsynced deployment indistinguishable from a synced one.
        with pytest.raises(ModelNotAvailableError):
            resolve_model_path("vijil/not-synced")

    def test_directory_without_config_json_is_not_a_model(self, model_dir):
        # A partial or interrupted sync leaves the directory but not the
        # weights. Treating that as success yields a confusing failure deeper
        # inside transformers.
        (model_dir / "vijil" / "half-synced").mkdir(parents=True)
        with pytest.raises(ModelNotAvailableError):
            resolve_model_path("vijil/half-synced")

    def test_error_names_the_path_and_the_fix(self, model_dir):
        # The failure a caller sees would otherwise be a network or model-card
        # error naming neither the model directory nor the sync that fills it.
        with pytest.raises(ModelNotAvailableError) as exc:
            resolve_model_path("vijil/not-synced")
        message = str(exc.value)
        assert "vijil/not-synced" in message
        assert str(model_dir) in message
        assert "s3://vijil-inference/models/" in message
        assert "aws s3 sync" in message

    def test_error_mentions_that_hub_is_never_used(self, model_dir):
        with pytest.raises(ModelNotAvailableError) as exc:
            resolve_model_path("vijil/not-synced")
        assert "HuggingFace" in str(exc.value)
